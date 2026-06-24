// GGUF block dequantization (Q4_K, Q6_K, Q8_0, F16, F32) -> bf16, plus bf16
// transposes. The Q4_K/Q6_K decoders are validated byte-exact against the gguf
// python reference (.cudaverify/deqtest.cu). Used to load GGUF weights: dense
// tensors are dequantized once at load; expert stacks are kept quantized in VRAM
// and dequantized per-layer into a reused scratch buffer.
//
// Portable CUDA — runs on sm_89 .. sm_120/sm_121 (RTX 5090 / PRO 6000 / Spark).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

// ggml type ids
enum { GGML_F32 = 0, GGML_F16 = 1, GGML_Q8_0 = 8, GGML_Q4_K = 12, GGML_Q6_K = 14 };

__device__ __forceinline__ float gg_h2f(const unsigned char* p) {
    __half h; *((unsigned short*)&h) = *(const unsigned short*)p; return __half2float(h);
}

__device__ __forceinline__ void gg_scale_min_k4(int j, const unsigned char* q, int* d, int* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

// one thread per 256-value block
__global__ void deq_q4k_kernel(const unsigned char* __restrict__ src, __nv_bfloat16* __restrict__ y, long nblocks) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nblocks) return;
    const unsigned char* blk = src + b * 144;
    float d = gg_h2f(blk), dmin = gg_h2f(blk + 2);
    const unsigned char* sc = blk + 4; const unsigned char* q = blk + 16;
    __nv_bfloat16* yy = y + b * 256; int is = 0;
    for (int j = 0; j < 256; j += 64) {
        int s, m;
        gg_scale_min_k4(is,   sc, &s, &m); float d1 = d * s, m1 = dmin * m;
        gg_scale_min_k4(is+1, sc, &s, &m); float d2 = d * s, m2 = dmin * m;
        for (int l = 0; l < 32; l++) yy[j + l]      = __float2bfloat16(d1 * (q[l] & 0xF) - m1);
        for (int l = 0; l < 32; l++) yy[j + 32 + l] = __float2bfloat16(d2 * (q[l] >> 4)  - m2);
        q += 32; is += 2;
    }
}

__global__ void deq_q6k_kernel(const unsigned char* __restrict__ src, __nv_bfloat16* __restrict__ y, long nblocks) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nblocks) return;
    const unsigned char* blk = src + b * 210;
    const unsigned char* ql = blk; const unsigned char* qh = blk + 128;
    const signed char* sc = (const signed char*)(blk + 192); float d = gg_h2f(blk + 208);
    __nv_bfloat16* yy = y + b * 256;
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            yy[l]    = __float2bfloat16(d * sc[is + 0] * q1);
            yy[l+32] = __float2bfloat16(d * sc[is + 2] * q2);
            yy[l+64] = __float2bfloat16(d * sc[is + 4] * q3);
            yy[l+96] = __float2bfloat16(d * sc[is + 6] * q4);
        }
        ql += 64; qh += 32; sc += 8; yy += 128;
    }
}

__global__ void deq_q8_0_kernel(const unsigned char* __restrict__ src, __nv_bfloat16* __restrict__ y, long nblocks) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nblocks) return;
    const unsigned char* blk = src + b * 34; float d = gg_h2f(blk);
    const signed char* q = (const signed char*)(blk + 2); __nv_bfloat16* yy = y + b * 32;
    for (int l = 0; l < 32; l++) yy[l] = __float2bfloat16(d * q[l]);
}

__global__ void deq_f16_kernel(const unsigned char* __restrict__ src, __nv_bfloat16* __restrict__ y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    y[i] = __float2bfloat16(gg_h2f(src + i * 2));
}
__global__ void deq_f32_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ y, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    y[i] = __float2bfloat16(src[i]);
}

__global__ void transpose2d_kernel(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst, int rows, int cols) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; if (idx >= (long)rows * cols) return;
    int r = idx / cols, c = idx % cols;
    dst[(long)c * rows + r] = src[idx];               // [rows,cols] -> [cols,rows]
}
__global__ void transpose3d_kernel(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst, int E, int A, int B) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; if (idx >= (long)E * A * B) return;
    int e = idx / ((long)A * B); int rem = idx % ((long)A * B); int a = rem / B, b = rem % B;
    dst[((long)e * B + b) * A + a] = src[idx];        // [E,A,B] -> [E,B,A]
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/quant.h"
#include <cstdio>

void launch_gguf_dequant(int ggml_type, const void* src, void* dst_bf16, long n_values, cudaStream_t stream) {
    auto* d = reinterpret_cast<__nv_bfloat16*>(dst_bf16);
    auto* s = reinterpret_cast<const unsigned char*>(src);
    const int T = 256;
    if (ggml_type == GGML_Q4_K) { long nb = n_values/256; deq_q4k_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == GGML_Q6_K) { long nb = n_values/256; deq_q6k_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == GGML_Q8_0) { long nb = n_values/32;  deq_q8_0_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == GGML_F16)  { deq_f16_kernel<<<(n_values+T-1)/T,T,0,stream>>>(s,d,n_values); }
    else if (ggml_type == GGML_F32)  { deq_f32_kernel<<<(n_values+T-1)/T,T,0,stream>>>(reinterpret_cast<const float*>(src),d,n_values); }
    else { fprintf(stderr, "[gguf] launch_gguf_dequant: unsupported ggml type %d — no decoder (e.g. Q5_K=13); refusing to reinterpret as F32\n", ggml_type); }
}

void launch_transpose_bf16(const void* src, void* dst, int rows, int cols, cudaStream_t stream) {
    long n = (long)rows*cols; const int T=256;
    transpose2d_kernel<<<(n+T-1)/T,T,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(src), reinterpret_cast<__nv_bfloat16*>(dst), rows, cols);
}
void launch_transpose3d_bf16(const void* src, void* dst, int E, int A, int B, cudaStream_t stream) {
    long n = (long)E*A*B; const int T=256;
    transpose3d_kernel<<<(n+T-1)/T,T,0,stream>>>(reinterpret_cast<const __nv_bfloat16*>(src), reinterpret_cast<__nv_bfloat16*>(dst), E, A, B);
}
#endif

} // namespace kernels
} // namespace sparkinfer
