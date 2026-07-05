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
enum { GGML_F32 = 0, GGML_F16 = 1, GGML_Q8_0 = 8, GGML_Q4_K = 12, GGML_Q5_K = 13, GGML_Q6_K = 14 };

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

// Q5_K: 176-byte super-block of 256 — d, dmin (fp16), 6-bit scales+mins (12B, like Q4_K),
// qh 1 high bit/quant (32B), qs 4 low bits/quant (128B). Byte-exact match to the ggml reference.
__global__ void deq_q5k_kernel(const unsigned char* __restrict__ src, __nv_bfloat16* __restrict__ y, long nblocks) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nblocks) return;
    const unsigned char* blk = src + b * 176;
    float d = gg_h2f(blk), dmin = gg_h2f(blk + 2);
    const unsigned char* sc = blk + 4;    // scales + mins (6-bit packed)
    const unsigned char* qh = blk + 16;   // high bit per quant
    const unsigned char* ql = blk + 48;   // low 4 bits per quant
    __nv_bfloat16* yy = y + b * 256; int is = 0; unsigned char u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        int s, m;
        gg_scale_min_k4(is,   sc, &s, &m); float d1 = d * s, m1 = dmin * m;
        gg_scale_min_k4(is+1, sc, &s, &m); float d2 = d * s, m2 = dmin * m;
        for (int l = 0; l < 32; l++) yy[j + l]      = __float2bfloat16(d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1);
        for (int l = 0; l < 32; l++) yy[j + 32 + l] = __float2bfloat16(d2 * ((ql[l] >> 4)  + ((qh[l] & u2) ? 16 : 0)) - m2);
        ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
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

// ---- bf16 -> Q6_K requantize (exact inverse of deq_q6k_kernel above) ------------------
// Ports ggml quantize_row_q6_K_ref + make_qx_quants (rmse_type=1, nmax=32): each 256-value
// superblock is 16 sub-blocks of 16; each sub-block gets an int8 scale, the superblock a fp16 d.
// Used to re-quantize the Q8_0 projection weights (attn/GDN q/k/v/o/qkv/gate/out) DOWN to Q6_K
// at load, so the decode GEMV moves 0.82 vs 2.0 bytes/weight; near-lossless (6.5 vs 8 bit).
__device__ __forceinline__ int q6_nearest_int(float f) { return __float2int_rn(f); }

// best scale for 16 values into L[0..16) (unsigned 6-bit, +32 bias), weighting error by x^2
__device__ float q6_make_qx_quants16(const float* x, signed char* L) {
    float amax = 0.f, vmax = 0.f;
    for (int i = 0; i < 16; ++i) { float ax = fabsf(x[i]); if (ax > amax) { amax = ax; vmax = x[i]; } }
    if (amax < 1e-30f) { for (int i = 0; i < 16; ++i) L[i] = 0; return 0.f; }
    float iscale = -32.f / vmax;
    float sumlx = 0.f, suml2 = 0.f;
    for (int i = 0; i < 16; ++i) {
        int l = q6_nearest_int(iscale * x[i]); l = max(-32, min(31, l)); L[i] = (signed char)(l + 32);
        float w = x[i] * x[i]; sumlx += w * x[i] * l; suml2 += w * (float)l * l;
    }
    float scale = suml2 > 0.f ? sumlx / suml2 : 0.f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        float is2 = -(32.f + 0.1f * is) / vmax;
        float slx = 0.f, sl2 = 0.f;
        for (int i = 0; i < 16; ++i) {
            int l = max(-32, min(31, q6_nearest_int(is2 * x[i])));
            float w = x[i] * x[i]; slx += w * x[i] * l; sl2 += w * (float)l * l;
        }
        if (sl2 > 0.f && slx * slx > best * sl2) {
            for (int i = 0; i < 16; ++i) {
                int l = max(-32, min(31, q6_nearest_int(is2 * x[i]))); L[i] = (signed char)(l + 32);
            }
            scale = slx / sl2; best = scale * slx;
        }
    }
    return scale;
}

__global__ void quant_q6k_kernel(const __nv_bfloat16* __restrict__ src, unsigned char* __restrict__ dst, long nblocks) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x; if (b >= nblocks) return;
    const __nv_bfloat16* xb = src + b * 256;
    float x[256];
    for (int i = 0; i < 256; ++i) x[i] = __bfloat162float(xb[i]);
    signed char L[256]; float scales[16];
    float max_scale = 0.f, max_abs = 0.f;
    for (int ib = 0; ib < 16; ++ib) {
        float s = q6_make_qx_quants16(x + 16 * ib, L + 16 * ib);
        scales[ib] = s; float a = fabsf(s);
        if (a > max_abs) { max_abs = a; max_scale = s; }
    }
    unsigned char* blk = dst + b * 210;
    if (max_abs < 1e-30f) { for (int i = 0; i < 210; ++i) blk[i] = 0; return; }
    unsigned char* ql = blk; unsigned char* qh = blk + 128; signed char* sc = (signed char*)(blk + 192);
    float iscale = -128.f / max_scale;
    __half dh = __float2half(1.f / iscale);
    *reinterpret_cast<unsigned short*>(blk + 208) = *reinterpret_cast<unsigned short*>(&dh);
    for (int ib = 0; ib < 16; ++ib) { int si = q6_nearest_int(iscale * scales[ib]); sc[ib] = (signed char)min(127, si); }
    float d_sb = __half2float(dh);
    for (int j = 0; j < 16; ++j) {
        float d = d_sb * sc[j];
        if (d == 0.f) continue;                                   // sub-block dequants to 0; keep make_qx_quants L
        for (int ii = 0; ii < 16; ++ii) {
            int l = max(-32, min(31, q6_nearest_int(x[16 * j + ii] / d))); L[16 * j + ii] = (signed char)(l + 32);
        }
    }
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            unsigned char q1 = L[n + l] & 0xF, q2 = L[n + l + 32] & 0xF, q3 = L[n + l + 64] & 0xF, q4 = L[n + l + 96] & 0xF;
            ql[l]      = q1 | (q3 << 4);
            ql[l + 32] = q2 | (q4 << 4);
            qh[l] = (L[n + l] >> 4) | ((L[n + l + 32] >> 4) << 2) | ((L[n + l + 64] >> 4) << 4) | ((L[n + l + 96] >> 4) << 6);
        }
        ql += 64; qh += 32;
    }
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

void launch_gguf_dequant(int ggml_type, const void* src, void* dst_bf16, long n_values, cudaStream_t stream) {
    auto* d = reinterpret_cast<__nv_bfloat16*>(dst_bf16);
    auto* s = reinterpret_cast<const unsigned char*>(src);
    const int T = 256;
    if (ggml_type == GGML_Q4_K) { long nb = n_values/256; deq_q4k_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == GGML_Q5_K) { long nb = n_values/256; deq_q5k_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == GGML_Q6_K) { long nb = n_values/256; deq_q6k_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == GGML_Q8_0) { long nb = n_values/32;  deq_q8_0_kernel<<<(nb+T-1)/T,T,0,stream>>>(s,d,nb); }
    else if (ggml_type == GGML_F16)  { deq_f16_kernel<<<(n_values+T-1)/T,T,0,stream>>>(s,d,n_values); }
    else /* F32 */                   { deq_f32_kernel<<<(n_values+T-1)/T,T,0,stream>>>(reinterpret_cast<const float*>(src),d,n_values); }
}

// Requantize a bf16 weight tensor to GGUF-native Q6_K (210 B / 256-value superblock). n_values
// must be a multiple of 256. dst holds (n_values/256)*210 bytes.
void launch_gguf_requant_q6k(const void* src_bf16, void* dst_q6k, long n_values, cudaStream_t stream) {
    long nb = n_values / 256;
    const int T = 64;
    quant_q6k_kernel<<<(nb + T - 1) / T, T, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src_bf16),
        reinterpret_cast<unsigned char*>(dst_q6k), nb);
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
