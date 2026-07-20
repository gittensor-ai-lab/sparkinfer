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

// ---------------------------------------------------------------------------
// Fused GGUF -> per-row-int8 dequant for the int8 prefill GEMM. Replaces the
// dequant-to-bf16 + pf_quantize_rows_i8 round trip (write 2B + read 2B per
// value) with one pass that decodes the superblocks twice (rows are L2-hot:
// a 12288-wide Q4_K row is ~7 KB) and writes 1B per value:
//   pass 1: block-wide amax of the exactly-dequantized row
//   pass 2: q = round(v / (amax/127)), matching pf_quantize_rows_i8
// One block (256 threads) per row; thread t decodes value t of each superblock.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float deq_q4k_val(const unsigned char* blk, int t) {
    const float d = gg_h2f(blk), dmin = gg_h2f(blk + 2);
    const unsigned char* sc = blk + 4; const unsigned char* qs = blk + 16;
    const int j64 = t >> 6, r = t & 63, l = r & 31, hi = r >> 5;
    const unsigned char byte = qs[j64 * 32 + l];
    const int nib = hi ? (byte >> 4) : (byte & 0xF);
    int s, m; gg_scale_min_k4(2 * j64 + hi, sc, &s, &m);
    return d * s * nib - dmin * m;
}
__device__ __forceinline__ float deq_q5k_val(const unsigned char* blk, int t) {
    const float d = gg_h2f(blk), dmin = gg_h2f(blk + 2);
    const unsigned char* sc = blk + 4; const unsigned char* qh = blk + 16;
    const unsigned char* ql = blk + 48;
    const int j64 = t >> 6, r = t & 63, l = r & 31, hi = r >> 5;
    const unsigned char byte = ql[j64 * 32 + l];
    const int nib = hi ? (byte >> 4) : (byte & 0xF);
    const int hbit = (qh[l] >> (2 * j64 + hi)) & 1;
    int s, m; gg_scale_min_k4(2 * j64 + hi, sc, &s, &m);
    return d * s * (nib + (hbit ? 16 : 0)) - dmin * m;
}
__device__ __forceinline__ float deq_q6k_val(const unsigned char* blk, int t) {
    const int half = t >> 7, r = t & 127, quad = r >> 5, l = r & 31;
    const unsigned char* ql = blk + half * 64;
    const unsigned char* qh = blk + 128 + half * 32;
    const signed char* sc = (const signed char*)(blk + 192) + half * 8;
    const float d = gg_h2f(blk + 208);
    const int is = l / 16;
    int qv;
    if (quad == 0)      qv = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
    else if (quad == 1) qv = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
    else if (quad == 2) qv = (int)((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
    else                qv = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
    return d * sc[is + 2 * quad] * qv;
}

// Single-pass Q→i8 for cols ≤ 2048 (Qwen3.6 MoE H/mffn). Values live in registers
// (templated NSB) so the int8 rows are bit-identical to the two-pass kernel — same
// dequant, same amax reduce, same roundf — while skipping the second decode pass.
// Wider rows (Qwythos attn K=4096, FFN K=12288) keep two-pass so we never reject the
// i8 path (the previous cols>4096 early-return regressed Qwythos @4k prefill ~10%).
static constexpr int kDeqRowsI8MaxNsb = 8;

template <int QT, int NSB>   // QT: 12=Q4_K, 13=Q5_K, 14=Q6_K; NSB = cols/256
__global__ void deq_rows_i8_kernel(const unsigned char* __restrict__ src,
                                   signed char* __restrict__ q, float* __restrict__ scale,
                                   int cols) {
    constexpr int BS = (QT == 12) ? 144 : (QT == 13) ? 176 : 210;
    const int row = blockIdx.x, t = threadIdx.x;
    const unsigned char* rbase = src + (size_t)row * NSB * BS;

    float vals[NSB];
    float amax = 0.f;
    #pragma unroll
    for (int sb = 0; sb < NSB; sb++) {
        const unsigned char* blk = rbase + (size_t)sb * BS;
        const float v = (QT == 12) ? deq_q4k_val(blk, t)
                      : (QT == 13) ? deq_q5k_val(blk, t) : deq_q6k_val(blk, t);
        vals[sb] = v;
        amax = fmaxf(amax, fabsf(v));
    }
    __shared__ float swarp[8];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((t & 31) == 0) swarp[t >> 5] = amax;
    __syncthreads();
    if (t < 32) {
        float v = (t < 8) ? swarp[t] : 0.f;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (t == 0) swarp[0] = v;
    }
    __syncthreads();
    const float d = swarp[0] / 127.f;
    if (t == 0) scale[row] = d;
    const float inv = (d > 0.f) ? (1.f / d) : 0.f;

    signed char* qrow = q + (size_t)row * cols;
    #pragma unroll
    for (int sb = 0; sb < NSB; sb++)
        qrow[sb * 256 + t] = (signed char)(int)roundf(vals[sb] * inv);
}

// Two-pass fallback for cols > 2048 (Qwythos projections). Same math as above.
template <int QT>
__global__ void deq_rows_i8_twopass_kernel(const unsigned char* __restrict__ src,
                                           signed char* __restrict__ q, float* __restrict__ scale,
                                           int cols) {
    constexpr int BS = (QT == 12) ? 144 : (QT == 13) ? 176 : 210;
    const int row = blockIdx.x, t = threadIdx.x;
    const int nsb = cols >> 8;
    const unsigned char* rbase = src + (size_t)row * nsb * BS;

    float amax = 0.f;
    for (int sb = 0; sb < nsb; sb++) {
        const unsigned char* blk = rbase + (size_t)sb * BS;
        const float v = (QT == 12) ? deq_q4k_val(blk, t)
                      : (QT == 13) ? deq_q5k_val(blk, t) : deq_q6k_val(blk, t);
        amax = fmaxf(amax, fabsf(v));
    }
    __shared__ float swarp[8];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((t & 31) == 0) swarp[t >> 5] = amax;
    __syncthreads();
    if (t < 32) {
        float v = (t < 8) ? swarp[t] : 0.f;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (t == 0) swarp[0] = v;
    }
    __syncthreads();
    const float d = swarp[0] / 127.f;
    if (t == 0) scale[row] = d;
    const float inv = (d > 0.f) ? (1.f / d) : 0.f;

    signed char* qrow = q + (size_t)row * cols;
    for (int sb = 0; sb < nsb; sb++) {
        const unsigned char* blk = rbase + (size_t)sb * BS;
        const float v = (QT == 12) ? deq_q4k_val(blk, t)
                      : (QT == 13) ? deq_q5k_val(blk, t) : deq_q6k_val(blk, t);
        qrow[sb * 256 + t] = (signed char)(int)roundf(v * inv);
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

bool launch_gguf_dequant_rows_i8(int ggml_type, const void* src, signed char* q, float* scale,
                                 int rows, int cols, cudaStream_t stream) {
    if ((cols & 255) != 0) return false;
    auto* s = reinterpret_cast<const unsigned char*>(src);
    const int nsb = cols >> 8;
    if (nsb > 0 && nsb <= kDeqRowsI8MaxNsb) {
        // Explicit NSB instantiations keep vals[] in registers (bit-same as two-pass).
        #define SI_LAUNCH(QT, NSB) deq_rows_i8_kernel<QT, NSB><<<rows, 256, 0, stream>>>(s, q, scale, cols)
        #define SI_BY_NSB(QT) \
            switch (nsb) { \
                case 1: SI_LAUNCH(QT, 1); break; \
                case 2: SI_LAUNCH(QT, 2); break; \
                case 3: SI_LAUNCH(QT, 3); break; \
                case 4: SI_LAUNCH(QT, 4); break; \
                case 5: SI_LAUNCH(QT, 5); break; \
                case 6: SI_LAUNCH(QT, 6); break; \
                case 7: SI_LAUNCH(QT, 7); break; \
                default: SI_LAUNCH(QT, 8); break; \
            }
        if (ggml_type == GGML_Q4_K)      { SI_BY_NSB(12); }
        else if (ggml_type == GGML_Q5_K) { SI_BY_NSB(13); }
        else if (ggml_type == GGML_Q6_K) { SI_BY_NSB(14); }
        else return false;
        #undef SI_BY_NSB
        #undef SI_LAUNCH
    } else {
        // Qwythos attn/FFN: keep two-pass so i8 GEMM still engages (do not return false).
        if (ggml_type == GGML_Q4_K)      deq_rows_i8_twopass_kernel<12><<<rows, 256, 0, stream>>>(s, q, scale, cols);
        else if (ggml_type == GGML_Q5_K) deq_rows_i8_twopass_kernel<13><<<rows, 256, 0, stream>>>(s, q, scale, cols);
        else if (ggml_type == GGML_Q6_K) deq_rows_i8_twopass_kernel<14><<<rows, 256, 0, stream>>>(s, q, scale, cols);
        else return false;
    }
    return true;
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
