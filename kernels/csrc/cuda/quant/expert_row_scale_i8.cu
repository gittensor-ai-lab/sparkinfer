// Load-time int8 row scales for the MoE expert stacks + the per-pass quantizer that consumes them.
// See expert_row_scale_i8.h for why this is split out of the fused dequant.
//
// Portable CUDA — runs on sm_89 .. sm_120/sm_121 (RTX 5090 / PRO 6000 / Spark).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "sparkinfer/kernels/expert_row_scale_i8.h"

namespace sparkinfer {
namespace kernels {
namespace {

enum { ERS_Q4_K = 12, ERS_Q5_K = 13, ERS_Q6_K = 14 };

// Block decoders, kept identical in value to dequant_gguf.cu's (that equality is the whole point:
// it is what makes the split scale bit-identical to the fused one). They are re-stated here rather
// than shared through a header so this path stays a self-contained translation unit.
__device__ __forceinline__ float ers_h2f(const unsigned char* p) {
    __half h; *((unsigned short*)&h) = *(const unsigned short*)p; return __half2float(h);
}

// Sub-block j of a Q4_K/Q5_K super-block, resolved all the way to the affine pair the callers
// actually want: value = a * quant - b. Folding the super-block scales in here (rather than handing
// back the raw 6-bit packed s/m) keeps the float op order identical to the fused kernel's
// `d * s * nib - dmin * m`, which is what the bit-identity guarantee rests on.
__device__ __forceinline__ void ers_affine(const unsigned char* blk, int j, float* a, float* b) {
    const unsigned char* p = blk + 4;
    int s, m;
    if (j < 4) { s = p[j] & 63; m = p[j + 4] & 63; }
    else {
        s = (p[j + 4] & 0xF) | ((p[j - 4] >> 6) << 4);
        m = (p[j + 4] >> 4)  | ((p[j]     >> 6) << 4);
    }
    *a = ers_h2f(blk) * s;
    *b = ers_h2f(blk + 2) * m;
}

__device__ __forceinline__ float ers_q4k_val(const unsigned char* blk, int t) {
    const int j64 = t >> 6, r = t & 63, l = r & 31, hi = r >> 5;
    const unsigned char byte = blk[16 + j64 * 32 + l];
    float a, b; ers_affine(blk, 2 * j64 + hi, &a, &b);
    return a * (hi ? (byte >> 4) : (byte & 0xF)) - b;
}

__device__ __forceinline__ float ers_q5k_val(const unsigned char* blk, int t) {
    const int j64 = t >> 6, r = t & 63, l = r & 31, hi = r >> 5;
    const int sub = 2 * j64 + hi;
    const unsigned char byte = blk[48 + j64 * 32 + l];
    const int nib = hi ? (byte >> 4) : (byte & 0xF);
    const int high = (blk[16 + l] >> sub) & 1;          // 5th bit lives in the qh plane
    float a, b; ers_affine(blk, sub, &a, &b);
    return a * (nib + (high ? 16 : 0)) - b;
}

__device__ __forceinline__ float ers_q6k_val(const unsigned char* blk, int t) {
    const int half = t >> 7, r = t & 127, quad = r >> 5, l = r & 31;
    const unsigned char* ql = blk + half * 64;
    const unsigned char* qh = blk + 128 + half * 32;
    const signed char* sc = (const signed char*)(blk + 192) + half * 8;
    const float d = ers_h2f(blk + 208);
    const int is = l / 16;
    int qv;
    if (quad == 0)      qv = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
    else if (quad == 1) qv = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
    else if (quad == 2) qv = (int)((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
    else                qv = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
    return d * sc[is + 2 * quad] * qv;
}

template <int QT>
__device__ __forceinline__ float ers_val(const unsigned char* blk, int t) {
    return (QT == ERS_Q4_K) ? ers_q4k_val(blk, t)
         : (QT == ERS_Q5_K) ? ers_q5k_val(blk, t) : ers_q6k_val(blk, t);
}

template <int QT>
__device__ __forceinline__ const unsigned char* ers_row(const unsigned char* src, int row, int nsb) {
    constexpr int BS = (QT == ERS_Q4_K) ? 144 : (QT == ERS_Q5_K) ? 176 : 210;
    return src + (size_t)row * nsb * BS;
}

// One block (256 threads) per row; thread t handles value t of each super-block, so a full row is
// covered in cols/256 steps. Load-time only, so plain scalar accesses are fine here.
template <int QT>
__global__ void ers_scale_kernel(const unsigned char* __restrict__ src,
                                 float* __restrict__ scale, int cols) {
    const int row = blockIdx.x, t = threadIdx.x, nsb = cols >> 8;
    constexpr int BS = (QT == ERS_Q4_K) ? 144 : (QT == ERS_Q5_K) ? 176 : 210;
    const unsigned char* rbase = ers_row<QT>(src, row, nsb);

    float peak = 0.f;
    for (int sb = 0; sb < nsb; sb++)
        peak = fmaxf(peak, fabsf(ers_val<QT>(rbase + (size_t)sb * BS, t)));

    // Block-wide max. fmaxf is exact and order-independent, so any reduction order yields the same
    // scale as the fused kernel's; this runs once at load, so take the plain tree over shared memory
    // rather than a tuned shuffle ladder.
    __shared__ float red[256];
    red[t] = peak;
    __syncthreads();
    for (int half = 128; half > 0; half >>= 1) {
        if (t < half) red[t] = fmaxf(red[t], red[t + half]);
        __syncthreads();
    }
    if (t == 0) scale[row] = red[0] / 127.f;
}

// The per-pass half: the row's scale is already known, so this is a single decode pass.
template <int QT>
__global__ void ers_quant_kernel(const unsigned char* __restrict__ src,
                                 signed char* __restrict__ q,
                                 const float* __restrict__ scale, int cols) {
    const int row = blockIdx.x, t = threadIdx.x, nsb = cols >> 8;
    constexpr int BS = (QT == ERS_Q4_K) ? 144 : (QT == ERS_Q5_K) ? 176 : 210;
    const unsigned char* rbase = ers_row<QT>(src, row, nsb);
    const float d = scale[row];
    const float inv = (d > 0.f) ? (1.f / d) : 0.f;

    signed char* qrow = q + (size_t)row * cols;
    for (int sb = 0; sb < nsb; sb++)
        qrow[sb * 256 + t] = (signed char)(int)roundf(ers_val<QT>(rbase + (size_t)sb * BS, t) * inv);
}

} // namespace

bool launch_expert_row_scales_i8(int ggml_type, const void* src, float* scale,
                                 int rows, int cols, cudaStream_t stream) {
    if ((cols & 255) != 0 || rows <= 0) return false;
    auto* s = reinterpret_cast<const unsigned char*>(src);
    switch (ggml_type) {
        case ERS_Q4_K: ers_scale_kernel<ERS_Q4_K><<<rows, 256, 0, stream>>>(s, scale, cols); return true;
        case ERS_Q5_K: ers_scale_kernel<ERS_Q5_K><<<rows, 256, 0, stream>>>(s, scale, cols); return true;
        case ERS_Q6_K: ers_scale_kernel<ERS_Q6_K><<<rows, 256, 0, stream>>>(s, scale, cols); return true;
        default: return false;
    }
}

bool launch_expert_rows_i8_scaled(int ggml_type, const void* src, signed char* q,
                                  const float* scale, int rows, int cols, cudaStream_t stream) {
    if ((cols & 255) != 0 || rows <= 0) return false;
    auto* s = reinterpret_cast<const unsigned char*>(src);
    switch (ggml_type) {
        case ERS_Q4_K: ers_quant_kernel<ERS_Q4_K><<<rows, 256, 0, stream>>>(s, q, scale, cols); return true;
        case ERS_Q5_K: ers_quant_kernel<ERS_Q5_K><<<rows, 256, 0, stream>>>(s, q, scale, cols); return true;
        case ERS_Q6_K: ers_quant_kernel<ERS_Q6_K><<<rows, 256, 0, stream>>>(s, q, scale, cols); return true;
        default: return false;
    }
}

} // namespace kernels
} // namespace sparkinfer
