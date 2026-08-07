// Row-batched DFlash verify kernels: GDN conv front-end + the fused norms.
//
// The verify loop forwards one target token per emitted token, so a window of R proposals
// streams the whole model R times. These kernels consume the window's R activation rows in
// one launch. Nothing in this file owns a weight matrix worth amortizing (the norm weight is
// a single row), so the cheap kernels take rows as an extra GRID dimension: each block still
// runs the single-row kernel verbatim on its own row, which is why rows=R is bit-identical to
// R single-row calls by construction. The depthwise conv cannot do that — it reads and then
// shifts conv_state, so token r depends on the state token r-1 left behind — and instead
// carries a compile-time-length sequential loop over the window inside the block.
//
// EXACTNESS: the DFlash gate asserts speculative output equals autoregressive output token
// for token, so every accumulation order, reduction tree and rounding step below is copied
// verbatim from conv_split_l2norm_fused_kernel / gated_norm_q8_warp_kernel /
// add_rmsnorm{2,3}_q8_kernel / rmsnorm_kernel. Batching only chooses which rows share a CTA.
//
// Self-contained by house convention: the small device helpers are duplicated here rather
// than shared through a header (as expert_ffn_q4k.cu duplicates gemv.cu's Q4_K decoder).
//
// Portable CUDA — sm_89 .. sm_120.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
// At file scope, not inside the namespace: the launchers below must attach to the
// declarations in this header (and to kDFlashMaxRows), not to nested copies of them.
#include "sparkinfer/kernels/dflash_rows.h"
#include <cassert>
#endif

namespace sparkinfer {
namespace kernels {

// ---- device helpers (copies of the qwen36.cu / rmsnorm.cu originals) --------

__device__ __forceinline__ float dfr_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float dfr_silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float dfr_wsum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float dfr_warp_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// 128-bit (uint4 = 8 bf16) coalesced access helpers for the bf16 row scan.
__device__ __forceinline__ void dfr_unpack8(const uint4& p, float out[8]) {
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&p);
    #pragma unroll
    for (int j = 0; j < 8; j++) out[j] = __bfloat162float(h[j]);
}
__device__ __forceinline__ uint4 dfr_pack8(const float in[8]) {
    uint4 p; __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&p);
    #pragma unroll
    for (int j = 0; j < 8; j++) h[j] = __float2bfloat16(in[j]);
    return p;
}

// llama mmvq activation block (matches si_block_q8_1 used by the int8 GEMVs/MMVQ).
struct dfr_blk_q8_1 { __half2 ds; signed char qs[32]; };

// ---- conv + split + L2(q,k) over a window of R tokens -----------------------
// Mirrors conv_split_l2norm_fused_kernel with the SAME grid — one block per head, one
// thread per channel — and walks the window inside the block. The rows CANNOT be spread
// over blocks: the causal conv reads conv_state and then shifts token r into it, so token
// r+1's convolution consumes what token r wrote. Keeping the window inside one block makes
// that dependency an ordinary sequential loop; each channel d is owned by exactly one
// thread in exactly one block, so the state shift needs no cross-block ordering.
//
// The block-wide L2 reduction forces two structural details: the per-token __syncthreads()
// pair must be executed by EVERY thread (hence the out-of-range early return becomes the
// `active` predicate — a thread that returned would starve the barriers and hang the
// block), and sw[] is reused across tokens, so the loop closes with one more barrier
// before the next token overwrites it. Barriers move no arithmetic; each token's operand
// order, reduction tree and rounding are exactly the single-token kernel's.
template <int R>
__global__ void dflash_rows_conv_split_l2norm_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ conv_w,
    __nv_bfloat16* __restrict__ conv_state,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    int q_heads, int v_heads, int head_dim, int conv_kernel, float eps)
{
    const int h  = blockIdx.x;
    const int t  = threadIdx.x;
    const int q_dim = q_heads * head_dim;
    const int v_dim = v_heads * head_dim;
    const int qkv_dim = 2 * q_dim + v_dim;

    bool do_norm = false;
    int d;
    __nv_bfloat16* out;
    if (h < q_heads) {
        d = h * head_dim + t;  out = q + d;           do_norm = true;
    } else if (h < 2 * q_heads) {
        d = q_dim + (h - q_heads) * head_dim + t;  out = k + d - q_dim;  do_norm = true;
    } else {
        d = 2 * q_dim + (h - 2 * q_heads) * head_dim + t;  out = v + d - 2 * q_dim;
    }
    const bool active = (d < qkv_dim);
    // q and k stack q_heads heads per token, v stacks v_heads.
    const int out_stride = do_norm ? q_dim : v_dim;

    __shared__ float sw[32];

    for (int r = 0; r < R; r++) {
        const __nv_bfloat16* qkv_r = qkv + (size_t)r * qkv_dim;

        // 1D conv + SiLU
        float y = 0.f;
        if (active) {
            for (int p = 0; p < conv_kernel - 1; p++)
                y += dfr_to_f(conv_state[(size_t)p * qkv_dim + d]) *
                     dfr_to_f(conv_w[(size_t)d * conv_kernel + p]);
            y += dfr_to_f(qkv_r[d]) * dfr_to_f(conv_w[(size_t)d * conv_kernel + (conv_kernel - 1)]);

            for (int p = 0; p < conv_kernel - 2; p++)
                conv_state[(size_t)p * qkv_dim + d] = conv_state[(size_t)(p + 1) * qkv_dim + d];
            if (conv_kernel > 1)
                conv_state[(size_t)(conv_kernel - 2) * qkv_dim + d] = qkv_r[d];
        }

        const float oy = dfr_silu(y);

        if (do_norm) {
            const float ss = dfr_wsum(oy * oy);
            if ((t & 31) == 0) sw[t >> 5] = ss;
            __syncthreads();
            if (t < 32) {
                float vv = (t < (head_dim + 31) / 32) ? sw[t] : 0.f;
                vv = dfr_wsum(vv);
                if (t == 0) sw[0] = rsqrtf(vv + eps);
            }
            __syncthreads();
            if (active) out[(size_t)r * out_stride] = __float2bfloat16(oy * sw[0]);
        } else {
            if (active) out[(size_t)r * out_stride] = __float2bfloat16(oy);
        }
        // Retire this token's readers of sw[0] before the next token rewrites sw[].
        __syncthreads();
    }
}

#ifndef _MSC_VER
template __global__ void dflash_rows_conv_split_l2norm_kernel<1>(const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, float);
template __global__ void dflash_rows_conv_split_l2norm_kernel<2>(const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, float);
template __global__ void dflash_rows_conv_split_l2norm_kernel<3>(const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, float);
template __global__ void dflash_rows_conv_split_l2norm_kernel<4>(const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, float);
template __global__ void dflash_rows_conv_split_l2norm_kernel<5>(const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, float);
template __global__ void dflash_rows_conv_split_l2norm_kernel<6>(const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, float);
template __global__ void dflash_rows_conv_split_l2norm_kernel<7>(const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, float);
template __global__ void dflash_rows_conv_split_l2norm_kernel<8>(const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, float);
#endif

// ---- gated norm -> Q8_1, one warp per (v-head, row) -------------------------
// Mirrors gated_norm_q8_warp_kernel. grid.y is the window index: the whole reduction is
// warp-local, so a row is just a base-pointer offset and the per-row math is untouched.
// The norm weight is shared by every row and is NOT offset.
template <int HEAD_DIM>
__global__ void dflash_rows_gated_norm_q8_kernel(const __nv_bfloat16* __restrict__ x,
                                                 const __nv_bfloat16* __restrict__ z,
                                                 const __nv_bfloat16* __restrict__ weight,
                                                 dfr_blk_q8_1* __restrict__ out_q8,
                                                 int v_heads, float eps) {
    constexpr int NROW = HEAD_DIM / 32;
    const int h = blockIdx.x;
    const int row = blockIdx.y;
    const int lane = threadIdx.x & 31;
    if (h >= v_heads) return;
    const size_t base = (size_t)(row * v_heads + h) * HEAD_DIM;
    const int qbase = (row * v_heads + h) * NROW;
    float ss = 0.f;
    float xv[NROW], zv[NROW], wv[NROW];
    #pragma unroll
    for (int r = 0; r < NROW; r++) {
        const int t = lane + r * 32;
        xv[r] = dfr_to_f(x[base + t]);
        zv[r] = dfr_to_f(z[base + t]);
        wv[r] = dfr_to_f(weight[t]);
        ss += xv[r] * xv[r];
    }
    const float inv = rsqrtf(dfr_wsum(ss) / HEAD_DIM + eps);
    #pragma unroll
    for (int r = 0; r < NROW; r++) {
        const float y = xv[r] * inv * wv[r] * dfr_silu(zv[r]);
        const float bv = __bfloat162float(__float2bfloat16(y));
        float amax = fabsf(bv);
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
        const float d = amax / 127.0f;
        const int qi = (amax == 0.0f) ? 0 : (int)roundf(bv / d);
        out_q8[qbase + r].qs[lane] = (signed char)qi;
        int s = qi;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
        if (lane == 0) out_q8[qbase + r].ds = __floats2half2_rn(d, d * (float)s);
    }
}

#ifndef _MSC_VER
template __global__ void dflash_rows_gated_norm_q8_kernel<128>(const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, dfr_blk_q8_1*, int, float);
#endif

// ---- residual add + RMSNorm + Q8_1(out_norm), one block per row -------------
// Mirrors add_rmsnorm2_q8_kernel, whose grid was a single block. blockIdx.x is now the
// window index; every pointer but `weight` is offset by the row. Still one 8-wide pack per
// thread, so a Q8_1 block is 4 consecutive threads inside one warp.
__global__ void dflash_rows_add_rmsnorm2_q8_kernel(const __nv_bfloat16* __restrict__ x,
                                                   const __nv_bfloat16* __restrict__ residual,
                                                   const __nv_bfloat16* __restrict__ weight,
                                                   __nv_bfloat16* __restrict__ out_sum,
                                                   __nv_bfloat16* __restrict__ out_norm,
                                                   dfr_blk_q8_1* __restrict__ out_q8,
                                                   int cols, float eps) {
    const int row = blockIdx.x;
    const size_t base = (size_t)row * cols;
    __shared__ float s_warp[32];
    const int t = threadIdx.x;
    const uint4* x4 = reinterpret_cast<const uint4*>(x + base);
    const uint4* r4 = reinterpret_cast<const uint4*>(residual + base);
    uint4* osum4 = reinterpret_cast<uint4*>(out_sum + base);
    dfr_blk_q8_1* q8 = out_q8 + (size_t)row * (cols >> 5);

    float xv[8], rv[8]; dfr_unpack8(__ldg(x4 + t), xv); dfr_unpack8(__ldg(r4 + t), rv);
    float sv[8]; float ss = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; j++) { sv[j] = xv[j] + rv[j]; ss = __fmaf_rn(sv[j], sv[j], ss); }
    osum4[t] = dfr_pack8(sv);
    ss = dfr_warp_sum(ss);
    if ((t & 31) == 0) s_warp[t >> 5] = ss;
    __syncthreads();
    if (t < 32) {
        float v = (t < (blockDim.x + 31) / 32) ? s_warp[t] : 0.f;
        v = dfr_warp_sum(v);
        if (t == 0) s_warp[0] = rsqrtf(v / cols + eps);
    }
    __syncthreads();
    const float inv_rms = s_warp[0];

    // Re-read the bf16-rounded residual sum (exactly as add_rmsnorm2_kernel does), so out_norm
    // is bit-identical to the unfused path; then derive Q8_1 from the bf16-rounded out_norm.
    const uint4* w4 = reinterpret_cast<const uint4*>(weight);
    const uint4* osum4r = reinterpret_cast<const uint4*>(out_sum + base);
    float svb[8], wv[8]; dfr_unpack8(__ldg(osum4r + t), svb); dfr_unpack8(__ldg(w4 + t), wv);
    float ov[8], bv[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) { ov[j] = svb[j] * inv_rms * wv[j]; bv[j] = __bfloat162float(__float2bfloat16(ov[j])); }
    reinterpret_cast<uint4*>(out_norm + base)[t] = dfr_pack8(ov);

    // Q8_1 of the bf16-rounded out_norm. 32-block = 4 consecutive threads (t&~3 .. +3).
    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(bv[j]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float d = amax / 127.0f;
    const int b = t >> 2, r = t & 3;
    int s = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int qi = (amax == 0.0f) ? 0 : (int)roundf(bv[j] / d);
        q8[b].qs[r * 8 + j] = (signed char)qi; s += qi;
    }
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2);
    if (r == 0) q8[b].ds = __floats2half2_rn(d, d * (float)s);
}

// 3-input form: out_sum = x + (res1 + res2), the inner sum rounded to bf16 first.
__global__ void dflash_rows_add_rmsnorm3_q8_kernel(const __nv_bfloat16* __restrict__ x,
                                                   const __nv_bfloat16* __restrict__ res1,
                                                   const __nv_bfloat16* __restrict__ res2,
                                                   const __nv_bfloat16* __restrict__ weight,
                                                   __nv_bfloat16* __restrict__ out_sum,
                                                   __nv_bfloat16* __restrict__ out_norm,
                                                   dfr_blk_q8_1* __restrict__ out_q8,
                                                   int cols, float eps) {
    const int row = blockIdx.x;
    const size_t base = (size_t)row * cols;
    __shared__ float s_warp[32];
    const int t = threadIdx.x;
    const uint4* x4  = reinterpret_cast<const uint4*>(x + base);
    const uint4* r14 = reinterpret_cast<const uint4*>(res1 + base);
    const uint4* r24 = reinterpret_cast<const uint4*>(res2 + base);
    uint4* osum4 = reinterpret_cast<uint4*>(out_sum + base);
    dfr_blk_q8_1* q8 = out_q8 + (size_t)row * (cols >> 5);

    float xv[8], r1v[8], r2v[8];
    dfr_unpack8(__ldg(x4 + t), xv); dfr_unpack8(__ldg(r14 + t), r1v); dfr_unpack8(__ldg(r24 + t), r2v);
    float sv[8]; float ss = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        float rs = __bfloat162float(__float2bfloat16(r1v[j] + r2v[j]));
        sv[j] = xv[j] + rs;
        ss = __fmaf_rn(sv[j], sv[j], ss);
    }
    osum4[t] = dfr_pack8(sv);
    ss = dfr_warp_sum(ss);
    if ((t & 31) == 0) s_warp[t >> 5] = ss;
    __syncthreads();
    if (t < 32) {
        float v = (t < (blockDim.x + 31) / 32) ? s_warp[t] : 0.f;
        v = dfr_warp_sum(v);
        if (t == 0) s_warp[0] = rsqrtf(v / cols + eps);
    }
    __syncthreads();
    const float inv_rms = s_warp[0];

    const uint4* w4 = reinterpret_cast<const uint4*>(weight);
    const uint4* osum4r = reinterpret_cast<const uint4*>(out_sum + base);
    float svb[8], wv[8]; dfr_unpack8(__ldg(osum4r + t), svb); dfr_unpack8(__ldg(w4 + t), wv);
    float ov[8], bv[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) { ov[j] = svb[j] * inv_rms * wv[j]; bv[j] = __bfloat162float(__float2bfloat16(ov[j])); }
    reinterpret_cast<uint4*>(out_norm + base)[t] = dfr_pack8(ov);

    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(bv[j]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float d = amax / 127.0f;
    const int b = t >> 2, r = t & 3;
    int s = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int qi = (amax == 0.0f) ? 0 : (int)roundf(bv[j] / d);
        q8[b].qs[r * 8 + j] = (signed char)qi; s += qi;
    }
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2);
    if (r == 0) q8[b].ds = __floats2half2_rn(d, d * (float)s);
}

// ---- plain RMSNorm ----------------------------------------------------------
// rmsnorm_kernel<0> with the residual branches constant-folded out (the DFlash prime norm
// at layer 0 has no residual). Already one block per row, so the window needs nothing but a
// larger grid.
__global__ void dflash_rows_rmsnorm_kernel(const __nv_bfloat16* __restrict__ x,
                                           const __nv_bfloat16* __restrict__ weight,
                                           __nv_bfloat16* __restrict__ out,
                                           int rows, int cols, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const size_t base = (size_t)row * cols;
    __shared__ float s_warp[32];

    const int npack = cols >> 3;   // cols / 8 (RMSNorm widths here are multiples of 8)
    const int tail  = npack << 3;  // first scalar column (handles cols % 8 != 0)
    const uint4* x4 = reinterpret_cast<const uint4*>(x + base);

    float ss = 0.f;
    for (int p = threadIdx.x; p < npack; p += blockDim.x) {
        float xv[8]; dfr_unpack8(__ldg(x4 + p), xv);
        #pragma unroll
        for (int j = 0; j < 8; j++) ss = __fmaf_rn(xv[j], xv[j], ss);
    }
    for (int c = tail + threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(x[base + c]);
        ss = __fmaf_rn(v, v, ss);
    }
    ss = dfr_warp_sum(ss);
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
        v = dfr_warp_sum(v);
        if (threadIdx.x == 0) s_warp[0] = rsqrtf(v / cols + eps);
    }
    __syncthreads();
    const float inv_rms = s_warp[0];

    const uint4* w4 = reinterpret_cast<const uint4*>(weight);
    uint4* o4 = reinterpret_cast<uint4*>(out + base);
    for (int p = threadIdx.x; p < npack; p += blockDim.x) {
        float xv[8]; dfr_unpack8(__ldg(x4 + p), xv);
        float wv[8]; dfr_unpack8(__ldg(w4 + p), wv);
        float ov[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) ov[j] = xv[j] * inv_rms * wv[j];
        o4[p] = dfr_pack8(ov);
    }
    for (int c = tail + threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(x[base + c]);
        out[base + c] = __float2bfloat16(v * inv_rms * __bfloat162float(weight[c]));
    }
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
void launch_dflash_rows_conv_split_l2norm(const void* qkv, const void* conv_w, void* conv_state,
                                          void* q, void* k, void* v,
                                          int q_heads, int v_heads, int head_dim,
                                          int conv_kernel, float eps, int rows,
                                          cudaStream_t stream) {
    if (rows < 1 || rows > kDFlashMaxRows) return;
    const int blocks = 2 * q_heads + v_heads;
#define SI_DFLASH_ROWS_CONV(R)                                                  \
    dflash_rows_conv_split_l2norm_kernel<R><<<blocks, head_dim, 0, stream>>>(   \
        reinterpret_cast<const __nv_bfloat16*>(qkv),                            \
        reinterpret_cast<const __nv_bfloat16*>(conv_w),                         \
        reinterpret_cast<__nv_bfloat16*>(conv_state),                           \
        reinterpret_cast<__nv_bfloat16*>(q),                                    \
        reinterpret_cast<__nv_bfloat16*>(k),                                    \
        reinterpret_cast<__nv_bfloat16*>(v),                                    \
        q_heads, v_heads, head_dim, conv_kernel, eps)
    switch (rows) {
        case 1: SI_DFLASH_ROWS_CONV(1); break;
        case 2: SI_DFLASH_ROWS_CONV(2); break;
        case 3: SI_DFLASH_ROWS_CONV(3); break;
        case 4: SI_DFLASH_ROWS_CONV(4); break;
        case 5: SI_DFLASH_ROWS_CONV(5); break;
        case 6: SI_DFLASH_ROWS_CONV(6); break;
        case 7: SI_DFLASH_ROWS_CONV(7); break;
        case 8: SI_DFLASH_ROWS_CONV(8); break;
        default: break;
    }
#undef SI_DFLASH_ROWS_CONV
}

void launch_dflash_rows_gated_norm_q8(const void* x, const void* z, const void* weight,
                                      void* out_q8, int v_heads, int head_dim, float eps,
                                      int rows, cudaStream_t stream) {
    (void)head_dim;
    if (rows < 1 || rows > kDFlashMaxRows) return;
    dflash_rows_gated_norm_q8_kernel<128><<<dim3(v_heads, rows), 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(z),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<dfr_blk_q8_1*>(out_q8),
        v_heads, eps);
}

// One 8-wide pack per thread and a Q8_1 block = 4 consecutive threads: requires
// cols % 256 == 0 (every warp full, each 32-block within one warp) and cols/8 <= 1024
// (max block dim). The decode residual width (hidden = 2048) satisfies both.
void launch_dflash_rows_add_rmsnorm2_q8(const void* x, const void* residual, const void* weight,
                                        void* out_sum, void* out_norm, void* out_q8,
                                        int cols, float eps, int rows,
                                        cudaStream_t stream) {
    if (rows < 1 || rows > kDFlashMaxRows) return;
    assert(cols % 256 == 0 && (cols >> 3) <= 1024);
    dflash_rows_add_rmsnorm2_q8_kernel<<<rows, cols >> 3, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out_sum),
        reinterpret_cast<__nv_bfloat16*>(out_norm),
        reinterpret_cast<dfr_blk_q8_1*>(out_q8), cols, eps);
}

void launch_dflash_rows_add_rmsnorm3_q8(const void* x, const void* res1, const void* res2,
                                        const void* weight, void* out_sum, void* out_norm,
                                        void* out_q8, int cols, float eps, int rows,
                                        cudaStream_t stream) {
    if (rows < 1 || rows > kDFlashMaxRows) return;
    assert(cols % 256 == 0 && (cols >> 3) <= 1024);
    dflash_rows_add_rmsnorm3_q8_kernel<<<rows, cols >> 3, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(res1),
        reinterpret_cast<const __nv_bfloat16*>(res2),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out_sum),
        reinterpret_cast<__nv_bfloat16*>(out_norm),
        reinterpret_cast<dfr_blk_q8_1*>(out_q8), cols, eps);
}

void launch_dflash_rows_rmsnorm(const void* x, const void* weight, void* out,
                                int cols, float eps, int rows, cudaStream_t stream) {
    if (rows < 1 || rows > kDFlashMaxRows) return;
    dflash_rows_rmsnorm_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out), rows, cols, eps);
}
#endif

} // namespace kernels
} // namespace sparkinfer
