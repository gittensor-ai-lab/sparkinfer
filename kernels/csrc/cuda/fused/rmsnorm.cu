// Fused RMSNorm (+ optional residual add). One block per row; block-reduces the
// sum of squares, then writes the normalized, weighted row. A small CODA-style
// epilogue building block kept on the portable CUDA path.
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

// llama mmvq activation block (matches si_block_q8_1 used by the int8 GEMVs/MMVQ).
struct si_blk_q8_1 { __half2 ds; signed char qs[32]; };

__device__ __forceinline__ float rn_warp_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// 128-bit (uint4 = 8 bf16) coalesced access helpers for the bf16 row scan. The
// hidden/ffn widths on this path are multiples of 256, so cols % 8 == 0 and each
// thread consumes whole 8-wide packs; the column loop issues 8x fewer load/store
// instructions than the scalar path. Math is unchanged (same per-element FMA).
__device__ __forceinline__ void rn_unpack8(const uint4& p, float out[8]) {
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&p);
    #pragma unroll
    for (int j = 0; j < 8; j++) out[j] = __bfloat162float(h[j]);
}
__device__ __forceinline__ uint4 rn_pack8(const float in[8]) {
    uint4 p; __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&p);
    #pragma unroll
    for (int j = 0; j < 8; j++) h[j] = __float2bfloat16(in[j]);
    return p;
}

template <int ADD_RESIDUAL>
__global__ void rmsnorm_kernel(const __nv_bfloat16* __restrict__ x,
                               const __nv_bfloat16* __restrict__ residual,
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
    const uint4* r4 = ADD_RESIDUAL ? reinterpret_cast<const uint4*>(residual + base) : nullptr;

    float ss = 0.f;
    for (int p = threadIdx.x; p < npack; p += blockDim.x) {
        float xv[8]; rn_unpack8(__ldg(x4 + p), xv);
        if (ADD_RESIDUAL) {
            float rv[8]; rn_unpack8(__ldg(r4 + p), rv);
            #pragma unroll
            for (int j = 0; j < 8; j++) xv[j] += rv[j];
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) ss = __fmaf_rn(xv[j], xv[j], ss);
    }
    for (int c = tail + threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(x[base + c]);
        if (ADD_RESIDUAL) v += __bfloat162float(residual[base + c]);
        ss = __fmaf_rn(v, v, ss);
    }
    ss = rn_warp_sum(ss);
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
        v = rn_warp_sum(v);
        if (threadIdx.x == 0) s_warp[0] = rsqrtf(v / cols + eps);
    }
    __syncthreads();
    const float inv_rms = s_warp[0];

    const uint4* w4 = reinterpret_cast<const uint4*>(weight);
    uint4* o4 = reinterpret_cast<uint4*>(out + base);
    for (int p = threadIdx.x; p < npack; p += blockDim.x) {
        float xv[8]; rn_unpack8(__ldg(x4 + p), xv);
        if (ADD_RESIDUAL) {
            float rv[8]; rn_unpack8(__ldg(r4 + p), rv);
            #pragma unroll
            for (int j = 0; j < 8; j++) xv[j] += rv[j];
        }
        float wv[8]; rn_unpack8(__ldg(w4 + p), wv);
        float ov[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) ov[j] = xv[j] * inv_rms * wv[j];
        o4[p] = rn_pack8(ov);
    }
    for (int c = tail + threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(x[base + c]);
        if (ADD_RESIDUAL) v += __bfloat162float(residual[base + c]);
        out[base + c] = __float2bfloat16(v * inv_rms * __bfloat162float(weight[c]));
    }
}

template __global__ void rmsnorm_kernel<0>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, float);
template __global__ void rmsnorm_kernel<1>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, float);

// Fused residual + RMSNorm that ALSO emits the residual sum:
//   sum = x + residual;  norm = (sum / rms(sum)) * weight
// One kernel replaces a residual_add + a rmsnorm (and keeps `sum` for the next
// residual), cutting the per-layer norm/residual kernel count from 4 to 2.
__global__ void add_rmsnorm2_kernel(const __nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ residual,
                                    const __nv_bfloat16* __restrict__ weight,
                                    __nv_bfloat16* __restrict__ out_sum,
                                    __nv_bfloat16* __restrict__ out_norm,
                                    int rows, int cols, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const size_t base = (size_t)row * cols;
    __shared__ float s_warp[32];

    const int npack = cols >> 3;   // cols / 8 (RMSNorm widths here are multiples of 8)
    const int tail  = npack << 3;  // first scalar column (handles cols % 8 != 0)
    const uint4* x4 = reinterpret_cast<const uint4*>(x + base);
    const uint4* r4 = reinterpret_cast<const uint4*>(residual + base);
    uint4* osum4 = reinterpret_cast<uint4*>(out_sum + base);

    float ss = 0.f;
    for (int p = threadIdx.x; p < npack; p += blockDim.x) {
        float xv[8], rv[8]; rn_unpack8(__ldg(x4 + p), xv); rn_unpack8(__ldg(r4 + p), rv);
        float sv[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) sv[j] = xv[j] + rv[j];
        osum4[p] = rn_pack8(sv);
        // ss accumulates on the fp32 sum (matches the original ss += v*v).
        #pragma unroll
        for (int j = 0; j < 8; j++) ss = __fmaf_rn(sv[j], sv[j], ss);
    }
    for (int c = tail + threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(x[base + c]) + __bfloat162float(residual[base + c]);
        out_sum[base + c] = __float2bfloat16(v);
        ss = __fmaf_rn(v, v, ss);
    }
    ss = rn_warp_sum(ss);
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
        v = rn_warp_sum(v);
        if (threadIdx.x == 0) s_warp[0] = rsqrtf(v / cols + eps);
    }
    __syncthreads();
    const float inv_rms = s_warp[0];

    const uint4* w4 = reinterpret_cast<const uint4*>(weight);
    const uint4* osum4r = reinterpret_cast<const uint4*>(out_sum + base);
    uint4* onorm4 = reinterpret_cast<uint4*>(out_norm + base);
    for (int p = threadIdx.x; p < npack; p += blockDim.x) {
        float sv[8], wv[8]; rn_unpack8(__ldg(osum4r + p), sv); rn_unpack8(__ldg(w4 + p), wv);
        float ov[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) ov[j] = sv[j] * inv_rms * wv[j];
        onorm4[p] = rn_pack8(ov);
    }
    for (int c = tail + threadIdx.x; c < cols; c += blockDim.x)
        out_norm[base + c] = __float2bfloat16(__bfloat162float(out_sum[base + c]) * inv_rms * __bfloat162float(weight[c]));
}

// add_rmsnorm2 that ALSO emits a Q8_1 quantization of out_norm (si_block_q8_1), so the
// downstream int8 GEMV/MMVQ skips its separate per-layer quantize node. Specialized to the
// decode residual width: one row, cols a multiple of 256, exactly one 8-wide pack per thread
// (blockDim*8 == cols), so each 32-element Q8_1 block maps to 4 consecutive threads. The Q8_1
// is computed from the bf16-rounded out_norm, so it is bit-identical to running the standalone
// quantizer on out_norm afterwards.
__global__ void add_rmsnorm2_q8_kernel(const __nv_bfloat16* __restrict__ x,
                                       const __nv_bfloat16* __restrict__ residual,
                                       const __nv_bfloat16* __restrict__ weight,
                                       __nv_bfloat16* __restrict__ out_sum,
                                       __nv_bfloat16* __restrict__ out_norm,
                                       si_blk_q8_1* __restrict__ out_q8,
                                       int cols, float eps) {
    const size_t base = 0;
    __shared__ float s_warp[32];
    const int t = threadIdx.x;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    const uint4* r4 = reinterpret_cast<const uint4*>(residual);
    uint4* osum4 = reinterpret_cast<uint4*>(out_sum);

    float xv[8], rv[8]; rn_unpack8(__ldg(x4 + t), xv); rn_unpack8(__ldg(r4 + t), rv);
    float sv[8]; float ss = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; j++) { sv[j] = xv[j] + rv[j]; ss = __fmaf_rn(sv[j], sv[j], ss); }
    osum4[t] = rn_pack8(sv);
    ss = rn_warp_sum(ss);
    if ((t & 31) == 0) s_warp[t >> 5] = ss;
    __syncthreads();
    if (t < 32) {
        float v = (t < (blockDim.x + 31) / 32) ? s_warp[t] : 0.f;
        v = rn_warp_sum(v);
        if (t == 0) s_warp[0] = rsqrtf(v / cols + eps);
    }
    __syncthreads();
    const float inv_rms = s_warp[0];

    // Re-read the bf16-rounded residual sum (exactly as add_rmsnorm2_kernel does), so out_norm
    // is bit-identical to the unfused path; then derive Q8_1 from the bf16-rounded out_norm.
    const uint4* w4 = reinterpret_cast<const uint4*>(weight);
    const uint4* osum4r = reinterpret_cast<const uint4*>(out_sum);
    float svb[8], wv[8]; rn_unpack8(__ldg(osum4r + t), svb); rn_unpack8(__ldg(w4 + t), wv);
    float ov[8], bv[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) { ov[j] = svb[j] * inv_rms * wv[j]; bv[j] = __bfloat162float(__float2bfloat16(ov[j])); }
    reinterpret_cast<uint4*>(out_norm)[t] = rn_pack8(ov);

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
        out_q8[b].qs[r * 8 + j] = (signed char)qi; s += qi;
    }
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2);
    if (r == 0) out_q8[b].ds = __floats2half2_rn(d, d * (float)s);
}

// 3-input add_rmsnorm2_q8: out_sum = x + (res1 + res2), folding a residual_add node.
__global__ void add_rmsnorm3_q8_kernel(const __nv_bfloat16* __restrict__ x,
                                       const __nv_bfloat16* __restrict__ res1,
                                       const __nv_bfloat16* __restrict__ res2,
                                       const __nv_bfloat16* __restrict__ weight,
                                       __nv_bfloat16* __restrict__ out_sum,
                                       __nv_bfloat16* __restrict__ out_norm,
                                       si_blk_q8_1* __restrict__ out_q8,
                                       int cols, float eps) {
    __shared__ float s_warp[32];
    const int t = threadIdx.x;
    const uint4* x4  = reinterpret_cast<const uint4*>(x);
    const uint4* r14 = reinterpret_cast<const uint4*>(res1);
    const uint4* r24 = reinterpret_cast<const uint4*>(res2);
    uint4* osum4 = reinterpret_cast<uint4*>(out_sum);

    float xv[8], r1v[8], r2v[8];
    rn_unpack8(__ldg(x4 + t), xv); rn_unpack8(__ldg(r14 + t), r1v); rn_unpack8(__ldg(r24 + t), r2v);
    float sv[8]; float ss = 0.f;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        float rs = __bfloat162float(__float2bfloat16(r1v[j] + r2v[j]));
        sv[j] = xv[j] + rs;
        ss = __fmaf_rn(sv[j], sv[j], ss);
    }
    osum4[t] = rn_pack8(sv);
    ss = rn_warp_sum(ss);
    if ((t & 31) == 0) s_warp[t >> 5] = ss;
    __syncthreads();
    if (t < 32) {
        float v = (t < (blockDim.x + 31) / 32) ? s_warp[t] : 0.f;
        v = rn_warp_sum(v);
        if (t == 0) s_warp[0] = rsqrtf(v / cols + eps);
    }
    __syncthreads();
    const float inv_rms = s_warp[0];

    const uint4* w4 = reinterpret_cast<const uint4*>(weight);
    const uint4* osum4r = reinterpret_cast<const uint4*>(out_sum);
    float svb[8], wv[8]; rn_unpack8(__ldg(osum4r + t), svb); rn_unpack8(__ldg(w4 + t), wv);
    float ov[8], bv[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) { ov[j] = svb[j] * inv_rms * wv[j]; bv[j] = __bfloat162float(__float2bfloat16(ov[j])); }
    reinterpret_cast<uint4*>(out_norm)[t] = rn_pack8(ov);

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
        out_q8[b].qs[r * 8 + j] = (signed char)qi; s += qi;
    }
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2);
    if (r == 0) out_q8[b].ds = __floats2half2_rn(d, d * (float)s);
}

__global__ void add_rmsnorm3_kernel(const __nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ res1,
                                    const __nv_bfloat16* __restrict__ res2,
                                    const __nv_bfloat16* __restrict__ weight,
                                    __nv_bfloat16* __restrict__ out_sum,
                                    __nv_bfloat16* __restrict__ out_norm,
                                    int rows, int cols, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const size_t base = (size_t)row * cols;
    __shared__ float s_warp[32];
    const int npack = cols >> 3;
    const int tail  = npack << 3;
    const uint4* x4  = reinterpret_cast<const uint4*>(x + base);
    const uint4* r14 = reinterpret_cast<const uint4*>(res1 + base);
    const uint4* r24 = reinterpret_cast<const uint4*>(res2 + base);
    uint4* osum4 = reinterpret_cast<uint4*>(out_sum + base);

    float ss = 0.f;
    for (int p = threadIdx.x; p < npack; p += blockDim.x) {
        float xv[8], r1v[8], r2v[8];
        rn_unpack8(__ldg(x4 + p), xv); rn_unpack8(__ldg(r14 + p), r1v); rn_unpack8(__ldg(r24 + p), r2v);
        float sv[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) sv[j] = xv[j] + __bfloat162float(__float2bfloat16(r1v[j] + r2v[j]));
        osum4[p] = rn_pack8(sv);
        #pragma unroll
        for (int j = 0; j < 8; j++) ss = __fmaf_rn(sv[j], sv[j], ss);
    }
    for (int c = tail + threadIdx.x; c < cols; c += blockDim.x) {
        float rs = __bfloat162float(__float2bfloat16(__bfloat162float(res1[base + c]) + __bfloat162float(res2[base + c])));
        float v = __bfloat162float(x[base + c]) + rs;
        out_sum[base + c] = __float2bfloat16(v);
        ss = __fmaf_rn(v, v, ss);
    }
    ss = rn_warp_sum(ss);
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
        v = rn_warp_sum(v);
        if (threadIdx.x == 0) s_warp[0] = rsqrtf(v / cols + eps);
    }
    __syncthreads();
    const float inv_rms = s_warp[0];
    const uint4* w4 = reinterpret_cast<const uint4*>(weight);
    const uint4* osum4r = reinterpret_cast<const uint4*>(out_sum + base);
    uint4* onorm4 = reinterpret_cast<uint4*>(out_norm + base);
    for (int p = threadIdx.x; p < npack; p += blockDim.x) {
        float sv[8], wv[8]; rn_unpack8(__ldg(osum4r + p), sv); rn_unpack8(__ldg(w4 + p), wv);
        float ov[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) ov[j] = sv[j] * inv_rms * wv[j];
        onorm4[p] = rn_pack8(ov);
    }
    for (int c = tail + threadIdx.x; c < cols; c += blockDim.x)
        out_norm[base + c] = __float2bfloat16(__bfloat162float(out_sum[base + c]) * inv_rms * __bfloat162float(weight[c]));
}

// Fused per-head Q-norm + K-norm: ONE kernel over (n_q_heads + n_kv_heads) heads
// (each block normalizes one head), vs two launch_rmsnorm calls. 1 graph node saved.
__global__ void rmsnorm_qk_kernel(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k,
                                  const __nv_bfloat16* __restrict__ q_w, const __nv_bfloat16* __restrict__ k_w,
                                  int n_q_heads, int head_dim, float eps) {
    const int hh = blockIdx.x;
    __nv_bfloat16* x; const __nv_bfloat16* w; int head;
    if (hh < n_q_heads) { x = q; w = q_w; head = hh; }
    else                { x = k; w = k_w; head = hh - n_q_heads; }
    const size_t base = (size_t)head * head_dim;
    __shared__ float s_warp[32];
    const int t = threadIdx.x;
    const float v = (t < head_dim) ? __bfloat162float(x[base + t]) : 0.f;
    float ss = rn_warp_sum(v * v);
    if ((t & 31) == 0) s_warp[t >> 5] = ss;
    __syncthreads();
    if (t < 32) {
        float vv = (t < (blockDim.x + 31) / 32) ? s_warp[t] : 0.f;
        vv = rn_warp_sum(vv);
        if (t == 0) s_warp[0] = rsqrtf(vv / head_dim + eps);
    }
    __syncthreads();
    if (t < head_dim) x[base + t] = __float2bfloat16(v * s_warp[0] * __bfloat162float(w[t]));
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/fused.h"
#include <cassert>

void launch_rmsnorm_qk(void* q, void* k, const void* q_w, const void* k_w,
                       int n_q_heads, int n_kv_heads, int head_dim, float eps, cudaStream_t stream) {
    rmsnorm_qk_kernel<<<n_q_heads + n_kv_heads, head_dim, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q), reinterpret_cast<__nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(q_w), reinterpret_cast<const __nv_bfloat16*>(k_w),
        n_q_heads, head_dim, eps);
}

void launch_rmsnorm(const void* x, const void* weight, void* out,
                    int rows, int cols, float eps, cudaStream_t stream) {
    rmsnorm_kernel<0><<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), nullptr,
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out), rows, cols, eps);
}

void launch_add_rmsnorm(const void* x, const void* residual, const void* weight, void* out,
                        int rows, int cols, float eps, cudaStream_t stream) {
    rmsnorm_kernel<1><<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out), rows, cols, eps);
}

void launch_add_rmsnorm2(const void* x, const void* residual, const void* weight,
                         void* out_sum, void* out_norm, int rows, int cols, float eps,
                         cudaStream_t stream) {
    add_rmsnorm2_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out_sum),
        reinterpret_cast<__nv_bfloat16*>(out_norm), rows, cols, eps);
}

// add_rmsnorm2 that also emits Q8_1(out_norm). Requires rows==1 and cols%256==0 (decode
// residual width); uses cols/8 threads so each thread owns one 8-wide pack.
void launch_add_rmsnorm2_q8(const void* x, const void* residual, const void* weight,
                            void* out_sum, void* out_norm, void* out_q8, int cols, float eps,
                            cudaStream_t stream) {
    // One row, one 8-wide pack per thread, a Q8_1 block = 4 consecutive threads: requires
    // cols % 256 == 0 (every warp full, each 32-block within one warp) and cols/8 <= 1024
    // (max block dim). The decode residual width (hidden = 2048) satisfies both.
    assert(cols % 256 == 0 && (cols >> 3) <= 1024);
    add_rmsnorm2_q8_kernel<<<1, cols >> 3, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out_sum),
        reinterpret_cast<__nv_bfloat16*>(out_norm),
        reinterpret_cast<si_blk_q8_1*>(out_q8), cols, eps);
}

void launch_add_rmsnorm3_q8(const void* x, const void* res1, const void* res2, const void* weight,
                            void* out_sum, void* out_norm, void* out_q8, int cols, float eps,
                            cudaStream_t stream) {
    assert(cols % 256 == 0 && (cols >> 3) <= 1024);
    add_rmsnorm3_q8_kernel<<<1, cols >> 3, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(res1),
        reinterpret_cast<const __nv_bfloat16*>(res2),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out_sum),
        reinterpret_cast<__nv_bfloat16*>(out_norm),
        reinterpret_cast<si_blk_q8_1*>(out_q8), cols, eps);
}

void launch_add_rmsnorm3(const void* x, const void* res1, const void* res2, const void* weight,
                         void* out_sum, void* out_norm, int rows, int cols, float eps,
                         cudaStream_t stream) {
    add_rmsnorm3_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(res1),
        reinterpret_cast<const __nv_bfloat16*>(res2),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out_sum),
        reinterpret_cast<__nv_bfloat16*>(out_norm), rows, cols, eps);
}
#endif

} // namespace kernels
} // namespace sparkinfer
