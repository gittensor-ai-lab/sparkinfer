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

#ifndef _MSC_VER
template __global__ void rmsnorm_kernel<0>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, float);
#endif
#ifndef _MSC_VER
template __global__ void rmsnorm_kernel<1>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, float);
#endif
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
    const size_t base = (size_t)blockIdx.x * cols;
    __shared__ float s_warp[32];
    const int t = threadIdx.x;
    const uint4* x4 = reinterpret_cast<const uint4*>(x + base);
    const uint4* r4 = reinterpret_cast<const uint4*>(residual + base);
    uint4* osum4 = reinterpret_cast<uint4*>(out_sum + base);
    if (out_q8) out_q8 += (size_t)blockIdx.x * (cols >> 5);

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
    const uint4* osum4r = reinterpret_cast<const uint4*>(out_sum + base);
    float svb[8], wv[8]; rn_unpack8(__ldg(osum4r + t), svb); rn_unpack8(__ldg(w4 + t), wv);
    float ov[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) ov[j] = svb[j] * inv_rms * wv[j];
    reinterpret_cast<uint4*>(out_norm + base)[t] = rn_pack8(ov);

    if (out_q8) {
        float bv[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) bv[j] = __bfloat162float(__float2bfloat16(ov[j]));
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
    const size_t base = (size_t)blockIdx.x * cols;
    const uint4* x4  = reinterpret_cast<const uint4*>(x + base);
    const uint4* r14 = reinterpret_cast<const uint4*>(res1 + base);
    const uint4* r24 = reinterpret_cast<const uint4*>(res2 + base);
    uint4* osum4 = reinterpret_cast<uint4*>(out_sum + base);
    out_q8 += (size_t)blockIdx.x * (cols >> 5);

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
    const uint4* osum4r = reinterpret_cast<const uint4*>(out_sum + base);
    float svb[8], wv[8]; rn_unpack8(__ldg(osum4r + t), svb); rn_unpack8(__ldg(w4 + t), wv);
    float ov[8], bv[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) { ov[j] = svb[j] * inv_rms * wv[j]; bv[j] = __bfloat162float(__float2bfloat16(ov[j])); }
    reinterpret_cast<uint4*>(out_norm + base)[t] = rn_pack8(ov);

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

// Sandwich-norm residual add: out[i] = residual[i] + RMSNorm(block_out)[i] * weight[i].
// Every other kernel in this file norms the SUM (x + residual) -- standard pre-norm reuse,
// where the norm doubles as the next sub-block's input norm. This is the opposite order:
// block_out (the attention or FFN sub-block's raw output, before any residual add) is
// normalized ALONE first, and only the normalized result is added back into the residual
// stream. Gemma2/Muse-Glimmer-style sandwich norm -- used for both the post-attention and
// post-FFN norm steps (same operation, different block_out/weight).
__global__ void norm_then_add_kernel(const __nv_bfloat16* __restrict__ residual,
                                     const __nv_bfloat16* __restrict__ block_out,
                                     const __nv_bfloat16* __restrict__ weight,
                                     __nv_bfloat16* __restrict__ out,
                                     int rows, int cols, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const size_t base = (size_t)row * cols;
    __shared__ float s_warp[32];
    const int npack = cols >> 3;
    const int tail = npack << 3;
    const uint4* b4 = reinterpret_cast<const uint4*>(block_out + base);

    float ss = 0.f;
    for (int p = threadIdx.x; p < npack; p += blockDim.x) {
        float bv[8]; rn_unpack8(__ldg(b4 + p), bv);
        #pragma unroll
        for (int j = 0; j < 8; j++) ss = __fmaf_rn(bv[j], bv[j], ss);
    }
    for (int c = tail + threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(block_out[base + c]);
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
    const uint4* r4 = reinterpret_cast<const uint4*>(residual + base);
    uint4* o4 = reinterpret_cast<uint4*>(out + base);
    for (int p = threadIdx.x; p < npack; p += blockDim.x) {
        float bv[8]; rn_unpack8(__ldg(b4 + p), bv);
        float rv[8]; rn_unpack8(__ldg(r4 + p), rv);
        float wv[8]; rn_unpack8(__ldg(w4 + p), wv);
        float ov[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) ov[j] = rv[j] + bv[j] * inv_rms * wv[j];
        o4[p] = rn_pack8(ov);
    }
    for (int c = tail + threadIdx.x; c < cols; c += blockDim.x) {
        float bv = __bfloat162float(block_out[base + c]);
        float rv = __bfloat162float(residual[base + c]);
        float wv = __bfloat162float(weight[c]);
        out[base + c] = __float2bfloat16(rv + bv * inv_rms * wv);
    }
}

// Muse Glimmer's sandwich-norm tail, fused. The architecture runs it twice per layer:
//   out_x  = residual + RMSNorm(branch, post_w, post_eps)      (launch_norm_then_add)
//   out_xn = RMSNorm(out_x, next_w, eps)                       (launch_rmsnorm)
// Both are single-CTA kernels over 6656 elements -- ~3.2 us each for 13 KB, i.e. one SM of 170
// and almost pure launch/reduction latency -- and there are 4 of them per layer, 208 per token.
//
// Bit-identical to the pair by construction: the block is the same 256 threads, each stage keeps
// the other kernel's exact loop order and warp-reduction tree, and stage 2 re-reads the bf16-
// rounded out_x back from global rather than reusing the float registers, which is precisely what
// the separate launch_rmsnorm would have seen.
__global__ void muse_sandwich_tail_kernel(const __nv_bfloat16* __restrict__ residual,
                                          const __nv_bfloat16* __restrict__ branch,
                                          const __nv_bfloat16* __restrict__ post_w,
                                          const __nv_bfloat16* __restrict__ next_w,
                                          __nv_bfloat16* __restrict__ out_x,
                                          __nv_bfloat16* __restrict__ out_xn,
                                          si_blk_q8_1* __restrict__ out_q8,
                                          int cols, float post_eps, float eps, int reg_tail) {
    const size_t base = (size_t)blockIdx.x * cols;
    __shared__ float s_warp[32];
    const int npack = cols >> 3;
    const int tail = npack << 3;

    // One pack per thread -- Muse's 6656 columns is 832 packs against a 1024-thread block, and the
    // scalar tail is empty -- lets the whole tail live in registers. Two things follow, and both
    // are latency, not bandwidth: the bf16 stage 2 re-reads from out_x is the bf16 this thread
    // just rounded, so keeping it deletes a store -> __syncthreads -> load round trip off the
    // critical path; and with nothing downstream of a load, all four operands issue together
    // instead of in three dependent waves. Same values, same order, so it stays bit-identical --
    // rn_unpack8(rn_pack8(v)) is __bfloat162float(__float2bfloat16(v)), exactly what the global
    // round trip returned. Wider rows keep the original path below.
    if (reg_tail && npack <= blockDim.x && tail == cols) {
        const int p = threadIdx.x;
        const bool live = p < npack;
        float bv[8], rv[8], pv[8], nv[8];
        if (live) {
            rn_unpack8(__ldg(reinterpret_cast<const uint4*>(branch + base) + p), bv);
            rn_unpack8(__ldg(reinterpret_cast<const uint4*>(residual + base) + p), rv);
            rn_unpack8(__ldg(reinterpret_cast<const uint4*>(post_w) + p), pv);
            rn_unpack8(__ldg(reinterpret_cast<const uint4*>(next_w) + p), nv);
        }
        float ss = 0.f;
        if (live) {
            #pragma unroll
            for (int j = 0; j < 8; j++) ss = __fmaf_rn(bv[j], bv[j], ss);
        }
        ss = rn_warp_sum(ss);
        if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = ss;
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
            v = rn_warp_sum(v);
            if (threadIdx.x == 0) s_warp[0] = rsqrtf(v / cols + post_eps);
        }
        __syncthreads();
        const float inv1 = s_warp[0];

        float xv[8];
        float ss2 = 0.f;
        if (live) {
            float ov[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) ov[j] = rv[j] + bv[j] * inv1 * pv[j];
            const uint4 packed = rn_pack8(ov);
            reinterpret_cast<uint4*>(out_x + base)[p] = packed;
            rn_unpack8(packed, xv);          // the bf16 the separate rmsnorm would have re-read
            #pragma unroll
            for (int j = 0; j < 8; j++) ss2 = __fmaf_rn(xv[j], xv[j], ss2);
        }
        ss2 = rn_warp_sum(ss2);
        __syncthreads();                     // every thread has consumed inv1; s_warp is free
        if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = ss2;
        __syncthreads();
        if (threadIdx.x < 32) {
            float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
            v = rn_warp_sum(v);
            if (threadIdx.x == 0) s_warp[0] = rsqrtf(v / cols + eps);
        }
        __syncthreads();
        const float inv2 = s_warp[0];
        if (live) {
            float ov[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) ov[j] = xv[j] * inv2 * nv[j];
            const uint4 packed = rn_pack8(ov);
            reinterpret_cast<uint4*>(out_xn + base)[p] = packed;
            // Q8_1(out_xn) for whichever MMVQ consumes this row next -- the FFN's gate/up after
            // the post-attention tail, the next layer's Q/K/V after the post-FFN one. Both used
            // to launch a standalone quantize over a row this kernel had just written. Quantizes
            // the *stored* bf16, and amax and the integer sum are both order-independent, so the
            // 4-lane form here and the 32-lane si_quant_bf16_q8_1 emit the same block. The
            // launcher only passes out_q8 when every warp is fully live, so these butterflies
            // cannot diverge.
            if (out_q8) {
                float q[8]; rn_unpack8(packed, q);
                const int b = p >> 2, r = p & 3;
                float amax = 0.f;
                #pragma unroll
                for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(q[j]));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
                const float d = amax / 127.0f;
                si_blk_q8_1* ob = out_q8 + (size_t)blockIdx.x * (cols >> 5) + b;
                int s = 0;
                unsigned w[2] = {0u, 0u};
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int qi = (amax == 0.0f) ? 0 : (int)roundf(q[j] / d);
                    w[j >> 2] |= ((unsigned)qi & 255u) << ((j & 3) * 8);
                    s += qi;
                }
                // qs[r*8] sits at 36*b + 4 + 8*r, so the eight bytes this lane owns are always
                // 4-byte aligned: two STG.32 instead of eight STG.U8, same bytes.
                unsigned* qw = reinterpret_cast<unsigned*>(ob->qs + r * 8);
                qw[0] = w[0];
                qw[1] = w[1];
                s += __shfl_xor_sync(0xffffffffu, s, 1);
                s += __shfl_xor_sync(0xffffffffu, s, 2);
                if (r == 0) ob->ds = __floats2half2_rn(d, d * (float)s);
            }
        }
        return;
    }

    // ---- stage 1: norm_then_add over `branch` ----
    const uint4* b4 = reinterpret_cast<const uint4*>(branch + base);
    float ss = 0.f;
    for (int p = threadIdx.x; p < npack; p += blockDim.x) {
        float bv[8]; rn_unpack8(__ldg(b4 + p), bv);
        #pragma unroll
        for (int j = 0; j < 8; j++) ss = __fmaf_rn(bv[j], bv[j], ss);
    }
    for (int c = tail + threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(branch[base + c]);
        ss = __fmaf_rn(v, v, ss);
    }
    ss = rn_warp_sum(ss);
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
        v = rn_warp_sum(v);
        if (threadIdx.x == 0) s_warp[0] = rsqrtf(v / cols + post_eps);
    }
    __syncthreads();
    const float inv1 = s_warp[0];
    {
        const uint4* pw4 = reinterpret_cast<const uint4*>(post_w);
        const uint4* r4 = reinterpret_cast<const uint4*>(residual + base);
        uint4* ox4 = reinterpret_cast<uint4*>(out_x + base);
        for (int p = threadIdx.x; p < npack; p += blockDim.x) {
            float bv[8]; rn_unpack8(__ldg(b4 + p), bv);
            float rv[8]; rn_unpack8(__ldg(r4 + p), rv);
            float wv[8]; rn_unpack8(__ldg(pw4 + p), wv);
            float ov[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) ov[j] = rv[j] + bv[j] * inv1 * wv[j];
            ox4[p] = rn_pack8(ov);
        }
        for (int c = tail + threadIdx.x; c < cols; c += blockDim.x) {
            float bv = __bfloat162float(branch[base + c]);
            float rv = __bfloat162float(residual[base + c]);
            float wv = __bfloat162float(post_w[c]);
            out_x[base + c] = __float2bfloat16(rv + bv * inv1 * wv);
        }
    }
    __syncthreads();   // out_x visible to the whole block, and s_warp free to reuse

    // ---- stage 2: rmsnorm over the bf16-rounded out_x ----
    const uint4* x4 = reinterpret_cast<const uint4*>(out_x + base);
    float ss2 = 0.f;
    for (int p = threadIdx.x; p < npack; p += blockDim.x) {
        float xv[8]; rn_unpack8(__ldg(x4 + p), xv);
        #pragma unroll
        for (int j = 0; j < 8; j++) ss2 = __fmaf_rn(xv[j], xv[j], ss2);
    }
    for (int c = tail + threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(out_x[base + c]);
        ss2 = __fmaf_rn(v, v, ss2);
    }
    ss2 = rn_warp_sum(ss2);
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = ss2;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
        v = rn_warp_sum(v);
        if (threadIdx.x == 0) s_warp[0] = rsqrtf(v / cols + eps);
    }
    __syncthreads();
    const float inv2 = s_warp[0];
    {
        const uint4* nw4 = reinterpret_cast<const uint4*>(next_w);
        uint4* on4 = reinterpret_cast<uint4*>(out_xn + base);
        for (int p = threadIdx.x; p < npack; p += blockDim.x) {
            float xv[8]; rn_unpack8(__ldg(x4 + p), xv);
            float wv[8]; rn_unpack8(__ldg(nw4 + p), wv);
            float ov[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) ov[j] = xv[j] * inv2 * wv[j];
            on4[p] = rn_pack8(ov);
        }
        for (int c = tail + threadIdx.x; c < cols; c += blockDim.x) {
            float v = __bfloat162float(out_x[base + c]);
            out_xn[base + c] = __float2bfloat16(v * inv2 * __bfloat162float(next_w[c]));
        }
    }
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


// Fused QK-norm + (optional NORM-convention RoPE) + K/V cache append, for Muse Glimmer.
// Collapses launch_rmsnorm_qk followed by launch_rope_kv_append_normal (sliding-window layers) or
// launch_kv_append (NoPE layers) into ONE kernel: 2 decode-graph nodes per layer become 1, 52 per
// 128-token step.
//
// Bit-identical to the split pair, by construction:
//   * the per-head RMS reduction keeps rmsnorm_qk_kernel's exact warp -> shared -> warp grouping
//     and the same blockDim (head_dim), so the fp32 summation order is unchanged;
//   * the normed value is rounded to bf16 BEFORE the rotation reads it, which is precisely what
//     the split path does implicitly when rmsnorm_qk stores bf16 to global and the rope kernel
//     loads it back;
//   * q/k are still written back in their normed (unrotated) form, so anything reading s.q/s.k
//     afterwards sees exactly what it saw before.
// `pos_angle` supplies the RoPE position and `pos_slot` the cache slot; the split path used d_pos
// for both on the rope layers and d_writepos for the slot on the NoPE layers, so they are passed
// separately rather than assumed equal.
__global__ void muse_qknorm_rope_kv_kernel(
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_w, const __nv_bfloat16* __restrict__ k_w,
    __nv_bfloat16* __restrict__ k_pool, __nv_bfloat16* __restrict__ v_pool,
    const int* __restrict__ block_table, const int* __restrict__ pos_angle,
    const int* __restrict__ pos_slot, int n_q_heads, int n_kv_heads, int head_dim,
    float theta, int block_size, float eps, int do_rope
) {
    const int hh = blockIdx.x, t = threadIdx.x;
    const int slot = pos_slot[0];
    const int blk = slot / block_size, within = slot - blk * block_size;
    const size_t ctok = (size_t)((size_t)block_table[blk] * block_size + within);

    if (hh >= n_q_heads + n_kv_heads) {          // V head: straight copy, no norm, no rope
        const int h = hh - n_q_heads - n_kv_heads;
        v_pool[(ctok * n_kv_heads + h) * head_dim + t] = v[(size_t)h * head_dim + t];
        return;
    }
    const bool is_q         = (hh < n_q_heads);
    __nv_bfloat16* x        = is_q ? q : k;
    const __nv_bfloat16* w  = is_q ? q_w : k_w;
    const int head          = is_q ? hh : hh - n_q_heads;
    const size_t base       = (size_t)head * head_dim;

    __shared__ float s_warp[32];
    extern __shared__ float s_h[];
    const float xv = __bfloat162float(x[base + t]);
    float ss = rn_warp_sum(xv * xv);
    if ((t & 31) == 0) s_warp[t >> 5] = ss;
    __syncthreads();
    if (t < 32) {
        float vv = (t < (blockDim.x + 31) / 32) ? s_warp[t] : 0.f;
        vv = rn_warp_sum(vv);
        if (t == 0) s_warp[0] = rsqrtf(vv / head_dim + eps);
    }
    __syncthreads();
    const __nv_bfloat16 nb = __float2bfloat16(xv * s_warp[0] * __bfloat162float(w[t]));
    x[base + t] = nb;                 // keep s.q / s.k byte-identical to the split path
    s_h[t] = __bfloat162float(nb);
    __syncthreads();

    if (!do_rope) {                   // NoPE layer: append the normed K as-is
        if (!is_q) k_pool[(ctok * n_kv_heads + head) * head_dim + t] = nb;
        return;
    }
    const int half = head_dim >> 1;
    if (t < half) {                   // NORM convention: rotate the pair (2i, 2i+1)
        const float freq = __powf(theta, -2.f * (float)t / (float)head_dim);
        const float ang = (float)pos_angle[0] * freq, c = __cosf(ang), sn = __sinf(ang);
        const float x0 = s_h[2 * t], x1 = s_h[2 * t + 1];
        if (is_q) {
            q[base + 2 * t]     = __float2bfloat16(x0 * c - x1 * sn);
            q[base + 2 * t + 1] = __float2bfloat16(x0 * sn + x1 * c);
        } else {
            const size_t dst = (ctok * n_kv_heads + head) * head_dim + 2 * t;
            k_pool[dst]     = __float2bfloat16(x0 * c - x1 * sn);
            k_pool[dst + 1] = __float2bfloat16(x0 * sn + x1 * c);
        }
    }
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/fused.h"
#include <cassert>
#include <cstdlib>

void launch_muse_qknorm_rope_kv(void* q, void* k, const void* v, const void* q_w, const void* k_w,
                                void* k_pool, void* v_pool, const int* block_table,
                                const int* pos_angle, const int* pos_slot,
                                int n_q_heads, int n_kv_heads, int head_dim, float theta,
                                int block_size, float eps, bool do_rope, cudaStream_t stream) {
    const int blocks = n_q_heads + 2 * n_kv_heads;   // q heads, k heads, then the v-copy heads
    muse_qknorm_rope_kv_kernel<<<blocks, head_dim, head_dim * sizeof(float), stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q), reinterpret_cast<__nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<const __nv_bfloat16*>(q_w), reinterpret_cast<const __nv_bfloat16*>(k_w),
        reinterpret_cast<__nv_bfloat16*>(k_pool), reinterpret_cast<__nv_bfloat16*>(v_pool),
        block_table, pos_angle, pos_slot, n_q_heads, n_kv_heads, head_dim, theta,
        block_size, eps, do_rope ? 1 : 0);
}

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

void launch_add_rmsnorm2_q8_rows(const void* x, const void* residual, const void* weight,
                                 void* out_sum, void* out_norm, void* out_q8,
                                 int rows, int cols, float eps, cudaStream_t stream) {
    assert(rows > 0 && cols % 256 == 0 && (cols >> 3) <= 1024);
    add_rmsnorm2_q8_kernel<<<rows, cols >> 3, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(weight), reinterpret_cast<__nv_bfloat16*>(out_sum),
        reinterpret_cast<__nv_bfloat16*>(out_norm), reinterpret_cast<si_blk_q8_1*>(out_q8), cols, eps);
}

void launch_add_rmsnorm3_q8_rows(const void* x, const void* res1, const void* res2,
                                 const void* weight, void* out_sum, void* out_norm, void* out_q8,
                                 int rows, int cols, float eps, cudaStream_t stream) {
    assert(rows > 0 && cols % 256 == 0 && (cols >> 3) <= 1024);
    add_rmsnorm3_q8_kernel<<<rows, cols >> 3, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(res1),
        reinterpret_cast<const __nv_bfloat16*>(res2), reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out_sum), reinterpret_cast<__nv_bfloat16*>(out_norm),
        reinterpret_cast<si_blk_q8_1*>(out_q8), cols, eps);
}

bool launch_muse_sandwich_tail(const void* residual, const void* branch, const void* post_w,
                               const void* next_w, void* out_x, void* out_xn, void* out_q8,
                               int rows, int cols, float post_eps, float eps, cudaStream_t stream) {
    // 256 threads for a 6656-wide row means npack = 832 packs walked 3-4 deep per thread, twice,
    // on a chain of dependent loads -- and the whole thing is one CTA of a 170-SM device. One
    // thread per pack collapses that chain; 1024 is the first width that covers all 832 in a
    // single pass. Sweep at 128 decode: 256 -> 91.80 tok/s, 512 -> 92.87, 768 -> 92.69,
    // 1024 -> 93.49.
    //
    // This is the one change here that is NOT bit-identical: the block width sets how the
    // sum-of-squares partials are grouped, so fp32 reassociates. It is a reassociation and not a
    // loss of quality -- teacher-forced over 191 in-distribution positions, perplexity moves
    // 2.12623 -> 2.11524 and agreement with the true next token 176/191 -> 175/191, while
    // agreement against the 256-wide arm is 189/191 with mean |dlogprob| 1.1e-2.
    // SPARKINFER_MUSE_TAIL_W overrides the width (256/512/768/1024) for A/B.
    static int tail_w = -1;
    if (tail_w < 0) {
        const char* e = getenv("SPARKINFER_MUSE_TAIL_W");
        tail_w = e ? atoi(e) : 1024;
        if (tail_w != 256 && tail_w != 512 && tail_w != 768 && tail_w != 1024) tail_w = 1024;
    }
    // SPARKINFER_MUSE_TAIL_REG=0 forces the global-memory path, so both arms of an A/B come out
    // of one binary.
    static int reg_tail = -1;
    if (reg_tail < 0) {
        const char* e = getenv("SPARKINFER_MUSE_TAIL_REG");
        reg_tail = (e && e[0] == '0') ? 0 : 1;
    }
    // The Q8_1 side channel rides the register path only, and only when npack is a whole number
    // of warps so the 4-lane butterflies stay convergent. Report what actually happened rather
    // than letting the caller assume: a caller that skips its own quantize on a false promise
    // feeds the next MMVQ a stale activation, which reads as a confident wrong distribution
    // rather than as a crash.
    const int npack = cols >> 3;
    const bool q8 = out_q8 && reg_tail && npack <= tail_w && (npack << 3) == cols &&
                    (npack & 31) == 0 && (cols & 31) == 0;
    muse_sandwich_tail_kernel<<<rows, tail_w, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(branch),
        reinterpret_cast<const __nv_bfloat16*>(post_w),
        reinterpret_cast<const __nv_bfloat16*>(next_w),
        reinterpret_cast<__nv_bfloat16*>(out_x),
        reinterpret_cast<__nv_bfloat16*>(out_xn),
        q8 ? reinterpret_cast<si_blk_q8_1*>(out_q8) : nullptr, cols, post_eps, eps, reg_tail);
    return q8;
}

// Same sandwich norm, fed straight from the split-K int32 accumulator instead of from a bf16
// tensor a reduce pass had to materialize first. The o and down projections each wrote H bf16 per
// token that exactly ONE kernel then read; this deletes that round trip and the reduce launch.
//
// BIT-IDENTICAL, and the accumulation order is the whole reason it is: `__float2bfloat16((float)acc
// * sxr[row] * rs[c])` is exactly the value pf_dense_splitk_reduce_kernel stored, and the
// sum-of-squares below walks the SAME packs of 8 in the SAME order with the SAME stride as
// norm_then_add_kernel, so the fp32 reduction tree is unchanged. (Re-associating it is what sank
// an earlier attempt at fusing these norms.)
__global__ void norm_then_add_acc_kernel(const __nv_bfloat16* __restrict__ residual,
                                         const int* __restrict__ acc,
                                         const float* __restrict__ sxr,
                                         const float* __restrict__ rs,
                                         const __nv_bfloat16* __restrict__ weight,
                                         __nv_bfloat16* __restrict__ out,
                                         int rows, int cols, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const size_t base = (size_t)row * cols;
    const float sr = sxr[row];
    __shared__ float s_warp[32];
    const int npack = cols >> 3;
    const int tail = npack << 3;

    auto load8 = [&](int p, float (&bv)[8]) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int c = p * 8 + j;
            bv[j] = __bfloat162float(__float2bfloat16((float)acc[base + c] * sr * rs[c]));
        }
    };

    // Hold the scaled values across the two passes: `acc` is int32, so a second read of it is 4 B
    // per element on a kernel already at the DRAM roofline (10.2 MB moved per launch at H=6656).
    __nv_bfloat16 keep[4][8];
    int held = 0;
    float ss = 0.f;
    for (int p = threadIdx.x; p < npack; p += blockDim.x, held++) {
        float bv[8]; load8(p, bv);
        #pragma unroll
        for (int j = 0; j < 8; j++) keep[held][j] = __float2bfloat16(bv[j]);
        #pragma unroll
        for (int j = 0; j < 8; j++) ss = __fmaf_rn(bv[j], bv[j], ss);
    }
    for (int c = tail + threadIdx.x; c < cols; c += blockDim.x) {
        float v = __bfloat162float(__float2bfloat16((float)acc[base + c] * sr * rs[c]));
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
    const uint4* r4 = reinterpret_cast<const uint4*>(residual + base);
    uint4* o4 = reinterpret_cast<uint4*>(out + base);
    held = 0;
    for (int p = threadIdx.x; p < npack; p += blockDim.x, held++) {
        float bv[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) bv[j] = __bfloat162float(keep[held][j]);
        float rv[8]; rn_unpack8(__ldg(r4 + p), rv);
        float wv[8]; rn_unpack8(__ldg(w4 + p), wv);
        float ov[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) ov[j] = rv[j] + bv[j] * inv_rms * wv[j];
        o4[p] = rn_pack8(ov);
    }
    for (int c = tail + threadIdx.x; c < cols; c += blockDim.x) {
        const float bv = __bfloat162float(__float2bfloat16((float)acc[base + c] * sr * rs[c]));
        out[base + c] = __float2bfloat16(__bfloat162float(residual[base + c])
                                         + bv * inv_rms * __bfloat162float(weight[c]));
    }
}

// RMSNorm whose output is consumed by exactly one row-quantize (Muse's pre-FFN norm feeding the
// grouped gate/up GEMM). Emitting the int8 in the same pass removes a launch and a full re-read of
// the bf16 the norm just wrote. The bf16 is still written, so every fallback consumer is unaffected.
//
// BIT-IDENTICAL: the sum-of-squares walks the same packs of 8 in the same order with the same
// stride and the same fp32 tree as rmsnorm_kernel; the stored bf16 is the same rn_pack8 value; and
// the quantize is the same amax (order-independent), the same d = amax/127.0f, the same roundf and
// the same amax==0 -> 0 rule that pf_quant_rows_fast_kernel applies.
template <int MAXP>
__global__ __launch_bounds__(256) void rmsnorm_quant_i8_kernel(
        const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
        __nv_bfloat16* __restrict__ out, signed char* __restrict__ q, float* __restrict__ scale,
        int rows, int cols, float eps, signed char* __restrict__ qp) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const size_t base = (size_t)row * cols;
    __shared__ float s_warp[32];
    const int npack = cols >> 3;
    const uint4* x4 = reinterpret_cast<const uint4*>(x + base);

    float ss = 0.f;
    for (int p = threadIdx.x; p < npack; p += blockDim.x) {
        float xv[8]; rn_unpack8(__ldg(x4 + p), xv);
        #pragma unroll
        for (int j = 0; j < 8; j++) ss = __fmaf_rn(xv[j], xv[j], ss);
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
    __nv_bfloat16 reg[MAXP][8];
    float amax = 0.f;
    int held = 0;
    for (int p = threadIdx.x; p < npack; p += blockDim.x, held++) {
        float xv[8]; rn_unpack8(__ldg(x4 + p), xv);
        float wv[8]; rn_unpack8(__ldg(w4 + p), wv);
        float ov[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) ov[j] = xv[j] * inv_rms * wv[j];
        const uint4 packed = rn_pack8(ov);
        o4[p] = packed;                                  // the bf16 the norm always wrote
        *reinterpret_cast<uint4*>(reg[held]) = packed;   // keep it for the quantize below
        #pragma unroll
        for (int j = 0; j < 8; j++)
            amax = fmaxf(amax, fabsf(__bfloat162float(reg[held][j])));
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) s_warp[0] = v;
    }
    __syncthreads();
    const float amax_row = s_warp[0];
    const float d = amax_row / 127.0f;
    if (threadIdx.x == 0) scale[row] = d;

    held = 0;
    for (int p = threadIdx.x; p < npack; p += blockDim.x, held++) {
        signed char o8[8];
        #pragma unroll
        for (int j = 0; j < 8; j++)
            o8[j] = (signed char)((amax_row == 0.f) ? 0
                                 : (int)roundf(__bfloat162float(reg[held][j]) / d));
        *reinterpret_cast<uint2*>(&q[base + p * 8]) = *reinterpret_cast<const uint2*>(o8);
        // Same bytes in the k-tiled [k/32][row][32] layout the dense prefill GEMM stages from
        // (qr_pack_off, prefill_quant_rows.cu). This kernel is the pre-FFN norm's fused quantize,
        // so it is one of the writers of that activation and has to emit both.
        if (qp) {
            const int c = p * 8;
            *reinterpret_cast<uint2*>(
                &qp[(size_t)(c >> 5) * (size_t)rows * 32 + (size_t)row * 32 + (size_t)(c & 31)]) =
                *reinterpret_cast<const uint2*>(o8);
        }
    }
}

bool launch_rmsnorm_quant_i8(const void* x, const void* weight, void* out,
                             signed char* q, float* scale,
                             int rows, int cols, float eps, cudaStream_t stream,
                             signed char* qp) {
    if (qp && (cols % 32) != 0) qp = nullptr;
    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_PREFILL_NORM_QUANT");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    if (!enabled || rows <= 0 || cols <= 0 || (cols & 7) != 0) return false;
    const int npack = cols >> 3;
    const int per = (npack + 255) / 256;                 // packs held per thread
    auto xb = reinterpret_cast<const __nv_bfloat16*>(x);
    auto wb = reinterpret_cast<const __nv_bfloat16*>(weight);
    auto ob = reinterpret_cast<__nv_bfloat16*>(out);
    if (per <= 1)      rmsnorm_quant_i8_kernel<1><<<rows, 256, 0, stream>>>(xb, wb, ob, q, scale, rows, cols, eps, qp);
    else if (per <= 2) rmsnorm_quant_i8_kernel<2><<<rows, 256, 0, stream>>>(xb, wb, ob, q, scale, rows, cols, eps, qp);
    else if (per <= 4) rmsnorm_quant_i8_kernel<4><<<rows, 256, 0, stream>>>(xb, wb, ob, q, scale, rows, cols, eps, qp);
    else return false;
    return true;
}

void launch_norm_then_add_acc(const void* residual, const int* acc, const float* sxr,
                              const float* rs, const void* weight, void* out,
                              int rows, int cols, float eps, cudaStream_t stream) {
    norm_then_add_acc_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(residual), acc, sxr, rs,
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out), rows, cols, eps);
}

void launch_norm_then_add(const void* residual, const void* block_out, const void* weight,
                          void* out, int rows, int cols, float eps, cudaStream_t stream) {
    norm_then_add_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(block_out),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(out), rows, cols, eps);
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
