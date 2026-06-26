// Fused RMSNorm (+ optional residual add). One block per row; block-reduces the
// sum of squares, then writes the normalized, weighted row. A small CODA-style
// epilogue building block kept on the portable CUDA path.
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

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

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/fused.h"

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
#endif

} // namespace kernels
} // namespace sparkinfer
