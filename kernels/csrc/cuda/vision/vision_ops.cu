// Vision-tower kernels: LayerNorm, bias-add, tanh-GELU, residual add, full bidirectional
// attention, and the 2x2 patch merge. See kernels/include/sparkinfer/kernels/vision.h for why
// these are new rather than reuses of the text path's RMSNorm/SwiGLU.
//
// Portable CUDA — sm_89/90/100/120, the set CMAKE_CUDA_ARCHITECTURES builds.
#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#include "sparkinfer/kernels/vision.h"
#endif

namespace sparkinfer {
namespace kernels {

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// One block per row, 256 threads. Two passes over the row (mean, then variance) rather than a
// fused sum/sum-of-squares: dim is only 1152, the row is in L1 after the first pass, and the
// numerically stabler form is worth more here than one saved read.
__global__ void vis_layernorm_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                     const bf16* __restrict__ b, bf16* __restrict__ out,
                                     int dim, float eps) {
    const int row = blockIdx.x;
    const bf16* xr = x + (size_t)row * dim;
    bf16* orow = out + (size_t)row * dim;

    __shared__ float s_mean, s_rstd;
    float acc = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) acc += b2f(xr[i]);
    acc = warp_sum(acc);
    __shared__ float warp_acc[8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) warp_acc[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (blockDim.x + 31) / 32; i++) t += warp_acc[i];
        s_mean = t / dim;
    }
    __syncthreads();
    const float mean = s_mean;

    float v = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float d = b2f(xr[i]) - mean; v += d * d; }
    v = warp_sum(v);
    if (lane == 0) warp_acc[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (blockDim.x + 31) / 32; i++) t += warp_acc[i];
        s_rstd = rsqrtf(t / dim + eps);
    }
    __syncthreads();
    const float rstd = s_rstd;

    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        orow[i] = f2b((b2f(xr[i]) - mean) * rstd * b2f(w[i]) + b2f(b[i]));
}

__global__ void vis_add_bias_kernel(bf16* __restrict__ x, const bf16* __restrict__ bias,
                                    long n_rows, int dim) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows * dim) return;
    x[i] = f2b(b2f(x[i]) + b2f(bias[i % dim]));
}

__global__ void vis_gelu_tanh_kernel(bf16* __restrict__ x, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = b2f(x[i]);
    // 0.7978845608028654 = sqrt(2/pi)
    x[i] = f2b(0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))));
}

__global__ void vis_residual_add_kernel(bf16* __restrict__ acc, const bf16* __restrict__ add, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    acc[i] = f2b(b2f(acc[i]) + b2f(add[i]));
}

// One warp per (head, query). Online softmax over all keys -- no mask, every query sees every
// patch. head_dim is 72 here, so each lane holds ceil(72/32)=3 accumulators.
__global__ void vis_attention_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k,
                                     const bf16* __restrict__ v, bf16* __restrict__ out,
                                     int n_tokens, int n_heads, int head_dim, float scale) {
    const int head = blockIdx.y;
    const int qi = blockIdx.x;
    if (qi >= n_tokens || head >= n_heads) return;
    const int lane = threadIdx.x;
    const int stride = n_heads * head_dim;
    const int off = head * head_dim;
    const int ELEMS = (head_dim + 31) / 32;

    float qr[8];
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const int d = lane + e * 32;
        qr[e] = (e < ELEMS && d < head_dim) ? b2f(q[(size_t)qi * stride + off + d]) : 0.f;
    }

    float m = -1e30f, l = 0.f, acc[8];
    #pragma unroll
    for (int e = 0; e < 8; e++) acc[e] = 0.f;

    for (int kj = 0; kj < n_tokens; kj++) {
        float dot = 0.f;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int d = lane + e * 32;
            if (e < ELEMS && d < head_dim) dot += qr[e] * b2f(k[(size_t)kj * stride + off + d]);
        }
        const float score = warp_sum(dot) * scale;
        const float mn = fmaxf(m, score);
        const float corr = __expf(m - mn), pe = __expf(score - mn);
        l = l * corr + pe;
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            const int d = lane + e * 32;
            const float vv = (e < ELEMS && d < head_dim) ? b2f(v[(size_t)kj * stride + off + d]) : 0.f;
            acc[e] = acc[e] * corr + pe * vv;
        }
        m = mn;
    }

    const float inv = l > 0.f ? 1.f / l : 0.f;
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        const int d = lane + e * 32;
        if (e < ELEMS && d < head_dim) out[(size_t)qi * stride + off + d] = f2b(acc[e] * inv);
    }
}

__global__ void vis_patch_merge_kernel(const bf16* __restrict__ x, bf16* __restrict__ out,
                                       int grid_h, int grid_w, int merge, int dim) {
    const int bw = grid_w / merge;
    const int blocks = (grid_h / merge) * bw;
    const long total = (long)blocks * merge * merge * dim;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        const int d = (int)(i % dim);
        const long t = i / dim;
        const int c = (int)(t % merge);
        const int r = (int)((t / merge) % merge);
        const long b = t / (merge * merge);
        const int bx = (int)(b % bw), by = (int)(b / bw);
        const long src = (long)((by * merge + r) * grid_w + (bx * merge + c)) * dim + d;
        out[i] = x[src];
    }
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
void launch_vision_layernorm(const void* x, const void* weight, const void* bias, void* out,
                             int n_rows, int dim, float eps, cudaStream_t stream) {
    if (n_rows <= 0 || dim <= 0) return;
    vis_layernorm_kernel<<<n_rows, 256, 0, stream>>>(
        (const bf16*)x, (const bf16*)weight, (const bf16*)bias, (bf16*)out, dim, eps);
}

void launch_vision_add_bias(void* x, const void* bias, int n_rows, int dim, cudaStream_t stream) {
    const long n = (long)n_rows * dim;
    if (n <= 0) return;
    vis_add_bias_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
        (bf16*)x, (const bf16*)bias, n_rows, dim);
}

void launch_vision_gelu_tanh(void* x, long n, cudaStream_t stream) {
    if (n <= 0) return;
    vis_gelu_tanh_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>((bf16*)x, n);
}

void launch_vision_residual_add(void* acc, const void* add, long n, cudaStream_t stream) {
    if (n <= 0) return;
    vis_residual_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(
        (bf16*)acc, (const bf16*)add, n);
}

void launch_vision_attention(const void* q, const void* k, const void* v, void* out,
                             int n_tokens, int n_heads, int head_dim, float scale,
                             cudaStream_t stream) {
    if (n_tokens <= 0 || n_heads <= 0 || head_dim <= 0 || head_dim > 256) return;
    dim3 grid((unsigned)n_tokens, (unsigned)n_heads);
    vis_attention_kernel<<<grid, 32, 0, stream>>>(
        (const bf16*)q, (const bf16*)k, (const bf16*)v, (bf16*)out,
        n_tokens, n_heads, head_dim, scale);
}

void launch_vision_patch_merge(const void* x, void* out, int grid_h, int grid_w, int merge,
                               int dim, cudaStream_t stream) {
    if (grid_h <= 0 || grid_w <= 0 || merge <= 0 || dim <= 0) return;
    if (grid_h % merge || grid_w % merge) return;   // caller must pad to the merge granularity
    const long total = (long)(grid_h / merge) * (grid_w / merge) * merge * merge * dim;
    const unsigned blocks = (unsigned)((total + 255) / 256 > 65535 ? 65535 : (total + 255) / 256);
    vis_patch_merge_kernel<<<blocks, 256, 0, stream>>>(
        (const bf16*)x, (bf16*)out, grid_h, grid_w, merge, dim);
}
#endif

}  // namespace kernels
}  // namespace sparkinfer
