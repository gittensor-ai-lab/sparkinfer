// Rotary position embedding (RoPE), HF "rotate-half" convention (GPT-NeoX style)
// as used by Qwen/Llama. Applied to Q and K after projection, before attention.
//
// For a head vector x[head_dim] at position p, with half = head_dim/2:
//   freq_i  = theta^(-2i/head_dim),  angle = p * freq_i
//   out[i]      = x[i]*cos - x[i+half]*sin
//   out[i+half] = x[i+half]*cos + x[i]*sin     for i in [0, half)
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

// grid = (n_tokens, n_heads); blockDim = head_dim/2 threads (one per rotated pair).
__global__ void rope_kernel(
    __nv_bfloat16* __restrict__ x,        // [n_tokens, n_heads, head_dim]
    const int* __restrict__ positions,    // [n_tokens]
    int n_heads, int head_dim, float theta
) {
    const int tok  = blockIdx.x;
    const int head = blockIdx.y;
    const int i    = threadIdx.x;
    const int half = head_dim / 2;
    if (i >= half) return;

    const float p    = (float)positions[tok];
    const float freq = __powf(theta, -2.f * (float)i / (float)head_dim);
    const float ang  = p * freq;
    const float c = __cosf(ang), s = __sinf(ang);

    const size_t base = ((size_t)tok * n_heads + head) * head_dim;
    const float x0 = __bfloat162float(x[base + i]);
    const float x1 = __bfloat162float(x[base + i + half]);
    x[base + i]        = __float2bfloat16(x0 * c - x1 * s);
    x[base + i + half] = __float2bfloat16(x1 * c + x0 * s);
}

// ---- Fused QK-norm + RoPE -----------------------------------------------------
// Decode applies q_norm, k_norm and RoPE as FOUR separate kernels: rmsnorm(q),
// rmsnorm(k), rope(q), rope(k). At batch=1 it is the per-kernel launch latency —
// not the (tiny) compute — that dominates these ops (the same bs=1 launch-latency
// bottleneck the PDL path in expert_ffn_q4k.cu targets). This fuses all four into
// ONE kernel: one block per head RMS-normalizes that head into a bf16 shared
// buffer, then rotates it in place. The arithmetic is identical to the separate
// kernels — same 256-thread block reduction for the RMS, the same bf16 round-trip
// of the normalized value before rotation (rmsnorm writes bf16, rope reads it),
// and the same RoPE — so every output bit is unchanged; only the kernel count
// (4 -> 1 per layer) drops.
__device__ __forceinline__ float qkr_warp_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

__global__ void qknorm_rope_kernel(
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ q_norm_w, const __nv_bfloat16* __restrict__ k_norm_w,
    const int* __restrict__ positions, int n_q_heads, int n_kv_heads,
    int head_dim, float eps, float theta
) {
    const int h     = blockIdx.x;                       // [0, n_q_heads + n_kv_heads)
    const bool is_q = (h < n_q_heads);
    __nv_bfloat16*       buf = is_q ? q : k;
    const __nv_bfloat16* wn  = is_q ? q_norm_w : k_norm_w;
    const int head  = is_q ? h : (h - n_q_heads);
    const size_t base = (size_t)head * head_dim;
    const int half  = head_dim / 2;

    __shared__ float s_warp[32];
    extern __shared__ __nv_bfloat16 s_n[];              // head_dim bf16 (normalized head)

    // RMSNorm over this head — identical block reduction to rmsnorm_kernel (blockDim=256).
    float ss = 0.f;
    for (int c = threadIdx.x; c < head_dim; c += blockDim.x) {
        float v = __bfloat162float(buf[base + c]);
        ss += v * v;
    }
    ss = qkr_warp_sum(ss);
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
        v = qkr_warp_sum(v);
        if (threadIdx.x == 0) s_warp[0] = rsqrtf(v / head_dim + eps);
    }
    __syncthreads();
    const float inv_rms = s_warp[0];
    // Normalized head -> bf16 shared (the same value rmsnorm would have written to global).
    for (int c = threadIdx.x; c < head_dim; c += blockDim.x)
        s_n[c] = __float2bfloat16(__bfloat162float(buf[base + c]) * inv_rms * __bfloat162float(wn[c]));
    __syncthreads();

    // RoPE on the normalized head — identical math to rope_kernel.
    const float p = (float)positions[0];
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float freq = __powf(theta, -2.f * (float)i / (float)head_dim);
        const float ang  = p * freq;
        const float c = __cosf(ang), s = __sinf(ang);
        const float x0 = __bfloat162float(s_n[i]);
        const float x1 = __bfloat162float(s_n[i + half]);
        buf[base + i]        = __float2bfloat16(x0 * c - x1 * s);
        buf[base + i + half] = __float2bfloat16(x1 * c + x0 * s);
    }
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/attention.h"

void launch_rope(void* q, void* k, const int* positions,
                 int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
                 float theta, cudaStream_t stream) {
    const int half = head_dim / 2;
    dim3 gq(n_tokens, n_q_heads);
    rope_kernel<<<gq, half, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(q), positions, n_q_heads, head_dim, theta);
    dim3 gk(n_tokens, n_kv_heads);
    rope_kernel<<<gk, half, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(k), positions, n_kv_heads, head_dim, theta);
}

// Single-token decode: fuse q_norm + k_norm + rope(q) + rope(k) into one launch.
void launch_qknorm_rope(void* q, void* k, const void* q_norm_w, const void* k_norm_w,
                        const int* positions, int n_q_heads, int n_kv_heads,
                        int head_dim, float eps, float theta, cudaStream_t stream) {
    const int blocks  = n_q_heads + n_kv_heads;
    const size_t smem = (size_t)head_dim * sizeof(__nv_bfloat16);
    qknorm_rope_kernel<<<blocks, 256, smem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q), reinterpret_cast<__nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(q_norm_w), reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
        positions, n_q_heads, n_kv_heads, head_dim, eps, theta);
}
#endif

} // namespace kernels
} // namespace sparkinfer
