// Rotary position embedding (RoPE), HF "rotate-half" convention (GPT-NeoX style)
// as used by Qwen/Llama. Applied to Q and K after projection, before attention.
//
// inv_freq[i] = theta^(-2i/head_dim) is precomputed into __constant__ memory at
// model init (CUDA-graph safe — no per-thread __powf in the decode hot path).
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#include <cmath>
#endif

namespace sparkinfer {
namespace kernels {

__constant__ float c_rope_inv_freq[128];   // head_dim/2, max head_dim=256

// grid = (n_tokens, n_heads); blockDim = head_dim/2 threads (one per rotated pair).
__global__ void rope_kernel(
    __nv_bfloat16* __restrict__ x,        // [n_tokens, n_heads, head_dim]
    const int* __restrict__ positions,    // [n_tokens]
    int n_heads, int head_dim
) {
    const int tok  = blockIdx.x;
    const int head = blockIdx.y;
    const int i    = threadIdx.x;
    const int half = head_dim / 2;
    if (i >= half) return;

    const float p    = (float)positions[tok];
    const float freq = c_rope_inv_freq[i];
    const float ang  = p * freq;
    const float c = __cosf(ang), s = __sinf(ang);

    const size_t base = ((size_t)tok * n_heads + head) * head_dim;
    const float x0 = __bfloat162float(x[base + i]);
    const float x1 = __bfloat162float(x[base + i + half]);
    x[base + i]        = __float2bfloat16(x0 * c - x1 * s);
    x[base + i + half] = __float2bfloat16(x1 * c + x0 * s);
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/attention.h"

void upload_rope_inv_freq(float theta, int head_dim, cudaStream_t stream) {
    const int half = head_dim / 2;
    float host[128];
    for (int i = 0; i < half; i++)
        host[i] = powf(theta, -2.f * (float)i / (float)head_dim);
    cudaMemcpyToSymbolAsync(c_rope_inv_freq, host, (size_t)half * sizeof(float), 0,
                            cudaMemcpyHostToDevice, stream);
}

void launch_rope(void* q, void* k, const int* positions,
                 int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
                 float theta, cudaStream_t stream) {
    (void)theta;   // table uploaded via upload_rope_inv_freq before graph capture
    const int half = head_dim / 2;
    dim3 gq(n_tokens, n_q_heads);
    rope_kernel<<<gq, half, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(q), positions, n_q_heads, head_dim);
    dim3 gk(n_tokens, n_kv_heads);
    rope_kernel<<<gk, half, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(k), positions, n_kv_heads, head_dim);
}
#endif

} // namespace kernels
} // namespace sparkinfer
