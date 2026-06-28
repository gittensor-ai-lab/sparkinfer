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

// Fused Q+K rope: ONE kernel over all (n_q_heads + n_kv_heads) heads with a flat
// 256-thread layout — 1 graph node instead of 2, and better occupancy than the
// head_dim/2-thread blocks. Mirrors llama's single rope_neox launch.
__global__ void rope_qk_kernel(
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k,
    const int* __restrict__ positions, int n_q_heads, int n_kv_heads, int head_dim, float theta
) {
    const int tok  = blockIdx.y;
    const int half = head_dim >> 1;
    const int total = (n_q_heads + n_kv_heads) * half;     // rotated pairs across Q|K
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    const int hh = gid / half, i = gid - hh * half;
    __nv_bfloat16* x; int head, nh;
    if (hh < n_q_heads) { x = q; head = hh;             nh = n_q_heads; }
    else                { x = k; head = hh - n_q_heads; nh = n_kv_heads; }
    const float p = (float)positions[tok];
    const float freq = __powf(theta, -2.f * (float)i / (float)head_dim);
    const float ang = p * freq;
    const float c = __cosf(ang), s = __sinf(ang);
    const size_t base = ((size_t)(tok * nh + head)) * head_dim;
    const float x0 = __bfloat162float(x[base + i]);
    const float x1 = __bfloat162float(x[base + i + half]);
    x[base + i]        = __float2bfloat16(x0 * c - x1 * s);
    x[base + i + half] = __float2bfloat16(x1 * c + x0 * s);
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/attention.h"
#include <cstdlib>

void launch_rope(void* q, void* k, const int* positions,
                 int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
                 float theta, cudaStream_t stream) {
    static int fuse = -1;   // default ON: fused Q+K rope (1 kernel). SPARKINFER_ROPEFUSE=0 disables
    if (fuse < 0) { const char* e = getenv("SPARKINFER_ROPEFUSE"); fuse = (e && e[0] == '0') ? 0 : 1; }
    if (fuse) {
        const int total = (n_q_heads + n_kv_heads) * (head_dim >> 1);
        dim3 grid((total + 255) / 256, n_tokens);
        rope_qk_kernel<<<grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(q), reinterpret_cast<__nv_bfloat16*>(k),
            positions, n_q_heads, n_kv_heads, head_dim, theta);
        return;
    }
    const int half = head_dim / 2;
    dim3 gq(n_tokens, n_q_heads);
    rope_kernel<<<gq, half, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(q), positions, n_q_heads, head_dim, theta);
    dim3 gk(n_tokens, n_kv_heads);
    rope_kernel<<<gk, half, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(k), positions, n_kv_heads, head_dim, theta);
}
#endif

} // namespace kernels
} // namespace sparkinfer
