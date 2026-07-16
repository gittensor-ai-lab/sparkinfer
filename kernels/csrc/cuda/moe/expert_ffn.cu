// Fused MoE expert FFN with SwiGLU.
//
// One block per token. The block stages the token's hidden vector in shared
// memory, then loops over its top_k experts: for each it computes the SwiGLU
// intermediate h = SiLU(X@gate) * (X@up) into shared memory, projects it back
// through down, and accumulates w * y into a shared hidden-size accumulator.
// Output is written once at the end — no cross-block atomics, fully deterministic.
//
//   h = SiLU(X @ gate_w[e]) * (X @ up_w[e])   // [ffn_dim]
//   y = h @ down_w[e]                          // [hidden_dim]
//   out[i] = sum_j  weight_j * y_(expert_j)
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#include <cstdio>
#endif

namespace sparkinfer {
namespace kernels {

__device__ __forceinline__ float ffn_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// dynamic smem layout: [ s_x(hidden) | s_acc(hidden) | s_h(ffn) ]
__global__ void moe_expert_ffn_kernel(
    const __nv_bfloat16* __restrict__ input,    // [num_tokens, hidden]
    const __nv_bfloat16* __restrict__ gate_w,   // [num_experts, hidden, ffn]
    const __nv_bfloat16* __restrict__ up_w,     // [num_experts, hidden, ffn]
    const __nv_bfloat16* __restrict__ down_w,   // [num_experts, ffn, hidden]
    const int*   __restrict__ expert_ids,       // [num_tokens, top_k]
    const float* __restrict__ expert_weights,   // [num_tokens, top_k]
    __nv_bfloat16* __restrict__ output,         // [num_tokens, hidden]
    int num_tokens, int top_k, int hidden, int ffn
) {
    const int tok = blockIdx.x;
    if (tok >= num_tokens) return;
    const int tid = threadIdx.x;
    const int nth = blockDim.x;

    extern __shared__ float smem[];
    float* s_x   = smem;
    float* s_acc = s_x + hidden;
    float* s_h   = s_acc + hidden;

    for (int h = tid; h < hidden; h += nth) {
        s_x[h]   = ffn_to_f(input[(size_t)tok * hidden + h]);
        s_acc[h] = 0.f;
    }
    __syncthreads();

    for (int j = 0; j < top_k; j++) {
        const int   e = expert_ids[tok * top_k + j];
        const float w = expert_weights[tok * top_k + j];
        const __nv_bfloat16* gptr = gate_w + (size_t)e * hidden * ffn;
        const __nv_bfloat16* uptr = up_w   + (size_t)e * hidden * ffn;
        const __nv_bfloat16* dptr = down_w + (size_t)e * ffn * hidden;

        // h_f = SiLU(sum_h X[h]*gate[h,f]) * (sum_h X[h]*up[h,f])
        for (int f = tid; f < ffn; f += nth) {
            float g = 0.f, u = 0.f;
            for (int h = 0; h < hidden; h++) {
                const float x = s_x[h];
                g += x * ffn_to_f(gptr[(size_t)h * ffn + f]);
                u += x * ffn_to_f(uptr[(size_t)h * ffn + f]);
            }
            s_h[f] = silu(g) * u;
        }
        __syncthreads();

        // y_h = sum_f h_f * down[f,h];  acc += w * y_h
        for (int h = tid; h < hidden; h += nth) {
            float y = 0.f;
            for (int f = 0; f < ffn; f++) y += s_h[f] * ffn_to_f(dptr[(size_t)f * hidden + h]);
            s_acc[h] += w * y;
        }
        __syncthreads();
    }

    for (int h = tid; h < hidden; h += nth)
        output[(size_t)tok * hidden + h] = __float2bfloat16(s_acc[h]);
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
void launch_moe_expert_ffn(
    const void* input, const void* gate_w, const void* up_w, const void* down_w,
    const int* expert_ids, const float* expert_weights, void* output,
    int num_tokens, int top_k, int num_experts,
    int hidden_dim, int ffn_dim, cudaStream_t stream
) {
    (void)num_experts;
    size_t smem = (size_t)(2 * hidden_dim + ffn_dim) * sizeof(float);
    static size_t cur = 0;
    if (smem > 48 * 1024 && smem != cur) {
        cudaError_t e = cudaFuncSetAttribute(moe_expert_ffn_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (e != cudaSuccess) fprintf(stderr, "[moe_ffn] smem opt-in failed: %s\n", cudaGetErrorString(e));
        cur = smem;
    }
    moe_expert_ffn_kernel<<<num_tokens, 256, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input),
        reinterpret_cast<const __nv_bfloat16*>(gate_w),
        reinterpret_cast<const __nv_bfloat16*>(up_w),
        reinterpret_cast<const __nv_bfloat16*>(down_w),
        expert_ids, expert_weights, reinterpret_cast<__nv_bfloat16*>(output),
        num_tokens, top_k, hidden_dim, ffn_dim);
}
#endif

} // namespace kernels
} // namespace sparkinfer
