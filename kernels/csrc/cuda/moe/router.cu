// MoE router — top-k expert selection with sync-free on-device token counting.
//
// One warp per token. Logits are staged in shared memory; top-k is found by k
// passes of warp-wide arg-max (k is small: 8 for Qwen3.5/Gemma4). The per-expert
// token counter is bumped with atomicAdd and stays on the GPU, so the dispatch
// that follows needs no host synchronization and the whole pass is CUDA-graph
// capturable.
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

// Upper bound on top_k. The per-token selection holds the chosen logits/ids in
// fixed-size registers (sel_logit/sel_id below), so top_k must not exceed this —
// the launcher rejects larger values instead of overrunning those arrays.
static constexpr int MOE_ROUTER_MAX_TOPK = 16;

// Warp arg-max: returns the max value across the warp; *idx is set on every lane
// to the index that owns it (ties resolved to the lowest index).
__device__ __forceinline__ float warp_argmax(float val, int& idx) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float oval = __shfl_xor_sync(0xffffffff, val, off);
        int   oidx = __shfl_xor_sync(0xffffffff, idx, off);
        if (oval > val || (oval == val && oidx < idx)) { val = oval; idx = oidx; }
    }
    return val;
}

__global__ void moe_router_kernel(
    const float* __restrict__ logits,    // [num_tokens, num_experts]
    int*   __restrict__ expert_ids,      // [num_tokens, top_k]
    float* __restrict__ expert_weights,  // [num_tokens, top_k]
    int*   __restrict__ tokens_per_expert,
    int num_tokens, int num_experts, int top_k, int normalize
) {
    const int tok  = blockIdx.x;
    const int lane = threadIdx.x;          // 0..31
    if (tok >= num_tokens) return;

    extern __shared__ float s_logits[];    // [num_experts]
    const float* row = logits + (size_t)tok * num_experts;
    for (int e = lane; e < num_experts; e += 32) s_logits[e] = row[e];
    __syncwarp();

    float sel_logit[MOE_ROUTER_MAX_TOPK];  // top_k <= MOE_ROUTER_MAX_TOPK (launcher-enforced)
    int   sel_id[MOE_ROUTER_MAX_TOPK];

    for (int j = 0; j < top_k; j++) {
        float best = -1e30f; int best_i = -1;
        for (int e = lane; e < num_experts; e += 32) {
            float v = s_logits[e];
            if (v > best || (v == best && e < best_i)) { best = v; best_i = e; }
        }
        int idx = best_i;
        float mx = warp_argmax(best, idx);   // idx now holds the winning expert on all lanes
        sel_logit[j] = mx;
        sel_id[j]    = idx;
        if (lane == 0) s_logits[idx] = -1e30f;  // mask so next pass skips it
        __syncwarp();
    }

    // Weights: softmax over the selected top-k logits (or raw exp if not normalizing).
    float denom = 1.f;
    if (normalize) {
        float mx = sel_logit[0];
        for (int j = 1; j < top_k; j++) mx = fmaxf(mx, sel_logit[j]);
        denom = 0.f;
        for (int j = 0; j < top_k; j++) denom += __expf(sel_logit[j] - mx);
        // store normalized weights
        if (lane == 0) {
            for (int j = 0; j < top_k; j++) {
                expert_ids[tok * top_k + j]     = sel_id[j];
                expert_weights[tok * top_k + j] = __expf(sel_logit[j] - mx) / denom;
            }
        }
    } else if (lane == 0) {
        for (int j = 0; j < top_k; j++) {
            expert_ids[tok * top_k + j]     = sel_id[j];
            expert_weights[tok * top_k + j] = sel_logit[j];
        }
    }

    if (tokens_per_expert && lane == 0) {
        for (int j = 0; j < top_k; j++) atomicAdd(&tokens_per_expert[sel_id[j]], 1);
    }
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
void launch_moe_router(
    const float* logits, int* expert_ids, float* expert_weights,
    int* tokens_per_expert, int num_tokens, int num_experts, int top_k,
    int normalize, cudaStream_t stream
) {
    if (num_tokens <= 0 || num_experts <= 0 || top_k <= 0 || top_k > num_experts) return;
    if (top_k > MOE_ROUTER_MAX_TOPK) return;   // sel_logit/sel_id are sized to MOE_ROUTER_MAX_TOPK
    size_t smem = (size_t)num_experts * sizeof(float);
    moe_router_kernel<<<num_tokens, 32, smem, stream>>>(
        logits, expert_ids, expert_weights, tokens_per_expert,
        num_tokens, num_experts, top_k, normalize);
}
#endif

} // namespace kernels
} // namespace sparkinfer
