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
#include <cstdlib>
#endif

namespace sparkinfer {
namespace kernels {

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

    float sel_logit[16];                   // top_k <= 16
    int   sel_id[16];

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

// Single-pass top-k: one thread per expert. Each thread counts how many experts
// outrank it (higher logit, or equal logit with a lower index — identical tie-break
// to moe_router_kernel's k-pass arg-max), giving its rank directly. Experts with
// rank < top_k are the selection, placed at slot == rank, so the output order
// (descending logit) and the softmax weights are bit-identical to the k-pass kernel,
// but the 8 serial arg-max passes collapse to one parallel comparison sweep.
__global__ void moe_router_kernel2(
    const float* __restrict__ logits, int* __restrict__ expert_ids,
    float* __restrict__ expert_weights, int* __restrict__ tokens_per_expert,
    int num_tokens, int num_experts, int top_k, int normalize
) {
    const int tok = blockIdx.x;
    const int e   = threadIdx.x;                 // one thread per expert
    if (tok >= num_tokens) return;
    extern __shared__ float s_logits[];          // [num_experts]
    __shared__ int   s_sel_id[16];               // top_k <= 16
    __shared__ float s_sel_logit[16];
    const float* rowp = logits + (size_t)tok * num_experts;
    if (e < num_experts) s_logits[e] = rowp[e];
    __syncthreads();

    if (e < num_experts) {
        const float my = s_logits[e];
        int rank = 0;
        for (int f = 0; f < num_experts; f++) {
            const float v = s_logits[f];
            if (v > my || (v == my && f < e)) rank++;
        }
        if (rank < top_k) { s_sel_id[rank] = e; s_sel_logit[rank] = my; }
    }
    __syncthreads();

    if (e == 0) {
        float denom = 1.f, mx = s_sel_logit[0];
        if (normalize) {
            for (int j = 1; j < top_k; j++) mx = fmaxf(mx, s_sel_logit[j]);
            denom = 0.f;
            for (int j = 0; j < top_k; j++) denom += __expf(s_sel_logit[j] - mx);
        }
        for (int j = 0; j < top_k; j++) {
            expert_ids[tok * top_k + j]     = s_sel_id[j];
            expert_weights[tok * top_k + j] = normalize ? __expf(s_sel_logit[j] - mx) / denom
                                                        : s_sel_logit[j];
        }
    }
    if (tokens_per_expert && e < num_experts) {
        // recompute membership cheaply (rank already known above only in the branch)
        const float my = s_logits[e]; int rank = 0;
        for (int f = 0; f < num_experts; f++) { const float v = s_logits[f];
            if (v > my || (v == my && f < e)) rank++; }
        if (rank < top_k) atomicAdd(&tokens_per_expert[e], 1);
    }
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
void launch_moe_router(
    const float* logits, int* expert_ids, float* expert_weights,
    int* tokens_per_expert, int num_tokens, int num_experts, int top_k,
    int normalize, cudaStream_t stream
) {
    if (num_tokens <= 0 || num_experts <= 0 || top_k <= 0 || top_k > num_experts) return;
    size_t smem = (size_t)num_experts * sizeof(float);
    // The rank-select router is still useful for larger/wider routing, but Qwen3.6 decode's
    // single-token 256-expert/top-8 case measured faster with the older k-pass warp kernel.
    // SPARKINFER_ROUTER2=1 forces rank-select; SPARKINFER_ROUTER2=0 forces the warp path.
    static int r2 = -1;
    if (r2 < 0) {
        const char* e = getenv("SPARKINFER_ROUTER2");
        r2 = e ? ((e[0] == '0') ? 0 : 1) : 2;
    }
    const bool qwen36_single = (num_tokens == 1 && num_experts == 256 && top_k == 8);
    if (r2 != 0 && !(r2 == 2 && qwen36_single) && top_k <= 16 && num_experts <= 1024) {
        const int bd = ((num_experts + 31) / 32) * 32;     // round up to a warp multiple
        moe_router_kernel2<<<num_tokens, bd, smem, stream>>>(
            logits, expert_ids, expert_weights, tokens_per_expert,
            num_tokens, num_experts, top_k, normalize);
        return;
    }
    moe_router_kernel<<<num_tokens, 32, smem, stream>>>(
        logits, expert_ids, expert_weights, tokens_per_expert,
        num_tokens, num_experts, top_k, normalize);
}
#endif

} // namespace kernels
} // namespace sparkinfer
