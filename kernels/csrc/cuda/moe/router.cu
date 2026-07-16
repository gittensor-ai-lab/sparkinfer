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
#include <cuda_bf16.h>
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

#define SI_CEXG(av, ai, bv, bi) do {                                     \
    bool _ge = ((av) > (bv)) || ((av) == (bv) && (ai) < (bi));          \
    if (!_ge) { float _t = (av); (av) = (bv); (bv) = _t;                \
                int _u = (ai); (ai) = (bi); (bi) = _u; }               \
  } while (0)

// Warp-0 bitonic top-8 over 256 logits staged in shared memory. Lane owns experts {lane,lane+32,...
// lane+224}; per-lane sort to descending, then a 5-step reduction tree bitonic-merging sorted-8 lists.
// Writes out_ids[0..top_k) / out_w[0..top_k) (softmax when normalize). Must be entered by all of warp 0.
__device__ __forceinline__ void si_warp_bitonic_top8(
    const float* __restrict__ s_lg, int lane, int top_k, int normalize,
    int* __restrict__ out_ids, float* __restrict__ out_w, int* __restrict__ counts)
{
    float r[8]; int ri[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) { r[i] = s_lg[lane + 32 * i]; ri[i] = lane + 32 * i; }
    #pragma unroll
    for (int k = 2; k <= 8; k <<= 1)
        #pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1)
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int l = i ^ j;
                if (l > i) {
                    if ((i & k) == 0) SI_CEXG(r[i], ri[i], r[l], ri[l]);
                    else              SI_CEXG(r[l], ri[l], r[i], ri[i]);
                }
            }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float pr[8]; int pri[8];
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            pr[m]  = __shfl_down_sync(0xffffffffu, r[m],  off);
            pri[m] = __shfl_down_sync(0xffffffffu, ri[m], off);
        }
        float cv[16]; int cci[16];
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            cv[m] = r[m];          cci[m] = ri[m];
            cv[8 + m] = pr[7 - m]; cci[8 + m] = pri[7 - m];
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) SI_CEXG(cv[i], cci[i], cv[i + 8], cci[i + 8]);
        #pragma unroll
        for (int stride = 4; stride > 0; stride >>= 1)
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int l = i ^ stride;
                if (l > i && l < 8) SI_CEXG(cv[i], cci[i], cv[l], cci[l]);
            }
        #pragma unroll
        for (int m = 0; m < 8; m++) { r[m] = cv[m]; ri[m] = cci[m]; }
    }
    if (lane == 0) {
        float denom = 1.f, mx = r[0];                   // r sorted descending -> r[0] is the max
        if (normalize) { denom = 0.f; for (int j = 0; j < top_k; j++) denom += __expf(r[j] - mx); }
        for (int j = 0; j < top_k; j++) {
            out_ids[j] = ri[j];
            out_w[j]   = normalize ? __expf(r[j] - mx) / denom : r[j];
            if (counts) atomicAdd(&counts[ri[j]], 1);
        }
    }
}

// FUSED router GEMV + top-8: a faithful copy of gemv_f32_sk<float,SPL> (each block writes its RPB
// final logits) followed by grid completion (atomicInc self-resets the counter to 0 for the next
// CUDA-graph replay); the last block to arrive reads all 256 logits and runs the same bitonic top-8
// in-kernel. Removes the separate top-k launch and its dependent-load gap from the decode critical
// path. Byte-identical logits + selection to gemv_f32 + kernel6. Requires num_experts==256, K%8==0.
template <int SPL>
__global__ void moe_router_fused_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ W,
    float* __restrict__ logits, unsigned int* __restrict__ gridctr,
    int* __restrict__ expert_ids, float* __restrict__ expert_weights,
    int N, int K, int top_k, int normalize)
{
    constexpr int RPB = 8 / SPL;
    __shared__ float s_part[8 / SPL][SPL];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row_local = warp / SPL, split = warp % SPL;
    const int n = blockIdx.x * RPB + row_local;
    float acc = 0.f;
    if (n < N) {
        const uint4* row4 = reinterpret_cast<const uint4*>(W + (size_t)n * K);
        const uint4* x4   = reinterpret_cast<const uint4*>(x);
        const int n4 = K / 8;
        for (int i = split * 32 + lane; i < n4; i += SPL * 32) {
            uint4 wv = row4[i], xv = x4[i];
            const __nv_bfloat162* wh = reinterpret_cast<const __nv_bfloat162*>(&wv);
            const __nv_bfloat162* xh = reinterpret_cast<const __nv_bfloat162*>(&xv);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 wf = __bfloat1622float2(wh[j]), xf = __bfloat1622float2(xh[j]);
                acc += wf.x * xf.x + wf.y * xf.y;
            }
        }
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
        if (lane == 0) s_part[row_local][split] = acc;
    }
    __syncthreads();
    if (n < N && split == 0 && lane == 0) {
        float o = 0.f;
        #pragma unroll
        for (int s = 0; s < SPL; s++) o += s_part[row_local][s];
        logits[n] = o;
    }
    // grid completion: make this block's logits visible, then signal. atomicInc(.,B-1) wraps to 0
    // after B increments -> the counter is self-clearing for the next graph replay.
    __threadfence();
    __syncthreads();
    __shared__ unsigned int s_last;
    if (threadIdx.x == 0) s_last = atomicInc(gridctr, gridDim.x - 1);
    __syncthreads();
    if (s_last != gridDim.x - 1) return;
    __shared__ float s_lg[256];
    for (int e = threadIdx.x; e < N; e += blockDim.x) s_lg[e] = logits[e];
    __syncthreads();
    if (threadIdx.x < 32)
        si_warp_bitonic_top8(s_lg, threadIdx.x & 31, top_k, normalize, expert_ids, expert_weights, nullptr);
}
#undef SI_CEXG


#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
void launch_moe_router(
    const float* logits, int* expert_ids, float* expert_weights,
    int* tokens_per_expert, int num_tokens, int num_experts, int top_k,
    int normalize, cudaStream_t stream
) {
    if (num_tokens <= 0 || num_experts <= 0 || top_k <= 0 || top_k > num_experts) return;
    size_t smem = (size_t)num_experts * sizeof(float);
    // Default ON: single-pass rank-select top-k (one thread/expert). SPARKINFER_ROUTER2=0
    // restores the k-pass single-warp kernel. Falls back automatically if num_experts > 1024.
    static int r2 = -1;
    if (r2 < 0) { const char* e = getenv("SPARKINFER_ROUTER2"); r2 = (e && e[0] == '0') ? 0 : 1; }
    if (r2 && top_k <= 16 && num_experts <= 1024) {
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

// Fused single-token router: split-K GEMV (writes `logits` scratch) + in-kernel bitonic top-8 in
// the grid's last block. `gridctr` is a persistent unsigned counter, zero-initialized once by the
// caller; atomicInc self-resets it each call so it is CUDA-graph-replay safe. Requires num_experts
// == 256 and K % 8 == 0 (both hold for the Qwen3.6 router). Byte-identical to gemv_f32 + kernel6.
void launch_router_fused(const void* x, const void* W, float* logits, unsigned int* gridctr,
                         int* expert_ids, float* expert_weights,
                         int num_experts, int K, int top_k, int normalize, cudaStream_t stream) {
    constexpr int SPL = 4, RPB = 8 / SPL;              // GEMV_WPB = 8, matches gemv_f32_sk<float,4>
    dim3 grid((num_experts + RPB - 1) / RPB);
    moe_router_fused_kernel<SPL><<<grid, 8 * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(W),
        logits, gridctr, expert_ids, expert_weights, num_experts, K, top_k, normalize);
}
#endif

} // namespace kernels
} // namespace sparkinfer
