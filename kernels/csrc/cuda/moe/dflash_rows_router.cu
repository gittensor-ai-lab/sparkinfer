// Row-batched MoE router for the DFlash verify window.
//
// One launch routes all `rows` proposal tokens of a verify window: the split-K GEMV of
// moe_router_fused_kernel gains a grid dimension over tokens, so the [num_experts, K]
// router weight is streamed for the whole window instead of once per single-token launch,
// and the top-8 selection still happens in-kernel (no separate top-k launch, no dependent
// -load gap) for every token.
//
// The arithmetic is the single-token kernel's, unchanged: each token owns its own blocks,
// its own accumulators and its own shared split-K partials, and each output logit is
// summed over the same K-stride in the same order and reduced by the same shfl_xor tree.
// Selection runs the same bitonic top-8 over the same 256 staged logits. A call with
// rows=R is therefore bit-identical to R calls of launch_router_fused, which is what the
// DFlash accuracy gate (DFlash output == autoregressive output, token for token) needs:
// the routed expert set and its softmax weights feed the MoE FFN, so any drift here moves
// the logits and eventually flips an argmax.
//
// Bit-exactness binds the build as well as the source: this TU must carry the same flags as
// router.cu (-O3 --use_fast_math, as every si_* kernel target does). --use_fast_math makes the
// `/ denom` of the softmax an approximate divide and flushes denormals, so a target that omits
// it yields different expert weights from the single-token path even with identical source.
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

// Per-TU copies of the router.cu selection helpers (house style: small __device__ helpers
// are duplicated rather than shared through a header). They are reproduced verbatim —
// the compare-exchange predicate below fixes both the ordering (descending logit) and the
// tie-break (lower expert index wins), and the softmax op sequence at the end fixes the
// weights, so neither may be rewritten.
#define DFR_CEXG(av, ai, bv, bi) do {                                     \
    bool _ge = ((av) > (bv)) || ((av) == (bv) && (ai) < (bi));          \
    if (!_ge) { float _t = (av); (av) = (bv); (bv) = _t;                \
                int _u = (ai); (ai) = (bi); (bi) = _u; }               \
  } while (0)

// Warp-0 bitonic top-8 over 256 logits staged in shared memory. Lane owns experts {lane,lane+32,...
// lane+224}; per-lane sort to descending, then a 5-step reduction tree bitonic-merging sorted-8 lists.
// Writes out_ids[0..top_k) / out_w[0..top_k) (softmax when normalize). Must be entered by all of warp 0.
__device__ __forceinline__ void dfr_warp_bitonic_top8(
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
                    if ((i & k) == 0) DFR_CEXG(r[i], ri[i], r[l], ri[l]);
                    else              DFR_CEXG(r[l], ri[l], r[i], ri[i]);
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
        for (int i = 0; i < 8; i++) DFR_CEXG(cv[i], cci[i], cv[i + 8], cci[i + 8]);
        #pragma unroll
        for (int stride = 4; stride > 0; stride >>= 1)
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int l = i ^ stride;
                if (l > i && l < 8) DFR_CEXG(cv[i], cci[i], cv[l], cci[l]);
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

// Row-batched fused router GEMV + top-8. blockIdx.x walks the expert rows exactly as in the
// single-token kernel (RPB logits per block, SPL warps cooperating per logit); blockIdx.y
// selects the proposal token, which offsets x / logits / expert_ids / expert_weights but NOT
// W — the weight is what the window shares.
//
// The row count is not a template parameter here: a token's work lives entirely in its own
// blocks, so there is nothing per-row to keep in registers and no gain from specializing.
//
// Grid completion is per token. Each token's gridDim.x blocks increment gridctr[blockIdx.y];
// a single shared counter would let blocks of different tokens elect each other and the
// elected block would read logits whose GEMV has not landed. atomicInc(., gridDim.x - 1)
// wraps to 0 after gridDim.x increments, so every counter is self-clearing for the next
// CUDA-graph replay. The __threadfence() must precede the atomicInc so this block's logits
// are visible to whichever block is elected; the __syncthreads() after it holds thread 0
// back until every thread here has fenced.
//
// Requires N == 256 (the s_lg staging array and the lane-owns-8-experts bitonic layout) and
// K % 8 == 0 (uint4 loads), both of which hold for the Qwen3.6 router.
template <int SPL>
__global__ void dflash_rows_router_fused_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ W,
    float* __restrict__ logits, unsigned int* __restrict__ gridctr,
    int* __restrict__ expert_ids, float* __restrict__ expert_weights,
    int N, int K, int top_k, int normalize)
{
    constexpr int RPB = 8 / SPL;
    __shared__ float s_part[8 / SPL][SPL];
    const int tok = blockIdx.y;
    const __nv_bfloat16* __restrict__ x_t = x + (size_t)tok * K;
    float* __restrict__ logits_t = logits + (size_t)tok * N;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row_local = warp / SPL, split = warp % SPL;
    const int n = blockIdx.x * RPB + row_local;
    float acc = 0.f;
    if (n < N) {
        const uint4* row4 = reinterpret_cast<const uint4*>(W + (size_t)n * K);
        const uint4* x4   = reinterpret_cast<const uint4*>(x_t);
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
        logits_t[n] = o;
    }
    __threadfence();
    __syncthreads();
    __shared__ unsigned int s_last;
    if (threadIdx.x == 0) s_last = atomicInc(&gridctr[tok], gridDim.x - 1);
    __syncthreads();
    if (s_last != gridDim.x - 1) return;
    __shared__ float s_lg[256];
    for (int e = threadIdx.x; e < N; e += blockDim.x) s_lg[e] = logits_t[e];
    __syncthreads();
    if (threadIdx.x < 32)
        dfr_warp_bitonic_top8(s_lg, threadIdx.x & 31, top_k, normalize,
                             expert_ids + (size_t)tok * top_k,
                             expert_weights + (size_t)tok * top_k, nullptr);
}
#ifndef _MSC_VER
template __global__ void dflash_rows_router_fused_kernel<4>(
    const __nv_bfloat16*, const __nv_bfloat16*, float*, unsigned int*, int*, float*, int, int, int, int);
#endif
#undef DFR_CEXG

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
// `gridctr` is a persistent unsigned counter array of `rows` entries, zero-initialized once by
// the caller; atomicInc self-resets each entry every call, so the launch is CUDA-graph-replay
// safe and needs no memset node.
//
// Bails (does nothing, leaving the outputs untouched) on every configuration this kernel does
// not compute exactly; the caller must then fall back to `rows` single-token launch_router_fused
// calls. num_experts must be exactly 256, not merely at most 256: the bitonic top-8 reads
// s_lg[lane + 32*i] for all i < 8 unconditionally, so a shorter row leaves the tail of the
// staging array uninitialized and lets garbage take a top-k slot. top_k must be <= 8 because the
// selection carries eight registers per lane and indexes them by j < top_k. K % 8 == 0 is what
// makes the uint4 loads legal, and here it is harder than in the single-token kernel: x + tok*K
// is 16B-aligned only when K % 8 == 0, so row 1 of the window would fault outright rather than
// silently drop a K tail. These are the same conditions the single-token call site already gates
// launch_router_fused on (n_experts == 256 && hidden % 8 == 0).
void launch_dflash_rows_router_fused(const void* x, const void* W, float* logits,
                                     unsigned int* gridctr, int* expert_ids,
                                     float* expert_weights, int num_experts, int K,
                                     int top_k, int normalize, int rows,
                                     cudaStream_t stream) {
    if (rows < 1 || rows > 8) return;
    if (num_experts != 256) return;
    if (top_k < 1 || top_k > 8) return;
    if (K < 8 || (K % 8) != 0) return;
    constexpr int SPL = 4, RPB = 8 / SPL;              // GEMV_WPB = 8, matches gemv_f32_sk<float,4>
    dim3 grid((num_experts + RPB - 1) / RPB, rows);
    dflash_rows_router_fused_kernel<SPL><<<grid, 8 * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(W),
        logits, gridctr, expert_ids, expert_weights, num_experts, K, top_k, normalize);
}
#endif

} // namespace kernels
} // namespace sparkinfer
