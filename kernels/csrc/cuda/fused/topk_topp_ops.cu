// top_k / top_p (nucleus) truncation sampling: a full descending sort of the vocab-sized logits
// row (CUB radix sort, keys=logits values=vocab-index) + cumulative-softmax scan (CUB inclusive
// sum) to find the top_p cutoff, then a mask+scatter kernel that writes -infinity back onto the
// original vocab-indexed logits array for everything outside the surviving set. See fused.h for
// the full design rationale (why this always processes the whole vocab regardless of top_k/top_p,
// and why it must always be launched unconditionally on a CUDA-graph-captured decode path).

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "sparkinfer/kernels/fused.h"

namespace sparkinfer {
namespace kernels {

namespace {

// Fixed grid for the deterministic normalizer sum -- the shape IS part of the summation order, so
// it must not vary with vocab, occupancy or anything else.
constexpr int kDenomThreads = 256;
constexpr int kDenomBlocks = 256;

__global__ void vocab_iota_kernel(int* vocab_iota, int vocab) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vocab; v += gridDim.x * blockDim.x)
        vocab_iota[v] = v;
}

// sorted_exp[i] = exp(sorted_logits[i] - sorted_logits[0]) -- sorted_logits[0] is the row max
// (descending sort), so this is a numerically-stable softmax numerator with no separate
// max-reduction pass needed. Also scatters the inverse permutation rank_by_id[sorted_idx[i]] = i
// -- free, since this loop already touches every rank exactly once and sorted_idx is a bijection
// over [0,vocab) after a full sort (every entry of rank_by_id is written exactly once per call,
// so no memset/stale-entry hazard is possible). Used by extract_chosen_logit_kernel below to find
// the raw (pre-mask, pre-temperature-noise) logit of whichever token launch_argmax later picks --
// needed for logprobs/top_logprobs, since the sampled token is not necessarily rank 0 once real
// temperature sampling is active.
__global__ void topk_topp_exp_kernel(const float* __restrict__ sorted_logits,
                                     const int* __restrict__ sorted_idx,
                                     float* __restrict__ sorted_exp,
                                     int* __restrict__ rank_by_id, int vocab) {
    const float row_max = sorted_logits[0];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vocab; i += gridDim.x * blockDim.x) {
        sorted_exp[i] = __expf(sorted_logits[i] - row_max);
        rank_by_id[sorted_idx[i]] = i;
    }
}

// One block; grid-stride over sorted rank. Reads the per-call top_k/top_p device scalars and
// scatters -infinity into the ORIGINAL vocab-indexed logits array (via sorted_idx) for every
// rank outside the surviving set. Rank 0 is always kept by construction -- see fused.h.
__global__ void topk_topp_mask_kernel(float* __restrict__ logits,
                                      const int* __restrict__ sorted_idx,
                                      const float* __restrict__ topk_cumsum,
                                      int vocab, const int* __restrict__ top_k_i32,
                                      const float* __restrict__ top_p_f32) {
    int effective_k = *top_k_i32;
    if (effective_k <= 0 || effective_k >= vocab) effective_k = vocab;
    if (effective_k < 1) effective_k = 1;  // defensive floor -- never trust the input blindly

    const float top_p = *top_p_f32;
    // OpenAI accepts top_p=0. Keep rank 0 unconditionally and mask every lower rank in that
    // case, rather than treating zero as the old internal "disabled" sentinel.
    const bool top_p_active = top_p >= 0.f && top_p < 1.f;
    const float total_topk = topk_cumsum[effective_k - 1];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vocab; i += gridDim.x * blockDim.x) {
        bool keep;
        if (i >= effective_k) {
            keep = false;
        } else if (i == 0 || !top_p_active) {
            keep = true;
        } else {
            keep = topk_cumsum[i - 1] < top_p * total_topk;
        }
        if (!keep) logits[sorted_idx[i]] = -INFINITY;
    }
}

// Single-thread lookup of the chosen token's raw (pre-mask, pre-temperature-noise) logit via the
// inverse permutation topk_topp_exp_kernel already scattered. Always launched, negligible cost --
// see fused.h for why this must run unconditionally regardless of whether logprobs was requested.
__global__ void extract_chosen_logit_kernel(const int* __restrict__ out_id,
                                            const int* __restrict__ rank_by_id,
                                            const float* __restrict__ sorted_logits,
                                            float* __restrict__ chosen_logit) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *chosen_logit = sorted_logits[rank_by_id[*out_id]];
}

// logits[v] -= frequency_penalty * counts[v] + presence_penalty * (counts[v] > 0 ? 1 : 0), for
// every v in [0, vocab). OpenAI's exact presence/frequency-penalty formula (counts[v] = number of
// times vocab id v has appeared in THIS request's generated completion so far, accumulated across
// prior decode steps by increment_penalty_count_kernel below -- NOT the prompt). Always launched
// unconditionally on the CUDA-graph-captured decode path -- presence_penalty_f32/
// frequency_penalty_f32 are read from device memory refreshed before every forward_token() call,
// since this node's topology is frozen at capture time and may be replayed for a LATER, different
// request with different penalty values (e.g. via use_prefix_session's shared seq_id=0). At
// presence_penalty==0 && frequency_penalty==0 this is a pure subtract-zero pass -- still a
// full-vocab memory-bound cost, no way to skip it without a host-side branch inside a captured
// graph (see fused.h's documented dead end on conditional graph nodes for why that path was
// abandoned for top_k/top_p and is not revisited here).
__global__ void apply_presence_frequency_penalty_kernel(float* __restrict__ logits,
                                                         const int* __restrict__ counts, int vocab,
                                                         const float* __restrict__ presence_penalty_f32,
                                                         const float* __restrict__ frequency_penalty_f32) {
    const float pp = *presence_penalty_f32, fp = *frequency_penalty_f32;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vocab; v += gridDim.x * blockDim.x) {
        const int c = counts[v];
        logits[v] -= fp * (float)c + pp * (c > 0 ? 1.f : 0.f);
    }
}

// counts[chosen_id] += 1 -- records that this decode step's sampled token was chosen_id, so the
// NEXT decode step's apply_presence_frequency_penalty_kernel sees the updated running count.
// <<<1,1>>>, negligible cost, mirrors extract_chosen_logit_kernel's shape. Always launched
// unconditionally (same graph-replay-safety rationale) -- this unconditional +=1 on session 0's
// shared buffer is exactly what makes a fresh per-request reset (Qwen35Model::
// reset_penalty_counts) load-bearing: without it, this buffer would keep accumulating counts
// across every request that has ever reused session 0.
__global__ void increment_penalty_count_kernel(int* __restrict__ counts, const int* __restrict__ out_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) counts[*out_id] += 1;
}

// logit_bias[v] += bias -- always launched unconditionally on the CUDA-graph-captured decode path,
// same rationale as apply_presence_frequency_penalty_kernel: the graph may replay for a LATER,
// different request via use_prefix_session's shared seq_id=0, so this must read whatever is
// CURRENTLY in the fixed-address `bias` buffer rather than being host-gated on this call's own
// request. Unlike presence/frequency penalty, `bias` is not refreshed every decode step -- it is
// set ONCE per request (Qwen35Model::set_logit_bias, outside this graph) via
// scatter_logit_bias_kernel below, then stays constant for the rest of the request's decode. At
// logit_bias unset, `bias` is all-zero and this is a pure add-zero pass -- same always-on cost
// story as the penalty kernel.
__global__ void apply_logit_bias_kernel(float* __restrict__ logits, const float* __restrict__ bias,
                                        int vocab) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < vocab; v += gridDim.x * blockDim.x)
        logits[v] += bias[v];
}

// bias[ids[i]] = vals[i] for i in [0,k) -- NOT part of the captured decode graph; called once per
// request from Qwen35Model::set_logit_bias, before the request's first forward_token(). The bound
// check is a defensive backstop (the real validation is parse_request_controls, which has access
// to the real vocab size); an out-of-range id here is silently dropped rather than corrupting
// adjacent device memory.
__global__ void scatter_logit_bias_kernel(float* __restrict__ bias, const int* __restrict__ ids,
                                          const float* __restrict__ vals, int k, int vocab) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
        const int id = ids[i];
        if (id >= 0 && id < vocab) bias[id] = vals[i];
    }
}

}  // namespace

size_t topk_sort_temp_storage_bytes(int vocab) {
    size_t bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending<float, int>(
        nullptr, bytes, nullptr, nullptr, nullptr, nullptr, vocab);
    return bytes;
}

size_t topk_scan_temp_storage_bytes(int vocab) {
    size_t bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, bytes, static_cast<float*>(nullptr),
                                  static_cast<float*>(nullptr), vocab);
    return bytes;
}

void launch_vocab_iota_init(int* vocab_iota, int vocab, cudaStream_t stream) {
    const int bx = (vocab + 255) / 256 > 1024 ? 1024 : (vocab + 255) / 256;
    vocab_iota_kernel<<<bx < 1 ? 1 : bx, 256, 0, stream>>>(vocab_iota, vocab);
}

// Fixed-order full-vocab sum of the softmax numerators, for deterministic mode.
//
// The default path reads the normalizer off the END of the CUB inclusive scan that top_p already
// needs (topk_cumsum[vocab-1]), which is free. But CUB's DeviceScan uses decoupled look-back: how
// many predecessor tile aggregates a tile folds together before it finds an inclusive prefix
// depends on how far along those predecessors happen to be, i.e. on TIMING. For a non-associative
// operator like fp32 addition that makes the grouping -- and so the last-place bit of the total --
// vary run to run. It is exactly one ULP, but it lands in logsumexp, which every reported logprob
// at that position is measured against, so it is visible in every logprob and it is the last thing
// standing between deterministic mode and bit-identical output.
//
// This is the same sum with the summation tree pinned: a fixed grid, each block folding a fixed
// strided subset in ascending index, then block partials folded in ascending block index by a
// single block. No atomics, no look-back, no timing dependence.
__global__ void logprob_denom_partial_kernel(const float* __restrict__ vals, int n,
                                             float* __restrict__ partials) {
    __shared__ float sm[kDenomThreads];
    float acc = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        acc += vals[i];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = sm[0];
}

__global__ void logprob_denom_final_kernel(const float* __restrict__ partials, int n,
                                           float* __restrict__ out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float acc = 0.f;
    for (int i = 0; i < n; i++) acc += partials[i];
    *out = acc;
}

void launch_logprob_denom_det(const float* topk_exp, int vocab, float* partials, float* out,
                              cudaStream_t stream) {
    logprob_denom_partial_kernel<<<kDenomBlocks, kDenomThreads, 0, stream>>>(topk_exp, vocab, partials);
    logprob_denom_final_kernel<<<1, 1, 0, stream>>>(partials, kDenomBlocks, out);
}

void launch_logprob_distribution(const float* logits, int vocab,
                                 const int* vocab_iota, float* sorted_logits, int* sorted_idx,
                                 float* topk_exp, float* topk_cumsum,
                                 void* sort_temp, size_t sort_temp_bytes,
                                 void* scan_temp, size_t scan_temp_bytes,
                                 int* rank_by_id,
                                 cudaStream_t stream) {
    size_t sort_bytes = sort_temp_bytes;
    cub::DeviceRadixSort::SortPairsDescending<float, int>(
        sort_temp, sort_bytes, logits, sorted_logits, vocab_iota, sorted_idx, vocab, 0,
        sizeof(float) * 8, stream);

    const int bx = (vocab + 255) / 256 > 1024 ? 1024 : (vocab + 255) / 256;
    topk_topp_exp_kernel<<<bx < 1 ? 1 : bx, 256, 0, stream>>>(sorted_logits, sorted_idx, topk_exp,
                                                              rank_by_id, vocab);

    size_t scan_bytes = scan_temp_bytes;
    cub::DeviceScan::InclusiveSum(scan_temp, scan_bytes, topk_exp, topk_cumsum, vocab, stream);
}

void launch_topk_topp_mask(float* logits, int vocab,
                           const int* vocab_iota, float* sorted_logits, int* sorted_idx,
                           float* topk_exp, float* topk_cumsum,
                           void* sort_temp, size_t sort_temp_bytes,
                           void* scan_temp, size_t scan_temp_bytes,
                           const int* top_k_i32, const float* top_p_f32,
                           int* rank_by_id,
                           cudaStream_t stream) {
    // Sort + exp/rank + cumsum, then mask. Split out so the prefill seed token can reuse the
    // distribution half verbatim -- see launch_logprob_distribution's doc comment in fused.h.
    launch_logprob_distribution(logits, vocab, vocab_iota, sorted_logits, sorted_idx, topk_exp,
                                topk_cumsum, sort_temp, sort_temp_bytes, scan_temp,
                                scan_temp_bytes, rank_by_id, stream);

    const int bx = (vocab + 255) / 256 > 1024 ? 1024 : (vocab + 255) / 256;
    topk_topp_mask_kernel<<<bx < 1 ? 1 : bx, 256, 0, stream>>>(
        logits, sorted_idx, topk_cumsum, vocab, top_k_i32, top_p_f32);
}

void launch_extract_chosen_logit(const int* out_id, const int* rank_by_id,
                                 const float* sorted_logits, float* chosen_logit,
                                 cudaStream_t stream) {
    extract_chosen_logit_kernel<<<1, 1, 0, stream>>>(out_id, rank_by_id, sorted_logits, chosen_logit);
}

void launch_presence_frequency_penalty(float* logits, const int* counts, int vocab,
                                       const float* presence_penalty_f32,
                                       const float* frequency_penalty_f32,
                                       cudaStream_t stream) {
    const int bx = (vocab + 255) / 256 > 1024 ? 1024 : (vocab + 255) / 256;
    apply_presence_frequency_penalty_kernel<<<bx < 1 ? 1 : bx, 256, 0, stream>>>(
        logits, counts, vocab, presence_penalty_f32, frequency_penalty_f32);
}

void launch_increment_penalty_count(int* counts, const int* out_id, cudaStream_t stream) {
    increment_penalty_count_kernel<<<1, 1, 0, stream>>>(counts, out_id);
}

void launch_logit_bias(float* logits, const float* bias, int vocab, cudaStream_t stream) {
    const int bx = (vocab + 255) / 256 > 1024 ? 1024 : (vocab + 255) / 256;
    apply_logit_bias_kernel<<<bx < 1 ? 1 : bx, 256, 0, stream>>>(logits, bias, vocab);
}

void launch_scatter_logit_bias(float* bias, const int* ids, const float* vals, int k, int vocab,
                               cudaStream_t stream) {
    if (k <= 0) return;
    const int bx = (k + 255) / 256 > 1024 ? 1024 : (k + 255) / 256;
    scatter_logit_bias_kernel<<<bx < 1 ? 1 : bx, 256, 0, stream>>>(bias, ids, vals, k, vocab);
}

}  // namespace kernels
}  // namespace sparkinfer
