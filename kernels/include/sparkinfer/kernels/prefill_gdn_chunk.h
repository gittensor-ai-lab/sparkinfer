// Chunk-parallel (WY / UT transform) Gated-DeltaNet prefill scan for Qwythos (Qwen3.5).
//
// See kernels/csrc/cuda/fused/prefill_gdn_chunk.cu for the derivation and design notes. The
// batched prompt prefill (#398) runs the gated delta rule as ONE sequential scan over all N
// prompt tokens (pf_gdn_scan_kernel): warp-per-state-column, one rank-1 state update per token,
// two 5-shuffle warp reductions per token on the critical path. It is the last sequential stage
// left in the batched prefill. Measured on an RTX 5090 (nsys, main @ cb34dc1, ctx=4096, reps-diff
// confirmed per-prefill): 59.3 ms = 20.7% of the 286.1 ms prefill, sustaining ~5.3 TFLOP/s
// (~5% of the fp32 peak) — it is not DRAM-bound (the tensors it touches need ~10.5 ms at
// 1792 GB/s) but L1/L2-bandwidth and serial-latency bound: every one of the 128 column-warps of
// a v-head re-loads the same k_t/q_t vector on every token.
//
// This translation unit rewrites the same recurrence in its chunk-parallel form, so the serial
// chain shortens from N to N/C and the per-chunk work becomes dense matmuls over shared-memory
// tiles that all state columns reuse.
#pragma once

#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {

// Launch the chunk-parallel Gated-DeltaNet prefill scan. Signature mirrors
// `launch_prefill_gdn_scan` (#398) so it drops in as a one-line guard at the top of that
// launcher, ahead of the sequential scan:
//
//     if (launch_prefill_gdn_chunk(...)) return;   // else fall through to the sequential scan
//
// Returns true if the kernels were launched, false if the caller should run its own scan (shape
// not specialized, disabled by env, or workspace allocation failed).
//
// Produces out[N, v_heads*head_dim] and leaves the recurrent state in the SAME transposed
// [v_head][col][row] layout the decode gdn_ar_fast kernel expects, so this is a drop-in
// replacement for the sequential scan and the decode path is untouched.
//
// Env knobs:
//   SPARKINFER_PREFILL_GDN_CHUNK         (default 1)   0 disables (A/B) -> sequential scan (#398).
//   SPARKINFER_PREFILL_GDN_CHUNK_MINCTX  (default 256) only chunk at n_tokens >= this; short
//                                                      prompts do not amortize the prep pass.
bool launch_prefill_gdn_chunk(const void* q, const void* k, const void* v,
                              const void* alpha, const void* beta,
                              const void* dt, const void* a,
                              float* state, void* out,
                              int n_tokens, int q_heads, int v_heads, int head_dim,
                              cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace sparkinfer
