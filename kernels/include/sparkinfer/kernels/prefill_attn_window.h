// Windowed / tiled prefill attention for Qwythos (Qwen3.5) long context.
// See kernels/csrc/cuda/fused/prefill_attn_window.cu for the design notes.
//
// Applies the merged decode sparse-KV window (StreamingLLM: sink block 0 + last
// `win_blocks` KV blocks, #379) to the batched prompt-prefill full-attention
// layers (paged int8 KV, PR #398), turning O(N^2) prompt attention into
// O(N * window) at long context with byte-identical online-softmax math.
#pragma once

#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {

// Launch the windowed (or smem-tiled) prefill attention over the paged int8 KV
// pool. Signature mirrors #398's `launch_prefill_attn_int8_paged` so it drops in
// as a one-line guard at the top of that launcher once #398 lands:
//
//     if (launch_prefill_attn_windowed(...)) return;   // else fall through to full attention
//
// Returns true if a kernel was launched, false if the caller should run its own
// attention (head_dim != 256, or both the window and tiling are disabled).
//
// Env knobs:
//   SPARKINFER_PREFILL_ATTN_WINDOW  (default 256) window size in KV blocks; 0 disables windowing.
//   SPARKINFER_PREFILL_ATTN_TILED   (default 0)   smem-tiled full attention when windowing is off.
bool launch_prefill_attn_windowed(
    const void* q, const signed char* k_pool, const signed char* v_pool,
    const void* k_scale, const void* v_scale, const int* block_table, void* attn,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, float scale, cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace sparkinfer
