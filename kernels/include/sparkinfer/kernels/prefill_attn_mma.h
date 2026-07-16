// Tensor-core (int8 wmma) prefill attention for Qwythos (Qwen3.5) hd256 full-attention layers.
//
// See kernels/csrc/cuda/fused/prefill_attn_mma.cu for the design notes. The merged prefill
// attention (#455) removed the O(N^2) *bandwidth* problem by tiling the paged int8 KV into
// shared memory; what remains is a *compute* problem — it evaluates QK^T and PV with scalar
// FMA plus a 5-shuffle warp reduction per key, which measures ~8 TFLOP/s on sm_120. This
// translation unit runs the same masked online-softmax attention on the int8 tensor cores.
#pragma once

#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {

// Launch the int8 tensor-core prefill attention over the paged int8 KV pool. Signature mirrors
// `launch_prefill_attn_int8_paged` (#398) so it drops in as a one-line guard at the top of that
// launcher, ahead of the scalar windowed/tiled path (#455):
//
//     if (launch_prefill_attn_mma(...)) return;   // else fall through to the scalar kernels
//
// Returns true if a kernel was launched, false if the caller should run its own attention
// (shape not specialized, or disabled by env).
//
// Env knobs:
//   SPARKINFER_PREFILL_ATTN_MMA         (default 1)    0 disables (A/B) -> falls through to #455.
//   SPARKINFER_PREFILL_ATTN_MMA_MINCTX  (default 0)    only use MMA at n_tokens >= this.
// The sink+sliding-window selection is read from SPARKINFER_PREFILL_ATTN_WINDOW (default 256
// blocks) so this path stays numerically consistent with the merged windowed prefill (#455) and
// the merged sparse-KV decode (#379).
bool launch_prefill_attn_mma(
    const void* q, const signed char* k_pool, const signed char* v_pool,
    const void* k_scale, const void* v_scale, const int* block_table, void* attn,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, float scale, cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace sparkinfer
