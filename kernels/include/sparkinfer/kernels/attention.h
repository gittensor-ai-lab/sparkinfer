#pragma once
#include <cuda_runtime.h>

namespace sparkinfer { namespace kernels {

// Flash decode: single-token decode attention over paged KV cache.
// q:           [num_seqs, num_heads, head_dim]  (fp16/bf16)
// k_pool:      [num_blocks, block_size, num_kv_heads, head_dim]
// v_pool:      same shape as k_pool
// block_table: [num_seqs, max_blocks_per_seq]  (int32)
// seq_lens:    [num_seqs]  (int32)
// out:         [num_seqs, num_heads, head_dim]
void launch_flash_decode(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens,
    void* out,
    int num_seqs, int num_heads, int num_kv_heads,
    int head_dim, int block_size, int max_blocks_per_seq,
    float scale, cudaStream_t stream = nullptr
);

// Rotary position embedding (RoPE, HF rotate-half) applied in-place to Q and K
// after projection. positions: [n_tokens] (int32, device).
//   q: [n_tokens, n_q_heads, head_dim]   k: [n_tokens, n_kv_heads, head_dim]
void launch_rope(
    void* q, void* k, const int* positions,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    float theta, cudaStream_t stream = nullptr);

// Fused per-head QK-norm + RoPE + KV-append: one kernel replacing launch_rmsnorm_qk +
// launch_rope_kv_append. Per-head RMSNorm(q_w/k_w), then RoPE, then writes Q in place / K,V
// into the paged cache. positions == write_pos (decode). Value-identical to the two-kernel path.
void launch_qknorm_rope_kv_append(
    void* q, void* k, const void* v, const void* q_w, const void* k_w,
    void* k_pool, void* v_pool, const int* block_table, const int* positions,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim, float theta,
    float eps, int block_size, int max_blocks_per_seq, cudaStream_t stream = nullptr,
    void* k_scale = nullptr, void* v_scale = nullptr, int int8_kv = 0);

// Fused RoPE + paged KV-append: ropes Q in place, ropes K straight into k_pool, copies V into
// v_pool — one kernel replacing launch_rope + launch_kv_append. positions == write_pos (decode).
void launch_rope_kv_append(
    void* q, const void* k, const void* v, void* k_pool, void* v_pool,
    const int* block_table, const int* positions,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim, float theta,
    int block_size, int max_blocks_per_seq, cudaStream_t stream = nullptr);

// Fused QK-norm + partial-RoPE + KV-append for Qwen3.6 full-attn (gated, rope_dim < head_dim).
void launch_qknorm_rope_kv_partial(
    void* q, void* k, const void* v, const void* q_w, const void* k_w,
    void* k_pool, void* v_pool, const int* block_table, const int* positions,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim,
    float theta, float eps, int block_size, int max_blocks_per_seq, cudaStream_t stream = nullptr);

// Fused QK-norm + partial-RoPE + int8 KV-append for Qwen3.6 hd256 full-attn layers.
void launch_qknorm_rope_kv_partial_int8(
    void* q, void* k, const void* v, const void* q_w, const void* k_w,
    void* k_pool, void* v_pool, void* k_scale, void* v_scale,
    const int* block_table, const int* positions,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim,
    float theta, float eps, int block_size, int max_blocks_per_seq, cudaStream_t stream = nullptr);

// Gated Qwen3.6: fused split_q_gate + QK-norm + partial-RoPE + int8 KV-append.
void launch_qknorm_rope_kv_partial_int8_gated(
    const void* qraw, void* q, void* qgate, void* k, const void* v, const void* q_w, const void* k_w,
    void* k_pool, void* v_pool, void* k_scale, void* v_scale,
    const int* block_table, const int* positions,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim,
    float theta, float eps, int block_size, int max_blocks_per_seq, cudaStream_t stream = nullptr);

// Variant for models with partial rotary embeddings (e.g. Qwen3.6: 64 of 256
// dimensions rotate, the remaining per-head dimensions are copied unchanged).
void launch_rope_kv_append_partial(
    void* q, const void* k, const void* v, void* k_pool, void* v_pool,
    const int* block_table, const int* positions,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim,
    float theta, int block_size, int max_blocks_per_seq, cudaStream_t stream = nullptr);
// int8-KV variant: K/V appended as int8 (per-(token,kv_head) fp16 scale), Q RoPE'd bf16 in-place.
void launch_rope_kv_append_partial_int8(
    void* q, const void* k, const void* v, void* k_pool, void* v_pool, void* k_scale, void* v_scale,
    const int* block_table, const int* positions,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim,
    float theta, int block_size, int max_blocks_per_seq, cudaStream_t stream = nullptr);

// Flash prefill: full causal attention for prompt processing.
// q/k/v:  [batch, seqlen, num_heads, head_dim]
// out:    same shape as q
void launch_flash_prefill(
    const void* q, const void* k, const void* v,
    void* out,
    int batch, int seqlen_q, int seqlen_kv,
    int num_heads, int num_kv_heads, int head_dim,
    float scale, bool causal, cudaStream_t stream = nullptr
);

// --- Qwen3.5-35B-A3B specialized kernels ---

// Flash decode for 8:1 GQA (16 Q heads / 2 KV heads), head_dim=128.
// 8 warps per CTA share one KV tile load.
void launch_flash_decode_gqa8(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens,
    void* out,
    int num_seqs, int num_kv_heads,
    int head_dim, int block_size, int max_blocks_per_seq,
    float scale, cudaStream_t stream = nullptr
);

// --- Gemma 4 26B-A4B specialized kernels ---

// Flash decode for LOCAL layers: sliding-window (1024 tokens), head_dim=256, GQA 2:1.
// 2 warps per CTA (one per Q-head pair), KV capped at window_size blocks.
void launch_flash_decode_local_hd256(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens,
    void* out,
    int num_seqs, int num_kv_heads,
    int block_size, int max_blocks_per_window,
    float scale, cudaStream_t stream = nullptr
);

// Flash-decoding (KV-split) for decode: one block per (seq, q_head, split) for
// high SM occupancy and long-context scaling, then a combine pass. Fixed grid
// (seq_len read in-kernel) so it is CUDA-graph capturable. head_dim=128.
//   part_m/part_l: [num_seqs*num_q_heads*n_splits] (fp32 scratch)
//   part_acc:      [num_seqs*num_q_heads*n_splits*head_dim] (fp32 scratch)
// out_q8 (optional): when non-null, the combine also emits Q8_1(out) into it (one si_block_q8_1
// per 32 attn dims), so the O-projection MMVQ can skip its standalone attn-quantize node.
void launch_flash_decode_split(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens, void* out,
    float* part_m, float* part_l, float* part_acc,
    int num_seqs, int num_q_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks, int n_splits, float scale,
    cudaStream_t stream = nullptr, void* out_q8 = nullptr, int seqlen = -1,
    const void* k_scale = nullptr, const void* v_scale = nullptr, int int8_kv = 0,
    const void* attn_gate = nullptr);

// Flash decode for GLOBAL layers: full context, head_dim=512, GQA 8:1.
// Two-phase dot product splits 512-dim head into two 256-dim halves.
// NOTE: No public FlashInfer/vLLM kernel handles head_dim=512 — this is novel.
// 8 warps per CTA (one per Q-head in GQA group), 64 KB smem (fits Blackwell).
void launch_flash_decode_global_hd512(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens,
    void* out,
    int num_seqs, int num_kv_heads,
    int block_size, int max_blocks_per_seq,
    float scale, cudaStream_t stream = nullptr
);

void launch_fa_combine_hd256(const float* part_m, const float* part_l, const float* part_acc,
    void* out, int num_q_heads, int n_splits, void* out_q8, cudaStream_t stream = nullptr,
    const void* attn_gate = nullptr);

// Sink + sliding-window sparse-KV (Qwythos GQA-4 hd256). Default on; SPARKINFER_SPARSE_KV=0 disables.
void launch_fa_kv_window_select(const int* seq_lens, int* sel_blk, int num_kv_heads,
    int block_size, int n_sel, int window_w, cudaStream_t stream = nullptr);
void launch_flash_decode_split_sparse(const void* q, const void* k_pool_layer, const void* v_pool_layer,
    const int* block_table, const int* seq_lens, const int* sel_blk, float* part_m, float* part_l,
    float* part_acc, int num_q_heads, int num_kv_heads, int head_dim, int block_size, int max_blocks,
    int n_splits, int n_sel, float scale, const void* k_scale_layer, const void* v_scale_layer,
    cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
