#pragma once
// Internal CUDA helpers for DFlash draft attention / RoPE / SwiGLU.

#include <cuda_runtime.h>

namespace sparkinfer {
namespace dflash_kernels {

// Upper bounds for the hd128 row-batched KV-split attention path, so the caller can size
// its per-split partial-state scratch once at load instead of per call.
constexpr int kDFlashAttnMaxSplits = 16;
constexpr int kDFlashAttnMaxRows = 16;
// Below this key count the split path is not worth taking: the unsplit kernel is already short
// enough that the extra combine pass eats the gain, and re-associating the reduction perturbs the
// draft's proposals -- at a 512-token context that cost more mean acceptance than the kernel saved
// (2.0317 -> 2.0000, net -0.7%). Long contexts, where the unsplit kernel is ~25x off roofline, are
// the case this exists for; everything shorter stays bit-for-bit on the original kernel.
constexpr int kDFlashAttnMinKv = 1024;

// GQA attention. q: [q_len, n_q, d], k/v: [kv_len, n_kv, d], out: [q_len, n_q, d] bf16.
// q_pos0 / k_pos0 are absolute positions of index 0. If window > 0, mask keys with
// (q_pos - k_pos) >= window (sliding window). scale = 1/sqrt(d).
// fa_m / fa_l / fa_acc: optional partial-state scratch for the hd128 KV-split path, sized
// [kDFlashAttnMaxRows * n_q * kDFlashAttnMaxSplits] (and * 128 for fa_acc). Pass nullptr to
// force the single-CTA-per-row kernel.
void launch_attn_gqa(const void* q, const void* k, const void* v, void* out,
                     int q_len, int kv_len, int n_q, int n_kv, int d,
                     int q_pos0, int k_pos0, int window, bool causal, float scale,
                     cudaStream_t stream, float* fa_m = nullptr, float* fa_l = nullptr,
                     float* fa_acc = nullptr);

// First key index launch_attn_gqa will actually read for a windowed layer, given a block whose
// smallest query position is q_pos0; 0 when it will read from the start. Everything below the
// return value is masked for every row of the block AND excluded from the split partition, so a
// caller that fills the KV cache may leave those rows unwritten. Exposed rather than left inline in
// the launcher so producer and consumer derive the bound from one place and cannot disagree -- a
// row the producer skipped but the kernel still read would be uninitialized memory.
int attn_gqa_kv_lo(int q_len, int kv_len, int n_q, int n_kv, int d,
                   int q_pos0, int k_pos0, int window);

// In-place RoPE on [seq, n_heads, d] bf16. positions[i] = pos0 + i.
void launch_rope_seq(void* x, int seq, int n_heads, int d, int pos0,
                     float theta, cudaStream_t stream);

// out[i] = silu(gate[i]) * up[i]  for n elements (bf16).
void launch_swiglu(const void* gate, const void* up, void* out, int n,
                   cudaStream_t stream);

// out[r, :] = x[r, :] + y[r, :]  (bf16), rows * cols elements.
void launch_add(const void* x, const void* y, void* out, int n,
                cudaStream_t stream);

// sum = a + b, then out = rms(sum) * w. One launch for the residual+norm pair.
void launch_add_rms(const void* a, const void* b, void* sum, const void* w, void* out,
                    int rows, int cols, float eps, cudaStream_t stream);

// RMSNorm over last dim: x [rows, cols] -> out [rows, cols], weight [cols].
void launch_rms(const void* x, const void* w, void* out, int rows, int cols,
                float eps, cudaStream_t stream);

// Batched GEMV: y[b,:] = x[b,:] @ W^T for a fixed batch of 16 rows in one launch.
// x: [16,K] bf16, W: [N,K] bf16 (native "out,in" layout, same as a single-row GEMV), y: [16,N] bf16.
// `batch` = active activation rows (the draft's diffusion width; 4, 8 or 16).
void launch_gemv_batched16(const void* x, const void* W, void* y, int N, int K,
                           cudaStream_t stream, int batch = 16);
void launch_gemv_batched16_fused2(const void* x,
                                  const void* W0, const void* W1,
                                  void* y0, void* y1,
                                  int N0, int N1, int K, cudaStream_t stream, int batch = 16);
void launch_gemv_batched16_fused3(const void* x,
                                  const void* W0, const void* W1, const void* W2,
                                  void* y0, void* y1, void* y2,
                                  int N0, int N1, int N2, int K, cudaStream_t stream, int batch = 16);

// Exact batched form of the small-N S=8 split-K BF16 GEMV used by launch_gemv.
// Collapses multiple row launches into grid.y without changing arithmetic order.
void launch_gemv_rows_exact(const void* x, const void* W, void* y,
                            int rows, int N, int K, cudaStream_t stream);
// Copy hidden row `x` [H] into hidden[(*cap_row) * row_elems + slot * H]. The row index lives in
// device memory so the launch can be captured into the decode graph and still target the right row.
void launch_capture_row(const void* x, void* hidden, const int* cap_row, int slot, int H,
                        int row_elems, int max_rows, cudaStream_t stream);

// bf16 -> asymmetric int4 (packed nibbles + fp16 scale/min per 32). Run once at load.
void launch_quantize_w_q4(const void* w, void* q, void* dm, int N, int K, cudaStream_t stream);

// int4-weight form of launch_gemv_batched16_fused3 (~5 bits/weight vs Q8_0's 9).
void launch_gemv_batched_q4_fused3(const void* x,
                                   const void* Q0, const void* Q1, const void* Q2,
                                   const void* D0, const void* D1, const void* D2,
                                   void* y0, void* y1, void* y2,
                                   int N0, int N1, int N2, int K, cudaStream_t stream,
                                   int batch = 16);

// bf16 -> Q8_0 (int8 + one fp32 scale per 32 values along K). Run once at load.
void launch_quantize_w_q8(const void* w, void* q, float* sc, int N, int K, cudaStream_t stream);

// Q8_0-weight form of launch_gemv_batched16_fused3 (half the weight bytes).
void launch_gemv_batched_q8_fused3(const void* x,
                                   const void* Q0, const void* Q1, const void* Q2,
                                   const float* S0, const float* S1, const float* S2,
                                   void* y0, void* y1, void* y2,
                                   int N0, int N1, int N2, int K, cudaStream_t stream,
                                   int batch = 16);

// Single-weight form of the row-batched context projection (weight streamed once per 8 rows).
void launch_gemv_rows_batched(const void* x, const void* W, void* y,
                              int rows, int N, int K, cudaStream_t stream);
void launch_gemv_rows_exact_fused2(const void* x,
                                   const void* W0, const void* W1,
                                   void* y0, void* y1,
                                   int rows, int N0, int N1, int K,
                                   cudaStream_t stream);

// Same, with fp32 output/accumulate (for the LM head's logits).
void launch_gemv_batched16_f32(const void* x, const void* W, float* y, int N, int K,
                               cudaStream_t stream);

// Per-head RMSNorm on [seq, n_heads, d] (in-place).
void launch_rms_heads(void* x, const void* w, int seq, int n_heads, int d,
                      float eps, cudaStream_t stream);

// Per-head RMSNorm followed by RoPE, same buffer, one launch. Same math and order as calling
// launch_rms_heads then launch_rope_seq.
// inv_freq (optional, [d/2]) + att_scale carry YaRN. Null/1.0f => the original inline
// theta^(-2i/d), bit-identical for every pre-existing caller. YaRN's NTK-by-parts ramp scales each
// frequency band differently, so it cannot be folded into theta; see k_rms_heads_rope.
void launch_rms_heads_rope(void* x, const void* w, int seq, int n_heads, int d, float eps,
                           int pos0, float theta, cudaStream_t stream,
                           const float* inv_freq = nullptr, float att_scale = 1.0f);

// Same as launch_rms_heads_rope, but "normal" (consecutive-pair, LLAMA_ROPE_TYPE_NORM) RoPE
// pairing instead of NeoX split-half. Needed for the Muse Glimmer DFlash draft checkpoint --
// see k_rms_heads_rope_normal in dflash_kernels.cu for the full reasoning, and
// DFlashDraftConfig::rope_normal for how a caller opts into this variant.
void launch_rms_heads_rope_normal(void* x, const void* w, int seq, int n_heads, int d, float eps,
                                  int pos0, float theta, cudaStream_t stream);

// Accepted-prefix GDN commit for EVERY linear-attention layer in one launch.
//
// The per-layer commits (kernels::launch_dflash_gdn_{conv,scan}_commit) are independent — each
// touches only its own slice of the live recurrent state — but they run back-to-back on one stream
// after the verify graph, and the next step's draft cannot start until they drain. That put 2 *
// n_gdn_layers tiny serialized launches on the critical path for work that is n_layers-way
// parallel. These forms take the base pointer plus the per-layer stride and use a grid dimension
// as the layer index. Every buffer is regularly strided by the FULL layer index (linear-attention
// layers are interleaved with full-attention ones), so the layer table carries the real index; the
// two per-layer weight vectors are not strided and are passed as device pointers.
//
// The per-layer arithmetic and reduction order are unchanged, so each layer's committed state is
// bit-identical to running the single-layer kernels one at a time -- PROVIDED qh_block is threaded
// through (fixed 2026-08-17, see launch_gdn_scan_commit_layers's own note below; this comment was
// wrong from the day the optimization was written -- k_gdn_scan_commit_layers hardcoded the
// vh % q_heads mapping and silently dropped the vh / (v_heads/q_heads) branch checkpoint kernels
// use for qh_block models like Qwen3.8-27B).
struct GdnCommitLayer { const void* dt; const void* a; int layer; };

void launch_gdn_conv_commit_layers(const void* qkv_base, size_t qkv_layer_stride,
                                   void* live_base, size_t live_layer_stride,
                                   const int* layer_ids, int n_layers, int n_tokens,
                                   int q_heads, int v_heads, int head_dim, int conv_kernel,
                                   cudaStream_t stream);

void launch_gdn_scan_commit_layers(const void* k_base, size_t k_layer_stride,
                                   const void* v_base, size_t v_layer_stride,
                                   const void* alpha_base, size_t ab_layer_stride,
                                   const void* beta_base, const GdnCommitLayer* layers,
                                   float* live_base, size_t live_layer_stride,
                                   int n_layers, int n_tokens, int q_heads, int v_heads,
                                   int head_dim, bool qh_block, cudaStream_t stream);

// Copy nodes are barriers inside the verify graph. The batched verify records two kinds of
// device-to-device copy per replay -- one 2-D copy per captured layer to stage the draft's target
// features, and one per row to replicate the KV block table -- and although they move only tens of
// kilobytes, a memcpy node does not schedule against its neighbours the way a kernel node does, so
// each one drains the graph. Measured on RTX 5090 at 4k with nsys: 14 copy nodes per verify, 40 us
// of actual copying and 2.07 ms of GPU idle behind them, against a 5.6 ms verify whose every other
// node runs back-to-back.
//
// These do the same byte-for-byte copies as kernel nodes, which schedule normally.
void launch_capture_rows(const void* src, void* dst, int rows, int hidden, int dst_row_stride,
                         cudaStream_t stream);

void launch_broadcast_rows_i32(const int* src, int* dst, int n, int rows, cudaStream_t stream);

// DSpark's Markov head: a low-rank learned bigram bias, added in place to one row of draft
// logits. bias[v] = sum_r(w1[prev_token][r] * w2[v][r]) -- w1 is a [verifier_vocab, rank]
// embedding table (indexed by the token id immediately preceding the position being predicted),
// w2 is a [draft_vocab, rank] projection back to vocab space. prev_token is read from DEVICE
// memory (not passed by value) so a chain of calls on the same stream can each depend on the
// previous call's own argmax result without a host round-trip in between -- required because the
// bias for block position k conditions on position k's own token, which for k>0 is only known
// once position k-1's (already Markov-corrected) argmax has been computed. out_latent (optional,
// nullptr to skip) receives the same [rank] latent vector this call already computed, for the
// confidence head below to reuse instead of re-doing the embedding lookup.
// int8 w2 twin of launch_markov_bias_add. w2q is [vocab, rank] int8 with per-32 scales in w2s;
// halves the stream and drops the table under L2, which is what the repeated per-row reads want.
// Requires rank == 256 (the released DSpark checkpoints); declines otherwise.
void launch_markov_bias_add_q8(const void* w1, const void* w2q, const float* w2s,
                               const int* prev_token, float* logits, int vocab, int rank,
                               cudaStream_t stream, float* out_latent);
void launch_markov_bias_add(const void* w1, const void* w2, const int* prev_token,
                            float* logits, int vocab, int rank, cudaStream_t stream,
                            float* out_latent = nullptr);

// DSpark's confidence head (AcceptRatePredictor): a single linear layer over concat(hidden[H],
// markov_latent[rank]) -> one logit, predicting this draft position's acceptance probability
// (sigmoid it on the host if a probability is needed; raw logit is enough for threshold
// comparisons). `latent` must already hold the SAME row's Markov latent (see out_latent above).
void launch_confidence_head(const void* hidden, const float* latent, const void* w, float bias,
                            int H, int rank, float* out_confidence, cudaStream_t stream);

} // namespace dflash_kernels
} // namespace sparkinfer
