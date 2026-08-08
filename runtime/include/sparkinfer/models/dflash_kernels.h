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
void launch_rms_heads_rope(void* x, const void* w, int seq, int n_heads, int d, float eps,
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
// bit-identical to running the single-layer kernels one at a time.
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
                                   int head_dim, cudaStream_t stream);

} // namespace dflash_kernels
} // namespace sparkinfer
