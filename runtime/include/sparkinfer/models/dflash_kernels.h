#pragma once
// Internal CUDA helpers for DFlash draft attention / RoPE / SwiGLU.

#include <cuda_runtime.h>

namespace sparkinfer {
namespace dflash_kernels {

// GQA attention. q: [q_len, n_q, d], k/v: [kv_len, n_kv, d], out: [q_len, n_q, d] bf16.
// q_pos0 / k_pos0 are absolute positions of index 0. If window > 0, mask keys with
// (q_pos - k_pos) >= window (sliding window). scale = 1/sqrt(d).
void launch_attn_gqa(const void* q, const void* k, const void* v, void* out,
                     int q_len, int kv_len, int n_q, int n_kv, int d,
                     int q_pos0, int k_pos0, int window, bool causal, float scale,
                     cudaStream_t stream);

// In-place RoPE on [seq, n_heads, d] bf16. positions[i] = pos0 + i.
void launch_rope_seq(void* x, int seq, int n_heads, int d, int pos0,
                     float theta, cudaStream_t stream);

// out[i] = silu(gate[i]) * up[i]  for n elements (bf16).
void launch_swiglu(const void* gate, const void* up, void* out, int n,
                   cudaStream_t stream);

// out[r, :] = x[r, :] + y[r, :]  (bf16), rows * cols elements.
void launch_add(const void* x, const void* y, void* out, int n,
                cudaStream_t stream);

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

} // namespace dflash_kernels
} // namespace sparkinfer
