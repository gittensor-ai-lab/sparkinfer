#pragma once
// Internal CUDA helpers for DFlash draft attention / RoPE / SwiGLU.

#include <cuda_runtime.h>

namespace sparkinfer {
namespace dflash_kernels {

// Non-causal GQA attention. q: [q_len, n_q, d], k/v: [kv_len, n_kv, d], out: [q_len, n_q, d] bf16.
// q_pos0 / k_pos0 are absolute positions of index 0. If window > 0, mask keys with
// (q_pos - k_pos) >= window (sliding window). scale = 1/sqrt(d).
void launch_attn_gqa(const void* q, const void* k, const void* v, void* out,
                     int q_len, int kv_len, int n_q, int n_kv, int d,
                     int q_pos0, int k_pos0, int window, float scale,
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

// Per-head RMSNorm on [seq, n_heads, d] (in-place).
void launch_rms_heads(void* x, const void* w, int seq, int n_heads, int d,
                      float eps, cudaStream_t stream);

} // namespace dflash_kernels
} // namespace sparkinfer
