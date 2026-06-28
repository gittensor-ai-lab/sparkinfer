#pragma once
#include <cuda_runtime.h>

namespace sparkinfer { namespace kernels {

// Fused RMSNorm:  out[r] = weight * x[r] / sqrt(mean(x[r]^2) + eps)
// Optionally adds a residual first (out and residual may alias x).
//   x / residual / out: [rows, cols] (bf16), weight: [cols] (bf16)
void launch_rmsnorm(const void* x_bf16, const void* weight_bf16, void* out_bf16,
                    int rows, int cols, float eps, cudaStream_t stream = nullptr);

void launch_add_rmsnorm(const void* x_bf16, const void* residual_bf16,
                        const void* weight_bf16, void* out_bf16,
                        int rows, int cols, float eps, cudaStream_t stream = nullptr);

// Fused residual+RMSNorm that also emits the residual sum:
//   out_sum = x + residual;  out_norm = (out_sum / rms(out_sum)) * weight
void launch_add_rmsnorm2(const void* x_bf16, const void* residual_bf16, const void* weight_bf16,
                         void* out_sum_bf16, void* out_norm_bf16,
                         int rows, int cols, float eps, cudaStream_t stream = nullptr);

// add_rmsnorm2 that additionally emits a Q8_1 quantization of out_norm (si_block_q8_1),
// so the downstream int8 GEMV skips its own quantize node. rows==1, cols % 256 == 0.
void launch_add_rmsnorm2_q8(const void* x_bf16, const void* residual_bf16, const void* weight_bf16,
                            void* out_sum_bf16, void* out_norm_bf16, void* out_q8,
                            int cols, float eps, cudaStream_t stream = nullptr);

// Fused per-head Q-norm + K-norm in one kernel (1 graph node vs 2). In-place on q/k.
void launch_rmsnorm_qk(void* q, void* k, const void* q_w, const void* k_w,
                       int n_q_heads, int n_kv_heads, int head_dim, float eps, cudaStream_t stream = nullptr);

// Token embedding gather: out[t,:] = table[ids[t],:]  (bf16).
//   ids: [n_tokens] (int32), table: [vocab, hidden], out: [n_tokens, hidden]
void launch_embedding(const int* ids, const void* table, void* out,
                      int n_tokens, int hidden, cudaStream_t stream = nullptr);

// Greedy argmax over each row of logits.  logits: [n_rows, vocab] (fp32),
// out_id: [n_rows] (int32).
void launch_argmax(const float* logits, int* out_id, int n_rows, int vocab,
                   cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
