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

// Fold residual_add(res1,res2) into add_rmsnorm2: out_sum = x + (res1 + res2).
void launch_add_rmsnorm3(const void* x_bf16, const void* res1_bf16, const void* res2_bf16,
                         const void* weight_bf16, void* out_sum_bf16, void* out_norm_bf16,
                         int rows, int cols, float eps, cudaStream_t stream = nullptr);
void launch_add_rmsnorm3_q8(const void* x_bf16, const void* res1_bf16, const void* res2_bf16,
                            const void* weight_bf16, void* out_sum_bf16, void* out_norm_bf16,
                            void* out_q8, int cols, float eps, cudaStream_t stream = nullptr);

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

// Benchmark-only decode feedback: tok = out_id; pos/writepos/seqlen += 1.
// Capturable, so a decode CUDA graph can self-feed during throughput timing.
void launch_decode_feedback(int* scalars, const int* out_id, cudaStream_t stream = nullptr);
// Qwen3.5/Qwen3.6 hybrid Gated DeltaNet helpers.
void launch_qwen36_split_q_gate(const void* qg_bf16, void* q_bf16, void* gate_bf16,
                                int n_heads, int head_dim, cudaStream_t stream = nullptr);

void launch_qwen36_mul_sigmoid(void* x_bf16, const void* gate_bf16, int n,
                               cudaStream_t stream = nullptr);

void launch_qwen36_sigmoid_scalar(const void* x_bf16, float* out_f32,
                                  cudaStream_t stream = nullptr);

// Shared-expert SwiGLU with folded gate scalar: out[i] = dw * SiLU(gate[i]) * up[i].
void launch_qwen36_shared_swiglu(const void* gate_bf16, const void* up_bf16,
                                 const float* dw_f32, void* out_bf16, int n,
                                 cudaStream_t stream = nullptr);

void launch_qwen36_conv_split_l2(const void* qkv_bf16, const void* conv_w_bf16,
                                 void* conv_state_bf16, void* q_bf16, void* k_bf16,
                                 void* v_bf16, int q_heads, int v_heads, int head_dim,
                                 int conv_kernel, float eps, cudaStream_t stream = nullptr);

// Fused conv_split + per-head l2_norm: one block per head, head_dim threads.
// Eliminates the two standalone l2_norm_heads kernel launches per GDN layer.
// SPARKINFER_GDN_FUSE=0 restores the split path for A/B.
void launch_qwen36_conv_split_l2norm_fused(const void* qkv_bf16, const void* conv_w_bf16,
                                 void* conv_state_bf16, void* q_bf16, void* k_bf16,
                                 void* v_bf16, int q_heads, int v_heads, int head_dim,
                                 int conv_kernel, float eps, cudaStream_t stream = nullptr);

void launch_qwen36_gdn_ar(const void* q_bf16, const void* k_bf16, const void* v_bf16,
                          const void* alpha_bf16, const void* beta_bf16,
                          const void* dt_bf16, const void* a_bf16,
                          float* state_f32, void* out_bf16,
                          int q_heads, int v_heads, int head_dim, cudaStream_t stream = nullptr);

void launch_qwen36_gated_norm(const void* x_bf16, const void* z_bf16,
                              const void* weight_bf16, void* out_bf16,
                              int v_heads, int head_dim, float eps,
                              cudaStream_t stream = nullptr);

// Gated norm + Q8_1 emit for ssm_out MMVQ (skips bf16 lin_norm + separate quantize).
void launch_qwen36_gated_norm_q8(const void* x_bf16, const void* z_bf16,
                                 const void* weight_bf16, void* out_q8,
                                 int v_heads, int head_dim, float eps,
                                 cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
