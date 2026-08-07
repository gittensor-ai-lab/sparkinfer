#pragma once
#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {

// Row-batched decode kernels for the DFlash verify window.
//
// The DFlash verify loop forwards one target token per emitted token, so the whole
// weight stream of the model is re-read once per proposal. These kernels take `rows`
// activation rows at once: the weight superblock is loaded into registers once and
// dotted against every row, so a window of R proposals reads the weights once instead
// of R times. Only the routed-expert weights still scale with R (each token routes to
// its own experts) -- everything else is amortized.
//
// EXACTNESS CONTRACT: every launcher below computes each row with the same operand
// order, the same partial-sum order and the same reduction tree as its single-row
// counterpart in gemv.cu / qwen36.cu / rmsnorm.cu / router.cu. Batching only changes
// which rows share a CTA, never the arithmetic, so the output of a call with rows=R is
// bit-identical to R calls of the single-row kernel. Greedy speculative decoding stays
// exact and DFlash output stays bit-equal to autoregressive output.
//
// rows must be in [1, kDFlashMaxRows].
constexpr int kDFlashMaxRows = 8;

// ---- Q4_K / Q8_0 MMVQ over `rows` activation rows sharing one weight matrix ----
// q81 : [rows, llama_q8_1_bytes(K)] (rows contiguous, stride llama_q8_1_bytes(K))
// W   : GGUF-native quantized [N, K]
// y   : [rows, N], row stride N
void launch_dflash_rows_mmvq_q4k(const void* q81, const void* W, void* y,
                                 int N, int K, int rows, cudaStream_t stream = nullptr);
void launch_dflash_rows_mmvq_q4k_f32(const void* q81, const void* W, float* y,
                                     int N, int K, int rows, cudaStream_t stream = nullptr);
void launch_dflash_rows_mmvq_q80(const void* q81, const void* W, void* y,
                                 int N, int K, int rows, cudaStream_t stream = nullptr);

// Fused GDN qkv + z projection (mirrors launch_mmvq_gdn_qkv_z_pack2): warps 0-3 produce
// qkv[row], warps 4-7 produce z[row], for every one of `rows` activation rows.
// qkv_out : [rows, n_qkv], z_out : [rows, n_z]
void launch_dflash_rows_mmvq_gdn_qkv_z(const void* q81, const void* qkv_w, const void* z_w,
                                       void* qkv_out, void* z_out,
                                       int n_qkv, int n_z, int K, int rows,
                                       cudaStream_t stream = nullptr);

// ---- fused norms / GDN front-end, row-batched ----
// Mirrors launch_qwen36_conv_split_l2norm_fused. The depthwise conv is causal over the
// window, so the `rows` tokens are consumed in order inside the kernel and conv_state is
// left exactly where `rows` successive single-token calls would leave it.
// qkv : [rows, qkv_dim]; q/k/v : [rows, *]
void launch_dflash_rows_conv_split_l2norm(const void* qkv, const void* conv_w, void* conv_state,
                                          void* q, void* k, void* v,
                                          int q_heads, int v_heads, int head_dim,
                                          int conv_kernel, float eps, int rows,
                                          cudaStream_t stream = nullptr);

// Mirrors launch_qwen36_gated_norm_q8. x/z : [rows, v_heads*head_dim];
// out_q8 : [rows, v_heads*head_dim/32] blocks.
void launch_dflash_rows_gated_norm_q8(const void* x, const void* z, const void* weight,
                                      void* out_q8, int v_heads, int head_dim, float eps,
                                      int rows, cudaStream_t stream = nullptr);

// Mirror launch_add_rmsnorm2_q8 / launch_add_rmsnorm3_q8 over `rows` rows of `cols`.
// out_q8 : [rows, cols/32] blocks.
void launch_dflash_rows_add_rmsnorm2_q8(const void* x, const void* residual, const void* weight,
                                        void* out_sum, void* out_norm, void* out_q8,
                                        int cols, float eps, int rows,
                                        cudaStream_t stream = nullptr);
void launch_dflash_rows_add_rmsnorm3_q8(const void* x, const void* res1, const void* res2,
                                        const void* weight, void* out_sum, void* out_norm,
                                        void* out_q8, int cols, float eps, int rows,
                                        cudaStream_t stream = nullptr);

// Mirrors launch_rmsnorm for `rows` rows (prime norm at layer 0).
void launch_dflash_rows_rmsnorm(const void* x, const void* weight, void* out,
                                int cols, float eps, int rows, cudaStream_t stream = nullptr);

// Mirrors launch_router_fused for `rows` tokens: split-K GEMV then the same in-kernel
// bitonic top-k, one independent grid-completion counter per token.
// gridctr : [rows] unsigned, zero-initialized once (atomicInc self-clears per replay).
// logits  : [rows, num_experts]; expert_ids / expert_weights : [rows, top_k]
void launch_dflash_rows_router_fused(const void* x, const void* W, float* logits,
                                     unsigned int* gridctr, int* expert_ids,
                                     float* expert_weights, int num_experts, int K,
                                     int top_k, int normalize, int rows,
                                     cudaStream_t stream = nullptr);

} // namespace kernels
} // namespace sparkinfer
