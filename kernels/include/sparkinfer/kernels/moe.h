#pragma once
#include <cuda_runtime.h>

namespace sparkinfer { namespace kernels {

// Router projection: logits = input @ router_w  (bf16 x bf16 -> fp32).
//   input:    [num_tokens, hidden_dim]      (bf16)
//   router_w: [hidden_dim, num_experts]     (bf16, pre-transposed)
//   logits:   [num_tokens, num_experts]     (fp32, output)
void launch_moe_router_gemm(
    const void* input, const void* router_w, float* logits,
    int num_tokens, int hidden_dim, int num_experts,
    cudaStream_t stream = nullptr);

// Token-to-expert router: top-k selection over per-expert logits.
//   logits:            [num_tokens, num_experts]  (float)
//   expert_ids:        [num_tokens, top_k]        (int32, output)
//   expert_weights:    [num_tokens, top_k]        (float, output)
//   tokens_per_expert: [num_experts]              (int32, output, device-only)
//
// tokens_per_expert is accumulated on-device and never copied to the host —
// this is the sync-free counter that lets the whole MoE forward pass be
// captured in a single CUDA graph. Pass nullptr to skip it.
// If normalize != 0, the top-k weights are softmax-normalized to sum to 1.
void launch_moe_router(
    const float* logits,
    int* expert_ids, float* expert_weights,
    int* tokens_per_expert,
    int num_tokens, int num_experts, int top_k,
    int normalize,
    cudaStream_t stream = nullptr);

// Fused MoE expert FFN with SwiGLU activation.
// For each token i and each of its top_k experts e (weight w):
//   h = SiLU(X[i] @ gate_w[e]) * (X[i] @ up_w[e])     // [ffn_dim]
//   y = h @ down_w[e]                                  // [hidden_dim]
//   out[i] += w * y                                    // accumulated over top_k
//
//   input:          [num_tokens, hidden_dim]            (bf16)
//   gate_w / up_w:  [num_experts, hidden_dim, ffn_dim]  (bf16)
//   down_w:         [num_experts, ffn_dim, hidden_dim]  (bf16)
//   expert_ids:     [num_tokens, top_k]                 (int32)
//   expert_weights: [num_tokens, top_k]                 (float)
//   output:         [num_tokens, hidden_dim]            (bf16, must be zeroed first)
void launch_moe_expert_ffn(
    const void* input,
    const void* gate_w, const void* up_w, const void* down_w,
    const int* expert_ids, const float* expert_weights,
    void* output,
    int num_tokens, int top_k, int num_experts,
    int hidden_dim, int ffn_dim,
    cudaStream_t stream = nullptr);

// Fused quantized expert FFN (decode-optimized): dequantizes only the top_k
// routed experts on-read, one warp per output row. gate_q/up_q are Q4_K
// [num_experts, ffn, hidden], down_q is Q6_K [num_experts, hidden, ffn] (GGUF
// native layout). h_scratch: [num_tokens*top_k*ffn] fp32; out_scratch:
// [num_tokens*hidden] fp32. output: [num_tokens, hidden] bf16. hidden,ffn % 256 == 0.
// gate_type/up_type/down_type are ggml type ids (12=Q4_K, 14=Q6_K); Q4_K_M mixes
// them per tensor, so each is dispatched independently inside the kernel.
void launch_moe_expert_ffn_q4k(
    const void* input, const void* gate_q, const void* up_q, const void* down_q,
    int gate_type, int up_type, int down_type,
    const int* expert_ids, const float* expert_weights, void* output,
    float* h_scratch, float* out_scratch,
    int num_tokens, int top_k, int hidden, int ffn,
    const void* input_q8 = nullptr,   // pre-quantized Q8_1(input) from the fused norm; nullptr = quantize internally
    cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
