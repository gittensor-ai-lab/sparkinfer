#pragma once
#include <cuda_runtime.h>

// Batched prompt-prefill kernels for the Qwen3.5 dense-hybrid (Qwythos) model.
//
// The decode path processes a prompt one token at a time (forward_token), so every
// prompt token pays a full bandwidth-bound weight reload for each projection. These
// kernels process the WHOLE prompt (N tokens) in a single pass: the weight-bound
// projections/FFN become tensor-core GEMMs (weight read O(N/tile) instead of O(N)),
// the Gated-DeltaNet recurrence runs as one sequential scan over all N tokens, and
// the full-attention layers fill the paged int8 KV cache in the exact layout the
// decode path reads. Nothing here touches the decode graph.
//
// All matrices are token-major row-major bf16 unless noted. Weights are the native
// GGUF [out,in] layout, already dequantized to bf16 by the caller.

namespace sparkinfer { namespace kernels {

// Tensor-core (wmma bf16) GEMM: C[M,N] = A[M,K] @ W^T, where W is the native GGUF
// weight [N,K] row-major (so C[m,n] = sum_k A[m,k]*W[n,k]). A,W,C bf16, fp32 accumulate.
void launch_prefill_gemm(const void* A, const void* W, void* C,
                         int M, int N, int K, cudaStream_t stream = nullptr);

// SwiGLU elementwise for the dense FFN: h[i] = silu(gate[i]) * up[i] over n elements.
void launch_prefill_swiglu(const void* gate, const void* up, void* h, long n,
                           cudaStream_t stream = nullptr);

// Batched residual add: out[i] = a[i] + b[i] (out may alias a). bf16.
void launch_prefill_add(const void* a, const void* b, void* out, long n,
                        cudaStream_t stream = nullptr);

// Split interleaved-per-head [q|gate]: qraw[N, 2*n_heads*hd] (each head is hd q then hd gate)
// -> q[N, n_heads*hd], gate[N, n_heads*hd]. Matches split_q_gate_kernel, batched over tokens.
void launch_prefill_split_q_gate(const void* qraw, void* q, void* gate,
                                 int n_tokens, int n_heads, int head_dim,
                                 cudaStream_t stream = nullptr);

// Batched attn *= sigmoid(gate), elementwise over n_tokens*dim (Qwen3.6 q-gate).
void launch_prefill_mul_sigmoid(void* attn, const void* gate, int n_tokens, int dim,
                                cudaStream_t stream = nullptr);

// Gated-DeltaNet causal depthwise conv (conv_kernel taps) + split(q,k,v) + SiLU + L2-norm(q,k),
// over all N tokens. Leaves the last conv_kernel-1 raw qkv rows in conv_state (decode layout).
//   qkv:        [N, 2*q_heads*hd + v_heads*hd]      conv_w: [qkv_dim, conv_kernel]
//   conv_state: [conv_kernel-1, qkv_dim] (bf16)     q/k: [N, q_heads*hd]   v: [N, v_heads*hd]
void launch_prefill_gdn_conv(const void* qkv, const void* conv_w, void* conv_state,
                             void* q, void* k, void* v,
                             int n_tokens, int q_heads, int v_heads, int head_dim,
                             int conv_kernel, float eps, cudaStream_t stream = nullptr);

// Gated-DeltaNet sequential recurrence scan over all N tokens (one launch). Produces
// out[N, v_heads*hd] and leaves the recurrent state in the SPARKINFER_GDN_FAST transposed
// layout [v_head][col][row] the decode gdn_ar_fast kernel expects. State starts at zero
// (fresh prefill). q/k: [N,q_heads*hd] (head_dim==128), v: [N,v_heads*hd]; alpha/beta:
// [N,v_heads]; dt/a: [v_heads] (per-head constants).
void launch_prefill_gdn_scan(const void* q, const void* k, const void* v,
                             const void* alpha, const void* beta,
                             const void* dt, const void* a,
                             float* state, void* out,
                             int n_tokens, int q_heads, int v_heads, int head_dim,
                             cudaStream_t stream = nullptr);

// Batched gated RMSNorm: out[t,h,:] = (x/rms(x)) * weight * silu(z), per (token, v_head).
//   x/z: [N, v_heads*hd]   weight: [hd]   out: [N, v_heads*hd]
void launch_prefill_gated_norm(const void* x, const void* z, const void* weight, void* out,
                               int n_tokens, int v_heads, int head_dim, float eps,
                               cudaStream_t stream = nullptr);

// Full-attention prefill: batched QK-norm + partial-RoPE (q,k in place, bf16) + int8 KV write
// into the single-sequence paged pool at positions 0..N-1. Matches the decode int8 layout
// (per-(token,kv_head) max-abs fp16 scale). block_table maps logical block -> physical block.
//   q: [N,n_q_heads*hd]  k/v: [N,n_kv_heads*hd]  q_w/k_w: [hd]
//   k_pool/v_pool: int8 [phys_tok, n_kv_heads, hd]   k_scale/v_scale: __half [phys_tok, n_kv_heads]
void launch_prefill_qknorm_rope_kv_int8(
    void* q, void* k, const void* v, const void* q_w, const void* k_w,
    signed char* k_pool, signed char* v_pool, void* k_scale, void* v_scale,
    const int* block_table, int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    int rotary_dim, float theta, float eps, int block_size, int max_blocks_per_seq,
    cudaStream_t stream = nullptr);

// Full-attention prefill: causal attention over the paged int8 KV pool just filled above.
// One warp per (token, q-head); online softmax over keys 0..token (causal). q is the rope'd
// bf16 query [N,n_q_heads*hd]; out attn[N,n_q_heads*hd].
void launch_prefill_attn_int8_paged(
    const void* q, const signed char* k_pool, const signed char* v_pool,
    const void* k_scale, const void* v_scale, const int* block_table, void* attn,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, float scale, cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
