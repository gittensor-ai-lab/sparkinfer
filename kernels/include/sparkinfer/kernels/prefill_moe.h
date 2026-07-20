#pragma once
#include <cuda_runtime.h>

// Batched (token-parallel) MoE FFN for Qwen3.6-35B-A3B prompt prefill.
//
// The decode expert FFN re-reads each routed expert's weights per token (~1.1 GB/token),
// which is also what a per-token prefill loop pays. These kernels instead bucket the
// N*top_k (token, expert) pairs by expert and run per-expert int8 tensor-core GEMMs, so
// each expert's weights are read ONCE per layer per pass:
//   router logits [N,E] (fp32, gemv_f32-order dot) -> launch_moe_router top-k ->
//   pair scatter by expert -> grouped GEMM (gate,up) -> SwiGLU -> grouped GEMM (down)
//   with the per-pair router weight folded into an fp32 scatter-add over tokens.
// Expert weights are dequantized once per layer to int8 rows + per-row scales with
// launch_gguf_dequant_rows_i8 (quant.h); activations use the per-row int8 quantization
// the merged int8 prefill GEMM already ships (prefill_i8.h).

namespace sparkinfer { namespace kernels {

// logits[N,E] fp32 = x[N,H] (bf16) @ W^T, W native [E,H] bf16. One warp per (token,expert)
// with the same lane-strided fp32 warp-reduce dot as launch_gemv_f32 (decode's reference
// router path), so per-row rounding matches it.
void launch_pfm_router_logits(const void* x, const void* W, float* logits,
                              int n_tokens, int n_experts, int H, cudaStream_t stream);

// Bucket the N*top_k routed pairs by expert. counts[E] comes from launch_moe_router.
//   offsets[E+1] (exclusive scan of counts), pair_tok[P], pair_w[P] grouped by expert,
//   tilemap[2*max_tiles] = (expert, local row tile) per BM=128-row GEMM tile,
//   d_ntiles[0] = tile count. cursors[E] is zeroed internally. P = n_tokens*top_k.
void launch_pfm_bucket_pairs(const int* expert_ids, const float* expert_weights,
                             const int* counts, int* offsets, int* cursors,
                             int* pair_tok, float* pair_w,
                             int* tilemap, int* d_ntiles,
                             int n_tokens, int n_experts, int top_k, cudaStream_t stream);

// Same as above but with explicit GEMM row-tile size (16 or 128). Short-N prefill
// uses bm=16 so underfilled experts don't pad to 128 WMMA rows.
void launch_pfm_bucket_pairs_bm(const int* expert_ids, const float* expert_weights,
                                const int* counts, int* offsets, int* cursors,
                                int* pair_tok, float* pair_w,
                                int* tilemap, int* d_ntiles,
                                int n_tokens, int n_experts, int top_k, int bm,
                                cudaStream_t stream);

// Grouped int8 GEMM over expert-partitioned pair tiles (BM=128, BN=128, BK=32, 8 warps).
//   A_i8/sx: per-row int8 activations + scales. A_INDIRECT: A row for pair p is
//   A_i8[pair_tok[p]*K ..] (gate/up: activations are per TOKEN); else A rows are the
//   contiguous pair rows (down: SwiGLU output is per PAIR).
//   W_i8/sw: per-expert int8 weights, W_i8 + (size_t)e*N_out*K, sw + (size_t)e*N_out.
//   C_SCATTER=false: C[p*N_out+n] = bf16(acc*sx*sw)  (pair-major h output).
//   C_SCATTER=true : atomicAdd(out_f32[pair_tok[p]*N_out+n], acc*sx*sw*pair_w[p]).
//   Launch bound: grid.y = max_tiles (host upper bound); tiles >= d_ntiles[0] exit.
//   bm: 128 (default) or 16 (short-N); must match launch_pfm_bucket_pairs_bm.
void launch_pfm_moe_gemm_i8(const signed char* A_i8, const float* sx,
                            const signed char* W_i8, const float* sw,
                            const int* pair_tok, const float* pair_w,
                            const int* offsets, const int* tilemap, const int* d_ntiles,
                            void* C_bf16, float* out_f32,
                            int N_out, int K, int max_tiles,
                            bool a_indirect, bool c_scatter, cudaStream_t stream);
void launch_pfm_moe_gemm_i8_bm(const signed char* A_i8, const float* sx,
                               const signed char* W_i8, const float* sw,
                               const int* pair_tok, const float* pair_w,
                               const int* offsets, const int* tilemap, const int* d_ntiles,
                               void* C_bf16, float* out_f32,
                               int N_out, int K, int max_tiles, int bm,
                               bool a_indirect, bool c_scatter, cudaStream_t stream);
// Like _bm but W_i8/sw hold a contiguous expert group starting at e_base (tilemap still
// stores global expert ids; W is indexed as e - e_base).
void launch_pfm_moe_gemm_i8_bm_base(const signed char* A_i8, const float* sx,
                                    const signed char* W_i8, const float* sw,
                                    const int* pair_tok, const float* pair_w,
                                    const int* offsets, const int* tilemap, const int* d_ntiles,
                                    void* C_bf16, float* out_f32,
                                    int N_out, int K, int max_tiles, int bm, int e_base,
                                    bool a_indirect, bool c_scatter, cudaStream_t stream);

// Single-expert GEMM: W_i8/sw already point at one expert's rows (no e-stride).
// grid.y = n_tiles = ceil(counts[expert]/bm). Used by the expert-serial L2 path
// (dequant one expert → GEMM while ~3 MB stays L2-hot).
void launch_pfm_moe_gemm_i8_one(const signed char* A_i8, const float* sx,
                                const signed char* W_i8, const float* sw,
                                const int* pair_tok, const float* pair_w,
                                const int* offsets, int expert_id, int n_tiles,
                                void* C_bf16, float* out_f32,
                                int N_out, int K, int bm,
                                bool a_indirect, bool c_scatter, cudaStream_t stream);

// Short-N fused path: same tilemap/pair API as launch_pfm_moe_gemm_i8, but B is read
// on-the-fly from GGUF Q4_K/Q5_K/Q6_K (ggml_type 12/13/14) into smem — no full-expert
// int8 materialize. A is still per-row int8+scale. Preferred when N is small enough that
// the ~768 MiB/layer int8 write dominates (see SPARKINFER_PREFILL_MOE_FUSED).
void launch_pfm_moe_gemm_qk(const signed char* A_i8, const float* sx,
                            const void* W_q, int wtype,
                            const int* pair_tok, const float* pair_w,
                            const int* offsets, const int* tilemap, const int* d_ntiles,
                            void* C_bf16, float* out_f32,
                            int N_out, int K, int max_tiles,
                            bool a_indirect, bool c_scatter, cudaStream_t stream);

// dw[N] = sigmoid(x[N,H] . w[H]) — batched shared-expert gate scalar (gemv+sigmoid order).
void launch_pfm_shared_gate(const void* x, const void* w, float* dw,
                            int n_tokens, int H, cudaStream_t stream);

// h[t,f] = SiLU(gate[t,f]) * up[t,f] * dw[t]   (batched launch_qwen36_shared_swiglu)
void launch_pfm_shared_swiglu(const void* gate, const void* up, const float* dw,
                              void* h, int n_tokens, int ffn, cudaStream_t stream);

// x[i] = h[i] + routed_f32[i] + shared[i]  (fp32 math, bf16 out; shared may be null)
void launch_pfm_resid3(const void* h, const float* routed_f32, const void* shared,
                       void* x, long n, cudaStream_t stream);

}} // namespace sparkinfer::kernels
