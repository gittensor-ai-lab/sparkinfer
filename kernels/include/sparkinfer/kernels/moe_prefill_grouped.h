#pragma once
// Grouped int8 tensor-core MoE FFN for batched prefill (Qwen3.6-35B-A3B).
//
// Decode routes one token at a time, so every prompt token reloads all 8 routed experts' Q4_K/Q6_K
// weights as bandwidth-bound GEMVs. This path instead runs the whole prompt through the MoE FFN in
// one pass: tokens are permuted into per-expert contiguous groups so each expert weight is streamed
// once and reused across all its tokens, and the gate/up/down products run as int8 mma.sync
// (m16n8k32) tensor-core GEMMs against the same per-row symmetric int8 requant the dense prefill
// projections already use. Output is numerically consistent with the decode MoE FFN so a subsequent
// decode step is faithful.
//
// The grouped GEMM is a single launch: a device-built tile schedule maps each CTA to (expert,
// row-tile), and expert groups are padded to the block-row tile so a tile never straddles two
// experts. No host sync in the per-layer loop.
#include <cuda_runtime.h>

namespace sparkinfer { namespace kernels {

// Padded row capacity for T = num_tokens * top_k assignments, groups rounded up to MOE_PF_BM (128).
int  moe_prefill_padded_rows(int num_tokens, int top_k, int num_experts);
// Upper bound on the number of row-tiles the grouped GEMM can emit (for grid.x sizing).
int  moe_prefill_max_tiles(int num_tokens, int top_k, int num_experts);

// Batched router logits: logits[N,E] = hn[N,H] @ router_w[E,H]^T. router_w native [E,H]; if
// router_w_type != 0 it is dequantized into rw_bf16 scratch [E,H] first (Q4_K/Q6_K/Q8_0).
void launch_moe_prefill_router_logits(
    const void* hn, const void* router_w, int router_w_type, void* rw_bf16,
    float* logits, int N, int H, int E, cudaStream_t stream);

// From top-k ids/weights + per-expert counts, build padded per-expert groups: exclusive-scan the
// counts into block-aligned offsets, scatter each (token,slot) assignment into its expert's group
// (recording source token + routing weight), and emit the (expert per row-tile) schedule. Padding
// rows get token = -1 (masked to zero downstream).
void launch_moe_prefill_build_groups(
    const int* ids, const float* weights, const int* counts,
    int* offsets, int* pos_token, float* pos_weight, int* tile_expert,
    int* d_num_tiles, int N, int top_k, int E, cudaStream_t stream);

// Gather permuted activations and per-row symmetric int8 quantize: for padded row t,
// Ai8[t,:] = round(hn[pos_token[t]/top_k,:] / sA[t]); pos_token[t] < 0 -> zero row.
void launch_moe_prefill_gather_quant_i8(
    const void* hn, const int* pos_token, int top_k, signed char* Ai8, float* sA,
    int padded_rows, int H, cudaStream_t stream);

// Single-launch grouped int8 GEMM: C[t, Nout] = Ai8[t,K] @ W_i8[e(t), Nout, K]^T, dequant folded as
// sA[t]*sW[e,n]. Rows are grouped by expert via block-aligned offsets; tile_expert[tile] names the
// expert for row-tile `tile` (>=0), or -1 to skip. grid = (max_tiles, ceil(Nout/128)).
void launch_moe_grouped_gemm_i8(
    const signed char* Ai8, const float* sA, const signed char* W_i8, const float* sW,
    const int* offsets, const int* tile_expert, int max_tiles,
    void* C, int Nout, int K, cudaStream_t stream);

// h[t,F] = silu(gate[t,F]) * up[t,F], then per-row symmetric int8 quantize -> Hi8[t,F], sH[t].
void launch_moe_prefill_swiglu_quant_i8(
    const void* gate, const void* up, signed char* Hi8, float* sH,
    int padded_rows, int F, cudaStream_t stream);

// Scatter the weighted down outputs back to per-token rows: routed[pos_token[t]/top_k] +=
// pos_weight[t] * D[t,:] (fp32 accumulate). routed is zero-initialized by the launcher. Padding
// rows (token < 0) are skipped.
void launch_moe_prefill_scatter_weighted(
    const void* D, const int* pos_token, int top_k, const float* pos_weight,
    void* routed, int padded_rows, int N, int H, cudaStream_t stream);

// Finalize: out[n,:] = h[n,:] + routed[n,:] + gate_scalar[n] * shared[n,:]. gate_scalar may be null
// (shared added ungated) or shared may be null (no shared expert).
void launch_moe_prefill_finalize(
    const void* h, const void* routed, const void* shared, const float* gate_scalar,
    void* out, int N, int H, cudaStream_t stream);

}} // namespace sparkinfer::kernels
