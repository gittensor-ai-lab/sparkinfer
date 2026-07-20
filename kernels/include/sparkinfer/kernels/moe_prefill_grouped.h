#pragma once
// Faithful weight-stationary grouped MoE FFN for batched prefill (Qwen3.6-35B-A3B).
//
// Decode routes one token at a time, reloading every routed expert's Q4_K/Q5_K weight per prompt
// token. This path permutes tokens into per-expert contiguous groups (block-aligned to the row
// tile), streams each expert weight into shared memory once per tile, and reuses it across every
// token routed to it. The dot products use the exact per-64-subblock scale+min affine dequant the
// decode MoE FFN uses, so the FFN output matches decode (no lossy per-row requant) — a subsequent
// decode step stays numerically faithful.
#include <cuda_runtime.h>

namespace sparkinfer { namespace kernels {

// Padded row capacity for T = num_tokens * top_k assignments (groups rounded up to the row tile).
int  moe_prefill_padded_rows(int num_tokens, int top_k, int num_experts);
// Upper bound on the number of row-tiles the grouped kernels can emit (for grid sizing).
int  moe_prefill_max_tiles(int num_tokens, int top_k, int num_experts);

// Batched router logits: logits[N,E] = hn[N,H] @ router_w[E,H]^T. router_w native [E,H]; if
// router_w_type != 0 it is dequantized into rw_bf16 scratch [E,H] first.
void launch_moe_prefill_router_logits(
    const void* hn, const void* router_w, int router_w_type, void* rw_bf16,
    float* logits, int N, int H, int E, cudaStream_t stream);

// From top-k ids/weights + per-expert counts, build padded per-expert groups: block-aligned offset
// scan, scatter each (token,slot) assignment into its expert's group (recording source token +
// routing weight), and emit the (expert per row-tile) schedule. Padding rows get token = -1.
void launch_moe_prefill_build_groups(
    const int* ids, const float* weights, const int* counts,
    int* offsets, int* pos_token, float* pos_weight, int* tile_expert,
    int* d_num_tiles, int N, int top_k, int E, cudaStream_t stream);

// Faithful grouped gate/up: h_out[row,f] = silu(<x_row, gate[e,f]>) * <x_row, up[e,f]>, x_row =
// hn[pos_token[row]/top_k]. Expert weights read native (Q4_K/Q5_K/Q6_K); weight-stationary per tile.
void launch_moe_grouped_gate_up(
    const void* hn, const int* pos_token, int top_k, const int* tile_expert, int max_tiles,
    const void* gate_q, const void* up_q, int gate_type, int up_type,
    void* h_out, int H, int F, int rows, cudaStream_t stream);

// Faithful grouped down: D[row,hh] = <h_in[row], down[e,hh]>. Expert weight read native.
void launch_moe_grouped_down(
    const void* h_in, const int* pos_token, const int* tile_expert, int max_tiles,
    const void* down_q, int down_type, void* D, int H, int F, int rows, cudaStream_t stream);

// Scatter weighted down outputs back to per-token rows: routed[pos_token[t]/top_k] += pos_weight[t]
// * D[t,:] (fp32 accumulate). routed is zero-initialized by the launcher; padding rows skipped.
void launch_moe_prefill_scatter_weighted(
    const void* D, const int* pos_token, int top_k, const float* pos_weight,
    void* routed, int padded_rows, int N, int H, cudaStream_t stream);

// Finalize: out[n,:] = h[n,:] + routed[n,:] + sigmoid(gate_scalar[n]) * shared[n,:]. gate_scalar may
// be null (shared added ungated) or shared may be null (no shared expert).
void launch_moe_prefill_finalize(
    const void* h, const void* routed, const void* shared, const float* gate_scalar,
    void* out, int N, int H, cudaStream_t stream);

}} // namespace sparkinfer::kernels
