#pragma once
#include <cuda_runtime.h>

// Direct-4-bit ("MMQ") grouped MoE expert GEMM for the batched prefill path.
//
// The int8 MoE prefill GEMM (launch_pfm_moe_gemm_i8_bm) first dequantizes each expert's
// GGUF weight to a full int8 row buffer (launch_gguf_dequant_rows_i8) and then runs an
// int8 tensor-core GEMM over it. At small token counts (few tokens per expert, e.g. the
// 128/512 prefill contexts) that dequant materialize dominates the layer: the weight is
// read as 4-bit, written back as int8 (2x the bytes), then re-read by the GEMM — ~5x the
// weight-DRAM traffic of reading the 4-bit block once. Since the per-expert GEMM is tiny
// there, the layer is weight-read-bound and the dequant, not the matmul, is the wall.
//
// This path folds the weight read into the matmul the way decode's MMVQ does: activations
// are quantized to Q8_1 and each expert's Q4_K/Q5_K block is dp4a'd directly, with no int8
// materialize. It reuses the exact pair/tilemap bucketing of the int8 path, so only the
// inner GEMM differs. It wins at small M (weight-bound) and is intentionally NOT used at
// large M, where the int8 tensor-core path is faster.

namespace sparkinfer {
namespace kernels {

// Quantize `rows` rows of `K` bf16 activations to Q8_1 blocks (36 B / 32 values,
// interleaved {ds, qs[32]}, one block per 32 elements) — the layout the dp4a dots consume.
// Output buffer must hold rows*(K/32) Q8_1 blocks.
void launch_prefill_quantize_rows_q8_1(const void* x_bf16, void* y_q8_1,
                                       int rows, int K, cudaStream_t stream);

// Grouped MoE expert GEMM reading GGUF-quantized weights directly (dp4a with Q8_1
// activations). Same pair/tilemap contract as launch_pfm_moe_gemm_i8_bm:
//   A_q8_1     : [rows][K/32] Q8_1 activations (rows = tokens if a_indirect, else pairs)
//   W_q        : raw GGUF expert weights [n_experts][N_out][K/256 super-blocks]
//   qtype      : ggml type id — 12 (Q4_K) and 13 (Q5_K) supported; else returns false
//   pair_tok/pair_w/offsets/tilemap/d_ntiles/e_base : bucketing built by
//                launch_pfm_bucket_pairs_bm (unchanged)
//   C_bf16     : dense pair-major output when !c_scatter (gate/up)
//   out_f32    : token-major scatter-add target when c_scatter (down; folds pair_w)
// Returns false (no launch) for unsupported qtype so the caller can fall back.
bool launch_pfm_moe_mmq(const void* A_q8_1, const void* W_q, int qtype,
                        const int* pair_tok, const float* pair_w,
                        const int* offsets, const int* tilemap, const int* d_ntiles,
                        void* C_bf16, float* out_f32,
                        int N_out, int K, int max_tiles, int e_base,
                        bool a_indirect, bool c_scatter, cudaStream_t stream);

}  // namespace kernels
}  // namespace sparkinfer
