#pragma once
// Routed-MoE grouped int8 GEMM that reads the experts in their NATIVE GGUF quantization and
// decodes to int8 inside the B-tile stage, so the per-layer int8 materialize never happens.
//
// The materialize path (deq_rows_i8 -> pfm_moe_gemm_i8) writes and re-reads the full expert
// pool every layer: for Qwen3.6-35B-A3B that is 805 MB of int8 out plus 805 MB back in, on top
// of the 486 MB of Q4_K/Q5_K the dequant already had to read. This kernel reads only that
// 486 MB. It is worth it exactly while each expert's weight slice is decoded ~once, i.e. while
// the pair count per expert stays within a couple of BM tiles -- see the caller's N gate.
//
// Bit-identical to the materialize path: the per-row int8 scale is the one the dequant kernel
// itself produced (precomputed once at load), and the value decode keeps the same
// (d*s)*nib - (dmin*m) evaluation order, the same roundf(v * inv) and the same int32 MMA
// operand order, so every int8 byte and every accumulator matches.
//
// Returns false when the quantization type has no fused decode (caller falls back).

#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {

// C[pair, n_out] = A_i8[pair-indirected token, K] * dequant(W_q[expert, n_out, K])^T
//   W_q        native GGUF expert pool, [E][n_out][K] blocks of `ggml_type`
//   row_scale  per (expert, n_out) int8 row scale, [E * n_out], as produced by
//              launch_gguf_dequant_rows_i8 (scale[row] = amax/127)
//   bm         16 or 128, must match the tilemap the caller built (both tiled shapes implemented)
bool launch_pfm_moe_gemm_qi8(int ggml_type, const signed char* A_i8, const float* sx,
                             const void* W_q, const float* row_scale,
                             const int* pair_tok, const float* pair_w,
                             const int* offsets, const int* tilemap, const int* d_ntiles,
                             void* C_bf16, float* out_f32,
                             int n_out, int K, int max_tiles, int bm,
                             bool a_indirect, bool c_scatter, cudaStream_t stream);

// True when launch_pfm_moe_gemm_qi8 has a decode for this ggml_type.
bool pfm_moe_gemm_qi8_supported(int ggml_type);

} // namespace kernels
} // namespace sparkinfer
