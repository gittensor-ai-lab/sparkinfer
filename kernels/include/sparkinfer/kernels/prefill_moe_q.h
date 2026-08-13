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

// Dense fused-decode GEMM only: Q4_K / Q5_K / Q6_K (the routed predicate above stays Q4_K/Q5_K).
bool pf_dense_gemm_qi8_supported(int ggml_type);

// Dense (non-routed) fused-decode int8 GEMM: C[M,N] = A_i8[M,K] @ dequant(W_q[N,K])^T, reading the
// weight in native Q4_K/Q5_K and decoding it to int8 inside the B-stage using a per-output-row
// scale precomputed at load (row_scale[n] = amax/127, == launch_gguf_dequant_rows_i8's scale). Skips
// the int8 materialize (dequant -> W_i8 -> reload) that launch_prefill_gemm_i8 pays. Returns false
// (launching nothing) for an unsupported ggml_type, null row_scale, K not a super-block multiple, or
// N not 64-aligned -- callers fall back to the materialize path. Used by Muse Glimmer dense prefill.
// `partials` (int32, >= partials_splits * M * N) enables a split-K fan-out: at prefill's M=128 the
// plain grid is only N/64 blocks, far under the device, so K is sliced across blockIdx.z and the
// int32 tiles are summed in a second pass. int32 accumulation is exact and associative => the
// result is BIT-IDENTICAL to the unsplit launch. partials=nullptr (or splits<=1) disables it;
// SPARKINFER_MUSE_QB_SPLITK=0 disables, >0 pins the slice count.
bool launch_prefill_gemm_qi8_dense(int ggml_type, const signed char* A_i8, const float* sx,
                                   const void* W_q, const float* row_scale, void* C_bf16,
                                   int M, int N, int K, cudaStream_t stream = nullptr,
                                   int* partials = nullptr, int partials_splits = 0,
                                   int* out_acc = nullptr,
                                   // Optional k-tiled [k/32][row][32] copy of A_i8 (see
                                   // qr_pack_off, prefill_quant_rows.cu). Same bytes, staged with
                                   // a quarter of the memory transactions; nullptr keeps the
                                   // row-major addressing. Output is bit-identical either way.
                                   const signed char* A_pack = nullptr);

// True when the split-K consumers hand their accumulator plane back zeroed, so the caller must
// zero the buffer once at allocation instead of the launcher zeroing it before every launch.
bool pf_dense_zero_on_read();

// Fuse up to 4 projections sharing A_i8/sx (same M, same K) into ONE grid. At prefill's M=128 a
// projection's grid is ceil(N/64) CTAs -- 64 for a 4096-wide q/gate but only 4 for a 256-wide
// k/v -- all far under a 5090's 170 SMs, so every launch costs a full CTA-duration regardless of
// how little work it carries. Bit-identical per output tile to the separate calls.
// W_q/row_scale/C_bf16/N are ngroup-long arrays; all groups must share ggml_type.
bool launch_prefill_gemm_qi8_dense_group(int ggml_type, const signed char* A_i8, const float* sx,
                                         const void* const* W_q, const float* const* row_scale,
                                         void* const* C_bf16, const int* N, int ngroup,
                                         int M, int K, cudaStream_t stream = nullptr,
                                         int* partials = nullptr, int partials_splits = 0,
                                         size_t partials_cap = 0,
                                         // Fuse the FFN's SwiGLU + int8 quantize into the split-K
                                         // epilogue: gate/up never become bf16. Sets *out_fused.
                                         signed char* fuse_q = nullptr, float* fuse_sx = nullptr,
                                         int* out_fused = nullptr,
                                         const signed char* A_pack = nullptr,
                                         // k-tiled copy of the fused SwiGLU's int8 output, for the
                                         // down projection that consumes it next.
                                         signed char* fuse_qp = nullptr);

} // namespace kernels
} // namespace sparkinfer
