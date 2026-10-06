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
// True when the BM=128 launch runs the pipelined m16n8k32 kernel (SPARKINFER_PREFILL_MOE_K32,
// default on). Callers choosing between 64- and 128-row tiles use it: that kernel moves the
// crossover from ~3072 to ~2048 tokens.
bool pfm_moe_gemm_qi8_k32_enabled();

// Dense fused-decode GEMM only: Q4_K / Q5_K / Q6_K (the routed predicate above stays Q4_K/Q5_K).
bool pf_dense_gemm_qi8_supported(int ggml_type);
// Largest M the fused quantized-B GEMM accepts (SPARKINFER_PREFILL_QB_MAX_M, default 512). Past it
// the launcher declines, so a caller that would give up a better path to try it can check first.
int pf_dense_gemm_qi8_max_m();
// This thread's override of that limit, for one pass (0 clears it): a caller that holds every
// weight the longer path needs already converted can send a short pass down that path, exactly as
// SPARKINFER_PREFILL_QB_MAX_M would, without moving it for anyone else.
void pf_dense_gemm_qi8_max_m_override(int m);

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
                                   const signed char* A_pack = nullptr,
                                   // QB_F_* bits below; 0 = the plain launch.
                                   int flags = 0);
// `partials` already holds zeros for [0, M*N) (its last reader cleared it), so the split-K
// atomic path skips its memset.
constexpr int QB_F_ZEROED = 1;
// Launch the 128-wide split kernel with programmatic stream serialization: its weight-decode warps
// start while the previous kernel (which must trigger launch_dependents) still runs, and the
// A-staging and MMA warps wait on that grid first. Only with QB_F_ZEROED -- a memset node between
// the two kernels would break the programmatic edge.
constexpr int QB_F_PDL = 2;

// The same GEMM with the residual add folded into its split-K reduce:
// X[m][n] = bf16(X[m][n] + bf16(acc * sx[m] * row_scale[n])), exactly what the reduce into a
// scratch output followed by launch_prefill_add(X, out, X) computes. Returns false, launching
// nothing, unless the launch takes the split-K arm (then the caller runs the two-step form).
// SPARKINFER_QB_RESID_REDUCE=0 disables it.
bool launch_prefill_gemm_qi8_dense_resid(int ggml_type, const signed char* A_i8, const float* sx,
                                         const void* W_q, const float* row_scale, void* X_bf16,
                                         int M, int N, int K, cudaStream_t stream, int* partials,
                                         int partials_splits, const signed char* A_pack = nullptr);

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
