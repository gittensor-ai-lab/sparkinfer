#pragma once
#include <cuda_runtime.h>

// int8 tensor-core GEMM for Qwythos (Qwen3.5) batched prefill.
//
// The batched-prefill projections/FFN are weight-bound bf16 tensor-core GEMMs. On sm_120 the
// int8 tensor cores run ~2-3x the bf16 throughput, and because the GGUF weights are already
// stored at 4-6 bit (Q4_K/Q6_K), quantizing the dequantized weight to int8 is *strictly higher
// precision than what is stored* — so the projection outputs are unchanged at the gate level
// (measured rel_l2 vs fp32 matches the bf16 kernel to 4 decimals).
//
// launch_prefill_gemm_i8 mirrors the bf16 prefill GEMM tiling exactly (128x128 output tile, 8
// warps, 2x4 accumulator fragments, BK=32, cp.async double-buffer) and folds the dequant into the
// store epilogue, so it is a 1:1 drop-in replacement that still emits bf16 C. Weights are the
// native GGUF [out,in] (=[N,K]) layout; the per-output-row weight scales are computed once and
// kept resident, the per-token activation scales are computed per prefill pass.

namespace sparkinfer { namespace kernels {

// Per-row symmetric int8 quantization: scale[r] = max_c|x[r,c]| / 127,
// q[r,c] = round(x[r,c] / scale[r]).  x: [rows,cols] bf16 -> q: [rows,cols] int8, scale: [rows] fp32.
// One warp per row. Used for both the per-token activation A and (once, resident) the weight W.
// Muse: fold attn *= sigmoid(gate) into the row-quantize load phase (bit-identical).
bool launch_prefill_gate_quant_rows_i8(const void* x, const void* gate, signed char* q,
                                      float* scale, int rows, int cols, cudaStream_t stream);

void launch_prefill_quantize_rows_i8(const void* x_bf16, signed char* q, float* scale,
                                     int rows, int cols, cudaStream_t stream = nullptr);

// int8 GEMM:  C[M,N] = A[M,K] @ W^T,  W native GGUF [N,K] row-major (so C[m,n]=sum_k A[m,k]*W[n,k]).
// A/W int8 with per-row scales sx[M] (per token) and sw[N] (per output channel). Output C is bf16
// with the dequant sx[m]*sw[n] fused into the store. Drop-in for the bf16 launch_prefill_gemm.
void launch_prefill_gemm_i8(const signed char* A, const signed char* W,
                            const float* sx, const float* sw, void* C,
                            int M, int N, int K, cudaStream_t stream = nullptr);

// Residual-fused int8 GEMM: C[m,n] = bf16(C[m,n] + bf16(acc*sx*sw)) -- pass the residual tensor as
// C to fold the post-projection "x += out" into the store (same two-step rounding as the separate
// add kernel, so the result is bit-identical while skipping the ao scratch round-trip + add pass).
void launch_prefill_gemm_i8_resid(const signed char* A, const signed char* W,
                                  const float* sx, const float* sw, void* C,
                                  int M, int N, int K, cudaStream_t stream = nullptr);

// Split-K variant for skinny GEMMs (one 128-row M tile, narrow n_out). The launch above puts one
// 128x128 output tile in a block, so a projection whose n_out is small leaves most of the device
// idle: on an RTX 5090 a 2-block launch (Muse Glimmer's attn k/v, n_out=256) takes the same ~69 us
// as a 32-block one, because a block streams its weight slice at only ~12.3 GB/s and the device
// does not saturate until ~80 blocks. This splits the K loop across blockIdx.z, accumulates the
// int32 tiles into `partials` (M*N int32, caller-owned, zeroed here) with atomicAdd, and scales
// them in a second pass. int32 accumulation is exact and associative, so the result is BIT-
// IDENTICAL to the single-block launcher; only the block count changes.
//
// Returns false -- caller must run launch_prefill_gemm_i8[_resid] itself -- when the shape does not
// want splitting (grid already fills the device, M > 128, K too short), when `partials` is null, or
// when SPARKINFER_PREFILL_GEMM_SPLITK=0 disables it (A/B).
bool launch_prefill_gemm_i8_splitk(const signed char* A, const signed char* W,
                                   const float* sx, const float* sw, void* C,
                                   int M, int N, int K, int* partials, bool resid,
                                   cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
