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

}} // namespace sparkinfer::kernels
