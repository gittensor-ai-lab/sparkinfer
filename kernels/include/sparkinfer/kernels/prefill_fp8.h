#pragma once
#include <cuda_runtime.h>

// fp8 (e4m3) tensor-core GEMM for Qwythos (Qwen3.5) batched prefill at long context.
//
// The dense long-context (>96k) fallback keeps the Gated-DeltaNet projections in bf16 because the
// near-1-decay recurrence amplifies per-row int8 activation-quant error over the sequence (128k
// top1 ~0.31 for int8 vs ~0.69 for bf16). bf16 runs at half the int8 tensor-core rate, so those
// projections dominate the 128k prefill. e4m3 keeps a floating range (uniform *relative* error,
// unlike int8's uniform absolute step), holding the recurrence to bf16-like fidelity while running
// on the fp8 tensor cores at the full int8 rate (fp16 accumulate; fp32 accumulate is throttled to
// half on GeForce Blackwell).
//
// launch_prefill_gemm_fp8 mirrors launch_prefill_gemm_i8's tiling exactly (128x128 tile, 8 warps,
// 2x8 fragments, BK=64, cp.async double-buffer) and folds the dequant into the bf16 store epilogue,
// so it is a drop-in replacement for the bf16 GDN projection GEMM.

namespace sparkinfer { namespace kernels {

// Per-row symmetric fp8 quantization to a fixed target amax (so the fp16-accumulate GEMM cannot
// overflow K=4096): scale[r] = max_c|x[r,c]| / 16, q[r,c] = e4m3(x[r,c] / scale[r]).
// x: [rows,cols] bf16 -> q: [rows,cols] e4m3 (1 byte), scale: [rows] fp32. One warp per row.
void launch_prefill_quantize_rows_fp8(const void* x_bf16, void* q, float* scale,
                                      int rows, int cols, cudaStream_t stream = nullptr);

// Checkpoint SI_QTYPE_FP8 stores per-row scales as bf16. The GEMM epilogue wants fp32 sw[N]
// with the same multiply convention (W_bf16 = e4m3 * scale).
void launch_prefill_fp8_wscales_bf16(const void* scale_bf16, float* sw, int n,
                                     cudaStream_t stream = nullptr);

// fp8 GEMM: C[M,N] = A[M,K] @ W^T, W dequantized bf16 [N,K] row-major (C[m,n]=sum_k A[m,k]*W[n,k]).
// A/W e4m3 with per-row scales sx[M] (per token) and sw[N] (per output channel). Output C is bf16
// with the dequant sx[m]*sw[n] fused into the store. fp16 accumulate with a per-BK-tile fp32 flush.
void launch_prefill_gemm_fp8(const void* A, const void* W,
                             const float* sx, const float* sw, void* C,
                             int M, int N, int K, cudaStream_t stream = nullptr);

// Fused SwiGLU + per-row int8 quantize: q[r,:] = int8(silu(gate[r,:]) * up[r,:]) with
// scale[r] = amax_c|.| / 127. Replaces launch_prefill_swiglu + launch_prefill_quantize_rows_i8 on
// the ffn-wide intermediate (one block per row), removing its DRAM round-trip. Bit-identical.
// SwiGLU + per-row int8 quantize fed directly from the grouped GEMM's split-K int32 accumulator,
// skipping the reduce pass that would otherwise materialize gate/up as bf16 first. Returns false if
// the shape is not eligible (caller keeps reduce + launch_prefill_swiglu_quant_i8).
bool launch_prefill_swiglu_quant_i8_acc(const int* acc_g, const int* acc_u, const float* sxr,
                                        const float* rs_g, const float* rs_u,
                                        signed char* q, float* scale, int rows, int cols,
                                        cudaStream_t stream, signed char* qp = nullptr);

// Returns false only when `qp` was requested but the path that ran cannot write it.
bool launch_prefill_swiglu_quant_i8(const void* gate, const void* up, signed char* q, float* scale,
                                    int rows, int cols, cudaStream_t stream = nullptr,
                                    signed char* qp = nullptr);

}} // namespace sparkinfer::kernels
