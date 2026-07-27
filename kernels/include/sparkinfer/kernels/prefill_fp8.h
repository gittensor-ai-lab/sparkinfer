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

// fp8 GEMM: C[M,N] = A[M,K] @ W^T, W dequantized bf16 [N,K] row-major (C[m,n]=sum_k A[m,k]*W[n,k]).
// A/W e4m3 with per-row scales sx[M] (per token) and sw[N] (per output channel). Output C is bf16
// with the dequant sx[m]*sw[n] fused into the store. fp16 accumulate with a per-BK-tile fp32 flush.
void launch_prefill_gemm_fp8(const void* A, const void* W,
                             const float* sx, const float* sw, void* C,
                             int M, int N, int K, cudaStream_t stream = nullptr);

// Fused SwiGLU + per-row int8 quantize: q[r,:] = int8(silu(gate[r,:]) * up[r,:]) with
// scale[r] = amax_c|.| / 127. Replaces launch_prefill_swiglu + launch_prefill_quantize_rows_i8 on
// the ffn-wide intermediate (one block per row), removing its DRAM round-trip. Bit-identical.
void launch_prefill_swiglu_quant_i8(const void* gate, const void* up, signed char* q, float* scale,
                                    int rows, int cols, cudaStream_t stream = nullptr);

// Fused gated RMSNorm + per-row fp8 (e4m3) quantize, for MoE (Qwen3.6) GDN layers whose o_proj
// always takes the fp8 tensor-core path (moe_fp8). Numerically identical to
// launch_prefill_gated_norm followed by launch_prefill_quantize_rows_fp8, but skips materializing
// the bf16 intermediate in DRAM. See prefill.h for the int8 counterpart (dense/Qwythos GDN).
//   x/z: [N, v_heads*hd]   weight: [hd]   q: [N, v_heads*hd] e4m3   scale: [N]
// Returns false when head_dim != 128 or v_heads is not a multiple of 8.
bool launch_prefill_gated_norm_quant_fp8(const void* x, const void* z, const void* weight,
                                         void* q, float* scale,
                                         int n_tokens, int v_heads, int head_dim, float eps,
                                         cudaStream_t stream = nullptr);

}} // namespace sparkinfer::kernels
