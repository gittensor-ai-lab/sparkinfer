// Vector-store fused GGUF -> int8 row dequantization.
//
// See kernels/csrc/cuda/quant/dequant_rows_i8_fast.cu for the design notes. deq_rows_i8_kernel
// (dequant_gguf.cu, #464) maps thread t to value t of a 256-value super-block, so each thread stores
// ONE byte and a warp emits a 32-byte store where the memory system wants 128. Measured on an
// RTX 5090 (main @ c9e20d1, ctx=4096): 19.5 ms per prefill over 200 calls, ~31% of DRAM peak, for a
// kernel whose output is 6.9 GB of pure streaming int8.
#pragma once

#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {

// Fused GGUF -> int8 rows with 4-byte vector stores. Signature mirrors `launch_gguf_dequant_rows_i8`
// so it drops in as a one-line guard at the top of that launcher:
//
//     if (launch_gguf_dequant_rows_i8_fast(ggml_type, src, q, scale, rows, cols, stream)) return true;
//
// Returns true if a kernel was launched, false if the caller should run its own (unsupported type,
// a row shape outside the register budget, or disabled by env).
//
// BIT-IDENTICAL to #464's kernel: same deq_q4k_val / deq_q6k_val decode, same amax over the row
// (max is associative and exact in fp, so the different thread->value map cannot move it), same
// d = amax/127.f, same inv = 1.f/d guarded on d > 0, and the same roundf(v * inv) — v * inv, not
// v / d; those differ in the last ulp.
//
// Env knobs:
//   SPARKINFER_DEQUANT_ROWS_I8_FAST  (default 1)  0 disables (A/B) -> falls through to #464's kernel.
bool launch_gguf_dequant_rows_i8_fast(int ggml_type, const void* src, signed char* q, float* scale,
                                      int rows, int cols, cudaStream_t stream = nullptr);

// One-grid gate+up (or any two same-type tensors): blockIdx.x < rows → tensor0, else tensor1.
// Same bit-identical decode as the single-tensor fast path. Used to fill SMs on one stream
// without a second CUDA stream (multi-stream MoE prefill poisons decode graphs on this runtime).
bool launch_gguf_dequant_rows_i8_fast_pair(int ggml_type,
                                           const void* src0, signed char* q0, float* scale0,
                                           const void* src1, signed char* q1, float* scale1,
                                           int rows, int cols, cudaStream_t stream = nullptr);

// Sparse MoE: dequant only live experts listed in live_le[n_live] (group-local indices).
// src0/q0/scale0 are group bases (expert 0 of the group). expert_bytes = rows_per_expert * row_bytes.
// One grid of n_live * rows_per_expert blocks — avoids thousands of tiny contiguous launches.
bool launch_gguf_dequant_rows_i8_fast_gather(
    int ggml_type, const void* src0, signed char* q0, float* scale0,
    const int* live_le, int n_live, int rows_per_expert, int cols,
    size_t expert_bytes, cudaStream_t stream = nullptr);

bool launch_gguf_dequant_rows_i8_fast_gather_pair(
    int ggml_type,
    const void* src0, signed char* q0, float* scale0,
    const void* src1, signed char* q1, float* scale1,
    const int* live_le, int n_live, int rows_per_expert, int cols,
    size_t expert_bytes0, size_t expert_bytes1, cudaStream_t stream = nullptr);

// MoE group mask: grid = n_in * rows_per_expert (pair: 2x). Skips experts with counts[e]==0.
// Avoids D2H counts + host live-index build when combined with device tilemap.
bool launch_gguf_dequant_rows_i8_fast_mask(
    int ggml_type, const void* src0, signed char* q0, float* scale0,
    const int* counts, int e_base, int n_in, int rows_per_expert, int cols,
    size_t expert_bytes, cudaStream_t stream = nullptr);

bool launch_gguf_dequant_rows_i8_fast_mask_pair(
    int ggml_type,
    const void* src0, signed char* q0, float* scale0,
    const void* src1, signed char* q1, float* scale1,
    const int* counts, int e_base, int n_in, int rows_per_expert, int cols,
    size_t expert_bytes0, size_t expert_bytes1, cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace sparkinfer
