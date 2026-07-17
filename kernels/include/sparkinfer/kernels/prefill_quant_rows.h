// Block-parallel, single-pass int8 row quantization for the prefill activations.
//
// See kernels/csrc/cuda/quant/prefill_quant_rows.cu for the design notes. pf_quantize_rows_i8
// (prefill_gemm_i8.cu, #422) gives each row ONE warp and reads the row twice — once for the amax
// reduction, once to divide and store. Measured on an RTX 5090 (main @ cb34dc1, ctx=4096): 17.1 ms
// per prefill over 200 calls, ~43% of DRAM peak, for a purely streaming op.
#pragma once

#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {

// Quantize x[rows, cols] (bf16) to int8 with one scale per row. Signature mirrors the launcher in
// prefill_gemm_i8.cu so it drops in as a one-line guard at the top of it:
//
//     if (launch_prefill_quant_rows_fast(x, q, scale, rows, cols, stream)) return;
//
// Returns true if a kernel was launched, false if the caller should run its own (ragged cols, or
// disabled by env).
//
// Numerics are unchanged: same amax over the row, same d = amax/127, same roundf, same zero-amax
// special case — so the int8 output and the per-row scale are bit-identical to the scalar path.
//
// Env knobs:
//   SPARKINFER_PREFILL_QUANT_ROWS  (default 1)  0 disables (A/B) -> falls through to #422's kernel.
bool launch_prefill_quant_rows_fast(const void* x, signed char* q, float* scale,
                                    int rows, int cols, cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace sparkinfer
