// Skinny bf16 prefill GEMM for narrow outputs (n_out << tile width).
//
// See kernels/csrc/cuda/fused/prefill_gemm_skinny.cu for the design notes. pf_gemm_kernel tiles the
// output 128x128, which is right for the big projections but pathological for the Gated-DeltaNet
// gate projections (ssm_alpha / ssm_beta, n_out == 32): the grid is
// ((32+127)/128, (4096+127)/128) = 1 x 32 = 32 blocks on a 170-SM part, and 96 of every 128 output
// columns are padding. Measured on an RTX 5090 (main @ cb34dc1, ctx=4096): 7.5 ms per prefill over
// 48 calls at ~6.9 TFLOP/s against a ~255 TFLOP/s bf16 tensor peak — compute-bound (A is L2-resident
// across the alpha/beta pair), so the fix is a full grid and tensor cores, not better staging.
#pragma once

#include <cuda_runtime.h>

namespace sparkinfer {
namespace kernels {

// Launch a bandwidth-shaped GEMM for narrow n_out. Signature mirrors `launch_prefill_gemm` so it
// drops in as a one-line guard at the top of that launcher:
//
//     if (launch_prefill_gemm_skinny(A, W, C, M, N, K, stream)) return;
//
// Returns true if a kernel was launched, false if the caller should run its own GEMM (n_out too
// wide to benefit, ragged K, or disabled by env).
//
// Accumulates in fp32 and emits bf16, exactly like pf_gemm_kernel; only the blocking changes, so
// the result matches to fp32 reassociation.
//
// Env knobs:
//   SPARKINFER_PREFILL_GEMM_SKINNY  (default 1)  0 disables (A/B) -> falls through to pf_gemm.
bool launch_prefill_gemm_skinny(const void* A, const void* W, void* C,
                                int M, int N, int K, cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace sparkinfer
