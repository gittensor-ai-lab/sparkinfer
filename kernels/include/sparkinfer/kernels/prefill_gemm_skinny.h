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

// Same GEMM for a PAIR of narrow weights over the same A (the GDN ssm_alpha / ssm_beta gates):
// C0 = A @ W0^T, C1 = A @ W1^T. One pass stages A and both weights, so the K-slice chain -- which is
// what the shape actually costs at short prompt length, not the mma work -- is walked once instead
// of twice, and the slices are cp.async double-buffered so the next one lands under the current
// one's math. Each output element still accumulates over K in the same 16-wide wmma steps in the
// same order as the single-weight kernel, so the results are bit-identical to two skinny calls.
//
// Returns false (having launched nothing) when the shape does not fit; the caller then runs the two
// projections separately.
//
// Env knobs:
//   SPARKINFER_PREFILL_GEMM_SKINNY2 (default 1)  0 disables (A/B) -> two separate projections.
bool launch_prefill_gemm_skinny_pair(const void* A, const void* W0, const void* W1,
                                     void* C0, void* C1,
                                     int M, int N, int K, cudaStream_t stream = nullptr);

}  // namespace kernels
}  // namespace sparkinfer
