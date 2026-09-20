#pragma once
// PTQ1_0 read in its stored form: 128 trits per 28-byte block with one FP16 scale, 1.75 bits per
// weight. Ternary-Bonsai-2's own packing, consumed directly rather than expanded.
//
// The loader's alternative is to un-rotate the weights and refit them to Q4_K, which is exact
// arithmetic and works on every kernel already here, but spends the packing: 0.5625 bytes/weight
// against the 0.21875 the file stores, so a 5.95 GB checkpoint occupies ~15 GB.
//
// The catch is the basis. These weights were rotated before being assigned trits, so `x` must
// already carry the matching rotation -- launch_hadamard_rotate_bf16 in kernels/hadamard.h. A
// caller that forgets is not approximately right; it is reading the weights in the wrong basis.
#include <cuda_runtime.h>

namespace sparkinfer { namespace kernels {

// y[n] = sum_k W[n,k] * x[k], W row-major over n with k contiguous, k a multiple of 128.
// x and y are bf16; x is expected in the weights' rotated basis.
void launch_gemv_ptq1(const void* x_bf16, const void* w_ptq1, void* y_bf16,
                      int n_rows, int k, cudaStream_t stream);

// The same, writing f32 -- the LM head's logits are f32 and are read as such downstream.
void launch_gemv_ptq1_f32(const void* x_bf16, const void* w_ptq1, float* y_f32,
                          int n_rows, int k, cudaStream_t stream);

}}  // namespace sparkinfer::kernels
