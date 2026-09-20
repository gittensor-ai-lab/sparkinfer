#pragma once
// The activation side of Ternary-Bonsai-2's weight basis.
//
// That checkpoint's weights were rotated before being assigned trits, so a kernel that reads them
// in their stored form has to hand them an activation rotated the same way: y = H . diag(s) . x,
// blockwise over `block` elements (1024 there), H the normalized Sylvester-Walsh Hadamard. This is
// the device twin of hadamard_rotate_activation() in runtime/include/sparkinfer/prism_hadamard.h,
// and the two are checked against each other.
//
// The alternative -- folding the rotation into the weights at load -- is what the loader does
// today. It is exactly equivalent arithmetically, but it leaves the weights dense, which costs the
// ternary packing: 0.5625 bytes/weight as Q4_K against 0.21875 stored.
#include <cuda_runtime.h>

namespace sparkinfer { namespace kernels {

// x and y are bf16 and may alias. `sign` holds one int8 (+1/-1) per index of the full `width`,
// which is the checkpoint's own sign vector for that width; `width` must be a multiple of `block`
// and `n_values` a multiple of `width`, so a batch of rows rotates in one call.
void launch_hadamard_rotate_bf16(const void* x_bf16, void* y_bf16, const signed char* sign,
                                 long n_values, int width, int block, cudaStream_t stream);

// The inverse, R^-1 = R^T = diag(s) . H: the transform first, then the signs.
void launch_hadamard_unrotate_bf16(const void* x_bf16, void* y_bf16, const signed char* sign,
                                   long n_values, int width, int block, cudaStream_t stream);

}}  // namespace sparkinfer::kernels
