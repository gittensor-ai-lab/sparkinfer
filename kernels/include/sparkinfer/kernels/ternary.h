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

// A batch of activations against the same weights: y[b,n] = sum_k W[n,k] * x[b,k], x row-major
// over the batch and y likewise. This is what prefill needs -- it projects N tokens at once, and
// a per-token GEMV would reload the whole weight matrix for each of them.
void launch_gemm_ptq1(const void* x_bf16, const void* w_ptq1, void* y_bf16,
                      int n_rows, int k, int batch, cudaStream_t stream);

// The same, writing f32 -- the LM head's logits are f32 and are read as such downstream.
void launch_gemv_ptq1_f32(const void* x_bf16, const void* w_ptq1, float* y_f32,
                          int n_rows, int k, cudaStream_t stream);

// A whole weight matrix decoded out of its ternary blocks and un-rotated into the architecture's
// basis, as ordinary bf16. Prefill uses this rather than a ternary GEMM so its existing projection
// branches -- FP8 GEMM, dequantize-then-requantize, plain GEMM -- keep working unchanged: they ask
// for bf16 weights and get them, into scratch, while the resident copy stays ternary.
void launch_ptq1_rows_unrotate_bf16(const void* w_ptq1, const signed char* sign, void* out_bf16,
                                    int n_rows, int k, int block, cudaStream_t stream);

// Embedding lookup from a ternary table: decodes one row per token into bf16. The row is still in
// the stored basis, so the caller applies launch_hadamard_unrotate_bf16 before using it as the
// residual -- token_embd is the one tensor whose rotation has to come off at runtime rather than
// being absorbed by a matmul.
void launch_embedding_ptq1(const int* tokens, const void* table_ptq1, void* out_bf16,
                           int n_tokens, int k, cudaStream_t stream);

// The same, with the stored rotation taken off in the same pass. Preferred: decoding to bf16 and
// rotating afterwards makes the transform sum 1024 already-rounded values, which is measurable.
void launch_embedding_ptq1_unrotate(const int* tokens, const void* table_ptq1,
                                    const signed char* sign, void* out_bf16,
                                    int n_tokens, int k, int block, cudaStream_t stream);

}}  // namespace sparkinfer::kernels
