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

// Batched and f32: a packed decode step scores every row of the batch against the head, and the
// head is the single largest weight in the model, so reading it once for the whole batch rather
// than once per row is most of what packing buys at the output end.
void launch_gemm_ptq1_f32(const void* x_bf16, const void* w_ptq1, float* y_f32,
                          int n_rows, int k, int batch, cudaStream_t stream);

// Decode's rotate-then-GEMV pair with the GEMV's int8 activation quantized once, in the rotation.
// launch_ptq1_rotate_quant is launch_hadamard_rotate_bf16 (same y_bf16, bit for bit) that also
// leaves x's quantized copy behind and returns a handle to it, or -1 when it did not (then it
// only rotated). launch_gemv_ptq1_q / _q_f32 read that handle instead of quantizing y_bf16
// themselves -- the same int8 values, so the same result -- and fall back to launch_gemv_ptq1 on
// -1. Several GEMVs can share one handle (the q/k/v legs, gate and up); it stays valid until
// about a dozen further ternary launches have gone by, i.e. within the layer that made it.
// The dp4a scratch the calls below quantize into (~10 MB). Reserved by the model that builds a
// decode shadow, before capturing any graph, and released with that shadow; until then, or if it
// fails, they decline and callers take the rotation + float path. No other model carries any.
bool ptq1_dp_reserve();
void ptq1_dp_release();
int launch_ptq1_rotate_quant(const void* x_bf16, void* y_bf16, const signed char* sign, int k,
                             int block, cudaStream_t stream);
void launch_gemv_ptq1_q(int handle, const void* x_bf16, const void* w_ptq1, void* y_bf16,
                        int n_rows, int k, cudaStream_t stream);
void launch_gemv_ptq1_q_f32(int handle, const void* x_bf16, const void* w_ptq1, float* y_f32,
                            int n_rows, int k, cudaStream_t stream);
// launch_prefill_swiglu(gate, up) followed by launch_ptq1_rotate_quant of the result, in one
// launch: y is the rotated bf16 SwiGLU and the handle is the same int8 copy. -1: not taken.
int launch_ptq1_swiglu_rotate_quant(const void* gate_bf16, const void* up_bf16, void* y_bf16,
                                    const signed char* sign, int k, int block,
                                    cudaStream_t stream);
// launch_add_rmsnorm2_q8(x, residual, weight -> out_sum, out_norm, out_q8) for one row followed by
// launch_ptq1_rotate_quant(out_norm -> y), in one launch, every output bit-identical to the pair.
// out_q8 may be null. -1: not taken (k past 8192 or not a multiple of the 1024 span).
int launch_ptq1_add_norm_rotate_quant(const void* x, const void* residual, const void* weight,
                                      void* out_sum, void* out_norm, void* out_q8, float eps,
                                      void* y_bf16, const signed char* sign, int k, int block,
                                      cudaStream_t stream);
// Two matrices of one shape against the same handle (gate and up) in one launch; each output is
// what launch_gemv_ptq1_q writes for it.
void launch_gemv_ptq1_q2(int handle, const void* x_bf16, const void* w0, const void* w1,
                         void* y0_bf16, void* y1_bf16, int n_rows, int k, cudaStream_t stream);
// The same pair over `batch` rows of x (row j at x + j*k), each row rotated and quantized exactly
// as launch_ptq1_rotate_quant does it alone, and a GEMM whose row j is bit-identical to
// launch_gemv_ptq1_q on that row. So a packed batch can read the decode shadow and still decode
// every row as it would alone. -1 / false: not taken (dp4a off, a shape it does not cover).
int launch_ptq1_rotate_quant_rows(const void* x_bf16, void* y_bf16, const signed char* sign, int k,
                                  int batch, int block, cudaStream_t stream);
bool launch_gemm_ptq1_q(int handle, const void* w_ptq1, void* y_bf16, int n_rows, int k,
                        int batch, cudaStream_t stream);

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

// ---- int8-activation arm ----------------------------------------------------------------------
// The activation is taken into the weights' basis and quantized in one pass: signs, a
// `block`-point Hadamard (1024 is the only span supported), then int8 with one float scale and
// one integer sum per 128 values -- the weight block's own granularity, so each block's dot
// product is exact in int32. q is [rows, k] int8, qd/qs are [rows, k/128]. Returns false for a
// shape it does not cover.
bool launch_ptq1_rotq_bf16(const void* x_bf16, const signed char* sign, signed char* q,
                           float* qd, int* qs, int rows, int k, int block, cudaStream_t stream);
// The same applied to bf16(silu(gate) * up), the value launch_prefill_swiglu would write: the
// down projection's activation straight from gate and up.
bool launch_ptq1_swiglu_rotq_bf16(const void* gate_bf16, const void* up_bf16,
                                  const signed char* sign, signed char* q, float* qd, int* qs,
                                  int rows, int k, int block, cudaStream_t stream);
// The same applied to the GDN gated RMSNorm of x with gate z and weight norm (head_dim 128 only),
// the value launch_qwen36_gated_norm / launch_prefill_gated_norm would write: ssm_out's
// activation straight from the recurrence output.
bool launch_ptq1_gnorm_rotq_bf16(const void* x_bf16, const void* z_bf16, const void* norm_bf16,
                                 float eps, const signed char* sign, signed char* q, float* qd,
                                 int* qs, int rows, int k, int head_dim, int block,
                                 cudaStream_t stream);
// The same applied to bf16(x * sigmoid(gate)), launch_qwen36_mul_sigmoid's value: the gated
// attention output straight into the output projection's activation.
bool launch_ptq1_gate_rotq_bf16(const void* x_bf16, const void* gate_bf16,
                                const signed char* sign, signed char* q, float* qd, int* qs,
                                int rows, int k, int block, cudaStream_t stream);

// y[n] = sum_k W[n,k] * x[k] with x given as launch_ptq1_rotq_*'s output (one row). w1/y1 run a
// second matrix of the same shape against the same activation in the same launch (gate and up);
// pass nullptr for one matrix.
bool launch_gemv_ptq1_i8_bf16(const signed char* xq, const float* xd, const int* xs,
                              const void* w0, const void* w1, void* y0_bf16, void* y1_bf16,
                              int n_rows, int k, cudaStream_t stream);
// The same for m activation rows (rotq output for m rows), y[m, n_rows] row-major. One row takes
// the GEMV above; up to 32 at a time take an int8 tensor-core kernel that decodes each weight
// block once for all of them. part (optional, part_cap floats): scratch for splitting k across
// CTAs when the matrix alone cannot fill the device (the down projection); results are then
// summed in split order by a second kernel.
bool launch_gemm_ptq1_i8_rows_bf16(const signed char* xq, const float* xd, const int* xs,
                                   const void* w0, const void* w1, void* y0_bf16, void* y1_bf16,
                                   int m, int n_rows, int k, cudaStream_t stream,
                                   float* part = nullptr, size_t part_cap = 0);
// The same two with fp32 output, for the LM head's logits: one matrix.
bool launch_gemv_ptq1_i8_f32(const signed char* xq, const float* xd, const int* xs,
                             const void* w, float* y, int n_rows, int k, cudaStream_t stream);
bool launch_gemm_ptq1_i8_rows_f32(const signed char* xq, const float* xd, const int* xs,
                                  const void* w, float* y, int m, int n_rows, int k,
                                  cudaStream_t stream, float* part = nullptr,
                                  size_t part_cap = 0);

// Prefill's int8 GEMM operands. launch_ptq1_rows_i8: weight rows [rows, k] -> int8 in the STORED
// (rotated) basis, t * round(s_b / scale), scale = max_b |s_b| / 127 per row -- the bytes the
// fused GEMM's PTQ1 arm decodes to. launch_ptq1_rotq_rows_i8: bf16 activation rows -> rotated,
// then int8 with one scale per row (d = amax/127), plus the k-tiled copy when qp is non-null.
// Only for a weight read against an activation that went through the second.
bool launch_ptq1_rows_i8(const void* w_ptq1, signed char* q, float* scale, int rows, int k,
                         cudaStream_t stream);
bool launch_ptq1_rotq_rows_i8(const void* x_bf16, const signed char* sign, signed char* q,
                              float* scale, signed char* qp, int rows, int k, int block,
                              cudaStream_t stream);
// launch_ptq1_rotq_rows_i8 with SwiGLU in front: the row rotated is bf16(silu(gate) * up), the bytes
// launch_prefill_swiglu_quant_i8 would have quantized. For the FFN's down leg in its stored
// (rotated) blocks; sign is that width's vector. k <= 17*block.
bool launch_ptq1_swiglu_rotq_rows_i8(const void* gate_bf16, const void* up_bf16,
                                     const signed char* sign, signed char* q, float* scale,
                                     signed char* qp, int rows, int k, int block, cudaStream_t st);

}}  // namespace sparkinfer::kernels
