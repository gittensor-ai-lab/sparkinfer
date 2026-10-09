#pragma once
// Short-prompt prefill GEMMs of a ternary (PTQ1_0) checkpoint on the FP4 tensor cores.
//
// launch_ptq1_rotq_fp4: bf16 activation rows (bf16(silu(x) * up) first when up_bf16 is set) rotated
// into the weights' basis exactly as launch_ptq1_rotq_rows_i8 rotates them, then quantized to NVFP4:
// `a` receives the packed e2m1 values (k/2 bytes a row, low nibble first) followed by a row-major
// ue4m3 scale per 16 values (k/16 bytes a row) -- ptq1_fp4_act_bytes(rows, k) in all.
//
// launch_ptq1_fp4_gemm: C_l = A @ W_l^T for up to three legs W_l that share A (row-major PTQ1_0
// blocks, n_l x k), each trit decoded straight into e2m1 in shared memory and the block's fp16 scale
// carried as an e2m1 magnitude times a ue4m3, then mma.sync kind::mxf4nvf4. resid: C_l += it.
// `part` (optional, part_cap floats) lets a grid too small for the device split K; the partials are
// then summed in a fixed order. False, launching nothing, where the shape does not fit.
#include <cstddef>
#include <cuda_runtime.h>

namespace sparkinfer { namespace kernels {

bool ptq1_fp4_gemm_supported(int m, int k);
size_t ptq1_fp4_act_bytes(int rows, int k);
bool launch_ptq1_rotq_fp4(const void* x_bf16, const void* up_bf16, const signed char* sign,
                          void* a, int rows, int k, int block, cudaStream_t stream);
// launch_ptq1_rotq_fp4 with the kernel that produced its input folded in, so its bf16 output is
// never written and read back; each computes that kernel's own values, so `a` receives the bytes
// the two passes wrote. norm: rmsnorm_kernel(x, weight) (out_norm, when set, still receives its
// bf16 row); gnorm: pf_gated_norm_kernel(x, z, weight) per head_dim-wide head (128 only); gate:
// pf_mul_sigmoid_kernel(x, gate), gate's rows gate_ld apart (0: k). False, launching nothing,
// where they do not apply.
bool launch_ptq1_norm_rotq_fp4(const void* x_bf16, const void* weight_bf16, float eps,
                               void* out_norm, const signed char* sign, void* a, int rows, int k,
                               int block, cudaStream_t stream);
bool launch_ptq1_gnorm_rotq_fp4(const void* x_bf16, const void* z_bf16, const void* weight_bf16,
                                float eps, int head_dim, const signed char* sign, void* a,
                                int rows, int k, int block, cudaStream_t stream);
bool launch_ptq1_gate_rotq_fp4(const void* x_bf16, const void* gate_bf16, int gate_ld,
                               const signed char* sign, void* a, int rows, int k, int block,
                               cudaStream_t stream);
bool launch_ptq1_fp4_gemm(const void* a, int m, int k, const void* const* w, void* const* c,
                          const int* n, int nleg, bool resid, float* part, size_t part_cap,
                          cudaStream_t stream);

}}  // namespace sparkinfer::kernels
