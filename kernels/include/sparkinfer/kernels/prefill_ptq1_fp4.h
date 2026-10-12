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
bool launch_ptq1_fp4_gemm(const void* a, int m, int k, const void* const* w, void* const* c,
                          const int* n, int nleg, bool resid, float* part, size_t part_cap,
                          cudaStream_t stream);
// The packed decode's rows GEMM (launch_gemm_ptq1_i8_rows_bf16's shapes, 17..32 rows) on the FP4
// tensor cores: xf / xsf are the activation's NVFP4 copy that ptq1_rotq_kernel writes beside its
// int8 one (64 bytes and 8 ue4m3 scales per 128-value block, in the int8 kernel's k order), xsum
// each block's sum of those NVFP4 values, [block][32 tokens]. The tiles and the k split are that
// kernel's. False, launching nothing, where the shape does not fit.
bool launch_ptq1_fp4_rows_bf16(const void* xf, const void* xsf, const float* xsum, const void* w0,
                               const void* w1, void* y0, void* y1, int m, int n_rows, int k,
                               cudaStream_t stream, float* part, size_t part_cap);

}}  // namespace sparkinfer::kernels
