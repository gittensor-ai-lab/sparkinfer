#include "sparkinfer/kernels/prefill_ptq1_fp4.h"

// Linkable fallbacks for a build without the sm_120a library; prefill_ptq1_fp4_sm120.cu replaces
// them when BUILD_NVFP4_KERNELS is on.
#ifndef SPARKINFER_BUILD_NVFP4
namespace sparkinfer { namespace kernels {
bool ptq1_fp4_gemm_supported(int, int) { return false; }
size_t ptq1_fp4_act_bytes(int rows, int k) { return (size_t)rows * k / 2 + (size_t)rows * k / 16; }
bool launch_ptq1_rotq_fp4(const void*, const void*, const signed char*, void*, int, int, int,
                          cudaStream_t) { return false; }
bool launch_ptq1_fp4_gemm(const void*, int, int, const void* const*, void* const*, const int*, int,
                          bool, float*, size_t, cudaStream_t) { return false; }
bool launch_ptq1_fp4_rows_bf16(const void*, const void*, const float*, const void*, const void*,
                               void*, void*, int, int, int, cudaStream_t, float*, size_t) {
    return false;
}
}}  // namespace sparkinfer::kernels
#endif
