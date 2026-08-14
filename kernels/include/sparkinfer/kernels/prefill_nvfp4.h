#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace sparkinfer::kernels {

// Experimental SM120 native block-scaled NVFP4 dense GEMM support.
bool prefill_nvfp4_supported(int m, int n, int k);
size_t prefill_nvfp4_data_bytes(int rows, int cols);
size_t prefill_nvfp4_scale_bytes_a(int m, int k);
size_t prefill_nvfp4_scale_bytes_b(int n, int k);
size_t prefill_nvfp4_workspace_bytes(int m, int n, int k);

bool launch_prefill_nvfp4_quant_a(const void* src_bf16, void* dst_fp4, void* dst_sf,
                                  int m, int k, cudaStream_t stream = nullptr);
// Muse's attention gate fused into the A-operand quantize: x * sigmoid(g) straight to FP4, the
// same fold launch_prefill_gate_quant_rows_i8 does for the int8 o-projection.
bool launch_prefill_nvfp4_gate_quant_a(const void* src_bf16, const void* gate_bf16,
                                       void* dst_fp4, void* dst_sf,
                                       int m, int k, cudaStream_t stream = nullptr);
bool launch_prefill_nvfp4_quant_b(const void* src_bf16, void* dst_fp4, void* dst_sf,
                                  int n, int k, cudaStream_t stream = nullptr);
bool launch_prefill_nvfp4_gemm(const void* a_fp4, const void* sfa,
                               const void* b_fp4, const void* sfb,
                               void* d_bf16, int m, int n, int k,
                               void* workspace, cudaStream_t stream = nullptr);

} // namespace sparkinfer::kernels
