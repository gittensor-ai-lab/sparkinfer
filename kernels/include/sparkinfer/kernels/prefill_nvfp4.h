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
// g_ld = row stride of the gate (0 => k); lets it be a column slice of the stacked
// q|gate|k|v GEMM output read in place.
bool launch_prefill_nvfp4_gate_quant_a(const void* src_bf16, const void* gate_bf16,
                                       void* dst_fp4, void* dst_sf,
                                       int m, int k, cudaStream_t stream = nullptr,
                                       int g_ld = 0);
// Fuse the dense FFN's bf16-rounded SwiGLU producer into the down projection's FP4 A quantize.
// ld_src = row stride of gate/up (0 => k); lets both be column slices of one stacked gate|up
// GEMM output. The FP4 destination stays contiguous [m, k].
bool launch_prefill_nvfp4_swiglu_quant_a(const void* gate_bf16, const void* up_bf16,
                                         void* dst_fp4, void* dst_sf,
                                         int m, int k, cudaStream_t stream = nullptr,
                                         int ld_src = 0);
bool launch_prefill_nvfp4_quant_b(const void* src_bf16, void* dst_fp4, void* dst_sf,
                                  int n, int k, cudaStream_t stream = nullptr);
bool launch_prefill_nvfp4_gemm(const void* a_fp4, const void* sfa,
                               const void* b_fp4, const void* sfb,
                               void* d_bf16, int m, int n, int k,
                               void* workspace, cudaStream_t stream = nullptr,
                               float alpha = 1.f);

// Scatter a compressed-tensors row-major UE4M3 scale [n, k/16] into the CUTLASS
// SFB layout launch_prefill_nvfp4_gemm expects for B. Packed E2M1 bytes are
// already the same nibble order as launch_prefill_nvfp4_quant_b, so they are
// used as-is. The checkpoint's tensor-wide global_scale is applied as GEMM
// alpha (1/global_scale), not folded into these UE4M3 bytes.
bool launch_ct_nvfp4_pack_sfb(const void* scale_rowmajor, void* sfb,
                              int n, int k, cudaStream_t stream = nullptr);

} // namespace sparkinfer::kernels
