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
bool launch_prefill_nvfp4_rmsnorm_quant_a(const void* src_bf16, const void* weight_bf16,
                                          void* dst_fp4, void* dst_sf,
                                          int m, int k, float eps,
                                          cudaStream_t stream = nullptr);
// Muse's attention gate fused into the A-operand quantize: x * sigmoid(g) straight to FP4, the
// same fold launch_prefill_gate_quant_rows_i8 does for the int8 o-projection.
bool launch_prefill_nvfp4_gate_quant_a(const void* src_bf16, const void* gate_bf16,
                                       void* dst_fp4, void* dst_sf,
                                       int m, int k, cudaStream_t stream = nullptr);
// Fuse the dense FFN's bf16-rounded SwiGLU producer into the down projection's FP4 A quantize.
bool launch_prefill_nvfp4_swiglu_quant_a(const void* gate_bf16, const void* up_bf16,
                                         void* dst_fp4, void* dst_sf,
                                         int m, int k, cudaStream_t stream = nullptr);
bool launch_prefill_nvfp4_quant_b(const void* src_bf16, void* dst_fp4, void* dst_sf,
                                  int n, int k, cudaStream_t stream = nullptr);
// Rows [n0, n0+rows) of the same `n`-row operand, read from a bf16 buffer holding ONLY those rows
// and written into the whole-operand dst_fp4/dst_sf. n0 and rows must be multiples of the 128-row
// scale-factor atom. A full sweep of slices is bit-identical to the whole-operand call above, so
// the caller's bf16 staging never has to be larger than one slice.
bool launch_prefill_nvfp4_quant_b_slice(const void* src_bf16, void* dst_fp4, void* dst_sf,
                                        int n, int n0, int rows, int k,
                                        cudaStream_t stream = nullptr);
// c_bf16 is the epilogue's source operand: non-null makes this compute
// D = alpha*(A*B) + C instead of D = alpha*(A*B), which is how a residual add is
// folded into the block-scaled GEMM rather than run as a separate full-tensor pass.
// c_bf16 may alias d_bf16 (the in-place x += proj form); the epilogue reads each
// output tile before it stores it.
size_t prefill_nvfp4_workspace_bytes_f32(int m, int n, int k);
bool launch_prefill_nvfp4_gemm_f32(const void* a, const void* sa, const void* b,
                                   const void* sb, void* d, int m, int n, int k,
                                   void* ws, cudaStream_t st, float alpha = 1.f);
bool launch_prefill_nvfp4_gemm(const void* a_fp4, const void* sfa,
                               const void* b_fp4, const void* sfb,
                               void* d_bf16, int m, int n, int k,
                               void* workspace, cudaStream_t stream = nullptr,
                               float alpha = 1.f, const void* c_bf16 = nullptr);

// Scatter a compressed-tensors row-major UE4M3 scale [n, k/16] into the CUTLASS
// SFB layout launch_prefill_nvfp4_gemm expects for B. Packed E2M1 bytes are
// already the same nibble order as launch_prefill_nvfp4_quant_b, so they are
// used as-is. The checkpoint's tensor-wide global_scale is applied as GEMM
// alpha (1/global_scale), not folded into these UE4M3 bytes.
bool launch_ct_nvfp4_pack_sfb(const void* scale_rowmajor, void* sfb,
                              int n, int k, cudaStream_t stream = nullptr);

} // namespace sparkinfer::kernels
