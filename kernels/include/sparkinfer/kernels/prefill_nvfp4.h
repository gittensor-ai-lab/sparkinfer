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
// Fuse the dense FFN's bf16-rounded SwiGLU producer into the down projection's FP4 A quantize.
bool launch_prefill_nvfp4_swiglu_quant_a(const void* gate_bf16, const void* up_bf16,
                                         void* dst_fp4, void* dst_sf,
                                         int m, int k, cudaStream_t stream = nullptr);
bool launch_prefill_nvfp4_quant_b(const void* src_bf16, void* dst_fp4, void* dst_sf,
                                  int n, int k, cudaStream_t stream = nullptr);
bool launch_prefill_nvfp4_gemm(const void* a_fp4, const void* sfa,
                               const void* b_fp4, const void* sfb,
                               void* d_bf16, int m, int n, int k,
                               void* workspace, cudaStream_t stream = nullptr,
                               float alpha = 1.f);

// ---- plain (non block-scaled) e4m3 GEMM on the same SM120 tensor cores --------------------------
// A [m,k] row-major e4m3, B [n,k] row-major e4m3 (i.e. TN, K-major on both, which is the only
// layout the SM120 F8F6F4 collective accepts), D [m,n] row-major bf16.
// The per-row activation scale sx[m] and per-column weight scale sw[n] are applied INSIDE the
// epilogue as (acc * sx) * sw with a single round to bf16 -- the same association and the same
// single rounding pf_gemm_fp8_sk_epi_kernel performs, so a projection can move between this path
// and the hand-written split-K one bit-identically. Emitting bf16 here means no fp32 tile is ever
// written to memory.
// Returns false when NVFP4 support is compiled out, the device is not sm_120, the shape is not
// 16-element aligned, or CUTLASS declines the problem -- caller falls back to its own kernel.
bool prefill_ct_fp8_gemm_supported(int m, int n, int k);
size_t prefill_ct_fp8_gemm_workspace_bytes(int m, int n, int k);
bool launch_prefill_ct_fp8_gemm(const void* a_e4m3, const void* b_e4m3, void* d_bf16,
                                const float* sx, const float* sw,
                                int m, int n, int k, void* workspace,
                                cudaStream_t stream = nullptr);

// Scatter a compressed-tensors row-major UE4M3 scale [n, k/16] into the CUTLASS
// SFB layout launch_prefill_nvfp4_gemm expects for B. Packed E2M1 bytes are
// already the same nibble order as launch_prefill_nvfp4_quant_b, so they are
// used as-is. The checkpoint's tensor-wide global_scale is applied as GEMM
// alpha (1/global_scale), not folded into these UE4M3 bytes.
bool launch_ct_nvfp4_pack_sfb(const void* scale_rowmajor, void* sfb,
                              int n, int k, cudaStream_t stream = nullptr);

} // namespace sparkinfer::kernels
