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
// The same fold, byte-identical to launch_rmsnorm (fast-math) + launch_prefill_nvfp4_quant_a,
// and it also writes the bf16 norm to `out` (skipped when null). k must be a multiple of 16.
bool launch_prefill_nvfp4_rmsnorm_quant_a_exact(const void* src_bf16, const void* weight_bf16,
                                                void* out_bf16, void* dst_fp4, void* dst_sf,
                                                int m, int k, float eps,
                                                cudaStream_t stream = nullptr);
// launch_norm_then_add_acc followed by the exact fold above, in one pass and byte-identical to
// both: out_sum = residual + RMSNorm(acc * sxr * rs) * w_post, out_norm = RMSNorm(out_sum) * w_pre
// and its FP4 operand. k must be a multiple of 16 and at most 8192.
bool launch_prefill_nvfp4_norm_add_norm_quant_exact(const void* residual, const int* acc,
                                                     const float* sxr, const float* rs,
                                                     const void* w_post, float eps_post,
                                                     void* out_sum, const void* w_pre,
                                                     float eps_pre, void* out_norm, void* dst_fp4,
                                                     void* dst_sf, int m, int k,
                                                     cudaStream_t stream = nullptr,
                                                     // write zeros back over acc[0, m*k) once read
                                                     bool zero_acc = false);
// launch_norm_then_add (si_fused) followed by the exact fold above, in one pass and byte-identical
// to both, for a branch that is already bf16: out_sum = residual + RMSNorm(branch) * w_post,
// out_norm = RMSNorm(out_sum) * w_pre and its FP4 operand. k must be a multiple of 16, <= 8192.
bool launch_prefill_nvfp4_norm_add_norm_quant_bf16_exact(const void* residual, const void* branch,
                                                          const void* w_post, float eps_post,
                                                          void* out_sum, const void* w_pre,
                                                          float eps_pre, void* out_norm,
                                                          void* dst_fp4, void* dst_sf, int m,
                                                          int k, cudaStream_t stream = nullptr);
// Muse's attention gate fused into the A-operand quantize: x * sigmoid(g) straight to FP4, the
// same fold launch_prefill_gate_quant_rows_i8 does for the int8 o-projection.
// gate_ld: row pitch of `gate` in elements, when it is a COLUMN SLICE of a wider packed
// buffer instead of a tight [rows, cols] array. 0 = tight (every pre-existing caller).
bool launch_prefill_nvfp4_gate_quant_a(const void* src_bf16, const void* gate_bf16,
                                       void* dst_fp4, void* dst_sf,
                                       int m, int k, cudaStream_t stream = nullptr,
                                       int gate_ld = 0);
// Fuse the dense FFN's bf16-rounded SwiGLU producer into the down projection's FP4 A quantize.
bool launch_prefill_nvfp4_swiglu_quant_a(const void* gate_bf16, const void* up_bf16,
                                         void* dst_fp4, void* dst_sf,
                                         int m, int k, cudaStream_t stream = nullptr);
// The up projection GEMM with SwiGLU and the down projection's FP4 A quantize in its epilogue:
// dst = quant(silu(gate_bf16) * (alpha * A.B^T)), dst_sf in the SFA layout of an m x n operand.
// Returns false (nothing launched) when the shape or the knob declines.
bool prefill_nvfp4_swiglu_epilogue_on();
bool launch_prefill_nvfp4_gemm_swiglu_quant(const void* a, const void* sa, const void* b,
                                            const void* sb, const void* gate_bf16,
                                            void* dst_fp4, void* dst_sf, int m, int n, int k,
                                            cudaStream_t stream, float alpha = 1.f);
bool launch_prefill_nvfp4_quant_b(const void* src_bf16, void* dst_fp4, void* dst_sf,
                                  int n, int k, cudaStream_t stream = nullptr);
// Rows [n0, n0+rows) of the same `n`-row operand, read from a bf16 buffer holding ONLY those rows
// and written into the whole-operand dst_fp4/dst_sf. n0 and rows must be multiples of the 128-row
// scale-factor atom. A full sweep of slices is bit-identical to the whole-operand call above, so
// the caller's bf16 staging never has to be larger than one slice.
bool launch_prefill_nvfp4_quant_b_slice(const void* src_bf16, void* dst_fp4, void* dst_sf,
                                        int n, int n0, int rows, int k,
                                        cudaStream_t stream = nullptr);
// The same rows read straight from GGUF Q4_K bytes (`q4k_rows` points at row n0; k must be a
// multiple of 256). Bit-identical to launch_gguf_dequant into bf16 followed by the slice call
// above, without the bf16 staging.
bool launch_prefill_nvfp4_quant_b_q4k(const void* q4k_rows, void* dst_fp4, void* dst_sf,
                                      int n, int n0, int rows, int k,
                                      cudaStream_t stream = nullptr);
// Same rows from GGUF Q6_K bytes (210 B super-blocks; k a multiple of 256). Muse ffn_down is
// Q6_K, so the streamed B operand used to dequant-to-bf16 (266 MB) then quantize; this is the
// Q4_K kernel's mapping on that layout, bit-identical to the two-launch path.
bool launch_prefill_nvfp4_quant_b_q6k(const void* q6k_rows, void* dst_fp4, void* dst_sf,
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
// The same GEMM launched as a programmatic dependent of the kernel ahead of it on `stream` (PDL):
// it waits for that kernel before its first global read, so only its launch and prologue overlap.
bool launch_prefill_nvfp4_gemm_pdl(const void* a_fp4, const void* sfa,
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
// The same scatter for an A operand: row-major ue4m3 [m, k/16] into the SFA layout.
bool launch_nvfp4_pack_sfa(const void* scale_rowmajor, void* sfa, int m, int k,
                           cudaStream_t stream = nullptr);

} // namespace sparkinfer::kernels
