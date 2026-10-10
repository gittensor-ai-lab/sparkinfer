#include "sparkinfer/kernels/prefill_nvfp4.h"
#include "sparkinfer/kernels/compressed_tensors.h"

#include <cstdio>
#include <cstdlib>

// Linkable fallbacks keep the runtime build independent of CUTLASS. The SM120 implementation
// replaces these symbols when BUILD_NVFP4_KERNELS is enabled for si_fused.
#ifndef SPARKINFER_BUILD_NVFP4
namespace sparkinfer::kernels {
bool prefill_nvfp4_supported(int, int, int) { return false; }
size_t prefill_nvfp4_data_bytes(int r, int c) { return ((size_t)r * c + 1) / 2; }
size_t prefill_nvfp4_scale_bytes_a(int, int) { return 0; }
size_t prefill_nvfp4_scale_bytes_b(int, int) { return 0; }
size_t prefill_nvfp4_workspace_bytes(int, int, int) { return 0; }
bool launch_prefill_nvfp4_quant_a(const void*, void*, void*, int, int, cudaStream_t) { return false; }
bool launch_prefill_nvfp4_rmsnorm_quant_a_exact(const void*, const void*, void*, void*, void*,
                                                int, int, float, cudaStream_t) { return false; }
bool launch_prefill_nvfp4_norm_add_norm_quant_exact(const void*, const int*, const float*,
                                                     const float*, const void*, float, void*,
                                                     const void*, float, void*, void*, void*, int,
                                                     int, cudaStream_t, bool) { return false; }
bool launch_prefill_nvfp4_norm_add_norm_quant_bf16_exact(const void*, const void*, const void*,
                                                          float, void*, const void*, float, void*,
                                                          void*, void*, int, int, cudaStream_t) {
    return false;
}
bool launch_prefill_nvfp4_gate_quant_a(const void*, const void*, void*, void*, int, int,
                                       cudaStream_t, int) { return false; }
bool launch_prefill_nvfp4_swiglu_quant_a(const void*, const void*, void*, void*, int, int,
                                         cudaStream_t) { return false; }
bool launch_prefill_nvfp4_interleave_gate_up(const void*, const void*, const void*, const void*,
                                             void*, void*, int, int, cudaStream_t) { return false; }
bool launch_prefill_nvfp4_gate_up_swiglu(const void*, const void*, const void*, const void*, void*,
                                         void*, int, int, int, float, float, void*,
                                         cudaStream_t) { return false; }
bool launch_prefill_nvfp4_gate_up_swiglu_pdl(const void*, const void*, const void*, const void*,
                                             void*, void*, int, int, int, float, float, void*,
                                             cudaStream_t) { return false; }
bool launch_prefill_nvfp4_swiglu_il_quant_a(const void*, void*, void*, int, int, cudaStream_t,
                                            bool) {
    return false;
}
bool launch_bf16_deinterleave_gate_up(const void*, void*, void*, int, int, cudaStream_t) {
    return false;
}
bool launch_muse_tail_fp4_exact(const void*, const void*, const void*, const void*, void*, void*,
                                void*, void*, void*, int, int, int, float, float, cudaStream_t,
                                bool) {
    return false;
}
bool launch_prefill_nvfp4_quant_b(const void*, void*, void*, int, int, cudaStream_t) { return false; }
bool launch_prefill_nvfp4_quant_b_slice(const void*, void*, void*, int, int, int, int,
                                        cudaStream_t) { return false; }
bool launch_prefill_nvfp4_quant_b_q4k(const void*, void*, void*, int, int, int, int,
                                      cudaStream_t) { return false; }
bool launch_prefill_nvfp4_quant_b_q6k(const void*, void*, void*, int, int, int, int,
                                      cudaStream_t) { return false; }
bool launch_prefill_nvfp4_gemm(const void*, const void*, const void*, const void*, void*, int, int,
                               int, void*, cudaStream_t, float, const void*) { return false; }
bool launch_prefill_nvfp4_gemm_pdl(const void*, const void*, const void*, const void*, void*, int,
                                   int, int, void*, cudaStream_t, float, const void*) {
    return false;
}
bool launch_prefill_nvfp4_gemm_fill(const void*, const void*, const void*, const void*, void*, int,
                                    int, int, void*, cudaStream_t, float, const void*) {
    return false;
}
bool prefill_nvfp4_swiglu_epilogue_on() { return false; }
bool launch_prefill_nvfp4_gemm_swiglu_quant(const void*, const void*, const void*, const void*,
                                            const void*, void*, void*, int, int, int, cudaStream_t,
                                            float) {
    return false;
}
bool launch_prefill_nvfp4_gate_up_swiglu_bf16(const void*, const void*, const void*, const void*,
                                              void*, int, int, int, float, float, cudaStream_t) {
    return false;
}
size_t prefill_nvfp4_workspace_bytes_f32(int, int, int) { return 0; }
bool launch_prefill_nvfp4_gemm_f32(const void*, const void*, const void*, const void*, void*, int,
                                   int, int, void*, cudaStream_t, float) { return false; }
bool launch_prefill_nvfp4_rmsnorm_quant_a(const void*, const void*, void*, void*, int, int, float,
                                          cudaStream_t) { return false; }
bool launch_ct_dequant_nvfp4_rows_i8(const void*, const void*, const float*, signed char*, float*,
                                     int, int, cudaStream_t) { return false; }
// These two have no way to report failure, and returning would leave the weights unwritten: an
// NVFP4 compressed-tensors checkpoint cannot be loaded without the NVFP4 kernels, so say so.
[[noreturn]] static void ct_nvfp4_unavailable() {
    fprintf(stderr, "[sparkinfer] this checkpoint stores NVFP4 weights, and this build has no NVFP4 "
                    "kernels (configured with SPARKINFER_NVFP4=OFF or without CUTLASS)\n");
    std::abort();
}
void launch_ct_dequant_nvfp4(const void*, const void*, float, void*, int, int, cudaStream_t) {
    ct_nvfp4_unavailable();
}
void launch_ct_dequant_nvfp4_dev(const void*, const void*, const float*, void*, int, int,
                                 cudaStream_t) {
    ct_nvfp4_unavailable();
}
bool launch_ct_nvfp4_pack_sfb(const void*, void*, int, int, cudaStream_t) { return false; }
bool launch_nvfp4_pack_sfa(const void*, void*, int, int, cudaStream_t) { return false; }
} // namespace sparkinfer::kernels
#endif
