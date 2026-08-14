#include "sparkinfer/kernels/prefill_nvfp4.h"

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
bool launch_prefill_nvfp4_gate_quant_a(const void*, const void*, void*, void*, int, int,
                                       cudaStream_t) { return false; }
bool launch_prefill_nvfp4_swiglu_quant_a(const void*, const void*, void*, void*, int, int,
                                         cudaStream_t) { return false; }
bool launch_prefill_nvfp4_quant_b(const void*, void*, void*, int, int, cudaStream_t) { return false; }
bool launch_prefill_nvfp4_gemm(const void*, const void*, const void*, const void*, void*, int, int,
                               int, void*, cudaStream_t) { return false; }
} // namespace sparkinfer::kernels
#endif
