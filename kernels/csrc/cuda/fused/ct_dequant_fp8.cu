// FP8 (E4M3) weight dequant for HuggingFace "compressed-tensors" checkpoints -- see
// sparkinfer/kernels/compressed_tensors.h. No CUTLASS dependency (unlike the NVFP4 dequant
// alongside it in prefill_nvfp4_sm120.cu), so this lives in si_fused (built unconditionally)
// rather than si_nvfp4 (CUTLASS-gated, BUILD_NVFP4_KERNELS).

#include "sparkinfer/kernels/compressed_tensors.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace sparkinfer { namespace kernels {
namespace {

__global__ void ct_dequant_fp8_kernel(const __nv_fp8_e4m3* __restrict__ w,
                                      const __nv_bfloat16* __restrict__ scale,
                                      __nv_bfloat16* __restrict__ out, int rows, long cols) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long n = (long)rows * cols;
    if (i >= n) return;
    const int r = (int)(i / cols);
    const float s = __bfloat162float(scale[r]);
    out[i] = __float2bfloat16(float(w[i]) * s);
}

} // namespace

void launch_ct_dequant_fp8(const void* w_e4m3, const void* scale_bf16, void* out_bf16,
                           int rows, int cols, cudaStream_t stream) {
    const long n = (long)rows * cols;
    const int threads = 256;
    const long blocks = (n + threads - 1) / threads;
    ct_dequant_fp8_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(w_e4m3),
        reinterpret_cast<const __nv_bfloat16*>(scale_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_bf16), rows, cols);
}

void launch_ct_dequant_fp8_packed(const void* packed, void* out_bf16,
                                  int rows, int cols, cudaStream_t stream) {
    if (!packed || !out_bf16 || rows <= 0 || cols <= 0) return;
    const char* p = reinterpret_cast<const char*>(packed);
    launch_ct_dequant_fp8(p + (size_t)rows * 2, packed, out_bf16, rows, cols, stream);
}

}} // namespace sparkinfer::kernels
