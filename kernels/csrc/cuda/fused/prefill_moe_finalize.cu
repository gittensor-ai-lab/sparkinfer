// MoE prefill finalize for the dequant-on-read routed-expert path.
//
// The grouped-GEMM MoE path accumulates the routed experts into an fp32 scratch buffer and folds
// the shared expert in from there. The MMVQ routed path instead writes its weighted top-k sum
// straight to bf16, so this variant takes a bf16 `routed` and applies the shared expert + its
// per-token scalar gate in one grid-stride pass:
//
//   out[t,h] = routed[t,h] + dsw[t] * shared[t,h]
//
// `shared` == nullptr (no shared expert) or `dsw` == nullptr (ungated shared) degrade cleanly.
// `dsw` is already sigmoid-applied by the shared-gate kernel, matching the grouped path.
#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

__global__ void prefill_moe_finalize_bf16_kernel(
    const __nv_bfloat16* __restrict__ routed, const __nv_bfloat16* __restrict__ shared,
    const float* __restrict__ dsw, __nv_bfloat16* __restrict__ out, long total, int H) {
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)blockDim.x * gridDim.x) {
        float v = __bfloat162float(routed[i]);
        if (shared) {
            const float g = dsw ? dsw[i / H] : 1.f;
            v += g * __bfloat162float(shared[i]);
        }
        out[i] = __float2bfloat16(v);
    }
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
void launch_prefill_moe_finalize(const void* routed, const void* shared, const float* dsw,
                                 void* out, int n_tokens, int hidden, cudaStream_t stream) {
    const long total = (long)n_tokens * hidden;
    const int block = 256;
    long g = (total + block - 1) / block;
    const int grid = (int)(g > 65535 ? 65535 : g);
    prefill_moe_finalize_bf16_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(routed), reinterpret_cast<const __nv_bfloat16*>(shared),
        dsw, reinterpret_cast<__nv_bfloat16*>(out), total, hidden);
}
#endif

} // namespace kernels
} // namespace sparkinfer
