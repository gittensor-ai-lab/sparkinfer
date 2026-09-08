// Spark-X2.5 elementwise kernels: GeGLU and the head-wise sigmoid attention gate.
// See sparkinfer/kernels/spark25.h for why these are separate kernels rather than modes on the
// existing SwiGLU / elementwise-gate paths.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "sparkinfer/kernels/spark25.h"

namespace sparkinfer { namespace kernels {

namespace {

// Exact (erf) GELU. erff is a real libdevice call, not one of the intrinsics --use_fast_math
// substitutes, so this stays the erf form under the fused target's compile options.
__device__ __forceinline__ float sp25_gelu(float x) {
    return 0.5f * x * (1.f + erff(x * 0.70710678118654752440f));   // 1/sqrt(2)
}

__global__ void sp25_geglu_kernel(const __nv_bfloat16* __restrict__ gate,
                                  const __nv_bfloat16* __restrict__ up,
                                  __nv_bfloat16* __restrict__ out, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(sp25_gelu(__bfloat162float(gate[i])) * __bfloat162float(up[i]));
}

__global__ void sp25_mul_sigmoid_headwise_kernel(__nv_bfloat16* __restrict__ attn,
                                                 const __nv_bfloat16* __restrict__ gate,
                                                 int n_heads, int head_dim) {
    // blockIdx.y = token, blockIdx.x covers this token's n_heads*head_dim elements.
    const int tok = blockIdx.y;
    const int i   = blockIdx.x * blockDim.x + threadIdx.x;
    const int qdim = n_heads * head_dim;
    if (i >= qdim) return;
    const int h = i / head_dim;
    const float g = __bfloat162float(gate[(size_t)tok * n_heads + h]);
    const float s = 1.f / (1.f + __expf(-g));
    const size_t o = (size_t)tok * qdim + i;
    attn[o] = __float2bfloat16(__bfloat162float(attn[o]) * s);
}

} // namespace

void launch_spark25_geglu(const void* gate, const void* up, void* out, long n,
                          cudaStream_t stream) {
    if (n <= 0) return;
    const int threads = 256;
    const long blocks = (n + threads - 1) / threads;
    sp25_geglu_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up, (__nv_bfloat16*)out, n);
}

void launch_spark25_mul_sigmoid_headwise(void* attn, const void* gate, int n_tokens,
                                         int n_heads, int head_dim, cudaStream_t stream) {
    if (n_tokens <= 0 || n_heads <= 0 || head_dim <= 0) return;
    const int threads = 256;
    const int qdim = n_heads * head_dim;
    dim3 grid((unsigned)((qdim + threads - 1) / threads), (unsigned)n_tokens);
    sp25_mul_sigmoid_headwise_kernel<<<grid, threads, 0, stream>>>(
        (__nv_bfloat16*)attn, (const __nv_bfloat16*)gate, n_heads, head_dim);
}

}} // namespace sparkinfer::kernels
