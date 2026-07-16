// Router projection GEMM: logits = input @ router_w, bf16 inputs, fp32 output.
// Small N (num_experts), so a simple tiled kernel with fp32 accumulation is
// plenty — this feeds straight into the top-k router kernel.

#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

template <int TILE>
__global__ void router_gemm_kernel(
    const __nv_bfloat16* __restrict__ input,    // [M, K]
    const __nv_bfloat16* __restrict__ router_w, // [K, N]
    float* __restrict__ logits,                 // [M, N]
    int M, int N, int K
) {
    __shared__ float sa[TILE][TILE];
    __shared__ float sb[TILE][TILE];
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        const int ak = k0 + threadIdx.x;
        const int bk = k0 + threadIdx.y;
        sa[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? __bfloat162float(input[(size_t)row * K + ak]) : 0.f;
        sb[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? __bfloat162float(router_w[(size_t)bk * N + col]) : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++) acc += sa[threadIdx.y][k] * sb[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) logits[(size_t)row * N + col] = acc;
}

#ifndef _MSC_VER
template __global__ void router_gemm_kernel<16>(const __nv_bfloat16*, const __nv_bfloat16*, float*, int, int, int);
#endif
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/moe.h"

void launch_moe_router_gemm(
    const void* input, const void* router_w, float* logits,
    int num_tokens, int hidden_dim, int num_experts, cudaStream_t stream
) {
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((num_experts + TILE - 1) / TILE, (num_tokens + TILE - 1) / TILE);
    router_gemm_kernel<TILE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input),
        reinterpret_cast<const __nv_bfloat16*>(router_w),
        logits, num_tokens, num_experts, hidden_dim);
}
#endif

} // namespace kernels
} // namespace sparkinfer
