// Tiled GEMM (bf16 in, fp32 accumulate, bf16 out).
//   C = alpha * A[M,K] @ B[K,N] + beta * C[M,N]   (all row-major)
//
// A shared-memory tiled kernel. This is the portable CUDA-core baseline that
// runs on every target including sm_120 (RTX 5090); the tensor-core / CuTe path
// is a separate experimental build. Correctness here is the reference the rest
// of the stack is validated against.

#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/gemm.h"   // at file scope so GemmConfig is visible
#endif

namespace sparkinfer {
namespace kernels {

template <int TILE>
__global__ void gemm_kernel(
    const __nv_bfloat16* __restrict__ A,   // [M,K]
    const __nv_bfloat16* __restrict__ B,   // [K,N]
    __nv_bfloat16* __restrict__ C,         // [M,N]
    int M, int N, int K, float alpha, float beta
) {
    __shared__ float sa[TILE][TILE];
    __shared__ float sb[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        const int ak = k0 + threadIdx.x;
        const int bk = k0 + threadIdx.y;
        sa[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? __bfloat162float(A[(size_t)row * K + ak]) : 0.f;
        sb[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? __bfloat162float(B[(size_t)bk * N + col]) : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++) acc += sa[threadIdx.y][k] * sb[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N) {
        const size_t idx = (size_t)row * N + col;
        const float prev = (beta != 0.f) ? __bfloat162float(C[idx]) : 0.f;
        C[idx] = __float2bfloat16(alpha * acc + beta * prev);
    }
}

#ifndef _MSC_VER
template __global__ void gemm_kernel<16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int, float, float);
#endif
// fp32-output variant for logit projections (LM head).
template <int TILE>
__global__ void gemm_f32_kernel(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C, int M, int N, int K
) {
    __shared__ float sa[TILE][TILE];
    __shared__ float sb[TILE][TILE];
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        const int ak = k0 + threadIdx.x;
        const int bk = k0 + threadIdx.y;
        sa[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? __bfloat162float(A[(size_t)row * K + ak]) : 0.f;
        sb[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? __bfloat162float(B[(size_t)bk * N + col]) : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++) acc += sa[threadIdx.y][k] * sb[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

#ifndef _MSC_VER
template __global__ void gemm_f32_kernel<16>(const __nv_bfloat16*, const __nv_bfloat16*, float*, int, int, int);
#endif
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
void launch_gemm(
    const void* A, const void* B, void* C,
    int M, int N, int K, float alpha, float beta,
    const GemmConfig& cfg, cudaStream_t stream
) {
    (void)cfg;
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_kernel<TILE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A),
        reinterpret_cast<const __nv_bfloat16*>(B),
        reinterpret_cast<__nv_bfloat16*>(C), M, N, K, alpha, beta);
}

void launch_batched_gemm(
    const void** A, const void** B, void** C,
    int batch, int M, int N, int K, float alpha, float beta,
    const GemmConfig& cfg, cudaStream_t stream
) {
    (void)cfg;
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    for (int i = 0; i < batch; i++) {
        gemm_kernel<TILE><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(A[i]),
            reinterpret_cast<const __nv_bfloat16*>(B[i]),
            reinterpret_cast<__nv_bfloat16*>(C[i]), M, N, K, alpha, beta);
    }
}

void launch_linear_f32(const void* A, const void* B, float* C,
                       int M, int N, int K, cudaStream_t stream) {
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_f32_kernel<TILE><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A),
        reinterpret_cast<const __nv_bfloat16*>(B), C, M, N, K);
}
#endif

} // namespace kernels
} // namespace sparkinfer
