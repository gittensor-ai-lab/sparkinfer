// Quantization helpers: int8 (per-tensor symmetric) and int4 block dequant.
// The int4 block path mirrors the structure of the Q4_K_M weights this stack
// serves (a shared scale per small block); it is a clean symmetric variant, not
// a byte-exact GGUF decoder.
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

// int8_t/uint8_t are signed char/unsigned char on this platform; use the
// builtin spellings in device code so NVRTC (no libstdc++) can compile it.

namespace sparkinfer {
namespace kernels {

__device__ __forceinline__ float q_warp_max(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, m));
    return v;
}

// Block-wide reduction of max|x| into scale[0], then quantize. One block total.
__global__ void quantize_i8_kernel(const __nv_bfloat16* __restrict__ in,
                                   signed char* __restrict__ out, float* __restrict__ scale, int n) {
    __shared__ float s_warp[32];
    float local = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) local = fmaxf(local, fabsf(__bfloat162float(in[i])));
    float wm = q_warp_max(local);
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = wm;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
        v = q_warp_max(v);
        if (threadIdx.x == 0) s_warp[0] = (v > 0.f ? v : 1.f) / 127.f;
    }
    __syncthreads();
    const float sc = s_warp[0];
    if (threadIdx.x == 0) *scale = sc;
    const float inv = 1.f / sc;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float q = roundf(__bfloat162float(in[i]) * inv);
        q = fminf(127.f, fmaxf(-127.f, q));
        out[i] = (signed char)q;
    }
}

__global__ void dequantize_i8_kernel(const signed char* __restrict__ in, const float* __restrict__ scale,
                                     __nv_bfloat16* __restrict__ out, int n) {
    const float sc = *scale;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = __float2bfloat16((float)in[i] * sc);
}

__global__ void dequant_int4_block_kernel(const unsigned char* __restrict__ packed,
                                          const __nv_bfloat16* __restrict__ scales,
                                          __nv_bfloat16* __restrict__ out, int n, int block) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const unsigned char byte = packed[i >> 1];
        int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
        int q = nib - 8;                       // signed [-8,7]
        float sc = __bfloat162float(scales[i / block]);
        out[i] = __float2bfloat16((float)q * sc);
    }
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/quant.h"

void launch_quantize_i8(const void* in_bf16, signed char* out, float* scale, int n, cudaStream_t stream) {
    quantize_i8_kernel<<<1, 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(in_bf16), out, scale, n);
}
void launch_dequantize_i8(const signed char* in, const float* scale, void* out_bf16, int n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    dequantize_i8_kernel<<<blocks, 256, 0, stream>>>(in, scale, reinterpret_cast<__nv_bfloat16*>(out_bf16), n);
}
void launch_dequant_int4_block(const unsigned char* packed, const void* scales_bf16, void* out_bf16,
                               int n, int block, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    dequant_int4_block_kernel<<<blocks, 256, 0, stream>>>(
        packed, reinterpret_cast<const __nv_bfloat16*>(scales_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_bf16), n, block);
}
#endif

} // namespace kernels
} // namespace sparkinfer
