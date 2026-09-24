// Blockwise normalized Walsh-Hadamard rotation of an activation, with a per-index sign.
// See sparkinfer/kernels/hadamard.h for why Ternary-Bonsai-2 needs it.
//
// One CUDA block per block-sized span, the span held in shared memory for the butterfly. At
// block = 1024 that is 4 KB of shared memory and ten butterfly stages, against the 1024 * width
// bytes of weight the matmul it feeds will read -- so it is not on the critical path, and is
// written for clarity rather than for the last few microseconds.
#include "sparkinfer/kernels/hadamard.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace sparkinfer { namespace kernels {

namespace {

constexpr int kThreads = 256;

// sign_first: R = H . diag(s), the rotation a stored weight expects.
// !sign_first: R^-1 = diag(s) . H, which undoes it.
template <bool sign_first>
__global__ void hadamard_span_kernel(const __nv_bfloat16* __restrict__ x,
                                     __nv_bfloat16* __restrict__ y,
                                     const signed char* __restrict__ sign,
                                     int width, int block, float norm) {
    extern __shared__ float sh[];
    const long span = blockIdx.x;                 // which block-sized span of the whole batch
    const long base = span * (long)block;
    const int sign_off = (int)(base % (long)width);   // signs repeat every `width` elements

    for (int i = threadIdx.x; i < block; i += blockDim.x) {
        float v = __bfloat162float(x[base + i]);
        if (sign_first) v *= (float)sign[sign_off + i];
        sh[i] = v;
    }
    __syncthreads();

    for (int len = 1; len < block; len <<= 1) {
        for (int i = threadIdx.x; i < block / 2; i += blockDim.x) {
            const int lo = ((i / len) * 2 * len) + (i % len);
            const int hi = lo + len;
            const float a = sh[lo], b = sh[hi];
            sh[lo] = a + b;
            sh[hi] = a - b;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < block; i += blockDim.x) {
        float v = sh[i] * norm;
        if (!sign_first) v *= (float)sign[sign_off + i];
        y[base + i] = __float2bfloat16(v);
    }
}

// The same transform for block = 1024, one warp per span and no barriers: each lane holds 32
// consecutive values, so the five butterfly stages with len < 32 are register-local and the five
// with len >= 32 pair a lane with lane ^ (len / 32) through a shuffle. The stages run in the same
// increasing-len order and each butterfly is the same a + b / a - b of the same two operands, so
// the result is bit-identical to hadamard_span_kernel's -- only the ten __syncthreads are gone.
// AR decode issues this once per ternary layer per token, where the barriers were most of it.
constexpr int kWarpSpan = 1024;
template <bool sign_first>
__global__ void hadamard_span1024_warp_kernel(const __nv_bfloat16* __restrict__ x,
                                              __nv_bfloat16* __restrict__ y,
                                              const signed char* __restrict__ sign,
                                              int width, long spans, float norm) {
    const long span = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (span >= spans) return;
    const int lane = threadIdx.x & 31;
    const long base = span * (long)kWarpSpan + lane * 32;
    const int sign_off = (int)(base % (long)width);
    float v[32];
    {
        const uint4* src = reinterpret_cast<const uint4*>(x + base);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const uint4 u = src[q];
            const unsigned int w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                v[q * 8 + h * 2] = __uint_as_float(w4[h] << 16);
                v[q * 8 + h * 2 + 1] = __uint_as_float(w4[h] & 0xFFFF0000u);
            }
        }
    }
    if (sign_first) {
#pragma unroll
        for (int i = 0; i < 32; ++i) v[i] *= (float)sign[sign_off + i];
    }
#pragma unroll
    for (int len = 1; len < 32; len <<= 1) {
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (i & len) continue;
            const float a = v[i], b = v[i + len];
            v[i] = a + b;
            v[i + len] = a - b;
        }
    }
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool hi = (lane & m) != 0;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const float o = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = hi ? o - v[i] : v[i] + o;   // lo: a + b; hi: a - b with a = partner's
        }
    }
    unsigned int out[16];
#pragma unroll
    for (int i = 0; i < 32; i += 2) {
        float a = v[i] * norm, b = v[i + 1] * norm;
        if (!sign_first) { a *= (float)sign[sign_off + i]; b *= (float)sign[sign_off + i + 1]; }
        const __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
        out[i / 2] = *reinterpret_cast<const unsigned int*>(&p);
    }
    uint4* dst = reinterpret_cast<uint4*>(y + base);
#pragma unroll
    for (int q = 0; q < 4; ++q) dst[q] = make_uint4(out[q * 4], out[q * 4 + 1], out[q * 4 + 2], out[q * 4 + 3]);
}

void launch(const void* x, void* y, const signed char* sign, long n_values, int width, int block,
            bool sign_first, cudaStream_t stream) {
    if (n_values <= 0 || block <= 0 || width <= 0) return;
    if (width % block != 0 || n_values % (long)width != 0) return;   // caller's contract
    const long spans = n_values / block;
    const float norm = rsqrtf((float)block);
    const bool aligned = (reinterpret_cast<uintptr_t>(x) & 15) == 0 &&
                         (reinterpret_cast<uintptr_t>(y) & 15) == 0;
    if (block == kWarpSpan && aligned) {
        constexpr int kWarps = 4;
        const unsigned grid = (unsigned)((spans + kWarps - 1) / kWarps);
        if (sign_first)
            hadamard_span1024_warp_kernel<true><<<grid, kWarps * 32, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(y),
                sign, width, spans, norm);
        else
            hadamard_span1024_warp_kernel<false><<<grid, kWarps * 32, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(y),
                sign, width, spans, norm);
        return;
    }
    const size_t shmem = (size_t)block * sizeof(float);
    if (sign_first)
        hadamard_span_kernel<true><<<(unsigned)spans, kThreads, shmem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(y),
            sign, width, block, norm);
    else
        hadamard_span_kernel<false><<<(unsigned)spans, kThreads, shmem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(y),
            sign, width, block, norm);
}

}  // namespace

void launch_hadamard_rotate_bf16(const void* x_bf16, void* y_bf16, const signed char* sign,
                                 long n_values, int width, int block, cudaStream_t stream) {
    launch(x_bf16, y_bf16, sign, n_values, width, block, /*sign_first=*/true, stream);
}

void launch_hadamard_unrotate_bf16(const void* x_bf16, void* y_bf16, const signed char* sign,
                                   long n_values, int width, int block, cudaStream_t stream) {
    launch(x_bf16, y_bf16, sign, n_values, width, block, /*sign_first=*/false, stream);
}

}}  // namespace sparkinfer::kernels
