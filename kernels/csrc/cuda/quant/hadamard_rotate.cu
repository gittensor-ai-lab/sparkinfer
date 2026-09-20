// Blockwise normalized Walsh-Hadamard rotation of an activation, with a per-index sign.
// See sparkinfer/kernels/hadamard.h for why Ternary-Bonsai-2 needs it.
//
// One CUDA block per block-sized span, the span held in shared memory for the butterfly. At
// block = 1024 that is 4 KB of shared memory and ten butterfly stages, against the 1024 * width
// bytes of weight the matmul it feeds will read -- so it is not on the critical path, and is
// written for clarity rather than for the last few microseconds.
#include "sparkinfer/kernels/hadamard.h"

#include <cuda_bf16.h>

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

void launch(const void* x, void* y, const signed char* sign, long n_values, int width, int block,
            bool sign_first, cudaStream_t stream) {
    if (n_values <= 0 || block <= 0 || width <= 0) return;
    if (width % block != 0 || n_values % (long)width != 0) return;   // caller's contract
    const long spans = n_values / block;
    const float norm = rsqrtf((float)block);
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
