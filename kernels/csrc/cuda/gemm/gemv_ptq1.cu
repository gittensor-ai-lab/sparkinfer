// GEMV against PTQ1_0 ternary weights, read in their stored 28-byte blocks.
// See sparkinfer/kernels/ternary.h for the format and for the basis the activation must be in.
#include "sparkinfer/kernels/ternary.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace sparkinfer { namespace kernels {

namespace {

constexpr int kBlockElems = 128;
constexpr int kBlockBytes = 28;   // 24 five-trit carriers + 2 four-trit carriers + fp16 scale
constexpr int kWarpsPerCta = 4;

// Which carrier byte and trit position a weight sits in. The 24 five-trit bytes are walked in two
// runs -- 16 then 8 -- each emitting its trits position-major, then the two four-trit bytes. This
// is ggml's TQ1_0 walk; reading it carrier-major instead gets every value right and every one in
// the wrong place, which is invisible until you compare against the unquantized checkpoint.
__device__ __forceinline__ int ptq1_trit(const unsigned char* __restrict__ qs, int idx) {
    int byte, m;
    if (idx < 80) {
        m = idx >> 4;
        byte = idx & 15;
    } else if (idx < 120) {
        const int r = idx - 80;
        m = r >> 3;
        byte = 16 + (r & 7);
    } else {
        const int r = idx - 120;
        m = r >> 1;
        byte = 24 + (r & 1);
    }
    // Carriers are scaled into the whole byte rather than packed as plain base 3, so the digit
    // comes back out by multiplying up and taking the high bits -- ggml's own extraction.
    const unsigned int pow3[5] = {1u, 3u, 9u, 27u, 81u};
    const unsigned int q = (unsigned char)(qs[byte] * pow3[m]);
    return (int)((q * 3u) >> 8) - 1;
}

template <typename OutT>
__device__ __forceinline__ void store_out(OutT* y, int row, float v);
template <>
__device__ __forceinline__ void store_out<__nv_bfloat16>(__nv_bfloat16* y, int row, float v) {
    y[row] = __float2bfloat16(v);
}
template <>
__device__ __forceinline__ void store_out<float>(float* y, int row, float v) {
    y[row] = v;
}

template <typename OutT>
__global__ void gemv_ptq1_kernel(const __nv_bfloat16* __restrict__ x,
                                 const unsigned char* __restrict__ w,
                                 OutT* __restrict__ y, int n_rows, int k) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarpsPerCta + warp;
    if (row >= n_rows) return;

    const int n_blocks = k / kBlockElems;
    const unsigned char* wrow = w + (size_t)row * n_blocks * kBlockBytes;

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* qs = wrow + (size_t)b * kBlockBytes;
        // The scale sits in the last two bytes. Blocks are 28 bytes and rows start block-aligned,
        // so this is 2-byte aligned.
        const __half scale_h = *reinterpret_cast<const __half*>(qs + kBlockBytes - 2);
        const float scale = __half2float(scale_h);

        const __nv_bfloat16* xb = x + (size_t)b * kBlockElems;
        float part = 0.0f;
#pragma unroll
        for (int t = 0; t < kBlockElems / 32; ++t) {
            const int idx = lane + t * 32;
            part += (float)ptq1_trit(qs, idx) * __bfloat162float(xb[idx]);
        }
        acc += scale * part;
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) store_out<OutT>(y, row, acc);
}

// Embedding lookup straight out of the ternary table: one row decoded per token. The row comes
// out in the basis it was stored in, so the caller un-rotates it before it becomes the residual.
__global__ void embedding_ptq1_kernel(const int* __restrict__ tok,
                                      const unsigned char* __restrict__ table,
                                      __nv_bfloat16* __restrict__ out, int k) {
    const int r = blockIdx.y;
    const int row = tok[r];
    const int n_blocks = k / kBlockElems;
    const unsigned char* wrow = table + (size_t)row * n_blocks * kBlockBytes;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
        const unsigned char* qs = wrow + (size_t)(i / kBlockElems) * kBlockBytes;
        const __half scale_h = *reinterpret_cast<const __half*>(qs + kBlockBytes - 2);
        out[(size_t)r * k + i] =
            __float2bfloat16((float)ptq1_trit(qs, i % kBlockElems) * __half2float(scale_h));
    }
}

template <typename OutT>
void launch_typed(const void* x, const void* w, OutT* y, int n_rows, int k, cudaStream_t stream) {
    if (n_rows <= 0 || k <= 0 || k % kBlockElems != 0) return;
    const int ctas = (n_rows + kWarpsPerCta - 1) / kWarpsPerCta;
    gemv_ptq1_kernel<OutT><<<ctas, kWarpsPerCta * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const unsigned char*>(w),
        y, n_rows, k);
}

}  // namespace

void launch_gemv_ptq1(const void* x_bf16, const void* w_ptq1, void* y_bf16,
                      int n_rows, int k, cudaStream_t stream) {
    launch_typed<__nv_bfloat16>(x_bf16, w_ptq1, reinterpret_cast<__nv_bfloat16*>(y_bf16),
                                n_rows, k, stream);
}

void launch_gemv_ptq1_f32(const void* x_bf16, const void* w_ptq1, float* y_f32,
                          int n_rows, int k, cudaStream_t stream) {
    launch_typed<float>(x_bf16, w_ptq1, y_f32, n_rows, k, stream);
}

void launch_embedding_ptq1(const int* tokens, const void* table_ptq1, void* out_bf16,
                           int n_tokens, int k, cudaStream_t stream) {
    if (n_tokens <= 0 || k <= 0 || k % kBlockElems != 0) return;
    const int threads = 256;
    const dim3 grid((unsigned)((k + threads - 1) / threads), (unsigned)n_tokens);
    embedding_ptq1_kernel<<<grid, threads, 0, stream>>>(
        tokens, reinterpret_cast<const unsigned char*>(table_ptq1),
        reinterpret_cast<__nv_bfloat16*>(out_bf16), k);
}

}}  // namespace sparkinfer::kernels
