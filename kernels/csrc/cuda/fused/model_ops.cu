// Model-level ops: token embedding gather and greedy argmax sampling.
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

// out[t, :] = table[ids[t], :]   grid = n_tokens, threads over hidden.
__global__ void embedding_kernel(const int* __restrict__ ids,
                                 const __nv_bfloat16* __restrict__ table,
                                 __nv_bfloat16* __restrict__ out,
                                 int hidden) {
    const int t  = blockIdx.x;
    const int id = ids[t];
    for (int h = threadIdx.x; h < hidden; h += blockDim.x)
        out[(size_t)t * hidden + h] = table[(size_t)id * hidden + h];
}

// out_id[r] = argmax_v logits[r, v]   (greedy).  One block per row.
// Decode runs a single row over a ~152k vocab, so size the block wide (1024 threads):
// at bs=1 there is only one row, so more warps per block — not more blocks — is what
// hides the global-load latency of the scan. ~165us -> tens of us.
__global__ void argmax_kernel(const float* __restrict__ logits, int* __restrict__ out_id,
                              int vocab) {
    const int row = blockIdx.x;
    const float* L = logits + (size_t)row * vocab;
    __shared__ float s_val[1024];
    __shared__ int   s_idx[1024];

    float best = -1e30f; int bi = 0;
    for (int v = threadIdx.x; v < vocab; v += blockDim.x)
        if (L[v] > best) { best = L[v]; bi = v; }
    s_val[threadIdx.x] = best; s_idx[threadIdx.x] = bi;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_val[threadIdx.x + stride] > s_val[threadIdx.x] ||
                (s_val[threadIdx.x + stride] == s_val[threadIdx.x] && s_idx[threadIdx.x + stride] < s_idx[threadIdx.x])) {
                s_val[threadIdx.x] = s_val[threadIdx.x + stride];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out_id[row] = s_idx[0];
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/fused.h"

void launch_embedding(const int* ids, const void* table, void* out,
                      int n_tokens, int hidden, cudaStream_t stream) {
    embedding_kernel<<<n_tokens, 256, 0, stream>>>(
        ids, reinterpret_cast<const __nv_bfloat16*>(table),
        reinterpret_cast<__nv_bfloat16*>(out), hidden);
}

void launch_argmax(const float* logits, int* out_id, int n_rows, int vocab, cudaStream_t stream) {
    argmax_kernel<<<n_rows, 1024, 0, stream>>>(logits, out_id, vocab);
}
#endif

} // namespace kernels
} // namespace sparkinfer
