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

// argmax tie-break (matches the reference): keep the smaller index on equal values.
__device__ __forceinline__ void argmax_merge(float& bv, int& bi, float ov, int oi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// out_id[r] = argmax_v logits[r, v]   (greedy).  One block per row.
// Fallback path for n_rows > 1 (prefill scoring); decode (n_rows == 1) uses the
// two-pass kernels below which scale across all SMs.
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
        if (threadIdx.x < stride)
            argmax_merge(s_val[threadIdx.x], s_idx[threadIdx.x],
                         s_val[threadIdx.x + stride], s_idx[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out_id[row] = s_idx[0];
}

// ---- two-pass single-row argmax (decode) --------------------------------------
// The bs=1 decode argmax scans one ~152k–262k row. The single-block kernel above
// pins that scan to ONE SM (1 of 170 on a 5090) — the rest idle. This splits the
// scan across ARGMAX_BLOCKS blocks (pass 1), each emitting one (val,idx) partial to
// a static device buffer, then a single block reduces the partials (pass 2). Result
// and tie-break are identical to argmax_kernel; only the scan is parallelized.
// Static scratch => no allocation, CUDA-graph safe (argmax is captured in the graph).
static constexpr int ARGMAX_BLOCKS = 512;
__device__ float g_argmax_pv[ARGMAX_BLOCKS];
__device__ int   g_argmax_pi[ARGMAX_BLOCKS];

__global__ void argmax_p1_kernel(const float* __restrict__ logits, int vocab) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    __shared__ float s_val[256];
    __shared__ int   s_idx[256];
    float best = -1e30f; int bi = 0;
    for (int v = tid; v < vocab; v += stride) if (logits[v] > best) { best = logits[v]; bi = v; }
    s_val[threadIdx.x] = best; s_idx[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            argmax_merge(s_val[threadIdx.x], s_idx[threadIdx.x], s_val[threadIdx.x + s], s_idx[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) { g_argmax_pv[blockIdx.x] = s_val[0]; g_argmax_pi[blockIdx.x] = s_idx[0]; }
}

__global__ void argmax_p2_kernel(int* __restrict__ out_id) {
    __shared__ float s_val[ARGMAX_BLOCKS];
    __shared__ int   s_idx[ARGMAX_BLOCKS];
    const int t = threadIdx.x;
    s_val[t] = g_argmax_pv[t]; s_idx[t] = g_argmax_pi[t];
    __syncthreads();
    for (int s = ARGMAX_BLOCKS / 2; s > 0; s >>= 1) {
        if (t < s) argmax_merge(s_val[t], s_idx[t], s_val[t + s], s_idx[t + s]);
        __syncthreads();
    }
    if (t == 0) out_id[0] = s_idx[0];
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
    if (n_rows == 1) {   // decode: parallelize the single-row scan across all SMs
        argmax_p1_kernel<<<ARGMAX_BLOCKS, 256, 0, stream>>>(logits, vocab);
        argmax_p2_kernel<<<1, ARGMAX_BLOCKS, 0, stream>>>(out_id);
    } else {
        argmax_kernel<<<n_rows, 1024, 0, stream>>>(logits, out_id, vocab);
    }
}
#endif

} // namespace kernels
} // namespace sparkinfer
