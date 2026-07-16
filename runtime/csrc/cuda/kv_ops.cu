// KV-cache append and residual add — small device ops the runtime uses to wire
// attention + MoE into a decode step.
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {

// grid = (num_seqs, num_kv_heads); blockDim = head_dim (<=1024).
__global__ void kv_append_kernel(
    __nv_bfloat16* __restrict__ k_pool, __nv_bfloat16* __restrict__ v_pool,
    const __nv_bfloat16* __restrict__ k_new, const __nv_bfloat16* __restrict__ v_new,
    const int* __restrict__ block_table, const int* __restrict__ write_pos,
    int num_kv_heads, int head_dim, int block_size, int max_blocks_per_seq
) {
    const int seq = blockIdx.x;
    const int h   = blockIdx.y;
    const int d   = threadIdx.x;
    if (d >= head_dim) return;

    const int pos    = write_pos[seq];
    const int blk    = pos / block_size;
    const int within = pos % block_size;
    const int phys   = block_table[seq * max_blocks_per_seq + blk];
    const size_t dst = ((size_t)(phys * block_size + within) * num_kv_heads + h) * head_dim + d;
    const size_t src = ((size_t)seq * num_kv_heads + h) * head_dim + d;
    k_pool[dst] = k_new[src];
    v_pool[dst] = v_new[src];
}

__global__ void residual_add_kernel(const __nv_bfloat16* __restrict__ a,
                                    const __nv_bfloat16* __restrict__ b,
                                    __nv_bfloat16* __restrict__ out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kv_ops.h"
#include <cstdio>

void launch_kv_append(void* k_pool, void* v_pool, const void* k_new, const void* v_new,
                      const int* block_table, const int* write_pos,
                      int num_seqs, int num_kv_heads, int head_dim,
                      int block_size, int max_blocks_per_seq, cudaStream_t stream) {
    // CUDA limits blockDim.x to 1024; launching with head_dim > 1024 is invalid.
    if (head_dim <= 0 || head_dim > 1024 || num_seqs <= 0 || num_kv_heads <= 0) {
        fprintf(stderr, "[kv] launch_kv_append: invalid dims (num_seqs=%d num_kv_heads=%d head_dim=%d)
",
                num_seqs, num_kv_heads, head_dim);
        return;
    }
    dim3 grid(num_seqs, num_kv_heads);
    kv_append_kernel<<<grid, head_dim, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(k_pool), reinterpret_cast<__nv_bfloat16*>(v_pool),
        reinterpret_cast<const __nv_bfloat16*>(k_new), reinterpret_cast<const __nv_bfloat16*>(v_new),
        block_table, write_pos, num_kv_heads, head_dim, block_size, max_blocks_per_seq);
}

void launch_residual_add(const void* a, const void* b, void* out, int n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    residual_add_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a), reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<__nv_bfloat16*>(out), n);
}
#endif

} // namespace sparkinfer
