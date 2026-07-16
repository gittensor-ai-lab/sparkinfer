// Flash decode for 8:1 GQA — Qwen3.5-35B-A3B (16 Q-heads, 2 KV-heads, head_dim=128).
//
// With only 2 KV heads, 8 query heads share each KV head. We assign 8 warps to
// one CTA (one warp per shared Q-head) and load each KV block into shared memory
// ONCE, so the 8 warps read it from smem instead of issuing 8x the global loads.
// That is the whole point of the specialization: KV global traffic drops 8x
// versus running the generic one-warp-per-head kernel.
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

__device__ __forceinline__ float g8_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float g8_warp_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// grid = (num_seqs, num_kv_heads); block = NWARPS*32 threads.
// warp w handles q_head = kv_head*GQA + w  (GQA == NWARPS).
template <int HEAD_DIM, int NWARPS, int BLOCK_SIZE>
__global__ void flash_decode_gqa8_kernel(
    const __nv_bfloat16* __restrict__ q,        // [num_seqs, num_q_heads, HEAD_DIM]
    const __nv_bfloat16* __restrict__ k_pool,   // [num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM]
    const __nv_bfloat16* __restrict__ v_pool,
    const int* __restrict__ block_table,        // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,
    __nv_bfloat16* __restrict__ out,            // [num_seqs, num_q_heads, HEAD_DIM]
    const float scale,
    const int num_q_heads,
    const int num_kv_heads,
    const int max_blocks_per_seq
) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const int seq     = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int warp    = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int q_head  = kv_head * NWARPS + warp;

    extern __shared__ float smem[];
    float* s_k = smem;                            // [BLOCK_SIZE, HEAD_DIM]
    float* s_v = s_k + BLOCK_SIZE * HEAD_DIM;     // [BLOCK_SIZE, HEAD_DIM]

    float q_reg[ELEMS];
    const __nv_bfloat16* qp = q + (size_t)(seq * num_q_heads + q_head) * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) q_reg[e] = g8_to_f(qp[lane + e * 32]);

    float m = -1e30f, l = 0.f, acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    const int seq_len   = seq_lens[seq];
    const int n_blocks  = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int blk = 0; blk < n_blocks; blk++) {
        const int phys  = block_table[seq * max_blocks_per_seq + blk];
        const int valid = min(BLOCK_SIZE, seq_len - blk * BLOCK_SIZE);

        // Cooperative load of this KV block into shared memory (all NWARPS warps).
        for (int i = threadIdx.x; i < valid * HEAD_DIM; i += blockDim.x) {
            const int within = i / HEAD_DIM;
            const int d      = i % HEAD_DIM;
            const size_t base = ((size_t)(phys * BLOCK_SIZE + within) * num_kv_heads + kv_head) * HEAD_DIM + d;
            s_k[i] = g8_to_f(k_pool[base]);
            s_v[i] = g8_to_f(v_pool[base]);
        }
        __syncthreads();

        for (int t = 0; t < valid; t++) {
            float partial = 0.f;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++)
                partial += q_reg[e] * s_k[t * HEAD_DIM + lane + e * 32];
            const float score = g8_warp_sum(partial) * scale;

            const float m_new = fmaxf(m, score);
            const float corr  = __expf(m - m_new);
            const float p     = __expf(score - m_new);
            l = l * corr + p;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++)
                acc[e] = acc[e] * corr + p * s_v[t * HEAD_DIM + lane + e * 32];
            m = m_new;
        }
        __syncthreads();   // done reading smem before next block overwrites it
    }

    const float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
    __nv_bfloat16* op = out + (size_t)(seq * num_q_heads + q_head) * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) op[lane + e * 32] = __float2bfloat16(acc[e] * inv_l);
}

#ifndef _MSC_VER
template __global__ void flash_decode_gqa8_kernel<128, 8, 16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const int*, const int*, __nv_bfloat16*, float, int, int, int);
#endif
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
void launch_flash_decode_gqa8(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens, void* out,
    int num_seqs, int num_kv_heads,
    int head_dim, int block_size, int max_blocks_per_seq,
    float scale, cudaStream_t stream
) {
    constexpr int NWARPS = 8, BLOCK_SIZE = 16, HEAD_DIM = 128;
    const int num_q_heads = num_kv_heads * NWARPS;
    dim3 grid(num_seqs, num_kv_heads);
    size_t smem = 2 * BLOCK_SIZE * HEAD_DIM * sizeof(float);
    flash_decode_gqa8_kernel<HEAD_DIM, NWARPS, BLOCK_SIZE><<<grid, NWARPS * 32, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(k_pool),
        reinterpret_cast<const __nv_bfloat16*>(v_pool),
        block_table, seq_lens, reinterpret_cast<__nv_bfloat16*>(out),
        scale, num_q_heads, num_kv_heads, max_blocks_per_seq);
    (void)head_dim; (void)block_size;
}
#endif

} // namespace kernels
} // namespace sparkinfer
