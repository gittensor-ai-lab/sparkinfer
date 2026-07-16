// Flash decode for Gemma 4 GLOBAL layers (5 of 30 layers).
//   16 Q-heads, 2 KV-heads (GQA 8:1), head_dim=512, full context.
//
// ECOSYSTEM GAP: as of 2026-06 no public kernel (FlashInfer, vLLM,
// FlashAttention, llama.cpp) handles head_dim=512 efficiently. They template
// HEAD_DIM in {64,128,256} and would pad or run two passes.
//
// We handle 512 directly: with the coalesced (lane + e*32) register layout each
// lane simply owns ELEMS = 512/32 = 16 head elements. No "two-phase" split is
// needed — the online-softmax math is identical to every other head_dim, only
// the register footprint grows. 8 warps share each KV block in 64 KB of smem
// (opt-in via cudaFuncAttributeMaxDynamicSharedMemorySize on the host side).
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#include <cstdio>
#endif

namespace sparkinfer {
namespace kernels {

__device__ __forceinline__ float gl_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float gl_warp_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

template <int HEAD_DIM, int NWARPS, int BLOCK_SIZE>
__global__ void flash_decode_global_kernel(
    const __nv_bfloat16* __restrict__ q,        // [num_seqs, num_q_heads, HEAD_DIM]
    const __nv_bfloat16* __restrict__ k_pool,   // [num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM]
    const __nv_bfloat16* __restrict__ v_pool,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    __nv_bfloat16* __restrict__ out,
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
    float* s_k = smem;
    float* s_v = s_k + BLOCK_SIZE * HEAD_DIM;

    float q_reg[ELEMS];
    const __nv_bfloat16* qp = q + (size_t)(seq * num_q_heads + q_head) * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) q_reg[e] = gl_to_f(qp[lane + e * 32]);

    float m = -1e30f, l = 0.f, acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    const int seq_len  = seq_lens[seq];
    const int n_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int blk = 0; blk < n_blocks; blk++) {
        const int phys  = block_table[seq * max_blocks_per_seq + blk];
        const int valid = min(BLOCK_SIZE, seq_len - blk * BLOCK_SIZE);

        for (int i = threadIdx.x; i < valid * HEAD_DIM; i += blockDim.x) {
            const int within = i / HEAD_DIM;
            const int d      = i % HEAD_DIM;
            const size_t base = ((size_t)(phys * BLOCK_SIZE + within) * num_kv_heads + kv_head) * HEAD_DIM + d;
            s_k[i] = gl_to_f(k_pool[base]);
            s_v[i] = gl_to_f(v_pool[base]);
        }
        __syncthreads();

        for (int t = 0; t < valid; t++) {
            float partial = 0.f;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++)
                partial += q_reg[e] * s_k[t * HEAD_DIM + lane + e * 32];
            const float score = gl_warp_sum(partial) * scale;

            const float m_new = fmaxf(m, score);
            const float corr  = __expf(m - m_new);
            const float p     = __expf(score - m_new);
            l = l * corr + p;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++)
                acc[e] = acc[e] * corr + p * s_v[t * HEAD_DIM + lane + e * 32];
            m = m_new;
        }
        __syncthreads();
    }

    const float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
    __nv_bfloat16* op = out + (size_t)(seq * num_q_heads + q_head) * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) op[lane + e * 32] = __float2bfloat16(acc[e] * inv_l);
}

#ifndef _MSC_VER
template __global__ void flash_decode_global_kernel<512, 8, 16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const int*, const int*, __nv_bfloat16*, float, int, int, int);
#endif
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
void launch_flash_decode_global_hd512(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens, void* out,
    int num_seqs, int num_kv_heads,
    int block_size, int max_blocks_per_seq,
    float scale, cudaStream_t stream
) {
    constexpr int HEAD_DIM = 512, NWARPS = 8, BLOCK_SIZE = 16;
    const int num_q_heads = num_kv_heads * NWARPS;
    dim3 grid(num_seqs, num_kv_heads);
    size_t smem = 2 * BLOCK_SIZE * HEAD_DIM * sizeof(float);   // 64 KB > 48 KB default

    static bool optin = false;
    if (!optin) {
        cudaError_t e = cudaFuncSetAttribute(
            flash_decode_global_kernel<HEAD_DIM, NWARPS, BLOCK_SIZE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (e != cudaSuccess) fprintf(stderr, "[hd512] smem opt-in failed: %s\n", cudaGetErrorString(e));
        optin = true;
    }
    flash_decode_global_kernel<HEAD_DIM, NWARPS, BLOCK_SIZE><<<grid, NWARPS * 32, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(k_pool),
        reinterpret_cast<const __nv_bfloat16*>(v_pool),
        block_table, seq_lens, reinterpret_cast<__nv_bfloat16*>(out),
        scale, num_q_heads, num_kv_heads, max_blocks_per_seq);
    (void)block_size;
}
#endif

} // namespace kernels
} // namespace sparkinfer
