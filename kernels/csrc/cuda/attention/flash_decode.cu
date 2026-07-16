// Flash decode attention — single-token decode over a paged KV cache.
//
// One warp computes one (sequence, query-head). Each lane owns HEAD_DIM/32
// elements of the head vector in a coalesced (lane + e*32) layout. KV positions
// are streamed once each with an online-softmax accumulation, so total memory
// traffic is exactly the KV cache size — optimal for batch-size-1 decode.
//
// Portable CUDA: runs on sm_89 (Ada) through sm_120 (RTX 5090). No tensor-core
// or arch-specific intrinsics, so correctness is identical across targets.
//
// References: FlashAttention-2 (Dao 2023), FlashDecoding, PagedAttention (Kwon 2023).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

// ---- scalar <-> float helpers (overloaded so kernels stay dtype-generic) ----
__device__ __forceinline__ float to_f(__half x)        { return __half2float(x); }
__device__ __forceinline__ float to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ void  from_f(float v, __half* o)        { *o = __float2half(v); }
__device__ __forceinline__ void  from_f(float v, __nv_bfloat16* o) { *o = __float2bfloat16(v); }

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// One warp per (seq, head). HEAD_DIM must be a multiple of 32.
template <typename scalar_t, int HEAD_DIM>
__global__ void flash_decode_kernel(
    const scalar_t* __restrict__ q,        // [num_seqs, num_heads, HEAD_DIM]
    const scalar_t* __restrict__ k_pool,   // [num_blocks, block_size, num_kv_heads, HEAD_DIM]
    const scalar_t* __restrict__ v_pool,   // same layout as k_pool
    const int*      __restrict__ block_table, // [num_seqs, max_blocks_per_seq]
    const int*      __restrict__ seq_lens,    // [num_seqs]
    scalar_t*       __restrict__ out,      // [num_seqs, num_heads, HEAD_DIM]
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int block_size,
    const int max_blocks_per_seq
) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const int seq  = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;                       // 0..31
    const int kv_head = head / (num_heads / num_kv_heads);

    // Load query fragment for this lane.
    float q_reg[ELEMS];
    const scalar_t* qp = q + (size_t)(seq * num_heads + head) * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) q_reg[e] = to_f(qp[lane + e * 32]);

    float m = -1e30f, l = 0.f;
    float acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    const int seq_len = seq_lens[seq];
    for (int t = 0; t < seq_len; t++) {
        const int blk    = t / block_size;
        const int within = t % block_size;
        const int phys   = block_table[seq * max_blocks_per_seq + blk];
        const size_t base = ((size_t)(phys * block_size + within) * num_kv_heads + kv_head) * HEAD_DIM;

        // score = scale * <q, k_t>
        float partial = 0.f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) partial += q_reg[e] * to_f(k_pool[base + lane + e * 32]);
        const float score = warp_reduce_sum(partial) * scale;

        // online softmax update
        const float m_new = fmaxf(m, score);
        const float corr  = __expf(m - m_new);
        const float p     = __expf(score - m_new);
        l = l * corr + p;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++)
            acc[e] = acc[e] * corr + p * to_f(v_pool[base + lane + e * 32]);
        m = m_new;
    }

    const float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
    scalar_t* op = out + (size_t)(seq * num_heads + head) * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) from_f(acc[e] * inv_l, &op[lane + e * 32]);
}

// Explicit instantiations — force kernel emission (into the static lib for the
// real build, and into PTX so NVRTC verification actually checks the body).
#ifndef _MSC_VER
template __global__ void flash_decode_kernel<__nv_bfloat16, 64>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const int*, const int*, __nv_bfloat16*, float, int, int, int, int);
#endif
#ifndef _MSC_VER
template __global__ void flash_decode_kernel<__nv_bfloat16, 128>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const int*, const int*, __nv_bfloat16*, float, int, int, int, int);
#endif
#ifndef _MSC_VER
template __global__ void flash_decode_kernel<__nv_bfloat16, 256>(
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const int*, const int*, __nv_bfloat16*, float, int, int, int, int);
#endif
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
template <int HEAD_DIM>
static void dispatch(const void* q, const void* k_pool, const void* v_pool,
                     const int* block_table, const int* seq_lens, void* out,
                     int num_seqs, int num_heads, int num_kv_heads,
                     int block_size, int max_blocks_per_seq,
                     float scale, cudaStream_t stream) {
    dim3 grid(num_seqs, num_heads);
    flash_decode_kernel<__nv_bfloat16, HEAD_DIM><<<grid, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(k_pool),
        reinterpret_cast<const __nv_bfloat16*>(v_pool),
        block_table, seq_lens, reinterpret_cast<__nv_bfloat16*>(out),
        scale, num_heads, num_kv_heads, block_size, max_blocks_per_seq);
}

void launch_flash_decode(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens, void* out,
    int num_seqs, int num_heads, int num_kv_heads,
    int head_dim, int block_size, int max_blocks_per_seq,
    float scale, cudaStream_t stream
) {
    switch (head_dim) {
        case 64:  dispatch<64> (q,k_pool,v_pool,block_table,seq_lens,out,num_seqs,num_heads,num_kv_heads,block_size,max_blocks_per_seq,scale,stream); break;
        case 128: dispatch<128>(q,k_pool,v_pool,block_table,seq_lens,out,num_seqs,num_heads,num_kv_heads,block_size,max_blocks_per_seq,scale,stream); break;
        case 256: dispatch<256>(q,k_pool,v_pool,block_table,seq_lens,out,num_seqs,num_heads,num_kv_heads,block_size,max_blocks_per_seq,scale,stream); break;
        default:  /* unsupported head_dim */ break;
    }
}
#endif

} // namespace kernels
} // namespace sparkinfer
