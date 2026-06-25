// Flash-decoding (KV-split) attention for decode.
//
// The plain decode kernel parallelizes only over (seq, kv_head) — e.g. 4 blocks
// for Qwen3-30B-A3B, leaving ~184 of 188 SMs idle. Flash-decoding instead splits
// the KV sequence into n_splits chunks and runs one block per (seq, q_head,
// split): each computes a partial online-softmax (m, l, acc) over its chunk, then
// a combine pass merges the partials with the standard log-sum-exp rescale. This
// fills the GPU at decode AND scales to long context (work grows with KV length,
// spread across many blocks). Grid is fixed (independent of seq_len, read in
// kernel), so it stays CUDA-graph capturable.
//
// One warp per block; head_dim=128 (Qwen3). Portable CUDA — sm_89 .. sm_120/121.
// K/V heads are loaded with 128-bit uint4 transactions (16 lanes x 8 bf16) and
// warp-shuffle broadcast — same online-softmax math, fewer global loads per token.

#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

__device__ __forceinline__ float fa_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float fa_wsum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// head_dim=128: lanes 0..15 each ld.global.v4 8 bf16; all lanes extract their
// lane+e*32 element via shfl from the owning lane. Math-identical to scalar loads.
__device__ __forceinline__ float fa_head128_elem(
    const uint4& pack, int owner, int sub
) {
    const unsigned int w0 = __shfl_sync(0xffffffffu, pack.x, owner);
    const unsigned int w1 = __shfl_sync(0xffffffffu, pack.y, owner);
    const unsigned int w2 = __shfl_sync(0xffffffffu, pack.z, owner);
    const unsigned int w3 = __shfl_sync(0xffffffffu, pack.w, owner);
    const uint4 p4 = {w0, w1, w2, w3};
    const __nv_bfloat16* hb = reinterpret_cast<const __nv_bfloat16*>(&p4);
    return fa_to_f(hb[sub]);
}

__device__ __forceinline__ float fa_dot_k128(
    const float qr[4], const __nv_bfloat16* __restrict__ kb, int lane
) {
    uint4 kpack{0, 0, 0, 0};
    if (lane < 16) kpack = __ldg(reinterpret_cast<const uint4*>(kb) + lane);
    float p = 0.f;
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        const int idx = lane + e * 32;
        p = __fmaf_rn(qr[e], fa_head128_elem(kpack, idx >> 3, idx & 7), p);
    }
    return p;
}

__device__ __forceinline__ void fa_acc_v128(
    float acc[4], float pe, float corr,
    const __nv_bfloat16* __restrict__ vb, int lane
) {
    uint4 vpack{0, 0, 0, 0};
    if (lane < 16) vpack = __ldg(reinterpret_cast<const uint4*>(vb) + lane);
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        const int idx = lane + e * 32;
        const float ve = fa_head128_elem(vpack, idx >> 3, idx & 7);
        acc[e] = __fmaf_rn(pe, ve, acc[e] * corr);
    }
}

template <int HEAD_DIM>
__global__ void fa_split_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    float scale, int num_q_heads, int num_kv_heads, int block_size, int max_blocks, int n_splits
) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const int seq   = blockIdx.y;
    const int split = blockIdx.x % n_splits;
    const int qh    = blockIdx.x / n_splits;
    const int lane  = threadIdx.x;
    const int kvh   = qh / (num_q_heads / num_kv_heads);

    float qr[ELEMS];
    const __nv_bfloat16* qp = q + (size_t)(seq * num_q_heads + qh) * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) qr[e] = fa_to_f(__ldg(qp + lane + e * 32));

    const int sl    = seq_lens[seq];
    const int chunk = (sl + n_splits - 1) / n_splits;
    const int start = split * chunk;
    const int end   = min(sl, start + chunk);

    float m = -1e30f, l = 0.f, acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    for (int t = start; t < end; t++) {
        const int blk = t / block_size, within = t % block_size;
        const int phys = __ldg(&block_table[seq * max_blocks + blk]);
        const size_t base = ((size_t)(phys * block_size + within) * num_kv_heads + kvh) * HEAD_DIM;
        const __nv_bfloat16* kb = k_pool + base;
        const __nv_bfloat16* vb = v_pool + base;

        const float score = fa_wsum(fa_dot_k128(qr, kb, lane)) * scale;
        const float mn = fmaxf(m, score), corr = __expf(m - mn), pe = __expf(score - mn);
        l = __fmaf_rn(l, corr, pe);
        fa_acc_v128(acc, pe, corr, vb, lane);
        m = mn;
    }

    const int idx = (seq * num_q_heads + qh) * n_splits + split;
    if (lane == 0) { part_m[idx] = m; part_l[idx] = l; }
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) part_acc[(size_t)idx * HEAD_DIM + lane + e * 32] = acc[e];
}

template <int HEAD_DIM>
__global__ void fa_combine_kernel(
    const float* __restrict__ part_m, const float* __restrict__ part_l,
    const float* __restrict__ part_acc, __nv_bfloat16* __restrict__ out,
    int num_q_heads, int n_splits
) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const int seq = blockIdx.y, qh = blockIdx.x, lane = threadIdx.x;
    const int idxbase = (seq * num_q_heads + qh) * n_splits;

    float gm = -1e30f;
    for (int s = 0; s < n_splits; s++) gm = fmaxf(gm, __ldg(&part_m[idxbase + s]));
    float gl = 0.f, acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;
    for (int s = 0; s < n_splits; s++) {
        const float ms = __ldg(&part_m[idxbase + s]), ls = __ldg(&part_l[idxbase + s]);
        const float sc = __expf(ms - gm);
        gl = __fmaf_rn(ls, sc, gl);   // gl += ls * sc
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) {
            const float pv = __ldg(&part_acc[(size_t)(idxbase + s) * HEAD_DIM + lane + e * 32]);
            acc[e] = __fmaf_rn(sc, pv, acc[e]);   // acc[e] += sc * pv
        }
    }
    const float inv = (gl > 0.f) ? (1.f / gl) : 0.f;
    __nv_bfloat16* op = out + (size_t)(seq * num_q_heads + qh) * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) op[lane + e * 32] = __float2bfloat16(acc[e] * inv);
}

template __global__ void fa_split_kernel<128>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int);
template __global__ void fa_combine_kernel<128>(const float*, const float*, const float*, __nv_bfloat16*, int, int);

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/attention.h"

void launch_flash_decode_split(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens, void* out,
    float* part_m, float* part_l, float* part_acc,
    int num_seqs, int num_q_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks, int n_splits, float scale, cudaStream_t stream
) {
    dim3 g1(num_q_heads * n_splits, num_seqs);
    fa_split_kernel<128><<<g1, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k_pool),
        reinterpret_cast<const __nv_bfloat16*>(v_pool), block_table, seq_lens,
        part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks, n_splits);
    dim3 g2(num_q_heads, num_seqs);
    fa_combine_kernel<128><<<g2, 32, 0, stream>>>(
        part_m, part_l, part_acc, reinterpret_cast<__nv_bfloat16*>(out), num_q_heads, n_splits);
    (void)head_dim;
}
#endif

} // namespace kernels
} // namespace sparkinfer
