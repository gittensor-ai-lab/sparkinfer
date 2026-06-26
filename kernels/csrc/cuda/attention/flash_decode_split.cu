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
    for (int e = 0; e < ELEMS; e++) qr[e] = fa_to_f(qp[lane + e * 32]);

    const int sl    = seq_lens[seq];
    const int chunk = (sl + n_splits - 1) / n_splits;
    const int start = split * chunk;
    const int end   = min(sl, start + chunk);

    float m = -1e30f, l = 0.f, acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    for (int t = start; t < end; t++) {
        const int blk = t / block_size, within = t % block_size;
        const int phys = block_table[seq * max_blocks + blk];
        const size_t base = ((size_t)(phys * block_size + within) * num_kv_heads + kvh) * HEAD_DIM;
        float p = 0.f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) p += qr[e] * fa_to_f(k_pool[base + lane + e * 32]);
        const float score = fa_wsum(p) * scale;
        const float mn = fmaxf(m, score), corr = __expf(m - mn), pe = __expf(score - mn);
        l = l * corr + pe;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) acc[e] = acc[e] * corr + pe * fa_to_f(v_pool[base + lane + e * 32]);
        m = mn;
    }

    const int idx = (seq * num_q_heads + qh) * n_splits + split;
    if (lane == 0) { part_m[idx] = m; part_l[idx] = l; }
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) part_acc[(size_t)idx * HEAD_DIM + lane + e * 32] = acc[e];
}

// Combine the split partials with DG x NW parallelism over the 1-block-per-head
// original (which idled at ~2% occupancy with a serial n_splits loop). DG head-dim
// groups -> DG x more blocks; NW warps per block each fold a 1/NW stripe of the
// splits, then a shared-memory log-sum-exp merge across warps. grid=(heads*DG,seqs).
template <int HEAD_DIM, int DG, int NW>
__global__ void fa_combine_kernel(
    const float* __restrict__ part_m, const float* __restrict__ part_l,
    const float* __restrict__ part_acc, __nv_bfloat16* __restrict__ out,
    int num_q_heads, int n_splits
) {
    constexpr int ELEMS = HEAD_DIM / (32 * DG);
    const int seq = blockIdx.y, qh = blockIdx.x / DG, dg = blockIdx.x % DG;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int idxbase = (seq * num_q_heads + qh) * n_splits;
    const int doff = dg * (HEAD_DIM / DG) + lane;     // first head-dim this lane owns

    // per-warp local combine over its split stripe (local max -> weighted l/acc)
    float lm = -1e30f;
    for (int s = warp; s < n_splits; s += NW) lm = fmaxf(lm, part_m[idxbase + s]);
    float ll = 0.f, lacc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) lacc[e] = 0.f;
    for (int s = warp; s < n_splits; s += NW) {
        const float sc = __expf(part_m[idxbase + s] - lm);
        ll += part_l[idxbase + s] * sc;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) lacc[e] += sc * part_acc[(size_t)(idxbase + s) * HEAD_DIM + doff + e * 32];
    }

    __shared__ float s_m[NW], s_l[NW], s_acc[NW][32 * ELEMS];
    if (lane == 0) { s_m[warp] = lm; s_l[warp] = ll; }
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) s_acc[warp][lane * ELEMS + e] = lacc[e];
    __syncthreads();
    if (warp != 0) return;

    float gm = -1e30f;
    #pragma unroll
    for (int w = 0; w < NW; w++) gm = fmaxf(gm, s_m[w]);
    float gl = 0.f, acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;
    #pragma unroll
    for (int w = 0; w < NW; w++) {
        const float sc = __expf(s_m[w] - gm);
        gl += s_l[w] * sc;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) acc[e] += sc * s_acc[w][lane * ELEMS + e];
    }
    const float inv = (gl > 0.f) ? (1.f / gl) : 0.f;
    __nv_bfloat16* op = out + (size_t)(seq * num_q_heads + qh) * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) op[doff + e * 32] = __float2bfloat16(acc[e] * inv);
}

#ifndef FA_COMBINE_DG
#define FA_COMBINE_DG 4     // head-dim groups (DG x blocks); sweepable
#endif
#ifndef FA_COMBINE_NW
#define FA_COMBINE_NW 4     // warps/block folding the split stripes; sweepable
#endif
template __global__ void fa_split_kernel<128>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int);
template __global__ void fa_combine_kernel<128, FA_COMBINE_DG, FA_COMBINE_NW>(const float*, const float*, const float*, __nv_bfloat16*, int, int);

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
    dim3 g2(num_q_heads * FA_COMBINE_DG, num_seqs);
    fa_combine_kernel<128, FA_COMBINE_DG, FA_COMBINE_NW><<<g2, FA_COMBINE_NW * 32, 0, stream>>>(
        part_m, part_l, part_acc, reinterpret_cast<__nv_bfloat16*>(out), num_q_heads, n_splits);
    (void)head_dim;
}
#endif

} // namespace kernels
} // namespace sparkinfer
