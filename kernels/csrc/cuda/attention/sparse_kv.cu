// Sink + sliding-window sparse-KV for Qwythos GQA-4 hd256 full-attention decode.
// [Paral1995] 2026-07-13. SPARKINFER_SPARSE_KV=0 disables; default ON for Qwythos GQA-4 hd256 int8-KV.
//
// Per full-attn layer after int8 KV append:
//   (1) fa_kv_window_select — sink block 0 + last W logical blocks (StreamingLLM-style)
//   (2) fa_split_gqa_sparse  — flash-split over the selected blocks only
// Positions/seqlen read from DEVICE pointers for CUDA-graph replay safety.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer { namespace kernels {

__device__ __forceinline__ float skv_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float skv_wsum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// (1) Build per-kv_head block list: sink (block 0) + recent window.
// Grid (num_kv_heads); block 32. Same list for every kv_head (logical blocks are shared).
__global__ void fa_kv_window_select(
    const int* __restrict__ seq_lens, int* __restrict__ sel_blk,
    int num_kv_heads, int block_size, int n_sel, int window_w
) {
    const int kvh = blockIdx.x;
    if (kvh >= num_kv_heads) return;
    int* sel = sel_blk + (size_t)kvh * n_sel;
    if (threadIdx.x != 0) return;

    const int sl = seq_lens[0];
    const int n_blk = (sl + block_size - 1) / block_size;
    int count = 0;
    if (n_blk > 0 && count < n_sel) sel[count++] = 0;   // attention sink
    const int recent_start = (window_w >= n_blk - 1) ? 1 : (n_blk - window_w);
    for (int b = recent_start; b < n_blk && count < n_sel; b++) sel[count++] = b;
    for (int i = count; i < n_sel; i++) sel[i] = -1;
}

// (2) Sparse flash split. Walks n_sel selected blocks instead of a contiguous chunk.
template <int HEAD_DIM, int GQA>
__global__ void fa_split_gqa_sparse(
    const __nv_bfloat16* __restrict__ q, const signed char* __restrict__ k_pool,
    const signed char* __restrict__ v_pool, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, const int* __restrict__ sel_blk,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    float scale, int num_q_heads, int num_kv_heads, int block_size, int max_blocks,
    int n_splits, int n_sel,
    const __half* __restrict__ k_scale, const __half* __restrict__ v_scale
) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const int split = blockIdx.x % n_splits;
    const int kvh   = blockIdx.x / n_splits;
    const int warp  = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int qh    = kvh * GQA + warp;

    float qr[ELEMS];
    const __nv_bfloat16* qp = q + (size_t)qh * HEAD_DIM;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) qr[e] = skv_to_f(qp[lane + e * 32]);

    const int sl = seq_lens[0];
    const int bps = (n_sel + n_splits - 1) / n_splits;
    const int bstart = split * bps, bend = min(n_sel, bstart + bps);
    const int* sel = sel_blk + (size_t)kvh * n_sel;

    float m = -1e30f, l = 0.f, acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    __shared__ __nv_bfloat16 s_k[16 * HEAD_DIM], s_v[16 * HEAD_DIM];
    __shared__ size_t s_rowbase[16];
    __shared__ float s_ksc[16], s_vsc[16];

    for (int i = bstart; i < bend; i++) {
        const int lblk = sel[i];
        if (lblk < 0) continue;
        const int phys  = block_table[lblk];
        const int valid = min(block_size, sl - lblk * block_size);
        if (valid <= 0) continue;
        if ((int)threadIdx.x < valid) {
            const size_t tokrow = (size_t)(phys * block_size + threadIdx.x) * num_kv_heads + kvh;
            s_rowbase[threadIdx.x] = tokrow * HEAD_DIM;
            s_ksc[threadIdx.x] = __half2float(k_scale[tokrow]);
            s_vsc[threadIdx.x] = __half2float(v_scale[tokrow]);
        }
        __syncthreads();
        for (int j = threadIdx.x * 8; j < valid * HEAD_DIM; j += blockDim.x * 8) {
            const int within = j / HEAD_DIM;
            const size_t base = s_rowbase[within] + (j % HEAD_DIM);
            const float ks = s_ksc[within], vs = s_vsc[within];
            const int2 kr = __ldg(reinterpret_cast<const int2*>(k_pool + base));
            const int2 vr = __ldg(reinterpret_cast<const int2*>(v_pool + base));
            const signed char* kc = reinterpret_cast<const signed char*>(&kr);
            const signed char* vc = reinterpret_cast<const signed char*>(&vr);
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                s_k[j + t] = __float2bfloat16((float)kc[t] * ks);
                s_v[j + t] = __float2bfloat16((float)vc[t] * vs);
            }
        }
        __syncthreads();
        for (int tt = 0; tt < valid; tt++) {
            float p = 0.f;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++) p += qr[e] * skv_to_f(s_k[tt * HEAD_DIM + lane + e * 32]);
            const float score = skv_wsum(p) * scale;
            const float mn = fmaxf(m, score), corr = __expf(m - mn), pe = __expf(score - mn);
            l = l * corr + pe;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++) acc[e] = acc[e] * corr + pe * skv_to_f(s_v[tt * HEAD_DIM + lane + e * 32]);
            m = mn;
        }
        __syncthreads();
    }

    const int idx = qh * n_splits + split;
    if (lane == 0) { part_m[idx] = m; part_l[idx] = l; }
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) part_acc[(size_t)idx * HEAD_DIM + lane + e * 32] = acc[e];
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
void launch_fa_kv_window_select(
    const int* seq_lens, int* sel_blk, int num_kv_heads, int block_size,
    int n_sel, int window_w, cudaStream_t stream
) {
    fa_kv_window_select<<<num_kv_heads, 32, 0, stream>>>(
        seq_lens, sel_blk, num_kv_heads, block_size, n_sel, window_w);
}

void launch_flash_decode_split_sparse(
    const void* q, const void* k_pool_layer, const void* v_pool_layer,
    const int* block_table, const int* seq_lens, const int* sel_blk,
    float* part_m, float* part_l, float* part_acc,
    int num_q_heads, int num_kv_heads, int head_dim, int block_size, int max_blocks,
    int n_splits, int n_sel, float scale,
    const void* k_scale_layer, const void* v_scale_layer, cudaStream_t stream
) {
    if (head_dim != 256 || num_q_heads != num_kv_heads * 4) return;
    dim3 grid(num_kv_heads * n_splits, 1);
    fa_split_gqa_sparse<256, 4><<<grid, 4 * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const signed char*>(k_pool_layer),
        reinterpret_cast<const signed char*>(v_pool_layer),
        block_table, seq_lens, sel_blk, part_m, part_l, part_acc, scale,
        num_q_heads, num_kv_heads, block_size, max_blocks, n_splits, n_sel,
        reinterpret_cast<const __half*>(k_scale_layer),
        reinterpret_cast<const __half*>(v_scale_layer));
}
#endif

}} // namespace sparkinfer::kernels
