// ============================================================================
// Tensor-core (int8 wmma) prefill attention for Qwythos (Qwen3.5), hd256 full-attn layers.
//
// WHY THIS EXISTS
// ---------------
// The batched prompt prefill (#398) computed the hd256 full-attention layers with a naive
// warp-per-query kernel; the merged windowed/tiled prefill attention (#455) then removed the
// O(N^2) *bandwidth* problem by restricting each query to an attention sink + sliding window
// (StreamingLLM, matching the merged sparse-KV decode #379) and by staging each KV tile in
// shared memory once per query tile.
//
// What is left is a *compute* problem. Both of those kernels evaluate QK^T and PV with scalar
// FMA plus a 5-shuffle warp reduction per key, and they stage K and V into shared memory as
// fp32 (2 * TK * 256 * 4B = 64 KB), which caps them at ~1 block/SM. Measured on an RTX 5090
// (nsys, ctx=32768): win_prefill_windowed_kernel = 262 ms per layer for ~2.08 TFLOP of work =
// ~8 TFLOP/s, i.e. 30.5% of prefill time at a small fraction of the achievable rate.
//
// This kernel runs the SAME masked online-softmax attention on the int8 tensor cores, reusing
// the pattern the merged int8-MMA flash-decode (fa_split_gqa_mma_i8, #338) already ships:
//   * K/V stay int8 and are fed to wmma DIRECTLY out of the paged pool -- a KV page is exactly
//     16 tokens and wmma's tile is 16x16, so a page IS a fragment with ldm = n_kv_heads*HEAD_DIM.
//     No fp32 KV staging, so shared memory drops 64 KB -> ~31 KB (3 blocks/SM).
//   * Q is quantized per query row to int8 (one scale per row); QK^T runs int8 x int8 -> int32
//     and the per-row Q scale, per-token K scale and softmax scale are applied to the int32.
//   * P is rescaled by the per-token V scale, then quantized per row, so PV also runs int8 on
//     the tensor cores with the row scale applied to the int32 accumulator.
//
// The mask (causal + sink/window) and the online-softmax recurrence are identical to #455, so
// the output matches the scalar windowed path to int8 round-off. The window is read from the
// SAME env knob (SPARKINFER_PREFILL_ATTN_WINDOW, default 256 blocks) so the three paths --
// scalar-windowed prefill, this MMA prefill, and the sparse-KV decode -- stay consistent.
//
// NOTE ON THE SCORE STRIDE: the decode reference stores the QK int32 tile with ldm=HEAD_DIM but
// reads it back at row stride 128; those agree only at HEAD_DIM==128. Here the score buffer is
// explicitly [BM][GN] with one stride (GN) used for both the wmma store and every read.
//
// A KV page is 16 tokens and the query tile is 16 rows aligned to 16, so every query in a tile
// shares one window start (n_blk_q = (t+16)/16 is constant across the tile) -- the sink/window
// range is computed once per block and only the causal bound varies per row.
// ============================================================================
#include "sparkinfer/kernels/prefill_attn_mma.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdlib>

namespace sparkinfer {
namespace kernels {

namespace {

// One block owns BM=16 query rows of ONE q-head; GROUP_BLKS KV pages (GN keys) are processed per
// iteration, one page per warp for the QK mma. WARPS must equal GROUP_BLKS and HEAD_DIM/16 must
// be divisible by WARPS (each warp owns HEAD_DIM/16/WARPS output d-tiles in the PV mma).
template <int HEAD_DIM, int GROUP_BLKS>
__global__ __launch_bounds__(GROUP_BLKS * 32, 3) void pf_attn_mma_i8_kernel(
    const __nv_bfloat16* __restrict__ q, const signed char* __restrict__ k_pool,
    const signed char* __restrict__ v_pool, const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale, const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ attn, int n_tokens, int n_q_heads, int n_kv_heads,
    int block_size, int max_blocks_per_seq, float scale, int win_blocks) {
    using namespace nvcuda::wmma;
    constexpr int BM    = 16;                    // query rows per block == wmma M == KV page size
    constexpr int GN    = GROUP_BLKS * 16;       // keys per group
    constexpr int KH    = HEAD_DIM / 16;         // QK k-steps
    constexpr int DTILE = HEAD_DIM / 16;         // PV output d-tiles
    constexpr int WARPS = GROUP_BLKS;
    constexpr int DPW   = DTILE / WARPS;         // d-tiles per warp
    constexpr int QE    = HEAD_DIM / 32;         // Q elements per lane per row

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    const int qbase = blockIdx.x * BM;
    const int head  = blockIdx.y;
    const int kvh   = head / (n_q_heads / n_kv_heads);
    const size_t KVLD = (size_t)n_kv_heads * HEAD_DIM;   // int8 token stride in the pool
    const int SLD = n_kv_heads;                          // scale stride per (token, kv_head)

    extern __shared__ char mma_smem[];
    signed char* s_qi = reinterpret_cast<signed char*>(mma_smem);   // [BM][HEAD_DIM]
    signed char* s_pi = s_qi + BM * HEAD_DIM;                       // [BM][GN]
    float* s_s  = reinterpret_cast<float*>(s_pi + BM * GN);         // [BM][GN] scores / P'
    float* s_o  = s_s + BM * GN;                                    // [BM][HEAD_DIM] O (epilogue only)
    float* s_ks = s_o + BM * HEAD_DIM;                              // [GN]
    float* s_vs = s_ks + GN;                                        // [GN]
    float* s_qs = s_vs + GN;                                        // [BM]
    float* s_ps = s_qs + BM;                                        // [BM]
    float* s_m  = s_ps + BM;                                        // [BM]
    float* s_l  = s_m + BM;                                         // [BM]
    float* s_corr = s_l + BM;                                       // [BM] per-group rescale

    // The running O lives in per-warp accumulator fragments (warp w owns d-tiles w*DPW..+DPW),
    // not in shared memory: the old path bounced every PV tile through a smem int landing zone
    // and rescaled all BM*HEAD_DIM floats of s_o through smem each group, at two extra
    // __syncthreads per group. Element rows for the rescale come from an index fragment loaded
    // once from a per-warp smem tile (value (row<<8)|col), so no accumulator-layout assumption
    // is made. All arithmetic keeps the old per-element op/rounding sequence -> bit-identical.
    fragment<accumulator, 16, 16, 16, float> ofr[DPW];
    fragment<accumulator, 16, 16, 16, int> idxf;
    {
        int* tile = reinterpret_cast<int*>(s_s) + warp * 256;       // disjoint per warp
        for (int i = lane; i < 256; i += 32) tile[i] = ((i >> 4) << 8) | (i & 15);
        __syncwarp();
        load_matrix_sync(idxf, tile, 16, mem_row_major);
    }
    #pragma unroll
    for (int dd = 0; dd < DPW; dd++) fill_fragment(ofr[dd], 0.f);

    // ---- load + quantize Q rows (warp w owns rows 2w, 2w+1 at WARPS=8) ----
    #pragma unroll
    for (int rr = 0; rr < BM / WARPS; rr++) {
        const int r = warp * (BM / WARPS) + rr;
        const int qtok = qbase + r;
        float qv[QE], amax = 0.f;
        #pragma unroll
        for (int e = 0; e < QE; e++) {
            qv[e] = (qtok < n_tokens)
                  ? __bfloat162float(q[((size_t)qtok * n_q_heads + head) * HEAD_DIM + lane + e * 32])
                  : 0.f;
            amax = fmaxf(amax, fabsf(qv[e]));
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = amax / 127.0f;
        if (lane == 0) s_qs[r] = d;
        #pragma unroll
        for (int e = 0; e < QE; e++)
            s_qi[r * HEAD_DIM + lane + e * 32] =
                (signed char)((amax == 0.f) ? 0 : (int)roundf(qv[e] / d));
    }
    if (tid < BM) { s_m[tid] = -1e30f; s_l[tid] = 0.f; }
    __syncthreads();

    // ---- sink/window range for this (16-aligned) query tile ----
    const int last_q = min(qbase + BM - 1, n_tokens - 1);
    int blk_rs = 0;                                   // first token of the recent window
    if (win_blocks > 0) {
        const int n_blk_q = (qbase + block_size) / block_size;   // constant across the tile
        const int rsb = (win_blocks >= n_blk_q - 1) ? 1 : (n_blk_q - win_blocks);
        blk_rs = rsb * block_size;
    }
    const bool split_sink = (win_blocks > 0) && (blk_rs > block_size);

    // Process a page-aligned key range [lo, hi) in GN-key groups.
    auto run_range = [&](int lo, int hi) {
        for (int k0 = lo; k0 < hi; k0 += GN) {
            const int nk   = min(GN, hi - k0);
            const int gblk = (nk + 15) / 16;          // pages touched by this group
            // stage per-token K/V dequant scales for the group
            for (int j = tid; j < gblk * 16; j += blockDim.x) {
                const int lb = (k0 / block_size) + j / 16, within = j & 15;
                const int pb = block_table[lb];
                const size_t si = (size_t)(pb * block_size + within) * SLD + kvh;
                s_ks[j] = __half2float(k_scale[si]);
                s_vs[j] = __half2float(v_scale[si]);
            }

            // ---- QK: int8 mma -> int32 scores, one page per warp ----
            if (warp < gblk) {
                const int pb = block_table[(k0 / block_size) + warp];
                const signed char* kb =
                    k_pool + ((size_t)pb * block_size * n_kv_heads + kvh) * HEAD_DIM;
                fragment<matrix_a, 16, 16, 16, signed char, row_major> af;
                fragment<matrix_b, 16, 16, 16, signed char, col_major> bf;
                fragment<accumulator, 16, 16, 16, int> cf;
                fill_fragment(cf, 0);
                #pragma unroll
                for (int ks = 0; ks < KH; ks++) {
                    load_matrix_sync(af, s_qi + ks * 16, HEAD_DIM);
                    load_matrix_sync(bf, kb + ks * 16, KVLD);
                    mma_sync(cf, af, bf, cf);
                }
                store_matrix_sync(reinterpret_cast<int*>(s_s) + warp * 16, cf, GN, mem_row_major);
            }
            __syncthreads();
            const int* s_si = reinterpret_cast<const int*>(s_s);

            // ---- online softmax; fold V scale into P', quantize P' per row ----
            #pragma unroll
            for (int rr = 0; rr < BM / WARPS; rr++) {
                const int r = warp * (BM / WARPS) + rr;
                const int qtok = qbase + r;
                float sc[GN / 32], mx = -1e30f;
                #pragma unroll
                for (int u = 0; u < GN / 32; u++) {
                    const int t = lane + u * 32, gtok = k0 + t;
                    // causal + (sink OR recent window); the window start is uniform across the tile
                    const bool live = (t < gblk * 16) && (gtok < hi) && (qtok < n_tokens) &&
                                      (gtok <= qtok) &&
                                      (win_blocks <= 0 || gtok < block_size || gtok >= blk_rs);
                    sc[u] = live ? (float)s_si[r * GN + t] * s_qs[r] * s_ks[t] * scale : -1e30f;
                    mx = fmaxf(mx, sc[u]);
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
                const float m_old = s_m[r], m_new = fmaxf(m_old, mx), corr = __expf(m_old - m_new);
                float sum = 0.f, pamax = 0.f;
                #pragma unroll
                for (int u = 0; u < GN / 32; u++) {
                    const int t = lane + u * 32;
                    float pv = 0.f;
                    if (sc[u] > -1e29f) {
                        const float p = __expf(sc[u] - m_new);
                        sum += p; pv = p * s_vs[t]; pamax = fmaxf(pamax, fabsf(pv));
                    }
                    s_s[r * GN + t] = pv;
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    sum   += __shfl_xor_sync(0xffffffffu, sum, o);
                    pamax  = fmaxf(pamax, __shfl_xor_sync(0xffffffffu, pamax, o));
                }
                const float pd = pamax / 127.0f;
                if (lane == 0) { s_m[r] = m_new; s_l[r] = s_l[r] * corr + sum;
                                 s_ps[r] = pd; s_corr[r] = corr; }
                for (int t = lane; t < gblk * 16; t += 32)
                    s_pi[r * GN + t] =
                        (signed char)((pamax == 0.f) ? 0 : (int)roundf(s_s[r * GN + t] / pd));
            }
            __syncthreads();

            // ---- PV: int8 mma -> int32, O = O*corr + int32 * per-row P' scale, in registers ----
            // No smem landing zone and no trailing barriers: the next group's after-QK barrier
            // already orders every cross-warp reuse (softmax g+1 writes s_pi/s_ps/s_corr only
            // after all warps passed it, i.e. after they finished this PV).
            #pragma unroll
            for (int dd = 0; dd < DPW; dd++) {
                const int dt = warp * DPW + dd;
                fragment<accumulator, 16, 16, 16, int> cf;
                fill_fragment(cf, 0);
                for (int ks = 0; ks < gblk; ks++) {
                    const int pb = block_table[(k0 / block_size) + ks];
                    const signed char* vb =
                        v_pool + ((size_t)pb * block_size * n_kv_heads + kvh) * HEAD_DIM + dt * 16;
                    fragment<matrix_a, 16, 16, 16, signed char, row_major> af;
                    fragment<matrix_b, 16, 16, 16, signed char, row_major> bf;
                    load_matrix_sync(af, s_pi + ks * 16, GN);
                    load_matrix_sync(bf, vb, KVLD);
                    mma_sync(cf, af, bf, cf);
                }
                // Rounding matches the old smem path exactly: the *= corr rescale was a separate
                // rounded multiply, while the += pv*ps accumulate compiled to an FMA -- so it is
                // __fmaf_rn over a rounded product here (verified bit-exact against the old
                // kernel; a plain mul+add differs).
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    const int r = idxf.x[e] >> 8;
                    ofr[dd].x[e] = __fmaf_rn((float)cf.x[e], s_ps[r],
                                             __fmul_rn(ofr[dd].x[e], s_corr[r]));
                }
            }
        }
    };

    if (split_sink) run_range(0, block_size);
    run_range(split_sink ? blk_rs : 0, last_q + 1);

    // Land the register O tiles in s_o once, so the epilogue below stays coalesced + unchanged.
    #pragma unroll
    for (int dd = 0; dd < DPW; dd++)
        store_matrix_sync(s_o + (warp * DPW + dd) * 16, ofr[dd], HEAD_DIM, mem_row_major);
    __syncthreads();

    // ---- epilogue ----
    for (int r = 0; r < BM; r++) {
        const int qtok = qbase + r;
        if (qtok >= n_tokens) break;
        const float l = s_l[r];
        const float inv = (l > 0.f) ? (1.f / l) : 0.f;
        for (int c = tid; c < HEAD_DIM; c += blockDim.x)
            attn[((size_t)qtok * n_q_heads + head) * HEAD_DIM + c] =
                __float2bfloat16(s_o[r * HEAD_DIM + c] * inv);
    }
}

}  // namespace

// ============================================================================
// GQA-fused int8 tensor-core prefill attention. One block owns BM query rows of
// RQH query heads that SHARE one kv-head, so each K page and V tile is loaded from
// the paged pool ONCE and fed to RQH mma's (one per q-head) instead of being
// re-read once per q-head. Qwen3.6 attention is GQA-8 (16 q-heads / 2 kv-heads),
// and the per-q-head kernel below re-loaded each kv-head's K/V 8x; that redundant
// int8 K/V traffic is the bound (nsys: attn_mma = 17% of qwen36 prefill @32k).
// RQH=1 is bit-identical to the per-head kernel. Math (mask, online softmax, int8
// round) is unchanged -- only the load ordering differs.
// ============================================================================
// Shared smem layout for the GQA kernel, so the kernel and its launcher cannot drift apart.
//
// s_o (the [BM][HEAD_DIM] float epilogue landing zone) is only ever touched AFTER the last key
// group has been consumed, and s_qi/s_pi/s_s are all dead by that point -- only s_l survives into
// the epilogue. So s_o does not need its own 16 KB: it can overlay the front of the block. That
// is what puts RQH=6 -- Qwen3.8's whole GQA group, so each K page and V tile is read exactly once
// instead of twice at RQH=3 -- under the 100 KB sm_120 cap at all: 105,344 B without this,
// 88,960 B with it. It also takes RQH=3 from 61,376 B to 44,992 B, back to 2 blocks/SM.
//
// The overlay costs one extra __syncthreads() before the epilogue: the PV loop deliberately runs
// without a trailing barrier (it leans on the next group's after-QK barrier), so without one a
// fast warp's first s_o store would land on s_pi while a slow warp was still reading it.
template <int HEAD_DIM, int GROUP_BLKS, int RQH>
struct attn_gqa_smem {
    static constexpr int    BM    = 16;
    static constexpr int    GN    = GROUP_BLKS * 16;
    static constexpr size_t front = (size_t)RQH * BM * HEAD_DIM                    // s_qi (int8)
                                  + (size_t)RQH * BM * GN                          // s_pi (int8)
                                  + (size_t)(RQH * BM * GN) * sizeof(float);       // s_s
    static constexpr size_t olen  = (size_t)(BM * HEAD_DIM) * sizeof(float);       // s_o
    static constexpr bool   alias = front >= olen;                                 // holds for RQH>=2
    static constexpr size_t bytes = front + (alias ? 0 : olen)
                                  + (size_t)(2 * GN + 5 * RQH * BM) * sizeof(float);
};

template <int HEAD_DIM, int GROUP_BLKS, int RQH>
__global__ __launch_bounds__(GROUP_BLKS * 32, (RQH <= 3 ? 2 : 1)) void pf_attn_mma_gqa_kernel(
    const __nv_bfloat16* __restrict__ q, const signed char* __restrict__ k_pool,
    const signed char* __restrict__ v_pool, const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale, const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ attn, int n_tokens, int n_q_heads, int n_kv_heads,
    int block_size, int max_blocks_per_seq, float scale, int win_blocks) {
    using namespace nvcuda::wmma;
    constexpr int BM    = 16;
    constexpr int GN    = GROUP_BLKS * 16;
    constexpr int KH    = HEAD_DIM / 16;
    constexpr int DTILE = HEAD_DIM / 16;
    constexpr int WARPS = GROUP_BLKS;
    constexpr int DPW   = DTILE / WARPS;
    constexpr int QE    = HEAD_DIM / 32;

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    const int qbase = blockIdx.x * BM;
    const int head0 = blockIdx.y * RQH;                       // first q-head this block owns
    const int gqa   = n_q_heads / n_kv_heads;
    const int kvh   = head0 / gqa;                            // all RQH heads share this kv-head
    const size_t KVLD = (size_t)n_kv_heads * HEAD_DIM;
    const int SLD = n_kv_heads;

    extern __shared__ char mma_smem[];
    // Per-q-head Q(int8), P(int8), scores(float); shared K/V scales; per-(qh,row) softmax state.
    signed char* s_qi = reinterpret_cast<signed char*>(mma_smem);   // [RQH][BM][HEAD_DIM]
    signed char* s_pi = s_qi + (size_t)RQH * BM * HEAD_DIM;          // [RQH][BM][GN]
    float* s_s  = reinterpret_cast<float*>(s_pi + (size_t)RQH * BM * GN); // [RQH][BM][GN]
    using SM = attn_gqa_smem<HEAD_DIM, GROUP_BLKS, RQH>;
    // [BM][HEAD_DIM] epilogue landing -- overlays s_qi/s_pi/s_s, which are dead by the epilogue.
    float* s_o  = SM::alias ? reinterpret_cast<float*>(mma_smem)
                            : reinterpret_cast<float*>(mma_smem + SM::front);
    float* s_ks = reinterpret_cast<float*>(mma_smem + SM::front + (SM::alias ? 0 : SM::olen));
    float* s_vs = s_ks + GN;                                         // [GN] shared
    float* s_qs = s_vs + GN;                                         // [RQH][BM]
    float* s_ps = s_qs + RQH * BM;                                   // [RQH][BM]
    float* s_m  = s_ps + RQH * BM;                                   // [RQH][BM]
    float* s_l  = s_m + RQH * BM;                                    // [RQH][BM]
    float* s_corr = s_l + RQH * BM;                                  // [RQH][BM]

    fragment<accumulator, 16, 16, 16, float> ofr[RQH][DPW];
    fragment<accumulator, 16, 16, 16, int> idxf;
    {
        int* tile = reinterpret_cast<int*>(s_s) + warp * 256;
        for (int i = lane; i < 256; i += 32) tile[i] = ((i >> 4) << 8) | (i & 15);
        __syncwarp();
        load_matrix_sync(idxf, tile, 16, mem_row_major);
    }
    #pragma unroll
    for (int h = 0; h < RQH; h++)
        #pragma unroll
        for (int dd = 0; dd < DPW; dd++) fill_fragment(ofr[h][dd], 0.f);

    // ---- load + quantize Q rows for each of the RQH heads ----
    #pragma unroll
    for (int h = 0; h < RQH; h++) {
        const int head = head0 + h;
        #pragma unroll
        for (int rr = 0; rr < BM / WARPS; rr++) {
            const int r = warp * (BM / WARPS) + rr;
            const int qtok = qbase + r;
            float qv[QE], amax = 0.f;
            #pragma unroll
            for (int e = 0; e < QE; e++) {
                qv[e] = (qtok < n_tokens)
                      ? __bfloat162float(q[((size_t)qtok * n_q_heads + head) * HEAD_DIM + lane + e * 32])
                      : 0.f;
                amax = fmaxf(amax, fabsf(qv[e]));
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
            const float d = amax / 127.0f;
            if (lane == 0) s_qs[h * BM + r] = d;
            #pragma unroll
            for (int e = 0; e < QE; e++)
                s_qi[((size_t)h * BM + r) * HEAD_DIM + lane + e * 32] =
                    (signed char)((amax == 0.f) ? 0 : (int)roundf(qv[e] / d));
        }
    }
    if (tid < RQH * BM) { s_m[tid] = -1e30f; s_l[tid] = 0.f; }
    __syncthreads();

    const int last_q = min(qbase + BM - 1, n_tokens - 1);
    int blk_rs = 0;
    if (win_blocks > 0) {
        const int n_blk_q = (qbase + block_size) / block_size;
        const int rsb = (win_blocks >= n_blk_q - 1) ? 1 : (n_blk_q - win_blocks);
        blk_rs = rsb * block_size;
    }
    const bool split_sink = (win_blocks > 0) && (blk_rs > block_size);

    auto run_range = [&](int lo, int hi) {
        for (int k0 = lo; k0 < hi; k0 += GN) {
            const int nk   = min(GN, hi - k0);
            const int gblk = (nk + 15) / 16;
            // K/V dequant scales for the group -- shared across all RQH heads (one kv-head).
            for (int j = tid; j < gblk * 16; j += blockDim.x) {
                const int lb = (k0 / block_size) + j / 16, within = j & 15;
                const int pb = block_table[lb];
                const size_t si = (size_t)(pb * block_size + within) * SLD + kvh;
                s_ks[j] = __half2float(k_scale[si]);
                s_vs[j] = __half2float(v_scale[si]);
            }

            // ---- QK: load each K page fragment ONCE, feed RQH q-heads ----
            if (warp < gblk) {
                const int pb = block_table[(k0 / block_size) + warp];
                const signed char* kb =
                    k_pool + ((size_t)pb * block_size * n_kv_heads + kvh) * HEAD_DIM;
                fragment<matrix_a, 16, 16, 16, signed char, row_major> af;
                fragment<matrix_b, 16, 16, 16, signed char, col_major> bf;
                fragment<accumulator, 16, 16, 16, int> cf[RQH];
                #pragma unroll
                for (int h = 0; h < RQH; h++) fill_fragment(cf[h], 0);
                #pragma unroll
                for (int ks = 0; ks < KH; ks++) {
                    load_matrix_sync(bf, kb + ks * 16, KVLD);        // K fragment: loaded once
                    #pragma unroll
                    for (int h = 0; h < RQH; h++) {
                        load_matrix_sync(af, s_qi + ((size_t)h * BM) * HEAD_DIM + ks * 16, HEAD_DIM);
                        mma_sync(cf[h], af, bf, cf[h]);
                    }
                }
                #pragma unroll
                for (int h = 0; h < RQH; h++)
                    store_matrix_sync(reinterpret_cast<int*>(s_s) + (size_t)h * BM * GN + warp * 16,
                                      cf[h], GN, mem_row_major);
            }
            __syncthreads();

            // ---- online softmax per head; quantize P' ----
            #pragma unroll
            for (int h = 0; h < RQH; h++) {
                const int qh_head = head0 + h;
                const int* s_si = reinterpret_cast<const int*>(s_s) + (size_t)h * BM * GN;
                float* s_sh = s_s + (size_t)h * BM * GN;
                signed char* s_pih = s_pi + (size_t)h * BM * GN;
                #pragma unroll
                for (int rr = 0; rr < BM / WARPS; rr++) {
                    const int r = warp * (BM / WARPS) + rr;
                    const int qtok = qbase + r;
                    float sc[GN / 32], mx = -1e30f;
                    #pragma unroll
                    for (int u = 0; u < GN / 32; u++) {
                        const int t = lane + u * 32, gtok = k0 + t;
                        const bool live = (t < gblk * 16) && (gtok < hi) && (qtok < n_tokens) &&
                                          (gtok <= qtok) &&
                                          (win_blocks <= 0 || gtok < block_size || gtok >= blk_rs);
                        sc[u] = live ? (float)s_si[r * GN + t] * s_qs[h * BM + r] * s_ks[t] * scale : -1e30f;
                        mx = fmaxf(mx, sc[u]);
                    }
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
                    const float m_old = s_m[h * BM + r], m_new = fmaxf(m_old, mx), corr = __expf(m_old - m_new);
                    float sum = 0.f, pamax = 0.f;
                    #pragma unroll
                    for (int u = 0; u < GN / 32; u++) {
                        const int t = lane + u * 32;
                        float pv = 0.f;
                        if (sc[u] > -1e29f) {
                            const float p = __expf(sc[u] - m_new);
                            sum += p; pv = p * s_vs[t]; pamax = fmaxf(pamax, fabsf(pv));
                        }
                        s_sh[r * GN + t] = pv;
                    }
                    #pragma unroll
                    for (int o = 16; o > 0; o >>= 1) {
                        sum   += __shfl_xor_sync(0xffffffffu, sum, o);
                        pamax  = fmaxf(pamax, __shfl_xor_sync(0xffffffffu, pamax, o));
                    }
                    const float pd = pamax / 127.0f;
                    if (lane == 0) { s_m[h * BM + r] = m_new; s_l[h * BM + r] = s_l[h * BM + r] * corr + sum;
                                     s_ps[h * BM + r] = pd; s_corr[h * BM + r] = corr; }
                    for (int t = lane; t < gblk * 16; t += 32)
                        s_pih[r * GN + t] =
                            (signed char)((pamax == 0.f) ? 0 : (int)roundf(s_sh[r * GN + t] / pd));
                }
            }
            __syncthreads();

            // ---- PV: load each V tile fragment ONCE, feed RQH q-heads ----
            #pragma unroll
            for (int dd = 0; dd < DPW; dd++) {
                const int dt = warp * DPW + dd;
                fragment<accumulator, 16, 16, 16, int> cf[RQH];
                #pragma unroll
                for (int h = 0; h < RQH; h++) fill_fragment(cf[h], 0);
                for (int ks = 0; ks < gblk; ks++) {
                    const int pb = block_table[(k0 / block_size) + ks];
                    const signed char* vb =
                        v_pool + ((size_t)pb * block_size * n_kv_heads + kvh) * HEAD_DIM + dt * 16;
                    fragment<matrix_a, 16, 16, 16, signed char, row_major> af;
                    fragment<matrix_b, 16, 16, 16, signed char, row_major> bf;
                    load_matrix_sync(bf, vb, KVLD);                  // V fragment: loaded once
                    #pragma unroll
                    for (int h = 0; h < RQH; h++) {
                        load_matrix_sync(af, s_pi + (size_t)h * BM * GN + ks * 16, GN);
                        mma_sync(cf[h], af, bf, cf[h]);
                    }
                }
                #pragma unroll
                for (int h = 0; h < RQH; h++)
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        const int r = idxf.x[e] >> 8;
                        ofr[h][dd].x[e] = __fmaf_rn((float)cf[h].x[e], s_ps[h * BM + r],
                                                    __fmul_rn(ofr[h][dd].x[e], s_corr[h * BM + r]));
                    }
            }
        }
    };

    if (split_sink) run_range(0, block_size);
    run_range(split_sink ? blk_rs : 0, last_q + 1);

    // ---- epilogue: one head at a time through the shared s_o landing zone ----
    // s_o overlays s_qi/s_pi/s_s (see attn_gqa_smem), and the PV loop above exits without a
    // trailing barrier, so fence here before the first store lands on a buffer another warp
    // may still be reading.
    __syncthreads();
    #pragma unroll
    for (int h = 0; h < RQH; h++) {
        const int head = head0 + h;
        #pragma unroll
        for (int dd = 0; dd < DPW; dd++)
            store_matrix_sync(s_o + (warp * DPW + dd) * 16, ofr[h][dd], HEAD_DIM, mem_row_major);
        __syncthreads();
        for (int r = 0; r < BM; r++) {
            const int qtok = qbase + r;
            if (qtok >= n_tokens) break;
            const float l = s_l[h * BM + r];
            const float inv = (l > 0.f) ? (1.f / l) : 0.f;
            for (int c = tid; c < HEAD_DIM; c += blockDim.x)
                attn[((size_t)qtok * n_q_heads + head) * HEAD_DIM + c] =
                    __float2bfloat16(s_o[r * HEAD_DIM + c] * inv);
        }
        __syncthreads();
    }
}

template <int HD, int GROUP_BLKS, int RQH>
static bool launch_attn_gqa(const void* q, const signed char* k_pool, const signed char* v_pool,
                            const void* k_scale, const void* v_scale, const int* block_table,
                            void* attn, int n_tokens, int n_q_heads, int n_kv_heads,
                            int block_size, int max_blocks_per_seq, float scale, int win_blocks,
                            cudaStream_t stream) {
    constexpr int BM = 16;
    const size_t sm = attn_gqa_smem<HD, GROUP_BLKS, RQH>::bytes;
    // With the s_o overlay this is 30,336 / 44,992 / 59,648 / 88,960 B at RQH=2/3/4/6 — every
    // tier above RQH=2 is past the 48 KB default, so the opt-in below is
    // REQUIRED for the launch to be valid, and both it and the launch itself have to
    // be checked: a discarded failure here used to report success to the caller, which
    // then skipped the scalar fallback and consumed whatever `attn` already held —
    // silently wrong logits, no diagnostic. cudaFuncSetAttribute is also a PER-DEVICE
    // setting, so the do-once latch is keyed on the device ordinal, not the process
    // (the old process-wide latch left every device but the first unconfigured, and
    // the launch then failed with cudaErrorInvalidValue on exactly the path that
    // needs the raise).
    constexpr int kMaxDevices = 16;
    static int cfg[kMaxDevices] = {0};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDevices) return false;
    if (!cfg[dev]) {
        const cudaError_t ce = cudaFuncSetAttribute(
            pf_attn_mma_gqa_kernel<HD, GROUP_BLKS, RQH>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
        if (ce != cudaSuccess && sm > 48u * 1024u) return false;  // opt-in refused where it is required
        cfg[dev] = 1;
    }
    dim3 grid((n_tokens + BM - 1) / BM, n_q_heads / RQH);
    pf_attn_mma_gqa_kernel<HD, GROUP_BLKS, RQH><<<grid, GROUP_BLKS * 32, sm, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool,
        reinterpret_cast<const __half*>(k_scale), reinterpret_cast<const __half*>(v_scale),
        block_table, reinterpret_cast<__nv_bfloat16*>(attn), n_tokens, n_q_heads, n_kv_heads,
        block_size, max_blocks_per_seq, scale, win_blocks);
    // A rejected launch (e.g. smem over the device limit) enqueues nothing; peek —
    // rather than get — so a pre-existing sticky error is not silently cleared here.
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launch_prefill_attn_mma(
    const void* q, const signed char* k_pool, const signed char* v_pool,
    const void* k_scale, const void* v_scale, const int* block_table, void* attn,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, float scale, cudaStream_t stream) {
    constexpr int HD = 256, GROUP_BLKS = 8, BM = 16;

    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_MMA");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    static const int minctx = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_MMA_MINCTX");
        return e ? atoi(e) : 0;
    }();
    static const int win_blocks = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_WINDOW");
        return e ? atoi(e) : 256;
    }();
    // GQA fusion: one block owns RQH q-heads sharing a kv-head, so each K page / V tile
    // is loaded once and fed RQH mma's instead of being re-read per q-head. RQH=1 disables.
    static const int gqa_rqh = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_GQA_RQH");
        const int v = e ? atoi(e) : 6;
        return (v == 1 || v == 2 || v == 3 || v == 4 || v == 6) ? v : 6;
    }();
    // Token floor for the widest (grid-narrowing) tier -- see the RQH=6 dispatch below.
    static const int minctx6 = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_GQA_MINCTX6");
        return e ? atoi(e) : 2048;
    }();

    if (!enabled || head_dim != HD || block_size != 16 || n_tokens < minctx) return false;
    if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0) return false;

    const int gqa = n_q_heads / n_kv_heads;
    // Each tier reports whether it actually launched; a refusal (opt-in rejected,
    // launch invalid) cascades to the next tier — RQH=2 needs 46,720 B, under the
    // 48 KB default — and finally to the per-q-head kernel below, instead of
    // returning success over an output buffer nothing wrote.
    // RQH=6 tier: the entire GQA group in one block, so each K page and V tile is read exactly
    // once per kv-head instead of twice at RQH=3. Only fits because s_o overlays the front of the
    // block (see attn_gqa_smem) -- 88,960 B against the 100 KB sm_120 cap, where the un-overlaid
    // layout wanted 105,344 B and could not launch at all. 1 block/SM either way, so unlike the
    // RQH=3 tier this is pure traffic reduction with no occupancy given up.
    //
    // Fusing RQH heads divides the grid's head dimension by RQH, so it only pays once the grid is
    // wide enough to fill the device without it: at ctx=128 the grid is 8 x (24/6) = 32 blocks and
    // RQH=6 measured -0.9% against RQH=3's 64, while at ctx=16384 it is 1024 x 4 and measured
    // +1.0%. Gate on token count -- a shape property, so the tier choice is identical on every box.
    if (gqa_rqh >= 6 && gqa % 6 == 0 && n_tokens >= minctx6 &&
        launch_attn_gqa<HD, GROUP_BLKS, 6>(q, k_pool, v_pool, k_scale, v_scale, block_table, attn,
            n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, win_blocks, stream))
        return true;
    // >= 4, not == 4: the default is now 6 (for GQA-6 checkpoints), and a gqa=4 model such as
    // Qwen3.6 must still take this tier rather than falling past it to RQH=2.
    if (gqa_rqh >= 4 && gqa % 4 == 0 &&
        launch_attn_gqa<HD, GROUP_BLKS, 4>(q, k_pool, v_pool, k_scale, v_scale, block_table, attn,
            n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, win_blocks, stream))
        return true;
    // RQH=3 tier: Qwen3.8 is GQA-6, which 4 cannot divide, so it used to fall straight to 2 and
    // re-read each K page / V tile 3x per kv-head instead of 2x. Measured +1.4% at prefill@16k.
    // With the s_o overlay it needs 44,992 B, so it keeps 2 blocks/SM (it cost 61,376 B and one
    // block/SM before) -- which is why the launch bound above is 2 for RQH<=3. Also the tier
    // RQH=6 falls back to below ctx=2048.
    if (gqa_rqh >= 3 && gqa % 3 == 0 &&
        launch_attn_gqa<HD, GROUP_BLKS, 3>(q, k_pool, v_pool, k_scale, v_scale, block_table, attn,
            n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, win_blocks, stream))
        return true;
    if (gqa_rqh >= 2 && gqa % 2 == 0 &&
        launch_attn_gqa<HD, GROUP_BLKS, 2>(q, k_pool, v_pool, k_scale, v_scale, block_table, attn,
            n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, win_blocks, stream))
        return true;

    // Fallback: original per-q-head kernel.
    constexpr int GN = GROUP_BLKS * 16;
    const size_t sm = (size_t)BM * HD
                    + (size_t)BM * GN
                    + (size_t)(BM * GN) * sizeof(float)
                    + (size_t)(BM * HD) * sizeof(float)
                    + (size_t)(2 * GN + 5 * BM) * sizeof(float);
    static int cfg = 0;
    if (!cfg) {
        cudaFuncSetAttribute(pf_attn_mma_i8_kernel<HD, GROUP_BLKS>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
        cfg = 1;
    }
    dim3 grid((n_tokens + BM - 1) / BM, n_q_heads);
    pf_attn_mma_i8_kernel<HD, GROUP_BLKS><<<grid, GROUP_BLKS * 32, sm, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool,
        reinterpret_cast<const __half*>(k_scale), reinterpret_cast<const __half*>(v_scale),
        block_table, reinterpret_cast<__nv_bfloat16*>(attn), n_tokens, n_q_heads, n_kv_heads,
        block_size, max_blocks_per_seq, scale, win_blocks);
    return true;
}


}  // namespace kernels
}  // namespace sparkinfer
