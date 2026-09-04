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
#include "sparkinfer/kernels/deterministic.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdio>
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
    int block_size, int max_blocks_per_seq, float scale, int win_blocks, int q_pos0) {
    // q_pos0 is where this pass's queries START in the sequence. It was implicitly 0 while
    // prefill always ingested [0, N) in a single pass; carrying it lets a long prompt be
    // ingested in windows. Queries and outputs stay addressed by the LOCAL row, while the
    // causal bound and key range below run in sequence coordinates, so a window attends
    // over the whole prefix that precedes it.
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
    const int last_q = q_pos0 + min(qbase + BM - 1, n_tokens - 1);
    int blk_rs = 0;                                   // first token of the recent window
    if (win_blocks > 0) {
        const int n_blk_q = (q_pos0 + qbase + block_size) / block_size;   // constant across the tile
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
                                      (gtok <= q_pos0 + qtok) &&
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
// Shared-memory row padding for the int8 wmma operands. Without it s_qi's row stride is HEAD_DIM
// (256 B = 64 banks) and s_pi's is GN (128 B = 32 banks), both exact multiples of the 128-byte bank
// row, so all 16 rows of a tile start on bank 0 and every ldmatrix replays 16-way. Measured on the
// unpadded kernel at ctx=16384: 2.48e9 shared-load bank conflicts over 3.27e9 wavefronts for
// 4.37e8 instructions -- 7.5 wavefronts per instruction against an ideal of 1, which is why the
// tensor pipe sits at 20.5% while the stalls are mio_throttle and short_scoreboard.
// +16 B keeps the 16-byte alignment int8 ldmatrix requires and takes gcd(stride/4, 32) from 32 to
// 4, i.e. 8 distinct starting banks instead of 1 (16-way -> 2-way).
// SPARKINFER_PREFILL_ATTN_SMEM_PAD=0 restores the packed layout (A/B in ONE binary).
inline int attn_smem_pad() {
    static const int v = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_SMEM_PAD");
        const int x = e ? atoi(e) : 16;
        return (x == 0 || x == 16 || x == 32) ? x : 16;
    }();
    return v;
}

// wmma's 16x16x16 int8 fragment can only reach the K=16 hardware shape: every mma_sync lowers to
// IMMA.16816. sm_120 also has IMMA.16832 -- the same tensor throughput per instruction-pair but
// twice the K per instruction -- so issuing mma.m16n8k32 directly halves the QK MMA instruction
// stream, and its 16x32 A operand halves the Q ldmatrix count with it (one 16-byte-per-lane
// ldmatrix.x4 where wmma needed two 8-byte fragment loads). Same lever, and the same reasoning,
// as the m16n8k32 path in prefill_moe_q.cu: this kernel is issue- and MIO-bound, not
// throughput-bound, so instruction count is what it pays in.
//
// BIT-IDENTICAL: int32 accumulation is exact and order-independent, and the operands are the same
// bytes out of the same s_qi rows and the same K pages -- only the grouping of the adds changes.
//
// Only the QK product converts. PV cannot: mma.m16n8k32 is .row.col, so BOTH operands need the
// reduction axis contiguous, and PV reduces over keys while the V pool is [key][dim] -- V's key
// axis is strided by n_kv_heads*HEAD_DIM. wmma's row_major B, which does not require that, is
// what lets V be fed to the tensor cores straight out of the paged pool.
template <bool B> struct pf_bool { static constexpr bool value = B; };

__device__ __forceinline__ void pf_ldsm_x4(unsigned (&r)[4], unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
template <int V> struct pf_int { static constexpr int value = V; };

__device__ __forceinline__ void pf_mma_16832(int (&d)[4], const unsigned (&a)[4],
                                             unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int HEAD_DIM, int GROUP_BLKS, int RQH, int PLANES = 0>
__global__ __launch_bounds__(GROUP_BLKS * 32, (GROUP_BLKS >= 16 ? 1 : (RQH <= 3 ? 2 : 1))) void pf_attn_mma_gqa_kernel(
    const __nv_bfloat16* __restrict__ q, const signed char* __restrict__ k_pool,
    const signed char* __restrict__ v_pool, const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale, const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ attn, int n_tokens, int n_q_heads, int n_kv_heads,
    int block_size, int max_blocks_per_seq, float scale, int win_blocks, int qld, int pld, int q_pos0) {
    // q_pos0 is where this pass's queries START in the sequence. It was implicitly 0 while
    // prefill always ingested [0, N) in a single pass; carrying it lets a long prompt be
    // ingested in windows. Queries and outputs stay addressed by the LOCAL row, while the
    // causal bound and key range below run in sequence coordinates, so a window attends
    // over the whole prefix that precedes it.
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
    signed char* s_qi = reinterpret_cast<signed char*>(mma_smem);   // [RQH][BM][qld]
    signed char* s_pi = s_qi + (size_t)RQH * BM * qld;               // [RQH][BM][pld]
    // s_o OVERLAYS s_s. The scores are dead by the epilogue -- the last read of s_s is the P
    // quantization inside the softmax section, and the PV mma after it touches only s_pi/s_ps/
    // s_corr -- so the landing zone costs nothing on top of the score buffer it lands in. That
    // is what makes RQH=3 fit: 46,528 B against the 51,200 a second resident block needs, where
    // holding both buffers is 62,912 and caps this kernel at one block per SM.
    // Score-plane row stride. GN alone is a whole multiple of the 128-byte bank row, so all 16
    // rows of a QK output tile start on bank 0 and store_matrix_sync replays 8-way -- the defect
    // attn_smem_pad() fixes for the two ldmatrix operands, on the one buffer it was never applied
    // to. +4 floats puts the 8 rows a single store phase touches on 8 different starting banks
    // (2-way; 1-way is unreachable, 8 row-starts x 4 column-pairs cannot tile 32 banks), and
    // keeps the 16-byte alignment the vectorized softmax load needs, for 768 B.
    // It MUST be a compile-time constant: passing the stride as a kernel argument costs
    // store_matrix_sync its immediate offsets and measured -4.7% on its own, swamping the
    // conflict it removes.
    constexpr int SPLD = GN + 4;
    // How many q-heads the score plane holds AT ONCE. It was always RQH -- QK wrote every head's
    // scores, then one softmax pass drained them all -- and at RQH*BM*GN floats that plane is what
    // caps RQH. Holding fewer heads at a time makes the plane a rolling buffer: QK fills PLANES
    // heads, the softmax drains them into s_pi, and the next group of heads reuses the same
    // floats. That is what lets a block own all six q-heads of a kv-head instead of three, which
    // is the whole point -- see the dispatch note on the RQH=6 tier.
    constexpr int SPL  = (PLANES > 0 && PLANES < RQH) ? PLANES : RQH;
    static_assert(RQH % SPL == 0, "the head loop must tile the score plane exactly");
    constexpr int SBLK = (SPL * BM * SPLD > BM * HEAD_DIM) ? SPL * BM * SPLD : BM * HEAD_DIM;
    float* s_s  = reinterpret_cast<float*>(s_pi + (size_t)RQH * BM * pld); // [SPL][BM][SPLD]
    float* s_o  = s_s;                                               // [BM][HEAD_DIM] epilogue landing
    // The K and V dequant scales arrive as __half and are only ever multiplied into a float, so
    // they stay __half in shared: half the bytes, and -- because the softmax hoists a lane's whole
    // column set into registers -- half the registers that hoist costs.
    __half* s_ks = reinterpret_cast<__half*>(s_s + SBLK);            // [GN] shared
    __half* s_vs = s_ks + GN;                                        // [GN] shared
    float* s_qs = reinterpret_cast<float*>(s_vs + GN);               // [RQH][BM]
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
    // A 16x16 accumulator gives every lane 8 elements spread over exactly TWO query rows (a row's
    // 16 elements sit in 4 lanes, 4 each), so the per-row P quantum and the online-softmax
    // correction the PV epilogue applies are two values per head, not eight. Resolve which of the
    // two each element belongs to once, from idxf, and the epilogue's 2*8*RQH shared loads per
    // group iteration become 2*2*RQH register reads. Nothing here assumes a layout: the rows and
    // the mask are read out of the index fragment itself.
    const int rlo = idxf.x[0] >> 8;
    int rhi = rlo;
    unsigned hi_mask = 0u;
    #pragma unroll
    for (int e = 1; e < 8; e++) {
        const int re = idxf.x[e] >> 8;
        if (re != rlo) { rhi = re; hi_mask |= 1u << e; }
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
            // The softmax scale and log2(e) are per-kernel constants and the Q scale is per row,
            // so all three fold into one number here -- once per row per block instead of once
            // per score. Folding log2(e) in is what lets the online softmax use a bare ex2:
            // __expf(x) is `ex2.approx(x * log2e)`, and every x it is given has already been
            // multiplied by this row scale.
            if (lane == 0) s_qs[h * BM + r] = d * scale * 1.4426950408889634f;
            #pragma unroll
            for (int e = 0; e < QE; e++)
                s_qi[((size_t)h * BM + r) * qld + lane + e * 32] =
                    (signed char)((amax == 0.f) ? 0 : (int)roundf(qv[e] / d));
        }
    }
    if (tid < RQH * BM) { s_m[tid] = -1e30f; s_l[tid] = 0.f; }
    __syncthreads();

    // ldmatrix.x4 lane->address mapping for the m16n8k32 A operand: the four 8x8 b16 matrices are
    // (rows 0-7 | rows 8-15) x (bytes 0-15 | bytes 16-31) of a 16x32 tile, so lane l supplies the
    // start of row (l & 15) at byte column (l >> 4) * 16. Only the k-step offset changes inside
    // the loop, so the whole per-head base is hoisted out of it.
    static_assert(HEAD_DIM % 32 == 0, "m16n8k32 needs whole 32-byte k steps");
    constexpr int KSTEPS = HEAD_DIM / 32;
    unsigned qa_base[RQH];
    #pragma unroll
    for (int h = 0; h < RQH; h++)
        qa_base[h] = (unsigned)__cvta_generic_to_shared(
            s_qi + ((size_t)h * BM + (lane & 15)) * qld + (lane >> 4) * 16);

    const int last_q = q_pos0 + min(qbase + BM - 1, n_tokens - 1);
    int blk_rs = 0;
    if (win_blocks > 0) {
        const int n_blk_q = (q_pos0 + qbase + block_size) / block_size;
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
                s_ks[j] = k_scale[si];
                s_vs[j] = v_scale[si];
            }

            // ---- QK: load each K page fragment ONCE, feed the block's q-heads ----
            // One 16x32 K slice covers two m16n8k32 B operands (keys 0-7 and 8-15). The B operand
            // is n x k col-major, which for K[key][dim] means lane l holds key (l>>2), dims
            // (l&3)*4 .. +3 and the same four 16 bytes later -- two 4-byte loads straight out of
            // the paged pool, exactly as before with no staging.
            //
            // With a rolling score plane the head loop runs SEVERAL times over the same keys, so
            // the K fragments are hoisted into registers first and the pool is read exactly once
            // per block per group however many q-heads the block owns. That is the whole traffic
            // argument -- see the RQH=6 dispatch note. At SPL == RQH there is only one pass and
            // the loads stay inside the k-step loop exactly as they were.
            constexpr int KREG = (SPL == RQH) ? 1 : KSTEPS;
            unsigned kfr[KREG][2][2];
            const signed char* kl = nullptr;
            if (warp < gblk) {
                const int pb = block_table[(k0 / block_size) + warp];
                kl = k_pool + ((size_t)pb * block_size * n_kv_heads + kvh) * HEAD_DIM
                   + (size_t)(lane >> 2) * KVLD + (lane & 3) * 4;
                if constexpr (SPL != RQH) {
                    #pragma unroll
                    for (int kk = 0; kk < KSTEPS; kk++)
                        #pragma unroll
                        for (int t = 0; t < 2; t++) {
                            const signed char* p = kl + (size_t)t * 8 * KVLD + kk * 32;
                            kfr[kk][t][0] = *reinterpret_cast<const unsigned*>(p);
                            kfr[kk][t][1] = *reinterpret_cast<const unsigned*>(p + 16);
                        }
                }
            }
            // H0T is a compile-time head base: qa_base lives in registers, so a runtime index
            // into it would push the whole array to local memory.
            auto qk_group = [&](auto H0T) {
                constexpr int h0 = decltype(H0T)::value;
                if (warp >= gblk) return;
                int acc[SPL][2][4];
                #pragma unroll
                for (int hp = 0; hp < SPL; hp++)
                    #pragma unroll
                    for (int t = 0; t < 2; t++)
                        #pragma unroll
                        for (int e = 0; e < 4; e++) acc[hp][t][e] = 0;
                #pragma unroll
                for (int kk = 0; kk < KSTEPS; kk++) {
                    unsigned bfr[2][2];
                    if constexpr (SPL == RQH) {
                        #pragma unroll
                        for (int t = 0; t < 2; t++) {
                            const signed char* p = kl + (size_t)t * 8 * KVLD + kk * 32;
                            bfr[t][0] = *reinterpret_cast<const unsigned*>(p);
                            bfr[t][1] = *reinterpret_cast<const unsigned*>(p + 16);
                        }
                    } else {
                        #pragma unroll
                        for (int t = 0; t < 2; t++) {
                            bfr[t][0] = kfr[kk][t][0];
                            bfr[t][1] = kfr[kk][t][1];
                        }
                    }
                    #pragma unroll
                    for (int hp = 0; hp < SPL; hp++) {
                        unsigned a[4];
                        pf_ldsm_x4(a, qa_base[h0 + hp] + (unsigned)(kk * 32));
                        #pragma unroll
                        for (int t = 0; t < 2; t++)
                            pf_mma_16832(acc[hp][t], a, bfr[t][0], bfr[t][1]);
                    }
                }
                // m16n8k32's D layout: lane holds rows (l>>2) and (l>>2)+8, columns 2*(l&3) and
                // +1 of each 16x8 tile. The two adjacent columns make each half of it one 8-byte
                // store, so the plane is written in 4 stores per head instead of 8.
                const int dr = lane >> 2, dc = 2 * (lane & 3);
                #pragma unroll
                for (int hp = 0; hp < SPL; hp++) {
                    int* sp = reinterpret_cast<int*>(s_s) + (size_t)hp * BM * SPLD + warp * 16;
                    #pragma unroll
                    for (int t = 0; t < 2; t++) {
                        *reinterpret_cast<int2*>(sp + dr * SPLD + t * 8 + dc) =
                            make_int2(acc[hp][t][0], acc[hp][t][1]);
                        *reinterpret_cast<int2*>(sp + (dr + 8) * SPLD + t * 8 + dc) =
                            make_int2(acc[hp][t][2], acc[hp][t][3]);
                    }
                }
            };
            qk_group(pf_int<0>{});
            __syncthreads();

            // Every mask term is monotone in the key index, so a group that is FULL and whose
            // last key precedes the first query row's causal bound needs no mask at all: the
            // per-column window/causal/bounds test, six integer ops on every one of the
            // BM*RQH*GN scores, disappears. At 256k almost every group is interior -- a query
            // tile at position p has p/GN interior groups and at most one masked one -- so this
            // is the common path, not the rare one.
            const bool grp_full =
                (nk == GN) && (qbase + BM <= n_tokens) && (k0 + GN <= q_pos0 + qbase + 1) &&
                (win_blocks <= 0 || k0 >= blk_rs || k0 + GN <= block_size);

            // ---- online softmax per head; quantize P' ----
            // Column ownership inside the warp is VECTOR, not strided. Lane `lane` used to own
            // the set {lane + 32u}, one column per 32, so every one of this section's shared
            // accesses was a separate 4-byte instruction: GN/32 score loads, GN/32 K-scale
            // loads, GN/32 V-scale loads and GN/32 single-byte P' stores, per query row and per
            // q-head. Giving the lane VW CONSECUTIVE columns instead makes each of those a
            // single 16-byte load or a single packed 4-byte store, so the whole section issues
            // 4x fewer shared instructions for exactly the same bytes. The access stays
            // conflict-free: 32 lanes x 16 B is four 128-byte bank rows, which the hardware
            // already splits into four phases, and the packed byte store lands in one row.
            // The padding note above measures the tensor pipe at 20.5% with the stalls on
            // mio_throttle and short_scoreboard, i.e. this kernel is bound by shared-memory
            // INSTRUCTIONS and their latency, not by bytes or by math -- so removing three of
            // every four is the axis.
            //
            // Only the softmax's `sum` is affected numerically: max and |P'|max are exact under
            // regrouping, the per-row P' quantum pd = pamax/127 is therefore identical, and so
            // are the int8 P' values and the whole PV accumulation. The one changed quantity is
            // the last-bit rounding of the softmax denominator s_l, which each lane now
            // accumulates over a different column subset before the same butterfly.
            constexpr int VW = 4;                       // consecutive columns per lane per step
            constexpr int VU = GN / (32 * VW);          // vector steps per (row, q-head)
            static_assert(GN % (32 * VW) == 0, "GN must cover whole vector steps");
            // The K and V dequant scales for a lane's columns depend only on (lane, step), so
            // all RQH heads -- and every row this warp owns -- read the same values. Hoisting
            // them out of the head loop turns 2*RQH*(BM/WARPS) vector loads per group into 2.
            __half2 ksr[GN / 64], vsr[GN / 64];
            #pragma unroll
            for (int v = 0; v < VU; v++) {
                const int t0 = (v * 32 + lane) * VW;
                const uint2 kw = *reinterpret_cast<const uint2*>(s_ks + t0);
                const uint2 vw = *reinterpret_cast<const uint2*>(s_vs + t0);
                ksr[v * 2 + 0] = *reinterpret_cast<const __half2*>(&kw.x);
                ksr[v * 2 + 1] = *reinterpret_cast<const __half2*>(&kw.y);
                vsr[v * 2 + 0] = *reinterpret_cast<const __half2*>(&vw.x);
                vsr[v * 2 + 1] = *reinterpret_cast<const __half2*>(&vw.y);
            }
            // u is a compile-time constant in every unrolled use below, so the half picked out of
            // the pair resolves statically and the pairs stay in registers.
            #define PF_KS(u) (((u) & 1) ? __high2float(ksr[(u) >> 1]) : __low2float(ksr[(u) >> 1]))
            #define PF_VS(u) (((u) & 1) ? __high2float(vsr[(u) >> 1]) : __low2float(vsr[(u) >> 1]))
            // One q-head, from the plane index it was written to. Splitting the head loop out of
            // the lambda is what lets the caller drain a plane that holds fewer heads than RQH;
            // at SPL == RQH the wrapper below reproduces the loop this used to be.
            // Row stride of the P' plane. The PV mma reads it with leading dimension `pld`, and
            // the bf16 sibling kernel writes `r * pld + t` to match; this one has always written
            // `r * GN + t`, so wherever the two differ (pld = GN + 16 at n_tokens >= 2048) row r
            // is read shifted by 16r columns, and the last row reads past everything any row
            // wrote. The shipped tiers keep that byte for byte -- it is their numerics and there
            // is nothing to gain by moving them -- but the new tier is new code and is written
            // the way the sibling already does it.
            const int pstr = (SPL == RQH) ? GN : pld;
            auto softmax_head = [&](auto FULLT, int h, int hp) {
                constexpr bool FULL = decltype(FULLT)::value;
                {
                    const int* s_si = reinterpret_cast<const int*>(s_s) + (size_t)hp * BM * SPLD;
                    signed char* s_pih = s_pi + (size_t)h * BM * pld;
                    #pragma unroll
                    for (int rr = 0; rr < BM / WARPS; rr++) {
                        const int r = warp * (BM / WARPS) + rr;
                        const int qtok = qbase + r;
                        const float qs = s_qs[h * BM + r];
                        float sc[GN / 32], mx = -1e30f;
                        #pragma unroll
                        for (int v = 0; v < VU; v++) {
                            const int t0 = (v * 32 + lane) * VW;
                            const int4 raw = *reinterpret_cast<const int4*>(s_si + r * SPLD + t0);
                            const int rw[VW] = {raw.x, raw.y, raw.z, raw.w};
                            #pragma unroll
                            for (int j = 0; j < VW; j++) {
                                const int u = v * VW + j;
                                if constexpr (FULL) {
                                    sc[u] = (float)rw[j] * qs * PF_KS(u);
                                } else {
                                    const int t = t0 + j, gtok = k0 + t;
                                    const bool live =
                                        (t < gblk * 16) && (gtok < hi) && (qtok < n_tokens) &&
                                        (gtok <= q_pos0 + qtok) &&
                                        (win_blocks <= 0 || gtok < block_size || gtok >= blk_rs);
                                    sc[u] = live ? (float)rw[j] * qs * PF_KS(u) : -1e30f;
                                }
                                mx = fmaxf(mx, sc[u]);
                            }
                        }
                        #pragma unroll
                        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
                        const float m_old = s_m[h * BM + r], m_new = fmaxf(m_old, mx),
                                    corr = exp2f(m_old - m_new);
                        float sum = 0.f, pamax = 0.f;
                        // P' overwrites the score register it was computed from: sc[u] is read at
                        // the top of the iteration and dead by the bottom, so the separate P'
                        // array the register staging used to need is free.
                        #pragma unroll
                        for (int u = 0; u < GN / 32; u++) {
                            float pv = 0.f;
                            if (FULL || sc[u] > -1e29f) {
                                const float p = exp2f(sc[u] - m_new);
                                // p is an exponential and the V dequant scale is an absmax/127,
                                // so pv is non-negative by construction and |pv| is pv.
                                sum += p; pv = p * PF_VS(u); pamax = fmaxf(pamax, pv);
                            }
                            sc[u] = pv;
                        }
                        #pragma unroll
                        for (int o = 16; o > 0; o >>= 1) {
                            sum   += __shfl_xor_sync(0xffffffffu, sum, o);
                            pamax  = fmaxf(pamax, __shfl_xor_sync(0xffffffffu, pamax, o));
                        }
                        const float pd = pamax / 127.0f;
                        // The quantum is per row, so its reciprocal and its zero test are per row
                        // too: an all-zero row makes ipd zero and every P' falls out as zero,
                        // which is what the per-column ternary used to spell out GN/32 times.
                        const float ipd = (pamax == 0.f) ? 0.f : 127.0f / pamax;
                        if (lane == 0) { s_m[h * BM + r] = m_new; s_l[h * BM + r] = s_l[h * BM + r] * corr + sum;
                                         s_ps[h * BM + r] = pd; s_corr[h * BM + r] = corr; }
                        // gblk*16 is a multiple of 16 and t0 of VW, so a lane's VW columns are
                        // either all live or all past the group -- the per-column bound becomes
                        // one test. __float2int_rn is the single cvt.rni the PV operand wants;
                        // roundf is ties-away-from-zero and lowers to a five-instruction sequence
                        // for a quantum that is already a per-row estimate. The bf16 sibling
                        // kernel has always quantized P' this way.
                        #pragma unroll
                        for (int v = 0; v < VU; v++) {
                            const int t0 = (v * 32 + lane) * VW;
                            if (FULL || t0 < gblk * 16) {
                                unsigned packed = 0u;
                                #pragma unroll
                                for (int j = 0; j < VW; j++) {
                                    const signed char q8 =
                                        (signed char)__float2int_rn(sc[v * VW + j] * ipd);
                                    packed |= ((unsigned)(unsigned char)q8) << (8 * j);
                                }
                                *reinterpret_cast<unsigned*>(s_pih + r * pstr + t0) = packed;
                            }
                        }
                    }
                }
            };
            auto softmax_planes = [&](int h0) {
                #pragma unroll
                for (int hp = 0; hp < SPL; hp++) {
                    if (grp_full) softmax_head(pf_bool<true>{},  h0 + hp, hp);
                    else          softmax_head(pf_bool<false>{}, h0 + hp, hp);
                }
            };
            // QK for head group 0 already ran above; drain it, then -- when the plane holds
            // fewer heads than the block owns -- run the remaining groups QK-then-drain against
            // the same floats. The head base has to be a compile-time constant (qa_base is a
            // register array), so the tail is spelled out rather than looped; RQH is at most 6.
            softmax_planes(0);
            auto roll = [&](auto H0T) {
                __syncthreads();                           // the plane is free again
                qk_group(H0T);
                __syncthreads();
                softmax_planes(decltype(H0T)::value);
            };
            if constexpr (1 * SPL < RQH) roll(pf_int<1 * SPL>{});
            if constexpr (2 * SPL < RQH) roll(pf_int<2 * SPL>{});
            if constexpr (3 * SPL < RQH) roll(pf_int<3 * SPL>{});
            if constexpr (4 * SPL < RQH) roll(pf_int<4 * SPL>{});
            if constexpr (5 * SPL < RQH) roll(pf_int<5 * SPL>{});
            #undef PF_KS
            #undef PF_VS
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
                        load_matrix_sync(af, s_pi + (size_t)h * BM * pld + ks * 16, pld);
                        mma_sync(cf[h], af, bf, cf[h]);
                    }
                }
                #pragma unroll
                for (int h = 0; h < RQH; h++) {
                    const float ps_lo = s_ps[h * BM + rlo],   ps_hi = s_ps[h * BM + rhi];
                    const float cr_lo = s_corr[h * BM + rlo], cr_hi = s_corr[h * BM + rhi];
                    #pragma unroll
                    for (int e = 0; e < 8; e++) {
                        const bool up = (hi_mask >> e) & 1u;
                        ofr[h][dd].x[e] = __fmaf_rn((float)cf[h].x[e], up ? ps_hi : ps_lo,
                                                    __fmul_rn(ofr[h][dd].x[e], up ? cr_hi : cr_lo));
                    }
                }
            }
        }
    };

    if (split_sink) run_range(0, block_size);
    run_range(split_sink ? blk_rs : 0, last_q + 1);

    // ---- epilogue: one head at a time through the shared s_o landing zone ----
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

template <int HD, int GROUP_BLKS, int RQH, int PLANES = 0>
static bool launch_attn_gqa(const void* q, const signed char* k_pool, const signed char* v_pool,
                            const void* k_scale, const void* v_scale, const int* block_table,
                            void* attn, int n_tokens, int n_q_heads, int n_kv_heads,
                            int block_size, int max_blocks_per_seq, float scale, int win_blocks,
                            cudaStream_t stream, int q_pos0) {
    constexpr int BM = 16, GN = GROUP_BLKS * 16;
    // Only at long context. The padding costs ~1 KB of shared memory per block, which is enough to
    // push this kernel past the 2-blocks-per-SM occupancy its __launch_bounds__ asks for at RQH<=2.
    // Where attention dominates (ctx>=2048) trading that occupancy for 8x fewer ldmatrix replays is
    // strongly positive; where it does not, the occupancy is worth more than the conflicts --
    // measured on the Qwen3.6 guard at ctx=512, padding unconditionally cost 5.8% (9993 -> 9410 pp)
    // while the same build at ctx=4096 was +0.3% and Qwen3.8 at ctx=16384 was +5.1%.
    const int pad = (n_tokens >= 2048) ? attn_smem_pad() : 0;
    const int qld = HD + pad, pld = GN + pad;
    // The opt-in below is latched once per device, so it MUST be raised to the largest size any
    // later launch can ask for. Sizing it from THIS call's pad locked in the unpadded size on a
    // small-context first call, after which every padded launch failed and silently fell back --
    // measured as -45% at ctx=16384 and -34% on the Qwen3.6 guard at ctx=4096, with no diagnostic.
    const int qld_max = HD + attn_smem_pad(), pld_max = GN + attn_smem_pad();
    // s_o lands in s_s (dead by the epilogue), so the pair costs the larger of the two.
    constexpr int SPLD = GN + 4;
    constexpr int SPL  = (PLANES > 0 && PLANES < RQH) ? PLANES : RQH;
    constexpr int SBLK = (SPL * BM * SPLD > BM * HD) ? SPL * BM * SPLD : BM * HD;
    const size_t sm = (size_t)RQH * BM * qld                         // s_qi (int8, padded)
                    + (size_t)RQH * BM * pld                         // s_pi (int8, padded)
                    + (size_t)SBLK * sizeof(float)                   // s_s, with s_o overlaid
                    + (size_t)2 * GN * sizeof(__half)                // s_ks, s_vs
                    + (size_t)5 * RQH * BM * sizeof(float);
    // At RQH=4 this is 76,032 B — past the 48 KB default, so the opt-in below is
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
        const size_t sm_max = (size_t)RQH * BM * qld_max
                            + (size_t)RQH * BM * pld_max
                            + (size_t)SBLK * sizeof(float)
                            + (size_t)2 * GN * sizeof(__half)
                            + (size_t)5 * RQH * BM * sizeof(float);
        const cudaError_t ce = cudaFuncSetAttribute(
            pf_attn_mma_gqa_kernel<HD, GROUP_BLKS, RQH, PLANES>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_max);
        if (ce != cudaSuccess && sm_max > 48u * 1024u) return false;  // opt-in refused where required
        cfg[dev] = 1;
    }
    dim3 grid((n_tokens + BM - 1) / BM, n_q_heads / RQH);
    pf_attn_mma_gqa_kernel<HD, GROUP_BLKS, RQH, PLANES><<<grid, GROUP_BLKS * 32, sm, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool,
        reinterpret_cast<const __half*>(k_scale), reinterpret_cast<const __half*>(v_scale),
        block_table, reinterpret_cast<__nv_bfloat16*>(attn), n_tokens, n_q_heads, n_kv_heads,
        block_size, max_blocks_per_seq, scale, win_blocks, qld, pld, q_pos0);
    // A rejected launch (e.g. smem over the device limit) enqueues nothing; peek —
    // rather than get — so a pre-existing sticky error is not silently cleared here.
    const bool ok = cudaPeekAtLastError() == cudaSuccess;
    // SPARKINFER_PREFILL_ATTN_TIER=1 prints the tier that ACTUALLY launched, once per
    // instantiation. A refused wide launch falls through to the next tier silently, which reads
    // exactly like "the change did nothing" -- host-side and once, so it costs nothing.
    static const bool tier_dbg = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_TIER");
        return e && e[0] != '0';
    }();
    static bool announced[kMaxDevices] = {false};
    if (tier_dbg && !announced[dev]) {
        announced[dev] = true;
        fprintf(stderr,
                "[pf-attn-tier] RQH=%d SPL=%d GB=%d GN=%d smem=%zu qld=%d pld=%d n=%d ok=%d\n",
                RQH, SPL, GROUP_BLKS, GN, sm, qld, pld, n_tokens, (int)ok);
    }
    return ok;
}

bool launch_prefill_attn_mma(
    const void* q, const signed char* k_pool, const signed char* v_pool,
    const void* k_scale, const void* v_scale, const int* block_table, void* attn,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, float scale, int win_blocks, cudaStream_t stream,
    int q_pos0) {
    constexpr int HD = 256, GROUP_BLKS = 8, BM = 16;

    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_MMA");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    static const int minctx = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_MMA_MINCTX");
        return e ? atoi(e) : 0;
    }();
    // GQA fusion: one block owns RQH q-heads sharing a kv-head, so each K page / V tile
    // is loaded once and fed RQH mma's instead of being re-read per q-head. RQH=1 disables.
    //
    // DEFAULTS TO 1 (fusion off) IN DETERMINISTIC MODE. The GQA-fused tiers are both
    // nondeterministic and materially INACCURATE once n_tokens passes ~2048 with int8 KV.
    // Measured on an RTX 5090, Qwen3.6-35B-A3B (GQA-6), qwen3_gguf_prefill_check against the
    // token-loop reference, mean KL over 16 teacher-forced positions:
    //
    //     prefix   1500      2000      2100      3000      4000
    //     fused    0.00043   0.00022   0.18672   0.20657   0.23978   <- and varies run to run
    //     RQH=1    ~0.0001   ~0.0001   ~0.0001   ~0.0001   0.00008   <- stable
    //
    // The cliff is exactly at 2048 and it is not the RQH=3 tier's `n_tokens >= 2048` gate:
    // RQH=2 is selected both below and above it and is equally wrong above, so the length
    // dependence lives inside launch_attn_gqa itself. Only RQH=1, which skips the fused family
    // for the per-q-head fallback, is correct there. That is a PRE-EXISTING defect independent of
    // this mode -- it is the default serving path today, and the server enables int8 KV whenever
    // max_seq >= 4096 -- and it is left ON by default here rather than silently changed, because
    // turning it off moves the long-context prefill numbers the eval scores against. It is
    // reported separately; deterministic mode simply refuses to build on top of it.
    static const int gqa_rqh = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_GQA_RQH");
        const int dflt = deterministic_mode() ? 1 : 4;
        const int v = e ? atoi(e) : dflt;
        return (v == 1 || v == 2 || v == 3 || v == 4) ? v : dflt;
    }();
    // KV pages per group iteration for the RQH=3 tier; 8 restores the previous width (A/B in ONE
    // binary). Only 8 and 16 are structurally valid -- GROUP_BLKS is also the warp count, and the
    // kernel needs both BM and HEAD_DIM/16 to divide by it.
    static const int gqa_gb = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_GQA_GB");
        const int v = e ? atoi(e) : 16;
        return (v == 8 || v == 16) ? v : 16;
    }();
    // Smallest KV span (prefix + this pass's queries) that takes the six-head tier below.
    // 0 disables it and restores the RQH=3 tier everywhere.
    static const long wide_minkeys = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_GQA6_MINKEYS");
        const long v = e ? atol(e) : 65536;
        return v < 0 ? 0 : v;
    }();

    if (!enabled || head_dim != HD || block_size != 16 || n_tokens < minctx) return false;
    if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0) return false;

    const int gqa = n_q_heads / n_kv_heads;
    // Each tier reports whether it actually launched; a refusal (opt-in rejected,
    // launch invalid) cascades to the next tier — RQH=2 needs 46,720 B, under the
    // 48 KB default — and finally to the per-q-head kernel below, instead of
    // returning success over an output buffer nothing wrote.
    if (gqa_rqh == 4 && gqa % 4 == 0 &&
        launch_attn_gqa<HD, GROUP_BLKS, 4>(q, k_pool, v_pool, k_scale, v_scale, block_table, attn,
            n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, win_blocks, stream, q_pos0))
        return true;
    // RQH=3 exists for GQA-6 (this checkpoint: 24 q-heads over 4 kv-heads), where 4 does not
    // divide the group and the tier above always falls through to 2. ncu at ctx=16384 puts this
    // kernel at 80.3% of peak L2 throughput with DRAM at 1.7% -- it is bound by re-reading the
    // same K/V pages out of L2, not by the tensor cores (SM 62.7%). RQH=3 loads each K page and V
    // tile once for THREE q-heads instead of two, which is 1/3 off the dominant traffic term, and
    // with s_o overlaid on s_s it still fits the two resident blocks its __launch_bounds__ asks
    // for. Ordered after 4 and before 2 so a group that divides by 4 keeps the wider tier.
    //
    // Long context only, for the same reason the shared-memory padding is: the win is the L2
    // re-read, which only dominates once the window is long. At ctx=128 there is no re-read to
    // save and the wider block costs registers -- measured +1.9% at ctx=16384 against -0.45% on
    // prefill@128, which is a no-regression floor.
    // GROUP_BLKS fixes BOTH the keys retired per group iteration (GN) and the warp count, so
    // doubling it to 16 halves the number of group iterations a block runs -- and with it every
    // per-iteration fixed cost. The one that matters is the online softmax's reductions: each
    // (query row, q-head) pass ends in two warp butterflies (max, then sum+|P'|max) worth ~30
    // instructions regardless of how many keys the group holds, and a block runs BM*RQH of them
    // per iteration. At GN=128 that is 48 passes per 128 keys; at GN=256 it is 48 per 256. The
    // two __syncthreads() per iteration halve per key with it. DPW also drops from 2 to 1, which
    // frees the second set of RQH float accumulators -- enough to retire the 160 bytes of stack
    // the RQH=3 tier was spilling.
    //
    // The cost is shared memory: 78,272 B against 46,528, so one block per SM instead of two. It
    // is a wash on occupancy (16 warps either way, since the block itself doubles to 16 warps)
    // and strongly positive on work per barrier. Measured on an RTX 5090 at the scored
    // target-prefill@256k point, interleaved x2: 2872.08 vs 2609.18 pp/s, +10.08%.
    //
    // Only this tier. RQH=4 at GN=256 needs 103,680 B, past the device's 101,376 B ceiling, and
    // RQH=2 would re-read each K page more often for a wider group -- so GQA-8 checkpoints
    // (Qwen3.6) keep exactly the tier and the numerics they had. Falls through to the GN=128 tier
    // if the wide launch is refused, so a device with a smaller ceiling still gets the old path.
    // Six heads per block, above the context where the KV stream stops fitting in L2.
    //
    // The comment above says this kernel is bound by re-reading the same K/V pages out of L2, and
    // that is why RQH went 2 -> 3. It stopped there because the score plane is RQH*BM*GN floats
    // and a fourth head does not fit beside the GN=256 group. But a block's K/V traffic is
    // (keys) x (blocks that share the kv-head), and that second factor is n_tokens/BM x gqa/RQH --
    // at GQA-6 and RQH=3 every kv-head's stream is pulled twice. Six heads pulls it once.
    //
    // At ctx=32768 six heads is worth almost nothing and costs the GN=256 group: a kv-head's
    // whole K+V there is 16 MB, L2 holds it, and the second read is very nearly free. At 262144
    // that stream is 134 MB against a 96 MB L2 and it is not free at all. Skip-probed on this
    // checkpoint at ctx=262144: stubbing the V pool read is +30.6% on the dimension and stubbing
    // the K pool read is +23.3%, so the two streams are over half of it -- which is what makes
    // halving them worth a tier, and why the tier is gated on length.
    //
    // What made six heads fit is the rolling score plane: SPL=2 keeps only two heads of scores
    // live, so the plane costs 2*BM*GN floats instead of 6 and the whole block is 88,448 B rather
    // than 155,008. The K fragments move into registers across the three head passes so the pool
    // is still read exactly once per block per group -- without that the head loop would re-read
    // K three times and give the traffic straight back.
    //
    // Gated on the KV SPAN, not on n_tokens: a windowed long-prompt pass ingests 16,384 tokens at
    // a time, so n_tokens alone cannot tell a 16k prompt from the tenth window of a 256k one. That
    // keeps every short-context caller -- the 4k/16k prefill dimensions and both cross-model
    // guards -- on exactly the tier, tile shape and arithmetic they have today.
    if (gqa_rqh >= 3 && gqa % 6 == 0 && wide_minkeys > 0 &&
        (long)q_pos0 + n_tokens >= wide_minkeys && n_tokens >= 2048 && gqa_gb >= 16 &&
        launch_attn_gqa<HD, 16, 6, 2>(q, k_pool, v_pool, k_scale, v_scale, block_table, attn,
            n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, win_blocks,
            stream, q_pos0))
        return true;
    if (gqa_rqh >= 3 && gqa % 3 == 0 && n_tokens >= 2048) {
        if (gqa_gb >= 16 &&
            launch_attn_gqa<HD, 16, 3>(q, k_pool, v_pool, k_scale, v_scale, block_table, attn,
                n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, win_blocks, stream, q_pos0))
            return true;
        if (launch_attn_gqa<HD, GROUP_BLKS, 3>(q, k_pool, v_pool, k_scale, v_scale, block_table, attn,
                n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, win_blocks, stream, q_pos0))
            return true;
    }
    if (gqa_rqh >= 2 && gqa % 2 == 0 &&
        launch_attn_gqa<HD, GROUP_BLKS, 2>(q, k_pool, v_pool, k_scale, v_scale, block_table, attn,
            n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, win_blocks, stream, q_pos0))
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
        block_size, max_blocks_per_seq, scale, win_blocks, q_pos0);
    return true;
}


// ============================================================================
// BF16-KV tensor-core prefill attention, hd256 GQA.
//
// The int8 kernels above are unreachable when the KV pool is bf16, and the bf16 pool is exactly
// what the DSpark harness runs (dspark_tau_check pins int8_kv=false) -- so the bf16 branch of
// launch_prefill_attn_bf16_paged fell to pf_attn_bf16_paged_kernel: one 32-thread block per
// (query, q-head) walking the causal history one key at a time out of global memory, with a
// 32-lane shuffle reduction per key. At ctx=32768 over 16 full-attention layers that is 2.1e14
// FLOP issued as scalar FFMA, and it dominates the prompt pass.
//
// This is the same schedule as pf_attn_mma_gqa_kernel with every quantization step deleted:
//   * Q is ALREADY bf16, so there is no per-row quantize and no s_qs.
//   * K/V are ALREADY bf16 in the pool, so there are no dequant scales (s_ks/s_vs) to stage.
//   * P is stored bf16 rather than int8, so there is no per-row P scale (s_ps) and no roundf.
// A KV page is 16 tokens and wmma's tile is 16x16, so a page IS a fragment read straight out of
// the pool with ldm = n_kv_heads*HEAD_DIM -- for QK as a col_major B (which is K^T) and for PV as
// a row_major B. Nothing but Q and P is staged in shared memory.
//
// Accuracy moves the RIGHT way relative to the int8 twin: bf16 P carries 8 mantissa bits against
// int8's 7, and the QK product is exact bf16xbf16->fp32 rather than a symmetric-quantized
// approximation, so the fused-GQA KL cliff documented on launch_prefill_attn_mma has no analogue
// here -- there is no quantization to lose the tail to.
// ============================================================================
// PSPLIT: carry P as a hi+lo bf16 PAIR instead of a single bf16.
// A single bf16 P has ~2^-9 relative error, and because attention is peaked the effective
// number of contributing keys is small, so that error does NOT average away -- it lands at
// ~1e-3 on the output, which is exactly the scale that flips a near-tied argmax. Measured:
// with single-bf16 P the 16k prompt's continuation diverges from main at the FIRST token and
// tau falls 1.6410 -> 1.5059 (ratio 0.918, under the 0.95 floor), even though the kernel is
// self-consistent (RQH=1/2/3 byte-identical). p_lo = p - float(bf16(p)) recovers the dropped
// mantissa, so P*V is evaluated as p_hi*V + p_lo*V: two mma's over the SAME V fragment, ~fp32
// accuracy in P for 1.5x the PV work. `l` is then summed from float(p_hi)+float(p_lo) so the
// denominator matches the numerator exactly rather than being the unrounded fp32 sum.
template <int HEAD_DIM, int GROUP_BLKS, int RQH, bool PSPLIT, bool VINT8 = false>
__global__ __launch_bounds__(GROUP_BLKS * 32) void pf_attn_mma_bf16_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_pool,
    const void* __restrict__ v_pool_raw, const __half* __restrict__ v_scale,
    const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ attn, int n_tokens, int n_q_heads, int n_kv_heads,
    int block_size, int max_blocks_per_seq, float scale, int qld, int pld, int q_pos0) {
    // q_pos0 is where this pass's queries START in the sequence. It was implicitly 0 while
    // prefill always ingested [0, N) in a single pass; carrying it lets a long prompt be
    // ingested in windows. Queries and outputs stay addressed by the LOCAL row, while the
    // causal bound and key range below run in sequence coordinates, so a window attends
    // over the whole prefix that precedes it.
    using namespace nvcuda::wmma;
    constexpr int BM    = 16;                    // query rows per block == wmma M == KV page size
    constexpr int GN    = GROUP_BLKS * 16;       // keys per iteration
    constexpr int KH    = HEAD_DIM / 16;         // k-tiles of the QK contraction
    constexpr int DTILE = HEAD_DIM / 16;         // output d-tiles
    constexpr int WARPS = GROUP_BLKS;
    constexpr int DPW   = DTILE / WARPS;         // output d-tiles per warp
    constexpr int QE    = HEAD_DIM / 32;         // Q elements per lane when staging
    constexpr int RPW   = BM / WARPS;            // softmax rows per warp

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    const int qbase = blockIdx.x * BM;
    const int head0 = blockIdx.y * RQH;                       // first q-head this block owns
    const int gqa   = n_q_heads / n_kv_heads;
    const int kvh   = head0 / gqa;                            // all RQH heads share this kv-head
    const size_t KVLD = (size_t)n_kv_heads * HEAD_DIM;
    const auto* v_pool_bf = reinterpret_cast<const __nv_bfloat16*>(v_pool_raw);
    const auto* v_pool_i8 = reinterpret_cast<const signed char*>(v_pool_raw);

    extern __shared__ char mma_smem_bf[];
    // Only Q and P are staged. s_o overlays s_s for exactly the reason it does in the int8 twin:
    // the scores are dead by the epilogue, so the landing zone is free.
    __nv_bfloat16* s_q = reinterpret_cast<__nv_bfloat16*>(mma_smem_bf);   // [RQH][BM][qld]
    __nv_bfloat16* s_p = s_q + (size_t)RQH * BM * qld;                    // [RQH][BM][pld]
    // Low half of the split P, immediately after the high half; zero-sized when PSPLIT is off.
    __nv_bfloat16* s_p2 = s_p + (size_t)RQH * BM * pld;                   // [RQH][BM][pld]
    constexpr int PPLANES = PSPLIT ? 2 : 1;
    constexpr int SBLK = (RQH * BM * GN > BM * HEAD_DIM) ? RQH * BM * GN : BM * HEAD_DIM;
    float* s_s    = reinterpret_cast<float*>(s_p + (size_t)PPLANES * RQH * BM * pld);  // [RQH][BM][GN]
    float* s_o    = s_s;                                                     // [BM][HEAD_DIM]
    float* s_m    = s_s + SBLK;                                              // [RQH][BM]
    float* s_l    = s_m + RQH * BM;                                          // [RQH][BM]
    float* s_corr = s_l + RQH * BM;                                          // [RQH][BM]
    float* s_ps   = s_corr + RQH * BM;                                        // [RQH][BM], VINT8

    fragment<accumulator, 16, 16, 16, float> ofr[RQH][DPW];
    // Row index of each accumulator lane element. Built with a FLOAT accumulator so the fragment
    // layout is the one the float accumulators below actually use, rather than assuming the int
    // accumulator maps identically. Rows 0..15 are exact in float.
    fragment<accumulator, 16, 16, 16, float> idxf;
    {
        float* tile = s_s + warp * 256;
        for (int i = lane; i < 256; i += 32) tile[i] = (float)(i >> 4);
        __syncwarp();
        load_matrix_sync(idxf, tile, 16, mem_row_major);
    }
    #pragma unroll
    for (int h = 0; h < RQH; h++)
        #pragma unroll
        for (int dd = 0; dd < DPW; dd++) fill_fragment(ofr[h][dd], 0.f);

    // ---- stage Q rows for each of the RQH heads (no quantize: Q is already bf16) ----
    #pragma unroll
    for (int h = 0; h < RQH; h++) {
        const int head = head0 + h;
        #pragma unroll
        for (int rr = 0; rr < RPW; rr++) {
            const int r = warp * RPW + rr;
            const int qtok = qbase + r;
            #pragma unroll
            for (int e = 0; e < QE; e++)
                s_q[((size_t)h * BM + r) * qld + lane + e * 32] =
                    (qtok < n_tokens)
                        ? q[((size_t)qtok * n_q_heads + head) * HEAD_DIM + lane + e * 32]
                        : __float2bfloat16(0.f);
        }
    }
    if (tid < RQH * BM) { s_m[tid] = -1e30f; s_l[tid] = 0.f; }
    __syncthreads();

    const int last_q = q_pos0 + min(qbase + BM - 1, n_tokens - 1);

    for (int k0 = 0; k0 < last_q + 1; k0 += GN) {
        const int nk   = min(GN, last_q + 1 - k0);
        const int gblk = (nk + 15) / 16;

        // VINT8 scales are shared by all BM rows and all RQH query heads, but loading them in the
        // softmax loop repeats the same half load BM*RQH times. Stage one copy per key into the
        // otherwise-unused 8-column padding of the first 2*BM Q rows. This consumes no additional
        // shared memory and preserves the exact __half2float conversion at the consumer.
        if constexpr (VINT8) {
            __half* s_vs_pad = reinterpret_cast<__half*>(s_q);
            for (int t = tid; t < GN; t += blockDim.x) {
                const int pr = t >> 3, pc = HEAD_DIM + (t & 7);
                s_vs_pad[pr * qld + pc] = (t < nk)
                    ? v_scale[(size_t)(k0 + t) * n_kv_heads + kvh]
                    : __float2half(0.f);
            }
        }

        // ---- QK: load each K page fragment ONCE, feed RQH q-heads ----
        if (warp < gblk) {
            const __nv_bfloat16* kb;
            if constexpr (VINT8) {
                kb = k_pool + ((size_t)(k0 + warp * 16) * n_kv_heads + kvh) * HEAD_DIM;
            } else {
                const int pb = block_table[(k0 / block_size) + warp];
                kb = k_pool + ((size_t)pb * block_size * n_kv_heads + kvh) * HEAD_DIM;
            }
            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> af;
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> bf;
            fragment<accumulator, 16, 16, 16, float> cf[RQH];
            #pragma unroll
            for (int h = 0; h < RQH; h++) fill_fragment(cf[h], 0.f);
            #pragma unroll
            for (int ks = 0; ks < KH; ks++) {
                // col_major with ldm=KVLD reads element (d, ktok) from kb[ktok*KVLD + d]: K^T.
                load_matrix_sync(bf, kb + ks * 16, KVLD);        // K fragment: loaded once
                #pragma unroll
                for (int h = 0; h < RQH; h++) {
                    load_matrix_sync(af, s_q + ((size_t)h * BM) * qld + ks * 16, qld);
                    mma_sync(cf[h], af, bf, cf[h]);
                }
            }
            #pragma unroll
            for (int h = 0; h < RQH; h++)
                store_matrix_sync(s_s + (size_t)h * BM * GN + warp * 16, cf[h], GN, mem_row_major);
        }
        __syncthreads();

        // ---- online softmax per head; write P as bf16 (no quantize) ----
        #pragma unroll
        for (int h = 0; h < RQH; h++) {
            const float* s_sh = s_s + (size_t)h * BM * GN;
            __nv_bfloat16* s_ph = s_p + (size_t)h * BM * pld;
            __nv_bfloat16* s_ph2 = s_p2 + (size_t)h * BM * pld;
            signed char* s_pih = reinterpret_cast<signed char*>(s_p) +
                                 (size_t)h * BM * (2 * pld);
            #pragma unroll
            for (int rr = 0; rr < RPW; rr++) {
                const int r = warp * RPW + rr;
                const int qtok = qbase + r;
                float sc[GN / 32], mx = -1e30f;
                #pragma unroll
                for (int u = 0; u < GN / 32; u++) {
                    const int t = lane + u * 32, gtok = k0 + t;
                    const bool live = (t < gblk * 16) && (qtok < n_tokens) && (gtok <= q_pos0 + qtok);
                    sc[u] = live ? s_sh[r * GN + t] * scale : -1e30f;
                    mx = fmaxf(mx, sc[u]);
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
                const float m_old = s_m[h * BM + r], m_new = fmaxf(m_old, mx);
                const float corr = __expf(m_old - m_new);
                float sum = 0.f, pamax = 0.f;
                #pragma unroll
                for (int u = 0; u < GN / 32; u++) {
                    const int t = lane + u * 32;
                    float p = 0.f;
                    if (sc[u] > -1e29f) p = __expf(sc[u] - m_new);
                    if constexpr (VINT8) {
                        const __half* s_vs_pad = reinterpret_cast<const __half*>(s_q);
                        const int pr = t >> 3, pc = HEAD_DIM + (t & 7);
                        const float pv = p * __half2float(s_vs_pad[pr * qld + pc]);
                        sc[u] = pv; pamax = fmaxf(pamax, fabsf(pv)); sum += p;
                    } else {
                        const __nv_bfloat16 hi = __float2bfloat16(p);
                        s_ph[r * pld + t] = hi;
                        if (PSPLIT) {
                            const float ph = __bfloat162float(hi);
                            const __nv_bfloat16 lo = __float2bfloat16(p - ph);
                            s_ph2[r * pld + t] = lo;
                            sum += ph + __bfloat162float(lo);
                        } else sum += __bfloat162float(hi);
                    }
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    sum += __shfl_xor_sync(0xffffffffu, sum, o);
                    if constexpr (VINT8)
                        pamax = fmaxf(pamax, __shfl_xor_sync(0xffffffffu, pamax, o));
                }
                if constexpr (VINT8) {
                    const float pd = pamax * (1.f / 127.f);
                    if (lane == 0) s_ps[h * BM + r] = pd;
                    #pragma unroll
                    for (int u = 0; u < GN / 32; ++u) {
                        const int t = lane + u * 32;
                        s_pih[r * (2 * pld) + t] =
                            (signed char)(pd == 0.f ? 0 : __float2int_rn(sc[u] / pd));
                    }
                }
                if (lane == 0) {
                    s_m[h * BM + r] = m_new;
                    s_l[h * BM + r] = s_l[h * BM + r] * corr + sum;
                    s_corr[h * BM + r] = corr;
                }
            }
        }
        __syncthreads();

        // ---- PV: load each V tile fragment ONCE, feed RQH q-heads ----
        if constexpr (!VINT8) {
        #pragma unroll
        for (int dd = 0; dd < DPW; dd++) {
            const int dt = warp * DPW + dd;
            fragment<accumulator, 16, 16, 16, float> cf[RQH];
            #pragma unroll
            for (int h = 0; h < RQH; h++) fill_fragment(cf[h], 0.f);
            for (int ks = 0; ks < gblk; ks++) {
                const int pb = block_table[(k0 / block_size) + ks];
                const __nv_bfloat16* vb =
                    v_pool_bf + ((size_t)pb * block_size * n_kv_heads + kvh) * HEAD_DIM + dt * 16;
                fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> af;
                fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> bf;
                load_matrix_sync(bf, vb, KVLD);                  // V fragment: loaded once
                #pragma unroll
                for (int h = 0; h < RQH; h++) {
                    load_matrix_sync(af, s_p + (size_t)h * BM * pld + ks * 16, pld);
                    mma_sync(cf[h], af, bf, cf[h]);
                    if (PSPLIT) {   // same V fragment, second pass for the recovered mantissa
                        load_matrix_sync(af, s_p2 + (size_t)h * BM * pld + ks * 16, pld);
                        mma_sync(cf[h], af, bf, cf[h]);
                    }
                }
            }
            #pragma unroll
            for (int h = 0; h < RQH; h++)
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    const int r = (int)idxf.x[e];
                    ofr[h][dd].x[e] = __fmaf_rn(ofr[h][dd].x[e], s_corr[h * BM + r], cf[h].x[e]);
                }
        }
        } else {
        #pragma unroll
        for (int dd = 0; dd < DPW; dd++) {
            const int dt = warp * DPW + dd;
            fragment<accumulator, 16, 16, 16, int> cf[RQH];
            #pragma unroll
            for (int h = 0; h < RQH; h++) fill_fragment(cf[h], 0);
            for (int ks = 0; ks < gblk; ks++) {
                const signed char* vb = v_pool_i8 +
                    ((size_t)(k0 + ks * 16) * n_kv_heads + kvh) * HEAD_DIM + dt * 16;
                fragment<matrix_a, 16, 16, 16, signed char, row_major> af;
                fragment<matrix_b, 16, 16, 16, signed char, row_major> bf;
                load_matrix_sync(bf, vb, KVLD);
                #pragma unroll
                for (int h = 0; h < RQH; h++) {
                    load_matrix_sync(af, reinterpret_cast<signed char*>(s_p) +
                                         (size_t)h * BM * (2 * pld) + ks * 16,
                                     2 * pld);
                    mma_sync(cf[h], af, bf, cf[h]);
                }
            }
            #pragma unroll
            for (int h = 0; h < RQH; h++)
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    const int r = (int)idxf.x[e];
                    ofr[h][dd].x[e] = __fmaf_rn(
                        ofr[h][dd].x[e], s_corr[h * BM + r],
                        (float)cf[h].x[e] * s_ps[h * BM + r]);
                }
        }
        }
        __syncthreads();   // s_p is rewritten by the next iteration's softmax
    }

    // ---- epilogue: one head at a time through the shared s_o landing zone ----
    #pragma unroll
    for (int h = 0; h < RQH; h++) {
        #pragma unroll
        for (int dd = 0; dd < DPW; dd++)
            store_matrix_sync(s_o + (warp * DPW + dd) * 16, ofr[h][dd], HEAD_DIM, mem_row_major);
        __syncthreads();
        const int head = head0 + h;
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

template <int HD, int GROUP_BLKS, int RQH, bool PSPLIT, bool VINT8 = false>
static bool launch_attn_bf16_gqa(const void* q, const void* k_pool, const void* v_pool,
                                 const void* v_scale, const int* block_table, void* attn, int n_tokens,
                                 int n_q_heads, int n_kv_heads, int block_size,
                                 int max_blocks_per_seq, float scale, cudaStream_t stream,
                                 int q_pos0) {
    constexpr int BM = 16, GN = GROUP_BLKS * 16;
    // Same bank-conflict argument as attn_smem_pad() above, in bf16 elements: an unpadded row
    // stride of HD (512 B) or GN (256 B) is a whole multiple of the 128-byte bank row, so all 16
    // rows of a tile start on bank 0 and every ldmatrix replays 16-way. +8 bf16 elements is 16 B,
    // which keeps the alignment ldmatrix requires.
    const int pad = attn_smem_pad() ? 8 : 0;
    const int qld = HD + pad, pld = GN + pad;
    constexpr int SBLK = (RQH * BM * GN > BM * HD) ? RQH * BM * GN : BM * HD;
    const size_t sm = (size_t)RQH * BM * qld * sizeof(__nv_bfloat16)
                    + (size_t)(PSPLIT ? 2 : 1) * RQH * BM * pld * sizeof(__nv_bfloat16)
                    + (size_t)(SBLK + (VINT8 ? 4 : 3) * RQH * BM) * sizeof(float);
    static int cfg = 0;
    if (!cfg) {
        if (cudaFuncSetAttribute(pf_attn_mma_bf16_kernel<HD, GROUP_BLKS, RQH, PSPLIT, VINT8>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) != cudaSuccess)
            return false;
        cfg = 1;
    }
    dim3 grid((n_tokens + BM - 1) / BM, n_q_heads / RQH);
    pf_attn_mma_bf16_kernel<HD, GROUP_BLKS, RQH, PSPLIT, VINT8><<<grid, GROUP_BLKS * 32, sm, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k_pool),
        v_pool, reinterpret_cast<const __half*>(v_scale), block_table,
        reinterpret_cast<__nv_bfloat16*>(attn), n_tokens, n_q_heads, n_kv_heads,
        block_size, max_blocks_per_seq, scale, qld, pld, q_pos0);
    // A rejected launch (e.g. smem over the device limit) enqueues nothing; peek --
    // rather than get -- so a pre-existing sticky error is not silently cleared here.
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launch_prefill_attn_mma_bf16(
    const void* q, const void* k_pool, const void* v_pool, const int* block_table, void* attn,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, float scale, cudaStream_t stream, int q_pos0) {
    constexpr int HD = 256;
    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_BF16_MMA");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    // One block owns RQH q-heads sharing a kv-head, so each K page / V tile is read once and fed
    // RQH mma's instead of being re-read per q-head. This checkpoint is GQA-6, so 3 divides the
    // group and 2 is the fallback; RQH=1 turns the fusion off.
    static const int rqh_env = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_BF16_RQH");
        const int v = e ? atoi(e) : 3;
        return (v == 1 || v == 2 || v == 3) ? v : 3;
    }();
    if (!enabled || head_dim != HD || block_size != 16) return false;
    if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0) return false;
    const int gqa = n_q_heads / n_kv_heads;

    // At 16k and above the wider 16-page group is the throughput shape, but its RQH=3 split-P
    // footprint is larger than the device's dynamic-smem ceiling. The single-P tier fits and is
    // lossless; shorter accuracy/decode paths stay byte-for-byte on the 8-page split-P tier.
    // RTX 5090, scored 16k prompt with 128 generated tokens: 9875.40 -> 11309.30 pp/s (+14.52%),
    // 128/128 lossless in three repeats and mean acceptance 1.4066 -> 1.4222.
    static const int psplit_env = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_BF16_PSPLIT");
        return e ? ((e[0] == '0') ? 0 : 1) : -1;
    }();
    const int psplit = psplit_env >= 0 ? psplit_env : (n_tokens < 16384 ? 1 : 0);
    static const int group_blks_env = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_BF16_GROUP_BLKS");
        return e ? ((atoi(e) == 16) ? 16 : 8) : 0;
    }();
    const int group_blks = group_blks_env ? group_blks_env : (n_tokens >= 16384 ? 16 : 8);
    if (group_blks == 16) {
        if (!psplit && rqh_env >= 3 && gqa % 3 == 0 &&
            launch_attn_bf16_gqa<HD, 16, 3, false>(
                q, k_pool, v_pool, nullptr, block_table, attn, n_tokens, n_q_heads, n_kv_heads,
                block_size, max_blocks_per_seq, scale, stream, q_pos0))
            return true;
        if (rqh_env >= 2 && gqa % 2 == 0) {
            if (psplit ? launch_attn_bf16_gqa<HD, 16, 2, true>(
                             q, k_pool, v_pool, nullptr, block_table, attn, n_tokens, n_q_heads,
                             n_kv_heads, block_size, max_blocks_per_seq, scale, stream, q_pos0)
                       : launch_attn_bf16_gqa<HD, 16, 2, false>(
                             q, k_pool, v_pool, nullptr, block_table, attn, n_tokens, n_q_heads,
                             n_kv_heads, block_size, max_blocks_per_seq, scale, stream, q_pos0))
                return true;
        }
        return psplit ? launch_attn_bf16_gqa<HD, 16, 1, true>(
                            q, k_pool, v_pool, nullptr, block_table, attn, n_tokens, n_q_heads,
                            n_kv_heads, block_size, max_blocks_per_seq, scale, stream, q_pos0)
                      : launch_attn_bf16_gqa<HD, 16, 1, false>(
                            q, k_pool, v_pool, nullptr, block_table, attn, n_tokens, n_q_heads,
                            n_kv_heads, block_size, max_blocks_per_seq, scale, stream, q_pos0);
    }
#define SI_MMA_BF16_TRY(RQH_)                                                                     \
    (psplit ? launch_attn_bf16_gqa<HD, 8, RQH_, true>(q, k_pool, v_pool, nullptr, block_table, attn,       \
                  n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, stream, q_pos0) \
            : launch_attn_bf16_gqa<HD, 8, RQH_, false>(q, k_pool, v_pool, nullptr, block_table, attn,      \
                  n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, stream, q_pos0))
    if (rqh_env >= 3 && gqa % 3 == 0 && SI_MMA_BF16_TRY(3)) return true;
    if (rqh_env >= 2 && gqa % 2 == 0 && SI_MMA_BF16_TRY(2)) return true;
    return SI_MMA_BF16_TRY(1);
#undef SI_MMA_BF16_TRY
}

bool launch_prefill_attn_mma_bf16_vi8(
    const void* q, const void* k_pool, const signed char* v_i8, const void* v_scale,
    const int* block_table, void* attn, int n_tokens, int n_q_heads, int n_kv_heads,
    int head_dim, int block_size, int max_blocks_per_seq, float scale, cudaStream_t stream,
    int q_pos0) {
    if (head_dim != 256 || block_size != 16 || n_tokens < 32768 ||
        n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0 ||
        (n_q_heads / n_kv_heads) % 3 != 0)
        return false;
    return launch_attn_bf16_gqa<256, 16, 3, false, true>(
        q, k_pool, v_i8, v_scale, block_table, attn, n_tokens, n_q_heads, n_kv_heads,
        block_size, max_blocks_per_seq, scale, stream, q_pos0);
}

}  // namespace kernels
}  // namespace sparkinfer
