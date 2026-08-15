// ============================================================================
// Windowed / tiled prefill attention for Qwythos (Qwen3.5) long context.
//
// The batched prompt prefill (PR #398, @fansilas) computes FULL O(N^2) attention
// over the prompt for the hd256 full-attention layers. Decode already restricts
// those same layers to an attention sink + sliding window (StreamingLLM: sink
// block 0 + the last `win_blocks` KV blocks) via the merged sparse-KV path
// (#379). This translation unit provides the drop-in kernels that apply the SAME
// window to the batched prefill attention, turning O(N^2) prompt attention into
// O(N * window) at long context, with byte-identical online-softmax math.
//
// It is intentionally self-contained: the kernels take the same raw paged-KV
// pointers #398's launcher already has (int8 K/V pools + per-slot fp16 scales +
// block table), define their own tiny device helpers, and expose ONE launcher
// (`launch_prefill_attn_windowed`) returning true when it handled the call. The
// integration once #398 lands is a single guard at the top of #398's
// `launch_prefill_attn_int8_paged`:
//
//     if (launch_prefill_attn_windowed(q, k_pool, v_pool, k_scale, v_scale,
//             block_table, attn, n_tokens, n_q_heads, n_kv_heads, head_dim,
//             block_size, max_blocks_per_seq, scale, stream)) return;
//
// DRAFT: depends on #398's batched prefill (paged int8 KV) for a call site.
// ============================================================================
#include "sparkinfer/kernels/prefill_attn_window.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>            // bf16 tensor-core path for Muse Glimmer's hd128 windowed attention

#include <cstdlib>
#include <type_traits>

namespace sparkinfer {
namespace kernels {

namespace {

// bf16 -> fp32 load helper (same value the naive prefill kernel uses).
__device__ __forceinline__ float win_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }

// full-warp (32-lane) reduction of a per-lane partial dot into every lane.
__device__ __forceinline__ float win_warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// ----------------------------------------------------------------------------
// TILED causal attention over the paged int8 KV pool. A block owns TQ consecutive
// query tokens (one warp each) for one q-head; the block cooperatively loads each
// KV tile (TK positions) into shared memory ONCE and every query-warp reuses it,
// cutting KV HBM reads ~TQ-fold vs a one-warp-per-query kernel. The per-key
// online-softmax math is identical to a naive per-query kernel, so the result is
// numerically the same — only the memory traffic changes.
// ----------------------------------------------------------------------------
template <int HEAD_DIM, int TQ, int TK>
__global__ void win_prefill_tiled_kernel(
    const __nv_bfloat16* __restrict__ q, const signed char* __restrict__ k_pool,
    const signed char* __restrict__ v_pool, const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale, const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ attn, int n_tokens, int n_q_heads, int n_kv_heads,
    int block_size, int max_blocks_per_seq, float scale) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = blockIdx.y;
    const int qbase = blockIdx.x * TQ;
    const int qtok = qbase + warp;
    const int kv_head = head / (n_q_heads / n_kv_heads);
    const bool active = (qtok < n_tokens) && (head < n_q_heads);

    // K/V tiles staged as half (2B): the values are dequantized from int8, so fp16 storage is
    // lossless vs what's stored, and it halves both the smem footprint (unlocking a 2nd block/SM)
    // and the smem bandwidth that limits this kernel. Dot products still accumulate in fp32.
    extern __shared__ __half smem[];
    __half* sK = smem;                      // [TK * HEAD_DIM]
    __half* sV = smem + TK * HEAD_DIM;      // [TK * HEAD_DIM]

    float q_reg[ELEMS];
    if (active) {
        const size_t q_off = ((size_t)qtok * n_q_heads + head) * HEAD_DIM;
#pragma unroll
        for (int e = 0; e < ELEMS; e++) q_reg[e] = win_to_f(q[q_off + lane + e * 32]);
    }
    float m = -1e30f, l = 0.f, acc[ELEMS];
#pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    // last key any query in this block needs (causal): the block's last valid query pos
    const int last_q = min(qbase + TQ - 1, n_tokens - 1);
    for (int k0 = 0; k0 <= last_q; k0 += TK) {
        const int tk = min(TK, last_q + 1 - k0);
        // cooperative dequant load of TK positions (k+v) into smem for this kv_head
        for (int idx = threadIdx.x; idx < tk * HEAD_DIM; idx += blockDim.x) {
            const int kk = idx / HEAD_DIM, d = idx - kk * HEAD_DIM;
            const int kpos = k0 + kk;
            const int blk = kpos / block_size, within = kpos - blk * block_size;
            const int phys = block_table[blk];
            const size_t ckt = (size_t)phys * block_size + within;
            const size_t off = (ckt * n_kv_heads + kv_head) * HEAD_DIM + d;
            const float ksc = __half2float(k_scale[ckt * n_kv_heads + kv_head]);
            const float vsc = __half2float(v_scale[ckt * n_kv_heads + kv_head]);
            sK[idx] = __float2half((float)k_pool[off] * ksc);
            sV[idx] = __float2half((float)v_pool[off] * vsc);
        }
        __syncthreads();
        if (active) {
            const int klim = min(k0 + tk, qtok + 1);   // causal cutoff for this warp's query
            for (int kpos = k0; kpos < klim; kpos++) {
                const int kk = kpos - k0;
                const __half* krow = sK + (size_t)kk * HEAD_DIM;
                float partial = 0.f;
#pragma unroll
                for (int e = 0; e < ELEMS; e++) partial += q_reg[e] * __half2float(krow[lane + e * 32]);
                const float score = win_warp_sum(partial) * scale;
                const float m_new = fmaxf(m, score);
                const float corr = __expf(m - m_new);
                const float p = __expf(score - m_new);
                l = l * corr + p;
                const __half* vrow = sV + (size_t)kk * HEAD_DIM;
#pragma unroll
                for (int e = 0; e < ELEMS; e++) acc[e] = acc[e] * corr + p * __half2float(vrow[lane + e * 32]);
                m = m_new;
            }
        }
        __syncthreads();
    }
    if (active) {
        const size_t q_off = ((size_t)qtok * n_q_heads + head) * HEAD_DIM;
        const float inv = (l > 0.f) ? (1.f / l) : 0.f;
#pragma unroll
        for (int e = 0; e < ELEMS; e++) attn[q_off + lane + e * 32] = __float2bfloat16(acc[e] * inv);
    }
}

// ----------------------------------------------------------------------------
// WINDOWED tiled prefill attention: each query attends to the attention sink
// (block 0) + the last `win_blocks` KV blocks, matching the merged sparse-KV
// decode selection (StreamingLLM). Turns O(N^2) prompt attention into
// O(N * window) at long context. Same online-softmax math + smem KV-tile reuse
// as the tiled kernel; win_blocks <= 0 => full attention.
// ----------------------------------------------------------------------------
template <int HEAD_DIM, int TQ, int TK>
__global__ void win_prefill_windowed_kernel(
    const __nv_bfloat16* __restrict__ q, const signed char* __restrict__ k_pool,
    const signed char* __restrict__ v_pool, const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale, const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ attn, int n_tokens, int n_q_heads, int n_kv_heads,
    int block_size, int max_blocks_per_seq, float scale, int win_blocks) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = blockIdx.y;
    const int qbase = blockIdx.x * TQ;
    const int qtok = qbase + warp;
    const int kv_head = head / (n_q_heads / n_kv_heads);
    const bool active = (qtok < n_tokens) && (head < n_q_heads);

    // recent-window start (block-aligned), per query, matching the decode selection:
    // n_blk_q = blocks up to qtok; recent_start_blk = (win >= n_blk_q-1) ? 1 : n_blk_q-win.
    auto win_start = [&](int t) -> int {
        const int n_blk_q = (t + block_size) / block_size;          // (t+1+bs-1)/bs
        const int rsb = (win_blocks >= n_blk_q - 1) ? 1 : (n_blk_q - win_blocks);
        return rsb * block_size;                                    // first token of recent window
    };
    const int my_rs = active ? win_start(qtok) : 0;                 // this query's recent-window start

    // K/V tiles staged as half (2B): dequantized-from-int8 values store losslessly in fp16, which
    // halves the smem footprint (2nd block/SM) and the smem bandwidth bottleneck. Dot in fp32.
    extern __shared__ __half smem[];
    __half* sK = smem;
    __half* sV = smem + TK * HEAD_DIM;

    float q_reg[ELEMS];
    if (active) {
        const size_t q_off = ((size_t)qtok * n_q_heads + head) * HEAD_DIM;
#pragma unroll
        for (int e = 0; e < ELEMS; e++) q_reg[e] = win_to_f(q[q_off + lane + e * 32]);
    }
    float m = -1e30f, l = 0.f, acc[ELEMS];
#pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    const int last_q = min(qbase + TQ - 1, n_tokens - 1);
    // block-wide recent-window start (earliest query qbase => widest union of windows)
    const int blk_rs = win_start(qbase);

    // process a contiguous key range [lo, hi) in TK tiles with per-query sink/window mask
    auto run_range = [&](int lo, int hi) {
        for (int k0 = lo; k0 < hi; k0 += TK) {
            const int tk = min(TK, hi - k0);
            for (int idx = threadIdx.x; idx < tk * HEAD_DIM; idx += blockDim.x) {
                const int kk = idx / HEAD_DIM, d = idx - kk * HEAD_DIM;
                const int kpos = k0 + kk;
                const int blk = kpos / block_size, within = kpos - blk * block_size;
                const int phys = block_table[blk];
                const size_t ckt = (size_t)phys * block_size + within;
                const size_t off = (ckt * n_kv_heads + kv_head) * HEAD_DIM + d;
                sK[idx] = __float2half((float)k_pool[off] * __half2float(k_scale[ckt * n_kv_heads + kv_head]));
                sV[idx] = __float2half((float)v_pool[off] * __half2float(v_scale[ckt * n_kv_heads + kv_head]));
            }
            __syncthreads();
            if (active) {
                for (int kpos = k0; kpos < k0 + tk; kpos++) {
                    // per-query membership: sink (block 0) OR recent window [my_rs, qtok]
                    const bool insink = kpos < block_size;
                    const bool inwin = (kpos >= my_rs) && (kpos <= qtok);
                    if (!insink && !inwin) continue;
                    const int kk = kpos - k0;
                    const __half* krow = sK + (size_t)kk * HEAD_DIM;
                    float partial = 0.f;
#pragma unroll
                    for (int e = 0; e < ELEMS; e++) partial += q_reg[e] * __half2float(krow[lane + e * 32]);
                    const float score = win_warp_sum(partial) * scale;
                    const float m_new = fmaxf(m, score);
                    const float corr = __expf(m - m_new);
                    const float p = __expf(score - m_new);
                    l = l * corr + p;
                    const __half* vrow = sV + (size_t)kk * HEAD_DIM;
#pragma unroll
                    for (int e = 0; e < ELEMS; e++) acc[e] = acc[e] * corr + p * __half2float(vrow[lane + e * 32]);
                    m = m_new;
                }
            }
            __syncthreads();
        }
    };

    // sink range: block 0, only if the block-wide window doesn't already start there
    if (blk_rs > block_size) run_range(0, block_size);
    // recent-window range: [blk_rs, last_q] (each query masks to its own [my_rs, qtok])
    const int wlo = (blk_rs > block_size) ? blk_rs : 0;
    run_range(wlo, last_q + 1);

    if (active) {
        const size_t q_off = ((size_t)qtok * n_q_heads + head) * HEAD_DIM;
        const float inv = (l > 0.f) ? (1.f / l) : 0.f;
#pragma unroll
        for (int e = 0; e < ELEMS; e++) attn[q_off + lane + e * 32] = __float2bfloat16(acc[e] * inv);
    }
}

// ----------------------------------------------------------------------------
// LANE-PARALLEL tiled prefill attention (windowed and full in one template).
//
// The kernels above walk keys one at a time per query-warp: every key pays a
// 5-shuffle warp reduction plus two dependent expf's inside the online-softmax
// carry chain, so the warp is latency-bound (~50 cycles/key) and the fp32 units
// sit idle. Here each of the 32 lanes owns ONE key of the TK=32 tile:
//   - scoring: lane j computes the full 256-dim dot q.K[j] from smem (K rows
//     padded to 260 floats -> lane-strided reads hit 32 distinct banks, float4
//     aligned), so 32 keys score in parallel with zero shuffles;
//   - softmax: ONE max-reduce + ONE sum-reduce per tile (amortized 10 shuffles
//     per 32 keys instead of 160), one expf per lane per tile;
//   - AV: each lane owns 8 contiguous output dims (float4-friendly sV reads,
//     coalesced attn writes); p_j broadcast by __shfl_sync is warp-uniform, so
//     all-masked keys are skipped without divergence.
// The sink+window selection, causal mask, and paged int8-KV dequant are
// identical to the kernels above; only the schedule (and thus fp32 rounding
// order) changes. SPARKINFER_PREFILL_ATTN_LANEPAR=0 restores the old kernels.
// ----------------------------------------------------------------------------
template <int HEAD_DIM, int QPW, int TK, bool WINDOWED>
__global__ void win_prefill_lanepar_kernel(
    const __nv_bfloat16* __restrict__ q, const signed char* __restrict__ k_pool,
    const signed char* __restrict__ v_pool, const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale, const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ attn, int n_tokens, int n_q_heads, int n_kv_heads,
    int block_size, int max_blocks_per_seq, float scale, int win_blocks) {
    constexpr int ELEMS = HEAD_DIM / 32;
    constexpr int KSTRIDE = HEAD_DIM + 4;   // +4 floats: lane-strided rows hit 32 banks, 16B-aligned
    constexpr int NWARP = 16;
    constexpr int TQ = NWARP * QPW;         // queries per block
    static_assert(TK == 32, "one key per lane");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = blockIdx.y;
    const int qbase = blockIdx.x * TQ;
    const int q0 = qbase + warp * QPW;      // this warp's first query
    const int kv_head = head / (n_q_heads / n_kv_heads);
    const bool active = (q0 < n_tokens) && (head < n_q_heads);

    auto win_start = [&](int t) -> int {
        const int n_blk_q = (t + block_size) / block_size;
        const int rsb = (win_blocks >= n_blk_q - 1) ? 1 : (n_blk_q - win_blocks);
        return rsb * block_size;
    };

    extern __shared__ float smem_lp[];   // distinct name: the kernels above declare __half smem[]
    float* sK = smem_lp;                                           // [TK][KSTRIDE] fp32
    float* sV = sK + TK * KSTRIDE;                                 // [TK][HEAD_DIM] fp32
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(sV + TK * HEAD_DIM);  // [TQ][HEAD_DIM] bf16

    // stage the block's query rows verbatim (q is bf16 in global -> no precision change)
    for (int idx = threadIdx.x; idx < TQ * (HEAD_DIM / 2); idx += blockDim.x) {
        const int qq = idx / (HEAD_DIM / 2), d2 = idx - qq * (HEAD_DIM / 2);
        const int qt = qbase + qq;
        __nv_bfloat162 v2 = {__float2bfloat16(0.f), __float2bfloat16(0.f)};
        if (qt < n_tokens && head < n_q_heads) {
            const size_t q_off = ((size_t)qt * n_q_heads + head) * HEAD_DIM;
            v2 = *reinterpret_cast<const __nv_bfloat162*>(q + q_off + d2 * 2);
        }
        *reinterpret_cast<__nv_bfloat162*>(sQ + (size_t)qq * HEAD_DIM + d2 * 2) = v2;
    }

    int qtok[QPW], my_rs[QPW];
    float m[QPW], l[QPW], acc[QPW][ELEMS];
#pragma unroll
    for (int i = 0; i < QPW; i++) {
        qtok[i] = q0 + i;
        my_rs[i] = (WINDOWED && active) ? win_start(min(qtok[i], n_tokens - 1)) : 0;
        m[i] = -1e30f; l[i] = 0.f;
#pragma unroll
        for (int e = 0; e < ELEMS; e++) acc[i][e] = 0.f;
    }

    const int last_q = min(qbase + TQ - 1, n_tokens - 1);
    const int blk_rs = WINDOWED ? win_start(qbase) : 0;

    auto run_range = [&](int lo, int hi) {
        for (int k0 = lo; k0 < hi; k0 += TK) {
            const int tk = min(TK, hi - k0);
            // cooperative dequant KV load, char4 global reads -> float4 smem writes
            for (int idx = threadIdx.x; idx < tk * (HEAD_DIM / 4); idx += blockDim.x) {
                const int kk = idx / (HEAD_DIM / 4), d4 = idx - kk * (HEAD_DIM / 4);
                const int kpos = k0 + kk;
                const int blk = kpos / block_size, within = kpos - blk * block_size;
                const int phys = block_table[blk];
                const size_t ckt = (size_t)phys * block_size + within;
                const size_t off = (ckt * n_kv_heads + kv_head) * HEAD_DIM + (size_t)d4 * 4;
                const float ksc = __half2float(k_scale[ckt * n_kv_heads + kv_head]);
                const float vsc = __half2float(v_scale[ckt * n_kv_heads + kv_head]);
                const char4 k4 = *reinterpret_cast<const char4*>(k_pool + off);
                const char4 v4 = *reinterpret_cast<const char4*>(v_pool + off);
                *reinterpret_cast<float4*>(sK + kk * KSTRIDE + d4 * 4) =
                    make_float4(k4.x * ksc, k4.y * ksc, k4.z * ksc, k4.w * ksc);
                *reinterpret_cast<float4*>(sV + kk * HEAD_DIM + d4 * 4) =
                    make_float4(v4.x * vsc, v4.y * vsc, v4.z * vsc, v4.w * vsc);
            }
            __syncthreads();
            if (active && k0 <= qtok[QPW - 1]) {
                // one key per lane; each K row read feeds all QPW query dots
                const int kpos = k0 + lane;
                bool live = lane < tk;
                bool in[QPW];
#pragma unroll
                for (int i = 0; i < QPW; i++) {
                    if (WINDOWED) {
                        const bool insink = kpos < block_size;
                        const bool inwin = (kpos >= my_rs[i]) && (kpos <= qtok[i]);
                        in[i] = live && (insink || inwin) && (qtok[i] < n_tokens);
                    } else {
                        in[i] = live && (kpos <= qtok[i]) && (qtok[i] < n_tokens);
                    }
                }
                float s[QPW];
                {
                    const float* krow = sK + lane * KSTRIDE;
                    const __nv_bfloat16* qrow = sQ + (size_t)(warp * QPW) * HEAD_DIM;
                    float sa[QPW], sb[QPW];
#pragma unroll
                    for (int i = 0; i < QPW; i++) { sa[i] = 0.f; sb[i] = 0.f; }
#pragma unroll
                    for (int d = 0; d < HEAD_DIM; d += 4) {
                        const float4 kv = *reinterpret_cast<const float4*>(krow + d);  // conflict-free (phase-split)
#pragma unroll
                        for (int i = 0; i < QPW; i++) {
                            const __nv_bfloat162 qa = *reinterpret_cast<const __nv_bfloat162*>(
                                qrow + (size_t)i * HEAD_DIM + d);           // broadcast reads
                            const __nv_bfloat162 qb = *reinterpret_cast<const __nv_bfloat162*>(
                                qrow + (size_t)i * HEAD_DIM + d + 2);
                            sa[i] += __bfloat162float(qa.x) * kv.x + __bfloat162float(qa.y) * kv.y;
                            sb[i] += __bfloat162float(qb.x) * kv.z + __bfloat162float(qb.y) * kv.w;
                        }
                    }
#pragma unroll
                    for (int i = 0; i < QPW; i++) s[i] = in[i] ? (sa[i] + sb[i]) * scale : -1e30f;
                }
#pragma unroll
                for (int i = 0; i < QPW; i++) {
                    float tmax = s[i];
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
                    if (tmax <= -1e29f) { s[i] = 0.f; continue; }   // no live key for query i
                    const float m_new = fmaxf(m[i], tmax);
                    const float corr = __expf(m[i] - m_new);
                    const float p = in[i] ? __expf(s[i] - m_new) : 0.f;
                    float tl = p;
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                        tl += __shfl_xor_sync(0xffffffffu, tl, o);
                    l[i] = l[i] * corr + tl;
                    m[i] = m_new;
#pragma unroll
                    for (int e = 0; e < ELEMS; e++) acc[i][e] *= corr;
                    s[i] = p;                                       // reuse s[] as this tile's p
                }
                // dim-parallel AV: lane owns dims {lane, lane+32, ...}; one sV read
                // (32 consecutive floats, bank-conflict-free) feeds all QPW queries
                for (int j = 0; j < tk; j++) {
                    float pj[QPW]; float any = 0.f;
#pragma unroll
                    for (int i = 0; i < QPW; i++) {
                        pj[i] = __shfl_sync(0xffffffffu, s[i], j);
                        any += pj[i];
                    }
                    if (any != 0.f) {             // warp-uniform: fully-masked keys skip whole warp
                        const float* vrow = sV + j * HEAD_DIM + lane;
#pragma unroll
                        for (int e = 0; e < ELEMS; e++) {
                            const float vv = vrow[e * 32];
#pragma unroll
                            for (int i = 0; i < QPW; i++) acc[i][e] += pj[i] * vv;
                        }
                    }
                }
            }
            __syncthreads();
        }
    };

    if (WINDOWED) {
        if (blk_rs > block_size) run_range(0, block_size);
        const int wlo = (blk_rs > block_size) ? blk_rs : 0;
        run_range(wlo, last_q + 1);
    } else {
        run_range(0, last_q + 1);
    }

    if (active) {
#pragma unroll
        for (int i = 0; i < QPW; i++) {
            if (qtok[i] >= n_tokens) break;
            const size_t q_off = ((size_t)qtok[i] * n_q_heads + head) * HEAD_DIM;
            const float inv = (l[i] > 0.f) ? (1.f / l[i]) : 0.f;
#pragma unroll
            for (int e = 0; e < ELEMS; e++)
                attn[q_off + lane + e * 32] = __float2bfloat16(acc[i][e] * inv);
        }
    }
}

// ----------------------------------------------------------------------------
// BF16-KV pure sliding-window prefill attention for Muse Glimmer: reads a BF16 K/V pool
// directly (Muse Glimmer runs a bf16 KV cache, not int8) -- no per-token dequant scale. win_blocks>0 =>
// last win_blocks blocks (SWA layers); win_blocks<=0 => full causal (global/NoPE
// layers), so this ONE kernel serves both Muse Glimmer attention types. Per-query
// window/causal predicate is warp-uniform, so the masked-key `continue` never
// splits a warp before the 32-lane reduction.
// ----------------------------------------------------------------------------
template <int HEAD_DIM, int TQ, int TK>
__global__ void win_prefill_pure_bf16_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool, const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ attn, int n_tokens, int n_q_heads, int n_kv_heads,
    int block_size, int max_blocks_per_seq, float scale, int win_blocks) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = blockIdx.y;
    const int qbase = blockIdx.x * TQ;
    const int qtok = qbase + warp;
    const int kv_head = head / (n_q_heads / n_kv_heads);
    const bool active = (qtok < n_tokens) && (head < n_q_heads);

    auto win_start = [&](int t) -> int {
        if (win_blocks <= 0) return 0;
        const int n_blk_q = (t + block_size) / block_size;
        const int rsb = (win_blocks >= n_blk_q) ? 0 : (n_blk_q - win_blocks);
        return rsb * block_size;
    };
    const int my_rs = active ? win_start(qtok) : 0;

    extern __shared__ __nv_bfloat16 smem_bf[];
    __nv_bfloat16* sK = smem_bf;
    __nv_bfloat16* sV = smem_bf + TK * HEAD_DIM;

    float q_reg[ELEMS];
    if (active) {
        const size_t q_off = ((size_t)qtok * n_q_heads + head) * HEAD_DIM;
#pragma unroll
        for (int e = 0; e < ELEMS; e++) q_reg[e] = win_to_f(q[q_off + lane + e * 32]);
    }
    float m = -1e30f, l = 0.f, acc[ELEMS];
#pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    const int last_q = min(qbase + TQ - 1, n_tokens - 1);
    const int blk_rs = win_start(qbase);

    for (int k0 = blk_rs; k0 <= last_q; k0 += TK) {
        const int tk = min(TK, last_q + 1 - k0);
        // One 16-byte copy per thread instead of eight 2-byte ones -- and, more to the point, the
        // paged-KV address math once per THREAD instead of once per element: `kpos / block_size` is
        // a division by a runtime value (~20 instructions) and `block_table[blk]` is a dependent
        // global load that has to land before the data load can even issue. The old loop paid both
        // eight times per thread per tile. HEAD_DIM is a multiple of 8 and the pools are
        // cudaMalloc'd, so every uint4 below is 16-byte aligned. Same bytes, same order.
        constexpr int VPT = 8;                    // bf16 per uint4
        constexpr int DCH = HEAD_DIM / VPT;       // uint4 chunks per key row
        for (int idx = threadIdx.x; idx < tk * DCH; idx += blockDim.x) {
            const int kk = idx / DCH, d = (idx - kk * DCH) * VPT;
            const int kpos = k0 + kk;
            const int blk = kpos / block_size, within = kpos - blk * block_size;
            const int phys = block_table[blk];
            const size_t ckt = (size_t)phys * block_size + within;
            const size_t off = (ckt * n_kv_heads + kv_head) * HEAD_DIM + d;
            *reinterpret_cast<uint4*>(sK + (size_t)kk * HEAD_DIM + d) =
                *reinterpret_cast<const uint4*>(k_pool + off);
            *reinterpret_cast<uint4*>(sV + (size_t)kk * HEAD_DIM + d) =
                *reinterpret_cast<const uint4*>(v_pool + off);
        }
        __syncthreads();
        if (active) {
            for (int kpos = k0; kpos < k0 + tk; kpos++) {
                if (kpos < my_rs || kpos > qtok) continue;
                const int kk = kpos - k0;
                const __nv_bfloat16* krow = sK + (size_t)kk * HEAD_DIM;
                float partial = 0.f;
#pragma unroll
                for (int e = 0; e < ELEMS; e++) partial += q_reg[e] * win_to_f(krow[lane + e * 32]);
                const float score = win_warp_sum(partial) * scale;
                const float m_new = fmaxf(m, score);
                const float corr = __expf(m - m_new);
                const float p = __expf(score - m_new);
                l = l * corr + p;
                const __nv_bfloat16* vrow = sV + (size_t)kk * HEAD_DIM;
#pragma unroll
                for (int e = 0; e < ELEMS; e++) acc[e] = acc[e] * corr + p * win_to_f(vrow[lane + e * 32]);
                m = m_new;
            }
        }
        __syncthreads();
    }

    if (active) {
        const size_t q_off = ((size_t)qtok * n_q_heads + head) * HEAD_DIM;
        const float inv = (l > 0.f) ? (1.f / l) : 0.f;
#pragma unroll
        for (int e = 0; e < ELEMS; e++) attn[q_off + lane + e * 32] = __float2bfloat16(acc[e] * inv);
    }
}

// ----------------------------------------------------------------------------
// Lane-parallel BF16-KV pure-window prefill attention (Muse Glimmer, hd128).
//
// Same schedule `win_prefill_lanepar_kernel` runs for the int8 hd256 layers,
// retargeted at the bf16 KV pool: ONE KEY PER LANE instead of one key at a time.
// `win_prefill_pure_bf16_kernel` walks its 128 keys sequentially and pays a full
// 32-lane butterfly plus two __expf inside the dependency chain of EVERY key --
// 32 butterflies and 64 exps per 32-key tile. Spreading the tile's keys across
// the lanes leaves the FMA count identical (each lane does the whole HEAD_DIM
// dot for its own key instead of HEAD_DIM/32 dots for all 32 keys) but collapses
// that to 2 butterflies and 2 exps per tile, which is what the kernel was
// actually bound by: it moved 2.3 MB/layer in 29 us, ~5% of DRAM bandwidth and
// ~1.5% of the SM's FMA throughput, so neither memory nor math was the limit.
//
// This is a rounding-order change, not a masking change -- the causal/window
// predicate, the sink-free window start, and the scale are copied verbatim from
// the kernel above. The online softmax becomes tile-wise (one max/sum over the
// 32 keys in flight) instead of strictly key-sequential, exactly the way the
// in-tree int8 lanepar kernel already differs from the int8 sequential ones.
//
// K/V stay bf16 in smem (the int8 kernel dequantizes to fp32 because it must):
// 18.9 KB/block keeps the same ~5 blocks/SM the sequential kernel gets, where
// fp32 tiles would cost 35 KB and halve occupancy. KSTRIDE pads the K row to
// HEAD_DIM+8 bf16 = 68 words, and 68 % 32 == 4, so the eight lanes of a 16-byte
// phase cover banks {0-3, 4-7, ... 28-31}: conflict-free.
// ----------------------------------------------------------------------------
template <int HEAD_DIM, int NWARP, int QPW, int TK>
__global__ void win_prefill_lanepar_bf16_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool, const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ attn, int n_tokens, int n_q_heads, int n_kv_heads,
    int block_size, int max_blocks_per_seq, float scale, int win_blocks) {
    constexpr int ELEMS = HEAD_DIM / 32;
    constexpr int KSTRIDE = HEAD_DIM + 8;   // bf16 elems; see bank note above
    constexpr int TQ = NWARP * QPW;         // queries per block
    static_assert(TK == 32, "one key per lane");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = blockIdx.y;
    const int qbase = blockIdx.x * TQ;
    const int q0 = qbase + warp * QPW;      // this warp's first query
    const int kv_head = head / (n_q_heads / n_kv_heads);
    const bool active = (q0 < n_tokens) && (head < n_q_heads);

    // verbatim from win_prefill_pure_bf16_kernel: no sink block, win_blocks<=0 => full causal
    auto win_start = [&](int t) -> int {
        if (win_blocks <= 0) return 0;
        const int n_blk_q = (t + block_size) / block_size;
        const int rsb = (win_blocks >= n_blk_q) ? 0 : (n_blk_q - win_blocks);
        return rsb * block_size;
    };
    int qtok[QPW], my_rs[QPW];
#pragma unroll
    for (int i = 0; i < QPW; i++) {
        qtok[i] = q0 + i;
        my_rs[i] = active ? win_start(min(qtok[i], n_tokens - 1)) : 0;
    }

    extern __shared__ __nv_bfloat16 smem_lpb[];
    __nv_bfloat16* sK = smem_lpb;                          // [TK][KSTRIDE]
    __nv_bfloat16* sV = sK + TK * KSTRIDE;                 // [TK][HEAD_DIM]
    __nv_bfloat16* sQ = sV + TK * HEAD_DIM;                // [TQ][HEAD_DIM]

    for (int idx = threadIdx.x; idx < TQ * (HEAD_DIM / 8); idx += blockDim.x) {
        const int qq = idx / (HEAD_DIM / 8), d8 = idx - qq * (HEAD_DIM / 8);
        const int qt = qbase + qq;
        uint4 v4 = make_uint4(0, 0, 0, 0);
        if (qt < n_tokens && head < n_q_heads) {
            const size_t q_off = ((size_t)qt * n_q_heads + head) * HEAD_DIM;
            v4 = *reinterpret_cast<const uint4*>(q + q_off + (size_t)d8 * 8);
        }
        *reinterpret_cast<uint4*>(sQ + (size_t)qq * HEAD_DIM + (size_t)d8 * 8) = v4;
    }

    float m[QPW], l[QPW], acc[QPW][ELEMS];
#pragma unroll
    for (int i = 0; i < QPW; i++) {
        m[i] = -1e30f; l[i] = 0.f;
#pragma unroll
        for (int e = 0; e < ELEMS; e++) acc[i][e] = 0.f;
    }

    const int last_q = min(qbase + TQ - 1, n_tokens - 1);
    const int blk_rs = win_start(qbase);

    for (int k0 = blk_rs; k0 <= last_q; k0 += TK) {
        const int tk = min(TK, last_q + 1 - k0);
        // identical staging to the sequential kernel: one uint4 per thread, paged
        // address math once per thread rather than once per element
        constexpr int VPT = 8, DCH = HEAD_DIM / VPT;
        for (int idx = threadIdx.x; idx < tk * DCH; idx += blockDim.x) {
            const int kk = idx / DCH, d = (idx - kk * DCH) * VPT;
            const int kpos = k0 + kk;
            const int blk = kpos / block_size, within = kpos - blk * block_size;
            const int phys = block_table[blk];
            const size_t ckt = (size_t)phys * block_size + within;
            const size_t off = (ckt * n_kv_heads + kv_head) * HEAD_DIM + d;
            *reinterpret_cast<uint4*>(sK + (size_t)kk * KSTRIDE + d) =
                *reinterpret_cast<const uint4*>(k_pool + off);
            *reinterpret_cast<uint4*>(sV + (size_t)kk * HEAD_DIM + d) =
                *reinterpret_cast<const uint4*>(v_pool + off);
        }
        __syncthreads();
        if (active && k0 <= qtok[QPW - 1]) {
            const int kpos = k0 + lane;
            const bool live = lane < tk;
            bool in[QPW];
#pragma unroll
            for (int i = 0; i < QPW; i++)
                in[i] = live && (kpos >= my_rs[i]) && (kpos <= qtok[i]) && (qtok[i] < n_tokens);
            // one key per lane: the whole HEAD_DIM dot lives in this lane, and the
            // single K row read feeds all QPW query dots
            float s[QPW];
            {
                const __nv_bfloat16* krow = sK + (size_t)lane * KSTRIDE;
                const __nv_bfloat16* qrow = sQ + (size_t)(warp * QPW) * HEAD_DIM;
                float sa[QPW], sb[QPW];
#pragma unroll
                for (int i = 0; i < QPW; i++) { sa[i] = 0.f; sb[i] = 0.f; }
#pragma unroll
                // Scalar __bfloat162float, deliberately: staging the K pair through
                // __bfloat1622float2 into a float2[4] costs more in registers than it saves in
                // conversions and measured ~0.5% SLOWER end to end. Do not "optimize" this.
                for (int d = 0; d < HEAD_DIM; d += 8) {
                    const uint4 kraw = *reinterpret_cast<const uint4*>(krow + d);
                    const __nv_bfloat162* kp = reinterpret_cast<const __nv_bfloat162*>(&kraw);
#pragma unroll
                    for (int i = 0; i < QPW; i++) {
                        const uint4 qraw = *reinterpret_cast<const uint4*>(
                            qrow + (size_t)i * HEAD_DIM + d);                  // broadcast read
                        const __nv_bfloat162* qp = reinterpret_cast<const __nv_bfloat162*>(&qraw);
#pragma unroll
                        for (int t = 0; t < 4; t++) {
                            sa[i] += __bfloat162float(qp[t].x) * __bfloat162float(kp[t].x);
                            sb[i] += __bfloat162float(qp[t].y) * __bfloat162float(kp[t].y);
                        }
                    }
                }
#pragma unroll
                for (int i = 0; i < QPW; i++) s[i] = in[i] ? (sa[i] + sb[i]) * scale : -1e30f;
            }
#pragma unroll
            for (int i = 0; i < QPW; i++) {
                float tmax = s[i];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
                if (tmax <= -1e29f) { s[i] = 0.f; continue; }  // no live key for this query
                const float m_new = fmaxf(m[i], tmax);
                const float corr = __expf(m[i] - m_new);
                const float p = in[i] ? __expf(s[i] - m_new) : 0.f;
                float tl = p;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) tl += __shfl_xor_sync(0xffffffffu, tl, o);
                l[i] = l[i] * corr + tl;
                m[i] = m_new;
#pragma unroll
                for (int e = 0; e < ELEMS; e++) acc[i][e] *= corr;
                s[i] = p;                                      // reuse s[] as this tile's p
            }
            // dim-parallel AV: lane owns dims {lane, lane+32, ...}; one sV read
            // feeds all QPW queries
            for (int j = 0; j < tk; j++) {
                float pj[QPW]; float any = 0.f;
#pragma unroll
                for (int i = 0; i < QPW; i++) {
                    pj[i] = __shfl_sync(0xffffffffu, s[i], j);
                    any += pj[i];
                }
                if (any != 0.f) {          // warp-uniform: fully-masked keys skip the warp
                    const __nv_bfloat16* vrow = sV + (size_t)j * HEAD_DIM + lane;
#pragma unroll
                    for (int e = 0; e < ELEMS; e++) {
                        const float vv = win_to_f(vrow[e * 32]);
#pragma unroll
                        for (int i = 0; i < QPW; i++) acc[i][e] += pj[i] * vv;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (active) {
#pragma unroll
        for (int i = 0; i < QPW; i++) {
            if (qtok[i] >= n_tokens) break;
            const size_t q_off = ((size_t)qtok[i] * n_q_heads + head) * HEAD_DIM;
            const float inv = (l[i] > 0.f) ? (1.f / l[i]) : 0.f;
#pragma unroll
            for (int e = 0; e < ELEMS; e++)
                attn[q_off + lane + e * 32] = __float2bfloat16(acc[i][e] * inv);
        }
    }
}

}  // namespace

// ----------------------------------------------------------------------------
// Host launcher. Returns true if a windowed/tiled kernel was launched; false if
// the caller should run its own attention (e.g. head_dim != 256, or the window
// and tiling are both disabled). Env knobs:
//   SPARKINFER_PREFILL_ATTN_WINDOW  (default 256) : window size in KV blocks; 0 disables.
//   SPARKINFER_PREFILL_ATTN_TILED   (default 0)   : use the smem-tiled full kernel when window off.
//   SPARKINFER_PREFILL_ATTN_LANEPAR (default 1)   : lane-parallel kernel; 0 = the older
//                                                   one-key-at-a-time warp kernels.
// ----------------------------------------------------------------------------
bool launch_prefill_attn_windowed(
    const void* q, const signed char* k_pool, const signed char* v_pool,
    const void* k_scale, const void* v_scale, const int* block_table, void* attn,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, float scale, cudaStream_t stream) {
    auto qb = reinterpret_cast<const __nv_bfloat16*>(q);
    auto ks = reinterpret_cast<const __half*>(k_scale);
    auto vs = reinterpret_cast<const __half*>(v_scale);
    auto ob = reinterpret_cast<__nv_bfloat16*>(attn);

    if (head_dim != 256) return false;   // only the hd256 full-attention layers are windowed
    constexpr int TQ = 16, TK = 32, HD = 256;
    const size_t sm = (size_t)2 * TK * HD * sizeof(__half);  // 32 KB smem: K + V tiles (half)
    // lane-parallel kernel: fp32 K tile (rows padded +4) + fp32 V tile + bf16 query tile
    constexpr int QPW = 4, TQ_LP = 16 * QPW;
    const size_t sm_lp = ((size_t)TK * (HD + 4) + (size_t)TK * HD) * sizeof(float)
                       + (size_t)TQ_LP * HD * sizeof(__nv_bfloat16);

    // The lane-parallel tiles need 98,816 B of dynamic shared memory — well past the
    // 48 KB default — so the cudaFuncSetAttribute opt-in below is REQUIRED for those
    // launches to be valid, and both it and the launch itself have to be checked: a
    // discarded failure used to report success to the caller, which then skipped its
    // full-attention fallback and consumed whatever `attn` already held — silently
    // wrong logits, no diagnostic. The attribute is also a PER-DEVICE setting, so the
    // do-once latches are keyed on the device ordinal, not the process (a process-wide
    // latch leaves every device but the first unconfigured, and the launch then fails
    // with cudaErrorInvalidValue on exactly the path that needs the raise). On a
    // refusal the windowed branch degrades to the 32 KB scalar windowed kernel, and
    // the full branch to `return false` — the contract the tail of this function
    // already documents ("caller runs full attention"). The launch check peeks —
    // rather than gets — the error, so a pre-existing sticky error is not silently
    // cleared here.
    constexpr size_t kSmDefault = 48u * 1024u;
    constexpr int kMaxDevices = 16;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= kMaxDevices) return false;

    static int lanepar = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_LANEPAR");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    static int win_blocks = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_WINDOW");
        return e ? atoi(e) : 256;
    }();
    if (win_blocks > 0) {
        bool lanepar_ok = lanepar != 0;
        if (lanepar_ok) {
            static int cfglw[kMaxDevices] = {0};
            if (!cfglw[dev]) {
                const cudaError_t ce =
                    cudaFuncSetAttribute(win_prefill_lanepar_kernel<HD, QPW, TK, true>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_lp);
                if (ce != cudaSuccess && sm_lp > kSmDefault) lanepar_ok = false;
                else cfglw[dev] = 1;
            }
            if (lanepar_ok) {
                dim3 gridw((n_tokens + TQ_LP - 1) / TQ_LP, n_q_heads);
                win_prefill_lanepar_kernel<HD, QPW, TK, true><<<gridw, 16 * 32, sm_lp, stream>>>(
                    qb, k_pool, v_pool, ks, vs, block_table, ob, n_tokens, n_q_heads, n_kv_heads,
                    block_size, max_blocks_per_seq, scale, win_blocks);
                if (cudaPeekAtLastError() == cudaSuccess) return true;
            }
            // opt-in refused or launch invalid: degrade to the 32 KB scalar windowed
            // kernel below rather than reporting success over an unwritten buffer.
        }
        static int cfgw[kMaxDevices] = {0};
        if (!cfgw[dev]) {
            cudaFuncSetAttribute(win_prefill_windowed_kernel<HD, TQ, TK>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
            cfgw[dev] = 1;   // 32 KB fits the default; the attribute is advisory here
        }
        dim3 gridw((n_tokens + TQ - 1) / TQ, n_q_heads);
        win_prefill_windowed_kernel<HD, TQ, TK><<<gridw, TQ * 32, sm, stream>>>(
            qb, k_pool, v_pool, ks, vs, block_table, ob, n_tokens, n_q_heads, n_kv_heads,
            block_size, max_blocks_per_seq, scale, win_blocks);
        return cudaPeekAtLastError() == cudaSuccess;
    }

    if (lanepar) {
        static int cfglf[kMaxDevices] = {0};
        bool ok = true;
        if (!cfglf[dev]) {
            const cudaError_t ce =
                cudaFuncSetAttribute(win_prefill_lanepar_kernel<HD, QPW, TK, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_lp);
            if (ce != cudaSuccess && sm_lp > kSmDefault) ok = false;
            else cfglf[dev] = 1;
        }
        if (ok) {
            dim3 grid((n_tokens + TQ_LP - 1) / TQ_LP, n_q_heads);
            win_prefill_lanepar_kernel<HD, QPW, TK, false><<<grid, 16 * 32, sm_lp, stream>>>(
                qb, k_pool, v_pool, ks, vs, block_table, ob, n_tokens, n_q_heads, n_kv_heads,
                block_size, max_blocks_per_seq, scale, 0);
            if (cudaPeekAtLastError() == cudaSuccess) return true;
        }
        // fall through: the tiled opt-in (if enabled) or the caller's full attention
        // computes the same unwindowed result this variant would have.
    }

    static int tiled = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ATTN_TILED");
        return (e && e[0] == '1') ? 1 : 0;
    }();
    if (tiled) {
        static int cfg[kMaxDevices] = {0};
        if (!cfg[dev]) {
            cudaFuncSetAttribute(win_prefill_tiled_kernel<HD, TQ, TK>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
            cfg[dev] = 1;   // 32 KB fits the default; the attribute is advisory here
        }
        dim3 grid((n_tokens + TQ - 1) / TQ, n_q_heads);
        win_prefill_tiled_kernel<HD, TQ, TK><<<grid, TQ * 32, sm, stream>>>(
            qb, k_pool, v_pool, ks, vs, block_table, ob, n_tokens, n_q_heads, n_kv_heads,
            block_size, max_blocks_per_seq, scale);
        return cudaPeekAtLastError() == cudaSuccess;
    }

    return false;   // window + tiling both off -> caller runs full attention
}

// BF16-KV pure-window prefill attention for Muse Glimmer (hd128). win_blocks>0 =>
// SWA layers (last win_blocks blocks); win_blocks<=0 => global/NoPE full causal.
namespace {

// ---------------------------------------------------------------------------------------------
// bf16 TENSOR-CORE windowed prefill attention (Muse Glimmer: hd128, bf16 paged KV).
//
// prefill_attn_mma.h already states the problem: the scalar path "evaluates QK^T and PV with scalar
// FMA plus a 5-shuffle warp reduction per key, which measures ~8 TFLOP/s on sm_120", and the repo
// already fixed it on the tensor cores -- for Qwythos hd256 + INT8 paged KV
// (pf_attn_mma_i8_kernel). Muse's windowed layers are hd128 + bf16 and never reach that path, so
// they still pay one warp reduction per key. Measured standalone at Muse's exact shape
// (n=128, 32 q heads, 2 kv heads, hd128, causal): scalar 28.67 us vs bf16 wmma 10.25 us = 2.80x,
// and 5.02x at n=512.
//
// This is the int8 kernel's schedule with the int8 machinery REMOVED, not a new design:
//   * QK and PV accumulate in fp32 directly, so the per-tile Q amax/round and the per-row P
//     amax/round (plus its extra warp reduction) that existed only to feed int8 tensor cores are
//     gone. P stays bf16.
//   * K and V fragments are loaded STRAIGHT FROM THE PAGED POOL, no smem staging: block_size is 16
//     and wmma's tile is 16x16, so a KV page IS a fragment with ldm = n_kv_heads*HEAD_DIM.
//   * The per-row `corr` rescale of the O accumulator makes NO assumption about the fp32
//     accumulator's element order -- an index fragment carrying (row<<8)|col is loaded once and
//     `idxf.x[e] >> 8` names the row, exactly as pf_attn_mma_i8_kernel does.
//
// Mask is the scalar kernel's, term for term: per-query window start (NOT the block-level start)
// and causal `gtok <= qtok`. NOT bit-identical to the scalar path -- fp32 tensor-core accumulation
// reassociates the sum -- so the batched-prefill accuracy gate is mandatory before this ships.
//
// WARPS must divide HEAD_DIM/16 (each warp owns HEAD_DIM/16/WARPS output d-tiles in the PV mma)
// and equals the number of KV pages consumed per group iteration.
template <int HEAD_DIM, int WARPS>
__global__ __launch_bounds__(WARPS * 32, 2) void win_prefill_mma_bf16_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool, const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ attn, int n_tokens, int n_q_heads, int n_kv_heads,
    int block_size, int max_blocks_per_seq, float scale, int win_blocks) {
    using namespace nvcuda::wmma;
    constexpr int BM = 16;                 // query rows per block == wmma M == KV page size
    constexpr int GN = 16 * WARPS;         // keys consumed per group iteration
    constexpr int DT = HEAD_DIM / 16;      // d-tiles
    constexpr int DPW = DT / WARPS;        // d-tiles per warp in the PV mma
    constexpr int RPW = BM / WARPS;        // query rows per warp in the softmax
    static_assert(DT % WARPS == 0, "HEAD_DIM/16 must be divisible by WARPS");
    static_assert(BM % WARPS == 0, "BM must be divisible by WARPS");
    static_assert(GN % 32 == 0, "GN must be a multiple of the warp width");

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int head = blockIdx.y, qbase = blockIdx.x * BM;
    const int kvh = head / (n_q_heads / n_kv_heads);
    const int KVLD = n_kv_heads * HEAD_DIM;          // ldm of a KV page in the pool
    (void)max_blocks_per_seq;

    extern __shared__ char mma_smem_bf[];
    __nv_bfloat16* s_q = reinterpret_cast<__nv_bfloat16*>(mma_smem_bf);   // [BM][HEAD_DIM]
    __nv_bfloat16* s_p = s_q + BM * HEAD_DIM;                             // [BM][GN]
    float* s_s    = reinterpret_cast<float*>(s_p + BM * GN);              // [BM][GN]
    float* s_o    = s_s + BM * GN;                                        // [BM][HEAD_DIM]
    float* s_m    = s_o + BM * HEAD_DIM;                                  // [BM]
    float* s_l    = s_m + BM;                                             // [BM]
    float* s_corr = s_l + BM;                                             // [BM]

    // Per-query window start, identical to the scalar kernel's win_start().
    auto win_start = [&](int t) -> int {
        if (win_blocks <= 0) return 0;
        const int n_blk_q = (t + block_size) / block_size;
        const int rsb = (win_blocks >= n_blk_q) ? 0 : (n_blk_q - win_blocks);
        return rsb * block_size;
    };

    // Accumulator row map: no assumption about the fp32 fragment's element order.
    fragment<accumulator, 16, 16, 16, float> ofr[DPW];
    fragment<accumulator, 16, 16, 16, int> idxf;
    {
        int* tile = reinterpret_cast<int*>(s_s) + warp * 256;   // disjoint per warp, reused below
        for (int i = lane; i < 256; i += 32) tile[i] = ((i >> 4) << 8) | (i & 15);
        __syncwarp();
        load_matrix_sync(idxf, tile, 16, mem_row_major);
    }
    #pragma unroll
    for (int dd = 0; dd < DPW; dd++) fill_fragment(ofr[dd], 0.f);

    for (int i = tid; i < BM * HEAD_DIM; i += blockDim.x) {
        const int r = i / HEAD_DIM, c = i - r * HEAD_DIM, t = qbase + r;
        s_q[i] = (t < n_tokens)
               ? q[((size_t)t * n_q_heads + head) * HEAD_DIM + c] : __float2bfloat16(0.f);
    }
    for (int r = tid; r < BM; r += blockDim.x) { s_m[r] = -1e30f; s_l[r] = 0.f; }
    __syncthreads();

    const int last_q = min(qbase + BM - 1, n_tokens - 1);
    const int blk_rs = win_start(qbase);   // monotonic in t, so this bounds every row in the tile

    for (int k0 = blk_rs; k0 <= last_q; k0 += GN) {
        const int gblk = min(WARPS, (last_q + 1 - k0 + 15) >> 4);   // live 16-key pages this group

        // ---- QK: one page per warp, fp32 accumulate, land the tile in s_s ----
        if (warp < gblk) {
            const int pb = block_table[(k0 / block_size) + warp];
            const __nv_bfloat16* kb =
                k_pool + ((size_t)pb * block_size * n_kv_heads + kvh) * HEAD_DIM;
            fragment<accumulator, 16, 16, 16, float> cf;
            fill_fragment(cf, 0.f);
            #pragma unroll
            for (int d = 0; d < DT; d++) {
                fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> af;
                fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> bf;
                load_matrix_sync(af, s_q + d * 16, HEAD_DIM);
                load_matrix_sync(bf, kb + d * 16, KVLD);
                mma_sync(cf, af, bf, cf);
            }
            store_matrix_sync(s_s + warp * 16, cf, GN, mem_row_major);
        }
        __syncthreads();

        // ---- online softmax over the whole GN-wide row, one warp per RPW rows ----
        #pragma unroll
        for (int rr = 0; rr < RPW; rr++) {
            const int r = warp * RPW + rr, qtok = qbase + r;
            const int my_rs = (qtok < n_tokens) ? win_start(qtok) : 0;
            float sc[GN / 32], mx = -1e30f;
            #pragma unroll
            for (int u = 0; u < GN / 32; u++) {
                const int t = lane + u * 32, gtok = k0 + t;
                const bool live = (t < gblk * 16) && (qtok < n_tokens) &&
                                  (gtok <= qtok) && (gtok >= my_rs);
                sc[u] = live ? s_s[r * GN + t] * scale : -1e30f;
                mx = fmaxf(mx, sc[u]);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = s_m[r], m_new = fmaxf(m_old, mx), corr = __expf(m_old - m_new);
            float sum = 0.f;
            #pragma unroll
            for (int u = 0; u < GN / 32; u++) {
                const int t = lane + u * 32;
                const float p = (sc[u] > -1e29f) ? __expf(sc[u] - m_new) : 0.f;
                sum += p;
                s_p[r * GN + t] = __float2bfloat16(p);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            if (lane == 0) {
                s_m[r] = m_new; s_l[r] = s_l[r] * corr + sum; s_corr[r] = corr;
            }
        }
        __syncthreads();

        // ---- PV: each warp owns DPW output d-tiles; V comes straight from the pool ----
        #pragma unroll
        for (int dd = 0; dd < DPW; dd++) {
            const int dt = warp * DPW + dd;
            fragment<accumulator, 16, 16, 16, float> cf;
            fill_fragment(cf, 0.f);
            for (int ks = 0; ks < gblk; ks++) {
                const int pb = block_table[(k0 / block_size) + ks];
                const __nv_bfloat16* vb =
                    v_pool + ((size_t)pb * block_size * n_kv_heads + kvh) * HEAD_DIM + dt * 16;
                fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> af;
                fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> bf;
                load_matrix_sync(af, s_p + ks * 16, GN);
                load_matrix_sync(bf, vb, KVLD);
                mma_sync(cf, af, bf, cf);
            }
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                const int r = idxf.x[e] >> 8;
                ofr[dd].x[e] = __fmaf_rn(ofr[dd].x[e], s_corr[r], cf.x[e]);
            }
        }
        __syncthreads();   // s_s / s_p are rewritten by the next group
    }

    #pragma unroll
    for (int dd = 0; dd < DPW; dd++)
        store_matrix_sync(s_o + (warp * DPW + dd) * 16, ofr[dd], HEAD_DIM, mem_row_major);
    __syncthreads();

    for (int r = 0; r < BM; r++) {
        const int qtok = qbase + r;
        if (qtok >= n_tokens) break;
        const float l = s_l[r], inv = (l > 0.f) ? (1.f / l) : 0.f;
        for (int c = tid; c < HEAD_DIM; c += blockDim.x)
            attn[((size_t)qtok * n_q_heads + head) * HEAD_DIM + c] =
                __float2bfloat16(s_o[r * HEAD_DIM + c] * inv);
    }
}

}  // namespace

void launch_prefill_attn_swa_pure_bf16(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, void* attn,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, float scale, int win_blocks,
    cudaStream_t stream) {
    (void)head_dim;   // Muse Glimmer attention is hd128 only; templated below.
    // TQ=8, not 16. This kernel is one warp per query, so TQ only sets how many queries share a
    // block's K/V tile -- each warp still walks its own keys in ascending kpos, so the fp32 dot and
    // the online-softmax chain are untouched and the result is bit-identical. At Muse's prefill@128
    // the old TQ=16 launched ceil(128/16) x 32 = 256 blocks of 512 threads against 170 SMs that
    // hold 4 such blocks each: 37.6% of the block slots, with the tail of a 1.5-block-per-SM
    // distribution setting the runtime. Halving TQ doubles the grid to 512 smaller blocks and
    // spreads them evenly, for the same total warps and the same K/V tile bytes per block.
    constexpr int TQ = 8, TK = 32, HD = 128;

    // bf16 tensor-core path, ahead of the scalar schedules. Requires a 16-token KV page (a page is
    // then exactly one wmma tile) and a GQA ratio, which Muse Glimmer satisfies (block_size 16,
    // 32 q heads over 2 kv heads). Anything else falls through to the scalar kernels below,
    // unchanged. SPARKINFER_MUSE_ATTN_MMA=0 disables (one-binary A/B control).
    static const bool attn_mma = [] {
        const char* e = getenv("SPARKINFER_MUSE_ATTN_MMA");
        return !e || atoi(e) != 0;
    }();
    if (attn_mma && block_size == 16 && n_kv_heads > 0 && (n_q_heads % n_kv_heads) == 0) {
        // WARPS is both the KV pages consumed per group iteration and the divisor of HD/16 output
        // d-tiles. At prefill@128 with WARPS=8 the group is 128 keys wide, so the whole causal
        // range for a 16-query tile is ONE iteration -- no group loop, no repeated barriers.
        // SPARKINFER_MUSE_ATTN_MMA_WARPS selects (4 or 8) for a one-binary A/B.
        static const int mw = [] {
            const char* e = getenv("SPARKINFER_MUSE_ATTN_MMA_WARPS");
            const int v = e ? atoi(e) : 8;
            return (v == 4 || v == 8) ? v : 8;
        }();
        constexpr int MBM = 16;
        auto sm_for = [&](int W) {
            const int GNw = 16 * W;
            return (size_t)MBM * HD * sizeof(__nv_bfloat16)          // s_q
                 + (size_t)MBM * GNw * sizeof(__nv_bfloat16)         // s_p
                 + (size_t)MBM * GNw * sizeof(float)                 // s_s (also the idx tile)
                 + (size_t)MBM * HD * sizeof(float)                  // s_o
                 + (size_t)3 * MBM * sizeof(float);                  // s_m / s_l / s_corr
        };
        dim3 gm((n_tokens + MBM - 1) / MBM, n_q_heads);
        if (mw == 8) {
            win_prefill_mma_bf16_kernel<HD, 8><<<gm, 8 * 32, sm_for(8), stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q),
                reinterpret_cast<const __nv_bfloat16*>(k_pool),
                reinterpret_cast<const __nv_bfloat16*>(v_pool), block_table,
                reinterpret_cast<__nv_bfloat16*>(attn),
                n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, win_blocks);
        } else {
            win_prefill_mma_bf16_kernel<HD, 4><<<gm, 4 * 32, sm_for(4), stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q),
                reinterpret_cast<const __nv_bfloat16*>(k_pool),
                reinterpret_cast<const __nv_bfloat16*>(v_pool), block_table,
                reinterpret_cast<__nv_bfloat16*>(attn),
                n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, win_blocks);
        }
        if (cudaPeekAtLastError() == cudaSuccess) return;
        // else fall through: the scalar kernels below compute the same attention.
    }

    // Lane-parallel schedule (one key per lane) by default; SPARKINFER_MUSE_ATTN_LANEPAR=0
    // restores the sequential one-key-at-a-time kernel. Same grid either way, so the only
    // difference is the schedule and the fp32 rounding order that comes with it.
    static const bool lanepar = [] {
        const char* e = getenv("SPARKINFER_MUSE_ATTN_LANEPAR");
        return !e || atoi(e) != 0;
    }();
    if (lanepar) {
        // NWARP=8 / QPW=1 keeps EXACTLY the grid the sequential kernel uses (512 blocks of 256
        // threads at prefill@128), so this changes the schedule and nothing else. Widening the
        // block to amortize the staged K/V tile over more queries looked attractive -- each tile
        // is re-staged by grid.x blocks per head -- but it measured monotonically worse
        // (prefill@128 medians: QPW=1 8709, QPW=2 8594, QPW=4 8536), the same way the sequential
        // kernel preferred TQ=8 over TQ=16: block count and machine coverage dominate that L2
        // traffic. Do not re-sweep QPW without a structural reason.
        constexpr int NWARP = 8, QPW = 1, KSTRIDE = HD + 8, TQB = NWARP * QPW;
        const size_t sm = (size_t)(TK * KSTRIDE + TK * HD + TQB * HD) * sizeof(__nv_bfloat16);
        dim3 g((n_tokens + TQB - 1) / TQB, n_q_heads);
        win_prefill_lanepar_bf16_kernel<HD, NWARP, QPW, TK><<<g, NWARP * 32, sm, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q),
            reinterpret_cast<const __nv_bfloat16*>(k_pool),
            reinterpret_cast<const __nv_bfloat16*>(v_pool), block_table,
            reinterpret_cast<__nv_bfloat16*>(attn),
            n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, win_blocks);
        return;
    }
    dim3 grid((n_tokens + TQ - 1) / TQ, n_q_heads);
    const size_t sm = (size_t)2 * TK * HD * sizeof(__nv_bfloat16);   // 16 KB: K + V bf16 tiles
    win_prefill_pure_bf16_kernel<HD, TQ, TK><<<grid, TQ * 32, sm, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k_pool),
        reinterpret_cast<const __nv_bfloat16*>(v_pool), block_table,
        reinterpret_cast<__nv_bfloat16*>(attn),
        n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, win_blocks);
}

}  // namespace kernels
}  // namespace sparkinfer
