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

#include <cstdlib>

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

// ----------------------------------------------------------------------------
// Faster and more accurate schedule for the kernel above. Same decomposition --
// one warp per query, keys walked in ascending kpos, the same fp32 online
// softmax, the same pure rolling-window predicate -- with three changes.
//
// 1. Lane owns 4 CONSECUTIVE head dims instead of `lane + e*32`. The strided
//    form is a 2-byte-per-lane shared load, so a warp moves 64 B per
//    instruction, and it pays that four times per key for K and four for V.
//    Consecutive-four is one __nv_bfloat162 pair: 256 B per warp instruction
//    and a quarter of the LDS. Q, the V accumulation and the store follow.
//
// 2. KG keys are scored before any of them is reduced. Per key the old loop is
//    a single dependent chain -- 4 FMAs, a 5-step shuffle butterfly, two
//    __expf, then the loop-carried m/l/acc update -- and only the FMAs are
//    work. Scoring KG first makes the butterflies independent so their shuffle
//    steps interleave. KG=4 measured best of {1,2,4,8} (+1.24 / +1.88 / +1.45%
//    end to end at 1,4,8); past 4 the live scores cost more than they buy.
//
// 3. The cross-lane butterfly runs ASCENDING (1,2,4,8,16). win_warp_sum() goes
//    16->1, which pairs each lane with the one holding dims 16 away before
//    pairing neighbours; ascending sums adjacent dims first, which are far
//    closer in magnitude, so the 32-term reduction is better conditioned. This
//    is why the kernel is MORE accurate than the one above, not less.
//
// ncu at prefill@128 on the old kernel: L1/TEX throughput 67.95% with DRAM at
// 1.61% and 39.7% of a 9.75-cycle issue gap in MIO short-scoreboard -- bound by
// shared-memory issue, not bandwidth and not math. After: 60.68% and 8.03.
//
// Numerics move (the 32-term dot is re-associated) and they move the right way.
// qwen3_gguf_prefill_check 128/512, SPARKINFER_KV_INT8=0, batched vs the token
// reference:  main 469/512 = 0.9160, KL 0.01818  ->  this 480/512 = 0.9375,
// KL 0.01427. The per-lane 4-term sum is exact in fp32 (a bf16*bf16 product
// carries at most 16 mantissa bits), so only the cross-lane order matters --
// reversing the per-lane order alone is bit-identical.
// SPARKINFER_MUSE_PREFILL_ATTN_WIDE=0 restores the kernel above.
// ----------------------------------------------------------------------------
template <int HEAD_DIM, int TQ, int TK, int KG>
__global__ void win_prefill_pure_bf16_wide_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool, const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ attn, int n_tokens, int n_q_heads, int n_kv_heads,
    int block_size, int max_blocks_per_seq, float scale, int win_blocks) {
    (void)max_blocks_per_seq;
    constexpr int ELEMS = HEAD_DIM / 32;
    static_assert(ELEMS == 4, "four consecutive dims per lane");
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int head = blockIdx.y;
    const int qbase = blockIdx.x * TQ;
    const int qtok = qbase + warp;
    const int kv_head = head / (n_q_heads / n_kv_heads);
    const bool active = (qtok < n_tokens) && (head < n_q_heads);
    const int d0 = lane * ELEMS;

    auto win_start = [&](int t) -> int {
        if (win_blocks <= 0) return 0;
        const int n_blk_q = (t + block_size) / block_size;
        const int rsb = (win_blocks >= n_blk_q) ? 0 : (n_blk_q - win_blocks);
        return rsb * block_size;
    };
    const int my_rs = active ? win_start(qtok) : 0;

    extern __shared__ __nv_bfloat16 smem_bfw[];
    __nv_bfloat16* sK = smem_bfw;
    __nv_bfloat16* sV = smem_bfw + TK * HEAD_DIM;

    float q_reg[ELEMS];
    if (active) {
        const size_t q_off = ((size_t)qtok * n_q_heads + head) * HEAD_DIM;
        const __nv_bfloat162* qp = reinterpret_cast<const __nv_bfloat162*>(q + q_off + d0);
        const __nv_bfloat162 a = qp[0], b = qp[1];
        q_reg[0] = __bfloat162float(a.x); q_reg[1] = __bfloat162float(a.y);
        q_reg[2] = __bfloat162float(b.x); q_reg[3] = __bfloat162float(b.y);
    }
    float m = -1e30f, l = 0.f, acc[ELEMS];
#pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    const int last_q = min(qbase + TQ - 1, n_tokens - 1);
    const int blk_rs = win_start(qbase);

    for (int k0 = blk_rs; k0 <= last_q; k0 += TK) {
        const int tk = min(TK, last_q + 1 - k0);
        // One 16-byte copy per thread, and the paged-KV address math once per THREAD rather than
        // once per element (see win_prefill_pure_bf16_kernel). Same bytes, same order.
        constexpr int VPT = 8;
        constexpr int DCH = HEAD_DIM / VPT;
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
            // Warp-uniform, so a key's validity is warp-uniform and the grouping below never
            // splits a warp before its butterfly.
            const int lo = max(k0, my_rs);
            const int hi = min(k0 + tk - 1, qtok);
            for (int base = lo; base <= hi; base += KG) {
                const int n = min(KG, hi + 1 - base);
                float part[KG];
#pragma unroll
                for (int g = 0; g < KG; g++) {
                    part[g] = 0.f;
                    if (g < n) {
                        const __nv_bfloat162* kp = reinterpret_cast<const __nv_bfloat162*>(
                            sK + (size_t)(base + g - k0) * HEAD_DIM + d0);
                        const __nv_bfloat162 a = kp[0], b = kp[1];
                        part[g] = q_reg[0] * __bfloat162float(a.x)
                                + q_reg[1] * __bfloat162float(a.y)
                                + q_reg[2] * __bfloat162float(b.x)
                                + q_reg[3] * __bfloat162float(b.y);
                    }
                }
#pragma unroll
                for (int d2 = 1; d2 <= 16; d2 <<= 1) {
#pragma unroll
                    for (int g = 0; g < KG; g++)
                        part[g] += __shfl_xor_sync(0xffffffffu, part[g], d2);
                }
#pragma unroll
                for (int g = 0; g < KG; g++) {
                    if (g >= n) break;
                    const float score = part[g] * scale;
                    const float m_new = fmaxf(m, score);
                    const float corr = __expf(m - m_new);
                    const float pr = __expf(score - m_new);
                    l = l * corr + pr;
                    const __nv_bfloat162* vp = reinterpret_cast<const __nv_bfloat162*>(
                        sV + (size_t)(base + g - k0) * HEAD_DIM + d0);
                    const __nv_bfloat162 a = vp[0], b = vp[1];
                    acc[0] = acc[0] * corr + pr * __bfloat162float(a.x);
                    acc[1] = acc[1] * corr + pr * __bfloat162float(a.y);
                    acc[2] = acc[2] * corr + pr * __bfloat162float(b.x);
                    acc[3] = acc[3] * corr + pr * __bfloat162float(b.y);
                    m = m_new;
                }
            }
        }
        __syncthreads();
    }

    if (active) {
        const size_t q_off = ((size_t)qtok * n_q_heads + head) * HEAD_DIM;
        const float inv = (l > 0.f) ? (1.f / l) : 0.f;
        __nv_bfloat162 o0, o1;
        o0.x = __float2bfloat16(acc[0] * inv); o0.y = __float2bfloat16(acc[1] * inv);
        o1.x = __float2bfloat16(acc[2] * inv); o1.y = __float2bfloat16(acc[3] * inv);
        __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(attn + q_off + d0);
        op[0] = o0; op[1] = o1;
    }
}

// BF16-KV pure-window prefill attention for Muse Glimmer (hd128). win_blocks>0 =>
// SWA layers (last win_blocks blocks); win_blocks<=0 => global/NoPE full causal.
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
    const size_t sm = (size_t)2 * TK * HD * sizeof(__nv_bfloat16);   // 16 KB: K + V bf16 tiles
    dim3 grid((n_tokens + TQ - 1) / TQ, n_q_heads);
    static const bool wide = [] {
        const char* e = getenv("SPARKINFER_MUSE_PREFILL_ATTN_WIDE");
        return !(e && e[0] == '0');
    }();
    if (wide) {
        win_prefill_pure_bf16_wide_kernel<HD, TQ, TK, 4><<<grid, TQ * 32, sm, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k_pool),
            reinterpret_cast<const __nv_bfloat16*>(v_pool), block_table,
            reinterpret_cast<__nv_bfloat16*>(attn),
            n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, win_blocks);
        return;
    }
    win_prefill_pure_bf16_kernel<HD, TQ, TK><<<grid, TQ * 32, sm, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k_pool),
        reinterpret_cast<const __nv_bfloat16*>(v_pool), block_table,
        reinterpret_cast<__nv_bfloat16*>(attn),
        n_tokens, n_q_heads, n_kv_heads, block_size, max_blocks_per_seq, scale, win_blocks);
}

}  // namespace kernels
}  // namespace sparkinfer
