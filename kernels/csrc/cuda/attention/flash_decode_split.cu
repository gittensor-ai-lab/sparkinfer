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
#include <cuda_fp16.h>
#include <climits>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#endif

namespace sparkinfer {
namespace kernels {

__device__ __forceinline__ float fa_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float fa_wsum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// int8_kv: k_pool/v_pool hold int8 and k_scale/v_scale one __half per (token, kv_head) head vector.
template <int HEAD_DIM>
__global__ void fa_split_kernel(
    const __nv_bfloat16* __restrict__ q, const void* __restrict__ k_pool,
    const void* __restrict__ v_pool, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    float scale, int num_q_heads, int num_kv_heads, int block_size, int max_blocks, int n_splits,
    const __half* __restrict__ k_scale, const __half* __restrict__ v_scale, int int8_kv
) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const int seq   = blockIdx.y;
    const int split = blockIdx.x % n_splits;
    const int qh    = blockIdx.x / n_splits;
    const int lane  = threadIdx.x;
    const int kvh   = qh / (num_q_heads / num_kv_heads);
    const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(k_pool);
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(v_pool);
    const signed char* ki = reinterpret_cast<const signed char*>(k_pool);
    const signed char* vi = reinterpret_cast<const signed char*>(v_pool);

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
        const float ks = int8_kv ? __half2float(k_scale[base / HEAD_DIM]) : 0.f;
        const float vs = int8_kv ? __half2float(v_scale[base / HEAD_DIM]) : 0.f;
        float p = 0.f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++)
            p += qr[e] * (int8_kv ? (float)ki[base + lane + e * 32] * ks : fa_to_f(kb[base + lane + e * 32]));
        const float score = fa_wsum(p) * scale;
        const float mn = fmaxf(m, score), corr = __expf(m - mn), pe = __expf(score - mn);
        l = l * corr + pe;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++)
            acc[e] = acc[e] * corr + pe * (int8_kv ? (float)vi[base + lane + e * 32] * vs : fa_to_f(vb[base + lane + e * 32]));
        m = mn;
    }

    const int idx = (seq * num_q_heads + qh) * n_splits + split;
    if (lane == 0) { part_m[idx] = m; part_l[idx] = l; }
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) part_acc[(size_t)idx * HEAD_DIM + lane + e * 32] = acc[e];
}

// GQA-shared split: one block per (seq, kv_head, split) with GQA warps (one per
// q-head in the group). The block stages a KV tile once into shared memory, then
// all GQA warps reuse it. For Qwen's 8:1 GQA this cuts long-context KV global
// reads in the split pass by up to 8x while preserving the same per-q-head
// partials consumed by the existing combine kernel.
// SEQ is how many verify rows one CTA answers. The K/V tile is staged in shared memory once per
// CTA, so at SEQ=1 (decode, and the shipped verify) every row re-streams the whole KV: measured at
// 4k, one launch costs 6.3 + 7.83*rows us, i.e. 88% of it is per-row at 6 rows. Folding S rows into
// a CTA divides the KV traffic by S; the cost is S extra accumulators per thread, which is why the
// fold has to stay narrow.

// Double-buffered KV staging with cp.async. Identical mathematics to fa_split_gqa_kernel below --
// same q registers, same per-token online-softmax update, same partials layout -- but the next
// tile's global->shared copy is issued BEFORE the current tile is consumed, so the load overlaps
// the compute instead of the block stalling on it at every barrier.
//
// Why: the synchronous kernel runs stage -> __syncthreads -> compute -> __syncthreads, so DRAM
// latency is exposed once per tile. At n_splits=1 and 4k that is hundreds of tiles per block, and
// the block sustains only ~7.5 GB/s. Deepening the tile put more loads in flight per barrier
// (+14%) but never overlapped the two phases; this does.
//
// bf16 KV only: cp.async copies bytes verbatim and cannot dequantize, so an int8 pool still needs
// the dequant-into-smem path of the kernel below.
//
// Bit-identical to that kernel: cp.async changes WHEN bytes land in shared memory, not their
// values, and the token walk (start..end, consecutive tiles, per-token update) is unchanged, so
// every row accumulates the same keys in the same order.
template <int HEAD_DIM, int GQA, int TILE, int SEQ = 1>
__global__ void fa_split_gqa_pipe_kernel(
    const __nv_bfloat16* __restrict__ q, const void* __restrict__ k_pool,
    const void* __restrict__ v_pool, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    float scale, int num_q_heads, int num_kv_heads, int block_size, int max_blocks, int n_splits
) {
    constexpr int ELEMS = HEAD_DIM / 32;
    constexpr int TELEMS = TILE * HEAD_DIM;
    const int seq0  = blockIdx.y * SEQ;
    const int split = blockIdx.x % n_splits;
    const int kvh   = blockIdx.x / n_splits;
    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & 31;
    const int qh    = kvh * GQA + warp;

    float qr[SEQ][ELEMS];
    #pragma unroll
    for (int v = 0; v < SEQ; v++) {
        const __nv_bfloat16* qp = q + (size_t)((seq0 + v) * num_q_heads + qh) * HEAD_DIM;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) qr[v][e] = fa_to_f(qp[lane + e * 32]);
    }

    int row_start[SEQ], row_end[SEQ];
    int start = INT_MAX, end = 0;
    #pragma unroll
    for (int v = 0; v < SEQ; v++) {
        const int slv = seq_lens[seq0 + v];
        const int ch  = (slv + n_splits - 1) / n_splits;
        row_start[v] = split * ch;
        row_end[v]   = min(slv, row_start[v] + ch);
        if (row_end[v] > row_start[v]) { start = min(start, row_start[v]); end = max(end, row_end[v]); }
    }
    if (start == INT_MAX) start = end = 0;

    float m[SEQ], l[SEQ], acc[SEQ][ELEMS];
    #pragma unroll
    for (int v = 0; v < SEQ; v++) {
        m[v] = -1e30f; l[v] = 0.f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) acc[v][e] = 0.f;
    }

    extern __shared__ __nv_bfloat16 s_kv[];
    __nv_bfloat16* s_k = s_kv;                       // [2][TILE*HEAD_DIM]
    __nv_bfloat16* s_v = s_kv + (size_t)2 * TELEMS;  // [2][TILE*HEAD_DIM]
    __shared__ size_t s_rowbase[2][TILE];
    const __nv_bfloat16* __restrict__ kb = reinterpret_cast<const __nv_bfloat16*>(k_pool);
    const __nv_bfloat16* __restrict__ vb = reinterpret_cast<const __nv_bfloat16*>(v_pool);

    // Resolve one tile's per-token global row bases into s_rowbase[buf]. Same address math as the
    // synchronous kernel, hoisted to one thread per token.
    auto resolve = [&](int buf, int t0v, int validv) {
        if ((int)threadIdx.x < validv) {
            const int t = t0v + threadIdx.x;
            const int blk = t / block_size, wb = t % block_size;
            const int phys = block_table[seq0 * max_blocks + blk];
            const size_t tokrow = (size_t)(phys * block_size + wb) * num_kv_heads + kvh;
            s_rowbase[buf][threadIdx.x] = tokrow * HEAD_DIM;
        }
    };
    // Issue the async copies for one tile. 16 B per thread-item, matching the uint4 load it replaces.
    auto issue = [&](int buf, int validv) {
        __nv_bfloat16* dk = s_k + (size_t)buf * TELEMS;
        __nv_bfloat16* dv = s_v + (size_t)buf * TELEMS;
        for (int i = threadIdx.x * 8; i < validv * HEAD_DIM; i += blockDim.x * 8) {
            const int within = i / HEAD_DIM, d = i % HEAD_DIM;
            const size_t base = s_rowbase[buf][within] + d;
            __pipeline_memcpy_async(dk + i, kb + base, 16);
            __pipeline_memcpy_async(dv + i, vb + base, 16);
        }
        __pipeline_commit();
    };

    if (start >= end) {
        #pragma unroll
        for (int v = 0; v < SEQ; v++) {
            const int idx = ((seq0 + v) * num_q_heads + qh) * n_splits + split;
            if (lane == 0) { part_m[idx] = m[v]; part_l[idx] = l[v]; }
            #pragma unroll
            for (int e = 0; e < ELEMS; e++) part_acc[(size_t)idx * HEAD_DIM + lane + e * 32] = acc[v][e];
        }
        return;
    }

    int buf = 0;
    resolve(0, start, min(TILE, end - start));
    __syncthreads();
    issue(0, min(TILE, end - start));

    for (int t0 = start; t0 < end; t0 += TILE) {
        const int valid = min(TILE, end - t0);
        const int nt0 = t0 + TILE;
        // Uniform across the block (depends only on t0/end), so the guarded barrier below is not
        // divergent.
        const int nvalid = nt0 < end ? min(TILE, end - nt0) : 0;
        if (nvalid > 0) {
            resolve(buf ^ 1, nt0, nvalid);
            __syncthreads();          // s_rowbase[buf^1] visible before its copies read it
            issue(buf ^ 1, nvalid);
        }
        __pipeline_wait_prior(nvalid > 0 ? 1 : 0);
        __syncthreads();
        const __nv_bfloat16* ck = s_k + (size_t)buf * TELEMS;
        const __nv_bfloat16* cv = s_v + (size_t)buf * TELEMS;
        for (int tt = 0; tt < valid; tt++) {
            float kv_k[ELEMS], kv_v[ELEMS];
            #pragma unroll
            for (int e = 0; e < ELEMS; e++) {
                kv_k[e] = fa_to_f(ck[tt * HEAD_DIM + lane + e * 32]);
                kv_v[e] = fa_to_f(cv[tt * HEAD_DIM + lane + e * 32]);
            }
            const int gtok = t0 + tt;
            #pragma unroll
            for (int v = 0; v < SEQ; v++) {
                if (gtok < row_start[v] || gtok >= row_end[v]) continue;
                float p = 0.f;
                #pragma unroll
                for (int e = 0; e < ELEMS; e++) p += qr[v][e] * kv_k[e];
                const float score = fa_wsum(p) * scale;
                const float mn = fmaxf(m[v], score), corr = __expf(m[v] - mn), pe = __expf(score - mn);
                l[v] = l[v] * corr + pe;
                #pragma unroll
                for (int e = 0; e < ELEMS; e++) acc[v][e] = acc[v][e] * corr + pe * kv_v[e];
                m[v] = mn;
            }
        }
        __syncthreads();
        buf ^= 1;
    }

    #pragma unroll
    for (int v = 0; v < SEQ; v++) {
        const int idx = ((seq0 + v) * num_q_heads + qh) * n_splits + split;
        if (lane == 0) { part_m[idx] = m[v]; part_l[idx] = l[v]; }
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) part_acc[(size_t)idx * HEAD_DIM + lane + e * 32] = acc[v][e];
    }
}


// KV-GROUP SPLIT of the cp.async pipeline kernel.
//
// #874 hid the staging latency but left the shape of the grid alone: at a small split count this
// launcher runs dim3(num_kv_heads * n_splits, num_seqs) = ~4 CTAs of GQA*32 = 192 threads, i.e.
// six warps on four SMs of 170. Measured with a skip-probe (mainloop removed) the walk is still
// ~47% of the decode step, so the work is there -- there are simply almost no warps doing it.
//
// Split the block's KV range across KVG warp-groups instead. Each group walks its own contiguous
// stripe with its own double-buffered tile and its own (m, l, acc), and the groups are merged once
// at the end with the standard log-sum-exp rescale. This multiplies warps per SM by KVG WITHOUT
// touching n_splits, the part_m/part_l/part_acc layout, or the combine kernel -- the block still
// emits exactly one split's partials per q-head.
//
// EXACTNESS. Within a stripe the token order is unchanged, and the cross-group merge folds groups
// in ascending index with a fixed formula, so the result is deterministic run to run. It is a
// different summation ORDER than the single-group walk (a partition of the same keys), so it is
// not bit-identical -- it is the same reassociation the existing n_splits>1 combine already
// performs, and acceptance is unaffected because the draft is untouched.
//
// SHARED MEMORY. sm_120 has 128 KB of L1/shared per SM, ~100 KB addressable as dynamic smem --
// NOT the 228 KB of datacenter Blackwell. KVG=2 at TILE=16 needs 4 * 16 * 256 * 2 B * 2 = 64 KB
// of tiles plus KVG*GQA*HEAD_DIM floats of merge scratch (12 KB), which fits; KVG=4 or TILE>=24
// does not. That cap is why this is a 2-group split and not a 4-group one.
template <int HEAD_DIM, int GQA, int TILE, int KVG>
__global__ void fa_split_gqa_pipeg_kernel(
    const __nv_bfloat16* __restrict__ q, const void* __restrict__ k_pool,
    const void* __restrict__ v_pool, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    float scale, int num_q_heads, int num_kv_heads, int block_size, int max_blocks, int n_splits
) {
    constexpr int ELEMS  = HEAD_DIM / 32;
    constexpr int TELEMS = TILE * HEAD_DIM;
    constexpr int GTHR   = GQA * 32;              // threads per KV group
    const int seq0  = blockIdx.y;
    const int split = blockIdx.x % n_splits;
    const int kvh   = blockIdx.x / n_splits;
    const int grp   = threadIdx.x / GTHR;
    const int gtid  = threadIdx.x - grp * GTHR;
    const int warp  = gtid >> 5;
    const int lane  = gtid & 31;
    const int qh    = kvh * GQA + warp;

    float qr[ELEMS];
    {
        const __nv_bfloat16* qp = q + (size_t)(seq0 * num_q_heads + qh) * HEAD_DIM;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) qr[e] = fa_to_f(qp[lane + e * 32]);
    }

    const int sl    = seq_lens[seq0];
    const int chunk = (sl + n_splits - 1) / n_splits;
    const int start = split * chunk;
    const int end   = min(sl, start + chunk);
    // Every group runs the SAME iteration count so the block-wide barriers stay uniform; a group
    // whose stripe is short simply contributes valid==0 tiles.
    const int span  = end > start ? end - start : 0;
    const int gch   = (span + KVG - 1) / KVG;
    const int gs    = start + grp * gch;
    const int ge    = min(end, gs + gch);
    const int niter = (gch + TILE - 1) / TILE;

    float m = -1e30f, l = 0.f, acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    extern __shared__ __nv_bfloat16 s_kv[];
    __nv_bfloat16* s_k = s_kv + (size_t)grp * 4 * TELEMS;
    __nv_bfloat16* s_v = s_k  + (size_t)2 * TELEMS;
    __shared__ size_t s_rowbase[KVG][2][TILE];
    __shared__ float  s_mm[KVG][GQA], s_ll[KVG][GQA];
    __shared__ float  s_ac[KVG][GQA][HEAD_DIM];
    const __nv_bfloat16* __restrict__ kb = reinterpret_cast<const __nv_bfloat16*>(k_pool);
    const __nv_bfloat16* __restrict__ vb = reinterpret_cast<const __nv_bfloat16*>(v_pool);

    auto resolve = [&](int buf, int t0v, int validv) {
        if (gtid < validv) {
            const int t = t0v + gtid;
            const int blk = t / block_size, wb = t % block_size;
            const int phys = block_table[seq0 * max_blocks + blk];
            const size_t tokrow = (size_t)(phys * block_size + wb) * num_kv_heads + kvh;
            s_rowbase[grp][buf][gtid] = tokrow * HEAD_DIM;
        }
    };
    auto issue = [&](int buf, int validv) {
        __nv_bfloat16* dk = s_k + (size_t)buf * TELEMS;
        __nv_bfloat16* dv = s_v + (size_t)buf * TELEMS;
        for (int i = gtid * 8; i < validv * HEAD_DIM; i += GTHR * 8) {
            const int within = i / HEAD_DIM, d = i % HEAD_DIM;
            const size_t base = s_rowbase[grp][buf][within] + d;
            __pipeline_memcpy_async(dk + i, kb + base, 16);
            __pipeline_memcpy_async(dv + i, vb + base, 16);
        }
        __pipeline_commit();
    };

    int buf = 0;
    {
        const int v0 = ge > gs ? min(TILE, ge - gs) : 0;
        resolve(0, gs, v0);
        __syncthreads();
        issue(0, v0);            // committed even when v0==0, so the group count stays uniform
    }
    for (int it = 0; it < niter; it++) {
        const int t0 = gs + it * TILE;
        const int valid  = (t0 < ge) ? min(TILE, ge - t0) : 0;
        const int nt0    = t0 + TILE;
        const int nvalid = (nt0 < ge) ? min(TILE, ge - nt0) : 0;
        const bool more  = (it + 1 < niter);
        if (more) {
            resolve(buf ^ 1, nt0, nvalid);
            __syncthreads();
            issue(buf ^ 1, nvalid);
        }
        __pipeline_wait_prior(more ? 1 : 0);
        __syncthreads();
        const __nv_bfloat16* ck = s_k + (size_t)buf * TELEMS;
        const __nv_bfloat16* cv = s_v + (size_t)buf * TELEMS;
        for (int tt = 0; tt < valid; tt++) {
            float kv_k[ELEMS], kv_v[ELEMS];
            #pragma unroll
            for (int e = 0; e < ELEMS; e++) {
                kv_k[e] = fa_to_f(ck[tt * HEAD_DIM + lane + e * 32]);
                kv_v[e] = fa_to_f(cv[tt * HEAD_DIM + lane + e * 32]);
            }
            float p = 0.f;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++) p += qr[e] * kv_k[e];
            const float score = fa_wsum(p) * scale;
            const float mn = fmaxf(m, score), corr = __expf(m - mn), pe = __expf(score - mn);
            l = l * corr + pe;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++) acc[e] = acc[e] * corr + pe * kv_v[e];
            m = mn;
        }
        __syncthreads();
        buf ^= 1;
    }

    // Cross-group merge: ascending group index, fixed formula -> deterministic.
    if (lane == 0) { s_mm[grp][warp] = m; s_ll[grp][warp] = l; }
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) s_ac[grp][warp][lane + e * 32] = acc[e];
    __syncthreads();
    if (grp == 0) {
        float mm = -1e30f;
        #pragma unroll
        for (int g = 0; g < KVG; g++) mm = fmaxf(mm, s_mm[g][warp]);
        float ll = 0.f, aa[ELEMS];
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) aa[e] = 0.f;
        #pragma unroll
        for (int g = 0; g < KVG; g++) {
            const float c = __expf(s_mm[g][warp] - mm);
            ll += s_ll[g][warp] * c;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++) aa[e] += s_ac[g][warp][lane + e * 32] * c;
        }
        const int idx = (seq0 * num_q_heads + qh) * n_splits + split;
        if (lane == 0) { part_m[idx] = mm; part_l[idx] = ll; }
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) part_acc[(size_t)idx * HEAD_DIM + lane + e * 32] = aa[e];
    }
}

template <int HEAD_DIM, int GQA, int TILE, bool INT8, int SEQ = 1>
__global__ void fa_split_gqa_kernel(
    const __nv_bfloat16* __restrict__ q, const void* __restrict__ k_pool,
    const void* __restrict__ v_pool, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    float scale, int num_q_heads, int num_kv_heads, int block_size, int max_blocks, int n_splits,
    const __half* __restrict__ k_scale, const __half* __restrict__ v_scale
) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const int seq0  = blockIdx.y * SEQ;
    const int split = blockIdx.x % n_splits;
    const int kvh   = blockIdx.x / n_splits;
    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & 31;
    const int qh    = kvh * GQA + warp;

    float qr[SEQ][ELEMS];
    #pragma unroll
    for (int v = 0; v < SEQ; v++) {
        const __nv_bfloat16* qp = q + (size_t)((seq0 + v) * num_q_heads + qh) * HEAD_DIM;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) qr[v][e] = fa_to_f(qp[lane + e * 32]);
    }

    // The folded rows differ in length (a verify block's row i ends at start_pos+i+1), so each row
    // keeps its OWN chunk/start/end -- exactly the range the SEQ=1 kernel would give it. The CTA
    // stages the union of those ranges (they differ by at most a token or two) and each row masks
    // itself back to its own bounds, so every row accumulates the same keys in the same order as
    // before: the fold is bit-exact by construction, not by luck.
    int row_start[SEQ], row_end[SEQ];
    int start = INT_MAX, end = 0;
    #pragma unroll
    for (int v = 0; v < SEQ; v++) {
        const int slv = seq_lens[seq0 + v];
        const int ch  = (slv + n_splits - 1) / n_splits;
        row_start[v] = split * ch;
        row_end[v]   = min(slv, row_start[v] + ch);
        if (row_end[v] > row_start[v]) { start = min(start, row_start[v]); end = max(end, row_end[v]); }
    }
    if (start == INT_MAX) start = end = 0;

    float m[SEQ], l[SEQ], acc[SEQ][ELEMS];
    #pragma unroll
    for (int v = 0; v < SEQ; v++) {
        m[v] = -1e30f; l[v] = 0.f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) acc[v][e] = 0.f;
    }

    extern __shared__ __nv_bfloat16 s_kv[];
    __nv_bfloat16* s_k = s_kv;
    __nv_bfloat16* s_v = s_kv + (size_t)TILE * HEAD_DIM;
    __shared__ size_t s_rowbase[TILE];   // per-token global row base, resolved once (not per head-dim)
    __shared__ float s_ksc[INT8 ? TILE : 1], s_vsc[INT8 ? TILE : 1];   // int8 only: per-token dequant scales

    for (int t0 = start; t0 < end; t0 += TILE) {
        const int valid = min(TILE, end - t0);
        // Hoist the block-table lookup + address math to ONCE per token (was redundantly
        // recomputed by all HEAD_DIM threads of a token). Byte-identical: same base offsets.
        if ((int)threadIdx.x < valid) {
            const int t = t0 + threadIdx.x;
            const int blk = t / block_size, wb = t % block_size;
            // One staged tile serves every folded row, so the fold requires the rows to share a
            // block table. The verify replicates one table across its rows (launch_broadcast_rows_i32),
            // which is the only caller that folds.
            const int phys = block_table[seq0 * max_blocks + blk];
            const size_t tokrow = (size_t)(phys * block_size + wb) * num_kv_heads + kvh;
            s_rowbase[threadIdx.x] = tokrow * HEAD_DIM;
            if constexpr (INT8) { s_ksc[threadIdx.x] = __half2float(k_scale[tokrow]); s_vsc[threadIdx.x] = __half2float(v_scale[tokrow]); }
        }
        __syncthreads();
        if constexpr (!INT8) {
            // Vectorized load: uint4 (8×bf16) via __ldg into bf16 smem. __restrict__ recast keeps the
            // no-alias load codegen identical to the pre-int8 (main) kernel — bf16 guard contexts unchanged.
            const __nv_bfloat16* __restrict__ kb = reinterpret_cast<const __nv_bfloat16*>(k_pool);
            const __nv_bfloat16* __restrict__ vb = reinterpret_cast<const __nv_bfloat16*>(v_pool);
            for (int i = threadIdx.x * 8; i < valid * HEAD_DIM; i += blockDim.x * 8) {
                const int within = i / HEAD_DIM, d = i % HEAD_DIM;
                const size_t base = s_rowbase[within] + d;
                *reinterpret_cast<uint4*>(s_k + i) = __ldg(reinterpret_cast<const uint4*>(kb + base));
                *reinterpret_cast<uint4*>(s_v + i) = __ldg(reinterpret_cast<const uint4*>(vb + base));
            }
        } else {
            // int8: load 8 int8 (int2) + per-token scale, dequant to bf16 into smem (dot loop unchanged).
            const signed char* __restrict__ ki = reinterpret_cast<const signed char*>(k_pool);
            const signed char* __restrict__ vi = reinterpret_cast<const signed char*>(v_pool);
            for (int i = threadIdx.x * 8; i < valid * HEAD_DIM; i += blockDim.x * 8) {
                const int within = i / HEAD_DIM, d = i % HEAD_DIM;
                const size_t base = s_rowbase[within] + d;
                const float ks = s_ksc[within], vs = s_vsc[within];
                const int2 kr = __ldg(reinterpret_cast<const int2*>(ki + base));
                const int2 vr = __ldg(reinterpret_cast<const int2*>(vi + base));
                const signed char* kc = reinterpret_cast<const signed char*>(&kr);
                const signed char* vc = reinterpret_cast<const signed char*>(&vr);
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    s_k[i + j] = __float2bfloat16((float)kc[j] * ks);
                    s_v[i + j] = __float2bfloat16((float)vc[j] * vs);
                }
            }
        }
        __syncthreads();
        for (int tt = 0; tt < valid; tt++) {
            // One shared-memory K/V read serves every folded row.
            float kv_k[ELEMS], kv_v[ELEMS];
            #pragma unroll
            for (int e = 0; e < ELEMS; e++) {
                kv_k[e] = fa_to_f(s_k[tt * HEAD_DIM + lane + e * 32]);
                kv_v[e] = fa_to_f(s_v[tt * HEAD_DIM + lane + e * 32]);
            }
            const int gtok = t0 + tt;
            #pragma unroll
            for (int v = 0; v < SEQ; v++) {
                if (gtok < row_start[v] || gtok >= row_end[v]) continue;   // this row's own range
                float p = 0.f;
                #pragma unroll
                for (int e = 0; e < ELEMS; e++) p += qr[v][e] * kv_k[e];
                const float score = fa_wsum(p) * scale;
                const float mn = fmaxf(m[v], score), corr = __expf(m[v] - mn), pe = __expf(score - mn);
                l[v] = l[v] * corr + pe;
                #pragma unroll
                for (int e = 0; e < ELEMS; e++) acc[v][e] = acc[v][e] * corr + pe * kv_v[e];
                m[v] = mn;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int v = 0; v < SEQ; v++) {
        const int idx = ((seq0 + v) * num_q_heads + qh) * n_splits + split;
        if (lane == 0) { part_m[idx] = m[v]; part_l[idx] = l[v]; }
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) part_acc[(size_t)idx * HEAD_DIM + lane + e * 32] = acc[v][e];
    }
}

// llama Q8_1 activation block (matches si_block_q8_1 used by the int8 MMVQ O-projection).
struct fa_block_q8_1 { __half2 ds; signed char qs[32]; };

// Combine the split partials with DG x NW parallelism over the 1-block-per-head
// original (which idled at ~2% occupancy with a serial n_splits loop). DG head-dim
// groups -> DG x more blocks; NW warps per block each fold a 1/NW stripe of the
// splits, then a shared-memory log-sum-exp merge across warps. grid=(heads*DG,seqs).
// When out_q8 != nullptr AND ELEMS==1 (DG*32==HEAD_DIM), each (qh,dg) block's warp 0 also
// emits the Q8_1 block for attn dims [qh*HEAD_DIM + dg*32, +32) from the bf16-rounded output,
// so the O-projection MMVQ skips its standalone attn-quantize node (bit-identical to running
// the quantizer on `out` afterwards). Q8_1 block index = qh*(HEAD_DIM/32) + dg.
template <int HEAD_DIM, int DG, int NW>
__global__ void fa_combine_kernel(
    const float* __restrict__ part_m, const float* __restrict__ part_l,
    const float* __restrict__ part_acc, __nv_bfloat16* __restrict__ out,
    int num_q_heads, int n_splits, fa_block_q8_1* __restrict__ out_q8 = nullptr
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

    // Fused Q8_1(attn) emit for the O-projection MMVQ (only the DG*32==HEAD_DIM layout, ELEMS==1,
    // where warp 0's 32 lanes hold exactly the 32 elements of one Q8_1 block).
    if (out_q8 != nullptr && ELEMS == 1) {
        const float bv = __bfloat162float(__float2bfloat16(acc[0] * inv));   // bf16-rounded, as `out`
        float amax = fabsf(bv);
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
        const float d = amax / 127.0f;
        const int qi = (amax == 0.0f) ? 0 : (int)roundf(bv / d);
        const int blk = (seq * num_q_heads + qh) * (HEAD_DIM / 32) + dg;
        out_q8[blk].qs[lane] = (signed char)qi;
        int s = qi;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
        if (lane == 0) out_q8[blk].ds = __floats2half2_rn(d, d * (float)s);
    }
}

// hd256 gated-Q: fold mul_sigmoid + O-proj Q8_1 quant into the combine tail (distinct from a
// standalone mul_sigmoid_q8 kernel — gate is applied inside the split-fold, per-dim).
template <int HEAD_DIM, int DG, int NW>
__global__ void fa_combine_gated_q8_kernel(
    const float* __restrict__ part_m, const float* __restrict__ part_l,
    const float* __restrict__ part_acc, __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ gate, int num_q_heads, int n_splits,
    fa_block_q8_1* __restrict__ out_q8
) {
    constexpr int ELEMS = HEAD_DIM / (32 * DG);
    const int seq = blockIdx.y, qh = blockIdx.x / DG, dg = blockIdx.x % DG;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int idxbase = (seq * num_q_heads + qh) * n_splits;
    const int doff = dg * (HEAD_DIM / DG) + lane;

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
    const size_t hbase = (size_t)(seq * num_q_heads + qh) * HEAD_DIM;
    __nv_bfloat16* op = out + hbase;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) {
        const int di = doff + e * 32;
        const float gated = __bfloat162float(__float2bfloat16(acc[e] * inv))
                          * (1.f / (1.f + __expf(-__bfloat162float(gate[hbase + di]))));
        const float bv = __bfloat162float(__float2bfloat16(gated));
        op[di] = __float2bfloat16(bv);
        float amax = fabsf(bv);
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
        const float d = amax / 127.0f;
        const int qi = (amax == 0.0f) ? 0 : (int)roundf(bv / d);
        const int blk = (seq * num_q_heads + qh) * (HEAD_DIM / 32) + di / 32;
        out_q8[blk].qs[lane] = (signed char)qi;
        int ssum = qi;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) ssum += __shfl_xor_sync(0xffffffffu, ssum, m);
        if (lane == 0) out_q8[blk].ds = __floats2half2_rn(d, d * (float)ssum);
    }
}

#ifndef FA_COMBINE_DG
#define FA_COMBINE_DG 4     // head-dim groups (DG x blocks); sweepable
#endif
#ifndef FA_COMBINE_NW
#define FA_COMBINE_NW 4     // warps/block folding the split stripes; sweepable
#endif
#ifndef FA_GQA_TILE
#define FA_GQA_TILE 14      // bf16 smem + uint4 ldg sweet spot at n_splits=128
#endif
#ifndef FA_GQA6_TILE
#define FA_GQA6_TILE 8     // Qwen3.8-27B hd256 GQA-6 (24Q/4KV); independently sweepable
#endif
// Deep tile for the SMALL-split regime. FA_GQA6_TILE above is a sweet spot at n_splits=128, where a
// block walks seqlen/n_splits ~= 32 keys and the staging loop runs two or three times. When the
// split count is small the same block walks the WHOLE range instead: at 4k with n_splits=1 that is
// 512 stage/sync/compute/sync cycles, each issuing only blockDim*16 B of loads before its barrier,
// so DRAM latency is exposed every iteration -- measured 1.11 ms/call, ~30 GB/s across 4 CTAs
// (~7.5 GB/s each) against a 18.7 us roofline for the bytes it reads. Staging a deeper tile puts
// proportionally more independent loads in flight per barrier and cuts the barrier count by the
// same factor. Capped by dynamic smem: 2 * TILE * 256 * 2 B = 1024*TILE, and this launcher never
// opts in past the 48 KB default, so 32 -> 32 KB is safe and 48 would not be.
#ifndef FA_GQA6_TILE_DEEP
#define FA_GQA6_TILE_DEEP 32
#endif
#ifndef FA_GQA4_TILE
#define FA_GQA4_TILE 8     // Qwythos hd256 GQA-4 (16Q/4KV); independently sweepable
#endif
#ifndef _MSC_VER
template __global__ void fa_split_kernel<128>(const __nv_bfloat16*, const void*, const void*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int, const __half*, const __half*, int);
#endif
#ifndef _MSC_VER
template __global__ void fa_split_gqa_kernel<128, 8, FA_GQA_TILE, false>(const __nv_bfloat16*, const void*, const void*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int, const __half*, const __half*);
#endif
#ifndef _MSC_VER
template __global__ void fa_split_gqa_kernel<128, 8, FA_GQA_TILE, true>(const __nv_bfloat16*, const void*, const void*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int, const __half*, const __half*);
#endif
#ifndef _MSC_VER
template __global__ void fa_combine_kernel<128, FA_COMBINE_DG, FA_COMBINE_NW>(const float*, const float*, const float*, __nv_bfloat16*, int, int, fa_block_q8_1*);
#endif
#ifndef _MSC_VER
template __global__ void fa_combine_kernel<128, FA_COMBINE_DG, 8>(const float*, const float*, const float*, __nv_bfloat16*, int, int, fa_block_q8_1*);
#endif
#ifndef _MSC_VER
template __global__ void fa_combine_kernel<128, FA_COMBINE_DG, 16>(const float*, const float*, const float*, __nv_bfloat16*, int, int, fa_block_q8_1*);
#endif
// Qwen3.6 full-attention head_dim=256 (bf16 KV): GQA-8 split + scalar fallback.
#ifndef _MSC_VER
template __global__ void fa_split_kernel<256>(const __nv_bfloat16*, const void*, const void*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int, const __half*, const __half*, int);
#endif
#ifndef _MSC_VER
template __global__ void fa_split_gqa_kernel<256, 8, FA_GQA_TILE, false>(const __nv_bfloat16*, const void*, const void*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int, const __half*, const __half*);
#endif
#ifndef _MSC_VER
template __global__ void fa_split_gqa_kernel<256, 8, FA_GQA_TILE, true>(const __nv_bfloat16*, const void*, const void*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int, const __half*, const __half*);
#endif
#ifndef _MSC_VER
template __global__ void fa_split_gqa_kernel<256, 4, FA_GQA4_TILE, false>(const __nv_bfloat16*, const void*, const void*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int, const __half*, const __half*);
#endif
#ifndef _MSC_VER
template __global__ void fa_split_gqa_kernel<256, 4, FA_GQA4_TILE, true>(const __nv_bfloat16*, const void*, const void*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int, const __half*, const __half*);
#endif
#ifndef _MSC_VER
template __global__ void fa_combine_kernel<256, FA_COMBINE_DG, FA_COMBINE_NW>(const float*, const float*, const float*, __nv_bfloat16*, int, int, fa_block_q8_1*);
#endif
#ifndef _MSC_VER
template __global__ void fa_combine_kernel<256, FA_COMBINE_DG, 8>(const float*, const float*, const float*, __nv_bfloat16*, int, int, fa_block_q8_1*);
#endif
#ifndef _MSC_VER
template __global__ void fa_combine_kernel<256, FA_COMBINE_DG, 16>(const float*, const float*, const float*, __nv_bfloat16*, int, int, fa_block_q8_1*);
#endif
#ifndef _MSC_VER
template __global__ void fa_combine_gated_q8_kernel<256, FA_COMBINE_DG, FA_COMBINE_NW>(
    const float*, const float*, const float*, __nv_bfloat16*, const __nv_bfloat16*, int, int, fa_block_q8_1*);
#endif
// head_dim=128 gated combine. The kernel is HEAD_DIM-generic (ELEMS = HEAD_DIM/(32*DG) = 1 here);
// only the hd256 architectures had ever been instantiated, which is what kept Muse Glimmer (128)
// on the split path -- a separate sigmoid-gate kernel plus a separate output quantize, two extra
// graph nodes per layer, 104 per step.
#ifndef _MSC_VER
template __global__ void fa_combine_gated_q8_kernel<128, FA_COMBINE_DG, FA_COMBINE_NW>(
    const float*, const float*, const float*, __nv_bfloat16*, const __nv_bfloat16*, int, int, fa_block_q8_1*);
#endif
#ifndef _MSC_VER
template __global__ void fa_combine_gated_q8_kernel<128, FA_COMBINE_DG, 8>(
    const float*, const float*, const float*, __nv_bfloat16*, const __nv_bfloat16*, int, int, fa_block_q8_1*);
#endif
#ifndef _MSC_VER
template __global__ void fa_combine_gated_q8_kernel<128, FA_COMBINE_DG, 16>(
    const float*, const float*, const float*, __nv_bfloat16*, const __nv_bfloat16*, int, int, fa_block_q8_1*);
#endif
#ifndef _MSC_VER
template __global__ void fa_combine_gated_q8_kernel<256, FA_COMBINE_DG, 8>(
    const float*, const float*, const float*, __nv_bfloat16*, const __nv_bfloat16*, int, int, fa_block_q8_1*);
#endif
#ifndef _MSC_VER
template __global__ void fa_combine_gated_q8_kernel<256, FA_COMBINE_DG, 16>(
    const float*, const float*, const float*, __nv_bfloat16*, const __nv_bfloat16*, int, int, fa_block_q8_1*);
#endif
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/attention.h"
#include <mma.h>

// hd256 GQA-4 MMA needs 8 warps (128-token KV groups) even though only 4 q-rows are live.
template <int HEAD_DIM, int GQA> struct fa_mma_block_threads { static constexpr int v = GQA * 32; };
template <> struct fa_mma_block_threads<256, 4> { static constexpr int v = 256; };
template <> struct fa_mma_block_threads<256, 6> { static constexpr int v = 256; };

// Tensor-core (wmma int8) GQA flash-decode split for long context. The 8 GQA q-heads of a kv-head are
// the batch (M) dim, so S = Q·Kᵀ and O = P·V become small matmuls on the tensor cores, replacing the
// per-lane FMA + 5-shuffle fa_wsum reduction that dominates the scalar kernel at long context. K/V are
// int8 with one fp16 scale per (token, kv_head) head vector. Q is quantized per-q-head and P (with the
// per-token V scale folded in) per-row, so QK and PV run on int8 tensor cores (int32 accumulate); the
// per-token/per-head fp16 scales are applied to the int32 results. This halves the KV global read (the
// bottleneck) and uses 2x-throughput int8 tensor cores. M is padded 8->16; partials (m,l,acc) stay
// byte-compatible with the combine kernel. sm_80+ (wmma). One block per (seq, kv_head, split); 8 warps.
template <int HEAD_DIM, int GQA>
__global__ void __launch_bounds__(fa_mma_block_threads<HEAD_DIM, GQA>::v, 5) fa_split_gqa_mma_i8_kernel(
    const __nv_bfloat16* __restrict__ q, const signed char* __restrict__ k_pool,
    const signed char* __restrict__ v_pool, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    float scale, int num_q_heads, int num_kv_heads, int block_size, int max_blocks, int n_splits,
    const __half* __restrict__ k_scale, const __half* __restrict__ v_scale
) {
    using namespace nvcuda::wmma;
    constexpr int KH = HEAD_DIM / 16;
    const int seq = blockIdx.y, split = blockIdx.x % n_splits, kvh = blockIdx.x / n_splits;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    const int sl = seq_lens[seq];
    const int chunk = (sl + n_splits - 1) / n_splits;
    const int start = split * chunk, end = min(sl, start + chunk);
    const size_t KVLD = (size_t)num_kv_heads * HEAD_DIM;   // int8 token stride in the pool
    const int SLD = num_kv_heads;                          // scale stride (one per token, kv_head)

    extern __shared__ char i8smem[];
    signed char* s_qi = reinterpret_cast<signed char*>(i8smem);       // [16][HD] quantized Q
    signed char* s_pi = s_qi + 16 * HEAD_DIM;                         // [16][HD] quantized P'
    float* s_s  = reinterpret_cast<float*>(s_pi + 16 * HEAD_DIM);     // [16][HD] scores / int32 mma scratch
    float* s_o  = s_s + 16 * HEAD_DIM;                                // [GQA][HD] running O (pad rows dropped)
    float* s_qs = s_o + GQA * HEAD_DIM;                               // [16] Q scale
    float* s_ps = s_qs + 16;                                          // [16] P' row scale
    float* s_ks = s_ps + 16;                                          // [128] group K scales
    float* s_vs = s_ks + 128;                                         // [128] group V scales
    float* s_m  = s_vs + 128;                                         // [16]
    float* s_l  = s_m + 16;                                           // [16]

    // Quantize Q per q-head row (warp w owns rows 2w, 2w+1; rows >= GQA are zero pad).
    // EPT spans the whole head vector: 4 elems/lane at hd128, 8 at hd256. Hardcoding 4 left
    // s_qi dims 128..255 uninitialized at hd256 (and computed amax over half the row), so the
    // QK mma k-tiles 8..15 multiplied against stale shared memory.
    constexpr int EPT = HEAD_DIM / 32;
    #pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        const int r = warp * 2 + rr;
        float qv[EPT], amax = 0.f;
        #pragma unroll
        for (int e = 0; e < EPT; e++) {
            qv[e] = (r < GQA) ? __bfloat162float(q[(size_t)(seq * num_q_heads + kvh * GQA + r) * HEAD_DIM + lane + e * 32]) : 0.f;
            amax = fmaxf(amax, fabsf(qv[e]));
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        const float d = amax / 127.0f;
        if (lane == 0) s_qs[r] = d;
        #pragma unroll
        for (int e = 0; e < EPT; e++)
            s_qi[r * HEAD_DIM + lane + e * 32] = (signed char)((amax == 0.f) ? 0 : (int)roundf(qv[e] / d));
    }
    for (int i = tid; i < GQA * HEAD_DIM; i += blockDim.x) s_o[i] = 0.f;
    if (tid < 16) { s_m[tid] = -1e30f; s_l[tid] = 0.f; }
    __syncthreads();

    const int first_blk = start / 16;
    const int nblk = (end > start) ? ((end - 1) / 16 - first_blk + 1) : 0;
    for (int g0 = 0; g0 < nblk; g0 += 8) {
        const int gblk = min(8, nblk - g0);
        const int gbase = (first_blk + g0) * 16;
        for (int j = tid; j < gblk * 16; j += blockDim.x) {   // stage per-token K/V scales for the group
            const int lb = first_blk + g0 + j / 16, within = j & 15;
            const int pb = block_table[seq * max_blocks + lb];
            const size_t si = (size_t)(pb * 16 + within) * SLD + kvh;
            s_ks[j] = __half2float(k_scale[si]);
            s_vs[j] = __half2float(v_scale[si]);
        }
        // No barrier: staged s_ks/s_vs are first read in the softmax, fenced by the post-QK-mma
        // __syncthreads below; the QK mma reads only s_qi and global KV, not the staged scales.

        // QK int8 mma -> int32; scale to float scores in s_s.
        if (warp < gblk) {
            const int pb = block_table[seq * max_blocks + first_blk + g0 + warp];
            const signed char* kb = k_pool + ((size_t)pb * 16 * num_kv_heads + kvh) * HEAD_DIM;
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
            // ldm = 128: the QK result is a [16 q-rows x up-to-128 tokens] score tile, so its row
            // stride is the group token width (128), not HEAD_DIM — the two only coincide at
            // hd128. With HEAD_DIM as ldm, the hd256 instantiation stored rows 256 apart while
            // the softmax below reads them 128 apart: rows interleave with garbage, and every
            // decoded token past the mma-engagement depth is wrong (verified: 100% argmax
            // divergence vs the exact tile path at >16k on Qwen3.6).
            store_matrix_sync(reinterpret_cast<int*>(s_s) + warp * 16, cf, 128, mem_row_major);
        }
        __syncthreads();
        // Read the raw int32 QK scores directly and apply the per-row/per-token scales inline in the
        // softmax below — this deletes a full 16x128 shared int32->float round-trip and one
        // __syncthreads per KV group (the flash-decode is latency-bound at high n_splits, so a barrier
        // matters). Math is bit-identical: ((int * q_scale) * k_scale) * softmax_scale, same order.
        const int* s_si = reinterpret_cast<const int*>(s_s);

        // Online softmax; fold V scale into P', quantize P' per-row into s_pi.
        #pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int r = warp * 2 + rr;
            // Cache this lane's 4 scaled QK scores (t = lane + u*32) once, reuse for max AND exp —
            // avoids reading s_si + re-applying the 3 scales twice. Invalid/masked positions get the
            // -inf sentinel so they drop out of the max and yield p=0 in the exp (no s_vs garbage read).
            float sc[4], mx = -1e30f;
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                const int t = lane + u * 32, gtok = gbase + t;
                sc[u] = (t < gblk * 16 && gtok >= start && gtok < end)
                        ? (float)s_si[r * 128 + t] * s_qs[r] * s_ks[t] * scale : -1e30f;
                mx = fmaxf(mx, sc[u]);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
            const float m_old = s_m[r], m_new = fmaxf(m_old, mx), corr = __expf(m_old - m_new);
            float sum = 0.f, pamax = 0.f;
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                const int t = lane + u * 32;
                float pv = 0.f;
                if (sc[u] > -1e29f) {
                    const float p = __expf(sc[u] - m_new);
                    sum += p; pv = p * s_vs[t]; pamax = fmaxf(pamax, fabsf(pv));
                }
                s_s[r * 128 + t] = pv;   // stash P' (score no longer needed for this row)
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) { sum += __shfl_xor_sync(0xffffffff, sum, o); pamax = fmaxf(pamax, __shfl_xor_sync(0xffffffff, pamax, o)); }
            const float pd = pamax / 127.0f;
            if (lane == 0) { s_m[r] = m_new; s_l[r] = s_l[r] * corr + sum; s_ps[r] = pd; }
            // Only quantize the gblk*16 P' columns the PV mma actually reads (it loops ks < gblk);
            // the tail columns are never loaded, so skipping them trims the per-row roundf work.
            for (int t = lane; t < gblk * 16; t += 32)
                s_pi[r * 128 + t] = (signed char)((pamax == 0.f) ? 0 : (int)roundf(s_s[r * 128 + t] / pd));
            if (r < GQA) for (int c = lane; c < HEAD_DIM; c += 32) s_o[r * HEAD_DIM + c] *= corr;
        }
        __syncthreads();

        // PV int8 mma -> int32; O += int32 * p_scale[m]. The 8 warps cover a 128-wide dim slab
        // per pass (warp*16 each), so hd128 takes one pass and hd256 two (dh = 0, 128). The
        // hd256 instantiation previously ran a single pass with HEAD_DIM strides: it computed
        // only dims 0..127 of O (128..255 stayed at their zero init) and read the 128-stride
        // P' rows at the wrong ldm — both fixed here; ldm for the P' fragment and the int32
        // store is 128 (the token/slab width), which coincided with HEAD_DIM only at hd128.
        for (int dh = 0; dh < HEAD_DIM; dh += 128) {
            fragment<accumulator, 16, 16, 16, int> cf;
            fill_fragment(cf, 0);
            for (int ks = 0; ks < gblk; ks++) {
                const int pb = block_table[seq * max_blocks + first_blk + g0 + ks];
                const signed char* vb = v_pool + ((size_t)pb * 16 * num_kv_heads + kvh) * HEAD_DIM + dh + warp * 16;
                fragment<matrix_a, 16, 16, 16, signed char, row_major> af;
                fragment<matrix_b, 16, 16, 16, signed char, row_major> bf;
                load_matrix_sync(af, s_pi + ks * 16, 128);
                load_matrix_sync(bf, vb, KVLD);
                mma_sync(cf, af, bf, cf);
            }
            store_matrix_sync(reinterpret_cast<int*>(s_s) + warp * 16, cf, 128, mem_row_major);
            __syncthreads();
            // Only the GQA real q-head rows are kept (rows GQA..15 are wmma M-padding, never
            // written to the partials) — accumulate this 128-wide slab into s_o at its dh offset.
            for (int i = tid; i < GQA * 128; i += blockDim.x)
                s_o[(i >> 7) * HEAD_DIM + dh + (i & 127)] += (float)reinterpret_cast<int*>(s_s)[i] * s_ps[i >> 7];
            __syncthreads();
        }
    }

    for (int r = 0; r < GQA; r++) {
        const int qh = kvh * GQA + r;
        const int idx = (seq * num_q_heads + qh) * n_splits + split;
        if (tid == 0) { part_m[idx] = s_m[r]; part_l[idx] = s_l[r]; }
        for (int c = tid; c < HEAD_DIM; c += blockDim.x)
            part_acc[(size_t)idx * HEAD_DIM + c] = s_o[r * HEAD_DIM + c];
    }
}
#ifndef _MSC_VER
template __global__ void fa_split_gqa_mma_i8_kernel<128, 8>(const __nv_bfloat16*, const signed char*,
    const signed char*, const int*, const int*, float*, float*, float*, float, int, int, int, int, int,
    const __half*, const __half*);
#endif
// Qwen3.6 full-attention head_dim=256 (hybrid). The kernel is HEAD_DIM-generic (KH=HEAD_DIM/16); this
// instantiation moves the 10 full-attn layers onto int8-KV tensor cores, halving their KV read at
// long context. i8 smem = ~33 KB (< 48 KB dynamic cap; 5 blocks/SM fits the 5090's ~228 KB).
#ifndef _MSC_VER
template __global__ void fa_split_gqa_mma_i8_kernel<256, 8>(const __nv_bfloat16*, const signed char*,
    const signed char*, const int*, const int*, float*, float*, float*, float, int, int, int, int, int,
    const __half*, const __half*);
#endif
// Qwythos full-attn: 16Q/4KV hd256 — same MMA kernel, 8 warps for 128-wide KV groups.
#ifndef _MSC_VER
template __global__ void fa_split_gqa_mma_i8_kernel<256, 4>(const __nv_bfloat16*, const signed char*,
    const signed char*, const int*, const int*, float*, float*, float*, float, int, int, int, int, int,
    const __half*, const __half*);
#endif
// Qwen3.8-27B full-attn: 24Q/4KV hd256 — same kernel again, M = 6 rows padded to the 16-row mma.
#ifndef _MSC_VER
template __global__ void fa_split_gqa_mma_i8_kernel<256, 6>(const __nv_bfloat16*, const signed char*,
    const signed char*, const int*, const int*, float*, float*, float*, float, int, int, int, int, int,
    const __half*, const __half*);
#endif
template <int NW>
static inline void fa_launch_combine(
    const float* part_m, const float* part_l, const float* part_acc,
    __nv_bfloat16* out, int num_q_heads, int n_splits, fa_block_q8_1* out_q8,
    int num_seqs, cudaStream_t stream
) {
    dim3 g(num_q_heads * FA_COMBINE_DG, num_seqs);
    fa_combine_kernel<128, FA_COMBINE_DG, NW><<<g, NW * 32, 0, stream>>>(
        part_m, part_l, part_acc, out, num_q_heads, n_splits, out_q8);
}

template <int NW>
static inline void fa_launch_combine_gated(
    const float* part_m, const float* part_l, const float* part_acc,
    __nv_bfloat16* out, const __nv_bfloat16* gate, int num_q_heads, int n_splits,
    fa_block_q8_1* out_q8, int num_seqs, cudaStream_t stream
) {
    dim3 g(num_q_heads * FA_COMBINE_DG, num_seqs);
    fa_combine_gated_q8_kernel<128, FA_COMBINE_DG, NW><<<g, NW * 32, 0, stream>>>(
        part_m, part_l, part_acc, out, gate, num_q_heads, n_splits, out_q8);
}

static inline void fa_launch_combine_gated_dispatch(
    const float* part_m, const float* part_l, const float* part_acc,
    __nv_bfloat16* out, const __nv_bfloat16* gate, int num_q_heads, int n_splits,
    fa_block_q8_1* out_q8, int num_seqs, cudaStream_t stream
) {
    if (n_splits >= 128)      fa_launch_combine_gated<16>(part_m, part_l, part_acc, out, gate, num_q_heads, n_splits, out_q8, num_seqs, stream);
    else if (n_splits >= 64)  fa_launch_combine_gated<8>(part_m, part_l, part_acc, out, gate, num_q_heads, n_splits, out_q8, num_seqs, stream);
    else                      fa_launch_combine_gated<FA_COMBINE_NW>(part_m, part_l, part_acc, out, gate, num_q_heads, n_splits, out_q8, num_seqs, stream);
}

static inline void fa_launch_combine_dispatch(
    const float* part_m, const float* part_l, const float* part_acc,
    __nv_bfloat16* out, int num_q_heads, int n_splits, fa_block_q8_1* out_q8,
    int num_seqs, cudaStream_t stream
) {
    if (n_splits >= 128)      fa_launch_combine<16>(part_m, part_l, part_acc, out, num_q_heads, n_splits, out_q8, num_seqs, stream);
    else if (n_splits >= 64)  fa_launch_combine<8>(part_m, part_l, part_acc, out, num_q_heads, n_splits, out_q8, num_seqs, stream);
    else                      fa_launch_combine<FA_COMBINE_NW>(part_m, part_l, part_acc, out, num_q_heads, n_splits, out_q8, num_seqs, stream);
}

template <int NW>
static inline void fa_launch_combine_hd256(
    const float* part_m, const float* part_l, const float* part_acc,
    __nv_bfloat16* out, int num_q_heads, int n_splits, fa_block_q8_1* out_q8,
    int num_seqs, cudaStream_t stream
) {
    dim3 g(num_q_heads * FA_COMBINE_DG, num_seqs);
    fa_combine_kernel<256, FA_COMBINE_DG, NW><<<g, NW * 32, 0, stream>>>(
        part_m, part_l, part_acc, out, num_q_heads, n_splits, out_q8);
}
template <int NW>
static inline void fa_launch_combine_gated_hd256(
    const float* part_m, const float* part_l, const float* part_acc,
    __nv_bfloat16* out, const __nv_bfloat16* gate, int num_q_heads, int n_splits,
    fa_block_q8_1* out_q8, int num_seqs, cudaStream_t stream
) {
    dim3 g(num_q_heads * FA_COMBINE_DG, num_seqs);
    fa_combine_gated_q8_kernel<256, FA_COMBINE_DG, NW><<<g, NW * 32, 0, stream>>>(
        part_m, part_l, part_acc, out, gate, num_q_heads, n_splits, out_q8);
}
static inline void fa_launch_combine_dispatch_hd256(
    const float* part_m, const float* part_l, const float* part_acc,
    __nv_bfloat16* out, int num_q_heads, int n_splits, fa_block_q8_1* out_q8,
    int num_seqs, cudaStream_t stream
) {
    if (n_splits >= 128)      fa_launch_combine_hd256<16>(part_m, part_l, part_acc, out, num_q_heads, n_splits, out_q8, num_seqs, stream);
    else if (n_splits >= 64)  fa_launch_combine_hd256<8>(part_m, part_l, part_acc, out, num_q_heads, n_splits, out_q8, num_seqs, stream);
    else                      fa_launch_combine_hd256<FA_COMBINE_NW>(part_m, part_l, part_acc, out, num_q_heads, n_splits, out_q8, num_seqs, stream);
}
static inline void fa_launch_combine_gated_dispatch_hd256(
    const float* part_m, const float* part_l, const float* part_acc,
    __nv_bfloat16* out, const __nv_bfloat16* gate, int num_q_heads, int n_splits,
    fa_block_q8_1* out_q8, int num_seqs, cudaStream_t stream
) {
    if (n_splits >= 128)      fa_launch_combine_gated_hd256<16>(part_m, part_l, part_acc, out, gate, num_q_heads, n_splits, out_q8, num_seqs, stream);
    else if (n_splits >= 64)  fa_launch_combine_gated_hd256<8>(part_m, part_l, part_acc, out, gate, num_q_heads, n_splits, out_q8, num_seqs, stream);
    else                      fa_launch_combine_gated_hd256<FA_COMBINE_NW>(part_m, part_l, part_acc, out, gate, num_q_heads, n_splits, out_q8, num_seqs, stream);
}

// Standalone hd256 combine (sparse-KV path: split then combine). num_seqs=1 (decode).
// attn_gate: optional per-element sigmoid output gate — same contract as the gated combine
// inside launch_flash_decode_split (gate && out_q8 selects the gated kernel).
void launch_fa_combine_hd256(
    const float* part_m, const float* part_l, const float* part_acc, void* out,
    int num_q_heads, int n_splits, void* out_q8, cudaStream_t stream,
    const void* attn_gate
) {
    const __nv_bfloat16* gate = reinterpret_cast<const __nv_bfloat16*>(attn_gate);
    if (gate && out_q8)
        fa_launch_combine_gated_dispatch_hd256(part_m, part_l, part_acc,
            reinterpret_cast<__nv_bfloat16*>(out), gate, num_q_heads, n_splits,
            reinterpret_cast<fa_block_q8_1*>(out_q8), 1, stream);
    else
        fa_launch_combine_dispatch_hd256(part_m, part_l, part_acc,
            reinterpret_cast<__nv_bfloat16*>(out), num_q_heads, n_splits,
            reinterpret_cast<fa_block_q8_1*>(out_q8), 1, stream);
}

void launch_flash_decode_split(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens, void* out,
    float* part_m, float* part_l, float* part_acc,
    int num_seqs, int num_q_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks, int n_splits, float scale, cudaStream_t stream,
    void* out_q8, int seqlen, const void* k_scale, const void* v_scale, int int8_kv,
    const void* attn_gate, int gated_combine_hd128
) {
    const __nv_bfloat16* gate = reinterpret_cast<const __nv_bfloat16*>(attn_gate);
    // hd256 already had a gated combine; hd128 only gets one when the caller explicitly opts in,
    // so every pre-existing model keeps byte-for-byte the dispatch it had before.
    const bool gate128 = gated_combine_hd128 && gate && out_q8;
    auto combine_hd256 = [&](void* oq8) {
        if (gate && oq8)
            fa_launch_combine_gated_dispatch_hd256(part_m, part_l, part_acc,
                reinterpret_cast<__nv_bfloat16*>(out), gate, num_q_heads, n_splits,
                reinterpret_cast<fa_block_q8_1*>(oq8), num_seqs, stream);
        else
            fa_launch_combine_dispatch_hd256(part_m, part_l, part_acc,
                reinterpret_cast<__nv_bfloat16*>(out), num_q_heads, n_splits,
                reinterpret_cast<fa_block_q8_1*>(oq8), num_seqs, stream);
    };
    // Qwen3.6 full-attention layers run head_dim=256 (bf16 KV). Use the GQA-8 shared-KV tile
    // path (same 8:1 grouping as Qwen3 hd=128) — cuts KV global reads ~8x vs one-warp-per-q-head.
    if (head_dim == 256) {
        dim3 g2(num_q_heads * FA_COMBINE_DG, num_seqs);
        // int8-KV tensor-core path for hd256 (long context): same gating as the hd128 MMA path
        // (block_size==16 so each warp maps to one physical block, chunk >= 2 blocks to fill the GPU).
        static int famma256 = -1;
        if (famma256 < 0) { const char* e = getenv("SPARKINFER_FAMMA"); famma256 = (e && e[0] == '0') ? 0 : 1; }
        const int mma_chunk256 = (n_splits > 0) ? (seqlen + n_splits - 1) / n_splits : 0;
        const bool mma_ok256 = famma256 && seqlen > 512 && block_size == 16 && mma_chunk256 >= 32;
        static int fagqa4 = -1;
        if (fagqa4 < 0) { const char* e = getenv("SPARKINFER_FAGQA4"); fagqa4 = (e && e[0] == '0') ? 0 : 1; }
        if (fagqa4 && num_kv_heads > 0 && num_q_heads == num_kv_heads * 4) {
            // Qwythos-9B: 16Q/4KV full-attn — GQA-4 shared-KV tile; int8 MMA at long ctx (>=8k).
            constexpr int GQA = 4, TILE = FA_GQA4_TILE;
            constexpr int MMA_THREADS = fa_mma_block_threads<256, GQA>::v;
            dim3 gq(num_kv_heads * n_splits, num_seqs);
            static int famma4 = -1;
            if (famma4 < 0) {
                const char* e = getenv("SPARKINFER_FAMMA4");
                famma4 = (e && e[0] == '0') ? 0 : 1;
            }
            if (mma_ok256 && int8_kv && famma4) {
                const size_t i8_smem = (size_t)2 * 16 * 256 * sizeof(signed char)
                                     + (size_t)(16 + GQA) * 256 * sizeof(float)
                                     + (size_t)(16 + 16 + 128 + 128 + 16 + 16) * sizeof(float);
                fa_split_gqa_mma_i8_kernel<256, GQA><<<gq, MMA_THREADS, i8_smem, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const signed char*>(k_pool),
                    reinterpret_cast<const signed char*>(v_pool), block_table, seq_lens,
                    part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks, n_splits,
                    reinterpret_cast<const __half*>(k_scale), reinterpret_cast<const __half*>(v_scale));
            } else {
                const size_t smem = (size_t)2 * TILE * 256 * sizeof(__nv_bfloat16);
                if (int8_kv)
                    fa_split_gqa_kernel<256, GQA, TILE, true><<<gq, GQA * 32, smem, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool, block_table, seq_lens,
                        part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks, n_splits,
                        reinterpret_cast<const __half*>(k_scale), reinterpret_cast<const __half*>(v_scale));
                else
                    fa_split_gqa_kernel<256, GQA, TILE, false><<<gq, GQA * 32, smem, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool, block_table, seq_lens,
                        part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks, n_splits,
                        reinterpret_cast<const __half*>(k_scale), reinterpret_cast<const __half*>(v_scale));
            }
            combine_hd256(out_q8);
            (void)seqlen;
            return;
        }
        // Qwen3.8-27B: 24Q/4KV full-attn. The shared-KV tile above is instantiated for GQA 4 and
        // 8 only, so a 6:1 group fell through to the scalar one-warp-per-block kernel, where every
        // q-head re-reads its group's K and V from global. At ctx=16384 that is the whole
        // long-context cost of decode: the split pass moves 67.1 MB per layer per token at
        // 635 GB/s, 35% of this part's bandwidth, and 13.8% of the decode step. Staging the tile
        // once per (kv head, split) reads each K/V byte once for all six q-heads instead of six
        // times. Same partials, same layout, same combine pass -- only who reads the KV changes.
        // SPARKINFER_FAGQA6=0 restores the scalar kernel (A/B in ONE binary).
        static int fagqa6 = -1;
        if (fagqa6 < 0) { const char* e = getenv("SPARKINFER_FAGQA6"); fagqa6 = (e && e[0] == '0') ? 0 : 1; }
        if (fagqa6 && num_kv_heads > 0 && num_q_heads == num_kv_heads * 6) {
            constexpr int GQA = 6, TILE = FA_GQA6_TILE;
            dim3 gq(num_kv_heads * n_splits, num_seqs);
            // int8 tensor-core arm, same as the 4:1 and 8:1 groups already take: Q and P go to
            // int8 so QK and PV run on the int8 tensor cores, and the KV global read halves
            // again. M is the group's 6 q-heads padded to the 16-row mma; blockDim stays 256
            // because the mainloop gives one of its 8 warps to each of 8 KV blocks per iteration,
            // which is independent of GQA. SPARKINFER_FAMMA6=0 keeps the bf16 tile kernel.
            static int famma6 = -1;
            if (famma6 < 0) { const char* e = getenv("SPARKINFER_FAMMA6"); famma6 = (e && e[0] == '0') ? 0 : 1; }
            if (mma_ok256 && int8_kv && famma6) {
                constexpr int MMA_THREADS = fa_mma_block_threads<256, GQA>::v;
                const size_t i8_smem = (size_t)2 * 16 * 256 * sizeof(signed char)
                                     + (size_t)(16 + GQA) * 256 * sizeof(float)
                                     + (size_t)(16 + 16 + 128 + 128 + 16 + 16) * sizeof(float);
                fa_split_gqa_mma_i8_kernel<256, GQA><<<gq, MMA_THREADS, i8_smem, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const signed char*>(k_pool),
                    reinterpret_cast<const signed char*>(v_pool), block_table, seq_lens,
                    part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks, n_splits,
                    reinterpret_cast<const __half*>(k_scale), reinterpret_cast<const __half*>(v_scale));
                combine_hd256(out_q8);
                (void)seqlen;
                return;
            }
            // How many keys ONE block walks -- not the sequence length. This, not seqlen, is what
            // sets the staging-loop iteration count, so it is what picks the tile depth.
            const long fa6_chunk = (long)(seqlen > 0 ? seqlen : 0) /
                                   (long)(n_splits > 0 ? n_splits : 1);
            // SPARKINFER_FA_TILE_DEEP: unset/2 = auto by chunk, 0 = always the shallow tile,
            // 1 = always deep. A/B in ONE binary, which is the only honest way to compare when the
            // kernels live in a shared library.
            static int fa6_deep_env = -1;
            if (fa6_deep_env < 0) {
                const char* e = getenv("SPARKINFER_FA_TILE_DEEP");
                fa6_deep_env = e ? atoi(e) : 2;
            }
            const bool fa6_deep = (fa6_deep_env == 2) ? (fa6_chunk >= 256) : (fa6_deep_env != 0);
            // int8_kv must use the <...,true> instantiation: it dequants int8 -> bf16 into the
            // staged tile, and reading the int8 pool as bf16 would be garbage.
#define SI_FA6_LAUNCH(TL, I8)                                                                     \
            do {                                                                                  \
                const size_t sm6 = (size_t)2 * (TL) * 256 * sizeof(__nv_bfloat16);                \
                fa_split_gqa_kernel<256, GQA, (TL), I8><<<gq, GQA * 32, sm6, stream>>>(           \
                    reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool, block_table,       \
                    seq_lens, part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads,         \
                    block_size, max_blocks, n_splits,                                             \
                    reinterpret_cast<const __half*>(k_scale),                                     \
                    reinterpret_cast<const __half*>(v_scale));                                     \
            } while (0)
            constexpr int TILE_DEEP = FA_GQA6_TILE_DEEP;
            // cp.async pipeline: issue the NEXT tile's global->shared copy before consuming this
            // one, so the load overlaps the compute instead of the block stalling at every barrier.
            // Deepening the tile alone raised loads-in-flight but kept the two phases strictly
            // serial. bf16 KV only -- cp.async copies bytes verbatim and cannot dequantize.
            // Double buffering needs 2x the staging smem (4 * TILE * 256 * 2 B), which is past the
            // 48 KB default, so the kernel opts in once. SPARKINFER_FA_PIPE=0 disables.
            static int fa6_pipe = -1;
            if (fa6_pipe < 0) { const char* e = getenv("SPARKINFER_FA_PIPE"); fa6_pipe = (e && e[0] == '0') ? 0 : 1; }
            // KV-GROUP split: KVG warp-groups per block, each walking its own stripe. Raises
            // warps/SM by KVG without changing n_splits or the combine. Off by default until
            // measured; SPARKINFER_FA_KVG=2 selects it. Tau-neutral -- the draft is untouched.
            static int fa6_kvg = -1;
            if (fa6_kvg < 0) { const char* e = getenv("SPARKINFER_FA_KVG"); fa6_kvg = e ? atoi(e) : 3; }
            // KT (tile per group) is swept independently of KVG: both trade against the same
            // ~100 KB dynamic-smem cap, so the best point is not obvious a priori.
            //   KVG=2 KT=16 -> 64 KB tiles + 13 KB merge = 77 KB
            //   KVG=2 KT=20 -> 80 KB          + 13 KB    = 93 KB
            //   KVG=3 KT=12 -> 72 KB          + 19 KB    = 91 KB
            static int fa6_kt = -1;
            if (fa6_kt < 0) { const char* e = getenv("SPARKINFER_FA_KVG_TILE"); fa6_kt = e ? atoi(e) : 16; }
#define SI_KVG_TRY(KVGV, KTV)                                                                     \
            do {                                                                                  \
                constexpr int KVG = (KVGV), KT = (KTV);                                           \
                const size_t gsm = (size_t)4 * KT * 256 * sizeof(__nv_bfloat16) * KVG;            \
                static int ok_##KVGV##_##KTV = -1;                                                \
                if (ok_##KVGV##_##KTV < 0) {                                                      \
                    ok_##KVGV##_##KTV = (cudaFuncSetAttribute(                                    \
                                  (const void*)fa_split_gqa_pipeg_kernel<256, GQA, KT, KVG>,      \
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,                    \
                                  (int)gsm) == cudaSuccess) ? 1 : 0;                              \
                    if (!ok_##KVGV##_##KTV) cudaGetLastError();                                   \
                }                                                                                 \
                if (ok_##KVGV##_##KTV) {                                                          \
                    fa_split_gqa_pipeg_kernel<256, GQA, KT, KVG>                                  \
                        <<<gq, GQA * 32 * KVG, gsm, stream>>>(                                    \
                        reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool, block_table,   \
                        seq_lens, part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads,     \
                        block_size, max_blocks, n_splits);                                        \
                    combine_hd256(out_q8);                                                        \
                    (void)seqlen;                                                                 \
                    return;                                                                       \
                }                                                                                 \
            } while (0)
            if (fa6_kvg >= 2 && fa6_pipe && !int8_kv && fa6_deep && num_seqs == 1) {
                if (fa6_kvg >= 3)      { SI_KVG_TRY(3, 12); }
                else if (fa6_kt >= 20) { SI_KVG_TRY(2, 20); }
                SI_KVG_TRY(2, 16);
            }
#undef SI_KVG_TRY
            if (fa6_pipe && !int8_kv && fa6_deep) {
                constexpr int PT = FA_GQA6_TILE_DEEP;
                const size_t psm = (size_t)4 * PT * 256 * sizeof(__nv_bfloat16);
                static int pipe_ok = -1;
                if (pipe_ok < 0) {
                    pipe_ok = (cudaFuncSetAttribute(
                                   (const void*)fa_split_gqa_pipe_kernel<256, GQA, PT>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   (int)psm) == cudaSuccess) ? 1 : 0;
                    if (!pipe_ok) cudaGetLastError();   // clear, fall through to the synchronous tile
                }
                if (pipe_ok) {
                    fa_split_gqa_pipe_kernel<256, GQA, PT><<<gq, GQA * 32, psm, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool, block_table,
                        seq_lens, part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads,
                        block_size, max_blocks, n_splits);
                    combine_hd256(out_q8);
                    (void)seqlen;
                    return;
                }
            }
            if (int8_kv) {
                if (fa6_deep) SI_FA6_LAUNCH(TILE_DEEP, true);
                else          SI_FA6_LAUNCH(TILE, true);
            } else {
                if (fa6_deep) SI_FA6_LAUNCH(TILE_DEEP, false);
                else          SI_FA6_LAUNCH(TILE, false);
            }
#undef SI_FA6_LAUNCH
            combine_hd256(out_q8);
            (void)seqlen;
            return;
        }
        if (num_kv_heads > 0 && num_q_heads == num_kv_heads * 8) {
            constexpr int GQA = 8, TILE = FA_GQA_TILE;
            dim3 gq(num_kv_heads * n_splits, num_seqs);
            if (mma_ok256 && int8_kv) {   // int8 tensor-core hd256 — halves the KV read for the 10 full-attn layers
                const size_t i8_smem = (size_t)2 * 16 * 256 * sizeof(signed char)
                                     + (size_t)(16 + GQA) * 256 * sizeof(float)     // s_s[16][256] + s_o[GQA][256]
                                     + (size_t)(16 + 16 + 128 + 128 + 16 + 16) * sizeof(float);
                fa_split_gqa_mma_i8_kernel<256, GQA><<<gq, GQA * 32, i8_smem, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const signed char*>(k_pool),
                    reinterpret_cast<const signed char*>(v_pool), block_table, seq_lens,
                    part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks, n_splits,
                    reinterpret_cast<const __half*>(k_scale), reinterpret_cast<const __half*>(v_scale));
                combine_hd256(out_q8);
                (void)seqlen;
                return;
            }
            // Fold S verify rows into one CTA so the staged K/V tile is read once for all of them
            // instead of once per row. Only when S divides num_seqs exactly, so no row is padded.
            static int seqfold = -1;
            // The fold is limited by the extra per-row accumulators it keeps live: 2, 3 and 4 are
            // close in isolation (951.6 / 943.8 us at 32k for 2 and 3) while 6 regresses to
            // 1548.0 us, worse than not folding at all. Since a fold only runs when it divides the
            // row count exactly, the default picks the widest instantiation that does: a 6-row
            // verify folds by 3 and an 8-row verify by 4, each leaving 2 CTAs per (kv head, split).
            // Measured at 32k on an 8-row verify: fold 4 = 518.6 tok/s against fold 2 = 512.4.
            // SPARKINFER_FA_SEQFOLD pins a specific width; 1 restores one row per CTA.
            if (seqfold < 0) { const char* e = getenv("SPARKINFER_FA_SEQFOLD"); seqfold = e ? atoi(e) : 0; }
            const int fold = seqfold > 0 ? seqfold
                           : (num_seqs % 4 == 0 ? 4 : (num_seqs % 3 == 0 ? 3 : (num_seqs % 2 == 0 ? 2 : 1)));
            if (!int8_kv && fold > 1 && num_seqs > 1 && (num_seqs % fold) == 0) {
                const size_t smem = (size_t)2 * TILE * 256 * sizeof(__nv_bfloat16);
                dim3 gqf(num_kv_heads * n_splits, num_seqs / fold);
                if (fold == 2)
                    fa_split_gqa_kernel<256, GQA, TILE, false, 2><<<gqf, GQA * 32, smem, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool, block_table, seq_lens,
                        part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks,
                        n_splits, reinterpret_cast<const __half*>(k_scale), reinterpret_cast<const __half*>(v_scale));
                else if (fold == 3)
                    fa_split_gqa_kernel<256, GQA, TILE, false, 3><<<gqf, GQA * 32, smem, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool, block_table, seq_lens,
                        part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks,
                        n_splits, reinterpret_cast<const __half*>(k_scale), reinterpret_cast<const __half*>(v_scale));
                else if (fold == 4)
                    fa_split_gqa_kernel<256, GQA, TILE, false, 4><<<gqf, GQA * 32, smem, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool, block_table, seq_lens,
                        part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks,
                        n_splits, reinterpret_cast<const __half*>(k_scale), reinterpret_cast<const __half*>(v_scale));
                else
                    fa_split_gqa_kernel<256, GQA, TILE, false, 6><<<gqf, GQA * 32, smem, stream>>>(
                        reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool, block_table, seq_lens,
                        part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks,
                        n_splits, reinterpret_cast<const __half*>(k_scale), reinterpret_cast<const __half*>(v_scale));
                combine_hd256(out_q8);
                (void)seqlen;
                return;
            }
            // Scalar/tile GQA fallback
            // whole run, so when int8_kv is on the tile kernel MUST dequant int8->bf16 in smem (the
            // <256,...,true> instantiation) — reading the int8 pool as bf16 would be garbage.
            const size_t smem = (size_t)2 * TILE * 256 * sizeof(__nv_bfloat16);
            if (int8_kv)
                fa_split_gqa_kernel<256, GQA, TILE, true><<<gq, GQA * 32, smem, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool, block_table, seq_lens,
                    part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks, n_splits,
                    reinterpret_cast<const __half*>(k_scale), reinterpret_cast<const __half*>(v_scale));
            else
                fa_split_gqa_kernel<256, GQA, TILE, false><<<gq, GQA * 32, smem, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool, block_table, seq_lens,
                    part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks, n_splits,
                    reinterpret_cast<const __half*>(k_scale), reinterpret_cast<const __half*>(v_scale));
        } else {
            dim3 g1(num_q_heads * n_splits, num_seqs);
            fa_split_kernel<256><<<g1, 32, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool, block_table, seq_lens,
                part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks, n_splits,
                reinterpret_cast<const __half*>(k_scale), reinterpret_cast<const __half*>(v_scale), int8_kv);
        }
        combine_hd256(out_q8);
        (void)seqlen;
        return;
    }
    static int fagqa = -1;
    if (fagqa < 0) {
        const char* e = getenv("SPARKINFER_FAGQA");
        fagqa = e ? ((e[0] == '0') ? 0 : 1) : -2;   // -2 = auto: long-context only
    }
    const bool use_gqa = (fagqa == 1) || (fagqa == -2 && n_splits >= 32);
    // Tensor-core (wmma int8) GQA split (SPARKINFER_FAMMA, default on): the 8 GQA q-heads become the
    // mma M dim, moving the QK/PV dot + reduction onto the int8 tensor cores while halving the KV read.
    // The kernel reads each 16-token physical block's fragments straight from the paged pool, so it is
    // only exact when every split's chunk is a multiple of block_size (16), and it needs the int8 cache;
    // otherwise the scalar split runs. bf16 (int8 off) always uses the scalar path (== main).
    static int famma = -1;
    if (famma < 0) { const char* e = getenv("SPARKINFER_FAMMA"); famma = (e && e[0] == '0') ? 0 : 1; }
    // Long-context regime only: requires block_size==16 (each warp maps to one physical block) AND a
    // large-enough per-split chunk (>=2 physical blocks). At tiny chunks the GQA-shared mma has too
    // few blocks/warps to fill the GPU and loses to the high-occupancy scalar split; those short
    // contexts use the scalar path. Robust to any chunk (partial blocks masked).
    const int mma_chunk = (n_splits > 0) ? (seqlen + n_splits - 1) / n_splits : 0;
    const bool mma_aligned = famma && seqlen > 512 && block_size == 16 && mma_chunk >= 32;
    const __half* ksc = reinterpret_cast<const __half*>(k_scale);
    const __half* vsc = reinterpret_cast<const __half*>(v_scale);
    if (use_gqa && num_kv_heads > 0 && num_q_heads == num_kv_heads * 8) {
        constexpr int GQA = 8, TILE = FA_GQA_TILE;
        dim3 gq(num_kv_heads * n_splits, num_seqs);
        if (mma_aligned && int8_kv) {   // int8 tensor-core (halved KV read) — the long-context win
            const size_t i8_smem = (size_t)2 * 16 * 128 * sizeof(signed char)
                                 + (size_t)(16 + GQA) * 128 * sizeof(float)   // s_s[16][HD] + s_o[GQA][HD]
                                 + (size_t)(16 + 16 + 128 + 128 + 16 + 16) * sizeof(float);
            fa_split_gqa_mma_i8_kernel<128, GQA><<<gq, GQA * 32, i8_smem, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const signed char*>(k_pool),
                reinterpret_cast<const signed char*>(v_pool), block_table, seq_lens,
                part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks, n_splits,
                ksc, vsc);
        } else {
            // Scalar split. The bf16 instantiation is byte-identical to the pre-int8 (main) kernel, so the
            // guard contexts (128/512/4k, int8 off) match main exactly; the int8 instantiation serves the
            // forced-int8 short/unaligned path (accuracy gate) and never touches the bf16 codegen.
            const size_t smem = (size_t)2 * TILE * 128 * sizeof(__nv_bfloat16);
            if (int8_kv)
                fa_split_gqa_kernel<128, GQA, TILE, true><<<gq, GQA * 32, smem, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool, block_table, seq_lens,
                    part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks, n_splits,
                    ksc, vsc);
            else
                fa_split_gqa_kernel<128, GQA, TILE, false><<<gq, GQA * 32, smem, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool, block_table, seq_lens,
                    part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks, n_splits,
                    ksc, vsc);
        }
        if (gate128)
            fa_launch_combine_gated_dispatch(part_m, part_l, part_acc, reinterpret_cast<__nv_bfloat16*>(out),
                                             gate, num_q_heads, n_splits,
                                             reinterpret_cast<fa_block_q8_1*>(out_q8), num_seqs, stream);
        else
            fa_launch_combine_dispatch(part_m, part_l, part_acc, reinterpret_cast<__nv_bfloat16*>(out),
                                       num_q_heads, n_splits, reinterpret_cast<fa_block_q8_1*>(out_q8), num_seqs, stream);
        (void)head_dim;
        return;
    }
    dim3 g1(num_q_heads * n_splits, num_seqs);
    fa_split_kernel<128><<<g1, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), k_pool, v_pool, block_table, seq_lens,
        part_m, part_l, part_acc, scale, num_q_heads, num_kv_heads, block_size, max_blocks, n_splits,
        ksc, vsc, int8_kv);
    if (gate128)
        fa_launch_combine_gated_dispatch(part_m, part_l, part_acc, reinterpret_cast<__nv_bfloat16*>(out),
                                         gate, num_q_heads, n_splits,
                                         reinterpret_cast<fa_block_q8_1*>(out_q8), num_seqs, stream);
    else
        fa_launch_combine_dispatch(part_m, part_l, part_acc, reinterpret_cast<__nv_bfloat16*>(out),
                                   num_q_heads, n_splits, reinterpret_cast<fa_block_q8_1*>(out_q8), num_seqs, stream);
    (void)head_dim;
}
#endif

} // namespace kernels
} // namespace sparkinfer
