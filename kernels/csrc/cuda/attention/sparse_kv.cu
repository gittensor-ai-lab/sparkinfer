// Sink + sliding-window sparse-KV for hd256 GQA-4 (Qwythos) and GQA-8 (Qwen3.6) decode.
// [Paral1995] 2026-07-13 GQA-4; extended GQA-8 + int8 wmma sparse.
// SPARKINFER_SPARSE_KV=0 disables; default ON for matching shapes with int8 KV.
//
// Per full-attn layer after int8 KV append:
//   (1) fa_kv_window_select      — sink block 0 + last W logical blocks (StreamingLLM-style)
//   (2) fa_split_gqa_sparse      — scalar flash-split over the selected blocks only
//   (2') fa_split_gqa_mma_i8_sparse — int8 tensor-core twin of (2); same block list and
//        partials, run on wmma instead of scalar FMA (GQA-8 default; SPARKINFER_SPARSE_MMA=0
//        falls back to scalar)
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
__global__ void __launch_bounds__(GQA * 32, 2) fa_split_gqa_sparse(
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
    __shared__ int s_valid;

    for (int i = bstart; i < bend; i++) {
        // Block-uniform early-outs: every thread in the CTA must agree on whether to
        // skip, otherwise __syncthreads below would hang. Clamp OOB logical blocks.
        int lblk = sel[i];
        if (lblk < 0 || lblk >= max_blocks) lblk = -1;
        const int phys = (lblk >= 0) ? block_table[lblk] : -1;
        int valid = 0;
        if (lblk >= 0 && phys >= 0)
            valid = min(block_size, sl - lblk * block_size);
        if (valid < 0) valid = 0;
        if (lane == 0 && warp == 0) s_valid = valid;
        __syncthreads();
        valid = s_valid;
        if (valid <= 0) continue;   // uniform across the CTA (all read the same s_valid)

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
    if (head_dim != 256) return;
    dim3 grid(num_kv_heads * n_splits, 1);
    if (num_q_heads == num_kv_heads * 4) {
        fa_split_gqa_sparse<256, 4><<<grid, 4 * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q),
            reinterpret_cast<const signed char*>(k_pool_layer),
            reinterpret_cast<const signed char*>(v_pool_layer),
            block_table, seq_lens, sel_blk, part_m, part_l, part_acc, scale,
            num_q_heads, num_kv_heads, block_size, max_blocks, n_splits, n_sel,
            reinterpret_cast<const __half*>(k_scale_layer),
            reinterpret_cast<const __half*>(v_scale_layer));
    } else if (num_q_heads == num_kv_heads * 8) {
        fa_split_gqa_sparse<256, 8><<<grid, 8 * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q),
            reinterpret_cast<const signed char*>(k_pool_layer),
            reinterpret_cast<const signed char*>(v_pool_layer),
            block_table, seq_lens, sel_blk, part_m, part_l, part_acc, scale,
            num_q_heads, num_kv_heads, block_size, max_blocks, n_splits, n_sel,
            reinterpret_cast<const __half*>(k_scale_layer),
            reinterpret_cast<const __half*>(v_scale_layer));
    }
}

#include <mma.h>

// Tensor-core (wmma int8) sparse flash split — the MMA twin of fa_split_gqa_sparse above.
//
// Motivation (RTX 5090 A/B on PR #560 / cd94c07): the SCALAR sparse walk beats dense
// flash-decode only once the dense pass reads ≳8x the window (32k+ for Qwen3.6). Between
// 8k and 32k the DENSE path is already on int8 tensor cores, so scanning 16k tokens with
// wmma beats scanning a 4k window with scalar FMAs — sparse@16k measured 419 tok/s vs
// dense 438. This kernel closes that gap: identical sink+window block list, but S=Q·Kᵀ
// and O=P·V run as int8 wmma tiles exactly like the dense MMA kernel, so the sparse path
// keeps the tensor-core throughput AND the O(window) KV read. Bot-verified on cd94c07:
// Qwen3.6 @32k 445.4 vs main 410.1 (+8.6%), accuracy pass (top1 92%, KL 0.056).
//
// Qwythos GQA-4 does NOT default onto this path: same-window scalar→MMA only changes
// compute inside a fixed window and full-attn is ~1/4 of hybrid layers — measured +0.3%
// (within noise) on PR #580. The win here is the KV-read cut vs dense full-cache MMA.
//
// Structure mirrors fa_split_gqa_mma_i8_kernel (same quantization, same online softmax,
// same partials layout — byte-compatible with launch_fa_combine_hd256). Differences:
//   • the KV walk is indirect: groups of up to 8 blocks come from sel_blk[] instead of a
//     contiguous logical range, so per-group physical ids are staged once into shared memory;
//   • per-token validity is a staged mask (a selected block may be the partially-filled tail
//     block), replacing the dense kernel's contiguous [start, end) window test;
//   • single sequence (decode), so the M dim is the GQA q-heads of one kv head, M padded to 16.
//
// One CTA per (kv_head, split); GQA*32 threads (8 warps at GQA-8 → one warp per KV block of
// the group in QK, one 16-wide dim slab per warp in PV). hd256 GQA-4 needs the same 8-warp
// launch as the dense kernel's fa_mma_block_threads<256,4> (kept as a specialization for
// optional use). sm_80+ (wmma int8).
template <int HEAD_DIM, int GQA> struct fa_mma_sparse_threads { static constexpr int v = GQA * 32; };
template <> struct fa_mma_sparse_threads<256, 4> { static constexpr int v = 256; };

template <int HEAD_DIM, int GQA>
__global__ void __launch_bounds__(fa_mma_sparse_threads<HEAD_DIM, GQA>::v, 5) fa_split_gqa_mma_i8_sparse(
    const __nv_bfloat16* __restrict__ q, const signed char* __restrict__ k_pool,
    const signed char* __restrict__ v_pool, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens, const int* __restrict__ sel_blk,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    float scale, int num_q_heads, int num_kv_heads, int max_blocks,
    int n_splits, int n_sel,
    const __half* __restrict__ k_scale, const __half* __restrict__ v_scale
) {
    using namespace nvcuda::wmma;
    constexpr int KH  = HEAD_DIM / 16;   // 16-wide k-tiles per head vector
    constexpr int EPT = HEAD_DIM / 32;   // head elems per lane
    const int split = blockIdx.x % n_splits;
    const int kvh   = blockIdx.x / n_splits;
    const int warp  = threadIdx.x >> 5, lane = threadIdx.x & 31, tid = threadIdx.x;
    const int sl    = seq_lens[0];
    const size_t KVLD = (size_t)num_kv_heads * HEAD_DIM;   // int8 token stride in the pool
    const int SLD = num_kv_heads;                          // scale stride (one per token, kv_head)

    // Shared layout mirrors the dense MMA kernel, plus the sparse staging tail
    // (per-group logical/physical block ids + per-token validity mask).
    extern __shared__ char sp_smem[];
    signed char* s_qi = reinterpret_cast<signed char*>(sp_smem);      // [16][HD] quantized Q
    signed char* s_pi = s_qi + 16 * HEAD_DIM;                         // [16][HD] quantized P'
    float* s_s  = reinterpret_cast<float*>(s_pi + 16 * HEAD_DIM);     // [16][128] scores / mma scratch
    float* s_o  = s_s + 16 * 128;                                     // [GQA][HD] running O
    float* s_qs = s_o + GQA * HEAD_DIM;                               // [16] Q scale
    float* s_ps = s_qs + 16;                                          // [16] P' row scale
    float* s_ks = s_ps + 16;                                          // [128] group K scales
    float* s_vs = s_ks + 128;                                         // [128] group V scales
    float* s_m  = s_vs + 128;                                         // [16]
    float* s_l  = s_m + 16;                                           // [16]
    int*   s_pb = reinterpret_cast<int*>(s_l + 16);                   // [8] group physical blocks
    int*   s_lb = s_pb + 8;                                           // [8] group logical blocks
    char*  s_ok = reinterpret_cast<char*>(s_lb + 8);                  // [128] token validity

    // Quantize Q per q-head row. GQA*32-thread launches (GQA-8) use one warp per 2 rows; the
    // GQA-4 8-warp launch has 4 spare warps that just fall through the r<GQA checks below (same
    // pattern the dense hd256 GQA-4 kernel already uses).
    #pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        const int r = warp * 2 + rr;
        float qv[EPT], amax = 0.f;
        #pragma unroll
        for (int e = 0; e < EPT; e++) {
            qv[e] = (r < GQA) ? __bfloat162float(q[(size_t)(kvh * GQA + r) * HEAD_DIM + lane + e * 32]) : 0.f;
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

    // This split's slice of the selected-block list.
    const int bps    = (n_sel + n_splits - 1) / n_splits;
    const int bstart = split * bps, bend = min(n_sel, bstart + bps);
    const int* sel   = sel_blk + (size_t)kvh * n_sel;

    for (int g0 = bstart; g0 < bend; g0 += 8) {
        const int gblk = min(8, bend - g0);

        // Stage group block ids: logical from sel[], physical via block_table. Invalid or
        // out-of-range entries become -1 and are masked everywhere below.
        if (tid < 8) {
            int lb = (tid < gblk) ? sel[g0 + tid] : -1;
            if (lb < 0 || lb >= max_blocks) lb = -1;
            s_lb[tid] = lb;
            s_pb[tid] = (lb >= 0) ? block_table[lb] : -1;
        }
        __syncthreads();   // QK warps read s_pb written by warp 0

        // Stage per-token K/V scales + validity for the group. A selected block may be the
        // partially-filled tail block, so each token checks its own logical position < sl.
        for (int j = tid; j < gblk * 16; j += blockDim.x) {
            const int e = j >> 4, within = j & 15;
            const int lb = s_lb[e], pb = s_pb[e];
            const bool ok = (lb >= 0) && (pb >= 0) && (lb * 16 + within < sl);
            s_ok[j] = ok ? 1 : 0;
            if (ok) {
                const size_t si = (size_t)(pb * 16 + within) * SLD + kvh;
                s_ks[j] = __half2float(k_scale[si]);
                s_vs[j] = __half2float(v_scale[si]);
            } else { s_ks[j] = 0.f; s_vs[j] = 0.f; }
        }
        // No extra barrier: QK mma reads only s_qi/s_pb + global K; the staged scales and
        // validity are first read in the softmax, fenced by the post-QK __syncthreads.

        // QK int8 mma -> int32 scores (warp w owns block w of the group).
        if (warp < gblk && s_pb[warp] >= 0) {
            const signed char* kb = k_pool + ((size_t)s_pb[warp] * 16 * num_kv_heads + kvh) * HEAD_DIM;
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
            // ldm = 128: the score tile is [16 q-rows x 128 group tokens] (see the dense
            // kernel's hd256 ldm note — row stride is the group width, not HEAD_DIM).
            store_matrix_sync(reinterpret_cast<int*>(s_s) + warp * 16, cf, 128, mem_row_major);
        }
        __syncthreads();
        const int* s_si = reinterpret_cast<const int*>(s_s);

        // Online softmax; fold V scale into P', quantize P' per-row into s_pi.
        #pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int r = warp * 2 + rr;
            float sc[4], mx = -1e30f;
            #pragma unroll
            for (int u = 0; u < 4; u++) {
                const int t = lane + u * 32;
                sc[u] = (t < gblk * 16 && s_ok[t])
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
            for (int o = 16; o > 0; o >>= 1) {
                sum   += __shfl_xor_sync(0xffffffff, sum, o);
                pamax  = fmaxf(pamax, __shfl_xor_sync(0xffffffff, pamax, o));
            }
            const float pd = pamax / 127.0f;
            if (lane == 0) { s_m[r] = m_new; s_l[r] = s_l[r] * corr + sum; s_ps[r] = pd; }
            for (int t = lane; t < gblk * 16; t += 32)
                s_pi[r * 128 + t] = (signed char)((pamax == 0.f) ? 0 : (int)roundf(s_s[r * 128 + t] / pd));
            if (r < GQA) for (int c = lane; c < HEAD_DIM; c += 32) s_o[r * HEAD_DIM + c] *= corr;
        }
        __syncthreads();

        // PV int8 mma -> int32; O += int32 * p_scale[m]. The kernel always runs 8 warps at
        // HEAD_DIM=256 (fa_mma_sparse_threads<256,*>::v == 256, GQA-4 included via the
        // specialization above), so each warp owns one 16-wide d-tile of a 128-wide slab and
        // HEAD_DIM=256 takes two passes (dh = 0, 128) regardless of GQA.
        constexpr int WARPS = fa_mma_sparse_threads<HEAD_DIM, GQA>::v / 32;
        for (int dh = 0; dh < HEAD_DIM; dh += WARPS * 16) {
            fragment<accumulator, 16, 16, 16, int> cf;
            fill_fragment(cf, 0);
            for (int ks = 0; ks < gblk; ks++) {
                const int pb = s_pb[ks];
                if (pb < 0) continue;   // invalid entry: its P' columns are all zero anyway
                const signed char* vb = v_pool + ((size_t)pb * 16 * num_kv_heads + kvh) * HEAD_DIM + dh + warp * 16;
                fragment<matrix_a, 16, 16, 16, signed char, row_major> af;
                fragment<matrix_b, 16, 16, 16, signed char, row_major> bf;
                load_matrix_sync(af, s_pi + ks * 16, 128);
                load_matrix_sync(bf, vb, KVLD);
                mma_sync(cf, af, bf, cf);
            }
            store_matrix_sync(reinterpret_cast<int*>(s_s) + warp * 16, cf, 128, mem_row_major);
            __syncthreads();
            for (int i = tid; i < GQA * 128; i += blockDim.x)
                s_o[(i >> 7) * HEAD_DIM + dh + (i & 127)] += (float)reinterpret_cast<int*>(s_s)[i] * s_ps[i >> 7];
            __syncthreads();
        }
    }

    // Partials: byte-compatible with the scalar sparse kernel / combine (single sequence).
    for (int r = 0; r < GQA; r++) {
        const int qh  = kvh * GQA + r;
        const int idx = qh * n_splits + split;
        if (tid == 0) { part_m[idx] = s_m[r]; part_l[idx] = s_l[r]; }
        for (int c = tid; c < HEAD_DIM; c += blockDim.x)
            part_acc[(size_t)idx * HEAD_DIM + c] = s_o[r * HEAD_DIM + c];
    }
}

#ifndef _MSC_VER
template __global__ void fa_split_gqa_mma_i8_sparse<256, 8>(const __nv_bfloat16*, const signed char*,
    const signed char*, const int*, const int*, const int*, float*, float*, float*, float, int, int,
    int, int, int, const __half*, const __half*);
template __global__ void fa_split_gqa_mma_i8_sparse<256, 4>(const __nv_bfloat16*, const signed char*,
    const signed char*, const int*, const int*, const int*, float*, float*, float*, float, int, int,
    int, int, int, const __half*, const __half*);
#endif

// MMA sparse launcher. hd256 GQA-8 (Qwen3.6) primary; GQA-4 accepted for optional A/B.
// block_size 16 required (wmma tile == KV page). Returns false otherwise so the caller
// falls back to the scalar sparse kernel above.
bool launch_flash_decode_split_sparse_mma(
    const void* q, const void* k_pool_layer, const void* v_pool_layer,
    const int* block_table, const int* seq_lens, const int* sel_blk,
    float* part_m, float* part_l, float* part_acc,
    int num_q_heads, int num_kv_heads, int head_dim, int block_size, int max_blocks,
    int n_splits, int n_sel, float scale,
    const void* k_scale_layer, const void* v_scale_layer, cudaStream_t stream
) {
    if (head_dim != 256 || block_size != 16 || num_kv_heads <= 0) return false;
    constexpr int HD = 256;
    dim3 grid(num_kv_heads * n_splits, 1);
    auto qb = reinterpret_cast<const __nv_bfloat16*>(q);
    auto kb = reinterpret_cast<const signed char*>(k_pool_layer);
    auto vb = reinterpret_cast<const signed char*>(v_pool_layer);
    auto ks = reinterpret_cast<const __half*>(k_scale_layer);
    auto vs = reinterpret_cast<const __half*>(v_scale_layer);
    if (num_q_heads == num_kv_heads * 8) {
        constexpr int GQA = 8, THREADS = fa_mma_sparse_threads<HD, GQA>::v;
        const size_t smem = (size_t)2 * 16 * HD
                          + (size_t)16 * 128 * sizeof(float)
                          + (size_t)GQA * HD * sizeof(float)
                          + (16 + 16 + 128 + 128 + 16 + 16) * sizeof(float)
                          + 2 * 8 * sizeof(int) + 128;
        fa_split_gqa_mma_i8_sparse<HD, GQA><<<grid, THREADS, smem, stream>>>(
            qb, kb, vb, block_table, seq_lens, sel_blk, part_m, part_l, part_acc, scale,
            num_q_heads, num_kv_heads, max_blocks, n_splits, n_sel, ks, vs);
        return true;
    }
    if (num_q_heads == num_kv_heads * 4) {
        constexpr int GQA = 4, THREADS = fa_mma_sparse_threads<HD, GQA>::v;
        const size_t smem = (size_t)2 * 16 * HD
                          + (size_t)16 * 128 * sizeof(float)
                          + (size_t)GQA * HD * sizeof(float)
                          + (16 + 16 + 128 + 128 + 16 + 16) * sizeof(float)
                          + 2 * 8 * sizeof(int) + 128;
        fa_split_gqa_mma_i8_sparse<HD, GQA><<<grid, THREADS, smem, stream>>>(
            qb, kb, vb, block_table, seq_lens, sel_blk, part_m, part_l, part_acc, scale,
            num_q_heads, num_kv_heads, max_blocks, n_splits, n_sel, ks, vs);
        return true;
    }
    return false;
}
#endif

}} // namespace sparkinfer::kernels
