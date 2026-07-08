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
template <int HEAD_DIM, int GQA, int TILE, bool INT8>
__global__ void fa_split_gqa_kernel(
    const __nv_bfloat16* __restrict__ q, const void* __restrict__ k_pool,
    const void* __restrict__ v_pool, const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_acc,
    float scale, int num_q_heads, int num_kv_heads, int block_size, int max_blocks, int n_splits,
    const __half* __restrict__ k_scale, const __half* __restrict__ v_scale
) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const int seq   = blockIdx.y;
    const int split = blockIdx.x % n_splits;
    const int kvh   = blockIdx.x / n_splits;
    const int warp  = threadIdx.x >> 5;
    const int lane  = threadIdx.x & 31;
    const int qh    = kvh * GQA + warp;

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
            const int phys = block_table[seq * max_blocks + blk];
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
            float p = 0.f;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++) p += qr[e] * fa_to_f(s_k[tt * HEAD_DIM + lane + e * 32]);
            const float score = fa_wsum(p) * scale;
            const float mn = fmaxf(m, score), corr = __expf(m - mn), pe = __expf(score - mn);
            l = l * corr + pe;
            #pragma unroll
            for (int e = 0; e < ELEMS; e++) acc[e] = acc[e] * corr + pe * fa_to_f(s_v[tt * HEAD_DIM + lane + e * 32]);
            m = mn;
        }
        __syncthreads();
    }

    const int idx = (seq * num_q_heads + qh) * n_splits + split;
    if (lane == 0) { part_m[idx] = m; part_l[idx] = l; }
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) part_acc[(size_t)idx * HEAD_DIM + lane + e * 32] = acc[e];
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

#ifndef FA_COMBINE_DG
#define FA_COMBINE_DG 4     // head-dim groups (DG x blocks); sweepable
#endif
#ifndef FA_COMBINE_NW
#define FA_COMBINE_NW 4     // warps/block folding the split stripes; sweepable
#endif
#ifndef FA_GQA_TILE
#define FA_GQA_TILE 14      // bf16 smem + uint4 ldg sweet spot at n_splits=128
#endif
template __global__ void fa_split_kernel<128>(const __nv_bfloat16*, const void*, const void*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int, const __half*, const __half*, int);
template __global__ void fa_split_gqa_kernel<128, 8, FA_GQA_TILE, false>(const __nv_bfloat16*, const void*, const void*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int, const __half*, const __half*);
template __global__ void fa_split_gqa_kernel<128, 8, FA_GQA_TILE, true>(const __nv_bfloat16*, const void*, const void*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int, const __half*, const __half*);
template __global__ void fa_combine_kernel<128, FA_COMBINE_DG, FA_COMBINE_NW>(const float*, const float*, const float*, __nv_bfloat16*, int, int, fa_block_q8_1*);
template __global__ void fa_combine_kernel<128, FA_COMBINE_DG, 8>(const float*, const float*, const float*, __nv_bfloat16*, int, int, fa_block_q8_1*);
template __global__ void fa_combine_kernel<128, FA_COMBINE_DG, 16>(const float*, const float*, const float*, __nv_bfloat16*, int, int, fa_block_q8_1*);
// Qwen3.6 full-attention head_dim=256 (bf16 KV): GQA-8 split + scalar fallback.
template __global__ void fa_split_kernel<256>(const __nv_bfloat16*, const void*, const void*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int, const __half*, const __half*, int);
template __global__ void fa_split_gqa_kernel<256, 8, FA_GQA_TILE, false>(const __nv_bfloat16*, const void*, const void*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int, const __half*, const __half*);
template __global__ void fa_split_gqa_kernel<256, 8, FA_GQA_TILE, true>(const __nv_bfloat16*, const void*, const void*,
    const int*, const int*, float*, float*, float*, float, int, int, int, int, int, const __half*, const __half*);
template __global__ void fa_combine_kernel<256, FA_COMBINE_DG, FA_COMBINE_NW>(const float*, const float*, const float*, __nv_bfloat16*, int, int, fa_block_q8_1*);
template __global__ void fa_combine_kernel<256, FA_COMBINE_DG, 8>(const float*, const float*, const float*, __nv_bfloat16*, int, int, fa_block_q8_1*);
template __global__ void fa_combine_kernel<256, FA_COMBINE_DG, 16>(const float*, const float*, const float*, __nv_bfloat16*, int, int, fa_block_q8_1*);

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/attention.h"
#include <mma.h>

// Tensor-core (wmma int8) GQA flash-decode split for long context. The 8 GQA q-heads of a kv-head are
// the batch (M) dim, so S = Q·Kᵀ and O = P·V become small matmuls on the tensor cores, replacing the
// per-lane FMA + 5-shuffle fa_wsum reduction that dominates the scalar kernel at long context. K/V are
// int8 with one fp16 scale per (token, kv_head) head vector. Q is quantized per-q-head and P (with the
// per-token V scale folded in) per-row, so QK and PV run on int8 tensor cores (int32 accumulate); the
// per-token/per-head fp16 scales are applied to the int32 results. This halves the KV global read (the
// bottleneck) and uses 2x-throughput int8 tensor cores. M is padded 8->16; partials (m,l,acc) stay
// byte-compatible with the combine kernel. sm_80+ (wmma). One block per (seq, kv_head, split); 8 warps.
template <int HEAD_DIM, int GQA>
__global__ void __launch_bounds__(GQA * 32, 5) fa_split_gqa_mma_i8_kernel(
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
    #pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        const int r = warp * 2 + rr;
        float qv[4], amax = 0.f;
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            qv[e] = (r < GQA) ? __bfloat162float(q[(size_t)(seq * num_q_heads + kvh * GQA + r) * HEAD_DIM + lane + e * 32]) : 0.f;
            amax = fmaxf(amax, fabsf(qv[e]));
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        const float d = amax / 127.0f;
        if (lane == 0) s_qs[r] = d;
        #pragma unroll
        for (int e = 0; e < 4; e++)
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
            store_matrix_sync(reinterpret_cast<int*>(s_s) + warp * 16, cf, HEAD_DIM, mem_row_major);
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

        // PV int8 mma -> int32; O += int32 * p_scale[m].
        {
            fragment<accumulator, 16, 16, 16, int> cf;
            fill_fragment(cf, 0);
            for (int ks = 0; ks < gblk; ks++) {
                const int pb = block_table[seq * max_blocks + first_blk + g0 + ks];
                const signed char* vb = v_pool + ((size_t)pb * 16 * num_kv_heads + kvh) * HEAD_DIM + warp * 16;
                fragment<matrix_a, 16, 16, 16, signed char, row_major> af;
                fragment<matrix_b, 16, 16, 16, signed char, row_major> bf;
                load_matrix_sync(af, s_pi + ks * 16, HEAD_DIM);
                load_matrix_sync(bf, vb, KVLD);
                mma_sync(cf, af, bf, cf);
            }
            store_matrix_sync(reinterpret_cast<int*>(s_s) + warp * 16, cf, HEAD_DIM, mem_row_major);
        }
        __syncthreads();
        // Only the GQA real q-head rows are kept (rows GQA..15 are wmma M-padding, never written to
        // the partials) — accumulate just those, halving this thread-parallel epilogue at GQA=8.
        for (int i = tid; i < GQA * 128; i += blockDim.x) s_o[i] += (float)reinterpret_cast<int*>(s_s)[i] * s_ps[i >> 7];
        __syncthreads();
    }

    for (int r = 0; r < GQA; r++) {
        const int qh = kvh * GQA + r;
        const int idx = (seq * num_q_heads + qh) * n_splits + split;
        if (tid == 0) { part_m[idx] = s_m[r]; part_l[idx] = s_l[r]; }
        for (int c = tid; c < HEAD_DIM; c += blockDim.x)
            part_acc[(size_t)idx * HEAD_DIM + c] = s_o[r * HEAD_DIM + c];
    }
}
template __global__ void fa_split_gqa_mma_i8_kernel<128, 8>(const __nv_bfloat16*, const signed char*,
    const signed char*, const int*, const int*, float*, float*, float*, float, int, int, int, int, int,
    const __half*, const __half*);
// Qwen3.6 full-attention head_dim=256 (hybrid). The kernel is HEAD_DIM-generic (KH=HEAD_DIM/16); this
// instantiation moves the 10 full-attn layers onto int8-KV tensor cores, halving their KV read at
// long context. i8 smem = ~33 KB (< 48 KB dynamic cap; 5 blocks/SM fits the 5090's ~228 KB).
template __global__ void fa_split_gqa_mma_i8_kernel<256, 8>(const __nv_bfloat16*, const signed char*,
    const signed char*, const int*, const int*, float*, float*, float*, float, int, int, int, int, int,
    const __half*, const __half*);

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

static inline void fa_launch_combine_dispatch(
    const float* part_m, const float* part_l, const float* part_acc,
    __nv_bfloat16* out, int num_q_heads, int n_splits, fa_block_q8_1* out_q8,
    int num_seqs, cudaStream_t stream
) {
    if (n_splits >= 128)      fa_launch_combine<16>(part_m, part_l, part_acc, out, num_q_heads, n_splits, out_q8, num_seqs, stream);
    else if (n_splits >= 64)  fa_launch_combine<8>(part_m, part_l, part_acc, out, num_q_heads, n_splits, out_q8, num_seqs, stream);
    else                      fa_launch_combine<FA_COMBINE_NW>(part_m, part_l, part_acc, out, num_q_heads, n_splits, out_q8, num_seqs, stream);
}

// hd256 combine: same NW-adaptive dispatch as hd128, scaling warps with n_splits
// so the per-warp serial split-folding stays bounded. NW=16 at n_splits>=128
// keeps each warp at ~8-16 splits (matching hd128 sweet spot).
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

static inline void fa_launch_combine_dispatch_hd256(
    const float* part_m, const float* part_l, const float* part_acc,
    __nv_bfloat16* out, int num_q_heads, int n_splits, fa_block_q8_1* out_q8,
    int num_seqs, cudaStream_t stream
) {
    if (n_splits >= 128)      fa_launch_combine_hd256<16>(part_m, part_l, part_acc, out, num_q_heads, n_splits, out_q8, num_seqs, stream);
    else if (n_splits >= 64)  fa_launch_combine_hd256<8>(part_m, part_l, part_acc, out, num_q_heads, n_splits, out_q8, num_seqs, stream);
    else                      fa_launch_combine_hd256<FA_COMBINE_NW>(part_m, part_l, part_acc, out, num_q_heads, n_splits, out_q8, num_seqs, stream);
}

void launch_flash_decode_split(
    const void* q, const void* k_pool, const void* v_pool,
    const int* block_table, const int* seq_lens, void* out,
    float* part_m, float* part_l, float* part_acc,
    int num_seqs, int num_q_heads, int num_kv_heads, int head_dim,
    int block_size, int max_blocks, int n_splits, float scale, cudaStream_t stream,
    void* out_q8, int seqlen, const void* k_scale, const void* v_scale, int int8_kv
) {
    // Qwen3.6 full-attention layers run head_dim=256 (bf16 KV). Use the GQA-8 shared-KV tile
    // path (same 8:1 grouping as Qwen3 hd=128) — cuts KV global reads ~8x vs one-warp-per-q-head.
    if (head_dim == 256) {
        // int8-KV tensor-core path for hd256 (long context): same gating as the hd128 MMA path
        // (block_size==16 so each warp maps to one physical block, chunk >= 2 blocks to fill the GPU).
        static int famma256 = -1;
        if (famma256 < 0) { const char* e = getenv("SPARKINFER_FAMMA"); famma256 = (e && e[0] == '0') ? 0 : 1; }
        const int mma_chunk256 = (n_splits > 0) ? (seqlen + n_splits - 1) / n_splits : 0;
        const bool mma_ok256 = famma256 && seqlen > 512 && block_size == 16 && mma_chunk256 >= 32;
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
                fa_launch_combine_dispatch_hd256(part_m, part_l, part_acc,
                    reinterpret_cast<__nv_bfloat16*>(out), num_q_heads, n_splits,
                    reinterpret_cast<fa_block_q8_1*>(out_q8), num_seqs, stream);
                (void)seqlen;
                return;
            }
            // Scalar/tile GQA fallback (short context or MMA off). The int8 cache is resident for the
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
        fa_launch_combine_dispatch_hd256(part_m, part_l, part_acc,
            reinterpret_cast<__nv_bfloat16*>(out), num_q_heads, n_splits,
            reinterpret_cast<fa_block_q8_1*>(out_q8), num_seqs, stream);
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
    fa_launch_combine_dispatch(part_m, part_l, part_acc, reinterpret_cast<__nv_bfloat16*>(out),
                               num_q_heads, n_splits, reinterpret_cast<fa_block_q8_1*>(out_q8), num_seqs, stream);
    (void)head_dim;
}
#endif

} // namespace kernels
} // namespace sparkinfer
