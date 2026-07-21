// Batched (token-parallel) MoE FFN kernels for Qwen3.6-35B-A3B prompt prefill.
// See prefill_moe.h for the pipeline; qwen35_prefill.cpp orchestrates per layer.
//
// The grouped GEMM mirrors the merged int8 prefill GEMM tiling (prefill_gemm_i8.cu:
// 128x128 tile, 8 warps, BK=32, cp.async double buffer) with two changes: the M
// dimension is a per-expert slice of the expert-bucketed pair list (a tile never
// spans experts), and the A rows / C rows can be indirected through pair_tok so
// gate/up read per-token activations and down scatter-adds per-token outputs.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cstdlib>

#include "sparkinfer/kernels/prefill_moe.h"

namespace sparkinfer {
namespace kernels {
namespace {

__device__ __forceinline__ float pm_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float pm_silu(float x) { return x / (1.f + __expf(-x)); }

// ---- router logits: one warp per (token, expert), gemv_f32-order dot ----
__global__ void pfm_router_logits_kernel(const __nv_bfloat16* __restrict__ x,
                                         const __nv_bfloat16* __restrict__ W,
                                         float* __restrict__ logits,
                                         int n_tokens, int n_experts, int H) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // tokens on grid.x (up to 2^31), expert-groups on grid.y (=32): grid.y = n_tokens would
    // overflow CUDA's 65535 grid-dim limit at N >= 64k and silently fail to launch.
    const int e = blockIdx.y * (blockDim.x >> 5) + warp;
    const int t = blockIdx.x;
    if (e >= n_experts || t >= n_tokens) return;
    const __nv_bfloat16* xr = x + (size_t)t * H;
    const __nv_bfloat16* wr = W + (size_t)e * H;
    float acc = 0.f;
    for (int i = lane; i < H; i += 32) acc += pm_to_f(xr[i]) * pm_to_f(wr[i]);
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) logits[(size_t)t * n_experts + e] = acc;
}

// ---- bucketing: exclusive scan of counts + tile map (one block), then pair scatter ----
// BM=128 for long-N (full WMMA tile). BM=16 for short-N where avg pairs/expert ≪ 128
// (at N=512 fill is only 12.5% with BM=128 — most tensor-core math multiplies zeros).
constexpr int PM_BM = 128;
constexpr int PM_BM_SHORT = 16;

__global__ void pfm_scan_tiles_kernel(const int* __restrict__ counts,
                                      int* __restrict__ offsets, int* __restrict__ cursors,
                                      int* __restrict__ tilemap, int* __restrict__ d_ntiles,
                                      int n_experts, int bm) {
    // single block, n_experts <= 1024 threads; simple shared-memory scan
    __shared__ int s_off[1025];
    const int t = threadIdx.x;
    if (t == 0) {
        int run = 0;
        for (int e = 0; e < n_experts; e++) { s_off[e] = run; run += counts[e]; }
        s_off[n_experts] = run;
        int nt = 0;
        for (int e = 0; e < n_experts; e++) {
            const int tiles = (counts[e] + bm - 1) / bm;
            for (int i = 0; i < tiles; i++) { tilemap[2 * nt] = e; tilemap[2 * nt + 1] = i; nt++; }
        }
        d_ntiles[0] = nt;
    }
    __syncthreads();
    if (t <= n_experts) offsets[t] = s_off[t];
    if (t < n_experts) cursors[t] = 0;
}

__global__ void pfm_scatter_kernel(const int* __restrict__ expert_ids,
                                   const float* __restrict__ expert_weights,
                                   const int* __restrict__ offsets, int* __restrict__ cursors,
                                   int* __restrict__ pair_tok, float* __restrict__ pair_w,
                                   int n_pairs, int top_k) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;
    const int e = expert_ids[p];
    const int slot = offsets[e] + atomicAdd(&cursors[e], 1);
    pair_tok[slot] = p / top_k;
    pair_w[slot]   = expert_weights[p];
}

// ---- grouped int8 GEMM over expert-partitioned pair tiles ----
#define PM_BN 128
#define PM_BK 32
__device__ __forceinline__ void pm_cp16(void* dst, const void* src, bool pred) {
    if (pred) __pipeline_memcpy_async(dst, src, 16);
    else      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
}

template <bool A_INDIRECT, bool C_SCATTER>
__global__ void pfm_moe_gemm_i8_kernel(const signed char* __restrict__ A_i8,
                                       const float* __restrict__ sx,
                                       const signed char* __restrict__ W_i8,
                                       const float* __restrict__ sw,
                                       const int* __restrict__ pair_tok,
                                       const float* __restrict__ pair_w,
                                       const int* __restrict__ offsets,
                                       const int* __restrict__ tilemap,
                                       const int* __restrict__ d_ntiles,
                                       __nv_bfloat16* __restrict__ C,
                                       float* __restrict__ out_f32,
                                       int N, int K, int e_base) {
    using namespace nvcuda;
    const int tile = blockIdx.y;
    if (tile >= d_ntiles[0]) return;
    const int e   = tilemap[2 * tile];
    const int mt  = tilemap[2 * tile + 1];
    const int p0  = offsets[e] + mt * PM_BM;        // first pair row of this tile
    const int cnt = offsets[e + 1] - offsets[e];    // pairs for this expert
    const int M   = min(PM_BM, cnt - mt * PM_BM);   // valid rows in tile

    __shared__ signed char As[2][PM_BM][PM_BK];
    __shared__ signed char Bs[2][PM_BN][PM_BK];
    __shared__ int Cs[8][16][16];
    __shared__ int s_tok[PM_BM];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 3, wn = warp >> 2;
    const int n0 = blockIdx.x * PM_BN;
    const int nk = (K + PM_BK - 1) / PM_BK;
    // e_base>0: W_i8 holds a contiguous expert group starting at e_base (L2-serial path).
    const signed char* We = W_i8 + (size_t)(e - e_base) * N * K;
    const float*       swe = sw + (size_t)(e - e_base) * N;

    for (int r = tid; r < PM_BM; r += blockDim.x)
        s_tok[r] = (r < M) ? (A_INDIRECT ? pair_tok[p0 + r] : (p0 + r)) : -1;
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> cf[2][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) wmma::fill_fragment(cf[i][j], 0);

    auto stage = [&](int buf, int k0) {
        // 256 threads, 16B each: A rows via s_tok, B rows from the expert's weight slice
        const int r = tid >> 1, c16 = (tid & 1) * 16;
        const int gk = k0 + c16;
        const int arow = s_tok[r];
        pm_cp16(&As[buf][r][c16], &A_i8[(size_t)max(arow, 0) * K + gk], arow >= 0 && gk < K);
        const int gn = n0 + r;
        pm_cp16(&Bs[buf][r][c16], &We[(size_t)gn * K + gk], gn < N && gk < K);
        __pipeline_commit();
    };

    stage(0, 0);
    int buf = 0;
    for (int t = 0; t < nk; t++) {
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * PM_BK);
        __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < PM_BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf[4];
#pragma unroll
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(af[i], &As[buf][wm * 32 + i * 16][kk], PM_BK);
#pragma unroll
            for (int j = 0; j < 4; j++) wmma::load_matrix_sync(bf[j], &Bs[buf][wn * 64 + j * 16][kk], PM_BK);
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) wmma::mma_sync(cf[i][j], af[i], bf[j], cf[i][j]);
        }
        __syncthreads();
        buf ^= 1;
    }

#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int rm0 = wm * 32 + i * 16, gn0 = n0 + wn * 64 + j * 16;
            wmma::store_matrix_sync(&Cs[warp][0][0], cf[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int el = lane; el < 256; el += 32) {
                const int r = el >> 4, cc = el & 15;
                const int rm = rm0 + r, rn = gn0 + cc;
                if (rm < M && rn < N) {
                    const int p = p0 + rm;
                    const float v = (float)Cs[warp][r][cc]
                                    * sx[A_INDIRECT ? s_tok[rm] : p] * swe[rn];
                    if (C_SCATTER) atomicAdd(&out_f32[(size_t)pair_tok[p] * N + rn], v * pair_w[p]);
                    else           C[(size_t)p * N + rn] = __float2bfloat16(v);
                }
            }
            __syncwarp();
        }
    }
}

// Short-N BM=16 legacy path: wmma.m16n16k16 (half the int8 MAC rate of mma.sync.k32).
// Kept for SPARKINFER_PREFILL_MOE_MMA=0 A/B; default launcher uses the mma path below.
template <bool A_INDIRECT, bool C_SCATTER>
__global__ void pfm_moe_gemm_i8_bm16_wmma_kernel(const signed char* __restrict__ A_i8,
                                                 const float* __restrict__ sx,
                                                 const signed char* __restrict__ W_i8,
                                                 const float* __restrict__ sw,
                                                 const int* __restrict__ pair_tok,
                                                 const float* __restrict__ pair_w,
                                                 const int* __restrict__ offsets,
                                                 const int* __restrict__ tilemap,
                                                 const int* __restrict__ d_ntiles,
                                                 __nv_bfloat16* __restrict__ C,
                                                 float* __restrict__ out_f32,
                                                 int N, int K, int e_base) {
    using namespace nvcuda;
    constexpr int BM = PM_BM_SHORT;
    const int tile = blockIdx.y;
    if (tile >= d_ntiles[0]) return;
    const int e   = tilemap[2 * tile];
    const int mt  = tilemap[2 * tile + 1];
    const int p0  = offsets[e] + mt * BM;
    const int cnt = offsets[e + 1] - offsets[e];
    const int M   = min(BM, cnt - mt * BM);
    const int n0  = blockIdx.x * PM_BN;

    __shared__ signed char As[2][BM][PM_BK];
    __shared__ signed char Bs[2][PM_BN][PM_BK];
    __shared__ int Cs[8][16][16];
    __shared__ int s_tok[BM];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wn = warp;   // 0..7 → N tile
    const signed char* We = W_i8 + (size_t)(e - e_base) * N * K;
    const float*       swe = sw + (size_t)(e - e_base) * N;

    for (int r = tid; r < BM; r += blockDim.x)
        s_tok[r] = (r < M) ? (A_INDIRECT ? pair_tok[p0 + r] : (p0 + r)) : -1;
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> cf;
    wmma::fill_fragment(cf, 0);

    auto stage = [&](int buf, int k0) {
        for (int idx = tid; idx < BM * 2; idx += blockDim.x) {
            const int r = idx >> 1, c16 = (idx & 1) * 16;
            const int gk = k0 + c16;
            const int arow = s_tok[r];
            pm_cp16(&As[buf][r][c16], &A_i8[(size_t)max(arow, 0) * K + gk], arow >= 0 && gk < K);
        }
        {
            const int r = tid >> 1, c16 = (tid & 1) * 16;
            const int gk = k0 + c16;
            const int gn = n0 + r;
            pm_cp16(&Bs[buf][r][c16], &We[(size_t)gn * K + gk], gn < N && gk < K);
        }
        __pipeline_commit();
    };

    stage(0, 0);
    const int nk = (K + PM_BK - 1) / PM_BK;
    int buf = 0;
    for (int t = 0; t < nk; t++) {
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * PM_BK);
        __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < PM_BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf;
            wmma::load_matrix_sync(af, &As[buf][0][kk], PM_BK);
            wmma::load_matrix_sync(bf, &Bs[buf][wn * 16][kk], PM_BK);
            wmma::mma_sync(cf, af, bf, cf);
        }
        __syncthreads();
        buf ^= 1;
    }

    {
        const int gn0 = n0 + wn * 16;
        wmma::store_matrix_sync(&Cs[warp][0][0], cf, 16, wmma::mem_row_major);
        __syncwarp();
        for (int el = lane; el < 256; el += 32) {
            const int r = el >> 4, cc = el & 15;
            const int rm = r, rn = gn0 + cc;
            if (rm < M && rn < N) {
                const int p = p0 + rm;
                const float v = (float)Cs[warp][r][cc]
                                * sx[A_INDIRECT ? s_tok[rm] : p] * swe[rn];
                if (C_SCATTER) atomicAdd(&out_f32[(size_t)pair_tok[p] * N + rn], v * pair_w[p]);
                else           C[(size_t)p * N + rn] = __float2bfloat16(v);
            }
        }
    }
}

// Short-N BM=16 mma.sync path (default). Same tile/API as the wmma kernel above, but:
//   - mma.sync.m16n8k32 (int8's native shape; wmma only exposes k16 → half the MAC rate)
//   - BK=64 (half the main-loop barriers vs BK=32)
//   - XOR-swizzled LDS so the 4B operand loads spread across banks
//   - register→global epilogue (no Cs smem bounce)
// Bit-identical to wmma: int8×int8 accumulate in int32 without overflow at these shapes
// (|sum| ≤ 127*127*4096 ≈ 6.6e7 < 2^31), so integer accumulation is order-independent and the
// sx*sw epilogue is unchanged.
constexpr int PM_BK_MMA = 64;
constexpr int PM_NFRAG_BM16 = 2;   // 16 cols / warp / 8

__device__ __forceinline__ int pm_swz(int k, int row) {
    return (((k >> 4) ^ (row & 3)) << 4) | (k & 15);
}
__device__ __forceinline__ unsigned pm_lds32(const signed char* p) {
    return *reinterpret_cast<const unsigned*>(p);
}

template <bool A_INDIRECT, bool C_SCATTER>
__global__ __launch_bounds__(256, 2) void pfm_moe_gemm_i8_bm16_mma_kernel(
        const signed char* __restrict__ A_i8, const float* __restrict__ sx,
        const signed char* __restrict__ W_i8, const float* __restrict__ sw,
        const int* __restrict__ pair_tok, const float* __restrict__ pair_w,
        const int* __restrict__ offsets, const int* __restrict__ tilemap,
        const int* __restrict__ d_ntiles, __nv_bfloat16* __restrict__ C,
        float* __restrict__ out_f32, int N, int K, int e_base) {
    constexpr int BM = PM_BM_SHORT;
    const int tile = blockIdx.y;
    if (tile >= d_ntiles[0]) return;
    const int e   = tilemap[2 * tile];
    const int mt  = tilemap[2 * tile + 1];
    const int p0  = offsets[e] + mt * BM;
    const int cnt = offsets[e + 1] - offsets[e];
    const int M   = min(BM, cnt - mt * BM);
    const int n0  = blockIdx.x * PM_BN;

    __shared__ signed char As[2][BM][PM_BK_MMA];
    __shared__ signed char Bs[2][PM_BN][PM_BK_MMA];
    __shared__ int s_tok[BM];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wn   = warp;                 // 0..7 → 16 output cols
    const int grp  = lane >> 2;            // 0..7
    const int tig  = lane & 3;             // thread-in-group
    const signed char* We = W_i8 + (size_t)(e - e_base) * N * K;
    const float*       swe = sw + (size_t)(e - e_base) * N;

    for (int r = tid; r < BM; r += blockDim.x)
        s_tok[r] = (r < M) ? (A_INDIRECT ? pair_tok[p0 + r] : (p0 + r)) : -1;
    __syncthreads();

    int acc[PM_NFRAG_BM16][4];
#pragma unroll
    for (int j = 0; j < PM_NFRAG_BM16; j++)
#pragma unroll
        for (int e4 = 0; e4 < 4; e4++) acc[j][e4] = 0;

    // A: 16×64 = 64 × 16B; B: 128×64 = 512 × 16B. 256 threads cover B in one pass and A with spare.
    auto stage = [&](int buf, int k0) {
        for (int s = tid; s < 64; s += 256) {
            const int r = s >> 2, c = s & 3, k = c << 4;
            const int gk = k0 + k, arow = s_tok[r];
            pm_cp16(&As[buf][r][pm_swz(k, r)], &A_i8[(size_t)max(arow, 0) * K + gk],
                    arow >= 0 && gk < K);
        }
#pragma unroll
        for (int s = tid; s < 512; s += 256) {
            const int r = s >> 2, c = s & 3, k = c << 4;
            const int gk = k0 + k, gn = n0 + r;
            pm_cp16(&Bs[buf][r][pm_swz(k, r)], &We[(size_t)gn * K + gk], gn < N && gk < K);
        }
        __pipeline_commit();
    };

    stage(0, 0);
    const int nk = (K + PM_BK_MMA - 1) / PM_BK_MMA;
    int buf = 0;
    for (int t = 0; t < nk; t++) {
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * PM_BK_MMA);
        __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < PM_BK_MMA; kk += 32) {
            const int ka = kk + tig * 4;
            unsigned af[4], bf[PM_NFRAG_BM16][2];
            const int rlo = grp, rhi = grp + 8;
            af[0] = pm_lds32(&As[buf][rlo][pm_swz(ka,      rlo)]);
            af[1] = pm_lds32(&As[buf][rhi][pm_swz(ka,      rhi)]);
            af[2] = pm_lds32(&As[buf][rlo][pm_swz(ka + 16, rlo)]);
            af[3] = pm_lds32(&As[buf][rhi][pm_swz(ka + 16, rhi)]);
#pragma unroll
            for (int j = 0; j < PM_NFRAG_BM16; j++) {
                const int col = wn * 16 + j * 8 + grp;
                bf[j][0] = pm_lds32(&Bs[buf][col][pm_swz(ka,      col)]);
                bf[j][1] = pm_lds32(&Bs[buf][col][pm_swz(ka + 16, col)]);
            }
#pragma unroll
            for (int j = 0; j < PM_NFRAG_BM16; j++)
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+r"(acc[j][0]), "+r"(acc[j][1]), "+r"(acc[j][2]), "+r"(acc[j][3])
                    : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]),
                      "r"(bf[j][0]), "r"(bf[j][1]));
        }
        __syncthreads();
        buf ^= 1;
    }

    // Registers straight to global / atomic (same sx*sw*pair_w math as the wmma epilogue).
#pragma unroll
    for (int j = 0; j < PM_NFRAG_BM16; j++) {
        const int gn = n0 + wn * 16 + j * 8 + tig * 2;
#pragma unroll
        for (int e4 = 0; e4 < 4; e4++) {
            const int gm = grp + (e4 >> 1) * 8;
            const int cn = gn + (e4 & 1);
            if (gm < M && cn < N) {
                const int p = p0 + gm;
                const float v = (float)acc[j][e4]
                                * sx[A_INDIRECT ? s_tok[gm] : p] * swe[cn];
                if (C_SCATTER) atomicAdd(&out_f32[(size_t)pair_tok[p] * N + cn], v * pair_w[p]);
                else           C[(size_t)p * N + cn] = __float2bfloat16(v);
            }
        }
    }
}

// Single-expert GEMM (W_i8 already the expert slice). blockIdx.y = local M-tile.
template <bool A_INDIRECT, bool C_SCATTER, int BM>
__global__ void pfm_moe_gemm_i8_one_kernel(const signed char* __restrict__ A_i8,
                                           const float* __restrict__ sx,
                                           const signed char* __restrict__ W_i8,
                                           const float* __restrict__ sw,
                                           const int* __restrict__ pair_tok,
                                           const float* __restrict__ pair_w,
                                           const int* __restrict__ offsets,
                                           int expert_id, int n_tiles,
                                           __nv_bfloat16* __restrict__ C,
                                           float* __restrict__ out_f32,
                                           int N, int K) {
    using namespace nvcuda;
    const int mt = blockIdx.y;
    if (mt >= n_tiles) return;
    const int p0  = offsets[expert_id] + mt * BM;
    const int cnt = offsets[expert_id + 1] - offsets[expert_id];
    const int M   = min(BM, cnt - mt * BM);
    const int n0  = blockIdx.x * PM_BN;

    __shared__ signed char As[2][BM][PM_BK];
    __shared__ signed char Bs[2][PM_BN][PM_BK];
    __shared__ int Cs[8][16][16];
    __shared__ int s_tok[BM];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;

    for (int r = tid; r < BM; r += blockDim.x)
        s_tok[r] = (r < M) ? (A_INDIRECT ? pair_tok[p0 + r] : (p0 + r)) : -1;
    __syncthreads();

    auto stage = [&](int buf, int k0) {
        if constexpr (BM == PM_BM) {
            const int r = tid >> 1, c16 = (tid & 1) * 16;
            const int gk = k0 + c16;
            const int arow = s_tok[r];
            pm_cp16(&As[buf][r][c16], &A_i8[(size_t)max(arow, 0) * K + gk], arow >= 0 && gk < K);
            const int gn = n0 + r;
            pm_cp16(&Bs[buf][r][c16], &W_i8[(size_t)gn * K + gk], gn < N && gk < K);
        } else {
            for (int idx = tid; idx < BM * 2; idx += blockDim.x) {
                const int r = idx >> 1, c16 = (idx & 1) * 16;
                const int gk = k0 + c16;
                const int arow = s_tok[r];
                pm_cp16(&As[buf][r][c16], &A_i8[(size_t)max(arow, 0) * K + gk], arow >= 0 && gk < K);
            }
            const int r = tid >> 1, c16 = (tid & 1) * 16;
            const int gk = k0 + c16;
            const int gn = n0 + r;
            pm_cp16(&Bs[buf][r][c16], &W_i8[(size_t)gn * K + gk], gn < N && gk < K);
        }
        __pipeline_commit();
    };

    if constexpr (BM == PM_BM) {
        const int wm = warp & 3, wn = warp >> 2;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> cf[2][4];
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) wmma::fill_fragment(cf[i][j], 0);

        stage(0, 0);
        const int nk = (K + PM_BK - 1) / PM_BK;
        int buf = 0;
        for (int t = 0; t < nk; t++) {
            if (t + 1 < nk) stage(buf ^ 1, (t + 1) * PM_BK);
            __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < PM_BK; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af[2];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf[4];
#pragma unroll
                for (int i = 0; i < 2; i++) wmma::load_matrix_sync(af[i], &As[buf][wm * 32 + i * 16][kk], PM_BK);
#pragma unroll
                for (int j = 0; j < 4; j++) wmma::load_matrix_sync(bf[j], &Bs[buf][wn * 64 + j * 16][kk], PM_BK);
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) wmma::mma_sync(cf[i][j], af[i], bf[j], cf[i][j]);
            }
            __syncthreads();
            buf ^= 1;
        }
#pragma unroll
        for (int i = 0; i < 2; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int rm0 = wm * 32 + i * 16, gn0 = n0 + wn * 64 + j * 16;
                wmma::store_matrix_sync(&Cs[warp][0][0], cf[i][j], 16, wmma::mem_row_major);
                __syncwarp();
                for (int el = lane; el < 256; el += 32) {
                    const int r = el >> 4, cc = el & 15;
                    const int rm = rm0 + r, rn = gn0 + cc;
                    if (rm < M && rn < N) {
                        const int p = p0 + rm;
                        const float v = (float)Cs[warp][r][cc]
                                        * sx[A_INDIRECT ? s_tok[rm] : p] * sw[rn];
                        if (C_SCATTER) atomicAdd(&out_f32[(size_t)pair_tok[p] * N + rn], v * pair_w[p]);
                        else           C[(size_t)p * N + rn] = __float2bfloat16(v);
                    }
                }
                __syncwarp();
            }
        }
    } else {
        const int wn = warp;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> cf;
        wmma::fill_fragment(cf, 0);
        stage(0, 0);
        const int nk = (K + PM_BK - 1) / PM_BK;
        int buf = 0;
        for (int t = 0; t < nk; t++) {
            if (t + 1 < nk) stage(buf ^ 1, (t + 1) * PM_BK);
            __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < PM_BK; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf;
                wmma::load_matrix_sync(af, &As[buf][0][kk], PM_BK);
                wmma::load_matrix_sync(bf, &Bs[buf][wn * 16][kk], PM_BK);
                wmma::mma_sync(cf, af, bf, cf);
            }
            __syncthreads();
            buf ^= 1;
        }
        const int gn0 = n0 + wn * 16;
        wmma::store_matrix_sync(&Cs[warp][0][0], cf, 16, wmma::mem_row_major);
        __syncwarp();
        for (int el = lane; el < 256; el += 32) {
            const int r = el >> 4, cc = el & 15;
            const int rm = r, rn = gn0 + cc;
            if (rm < M && rn < N) {
                const int p = p0 + rm;
                const float v = (float)Cs[warp][r][cc]
                                * sx[A_INDIRECT ? s_tok[rm] : p] * sw[rn];
                if (C_SCATTER) atomicAdd(&out_f32[(size_t)pair_tok[p] * N + rn], v * pair_w[p]);
                else           C[(size_t)p * N + rn] = __float2bfloat16(v);
            }
        }
    }
}

// ---- fused Q4_K/Q5_K/Q6_K grouped GEMM (no full-expert int8 materialize) ----
// B is dequantized on-the-fly into smem as float; A stays int8*sx. Same tilemap as the
// int8 path. Avoids the ~768 MiB/layer HBM write that dominates short-N prefill.
__device__ __forceinline__ float pfm_h2f(const unsigned char* p) {
    __half h; *reinterpret_cast<unsigned short*>(&h) = *reinterpret_cast<const unsigned short*>(p);
    return __half2float(h);
}
__device__ __forceinline__ void pfm_scale_min_k4(int j, const unsigned char* q, int* d, int* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}
__device__ __forceinline__ float pfm_deq_q4k(const unsigned char* blk, int t) {
    const float d = pfm_h2f(blk), dmin = pfm_h2f(blk + 2);
    const unsigned char* sc = blk + 4; const unsigned char* qs = blk + 16;
    const int j64 = t >> 6, r = t & 63, l = r & 31, hi = r >> 5;
    const unsigned char byte = qs[j64 * 32 + l];
    const int nib = hi ? (byte >> 4) : (byte & 0xF);
    int s, m; pfm_scale_min_k4(2 * j64 + hi, sc, &s, &m);
    return d * s * nib - dmin * m;
}
__device__ __forceinline__ float pfm_deq_q5k(const unsigned char* blk, int t) {
    const float d = pfm_h2f(blk), dmin = pfm_h2f(blk + 2);
    const unsigned char* sc = blk + 4; const unsigned char* qh = blk + 16;
    const unsigned char* ql = blk + 48;
    const int j64 = t >> 6, r = t & 63, l = r & 31, hi = r >> 5;
    const unsigned char byte = ql[j64 * 32 + l];
    const int nib = hi ? (byte >> 4) : (byte & 0xF);
    const int hbit = (qh[l] >> (2 * j64 + hi)) & 1;
    int s, m; pfm_scale_min_k4(2 * j64 + hi, sc, &s, &m);
    return d * s * (nib + (hbit ? 16 : 0)) - dmin * m;
}
__device__ __forceinline__ float pfm_deq_q6k(const unsigned char* blk, int t) {
    const int half = t >> 7, r = t & 127, quad = r >> 5, l = r & 31;
    const unsigned char* ql = blk + half * 64;
    const unsigned char* qh = blk + 128 + half * 32;
    const signed char* sc = (const signed char*)(blk + 192) + half * 8;
    const float d = pfm_h2f(blk + 208);
    const int is = l / 16;
    int qv;
    if (quad == 0)      qv = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
    else if (quad == 1) qv = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
    else if (quad == 2) qv = (int)((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
    else                qv = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
    return d * sc[is + 2 * quad] * qv;
}

template <int QT, bool A_INDIRECT, bool C_SCATTER>
__global__ void pfm_moe_gemm_qk_kernel(const signed char* __restrict__ A_i8,
                                       const float* __restrict__ sx,
                                       const unsigned char* __restrict__ W_q,
                                       const int* __restrict__ pair_tok,
                                       const float* __restrict__ pair_w,
                                       const int* __restrict__ offsets,
                                       const int* __restrict__ tilemap,
                                       const int* __restrict__ d_ntiles,
                                       __nv_bfloat16* __restrict__ C,
                                       float* __restrict__ out_f32,
                                       int N, int K) {
    using namespace nvcuda;
    constexpr int BS = (QT == 12) ? 144 : (QT == 13) ? 176 : 210;
    const int tile = blockIdx.y;
    if (tile >= d_ntiles[0]) return;
    const int e   = tilemap[2 * tile];
    const int mt  = tilemap[2 * tile + 1];
    const int p0  = offsets[e] + mt * PM_BM;
    const int cnt = offsets[e + 1] - offsets[e];
    const int M   = min(PM_BM, cnt - mt * PM_BM);
    const int n0  = blockIdx.x * PM_BN;
    const int nsb = K >> 8;

    // bf16 WMMA tiles; A = int8*sx promoted, B = QK dequant. Double-buffer both.
    __shared__ __nv_bfloat16 As[2][PM_BM][PM_BK];
    __shared__ __nv_bfloat16 Bs[2][PM_BN][PM_BK];
    __shared__ float Cs[8][16][16];
    __shared__ int s_tok[PM_BM];
    __shared__ float s_sx[PM_BM];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 3, wn = warp >> 2;

    for (int r = tid; r < PM_BM; r += blockDim.x) {
        if (r < M) {
            const int arow = A_INDIRECT ? pair_tok[p0 + r] : (p0 + r);
            s_tok[r] = arow;
            s_sx[r]  = sx[arow];
        } else {
            s_tok[r] = -1;
            s_sx[r]  = 0.f;
        }
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf[2][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) wmma::fill_fragment(cf[i][j], 0.f);

    auto stage = [&](int buf, int k0) {
        // A: 256 threads each write 16 bf16 = one half-row of BK=32
        {
            const int r = tid >> 1, c16 = (tid & 1) * 16;
            const int arow = s_tok[r];
            const float sc = s_sx[r];
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const int gk = k0 + c16 + i;
                float v = 0.f;
                if (arow >= 0 && gk < K) v = (float)A_i8[(size_t)arow * K + gk] * sc;
                As[buf][r][c16 + i] = __float2bfloat16(v);
            }
        }
        // B: dequant QK → bf16 for BN×BK
        for (int idx = tid; idx < PM_BN * PM_BK; idx += blockDim.x) {
            const int rn = idx / PM_BK, ck = idx - rn * PM_BK;
            const int gn = n0 + rn, gk = k0 + ck;
            float v = 0.f;
            if (gn < N && gk < K) {
                const unsigned char* row = W_q + ((size_t)e * N + gn) * (size_t)nsb * BS;
                const unsigned char* blk = row + (size_t)(gk >> 8) * BS;
                const int t = gk & 255;
                v = (QT == 12) ? pfm_deq_q4k(blk, t)
                  : (QT == 13) ? pfm_deq_q5k(blk, t) : pfm_deq_q6k(blk, t);
            }
            Bs[buf][rn][ck] = __float2bfloat16(v);
        }
    };

    stage(0, 0);
    const int nk = (K + PM_BK - 1) / PM_BK;
    int buf = 0;
    for (int t = 0; t < nk; t++) {
        if (t + 1 < nk) {
            __syncthreads();
            stage(buf ^ 1, (t + 1) * PM_BK);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < PM_BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf[4];
#pragma unroll
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(af[i], &As[buf][wm * 32 + i * 16][kk], PM_BK);
#pragma unroll
            for (int j = 0; j < 4; j++) wmma::load_matrix_sync(bf[j], &Bs[buf][wn * 64 + j * 16][kk], PM_BK);
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) wmma::mma_sync(cf[i][j], af[i], bf[j], cf[i][j]);
        }
        buf ^= 1;
    }

#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int rm0 = wm * 32 + i * 16, gn0 = n0 + wn * 64 + j * 16;
            wmma::store_matrix_sync(&Cs[warp][0][0], cf[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int el = lane; el < 256; el += 32) {
                const int r = el >> 4, cc = el & 15;
                const int rm = rm0 + r, rn = gn0 + cc;
                if (rm < M && rn < N) {
                    const int p = p0 + rm;
                    const float v = Cs[warp][r][cc];
                    if (C_SCATTER) atomicAdd(&out_f32[(size_t)pair_tok[p] * N + rn], v * pair_w[p]);
                    else           C[(size_t)p * N + rn] = __float2bfloat16(v);
                }
            }
            __syncwarp();
        }
    }
}

// ---- shared-expert helpers ----
__global__ void pfm_shared_gate_kernel(const __nv_bfloat16* __restrict__ x,
                                       const __nv_bfloat16* __restrict__ w,
                                       float* __restrict__ dw, int n_tokens, int H) {
    const int t = blockIdx.x;
    const int lane = threadIdx.x;
    if (t >= n_tokens) return;
    const __nv_bfloat16* xr = x + (size_t)t * H;
    float acc = 0.f;
    for (int i = lane; i < H; i += 32) acc += pm_to_f(xr[i]) * pm_to_f(w[i]);
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) dw[t] = 1.f / (1.f + __expf(-acc));
}

__global__ void pfm_shared_swiglu_kernel(const __nv_bfloat16* __restrict__ gate,
                                         const __nv_bfloat16* __restrict__ up,
                                         const float* __restrict__ dw,
                                         __nv_bfloat16* __restrict__ h, long n, int ffn) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float d = dw ? dw[(int)(i / ffn)] : 1.f;
    h[i] = __float2bfloat16(pm_silu(pm_to_f(gate[i])) * pm_to_f(up[i]) * d);
}

__global__ void pfm_resid3_kernel(const __nv_bfloat16* __restrict__ h,
                                  const float* __restrict__ routed,
                                  const __nv_bfloat16* __restrict__ shared,
                                  __nv_bfloat16* __restrict__ x, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = pm_to_f(h[i]) + routed[i];
    if (shared) v += pm_to_f(shared[i]);
    x[i] = __float2bfloat16(v);
}

} // namespace

void launch_pfm_router_logits(const void* x, const void* W, float* logits,
                              int n_tokens, int n_experts, int H, cudaStream_t stream) {
    dim3 grid(n_tokens, (n_experts + 7) / 8);
    pfm_router_logits_kernel<<<grid, 8 * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(W),
        logits, n_tokens, n_experts, H);
}

void launch_pfm_bucket_pairs(const int* expert_ids, const float* expert_weights,
                             const int* counts, int* offsets, int* cursors,
                             int* pair_tok, float* pair_w,
                             int* tilemap, int* d_ntiles,
                             int n_tokens, int n_experts, int top_k, cudaStream_t stream) {
    launch_pfm_bucket_pairs_bm(expert_ids, expert_weights, counts, offsets, cursors,
                               pair_tok, pair_w, tilemap, d_ntiles,
                               n_tokens, n_experts, top_k, PM_BM, stream);
}

void launch_pfm_bucket_pairs_bm(const int* expert_ids, const float* expert_weights,
                                const int* counts, int* offsets, int* cursors,
                                int* pair_tok, float* pair_w,
                                int* tilemap, int* d_ntiles,
                                int n_tokens, int n_experts, int top_k, int bm,
                                cudaStream_t stream) {
    pfm_scan_tiles_kernel<<<1, n_experts + 1, 0, stream>>>(counts, offsets, cursors,
                                                           tilemap, d_ntiles, n_experts, bm);
    const int P = n_tokens * top_k;
    pfm_scatter_kernel<<<(P + 255) / 256, 256, 0, stream>>>(
        expert_ids, expert_weights, offsets, cursors, pair_tok, pair_w, P, top_k);
}

void launch_pfm_moe_gemm_i8(const signed char* A_i8, const float* sx,
                            const signed char* W_i8, const float* sw,
                            const int* pair_tok, const float* pair_w,
                            const int* offsets, const int* tilemap, const int* d_ntiles,
                            void* C_bf16, float* out_f32,
                            int N_out, int K, int max_tiles,
                            bool a_indirect, bool c_scatter, cudaStream_t stream) {
    launch_pfm_moe_gemm_i8_bm(A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles,
                              C_bf16, out_f32, N_out, K, max_tiles, PM_BM,
                              a_indirect, c_scatter, stream);
}

void launch_pfm_moe_gemm_i8_bm(const signed char* A_i8, const float* sx,
                               const signed char* W_i8, const float* sw,
                               const int* pair_tok, const float* pair_w,
                               const int* offsets, const int* tilemap, const int* d_ntiles,
                               void* C_bf16, float* out_f32,
                               int N_out, int K, int max_tiles, int bm,
                               bool a_indirect, bool c_scatter, cudaStream_t stream) {
    launch_pfm_moe_gemm_i8_bm_base(A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles,
                                   C_bf16, out_f32, N_out, K, max_tiles, bm, /*e_base=*/0,
                                   a_indirect, c_scatter, stream);
}

void launch_pfm_moe_gemm_i8_bm_base(const signed char* A_i8, const float* sx,
                                    const signed char* W_i8, const float* sw,
                                    const int* pair_tok, const float* pair_w,
                                    const int* offsets, const int* tilemap, const int* d_ntiles,
                                    void* C_bf16, float* out_f32,
                                    int N_out, int K, int max_tiles, int bm, int e_base,
                                    bool a_indirect, bool c_scatter, cudaStream_t stream) {
    dim3 grid((N_out + PM_BN - 1) / PM_BN, max_tiles);
    auto* C = reinterpret_cast<__nv_bfloat16*>(C_bf16);
    if (bm == PM_BM_SHORT) {
        // Default: mma.sync.m16n8k32 (+BK=64, register epilogue). SPARKINFER_PREFILL_MOE_MMA=0
        // restores the legacy wmma.m16n16k16 path for A/B.
        static int use_mma = [] {
            const char* e = getenv("SPARKINFER_PREFILL_MOE_MMA");
            return (e && e[0] == '0') ? 0 : 1;
        }();
        if (use_mma) {
            if (a_indirect && !c_scatter)
                pfm_moe_gemm_i8_bm16_mma_kernel<true, false><<<grid, 256, 0, stream>>>(
                    A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
            else if (!a_indirect && c_scatter)
                pfm_moe_gemm_i8_bm16_mma_kernel<false, true><<<grid, 256, 0, stream>>>(
                    A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
            else if (a_indirect && c_scatter)
                pfm_moe_gemm_i8_bm16_mma_kernel<true, true><<<grid, 256, 0, stream>>>(
                    A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
            else
                pfm_moe_gemm_i8_bm16_mma_kernel<false, false><<<grid, 256, 0, stream>>>(
                    A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
        } else {
            if (a_indirect && !c_scatter)
                pfm_moe_gemm_i8_bm16_wmma_kernel<true, false><<<grid, 256, 0, stream>>>(
                    A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
            else if (!a_indirect && c_scatter)
                pfm_moe_gemm_i8_bm16_wmma_kernel<false, true><<<grid, 256, 0, stream>>>(
                    A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
            else if (a_indirect && c_scatter)
                pfm_moe_gemm_i8_bm16_wmma_kernel<true, true><<<grid, 256, 0, stream>>>(
                    A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
            else
                pfm_moe_gemm_i8_bm16_wmma_kernel<false, false><<<grid, 256, 0, stream>>>(
                    A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
        }
        return;
    }
    if (a_indirect && !c_scatter)
        pfm_moe_gemm_i8_kernel<true, false><<<grid, 256, 0, stream>>>(
            A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
    else if (!a_indirect && c_scatter)
        pfm_moe_gemm_i8_kernel<false, true><<<grid, 256, 0, stream>>>(
            A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
    else if (a_indirect && c_scatter)
        pfm_moe_gemm_i8_kernel<true, true><<<grid, 256, 0, stream>>>(
            A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
    else
        pfm_moe_gemm_i8_kernel<false, false><<<grid, 256, 0, stream>>>(
            A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
}

void launch_pfm_moe_gemm_i8_one(const signed char* A_i8, const float* sx,
                                const signed char* W_i8, const float* sw,
                                const int* pair_tok, const float* pair_w,
                                const int* offsets, int expert_id, int n_tiles,
                                void* C_bf16, float* out_f32,
                                int N_out, int K, int bm,
                                bool a_indirect, bool c_scatter, cudaStream_t stream) {
    if (n_tiles <= 0) return;
    dim3 grid((N_out + PM_BN - 1) / PM_BN, n_tiles);
    auto* C = reinterpret_cast<__nv_bfloat16*>(C_bf16);
#define PFM_ONE(BM, AI, CS) \
    pfm_moe_gemm_i8_one_kernel<AI, CS, BM><<<grid, 256, 0, stream>>>( \
        A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, expert_id, n_tiles, C, out_f32, N_out, K)
    if (bm == PM_BM_SHORT) {
        if (a_indirect && !c_scatter)      PFM_ONE(PM_BM_SHORT, true,  false);
        else if (!a_indirect && c_scatter) PFM_ONE(PM_BM_SHORT, false, true);
        else if (a_indirect && c_scatter)  PFM_ONE(PM_BM_SHORT, true,  true);
        else                               PFM_ONE(PM_BM_SHORT, false, false);
    } else {
        if (a_indirect && !c_scatter)      PFM_ONE(PM_BM, true,  false);
        else if (!a_indirect && c_scatter) PFM_ONE(PM_BM, false, true);
        else if (a_indirect && c_scatter)  PFM_ONE(PM_BM, true,  true);
        else                               PFM_ONE(PM_BM, false, false);
    }
#undef PFM_ONE
}

void launch_pfm_moe_gemm_qk(const signed char* A_i8, const float* sx,
                            const void* W_q, int wtype,
                            const int* pair_tok, const float* pair_w,
                            const int* offsets, const int* tilemap, const int* d_ntiles,
                            void* C_bf16, float* out_f32,
                            int N_out, int K, int max_tiles,
                            bool a_indirect, bool c_scatter, cudaStream_t stream) {
    if ((K & 255) != 0 || (wtype != 12 && wtype != 13 && wtype != 14)) return;
    dim3 grid((N_out + PM_BN - 1) / PM_BN, max_tiles);
    auto* C = reinterpret_cast<__nv_bfloat16*>(C_bf16);
    auto* W = reinterpret_cast<const unsigned char*>(W_q);
#define PFM_QK_LAUNCH(QT, AI, CS) \
    pfm_moe_gemm_qk_kernel<QT, AI, CS><<<grid, 256, 0, stream>>>( \
        A_i8, sx, W, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K)
    if (wtype == 12) {
        if (a_indirect && !c_scatter)      PFM_QK_LAUNCH(12, true,  false);
        else if (!a_indirect && c_scatter) PFM_QK_LAUNCH(12, false, true);
        else if (a_indirect && c_scatter)  PFM_QK_LAUNCH(12, true,  true);
        else                               PFM_QK_LAUNCH(12, false, false);
    } else if (wtype == 13) {
        if (a_indirect && !c_scatter)      PFM_QK_LAUNCH(13, true,  false);
        else if (!a_indirect && c_scatter) PFM_QK_LAUNCH(13, false, true);
        else if (a_indirect && c_scatter)  PFM_QK_LAUNCH(13, true,  true);
        else                               PFM_QK_LAUNCH(13, false, false);
    } else {
        if (a_indirect && !c_scatter)      PFM_QK_LAUNCH(14, true,  false);
        else if (!a_indirect && c_scatter) PFM_QK_LAUNCH(14, false, true);
        else if (a_indirect && c_scatter)  PFM_QK_LAUNCH(14, true,  true);
        else                               PFM_QK_LAUNCH(14, false, false);
    }
#undef PFM_QK_LAUNCH
}

void launch_pfm_shared_gate(const void* x, const void* w, float* dw,
                            int n_tokens, int H, cudaStream_t stream) {
    pfm_shared_gate_kernel<<<n_tokens, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(w),
        dw, n_tokens, H);
}

void launch_pfm_shared_swiglu(const void* gate, const void* up, const float* dw,
                              void* h, int n_tokens, int ffn, cudaStream_t stream) {
    const long n = (long)n_tokens * ffn;
    pfm_shared_swiglu_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate), reinterpret_cast<const __nv_bfloat16*>(up),
        dw, reinterpret_cast<__nv_bfloat16*>(h), n, ffn);
}

void launch_pfm_resid3(const void* h, const float* routed_f32, const void* shared,
                       void* x, long n, cudaStream_t stream) {
    pfm_resid3_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(h), routed_f32,
        reinterpret_cast<const __nv_bfloat16*>(shared),
        reinterpret_cast<__nv_bfloat16*>(x), n);
}

} // namespace kernels
} // namespace sparkinfer
