// ============================================================================
// Routed-MoE grouped int8 GEMM straight off the native GGUF expert weights.
//
// WHY THIS EXISTS
// ---------------
// The batched MoE prefill runs the routed experts as: dequantize the whole expert pool to int8
// rows (deq_rows_i8*), then a grouped int8 WMMA GEMM over pair tiles (pfm_moe_gemm_i8*). For
// Qwen3.6-35B-A3B (256 experts, per-expert ffn 512, H 2048) the dequant is a FIXED per-pass cost
// -- it materializes all 256 experts for every one of the 40 layers whatever the prompt length:
//
//     read  486 MB of Q4_K/Q5_K   (gate+up Q4_K, down Q5_K)
//     write 805 MB of int8        <- pure overhead
//     read  805 MB of int8 back   <- pure overhead, in the GEMM
//
// Measured on an RTX 5090 (nsys --cuda-graph-trace=node, main @ 37b3bb5): the dequant is 65.3 ms
// of the 237.8 ms 4k prefill (27.5%) and 32.4 ms of the 85.4 ms 512 prefill (38%), while at 32k it
// is only 4.9% -- it does not scale with N, so it dominates exactly where the prompt is short.
// Both dequant kernels already run at 0.8-1.6 TB/s, i.e. near DRAM peak: the traffic is the cost,
// not the code. The only way to win is to not move those bytes at all.
//
// THIS KERNEL decodes the expert weight to int8 inside the B-tile stage, so the 1.6 GB/layer
// round trip disappears and the GEMM reads only the 486 MB it fundamentally needs. It reads one
// whole 256-value super-block per row per stage, which is what makes the quantized read cheaper
// than the int8 one: at a 32-value granularity a Q4_K row costs 16 B of block header for every
// 16 B of nibbles, so it would be no smaller than int8. At 256 values the header amortizes to
// 4.5 bits/value.
//
// WHEN IT WINS. The materialize path decodes each weight element ONCE PER LAYER; this kernel
// decodes it once per M-tile that touches it. Tiles per expert = ceil(N*top_k/E/BM), so at
// BM=128 that is 1 at N=4k, 2 at 8k, but 8 at 32k -- where re-decoding loses to just
// materializing. The caller gates on N; above the gate nothing changes.
//
// BIT-IDENTICAL to the materialize path, deliberately:
//   * the per-row int8 scale is the one deq_rows_i8 itself produced (precomputed once at load and
//     handed in as row_scale), so it is identical by construction rather than by re-derivation;
//   * the value decode keeps the same evaluation order. Hoisting ds = d*s and dm = dmin*m to the
//     32-value sub-block is exact: the reference computes ((d*s)*nib) - (dmin*m) left to right, so
//     the hoisted form multiplies and subtracts the same floats. This is also what makes the
//     decode cheap (~5 ops/value instead of re-deriving d/dmin/s/m per value);
//   * same roundf(v * inv) and same (signed char)(int) cast, so every int8 byte matches;
//   * int32 WMMA accumulation is exact, and the epilogue applies sx * row_scale in the same order.
// ============================================================================
#include "sparkinfer/kernels/prefill_moe_q.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>

namespace sparkinfer {
namespace kernels {

namespace {

enum { QMQ_Q4_K = 12, QMQ_Q5_K = 13 };

constexpr int QM_BM = 128;    // pair rows per tile (must match the caller's tilemap)
constexpr int QM_BN = 64;     // weight rows (output channels) per block
constexpr int QM_SB = 256;    // values per GGUF super-block == the B-stage K depth
constexpr int QM_BK = 32;     // WMMA K step
constexpr int QM_LD = QM_SB + 16;   // Bs row stride: 16B-aligned for WMMA, +16 breaks bank conflicts

template <int QT>
__device__ __forceinline__ constexpr int qm_bs() {
    return (QT == QMQ_Q4_K) ? 144 : 176;
}

__device__ __forceinline__ float qm_h2f(const unsigned char* p) {
    __half h; *((unsigned short*)&h) = *(const unsigned short*)p; return __half2float(h);
}

// 6-bit packed (scale, min) pair j of a K-quant super-block -- the unpack deq_rows_i8 uses.
__device__ __forceinline__ void qm_scale_min(const unsigned char* sc, int j, int* s, int* m) {
    if (j < 4) { *s = sc[j] & 63; *m = sc[j + 4] & 63; }
    else {
        *s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        *m = (sc[j + 4] >> 4)  | ((sc[j]     >> 6) << 4);
    }
}

__device__ __forceinline__ void qm_cp16(void* dst, const void* src, bool pred) {
    if (pred) __pipeline_memcpy_async(dst, src, 16);
    else      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
}

// Decode the 64 values of sub-block pair `j64` of one super-block into int8.
// Emits t = j64*64 + l (low nibbles, sub-block 2*j64) and t = j64*64 + 32 + l (high nibbles,
// sub-block 2*j64+1), i.e. 64 CONSECUTIVE int8 at dst + j64*64.
//
// Everything stays in registers: the 32 quant bytes come in as two 16-byte loads and the int8
// results are packed into words before being stored, so no byte array is ever addressed
// dynamically (a dynamically indexed local array would land in local memory).
template <int QT>
__device__ __forceinline__ void qm_decode_j64(const unsigned char* __restrict__ blk, int j64,
                                              float inv, signed char* __restrict__ dst) {
    const float bd = qm_h2f(blk), bdmin = qm_h2f(blk + 2);
    const unsigned char* sc = blk + 4;
    const unsigned char* qs = (QT == QMQ_Q4_K) ? (blk + 16) : (blk + 48);

    int s0, m0, s1, m1;
    qm_scale_min(sc, 2 * j64,     &s0, &m0);
    qm_scale_min(sc, 2 * j64 + 1, &s1, &m1);
    // Same left-to-right products the reference evaluates, hoisted out of the value loop:
    // reference is ((bd*s)*nib) - (bdmin*m), so these are the identical intermediate floats.
    const float ds0 = bd * s0, dm0 = bdmin * m0;
    const float ds1 = bd * s1, dm1 = bdmin * m1;

    const unsigned char* qb = qs + j64 * 32;
    const unsigned char* qh = blk + 16;               // Q5_K high-bit plane (unused for Q4_K)
    const int shl = 2 * j64, shh = 2 * j64 + 1;
    // One 16-byte load per 16 values and one 16-byte store, for both nibble halves. 16 B is the
    // widest load the block strides permit and they are all aligned: the Q4_K/Q5_K block sizes
    // (144/176) and the qs/qh offsets (16/48) and j64*32 are every one a multiple of 16, on a
    // cudaMalloc'd base. Loading a uint4 instead of 4 separate words cuts the decode's load
    // instruction count 4x, which matters because this kernel is decode-bound, not MMA-bound.
#pragma unroll
    for (int c = 0; c < 2; c++) {
        const uint4 qv = *reinterpret_cast<const uint4*>(qb + c * 16);
        uint4 hv = make_uint4(0u, 0u, 0u, 0u);
        if (QT == QMQ_Q5_K) hv = *reinterpret_cast<const uint4*>(qh + c * 16);
        uint4 vlo, vhi;
#pragma unroll
        for (int w = 0; w < 4; w++) {
            const unsigned qwd = (w == 0) ? qv.x : (w == 1) ? qv.y : (w == 2) ? qv.z : qv.w;
            const unsigned hwd = (w == 0) ? hv.x : (w == 1) ? hv.y : (w == 2) ? hv.z : hv.w;
            unsigned alo = 0, ahi = 0;
#pragma unroll
            for (int b = 0; b < 4; b++) {
                const int sb8 = 8 * b;
                const unsigned byte = (qwd >> sb8) & 0xFFu;
                int nlo = (int)(byte & 0xFu), nhi = (int)(byte >> 4);
                if (QT == QMQ_Q5_K) {
                    const unsigned hb = (hwd >> sb8) & 0xFFu;
                    nlo += ((hb >> shl) & 1u) ? 16 : 0;
                    nhi += ((hb >> shh) & 1u) ? 16 : 0;
                }
                const int qlo = (int)roundf((ds0 * nlo - dm0) * inv);
                const int qhi = (int)roundf((ds1 * nhi - dm1) * inv);
                alo |= ((unsigned)(qlo & 0xFF)) << sb8;
                ahi |= ((unsigned)(qhi & 0xFF)) << sb8;
            }
            if (w == 0)      { vlo.x = alo; vhi.x = ahi; }
            else if (w == 1) { vlo.y = alo; vhi.y = ahi; }
            else if (w == 2) { vlo.z = alo; vhi.z = ahi; }
            else             { vlo.w = alo; vhi.w = ahi; }
        }
        *reinterpret_cast<uint4*>(dst + j64 * 64 + c * 16) = vlo;
        *reinterpret_cast<uint4*>(dst + j64 * 64 + 32 + c * 16) = vhi;
    }
}

// 3 resident blocks/SM. Occupancy is what this kernel lives on: it alternates a decode phase
// (global reads + ALU) with an MMA phase, so the other resident blocks are what keep the tensor
// cores fed while one block is decoding. BN=64 rather than 128 is chosen for exactly this -- it puts
// Bs at 17408 B so three blocks fit the 100 KB an sm_120 SM has, and it halves the accumulator
// fragments so ptxas can reach the matching register budget. The cost is that the A tile is re-read
// by twice as many blocks, which is cheap because A is small enough to stay L2-resident.
template <int QT, bool A_INDIRECT, bool C_SCATTER>
__global__ __launch_bounds__(256, 3) void pfm_moe_gemm_qi8_kernel(
        const signed char* __restrict__ A_i8, const float* __restrict__ sx,
        const unsigned char* __restrict__ W_q, const float* __restrict__ row_scale,
        const int* __restrict__ pair_tok, const float* __restrict__ pair_w,
        const int* __restrict__ offsets, const int* __restrict__ tilemap,
        const int* __restrict__ d_ntiles,
        __nv_bfloat16* __restrict__ C, float* __restrict__ out_f32,
        int N, int K) {
    using namespace nvcuda;
    constexpr int BS = qm_bs<QT>();
    const int tile = blockIdx.y;
    if (tile >= d_ntiles[0]) return;
    const int e   = tilemap[2 * tile];
    const int mt  = tilemap[2 * tile + 1];
    const int p0  = offsets[e] + mt * QM_BM;
    const int cnt = offsets[e + 1] - offsets[e];
    const int M   = min(QM_BM, cnt - mt * QM_BM);
    const int n0  = blockIdx.x * QM_BN;
    const int nsb = K >> 8;

    __shared__ __align__(16) signed char Bs[QM_BN][QM_LD];
    __shared__ __align__(16) signed char As[2][QM_BM][QM_BK];
    __shared__ int s_tok[QM_BM];

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 3, wn = warp >> 2;

    // This block's 128 weight rows and their int8 row scales.
    const float* swe = row_scale + (size_t)e * N;

    for (int r = tid; r < QM_BM; r += blockDim.x)
        s_tok[r] = (r < M) ? (A_INDIRECT ? pair_tok[p0 + r] : (p0 + r)) : -1;

    // Decode assignment: 4 threads per weight row, one 64-value sub-block pair (j64) each.
    const int dr = tid >> 2, dj = tid & 3;
    const int dgn = n0 + dr;
    const bool drow_ok = dgn < N;
    const unsigned char* drow = W_q + ((size_t)e * N + (drow_ok ? dgn : 0)) * (size_t)nsb * BS;
    const float dscale = drow_ok ? swe[dgn] : 0.f;
    const float dinv = (dscale > 0.f) ? (1.f / dscale) : 0.f;

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> cf[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) wmma::fill_fragment(cf[i][j], 0);

    auto stageA = [&](int buf, int k0) {
        for (int idx = tid; idx < QM_BM * 2; idx += blockDim.x) {
            const int r = idx >> 1, c16 = (idx & 1) * 16;
            const int arow = s_tok[r];
            qm_cp16(&As[buf][r][c16], &A_i8[(size_t)max(arow, 0) * K + k0 + c16],
                    arow >= 0 && (k0 + c16) < K);
        }
        __pipeline_commit();
    };

    __syncthreads();          // s_tok published before stageA reads it
    stageA(0, 0);
    int abuf = 0;

    for (int sb = 0; sb < nsb; sb++) {
        __syncthreads();      // previous super-block's MMA finished reading Bs
        if (drow_ok) {
            const unsigned char* blk = drow + (size_t)sb * BS;
            qm_decode_j64<QT>(blk, dj, dinv, &Bs[dr][0]);
        } else if (dj == 0) {
#pragma unroll
            for (int i = 0; i < QM_SB / 16; i++)
                *reinterpret_cast<uint4*>(&Bs[dr][i * 16]) = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();      // Bs ready

        for (int kk = 0; kk < QM_SB; kk += QM_BK) {
            const int knext = sb * QM_SB + kk + QM_BK;
            if (knext < K) stageA(abuf ^ 1, knext);
            __pipeline_wait_prior(knext < K ? 1 : 0);
            __syncthreads();
#pragma unroll
            for (int k16 = 0; k16 < QM_BK; k16 += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af[2];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf[2];
#pragma unroll
                for (int i = 0; i < 2; i++)
                    wmma::load_matrix_sync(af[i], &As[abuf][wm * 32 + i * 16][k16], QM_BK);
#pragma unroll
                for (int j = 0; j < 2; j++)
                    wmma::load_matrix_sync(bf[j], &Bs[wn * 32 + j * 16][kk + k16], QM_LD);
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 2; j++) wmma::mma_sync(cf[i][j], af[i], bf[j], cf[i][j]);
            }
            __syncthreads();
            abuf ^= 1;
        }
    }

    // Epilogue staging reuses Bs (the K loop is done with it).
    __syncthreads();
    int* Cs = reinterpret_cast<int*>(&Bs[0][0]);
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int rm0 = wm * 32 + i * 16, gn0 = n0 + wn * 32 + j * 16;
            wmma::store_matrix_sync(&Cs[warp * 256], cf[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int el = lane; el < 256; el += 32) {
                const int r = el >> 4, cc = el & 15;
                const int rm = rm0 + r, rn = gn0 + cc;
                if (rm < M && rn < N) {
                    const int p = p0 + rm;
                    const float v = (float)Cs[warp * 256 + el]
                                    * sx[A_INDIRECT ? s_tok[rm] : p] * swe[rn];
                    if (C_SCATTER) atomicAdd(&out_f32[(size_t)pair_tok[p] * N + rn], v * pair_w[p]);
                    else           C[(size_t)p * N + rn] = __float2bfloat16(v);
                }
            }
            __syncwarp();
        }
    }
}

template <int QT>
void dispatch_qi8(const signed char* A_i8, const float* sx, const void* W_q, const float* row_scale,
                  const int* pair_tok, const float* pair_w, const int* offsets,
                  const int* tilemap, const int* d_ntiles,
                  __nv_bfloat16* C, float* out_f32, int n_out, int K, int max_tiles,
                  bool a_indirect, bool c_scatter, cudaStream_t stream) {
    dim3 grid((n_out + QM_BN - 1) / QM_BN, max_tiles);
    const auto* W = reinterpret_cast<const unsigned char*>(W_q);
    if (a_indirect && !c_scatter)
        pfm_moe_gemm_qi8_kernel<QT, true, false><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
    else if (!a_indirect && c_scatter)
        pfm_moe_gemm_qi8_kernel<QT, false, true><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
    else if (a_indirect && c_scatter)
        pfm_moe_gemm_qi8_kernel<QT, true, true><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
    else
        pfm_moe_gemm_qi8_kernel<QT, false, false><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
}

// ============================================================================
// BM=16 variant: the same native-quant fused-decode B-tile technique, tiled for the short-N
// path (moe_serial's regime, N<=512) where pairs/expert average N*top_k/E — far below a
// BM=128 tile's fill (prefill_moe.cu's own pfm_moe_gemm_i8_bm16_kernel exists for exactly this
// reason). Before this, N<=512 could only reach the routed GEMM via deq_rows_i8_vec_gather* ->
// materialize -> pfm_moe_gemm_i8_bm16, paying the SAME fixed per-layer dequant this file's BM=128
// kernel already eliminates for N>512 -- measured 40%+ of the ctx=512 prefill wall (nsys,
// --cuda-graph-trace=node, frontier c66e87f). BM=16 has no M-subdivision (one 16-row wmma tile
// covers the whole M dimension), so the warp/epilogue layout collapses to prefill_moe.cu's bm16
// kernel's shape (8 warps, each owns one 16-col N-tile, BN=128) rather than the BM=128 kernel's
// 2x2 warp grid. The decode primitive (qm_decode_j64) is UNCHANGED and reused verbatim: BN doubling
// (64->128) is covered by assigning 2 threads per weight row instead of 4 (dr = tid>>1 spans all
// 128 rows; each thread decodes 2 of the super-block's 4 j64 groups instead of 1), so the same
// bit-identical per-value decode still runs, just distributed differently across threads.
constexpr int QM_BM16 = 16;
constexpr int QM_BN16 = 128;

template <int QT, bool A_INDIRECT, bool C_SCATTER>
__global__ __launch_bounds__(256, 2) void pfm_moe_gemm_qi8_bm16_kernel(
        const signed char* __restrict__ A_i8, const float* __restrict__ sx,
        const unsigned char* __restrict__ W_q, const float* __restrict__ row_scale,
        const int* __restrict__ pair_tok, const float* __restrict__ pair_w,
        const int* __restrict__ offsets, const int* __restrict__ tilemap,
        const int* __restrict__ d_ntiles,
        __nv_bfloat16* __restrict__ C, float* __restrict__ out_f32,
        int N, int K) {
    using namespace nvcuda;
    constexpr int BS = qm_bs<QT>();
    const int tile = blockIdx.y;
    if (tile >= d_ntiles[0]) return;
    const int e   = tilemap[2 * tile];
    const int mt  = tilemap[2 * tile + 1];
    const int p0  = offsets[e] + mt * QM_BM16;
    const int cnt = offsets[e + 1] - offsets[e];
    const int M   = min(QM_BM16, cnt - mt * QM_BM16);
    const int n0  = blockIdx.x * QM_BN16;
    const int nsb = K >> 8;

    __shared__ __align__(16) signed char Bs[QM_BN16][QM_LD];
    __shared__ __align__(16) signed char As[2][QM_BM16][QM_BK];
    __shared__ int s_tok[QM_BM16];

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wn = warp;                    // 0..7 -> 16-col N tile (single M tile, no wm)

    const float* swe = row_scale + (size_t)e * N;

    for (int r = tid; r < QM_BM16; r += blockDim.x)
        s_tok[r] = (r < M) ? (A_INDIRECT ? pair_tok[p0 + r] : (p0 + r)) : -1;

    // 2 threads per weight row (covers all QM_BN16=128 rows); each decodes 2 of the 4 j64 groups
    // of the super-block (dj, dj+2) instead of the BM=128 kernel's 1-of-4 (4 threads/row). Same
    // qm_decode_j64 primitive, same bit-identical per-value math -- only the thread->work mapping
    // changes to cover twice the rows with the same 256 threads.
    const int dr = tid >> 1, dj = tid & 1;
    const int dgn = n0 + dr;
    const bool drow_ok = dgn < N;
    const unsigned char* drow = W_q + ((size_t)e * N + (drow_ok ? dgn : 0)) * (size_t)nsb * BS;
    const float dscale = drow_ok ? swe[dgn] : 0.f;
    const float dinv = (dscale > 0.f) ? (1.f / dscale) : 0.f;

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> cf;
    wmma::fill_fragment(cf, 0);

    auto stageA = [&](int buf, int k0) {
        for (int idx = tid; idx < QM_BM16 * 2; idx += blockDim.x) {
            const int r = idx >> 1, c16 = (idx & 1) * 16;
            const int arow = s_tok[r];
            qm_cp16(&As[buf][r][c16], &A_i8[(size_t)max(arow, 0) * K + k0 + c16],
                    arow >= 0 && (k0 + c16) < K);
        }
        __pipeline_commit();
    };

    __syncthreads();          // s_tok published before stageA reads it
    stageA(0, 0);
    int abuf = 0;

    for (int sb = 0; sb < nsb; sb++) {
        __syncthreads();      // previous super-block's MMA finished reading Bs
        if (drow_ok) {
            const unsigned char* blk = drow + (size_t)sb * BS;
            qm_decode_j64<QT>(blk, dj,     dinv, &Bs[dr][0]);
            qm_decode_j64<QT>(blk, dj + 2, dinv, &Bs[dr][0]);
        } else if (dj == 0) {
#pragma unroll
            for (int i = 0; i < QM_SB / 16; i++)
                *reinterpret_cast<uint4*>(&Bs[dr][i * 16]) = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();      // Bs ready

        for (int kk = 0; kk < QM_SB; kk += QM_BK) {
            const int knext = sb * QM_SB + kk + QM_BK;
            if (knext < K) stageA(abuf ^ 1, knext);
            __pipeline_wait_prior(knext < K ? 1 : 0);
            __syncthreads();
#pragma unroll
            for (int k16 = 0; k16 < QM_BK; k16 += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf;
                wmma::load_matrix_sync(af, &As[abuf][0][k16], QM_BK);
                wmma::load_matrix_sync(bf, &Bs[wn * 16][kk + k16], QM_LD);
                wmma::mma_sync(cf, af, bf, cf);
            }
            __syncthreads();
            abuf ^= 1;
        }
    }

    // Epilogue staging reuses Bs (the K loop is done with it).
    __syncthreads();
    int* Cs = reinterpret_cast<int*>(&Bs[0][0]);
    {
        const int gn0 = n0 + wn * 16;
        wmma::store_matrix_sync(&Cs[warp * 256], cf, 16, wmma::mem_row_major);
        __syncwarp();
        for (int el = lane; el < 256; el += 32) {
            const int r = el >> 4, cc = el & 15;
            const int rm = r, rn = gn0 + cc;
            if (rm < M && rn < N) {
                const int p = p0 + rm;
                const float v = (float)Cs[warp * 256 + el]
                                * sx[A_INDIRECT ? s_tok[rm] : p] * swe[rn];
                if (C_SCATTER) atomicAdd(&out_f32[(size_t)pair_tok[p] * N + rn], v * pair_w[p]);
                else           C[(size_t)p * N + rn] = __float2bfloat16(v);
            }
        }
        __syncwarp();
    }
}

template <int QT>
void dispatch_qi8_bm16(const signed char* A_i8, const float* sx, const void* W_q, const float* row_scale,
                       const int* pair_tok, const float* pair_w, const int* offsets,
                       const int* tilemap, const int* d_ntiles,
                       __nv_bfloat16* C, float* out_f32, int n_out, int K, int max_tiles,
                       bool a_indirect, bool c_scatter, cudaStream_t stream) {
    dim3 grid((n_out + QM_BN16 - 1) / QM_BN16, max_tiles);
    const auto* W = reinterpret_cast<const unsigned char*>(W_q);
    if (a_indirect && !c_scatter)
        pfm_moe_gemm_qi8_bm16_kernel<QT, true, false><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
    else if (!a_indirect && c_scatter)
        pfm_moe_gemm_qi8_bm16_kernel<QT, false, true><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
    else if (a_indirect && c_scatter)
        pfm_moe_gemm_qi8_bm16_kernel<QT, true, true><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
    else
        pfm_moe_gemm_qi8_bm16_kernel<QT, false, false><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
}

} // namespace

bool pfm_moe_gemm_qi8_supported(int ggml_type) {
    return ggml_type == QMQ_Q4_K || ggml_type == QMQ_Q5_K;
}

bool launch_pfm_moe_gemm_qi8(int ggml_type, const signed char* A_i8, const float* sx,
                             const void* W_q, const float* row_scale,
                             const int* pair_tok, const float* pair_w,
                             const int* offsets, const int* tilemap, const int* d_ntiles,
                             void* C_bf16, float* out_f32,
                             int n_out, int K, int max_tiles, int bm,
                             bool a_indirect, bool c_scatter, cudaStream_t stream) {
    if (!pfm_moe_gemm_qi8_supported(ggml_type)) return false;
    if (bm != QM_BM && bm != QM_BM16) return false;       // tilemap must match one of these
    if (K <= 0 || (K & (QM_SB - 1)) != 0) return false;   // whole super-blocks only
    if (!row_scale || max_tiles <= 0) return false;
    auto* C = reinterpret_cast<__nv_bfloat16*>(C_bf16);
    if (bm == QM_BM16) {
        if (n_out <= 0 || (n_out % QM_BN16) != 0) return false;
        if (ggml_type == QMQ_Q4_K)
            dispatch_qi8_bm16<QMQ_Q4_K>(A_i8, sx, W_q, row_scale, pair_tok, pair_w, offsets, tilemap,
                                        d_ntiles, C, out_f32, n_out, K, max_tiles, a_indirect, c_scatter, stream);
        else
            dispatch_qi8_bm16<QMQ_Q5_K>(A_i8, sx, W_q, row_scale, pair_tok, pair_w, offsets, tilemap,
                                        d_ntiles, C, out_f32, n_out, K, max_tiles, a_indirect, c_scatter, stream);
        return true;
    }
    if (n_out <= 0 || (n_out % QM_BN) != 0) return false;
    if (ggml_type == QMQ_Q4_K)
        dispatch_qi8<QMQ_Q4_K>(A_i8, sx, W_q, row_scale, pair_tok, pair_w, offsets, tilemap,
                               d_ntiles, C, out_f32, n_out, K, max_tiles, a_indirect, c_scatter, stream);
    else
        dispatch_qi8<QMQ_Q5_K>(A_i8, sx, W_q, row_scale, pair_tok, pair_w, offsets, tilemap,
                               d_ntiles, C, out_f32, n_out, K, max_tiles, a_indirect, c_scatter, stream);
    return true;
}

} // namespace kernels
} // namespace sparkinfer
