// ============================================================================
// Skinny bf16 prefill GEMM: C[M,n_out] = A[M,K] @ W^T for narrow n_out.
//
// WHY THIS EXISTS
// ---------------
// pf_gemm_kernel (batched_prefill.cu, #398) tiles the output 128x128 with cp.async double buffering
// — the right shape for the big projections, which are where prefill's arithmetic lives. The Gated-
// DeltaNet gate projections are the opposite shape: ssm_alpha and ssm_beta are [n_out=32, K=4096],
// deliberately left in bf16 (per-row int8 quant of a 32-wide weight costs more accuracy than the
// time it saves), so they fall through to pf_gemm and get:
//
//     grid = ((32 + 127)/128, (4096 + 127)/128) = 1 x 32 = 32 blocks   on a 170-SM part
//
// 32 blocks use 19% of the GPU, and 96 of every 128 output columns are padding — so ~75% of the
// tensor-core work is multiplying zeros. Measured on an RTX 5090 (nsys, main @ cb34dc1, ctx=4096):
// 7.5 ms per prefill across 48 calls (24 linear layers x {alpha, beta}).
//
// It is COMPUTE-bound, not bandwidth-bound — worth stating because the shape invites the opposite
// guess. A is [4096,4096] bf16 = 33.5 MB per call, but it is the same `xn` for alpha and beta (and
// for the qkv/gate projections beside them), so it is L2-resident and real DRAM traffic is ~0.45 ms
// across all 48 calls. The work is 4096*32*4096 = 537 MMAC per call (51.5 GFLOP over 48), and the
// existing path sustains ~6.9 TFLOP/s against a ~255 TFLOP/s bf16 tensor peak. An fp32 register-
// tiled rewrite was tried first and changed nothing (7.49 vs 7.50 ms): staging was never the issue.
//
// THE SHAPE THIS WANTS. One block per BT-row strip of A, all n_out columns held at once, K streamed
// in KT-wide tiles through shared memory, and the math on tensor cores:
//   * grid = M/BT = 128 blocks (vs 32) — 4x the SMs, and every block does useful work.
//   * one warp per 16x16 wmma output tile; no padding — n_out is masked, never rounded to a tile.
//   * 16-byte vector staging, so a warp moves 512 contiguous bytes per slot.
// Accumulation is fp32 and the store is bf16, matching pf_gemm_kernel; only the blocking and the
// mma shape differ. Measured: 7.5 -> 3.54 ms.
// ============================================================================
#include "sparkinfer/kernels/prefill_gemm_skinny.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <cstdlib>

namespace sparkinfer {
namespace kernels {

namespace {

constexpr int SK_PAD = 8;    // break the power-of-two smem row stride

// One block per BT-row strip; NMAX output columns held at once; K streamed KT at a time.
// WARPS warps, each owning one 16x16 wmma accumulator tile of the [BT, NMAX] output.
template <int BT, int NMAX, int KT, int WARPS>
__global__ __launch_bounds__(WARPS * 32) void pf_gemm_skinny_kernel(
        const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ W,
        __nv_bfloat16* __restrict__ C, int M, int n_out, int K) {
    using namespace nvcuda;
    __shared__ __nv_bfloat16 sA[BT][KT + SK_PAD];
    __shared__ __nv_bfloat16 sW[NMAX][KT + SK_PAD];
    __shared__ float sC[16][16];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int m0   = blockIdx.x * BT;
    const int wm   = warp / (NMAX / 16);          // which 16-row tile
    const int wn   = warp % (NMAX / 16);          // which 16-col tile

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.f);

    for (int k0 = 0; k0 < K; k0 += KT) {
        // 16-byte vector staging: 8 bf16 per thread per slot, so a warp moves 512 contiguous bytes.
        for (int e = tid; e < (BT * KT) / 8; e += WARPS * 32) {
            const int i = e / (KT / 8), kv = (e % (KT / 8)) * 8;
            const int gm = m0 + i, gk = k0 + kv;
            const bool ok = gm < M && gk + 7 < K;
            *reinterpret_cast<uint4*>(&sA[i][kv]) =
                ok ? *reinterpret_cast<const uint4*>(&A[(size_t)gm * K + gk]) : make_uint4(0, 0, 0, 0);
        }
        for (int e = tid; e < (NMAX * KT) / 8; e += WARPS * 32) {
            const int j = e / (KT / 8), kv = (e % (KT / 8)) * 8;
            const int gk = k0 + kv;
            const bool ok = j < n_out && gk + 7 < K;
            *reinterpret_cast<uint4*>(&sW[j][kv]) =
                ok ? *reinterpret_cast<const uint4*>(&W[(size_t)j * K + gk]) : make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < KT; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
            wmma::load_matrix_sync(af, &sA[wm * 16][kk], KT + SK_PAD);
            wmma::load_matrix_sync(bf, &sW[wn * 16][kk], KT + SK_PAD);   // col_major => W^T
            wmma::mma_sync(cf, af, bf, cf);
        }
        __syncthreads();
    }

    // staged store: one warp at a time reuses sC, then writes bf16 with bounds+n_out masking
    for (int w = 0; w < WARPS; w++) {
        __syncthreads();
        if (warp != w) continue;
        wmma::store_matrix_sync(&sC[0][0], cf, 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int r = e >> 4, c = e & 15;
            const int gm = m0 + wm * 16 + r, gn = wn * 16 + c;
            if (gm < M && gn < n_out) C[(size_t)gm * n_out + gn] = __float2bfloat16(sC[r][c]);
        }
    }
}

}  // namespace

bool launch_prefill_gemm_skinny(const void* A, const void* W, void* C,
                                int M, int N, int K, cudaStream_t stream) {
    constexpr int BT = 32, NMAX = 32, KT = 64, WARPS = (BT / 16) * (NMAX / 16);
    static_assert(WARPS * 32 <= 1024 && (KT % 16) == 0, "one warp per 16x16 output tile");

    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_PREFILL_GEMM_SKINNY");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    // Only worth it while the 128-wide tile is mostly padding; wider outputs keep the tiled GEMM,
    // which is tensor-core bound and already the right shape for them.
    if (!enabled || N > NMAX || M <= 0 || K <= 0 || (K % KT) != 0) return false;

    dim3 grid((M + BT - 1) / BT);
    pf_gemm_skinny_kernel<BT, NMAX, KT, WARPS><<<grid, WARPS * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A), reinterpret_cast<const __nv_bfloat16*>(W),
        reinterpret_cast<__nv_bfloat16*>(C), M, N, K);
    return true;
}

}  // namespace kernels
}  // namespace sparkinfer
