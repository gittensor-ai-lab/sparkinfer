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
#include <cuda_pipeline.h>
#include <mma.h>

#include <cstdlib>

namespace sparkinfer {
namespace kernels {

namespace {

constexpr int SK_PAD = 8;    // break the power-of-two smem row stride

// 16-byte cp.async when in-bounds, zero-fill otherwise. Same helper pf_gemm_kernel uses.
__device__ __forceinline__ void sk_cp16(void* dst, const void* src, bool pred) {
    if (pred) __pipeline_memcpy_async(dst, src, 16);
    else      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
}

// One block per BT-row strip; NMAX output columns held at once; K streamed KT at a time.
// WARPS warps, each owning one 16x16 wmma accumulator tile of the [BT, NMAX] output.
template <int BT, int NMAX, int KT, int WARPS>
__global__ __launch_bounds__(WARPS * 32) void pf_gemm_skinny_kernel(
        const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ W,
        __nv_bfloat16* __restrict__ C, int M, int n_out, int K) {
    using namespace nvcuda;
    // __align__(16): cp.async and the uint4 zero-fill both need 16B-aligned destinations, and the
    // padded row stride (KT+SK_PAD elements = 144B) only keeps that if the base is aligned too.
    __shared__ __align__(16) __nv_bfloat16 sA[2][BT][KT + SK_PAD];
    __shared__ __align__(16) __nv_bfloat16 sW[2][NMAX][KT + SK_PAD];
    __shared__ float sC[16][16];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int m0   = blockIdx.x * BT;
    const int wm   = warp / (NMAX / 16);          // which 16-row tile
    const int wn   = warp % (NMAX / 16);          // which 16-col tile

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.f);

    // 16-byte cp.async staging: 8 bf16 per thread per slot, so a warp moves 512 contiguous bytes.
    // Double-buffered exactly like pf_gemm_kernel -- with only WARPS*32 threads issuing plain
    // loads, the K loop otherwise pays full global latency on every one of its K/KT trips with no
    // other warp resident to cover it, which is what left this kernel at 118 us for a 128x48x5120
    // GEMM whose tensor-core work is a few microseconds.
    auto stage = [&](int buf, int k0) {
        for (int e = tid; e < (BT * KT) / 8; e += WARPS * 32) {
            const int i = e / (KT / 8), kv = (e % (KT / 8)) * 8;
            const int gm = m0 + i, gk = k0 + kv;
            sk_cp16(&sA[buf][i][kv], &A[(size_t)gm * K + gk], gm < M && gk + 7 < K);
        }
        for (int e = tid; e < (NMAX * KT) / 8; e += WARPS * 32) {
            const int j = e / (KT / 8), kv = (e % (KT / 8)) * 8;
            const int gk = k0 + kv;
            sk_cp16(&sW[buf][j][kv], &W[(size_t)j * K + gk], j < n_out && gk + 7 < K);
        }
        __pipeline_commit();
    };

    const int nk = K / KT;              // launcher guarantees K % KT == 0
    stage(0, 0);
    int buf = 0;
    for (int t = 0; t < nk; t++) {
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * KT);
        __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < KT; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
            wmma::load_matrix_sync(af, &sA[buf][wm * 16][kk], KT + SK_PAD);
            wmma::load_matrix_sync(bf, &sW[buf][wn * 16][kk], KT + SK_PAD);   // col_major => W^T
            wmma::mma_sync(cf, af, bf, cf);
        }
        __syncthreads();
        buf ^= 1;
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

// Same cp.async staging, K partitioned across blockIdx.y so M=128 (4 row-tiles)
// fills the 170-SM 5090. Partials atomicAdd into fp32 P[M,N].
template <int BT, int NMAX, int KT, int WARPS>
__global__ __launch_bounds__(WARPS * 32) void pf_gemm_skinny_sk_kernel(
        const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ W,
        float* __restrict__ P, int M, int n_out, int K, int ktiles) {
    using namespace nvcuda;
    __shared__ __align__(16) __nv_bfloat16 sA[2][BT][KT + SK_PAD];
    __shared__ __align__(16) __nv_bfloat16 sW[2][NMAX][KT + SK_PAD];
    __shared__ float sC[16][16];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int m0   = blockIdx.x * BT;
    const int wm   = warp / (NMAX / 16);
    const int wn   = warp % (NMAX / 16);
    const int nk   = K / KT;
    const int t0   = blockIdx.y * ktiles;
    int t1         = t0 + ktiles;
    if (t1 > nk) t1 = nk;
    if (t0 >= t1) return;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.f);

    auto stage = [&](int buf, int k0) {
        for (int e = tid; e < (BT * KT) / 8; e += WARPS * 32) {
            const int i = e / (KT / 8), kv = (e % (KT / 8)) * 8;
            const int gm = m0 + i, gk = k0 + kv;
            sk_cp16(&sA[buf][i][kv], &A[(size_t)gm * K + gk], gm < M && gk + 7 < K);
        }
        for (int e = tid; e < (NMAX * KT) / 8; e += WARPS * 32) {
            const int j = e / (KT / 8), kv = (e % (KT / 8)) * 8;
            const int gk = k0 + kv;
            sk_cp16(&sW[buf][j][kv], &W[(size_t)j * K + gk], j < n_out && gk + 7 < K);
        }
        __pipeline_commit();
    };

    stage(0, t0 * KT);
    int buf = 0;
    for (int t = t0; t < t1; t++) {
        if (t + 1 < t1) stage(buf ^ 1, (t + 1) * KT);
        __pipeline_wait_prior(t + 1 < t1 ? 1 : 0);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < KT; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
            wmma::load_matrix_sync(af, &sA[buf][wm * 16][kk], KT + SK_PAD);
            wmma::load_matrix_sync(bf, &sW[buf][wn * 16][kk], KT + SK_PAD);
            wmma::mma_sync(cf, af, bf, cf);
        }
        __syncthreads();
        buf ^= 1;
    }

    for (int w = 0; w < WARPS; w++) {
        __syncthreads();
        if (warp != w) continue;
        wmma::store_matrix_sync(&sC[0][0], cf, 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int r = e >> 4, c = e & 15;
            const int gm = m0 + wm * 16 + r, gn = wn * 16 + c;
            if (gm < M && gn < n_out) atomicAdd(&P[(size_t)gm * n_out + gn], sC[r][c]);
        }
    }
}

__global__ void pf_gemm_skinny_reduce(const float* __restrict__ P,
                                      __nv_bfloat16* __restrict__ C, int M, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * N) C[i] = __float2bfloat16(P[i]);
}

}  // namespace

// One instantiation of the kernel above. n_out is masked inside, so NMAX only has to be the
// smallest multiple of 16 that covers it -- the padding columns cost tensor-core slots.
#define SI_SKINNY_LAUNCH(BTV, NMAXV)                                                        \
    do {                                                                                    \
        constexpr int BT_ = (BTV), NMAX_ = (NMAXV), W_ = (BT_ / 16) * (NMAX_ / 16);         \
        static_assert(W_ * 32 <= 1024 && (KT % 16) == 0, "one warp per 16x16 output tile"); \
        const int mtiles = (M + BT_ - 1) / BT_;                                             \
        const int nk = K / KT;                                                              \
        int splits = 1;                                                                     \
        if (sk_on && mtiles > 0 && mtiles < 170 && nk >= 4) {                               \
            splits = (170 + mtiles - 1) / mtiles;                                           \
            if (splits > nk / 2) splits = nk / 2;                                           \
            if (splits < 1) splits = 1;                                                     \
        }                                                                                   \
        if (splits <= 1) {                                                                  \
            pf_gemm_skinny_kernel<BT_, NMAX_, KT, W_><<<mtiles, W_ * 32, 0, stream>>>(      \
                reinterpret_cast<const __nv_bfloat16*>(A),                                  \
                reinterpret_cast<const __nv_bfloat16*>(W),                                  \
                reinterpret_cast<__nv_bfloat16*>(C), M, N, K);                              \
            return true;                                                                    \
        }                                                                                   \
        static float* parts = nullptr;                                                      \
        static size_t parts_n = 0;                                                          \
        const size_t need = (size_t)M * (size_t)N;                                          \
        if (parts_n < need) {                                                               \
            if (parts) cudaFree(parts);                                                     \
            parts = nullptr; parts_n = 0;                                                   \
            if (cudaMalloc(&parts, need * sizeof(float)) != cudaSuccess) return false;      \
            parts_n = need;                                                                 \
        }                                                                                   \
        if (cudaMemsetAsync(parts, 0, need * sizeof(float), stream) != cudaSuccess)         \
            return false;                                                                   \
        const int ktiles = (nk + splits - 1) / splits;                                      \
        const int nz = (nk + ktiles - 1) / ktiles;                                          \
        pf_gemm_skinny_sk_kernel<BT_, NMAX_, KT, W_><<<dim3(mtiles, nz), W_ * 32, 0, stream>>>( \
            reinterpret_cast<const __nv_bfloat16*>(A),                                      \
            reinterpret_cast<const __nv_bfloat16*>(W),                                      \
            parts, M, N, K, ktiles);                                                        \
        pf_gemm_skinny_reduce<<<(int)((need + 255) / 256), 256, 0, stream>>>(               \
            parts, reinterpret_cast<__nv_bfloat16*>(C), M, N);                              \
        return true;                                                                        \
    } while (0)

bool launch_prefill_gemm_skinny(const void* A, const void* W, void* C,
                                int M, int N, int K, cudaStream_t stream) {
    constexpr int KT = 64, NWIDE = 64;

    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_PREFILL_GEMM_SKINNY");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    // Split-K fills the 5090 at M=128 (4 row-tiles). #842 kept one wave of 4 CTAs
    // because shrinking BT made staging worse; splitting K keeps BT=32. Default ON.
    // SPARKINFER_PREFILL_SKINNY_SPLITK=0 restores #842's single-wave grid.
    static const bool sk_on = [] {
        const char* e = getenv("SPARKINFER_PREFILL_SKINNY_SPLITK");
        return !(e && e[0] == '0');
    }();
    // Only worth it while the 128-wide tile is mostly padding; wider outputs keep the tiled GEMM,
    // which is tensor-core bound and already the right shape for them.
    if (!enabled || N > NWIDE || M <= 0 || K <= 0 || (K % KT) != 0) return false;

    // Qwen3.5 / Muse Glimmer: 32 v-heads, long prompts. Unchanged shape, so unchanged bytes.
    if (N <= 32) SI_SKINNY_LAUNCH(32, 32);

    // Qwen3.8-27B has 48 v-heads, so its ssm_alpha/ssm_beta are [48,K] -- one column tile too wide
    // for the NMAX=32 instantiation above, which is why they were still falling through to
    // pf_gemm_kernel. At the scored ctx=128 that grids to ((48+127)/128, (128+127)/128) = ONE
    // block: a single SM runs the whole 128x48x5120 GEMM, 96 times per prefill.
    //
    // BT stays 32. At M=128 that is only 4 blocks, so widening the grid by halving the row strip
    // looks like the obvious next move -- it measures WORSE (+5.6% vs +15.0% over the pf_gemm
    // fallback). Total work is fixed at (M/16)*(NMAX/16) warp tiles of K/16 mma steps whatever the
    // blocking, and what actually costs time here is the staging: fewer threads per block means
    // more sequential 16B load rounds per K tile, and the block cannot spare warps to cover them.
    // SPARKINFER_PREFILL_SKINNY_BT re-runs that A/B.
    static const int bt_env = [] {
        const char* e = getenv("SPARKINFER_PREFILL_SKINNY_BT");
        return e ? atoi(e) : 32;
    }();

    if (N <= 48) {
        if (bt_env <= 16) SI_SKINNY_LAUNCH(16, 48);
        if (bt_env >= 64) SI_SKINNY_LAUNCH(64, 48);
        SI_SKINNY_LAUNCH(32, 48);
    }
    if (bt_env <= 16) SI_SKINNY_LAUNCH(16, 64);
    if (bt_env >= 64) SI_SKINNY_LAUNCH(64, 64);
    SI_SKINNY_LAUNCH(32, 64);
}

#undef SI_SKINNY_LAUNCH

}  // namespace kernels
}  // namespace sparkinfer
