// ============================================================================
// Tensor-core router-logits GEMM for the Qwen3.6 MoE batched prefill.
//
// WHY THIS EXISTS
// ---------------
// Every MoE layer of the batched prefill computes router logits [N, E] from the
// normed hidden states [N, H] and the router weight [E, H] with a warp-per-
// (token, expert) fp32 dot (pfm_router_logits_kernel). That shape does no
// operand reuse at all: each warp re-reads one full x row and one full W row per
// output element. Measured on an RTX 5090 (nsys --cuda-graph-trace=node,
// ctx=32768, Qwen3.6-35B-A3B): 194.9 ms per prefill across 40 layers — 4.9 ms
// per layer for N*H*E = 17.2 GMAC, i.e. ~7 TFLOP/s against a >200 TFLOP/s bf16
// tensor peak, and 10-13% of the whole prefill wall at 4k-32k. The router is a
// plain [N,H] x [H,E] GEMM; this file runs it on the bf16 tensor cores.
//
// NUMERICS
// --------
// Inputs are the SAME bf16 values the reference dot reads (x rows and the
// dequantized router weight), and accumulation is fp32 in both paths. A bf16
// product is exact in fp32 (8-bit mantissas), so each multiply contributes the
// identical value in either kernel; the ONLY difference is the order the fp32
// partial sums are reduced in (lane-strided + shuffle tree there, 16-wide wmma
// k-groups here). The logits agree to fp32 rounding of a 2048-term sum
// (|Δ| ~ 1e-6 relative); top-k selection can differ only where two experts'
// logits tie to that precision. Batched-vs-token fidelity is quantified in the
// PR with qwen3_gguf_prefill_check; the decode/scoring path never calls this
// kernel (it uses the fused decode router), so gate accuracy is untouched.
//
// SHAPE
// -----
// One block owns a 64-token x 128-expert C tile: 8 warps in a 2x4 grid, each
// warp a 32x32 quadrant held in four fp32 accumulator fragments. K is streamed
// in 32-wide slices through a cp.async double buffer, so global loads overlap
// the mma work. E and H are tile-aligned for this model (256, 2048); N is
// masked in the tail block. Occupancy is 3 blocks/SM (vs 1 for the big bf16
// projection GEMM) and the grid at 32k is 512x2 = 1024 blocks on 170 SMs.
// ============================================================================
#include "sparkinfer/kernels/prefill_router_mma.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include <cstdlib>

namespace sparkinfer {
namespace kernels {
namespace {

constexpr int RT_BM = 64;    // tokens per block
constexpr int RT_BN = 128;   // experts per block
constexpr int RT_BK = 32;    // K slice per stage
constexpr int RT_LD = RT_BK + 8;   // smem row stride (elements): breaks bank conflicts, 16B-aligned

__global__ __launch_bounds__(256) void pfr_logits_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ W,
    float* __restrict__ logits, int n_tokens, int n_experts, int H) {
    using namespace nvcuda::wmma;

    __shared__ __nv_bfloat16 s_a[2][RT_BM][RT_LD];
    __shared__ __nv_bfloat16 s_b[2][RT_BN][RT_LD];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int row0 = blockIdx.x * RT_BM;     // first token of this block
    const int col0 = blockIdx.y * RT_BN;     // first expert of this block
    const bool tail = row0 + RT_BM > n_tokens;

    // warp (wr, wc) owns rows [wr*32, +32) x cols [wc*32, +32) of the C tile
    const int wr = warp >> 2, wc = warp & 3;

    fragment<accumulator, 16, 16, 16, float> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++) fill_fragment(acc[i][j], 0.f);

    // ---- stage one K slice: A rows via cp.async (8 bf16 = 16B per copy) ----
    // 64 rows x 32 cols = 256 copies for A (1/thread), 128 x 32 = 512 for B (2/thread).
    auto stage = [&](int kb, int buf) {
        {
            const int r = tid >> 2, c4 = (tid & 3) << 3;          // 4 copies cover a row
            const int gr = row0 + r;
            if (!tail || gr < n_tokens) {
                __pipeline_memcpy_async(&s_a[buf][r][c4],
                                        x + (size_t)gr * H + kb + c4, 16);
            } else {
                *reinterpret_cast<float4*>(&s_a[buf][r][c4]) = float4{0.f, 0.f, 0.f, 0.f};
            }
        }
        #pragma unroll
        for (int rep = 0; rep < 2; rep++) {
            const int e = (tid >> 2) + (rep << 6), c4 = (tid & 3) << 3;
            __pipeline_memcpy_async(&s_b[buf][e][c4],
                                    W + (size_t)(col0 + e) * H + kb + c4, 16);
        }
        __pipeline_commit();
    };

    stage(0, 0);
    for (int kb = 0, p = 0; kb < H; kb += RT_BK, p ^= 1) {
        __pipeline_wait_prior(0);
        __syncthreads();
        if (kb + RT_BK < H) stage(kb + RT_BK, p ^ 1);

        #pragma unroll
        for (int kt = 0; kt < RT_BK; kt += 16) {
            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> af[2];
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> bf[2];
            #pragma unroll
            for (int i = 0; i < 2; i++)
                load_matrix_sync(af[i], &s_a[p][wr * 32 + i * 16][kt], RT_LD);
            // col_major over the row-major W tile reads W^T -- [K, E] as mma wants it
            #pragma unroll
            for (int j = 0; j < 2; j++)
                load_matrix_sync(bf[j], &s_b[p][wc * 32 + j * 16][kt], RT_LD);
            #pragma unroll
            for (int i = 0; i < 2; i++)
                #pragma unroll
                for (int j = 0; j < 2; j++) mma_sync(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        __syncthreads();
    }

    // ---- store: fp32 accumulators straight to the fp32 logits ----
    if (!tail) {
        #pragma unroll
        for (int i = 0; i < 2; i++)
            #pragma unroll
            for (int j = 0; j < 2; j++)
                store_matrix_sync(logits + (size_t)(row0 + wr * 32 + i * 16) * n_experts
                                         + col0 + wc * 32 + j * 16,
                                  acc[i][j], n_experts, mem_row_major);
    } else {
        // tail block: bounce each fragment through smem and write row-guarded
        float* s_t = reinterpret_cast<float*>(&s_a[0][0][0]) + warp * 256;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                store_matrix_sync(s_t, acc[i][j], 16, mem_row_major);
                __syncwarp();
                const int gr0 = row0 + wr * 32 + i * 16;
                for (int e = lane; e < 256; e += 32) {
                    const int r = e >> 4, c = e & 15;
                    if (gr0 + r < n_tokens)
                        logits[(size_t)(gr0 + r) * n_experts + col0 + wc * 32 + j * 16 + c] =
                            s_t[r * 16 + c];
                }
                __syncwarp();
            }
        }
    }
}

}  // namespace

bool launch_pfm_router_logits_mma(const void* x, const void* W, float* logits,
                                  int n_tokens, int n_experts, int H,
                                  cudaStream_t stream) {
    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_PREFILL_ROUTER_MMA");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    // Tile-alignment guards (Qwen3.6: E=256, H=2048). Small N stays on the warp
    // dot -- below ~2 row tiles the GEMV's launch is cheaper than the GEMM's.
    if (!enabled || n_experts % RT_BN != 0 || H % RT_BK != 0 || n_tokens < 128)
        return false;
    dim3 grid((n_tokens + RT_BM - 1) / RT_BM, n_experts / RT_BN);
    pfr_logits_mma_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(W), logits, n_tokens, n_experts, H);
    return true;
}

}  // namespace kernels
}  // namespace sparkinfer
