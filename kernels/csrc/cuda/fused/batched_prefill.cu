// Batched prompt-prefill kernels for the Qwen3.5 dense-hybrid (Qwythos) model.
//
// forward_token processes a prompt one token at a time: each token pays a full
// bandwidth-bound weight reload per projection (a GEMV). These kernels process the
// whole prompt (N tokens) at once so the weight-bound work becomes tensor-core GEMMs
// (weight read O(N/tile) instead of O(N)), the Gated-DeltaNet recurrence runs as one
// sequential scan over all N tokens, and the full-attention layers fill the paged int8
// KV cache in the exact layout the decode path reads. The decode graph is untouched.
//
// Numerics mirror the decode kernels (qwen36.cu / rope.cu) so the KV cache and recurrent
// state left behind are faithful to the token-by-token path.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include <cstdlib>

#include "sparkinfer/kernels/prefill.h"
#include "sparkinfer/kernels/prefill_attn_mma.h"
#include "sparkinfer/kernels/prefill_attn_window.h"
#include "sparkinfer/kernels/prefill_gdn_chunk.h"
#include "sparkinfer/kernels/prefill_gemm_skinny.h"

namespace sparkinfer {
namespace kernels {

// ---- shared device helpers (byte-for-byte the decode-path math) -------------
namespace {
__device__ __forceinline__ float pf_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float pf_silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float pf_sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float pf_softplus(float x) { return x > 20.f ? x : __logf(1.f + __expf(x)); }
__device__ __forceinline__ float pf_wsum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}
} // namespace

// ============================================================================
// Tensor-core bf16 GEMM:  C[M,N] = A[M,K] @ W^T,  W native GGUF [N,K] row-major.
// So C[m,n] = sum_k A[m,k] * W[n*K + k]. 64x64 output tile, 16 warps, BK=16.
// ============================================================================
// 128x128 output tile, 256 threads (8 warps), warp tile 32x64 (each warp owns 2x4 = 8 accumulator
// fragments), BK=32, cp.async double-buffered global->shared so the MMA pipe stays fed at low
// occupancy. Both A [M,K] and the native weight W [N,K] are loaded as [row][k] (k contiguous ->
// coalesced); B is consumed as a col_major fragment (ld=BK) so C[m,n] = sum_k A[m,k]*W[n,k].
#define PF_BM 128
#define PF_BN 128
#define PF_BK 32
__device__ __forceinline__ void pf_cp16(void* dst, const void* src, bool pred) {
    // 16-byte cp.async when in-bounds; zero the shared slot otherwise.
    if (pred) __pipeline_memcpy_async(dst, src, 16);
    else      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
}
// bf16-element XOR swizzle (8 bf16 = one 16B chunk): element e of row r lives at element
// (((e>>3) ^ (r&3)) << 3) | (e&7). Mirrors the int8 GEMM's 16B-granularity swizzle so the 4B
// mma operand loads (2 bf16) spread across banks; rows 0..3 land on disjoint banks.
__device__ __forceinline__ int pf_swz_e(int e, int row) {
    return (((e >> 3) ^ (row & 3)) << 3) | (e & 7);
}
__device__ __forceinline__ unsigned pf_lds32b(const __nv_bfloat16* p) {
    return *reinterpret_cast<const unsigned*>(p);   // load 2 contiguous bf16 as one 32b operand
}
__global__ void pf_gemm_kernel(const __nv_bfloat16* __restrict__ A,
                               const __nv_bfloat16* __restrict__ W,
                               __nv_bfloat16* __restrict__ C, int M, int N, int K) {
    using namespace nvcuda;
    __shared__ __nv_bfloat16 As[2][PF_BM][PF_BK];   // [buf][m][k]
    __shared__ __nv_bfloat16 Bs[2][PF_BN][PF_BK];   // [buf][n][k]
    __shared__ float Cs[8][16][16];                 // per-warp fp32 fragment staging

    const int tid  = threadIdx.x;          // 0..255
    const int warp = tid >> 5;             // 0..7
    const int lane = tid & 31;
    const int wm   = warp & 3;             // 0..3 -> rows [wm*32, +32)
    const int wn   = warp >> 2;            // 0..1 -> cols [wn*64, +64)
    const int m0   = blockIdx.y * PF_BM;
    const int n0   = blockIdx.x * PF_BN;
    const int nk   = (K + PF_BK - 1) / PF_BK;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf[2][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) wmma::fill_fragment(cf[i][j], 0.f);

    // Stage the (buf, k0) tiles: 512 vec8 slots each for A and B, 256 threads -> 2 slots each.
    auto stage = [&](int buf, int k0) {
        #pragma unroll
        for (int v = 0; v < 2; v++) {
            const int s = tid + v * 256;
            const int r = s >> 2, c8 = (s & 3) * 8;
            const int gm = m0 + r, gk = k0 + c8;
            pf_cp16(&As[buf][r][c8], &A[(size_t)gm * K + gk], gm < M && gk < K);
            const int gn = n0 + r;
            pf_cp16(&Bs[buf][r][c8], &W[(size_t)gn * K + gk], gn < N && gk < K);
        }
        __pipeline_commit();
    };

    stage(0, 0);
    int buf = 0;
    for (int t = 0; t < nk; t++) {
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * PF_BK);
        __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < PF_BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf[4];
            #pragma unroll
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(af[i], &As[buf][wm * 32 + i * 16][kk], PF_BK);
            #pragma unroll
            for (int j = 0; j < 4; j++) wmma::load_matrix_sync(bf[j], &Bs[buf][wn * 64 + j * 16][kk], PF_BK);
            #pragma unroll
            for (int i = 0; i < 2; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) wmma::mma_sync(cf[i][j], af[i], bf[j], cf[i][j]);
        }
        __syncthreads();
        buf ^= 1;
    }
    // store 8 fragments via per-warp fp32 staging -> bf16 with bounds check.
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int gm = m0 + wm * 32 + i * 16, gn = n0 + wn * 64 + j * 16;
            wmma::store_matrix_sync(&Cs[warp][0][0], cf[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int e = lane; e < 256; e += 32) {
                const int r = e >> 4, cc = e & 15;
                const int rm = gm + r, rn = gn + cc;
                if (rm < M && rn < N) C[(size_t)rm * N + rn] = __float2bfloat16(Cs[warp][r][cc]);
            }
            __syncwarp();
        }
    }
}

// ============================================================================
// bf16 tensor-core GEMM, mma.sync m16n8k16 variant. Same C[M,N] = A[M,K] @ W[N,K]^T as
// pf_gemm_kernel, but shaped like the int8 GEMM (prefill_gemm_i8.cu) instead of the wmma path:
// direct mma.sync (register accumulators, no shared-memory epilogue staging), an XOR-swizzled
// smem layout so the 4B operand loads avoid bank conflicts, and a register->global bf16 store.
// The wmma path stages fp32 fragments through smem and loads unswizzled, which caps it near ~60%
// of the bf16 tensor roofline; this reaches ~the int8 kernel's efficiency at half the MAC rate.
// fp32 accumulate, so it stays bf16-faithful (KL parity with the wmma GEMM).
__global__ __launch_bounds__(256, 2) void pf_gemm_bf16_mma_kernel(
        const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ W,
        __nv_bfloat16* __restrict__ C, int M, int N, int K) {
    constexpr int MFRAG = 2;          // 32 rows per warp / 16
    constexpr int NFRAG = 8;          // 64 cols per warp / 8
    __shared__ __nv_bfloat16 As[2][PF_BM][PF_BK];
    __shared__ __nv_bfloat16 Bs[2][PF_BN][PF_BK];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 2;                       // 0..7
    const int tig  = lane & 3;                        // thread-in-group
    const int wm   = warp & 3;                        // rows [wm*32, +32)
    const int wn   = warp >> 2;                       // cols [wn*64, +64)
    const int m0   = blockIdx.y * PF_BM;
    const int n0   = blockIdx.x * PF_BN;
    const int nk   = (K + PF_BK - 1) / PF_BK;

    float acc[MFRAG][NFRAG][4];
    #pragma unroll
    for (int i = 0; i < MFRAG; i++)
        #pragma unroll
        for (int j = 0; j < NFRAG; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

    // 128 rows x 32 bf16 (64B) = 512 16B chunks per tile; 256 threads stage 2 chunks each for A and B.
    auto stage = [&](int buf, int k0) {
        #pragma unroll
        for (int s = tid; s < 512; s += 256) {
            const int r = s >> 2, c = s & 3, e = c << 3;   // 8 bf16 per 16B chunk
            const int gm = m0 + r, gn = n0 + r, gk = k0 + e;
            pf_cp16(&As[buf][r][pf_swz_e(e, r)], &A[(size_t)gm * K + gk], gm < M && gk < K);
            pf_cp16(&Bs[buf][r][pf_swz_e(e, r)], &W[(size_t)gn * K + gk], gn < N && gk < K);
        }
        __pipeline_commit();
    };

    stage(0, 0);
    int buf = 0;
    for (int t = 0; t < nk; t++) {
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * PF_BK);
        __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < PF_BK; kk += 16) {          // two m16n8k16 sub-steps per BK=32 tile
            const int kb = kk + tig * 2;
            unsigned af[MFRAG][4], bf[NFRAG][2];
            #pragma unroll
            for (int i = 0; i < MFRAG; i++) {
                const int rlo = wm * 32 + i * 16 + grp, rhi = rlo + 8;
                af[i][0] = pf_lds32b(&As[buf][rlo][pf_swz_e(kb,     rlo)]);
                af[i][1] = pf_lds32b(&As[buf][rhi][pf_swz_e(kb,     rhi)]);
                af[i][2] = pf_lds32b(&As[buf][rlo][pf_swz_e(kb + 8, rlo)]);
                af[i][3] = pf_lds32b(&As[buf][rhi][pf_swz_e(kb + 8, rhi)]);
            }
            #pragma unroll
            for (int j = 0; j < NFRAG; j++) {
                const int col = wn * 64 + j * 8 + grp;
                bf[j][0] = pf_lds32b(&Bs[buf][col][pf_swz_e(kb,     col)]);
                bf[j][1] = pf_lds32b(&Bs[buf][col][pf_swz_e(kb + 8, col)]);
            }
            #pragma unroll
            for (int i = 0; i < MFRAG; i++)
                #pragma unroll
                for (int j = 0; j < NFRAG; j++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]),
                          "r"(bf[j][0]), "r"(bf[j][1]));
        }
        __syncthreads();
        buf ^= 1;
    }

    // Registers straight to global: c0/c1 (and c2/c3) are adjacent columns -> one bf16x2 store each.
    #pragma unroll
    for (int i = 0; i < MFRAG; i++) {
        #pragma unroll
        for (int j = 0; j < NFRAG; j++) {
            const int gn = n0 + wn * 64 + j * 8 + tig * 2;
            if (gn + 1 >= N) {                            // tail: scalar path
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int gm = m0 + wm * 32 + i * 16 + grp + (e >> 1) * 8;
                    const int cn = gn + (e & 1);
                    if (gm < M && cn < N) C[(size_t)gm * N + cn] = __float2bfloat16(acc[i][j][e]);
                }
                continue;
            }
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int gm = m0 + wm * 32 + i * 16 + grp + h * 8;
                if (gm >= M) continue;
                const __nv_bfloat162 v = __floats2bfloat162_rn(acc[i][j][h * 2], acc[i][j][h * 2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)gm * N + gn]) = v;
            }
        }
    }
}

// ============================================================================
// Elementwise helpers
// ============================================================================
__global__ void pf_swiglu_kernel(const __nv_bfloat16* __restrict__ gate,
                                 const __nv_bfloat16* __restrict__ up,
                                 __nv_bfloat16* __restrict__ h, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    h[i] = __float2bfloat16(pf_silu(pf_to_f(gate[i])) * pf_to_f(up[i]));
}

__global__ void pf_add_kernel(const __nv_bfloat16* __restrict__ a,
                              const __nv_bfloat16* __restrict__ b,
                              __nv_bfloat16* __restrict__ out, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(pf_to_f(a[i]) + pf_to_f(b[i]));
}

__global__ void pf_split_q_gate_kernel(const __nv_bfloat16* __restrict__ qraw,
                                       __nv_bfloat16* __restrict__ q,
                                       __nv_bfloat16* __restrict__ gate,
                                       int n_tokens, int n_heads, int head_dim) {
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long per = (long)n_heads * head_dim;
    if (gid >= (long)n_tokens * per) return;
    const int t = gid / per;
    const int r = gid % per;                       // within-token index
    const int h = r / head_dim, d = r % head_dim;
    const size_t src = (size_t)t * 2 * per + (size_t)h * 2 * head_dim + d;
    q[gid]    = qraw[src];
    gate[gid] = qraw[src + head_dim];
}

__global__ void pf_mul_sigmoid_kernel(__nv_bfloat16* __restrict__ attn,
                                      const __nv_bfloat16* __restrict__ gate, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    attn[i] = __float2bfloat16(pf_to_f(attn[i]) * pf_sigmoid(pf_to_f(gate[i])));
}

// ============================================================================
// Gated-DeltaNet causal conv + split + SiLU + L2-norm(q,k), scanned over N tokens.
// One block per output head (blockDim = head_dim); each thread owns one channel and
// keeps the (conv_kernel-1) most recent raw qkv values in registers.
// ============================================================================
__global__ void pf_gdn_conv_kernel(const __nv_bfloat16* __restrict__ qkv,
                                   const __nv_bfloat16* __restrict__ conv_w,
                                   __nv_bfloat16* __restrict__ conv_state,
                                   __nv_bfloat16* __restrict__ q,
                                   __nv_bfloat16* __restrict__ k,
                                   __nv_bfloat16* __restrict__ v,
                                   int n_tokens, int q_heads, int v_heads,
                                   int head_dim, int qkv_dim, int conv_kernel, float eps) {
    const int q_dim = q_heads * head_dim;
    const int v_dim = v_heads * head_dim;
    const int blk = blockIdx.x;                    // output head
    const int t   = threadIdx.x;                   // channel within head
    int d; __nv_bfloat16* out; int out_dim; int hh; bool do_norm;
    if (blk < q_heads)            { hh = blk;                d = hh * head_dim + t;               out = q; out_dim = q_dim; do_norm = true;  }
    else if (blk < 2 * q_heads)   { hh = blk - q_heads;      d = q_dim + hh * head_dim + t;       out = k; out_dim = q_dim; do_norm = true;  }
    else                          { hh = blk - 2 * q_heads;  d = 2 * q_dim + hh * head_dim + t;   out = v; out_dim = v_dim; do_norm = false; }

    float w[8];                                    // conv taps (conv_kernel <= 8)
    #pragma unroll
    for (int c = 0; c < 8; c++) w[c] = (c < conv_kernel) ? pf_to_f(conv_w[(size_t)d * conv_kernel + c]) : 0.f;
    float hist[7];                                 // last conv_kernel-1 raw qkv (<=7)
    #pragma unroll
    for (int c = 0; c < 7; c++) hist[c] = 0.f;

    __shared__ float sw[32];
    const int nwarp = (blockDim.x + 31) / 32;
    for (int tok = 0; tok < n_tokens; tok++) {
        const float cur = pf_to_f(qkv[(size_t)tok * qkv_dim + d]);
        float y = cur * w[conv_kernel - 1];
        #pragma unroll
        for (int c = 0; c < 7; c++)
            if (c < conv_kernel - 1) y += hist[c] * w[c];
        // shift history: hist[0..K-3] <- hist[1..K-2], hist[K-2] <- cur
        #pragma unroll
        for (int c = 0; c < 6; c++)
            if (c < conv_kernel - 2) hist[c] = hist[c + 1];
        if (conv_kernel >= 2) hist[conv_kernel - 2] = cur;

        float cval = pf_silu(y);
        if (do_norm) {
            float ss = pf_wsum(cval * cval);
            if ((t & 31) == 0) sw[t >> 5] = ss;
            __syncthreads();
            if (t < 32) {
                float vv = (t < nwarp) ? sw[t] : 0.f;
                vv = pf_wsum(vv);
                if (t == 0) sw[0] = rsqrtf(vv + eps);
            }
            __syncthreads();
            cval *= sw[0];
            // sw[] is reused by the next token's reduction, so without a barrier here a warp
            // that has already advanced can overwrite sw[t>>5] while a slower warp is still
            // reading sw[0] for THIS token (#705). do_norm is block-uniform, so the barrier is
            // reached by every thread in the block.
            __syncthreads();
        }
        out[(size_t)tok * out_dim + hh * head_dim + t] = __float2bfloat16(cval);
    }
    // persist the conv window (last conv_kernel-1 raw qkv) in the decode layout.
    #pragma unroll
    for (int c = 0; c < 7; c++)
        if (c < conv_kernel - 1) conv_state[(size_t)c * qkv_dim + d] = __float2bfloat16(hist[c]);
}

// ============================================================================
// Token-parallel variant of pf_gdn_conv_kernel. The causal conv needs only the
// conv_kernel-1 previous RAW qkv rows, which are all in the input buffer (the
// prompt starts at position 0, so the incoming conv state is zero) — the token
// loop above is a needless serialization. One block per (token, output head);
// same taps, SiLU, and L2-norm math, same conv_state layout on exit.
// Token on grid.x (grid.y stays under the 65535 cap at any context).
// ============================================================================
__global__ void pf_gdn_conv_par_kernel(const __nv_bfloat16* __restrict__ qkv,
                                       const __nv_bfloat16* __restrict__ conv_w,
                                       __nv_bfloat16* __restrict__ conv_state,
                                       __nv_bfloat16* __restrict__ q,
                                       __nv_bfloat16* __restrict__ k,
                                       __nv_bfloat16* __restrict__ v,
                                       int n_tokens, int q_heads, int v_heads,
                                       int head_dim, int qkv_dim, int conv_kernel, float eps) {
    const int q_dim = q_heads * head_dim;
    const int v_dim = v_heads * head_dim;
    const int tok = blockIdx.x;
    const int blk = blockIdx.y;                    // output head
    const int t   = threadIdx.x;                   // channel within head
    int d; __nv_bfloat16* out; int out_dim; int hh; bool do_norm;
    if (blk < q_heads)            { hh = blk;                d = hh * head_dim + t;               out = q; out_dim = q_dim; do_norm = true;  }
    else if (blk < 2 * q_heads)   { hh = blk - q_heads;      d = q_dim + hh * head_dim + t;       out = k; out_dim = q_dim; do_norm = true;  }
    else                          { hh = blk - 2 * q_heads;  d = 2 * q_dim + hh * head_dim + t;   out = v; out_dim = v_dim; do_norm = false; }

    float y = 0.f;
    #pragma unroll
    for (int c = 0; c < 8; c++) {
        if (c >= conv_kernel) break;
        const int src = tok - (conv_kernel - 1) + c;
        if (src >= 0)
            y += pf_to_f(conv_w[(size_t)d * conv_kernel + c]) * pf_to_f(qkv[(size_t)src * qkv_dim + d]);
    }

    float cval = pf_silu(y);
    if (do_norm) {
        __shared__ float sw[32];
        const int nwarp = (blockDim.x + 31) / 32;
        float ss = pf_wsum(cval * cval);
        if ((t & 31) == 0) sw[t >> 5] = ss;
        __syncthreads();
        if (t < 32) {
            float vv = (t < nwarp) ? sw[t] : 0.f;
            vv = pf_wsum(vv);
            if (t == 0) sw[0] = rsqrtf(vv + eps);
        }
        __syncthreads();
        cval *= sw[0];
    }
    out[(size_t)tok * out_dim + hh * head_dim + t] = __float2bfloat16(cval);

    // persist the conv window (last conv_kernel-1 raw qkv) in the decode layout.
    if (tok == n_tokens - 1) {
        #pragma unroll
        for (int c = 0; c < 7; c++) {
            if (c >= conv_kernel - 1) break;
            const int src = n_tokens - 1 - (conv_kernel - 2 - c);
            conv_state[(size_t)c * qkv_dim + d] =
                (src >= 0) ? qkv[(size_t)src * qkv_dim + d] : __float2bfloat16(0.f);
        }
    }
}

// ============================================================================
// Token-tiled variant of pf_gdn_conv_par_kernel. That kernel launches one block per (token, head),
// so every raw qkv row is re-read from DRAM by the conv_kernel overlapping token blocks -- a ~4x
// read amplification of the largest input (measured at its DRAM roofline *including* that
// amplification). One block per (16-token tile, head) instead: each thread seeds a register
// window with its channel's conv_kernel-1 boundary rows and slides it across the tile, so the
// tile reads 16+conv_kernel-1 rows for 16 outputs (~1.2x). Tap order, SiLU, and the two-stage
// L2-norm reduction are unchanged, and the boundary "add w*0" equals the parallel kernel's
// predicated skip exactly -- output is bit-identical.
// ============================================================================
constexpr int PF_CONV_TT = 16;                     // tokens per block
__global__ void pf_gdn_conv_tile_kernel(const __nv_bfloat16* __restrict__ qkv,
                                        const __nv_bfloat16* __restrict__ conv_w,
                                        __nv_bfloat16* __restrict__ conv_state,
                                        __nv_bfloat16* __restrict__ q,
                                        __nv_bfloat16* __restrict__ k,
                                        __nv_bfloat16* __restrict__ v,
                                        int n_tokens, int q_heads, int v_heads,
                                        int head_dim, int qkv_dim, int conv_kernel, float eps) {
    const int q_dim = q_heads * head_dim;
    const int v_dim = v_heads * head_dim;
    const int tok0 = blockIdx.x * PF_CONV_TT;
    const int blk = blockIdx.y;                    // output head
    const int t   = threadIdx.x;                   // channel within head
    int d; __nv_bfloat16* out; int out_dim; int hh; bool do_norm;
    if (blk < q_heads)            { hh = blk;                d = hh * head_dim + t;               out = q; out_dim = q_dim; do_norm = true;  }
    else if (blk < 2 * q_heads)   { hh = blk - q_heads;      d = q_dim + hh * head_dim + t;       out = k; out_dim = q_dim; do_norm = true;  }
    else                          { hh = blk - 2 * q_heads;  d = 2 * q_dim + hh * head_dim + t;   out = v; out_dim = v_dim; do_norm = false; }

    float w[8];                                    // conv taps (conv_kernel <= 8)
    #pragma unroll
    for (int c = 0; c < 8; c++) w[c] = (c < conv_kernel) ? pf_to_f(conv_w[(size_t)d * conv_kernel + c]) : 0.f;
    float hist[7];                                 // rolling raw-qkv window for this channel (<=7)
    #pragma unroll
    for (int c = 0; c < 7; c++) {
        const int src = tok0 - (conv_kernel - 1) + c;
        hist[c] = (c < conv_kernel - 1 && src >= 0) ? pf_to_f(qkv[(size_t)src * qkv_dim + d]) : 0.f;
    }

    __shared__ float sw[32];
    const int nwarp = (blockDim.x + 31) / 32;
    #pragma unroll
    for (int tt = 0; tt < PF_CONV_TT; tt++) {
        const int tok = tok0 + tt;
        if (tok >= n_tokens) break;                // uniform across the block
        const float cur = pf_to_f(qkv[(size_t)tok * qkv_dim + d]);
        float y = 0.f;
        #pragma unroll
        for (int c = 0; c < 7; c++)
            if (c < conv_kernel - 1) y += w[c] * hist[c];
        y += w[conv_kernel - 1] * cur;
        #pragma unroll
        for (int c = 0; c < 6; c++)
            if (c < conv_kernel - 2) hist[c] = hist[c + 1];
        if (conv_kernel >= 2) hist[conv_kernel - 2] = cur;

        float cval = pf_silu(y);
        if (do_norm) {
            float ss = pf_wsum(cval * cval);
            if ((t & 31) == 0) sw[t >> 5] = ss;
            __syncthreads();
            if (t < 32) {
                float vv = (t < nwarp) ? sw[t] : 0.f;
                vv = pf_wsum(vv);
                if (t == 0) sw[0] = rsqrtf(vv + eps);
            }
            __syncthreads();
            cval *= sw[0];
            // sw[] is reused by the next token's reduction, so without a barrier here a warp
            // that has already advanced can overwrite sw[t>>5] while a slower warp is still
            // reading sw[0] for THIS token (#705). do_norm is block-uniform, so the barrier is
            // reached by every thread in the block.
            __syncthreads();
        }
        out[(size_t)tok * out_dim + hh * head_dim + t] = __float2bfloat16(cval);
    }

    // persist the conv window (last conv_kernel-1 raw qkv) in the decode layout.
    if (tok0 <= n_tokens - 1 && n_tokens - 1 < tok0 + PF_CONV_TT) {
        #pragma unroll
        for (int c = 0; c < 7; c++) {
            if (c >= conv_kernel - 1) break;
            const int src = n_tokens - 1 - (conv_kernel - 2 - c);
            conv_state[(size_t)c * qkv_dim + d] =
                (src >= 0) ? qkv[(size_t)src * qkv_dim + d] : __float2bfloat16(0.f);
        }
    }
}

// ============================================================================
// Gated-DeltaNet sequential recurrence, scanned over N tokens in one launch.
// Warp-per-state-column, register-cached column, transposed final state layout —
// bit-identical to running gdn_ar_fast (the SPARKINFER_GDN_FAST default) N times.
// ============================================================================
template <int COLS, int HEAD_DIM>
__global__ void pf_gdn_scan_kernel(const __nv_bfloat16* __restrict__ q,
                                   const __nv_bfloat16* __restrict__ k,
                                   const __nv_bfloat16* __restrict__ v,
                                   const __nv_bfloat16* __restrict__ alpha,
                                   const __nv_bfloat16* __restrict__ beta,
                                   const __nv_bfloat16* __restrict__ dt,
                                   const __nv_bfloat16* __restrict__ a,
                                   float* __restrict__ state,
                                   __nv_bfloat16* __restrict__ out,
                                   int n_tokens, int q_heads, int v_heads, bool qh_block) {
    constexpr int NROW = HEAD_DIM / 32;
    const int vh   = blockIdx.x;
    const int j    = blockIdx.y * COLS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (vh >= v_heads || j >= HEAD_DIM) return;
    const int qh = qh_block ? (vh / (v_heads / q_heads)) : (vh % q_heads);
    const int q_dim = q_heads * HEAD_DIM;
    const int v_dim = v_heads * HEAD_DIM;
    const float scale = rsqrtf((float)HEAD_DIM);
    const float a_h  = pf_to_f(a[vh]);
    const float dt_h = pf_to_f(dt[vh]);

    float sloc[NROW];
    #pragma unroll
    for (int r = 0; r < NROW; r++) sloc[r] = 0.f;   // fresh prefill: state starts at zero

    for (int t = 0; t < n_tokens; t++) {
        const float bb = pf_sigmoid(pf_to_f(beta[(size_t)t * v_heads + vh]));
        const float g  = __expf(pf_softplus(pf_to_f(alpha[(size_t)t * v_heads + vh]) + dt_h) * a_h);
        const __nv_bfloat16* qp = q + ((size_t)t * q_dim + qh * HEAD_DIM);
        const __nv_bfloat16* kp = k + ((size_t)t * q_dim + qh * HEAD_DIM);
        const __nv_bfloat16* vp = v + ((size_t)t * v_dim + vh * HEAD_DIM);
        float part_sk = 0.f;
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int i = lane + r * 32;
            part_sk += sloc[r] * pf_to_f(kp[i]);
        }
        const float sk = g * pf_wsum(part_sk);
        const float delta = (pf_to_f(vp[j]) - sk) * bb;
        float part_y = 0.f;
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int i = lane + r * 32;
            const float s_new = sloc[r] * g + pf_to_f(kp[i]) * delta;
            sloc[r] = s_new;
            part_y += s_new * pf_to_f(qp[i]) * scale;
        }
        const float y = pf_wsum(part_y);
        if (lane == 0) out[(size_t)t * v_dim + vh * HEAD_DIM + j] = __float2bfloat16(y);
    }
    float* col = state + ((size_t)vh * HEAD_DIM + j) * HEAD_DIM;   // transposed [vh][col][row]
    #pragma unroll
    for (int r = 0; r < NROW; r++) col[lane + r * 32] = sloc[r];
}

// DFlash verification scan. Arithmetic and reduction order match pf_gdn_scan_kernel, but the
// register state is seeded from decode and each candidate's post-token state is checkpointed.
// The live state is read-only, so a rejected suffix never needs rollback or replay.
template <int COLS, int HEAD_DIM, bool WRITE_CHECKPOINT>
__global__ void df_gdn_scan_checkpoint_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const __nv_bfloat16* __restrict__ alpha,
    const __nv_bfloat16* __restrict__ beta, const __nv_bfloat16* __restrict__ dt,
    const __nv_bfloat16* __restrict__ a, const float* __restrict__ live_state,
    __nv_bfloat16* __restrict__ out, float* __restrict__ checkpoints,
    int n_tokens, int q_heads, int v_heads, bool qh_block) {
    constexpr int NROW = HEAD_DIM / 32;
    const int vh = blockIdx.x;
    const int j = blockIdx.y * COLS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (vh >= v_heads || j >= HEAD_DIM) return;
    const int qh = qh_block ? (vh / (v_heads / q_heads)) : (vh % q_heads);
    const int q_dim = q_heads * HEAD_DIM;
    const int v_dim = v_heads * HEAD_DIM;
    const size_t state_elems = (size_t)v_heads * HEAD_DIM * HEAD_DIM;
    const size_t col_off = ((size_t)vh * HEAD_DIM + j) * HEAD_DIM;
    const float scale = rsqrtf((float)HEAD_DIM);
    const float a_h = pf_to_f(a[vh]);
    const float dt_h = pf_to_f(dt[vh]);

    float sloc[NROW];
    #pragma unroll
    for (int r = 0; r < NROW; r++) sloc[r] = live_state[col_off + lane + r * 32];

    for (int t = 0; t < n_tokens; t++) {
        const float bb = pf_sigmoid(pf_to_f(beta[(size_t)t * v_heads + vh]));
        const float g = __expf(pf_softplus(pf_to_f(alpha[(size_t)t * v_heads + vh]) + dt_h) * a_h);
        const __nv_bfloat16* qp = q + (size_t)t * q_dim + qh * HEAD_DIM;
        const __nv_bfloat16* kp = k + (size_t)t * q_dim + qh * HEAD_DIM;
        const __nv_bfloat16* vp = v + (size_t)t * v_dim + vh * HEAD_DIM;
        float part_sk = 0.f;
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int i = lane + r * 32;
            part_sk += sloc[r] * pf_to_f(kp[i]);
        }
        const float sk = g * pf_wsum(part_sk);
        const float delta = (pf_to_f(vp[j]) - sk) * bb;
        float part_y = 0.f;
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int i = lane + r * 32;
            const float s_new = sloc[r] * g + pf_to_f(kp[i]) * delta;
            sloc[r] = s_new;
            part_y += s_new * pf_to_f(qp[i]) * scale;
            if constexpr (WRITE_CHECKPOINT)
                checkpoints[(size_t)t * state_elems + col_off + i] = s_new;
        }
        const float y = pf_wsum(part_y);
        if (lane == 0) out[(size_t)t * v_dim + vh * HEAD_DIM + j] = __float2bfloat16(y);
    }
}

// Commit only the accepted compact recurrence inputs. Verification already retained k/v and the
// scalar gates for each row, so materializing N full state checkpoints is unnecessary: replay the
// accepted prefix once into the live state. The update expression and reduction order are the
// same as df_gdn_scan_checkpoint_kernel; q/output work is intentionally omitted because commit
// happens after posterior selection.
template <int COLS, int HEAD_DIM>
__global__ void df_gdn_scan_commit_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ alpha, const __nv_bfloat16* __restrict__ beta,
    const __nv_bfloat16* __restrict__ dt, const __nv_bfloat16* __restrict__ a,
    float* __restrict__ live_state, int n_tokens, int q_heads, int v_heads, bool qh_block) {
    constexpr int NROW = HEAD_DIM / 32;
    const int vh = blockIdx.x;
    const int j = blockIdx.y * COLS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (vh >= v_heads || j >= HEAD_DIM) return;
    const int qh = qh_block ? (vh / (v_heads / q_heads)) : (vh % q_heads);
    const int q_dim = q_heads * HEAD_DIM;
    const int v_dim = v_heads * HEAD_DIM;
    const size_t col_off = ((size_t)vh * HEAD_DIM + j) * HEAD_DIM;
    const float a_h = pf_to_f(a[vh]);
    const float dt_h = pf_to_f(dt[vh]);

    float sloc[NROW];
    #pragma unroll
    for (int r = 0; r < NROW; r++) sloc[r] = live_state[col_off + lane + r * 32];
    for (int t = 0; t < n_tokens; t++) {
        const float bb = pf_sigmoid(pf_to_f(beta[(size_t)t * v_heads + vh]));
        const float g = __expf(pf_softplus(pf_to_f(alpha[(size_t)t * v_heads + vh]) + dt_h) * a_h);
        const __nv_bfloat16* kp = k + (size_t)t * q_dim + qh * HEAD_DIM;
        const __nv_bfloat16* vp = v + (size_t)t * v_dim + vh * HEAD_DIM;
        float part_sk = 0.f;
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int i = lane + r * 32;
            part_sk += sloc[r] * pf_to_f(kp[i]);
        }
        const float sk = g * pf_wsum(part_sk);
        const float delta = (pf_to_f(vp[j]) - sk) * bb;
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int i = lane + r * 32;
            sloc[r] = sloc[r] * g + pf_to_f(kp[i]) * delta;
        }
    }
    #pragma unroll
    for (int r = 0; r < NROW; r++) live_state[col_off + lane + r * 32] = sloc[r];
}

// The convolution state is only the last conv_kernel-1 raw qkv rows. Commit therefore reduces to
// a window splice from the original live history and the accepted qkv prefix; no convolution or
// normalization needs to be repeated.
__global__ void df_gdn_conv_commit_kernel(const __nv_bfloat16* __restrict__ qkv,
                                          __nv_bfloat16* __restrict__ live_state,
                                          int n_tokens, int qkv_dim, int conv_kernel) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= qkv_dim) return;
    const int history = conv_kernel - 1;
    for (int c = 0; c < history; c++) {
        const int src_tok = n_tokens - history + c;
        live_state[(size_t)c * qkv_dim + d] = src_tok >= 0
            ? qkv[(size_t)src_tok * qkv_dim + d]
            : live_state[(size_t)(c + n_tokens) * qkv_dim + d];
    }
}

// DFlash causal-convolution counterpart. A single block owns one output head, seeds its rolling
// window from decode state, and records the post-token window in decode layout for every row.
template <bool WRITE_CHECKPOINT>
__global__ void df_gdn_conv_checkpoint_kernel(
    const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ conv_w,
    const __nv_bfloat16* __restrict__ live_state, __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ checkpoints, int n_tokens, int q_heads, int v_heads,
    int head_dim, int qkv_dim, int conv_kernel, float eps) {
    const int q_dim = q_heads * head_dim;
    const int v_dim = v_heads * head_dim;
    const int blk = blockIdx.x, t = threadIdx.x;
    int d, out_dim, hh; __nv_bfloat16* out; bool do_norm;
    if (blk < q_heads) { hh = blk; d = hh * head_dim + t; out = q; out_dim = q_dim; do_norm = true; }
    else if (blk < 2 * q_heads) { hh = blk - q_heads; d = q_dim + hh * head_dim + t; out = k; out_dim = q_dim; do_norm = true; }
    else { hh = blk - 2 * q_heads; d = 2 * q_dim + hh * head_dim + t; out = v; out_dim = v_dim; do_norm = false; }
    float w[8], hist[7];
    #pragma unroll
    for (int c = 0; c < 8; c++) w[c] = c < conv_kernel ? pf_to_f(conv_w[(size_t)d * conv_kernel + c]) : 0.f;
    #pragma unroll
    for (int c = 0; c < 7; c++) hist[c] = c < conv_kernel - 1 ? pf_to_f(live_state[(size_t)c * qkv_dim + d]) : 0.f;
    __shared__ float sw[32];
    const int nwarp = (blockDim.x + 31) / 32;
    const size_t state_elems = (size_t)(conv_kernel - 1) * qkv_dim;
    for (int tok = 0; tok < n_tokens; tok++) {
        const float cur = pf_to_f(qkv[(size_t)tok * qkv_dim + d]);
        // Accumulate in EXACTLY the order the single-token decode conv uses
        // (conv_split_l2norm_fused_kernel in qwen36.cu): every history tap in ascending order
        // first, then the current token's tap last. Float addition is not associative, so summing
        // `cur` first -- as this kernel used to -- lands a different low bit in q/k/v than decode
        // does for identical inputs. That difference is ~1 ulp, but k and v feed the GDN
        // recurrence (S = S*g + k*delta), whose state persists across decode steps, so it
        // compounds until it flips an argmax and DFlash stops reproducing AR (#712). Matching the
        // order makes the batched and single-token paths bit-identical.
        float y = 0.f;
        #pragma unroll
        for (int c = 0; c < 7; c++) if (c < conv_kernel - 1) y += hist[c] * w[c];
        y += cur * w[conv_kernel - 1];
        #pragma unroll
        for (int c = 0; c < 6; c++) if (c < conv_kernel - 2) hist[c] = hist[c + 1];
        if (conv_kernel >= 2) hist[conv_kernel - 2] = cur;
        float cval = pf_silu(y);
        if (do_norm) {
            float ss = pf_wsum(cval * cval);
            if ((t & 31) == 0) sw[t >> 5] = ss;
            __syncthreads();
            if (t < 32) {
                float vv = t < nwarp ? sw[t] : 0.f;
                vv = pf_wsum(vv);
                if (t == 0) sw[0] = rsqrtf(vv + eps);
            }
            __syncthreads();
            cval *= sw[0];
            // Same sw[] reuse hazard as the prefill convs above (#705). This kernel is the one
            // the DFlash compact verify runs, and a non-deterministic q/k norm here feeds k and v
            // straight into the GDN recurrence, whose state persists across decode steps -- so the
            // race compounds until it flips an argmax and the batched path stops reproducing AR.
            __syncthreads();
        }
        out[(size_t)tok * out_dim + hh * head_dim + t] = __float2bfloat16(cval);
        if constexpr (WRITE_CHECKPOINT) {
            #pragma unroll
            for (int c = 0; c < 7; c++) if (c < conv_kernel - 1)
                checkpoints[(size_t)tok * state_elems + (size_t)c * qkv_dim + d] = __float2bfloat16(hist[c]);
        }
    }
}

// ============================================================================
// Batched gated RMSNorm: out = (x/rms(x)) * weight * silu(z), per (token, v_head).
// One warp per (token, v_head). Mirrors gated_norm_warp_kernel.
// ============================================================================
template <int HEAD_DIM>
__global__ void pf_gated_norm_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ z,
                                     const __nv_bfloat16* __restrict__ weight,
                                     __nv_bfloat16* __restrict__ out,
                                     int n_tokens, int v_heads, float eps) {
    constexpr int NROW = HEAD_DIM / 32;
    const int h    = blockIdx.y;                    // v_head
    const int t    = blockIdx.x;                    // token (grid.x avoids the 65535 grid.y cap)
    const int lane = threadIdx.x & 31;
    if (h >= v_heads || t >= n_tokens) return;
    const size_t base = ((size_t)t * v_heads + h) * HEAD_DIM;
    float xv[NROW], zv[NROW], wv[NROW], ss = 0.f;
    #pragma unroll
    for (int r = 0; r < NROW; r++) {
        const int d = lane + r * 32;
        xv[r] = pf_to_f(x[base + d]);
        zv[r] = pf_to_f(z[base + d]);
        wv[r] = pf_to_f(weight[d]);
        ss += xv[r] * xv[r];
    }
    const float inv = rsqrtf(pf_wsum(ss) / HEAD_DIM + eps);
    #pragma unroll
    for (int r = 0; r < NROW; r++) {
        const int d = lane + r * 32;
        out[base + d] = __float2bfloat16(xv[r] * inv * wv[r] * pf_silu(zv[r]));
    }
}

// ============================================================================
// Full-attention prefill: QK-norm + partial-RoPE (q,k in place) + int8 KV write into
// the single-sequence paged pool. Mirrors qknorm_rope_kv_partial_int8_kernel (rope.cu)
// but indexes the block table for a single sequence and grids over all N prompt tokens.
// grid = (n_q_heads + 2*n_kv_heads, N); blockDim = head_dim.
// ============================================================================
__global__ void pf_qknorm_rope_kv_int8_kernel(
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_w, const __nv_bfloat16* __restrict__ k_w,
    signed char* __restrict__ k_pool, signed char* __restrict__ v_pool,
    __half* __restrict__ k_scale, __half* __restrict__ v_scale,
    const int* __restrict__ block_table,
    int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim, float theta, float eps,
    int block_size, int max_blocks_per_seq) {
    const int tok  = blockIdx.x;                    // token (grid.x avoids 65535 grid.y cap)
    const int unit = blockIdx.y;
    const int t    = threadIdx.x;
    const int rhalf = rotary_dim >> 1;
    const int pos   = tok;                              // prefill: position == token index
    const int blk   = pos / block_size, within = pos % block_size;
    const int phys  = block_table[blk];                // single sequence
    const size_t ctok = (size_t)phys * block_size + within;

    extern __shared__ float s_h[];
    __shared__ float s_warp[32];
    __shared__ float s_red[8];

    if (unit < n_q_heads) {                            // Q: RMSNorm + partial RoPE in place
        const size_t base = ((size_t)tok * n_q_heads + unit) * head_dim;
        const float xv = pf_to_f(q[base + t]);
        float ss = xv * xv;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
        if ((t & 31) == 0) s_warp[t >> 5] = ss;
        __syncthreads();
        if (t < 32) {
            float vv = (t < (head_dim + 31) / 32) ? s_warp[t] : 0.f;
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) vv += __shfl_xor_sync(0xffffffff, vv, m);
            if (t == 0) s_warp[0] = rsqrtf(vv / head_dim + eps);
        }
        __syncthreads();
        s_h[t] = pf_to_f(__float2bfloat16(xv * s_warp[0] * pf_to_f(q_w[t])));
        __syncthreads();
        // Rotate in s_h, then write the whole head vector — see qknorm_rope_kv_partial_int8_kernel
        // (rope.cu). Storing only t < rhalf left dims rope_dim..head_dim-1 (192 of 256 at
        // hd256/rope_dim=64) holding the raw wq output, un-normalized. The K branch below already
        // does this correctly via its `else { val = s_h[t]; }` arm; Q was asymmetric with it.
        if (t < rhalf) {
            const float freq = __powf(theta, -2.f * (float)t / (float)rotary_dim);
            const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
            const float x0 = s_h[t], x1 = s_h[t + rhalf];
            s_h[t]         = pf_to_f(__float2bfloat16(x0 * c - x1 * s));
            s_h[t + rhalf] = pf_to_f(__float2bfloat16(x1 * c + x0 * s));
        }
        __syncthreads();
        if (t < head_dim) q[base + t] = __float2bfloat16(s_h[t]);
        return;
    }

    const bool is_k = unit < n_q_heads + n_kv_heads;
    const int  hh   = is_k ? (unit - n_q_heads) : (unit - n_q_heads - n_kv_heads);
    const size_t base = ((size_t)tok * n_kv_heads + hh) * head_dim;
    const size_t dst  = (ctok * n_kv_heads + hh) * head_dim;

    float val;
    if (is_k) {
        const float xv = pf_to_f(k[base + t]);
        float ss = xv * xv;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
        if ((t & 31) == 0) s_warp[t >> 5] = ss;
        __syncthreads();
        if (t < 32) {
            float vv = (t < (head_dim + 31) / 32) ? s_warp[t] : 0.f;
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) vv += __shfl_xor_sync(0xffffffff, vv, m);
            if (t == 0) s_warp[0] = rsqrtf(vv / head_dim + eps);
        }
        __syncthreads();
        s_h[t] = pf_to_f(__float2bfloat16(xv * s_warp[0] * pf_to_f(k_w[t])));
        __syncthreads();
        if (t < rotary_dim) {
            const int i = (t < rhalf) ? t : (t - rhalf);
            const float freq = __powf(theta, -2.f * (float)i / (float)rotary_dim);
            const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
            const float x0 = s_h[i], x1 = s_h[i + rhalf];
            val = (t < rhalf) ? (x0 * c - x1 * s) : (x1 * c + x0 * s);
        } else {
            val = s_h[t];
        }
    } else {
        val = pf_to_f(v[base + t]);
    }

    float amax = fabsf(val);
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m));
    if ((t & 31) == 0) s_red[t >> 5] = amax;
    __syncthreads();
    if (t == 0) {
        float aa = 0.f;
        for (int w = 0; w < (head_dim >> 5); w++) aa = fmaxf(aa, s_red[w]);
        s_red[0] = aa;
    }
    __syncthreads();
    const float d  = s_red[0] / 127.0f;
    const int   qi = (s_red[0] == 0.f) ? 0 : (int)roundf(val / d);
    if (is_k) {
        k_pool[dst + t] = (signed char)qi;
        if (t == 0) k_scale[ctok * n_kv_heads + hh] = __float2half(d);
    } else {
        v_pool[dst + t] = (signed char)qi;
        if (t == 0) v_scale[ctok * n_kv_heads + hh] = __float2half(d);
    }
}

// ============================================================================
// Muse Glimmer full-attention prefill: QK-norm + NORMAL RoPE (consecutive-pair,
// LLAMA_ROPE_TYPE_NORM) + BF16 KV write into the single-sequence paged pool. The
// bf16 counterpart of pf_qknorm_rope_kv_int8_kernel: same prefill block-table
// layout (pos==tok, block_table[blk]), but writes bf16 K/V (no int8 quant/scale)
// and rotates consecutive pairs (2i,2i+1) instead of NeoX (i,i+half) -- Muse
// Glimmer is LLAMA_ROPE_TYPE_NORM in llama.cpp (see rope_kv_append_normal_kernel,
// rope.cu). rotary_dim>0 => rope (SWA layers, rotary_dim=head_dim); rotary_dim==0
// => NoPE (global layers, QK-norm then append as-is). Bit-for-bit the same KV a
// forward_token decode step writes via launch_rmsnorm + launch_rope_kv_append_normal
// / launch_kv_append, so a subsequent decode reads a consistent cache.
// grid = (N, n_q_heads + 2*n_kv_heads); blockDim = head_dim.
// ============================================================================
__global__ void pf_qknorm_ropenorm_kv_bf16_kernel(
    const __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ q_out,
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_w, const __nv_bfloat16* __restrict__ k_w,
    __nv_bfloat16* __restrict__ k_pool, __nv_bfloat16* __restrict__ v_pool,
    const int* __restrict__ block_table,
    int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim, float theta, float eps,
    int block_size, int max_blocks_per_seq, int q_ld, int kv_ld) {
    const int tok  = blockIdx.x;
    const int unit = blockIdx.y;
    const int t    = threadIdx.x;
    const int pos  = tok;                              // prefill: position == token index
    const int blk  = pos / block_size, within = pos % block_size;
    const int phys = block_table[blk];                // single sequence
    const size_t ctok = (size_t)phys * block_size + within;

    extern __shared__ float s_h[];
    __shared__ float s_warp[32];

    const bool is_q = unit < n_q_heads;
    const bool is_k = !is_q && unit < n_q_heads + n_kv_heads;
    if (unit >= n_q_heads + 2 * n_kv_heads) return;

    if (is_q || is_k) {
        const int hh = is_q ? unit : (unit - n_q_heads);
        // Row stride of the SOURCE: tight (heads*head_dim) for split inputs, or the stacked
        // q|gate|k|v GEMM output's full row width when reading it in place (no split copy).
        const size_t base = (size_t)tok * (is_q ? q_ld : kv_ld) + (size_t)hh * head_dim;
        const __nv_bfloat16* src = is_q ? q : k;
        const __nv_bfloat16* nrm = is_q ? q_w : k_w;
        const float xv = pf_to_f(src[base + t]);
        float ss = xv * xv;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
        if ((t & 31) == 0) s_warp[t >> 5] = ss;
        __syncthreads();
        if (t < 32) {
            float vv = (t < (head_dim + 31) / 32) ? s_warp[t] : 0.f;
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) vv += __shfl_xor_sync(0xffffffff, vv, m);
            if (t == 0) s_warp[0] = rsqrtf(vv / head_dim + eps);
        }
        __syncthreads();
        s_h[t] = pf_to_f(__float2bfloat16(xv * s_warp[0] * pf_to_f(nrm[t])));
        __syncthreads();
        // NORMAL (consecutive-pair) RoPE of the normed head vector: pair (2i, 2i+1).
        float out = s_h[t];
        if (rotary_dim > 0 && t < rotary_dim) {
            const int i = t >> 1;
            const float freq = __powf(theta, -2.f * (float)i / (float)rotary_dim);
            const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
            const float x0 = s_h[2 * i], x1 = s_h[2 * i + 1];
            out = ((t & 1) == 0) ? (x0 * c - x1 * s) : (x0 * s + x1 * c);
        }
        if (is_q) {
            // The normed/roped q always lands TIGHT ([tok, n_q_heads*head_dim]) -- the
            // attention kernels read that layout regardless of where the raw q came from.
            q_out[((size_t)tok * n_q_heads + hh) * head_dim + t] = __float2bfloat16(out);
        } else {
            const size_t dst = (ctok * n_kv_heads + hh) * head_dim;
            k_pool[dst + t] = __float2bfloat16(out);
        }
    } else {                                          // V: append as-is (no norm, no rope)
        const int hh = unit - n_q_heads - n_kv_heads;
        const size_t base = (size_t)tok * kv_ld + (size_t)hh * head_dim;
        const size_t dst  = (ctok * n_kv_heads + hh) * head_dim;
        v_pool[dst + t] = v[base + t];
    }
}

// ============================================================================
// Causal attention over the paged int8 KV pool. One warp per (token, q-head);
// online softmax over keys 0..token. head_dim <= 256 -> ELEMS <= 8 per lane.
// ============================================================================
template <int HEAD_DIM>
__global__ void pf_attn_int8_paged_kernel(
    const __nv_bfloat16* __restrict__ q, const signed char* __restrict__ k_pool,
    const signed char* __restrict__ v_pool, const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale, const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ attn, int n_tokens, int n_q_heads, int n_kv_heads,
    int block_size, int max_blocks_per_seq, float scale) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const int head = blockIdx.y;                    // q-head
    const int qtok = blockIdx.x;                    // query token (grid.x avoids 65535 grid.y cap)
    const int lane = threadIdx.x;
    if (head >= n_q_heads || qtok >= n_tokens) return;
    const int kv_head = head / (n_q_heads / n_kv_heads);

    const size_t q_off = ((size_t)qtok * n_q_heads + head) * HEAD_DIM;
    float q_reg[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) q_reg[e] = pf_to_f(q[q_off + lane + e * 32]);

    float m = -1e30f, l = 0.f, acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    for (int kpos = 0; kpos <= qtok; kpos++) {
        const int blk = kpos / block_size, within = kpos % block_size;
        const int phys = block_table[blk];
        const size_t ckt = ((size_t)phys * block_size + within);
        const size_t kv_off = (ckt * n_kv_heads + kv_head) * HEAD_DIM;
        const float ksc = __half2float(k_scale[ckt * n_kv_heads + kv_head]);
        float partial = 0.f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++)
            partial += q_reg[e] * ((float)k_pool[kv_off + lane + e * 32] * ksc);
        const float score = pf_wsum(partial) * scale;

        const float m_new = fmaxf(m, score);
        const float corr  = __expf(m - m_new);
        const float p     = __expf(score - m_new);
        l = l * corr + p;
        const float vsc = __half2float(v_scale[ckt * n_kv_heads + kv_head]);
        #pragma unroll
        for (int e = 0; e < ELEMS; e++)
            acc[e] = acc[e] * corr + p * ((float)v_pool[kv_off + lane + e * 32] * vsc);
        m = m_new;
    }
    const float inv = (l > 0.f) ? (1.f / l) : 0.f;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) attn[q_off + lane + e * 32] = __float2bfloat16(acc[e] * inv);
}

// ============================================================================
// bf16-KV batched prefill for the hybrid dense models (Qwen3.8): a NeoX-RoPE
// QK-norm/append and a full-causal paged attention that need no int8 KV.
//
// Batched prefill previously required an int8 KV cache, and the bench only turns int8 on at
// ctx>=4096, so Qwen3.8's prefill@128 declined to the sequential per-token path -- 88.0 tok/s
// against 84.2 tok/s of decode, i.e. 128 full weight-streaming passes to fill 128 positions.
// These are bf16 twins of pf_qknorm_rope_kv_int8_kernel and pf_attn_int8_paged_kernel<256>:
// same schedule, same fp32 online softmax, same NeoX (i, i+half) partial rotation, with the
// int8 quantize/dequantize removed and the pool typed bf16 -- which is exactly the KV a
// forward_token decode step writes when kv8 is off, so a decode continuing from this cache
// reads a consistent one.
// ============================================================================
__global__ void pf_qknorm_rope_kv_bf16_kernel(
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_w, const __nv_bfloat16* __restrict__ k_w,
    __nv_bfloat16* __restrict__ k_pool, __nv_bfloat16* __restrict__ v_pool,
    const int* __restrict__ block_table,
    int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim, float theta, float eps,
    int block_size, int max_blocks_per_seq) {
    const int tok  = blockIdx.x;
    const int unit = blockIdx.y;
    const int t    = threadIdx.x;
    const int pos  = tok;                              // prefill: position == token index
    const int blk  = pos / block_size, within = pos % block_size;
    const int phys = block_table[blk];
    const size_t ctok = (size_t)phys * block_size + within;

    extern __shared__ float s_h[];
    __shared__ float s_warp[32];

    const bool is_q = unit < n_q_heads;
    const bool is_k = !is_q && unit < n_q_heads + n_kv_heads;
    if (unit >= n_q_heads + 2 * n_kv_heads) return;

    if (is_q || is_k) {
        const int hh = is_q ? unit : (unit - n_q_heads);
        const size_t base = ((size_t)tok * (is_q ? n_q_heads : n_kv_heads) + hh) * head_dim;
        const __nv_bfloat16* src = is_q ? q : k;
        const __nv_bfloat16* nrm = is_q ? q_w : k_w;
        const float xv = pf_to_f(src[base + t]);
        float ss = xv * xv;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
        if ((t & 31) == 0) s_warp[t >> 5] = ss;
        __syncthreads();
        if (t < 32) {
            float vv = (t < (head_dim + 31) / 32) ? s_warp[t] : 0.f;
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) vv += __shfl_xor_sync(0xffffffff, vv, m);
            if (t == 0) s_warp[0] = rsqrtf(vv / head_dim + eps);
        }
        __syncthreads();
        s_h[t] = pf_to_f(__float2bfloat16(xv * s_warp[0] * pf_to_f(nrm[t])));
        __syncthreads();
        // NeoX rotate-half over the first rotary_dim dims (partial rotary): pair (i, i+half),
        // out[i] = x[i]*cos - x[i+half]*sin, out[i+half] = x[i+half]*cos + x[i]*sin.
        float out = s_h[t];
        if (rotary_dim > 0 && t < rotary_dim) {
            const int half = rotary_dim >> 1;
            const int i    = (t < half) ? t : (t - half);
            const float freq = __powf(theta, -2.f * (float)i / (float)rotary_dim);
            const float ang = (float)pos * freq, c = __cosf(ang), sn = __sinf(ang);
            const float x0 = s_h[i], x1 = s_h[i + half];
            out = (t < half) ? (x0 * c - x1 * sn) : (x1 * c + x0 * sn);
        }
        if (is_q) {
            q[base + t] = __float2bfloat16(out);
        } else {
            const size_t dst = (ctok * n_kv_heads + hh) * head_dim;
            k_pool[dst + t] = __float2bfloat16(out);
        }
    } else {
        const int hh = unit - n_q_heads - n_kv_heads;
        const size_t base = ((size_t)tok * n_kv_heads + hh) * head_dim;
        const size_t dst  = (ctok * n_kv_heads + hh) * head_dim;
        v_pool[dst + t] = v[base + t];
    }
}

template <int HEAD_DIM>
__global__ void pf_attn_bf16_paged_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_pool,
    const __nv_bfloat16* __restrict__ v_pool, const int* __restrict__ block_table,
    __nv_bfloat16* __restrict__ attn, int n_tokens, int n_q_heads, int n_kv_heads,
    int block_size, int max_blocks_per_seq, float scale) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const int head = blockIdx.y;
    const int qtok = blockIdx.x;
    const int lane = threadIdx.x;
    if (head >= n_q_heads || qtok >= n_tokens) return;
    const int kv_head = head / (n_q_heads / n_kv_heads);

    const size_t q_off = ((size_t)qtok * n_q_heads + head) * HEAD_DIM;
    float q_reg[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) q_reg[e] = pf_to_f(q[q_off + lane + e * 32]);

    float m = -1e30f, l = 0.f, acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    for (int kpos = 0; kpos <= qtok; kpos++) {
        const int blk = kpos / block_size, within = kpos % block_size;
        const int phys = block_table[blk];
        const size_t ckt = ((size_t)phys * block_size + within);
        const size_t kv_off = (ckt * n_kv_heads + kv_head) * HEAD_DIM;
        float partial = 0.f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++)
            partial += q_reg[e] * pf_to_f(k_pool[kv_off + lane + e * 32]);
        const float score = pf_wsum(partial) * scale;

        const float m_new = fmaxf(m, score);
        const float corr  = __expf(m - m_new);
        const float p     = __expf(score - m_new);
        l = l * corr + p;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++)
            acc[e] = acc[e] * corr + p * pf_to_f(v_pool[kv_off + lane + e * 32]);
        m = m_new;
    }
    const float inv = (l > 0.f) ? (1.f / l) : 0.f;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) attn[q_off + lane + e * 32] = __float2bfloat16(acc[e] * inv);
}

// ============================================================================
// Host launchers
// ============================================================================
void launch_prefill_gemm(const void* A, const void* W, void* C,
                         int M, int N, int K, cudaStream_t stream, bool prefer_mma) {
    // Narrow n_out (the GDN gate projections) wastes most of a 128-wide tile and grids to only
    // 32 blocks; hand those to a tensor-core kernel with a full grid. =0 restores the tiled GEMM.
    if (launch_prefill_gemm_skinny(A, W, C, M, N, K, stream)) return;
    dim3 grid((N + PF_BN - 1) / PF_BN, (M + PF_BM - 1) / PF_BM);
    // Default = proven wmma kernel (byte-identical to main for MoE / short ctx). The mma.sync
    // path is opt-in via prefer_mma (dense long-context caller) and can still be forced off with
    // SPARKINFER_PREFILL_BF16_WMMA=1 for A/B. Never auto-select by M alone — MoE keeps bf16
    // projections at every context, and a silent M-threshold switch regressed its accuracy gate.
    static int force_wmma = []{ const char* e = getenv("SPARKINFER_PREFILL_BF16_WMMA"); return e && e[0] != '0' ? 1 : 0; }();
    if (force_wmma || !prefer_mma)
        pf_gemm_kernel<<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(A), reinterpret_cast<const __nv_bfloat16*>(W),
            reinterpret_cast<__nv_bfloat16*>(C), M, N, K);
    else
        pf_gemm_bf16_mma_kernel<<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(A), reinterpret_cast<const __nv_bfloat16*>(W),
            reinterpret_cast<__nv_bfloat16*>(C), M, N, K);
}

void launch_prefill_swiglu(const void* gate, const void* up, void* h, long n, cudaStream_t stream) {
    pf_swiglu_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate), reinterpret_cast<const __nv_bfloat16*>(up),
        reinterpret_cast<__nv_bfloat16*>(h), n);
}

// Strided-source SwiGLU: gate/up are column slices of one [rows, ld] tensor (the stacked gate|up
// GEMM output), the result is written contiguous [rows, cols]. Per element this is exactly
// pf_swiglu_kernel's value chain, so it is bit-identical to the unstrided form.
__global__ void pf_swiglu_ld_kernel(const __nv_bfloat16* __restrict__ gate,
                                    const __nv_bfloat16* __restrict__ up,
                                    __nv_bfloat16* __restrict__ h, int rows, int cols, int ld) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)rows * cols) return;
    const int r = (int)(i / cols);
    const long s = (long)r * ld + (int)(i - (long)r * cols);
    h[i] = __float2bfloat16(pf_silu(pf_to_f(gate[s])) * pf_to_f(up[s]));
}

void launch_prefill_swiglu_ld(const void* gate, const void* up, void* h,
                              int rows, int cols, int ld, cudaStream_t stream) {
    const long n = (long)rows * cols;
    pf_swiglu_ld_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate), reinterpret_cast<const __nv_bfloat16*>(up),
        reinterpret_cast<__nv_bfloat16*>(h), rows, cols, ld);
}

// One-launch split of the stacked [rows, 2*qdim + 2*kvdim] q|gate|k|v GEMM output into the four
// tight-stride destinations the rest of the layer expects -- replaces four cudaMemcpy2DAsync
// nodes per layer with a single kernel moving the same bytes. 16 B vectors: every column
// boundary (qdim, 2*qdim, 2*qdim + kvdim) is a multiple of 8 bf16, so a vector never straddles
// two destinations. Byte-identical to the copies by construction.
__global__ void pf_qkvg_split_kernel(const uint4* __restrict__ src, uint4* __restrict__ qb,
                                     uint4* __restrict__ qg, uint4* __restrict__ kf,
                                     uint4* __restrict__ vf, int rows, int qd8, int kd8) {
    const int rw = 2 * qd8 + 2 * kd8;                 // row width in 8-bf16 vectors
    const long total = (long)rows * rw;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        const int r = (int)(i / rw), c = (int)(i - (long)r * rw);
        const uint4 v = src[i];
        if (c < qd8)                 qb[(long)r * qd8 + c] = v;
        else if (c < 2 * qd8)        qg[(long)r * qd8 + (c - qd8)] = v;
        else if (c < 2 * qd8 + kd8)  kf[(long)r * kd8 + (c - 2 * qd8)] = v;
        else                         vf[(long)r * kd8 + (c - 2 * qd8 - kd8)] = v;
    }
}

bool launch_prefill_qkvg_split(const void* src, void* qb, void* qg, void* kf, void* vf,
                               int rows, int qdim, int kvdim, cudaStream_t stream) {
    if ((qdim & 7) || (kvdim & 7) || rows <= 0) return false;
    // The destinations are aliases into scratch; 16 B vectors need 16 B bases.
    if ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(qb) |
         reinterpret_cast<uintptr_t>(qg) | reinterpret_cast<uintptr_t>(kf) |
         reinterpret_cast<uintptr_t>(vf)) & 15) return false;
    const int qd8 = qdim >> 3, kd8 = kvdim >> 3;
    const long total = (long)rows * (2 * qd8 + 2 * kd8);
    int blocks = (int)((total + 255) / 256); if (blocks > 4096) blocks = 4096;
    pf_qkvg_split_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(src), reinterpret_cast<uint4*>(qb),
        reinterpret_cast<uint4*>(qg), reinterpret_cast<uint4*>(kf),
        reinterpret_cast<uint4*>(vf), rows, qd8, kd8);
    return cudaPeekAtLastError() == cudaSuccess;
}

void launch_prefill_add(const void* a, const void* b, void* out, long n, cudaStream_t stream) {
    pf_add_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a), reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<__nv_bfloat16*>(out), n);
}

void launch_prefill_split_q_gate(const void* qraw, void* q, void* gate,
                                 int n_tokens, int n_heads, int head_dim, cudaStream_t stream) {
    const long n = (long)n_tokens * n_heads * head_dim;
    pf_split_q_gate_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qraw), reinterpret_cast<__nv_bfloat16*>(q),
        reinterpret_cast<__nv_bfloat16*>(gate), n_tokens, n_heads, head_dim);
}

void launch_prefill_mul_sigmoid(void* attn, const void* gate, int n_tokens, int dim, cudaStream_t stream) {
    const long n = (long)n_tokens * dim;
    pf_mul_sigmoid_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(attn), reinterpret_cast<const __nv_bfloat16*>(gate), n);
}

void launch_prefill_gdn_conv(const void* qkv, const void* conv_w, void* conv_state,
                             void* q, void* k, void* v, int n_tokens, int q_heads, int v_heads,
                             int head_dim, int conv_kernel, float eps, cudaStream_t stream) {
    const int qkv_dim = 2 * q_heads * head_dim + v_heads * head_dim;
    const int blocks = 2 * q_heads + v_heads;
    static int seq = [] {   // SPARKINFER_PREFILL_GDN_CONV_SEQ=1 restores the token-loop kernel
        const char* e = getenv("SPARKINFER_PREFILL_GDN_CONV_SEQ");
        return (e && e[0] == '1') ? 1 : 0;
    }();
    static int tile = [] {  // SPARKINFER_PREFILL_GDN_CONV_TILE=0 restores the token-parallel kernel
        const char* e = getenv("SPARKINFER_PREFILL_GDN_CONV_TILE");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    if (!seq && tile && conv_kernel <= 8) {
        dim3 grid((n_tokens + PF_CONV_TT - 1) / PF_CONV_TT, blocks);
        pf_gdn_conv_tile_kernel<<<grid, head_dim, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(qkv), reinterpret_cast<const __nv_bfloat16*>(conv_w),
            reinterpret_cast<__nv_bfloat16*>(conv_state), reinterpret_cast<__nv_bfloat16*>(q),
            reinterpret_cast<__nv_bfloat16*>(k), reinterpret_cast<__nv_bfloat16*>(v),
            n_tokens, q_heads, v_heads, head_dim, qkv_dim, conv_kernel, eps);
        return;
    }
    if (!seq) {
        dim3 grid(n_tokens, blocks);
        pf_gdn_conv_par_kernel<<<grid, head_dim, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(qkv), reinterpret_cast<const __nv_bfloat16*>(conv_w),
            reinterpret_cast<__nv_bfloat16*>(conv_state), reinterpret_cast<__nv_bfloat16*>(q),
            reinterpret_cast<__nv_bfloat16*>(k), reinterpret_cast<__nv_bfloat16*>(v),
            n_tokens, q_heads, v_heads, head_dim, qkv_dim, conv_kernel, eps);
        return;
    }
    pf_gdn_conv_kernel<<<blocks, head_dim, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv), reinterpret_cast<const __nv_bfloat16*>(conv_w),
        reinterpret_cast<__nv_bfloat16*>(conv_state), reinterpret_cast<__nv_bfloat16*>(q),
        reinterpret_cast<__nv_bfloat16*>(k), reinterpret_cast<__nv_bfloat16*>(v),
        n_tokens, q_heads, v_heads, head_dim, qkv_dim, conv_kernel, eps);
}

void launch_prefill_gdn_scan(const void* q, const void* k, const void* v,
                             const void* alpha, const void* beta, const void* dt, const void* a,
                             float* state, void* out, int n_tokens, int q_heads, int v_heads,
                             int head_dim, bool qh_block, cudaStream_t stream) {
    // Chunk-parallel (WY/UT transform) scan: shortens the serial chain N -> N/C. Falls through to
    // the sequential scan below when disabled (SPARKINFER_PREFILL_GDN_CHUNK=0) or shape-unsupported.
    if (launch_prefill_gdn_chunk(q, k, v, alpha, beta, dt, a, state, out,
                                 n_tokens, q_heads, v_heads, head_dim, qh_block, stream)) return;
    constexpr int COLS = 4;
    dim3 grid(v_heads, (head_dim + COLS - 1) / COLS);
    auto qb = reinterpret_cast<const __nv_bfloat16*>(q);
    auto kb = reinterpret_cast<const __nv_bfloat16*>(k);
    auto vb = reinterpret_cast<const __nv_bfloat16*>(v);
    auto ab = reinterpret_cast<const __nv_bfloat16*>(alpha);
    auto bb = reinterpret_cast<const __nv_bfloat16*>(beta);
    auto db = reinterpret_cast<const __nv_bfloat16*>(dt);
    auto aa = reinterpret_cast<const __nv_bfloat16*>(a);
    auto ob = reinterpret_cast<__nv_bfloat16*>(out);
    if (head_dim == 128)
        pf_gdn_scan_kernel<COLS, 128><<<grid, COLS * 32, 0, stream>>>(
            qb, kb, vb, ab, bb, db, aa, state, ob, n_tokens, q_heads, v_heads, qh_block);
}

void launch_dflash_gdn_conv(const void* qkv, const void* conv_w, const void* live_state,
                            void* q, void* k, void* v, void* checkpoints,
                            int n_tokens, int q_heads, int v_heads, int head_dim,
                            int conv_kernel, float eps, cudaStream_t stream) {
    if (n_tokens <= 0 || head_dim <= 0 || conv_kernel < 2 || conv_kernel > 8) return;
    const int qkv_dim = 2 * q_heads * head_dim + v_heads * head_dim;
    df_gdn_conv_checkpoint_kernel<true><<<2 * q_heads + v_heads, head_dim, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv), reinterpret_cast<const __nv_bfloat16*>(conv_w),
        reinterpret_cast<const __nv_bfloat16*>(live_state), reinterpret_cast<__nv_bfloat16*>(q),
        reinterpret_cast<__nv_bfloat16*>(k), reinterpret_cast<__nv_bfloat16*>(v),
        reinterpret_cast<__nv_bfloat16*>(checkpoints), n_tokens, q_heads, v_heads,
        head_dim, qkv_dim, conv_kernel, eps);
}

void launch_dflash_gdn_scan(const void* q, const void* k, const void* v,
                            const void* alpha, const void* beta, const void* dt, const void* a,
                            const float* live_state, void* out, float* checkpoints,
                            int n_tokens, int q_heads, int v_heads, int head_dim,
                            bool qh_block, cudaStream_t stream) {
    if (n_tokens <= 0 || head_dim != 128) return;
    constexpr int COLS = 4;
    dim3 grid(v_heads, (head_dim + COLS - 1) / COLS);
    df_gdn_scan_checkpoint_kernel<COLS, 128, true><<<grid, COLS * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v), reinterpret_cast<const __nv_bfloat16*>(alpha),
        reinterpret_cast<const __nv_bfloat16*>(beta), reinterpret_cast<const __nv_bfloat16*>(dt),
        reinterpret_cast<const __nv_bfloat16*>(a), live_state,
        reinterpret_cast<__nv_bfloat16*>(out), checkpoints, n_tokens, q_heads, v_heads, qh_block);
}

void launch_dflash_gdn_conv_compact(const void* qkv, const void* conv_w,
                                    const void* live_state, void* q, void* k, void* v,
                                    int n_tokens, int q_heads, int v_heads, int head_dim,
                                    int conv_kernel, float eps, cudaStream_t stream) {
    if (n_tokens <= 0 || head_dim <= 0 || conv_kernel < 2 || conv_kernel > 8) return;
    const int qkv_dim = 2 * q_heads * head_dim + v_heads * head_dim;
    df_gdn_conv_checkpoint_kernel<false><<<2 * q_heads + v_heads, head_dim, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv), reinterpret_cast<const __nv_bfloat16*>(conv_w),
        reinterpret_cast<const __nv_bfloat16*>(live_state), reinterpret_cast<__nv_bfloat16*>(q),
        reinterpret_cast<__nv_bfloat16*>(k), reinterpret_cast<__nv_bfloat16*>(v), nullptr,
        n_tokens, q_heads, v_heads, head_dim, qkv_dim, conv_kernel, eps);
}

void launch_dflash_gdn_scan_compact(const void* q, const void* k, const void* v,
                                    const void* alpha, const void* beta,
                                    const void* dt, const void* a, const float* live_state,
                                    void* out, int n_tokens, int q_heads, int v_heads,
                                    int head_dim, bool qh_block, cudaStream_t stream) {
    if (n_tokens <= 0 || head_dim != 128) return;
    constexpr int COLS = 4;
    dim3 grid(v_heads, (head_dim + COLS - 1) / COLS);
    df_gdn_scan_checkpoint_kernel<COLS, 128, false><<<grid, COLS * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v), reinterpret_cast<const __nv_bfloat16*>(alpha),
        reinterpret_cast<const __nv_bfloat16*>(beta), reinterpret_cast<const __nv_bfloat16*>(dt),
        reinterpret_cast<const __nv_bfloat16*>(a), live_state,
        reinterpret_cast<__nv_bfloat16*>(out), nullptr, n_tokens, q_heads, v_heads, qh_block);
}

void launch_dflash_gdn_conv_commit(const void* qkv, void* live_state, int n_tokens,
                                   int q_heads, int v_heads, int head_dim, int conv_kernel,
                                   cudaStream_t stream) {
    if (n_tokens <= 0 || head_dim <= 0 || conv_kernel < 2 || conv_kernel > 8) return;
    const int qkv_dim = 2 * q_heads * head_dim + v_heads * head_dim;
    constexpr int BLOCK = 256;
    df_gdn_conv_commit_kernel<<<(qkv_dim + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<__nv_bfloat16*>(live_state), n_tokens, qkv_dim, conv_kernel);
}

void launch_dflash_gdn_scan_commit(const void* k, const void* v,
                                   const void* alpha, const void* beta,
                                   const void* dt, const void* a, float* live_state,
                                   int n_tokens, int q_heads, int v_heads, int head_dim,
                                   bool qh_block, cudaStream_t stream) {
    if (n_tokens <= 0 || head_dim != 128) return;
    constexpr int COLS = 4;
    dim3 grid(v_heads, (head_dim + COLS - 1) / COLS);
    df_gdn_scan_commit_kernel<COLS, 128><<<grid, COLS * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(k), reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<const __nv_bfloat16*>(alpha), reinterpret_cast<const __nv_bfloat16*>(beta),
        reinterpret_cast<const __nv_bfloat16*>(dt), reinterpret_cast<const __nv_bfloat16*>(a),
        live_state, n_tokens, q_heads, v_heads, qh_block);
}

void launch_prefill_gated_norm(const void* x, const void* z, const void* weight, void* out,
                               int n_tokens, int v_heads, int head_dim, float eps, cudaStream_t stream) {
    dim3 grid(n_tokens, v_heads);   // token on grid.x (grid.y capped at 65535)
    auto xb = reinterpret_cast<const __nv_bfloat16*>(x);
    auto zb = reinterpret_cast<const __nv_bfloat16*>(z);
    auto wb = reinterpret_cast<const __nv_bfloat16*>(weight);
    auto ob = reinterpret_cast<__nv_bfloat16*>(out);
    if (head_dim == 128)
        pf_gated_norm_kernel<128><<<grid, 32, 0, stream>>>(xb, zb, wb, ob, n_tokens, v_heads, eps);
}

void launch_prefill_qknorm_rope_kv_int8(
    void* q, void* k, const void* v, const void* q_w, const void* k_w,
    signed char* k_pool, signed char* v_pool, void* k_scale, void* v_scale,
    const int* block_table, int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    int rotary_dim, float theta, float eps, int block_size, int max_blocks_per_seq,
    cudaStream_t stream) {
    dim3 grid(n_tokens, n_q_heads + 2 * n_kv_heads);   // token on grid.x
    const size_t shmem = (size_t)head_dim * sizeof(float);
    pf_qknorm_rope_kv_int8_kernel<<<grid, head_dim, shmem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q), reinterpret_cast<__nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v), reinterpret_cast<const __nv_bfloat16*>(q_w),
        reinterpret_cast<const __nv_bfloat16*>(k_w), k_pool, v_pool,
        reinterpret_cast<__half*>(k_scale), reinterpret_cast<__half*>(v_scale),
        block_table, n_q_heads, n_kv_heads, head_dim, rotary_dim, theta, eps,
        block_size, max_blocks_per_seq);
}

// Muse Glimmer bf16-KV counterpart: QK-norm + NORMAL RoPE (rotary_dim>0) / NoPE
// (rotary_dim==0) + bf16 KV write. k_pool/v_pool are bf16 (muse uses a bf16 KV cache).
void launch_prefill_qknorm_ropenorm_kv_bf16(
    const void* q, void* q_out, const void* k, const void* v,
    const void* q_w, const void* k_w,
    void* k_pool, void* v_pool,
    const int* block_table, int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    int rotary_dim, float theta, float eps, int block_size, int max_blocks_per_seq,
    cudaStream_t stream, int q_ld, int kv_ld) {
    dim3 grid(n_tokens, n_q_heads + 2 * n_kv_heads);   // token on grid.x
    const size_t shmem = (size_t)head_dim * sizeof(float);
    if (q_ld <= 0) q_ld = n_q_heads * head_dim;        // tight (split) inputs
    if (kv_ld <= 0) kv_ld = n_kv_heads * head_dim;
    pf_qknorm_ropenorm_kv_bf16_kernel<<<grid, head_dim, shmem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<__nv_bfloat16*>(q_out),
        reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v), reinterpret_cast<const __nv_bfloat16*>(q_w),
        reinterpret_cast<const __nv_bfloat16*>(k_w),
        reinterpret_cast<__nv_bfloat16*>(k_pool), reinterpret_cast<__nv_bfloat16*>(v_pool),
        block_table, n_q_heads, n_kv_heads, head_dim, rotary_dim, theta, eps,
        block_size, max_blocks_per_seq, q_ld, kv_ld);
}

void launch_prefill_attn_int8_paged(
    const void* q, const signed char* k_pool, const signed char* v_pool,
    const void* k_scale, const void* v_scale, const int* block_table, void* attn,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, float scale, cudaStream_t stream) {
    // int8 tensor-core prefill attention: same mask + online softmax as the scalar path below,
    // run on the wmma int8 cores (the scalar kernels are compute-bound at ~8 TFLOP/s). Honours the
    // same SPARKINFER_PREFILL_ATTN_WINDOW selection. SPARKINFER_PREFILL_ATTN_MMA=0 falls through.
    if (launch_prefill_attn_mma(q, k_pool, v_pool, k_scale, v_scale, block_table, attn,
            n_tokens, n_q_heads, n_kv_heads, head_dim, block_size, max_blocks_per_seq, scale, stream))
        return;
    // Sink + sliding-window sparse prefill attention (StreamingLLM, matches the merged decode
    // sparse-KV #379): O(N*window) instead of O(N^2) at long context. Default on; returns false
    // (SPARKINFER_PREFILL_ATTN_WINDOW=0, or head_dim != 256) to fall through to full attention.
    if (launch_prefill_attn_windowed(q, k_pool, v_pool, k_scale, v_scale, block_table, attn,
            n_tokens, n_q_heads, n_kv_heads, head_dim, block_size, max_blocks_per_seq, scale, stream))
        return;
    dim3 grid(n_tokens, n_q_heads);   // token on grid.x
    auto qb = reinterpret_cast<const __nv_bfloat16*>(q);
    auto ks = reinterpret_cast<const __half*>(k_scale);
    auto vs = reinterpret_cast<const __half*>(v_scale);
    auto ob = reinterpret_cast<__nv_bfloat16*>(attn);
    if (head_dim == 256)
        pf_attn_int8_paged_kernel<256><<<grid, 32, 0, stream>>>(
            qb, k_pool, v_pool, ks, vs, block_table, ob, n_tokens, n_q_heads, n_kv_heads,
            block_size, max_blocks_per_seq, scale);
}

void launch_prefill_qknorm_rope_kv_bf16(
    void* q, void* k, const void* v, const void* q_w, const void* k_w,
    void* k_pool, void* v_pool,
    const int* block_table, int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    int rotary_dim, float theta, float eps, int block_size, int max_blocks_per_seq,
    cudaStream_t stream) {
    dim3 grid(n_tokens, n_q_heads + 2 * n_kv_heads);
    const size_t shmem = (size_t)head_dim * sizeof(float);
    pf_qknorm_rope_kv_bf16_kernel<<<grid, head_dim, shmem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q), reinterpret_cast<__nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v), reinterpret_cast<const __nv_bfloat16*>(q_w),
        reinterpret_cast<const __nv_bfloat16*>(k_w),
        reinterpret_cast<__nv_bfloat16*>(k_pool), reinterpret_cast<__nv_bfloat16*>(v_pool),
        block_table, n_q_heads, n_kv_heads, head_dim, rotary_dim, theta, eps,
        block_size, max_blocks_per_seq);
}

bool launch_prefill_attn_bf16_paged(
    const void* q, const void* k_pool, const void* v_pool, const int* block_table, void* attn,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
    int block_size, int max_blocks_per_seq, float scale, cudaStream_t stream) {
    dim3 grid(n_tokens, n_q_heads);
    auto qb = reinterpret_cast<const __nv_bfloat16*>(q);
    auto kb = reinterpret_cast<const __nv_bfloat16*>(k_pool);
    auto vb = reinterpret_cast<const __nv_bfloat16*>(v_pool);
    auto ob = reinterpret_cast<__nv_bfloat16*>(attn);
    if (head_dim == 256) {
        pf_attn_bf16_paged_kernel<256><<<grid, 32, 0, stream>>>(
            qb, kb, vb, block_table, ob, n_tokens, n_q_heads, n_kv_heads,
            block_size, max_blocks_per_seq, scale);
        return true;
    }
    if (head_dim == 128) {
        pf_attn_bf16_paged_kernel<128><<<grid, 32, 0, stream>>>(
            qb, kb, vb, block_table, ob, n_tokens, n_q_heads, n_kv_heads,
            block_size, max_blocks_per_seq, scale);
        return true;
    }
    return false;
}

} // namespace kernels
} // namespace sparkinfer
