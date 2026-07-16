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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include <cstdlib>

#include "sparkinfer/kernels/prefill.h"
#include "sparkinfer/kernels/prefill_attn_mma.h"
#include "sparkinfer/kernels/prefill_attn_window.h"

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
                                   int n_tokens, int q_heads, int v_heads) {
    constexpr int NROW = HEAD_DIM / 32;
    const int vh   = blockIdx.x;
    const int j    = blockIdx.y * COLS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (vh >= v_heads || j >= HEAD_DIM) return;
    const int qh = vh % q_heads;
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
        if (t < rhalf) {
            const float freq = __powf(theta, -2.f * (float)t / (float)rotary_dim);
            const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
            const float x0 = s_h[t], x1 = s_h[t + rhalf];
            q[base + t]         = __float2bfloat16(x0 * c - x1 * s);
            q[base + t + rhalf] = __float2bfloat16(x1 * c + x0 * s);
        }
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
// Host launchers
// ============================================================================
void launch_prefill_gemm(const void* A, const void* W, void* C,
                         int M, int N, int K, cudaStream_t stream) {
    dim3 grid((N + PF_BN - 1) / PF_BN, (M + PF_BM - 1) / PF_BM);
    pf_gemm_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(A), reinterpret_cast<const __nv_bfloat16*>(W),
        reinterpret_cast<__nv_bfloat16*>(C), M, N, K);
}

void launch_prefill_swiglu(const void* gate, const void* up, void* h, long n, cudaStream_t stream) {
    pf_swiglu_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate), reinterpret_cast<const __nv_bfloat16*>(up),
        reinterpret_cast<__nv_bfloat16*>(h), n);
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
                             int head_dim, cudaStream_t stream) {
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
            qb, kb, vb, ab, bb, db, aa, state, ob, n_tokens, q_heads, v_heads);
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

} // namespace kernels
} // namespace sparkinfer
