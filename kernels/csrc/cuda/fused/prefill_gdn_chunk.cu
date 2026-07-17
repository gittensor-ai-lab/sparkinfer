// ============================================================================
// Chunk-parallel (WY / UT transform) Gated-DeltaNet prefill scan for Qwythos (Qwen3.5).
//
// WHY THIS EXISTS
// ---------------
// The batched prompt prefill (#398) runs the gated delta rule as ONE sequential scan over all N
// prompt tokens (pf_gdn_scan_kernel, batched_prefill.cu): one warp per state column, one rank-1
// state update per token, two 5-shuffle warp reductions per token on the dependency chain. It is
// the last sequential stage left in the batched prefill.
//
// Measured on an RTX 5090 (nsys --cuda-graph-trace=node, main @ cb34dc1, ctx=4096, and confirmed
// per-prefill by a reps=1-vs-2 instance diff: 24 -> 48 instances, 59.31 -> 118.64 ms): 59.3 ms per
// prefill = 20.7% of the 286.1 ms prefill, sustaining ~5.3 TFLOP/s — about 5% of the fp32 peak.
// After #464 (fused Q4K->int8 dequant + token-parallel GDN conv) took the prefill from 8526 to
// 14314 pp, this is the largest remaining slice that is not already at hardware peak: pf_gemm_i8 is
// 52% of the wall but runs at 379 TOPS = 90% of the int8 tensor peak, so it has no headroom.
//
// The scan is NOT DRAM-bound: q/k/v/out for one layer need only ~10.5 ms at 1792 GB/s. The cost is
// L1/L2 bandwidth and serial latency — the grid is (v_heads=32, head_dim/4=32) x 4 warps, so all
// 128 column-warps of a v-head re-load the same k_t and q_t vector on every one of the N steps
// (~200x read amplification, ~101 GB/layer of L2 traffic, which alone accounts for the time).
//
// THE RECURRENCE AND ITS CHUNK FORM
// ---------------------------------
// Per v-head, with S in [d_k=128, d_v=128] (the same S[row][col] the sequential kernel keeps):
//     S_t = g_t (I - b_t k_t k_t^T) S_{t-1} + b_t k_t v_t^T,   y_t = S_t^T q_t * scale
//     g_t = exp(softplus(alpha_t + dt) * a),  b_t = sigmoid(beta_t)
//
// Substituting S~_t = S_t / Gamma_t with Gamma_t = prod_{s<=t} g_s cancels the gate and leaves a
// pure delta rule, whose rank-1 chain collapses into a triangular solve (the WY / UT transform):
//     u~_t = b_t (v~_t - S~_{t-1}^T k_t)   =>   S~_t = S~_{t-1} + k_t u~_t^T
// Writing u^_t = Gamma_t u~_t keeps every coefficient a RATIO exp(G_t - G_s) with s <= t (G =
// log Gamma), which is bounded by 1 — the naive form would need v_t / Gamma_t, which overflows.
// Per chunk of C tokens, with G_i the cumulative log-gate inside the chunk:
//     (I + A) U^ = B (V - diag(exp G) K S_in),   A[i][j] = b_i (k_i.k_j) exp(G_i - G_j), j < i
//     Y         = [ diag(exp G) (Q S_in) + M U^ ] * scale,  M[i][j] = (q_i.k_j) exp(G_i-G_j), j<=i
//     S_out     = exp(G_last) S_in + K^T U~,     U~[j] = exp(G_last - G_j) U^[j]
// (I + A) is unit lower triangular, so T = (I+A)^-1 is too and costs C^3/6 MACs by forward
// substitution. The serial chain shortens from N to N/C and the inner work becomes dense matmuls
// over shared-memory tiles that every state column reuses.
//
// NUMERICS. ssm_a is negative for all 24 linear layers x 32 v-heads of this checkpoint (read from
// the GGUF: min -76.996, max -0.0089), so log g = softplus(alpha+dt)*a <= 0 and G decreases
// monotonically => every exp(G_i - G_j) with j <= i is in (0, 1]. Per-token log-gates reach -8 and
// below, so Gamma UNDERFLOWS to zero over a chunk; G is therefore kept in LOG space and only ever
// exponentiated after a subtraction (never as a ratio exp(G_last)/exp(G_i), which would be 0/0).
// A zero decay is the correct answer there — the state is simply fully forgotten.
//
// STRUCTURE. The state columns are independent: T, A and M depend only on k, q and the gates, and
// U^[:,j], Y[:,j], S[:,j] each depend only on S_in[:,j]. So the work splits into
//   1. pf_gdnc_prep_kernel  — grid (N/C, v_heads), FULLY parallel: builds T, then
//      W^ = T B diag(exp G) K and U0 = T B V and M. No dependence on S_in.
//   2. pf_gdnc_scan_kernel  — grid (v_heads, head_dim/JC), sequential over chunks only: applies
//      U^ = U0 - W^ S_in, Y, and the state update. Two matmuls and a triangular apply per chunk.
// The state stays fp32 and is written back in the SAME transposed [v_head][col][row] layout the
// decode gdn_ar_fast kernel expects, so this is a drop-in replacement and decode is untouched.
// ============================================================================
#include "sparkinfer/kernels/prefill_gdn_chunk.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>

namespace sparkinfer {
namespace kernels {

namespace {

__device__ __forceinline__ float gc_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float gc_sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float gc_softplus(float x) { return x > 20.f ? x : __logf(1.f + __expf(x)); }

// Shared-memory row padding (in elements) to break the power-of-two bank stride.
constexpr int PAD = 8;

// ---------------------------------------------------------------------------
// Kernel 1: per-chunk prep. grid = (n_chunks, v_heads), fully parallel.
//
// Emits, for chunk c of v-head h:
//   g_buf[t][h]        = G_i, the cumulative log-gate inside the chunk (log space)
//   w_buf[t][h][:]     = W^ = T B diag(exp G) K      [C, HD]
//   u_buf[t][h][:]     = U0 = T B V                  [C, HD]
//   m_buf[c][h][i][j]  = M  = (q_i.k_j) exp(G_i-G_j) for j <= i, else 0   [C, C]
// Rows past the end of a short final chunk get b = 0 and log-gate 0, which makes W^, U0 and M
// vanish there, so the scan kernel needs no tail special-casing beyond bounds-checking its writes.
// ---------------------------------------------------------------------------
template <int C, int HD>
__global__ void pf_gdnc_prep_kernel(const __nv_bfloat16* __restrict__ q,
                                    const __nv_bfloat16* __restrict__ k,
                                    const __nv_bfloat16* __restrict__ v,
                                    const __nv_bfloat16* __restrict__ alpha,
                                    const __nv_bfloat16* __restrict__ beta,
                                    const __nv_bfloat16* __restrict__ dt,
                                    const __nv_bfloat16* __restrict__ a,
                                    float* __restrict__ g_buf,
                                    __nv_bfloat16* __restrict__ w_buf,
                                    __nv_bfloat16* __restrict__ u_buf,
                                    float* __restrict__ m_buf,
                                    int n_tokens, int q_heads, int v_heads) {
    extern __shared__ char s_raw[];
    __nv_bfloat16* s_k = reinterpret_cast<__nv_bfloat16*>(s_raw);              // [C][HD+PAD]
    __nv_bfloat16* s_x = s_k + (size_t)C * (HD + PAD);                         // [C][HD+PAD] q then v
    float* s_A = reinterpret_cast<float*>(s_x + (size_t)C * (HD + PAD));       // [C][C+PAD]
    float* s_g = s_A + (size_t)C * (C + PAD);                                  // [C]
    float* s_b = s_g + C;                                                      // [C]
    float* s_t = s_b + C;                                                      // [C] scratch row

    const int c    = blockIdx.x;
    const int h    = blockIdx.y;
    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;
    const int t0   = c * C;
    const int len  = min(C, n_tokens - t0);
    if (len <= 0) return;

    const int qh    = h % q_heads;
    const int q_dim = q_heads * HD;
    const int v_dim = v_heads * HD;
    const float a_h  = gc_to_f(a[h]);
    const float dt_h = gc_to_f(dt[h]);

    // ---- gates: per-token log-gate and b, then an inclusive prefix sum over the chunk ----
    for (int i = tid; i < C; i += nthr) {
        if (i < len) {
            const float al = gc_to_f(alpha[(size_t)(t0 + i) * v_heads + h]);
            s_t[i] = gc_softplus(al + dt_h) * a_h;                       // log g_i  (<= 0)
            s_b[i] = gc_sigmoid(gc_to_f(beta[(size_t)(t0 + i) * v_heads + h]));
        } else {
            s_t[i] = 0.f;                                                // tail: no decay, no update
            s_b[i] = 0.f;
        }
    }
    __syncthreads();
    if (tid == 0) {                                  // C=64 serial adds, once per block
        float acc = 0.f;
        for (int i = 0; i < C; i++) { acc += s_t[i]; s_g[i] = acc; }
    }
    __syncthreads();
    for (int i = tid; i < len; i += nthr) g_buf[(size_t)(t0 + i) * v_heads + h] = s_g[i];

    // ---- stage K and Q ----
    for (int e = tid; e < C * HD; e += nthr) {
        const int i = e / HD, d = e - i * HD;
        const bool live = i < len;
        s_k[i * (HD + PAD) + d] = live ? k[(size_t)(t0 + i) * q_dim + qh * HD + d] : __float2bfloat16(0.f);
        s_x[i * (HD + PAD) + d] = live ? q[(size_t)(t0 + i) * q_dim + qh * HD + d] : __float2bfloat16(0.f);
    }
    __syncthreads();

    // ---- A[i][j] = b_i (k_i.k_j) exp(G_i-G_j) for j<i (unit diagonal), and M = tril(Q K^T . decay) ----
    // K K^T and Q K^T are 80% of this kernel's MACs and both contract over HD against the same
    // K tile, so they run on tensor cores: 8 warps, 8 output tiles of 16x16 (4 for K K^T, 4 for
    // Q K^T). K and Q are already bf16 (they are the raw conv outputs), so nothing is narrowed here
    // — only the decay/b scaling and the triangular mask stay scalar, applied to the fp32 results.
    {
        using namespace nvcuda;
        const int warp = tid >> 5;
        const bool isKK = warp < 4;
        const int t = warp & 3;
        const int ti = (t >> 1) * 16, tj = (t & 1) * 16;
        const __nv_bfloat16* Aop = isKK ? s_k : s_x;      // s_x holds Q at this point
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
        wmma::fill_fragment(cf, 0.f);
        #pragma unroll
        for (int d = 0; d < HD; d += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
            wmma::load_matrix_sync(af, Aop + (size_t)ti * (HD + PAD) + d, HD + PAD);
            wmma::load_matrix_sync(bf, s_k + (size_t)tj * (HD + PAD) + d, HD + PAD);  // col_major => K^T
            wmma::mma_sync(cf, af, bf, cf);
        }
        __shared__ float sD[8][16][16];
        wmma::store_matrix_sync(&sD[warp][0][0], cf, 16, wmma::mem_row_major);
        __syncthreads();

        const size_t mbase = ((size_t)c * v_heads + h) * C * C;
        for (int e = tid; e < C * C; e += nthr) {
            const int i = e / C, j = e - i * C;
            const int w = ((i >> 4) << 1) | (j >> 4);
            if (j > i) { s_A[i * (C + PAD) + j] = 0.f; m_buf[mbase + e] = 0.f; continue; }
            const float decay = __expf(s_g[i] - s_g[j]);                 // <= 1 (G is decreasing)
            const float kk = sD[w][i & 15][j & 15];
            const float qk = sD[4 + w][i & 15][j & 15];
            s_A[i * (C + PAD) + j] = (j < i) ? s_b[i] * kk * decay : 1.f;
            m_buf[mbase + e] = qk * decay;
        }
    }
    __syncthreads();

    // ---- T = (I + A)^-1 in place, by forward substitution over rows ----
    //   T[i][j] = -A[i][j] - sum_{m=j+1}^{i-1} A[i][m] T[m][j]      (T[j][j] = 1)
    // Row i still holds A while it is being read; rows m < i already hold T.
    for (int i = 1; i < C; i++) {
        for (int j = tid; j < i; j += nthr) {
            float acc = s_A[i * (C + PAD) + j];
            for (int m = j + 1; m < i; m++) acc += s_A[i * (C + PAD) + m] * s_A[m * (C + PAD) + j];
            s_t[j] = -acc;
        }
        __syncthreads();
        for (int j = tid; j < i; j += nthr) s_A[i * (C + PAD) + j] = s_t[j];
        __syncthreads();
    }

    // ---- W^ = T . (b_m exp(G_m) k_m) ----
    for (int e = tid; e < C * HD; e += nthr) {
        const int i = e / HD, d = e - i * HD;
        float acc = 0.f;
        for (int m = 0; m <= i; m++)
            acc += s_A[i * (C + PAD) + m] * (s_b[m] * __expf(s_g[m]) * gc_to_f(s_k[m * (HD + PAD) + d]));
        w_buf[((size_t)(t0 + i) * v_heads + h) * HD + d] = __float2bfloat16(acc);
    }
    __syncthreads();

    // ---- reuse the Q tile for V, then U0 = T . (b_m v_m) ----
    for (int e = tid; e < C * HD; e += nthr) {
        const int i = e / HD, d = e - i * HD;
        s_x[i * (HD + PAD) + d] = (i < len) ? v[(size_t)(t0 + i) * v_dim + h * HD + d] : __float2bfloat16(0.f);
    }
    __syncthreads();
    for (int e = tid; e < C * HD; e += nthr) {
        const int i = e / HD, d = e - i * HD;
        float acc = 0.f;
        for (int m = 0; m <= i; m++)
            acc += s_A[i * (C + PAD) + m] * (s_b[m] * gc_to_f(s_x[m * (HD + PAD) + d]));
        u_buf[((size_t)(t0 + i) * v_heads + h) * HD + d] = __float2bfloat16(acc);
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: the (now C-times shorter) sequential scan over chunks.
// grid = (v_heads, HD/JC); each block owns JC state columns of one v-head and walks the chunks.
// Per chunk: U^ = U0 - W^ S ; Y = [diag(exp G)(Q S) + M U^] * scale ; S = exp(G_last) S + K^T U~.
// ---------------------------------------------------------------------------
template <int C, int HD, int JC>
__global__ void pf_gdnc_scan_kernel(const __nv_bfloat16* __restrict__ q,
                                    const __nv_bfloat16* __restrict__ k,
                                    const float* __restrict__ g_buf,
                                    const __nv_bfloat16* __restrict__ w_buf,
                                    const __nv_bfloat16* __restrict__ u_buf,
                                    const float* __restrict__ m_buf,
                                    float* __restrict__ state,
                                    __nv_bfloat16* __restrict__ out,
                                    int n_tokens, int q_heads, int v_heads, int n_chunks) {
    extern __shared__ char s_raw[];
    float* s_S = reinterpret_cast<float*>(s_raw);                              // [HD][JC]  fp32 carrier
    float* s_U = s_S + (size_t)HD * JC;                                        // [C][JC]
    float* s_M = s_U + (size_t)C * JC;                                         // [C][C+PAD]
    float* s_g = s_M + (size_t)C * (C + PAD);                                  // [C]
    float* s_Y = s_g + C;                                                      // [C][JC]  Q S staging
    __nv_bfloat16* s_W =
        reinterpret_cast<__nv_bfloat16*>(s_Y + (size_t)C * JC);                // [C][HD+PAD]
    __nv_bfloat16* s_K = s_W + (size_t)C * (HD + PAD);                         // [C][HD+PAD]
    __nv_bfloat16* s_Q = s_K + (size_t)C * (HD + PAD);                         // [C][HD+PAD]
    // bf16 operand mirrors. S stays fp32 across chunks (it is the recurrent carrier); it is
    // narrowed to bf16 ONCE per chunk to feed the tensor cores, which is the only precision the
    // wmma path costs over the fp32 register-tiled one — W^, K and Q are already bf16.
    __nv_bfloat16* s_Sb = s_Q + (size_t)C * (HD + PAD);                        // [HD][JC+PAD]
    __nv_bfloat16* s_Ub = s_Sb + (size_t)HD * (JC + PAD);                      // [C][JC+PAD]

    const int h    = blockIdx.x;
    const int j0   = blockIdx.y * JC;
    const int tid  = threadIdx.x;
    const int nthr = blockDim.x;

    const int qh    = h % q_heads;
    const int q_dim = q_heads * HD;
    const int v_dim = v_heads * HD;
    const float scale = rsqrtf((float)HD);

    for (int e = tid; e < HD * JC; e += nthr) s_S[e] = 0.f;    // fresh prefill: state starts at zero
    __syncthreads();

    for (int c = 0; c < n_chunks; c++) {
        const int t0  = c * C;
        const int len = min(C, n_tokens - t0);

        // ---- stage this chunk ----
        for (int i = tid; i < C; i += nthr)
            s_g[i] = (i < len) ? g_buf[(size_t)(t0 + i) * v_heads + h] : 0.f;
        for (int e = tid; e < C * JC; e += nthr) {
            const int i = e / JC, jj = e - i * JC;
            s_U[e] = (i < len) ? gc_to_f(u_buf[((size_t)(t0 + i) * v_heads + h) * HD + j0 + jj]) : 0.f;
        }
        for (int e = tid; e < C * C; e += nthr) {
            const int i = e / C, j = e - i * C;
            s_M[i * (C + PAD) + j] = m_buf[(size_t)e + ((size_t)c * v_heads + h) * C * C];
        }
        for (int e = tid; e < C * HD; e += nthr) {
            const int i = e / HD, d = e - i * HD;
            const bool live = i < len;
            s_W[i * (HD + PAD) + d] = live ? w_buf[((size_t)(t0 + i) * v_heads + h) * HD + d] : __float2bfloat16(0.f);
            s_K[i * (HD + PAD) + d] = live ? k[(size_t)(t0 + i) * q_dim + qh * HD + d] : __float2bfloat16(0.f);
            s_Q[i * (HD + PAD) + d] = live ? q[(size_t)(t0 + i) * q_dim + qh * HD + d] : __float2bfloat16(0.f);
        }
        __syncthreads();

        // ---- U^ = U0 - W^ S   [C,HD] x [HD,JC] ----
        // REGISTER-TILED 2x2. A scalar `acc += A[m]*B[m]` matmul reads TWO shared-memory operands per
        // FMA, which caps it at ~16 FMA/cycle/SM against a 128 FMA/cycle peak — that (not the serial
        // chain) is why the naive chunk kernel only matched the sequential scan. A 2x2 tile reuses
        // each loaded value twice: 4 loads per 4 FMAs, and 4 independent accumulators to hide latency.
        // [C=32,JC=32] outputs / 256 threads = exactly 4 each, so the tiling divides evenly.
        // ---- narrow S to bf16 once, then run U^ = U0 - W^ S and Y0 = Q S on tensor cores ----
        for (int e = tid; e < HD * JC; e += nthr) {
            const int m = e / JC, jj = e - m * JC;
            s_Sb[m * (JC + PAD) + jj] = __float2bfloat16(s_S[e]);
        }
        __syncthreads();
        {
            using namespace nvcuda;
            // 8 warps, 8 output tiles of 16x16: warps 0-3 own U^ = W^ S, warps 4-7 own Y0 = Q S.
            // Both contract over HD against the same s_Sb, so the two matmuls share its fragments.
            const int warp = tid >> 5;
            const bool isU = warp < 4;
            const int t    = warp & 3;                       // 2x2 tile grid over [C=32, JC=32]
            const int ti = (t >> 1) * 16, tj = (t & 1) * 16;
            const __nv_bfloat16* Aop = isU ? s_W : s_Q;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
            wmma::fill_fragment(cf, 0.f);
            #pragma unroll
            for (int kk = 0; kk < HD; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(af, Aop + (size_t)ti * (HD + PAD) + kk, HD + PAD);
                wmma::load_matrix_sync(bf, s_Sb + (size_t)kk * (JC + PAD) + tj, JC + PAD);
                wmma::mma_sync(cf, af, bf, cf);
            }
            __shared__ float sC[8][16][16];
            wmma::store_matrix_sync(&sC[warp][0][0], cf, 16, wmma::mem_row_major);
            __syncthreads();
            for (int e = tid; e < C * JC; e += nthr) {
                const int i = e / JC, jj = e - i * JC;
                const int w = ((i >> 4) << 1) | (jj >> 4);
                s_U[e] -= sC[w][i & 15][jj & 15];             // U^ = U0 - W^ S
                s_Y[e]  = sC[4 + w][i & 15][jj & 15];         // Y0 = Q S
            }
        }
        __syncthreads();

        // ---- Y = [diag(exp G) Y0 + M U^] * scale ----
        for (int e = tid; e < C * JC; e += nthr) {
            const int i = e / JC, jj = e - i * JC;
            if (t0 + i >= n_tokens) continue;
            float mu = 0.f;
            for (int p = 0; p <= i; p++) mu += s_M[i * (C + PAD) + p] * s_U[p * JC + jj];
            const float y = (__expf(s_g[i]) * s_Y[e] + mu) * scale;
            out[(size_t)(t0 + i) * v_dim + h * HD + j0 + jj] = __float2bfloat16(y);
        }
        __syncthreads();

        // ---- U~[p] = exp(G_last - G_p) U^[p]  (tail rows already carry U^ = 0) ----
        const float g_last = s_g[C - 1];               // == s_g[len-1]: tail log-gates are 0
        for (int e = tid; e < C * JC; e += nthr) {
            const int i = e / JC;
            s_U[e] *= __expf(g_last - s_g[i]);
        }
        __syncthreads();

        // ---- S = exp(G_last) S + K^T U~   [HD,C] x [C,JC], on tensor cores ----
        for (int e = tid; e < C * JC; e += nthr) {
            const int i = e / JC, jj = e - i * JC;
            s_Ub[i * (JC + PAD) + jj] = __float2bfloat16(s_U[e]);
        }
        __syncthreads();
        {
            using namespace nvcuda;
            const float gl = __expf(g_last);
            const int warp = tid >> 5;                       // 8 warps x 2 tiles = 16 tiles [128,32]
            __shared__ float sC2[8][16][16];
            #pragma unroll
            for (int rep = 0; rep < 2; rep++) {
                const int tile = warp * 2 + rep;             // 0..15
                const int ti = (tile >> 1) * 16, tj = (tile & 1) * 16;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
                wmma::fill_fragment(cf, 0.f);
                #pragma unroll
                for (int kk = 0; kk < C; kk += 16) {
                    // K^T: A[m][p] = s_K[p][m] -> col_major over s_K gives the transpose for free
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> af;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> bf;
                    wmma::load_matrix_sync(af, s_K + (size_t)kk * (HD + PAD) + ti, HD + PAD);
                    wmma::load_matrix_sync(bf, s_Ub + (size_t)kk * (JC + PAD) + tj, JC + PAD);
                    wmma::mma_sync(cf, af, bf, cf);
                }
                wmma::store_matrix_sync(&sC2[warp][0][0], cf, 16, wmma::mem_row_major);
                __syncwarp();
                for (int e = (tid & 31); e < 256; e += 32) {
                    const int r = e >> 4, cc = e & 15;
                    const int m = ti + r, jj = tj + cc;
                    s_S[m * JC + jj] = gl * s_S[m * JC + jj] + sC2[warp][r][cc];
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }

    // ---- final state, in the transposed [v_head][col][row] layout decode expects ----
    for (int e = tid; e < HD * JC; e += nthr) {
        const int m = e / JC, jj = e - m * JC;
        state[((size_t)h * HD + (j0 + jj)) * HD + m] = s_S[e];
    }
}

// Workspace cache. The scan is called once per linear layer with the same N, so one allocation is
// reused across all 24 layers and every subsequent prefill; it only ever grows.
void* g_ws = nullptr;
size_t g_ws_bytes = 0;

bool ws_reserve(size_t bytes) {
    if (bytes <= g_ws_bytes) return true;
    if (g_ws) cudaFree(g_ws);
    g_ws = nullptr;
    g_ws_bytes = 0;
    if (cudaMalloc(&g_ws, bytes) != cudaSuccess) { g_ws = nullptr; return false; }
    g_ws_bytes = bytes;
    return true;
}

}  // namespace

bool launch_prefill_gdn_chunk(const void* q, const void* k, const void* v,
                              const void* alpha, const void* beta,
                              const void* dt, const void* a,
                              float* state, void* out,
                              int n_tokens, int q_heads, int v_heads, int head_dim,
                              cudaStream_t stream) {
    constexpr int C = 32, HD = 128, JC = 32, SCAN_THREADS = 256, PREP_THREADS = 256;

    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_PREFILL_GDN_CHUNK");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    static const int minctx = [] {
        const char* e = getenv("SPARKINFER_PREFILL_GDN_CHUNK_MINCTX");
        return e ? atoi(e) : 256;
    }();

    if (!enabled || head_dim != HD || n_tokens < minctx) return false;
    if (q_heads <= 0 || v_heads <= 0 || (HD % JC) != 0) return false;

    const int n_chunks = (n_tokens + C - 1) / C;

    // Workspace: G and M in fp32, W^ and U0 in bf16. W^ is the scan's largest read — every one of
    // the HD/JC column groups re-reads all of it each chunk — so bf16 there is worth ~1 pp of
    // end-to-end prefill. It does cost a little accuracy, because the scan forms U^ = U0 - W^ S and
    // that subtraction cancels: measured at ctx=4096 over 512 positions, top-1 0.9336 (fp32) ->
    // 0.9277 (bf16), KL 0.01069 -> 0.01096. Both stay well clear of the 0.90 / 0.20 gates and the
    // KL is still below the sequential scan's own 0.01136, so the trade is taken. M stays fp32: it
    // is only C*C per chunk, so shrinking it buys little.
    const size_t n_g = (size_t)n_tokens * v_heads;
    const size_t n_w = (size_t)n_tokens * v_heads * HD;
    const size_t n_m = (size_t)n_chunks * v_heads * C * C;
    const size_t off_g = 0;
    const size_t off_m = off_g + n_g * sizeof(float);
    const size_t off_w = off_m + n_m * sizeof(float);
    const size_t off_u = off_w + n_w * sizeof(__nv_bfloat16);
    const size_t total = off_u + n_w * sizeof(__nv_bfloat16);
    if (!ws_reserve(total)) return false;

    char* base  = reinterpret_cast<char*>(g_ws);
    float* g_buf = reinterpret_cast<float*>(base + off_g);
    float* m_buf = reinterpret_cast<float*>(base + off_m);
    __nv_bfloat16* w_buf = reinterpret_cast<__nv_bfloat16*>(base + off_w);
    __nv_bfloat16* u_buf = reinterpret_cast<__nv_bfloat16*>(base + off_u);

    const size_t sm_prep = (size_t)2 * C * (HD + PAD) * sizeof(__nv_bfloat16)
                         + (size_t)C * (C + PAD) * sizeof(float)
                         + (size_t)3 * C * sizeof(float);
    const size_t sm_scan = (size_t)HD * JC * sizeof(float)          // s_S (fp32 carrier)
                         + (size_t)C * JC * sizeof(float)            // s_U
                         + (size_t)C * (C + PAD) * sizeof(float)     // s_M
                         + (size_t)C * sizeof(float)                 // s_g
                         + (size_t)C * JC * sizeof(float)            // s_Y
                         + (size_t)3 * C * (HD + PAD) * sizeof(__nv_bfloat16)   // s_W, s_K, s_Q
                         + (size_t)HD * (JC + PAD) * sizeof(__nv_bfloat16)      // s_Sb
                         + (size_t)C * (JC + PAD) * sizeof(__nv_bfloat16);      // s_Ub

    static int cfg = 0;
    if (!cfg) {
        if (cudaFuncSetAttribute(pf_gdnc_prep_kernel<C, HD>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_prep) != cudaSuccess)
            return false;
        if (cudaFuncSetAttribute(pf_gdnc_scan_kernel<C, HD, JC>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm_scan) != cudaSuccess)
            return false;
        cfg = 1;
    }

    auto qb = reinterpret_cast<const __nv_bfloat16*>(q);
    auto kb = reinterpret_cast<const __nv_bfloat16*>(k);
    auto vb = reinterpret_cast<const __nv_bfloat16*>(v);
    auto ab = reinterpret_cast<const __nv_bfloat16*>(alpha);
    auto bb = reinterpret_cast<const __nv_bfloat16*>(beta);
    auto db = reinterpret_cast<const __nv_bfloat16*>(dt);
    auto aa = reinterpret_cast<const __nv_bfloat16*>(a);
    auto ob = reinterpret_cast<__nv_bfloat16*>(out);

    // PREP_THREADS is fixed by the 2x2 (i,j) tiling in the A/M pass: C*C == 4*PREP_THREADS.
    static_assert(C * C == PREP_THREADS * 4, "2x2 A/M tiling requires C*C == 4*PREP_THREADS");
    dim3 gprep(n_chunks, v_heads);
    pf_gdnc_prep_kernel<C, HD><<<gprep, PREP_THREADS, sm_prep, stream>>>(
        qb, kb, vb, ab, bb, db, aa, g_buf, w_buf, u_buf, m_buf, n_tokens, q_heads, v_heads);

    // SCAN_THREADS is not free tuning: the register tiling below partitions the [C,JC] and [HD,JC]
    // outputs across exactly this many threads (2x2 and 4x4 tiles respectively).
    static_assert(C * JC == SCAN_THREADS * 4, "2x2 tiling requires C*JC == 4*SCAN_THREADS");
    static_assert(HD * JC == SCAN_THREADS * 16, "4x4 tiling requires HD*JC == 16*SCAN_THREADS");
    dim3 gscan(v_heads, HD / JC);
    pf_gdnc_scan_kernel<C, HD, JC><<<gscan, SCAN_THREADS, sm_scan, stream>>>(
        qb, kb, g_buf, w_buf, u_buf, m_buf, state, ob, n_tokens, q_heads, v_heads, n_chunks);
    return true;
}

}  // namespace kernels
}  // namespace sparkinfer
