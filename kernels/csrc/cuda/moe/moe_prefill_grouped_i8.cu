// Grouped int8 tensor-core MoE FFN for batched prefill — see moe_prefill_grouped.h.
//
// The whole prompt runs through the MoE FFN in one pass: top-k routing permutes tokens into
// per-expert contiguous groups (block-aligned to the 128-row GEMM tile), so each Q4_K/Q6_K expert
// weight is requantized to int8 once and streamed once, then reused across every token routed to it.
// gate/up/down run as int8 mma.sync m16n8k32 GEMMs — the same per-row symmetric int8 scheme and
// accumulation order the dense prefill projections already use, so the FFN output tracks the decode
// MoE FFN and a following decode step stays faithful.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include "sparkinfer/kernels/moe_prefill_grouped.h"
#include "sparkinfer/kernels/quant.h"

namespace sparkinfer { namespace kernels {

namespace {
constexpr int MBM = 128, MBN = 128, MBK = 64;   // output tile 128x128, K-step 64 (mirrors pf_gemm_i8)
constexpr int MFRAG = 2, NFRAG = 8;             // 32 rows / 16, 64 cols / 8 per warp

__device__ __forceinline__ void mp_cp16(void* dst, const void* src, bool pred) {
    if (pred) __pipeline_memcpy_async(dst, src, 16);
    else      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
}
__device__ __forceinline__ int mp_swz(int k, int row) { return (((k >> 4) ^ (row & 3)) << 4) | (k & 15); }
__device__ __forceinline__ unsigned mp_lds32(const signed char* p) { return *reinterpret_cast<const unsigned*>(p); }

// ---- routing ---------------------------------------------------------------------------------
// logits[n,e] = <hn[n], rw[e]>, one warp per (token, expert). rw is bf16 [E,H].
__global__ void mp_router_logits_kernel(const __nv_bfloat16* __restrict__ hn,
                                        const __nv_bfloat16* __restrict__ rw,
                                        float* __restrict__ logits, int N, int H, int E) {
    const int n = blockIdx.x, e = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (n >= N || e >= E) return;
    const int lane = threadIdx.x & 31;
    const __nv_bfloat16* a = hn + (size_t)n * H;
    const __nv_bfloat16* w = rw + (size_t)e * H;
    float acc = 0.f;
    for (int c = lane; c < H; c += 32) acc += __bfloat162float(a[c]) * __bfloat162float(w[c]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) logits[(size_t)n * E + e] = acc;
}

// ---- group construction ----------------------------------------------------------------------
// One block, E threads: block-align the per-expert counts, exclusive-scan into padded row offsets,
// stamp the (row-tile -> expert) schedule, and clear the per-expert scatter cursors.
__global__ void mp_build_offsets_kernel(const int* __restrict__ counts, int* __restrict__ offsets,
                                        int* __restrict__ tile_expert, int* __restrict__ cursor,
                                        int E, int max_tiles) {
    extern __shared__ int s_aligned[];   // E padded counts
    const int e = threadIdx.x;
    if (e < E) {
        int c = counts[e];
        s_aligned[e] = ((c + MBM - 1) / MBM) * MBM;   // round group up to the 128-row tile
        cursor[e] = 0;
    }
    __syncthreads();
    if (e == 0) {   // serial prefix over E=256 experts (cheap, once per layer)
        int off = 0;
        for (int j = 0; j < E; j++) {
            offsets[j] = off;
            const int t0 = off / MBM, nt = s_aligned[j] / MBM;
            for (int t = 0; t < nt; t++) tile_expert[t0 + t] = j;
            off += s_aligned[j];
        }
        offsets[E] = off;
        for (int t = off / MBM; t < max_tiles; t++) tile_expert[t] = -1;   // beyond real tiles
    }
}

// Scatter each (token, slot) assignment into its expert's padded group.
__global__ void mp_scatter_assign_kernel(const int* __restrict__ ids, const float* __restrict__ weights,
                                         const int* __restrict__ offsets, int* __restrict__ cursor,
                                         int* __restrict__ pos_token, float* __restrict__ pos_weight,
                                         int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T) return;
    const int e = ids[i];
    const int p = offsets[e] + atomicAdd(&cursor[e], 1);
    pos_token[p] = i;            // flat token*top_k+slot index; token recovered as i/top_k downstream
    pos_weight[p] = weights[i];
}

// ---- gather + per-row int8 quantize ----------------------------------------------------------
__global__ void mp_gather_quant_i8_kernel(const __nv_bfloat16* __restrict__ hn,
                                          const int* __restrict__ pos_token, int top_k,
                                          signed char* __restrict__ Ai8, float* __restrict__ sA,
                                          int rows, int H) {
    const int r = blockIdx.x, lane = threadIdx.x;
    if (r >= rows) return;
    const int flat = pos_token[r];
    signed char* out = Ai8 + (size_t)r * H;
    if (flat < 0) {   // padding row -> zero
        for (int c = lane; c < H; c += 32) out[c] = 0;
        if (lane == 0) sA[r] = 0.f;
        return;
    }
    const __nv_bfloat16* a = hn + (size_t)(flat / top_k) * H;
    float amax = 0.f;
    for (int c = lane; c < H; c += 32) amax = fmaxf(amax, fabsf(__bfloat162float(a[c])));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    if (lane == 0) sA[r] = d;
    for (int c = lane; c < H; c += 32)
        out[c] = (signed char)((amax == 0.f) ? 0 : (int)roundf(__bfloat162float(a[c]) / d));
}

// ---- grouped int8 GEMM -----------------------------------------------------------------------
// C[t,Nout] = Ai8[t,K] @ W_i8[e,Nout,K]^T for the expert e = tile_expert[blockIdx.x]. Rows are the
// 128-row block-aligned group tile; dims are tile-aligned (Nout%128==0, K%64==0) by construction.
__global__ __launch_bounds__(256, 2) void mp_grouped_gemm_i8_kernel(
        const signed char* __restrict__ A, const float* __restrict__ sA,
        const signed char* __restrict__ W, const float* __restrict__ sW,
        const int* __restrict__ tile_expert, __nv_bfloat16* __restrict__ C, int Nout, int K) {
    const int e = tile_expert[blockIdx.x];
    if (e < 0) return;
    const signed char* We = W + (size_t)e * Nout * K;
    const float* sWe = sW + (size_t)e * Nout;

    __shared__ signed char As[2][MBM][MBK];
    __shared__ signed char Bs[2][MBN][MBK];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grp = lane >> 2, tig = lane & 3, wm = warp & 3, wn = warp >> 2;
    const int m0 = blockIdx.x * MBM, n0 = blockIdx.y * MBN;
    const int nk = (K + MBK - 1) / MBK;

    int acc[MFRAG][NFRAG][4];
    #pragma unroll
    for (int i = 0; i < MFRAG; i++)
        #pragma unroll
        for (int j = 0; j < NFRAG; j++)
            #pragma unroll
            for (int el = 0; el < 4; el++) acc[i][j][el] = 0;

    auto stage = [&](int buf, int k0) {
        #pragma unroll
        for (int s = tid; s < 512; s += 256) {
            const int r = s >> 2, c = s & 3, k = c << 4;
            const int gk = k0 + k;
            mp_cp16(&As[buf][r][mp_swz(k, r)], &A[(size_t)(m0 + r) * K + gk], gk < K);
            mp_cp16(&Bs[buf][r][mp_swz(k, r)], &We[(size_t)(n0 + r) * K + gk], gk < K);
        }
        __pipeline_commit();
    };

    stage(0, 0);
    int buf = 0;
    for (int t = 0; t < nk; t++) {
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * MBK);
        __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < MBK; kk += 32) {
            const int ka = kk + tig * 4;
            unsigned af[MFRAG][4], bf[NFRAG][2];
            #pragma unroll
            for (int i = 0; i < MFRAG; i++) {
                const int rlo = wm * 32 + i * 16 + grp, rhi = rlo + 8;
                af[i][0] = mp_lds32(&As[buf][rlo][mp_swz(ka,      rlo)]);
                af[i][1] = mp_lds32(&As[buf][rhi][mp_swz(ka,      rhi)]);
                af[i][2] = mp_lds32(&As[buf][rlo][mp_swz(ka + 16, rlo)]);
                af[i][3] = mp_lds32(&As[buf][rhi][mp_swz(ka + 16, rhi)]);
            }
            #pragma unroll
            for (int j = 0; j < NFRAG; j++) {
                const int col = wn * 64 + j * 8 + grp;
                bf[j][0] = mp_lds32(&Bs[buf][col][mp_swz(ka,      col)]);
                bf[j][1] = mp_lds32(&Bs[buf][col][mp_swz(ka + 16, col)]);
            }
            #pragma unroll
            for (int i = 0; i < MFRAG; i++)
                #pragma unroll
                for (int j = 0; j < NFRAG; j++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]), "+r"(acc[i][j][2]), "+r"(acc[i][j][3])
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]),
                          "r"(bf[j][0]), "r"(bf[j][1]));
        }
        __syncthreads();
        buf ^= 1;
    }

    #pragma unroll
    for (int i = 0; i < MFRAG; i++) {
        #pragma unroll
        for (int j = 0; j < NFRAG; j++) {
            const int gn = n0 + wn * 64 + j * 8 + tig * 2;
            const float w0 = sWe[gn], w1 = sWe[gn + 1];
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int gm = m0 + wm * 32 + i * 16 + grp + h * 8;
                const float s = sA[gm];
                const __nv_bfloat162 v = __floats2bfloat162_rn((float)acc[i][j][h * 2] * s * w0,
                                                               (float)acc[i][j][h * 2 + 1] * s * w1);
                *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)gm * Nout + gn]) = v;
            }
        }
    }
}

// ---- SwiGLU -> int8 --------------------------------------------------------------------------
__global__ void mp_swiglu_quant_i8_kernel(const __nv_bfloat16* __restrict__ gate,
                                          const __nv_bfloat16* __restrict__ up,
                                          signed char* __restrict__ Hi8, float* __restrict__ sH,
                                          int rows, int F) {
    const int r = blockIdx.x, lane = threadIdx.x;
    if (r >= rows) return;
    const __nv_bfloat16* g = gate + (size_t)r * F;
    const __nv_bfloat16* u = up   + (size_t)r * F;
    float amax = 0.f;
    for (int c = lane; c < F; c += 32) {
        const float gv = __bfloat162float(g[c]);
        const float h = (gv / (1.f + __expf(-gv))) * __bfloat162float(u[c]);
        amax = fmaxf(amax, fabsf(h));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    if (lane == 0) sH[r] = d;
    signed char* out = Hi8 + (size_t)r * F;
    for (int c = lane; c < F; c += 32) {
        const float gv = __bfloat162float(g[c]);
        const float h = (gv / (1.f + __expf(-gv))) * __bfloat162float(u[c]);
        out[c] = (signed char)((amax == 0.f) ? 0 : (int)roundf(h / d));
    }
}

// ---- weighted scatter-back -------------------------------------------------------------------
__global__ void mp_scatter_weighted_kernel(const __nv_bfloat16* __restrict__ D,
                                           const int* __restrict__ pos_token, int top_k,
                                           const float* __restrict__ pos_weight,
                                           float* __restrict__ routed, int rows, int H) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int flat = pos_token[r];
    if (flat < 0) return;
    const int n = flat / top_k;
    const float wgt = pos_weight[r];
    const __nv_bfloat16* d = D + (size_t)r * H;
    float* dst = routed + (size_t)n * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x)
        atomicAdd(&dst[c], wgt * __bfloat162float(d[c]));
}

// ---- finalize --------------------------------------------------------------------------------
__global__ void mp_finalize_kernel(const __nv_bfloat16* __restrict__ h, const float* __restrict__ routed,
                                   const __nv_bfloat16* __restrict__ shared, const float* __restrict__ gate_logit,
                                   __nv_bfloat16* __restrict__ out, int N, int H) {
    const int n = blockIdx.x;
    if (n >= N) return;
    const float g = (gate_logit && shared) ? 1.f / (1.f + __expf(-gate_logit[n])) : 0.f;
    const __nv_bfloat16* hn = h + (size_t)n * H;
    const float* rn = routed + (size_t)n * H;
    const __nv_bfloat16* sn = shared ? shared + (size_t)n * H : nullptr;
    __nv_bfloat16* on = out + (size_t)n * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
        float v = __bfloat162float(hn[c]) + rn[c];
        if (sn) v += g * __bfloat162float(sn[c]);
        on[c] = __float2bfloat16(v);
    }
}
} // namespace

// ---- host launchers --------------------------------------------------------------------------
int moe_prefill_max_tiles(int N, int top_k, int E) { return (N * top_k + MBM - 1) / MBM + E; }
int moe_prefill_padded_rows(int N, int top_k, int E) { return moe_prefill_max_tiles(N, top_k, E) * MBM; }

void launch_moe_prefill_router_logits(const void* hn, const void* router_w, int router_w_type,
                                      void* rw_bf16, float* logits, int N, int H, int E, cudaStream_t st) {
    const __nv_bfloat16* rw;
    if (router_w_type != 0) {
        launch_gguf_dequant(router_w_type, router_w, rw_bf16, (long)E * H, st);
        rw = reinterpret_cast<const __nv_bfloat16*>(rw_bf16);
    } else {
        rw = reinterpret_cast<const __nv_bfloat16*>(router_w);
    }
    const int WPB = 8;
    dim3 grid(N, (E + WPB - 1) / WPB);
    mp_router_logits_kernel<<<grid, WPB * 32, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(hn), rw, logits, N, H, E);
}

void launch_moe_prefill_build_groups(const int* ids, const float* weights, const int* counts,
                                     int* offsets, int* pos_token, float* pos_weight, int* tile_expert,
                                     int* /*d_num_tiles*/, int N, int top_k, int E, cudaStream_t st) {
    const int T = N * top_k;
    const int max_tiles = moe_prefill_max_tiles(N, top_k, E);
    cudaMemsetAsync(pos_token, 0xFF, (size_t)max_tiles * MBM * sizeof(int), st);   // -1 padding sentinel
    int* cursor = offsets + (E + 1);   // reuse the tail of the offsets scratch as the E cursors
    mp_build_offsets_kernel<<<1, ((E + 31) / 32) * 32, E * sizeof(int), st>>>(
        counts, offsets, tile_expert, cursor, E, max_tiles);
    mp_scatter_assign_kernel<<<(T + 255) / 256, 256, 0, st>>>(
        ids, weights, offsets, cursor, pos_token, pos_weight, T);
}

void launch_moe_prefill_gather_quant_i8(const void* hn, const int* pos_token, int top_k,
                                        signed char* Ai8, float* sA, int rows, int H, cudaStream_t st) {
    mp_gather_quant_i8_kernel<<<rows, 32, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(hn), pos_token, top_k, Ai8, sA, rows, H);
}

void launch_moe_grouped_gemm_i8(const signed char* Ai8, const float* sA, const signed char* W_i8,
                                const float* sW, const int* /*offsets*/, const int* tile_expert,
                                int max_tiles, void* C, int Nout, int K, cudaStream_t st) {
    dim3 grid(max_tiles, (Nout + MBN - 1) / MBN);
    mp_grouped_gemm_i8_kernel<<<grid, 256, 0, st>>>(
        Ai8, sA, W_i8, sW, tile_expert, reinterpret_cast<__nv_bfloat16*>(C), Nout, K);
}

void launch_moe_prefill_swiglu_quant_i8(const void* gate, const void* up, signed char* Hi8,
                                        float* sH, int rows, int F, cudaStream_t st) {
    mp_swiglu_quant_i8_kernel<<<rows, 32, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate), reinterpret_cast<const __nv_bfloat16*>(up),
        Hi8, sH, rows, F);
}

void launch_moe_prefill_scatter_weighted(const void* D, const int* pos_token, int top_k,
                                         const float* pos_weight, void* routed, int rows, int N,
                                         int H, cudaStream_t st) {
    cudaMemsetAsync(routed, 0, (size_t)N * H * sizeof(float), st);
    mp_scatter_weighted_kernel<<<rows, 256, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(D), pos_token, top_k, pos_weight,
        reinterpret_cast<float*>(routed), rows, H);
}

void launch_moe_prefill_finalize(const void* h, const void* routed, const void* shared,
                                 const float* gate_scalar, void* out, int N, int H, cudaStream_t st) {
    mp_finalize_kernel<<<N, 256, 0, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(h), reinterpret_cast<const float*>(routed),
        reinterpret_cast<const __nv_bfloat16*>(shared), gate_scalar,
        reinterpret_cast<__nv_bfloat16*>(out), N, H);
}

}} // namespace sparkinfer::kernels
