// Faithful weight-stationary grouped MoE FFN for batched prefill — see moe_prefill_grouped.h.
//
// Decode routes one token at a time, so every prompt token reloads all 8 routed experts' Q4_K/Q5_K
// weights as bandwidth-bound GEMVs. This path permutes the prompt's tokens into per-expert
// contiguous groups (block-aligned to the row tile), streams each expert weight into shared memory
// ONCE per tile, and reuses it across every token routed to it. The dot products use the exact
// byte-faithful K-quant dequant the decode MoE FFN uses (per-64 sub-block scale+min affine), so the
// FFN output — and thus a following decode step — matches decode; only the weight HBM traffic is
// amortized. Output is numerically faithful (no lossy per-row requant).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "sparkinfer/kernels/moe_prefill_grouped.h"
#include "sparkinfer/kernels/quant.h"

namespace sparkinfer { namespace kernels {

namespace {
constexpr int MP_TILE_M = 64;    // rows per weight-stationary tile (weight HBM read amortized over these)
constexpr int MP_WPB    = 4;     // output features per block (warps per block)

// ---- byte-faithful K-quant dequant-dot (mirrors the validated decode MoE FFN math) --------------
__device__ __forceinline__ float mp_h2f(const unsigned char* p) {
    __half h; *((unsigned short*)&h) = *(const unsigned short*)p; return __half2float(h);
}
__device__ __forceinline__ float mp_wsum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}
__device__ __forceinline__ void mp_scale_min(int j, const unsigned char* q, int* d, int* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
           *m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4); }
}
__device__ __forceinline__ float mp_silu(float x) { return x / (1.f + __expf(-x)); }
__host__ __device__ __forceinline__ int mp_qbytes(int t) { return t == 14 ? 210 : (t == 13 ? 176 : 144); }

// This lane's partial dot of one 256-value super-block `b` (native Q4_K/Q5_K/Q6_K) with sx[0..255].
__device__ __forceinline__ float mp_deq_dot(int t, const unsigned char* b, const float* sx, int lane) {
    float p = 0.f;
    if (t == 14) {   // Q6_K
        const unsigned char* ql = b; const unsigned char* qh = b + 128;
        const signed char* sc = (const signed char*)(b + 192); float d = mp_h2f(b + 208);
        #pragma unroll
        for (int nn = 0; nn < 2; nn++) {
            const unsigned char* qln = ql + nn*64; const unsigned char* qhn = qh + nn*32; const signed char* scn = sc + nn*8;
            int is = lane / 16;
            int q1 = (int)((qln[lane]    & 0xF) | (((qhn[lane] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((qln[lane+32] & 0xF) | (((qhn[lane] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((qln[lane]    >> 4)  | (((qhn[lane] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((qln[lane+32] >> 4)  | (((qhn[lane] >> 6) & 3) << 4)) - 32;
            p += d * scn[is+0] * q1 * sx[nn*128 + lane];
            p += d * scn[is+2] * q2 * sx[nn*128 + lane + 32];
            p += d * scn[is+4] * q3 * sx[nn*128 + lane + 64];
            p += d * scn[is+6] * q4 * sx[nn*128 + lane + 96];
        }
    } else if (t == 13) {   // Q5_K
        float d = mp_h2f(b), dmin = mp_h2f(b + 2);
        const unsigned char* sc = b + 4; const unsigned char* qh = b + 16; const unsigned char* ql = b + 48;
        const unsigned char hb = qh[lane];
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            int s1, m1, s2, m2;
            mp_scale_min(2*g, sc, &s1, &m1); mp_scale_min(2*g+1, sc, &s2, &m2);
            float d1 = d*s1, mm1 = dmin*m1, d2 = d*s2, mm2 = dmin*m2;
            unsigned char qb = ql[g*32 + lane];
            const unsigned char u1 = (unsigned char)(1u << (2*g)), u2 = (unsigned char)(2u << (2*g));
            int v_lo = (qb & 0xF) + ((hb & u1) ? 16 : 0);
            int v_hi = (qb >> 4)  + ((hb & u2) ? 16 : 0);
            p += (d1 * v_lo - mm1) * sx[g*64 + lane];
            p += (d2 * v_hi - mm2) * sx[g*64 + 32 + lane];
        }
    } else {         // Q4_K
        float d = mp_h2f(b), dmin = mp_h2f(b + 2);
        const unsigned char* sc = b + 4; const unsigned char* qs = b + 16;
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            int s1, m1, s2, m2;
            mp_scale_min(2*g, sc, &s1, &m1); mp_scale_min(2*g+1, sc, &s2, &m2);
            float d1 = d*s1, mm1 = dmin*m1, d2 = d*s2, mm2 = dmin*m2;
            unsigned char qb = qs[g*32 + lane];
            p += (d1 * (qb & 0xF) - mm1) * sx[g*64 + lane];
            p += (d2 * (qb >> 4)  - mm2) * sx[g*64 + 32 + lane];
        }
    }
    return p;
}

// ---- routing ---------------------------------------------------------------------------------
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
    acc = mp_wsum(acc);
    if (lane == 0) logits[(size_t)n * E + e] = acc;
}

// ---- group construction ----------------------------------------------------------------------
__global__ void mp_build_offsets_kernel(const int* __restrict__ counts, int* __restrict__ offsets,
                                        int* __restrict__ tile_expert, int* __restrict__ cursor,
                                        int E, int max_tiles) {
    extern __shared__ int s_aligned[];
    const int e = threadIdx.x;
    if (e < E) { int c = counts[e]; s_aligned[e] = ((c + MP_TILE_M - 1) / MP_TILE_M) * MP_TILE_M; cursor[e] = 0; }
    __syncthreads();
    if (e == 0) {
        int off = 0;
        for (int j = 0; j < E; j++) {
            offsets[j] = off;
            const int t0 = off / MP_TILE_M, nt = s_aligned[j] / MP_TILE_M;
            for (int t = 0; t < nt; t++) tile_expert[t0 + t] = j;
            off += s_aligned[j];
        }
        offsets[E] = off;
        for (int t = off / MP_TILE_M; t < max_tiles; t++) tile_expert[t] = -1;
    }
}

__global__ void mp_scatter_assign_kernel(const int* __restrict__ ids, const float* __restrict__ weights,
                                         const int* __restrict__ offsets, int* __restrict__ cursor,
                                         int* __restrict__ pos_token, float* __restrict__ pos_weight, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T) return;
    const int e = ids[i];
    const int p = offsets[e] + atomicAdd(&cursor[e], 1);
    pos_token[p] = i;
    pos_weight[p] = weights[i];
}

// Dequant one native 256-value super-block `b` straight to bf16 smem out[0..255], byte-faithful.
__device__ __forceinline__ void mp_deq_store(int t, const unsigned char* b, __nv_bfloat16* out, int lane) {
    if (t == 14) {   // Q6_K
        const unsigned char* ql = b; const unsigned char* qh = b + 128;
        const signed char* sc = (const signed char*)(b + 192); float d = mp_h2f(b + 208);
        #pragma unroll
        for (int nn = 0; nn < 2; nn++) {
            const unsigned char* qln = ql + nn*64; const unsigned char* qhn = qh + nn*32; const signed char* scn = sc + nn*8;
            int is = lane / 16;
            int q1 = (int)((qln[lane]    & 0xF) | (((qhn[lane] >> 0) & 3) << 4)) - 32;
            int q2 = (int)((qln[lane+32] & 0xF) | (((qhn[lane] >> 2) & 3) << 4)) - 32;
            int q3 = (int)((qln[lane]    >> 4)  | (((qhn[lane] >> 4) & 3) << 4)) - 32;
            int q4 = (int)((qln[lane+32] >> 4)  | (((qhn[lane] >> 6) & 3) << 4)) - 32;
            out[nn*128 + lane]      = __float2bfloat16(d * scn[is+0] * q1);
            out[nn*128 + lane + 32] = __float2bfloat16(d * scn[is+2] * q2);
            out[nn*128 + lane + 64] = __float2bfloat16(d * scn[is+4] * q3);
            out[nn*128 + lane + 96] = __float2bfloat16(d * scn[is+6] * q4);
        }
    } else if (t == 13) {   // Q5_K
        float d = mp_h2f(b), dmin = mp_h2f(b + 2);
        const unsigned char* sc = b + 4; const unsigned char* qh = b + 16; const unsigned char* ql = b + 48;
        const unsigned char hb = qh[lane];
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            int s1, m1, s2, m2;
            mp_scale_min(2*g, sc, &s1, &m1); mp_scale_min(2*g+1, sc, &s2, &m2);
            float d1 = d*s1, mm1 = dmin*m1, d2 = d*s2, mm2 = dmin*m2;
            unsigned char qb = ql[g*32 + lane];
            const unsigned char u1 = (unsigned char)(1u << (2*g)), u2 = (unsigned char)(2u << (2*g));
            int v_lo = (qb & 0xF) + ((hb & u1) ? 16 : 0);
            int v_hi = (qb >> 4)  + ((hb & u2) ? 16 : 0);
            out[g*64 + lane]      = __float2bfloat16(d1 * v_lo - mm1);
            out[g*64 + 32 + lane] = __float2bfloat16(d2 * v_hi - mm2);
        }
    } else {         // Q4_K
        float d = mp_h2f(b), dmin = mp_h2f(b + 2);
        const unsigned char* sc = b + 4; const unsigned char* qs = b + 16;
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            int s1, m1, s2, m2;
            mp_scale_min(2*g, sc, &s1, &m1); mp_scale_min(2*g+1, sc, &s2, &m2);
            float d1 = d*s1, mm1 = dmin*m1, d2 = d*s2, mm2 = dmin*m2;
            unsigned char qb = qs[g*32 + lane];
            out[g*64 + lane]      = __float2bfloat16(d1 * (qb & 0xF) - mm1);
            out[g*64 + 32 + lane] = __float2bfloat16(d2 * (qb >> 4)  - mm2);
        }
    }
}

// ---- faithful grouped gate_up: h[row,f] = silu(<x,gate[e,f]>) * <x,up[e,f]> --------------------
// Weight-stationary: dequant the MP_WPB feature weights (gate+up) for expert e to bf16 in smem ONCE
// per tile (byte-faithful), then reuse across the tile's rows — amortizing both the weight HBM read
// and the dequant. Each row reads its activation straight from L2 (no per-row barrier).
__global__ void mp_gate_up_faithful_kernel(
        const __nv_bfloat16* __restrict__ hn, const int* __restrict__ pos_token, int top_k,
        const int* __restrict__ tile_expert, const unsigned char* __restrict__ gate_q,
        const unsigned char* __restrict__ up_q, int gate_type, int up_type,
        __nv_bfloat16* __restrict__ h_out, int H, int F, int rows) {
    const int e = tile_expert[blockIdx.x];
    if (e < 0) return;
    const int nsb = H >> 8;
    const int gbb = mp_qbytes(gate_type), ubb = mp_qbytes(up_type);
    const int fbase = blockIdx.y * MP_WPB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int f = fbase + warp;

    extern __shared__ __nv_bfloat16 s_w[];   // [2*MP_WPB][H] : gate then up, dequanted
    __nv_bfloat16* s_g = s_w;
    __nv_bfloat16* s_u = s_w + (size_t)MP_WPB * H;
    // dequant each feature's gate+up superblocks to bf16 (one warp per feature)
    if (f < F) {
        const unsigned char* gb = gate_q + ((size_t)e * F + f) * nsb * gbb;
        const unsigned char* ub = up_q   + ((size_t)e * F + f) * nsb * ubb;
        for (int sb = 0; sb < nsb; sb++) {
            mp_deq_store(gate_type, gb + (size_t)sb * gbb, s_g + (size_t)warp * H + sb * 256, lane);
            mp_deq_store(up_type,   ub + (size_t)sb * ubb, s_u + (size_t)warp * H + sb * 256, lane);
        }
    }
    __syncthreads();
    if (f >= F) return;

    const int row0 = blockIdx.x * MP_TILE_M;
    const __nv_bfloat16* wg = s_g + (size_t)warp * H;
    const __nv_bfloat16* wu = s_u + (size_t)warp * H;
    for (int r = 0; r < MP_TILE_M; r++) {
        const int row = row0 + r;
        if (row >= rows) break;
        const int tok = pos_token[row];
        if (tok < 0) continue;
        const __nv_bfloat16* a = hn + (size_t)(tok / top_k) * H;
        float g = 0.f, u = 0.f;
        for (int c = lane; c < H; c += 32) {
            float av = __bfloat162float(a[c]);
            g += __bfloat162float(wg[c]) * av;
            u += __bfloat162float(wu[c]) * av;
        }
        g = mp_wsum(g); u = mp_wsum(u);
        if (lane == 0) h_out[(size_t)row * F + f] = __float2bfloat16(mp_silu(g) * u);
    }
}

// ---- faithful grouped down: D[row,hh] = <h[row], down[e,hh]> -----------------------------------
__global__ void mp_down_faithful_kernel(
        const __nv_bfloat16* __restrict__ h_in, const int* __restrict__ pos_token,
        const int* __restrict__ tile_expert, const unsigned char* __restrict__ down_q, int down_type,
        __nv_bfloat16* __restrict__ D, int H, int F, int rows) {
    const int e = tile_expert[blockIdx.x];
    if (e < 0) return;
    const int nsb = F >> 8;
    const int dbb = mp_qbytes(down_type);
    const int hhbase = blockIdx.y * MP_WPB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int hh = hhbase + warp;

    extern __shared__ __nv_bfloat16 s_d[];   // [MP_WPB][F] dequanted
    if (hh < H) {
        const unsigned char* db = down_q + ((size_t)e * H + hh) * nsb * dbb;
        for (int sb = 0; sb < nsb; sb++)
            mp_deq_store(down_type, db + (size_t)sb * dbb, s_d + (size_t)warp * F + sb * 256, lane);
    }
    __syncthreads();
    if (hh >= H) return;

    const int row0 = blockIdx.x * MP_TILE_M;
    const __nv_bfloat16* wd = s_d + (size_t)warp * F;
    for (int r = 0; r < MP_TILE_M; r++) {
        const int row = row0 + r;
        if (row >= rows) break;
        if (pos_token[row] < 0) continue;
        const __nv_bfloat16* hp = h_in + (size_t)row * F;
        float acc = 0.f;
        for (int c = lane; c < F; c += 32) acc += __bfloat162float(wd[c]) * __bfloat162float(hp[c]);
        acc = mp_wsum(acc);
        if (lane == 0) D[(size_t)row * H + hh] = __float2bfloat16(acc);
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
int moe_prefill_max_tiles(int N, int top_k, int E) { return (N * top_k + MP_TILE_M - 1) / MP_TILE_M + E; }
int moe_prefill_padded_rows(int N, int top_k, int E) { return moe_prefill_max_tiles(N, top_k, E) * MP_TILE_M; }

void launch_moe_prefill_router_logits(const void* hn, const void* router_w, int router_w_type,
                                      void* rw_bf16, float* logits, int N, int H, int E, cudaStream_t st) {
    const __nv_bfloat16* rw;
    if (router_w_type != 0) {
        launch_gguf_dequant(router_w_type, router_w, rw_bf16, (long)E * H, st);
        rw = reinterpret_cast<const __nv_bfloat16*>(rw_bf16);
    } else rw = reinterpret_cast<const __nv_bfloat16*>(router_w);
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
    cudaMemsetAsync(pos_token, 0xFF, (size_t)max_tiles * MP_TILE_M * sizeof(int), st);
    int* cursor = offsets + (E + 1);
    mp_build_offsets_kernel<<<1, ((E + 31) / 32) * 32, E * sizeof(int), st>>>(
        counts, offsets, tile_expert, cursor, E, max_tiles);
    mp_scatter_assign_kernel<<<(T + 255) / 256, 256, 0, st>>>(
        ids, weights, offsets, cursor, pos_token, pos_weight, T);
}

void launch_moe_grouped_gate_up(const void* hn, const int* pos_token, int top_k, const int* tile_expert,
                                int max_tiles, const void* gate_q, const void* up_q, int gate_type, int up_type,
                                void* h_out, int H, int F, int rows, cudaStream_t st) {
    size_t smem = (size_t)2 * MP_WPB * H * sizeof(__nv_bfloat16);
    dim3 grid(max_tiles, (F + MP_WPB - 1) / MP_WPB);
    mp_gate_up_faithful_kernel<<<grid, MP_WPB * 32, smem, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(hn), pos_token, top_k, tile_expert,
        reinterpret_cast<const unsigned char*>(gate_q), reinterpret_cast<const unsigned char*>(up_q),
        gate_type, up_type, reinterpret_cast<__nv_bfloat16*>(h_out), H, F, rows);
}

void launch_moe_grouped_down(const void* h_in, const int* pos_token, const int* tile_expert, int max_tiles,
                             const void* down_q, int down_type, void* D, int H, int F, int rows, cudaStream_t st) {
    size_t smem = (size_t)MP_WPB * F * sizeof(__nv_bfloat16);
    dim3 grid(max_tiles, (H + MP_WPB - 1) / MP_WPB);
    mp_down_faithful_kernel<<<grid, MP_WPB * 32, smem, st>>>(
        reinterpret_cast<const __nv_bfloat16*>(h_in), pos_token, tile_expert,
        reinterpret_cast<const unsigned char*>(down_q), down_type, reinterpret_cast<__nv_bfloat16*>(D), H, F, rows);
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
