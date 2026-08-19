// Fused quantized MoE expert FFN for decode (batch small).
//
// Closes most of the gap vs llama.cpp on Qwen3-MoE decode:
//   - dequantizes ONLY the top_k routed experts, on-read inside the GEMV — no
//     bf16 materialization, no 16x wasted dequant of unused experts.
//   - one warp per output row; thousands of warps fill the GPU (vs one CTA).
//   - reads GGUF-native quantized weights directly (gate/up = Q4_K [E,F,H],
//     down = Q6_K [E,H,F]). Decode is memory-bound on the quantized weight reads
//     — the right regime for a CUDA-core GEMV.
//   - down pass accumulates the top_k experts inside each warp and writes the
//     output once (no atomics, no scratch).
//
// Q4_K/Q6_K decoders are the byte-exact ones validated in dequant_gguf.cu.
// Requires hidden and ffn to be multiples of 256 (Qwen3-30B-A3B: 2048, 768).
//
// Portable CUDA — sm_89 .. sm_120/sm_121.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#include "sparkinfer/kernels/qtype.h"   // SI_QTYPE_Q3A (moe.h below is included from *inside* the namespace)
#endif

namespace sparkinfer {
namespace kernels {

static constexpr int WPB = 8;   // warps per block

// Programmatic Dependent Launch (PDL): overlap a kernel's grid spin-up with its
// predecessor's tail to hide bs=1 decode launch latency (the ncu-confirmed bottleneck).
// No-op unless the kernel is launched programmatic (cudaLaunchKernelEx +
// ProgrammaticStreamSerialization) on sm_90+. NVRTC device path stays a no-op.
__device__ __forceinline__ void si_pdl_lc() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(SPARKINFER_NVRTC_DEVICE_ONLY)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}
__device__ __forceinline__ void si_pdl_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && !defined(SPARKINFER_NVRTC_DEVICE_ONLY)
    cudaGridDependencySynchronize();
#endif
}

__device__ __forceinline__ float q4kf_h2f(const unsigned char* p) {
    __half h; *((unsigned short*)&h) = *(const unsigned short*)p; return __half2float(h);
}
__device__ __forceinline__ float q4kf_wsum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}
__device__ __forceinline__ void q4kf_scale_min(int j, const unsigned char* q, int* d, int* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
           *m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4); }
}
__device__ __forceinline__ float q4kf_silu(float x) { return x / (1.f + __expf(-x)); }

// Dequant a 256-block in registers and return THIS lane's partial dot with sx[0..255]
// (8 weights/lane). No shared round-trip — caller warp-reduces the accumulated partials
// once at the end. Same math as warp_deq + the shared dot, just fused and register-resident.
__device__ __forceinline__ float q4kf_deq_dot(int t, const unsigned char* b, const float* sx, int lane) {
    float p = 0.f;
    if (t == 14) {   // Q6_K
        const unsigned char* ql = b; const unsigned char* qh = b + 128;
        const signed char* sc = (const signed char*)(b + 192); float d = q4kf_h2f(b + 208);
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
    } else if (t == 13) {   // Q5_K, 176 B/256 — 4-bit ql (b+48) + 1 high bit/quant (qh, b+16).
        float d = q4kf_h2f(b), dmin = q4kf_h2f(b + 2);
        const unsigned char* sc = b + 4; const unsigned char* qh = b + 16; const unsigned char* ql = b + 48;
        const unsigned char hb = qh[lane];
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            int s1, m1, s2, m2;
            q4kf_scale_min(2*g, sc, &s1, &m1); q4kf_scale_min(2*g+1, sc, &s2, &m2);
            float d1 = d*s1, mm1 = dmin*m1, d2 = d*s2, mm2 = dmin*m2;
            unsigned char qb = ql[g*32 + lane];
            const unsigned char u1 = (unsigned char)(1u << (2*g)), u2 = (unsigned char)(2u << (2*g));
            int v_lo = (qb & 0xF) + ((hb & u1) ? 16 : 0);
            int v_hi = (qb >> 4)  + ((hb & u2) ? 16 : 0);
            p += (d1 * v_lo - mm1) * sx[g*64 + lane];
            p += (d2 * v_hi - mm2) * sx[g*64 + 32 + lane];
        }
    } else {         // Q4_K
        float d = q4kf_h2f(b), dmin = q4kf_h2f(b + 2);
        const unsigned char* sc = b + 4; const unsigned char* qs = b + 16;
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            int s1, m1, s2, m2;
            q4kf_scale_min(2*g, sc, &s1, &m1); q4kf_scale_min(2*g+1, sc, &s2, &m2);
            float d1 = d*s1, mm1 = dmin*m1, d2 = d*s2, mm2 = dmin*m2;
            unsigned char qb = qs[g*32 + lane];
            p += (d1 * (qb & 0xF) - mm1) * sx[g*64 + lane];
            p += (d2 * (qb >> 4)  - mm2) * sx[g*64 + 32 + lane];
        }
    }
    return p;
}

// ggml types: Q4_K=12 (144 B/256), Q5_K=13 (176 B/256), Q6_K=14 (210 B/256). UD quants mix them per tensor.
__device__ __forceinline__ int q_block_bytes(int t) { return t == 14 ? 210 : (t == 13 ? 176 : 144); }

// gate_up: h[ts,f] = SiLU(<x, gate[e,f]>) * <x, up[e,f]>.  one warp per f.
// grid=(num_tokens*top_k, ffn/WPB), block=WPB*32. smem: s_x[hidden] + WPB*256.
__global__ void gate_up_q4k_kernel(
    const __nv_bfloat16* __restrict__ input, const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q, const int* __restrict__ expert_ids,
    float* __restrict__ h_scratch, int H, int F, int top_k, int gate_type, int up_type
) {
    extern __shared__ float s_x[];           // s_x[H]
    const int ts = blockIdx.x, tok = ts / top_k;
    const int e = expert_ids[ts];
    for (int i = threadIdx.x; i < H; i += blockDim.x) s_x[i] = __bfloat162float(input[(size_t)tok * H + i]);
    __syncthreads();

    const int lane = threadIdx.x % 32;
    const int f = blockIdx.y * WPB + (threadIdx.x / 32);
    if (f >= F) return;
    const int nblk = H / 256;
    const int gbb = q_block_bytes(gate_type), ubb = q_block_bytes(up_type);
    const unsigned char* gbase = gate_q + ((size_t)e * F + f) * nblk * gbb;
    const unsigned char* ubase = up_q   + ((size_t)e * F + f) * nblk * ubb;
    float g = 0.f, u = 0.f;
    for (int blk = 0; blk < nblk; blk++) {
        const float* sx = s_x + blk * 256;
        g += q4kf_deq_dot(gate_type, gbase + (size_t)blk * gbb, sx, lane);
        u += q4kf_deq_dot(up_type,   ubase + (size_t)blk * ubb, sx, lane);
    }
    g = q4kf_wsum(g); u = q4kf_wsum(u);
    if (lane == 0) h_scratch[(size_t)ts * F + f] = q4kf_silu(g) * u;
}

// ---- int8 dp4a MMVQ gate/up (SPARKINFER_MMVQ=1) -------------------------------
// Same result as gate_up_q4k_kernel but stays in int8: the activation x is
// quantized to Q8_1 once per token (s_xq8 + per-32-block scale s_xd and the
// Q8_1 sum term s_xs), and the Q4_K weight nibbles are dp4a'd directly against
// it — no dequant-to-fp, no shared round-trip. Math is the faithful llama.cpp
// vec_dot_q4_K_q8_1 identity, derived to match the byte-exact warp_deq_q4k:
//   <w,a>_sub = d*sc*xd*dp4a(q4, xq8) - dmin*m*(xd*sum xq8).
// Each lane owns whole 32-sub-blocks (2 of them for H=2048) and reduces once.
// Assumes Q4_K (ggml type 12) gate+up; launcher falls back otherwise.
__global__ void gate_up_q4k_mmvq_kernel(
    const __nv_bfloat16* __restrict__ input, const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q, const int* __restrict__ expert_ids,
    float* __restrict__ h_scratch, int H, int F, int top_k, int pdl
) {
    extern __shared__ char smem_mmvq[];
    float* s_xd = reinterpret_cast<float*>(smem_mmvq);   // [H/32] activation scales
    float* s_xs = s_xd + (H >> 5);                        // [H/32] Q8_1 sum (d*sum)
    signed char* s_xq8 = reinterpret_cast<signed char*>(s_xs + (H >> 5)); // [H] int8

    const int ts = blockIdx.x, tok = ts / top_k;
    const int e = expert_ids[ts];
    const int warpId = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nsb = H >> 5;   // sub-blocks of 32

    // quantize activation -> Q8_1, one 32-block per warp-iteration (lane = element)
    for (int b = warpId; b < nsb; b += WPB) {
        float xv = __bfloat162float(input[(size_t)tok * H + b * 32 + lane]);
        float a = fabsf(xv);
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
        float d = a / 127.0f;                                  // faithful to llama Q8_1:
        int qi = (a == 0.0f) ? 0 : (int)roundf(xv / d);        // roundf(xi/d), not rn(xi*inv)
        s_xq8[b * 32 + lane] = (signed char)qi;
        int sm = qi;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, m);
        if (lane == 0) { s_xd[b] = d; s_xs[b] = d * (float)sm; }
    }
    __syncthreads();

    const int f = blockIdx.y * WPB + warpId;
    if (f >= F) return;
    const int nsuper = H >> 8;   // super-blocks of 256
    const unsigned char* gbase = gate_q + ((size_t)e * F + f) * nsuper * 144;
    const unsigned char* ubase = up_q   + ((size_t)e * F + f) * nsuper * 144;

    float acc_g = 0.f, acc_u = 0.f;
    for (int sb = lane; sb < nsb; sb += 32) {
        const int super = sb >> 3, sib = sb & 7;
        const int* aint = reinterpret_cast<const int*>(s_xq8 + (sb << 5));
        const float xd = s_xd[sb], xs = s_xs[sb];
        const int boff = (sib >> 1) * 32;     // quant byte group within super-block
        const bool hi = sib & 1;
        // gate
        {
            const unsigned char* blk = gbase + (size_t)super * 144;
            float d = q4kf_h2f(blk), dmin = q4kf_h2f(blk + 2);
            int scd, scm; q4kf_scale_min(sib, blk + 4, &scd, &scm);
            const int* q = reinterpret_cast<const int*>(blk + 16 + boff);
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int w = hi ? ((q[k] >> 4) & 0x0F0F0F0F) : (q[k] & 0x0F0F0F0F);
                sumi = __dp4a(w, aint[k], sumi);
            }
            acc_g += d * (float)scd * xd * (float)sumi - dmin * (float)scm * xs;
        }
        // up
        {
            const unsigned char* blk = ubase + (size_t)super * 144;
            float d = q4kf_h2f(blk), dmin = q4kf_h2f(blk + 2);
            int scd, scm; q4kf_scale_min(sib, blk + 4, &scd, &scm);
            const int* q = reinterpret_cast<const int*>(blk + 16 + boff);
            int sumi = 0;
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                int w = hi ? ((q[k] >> 4) & 0x0F0F0F0F) : (q[k] & 0x0F0F0F0F);
                sumi = __dp4a(w, aint[k], sumi);
            }
            acc_u += d * (float)scd * xd * (float)sumi - dmin * (float)scm * xs;
        }
    }
    float g = q4kf_wsum(acc_g), u = q4kf_wsum(acc_u);
    if (lane == 0) h_scratch[(size_t)ts * F + f] = q4kf_silu(g) * u;
    if (pdl) si_pdl_lc();   // PDL: signal quant_h/down after h_scratch is written
}

// down: out[tok,hh] = sum_j weight_j * <h[tok,j], down[e_j, hh]>.
// one warp per (token, hh); loops over top_k experts internally and writes once.
// grid=(num_tokens, hidden/WPB), block=WPB*32. smem: WPB*256 (s_deq per warp).
__global__ void down_q6k_kernel(
    const unsigned char* __restrict__ down_q, const int* __restrict__ expert_ids,
    const float* __restrict__ expert_weights, const float* __restrict__ h_scratch,
    __nv_bfloat16* __restrict__ output, int H, int F, int top_k, int down_type
) {
    const int token = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int hh = blockIdx.y * WPB + (threadIdx.x / 32);
    if (hh >= H) return;
    const int nblk = F / 256;
    const int dbb = q_block_bytes(down_type);

    float acc = 0.f;   // sum_j w_j * <h_j, down[e_j, hh]> ; fold w into the per-lane partials
    for (int j = 0; j < top_k; j++) {
        const int ts = token * top_k + j;
        const int e = expert_ids[ts];
        const float w = expert_weights[ts];
        const unsigned char* dbase = down_q + ((size_t)e * H + hh) * nblk * dbb;
        const float* hbase = h_scratch + (size_t)ts * F;
        for (int blk = 0; blk < nblk; blk++)
            acc += w * q4kf_deq_dot(down_type, dbase + (size_t)blk * dbb, hbase + blk*256, lane);
    }
    acc = q4kf_wsum(acc);
    if (lane == 0) output[(size_t)token * H + hh] = __float2bfloat16(acc);
}

// split-K down: S warps cooperate per output row hh (each does a stride of the
// top_k*Fblocks work, then the S partials are summed in shared). At bs=1 the plain
// one-warp-per-row down has only H rows = H warps -> ~19% occupancy; this puts S*H
// warps in flight to hide latency. Accuracy-safe: same fp math, only the reduction
// order changes. ncu said decode is occupancy-bound — this is the measured lever.
__global__ void down_q6k_splitk_kernel(
    const unsigned char* __restrict__ down_q, const int* __restrict__ expert_ids,
    const float* __restrict__ expert_weights, const float* __restrict__ h_scratch,
    __nv_bfloat16* __restrict__ output, int H, int F, int top_k, int down_type
) {
    constexpr int S = 4, RPB = WPB / S;     // splits per row, rows per block
    __shared__ float s_part[RPB][S];
    const int token = blockIdx.x, lane = threadIdx.x & 31, warpId = threadIdx.x >> 5;
    const int hh_local = warpId / S, split = warpId % S;
    const int hh = blockIdx.y * RPB + hh_local;
    const int nblk = F >> 8, dbb = q_block_bytes(down_type);
    float acc = 0.f;
    si_pdl_sync();   // PDL: wait for gate_up's h_scratch writes before reading them
    if (hh < H) {
        const int total = top_k * nblk;
        for (int wi = split; wi < total; wi += S) {
            const int j = wi / nblk, blk = wi % nblk;
            const int ts = token * top_k + j, e = expert_ids[ts];
            const float w = expert_weights[ts];
            const unsigned char* drow = down_q + ((size_t)e * H + hh) * nblk * dbb;
            acc += w * q4kf_deq_dot(down_type, drow + (size_t)blk * dbb,
                                    h_scratch + (size_t)ts * F + blk * 256, lane);
        }
        acc = q4kf_wsum(acc);
        if (lane == 0) s_part[hh_local][split] = acc;
    }
    __syncthreads();
    if (hh < H && split == 0 && lane == 0) {
        float o = 0.f;
        #pragma unroll
        for (int s = 0; s < S; s++) o += s_part[hh_local][s];
        output[(size_t)token * H + hh] = __float2bfloat16(o);
    }
}

// ---- int8 dp4a MMVQ down (Q6_K) — SPARKINFER_DOWN_MMVQ=1 -----------------------
// The MoE down projection (Q6_K [E,H,F]) is the single biggest decode kernel and the
// only major GEMV still on the fp register-dequant path (gate/up + attention already
// run int8 MMVQ). This ports llama.cpp's vec_dot_q6_K_q8_1 faithfully: the down
// activation h (per token*expert, F floats) is quantized to Q8_1 once, then the Q6_K
// weight is dp4a'd against it. Lower register pressure than the fp dequant -> more
// warps resident on the occupancy-bound bs=1 down, plus integer dp4a math.
struct si_block_q8_1 { __half2 ds; signed char qs[32]; };   // 36 B / 32 values (llama layout)

// Quantize the down activation h (fp32, n_blocks*32 values) to Q8_1 in natural element
// order (block ib covers elements [ib*32, ib*32+32)). One 32-value block per warp.
__global__ void quant_h_q8_1_kernel(const float* __restrict__ h,
                                    si_block_q8_1* __restrict__ y, int n_blocks, int pdl) {
    if (pdl) si_pdl_sync();   // PDL: wait for gate_up's h_scratch writes
    const int warpsPB = blockDim.x >> 5;
    const int ib = blockIdx.x * warpsPB + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (ib >= n_blocks) return;
    float xv = h[(size_t)ib * 32 + lane], a = fabsf(xv);
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
    float d = a / 127.0f;
    int qi = (a == 0.0f) ? 0 : (int)roundf(xv / d);
    y[ib].qs[lane] = (signed char)qi;
    int s = qi;
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
    if (lane == 0) y[ib].ds = __floats2half2_rn(d, d * (float)s);
    if (pdl) si_pdl_lc();   // PDL: signal down after Q8_1(h) is ready
}

// Faithful llama.cpp vec_dot_q6_K_q8_1 for one 256-superblock at quant-index iqs (0..31).
// bq6 -> ggml block_q6_K (ql[128], qh[64], int8 scales[16], fp16 d); bq8 -> that
// superblock's 8 Q8_1 activation blocks.
// Unaligned 32-bit load from a >=2-byte-aligned byte array (Q6_K blocks are 210 B,
// only even-aligned), faithful to llama.cpp's get_int_b2.
__device__ __forceinline__ int si_get_int_b2(const void* x, int i32) {
    const unsigned short* x16 = reinterpret_cast<const unsigned short*>(x);
    return (int)x16[2 * i32] | ((int)x16[2 * i32 + 1] << 16);
}

__device__ __forceinline__ float si_vec_dot_q6_K(const unsigned char* __restrict__ bq6,
                                                 const si_block_q8_1* __restrict__ bq8, int iqs) {
    const signed char* scales = reinterpret_cast<const signed char*>(bq6 + 192);
    const float d = q4kf_h2f(bq6 + 208);
    const int bq8_offset   = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift     = 2 * ((iqs % 16) / 8);
    const int vl = si_get_int_b2(bq6, iqs);                                  // ql[128]
    const int vh = si_get_int_b2(bq6 + 128, 8 * (iqs / 16) + (iqs % 8)) >> vh_shift;  // qh[64]
    const signed char* sc = scales + scale_offset;
    float sumf = 0.f;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const si_block_q8_1* b8 = bq8 + bq8_offset + 2 * i;
        const int u = reinterpret_cast<const int*>(b8->qs)[iqs % 8];
        const float d8 = __low2float(b8->ds);
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi  = __vsubss4((vil | vih), 0x20202020);   // (vil|vih) - 32
        sumf += d8 * (__dp4a(vi, u, 0) * (int)sc[4 * i]);
    }
    return d * sumf;
}

// down (MMVQ): out[token,hh] = sum_j w_j * <h_j, down[e_j,hh]>. one warp per (token,hh),
// loops top_k experts and dp4a's Q6_K weights against the pre-quantized Q8_1 activation.
__global__ void down_q6k_mmvq_kernel(
    const unsigned char* __restrict__ down_q, const int* __restrict__ expert_ids,
    const float* __restrict__ expert_weights, const si_block_q8_1* __restrict__ hq8,
    __nv_bfloat16* __restrict__ output, int H, int F, int top_k, int pdl
) {
    if (pdl) si_pdl_sync();
    const int token = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int hh = blockIdx.y * WPB + (threadIdx.x >> 5);
    if (hh >= H) return;
    const int nblk = F >> 8;        // 256-superblocks per row
    const int q8pb = F >> 5;        // Q8_1 blocks per expert activation row
    float acc = 0.f;
    for (int j = 0; j < top_k; j++) {
        const int ts = token * top_k + j;
        const int e = expert_ids[ts];
        const float w = expert_weights[ts];
        const unsigned char* drow = down_q + ((size_t)e * H + hh) * nblk * 210;
        const si_block_q8_1* h8 = hq8 + (size_t)ts * q8pb;
        float t = 0.f;
        for (int kbx = 0; kbx < nblk; kbx++)
            t += si_vec_dot_q6_K(drow + (size_t)kbx * 210, h8 + (size_t)kbx * 8, lane);
        acc += w * t;
    }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) output[(size_t)token * H + hh] = __float2bfloat16(acc);
}

// ===== faithful llama Q4_K mmvq for the gate/up experts (4 warps/row; bundle with the
// Q6_K-mmvq V+LM-head win). Quantize hn once -> block_q8_1, one block per (ts,f). =====
struct si_block_q4_K { __half2 dm; unsigned char scales[12]; unsigned char qs[128]; };  // 144 B
__global__ void si_quant_bf16_q8_1(const __nv_bfloat16* __restrict__ x, si_block_q8_1* __restrict__ y, int K) {
    const int warpsPB = blockDim.x >> 5, ib = blockIdx.x * warpsPB + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (ib >= (K >> 5)) return;
    float xv = __bfloat162float(x[ib * 32 + lane]), a = fabsf(xv);
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
    float d = a / 127.0f;
    int qi = (a == 0.0f) ? 0 : (int)roundf(xv / d);
    y[ib].qs[lane] = (signed char)qi;
    int s = qi;
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
    if (lane == 0) y[ib].ds = __floats2half2_rn(d, d * (float)s);
}
// Q8_0 weight block (34 B / 32 values) dotted against a Q8_1 activation block — the
// faithful llama.cpp vec_dot_q8_0_q8_1 identity (no sum term; both sides are plain int8).
__device__ __forceinline__ int si_ld4(const unsigned char* p) {
    int v;
    memcpy(&v, p, sizeof(v));
    return v;
}
__device__ __forceinline__ float si_vec_dot_q8_0(const unsigned char* __restrict__ wblk,
                                                 const si_block_q8_1* __restrict__ ablk) {
    float d_w = q4kf_h2f(wblk);
    float d_a = __low2float(ablk->ds);
    const unsigned char* qw = wblk + 2;
    const unsigned char* qa = reinterpret_cast<const unsigned char*>(ablk->qs);
    int sumi = 0;
    #pragma unroll
    for (int k = 0; k < 8; k++) sumi = __dp4a(si_ld4(qw + k * 4), si_ld4(qa + k * 4), sumi);
    return d_w * d_a * (float)sumi;
}

template <int R>
__device__ __forceinline__ void si_vec_dot_q8_0_rows(
        const unsigned char* __restrict__ wblk,
        const si_block_q8_1* __restrict__ ablk, int astride, float (&out)[R]) {
    const float d_w = q4kf_h2f(wblk);
    int wv[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) wv[k] = si_ld4(wblk + 2 + k * 4);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const si_block_q8_1* a = ablk + (size_t)r * astride;
        const float d_a = __low2float(a->ds);
        const unsigned char* qa = reinterpret_cast<const unsigned char*>(a->qs);
        int sumi = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) sumi = __dp4a(wv[k], si_ld4(qa + k * 4), sumi);
        out[r] += d_w * d_a * (float)sumi;
    }
}

__device__ __forceinline__ float si_vec_dot_q4_K(const si_block_q4_K* bq4, const si_block_q8_1* bq8_1, int iqs) {
    int v[2], u[4]; float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* q4 = (const int*)(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0]; v[1] = q4[4];
    const unsigned short* scales = (const unsigned short*)bq4->scales;
    unsigned short aux[2]; const int j = bq8_offset / 2;
    if (j < 2) { aux[0] = scales[j] & 0x3f3f; aux[1] = scales[j + 2] & 0x3f3f; }
    else { aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
           aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j]     & 0xc0c0) >> 2); }
    const unsigned char* sc = (const unsigned char*)aux; const unsigned char* m = sc + 2;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const si_block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i] = q8[0]; u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.0f, sumf_m = 0.0f;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F, v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    float2 dm4f = __half22float2(bq4->dm);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

// Q5_K int8 dp4a vec-dot for the MoE down. Q5_K is Q4_K plus one high bit per quant (qh[32]), so
// this reuses the faithful Q4_K vec_dot structure (same qs / scales / activation indexing, same
// iqs = 2*L convention as si_vec_dot_q4_K) and only widens each 4-bit weight to 5 bits. The qh bit
// for a nibble is derived from the fp reference (q4kf_deq_dot, t==13): for a qs byte b the low
// nibble takes qh bit 2*(b/32) and the high nibble bit 2*(b/32)+1; across the four bytes v[0]/v[1]
// cover, b/32 == bq8_offset/2, so nibble half i uses qh bit (bq8_offset + i). Result matches the
// fp Q5_K down up to the int8 activation rounding (validated by self-consistency).
struct si_block_q5_K { __half2 dm; unsigned char scales[12]; unsigned char qh[32]; unsigned char qs[128]; };  // 176 B
__device__ __forceinline__ float si_vec_dot_q5_K(const si_block_q5_K* bq5, const si_block_q8_1* bq8_1, int iqs) {
    int v[2], u[4]; float d8[2];
    const int L = iqs >> 1;                              // 0..15, same position index as the Q4_K path
    const int bq8_offset = 2 * (L / 4);
    const int* q4 = (const int*)(bq5->qs + 16 * bq8_offset + 4 * (L % 4));
    v[0] = q4[0]; v[1] = q4[4];
    const int* qhp = (const int*)(bq5->qh + 4 * (L % 4));  // qh ints aligned with q4[0] and q4[4]
    const int qh0 = qhp[0], qh1 = qhp[4];
    const unsigned short* scales = (const unsigned short*)bq5->scales;   // 6-bit scales/mins, as Q4_K
    unsigned short aux[2]; const int j = bq8_offset / 2;
    if (j < 2) { aux[0] = scales[j] & 0x3f3f; aux[1] = scales[j + 2] & 0x3f3f; }
    else { aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
           aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j]     & 0xc0c0) >> 2); }
    const unsigned char* sc = (const unsigned char*)aux; const unsigned char* m = sc + 2;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const si_block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*)bq8i->qs + (L % 4);
        u[2 * i] = q8[0]; u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.f, sumf_m = 0.f;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int hs = bq8_offset + i;                    // qh bit for this nibble half (low=+0, high=+1)
        const int v0i = ((v[0] >> (4 * i)) & 0x0F0F0F0F) | (((qh0 >> hs) & 0x01010101) << 4);
        const int v1i = ((v[1] >> (4 * i)) & 0x0F0F0F0F) | (((qh1 >> hs) & 0x01010101) << 4);
        const int dot1 = __dp4a(v0i, u[2 * i], __dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i], __dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    float2 dm5f = __half22float2(bq5->dm);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}
// ---- Q3_A: sparkinfer's internal 3.5 bits/weight super-block ------------------------------
// Q4_K with the 4-bit quant plane replaced by a 3-bit one, keeping Q4_K's asymmetric per-32 scale
// AND min and Q4_K's exact 6-bit scale/min packing -- 112 B per 256 weights against Q4_K's 144 B,
// so a converted matrix is read 22.2% smaller on every decode token. Produced only by
// launch_ffn_requant_q3a at model load; never read from or written to a GGUF, so the layout is
// picked to keep this vec-dot as cheap as si_vec_dot_q4_K rather than to match any ggml type.
//
// Same iqs = 2*L convention as si_vec_dot_q4_K (L = 0..15): j = L/4 selects the sub-block pair
// {2j, 2j+1}, m = L%4 selects positions 8m..8m+7. qs holds the low 2 bits as one int per (j,m) at
// byte 16*j + 4*m, whose byte lane b = p%4 packs four 2-bit fields f = (s%2) | ((p%8)/4 << 1);
// qh[p] holds the high bit of every sub-block at position p in bit s, so one int load covers four
// positions for all eight sub-blocks. Both planes extract with one shift + one mask per four
// weights, exactly like Q4_K's nibble plane.
struct si_block_q3_A { __half2 dm; unsigned char scales[12]; unsigned char qs[64]; unsigned char qh[32]; };  // 112 B
__device__ __forceinline__ float si_vec_dot_q3_A(const si_block_q3_A* bq3, const si_block_q8_1* bq8_1, int iqs) {
    const int L = iqs >> 1;
    const int j = L >> 2, m = L & 3;
    const int vl  = *(const int*)(bq3->qs + 16 * j + 4 * m);
    const int hlo = *(const int*)(bq3->qh + 8 * m);
    const int hhi = *(const int*)(bq3->qh + 8 * m + 4);
    // 6-bit scales/mins: identical packing and identical unpack to si_vec_dot_q4_K, with
    // bq8_offset == 2*j so the aux index j is the same one that function uses.
    const unsigned short* scales = (const unsigned short*)bq3->scales;
    unsigned short aux[2];
    if (j < 2) { aux[0] = scales[j] & 0x3f3f; aux[1] = scales[j + 2] & 0x3f3f; }
    else { aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
           aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j]     & 0xc0c0) >> 2); }
    const unsigned char* sc = (const unsigned char*)aux; const unsigned char* mn = sc + 2;
    float sumf_d = 0.f, sumf_m = 0.f;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int s = 2 * j + i;                             // sub-block == q8_1 block index
        const si_block_q8_1* bq8i = bq8_1 + s;
        const float d8 = __low2float(bq8i->ds);
        const int* q8 = (const int*)bq8i->qs;
        const int u0 = q8[2 * m], u1 = q8[2 * m + 1];
        const int v0 = ((vl >> (2 * i))     & 0x03030303) | (((hlo >> s) & 0x01010101) << 2);
        const int v1 = ((vl >> (2 * i + 4)) & 0x03030303) | (((hhi >> s) & 0x01010101) << 2);
        const int dot1 = __dp4a(v0, u0, __dp4a(v1, u1, 0));
        const int dot2 = __dp4a(0x01010101, u0, __dp4a(0x01010101, u1, 0));
        sumf_d += d8 * (dot1 * sc[i]);
        sumf_m += d8 * (dot2 * mn[i]);
    }
    float2 dm3f = __half22float2(bq3->dm);
    return dm3f.x * sumf_d - dm3f.y * sumf_m;
}

// One CTA per output row, weights read as Q3_A. Same tiling, same accumulation order and same
// 4-warp reduction as gate_up_mmvq2_kernel below -- only the block format differs.
__global__ void gate_up_q3a_kernel(
    const si_block_q8_1* __restrict__ vy, const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q, const int* __restrict__ expert_ids,
    float* __restrict__ h_scratch, int H, int F, int top_k, int pdl
) {
    constexpr int NW = 4, WS = 32, vdr = 2, qi = 32;
    const int row = blockIdx.x, ts = row / F, f = row % F, tok = ts / top_k;
    const int e = expert_ids[ts];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
    const si_block_q8_1* vrow = vy + (size_t)tok * (H >> 5);
    const si_block_q3_A* g_row = (const si_block_q3_A*)(gate_q + ((size_t)e * F + f) * (H >> 8) * 112);
    const si_block_q3_A* u_row = (const si_block_q3_A*)(up_q   + ((size_t)e * F + f) * (H >> 8) * 112);
    const int blocks_per_row = H >> 8, blocks_per_iter = vdr * NW * WS / qi;   // = 8
    float tg = 0.f, tu = 0.f;
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * 8, kqs = vdr * (tid % (qi / vdr));
        tg += si_vec_dot_q3_A(g_row + kbx, vrow + kby, kqs);
        tu += si_vec_dot_q3_A(u_row + kbx, vrow + kby, kqs);
    }
    __shared__ float sg[NW - 1][WS], su[NW - 1][WS];
    if (warp > 0) { sg[warp - 1][lane] = tg; su[warp - 1][lane] = tu; }
    __syncthreads();
    if (warp > 0) return;
    #pragma unroll
    for (int l = 0; l < NW - 1; l++) { tg += sg[l][lane]; tu += su[l][lane]; }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) { tg += __shfl_xor_sync(0xffffffff, tg, m); tu += __shfl_xor_sync(0xffffffff, tu, m); }
    if (lane == 0) h_scratch[(size_t)ts * F + f] = q4kf_silu(tg) * tu;
    if (pdl) si_pdl_lc();
}

template <int H, int F>
__global__ void gate_up_q3a_muse_kernel(
    const si_block_q8_1* __restrict__ vy, const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q, const int* __restrict__ expert_ids,
    float* __restrict__ h_scratch, int pdl
) {
    constexpr int NW = 4, NB = H >> 8;
    const int f = blockIdx.x;
    const int e = expert_ids[0];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
    const int kbx0 = tid >> 4, kqs = 2 * (tid & 15);
    const si_block_q3_A* g_row = reinterpret_cast<const si_block_q3_A*>(gate_q + ((size_t)e * F + f) * NB * sizeof(si_block_q3_A));
    const si_block_q3_A* u_row = reinterpret_cast<const si_block_q3_A*>(up_q + ((size_t)e * F + f) * NB * sizeof(si_block_q3_A));
    float tg = 0.f, tu = 0.f;
    #pragma unroll
    for (int kbx = kbx0; kbx < NB; kbx += 8) {
        tg += si_vec_dot_q3_A(g_row + kbx, vy + (size_t)kbx * 8, kqs);
        tu += si_vec_dot_q3_A(u_row + kbx, vy + (size_t)kbx * 8, kqs);
    }
    __shared__ float sg[NW - 1][32], su[NW - 1][32];
    if (warp > 0) { sg[warp - 1][lane] = tg; su[warp - 1][lane] = tu; }
    __syncthreads();
    if (warp > 0) return;
    #pragma unroll
    for (int w = 0; w < NW - 1; ++w) { tg += sg[w][lane]; tu += su[w][lane]; }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) { tg += __shfl_xor_sync(0xffffffff, tg, m); tu += __shfl_xor_sync(0xffffffff, tu, m); }
    if (lane == 0) h_scratch[f] = q4kf_silu(tg) * tu;
    if (pdl) si_pdl_lc();
}

__global__ void gate_up_mmvq2_kernel(
    const si_block_q8_1* __restrict__ vy, const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q, const int* __restrict__ expert_ids,
    float* __restrict__ h_scratch, int H, int F, int top_k, int pdl
) {
    constexpr int NW = 4, WS = 32, vdr = 2, qi = 32;
    const int row = blockIdx.x, ts = row / F, f = row % F, tok = ts / top_k;
    const int e = expert_ids[ts];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
    const si_block_q8_1* vrow = vy + (size_t)tok * (H >> 5);
    const si_block_q4_K* g_row = (const si_block_q4_K*)(gate_q + ((size_t)e * F + f) * (H >> 8) * 144);
    const si_block_q4_K* u_row = (const si_block_q4_K*)(up_q   + ((size_t)e * F + f) * (H >> 8) * 144);
    const int blocks_per_row = H >> 8, blocks_per_iter = vdr * NW * WS / qi;   // = 8
    float tg = 0.f, tu = 0.f;
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * 8, kqs = vdr * (tid % (qi / vdr));
        tg += si_vec_dot_q4_K(g_row + kbx, vrow + kby, kqs);
        tu += si_vec_dot_q4_K(u_row + kbx, vrow + kby, kqs);
    }
    __shared__ float sg[NW - 1][WS], su[NW - 1][WS];
    if (warp > 0) { sg[warp - 1][lane] = tg; su[warp - 1][lane] = tu; }
    __syncthreads();
    if (warp > 0) return;
    #pragma unroll
    for (int l = 0; l < NW - 1; l++) { tg += sg[l][lane]; tu += su[l][lane]; }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) { tg += __shfl_xor_sync(0xffffffff, tg, m); tu += __shfl_xor_sync(0xffffffff, tu, m); }
    if (lane == 0) h_scratch[(size_t)ts * F + f] = q4kf_silu(tg) * tu;
    if (pdl) si_pdl_lc();
}

// ---- Qwen3.6 shared expert (Q8_0 gate/up/down, F=512) -------------------------
// Reuses the FNQ Q8_1(hn) buffer for gate/up dp4a (same trick as routed MoE mmvq),
// then overwrites that scratch with Q8_1(h) for the down dp4a. One 4-warp block/row
// for gate/up (64 Q8_0 blocks/row); one warp/row for down (2048 rows, K=512).
template <int H, int F>
__global__ void shared_gate_up_q8_mmvq_kernel(
    const si_block_q8_1* __restrict__ vy, const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q, const float* __restrict__ dw,
    float* __restrict__ h_scratch) {
    constexpr int NW = 4, WS = 32;
    const int f = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
    const int nblk = H >> 5;
    const unsigned char* gbase = gate_q + (size_t)f * nblk * 34;
    const unsigned char* ubase = up_q   + (size_t)f * nblk * 34;
    float tg = 0.f, tu = 0.f;
    for (int b = tid; b < nblk; b += NW * WS) {
        tg += si_vec_dot_q8_0(gbase + (size_t)b * 34, vy + b);
        tu += si_vec_dot_q8_0(ubase + (size_t)b * 34, vy + b);
    }
    __shared__ float sg[NW - 1][WS], su[NW - 1][WS];
    if (warp > 0) { sg[warp - 1][lane] = tg; su[warp - 1][lane] = tu; }
    __syncthreads();
    if (warp > 0) return;
    #pragma unroll
    for (int l = 0; l < NW - 1; l++) { tg += sg[l][lane]; tu += su[l][lane]; }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) { tg += __shfl_xor_sync(0xffffffff, tg, m); tu += __shfl_xor_sync(0xffffffff, tu, m); }
    if (lane == 0) {
        float w = dw ? __ldg(dw) : 1.f;
        h_scratch[f] = w * q4kf_silu(tg) * tu;
    }
}

template <int H, int F, bool ACCUM>
__global__ void shared_down_q8_mmvq_kernel(
    const si_block_q8_1* __restrict__ hq8, const unsigned char* __restrict__ down_q,
    __nv_bfloat16* __restrict__ out) {
    const int h = blockIdx.x * WPB + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (h >= H) return;
    const int nblk = F >> 5;
    const unsigned char* dbase = down_q + (size_t)h * nblk * 34;
    float acc = 0.f;
    for (int b = lane; b < nblk; b += 32)
        acc += si_vec_dot_q8_0(dbase + (size_t)b * 34, hq8 + b);
    acc = q4kf_wsum(acc);
    if (lane == 0) {
        if constexpr (ACCUM) acc += __bfloat162float(out[h]);
        out[h] = __float2bfloat16(acc);
    }
}

// Warps per CTA sized to the work that exists: one row is H/32 Q8_0 blocks and the loop strides
// by NW*WS, so with H=2048 (64 blocks) a 4-warp CTA leaves threads 64..127 with zero iterations.
// Those dead warps still occupy the CTA and still push a 0.0f into the shared-memory reduction.
// Dropping them is numerically inert (the reduction just stops adding those zeros) and doubles the
// CTAs that fit per SM. This kernel is hidden behind the routed MoE on stream_k in the DECODE path,
// but the DFlash verify issues it on the main stream, where it is squarely on the critical path.
template <int H, int WS>
__device__ __host__ constexpr int si_shexp_nw() {
    return (H / 32) >= 4 * WS ? 4 : ((H / 32) + WS - 1) / WS;
}

template <int H, int F, int R>
__global__ void shared_gate_up_q8_mmvq_rows_kernel(
    const si_block_q8_1* __restrict__ vy, const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q, const float* __restrict__ dw,
    float* __restrict__ h_scratch) {
    constexpr int WS = 32;
    constexpr int NW = si_shexp_nw<H, WS>();
    const int f = blockIdx.x, lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5, tid = threadIdx.x;
    const int nblk = H >> 5;
    const unsigned char* gbase = gate_q + (size_t)f * nblk * 34;
    const unsigned char* ubase = up_q + (size_t)f * nblk * 34;
    float tg[R], tu[R];
#pragma unroll
    for (int r = 0; r < R; ++r) { tg[r] = 0.f; tu[r] = 0.f; }
    for (int b = tid; b < nblk; b += NW * WS) {
        si_vec_dot_q8_0_rows<R>(gbase + (size_t)b * 34, vy + b, nblk, tg);
        si_vec_dot_q8_0_rows<R>(ubase + (size_t)b * 34, vy + b, nblk, tu);
    }
    __shared__ float sg[R][NW - 1][WS], su[R][NW - 1][WS];
    if (warp > 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) { sg[r][warp - 1][lane] = tg[r]; su[r][warp - 1][lane] = tu[r]; }
    }
    __syncthreads();
    if (warp > 0) return;
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int l = 0; l < NW - 1; ++l) { tg[r] += sg[r][l][lane]; tu[r] += su[r][l][lane]; }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            tg[r] += __shfl_xor_sync(0xffffffffu, tg[r], m);
            tu[r] += __shfl_xor_sync(0xffffffffu, tu[r], m);
        }
        if (lane == 0) h_scratch[(size_t)r * F + f] = __ldg(dw + r) * q4kf_silu(tg[r]) * tu[r];
    }
}

// F/32 Q8_0 blocks per output row (16 for the Qwen3.6 shared expert) against a 32-lane stride left
// lanes 16..31 with nothing to do, and the 32-lane reduction then folded their zeros in. Give each
// half-warp its own output row instead: the surviving reduction is the same 16-lane tree the low
// half already ran, minus a `+ 0.0f`, so every output is unchanged while the CTA covers twice the
// rows. Only worthwhile because the DFlash verify puts this on the critical path — in decode it
// hides on stream_k behind the routed MoE.
template <int H, int F, int R>
__global__ void shared_down_q8_mmvq_rows_kernel(
    const si_block_q8_1* __restrict__ hq8, const unsigned char* __restrict__ down_q,
    __nv_bfloat16* __restrict__ out) {
    constexpr int nblk = F >> 5;
    constexpr int HALF = (nblk <= 16) ? 1 : 0;          // two rows per warp when 16 lanes suffice
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int sub = HALF ? (lane >> 4) : 0;             // which half-warp
    const int llane = HALF ? (lane & 15) : lane;
    const int rows_per_warp = HALF ? 2 : 1;
    const int h = (blockIdx.x * WPB + warp) * rows_per_warp + sub;
    if (h >= H) return;
    const unsigned char* dbase = down_q + (size_t)h * nblk * 34;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int b = llane; b < nblk; b += (HALF ? 16 : 32))
        si_vec_dot_q8_0_rows<R>(dbase + (size_t)b * 34, hq8 + b, nblk, acc);
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = (HALF ? 8 : 16); m > 0; m >>= 1)
            acc[r] += __shfl_xor_sync(0xffffffff, acc[r], m);
        if (llane == 0) out[(size_t)r * H + h] = __float2bfloat16(acc[r]);
    }
}

template <int H, int F, int TOPK>
__global__ void gate_up_mmvq2_qwen_kernel(
    const si_block_q8_1* __restrict__ vy, const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q, const int* __restrict__ expert_ids,
    float* __restrict__ h_scratch, int pdl
) {
    constexpr int NW = 4, WS = 32;
    const int row = blockIdx.x, ts = row / F, f = row - ts * F, tok = ts / TOPK;
    const int e = expert_ids[ts];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
    const int kbx0 = tid >> 4;
    const int kqs = 2 * (tid & 15);
    const si_block_q8_1* vrow = vy + (size_t)tok * (H >> 5);
    const si_block_q4_K* g_row = (const si_block_q4_K*)(gate_q + ((size_t)e * F + f) * (H >> 8) * 144);
    const si_block_q4_K* u_row = (const si_block_q4_K*)(up_q   + ((size_t)e * F + f) * (H >> 8) * 144);
    constexpr int NB = H >> 8;
    float tg = 0.f, tu = 0.f;
    for (int kbx = kbx0; kbx < NB; kbx += 8) {
        tg += si_vec_dot_q4_K(g_row + kbx, vrow + (size_t)kbx * 8, kqs);
        tu += si_vec_dot_q4_K(u_row + kbx, vrow + (size_t)kbx * 8, kqs);
    }
    __shared__ float sg[NW - 1][WS], su[NW - 1][WS];
    if (warp > 0) { sg[warp - 1][lane] = tg; su[warp - 1][lane] = tu; }
    __syncthreads();
    if (warp > 0) return;
    #pragma unroll
    for (int l = 0; l < NW - 1; l++) { tg += sg[l][lane]; tu += su[l][lane]; }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) { tg += __shfl_xor_sync(0xffffffff, tg, m); tu += __shfl_xor_sync(0xffffffff, tu, m); }
    if (lane == 0) h_scratch[(size_t)ts * F + f] = q4kf_silu(tg) * tu;
    if (pdl) si_pdl_lc();
}

// Row-batched dense gate/up: T token rows against ONE weight stream.
//
// gate_up_mmvq2_qwen_kernel is launched with grid = num_tokens * F, i.e. one CTA per (token,
// ffn row), and every CTA loads the gate and up rows it needs itself. At num_tokens == 1 that is
// exactly right. At the speculative verify's num_tokens == N it reads the whole FFN N times, and
// the FFN is the largest weight in the model -- so the verify pays N x the DRAM traffic for a
// pass whose weights do not depend on the token at all. Decode here runs at ~85% of this card's
// bandwidth, so that traffic is the cost.
//
// One CTA per ffn row, T tokens folded inside: the weight super-blocks are read once and each
// token's activation blocks are dotted against them in the SAME order the one-token kernel uses
// (same lane -> kbx/kqs mapping, same four-warp shared reduce, same silu(tg)*tu tail). Every row
// therefore produces the bits the per-token kernel produced for it -- the verify's losslessness
// argument (N one-row calls are bit-identical to N AR steps) survives unchanged.
template <int H, int F, int TOPK, int T>
__global__ void gate_up_mmvq2_qwen_rows_kernel(
    const si_block_q8_1* __restrict__ vy, const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q, const int* __restrict__ expert_ids,
    float* __restrict__ h_scratch, int pdl
) {
    constexpr int NW = 4, WS = 32;
    const int f = blockIdx.x;                    // ffn row, shared by every token
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
    const int kbx0 = tid >> 4;
    const int kqs = 2 * (tid & 15);
    constexpr int NB = H >> 8;
    // TOPK == 1 on every shape this arm serves, so token t's slot is t and its expert is
    // expert_ids[t]; the weight rows below are the same for all of them when the expert matches,
    // which is the case this kernel exists for (dense FFN, one expert).
    const int e0 = expert_ids[0];
    const si_block_q4_K* g_row = (const si_block_q4_K*)(gate_q + ((size_t)e0 * F + f) * (H >> 8) * 144);
    const si_block_q4_K* u_row = (const si_block_q4_K*)(up_q   + ((size_t)e0 * F + f) * (H >> 8) * 144);
    float tg[T], tu[T];
#pragma unroll
    for (int t = 0; t < T; t++) { tg[t] = 0.f; tu[t] = 0.f; }
    // The weight block stays in global and is read through the same __ldg path the one-token
    // kernel uses; what makes it cheap for token t>0 is that the CTA touched those exact bytes
    // microseconds earlier, so they are in L1/L2 rather than DRAM. Copying the 144-byte
    // super-block into registers instead looks like reuse and is not: it spills to local memory
    // and measured 3.2x SLOWER than the per-token kernel it replaces.
    for (int kbx = kbx0; kbx < NB; kbx += 8) {
#pragma unroll
        for (int t = 0; t < T; t++) {
            const si_block_q8_1* vrow = vy + (size_t)t * (H >> 5);
            tg[t] += si_vec_dot_q4_K(g_row + kbx, vrow + (size_t)kbx * 8, kqs);
            tu[t] += si_vec_dot_q4_K(u_row + kbx, vrow + (size_t)kbx * 8, kqs);
        }
    }
    __shared__ float sg[T][NW - 1][WS], su[T][NW - 1][WS];
    if (warp > 0) {
#pragma unroll
        for (int t = 0; t < T; t++) { sg[t][warp - 1][lane] = tg[t]; su[t][warp - 1][lane] = tu[t]; }
    }
    __syncthreads();
    if (warp > 0) return;
#pragma unroll
    for (int t = 0; t < T; t++) {
        #pragma unroll
        for (int l = 0; l < NW - 1; l++) { tg[t] += sg[t][l][lane]; tu[t] += su[t][l][lane]; }
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
            tg[t] += __shfl_xor_sync(0xffffffff, tg[t], m);
            tu[t] += __shfl_xor_sync(0xffffffff, tu[t], m);
        }
        if (lane == 0) h_scratch[(size_t)t * F + f] = q4kf_silu(tg[t]) * tu[t];
    }
    if (pdl) si_pdl_lc();
}

template <int H, int F, int TOPK>
__global__ void gate_up_mmvq2_pack2_qwen_kernel(
    const si_block_q8_1* __restrict__ vy, const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q, const int* __restrict__ expert_ids,
    float* __restrict__ h_scratch, int n_rows, int pdl
) {
    constexpr int NW = 4, WS = 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int group = warp >> 2, group_warp = warp & 3;
    const int row = blockIdx.x * 2 + group;
    if (row >= n_rows) return;
    const int ts = row / F, f = row - ts * F, tok = ts / TOPK;
    const int e = expert_ids[ts];
    const int tid4 = group_warp * WS + lane;
    const int kbx0 = tid4 >> 4;
    const int kqs = 2 * (tid4 & 15);
    const si_block_q8_1* vrow = vy + (size_t)tok * (H >> 5);
    const si_block_q4_K* g_row = (const si_block_q4_K*)(gate_q + ((size_t)e * F + f) * (H >> 8) * 144);
    const si_block_q4_K* u_row = (const si_block_q4_K*)(up_q   + ((size_t)e * F + f) * (H >> 8) * 144);
    constexpr int NB = H >> 8;
    float tg = 0.f, tu = 0.f;
    for (int kbx = kbx0; kbx < NB; kbx += 8) {
        tg += si_vec_dot_q4_K(g_row + kbx, vrow + (size_t)kbx * 8, kqs);
        tu += si_vec_dot_q4_K(u_row + kbx, vrow + (size_t)kbx * 8, kqs);
    }
    __shared__ float sg[2][NW - 1][WS], su[2][NW - 1][WS];
    if (group_warp > 0) { sg[group][group_warp - 1][lane] = tg; su[group][group_warp - 1][lane] = tu; }
    __syncthreads();
    if (group_warp > 0) return;
    #pragma unroll
    for (int l = 0; l < NW - 1; l++) { tg += sg[group][l][lane]; tu += su[group][l][lane]; }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) { tg += __shfl_xor_sync(0xffffffff, tg, m); tu += __shfl_xor_sync(0xffffffff, tu, m); }
    if (lane == 0) h_scratch[(size_t)ts * F + f] = q4kf_silu(tg) * tu;
    if (pdl) si_pdl_lc();
}

// Same output as gate_up_mmvq2_pack2_qwen_kernel, one warp per element instead of four.
//
// pack2 splits each output across 4 warps because AR decode calls this with num_tokens=1, where
// top_k*ffn = 4096 elements is not enough to fill the GPU and the extra warps buy occupancy. The
// DFlash compact verify calls it with num_tokens=B, so the grid is already B times larger and the
// split stops buying anything -- every thread runs a single si_vec_dot_q4_K pair (NB = H>>8 = 8,
// stride 8, so the loop body executes once) and then pays a shared-memory round trip and a
// __syncthreads to reassemble it.
//
// This is bit-identical to pack2, not merely equivalent. There, lane l of warp 0 finishes with
//   p(kbx0) + p(kbx0+2) + p(kbx0+4) + p(kbx0+6),   kbx0 = l>>4
// accumulated in that order: warp w covers tid4 = w*32 + l, hence kbx = kbx0 + 2w and the same
// kqs = 2*(l&15) (32 is a multiple of 16), and the shared reduce folds warps 1,2,3 in ascending
// order. Walking that same kbx sequence inside one warp produces the same partials in the same
// order, so the butterfly below sees identical inputs. Only the thread mapping changes.
template <int H, int F, int TOPK, int WARPS>
__global__ void gate_up_mmvq2_warp_qwen_kernel(
    const si_block_q8_1* __restrict__ vy, const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q, const int* __restrict__ expert_ids,
    float* __restrict__ h_scratch, int n_rows, int pdl
) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * WARPS + (int)(threadIdx.x >> 5);
    if (row >= n_rows) { if (pdl) si_pdl_lc(); return; }
    const int ts = row / F, f = row - ts * F, tok = ts / TOPK;
    const int e = expert_ids[ts];
    const int kbx0 = lane >> 4;
    const int kqs = 2 * (lane & 15);
    const si_block_q8_1* vrow = vy + (size_t)tok * (H >> 5);
    const si_block_q4_K* g_row = (const si_block_q4_K*)(gate_q + ((size_t)e * F + f) * (H >> 8) * 144);
    const si_block_q4_K* u_row = (const si_block_q4_K*)(up_q   + ((size_t)e * F + f) * (H >> 8) * 144);
    constexpr int NB = H >> 8;
    float tg = 0.f, tu = 0.f;
    for (int kbx = kbx0; kbx < NB; kbx += 2) {
        tg += si_vec_dot_q4_K(g_row + kbx, vrow + (size_t)kbx * 8, kqs);
        tu += si_vec_dot_q4_K(u_row + kbx, vrow + (size_t)kbx * 8, kqs);
    }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) { tg += __shfl_xor_sync(0xffffffff, tg, m); tu += __shfl_xor_sync(0xffffffff, tu, m); }
    if (lane == 0) h_scratch[(size_t)ts * F + f] = q4kf_silu(tg) * tu;
    if (pdl) si_pdl_lc();
}

template <int H, int F, int TOPK>
__global__ void gate_up_mmvq2_pack4_qwen_kernel(
    const si_block_q8_1* __restrict__ vy, const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q, const int* __restrict__ expert_ids,
    float* __restrict__ h_scratch, int n_rows
) {
    si_pdl_lc();
    constexpr int NW = 4, WS = 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int group = warp >> 2, group_warp = warp & 3;
    const int row = blockIdx.x * 4 + group;
    if (row >= n_rows) return;
    const int ts = row / F, f = row - ts * F, tok = ts / TOPK;
    const int e = expert_ids[ts];
    const int tid4 = group_warp * WS + lane;
    const int kbx0 = tid4 >> 4;
    const int kqs = 2 * (tid4 & 15);
    const si_block_q8_1* vrow = vy + (size_t)tok * (H >> 5);
    const si_block_q4_K* g_row = (const si_block_q4_K*)(gate_q + ((size_t)e * F + f) * (H >> 8) * 144);
    const si_block_q4_K* u_row = (const si_block_q4_K*)(up_q   + ((size_t)e * F + f) * (H >> 8) * 144);
    constexpr int NB = H >> 8;
    float tg = 0.f, tu = 0.f;
    for (int kbx = kbx0; kbx < NB; kbx += 8) {
        tg += si_vec_dot_q4_K(g_row + kbx, vrow + (size_t)kbx * 8, kqs);
        tu += si_vec_dot_q4_K(u_row + kbx, vrow + (size_t)kbx * 8, kqs);
    }
    __shared__ float sg[4][NW - 1][WS], su[4][NW - 1][WS];
    if (group_warp > 0) { sg[group][group_warp - 1][lane] = tg; su[group][group_warp - 1][lane] = tu; }
    __syncthreads();
    if (group_warp > 0) return;
    #pragma unroll
    for (int l = 0; l < NW - 1; l++) { tg += sg[group][l][lane]; tu += su[group][l][lane]; }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) { tg += __shfl_xor_sync(0xffffffff, tg, m); tu += __shfl_xor_sync(0xffffffff, tu, m); }
    if (lane == 0) h_scratch[(size_t)ts * F + f] = q4kf_silu(tg) * tu;
}

// Dense hybrid gate/up (Qwythos): same pack2 math as gate_up_mmvq2_pack2_qwen_kernel but
// without expert_ids indirection (single expert 0 baked into the weight layout).
template <int H, int F>
__global__ void dense_gate_up_q4k_pack2_kernel(
    const si_block_q8_1* __restrict__ vy, const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q, float* __restrict__ h_scratch) {
    si_pdl_lc();
    constexpr int NW = 4, WS = 32, NB = H >> 8;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int group = warp >> 2, group_warp = warp & 3;
    const int f = blockIdx.x * 2 + group;
    if (f >= F) return;
    const int tid4 = group_warp * WS + lane;
    const int kbx0 = tid4 >> 4;
    const int kqs = 2 * (tid4 & 15);
    const si_block_q4_K* g_row = (const si_block_q4_K*)(gate_q + (size_t)f * NB * 144);
    const si_block_q4_K* u_row = (const si_block_q4_K*)(up_q   + (size_t)f * NB * 144);
    float tg = 0.f, tu = 0.f;
    for (int kbx = kbx0; kbx < NB; kbx += 8) {
        tg += si_vec_dot_q4_K(g_row + kbx, vy + (size_t)kbx * 8, kqs);
        tu += si_vec_dot_q4_K(u_row + kbx, vy + (size_t)kbx * 8, kqs);
    }
    __shared__ float sg[2][NW - 1][WS], su[2][NW - 1][WS];
    if (group_warp > 0) { sg[group][group_warp - 1][lane] = tg; su[group][group_warp - 1][lane] = tu; }
    __syncthreads();
    if (group_warp > 0) return;
    #pragma unroll
    for (int l = 0; l < NW - 1; l++) { tg += sg[group][l][lane]; tu += su[group][l][lane]; }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) { tg += __shfl_xor_sync(0xffffffff, tg, m); tu += __shfl_xor_sync(0xffffffff, tu, m); }
    if (lane == 0) h_scratch[f] = q4kf_silu(tg) * tu;
}

// int8 dp4a MMVQ down (Q4_K). The Q4_K-quantized down rows in Q4_K_M were the last MoE GEMV
// still on the fp register-dequant path (Q6_K down + gate/up + attention already run int8).
// Reuses the Q8_1-quantized activation and the faithful vec_dot_q4_K_q8_1, one warp per
// (token,hh) folding the top_k experts. Each 256-superblock has 16 vdr=2 positions; the warp
// strides those (nblk*16 items) so the dp4a math matches the gate/up path exactly.
__global__ void down_q4k_mmvq_kernel(
    const unsigned char* __restrict__ down_q, const int* __restrict__ expert_ids,
    const float* __restrict__ expert_weights, const si_block_q8_1* __restrict__ hq8,
    __nv_bfloat16* __restrict__ output, int H, int F, int top_k, int pdl
) {
    if (pdl) si_pdl_sync();
    const int token = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int hh = blockIdx.y * WPB + (threadIdx.x >> 5);
    if (hh >= H) return;
    const int nblk = F >> 8;        // 256-superblocks per row
    const int q8pb = F >> 5;        // Q8_1 blocks per expert activation row
    const int work = nblk * 16;     // vdr=2 positions per superblock
    float acc = 0.f;
    for (int j = 0; j < top_k; j++) {
        const int ts = token * top_k + j;
        const int e = expert_ids[ts];
        const float w = expert_weights[ts];
        const si_block_q4_K* drow = reinterpret_cast<const si_block_q4_K*>(
            down_q + ((size_t)e * H + hh) * nblk * 144);
        const si_block_q8_1* h8 = hq8 + (size_t)ts * q8pb;
        float t = 0.f;
        for (int wi = lane; wi < work; wi += 32) {
            const int kbx = wi >> 4, kqs = (wi & 15) << 1;
            t += si_vec_dot_q4_K(drow + kbx, h8 + (size_t)kbx * 8, kqs);
        }
        acc += w * t;
    }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) output[(size_t)token * H + hh] = __float2bfloat16(acc);
}

// ---- split-K int8 MMVQ down (Q6_K / Q4_K) -------------------------------------
// The default MMVQ down kernels above are one-warp-per-row: at bs=1 that's only H
// warps in flight (~19% occupancy on the 5090 — the same occupancy wall the fp
// down hit before split-K). The int8 dp4a math recovered most of the bandwidth,
// but the down GEMV stayed launch/occupancy-bound. This applies the proven split-K
// lever (down_q6k_splitk_kernel) to the MMVQ path: S warps cooperate per output
// row, each striding a slice of the flattened (top_k * superblock) work, then the
// S partials sum in shared. S*H warps in flight hides the bs=1 latency. The expert
// weight w is folded per (expert,block) item so a split spanning experts is exact;
// only the float reduction order changes vs the one-warp kernel — accuracy-safe.
template <int S>
__global__ void down_q6k_mmvq_splitk_kernel(
    const unsigned char* __restrict__ down_q, const int* __restrict__ expert_ids,
    const float* __restrict__ expert_weights, const si_block_q8_1* __restrict__ hq8,
    __nv_bfloat16* __restrict__ output, int H, int F, int top_k, int pdl
) {
    if (pdl) si_pdl_sync();
    constexpr int RPB = WPB / S;            // output rows per block
    __shared__ float s_part[RPB][S];
    const int token = blockIdx.x, lane = threadIdx.x & 31, warpId = threadIdx.x >> 5;
    const int hh_local = warpId / S, split = warpId % S;
    const int hh = blockIdx.y * RPB + hh_local;
    const int nblk = F >> 8;                 // 256-superblocks per row
    const int q8pb = F >> 5;                 // Q8_1 blocks per expert activation row
    float acc = 0.f;
    if (hh < H) {
        const int total = top_k * nblk;      // flattened (expert, superblock) work items
        for (int wi = split; wi < total; wi += S) {
            const int j = wi / nblk, kbx = wi % nblk;
            const int ts = token * top_k + j, e = expert_ids[ts];
            const float w = expert_weights[ts];
            const unsigned char* drow = down_q + ((size_t)e * H + hh) * nblk * 210;
            const si_block_q8_1* h8 = hq8 + (size_t)ts * q8pb;
            acc += w * si_vec_dot_q6_K(drow + (size_t)kbx * 210, h8 + (size_t)kbx * 8, lane);
        }
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
        if (lane == 0) s_part[hh_local][split] = acc;
    }
    __syncthreads();
    if (hh < H && split == 0 && lane == 0) {
        float o = 0.f;
        #pragma unroll
        for (int s = 0; s < S; s++) o += s_part[hh_local][s];
        output[(size_t)token * H + hh] = __float2bfloat16(o);
    }
}

template <int S>
__global__ void down_q4k_mmvq_splitk_kernel(
    const unsigned char* __restrict__ down_q, const int* __restrict__ expert_ids,
    const float* __restrict__ expert_weights, const si_block_q8_1* __restrict__ hq8,
    __nv_bfloat16* __restrict__ output, int H, int F, int top_k, int pdl
) {
    if (pdl) si_pdl_sync();
    constexpr int RPB = WPB / S;
    __shared__ float s_part[RPB][S];
    const int token = blockIdx.x, lane = threadIdx.x & 31, warpId = threadIdx.x >> 5;
    const int hh_local = warpId / S, split = warpId % S;
    const int hh = blockIdx.y * RPB + hh_local;
    const int nblk = F >> 8;
    const int q8pb = F >> 5;
    const int work = nblk * 16;              // vdr=2 positions per superblock
    float acc = 0.f;
    if (hh < H) {
        // flatten (expert, vdr-position) so S warps split the whole row evenly
        const int total = top_k * work;
        for (int wi = split * 32 + lane; wi < total; wi += S * 32) {
            const int j = wi / work, r = wi % work;
            const int kbx = r >> 4, kqs = (r & 15) << 1;
            const int ts = token * top_k + j, e = expert_ids[ts];
            const float w = expert_weights[ts];
            const si_block_q4_K* drow = reinterpret_cast<const si_block_q4_K*>(
                down_q + ((size_t)e * H + hh) * nblk * 144);
            const si_block_q8_1* h8 = hq8 + (size_t)ts * q8pb;
            acc += w * si_vec_dot_q4_K(drow + kbx, h8 + (size_t)kbx * 8, kqs);
        }
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
        if (lane == 0) s_part[hh_local][split] = acc;
    }
    __syncthreads();
    if (hh < H && split == 0 && lane == 0) {
        float o = 0.f;
        #pragma unroll
        for (int s = 0; s < S; s++) o += s_part[hh_local][s];
        output[(size_t)token * H + hh] = __float2bfloat16(o);
    }
}

// int8 dp4a MMVQ down (Q5_K). Mirrors the Q4_K down (one warp per output row, top_k experts folded,
// vdr=2 positions per superblock strided across the warp) with the Q5_K block size (176 B) and the
// 5-bit vec_dot above. This is the down projection quant that the UD Q4_K_M Qwen3.6 experts use, and
// the only MoE GEMV still on the fp register-dequant path.
__global__ void down_q5k_mmvq_kernel(
    const unsigned char* __restrict__ down_q, const int* __restrict__ expert_ids,
    const float* __restrict__ expert_weights, const si_block_q8_1* __restrict__ hq8,
    __nv_bfloat16* __restrict__ output, int H, int F, int top_k, int pdl
) {
    if (pdl) si_pdl_sync();
    const int token = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int hh = blockIdx.y * WPB + (threadIdx.x >> 5);
    if (hh >= H) return;
    const int nblk = F >> 8;
    const int q8pb = F >> 5;
    const int work = nblk * 16;
    float acc = 0.f;
    for (int j = 0; j < top_k; j++) {
        const int ts = token * top_k + j;
        const int e = expert_ids[ts];
        const float w = expert_weights[ts];
        const si_block_q5_K* drow = reinterpret_cast<const si_block_q5_K*>(
            down_q + ((size_t)e * H + hh) * nblk * 176);
        const si_block_q8_1* h8 = hq8 + (size_t)ts * q8pb;
        float t = 0.f;
        for (int wi = lane; wi < work; wi += 32) {
            const int kbx = wi >> 4, kqs = (wi & 15) << 1;
            t += si_vec_dot_q5_K(drow + kbx, h8 + (size_t)kbx * 8, kqs);
        }
        acc += w * t;
    }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) output[(size_t)token * H + hh] = __float2bfloat16(acc);
}

template <int S, int WPBK = WPB>
__global__ void down_q5k_mmvq_splitk_kernel(
    const unsigned char* __restrict__ down_q, const int* __restrict__ expert_ids,
    const float* __restrict__ expert_weights, const si_block_q8_1* __restrict__ hq8,
    __nv_bfloat16* __restrict__ output, int H, int F, int top_k, int pdl
) {
    if (pdl) si_pdl_sync();
    // RPB = rows (hidden columns) per block. At the shipped WPBK = WPB = 8 and the exact-split
    // S = 8 this is 1, i.e. a 256-thread block per single bf16 output -- with total = top_k*work
    // = 256 that is one si_vec_dot_q5_K per thread, then a 32-lane butterfly, a shared store and a
    // __syncthreads. Correct for AR decode, where grid.x = 1 and the block count is the only thing
    // filling the GPU; wasteful for the DFlash verify, where grid.x = B already does. Raising WPBK
    // packs several hidden columns into one block, and since s_part is indexed per column and each
    // column keeps its own splits, warps and reduction order, the result is bit-identical -- the
    // adjacent hh also share the same expert rows, so the wider block reads them contiguously.
    constexpr int RPB = WPBK / S;
    __shared__ float s_part[RPB][S];
    const int token = blockIdx.x, lane = threadIdx.x & 31, warpId = threadIdx.x >> 5;
    const int hh_local = warpId / S, split = warpId % S;
    const int hh = blockIdx.y * RPB + hh_local;
    const int nblk = F >> 8;
    const int q8pb = F >> 5;
    const int work = nblk * 16;
    float acc = 0.f;
    if (hh < H) {
        const int total = top_k * work;
        for (int wi = split * 32 + lane; wi < total; wi += S * 32) {
            const int j = wi / work, r = wi % work;
            const int kbx = r >> 4, kqs = (r & 15) << 1;
            const int ts = token * top_k + j, e = expert_ids[ts];
            const float w = expert_weights[ts];
            const si_block_q5_K* drow = reinterpret_cast<const si_block_q5_K*>(
                down_q + ((size_t)e * H + hh) * nblk * 176);
            const si_block_q8_1* h8 = hq8 + (size_t)ts * q8pb;
            acc += w * si_vec_dot_q5_K(drow + kbx, h8 + (size_t)kbx * 8, kqs);
        }
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
        if (lane == 0) s_part[hh_local][split] = acc;
    }
    __syncthreads();
    if (hh < H && split == 0 && lane == 0) {
        float o = 0.f;
        #pragma unroll
        for (int s = 0; s < S; s++) o += s_part[hh_local][s];
        output[(size_t)token * H + hh] = __float2bfloat16(o);
    }
}

template <int S, int NBLK, int TOPK>
__global__ void down_q6k_mmvq_splitk_qwen_kernel(
    const unsigned char* __restrict__ down_q, const int* __restrict__ expert_ids,
    const float* __restrict__ expert_weights, const si_block_q8_1* __restrict__ hq8,
    __nv_bfloat16* __restrict__ output, int H, int pdl
) {
    if (pdl) si_pdl_sync();
    constexpr int RPB = WPB / S;
    constexpr int Q8PB = NBLK * 8;
    __shared__ float s_part[RPB][S];
    const int token = blockIdx.x, lane = threadIdx.x & 31, warpId = threadIdx.x >> 5;
    const int hh_local = warpId / S, split = warpId % S;
    const int hh = blockIdx.y * RPB + hh_local;
    float acc = 0.f;
    if (hh < H) {
        #pragma unroll
        for (int wi = split; wi < TOPK * NBLK; wi += S) {
            const int j = wi / NBLK, kbx = wi - j * NBLK;
            const int ts = token * TOPK + j, e = expert_ids[ts];
            const float w = expert_weights[ts];
            const unsigned char* drow = down_q + ((size_t)e * H + hh) * NBLK * 210;
            const si_block_q8_1* h8 = hq8 + (size_t)ts * Q8PB;
            acc += w * si_vec_dot_q6_K(drow + (size_t)kbx * 210, h8 + (size_t)kbx * 8, lane);
        }
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
        if (lane == 0) s_part[hh_local][split] = acc;
    }
    __syncthreads();
    if (hh < H && split == 0 && lane == 0) {
        float o = 0.f;
        #pragma unroll
        for (int s = 0; s < S; s++) o += s_part[hh_local][s];
        output[(size_t)token * H + hh] = __float2bfloat16(o);
    }
}

template <int S, int NBLK, int TOPK>
__global__ void down_q4k_mmvq_splitk_qwen_kernel(
    const unsigned char* __restrict__ down_q, const int* __restrict__ expert_ids,
    const float* __restrict__ expert_weights, const si_block_q8_1* __restrict__ hq8,
    __nv_bfloat16* __restrict__ output, int H, int pdl
) {
    if (pdl) si_pdl_sync();
    constexpr int RPB = WPB / S;
    constexpr int Q8PB = NBLK * 8;
    constexpr int WORK = NBLK * 16;
    __shared__ float s_part[RPB][S];
    const int token = blockIdx.x, lane = threadIdx.x & 31, warpId = threadIdx.x >> 5;
    const int hh_local = warpId / S, split = warpId % S;
    const int hh = blockIdx.y * RPB + hh_local;
    float acc = 0.f;
    if (hh < H) {
        #pragma unroll 1
        for (int wi = split * 32 + lane; wi < TOPK * WORK; wi += S * 32) {
            const int j = wi / WORK, r = wi - j * WORK;
            const int kbx = r >> 4, kqs = (r & 15) << 1;
            const int ts = token * TOPK + j, e = expert_ids[ts];
            const float w = expert_weights[ts];
            const si_block_q4_K* drow = reinterpret_cast<const si_block_q4_K*>(
                down_q + ((size_t)e * H + hh) * NBLK * 144);
            const si_block_q8_1* h8 = hq8 + (size_t)ts * Q8PB;
            acc += w * si_vec_dot_q4_K(drow + kbx, h8 + (size_t)kbx * 8, kqs);
        }
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
        if (lane == 0) s_part[hh_local][split] = acc;
    }
    __syncthreads();
    if (hh < H && split == 0 && lane == 0) {
        float o = 0.f;
        #pragma unroll
        for (int s = 0; s < S; s++) o += s_part[hh_local][s];
        output[(size_t)token * H + hh] = __float2bfloat16(o);
    }
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/moe.h"
#include <cstdlib>

// Split count for the split-K MMVQ down (SPARKINFER_DOWN_SPLITK_S, default 2).
// Swept on the RTX 5090 / Qwen3-30B-A3B: S=2 is the decode optimum (S=2 ≈ S=8 >
// S=4 > one-warp); S=2 wins on the least cross-split reduction overhead. 0 or 1
// disables split-K and restores the one-warp-per-row MMVQ down.
static inline int down_splitk_s() {
    static int s = -2;
    if (s == -2) {
        const char* v = getenv("SPARKINFER_DOWN_SPLITK_S");
        s = v ? atoi(v) : 2;
        if (!(s == 0 || s == 1 || s == 2 || s == 4 || s == 8)) s = 2;
    }
    return s;
}

static inline int down_splitk_s_q6() {
    static int s = -2;
    if (s == -2) {
        const char* v = getenv("SPARKINFER_DOWN_SPLITK_S_Q6");
        if (!v) return down_splitk_s();
        s = atoi(v);
        if (!(s == 0 || s == 1 || s == 2 || s == 4 || s == 8)) s = down_splitk_s();
    }
    return s;
}

// Shape-tuned Q6_K down split count. Dense hybrid Qwythos (4096,12288,top-1): S=8 wins on 5090.
static inline int down_splitk_s_q6_ffn(int hidden, int ffn, int top_k) {
    if (const char* v = getenv("SPARKINFER_DOWN_SPLITK_S_Q6")) {
        int s = atoi(v);
        if (s == 0 || s == 1 || s == 2 || s == 4 || s == 8) return s;
    }
    if (const char* v = getenv("SPARKINFER_DOWN_SPLITK_S")) {
        int s = atoi(v);
        if (s == 0 || s == 1 || s == 2 || s == 4 || s == 8) return s;
    }
    if (hidden == 4096 && ffn == 12288 && top_k == 1) return 8;
    return down_splitk_s_q6();
}

static inline int down_splitk_s_q4() {
    static int s = -2;
    if (s == -2) {
        const char* v = getenv("SPARKINFER_DOWN_SPLITK_S_Q4");
        if (!v) return down_splitk_s();
        s = atoi(v);
        if (!(s == 0 || s == 1 || s == 2 || s == 4 || s == 8)) s = down_splitk_s();
    }
    return s;
}

// Shape-tuned Q4_K down split count. Dense hybrid Qwythos after #323 requant: S=8 on 5090.
static inline int down_splitk_s_q4_ffn(int hidden, int ffn, int top_k) {
    if (const char* v = getenv("SPARKINFER_DOWN_SPLITK_S_Q4")) {
        int s = atoi(v);
        if (s == 0 || s == 1 || s == 2 || s == 4 || s == 8) return s;
    }
    if (const char* v = getenv("SPARKINFER_DOWN_SPLITK_S")) {
        int s = atoi(v);
        if (s == 0 || s == 1 || s == 2 || s == 4 || s == 8) return s;
    }
    if (hidden == 4096 && ffn == 12288 && top_k == 1) return 8;
    return down_splitk_s_q4();
}

static inline int down_splitk_s_q5() {
    static int s = -2;
    if (s == -2) {
        const char* v = getenv("SPARKINFER_DOWN_SPLITK_S_Q5");
        if (!v) return 8;   // Q5_K down: S=8 wins on Qwen3.6 UD decode (5090 sweep)
        s = atoi(v);
        if (!(s == 0 || s == 1 || s == 2 || s == 4 || s == 8)) s = 8;
    }
    return s;
}

static inline int down_mmvq_pdl() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("SPARKINFER_DOWN_PDL");
        v = (e && e[0] == '0') ? 0 : 1;
    }
    return v;
}

static inline int gu_mmvq_pdl() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("SPARKINFER_GU_PDL");
        v = (e && e[0] == '0') ? 0 : 1;
    }
    return v;
}

// Warps per block for the multi-row gate/up tiling; 0 disables it and restores the pack2 path for
// every caller. Only reached when num_tokens > 1. Swept on RTX 5090 at both scored contexts
// (pack2 -> 4 -> 8 -> 16 warps): 128-ctx 832.7 -> 842.5 -> 841.2 -> 838.6, 4k 514.1 -> 520.5 ->
// 519.3 -> 519.0. Four warps per block wins at both.
static inline int gu_batch_warps() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("SPARKINFER_GU_BATCH_WARPS");
        v = e ? atoi(e) : 4;
        if (v < 0) v = 0;
        if (v > 32) v = 32;
    }
    return v;
}

// Warps per block for the multi-row Q5_K down split-K. WPB is the shipped tiling and stays the
// default: widening the block so one block covers several hidden columns is bit-identical but
// measurably slower here (4k, 32 warps vs WPB: 496.5 vs 519.3), because a 1024-thread block costs
// more occupancy than the block count it saves. Kept as a knob, not a default.
static inline int down_batch_warps() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("SPARKINFER_DOWN_BATCH_WARPS");
        v = e ? atoi(e) : WPB;
        if (v != WPB && v != 2 * WPB && v != 4 * WPB) v = WPB;
    }
    return v;
}

template <int H, int F, int TOPK, typename... Args>
static inline void launch_gate_up_warp_qwen(int warps, int pdl, int n_rows, cudaStream_t stream,
                                            Args... args) {
    const dim3 grid((n_rows + warps - 1) / warps);
    switch (warps) {
        case 4:  launch_pdl_kernel(pdl, grid, dim3(4 * 32), 0, stream,
                     gate_up_mmvq2_warp_qwen_kernel<H, F, TOPK, 4>, args..., n_rows, pdl); return;
        case 16: launch_pdl_kernel(pdl, grid, dim3(16 * 32), 0, stream,
                     gate_up_mmvq2_warp_qwen_kernel<H, F, TOPK, 16>, args..., n_rows, pdl); return;
        default: launch_pdl_kernel(pdl, dim3((n_rows + 7) / 8), dim3(8 * 32), 0, stream,
                     gate_up_mmvq2_warp_qwen_kernel<H, F, TOPK, 8>, args..., n_rows, pdl); return;
    }
}

static inline int dense_top1_down_splitk(int current, int top_k, const char* specific_env) {
    if (top_k == 1 && !getenv("SPARKINFER_DOWN_SPLITK_S") && !getenv(specific_env))
        return 8;
    return current;
}

template <typename Kernel, typename... Args>
static inline void launch_pdl_kernel(
    int pdl, dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Kernel kernel, Args... args
) {
    if (!pdl) {
        kernel<<<grid, block, smem, stream>>>(args...);
        return;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, args...);
}

template <typename Kernel, typename... Args>
static inline void launch_mmvq_down_kernel(
    int pdl, dim3 grid, dim3 block, cudaStream_t stream, Kernel kernel, Args... args
) {
    launch_pdl_kernel(pdl, grid, block, 0, stream, kernel, args...);
}

// Dispatch the templated split-K MMVQ down on the runtime split count. WPB=8, so
// S in {1,2,4,8} keeps RPB=WPB/S a positive divisor. Returns false (no launch) when
// split-K is disabled or S is unsupported, so the caller runs the one-warp kernel.
static inline bool launch_down_q6k_mmvq_splitk(
    int S, int pdl, dim3 grid, const unsigned char* down_q, const int* expert_ids,
    const float* expert_weights, const si_block_q8_1* hq8, __nv_bfloat16* output,
    int H, int F, int top_k, cudaStream_t stream
) {
    static int spec = -1;
    if (spec < 0) { const char* e = getenv("SPARKINFER_DOWN_SPEC"); spec = (e && e[0] == '0') ? 0 : 1; }
    const dim3 block(WPB * 32);
    if (spec && S == 2 && F == 512 && top_k == 8) {
        launch_mmvq_down_kernel(pdl, grid, block, stream, down_q6k_mmvq_splitk_qwen_kernel<2, 2, 8>,
            down_q, expert_ids, expert_weights, hq8, output, H, pdl);
        return true;
    }
    if (spec && S == 2 && F == 768 && top_k == 8) {
        launch_mmvq_down_kernel(pdl, grid, block, stream, down_q6k_mmvq_splitk_qwen_kernel<2, 3, 8>,
            down_q, expert_ids, expert_weights, hq8, output, H, pdl);
        return true;
    }
    if (spec && top_k == 1 && F == 12288) {
        if (S == 8) {
            launch_mmvq_down_kernel(pdl, grid, block, stream, down_q6k_mmvq_splitk_qwen_kernel<8, 48, 1>,
                down_q, expert_ids, expert_weights, hq8, output, H, pdl);
            return true;
        }
        if (S == 4) {
            launch_mmvq_down_kernel(pdl, grid, block, stream, down_q6k_mmvq_splitk_qwen_kernel<4, 48, 1>,
                down_q, expert_ids, expert_weights, hq8, output, H, pdl);
            return true;
        }
        if (S == 2) {
            launch_mmvq_down_kernel(pdl, grid, block, stream, down_q6k_mmvq_splitk_qwen_kernel<2, 48, 1>,
                down_q, expert_ids, expert_weights, hq8, output, H, pdl);
            return true;
        }
    }
    switch (S) {
        case 2: launch_mmvq_down_kernel(pdl, grid, block, stream, down_q6k_mmvq_splitk_kernel<2>, down_q, expert_ids, expert_weights, hq8, output, H, F, top_k, pdl); return true;
        case 4: launch_mmvq_down_kernel(pdl, grid, block, stream, down_q6k_mmvq_splitk_kernel<4>, down_q, expert_ids, expert_weights, hq8, output, H, F, top_k, pdl); return true;
        case 8: launch_mmvq_down_kernel(pdl, grid, block, stream, down_q6k_mmvq_splitk_kernel<8>, down_q, expert_ids, expert_weights, hq8, output, H, F, top_k, pdl); return true;
        default: return false;
    }
}
static inline bool launch_down_q4k_mmvq_splitk(
    int S, int pdl, dim3 grid, const unsigned char* down_q, const int* expert_ids,
    const float* expert_weights, const si_block_q8_1* hq8, __nv_bfloat16* output,
    int H, int F, int top_k, cudaStream_t stream
) {
    static int spec = -1;
    if (spec < 0) { const char* e = getenv("SPARKINFER_DOWN_SPEC"); spec = (e && e[0] == '0') ? 0 : 1; }
    const dim3 block(WPB * 32);
    if (spec && S == 2 && F == 512 && top_k == 8) {
        launch_mmvq_down_kernel(pdl, grid, block, stream, down_q4k_mmvq_splitk_qwen_kernel<2, 2, 8>,
            down_q, expert_ids, expert_weights, hq8, output, H, pdl);
        return true;
    }
    if (spec && S == 2 && F == 768 && top_k == 8) {
        launch_mmvq_down_kernel(pdl, grid, block, stream, down_q4k_mmvq_splitk_qwen_kernel<2, 3, 8>,
            down_q, expert_ids, expert_weights, hq8, output, H, pdl);
        return true;
    }
    if (spec && top_k == 1 && F == 12288) {
        if (S == 8) {
            launch_mmvq_down_kernel(pdl, grid, block, stream, down_q4k_mmvq_splitk_qwen_kernel<8, 48, 1>,
                down_q, expert_ids, expert_weights, hq8, output, H, pdl);
            return true;
        }
        if (S == 4) {
            launch_mmvq_down_kernel(pdl, grid, block, stream, down_q4k_mmvq_splitk_qwen_kernel<4, 48, 1>,
                down_q, expert_ids, expert_weights, hq8, output, H, pdl);
            return true;
        }
        if (S == 2) {
            launch_mmvq_down_kernel(pdl, grid, block, stream, down_q4k_mmvq_splitk_qwen_kernel<2, 48, 1>,
                down_q, expert_ids, expert_weights, hq8, output, H, pdl);
            return true;
        }
    }
    // Muse Glimmer's dense FFN down (F = 19968 -> NBLK 78, single expert). The counterpart to the
    // gate/up arm: this was the other decode GEMV still landing on the runtime-parameterised
    // kernel, 2.49 ms of the step. NBLK constant makes WORK and the wi loop bound compile-time.
    // Identical dot products in identical order, so the result is bit-identical.
    if (spec && top_k == 1 && F == 19968) {
        if (S == 8) {
            launch_mmvq_down_kernel(pdl, grid, block, stream, down_q4k_mmvq_splitk_qwen_kernel<8, 78, 1>,
                down_q, expert_ids, expert_weights, hq8, output, H, pdl);
            return true;
        }
        if (S == 4) {
            launch_mmvq_down_kernel(pdl, grid, block, stream, down_q4k_mmvq_splitk_qwen_kernel<4, 78, 1>,
                down_q, expert_ids, expert_weights, hq8, output, H, pdl);
            return true;
        }
        if (S == 2) {
            launch_mmvq_down_kernel(pdl, grid, block, stream, down_q4k_mmvq_splitk_qwen_kernel<2, 78, 1>,
                down_q, expert_ids, expert_weights, hq8, output, H, pdl);
            return true;
        }
    }
    switch (S) {
        case 2: launch_mmvq_down_kernel(pdl, grid, block, stream, down_q4k_mmvq_splitk_kernel<2>, down_q, expert_ids, expert_weights, hq8, output, H, F, top_k, pdl); return true;
        case 4: launch_mmvq_down_kernel(pdl, grid, block, stream, down_q4k_mmvq_splitk_kernel<4>, down_q, expert_ids, expert_weights, hq8, output, H, F, top_k, pdl); return true;
        case 8: launch_mmvq_down_kernel(pdl, grid, block, stream, down_q4k_mmvq_splitk_kernel<8>, down_q, expert_ids, expert_weights, hq8, output, H, F, top_k, pdl); return true;
        default: return false;
    }
}
// wpbk = warps per block. WPB is the shipped value and the only one AR decode uses; the DFlash
// verify passes a wider block so one block covers RPB = wpbk/S hidden columns instead of one.
// Every column keeps its own splits and its own reduction, so widening is bit-identical.
static inline bool launch_down_q5k_mmvq_splitk(
    int S, int pdl, dim3 grid, const unsigned char* down_q, const int* expert_ids,
    const float* expert_weights, const si_block_q8_1* hq8, __nv_bfloat16* output,
    int H, int F, int top_k, cudaStream_t stream, int wpbk = WPB
) {
    const dim3 block(wpbk * 32);
    if (wpbk == 2 * WPB) {
        switch (S) {
            case 4: launch_mmvq_down_kernel(pdl, grid, block, stream, down_q5k_mmvq_splitk_kernel<4, 2 * WPB>, down_q, expert_ids, expert_weights, hq8, output, H, F, top_k, pdl); return true;
            case 8: launch_mmvq_down_kernel(pdl, grid, block, stream, down_q5k_mmvq_splitk_kernel<8, 2 * WPB>, down_q, expert_ids, expert_weights, hq8, output, H, F, top_k, pdl); return true;
            default: return false;
        }
    }
    if (wpbk == 4 * WPB) {
        switch (S) {
            case 4: launch_mmvq_down_kernel(pdl, grid, block, stream, down_q5k_mmvq_splitk_kernel<4, 4 * WPB>, down_q, expert_ids, expert_weights, hq8, output, H, F, top_k, pdl); return true;
            case 8: launch_mmvq_down_kernel(pdl, grid, block, stream, down_q5k_mmvq_splitk_kernel<8, 4 * WPB>, down_q, expert_ids, expert_weights, hq8, output, H, F, top_k, pdl); return true;
            default: return false;
        }
    }
    switch (S) {
        case 2: launch_mmvq_down_kernel(pdl, grid, block, stream, down_q5k_mmvq_splitk_kernel<2>, down_q, expert_ids, expert_weights, hq8, output, H, F, top_k, pdl); return true;
        case 4: launch_mmvq_down_kernel(pdl, grid, block, stream, down_q5k_mmvq_splitk_kernel<4>, down_q, expert_ids, expert_weights, hq8, output, H, F, top_k, pdl); return true;
        case 8: launch_mmvq_down_kernel(pdl, grid, block, stream, down_q5k_mmvq_splitk_kernel<8>, down_q, expert_ids, expert_weights, hq8, output, H, F, top_k, pdl); return true;
        default: return false;
    }
}

// ---- Muse Glimmer contextual-sparsity FFN -------------------------------------------------
// SwiGLU gates each FFN neuron by silu(gate.x); a neuron whose |silu(gate.x)| ~ 0 contributes
// ~nothing to h = silu(gate.x)*up.x (and thus to the down projection). Muse Glimmer is strongly
// contextually sparse per token, so this fused gate/up kernel computes the gate projection
// first and, for a gated-off neuron, writes h = 0 and SKIPS that neuron's up-projection Q4_K
// weight read -- reading fewer weight bytes per token, which is the decode bottleneck. The mask
// |silu(gate.x)| < tau is uniform across the CTA (one CTA per neuron f), so there is no warp
// divergence and the launch grid is unchanged -- the captured decode graph is unaffected. tau is
// read once from the environment (default 0.06); SPARKINFER_MG_SPARSE_FFN=0 disables it (falls
// back to the dense gate/up arm) and SPARKINFER_MG_SPARSE_TAU overrides tau, both for A/B.
static float g_mg_sparse_tau = -1.f;   // <0 = uninit; env-set once (host), passed by value

static void mg_sparse_ffn_init() {
    if (g_mg_sparse_tau >= 0.f) return;
    const char* off = getenv("SPARKINFER_MG_SPARSE_FFN");
    if (off && off[0] == '0') { g_mg_sparse_tau = 0.f; return; }
    const char* t = getenv("SPARKINFER_MG_SPARSE_TAU");
    g_mg_sparse_tau = t ? (float)atof(t) : 0.06f;
}

template <int H, int F, int TOPK>
__global__ void gate_up_mmvq2_qwen_sparse_kernel(
    const si_block_q8_1* __restrict__ vy, const unsigned char* __restrict__ gate_q,
    const unsigned char* __restrict__ up_q, const int* __restrict__ expert_ids,
    float* __restrict__ h_scratch, float tau, int pdl
) {
    constexpr int NW = 4;
    const int row = blockIdx.x, ts = row / F, f = row - ts * F, tok = ts / TOPK;
    const int e = expert_ids[ts];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
    const int kbx0 = tid >> 4;
    const int kqs = 2 * (tid & 15);
    const si_block_q8_1* vrow = vy + (size_t)tok * (H >> 5);
    const si_block_q4_K* g_row = (const si_block_q4_K*)(gate_q + ((size_t)e * F + f) * (H >> 8) * 144);
    const si_block_q4_K* u_row = (const si_block_q4_K*)(up_q   + ((size_t)e * F + f) * (H >> 8) * 144);
    constexpr int NB = H >> 8;
    __shared__ float sred[NW];
    // Phase 1: gate projection -> reduce to ALL threads -> silu (uniform; decides the mask).
    float tg = 0.f;
    for (int kbx = kbx0; kbx < NB; kbx += 8)
        tg += si_vec_dot_q4_K(g_row + kbx, vrow + (size_t)kbx * 8, kqs);
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) tg += __shfl_xor_sync(0xffffffff, tg, m);
    if (lane == 0) sred[warp] = tg;
    __syncthreads();
    float tgf = 0.f;
    #pragma unroll
    for (int w = 0; w < NW; w++) tgf += sred[w];
    const float g = q4kf_silu(tgf);
    float hval = 0.f;
    if (fabsf(g) >= tau) {                 // active neuron -> read up (Phase 2)
        float tu = 0.f;
        for (int kbx = kbx0; kbx < NB; kbx += 8)
            tu += si_vec_dot_q4_K(u_row + kbx, vrow + (size_t)kbx * 8, kqs);
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) tu += __shfl_xor_sync(0xffffffff, tu, m);
        __syncthreads();
        if (lane == 0) sred[warp] = tu;
        __syncthreads();
        float tuf = 0.f;
        #pragma unroll
        for (int w = 0; w < NW; w++) tuf += sred[w];
        hval = g * tuf;
    }
    if (tid == 0) h_scratch[(size_t)ts * F + f] = hval;
    if (pdl) si_pdl_lc();
}

void launch_moe_expert_ffn_q4k(
    const void* input, const void* gate_q, const void* up_q, const void* down_q,
    int gate_type, int up_type, int down_type,
    const int* expert_ids, const float* expert_weights, void* output,
    float* h_scratch, float* out_scratch,
    int num_tokens, int top_k, int hidden, int ffn, const void* input_q8, cudaStream_t stream,
    bool ar_exact_splitk
) {
    mg_sparse_ffn_init();
    // Qwythos dense hybrid fast path: pack2 gate/up without expert lookup + PDL-chained
    // quant/down. SPARKINFER_DENSE_FFN_FUSE=0 restores the generic MoE dispatch below.
    static int dense_fuse = -1;
    if (dense_fuse < 0) {
        const char* e = getenv("SPARKINFER_DENSE_FFN_FUSE");
        dense_fuse = (e && e[0] == '0') ? 0 : 1;   // default on for Qwythos dense 4096x12288
    }
    if (dense_fuse && num_tokens == 1 && top_k == 1 && hidden == 4096 && ffn == 12288
        && gate_type == 12 && up_type == 12 && (down_type == 12 || down_type == 14)) {
        constexpr int H = 4096, F = 12288;
        const si_block_q8_1* vy;
        si_block_q8_1* qbuf = reinterpret_cast<si_block_q8_1*>(out_scratch);
        if (input_q8) {
            vy = reinterpret_cast<const si_block_q8_1*>(input_q8);
        } else {
            si_quant_bf16_q8_1<<<(H >> 5), 32, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input), qbuf, H);
            vy = qbuf;
        }
        dense_gate_up_q4k_pack2_kernel<H, F><<<(F + 1) / 2, 8 * 32, 0, stream>>>(
            vy, reinterpret_cast<const unsigned char*>(gate_q),
            reinterpret_cast<const unsigned char*>(up_q), h_scratch);
        const int nqb = F >> 5;
        const int pdl = down_mmvq_pdl();
        quant_h_q8_1_kernel<<<(nqb + 7) / 8, 8 * 32, 0, stream>>>(
            h_scratch, qbuf, nqb, pdl);
        const int S = (down_type == 14) ? down_splitk_s_q6_ffn(H, F, 1)
                                        : down_splitk_s_q4_ffn(H, F, 1);
        if (S > 1) {
            const int RPB = WPB / S;
            dim3 dns(1, (H + RPB - 1) / RPB);
            if (down_type == 14) {
                if (launch_down_q6k_mmvq_splitk(S, pdl, dns,
                        reinterpret_cast<const unsigned char*>(down_q), expert_ids, expert_weights, qbuf,
                        reinterpret_cast<__nv_bfloat16*>(output), H, F, 1, stream))
                    return;
            } else if (launch_down_q4k_mmvq_splitk(S, pdl, dns,
                        reinterpret_cast<const unsigned char*>(down_q), expert_ids, expert_weights, qbuf,
                        reinterpret_cast<__nv_bfloat16*>(output), H, F, 1, stream))
                return;
        }
        dim3 dnm(1, (H + WPB - 1) / WPB);
        if (down_type == 14) {
            launch_mmvq_down_kernel(pdl, dnm, dim3(WPB * 32), stream, down_q6k_mmvq_kernel,
                reinterpret_cast<const unsigned char*>(down_q), expert_ids, expert_weights, qbuf,
                reinterpret_cast<__nv_bfloat16*>(output), H, F, 1, pdl);
        } else {
            launch_mmvq_down_kernel(pdl, dnm, dim3(WPB * 32), stream, down_q4k_mmvq_kernel,
                reinterpret_cast<const unsigned char*>(down_q), expert_ids, expert_weights, qbuf,
                reinterpret_cast<__nv_bfloat16*>(output), H, F, 1, pdl);
        }
        return;
    }

    // int8 dp4a path for Q4_K gate/up (decode parity with llama.cpp's MMVQ). Default
    // ON — the largest single decode cost; down stays on the fp path (Q6_K). Set
    // SPARKINFER_MMVQ=0 to fall back to the bf16 dequant-GEMV.
    static int mmvq = -1;
    if (mmvq < 0) { const char* ev = getenv("SPARKINFER_MMVQ"); mmvq = (ev && ev[0] == '0') ? 0 : 1; }

    static int gu2 = -1;   // default ON: faithful 4-warp Q4_K mmvq gate/up. =0 falls back to #50 path
    if (gu2 < 0) { const char* g2 = getenv("SPARKINFER_GU2"); gu2 = (g2 && g2[0] == '0') ? 0 : 1; }
    static int gu_spec = -1;
    if (gu_spec < 0) { const char* gs = getenv("SPARKINFER_GU_SPEC"); gu_spec = (gs && gs[0] == '0') ? 0 : 1; }
    static int gu_pack2 = -1;
    if (gu_pack2 < 0) { const char* gp = getenv("SPARKINFER_GU_PACK2"); gu_pack2 = (gp && gp[0] == '0') ? 0 : 1; }
    const int gu_pdl = gu_mmvq_pdl();
    dim3 gu(num_tokens * top_k, (ffn + WPB - 1) / WPB);
    if (mmvq && gu2 && ((gate_type == 12 && up_type == 12) ||
                       (gate_type == SI_QTYPE_Q3A && up_type == SI_QTYPE_Q3A))) {   // faithful 4-warp mmvq gate/up
        const si_block_q8_1* q;
        if (input_q8) {   // pre-quantized Q8_1(hn) from the fused norm: skip the quantize node
            q = reinterpret_cast<const si_block_q8_1*>(input_q8);
        } else {
            si_block_q8_1* qbuf = reinterpret_cast<si_block_q8_1*>(out_scratch);  // Q8_1(hn) once
            const int nqb = num_tokens * (hidden >> 5);
            si_quant_bf16_q8_1<<<nqb, 32, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(input), qbuf, num_tokens * hidden);
            q = qbuf;
        }
        // Multi-row callers (the DFlash compact verify) get the one-warp-per-element tiling: same
        // arithmetic, same order, same result, but without the 4-warp shared-memory reduce that
        // only earns its keep when num_tokens == 1 leaves the GPU short of warps.
        const int gu_warps = gu_batch_warps();
        // Q3_A gate/up first: every arm below hard-codes the 144-byte Q4_K super-block, so a
        // converted tensor reaching one of them would read the wrong stride.
        if (gate_type == SI_QTYPE_Q3A) {
            static int q3_spec = -1;
            if (q3_spec < 0) { const char* e = getenv("SPARKINFER_MUSE_Q3A_SPEC"); q3_spec = (e && e[0] == '0') ? 0 : 1; }
            if (q3_spec && num_tokens == 1 && top_k == 1 && hidden == 6656 && ffn == 19968)
                launch_pdl_kernel(gu_pdl, dim3(19968), dim3(4 * 32), 0, stream, gate_up_q3a_muse_kernel<6656, 19968>,
                    q, reinterpret_cast<const unsigned char*>(gate_q), reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch, gu_pdl);
            else
                launch_pdl_kernel(gu_pdl, dim3(num_tokens * top_k * ffn), dim3(4 * 32), 0, stream, gate_up_q3a_kernel,
                    q, reinterpret_cast<const unsigned char*>(gate_q), reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch,
                    hidden, ffn, top_k, gu_pdl);
        } else if (num_tokens > 1 && gu_warps > 0 && gu_spec && hidden == 2048 && ffn == 512 && top_k == 8) {
            const int n_rows = num_tokens * top_k * ffn;
            launch_gate_up_warp_qwen<2048, 512, 8>(gu_warps, gu_pdl, n_rows, stream,
                q, reinterpret_cast<const unsigned char*>(gate_q),
                reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch);
        } else if (num_tokens > 1 && gu_warps > 0 && gu_spec && hidden == 2048 && ffn == 768 && top_k == 8) {
            const int n_rows = num_tokens * top_k * ffn;
            launch_gate_up_warp_qwen<2048, 768, 8>(gu_warps, gu_pdl, n_rows, stream,
                q, reinterpret_cast<const unsigned char*>(gate_q),
                reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch);
        } else if (gu_pack2 && gu_spec && hidden == 2048 && ffn == 512 && top_k == 8)
            launch_pdl_kernel(gu_pdl, dim3((num_tokens * top_k * ffn + 1) / 2), dim3(8 * 32), 0, stream,
                gate_up_mmvq2_pack2_qwen_kernel<2048, 512, 8>,
                q, reinterpret_cast<const unsigned char*>(gate_q),
                reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch,
                num_tokens * top_k * ffn, gu_pdl);
        else if (gu_spec && hidden == 2048 && ffn == 512 && top_k == 8)
            launch_pdl_kernel(gu_pdl, dim3(num_tokens * top_k * ffn), dim3(4 * 32), 0, stream,
                gate_up_mmvq2_qwen_kernel<2048, 512, 8>,
                q, reinterpret_cast<const unsigned char*>(gate_q),
                reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch, gu_pdl);
        else if (gu_pack2 && gu_spec && hidden == 2048 && ffn == 768 && top_k == 8)
            launch_pdl_kernel(gu_pdl, dim3((num_tokens * top_k * ffn + 1) / 2), dim3(8 * 32), 0, stream,
                gate_up_mmvq2_pack2_qwen_kernel<2048, 768, 8>,
                q, reinterpret_cast<const unsigned char*>(gate_q),
                reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch,
                num_tokens * top_k * ffn, gu_pdl);
        else if (gu_spec && hidden == 2048 && ffn == 768 && top_k == 8)
            launch_pdl_kernel(gu_pdl, dim3(num_tokens * top_k * ffn), dim3(4 * 32), 0, stream,
                gate_up_mmvq2_qwen_kernel<2048, 768, 8>,
                q, reinterpret_cast<const unsigned char*>(gate_q),
                reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch, gu_pdl);
        else if (gu_pack2 && gu_spec && hidden == 4096 && ffn == 12288 && top_k == 1)
            launch_pdl_kernel(gu_pdl, dim3((num_tokens * top_k * ffn + 1) / 2), dim3(8 * 32), 0, stream,
                gate_up_mmvq2_pack2_qwen_kernel<4096, 12288, 1>,
                q, reinterpret_cast<const unsigned char*>(gate_q),
                reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch,
                num_tokens * top_k * ffn, gu_pdl);
        else if (gu_spec && hidden == 4096 && ffn == 12288 && top_k == 1)
            launch_pdl_kernel(gu_pdl, dim3(num_tokens * top_k * ffn), dim3(4 * 32), 0, stream,
                gate_up_mmvq2_qwen_kernel<4096, 12288, 1>,
                q, reinterpret_cast<const unsigned char*>(gate_q),
                reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch, gu_pdl);
        // Muse Glimmer's dense FFN (hidden 6656, ffn 19968, single expert). This was the last
        // shape on the runtime-parameterised kernel, and it is the largest kernel on the decode
        // step -- 5.07 ms of a 12.7 ms token, 7.77 GB at 1532 GB/s. With H and F constant the
        // per-CTA row/expert decode stops issuing true integer divides and NB = H>>8 = 26 becomes
        // a known trip count. Same dot products in the same order as the generic kernel, so the
        // result is bit-identical.
        //
        // No pack2 arm here, unlike the three shapes above. pack2 exists because at top_k*ffn =
        // 4096 a one-row 4-warp CTA leaves the GPU short of blocks; F = 19968 already launches
        // five times that, so pairing rows into 8-warp CTAs only coarsens scheduling. Measured
        // on a 5090 at 128 decode, a pack2 arm for this shape gives back about four fifths of
        // what this arm wins, so it is deliberately absent rather than merely unwritten.
        else if (gu_spec && hidden == 6656 && ffn == 19968 && top_k == 1) {
            if (g_mg_sparse_tau > 0.f)   // Muse Glimmer contextual-sparsity FFN (skip gated-off up reads)
                launch_pdl_kernel(gu_pdl, dim3(num_tokens * top_k * ffn), dim3(4 * 32), 0, stream,
                    gate_up_mmvq2_qwen_sparse_kernel<6656, 19968, 1>,
                    q, reinterpret_cast<const unsigned char*>(gate_q),
                    reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch, g_mg_sparse_tau, gu_pdl);
            else
                launch_pdl_kernel(gu_pdl, dim3(num_tokens * top_k * ffn), dim3(4 * 32), 0, stream,
                    gate_up_mmvq2_qwen_kernel<6656, 19968, 1>,
                    q, reinterpret_cast<const unsigned char*>(gate_q),
                    reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch, gu_pdl);
        }
        // Qwen3.8-27B dense hybrid (hidden 5120, ffn 17408). Same recipe as Muse's 6656x19968
        // arm: F is already large enough that pack2 coarsens scheduling. Compile-time H/F
        // makes NB = H>>8 = 20 a known trip count; same dots in the same order as the
        // generic kernel, so the result is bit-identical.
        else if (gu_spec && hidden == 5120 && ffn == 17408 && top_k == 1 &&
                 num_tokens >= 2 && num_tokens <= 4) {
            // Verify rows share the FFN weights; read them once. See
            // gate_up_mmvq2_qwen_rows_kernel -- bit-identical per row, one CTA per ffn row.
            static const int rows_on = []{ const char* e = getenv("SPARKINFER_GU_ROWS");
                                           return (e && e[0] == '0') ? 0 : 1; }();
            if (rows_on) {
                const dim3 gr(ffn), gb(4 * 32);
                if (num_tokens == 2)
                    launch_pdl_kernel(gu_pdl, gr, gb, 0, stream, gate_up_mmvq2_qwen_rows_kernel<5120, 17408, 1, 2>,
                        q, reinterpret_cast<const unsigned char*>(gate_q),
                        reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch, gu_pdl);
                else if (num_tokens == 3)
                    launch_pdl_kernel(gu_pdl, gr, gb, 0, stream, gate_up_mmvq2_qwen_rows_kernel<5120, 17408, 1, 3>,
                        q, reinterpret_cast<const unsigned char*>(gate_q),
                        reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch, gu_pdl);
                else
                    launch_pdl_kernel(gu_pdl, gr, gb, 0, stream, gate_up_mmvq2_qwen_rows_kernel<5120, 17408, 1, 4>,
                        q, reinterpret_cast<const unsigned char*>(gate_q),
                        reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch, gu_pdl);
            } else {
                launch_pdl_kernel(gu_pdl, dim3(num_tokens * top_k * ffn), dim3(4 * 32), 0, stream,
                    gate_up_mmvq2_qwen_kernel<5120, 17408, 1>,
                    q, reinterpret_cast<const unsigned char*>(gate_q),
                    reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch, gu_pdl);
            }
        }
        else if (gu_spec && hidden == 5120 && ffn == 17408 && top_k == 1)
            launch_pdl_kernel(gu_pdl, dim3(num_tokens * top_k * ffn), dim3(4 * 32), 0, stream,
                gate_up_mmvq2_qwen_kernel<5120, 17408, 1>,
                q, reinterpret_cast<const unsigned char*>(gate_q),
                reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch, gu_pdl);
        else
            launch_pdl_kernel(gu_pdl, dim3(num_tokens * top_k * ffn), dim3(4 * 32), 0, stream,
                gate_up_mmvq2_kernel,
                q, reinterpret_cast<const unsigned char*>(gate_q),
                reinterpret_cast<const unsigned char*>(up_q), expert_ids, h_scratch,
                hidden, ffn, top_k, gu_pdl);
    } else if (mmvq && gate_type == 12 && up_type == 12) {   // 12 = ggml Q4_K
        size_t sm = 2 * (size_t)(hidden >> 5) * sizeof(float) + (size_t)hidden;  // s_xd+s_xs+s_xq8
        launch_pdl_kernel(gu_pdl, gu, dim3(WPB * 32), sm, stream, gate_up_q4k_mmvq_kernel,
            reinterpret_cast<const __nv_bfloat16*>(input),
            reinterpret_cast<const unsigned char*>(gate_q),
            reinterpret_cast<const unsigned char*>(up_q),
            expert_ids, h_scratch, hidden, ffn, top_k, gu_pdl);
    } else {
        size_t gu_smem = (size_t)hidden * sizeof(float);   // s_x only; s_deq is static
        gate_up_q4k_kernel<<<gu, WPB * 32, gu_smem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input),
            reinterpret_cast<const unsigned char*>(gate_q),
            reinterpret_cast<const unsigned char*>(up_q),
            expert_ids, h_scratch, hidden, ffn, top_k, gate_type, up_type);
    }

    // int8 dp4a MMVQ down (Q6_K) — default ON; SPARKINFER_DOWN_MMVQ=0 falls back to the
    // fp register-dequant down. The MoE down is the last major decode GEMV still on the
    // fp path (gate/up + attention already run int8 MMVQ): quantize the activation h to
    // Q8_1 once (into the otherwise-unused out_scratch) and dp4a the Q6_K weights against
    // it, faithful to llama.cpp vec_dot_q6_K_q8_1.
    static int down_mmvq = -1;
    if (down_mmvq < 0) { const char* dv = getenv("SPARKINFER_DOWN_MMVQ"); down_mmvq = (dv && dv[0] == '0') ? 0 : 1; }
    if (down_mmvq && down_type == 14) {   // 14 = ggml Q6_K
        si_block_q8_1* hq8 = reinterpret_cast<si_block_q8_1*>(out_scratch);   // <= hidden floats; fits
        const int nqb = num_tokens * top_k * (ffn >> 5);
        const int qthreads = 256;
        const int pdl = down_mmvq_pdl();
        const int q_pdl = gu_pdl && pdl;
        launch_pdl_kernel(q_pdl, dim3((nqb + (qthreads >> 5) - 1) / (qthreads >> 5)), dim3(qthreads), 0, stream,
            quant_h_q8_1_kernel, h_scratch, hq8, nqb, q_pdl);
        // split-K MMVQ down: S warps/row -> S*H warps in flight, hiding the bs=1
        // occupancy stall the one-warp kernel hits. Dense top-1 defaults to S=8 unless
        // an explicit split-K env override is set; routed MoE keeps its existing default.
        const int S = dense_top1_down_splitk(down_splitk_s_q6(), top_k, "SPARKINFER_DOWN_SPLITK_S_Q6");
        if (S > 1) {
            const int RPB = WPB / S;
            dim3 dns(num_tokens, (hidden + RPB - 1) / RPB);
            if (launch_down_q6k_mmvq_splitk(S, pdl, dns,
                    reinterpret_cast<const unsigned char*>(down_q), expert_ids, expert_weights, hq8,
                    reinterpret_cast<__nv_bfloat16*>(output), hidden, ffn, top_k, stream))
                return;
        }
        dim3 dnm(num_tokens, (hidden + WPB - 1) / WPB);
        launch_mmvq_down_kernel(pdl, dnm, dim3(WPB * 32), stream, down_q6k_mmvq_kernel,
            reinterpret_cast<const unsigned char*>(down_q), expert_ids, expert_weights, hq8,
            reinterpret_cast<__nv_bfloat16*>(output), hidden, ffn, top_k, pdl);
        return;
    }

    // int8 dp4a MMVQ down (Q4_K) — default ON; SPARKINFER_DOWN_Q4K=0 restores the fp dequant
    // down for the Q4_K rows. Same quantize-once + faithful vec_dot path as the Q6_K down.
    static int down_q4k = -1;
    if (down_q4k < 0) { const char* qv = getenv("SPARKINFER_DOWN_Q4K"); down_q4k = (qv && qv[0] == '0') ? 0 : 1; }
    if (down_mmvq && down_q4k && down_type == 12) {   // 12 = ggml Q4_K
        si_block_q8_1* hq8 = reinterpret_cast<si_block_q8_1*>(out_scratch);
        const int nqb = num_tokens * top_k * (ffn >> 5);
        const int qthreads = 256;
        const int pdl = down_mmvq_pdl();
        const int q_pdl = gu_pdl && pdl;
        launch_pdl_kernel(q_pdl, dim3((nqb + (qthreads >> 5) - 1) / (qthreads >> 5)), dim3(qthreads), 0, stream,
            quant_h_q8_1_kernel, h_scratch, hq8, nqb, q_pdl);
        const int S = dense_top1_down_splitk(down_splitk_s_q4(), top_k, "SPARKINFER_DOWN_SPLITK_S_Q4");
        if (S > 1) {
            const int RPB = WPB / S;
            dim3 dns(num_tokens, (hidden + RPB - 1) / RPB);
            if (launch_down_q4k_mmvq_splitk(S, pdl, dns,
                    reinterpret_cast<const unsigned char*>(down_q), expert_ids, expert_weights, hq8,
                    reinterpret_cast<__nv_bfloat16*>(output), hidden, ffn, top_k, stream))
                return;
        }
        dim3 dnm(num_tokens, (hidden + WPB - 1) / WPB);
        launch_mmvq_down_kernel(pdl, dnm, dim3(WPB * 32), stream, down_q4k_mmvq_kernel,
            reinterpret_cast<const unsigned char*>(down_q), expert_ids, expert_weights, hq8,
            reinterpret_cast<__nv_bfloat16*>(output), hidden, ffn, top_k, pdl);
        return;
    }

    // int8 dp4a MMVQ down (Q5_K) — default ON; SPARKINFER_DOWN_Q5K=0 restores the fp dequant down for
    // the Q5_K rows. The UD Q4_K_M Qwen3.6 experts keep their down at Q5_K, which fell through both the
    // Q6_K and Q4_K MMVQ paths onto the fp register-dequant down. Same quantize-once + faithful vec_dot
    // pipeline, widened to the 5th bit.
    static int down_q5k = -1;
    if (down_q5k < 0) { const char* qv = getenv("SPARKINFER_DOWN_Q5K"); down_q5k = (qv && qv[0] == '0') ? 0 : 1; }
    if (down_mmvq && down_q5k && down_type == 13) {   // 13 = ggml Q5_K
        si_block_q8_1* hq8 = reinterpret_cast<si_block_q8_1*>(out_scratch);
        const int nqb = num_tokens * top_k * (ffn >> 5);
        const int qthreads = 256;
        const int pdl = down_mmvq_pdl();
        const int q_pdl = gu_pdl && pdl;
        launch_pdl_kernel(q_pdl, dim3((nqb + (qthreads >> 5) - 1) / (qthreads >> 5)), dim3(qthreads), 0, stream,
            quant_h_q8_1_kernel, h_scratch, hq8, nqb, q_pdl);
        // Row-count-aware split-K: S=8 (July-2026 sweep) is tuned for 1-row AR decode; the DFlash
        // compact verify runs num_tokens=6 rows -> already high occupancy, so split-K reduction
        // overhead dominates and S=1 wins (+2.7% DFlash decode @2550, bit-exact @128, SPEC 32/32).
        //
        // S also fixes the reduction order, so it is the one thing here that is not row-count
        // invariant: at S=1 each output row is one warp's serial sum over ffn, at S=8 it is eight
        // partial sums combined. Both are correct, but they round differently, which is why a
        // batched call did not reproduce what AR computes one token at a time. A caller that needs
        // that reproduction passes ar_exact_splitk and gets AR's split count for any num_tokens;
        // with S pinned the kernel's grid is dim3(num_tokens, ...), one block column per token, so
        // every row's arithmetic is identical to the num_tokens == 1 call.
        const char* s5env = getenv("SPARKINFER_DOWN_SPLITK_S_Q5");
        const int Sbase = (num_tokens > 1 && !s5env && !ar_exact_splitk) ? 1 : down_splitk_s_q5();
        const int S = dense_top1_down_splitk(Sbase, top_k, "SPARKINFER_DOWN_SPLITK_S_Q5");
        if (S > 1) {
            // A caller that pinned S for exactness (ar_exact_splitk) is the compact verify, which
            // runs many rows; give it a block wide enough that the pinned split count no longer
            // means one block per output. AR decode keeps WPB.
            const int wpbk = num_tokens > 1 ? down_batch_warps() : WPB;
            const int RPB = wpbk / S;
            dim3 dns(num_tokens, (hidden + RPB - 1) / RPB);
            if (launch_down_q5k_mmvq_splitk(S, pdl, dns,
                    reinterpret_cast<const unsigned char*>(down_q), expert_ids, expert_weights, hq8,
                    reinterpret_cast<__nv_bfloat16*>(output), hidden, ffn, top_k, stream, wpbk))
                return;
        }
        dim3 dnm(num_tokens, (hidden + WPB - 1) / WPB);
        launch_mmvq_down_kernel(pdl, dnm, dim3(WPB * 32), stream, down_q5k_mmvq_kernel,
            reinterpret_cast<const unsigned char*>(down_q), expert_ids, expert_weights, hq8,
            reinterpret_cast<__nv_bfloat16*>(output), hidden, ffn, top_k, pdl);
        return;
    }

    static int splitk = -1;
    if (splitk < 0) { const char* sv = getenv("SPARKINFER_SPLITK"); splitk = (sv && sv[0] == '0') ? 0 : 1; }
    static int pdl = -1;
    if (pdl < 0) { const char* pv = getenv("SPARKINFER_PDL"); pdl = (pv && pv[0] == '1') ? 1 : 0; }
    if (splitk) {   // split-K down: 4 warps/row -> 4x warps in flight (occupancy lever)
        const int RPB = WPB / 4;
        dim3 dns(num_tokens, (hidden + RPB - 1) / RPB);
        if (pdl) {   // PDL: down's grid spin-up overlaps gate_up's tail (programmatic dependent launch)
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = dns; cfg.blockDim = dim3(WPB * 32); cfg.dynamicSmemBytes = 0; cfg.stream = stream;
            cudaLaunchAttribute attr; attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr.val.programmaticStreamSerializationAllowed = 1;
            cfg.attrs = &attr; cfg.numAttrs = 1;
            cudaLaunchKernelEx(&cfg, down_q6k_splitk_kernel,
                reinterpret_cast<const unsigned char*>(down_q), expert_ids, expert_weights, h_scratch,
                reinterpret_cast<__nv_bfloat16*>(output), hidden, ffn, top_k, down_type);
        } else {
            down_q6k_splitk_kernel<<<dns, WPB * 32, 0, stream>>>(
                reinterpret_cast<const unsigned char*>(down_q),
                expert_ids, expert_weights, h_scratch,
                reinterpret_cast<__nv_bfloat16*>(output), hidden, ffn, top_k, down_type);
        }
    } else {
        dim3 dn(num_tokens, (hidden + WPB - 1) / WPB);
        down_q6k_kernel<<<dn, WPB * 32, 0, stream>>>(
            reinterpret_cast<const unsigned char*>(down_q),
            expert_ids, expert_weights, h_scratch,
            reinterpret_cast<__nv_bfloat16*>(output), hidden, ffn, top_k, down_type);
    }
}

// Qwen3.6 UD shared expert: Q8_0 weights + int8 dp4a MMVQ (reuses FNQ Q8_1(hn)).
// input_q8 may be overwritten with Q8_1(h) after gate/up. h_scratch: [ffn] fp32.
void launch_shared_expert_q8_mmvq(
    const void* input, const void* input_q8,
    const void* gate_q, const void* up_q, const void* down_q,
    const float* dw, void* output, float* h_scratch, void* h_q8_buf,
    int hidden, int ffn, cudaStream_t stream, bool accum = false) {
    const si_block_q8_1* vy;
    si_block_q8_1* qbuf = reinterpret_cast<si_block_q8_1*>(h_q8_buf);
    if (input_q8) {
        vy = reinterpret_cast<const si_block_q8_1*>(input_q8);
    } else {
        si_quant_bf16_q8_1<<<(hidden >> 5), 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input), qbuf, hidden);
        vy = qbuf;
    }
    shared_gate_up_q8_mmvq_kernel<2048, 512><<<ffn, 4 * 32, 0, stream>>>(
        vy, reinterpret_cast<const unsigned char*>(gate_q),
        reinterpret_cast<const unsigned char*>(up_q), dw, h_scratch);
    const int nqb = ffn >> 5;
    quant_h_q8_1_kernel<<<(nqb + 7) / 8, 8 * 32, 0, stream>>>(
        h_scratch, qbuf, nqb, 0);
    dim3 dn((hidden + WPB - 1) / WPB);
    if (accum) {
        shared_down_q8_mmvq_kernel<2048, 512, true><<<dn, WPB * 32, 0, stream>>>(
            qbuf, reinterpret_cast<const unsigned char*>(down_q),
            reinterpret_cast<__nv_bfloat16*>(output));
    } else {
        shared_down_q8_mmvq_kernel<2048, 512, false><<<dn, WPB * 32, 0, stream>>>(
            qbuf, reinterpret_cast<const unsigned char*>(down_q),
            reinterpret_cast<__nv_bfloat16*>(output));
    }
}

void launch_shared_expert_q8_mmvq_rows(
    const void* input_q8, const void* gate_q, const void* up_q, const void* down_q,
    const float* dw, void* output, float* h_scratch, void* h_q8_buf,
    int hidden, int ffn, int rows, cudaStream_t stream) {
    if (!input_q8 || !gate_q || !up_q || !down_q || !dw || !output || !h_scratch ||
        !h_q8_buf || hidden != 2048 || ffn != 512 || rows < 1 || rows > 8) return;
    const auto* q = reinterpret_cast<const si_block_q8_1*>(input_q8);
    auto* hq = reinterpret_cast<si_block_q8_1*>(h_q8_buf);
    auto* out = reinterpret_cast<__nv_bfloat16*>(output);
#define SI_SHARED_ROWS(R) do { \
    shared_gate_up_q8_mmvq_rows_kernel<2048, 512, R><<<512, si_shexp_nw<2048, 32>() * 32, 0, stream>>>( \
        q, reinterpret_cast<const unsigned char*>(gate_q), \
        reinterpret_cast<const unsigned char*>(up_q), dw, h_scratch); \
    quant_h_q8_1_kernel<<<((R * (512 >> 5)) + 7) / 8, 8 * 32, 0, stream>>>( \
        h_scratch, hq, R * (512 >> 5), 0); \
    shared_down_q8_mmvq_rows_kernel<2048, 512, R><<<(2048 + WPB * 2 - 1) / (WPB * 2), WPB * 32, 0, stream>>>( \
        hq, reinterpret_cast<const unsigned char*>(down_q), out); \
} while (0)
    switch (rows) {
        case 1: SI_SHARED_ROWS(1); break;
        case 2: SI_SHARED_ROWS(2); break;
        case 3: SI_SHARED_ROWS(3); break;
        case 4: SI_SHARED_ROWS(4); break;
        case 5: SI_SHARED_ROWS(5); break;
        case 6: SI_SHARED_ROWS(6); break;
        case 7: SI_SHARED_ROWS(7); break;
        case 8: SI_SHARED_ROWS(8); break;
        default: break;
    }
#undef SI_SHARED_ROWS
}
#endif

} // namespace kernels
} // namespace sparkinfer
