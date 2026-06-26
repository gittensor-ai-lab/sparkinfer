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

// ggml types: Q4_K=12 (144 B/256), Q6_K=14 (210 B/256). Q4_K_M mixes them per tensor.
__device__ __forceinline__ int q_block_bytes(int t) { return t == 14 ? 210 : 144; }

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
    float* __restrict__ h_scratch, int H, int F, int top_k
) {
    si_pdl_lc();   // PDL: let the dependent down kernel begin its grid spin-up now
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
                                    si_block_q8_1* __restrict__ y, int n_blocks) {
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
    __nv_bfloat16* __restrict__ output, int H, int F, int top_k
) {
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

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/moe.h"
#include <cstdlib>

void launch_moe_expert_ffn_q4k(
    const void* input, const void* gate_q, const void* up_q, const void* down_q,
    int gate_type, int up_type, int down_type,
    const int* expert_ids, const float* expert_weights, void* output,
    float* h_scratch, float* out_scratch,
    int num_tokens, int top_k, int hidden, int ffn, cudaStream_t stream
) {
    // int8 dp4a path for Q4_K gate/up (decode parity with llama.cpp's MMVQ). Default
    // ON — the largest single decode cost; down stays on the fp path (Q6_K). Set
    // SPARKINFER_MMVQ=0 to fall back to the bf16 dequant-GEMV.
    static int mmvq = -1;
    if (mmvq < 0) { const char* ev = getenv("SPARKINFER_MMVQ"); mmvq = (ev && ev[0] == '0') ? 0 : 1; }

    dim3 gu(num_tokens * top_k, (ffn + WPB - 1) / WPB);
    if (mmvq && gate_type == 12 && up_type == 12) {   // 12 = ggml Q4_K
        size_t sm = 2 * (size_t)(hidden >> 5) * sizeof(float) + (size_t)hidden;  // s_xd+s_xs+s_xq8
        gate_up_q4k_mmvq_kernel<<<gu, WPB * 32, sm, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input),
            reinterpret_cast<const unsigned char*>(gate_q),
            reinterpret_cast<const unsigned char*>(up_q),
            expert_ids, h_scratch, hidden, ffn, top_k);
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
        quant_h_q8_1_kernel<<<(nqb + (qthreads >> 5) - 1) / (qthreads >> 5), qthreads, 0, stream>>>(
            h_scratch, hq8, nqb);
        dim3 dnm(num_tokens, (hidden + WPB - 1) / WPB);
        down_q6k_mmvq_kernel<<<dnm, WPB * 32, 0, stream>>>(
            reinterpret_cast<const unsigned char*>(down_q), expert_ids, expert_weights, hq8,
            reinterpret_cast<__nv_bfloat16*>(output), hidden, ffn, top_k);
        return;
    }

    static int splitk = -1;
    if (splitk < 0) { const char* sv = getenv("SPARKINFER_SPLITK"); splitk = (sv && sv[0] == '1') ? 1 : 0; }
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
#endif

} // namespace kernels
} // namespace sparkinfer
