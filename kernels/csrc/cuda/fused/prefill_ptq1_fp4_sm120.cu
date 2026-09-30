// Ternary-Bonsai-2's short-prompt prefill GEMMs on the FP4 tensor cores (sm_120a only: the
// block-scaled FP4 MMA does not assemble for plain sm_120, so this file builds in si_nvfp4).
//
// Up to 512 rows these GEMMs ran on the fused int8 GEMM, which decodes the stored ternary blocks to
// int8 in shared memory and issues m16n8k32 int8 MMAs. A trit is exact in e2m1, and m16n8k64
// kind::mxf4nvf4 does twice the MACs an instruction (measured 1985 vs 993 TOPS on an RTX 5090) with
// half the shared-memory bytes on both operands. The block's fp16 scale s rides in the MMA's per-16
// scale: s * 2^10 is written as an e2m1 magnitude (the nonzero trits' code) times a ue4m3, the
// pair nearest it, and the epilogue takes the 2^-10 back off. The activation is rotated as the
// int8 path rotates it and quantized to NVFP4 by prefill_nvfp4's rule, ue4m3(amax / 6) per 16.
//
// The card runs these GEMMs at its 575 W limit, so the decode is built for instruction count: a
// GEMM sums over K, so the 128 values of a block may sit in any order as long as A's sit in the
// same one. Each block is laid out in the order its bytes store the trits (fp4_perm), which lets a
// 256-entry table turn every stored byte straight into its five e2m1 nibbles -- one shared load
// and a shift per five trits, and two 16-byte stores per half-block.
#include "sparkinfer/kernels/prefill_ptq1_fp4.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdlib>

namespace sparkinfer { namespace kernels {

namespace {

constexpr int kBlk = 128;          // weights per PTQ1_0 block
constexpr int kBlkBytes = 28;
constexpr int kSpan = 1024;        // Hadamard span of this checkpoint
constexpr float kWScale = 1024.f;  // the weight scale's lift into ue4m3's normal range

// A PTQ1_0 block stores its 128 trits base-3, five to a byte in bytes 0..23 and four in bytes 24
// and 25 (fp16 scale in 26..27); digit m of a byte is floor(3 * x_m / 256), x_{m+1} = 3 * x_m mod
// 256. In the block's natural order byte b's digit m is value 16m + b (b < 16), 80 + 8m + (b - 16)
// (b < 24) or 120 + 2m + (b - 24). The FP4 slot order here: half h (the thread decoding it) owns
// bytes 8h..8h+7, 16+4h..19+4h and 24+h, and slot 64h + 5i + m holds digit m of its i-th byte
// (60 + m for the four-digit byte). fp4_perm maps a slot to the natural position A reads from.
__device__ __forceinline__ int fp4_perm(int q) {
    const int h = q >> 6, r = q & 63;
    if (r >= 60) return 120 + 2 * (r - 60) + h;
    const int i = r / 5, m = r - 5 * i;
    const int b = i < 8 ? 8 * h + i : 16 + 4 * h + (i - 8);
    return b < 16 ? 16 * m + b : 80 + 8 * m + (b - 16);
}

// ---- activation: rotate, then NVFP4 ---------------------------------------------------------
// One CTA per (row, span), 256 threads holding four consecutive values of the 1024-span: the
// rotation is ptq1_rotq_rows_i8_kernel's, value for value. The quantize then reads the span back in
// slot order, so each four-lane quad owns one 16-slot group. Spans share nothing (the rotation, the
// scales and the output bytes are all per span), so gridDim.y CTAs split a row's spans between
// them; gridDim.y = 1 walks them in one CTA.
template <bool SWIGLU>
__global__ void __launch_bounds__(256)
ptq1_rotq_fp4_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ u,
                     const signed char* __restrict__ sign, unsigned char* __restrict__ q,
                     int rows, int k) {
    __shared__ float sh[kSpan];
    const int t = threadIdx.x, lane = t & 31;
    const int row = blockIdx.x;
    unsigned char* qrow = q + (size_t)row * (k / 2);
    unsigned char* srow = q + (size_t)rows * (k / 2) + (size_t)row * (k / 16);
    int src[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int s = t * 4 + i;
        src[i] = (s & ~(kBlk - 1)) + fp4_perm(s & (kBlk - 1));
    }
    for (int sp = blockIdx.y; sp < k / kSpan; sp += gridDim.y) {
        const int e0 = sp * kSpan + t * 4;
        uint2 raw = *reinterpret_cast<const uint2*>(x + (size_t)row * k + e0);
        if constexpr (SWIGLU) {
            const uint2 ur = *reinterpret_cast<const uint2*>(u + (size_t)row * k + e0);
            const __nv_bfloat16* gh = reinterpret_cast<const __nv_bfloat16*>(&raw);
            const __nv_bfloat16* uh = reinterpret_cast<const __nv_bfloat16*>(&ur);
            __nv_bfloat16 o[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float g = __bfloat162float(gh[j]);
                // __fdividef: what launch_ptq1_swiglu_rotq_rows_i8's --use_fast_math build emits,
                // so the bf16 SwiGLU output is the int8 path's, value for value.
                o[j] = __float2bfloat16(__fdividef(g, 1.f + __expf(-g)) * __bfloat162float(uh[j]));
            }
            raw = *reinterpret_cast<const uint2*>(o);
        }
        const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        const char4 sg = *reinterpret_cast<const char4*>(sign + e0);
        const float w0 = __low2float(a) * (float)sg.x, w1 = __high2float(a) * (float)sg.y;
        const float w2 = __low2float(b) * (float)sg.z, w3 = __high2float(b) * (float)sg.w;
        const float a0 = w0 + w1, a1 = w0 - w1, a2 = w2 + w3, a3 = w2 - w3;
        float r[4] = {a0 + a2, a1 + a3, a0 - a2, a1 - a3};
#pragma unroll
        for (int m = 1; m < 32; m <<= 1) {
            const bool hi = lane & m;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float p = __shfl_xor_sync(0xffffffffu, r[i], m);
                r[i] = hi ? p - r[i] : r[i] + p;
            }
        }
        __syncthreads();   // the previous span's readers are done with sh
#pragma unroll
        for (int i = 0; i < 4; ++i) sh[t * 4 + i] = r[i];
        __syncthreads();
        if (t < 128) {
            float c[8];
#pragma unroll
            for (int w8 = 0; w8 < 8; ++w8) c[w8] = sh[t + 128 * w8];
#pragma unroll
            for (int len = 1; len < 8; len <<= 1)
#pragma unroll
                for (int w8 = 0; w8 < 8; ++w8)
                    if (!(w8 & len)) {
                        const float xx = c[w8], yy = c[w8 + len];
                        c[w8] = xx + yy; c[w8 + len] = xx - yy;
                    }
#pragma unroll
            for (int w8 = 0; w8 < 8; ++w8) sh[t + 128 * w8] = c[w8];
        }
        __syncthreads();
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = sh[src[i]] * 0.03125f;
        float ga = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
        ga = fmaxf(ga, __shfl_xor_sync(0xffffffffu, ga, 1));
        ga = fmaxf(ga, __shfl_xor_sync(0xffffffffu, ga, 2));
        const __nv_fp8_storage_t qb =
            __nv_cvt_float_to_fp8(fmaxf(ga * (1.f / 6.f), 0x1p-9f), __NV_SATFINITE, __NV_E4M3);
        const float rq = __frcp_rn(__half2float(__half(__nv_cvt_fp8_to_halfraw(qb, __NV_E4M3))));
        const unsigned lo = __nv_cvt_float2_to_fp4x2(make_float2(v[0] * rq, v[1] * rq), __NV_E2M1,
                                                     cudaRoundNearest);
        const unsigned hi = __nv_cvt_float2_to_fp4x2(make_float2(v[2] * rq, v[3] * rq), __NV_E2M1,
                                                     cudaRoundNearest);
        *reinterpret_cast<unsigned short*>(qrow + e0 / 2) = (unsigned short)(lo | (hi << 8));
        if ((t & 3) == 0) srow[e0 / 16] = qb;
    }
}

// ---- weights: trits -> e2m1 -------------------------------------------------------------------
// s * 2^10 -> (e2m1 code of the magnitude m, ue4m3 sf) with m * sf nearest it. The checkpoint's
// scales (2.7e-3 .. 0.16) put s * 2^10 at 2.8 .. 164, where every ue4m3 is normal, so m = 2, 4
// only rescale m = 1 by a power of two and m = 3, 6 rescale 1.5: the two candidates below are the
// whole search (m = 1 wins a tie).
__device__ __forceinline__ void fp4_wscale(float s, unsigned& code, unsigned& sf) {
    const float T = fabsf(s) * kWScale;
    const __nv_fp8_storage_t f1 = __nv_cvt_float_to_fp8(T, __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8_storage_t f2 =
        __nv_cvt_float_to_fp8(T * (2.f / 3.f), __NV_SATFINITE, __NV_E4M3);
    const float e1 = fabsf(__half2float(__half(__nv_cvt_fp8_to_halfraw(f1, __NV_E4M3))) - T);
    const float e2 = fabsf(1.5f * __half2float(__half(__nv_cvt_fp8_to_halfraw(f2, __NV_E4M3))) - T);
    const bool one = e1 <= e2;
    code = one ? 2u : 3u;
    sf = one ? f1 : f2;
}

// The byte table: entry v * 256 + x holds byte x's five digits as e2m1 nibbles (digit m at nibble
// m), for variant v = 2 * (s < 0) + (code == 3): digit 0 is -m, 1 is 0 and 2 is +m, the sign
// swapped for a negative scale.
constexpr int kLutN = 4 * 256;
__device__ __forceinline__ unsigned fp4_lut_entry(int v, unsigned x) {
    const unsigned code = (v & 1) ? 3u : 2u;
    const unsigned lo = (v & 2) ? code : (code | 8u), hi = (v & 2) ? (code | 8u) : code;
    unsigned e = 0;
#pragma unroll
    for (int m = 0; m < 5; ++m) {
        x *= 3u;
        const unsigned d = x >> 8;
        x &= 0xFFu;
        e |= (d == 0 ? lo : d == 2 ? hi : 0u) << (4 * m);
    }
    return e;
}

// ---- the GEMM ---------------------------------------------------------------------------------
// 128x128 output tile, K staged one PTQ1 block (128 values = 64 FP4 bytes a row) at a time. The
// roles run concurrently, synchronized by mbarriers as pf_dense_gemm_qi8_ws_kernel does it: one A
// warp streams A stages into a ring with cp.async, eight decode warps turn the weight blocks into
// e2m1 stages of a second ring (a half-block a thread, two stages an iteration, the words a pair
// ahead), and eight MMA warps (2 x 4, 64x32 each) only wait, ldmatrix and mma. The grid is
// persistent over (M-tile, N-tile, K-slice) items, so every role walks the same item list and the
// rings run straight through item edges. Rows are 64 bytes with the 16-byte chunks XOR-swizzled by
// (row/2)&3, so every ldmatrix phase (8 rows, one chunk) hits 32 distinct banks.
// mbarriers: fullA[NS] (A lanes' cp.async arrivals), emptyA[NS] and emptyB[NS] (one per MMA
// warp), fullB[NS] (every decode thread).
constexpr int BM = 128, BN = 128, BKB = 64, MAX_LEGS = 3;
constexpr int NS = 4, N_MMA = 256, N_DEC = 256, N_AW = 32;
constexpr int THREADS = N_MMA + N_DEC + N_AW;

struct Smem {
    unsigned char a[NS][BM * BKB];
    unsigned char asf[NS][BM * 8];
    unsigned char b[NS][BN * BKB];
    unsigned bsf[NS][BN];
    unsigned lut[kLutN];
    unsigned long long mb[4 * NS];
};

struct Legs {
    const unsigned char* w[MAX_LEGS];
    __nv_bfloat16* c[MAX_LEGS];
    size_t poff[MAX_LEGS];     // the leg's first float in the split-K partials, [z][m][n] inside
    int n[MAX_LEGS];
    int first[MAX_LEGS];       // the leg's first N-tile
    int nleg;
};

__device__ __forceinline__ int swz(int r, int ch) { return r * BKB + ((ch ^ ((r >> 1) & 3)) << 4); }
__device__ __forceinline__ unsigned smem_u32(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp16(unsigned dst, const void* src, int n) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(n) : "memory");
}
__device__ __forceinline__ void cp8(unsigned dst, const void* src, int n) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(dst), "l"(src), "r"(n) : "memory");
}
__device__ __forceinline__ void ldsm4(unsigned addr, unsigned& r0, unsigned& r1, unsigned& r2,
                                      unsigned& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_fp4(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2],
                                        unsigned sfa, unsigned sfb) {
    asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0),
          "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0));
}
__device__ __forceinline__ void mb_init(unsigned a, unsigned cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mb_arrive(unsigned a) {
    asm volatile("{ .reg .b64 st; mbarrier.arrive.shared::cta.b64 st, [%0]; }" :: "r"(a) : "memory");
}
__device__ __forceinline__ void mb_cp_arrive(unsigned a) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(a) : "memory");
}
__device__ __forceinline__ void mb_wait(unsigned a, unsigned parity) {
    asm volatile("{ .reg .pred p; W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @!p bra W%=; }"
                 :: "r"(a), "r"(parity) : "memory");
}

// Half h of weight row br's block, from its words tw (2h, 2h+1, 4+h and 6), into the row's 32 FP4
// bytes at 32h of its stage row; h = 0 also writes the row's ue4m3 scale, replicated over the four
// k groups of a 64-deep step. Twelve five-digit bytes and one four-digit byte -> 64 nibbles.
__device__ __forceinline__ void decode_half(const unsigned (&tw)[4], int h, int br,
                                            const unsigned* __restrict__ lut,
                                            unsigned char* __restrict__ dst,
                                            unsigned* __restrict__ bsf) {
    const float sb = __half2float(__ushort_as_half((unsigned short)(tw[3] >> 16)));
    unsigned code, sf;
    fp4_wscale(sb, code, sf);
    const unsigned* T = lut + ((sb < 0.f ? 2 : 0) + (code == 3u ? 1 : 0)) * 256;
    unsigned L[13];
#pragma unroll
    for (int i = 0; i < 12; ++i) L[i] = T[(tw[i >> 2] >> (8 * (i & 3))) & 0xFFu];
    L[12] = T[(tw[3] >> (8 * h)) & 0xFFu] & 0xFFFFu;
    uint4 w0, w1;
    w0.x = L[0] | (L[1] << 20);
    w0.y = (L[1] >> 12) | (L[2] << 8) | (L[3] << 28);
    w0.z = (L[3] >> 4) | (L[4] << 16);
    w0.w = (L[4] >> 16) | (L[5] << 4) | (L[6] << 24);
    w1.x = (L[6] >> 8) | (L[7] << 12);
    w1.y = L[8] | (L[9] << 20);
    w1.z = (L[9] >> 12) | (L[10] << 8) | (L[11] << 28);
    w1.w = (L[11] >> 4) | (L[12] << 16);
    *reinterpret_cast<uint4*>(dst + swz(br, 2 * h)) = w0;
    *reinterpret_cast<uint4*>(dst + swz(br, 2 * h + 1)) = w1;
    if (!h) *bsf = sf * 0x01010101u;
}

// One stage (two 64-deep k steps) of a warp's 64x32 tile. Each step loads all four A and four B
// fragments before its 16 MMAs: ldmatrix stays volatile (it must not rise above the mbarrier wait),
// so a fragment loaded between MMAs would expose its latency to the two warps a scheduler has.
__device__ __forceinline__ void mma_stage(const unsigned char* as, const unsigned char* asf,
                                          const unsigned char* bs, const unsigned* bsf, int wm,
                                          int wn, int lane, float (&acc)[4][4][4]) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
        unsigned bf[4][2], sfb[4], af[4][4], sfa[4];
#pragma unroll
        for (int g = 0; g < 2; g++) {
            const int row = wn * 32 + (2 * g + (lane >> 4)) * 8 + (lane & 7);
            ldsm4(smem_u32(bs + swz(row, 2 * j + ((lane >> 3) & 1))),
                  bf[2 * g][0], bf[2 * g][1], bf[2 * g + 1][0], bf[2 * g + 1][1]);
        }
#pragma unroll
        for (int f = 0; f < 4; f++) {
            const int row = wm * 64 + f * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            ldsm4(smem_u32(as + swz(row, 2 * j + (lane >> 4))), af[f][0], af[f][1], af[f][2],
                  af[f][3]);
        }
#pragma unroll
        for (int g = 0; g < 4; g++) sfb[g] = bsf[wn * 32 + g * 8 + (lane >> 2)];
#pragma unroll
        for (int f = 0; f < 4; f++)
            sfa[f] = *reinterpret_cast<const unsigned*>(
                asf + (wm * 64 + f * 16 + (lane & 1) * 8 + (lane >> 2)) * 8 + 4 * j);
#pragma unroll
        for (int f = 0; f < 4; f++)
#pragma unroll
            for (int g = 0; g < 4; g++) mma_fp4(acc[f][g], af[f], bf[g], sfa[f], sfb[g]);
    }
}

// A warp's 64x32 tile out: C fragment rows lane/4 and lane/4 + 8, columns 2*(lane%4) and +1.
// With `part` the raw sums go to split-K slice z, else alpha * acc (+ C) to the leg's bf16 C.
template <bool RESID>
__device__ __forceinline__ void store_tile(const float (&acc)[4][4][4], const Legs& L, int leg,
                                           float* __restrict__ part, int z, int M, int m0, int n0,
                                           int wm, int wn, int lane, float alpha) {
    const int N = L.n[leg];
#pragma unroll
    for (int f = 0; f < 4; f++)
#pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            const int row = m0 + wm * 64 + f * 16 + (lane >> 2) + hh * 8;
            if (row >= M) continue;
#pragma unroll
            for (int g = 0; g < 4; g++) {
                const int col = n0 + wn * 32 + g * 8 + (lane & 3) * 2;
                const float v0 = acc[f][g][2 * hh], v1 = acc[f][g][2 * hh + 1];
                if (part) {
                    float* P = part + L.poff[leg] + ((size_t)z * M + row) * N + col;
                    *reinterpret_cast<float2*>(P) = make_float2(v0, v1);
                } else {
                    __nv_bfloat162* C =
                        reinterpret_cast<__nv_bfloat162*>(L.c[leg] + (size_t)row * N + col);
                    float2 o = make_float2(v0 * alpha, v1 * alpha);
                    if (RESID) {
                        const float2 p = __bfloat1622float2(*C);
                        o.x += p.x; o.y += p.y;
                    }
                    *C = __float22bfloat162_rn(o);
                }
            }
        }
}

template <bool RESID>
__global__ void __launch_bounds__(THREADS, 1)
ptq1_fp4_gemm_kernel(const unsigned char* __restrict__ a, int M, int K, Legs L,
                     float* __restrict__ part, int nblk_split, int mtiles, int ntiles, int nitems,
                     float alpha) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    const unsigned mb0 = smem_u32(&sm.mb[0]);
    auto fullA  = [&](int s) { return mb0 + 8u * (unsigned)s; };
    auto emptyA = [&](int s) { return mb0 + 8u * (unsigned)(NS + s); };
    auto fullB  = [&](int s) { return mb0 + 8u * (unsigned)(2 * NS + s); };
    auto emptyB = [&](int s) { return mb0 + 8u * (unsigned)(3 * NS + s); };
    const int tid = threadIdx.x;
    for (int e = tid; e < kLutN; e += THREADS) sm.lut[e] = fp4_lut_entry(e >> 8, e & 0xFF);
    if (tid == 0) {
        for (int s = 0; s < NS; s++) {
            mb_init(fullA(s), N_AW);
            mb_init(emptyA(s), N_MMA / 32);
            mb_init(fullB(s), N_DEC);
            mb_init(emptyB(s), N_MMA / 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const int nblk = K / kBlk;
    auto item = [&](int w, int& mt, int& leg, int& n0, int& kb0, int& nst) {
        mt = w % mtiles;
        const int r = w / mtiles, nt = r % ntiles, z = r / ntiles;
        leg = 0;
#pragma unroll
        for (int i = 1; i < MAX_LEGS; i++)
            if (i < L.nleg && nt >= L.first[i]) leg = i;
        n0 = (nt - L.first[leg]) * BN;
        kb0 = z * nblk_split;
        nst = min(nblk, kb0 + nblk_split) - kb0;
    };

    if (tid >= N_MMA + N_DEC) {
        // ---------------- A warp: 16 A chunks and 4 scale chunks a lane per stage ----------------
        const int lane = tid - (N_MMA + N_DEC);
        const size_t arow = (size_t)K / 2, asrow = (size_t)K / 16;
        const unsigned char* asf_g = a + (size_t)M * arow;
        int g = 0;
        for (int w = blockIdx.x; w < nitems; w += gridDim.x) {
            int mt, leg, n0, kb0, nst;
            item(w, mt, leg, n0, kb0, nst);
            const int m0 = mt * BM;
            for (int i = 0; i < nst; i++, g++) {
                const int s = g % NS;
                if (g >= NS) mb_wait(emptyA(s), ((g / NS) - 1) & 1);
                const int kb = kb0 + i;
#pragma unroll
                for (int u = 0; u < 16; u++) {
                    const int c = lane + 32 * u, r = c >> 2, ch = c & 3;
                    const bool ok = m0 + r < M;
                    cp16(smem_u32(&sm.a[s][swz(r, ch)]),
                         a + (size_t)(ok ? m0 + r : 0) * arow + (size_t)kb * BKB + ch * 16,
                         ok ? 16 : 0);
                }
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    const int r = lane + 32 * u;
                    const bool ok = m0 + r < M;
                    cp8(smem_u32(&sm.asf[s][r * 8]),
                        asf_g + (size_t)(ok ? m0 + r : 0) * asrow + (size_t)kb * 8, ok ? 8 : 0);
                }
                mb_cp_arrive(fullA(s));
            }
        }
        return;
    }
    if (tid >= N_MMA) {
        // ---------------- decode warps: one half-block a thread per stage ----------------
        const int d = tid - N_MMA, br = d >> 1, h = d & 1;
        int g = 0;
        for (int w = blockIdx.x; w < nitems; w += gridDim.x) {
            int mt, leg, n0, kb0, nst;
            item(w, mt, leg, n0, kb0, nst);
            const unsigned* wrow = reinterpret_cast<const unsigned*>(
                L.w[leg] + (size_t)(n0 + br) * nblk * kBlkBytes + (size_t)kb0 * kBlkBytes);
            // Two stages an iteration (independent decode chains), their words (2h, 2h+1, 4+h and 6
            // of the block) a pair ahead.
            unsigned tw[2][4], nx[2][4];
            auto fetch = [&](int i, unsigned (&t)[4]) {
                if (i >= nst) return;
                const unsigned* bw = wrow + (size_t)i * (kBlkBytes / 4);
                t[0] = __ldg(bw + 2 * h);
                t[1] = __ldg(bw + 2 * h + 1);
                t[2] = __ldg(bw + 4 + h);
                t[3] = __ldg(bw + 6);
            };
            fetch(0, tw[0]);
            fetch(1, tw[1]);
            for (int i = 0; i < nst; i += 2) {
                fetch(i + 2, nx[0]);
                fetch(i + 3, nx[1]);
                const bool two = i + 1 < nst;
                const int s0 = g % NS, s1 = (g + 1) % NS;
                if (g >= NS) mb_wait(emptyB(s0), ((g / NS) - 1) & 1);
                if (two && g + 1 >= NS) mb_wait(emptyB(s1), (((g + 1) / NS) - 1) & 1);
                decode_half(tw[0], h, br, sm.lut, sm.b[s0], &sm.bsf[s0][br]);
                if (two) decode_half(tw[1], h, br, sm.lut, sm.b[s1], &sm.bsf[s1][br]);
                mb_arrive(fullB(s0));
                if (two) mb_arrive(fullB(s1));
                g += two ? 2 : 1;
#pragma unroll
                for (int e = 0; e < 4; e++) { tw[0][e] = nx[0][e]; tw[1][e] = nx[1][e]; }
            }
        }
        return;
    }

    // ---------------- MMA warps ----------------
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 1, wn = warp >> 1;
    int g = 0;
    for (int w = blockIdx.x; w < nitems; w += gridDim.x) {
        int mt, leg, n0, kb0, nst;
        item(w, mt, leg, n0, kb0, nst);
        float acc[4][4][4];
#pragma unroll
        for (int f = 0; f < 4; f++)
#pragma unroll
            for (int q = 0; q < 4; q++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[f][q][e] = 0.f;
        for (int i = 0; i < nst; i++, g++) {
            const int s = g % NS;
            const unsigned par = (g / NS) & 1;
            mb_wait(fullB(s), par);
            mb_wait(fullA(s), par);
            mma_stage(sm.a[s], sm.asf[s], sm.b[s], sm.bsf[s], wm, wn, lane, acc);
            __syncwarp();
            if (lane == 0) { mb_arrive(emptyA(s)); mb_arrive(emptyB(s)); }
        }
        store_tile<RESID>(acc, L, leg, part, kb0 / nblk_split, M, mt * BM, n0, wm, wn, lane,
                          alpha);
    }
}

// Split-K partials [z][m*n] -> C, the slices summed in order.
template <bool RESID>
__global__ void ptq1_fp4_reduce_kernel(const float4* __restrict__ p, __nv_bfloat16* __restrict__ c,
                                       size_t n4, int splits, float alpha) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        float4 s = p[i];
        for (int z = 1; z < splits; z++) {
            const float4 q = p[(size_t)z * n4 + i];
            s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
        }
        __nv_bfloat162* C = reinterpret_cast<__nv_bfloat162*>(c) + 2 * i;
        float2 lo = make_float2(s.x * alpha, s.y * alpha), hi = make_float2(s.z * alpha, s.w * alpha);
        if (RESID) {
            const float2 a = __bfloat1622float2(C[0]), b = __bfloat1622float2(C[1]);
            lo.x += a.x; lo.y += a.y; hi.x += b.x; hi.y += b.y;
        }
        C[0] = __float22bfloat162_rn(lo);
        C[1] = __float22bfloat162_rn(hi);
    }
}

int sm_count() {
    static int n = [] {
        int dev = 0, v = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
        return v > 0 ? v : 1;
    }();
    return n;
}

// Split K only while the persistent grid would leave SMs short: pick the slice count whose last
// round of items is fullest, each extra slice paying for its partials and the reduce.
// SPARKINFER_PTQ1_FP4_SPLITK=<n> forces n (A/B).
int pick_splits(int tiles, int nblk, size_t per_split_floats, size_t part_cap) {
    static const int env = [] {
        const char* e = getenv("SPARKINFER_PTQ1_FP4_SPLITK");
        return e ? atoi(e) : 0;
    }();
    const int smax = nblk / 4 < 16 ? nblk / 4 : 16;
    auto fits = [&](int s) { return (size_t)s * per_split_floats <= part_cap; };
    if (env > 0) return (env <= smax && fits(env)) ? env : 1;
    const double sms = sm_count();
    int best = 1;
    double best_score = -1.0;
    for (int s = 1; s <= smax && fits(s); s++) {
        const double w = tiles * (double)s / sms;
        const double score = w / ceil(w) - 0.08 * (s - 1);
        if (score > best_score + 1e-9) { best_score = score; best = s; }
    }
    return best;
}

}  // namespace

bool ptq1_fp4_gemm_supported(int m, int k) {
    static const bool dev_ok = [] {
        int dev = 0, major = 0, minor = 0;
        return cudaGetDevice(&dev) == cudaSuccess &&
               cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
               cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess &&
               major == 12 && minor == 0;
    }();
    return dev_ok && m > 0 && k > 0 && k % kSpan == 0;
}

size_t ptq1_fp4_act_bytes(int rows, int k) {
    return (size_t)rows * k / 2 + (size_t)rows * k / 16;
}

bool launch_ptq1_rotq_fp4(const void* x_bf16, const void* up_bf16, const signed char* sign,
                          void* a, int rows, int k, int block, cudaStream_t st) {
    if (rows <= 0 || block != kSpan || k % kSpan != 0 || !ptq1_fp4_gemm_supported(rows, k))
        return false;
    // Fewer rows than SMs: a CTA per (row, span). A CTA per row walks its row's 5-17 spans back
    // to back, and at 128 rows that is 128 CTAs on 170 SMs, each a serial chain of span latencies
    // (the SwiGLU leg's 17 spans ran 14.2 us for 9 MB; split, 6.0 us). From the SM count up the
    // row grid already covers the device and keeps its launch.
    // SPARKINFER_PTQ1_ROTQ_SPANS=0 restores a CTA per row everywhere.
    static const bool span_grid = [] {
        const char* e = getenv("SPARKINFER_PTQ1_ROTQ_SPANS");
        return !(e && e[0] == '0');
    }();
    const dim3 grid(rows, span_grid && rows < sm_count() ? k / kSpan : 1);
    const auto* x = static_cast<const __nv_bfloat16*>(x_bf16);
    auto* q = static_cast<unsigned char*>(a);
    if (up_bf16)
        ptq1_rotq_fp4_kernel<true><<<grid, 256, 0, st>>>(
            x, static_cast<const __nv_bfloat16*>(up_bf16), sign, q, rows, k);
    else
        ptq1_rotq_fp4_kernel<false><<<grid, 256, 0, st>>>(x, nullptr, sign, q, rows, k);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launch_ptq1_fp4_gemm(const void* a, int m, int k, const void* const* w, void* const* c,
                          const int* n, int nleg, bool resid, float* part, size_t part_cap,
                          cudaStream_t st) {
    if (!a || nleg < 1 || nleg > MAX_LEGS || !ptq1_fp4_gemm_supported(m, k)) return false;
    Legs L{};
    L.nleg = nleg;
    int tiles_n = 0;
    size_t sum_n = 0;
    for (int i = 0; i < nleg; i++) {
        if (!w[i] || !c[i] || n[i] <= 0 || n[i] % BN) return false;
        L.w[i] = static_cast<const unsigned char*>(w[i]);
        L.c[i] = static_cast<__nv_bfloat16*>(c[i]);
        L.n[i] = n[i];
        L.first[i] = tiles_n;
        tiles_n += n[i] / BN;
        sum_n += (size_t)n[i];
    }
    const int mtiles = (m + BM - 1) / BM;
    const int nblk = k / kBlk;
    int splits = part ? pick_splits(mtiles * tiles_n, nblk, (size_t)m * sum_n, part_cap) : 1;
    const int per = (nblk + splits - 1) / splits;
    splits = (nblk + per - 1) / per;   // every launched slice owns at least one block
    size_t off = 0;
    for (int i = 0; i < nleg; i++) { L.poff[i] = off; off += (size_t)splits * m * n[i]; }
    const float alpha = 1.f / kWScale;
    const int nitems = mtiles * tiles_n * splits;
    const int grid = nitems < sm_count() ? nitems : sm_count();
    constexpr size_t smem = sizeof(Smem);
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(ptq1_fp4_gemm_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncSetAttribute(ptq1_fp4_gemm_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        attr = true;
    }
    const auto* A = static_cast<const unsigned char*>(a);
    float* P = splits > 1 ? part : nullptr;
    if (resid && !P)
        ptq1_fp4_gemm_kernel<true><<<grid, THREADS, smem, st>>>(A, m, k, L, P, per, mtiles,
                                                                tiles_n, nitems, alpha);
    else
        ptq1_fp4_gemm_kernel<false><<<grid, THREADS, smem, st>>>(A, m, k, L, P, per, mtiles,
                                                                 tiles_n, nitems, alpha);
    if (P) {
        for (int i = 0; i < nleg; i++) {
            const size_t n4 = (size_t)m * n[i] / 4;
            const int blocks = (int)((n4 + 255) / 256 < 4096 ? (n4 + 255) / 256 : 4096);
            const auto* p4 = reinterpret_cast<const float4*>(part + L.poff[i]);
            if (resid)
                ptq1_fp4_reduce_kernel<true><<<blocks, 256, 0, st>>>(p4, L.c[i], n4, splits, alpha);
            else
                ptq1_fp4_reduce_kernel<false><<<blocks, 256, 0, st>>>(p4, L.c[i], n4, splits, alpha);
        }
    }
    return cudaPeekAtLastError() == cudaSuccess;
}

}}  // namespace sparkinfer::kernels
