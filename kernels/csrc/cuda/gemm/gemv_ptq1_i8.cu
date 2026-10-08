// PTQ1_0 ternary weights against an int8 activation: the kernels that let Ternary-Bonsai-2 read
// its FFN in the stored 1.75-bit blocks -- gate/up on every path, and the decode shadow's down in
// the packed batch -- and the decode shadow's attention and output projections.
//
//   decode      ptq1_rotq_kernel takes the activation into the weights' basis (sign flip, then a
//               1024-point Hadamard) and quantizes it to int8 in the same pass, one scale per 128
//               values -- the same 128 the weight blocks use, so a block's dot product is one exact
//               integer. ptq1_gemv_i8_kernel multiplies trit digits against it with dp4a and
//               folds the two scales once per block.
//   packed      ptq1_mma_rows_kernel: the same arithmetic for up to 32 rows on the int8 tensor
//               cores, each weight block decoded once for all of them, with k split across CTAs
//               when the matrix alone cannot fill the device.
//   prefill     launch_ptq1_rows_i8 / launch_ptq1_rotq_rows_i8: the per-row int8 operands the
//               existing int8 GEMMs read (and the fused GEMM's PTQ1 arm decodes to).
//
// The weights are read exactly as the checkpoint stores them; the arithmetic's own rounding is
// the activation's int8 step (and, in prefill, the per-row weight scale).
#include "sparkinfer/kernels/ternary.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include <cstdlib>
#include <mutex>

namespace sparkinfer { namespace kernels {

namespace {

constexpr int kBlk = 128;         // weights per PTQ1_0 block, and activation values per scale
constexpr int kBlkBytes = 28;
constexpr int kSpan = 1024;       // Hadamard span of this checkpoint

// Programmatic dependent launch. A packed rows kernel's weights do not depend on the rotation
// that feeds it, so it is launched programmatic and streams its first weight stage in while
// that kernel still runs, then waits for it before touching the activation. Both are no-ops
// for a kernel launched the ordinary way.
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
}
// Byte offset of the ue4m3 scale for (row r, 16-value group g) in the CUTLASS sm1xx block-scaled
// layout (launch_ct_nvfp4_pack_sfb / launch_nvfp4_pack_sfa's target): 128-row x 4-group atoms of
// 512 bytes, the K atoms fastest; inside one, row r%32 strides 16, r%128/32 strides 4, g%4 is 1.
__device__ __forceinline__ size_t sf_cutlass_off(int r, int g, int ng) {
    return ((size_t)(r >> 7) * (size_t)(ng >> 2) + (size_t)(g >> 2)) * 512 +
           (size_t)((r & 31) * 16 + ((r >> 5) & 3) * 4 + (g & 3));
}
// e2m1 code of v, nearest-even, saturating at 6, sign kept (-0 and a negative underflow give 8, a
// NaN gives +6): what __nv_cvt_float2_to_fp4x2(.., __NV_E2M1, cudaRoundNearest) returns for each
// half. That intrinsic is one cvt only on an arch-specific target (sm_120a, as si_nvfp4 builds);
// this file builds for plain sm_120 with the rest of si_gemm, where cuda_fp4.hpp widens every
// value to double and rounds it in software -- FP64 at 1/64 rate on a GeForce part, and the
// long-prompt FP4 activation below converts every value of every leg. The thresholds are e2m1's
// midpoints (0.25 .. 5), each tie going to the even code, so the codes are the intrinsic's. The
// code is the count of thresholds the magnitude clears, summed: the same codes as a chain of
// selects, which compiled to divergent branches -- the rotations' largest instruction share.
__device__ __forceinline__ unsigned e2m1_rn(float v) {
    const float a = fabsf(v);
    const unsigned c = (unsigned)(a > 0.25f) + (unsigned)(a >= 0.75f) + (unsigned)(a > 1.25f) +
                       (unsigned)(a >= 1.75f) + (unsigned)(a > 2.5f) + (unsigned)(a >= 3.5f) +
                       (unsigned)(a > 5.f);
    return a != a ? 7u : c | ((__float_as_uint(v) >> 28) & 8u);
}
// The pair packed as the intrinsic packs it: x in the low nibble, y in the high one.
__device__ __forceinline__ unsigned e2m1x2_rn(float x, float y) {
    return e2m1_rn(x) | (e2m1_rn(y) << 4);
}

// ---------------------------------------------------------------------------------------------
// Rotate + quantize. One CTA per (1024-span, row), 256 threads holding four consecutive values
// each: bits 0-1 of the in-span index are the value, bits 2-6 the lane, bits 7-9 the warp, so the
// ten butterfly stages are two in registers, five across lanes and three across warps. Each warp
// then owns exactly one 128-value quantization block.
__device__ __forceinline__ void ld4_bf16(const __nv_bfloat16* p, float v[4]) {
    const uint2 raw = *reinterpret_cast<const uint2*>(p);
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
    v[0] = __low2float(a); v[1] = __high2float(a); v[2] = __low2float(b); v[3] = __high2float(b);
}

// What is rotated, each rounded to bf16 exactly as the kernel it replaces writes it (every
// translation unit involved is fast-math), so the result equals that kernel followed by this one:
//   kRotPlain   x itself.
//   kRotSwiglu  x the gate, u the up projection: launch_prefill_swiglu's bf16(g/(1+exp(-g)) * u).
//   kRotGnorm   x the GDN output, u its z gate: the gated RMSNorm of one 128-wide v head per warp,
//               bf16(x * rsqrt(mean(x^2) + eps) * nw * silu(z)), the square sum formed in
//               gated_norm_warp_kernel's lane order so the norm is that kernel's to the bit.
//   kRotGate    x the attention output, u its gate: launch_qwen36_mul_sigmoid's bf16(x*sigmoid(u)).
//   kRotAddNorm x + u through add_rmsnorm2_q8 with weight nw: that kernel's sum, norm and Q8_1
//               written for this CTA's span (out_sum, out_norm, out_q8), and its norm rotated. The
//               row's square sum is formed in the 640-thread kernel's own order: virtual thread v
//               owns elements 8v..8v+7, virtual warps fold with the same xor tree, and so do their
//               partials.
//   kRotNorm    x through rmsnorm_kernel with weight nw (prefill rows only, below).
enum : int { kRotPlain = 0, kRotSwiglu = 1, kRotGnorm = 2, kRotGate = 3, kRotAddNorm = 4, kRotNorm = 5 };
struct i8_blk_q8_1 { __half2 ds; signed char qs[32]; };
template <int MODE>
__global__ void __launch_bounds__(256)
ptq1_rotq_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ u,
                 const __nv_bfloat16* __restrict__ nw, float eps,
                 const signed char* __restrict__ sign, signed char* __restrict__ q,
                 float* __restrict__ qd, int* __restrict__ qs, int k,
                 __nv_bfloat16* __restrict__ out_sum = nullptr,
                 __nv_bfloat16* __restrict__ out_norm = nullptr,
                 i8_blk_q8_1* __restrict__ out_q8 = nullptr) {
    __shared__ float sh[kSpan];
    pdl_wait();      // launched programmatic (rotq_launch): x and u are the kernel before's output
    pdl_trigger();   // the rows kernel after this may start fetching its weights
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const long row = blockIdx.y;
    const int e0 = blockIdx.x * kSpan + t * 4;
    const size_t off = (size_t)row * k + e0;
    float v[4];
    ld4_bf16(x + off, v);
    if (MODE == kRotSwiglu) {
        float w[4];
        ld4_bf16(u + off, w);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            v[i] = __bfloat162float(__float2bfloat16(v[i] / (1.f + __expf(-v[i])) * w[i]));
    } else if (MODE == kRotGate) {
        float g[4];
        ld4_bf16(u + off, g);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            v[i] = __bfloat162float(__float2bfloat16(v[i] * (1.f / (1.f + __expf(-g[i])))));
    } else if (MODE == kRotGnorm) {
        const __nv_bfloat16* hx = x + (off - (size_t)lane * 4);   // this warp's head
        float ss = 0.f;
#pragma unroll
        for (int r = 0; r < kBlk / 32; ++r) {
            const float xr = __bfloat162float(hx[lane + 32 * r]);
            ss += xr * xr;
        }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
        const float inv = rsqrtf(ss / kBlk + eps);
        float z[4], w[4];
        ld4_bf16(u + off, z);
        ld4_bf16(nw + lane * 4, w);
#pragma unroll
        for (int i = 0; i < 4; ++i)
            v[i] = __bfloat162float(
                __float2bfloat16(v[i] * inv * w[i] * (z[i] / (1.f + __expf(-z[i])))));
    } else if (MODE == kRotAddNorm) {
        __shared__ float s_warp[32];
        const int nvw = k / 256;                  // add_rmsnorm2_q8's warps: k/8 threads
        constexpr int kVw = 8192 / 256 / 8;       // virtual warps per real warp, at most
        const uint4* x8 = reinterpret_cast<const uint4*>(x + (size_t)row * k);
        const uint4* r8 = reinterpret_cast<const uint4*>(u + (size_t)row * k);
        uint4 xp[kVw], rp[kVw];
#pragma unroll
        for (int i = 0; i < kVw; ++i) {
            const int vw = warp + 8 * i;
            if (vw < nvw) { xp[i] = __ldg(x8 + vw * 32 + lane); rp[i] = __ldg(r8 + vw * 32 + lane); }
        }
        float rv[4], wv[4];
        ld4_bf16(u + off, rv);
        ld4_bf16(nw + e0, wv);
#pragma unroll
        for (int i = 0; i < kVw; ++i) {
            const int vw = warp + 8 * i;
            if (vw < nvw) {
                const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xp[i]);
                const __nv_bfloat16* rh = reinterpret_cast<const __nv_bfloat16*>(&rp[i]);
                float ss = 0.f;
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    const float sv = __bfloat162float(xh[j]) + __bfloat162float(rh[j]);
                    ss = __fmaf_rn(sv, sv, ss);
                }
#pragma unroll
                for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
                if (lane == 0) s_warp[vw] = ss;
            }
        }
        __syncthreads();
        float red = lane < nvw ? s_warp[lane] : 0.f;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) red += __shfl_xor_sync(0xffffffffu, red, m);
        const float inv_rms = rsqrtf(red / k + eps);
        float sv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sv[i] = v[i] + rv[i];
            const float svb = __bfloat162float(__float2bfloat16(sv[i]));
            v[i] = __bfloat162float(__float2bfloat16(svb * inv_rms * wv[i]));
        }
        const __nv_bfloat162 s01 = __floats2bfloat162_rn(sv[0], sv[1]);
        const __nv_bfloat162 s23 = __floats2bfloat162_rn(sv[2], sv[3]);
        const __nv_bfloat162 n01 = __floats2bfloat162_rn(v[0], v[1]);
        const __nv_bfloat162 n23 = __floats2bfloat162_rn(v[2], v[3]);
        *reinterpret_cast<uint2*>(out_sum + off) =
            make_uint2(*reinterpret_cast<const unsigned*>(&s01), *reinterpret_cast<const unsigned*>(&s23));
        *reinterpret_cast<uint2*>(out_norm + off) =
            make_uint2(*reinterpret_cast<const unsigned*>(&n01), *reinterpret_cast<const unsigned*>(&n23));
        if (out_q8) {
            // Q8_1 of the bf16-rounded norm: a 32-block is 8 consecutive threads here.
            float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
            const float d = amax / 127.0f;
            int s8 = 0;
            unsigned word = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int qi = (amax == 0.0f) ? 0 : (int)roundf(v[i] / d);
                s8 += qi;
                word |= ((unsigned)(unsigned char)(signed char)qi) << (8 * i);
            }
            i8_blk_q8_1* blk = out_q8 + (off >> 5);
            reinterpret_cast<unsigned*>(blk->qs)[t & 7] = word;
            s8 += __shfl_xor_sync(0xffffffffu, s8, 1);
            s8 += __shfl_xor_sync(0xffffffffu, s8, 2);
            s8 += __shfl_xor_sync(0xffffffffu, s8, 4);
            if ((t & 7) == 0) blk->ds = __floats2half2_rn(d, d * (float)s8);
        }
    }
    const char4 sg = *reinterpret_cast<const char4*>(sign + e0);
    v[0] *= (float)sg.x; v[1] *= (float)sg.y; v[2] *= (float)sg.z; v[3] *= (float)sg.w;
    // bits 0 and 1
    float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
    v[0] = a0 + a2; v[2] = a0 - a2; v[1] = a1 + a3; v[3] = a1 - a3;
    // bits 2..6: across lanes
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool hi = lane & m;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float p = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = hi ? p - v[i] : v[i] + p;
        }
    }
    // bits 7..9: across warps, through shared memory, one 8-point transform per column
#pragma unroll
    for (int i = 0; i < 4; ++i) sh[t * 4 + i] = v[i];
    __syncthreads();
    if (t < 128) {
        float c[8];
#pragma unroll
        for (int w = 0; w < 8; ++w) c[w] = sh[t + 128 * w];
#pragma unroll
        for (int len = 1; len < 8; len <<= 1)
#pragma unroll
            for (int w = 0; w < 8; ++w)
                if (!(w & len)) { const float p0 = c[w], p1 = c[w + len]; c[w] = p0 + p1; c[w + len] = p0 - p1; }
#pragma unroll
        for (int w = 0; w < 8; ++w) sh[t + 128 * w] = c[w];
    }
    __syncthreads();
    constexpr float kNorm = 0.03125f;   // 1/sqrt(1024)
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = sh[t * 4 + i] * kNorm;
    float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
    for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float id = am > 0.f ? 127.f / am : 0.f;
    const int q0 = __float2int_rn(v[0] * id), q1 = __float2int_rn(v[1] * id);
    const int q2 = __float2int_rn(v[2] * id), q3 = __float2int_rn(v[3] * id);
    int sum = q0 + q1 + q2 + q3;
#pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    *reinterpret_cast<char4*>(q + off) = make_char4((signed char)q0, (signed char)q1,
                                                    (signed char)q2, (signed char)q3);
    if (lane == 0) {
        const size_t b = (size_t)row * (k / kBlk) + blockIdx.x * (kSpan / kBlk) + warp;
        qd[b] = am * (1.f / 127.f);
        qs[b] = sum;
    }
}

// ---------------------------------------------------------------------------------------------
// GEMV. Trit digits are recovered four at a time: the four carrier bytes of a word are split into
// even and odd bytes, each held in the low byte of a 16-bit half, so multiplying by 3 keeps every
// carrier in its own half (255*3 < 65536). After the multiply the digit for multiplier 3^m sits in
// the half's high byte and the masked low byte is the carrier for 3^(m+1) -- two operations per
// half per digit, one byte_perm to gather the four digits. dp4a takes the digits unsigned (0..2);
// the -1 offset comes back as the activation block's integer sum.
__device__ __forceinline__ int dp4a_us(unsigned a, int b, int c) {
    int r;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

template <int STRIDE>
__device__ __forceinline__ int word_dot5(unsigned w, const int* a, int j0, int acc) {
    unsigned ql = w & 0x00FF00FFu, qh = (w >> 8) & 0x00FF00FFu;
#pragma unroll
    for (int m = 0; m < 5; ++m) {
        const unsigned ul = ql * 3u, uh = qh * 3u;
        acc = dp4a_us(__byte_perm(ul, uh, 0x7351), a[j0 + STRIDE * m], acc);
        ql = ul & 0x00FF00FFu;
        qh = uh & 0x00FF00FFu;
    }
    return acc;
}

// sum_k digit_k * a_k over one 128-trit block, digits in 0..2. Word layout (see ternary_ptq1.h):
// bytes 0..15 carry trits m*16+byte, bytes 16..23 carry 80+m*8+(byte-16), bytes 24..25 carry
// 120+2m+(byte-24); every (word, m) pair is four CONSECUTIVE trits, i.e. one int32 of activation.
__device__ __forceinline__ int ptq1_block_dot(const unsigned* w, const int* a) {
    int acc = 0;
    acc = word_dot5<4>(w[0], a, 0, acc);
    acc = word_dot5<4>(w[1], a, 1, acc);
    acc = word_dot5<4>(w[2], a, 2, acc);
    acc = word_dot5<4>(w[3], a, 3, acc);
    acc = word_dot5<2>(w[4], a, 20, acc);
    acc = word_dot5<2>(w[5], a, 21, acc);
    // The two four-trit carriers: activation word 30 is {b24 m0, b25 m0, b24 m1, b25 m1}, 31 the
    // same for m = 2, 3.
    const unsigned x = (w[6] & 0xFFu) | ((w[6] & 0xFF00u) << 8);
    const unsigned u0 = x * 3u, u1 = (u0 & 0x00FF00FFu) * 3u;
    const unsigned u2 = (u1 & 0x00FF00FFu) * 3u, u3 = (u2 & 0x00FF00FFu) * 3u;
    acc = dp4a_us(__byte_perm(u0, u1, 0x7531), a[30], acc);
    acc = dp4a_us(__byte_perm(u2, u3, 0x7531), a[31], acc);
    return acc;
}

template <typename OutT> __device__ __forceinline__ void put(OutT* y, size_t i, float v);
template <> __device__ __forceinline__ void put<float>(float* y, size_t i, float v) { y[i] = v; }
template <> __device__ __forceinline__ void put<__nv_bfloat16>(__nv_bfloat16* y, size_t i, float v) {
    y[i] = __float2bfloat16(v);
}

// One block's contribution, and the order contributions are summed in. Both kernels below --
// the GEMV and the tensor-core rows kernel -- use exactly this, so a row comes out bit-identical
// whichever of them computed it and however many other rows shared the launch:
//   * a step is KB = 4 consecutive blocks; its terms add in block order,
//   * steps add to a running sum in order, starting from zero,
//   * with k split S ways (row_splits, a function of the weight shape alone), each split sums its
//     own steps that way and the splits then add in split order.
// Written with _rn intrinsics so no contraction into an FMA can differ between the two.
constexpr int kStepBlocks = 4;
__device__ __forceinline__ float blk_term(float sw, float d, int dot) {
    return __fmul_rn(__fmul_rn(sw, d), (float)dot);
}

// Fewest waves x steps per CTA for 128-row CTAs at one per SM, ties to fewer splits, no split
// left empty. Down (5120 rows = 40 CTAs, 34 steps) splits 4 ways; gate/up (272 CTAs) never does.
int num_sms() {
    static const int sms = [] {
        int dev = 0, n = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n > 0 ? n : 170;
    }();
    return sms;
}

int row_splits(int n_rows, int nblk, int nmat) {
    const int sms = num_sms();
    const int ctas = (n_rows + 127) / 128 * nmat, nsteps = nblk / kStepBlocks;
    // Four waves and more (the LM head's 1940) already fill the device; a split there only
    // trades a wave-rounding sliver for a partial plane of vocab floats per row.
    if (ctas >= 4 * sms) return 1;
    int best = 1;
    long best_cost = (long)((ctas + sms - 1) / sms) * nsteps;
    for (int S = 2; S <= 8; ++S) {
        const int sps = (nsteps + S - 1) / S;
        if (sps < 4 || (S - 1) * sps >= nsteps) continue;
        const long cost = (long)((ctas * S + sms - 1) / sms) * sps;
        if (cost < best_cost) { best_cost = cost; best = S; }
    }
    return best;
}

// A warp owns R consecutive rows, i.e. R*nblk consecutive 28-byte blocks, and each lane takes one
// block per 32-block step, so the warp's weight loads cover one contiguous 896-byte run. The
// activation is staged once per CTA in shared memory: read straight from global, 32 lanes asking
// for 32 different 128-byte blocks cost 32 L1 wavefronts per load and were 60% of the kernel. The
// 16-byte units are XOR-swizzled by block so eight lanes reading eight consecutive blocks' unit u
// fall on eight different bank groups.
//
// Two weight matrices of one shape against one activation (gate and up) share a launch: CTAs past
// the first matrix's rows take the second.
//
// Summation order: see blk_term. A group of four lanes holds one step; its partial is formed in
// lane order, and the warp's eight step partials are added to the owning row's running sum one
// after another -- every lane carries the same sums, so the row's result is lane 0's.
template <int R, int WPC, bool SPLIT, typename OutT>
__global__ void __launch_bounds__(WPC * 32)
ptq1_gemv_i8_kernel(const signed char* __restrict__ xq, const float* __restrict__ xd,
                    const int* __restrict__ xs, const unsigned char* __restrict__ w0,
                    const unsigned char* __restrict__ w1, OutT* __restrict__ y0,
                    OutT* __restrict__ y1, int n_rows, int nblk, int ctas_per_mat, int sps) {
    extern __shared__ uint4 smem[];
    uint4* sa = smem;
    float* sd = reinterpret_cast<float*>(sa + nblk * 8);
    int* ss = reinterpret_cast<int*>(sd + nblk);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    {
        const uint4* g = reinterpret_cast<const uint4*>(xq);
        for (int i = threadIdx.x; i < nblk * 8; i += WPC * 32) {
            const int b = i >> 3, u = i & 7;
            sa[b * 8 + (u ^ (b & 7))] = __ldg(g + i);
        }
        for (int i = threadIdx.x; i < nblk; i += WPC * 32) {
            sd[i] = __ldg(xd + i);
            ss[i] = __ldg(xs + i);
        }
    }
    __syncthreads();
    int cta = blockIdx.x;
    const unsigned char* W = w0;
    OutT* y = y0;
    if (cta >= ctas_per_mat) { cta -= ctas_per_mat; W = w1; y = y1; }
    const int row0 = (cta * WPC + warp) * R;
    if (row0 >= n_rows) return;
    const int nr = n_rows - row0 < R ? n_rows - row0 : R;
    const int total = nr * nblk;
    const int steps = (total + 31) >> 5;
    const unsigned* wb = reinterpret_cast<const unsigned*>(W + (size_t)row0 * nblk * kBlkBytes);
    const int spr = nblk / kStepBlocks;   // steps per row
    int blk = lane % nblk;
    float acc[R], tot[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = tot[r] = 0.f;
    // Where the warp's next step lands: row r, step sr within it, cnt steps left in its split.
    // Uniform across the warp.
    int r = 0, sr = 0, cnt = sps;
    unsigned nw[7];
#pragma unroll
    for (int j = 0; j < 7; ++j) nw[j] = lane < total ? __ldg(wb + lane * 7 + j) : 0u;
    for (int st = 0; st < steps; ++st) {
        unsigned w[7];
#pragma unroll
        for (int j = 0; j < 7; ++j) w[j] = nw[j];
        const int nit = (st + 1) * 32 + lane;
#pragma unroll
        for (int j = 0; j < 7; ++j) nw[j] = nit < total ? __ldg(wb + (size_t)nit * 7 + j) : 0u;
        float v = 0.f;
        if (st * 32 + lane < total) {
            int a[32];
            const uint4* ap = sa + blk * 8;
            const int sx = blk & 7;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const uint4 tv = ap[u ^ sx];
                a[4 * u] = (int)tv.x; a[4 * u + 1] = (int)tv.y;
                a[4 * u + 2] = (int)tv.z; a[4 * u + 3] = (int)tv.w;
            }
            const int dot = ptq1_block_dot(w, a) - ss[blk];
            const float sw = __half2float(__ushort_as_half((unsigned short)(w[6] >> 16)));
            v = blk_term(sw, sd[blk], dot);
        }
        const float v1 = __shfl_down_sync(0xffffffffu, v, 1);
        const float v2 = __shfl_down_sync(0xffffffffu, v, 2);
        const float v3 = __shfl_down_sync(0xffffffffu, v, 3);
        const float stp_sum = __fadd_rn(__fadd_rn(__fadd_rn(v, v1), v2), v3);
#pragma unroll
        for (int j = 0; j < 32 / kStepBlocks; ++j) {
            const float pj = __shfl_sync(0xffffffffu, stp_sum, kStepBlocks * j);
            if (r >= nr) break;
            if (SPLIT && cnt == 0) {   // the row's next split begins
#pragma unroll
                for (int rr = 0; rr < R; ++rr)
                    if (rr == r) {
                        tot[rr] = sr == sps ? acc[rr] : __fadd_rn(tot[rr], acc[rr]);
                        acc[rr] = 0.f;
                    }
                cnt = sps;
            }
#pragma unroll
            for (int rr = 0; rr < R; ++rr)
                if (rr == r) acc[rr] = __fadd_rn(acc[rr], pj);
            --cnt;
            if (++sr == spr) { sr = 0; ++r; cnt = sps; }
        }
        blk += 32;
        while (blk >= nblk) blk -= nblk;
    }
    if (lane == 0) {
#pragma unroll
        for (int q = 0; q < R; ++q)
            if (q < nr) put<OutT>(y, (size_t)row0 + q, SPLIT ? __fadd_rn(tot[q], acc[q]) : acc[q]);
    }
}

template <int R, int WPC, bool SPLIT, typename OutT>
void launch_gemv_i8_t(const signed char* xq, const float* xd, const int* xs, const void* w0,
                      const void* w1, OutT* y0, OutT* y1, int n_rows, int nblk, int sps,
                      size_t shm, cudaStream_t st) {
    static bool attr = false;
    if (!attr && shm > 48 * 1024) {
        cudaFuncSetAttribute(ptq1_gemv_i8_kernel<R, WPC, SPLIT, OutT>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        attr = true;
    }
    const int ctas = (n_rows + R * WPC - 1) / (R * WPC);
    ptq1_gemv_i8_kernel<R, WPC, SPLIT, OutT><<<ctas * (w1 ? 2 : 1), WPC * 32, shm, st>>>(
        xq, xd, xs, static_cast<const unsigned char*>(w0), static_cast<const unsigned char*>(w1),
        y0, y1, n_rows, nblk, ctas, sps);
}

template <typename OutT>
bool launch_gemv_i8(const signed char* xq, const float* xd, const int* xs, const void* w0,
                    const void* w1, OutT* y0, OutT* y1, int n_rows, int k, cudaStream_t st) {
    if (n_rows <= 0 || k <= 0 || k % (kBlk * kStepBlocks) != 0) return false;
    constexpr int R = 2, WPC = 4;
    const int nblk = k / kBlk;
    const int spr = nblk / kStepBlocks;
    const int S = row_splits(n_rows, nblk, w1 ? 2 : 1);
    const int sps = (spr + S - 1) / S;
    const size_t shm = (size_t)nblk * 8 * 16 + (size_t)nblk * 8;
    if (shm > 96 * 1024) return false;
    if (S > 1)
        launch_gemv_i8_t<R, WPC, true, OutT>(xq, xd, xs, w0, w1, y0, y1, n_rows, nblk, sps, shm, st);
    else
        launch_gemv_i8_t<R, WPC, false, OutT>(xq, xd, xs, w0, w1, y0, y1, n_rows, nblk, sps, shm,
                                              st);
    return true;
}


// ---------------------------------------------------------------------------------------------
// A few activation rows (a packed decode step): mma.sync m16n8k32 with the weight digits as the
// unsigned A operand (16 weight rows) and the int8 activation as B (8 tokens). Per thread
// (g = lane/4, t = lane%4) and weight row, one block's eight A-fragment words are the digit
// groups 8s+t and 8s+4+t for k-steps s = 0..3: all five digits of carrier word t, three of word
// 4 + (t&1), and for t >= 2 one pair from the four-trit carriers. So every block is decoded once
// per weight row, by the four threads that own it, with no redundant work.
__device__ __forceinline__ void mma_u8s8(int* c, unsigned a0, unsigned a1, unsigned a2,
                                         unsigned a3, int b0, int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm_x4(const void* p, int& r0, int& r1, int& r2, int& r3) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}

__device__ __forceinline__ void digits5(unsigned w, unsigned (&d)[5]) {
    unsigned ql = w & 0x00FF00FFu, qh = (w >> 8) & 0x00FF00FFu;
#pragma unroll
    for (int m = 0; m < 5; ++m) {
        const unsigned ul = ql * 3u, uh = qh * 3u;
        d[m] = __byte_perm(ul, uh, 0x7351);
        ql = ul & 0x00FF00FFu;
        qh = uh & 0x00FF00FFu;
    }
}

// wt = word t, w45 = word 4 + (t & 1), w6 = the four-trit carriers + scale.
//
// Every lane decodes a carrier pair, so nothing here branches on t: t == 3 starts two trits in,
// at r_2 = 9 r_0 mod 256 in each 16-bit lane (the digit chain is r_{m+1} = 3 r_m mod 256), and a
// byte select keeps b[4] for t < 2. Selecting between the lanes' own chains instead compiled to a
// divergent branch per row and block, and the reconvergence points kept the scheduler from
// overlapping the decode with the MMAs around it.
__device__ __forceinline__ void row_frags(unsigned wt, unsigned w45, unsigned w6, int t,
                                          unsigned (&f)[4][2]) {
    unsigned a[5], b[5];
    digits5(wt, a);
    digits5(w45, b);
    f[0][0] = a[0]; f[0][1] = a[1]; f[1][0] = a[2]; f[1][1] = a[3]; f[2][0] = a[4];
    const bool hm = t >> 1;
    f[2][1] = hm ? b[1] : b[0];
    f[3][0] = hm ? b[3] : b[2];
    const unsigned x = (w6 & 0xFFu) | ((w6 & 0xFF00u) << 8);
    const unsigned xm = (x * (t == 3 ? 9u : 1u)) & 0x00FF00FFu;
    const unsigned u0 = xm * 3u, u1 = (u0 & 0x00FF00FFu) * 3u;
    f[3][1] = __byte_perm(b[4], __byte_perm(u0, u1, 0x7531), t < 2 ? 0x3210 : 0x7654);
}

// WARPS x 16 weight rows per CTA, NT tiles of 8 tokens (M <= 8*NT). Each pipeline step covers
// KB weight blocks: the CTA's weight rows and the activation for those blocks are brought into
// shared memory with 16-byte cp.async (a row's KB blocks are one contiguous, 16-byte aligned run
// when KB is a multiple of 4), ST steps in flight. Reading the weights straight into registers
// instead -- three scattered 4-byte loads per thread per block, one block ahead -- left the
// kernel latency-bound at ~60 us for gate+up whatever M was; staged, M <= 8 runs at ~28 us, about
// the single-row GEMV's time for the same bytes.
//
// SPLIT: blockIdx.y takes steps [y*sps, (y+1)*sps) and writes f32 partials to
// part[((mat * gridDim.y + y) * M + token) * N + row], which are then summed in split order. For the
// down projection, whose 5120 rows are only 40 CTAs on a 170-SM part. With `cnt` (one counter per
// tile, zero between launches) the tile's last CTA to finish sums them and resets its counter;
// without it ptq1_split_reduce_kernel does, as a second launch. Either way each output is
// part[0] + part[1] + ... in split order, so which CTA finishes last changes nothing.
//
// WARPS is 8 (128-row tiles, which the matrix divides) except for the balanced launches in
// launch_rows_i8, whose last tile may be short: its spare warps still stage and sync, and skip the
// math and the stores.
template <int NT, int WARPS, int KB, int ST, typename OutT, bool SPLIT = false>
__global__ void __launch_bounds__(WARPS * 32)
ptq1_mma_rows_kernel(const signed char* __restrict__ xq, const float* __restrict__ xd,
                     const int* __restrict__ xs, const unsigned char* __restrict__ w0,
                     const unsigned char* __restrict__ w1, OutT* __restrict__ y0,
                     OutT* __restrict__ y1, int M, int N, int nblk, int ctas_per_mat,
                     float* __restrict__ part = nullptr, int sps = 0,
                     unsigned* __restrict__ cnt = nullptr) {
    constexpr int TOK = NT * 8;
    constexpr int ROWB = KB * kBlk + 16;       // +16: a B-fragment load's 8 tokens hit 8 banks
    constexpr int WSEG = KB * kBlkBytes;       // one weight row's bytes per step
    constexpr int WCH = WSEG / 16;
    constexpr int WROWS = WARPS * 16;
    extern __shared__ __align__(16) unsigned char smem_mma[];
    unsigned char* sw = smem_mma;                                              // [ST][WROWS][WSEG]
    signed char* sx = reinterpret_cast<signed char*>(sw + ST * WROWS * WSEG);   // [ST][TOK][ROWB]
    // A token's scale and sum sit side by side, and consecutive tokens side by side, so the two
    // tokens a thread's C fragment holds are one 16-byte load: {d0, s0, d1, s1}.
    int* sds = reinterpret_cast<int*>(sx + ST * TOK * ROWB);              // [ST][KB][TOK][2]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int cta = blockIdx.x, mat = 0;
    const unsigned char* W = w0;
    OutT* y = y0;
    if (cta >= ctas_per_mat) { cta -= ctas_per_mat; W = w1; y = y1; mat = 1; }
    const int row0 = cta * WROWS;
    const int s_beg = SPLIT ? blockIdx.y * sps : 0;
    const int s_end = SPLIT ? min(nblk / KB, s_beg + sps) : nblk / KB;
    // A thread copies the same chunks every step; only b0 moves. Weight chunk i is row i / WCH,
    // unit i % WCH, and lands at i * 16 in its stage (WSEG == WCH * 16); activation chunk j is
    // token warp + WARPS * j, unit lane. So the offsets are formed once, not divided out per step.
    static_assert(KB * 8 == 32 && KB == 4, "issue map");
    constexpr bool FIT = WARPS == 8;   // the tile never runs past the matrix
    constexpr int NTH = WARPS * 32;
    constexpr int WJ = (WROWS * WCH + NTH - 1) / NTH;
    constexpr int XJ = (TOK + WARPS - 1) / WARPS;
    unsigned wsrc[WJ];
    unsigned wok = 0;   // !FIT: bit j set when chunk j is a row of this matrix
#pragma unroll
    for (int j = 0; j < WJ; ++j) {
        const int i = threadIdx.x + j * NTH, r = i / WCH, c = i - r * WCH;
        wsrc[j] = (unsigned)(r * nblk * kBlkBytes + c * 16);
        if (!FIT && i < WROWS * WCH && row0 + r < N) wok |= 1u << j;
    }
    const bool live = FIT || row0 + warp * 16 < N;
    const unsigned char* wbase = W + (size_t)row0 * nblk * kBlkBytes;
    const signed char* xbase = xq + (size_t)warp * nblk * kBlk + lane * 16;
    const size_t xstride = (size_t)WARPS * nblk * kBlk;
    // Tokens past M are zero in every stage from the start; no copy ever lands on them.
    if (M < TOK)
        for (int i = threadIdx.x; i < ST * TOK * KB * 8; i += NTH) {
            const int row = i / (KB * 8), tok = row % TOK;
            if (tok >= M)
                *reinterpret_cast<uint4*>(sx + (size_t)row * ROWB + (i % (KB * 8)) * 16) =
                    make_uint4(0, 0, 0, 0);
        }
    auto issue_w = [&](int stp, int buf) {
        unsigned char* sws = sw + (size_t)buf * WROWS * WSEG;
        const unsigned char* wstep = wbase + (size_t)stp * KB * kBlkBytes;
#pragma unroll
        for (int j = 0; j < WJ; ++j) {
            const int i = threadIdx.x + j * NTH;
            if (FIT ? (WROWS * WCH % NTH == 0 || i < WROWS * WCH) : ((wok >> j) & 1u))
                __pipeline_memcpy_async(sws + i * 16, wstep + wsrc[j], 16);
        }
    };
    auto issue_x = [&](int stp, int buf) {
        const int b0 = stp * KB;
        signed char* sxs = sx + ((size_t)buf * TOK + warp) * ROWB + lane * 16;
        const signed char* xstep = xbase + (size_t)b0 * kBlk;
#pragma unroll
        for (int j = 0; j < XJ; ++j)
            if (warp + WARPS * j < M)
                __pipeline_memcpy_async(sxs + j * WARPS * ROWB, xstep + j * xstride, 16);
        // Scales and sums are copied like the rest, so no warp stalls on a load before its MMAs,
        // but one 4-byte value at a time into their interleaved slots (see sds).
        for (int i = threadIdx.x; i < 2 * KB * TOK; i += NTH) {
            const int which = i & 1, tok = i >> 1 & (TOK - 1), bb = i / (2 * TOK);
            const bool ok = tok < M;
            const size_t src = (size_t)(ok ? tok : 0) * nblk + b0 + bb;
            int* dst = sds + (((size_t)buf * KB + bb) * TOK + tok) * 2 + which;
            if (which) __pipeline_memcpy_async(dst, xs + src, 4, ok ? 0 : 4);
            else       __pipeline_memcpy_async(dst, xd + src, 4, ok ? 0 : 4);
        }
    };
    auto issue = [&](int stp, int buf) {
        issue_w(stp, buf);
        issue_x(stp, buf);
        __pipeline_commit();
    };
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
    // The weights of the first stages are the kernel's own; only the activation comes from the
    // launch before it. So they are in flight before the wait (see pdl_wait), and each stage's
    // group commits once its activation is issued too.
#pragma unroll
    for (int s0 = 0; s0 < ST - 1; ++s0)
        if (s_beg + s0 < s_end) issue_w(s_beg + s0, s0);
    pdl_wait();
    pdl_trigger();
#pragma unroll
    for (int s0 = 0; s0 < ST - 1; ++s0) {
        if (s_beg + s0 < s_end) issue_x(s_beg + s0, s0);
        __pipeline_commit();
    }
    const int o45 = 4 + (t & 1);
    // ldmatrix row address of this lane: matrix lane / 8 of an x4 is k-step (lane / 16) of the
    // pair, low or high 16 bytes by bit 3, row lane % 8 is the tile's token. Thread (g, t) then
    // receives bytes 4t..4t+3 of token g's segment, which is the MMA's B fragment. The padded
    // token rows (ROWB) put a matrix's eight rows on eight different bank groups.
    const signed char* xl = sx + (size_t)(lane & 7) * ROWB + (lane >> 4) * 32 + (lane >> 3 & 1) * 16;
    for (int stp = s_beg; stp < s_end; ++stp) {
        const int buf = (stp - s_beg) % ST;
        __pipeline_wait_prior(ST - 2);
        __syncthreads();   // this step's data is visible, and the buffer refilled next is idle
        {
            const int nx = stp + ST - 1;
            if (nx < s_end) issue(nx, (nx - s_beg) % ST);
            else __pipeline_commit();
        }
        if (!live) continue;
        const unsigned char* wa = sw + ((size_t)buf * WROWS + warp * 16 + g) * WSEG;
        const unsigned char* wbr = wa + 8 * WSEG;
        float p[NT][4];   // this step's partial sums (blk_term's order)
#pragma unroll
        for (int bb = 0; bb < KB; ++bb) {
            const unsigned* A = reinterpret_cast<const unsigned*>(wa + bb * kBlkBytes);
            const unsigned* B = reinterpret_cast<const unsigned*>(wbr + bb * kBlkBytes);
            const unsigned a6 = A[6], b6 = B[6];
            unsigned fa[4][2], fb[4][2];
            row_frags(A[t], A[o45], a6, t, fa);
            row_frags(B[t], B[o45], b6, t, fb);
            const float swA = __half2float(__ushort_as_half((unsigned short)(a6 >> 16)));
            const float swB = __half2float(__ushort_as_half((unsigned short)(b6 >> 16)));
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                int cc[4] = {0, 0, 0, 0};
                // The four k-steps' B fragments in two ldmatrix.x4 (see xl) instead of eight
                // 4-byte loads: at 32 tokens those loads, not the MMAs, were the kernel's limit.
                const signed char* xn = xl + ((size_t)buf * TOK + n * 8) * ROWB + bb * kBlk;
                int bv[4][2];
                ldsm_x4(xn, bv[0][0], bv[0][1], bv[1][0], bv[1][1]);
                ldsm_x4(xn + 64, bv[2][0], bv[2][1], bv[3][0], bv[3][1]);
#pragma unroll
                for (int s = 0; s < 4; ++s)
                    mma_u8s8(cc, fa[s][0], fb[s][0], fa[s][1], fb[s][1], bv[s][0], bv[s][1]);
                const int t0 = n * 8 + 2 * t;
                const int4 ds = *reinterpret_cast<const int4*>(
                    sds + (((size_t)buf * KB + bb) * TOK + t0) * 2);
                const float d0 = __int_as_float(ds.x), d1 = __int_as_float(ds.z);
                const int s0 = ds.y, s1 = ds.w;
                const float v[4] = {blk_term(swA, d0, cc[0] - s0), blk_term(swA, d1, cc[1] - s1),
                                    blk_term(swB, d0, cc[2] - s0), blk_term(swB, d1, cc[3] - s1)};
#pragma unroll
                for (int i = 0; i < 4; ++i) p[n][i] = bb == 0 ? v[i] : __fadd_rn(p[n][i], v[i]);
            }
        }
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[n][i] = __fadd_rn(acc[n][i], p[n][i]);
    }
    if (!live) return;
    const int rA = row0 + warp * 16 + g, rB = rA + 8;
    if (SPLIT) {
        float* pp = part + ((size_t)mat * gridDim.y + blockIdx.y) * M * N;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int t0 = n * 8 + 2 * t, t1 = t0 + 1;
            if (t0 < M) { pp[(size_t)t0 * N + rA] = acc[n][0]; pp[(size_t)t0 * N + rB] = acc[n][2]; }
            if (t1 < M) { pp[(size_t)t1 * N + rA] = acc[n][1]; pp[(size_t)t1 * N + rB] = acc[n][3]; }
        }
        // Every warp is live on a split launch (FIT), so all of them reach the barriers below.
        if (!FIT || !cnt) return;
        __shared__ unsigned s_last;
        __threadfence();   // this CTA's partials are visible before its arrival is counted
        __syncthreads();
        unsigned* c = cnt + (size_t)mat * ctas_per_mat + cta;
        if (threadIdx.x == 0) s_last = atomicAdd(c, 1u) == gridDim.y - 1;
        __syncthreads();
        if (!s_last) return;
        __threadfence();
        // Four consecutive rows a thread (N % 4 == 0 on a split launch), every split's load issued
        // before the first add.
        const size_t plane = (size_t)M * N;
        const float* pm = part + (size_t)mat * gridDim.y * plane;
        const int S = gridDim.y;
        for (int i = threadIdx.x; i < M * (WROWS / 4); i += NTH) {
            const int tok = i / (WROWS / 4);
            const size_t o = (size_t)tok * N + row0 + 4 * (i - tok * (WROWS / 4));
            float4 v[8];
#pragma unroll
            for (int q = 0; q < 8; ++q)
                if (q < S) v[q] = __ldcg(reinterpret_cast<const float4*>(pm + q * plane + o));
            float4 a = v[0];
#pragma unroll
            for (int q = 1; q < 8; ++q)
                if (q < S) {
                    a.x = __fadd_rn(a.x, v[q].x); a.y = __fadd_rn(a.y, v[q].y);
                    a.z = __fadd_rn(a.z, v[q].z); a.w = __fadd_rn(a.w, v[q].w);
                }
            put<OutT>(y, o, a.x); put<OutT>(y, o + 1, a.y);
            put<OutT>(y, o + 2, a.z); put<OutT>(y, o + 3, a.w);
        }
        if (threadIdx.x == 0) *c = 0u;
        return;
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int t0 = n * 8 + 2 * t, t1 = t0 + 1;
        if (t0 < M) {
            put<OutT>(y, (size_t)t0 * N + rA, acc[n][0]);
            put<OutT>(y, (size_t)t0 * N + rB, acc[n][2]);
        }
        if (t1 < M) {
            put<OutT>(y, (size_t)t1 * N + rA, acc[n][1]);
            put<OutT>(y, (size_t)t1 * N + rB, acc[n][3]);
        }
    }
}

// y (and y1 for a pair) = the S partials summed in split order. One thread per 4 outputs.
template <typename OutT>
__global__ void ptq1_split_reduce_kernel(const float* __restrict__ part, OutT* __restrict__ y0,
                                         OutT* __restrict__ y1, int mn, int S) {
    pdl_wait();
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int mat = blockIdx.y;
    if (i >= mn) return;
    const float* p = part + (size_t)mat * S * mn + i;
    float4 a = *reinterpret_cast<const float4*>(p);
    for (int s = 1; s < S; ++s) {
        const float4 b = *reinterpret_cast<const float4*>(p + (size_t)s * mn);
        a.x = __fadd_rn(a.x, b.x); a.y = __fadd_rn(a.y, b.y);
        a.z = __fadd_rn(a.z, b.z); a.w = __fadd_rn(a.w, b.w);
    }
    OutT* y = mat ? y1 : y0;
    put<OutT>(y, i, a.x); put<OutT>(y, i + 1, a.y); put<OutT>(y, i + 2, a.z); put<OutT>(y, i + 3, a.w);
}

// Launches a packed-row kernel programmatic (see pdl_trigger) when `pdl`; SPARKINFER_ROWS_PDL=0
// launches every one the ordinary way, for an A/B out of one binary.
//
// Only the 8-token tile takes it. There a launch is mostly its weight stream and its fixed start
// cost, which the early weight fetch hides (c2 +3.8%). Past 8 rows the kernel holds 46-65 KB of
// shared memory per CTA, and CTAs parked on the wait crowd the side stream's kernels: c16 -1.6%.
template <typename... KArgs, typename... Args>
void launch_rows_pdl(bool pdl, void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t shm,
                     cudaStream_t st, Args... args) {
    static const bool on = [] {
        const char* e = getenv("SPARKINFER_ROWS_PDL");
        return !(e && e[0] == '0');
    }();
    if (!pdl || !on) {
        kernel<<<grid, block, shm, st>>>(args...);
        return;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = shm;
    cfg.stream = st;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = &attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kernel, args...);
}

// The split launches' arrival counters: one set per stream, since a stream's launches run one
// after another (a programmatic one touches its counters only after its wait), while the side
// stream's split launches may run beside the main stream's. Zero at load, and every launch leaves
// them zero. SPARKINFER_ROWS_SPLIT_FUSED=0 keeps the separate reduce launch (A/B in one binary).
constexpr int kCntSlots = 8, kCntTiles = 1024;
__device__ unsigned g_split_cnt[kCntSlots][kCntTiles];

unsigned* split_cnt_for(cudaStream_t st, int tiles) {
    static const bool on = [] {
        const char* e = getenv("SPARKINFER_ROWS_SPLIT_FUSED");
        return !(e && e[0] == '0');
    }();
    if (!on || tiles > kCntTiles) return nullptr;
    static std::mutex mu;
    static cudaStream_t owners[kCntSlots] = {};
    static int used = 0;
    static unsigned* base = nullptr;
    std::lock_guard<std::mutex> lk(mu);
    if (!base && cudaGetSymbolAddress(reinterpret_cast<void**>(&base), g_split_cnt) != cudaSuccess) {
        base = nullptr;
        return nullptr;
    }
    for (int i = 0; i < used; ++i)
        if (owners[i] == st) return base + (size_t)i * kCntTiles;
    if (used >= kCntSlots) return nullptr;
    owners[used] = st;
    return base + (size_t)(used++) * kCntTiles;
}

template <int NT, int ST, typename OutT, bool SPLIT, int WARPS = 8>
void launch_mma_rows_t(const signed char* xq, const float* xd, const int* xs, const void* w0,
                       const void* w1, OutT* y0, OutT* y1, int m, int n_rows, int nblk, int S,
                       float* part, cudaStream_t st) {
    constexpr int KB = 4;
    constexpr size_t shm = (size_t)ST * (WARPS * 16 * KB * kBlkBytes + NT * 8 * (KB * kBlk + 16) +
                                         NT * 8 * KB * 8);
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(ptq1_mma_rows_kernel<NT, WARPS, KB, ST, OutT, SPLIT>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
        attr = true;
    }
    const int ctas = (n_rows + 16 * WARPS - 1) / (16 * WARPS), nmat = w1 ? 2 : 1;
    const int nsteps = nblk / KB, sps = (nsteps + S - 1) / S;
    // Not for the programmatic (8-token) launches: there the reduce launch is resident before the
    // GEMM ends and costs next to nothing, and the last CTA's pass would only lengthen the tail
    // (single-row decode 1% slower).
    unsigned* cnt = SPLIT && WARPS == 8 && NT > 1 ? split_cnt_for(st, ctas * nmat) : nullptr;
    launch_rows_pdl(NT == 1, ptq1_mma_rows_kernel<NT, WARPS, KB, ST, OutT, SPLIT>,
                    dim3(ctas * nmat, SPLIT ? S : 1), dim3(WARPS * 32), shm, st, xq, xd, xs,
                    static_cast<const unsigned char*>(w0), static_cast<const unsigned char*>(w1),
                    y0, y1, m, n_rows, nblk, ctas, part, sps, cnt);
    if (SPLIT && !cnt) {
        const int mn = m * n_rows;
        launch_rows_pdl(NT == 1, ptq1_split_reduce_kernel<OutT>, dim3((mn / 4 + 255) / 256, nmat),
                        dim3(256), 0, st, (const float*)part, y0, y1, mn, S);
    }
}

template <int NT, int ST, typename OutT>
void launch_mma_rows(const signed char* xq, const float* xd, const int* xs, const void* w0,
                     const void* w1, OutT* y0, OutT* y1, int m, int n_rows, int nblk, int S,
                     float* part, cudaStream_t st) {
    if (S > 1)
        launch_mma_rows_t<NT, ST, OutT, true>(xq, xd, xs, w0, w1, y0, y1, m, n_rows, nblk, S,
                                              part, st);
    else
        launch_mma_rows_t<NT, ST, OutT, false>(xq, xd, xs, w0, w1, y0, y1, m, n_rows, nblk, 1,
                                               nullptr, st);
}

// SPARKINFER_PTQ1_ROW1_TILES=0 keeps a single row on the packed launches' tiles and stages.
bool row1_tiles_on() {
    static const bool v = [] {
        const char* e = getenv("SPARKINFER_PTQ1_ROW1_TILES");
        return !(e && e[0] == '0');
    }();
    return v;
}

template <typename OutT>
bool launch_rows_i8(const signed char* xq, const float* xd, const int* xs, const void* w0,
                    const void* w1, OutT* y0, OutT* y1, int m, int n_rows, int k,
                    cudaStream_t st, float* part = nullptr, size_t part_cap = 0) {
    if (m <= 0 || n_rows <= 0 || k <= 0 || k % (kBlk * kStepBlocks) != 0) return false;
    const int nblk = k / kBlk;
    // The tensor-core kernel tiles 128 weight rows; its k split must be the GEMV's (row_splits),
    // so a launch that cannot hold the partials declines rather than summing differently. One
    // row takes it too: per row it is the GEMV's arithmetic, and at one row it is 3-10% faster
    // (fewer, fuller CTAs than the GEMV's lane-per-block walk).
    const int nmat = w1 ? 2 : 1;
    const int S = row_splits(n_rows, nblk, nmat);
    const bool mma_ok = n_rows % 128 == 0;
    const bool split_fits =
        S == 1 || (part && (size_t)S * nmat * (m < 32 ? m : 32) * n_rows <= part_cap && n_rows % 4 == 0);
    if (mma_ok && m > 1 && !split_fits) return false;
    // One wave, rows spread evenly. Where 128-row tiles need between one and two waves -- gate
    // and up's 2 x 17408 rows are 272 tiles on 170 SMs, and the second wave ran 102 of them with
    // 68 SMs idle -- 13-warp CTAs hold the same rows in one (168 of them). A step's time follows
    // the rows an SM holds, so this is the 256 -> 208 rows the busiest SMs carried; the k split
    // (none) and every row's sums are unchanged.
    constexpr int kBalWarps = 13;
    const int groups = n_rows / 16 * nmat;
    const bool bal = mma_ok && S == 1 && groups > 8 * num_sms() &&
                     (groups + num_sms() - 1) / num_sms() <= kBalWarps;
    for (int m0 = 0; m0 < m; m0 += 32) {
        const int mc = m - m0 < 32 ? m - m0 : 32;
        const signed char* q = xq + (size_t)m0 * k;
        const float* d = xd + (size_t)m0 * nblk;
        const int* s = xs + (size_t)m0 * nblk;
        OutT* a = y0 + (size_t)m0 * n_rows;
        OutT* b = y1 ? y1 + (size_t)m0 * n_rows : nullptr;
        if (!mma_ok || (mc == 1 && !split_fits)) {
            for (int r = 0; r < mc; ++r)
                if (!launch_gemv_i8<OutT>(q + (size_t)r * k, d + (size_t)r * nblk, s + (size_t)r * nblk,
                                          w0, w1, a + (size_t)r * n_rows,
                                          b ? b + (size_t)r * n_rows : nullptr, n_rows, k, st))
                    return false;
        } else if (bal) {
            if (mc <= 8)
                launch_mma_rows_t<1, 2, OutT, false, kBalWarps>(q, d, s, w0, w1, a, b, mc, n_rows,
                                                                nblk, 1, nullptr, st);
            else if (mc <= 16)
                launch_mma_rows_t<2, 2, OutT, false, kBalWarps>(q, d, s, w0, w1, a, b, mc, n_rows,
                                                                nblk, 1, nullptr, st);
            else
                launch_mma_rows_t<4, 2, OutT, false, kBalWarps>(q, d, s, w0, w1, a, b, mc, n_rows,
                                                                nblk, 1, nullptr, st);
        } else if (mc == 1 && row1_tiles_on()) {
            // One row: split launches take 64-row tiles (twice the CTAs at half the shared
            // memory, so a split's few steps have more of them in flight per SM), and unsplit
            // ones (the head) a third weight stage. Tiles and stages never touch a row's sums.
            if (S > 1)
                launch_mma_rows_t<1, 2, OutT, true, 4>(q, d, s, w0, w1, a, b, mc, n_rows, nblk,
                                                       S, part, st);
            else
                launch_mma_rows_t<1, 3, OutT, false>(q, d, s, w0, w1, a, b, mc, n_rows, nblk, 1,
                                                     nullptr, st);
        } else if (mc <= 8) {
            launch_mma_rows<1, 2, OutT>(q, d, s, w0, w1, a, b, mc, n_rows, nblk, S, part, st);
        } else if (mc <= 16) {
            launch_mma_rows<2, 2, OutT>(q, d, s, w0, w1, a, b, mc, n_rows, nblk, S, part, st);
        } else {
            launch_mma_rows<4, 2, OutT>(q, d, s, w0, w1, a, b, mc, n_rows, nblk, S, part, st);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// Prefill: the per-row int8 form the int8 tensor-core GEMMs read.
//
// Weights: t * round(s_b / row_scale), row_scale = max_b |s_b| / 127 -- written here for the
// materialize path, and decoded the same way inside the fused GEMM's B stage (prefill_moe_q.cu,
// the PTQ1 arm of qm_stage_decode). The helpers below are that file's, byte for byte, so the two
// paths hold identical int8 weights.
__device__ __forceinline__ unsigned t_lut(unsigned w6, float inv) {
    const float sb = __half2float(__ushort_as_half((unsigned short)(w6 >> 16)));
    int mq = (int)roundf(sb * inv);
    mq = mq > 127 ? 127 : (mq < -127 ? -127 : mq);
    return ((unsigned)(-mq) & 0xFFu) | (((unsigned)mq & 0xFFu) << 16);
}
__device__ __forceinline__ unsigned t_sel(unsigned lut, unsigned dg) {
    const unsigned w = dg | (dg >> 4);
    return __byte_perm(lut, 0u, __byte_perm(w, 0u, 0x4420u));
}
__device__ __forceinline__ void t_word5(unsigned w, unsigned lut, unsigned (&o)[5]) {
    unsigned ql = w & 0x00FF00FFu, qh = (w >> 8) & 0x00FF00FFu;
#pragma unroll
    for (int m = 0; m < 5; ++m) {
        const unsigned ul = ql * 3u, uh = qh * 3u;
        o[m] = t_sel(lut, __byte_perm(ul, uh, 0x7351));
        ql = ul & 0x00FF00FFu;
        qh = uh & 0x00FF00FFu;
    }
}
__device__ __forceinline__ void t_half(const unsigned (&tw)[4], int h, float inv,
                                       signed char* __restrict__ dst) {
    const unsigned lut = t_lut(tw[3], inv);
    unsigned a[5], b[5], c[5];
    t_word5(tw[0], lut, a);
    t_word5(tw[1], lut, b);
    t_word5(tw[2], lut, c);
#pragma unroll
    for (int m = 0; m < 5; ++m) {
        *reinterpret_cast<uint2*>(dst + m * 16 + 8 * h) = make_uint2(a[m], b[m]);
        *reinterpret_cast<unsigned*>(dst + 80 + 8 * m + 4 * h) = c[m];
    }
    const unsigned x = (tw[3] & 0xFFu) | ((tw[3] & 0xFF00u) << 8);
    const unsigned u0 = x * 3u, u1 = (u0 & 0x00FF00FFu) * 3u;
    const unsigned u2 = (u1 & 0x00FF00FFu) * 3u, u3 = (u2 & 0x00FF00FFu) * 3u;
    const unsigned dg = h ? __byte_perm(u2, u3, 0x7531) : __byte_perm(u0, u1, 0x7531);
    *reinterpret_cast<unsigned*>(dst + 120 + 4 * h) = t_sel(lut, dg);
}

// One warp per weight row: the block scales' max gives the row scale, then each lane decodes
// half-blocks into a shared row buffer that is written out with 16-byte stores.
template <int WPC>
__global__ void __launch_bounds__(WPC * 32)
ptq1_rows_i8_kernel(const unsigned char* __restrict__ w, signed char* __restrict__ q,
                    float* __restrict__ scale, int rows, int nblk) {
    extern __shared__ uint4 srow[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * WPC + warp;
    if (row >= rows) return;
    const unsigned char* wr = w + (size_t)row * nblk * kBlkBytes;
    float am = 0.f;
    for (int b = lane; b < nblk; b += 32) {
        const unsigned short hs = *reinterpret_cast<const unsigned short*>(wr + b * kBlkBytes + 26);
        am = fmaxf(am, fabsf(__half2float(__ushort_as_half(hs))));
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float rs = am / 127.0f;
    const float inv = (rs > 0.f) ? (1.f / rs) : 0.f;
    if (lane == 0) scale[row] = rs;
    signed char* buf = reinterpret_cast<signed char*>(srow) + (size_t)warp * nblk * kBlk;
    for (int i = lane; i < 2 * nblk; i += 32) {
        const int b = i >> 1, h = i & 1;
        const unsigned* bw = reinterpret_cast<const unsigned*>(wr + b * kBlkBytes);
        const unsigned tw[4] = {__ldg(bw + 2 * h), __ldg(bw + 2 * h + 1), __ldg(bw + 4 + h),
                                __ldg(bw + 6)};
        t_half(tw, h, inv, buf + b * kBlk);
    }
    __syncwarp();
    const uint4* src = reinterpret_cast<const uint4*>(buf);
    uint4* dst = reinterpret_cast<uint4*>(q + (size_t)row * nblk * kBlk);
    for (int i = lane; i < nblk * 8; i += 32) dst[i] = src[i];
}

// Activation for the same GEMMs: rotate each 1024-span into the weights' basis, then quantize the
// whole row to int8 with one scale -- d = amax/127, q = round(v/d), the per-row quantizer's rule
// -- writing the row-major copy and, when asked, the k-tiled [k/32][row][32] copy the fused GEMM
// stages from. One CTA per row; the row stays in registers between the two passes.
// MODE, the row rotated (the kernel it replaces never writes its bf16 output, which out_norm can
// still receive):
//   kRotPlain   x itself.
//   kRotSwiglu  x the gate, u the up projection: SwiGLU's output rounded to bf16 exactly as
//               launch_prefill_swiglu_quant_i8 forms it, bf16(g / (1 + exp(-g)) * u), which is also
//               what the decode shadow's down reads (launch_ptq1_swiglu_rotq_bf16).
//   kRotNorm    x through rmsnorm_kernel with weight nw: the row's square sum formed in that
//               kernel's order (256 threads, thread t's 8-value packs t, t + 256, ..., then its two
//               xor trees), and bf16(x * inv_rms * w).
//   kRotGnorm   x the GDN output, u its z gate: pf_gated_norm_kernel per 128-wide v head, which is
//               one warp's 128 values here -- the square sum in that kernel's lane order (lane l
//               takes l, l + 32, l + 64, l + 96), then bf16(x * inv * w * silu(z)).
//   kRotGate    x the attention output, u its gate (row pitch u_ld): pf_mul_sigmoid_kernel's
//               bf16(x * sigmoid(g)).
// Every one is in the kernel it replaces' expression and order, in a TU with the same flags
// (si_gemm and si_fused both build with --use_fast_math), so the rotated values are its own.
template <int NS, int MODE = kRotPlain, bool FP4 = false>
__global__ void __launch_bounds__(256)
ptq1_rotq_rows_i8_kernel(const __nv_bfloat16* __restrict__ x, const signed char* __restrict__ sign,
                         signed char* __restrict__ q, float* __restrict__ scale,
                         signed char* __restrict__ qp, int rows, int k,
                         const __nv_bfloat16* __restrict__ u = nullptr,
                         unsigned char* __restrict__ sfl = nullptr,
                         const __nv_bfloat16* __restrict__ nw = nullptr, float eps = 0.f,
                         int u_ld = 0, __nv_bfloat16* __restrict__ out_norm = nullptr) {
    __shared__ float sh[kSpan];
    __shared__ float sred[8];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int row = blockIdx.x;
    const int ns = k / kSpan;
    const __nv_bfloat16* xr = x + (size_t)row * k;
    const __nv_bfloat16* ur = u ? u + (size_t)row * (u_ld ? u_ld : k) : nullptr;
    float inv_rms = 0.f;
    if constexpr (MODE == kRotNorm) {
        const uint4* x4 = reinterpret_cast<const uint4*>(xr);
        float ss = 0.f;
        for (int p = t; p < (k >> 3); p += 256) {
            const uint4 pk = __ldg(x4 + p);
            const __nv_bfloat16* hp = reinterpret_cast<const __nv_bfloat16*>(&pk);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float xv = __bfloat162float(hp[j]);
                ss = __fmaf_rn(xv, xv, ss);
            }
        }
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
        if (lane == 0) sred[warp] = ss;
        __syncthreads();
        if (t < 32) {
            float r = t < 8 ? sred[t] : 0.f;
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) r += __shfl_xor_sync(0xffffffffu, r, m);
            if (t == 0) sred[0] = rsqrtf(r / k + eps);
        }
        __syncthreads();
        inv_rms = sred[0];
        __syncthreads();   // sred holds the row amax below
    }
    float v[NS][4];
    float am = 0.f;
#pragma unroll
    for (int sp = 0; sp < NS; ++sp) {
        if (sp < ns) {
            const int e0 = sp * kSpan + t * 4;
            uint2 raw = *reinterpret_cast<const uint2*>(xr + e0);
            if constexpr (MODE != kRotPlain) {
                const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&raw);
                __nv_bfloat16 o[4];
                if constexpr (MODE == kRotSwiglu) {
                    const uint2 up = *reinterpret_cast<const uint2*>(ur + e0);
                    const __nv_bfloat16* uh = reinterpret_cast<const __nv_bfloat16*>(&up);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const float g = __bfloat162float(xh[j]);
                        o[j] = __float2bfloat16(g / (1.f + __expf(-g)) * __bfloat162float(uh[j]));
                    }
                } else if constexpr (MODE == kRotNorm) {
                    const uint2 wp = *reinterpret_cast<const uint2*>(nw + e0);
                    const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wp);
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        o[j] = __float2bfloat16(__bfloat162float(xh[j]) * inv_rms *
                                                __bfloat162float(wh[j]));
                } else if constexpr (MODE == kRotGnorm) {
                    const __nv_bfloat16* hx = xr + sp * kSpan + warp * kBlk;   // this warp's head
                    float ss = 0.f;
#pragma unroll
                    for (int r = 0; r < kBlk / 32; r++) {
                        const float xv = __bfloat162float(hx[lane + 32 * r]);
                        ss += xv * xv;
                    }
#pragma unroll
                    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
                    const float inv = rsqrtf(ss / kBlk + eps);
                    const uint2 zp = *reinterpret_cast<const uint2*>(ur + e0);
                    const uint2 wp = *reinterpret_cast<const uint2*>(nw + lane * 4);
                    const __nv_bfloat16* zh = reinterpret_cast<const __nv_bfloat16*>(&zp);
                    const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wp);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const float z = __bfloat162float(zh[j]);
                        o[j] = __float2bfloat16(__bfloat162float(xh[j]) * inv *
                                                __bfloat162float(wh[j]) * (z / (1.f + __expf(-z))));
                    }
                } else {   // kRotGate
                    const uint2 gp = *reinterpret_cast<const uint2*>(ur + e0);
                    const __nv_bfloat16* gh = reinterpret_cast<const __nv_bfloat16*>(&gp);
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        o[j] = __float2bfloat16(__bfloat162float(xh[j]) *
                                                (1.f / (1.f + __expf(-__bfloat162float(gh[j])))));
                }
                raw = *reinterpret_cast<const uint2*>(o);
                if (out_norm) *reinterpret_cast<uint2*>(out_norm + (size_t)row * k + e0) = raw;
            }
            const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
            const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
            const char4 sg = *reinterpret_cast<const char4*>(sign + e0);
            float w0 = __low2float(a) * (float)sg.x, w1 = __high2float(a) * (float)sg.y;
            float w2 = __low2float(b) * (float)sg.z, w3 = __high2float(b) * (float)sg.w;
            const float a0 = w0 + w1, a1 = w0 - w1, a2 = w2 + w3, a3 = w2 - w3;
            float r[4] = {a0 + a2, a1 + a3, a0 - a2, a1 - a3};
#pragma unroll
            for (int m = 1; m < 32; m <<= 1) {
                // p - r on the upper lane, r + p on the lower: one fma with r's sign, which rounds
                // the exact sum once, as the add does.
                const float sg = (lane & m) ? -1.f : 1.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float p = __shfl_xor_sync(0xffffffffu, r[i], m);
                    r[i] = __fmaf_rn(sg, r[i], p);
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
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[sp][i] = sh[t * 4 + i] * 0.03125f;
                am = fmaxf(am, fabsf(v[sp][i]));
            }
            if constexpr (FP4) {
                // NVFP4 instead of int8: q (the packed nibbles, k/2 bytes a row) and a row-major
                // ue4m3 scale per 16 values in `qp`. Four lanes hold a 16-value group; the rule is
                // prefill_nvfp4's quant_rows_t: qs = ue4m3(max(amax / 6, 2^-9)), e2m1(v / qs).
                float ga = fmaxf(fmaxf(fabsf(v[sp][0]), fabsf(v[sp][1])),
                                 fmaxf(fabsf(v[sp][2]), fabsf(v[sp][3])));
                ga = fmaxf(ga, __shfl_xor_sync(0xffffffffu, ga, 1));
                ga = fmaxf(ga, __shfl_xor_sync(0xffffffffu, ga, 2));
                const __nv_fp8_storage_t qb =
                    __nv_cvt_float_to_fp8(fmaxf(ga * (1.f / 6.f), 0x1p-9f), __NV_SATFINITE, __NV_E4M3);
                const float qs = __half2float(__half(__nv_cvt_fp8_to_halfraw(qb, __NV_E4M3)));
                const int e0 = sp * kSpan + t * 4;
                const unsigned lo = e2m1x2_rn(v[sp][0] / qs, v[sp][1] / qs);
                const unsigned hi = e2m1x2_rn(v[sp][2] / qs, v[sp][3] / qs);
                *reinterpret_cast<unsigned short*>(
                    reinterpret_cast<unsigned char*>(q) + ((size_t)row * k + e0) / 2) =
                    (unsigned short)(lo | (hi << 8));
                if ((t & 3) == 0) {
                    if (sfl) sfl[sf_cutlass_off(row, e0 / 16, k / 16)] = qb;
                    else reinterpret_cast<unsigned char*>(qp)[(size_t)row * (k / 16) + e0 / 16] = qb;
                }
            }
        }
    }
    if constexpr (FP4) return;
#pragma unroll
    for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    if (lane == 0) sred[warp] = am;
    __syncthreads();
    float amax = sred[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) amax = fmaxf(amax, sred[i]);
    const float d = amax / 127.0f;
    if (t == 0) scale[row] = d;
#pragma unroll
    for (int sp = 0; sp < NS; ++sp) {
        if (sp < ns) {
            const int e0 = sp * kSpan + t * 4;
            int qv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) qv[i] = (amax == 0.f) ? 0 : (int)roundf(v[sp][i] / d);
            const char4 c4 = make_char4((signed char)qv[0], (signed char)qv[1], (signed char)qv[2],
                                        (signed char)qv[3]);
            *reinterpret_cast<char4*>(q + (size_t)row * k + e0) = c4;
            if (qp)
                *reinterpret_cast<char4*>(qp + (size_t)(e0 >> 5) * rows * 32 + (size_t)row * 32 +
                                          (e0 & 31)) = c4;
        }
    }
}

// The NVFP4 arm of ptq1_rotq_rows_i8_kernel with one warp per 1024-span. Lane l holds the span's
// values 128 h + 4 l + j (h = 0..7, j = 0..3): what thread (warp h, lane l) of that kernel holds.
// The butterflies run in its order -- j, then the lanes, then h -- each on the same two values, so
// every nibble and scale byte is its own; only the h levels move, from a shared-memory pass with
// three barriers a span into registers. Each warp takes span item 8 * block + warp of rows x
// k/1024, except under kRotNorm, whose row square sum comes first: a CTA a row runs that kernel's
// prologue -- its 256 threads' square sums and two xor trees, each real warp w also standing in for
// virtual warps w + ns, w + 2 ns, ... -- then each of its ns warps takes one span.
template <int MODE>
__global__ void __launch_bounds__(256)
ptq1_rotq_rows_fp4_warp_kernel(const __nv_bfloat16* __restrict__ x, const signed char* __restrict__ sign,
                               unsigned char* __restrict__ q, unsigned char* __restrict__ sfr,
                               int rows, int k, const __nv_bfloat16* __restrict__ u,
                               unsigned char* __restrict__ sfl, const __nv_bfloat16* __restrict__ nw,
                               float eps, int u_ld, __nv_bfloat16* __restrict__ out_norm) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int ns = k / kSpan;
    int row, sp, sp_step;
    float inv_rms = 0.f;
    if constexpr (MODE == kRotNorm) {
        __shared__ float sred[8];
        row = blockIdx.x;
        const uint4* x4 = reinterpret_cast<const uint4*>(x + (size_t)row * k);
        for (int vw = warp; vw < 8; vw += ns) {   // virtual warp vw of a 256-thread block
            const int vt = vw * 32 + lane;
            float ss = 0.f;
            for (int p = vt; p < (k >> 3); p += 256) {
                const uint4 pk = __ldg(x4 + p);
                const __nv_bfloat16* hp = reinterpret_cast<const __nv_bfloat16*>(&pk);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float xv = __bfloat162float(hp[j]);
                    ss = __fmaf_rn(xv, xv, ss);
                }
            }
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
            if (lane == 0) sred[vw] = ss;
        }
        __syncthreads();
        if (t < 32) {
            float r = t < 8 ? sred[t] : 0.f;
#pragma unroll
            for (int m = 16; m > 0; m >>= 1) r += __shfl_xor_sync(0xffffffffu, r, m);
            if (t == 0) sred[0] = rsqrtf(r / k + eps);
        }
        __syncthreads();
        inv_rms = sred[0];
        sp = warp;
        sp_step = ns;
    } else {
        const int item = blockIdx.x * 8 + warp;
        if (item >= rows * ns) return;
        row = item / ns;
        sp = item - row * ns;
        sp_step = ns;
    }
    const __nv_bfloat16* xr = x + (size_t)row * k;
    const __nv_bfloat16* ur = u ? u + (size_t)row * (u_ld ? u_ld : k) : nullptr;
    for (; sp < ns; sp += sp_step) {
        uint2 raw[8];
#pragma unroll
        for (int h = 0; h < 8; ++h) raw[h] = *reinterpret_cast<const uint2*>(xr + sp * kSpan + h * kBlk + lane * 4);
        float r[8][4];
#pragma unroll
        for (int h = 0; h < 8; ++h) {
            const int e0 = sp * kSpan + h * kBlk + lane * 4;
            if constexpr (MODE != kRotPlain) {
                const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&raw[h]);
                __nv_bfloat16 o[4];
                if constexpr (MODE == kRotSwiglu) {
                    const uint2 up = *reinterpret_cast<const uint2*>(ur + e0);
                    const __nv_bfloat16* uh = reinterpret_cast<const __nv_bfloat16*>(&up);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const float g = __bfloat162float(xh[j]);
                        o[j] = __float2bfloat16(g / (1.f + __expf(-g)) * __bfloat162float(uh[j]));
                    }
                } else if constexpr (MODE == kRotNorm) {
                    const uint2 wp = *reinterpret_cast<const uint2*>(nw + e0);
                    const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wp);
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        o[j] = __float2bfloat16(__bfloat162float(xh[j]) * inv_rms *
                                                __bfloat162float(wh[j]));
                } else if constexpr (MODE == kRotGnorm) {
                    const __nv_bfloat16* hx = xr + sp * kSpan + h * kBlk;   // this head
                    float ss = 0.f;
#pragma unroll
                    for (int rr = 0; rr < kBlk / 32; rr++) {
                        const float xv = __bfloat162float(hx[lane + 32 * rr]);
                        ss += xv * xv;
                    }
#pragma unroll
                    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);
                    const float inv = rsqrtf(ss / kBlk + eps);
                    const uint2 zp = *reinterpret_cast<const uint2*>(ur + e0);
                    const uint2 wp = *reinterpret_cast<const uint2*>(nw + lane * 4);
                    const __nv_bfloat16* zh = reinterpret_cast<const __nv_bfloat16*>(&zp);
                    const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wp);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const float z = __bfloat162float(zh[j]);
                        o[j] = __float2bfloat16(__bfloat162float(xh[j]) * inv *
                                                __bfloat162float(wh[j]) * (z / (1.f + __expf(-z))));
                    }
                } else {   // kRotGate
                    const uint2 gp = *reinterpret_cast<const uint2*>(ur + e0);
                    const __nv_bfloat16* gh = reinterpret_cast<const __nv_bfloat16*>(&gp);
#pragma unroll
                    for (int j = 0; j < 4; j++)
                        o[j] = __float2bfloat16(__bfloat162float(xh[j]) *
                                                (1.f / (1.f + __expf(-__bfloat162float(gh[j])))));
                }
                raw[h] = *reinterpret_cast<const uint2*>(o);
                if (out_norm) *reinterpret_cast<uint2*>(out_norm + (size_t)row * k + e0) = raw[h];
            }
            const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&raw[h].x);
            const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&raw[h].y);
            const char4 sg = *reinterpret_cast<const char4*>(sign + e0);
            float w0 = __low2float(a) * (float)sg.x, w1 = __high2float(a) * (float)sg.y;
            float w2 = __low2float(b) * (float)sg.z, w3 = __high2float(b) * (float)sg.w;
            const float a0 = w0 + w1, a1 = w0 - w1, a2 = w2 + w3, a3 = w2 - w3;
            r[h][0] = a0 + a2; r[h][1] = a1 + a3; r[h][2] = a0 - a2; r[h][3] = a1 - a3;
        }
#pragma unroll
        for (int m = 1; m < 32; m <<= 1) {
            const float sgn = (lane & m) ? -1.f : 1.f;
#pragma unroll
            for (int h = 0; h < 8; ++h)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float p = __shfl_xor_sync(0xffffffffu, r[h][i], m);
                    r[h][i] = __fmaf_rn(sgn, r[h][i], p);
                }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int len = 1; len < 8; len <<= 1)
#pragma unroll
                for (int h = 0; h < 8; ++h)
                    if (!(h & len)) {
                        const float xx = r[h][i], yy = r[h + len][i];
                        r[h][i] = xx + yy; r[h + len][i] = xx - yy;
                    }
#pragma unroll
        for (int h = 0; h < 8; ++h) {
            float v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = r[h][i] * 0.03125f;
            float ga = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
            ga = fmaxf(ga, __shfl_xor_sync(0xffffffffu, ga, 1));
            ga = fmaxf(ga, __shfl_xor_sync(0xffffffffu, ga, 2));
            const __nv_fp8_storage_t qb =
                __nv_cvt_float_to_fp8(fmaxf(ga * (1.f / 6.f), 0x1p-9f), __NV_SATFINITE, __NV_E4M3);
            const float qs = __half2float(__half(__nv_cvt_fp8_to_halfraw(qb, __NV_E4M3)));
            const int e0 = sp * kSpan + h * kBlk + lane * 4;
            const unsigned lo = e2m1x2_rn(v[0] / qs, v[1] / qs);
            const unsigned hi = e2m1x2_rn(v[2] / qs, v[3] / qs);
            *reinterpret_cast<unsigned short*>(q + ((size_t)row * k + e0) / 2) =
                (unsigned short)(lo | (hi << 8));
            if ((lane & 3) == 0) {
                if (sfl) sfl[sf_cutlass_off(row, e0 / 16, k / 16)] = qb;
                else sfr[(size_t)row * (k / 16) + e0 / 16] = qb;
            }
        }
    }
}

// SPARKINFER_PTQ1_ROTQ_WARP=0 keeps the NVFP4 rotations on ptq1_rotq_rows_i8_kernel (A/B in one binary).
bool rotq_fp4_warp_on() {
    static const bool on = [] {
        const char* e = getenv("SPARKINFER_PTQ1_ROTQ_WARP");
        return !(e && e[0] == '0');
    }();
    return on;
}
template <int MODE>
void rotq_fp4_warp_launch(const __nv_bfloat16* x, const signed char* sign, void* q, void* sfr,
                          int rows, int k, const __nv_bfloat16* u, unsigned char* sfl,
                          const __nv_bfloat16* nw, float eps, int u_ld, __nv_bfloat16* out_norm,
                          cudaStream_t st) {
    const long items = (long)rows * (k / kSpan);
    const unsigned grid = MODE == kRotNorm ? (unsigned)rows : (unsigned)((items + 7) / 8);
    const unsigned block = MODE == kRotNorm ? 32u * (unsigned)(k / kSpan) : 256u;
    ptq1_rotq_rows_fp4_warp_kernel<MODE><<<grid, block, 0, st>>>(
        x, sign, static_cast<unsigned char*>(q), static_cast<unsigned char*>(sfr), rows, k, u, sfl,
        nw, eps, u_ld, out_norm);
}

// Prefill's NVFP4 B operand from the stored blocks. A trit is exact in e2m1, so the only rounding
// is the block scale: s_b * 2^10 is written as m * sf, m one of e2m1's magnitudes {1, 1.5, 2, 3,
// 4, 6} and sf a ue4m3, the pair closest to it (the GEMM's alpha takes the 2^-10 back off). The
// scale range of the checkpoint (2.7e-3 .. 0.16) lands on normal ue4m3 values. Output: the packed
// nibbles (k/2 bytes a row, low nibble first) and a row-major ue4m3 per 16 values.
constexpr float kFp4WScale = 1024.f;
__device__ __forceinline__ void ptq1_fp4_scale(float sb, unsigned& code, unsigned char& sfb) {
    const float T = fabsf(sb) * kFp4WScale;
    const float mags[6] = {1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float best = 3.4e38f;
    code = 2u; sfb = 0;
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        const __nv_fp8_storage_t f = __nv_cvt_float_to_fp8(T / mags[i], __NV_SATFINITE, __NV_E4M3);
        const float e = fabsf(mags[i] * __half2float(__half(__nv_cvt_fp8_to_halfraw(f, __NV_E4M3))) - T);
        if (e < best) { best = e; code = 2u + (unsigned)i; sfb = f; }
    }
}
// A warp stages at most kRowsChunk blocks of its row (8 KB of trits) at a time, so every width runs
// WPC rows a CTA: a whole 17408-wide row staged at once took 17 KB a warp, and the FFN down leg
// ran one warp a CTA at ~0.74 TB/s against the gate/up legs' ~1.34 over the same bytes. Each
// 16-value group reads the same trits and the same block scale either way. Rows up to 8192 wide
// are one chunk, as before.
constexpr int kRowsChunk = 64;
template <int WPC>
__global__ void __launch_bounds__(WPC * 32)
ptq1_rows_nvfp4_kernel(const unsigned char* __restrict__ w, unsigned char* __restrict__ q,
                       unsigned char* __restrict__ sf, int rows, int nblk,
                       unsigned char* __restrict__ sfl, int omul = 1, int oadd = 0) {
    extern __shared__ uint4 srow[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * WPC + warp;
    if (row >= rows) return;
    const int orow = row * omul + oadd;   // the row it lands in (2r, 2r + 1: gate|up interleaved)
    const unsigned char* wr = w + (size_t)row * nblk * kBlkBytes;
    const int cb = min(nblk, kRowsChunk);
    signed char* buf = reinterpret_cast<signed char*>(srow) + (size_t)warp * cb * kBlk;
    const int ng = nblk * (kBlk / 16);
    for (int c0 = 0; c0 < nblk; c0 += cb) {
        const int cn = min(cb, nblk - c0);
        if (c0) __syncwarp();   // the previous chunk's groups are read out
        for (int i = lane; i < 2 * cn; i += 32) {
            const int b = i >> 1, h = i & 1;
            const unsigned* bw = reinterpret_cast<const unsigned*>(wr + (c0 + b) * kBlkBytes);
            const unsigned tw[4] = {__ldg(bw + 2 * h), __ldg(bw + 2 * h + 1), __ldg(bw + 4 + h),
                                    (__ldg(bw + 6) & 0xFFFFu) | 0x3C000000u};   // scale 1.0: bare trits
            t_half(tw, h, 1.f, buf + b * kBlk);
        }
        __syncwarp();
        for (int gc = lane; gc < cn * (kBlk / 16); gc += 32) {
            const int g = c0 * (kBlk / 16) + gc;
            const int b = g >> 3;
            const unsigned short hs = *reinterpret_cast<const unsigned short*>(wr + b * kBlkBytes + 26);
            unsigned code; unsigned char sfb;
            ptq1_fp4_scale(__half2float(__ushort_as_half(hs)), code, sfb);
            const signed char* tv = buf + gc * 16;
            unsigned o[2] = {0u, 0u};
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const int tr = tv[j];
                const unsigned nib = tr == 0 ? 0u : (code | (tr < 0 ? 8u : 0u));
                o[j >> 3] |= nib << (4 * (j & 7));
            }
            *reinterpret_cast<uint2*>(q + (size_t)orow * nblk * (kBlk / 2) + g * 8) = make_uint2(o[0], o[1]);
            if (sfl) sfl[sf_cutlass_off(orow, g, ng)] = sfb;
            else sf[(size_t)orow * ng + g] = sfb;
        }
    }
}

}  // namespace

// A partial last 128-row atom leaves bytes no row writes; they are zeroed as the pack does.
static bool sf_cutlass_clear(void* sfl, int rows, int k, cudaStream_t st) {
    if (!sfl || (rows & 127) == 0) return true;
    return cudaMemsetAsync(sfl, 0, (size_t)((rows + 127) & ~127) * (k / 16), st) == cudaSuccess;
}

bool launch_ptq1_rows_nvfp4(const void* w_ptq1, void* q, void* sf_rowmajor, int rows, int k,
                            cudaStream_t st, void* sf_cutlass) {
    if (rows <= 0 || k <= 0 || k % kBlk != 0 || k > 96 * 1024) return false;
    if (!sf_cutlass_clear(sf_cutlass, rows, k, st)) return false;
    auto* sfl = static_cast<unsigned char*>(sf_cutlass);
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(ptq1_rows_nvfp4_kernel<4>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        attr = true;
    }
    const auto* w = static_cast<const unsigned char*>(w_ptq1);
    auto* qq = static_cast<unsigned char*>(q);
    auto* sf = static_cast<unsigned char*>(sf_rowmajor);
    // A row stages at most kRowsChunk blocks of shared memory at a time, so four rows a CTA at
    // every width (the FFN's 17408 used to go one row to a CTA).
    const int nblk = k / kBlk;
    const size_t smem = (size_t)4 * (nblk < kRowsChunk ? nblk : kRowsChunk) * kBlk;
    ptq1_rows_nvfp4_kernel<4><<<(rows + 3) / 4, 128, smem, st>>>(w, qq, sf, rows, nblk, sfl);
    return true;
}
// Gate and up of one FFN as the single [2 * rows, k] operand the gate|up GEMM reads, row 2j gate
// row j and row 2j + 1 up row j (launch_prefill_nvfp4_interleave_gate_up's layout): each written
// where launch_ptq1_rows_nvfp4 writes it, two rows apart.
bool launch_ptq1_rows_nvfp4_gate_up(const void* gate_ptq1, const void* up_ptq1, void* q, int rows,
                                    int k, cudaStream_t st, void* sf_cutlass) {
    if (!gate_ptq1 || !up_ptq1 || !sf_cutlass || rows <= 0 || k <= 0 || k % kBlk != 0 ||
        k > 96 * 1024)
        return false;
    if (!sf_cutlass_clear(sf_cutlass, 2 * rows, k, st)) return false;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(ptq1_rows_nvfp4_kernel<4>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024);
        attr = true;
    }
    const int nblk = k / kBlk;
    const size_t smem = (size_t)4 * (nblk < kRowsChunk ? nblk : kRowsChunk) * kBlk;
    auto* qq = static_cast<unsigned char*>(q);
    auto* sfl = static_cast<unsigned char*>(sf_cutlass);
    ptq1_rows_nvfp4_kernel<4><<<(rows + 3) / 4, 128, smem, st>>>(
        static_cast<const unsigned char*>(gate_ptq1), qq, nullptr, rows, nblk, sfl, 2, 0);
    ptq1_rows_nvfp4_kernel<4><<<(rows + 3) / 4, 128, smem, st>>>(
        static_cast<const unsigned char*>(up_ptq1), qq, nullptr, rows, nblk, sfl, 2, 1);
    return true;
}
float ptq1_nvfp4_alpha() { return 1.f / kFp4WScale; }

// A rotation sits between two GEMMs on every leg, so its launch and CTA dispatch were paid in
// full after the GEMM before it drained. Launched programmatic, its CTAs are resident when that
// GEMM completes (the rows kernels trigger at their start) and wait there (pdl_wait) before
// reading anything. SPARKINFER_ROTQ_PDL=0 launches them the ordinary way (A/B in one binary).
template <int MODE>
void rotq_launch(int rows, int k, cudaStream_t st, const __nv_bfloat16* x, const __nv_bfloat16* u,
                 const __nv_bfloat16* nw, float eps, const signed char* sign, signed char* q,
                 float* qd, int* qs, __nv_bfloat16* out_sum = nullptr,
                 __nv_bfloat16* out_norm = nullptr, i8_blk_q8_1* out_q8 = nullptr) {
    static const bool pdl = [] {
        const char* e = getenv("SPARKINFER_ROTQ_PDL");
        return !(e && e[0] == '0');
    }();
    launch_rows_pdl(pdl, ptq1_rotq_kernel<MODE>, dim3((unsigned)(k / kSpan), (unsigned)rows),
                    dim3(256), 0, st, x, u, nw, eps, sign, q, qd, qs, k, out_sum, out_norm, out_q8);
}

bool launch_ptq1_rotq_bf16(const void* x_bf16, const signed char* sign, signed char* q,
                           float* qd, int* qs, int rows, int k, int block, cudaStream_t st) {
    if (rows <= 0 || block != kSpan || k % kSpan != 0) return false;
    rotq_launch<kRotPlain>(rows, k, st, static_cast<const __nv_bfloat16*>(x_bf16), nullptr,
                           nullptr, 0.f, sign, q, qd, qs);
    return true;
}

bool launch_ptq1_add_norm_rotq_bf16(const void* x_bf16, const void* residual_bf16,
                                    const void* weight_bf16, void* out_sum, void* out_norm,
                                    void* out_q8, float eps, const signed char* sign,
                                    signed char* q, float* qd, int* qs, int rows, int k,
                                    int block, cudaStream_t st) {
    if (rows <= 0 || block != kSpan || k % kSpan != 0 || k > 8192 || !out_sum || !out_norm)
        return false;
    static const bool on = [] {
        const char* e = getenv("SPARKINFER_PTQ1_NORM_ROTQ");
        return !(e && e[0] == '0');
    }();
    if (!on) return false;
    rotq_launch<kRotAddNorm>(rows, k, st, static_cast<const __nv_bfloat16*>(x_bf16),
                             static_cast<const __nv_bfloat16*>(residual_bf16),
                             static_cast<const __nv_bfloat16*>(weight_bf16), eps, sign, q, qd, qs,
                             static_cast<__nv_bfloat16*>(out_sum),
                             static_cast<__nv_bfloat16*>(out_norm),
                             static_cast<i8_blk_q8_1*>(out_q8));
    return true;
}

bool launch_ptq1_swiglu_rotq_bf16(const void* gate_bf16, const void* up_bf16,
                                  const signed char* sign, signed char* q, float* qd, int* qs,
                                  int rows, int k, int block, cudaStream_t st) {
    if (rows <= 0 || block != kSpan || k % kSpan != 0) return false;
    rotq_launch<kRotSwiglu>(rows, k, st, static_cast<const __nv_bfloat16*>(gate_bf16),
                            static_cast<const __nv_bfloat16*>(up_bf16), nullptr, 0.f, sign, q, qd,
                            qs);
    return true;
}

bool launch_ptq1_gnorm_rotq_bf16(const void* x_bf16, const void* z_bf16, const void* norm_bf16,
                                 float eps, const signed char* sign, signed char* q, float* qd,
                                 int* qs, int rows, int k, int head_dim, int block,
                                 cudaStream_t st) {
    if (rows <= 0 || block != kSpan || k % kSpan != 0 || head_dim != kBlk) return false;
    rotq_launch<kRotGnorm>(rows, k, st, static_cast<const __nv_bfloat16*>(x_bf16),
                           static_cast<const __nv_bfloat16*>(z_bf16),
                           static_cast<const __nv_bfloat16*>(norm_bf16), eps, sign, q, qd, qs);
    return true;
}

bool launch_ptq1_gate_rotq_bf16(const void* x_bf16, const void* gate_bf16,
                                const signed char* sign, signed char* q, float* qd, int* qs,
                                int rows, int k, int block, cudaStream_t st) {
    if (rows <= 0 || block != kSpan || k % kSpan != 0) return false;
    rotq_launch<kRotGate>(rows, k, st, static_cast<const __nv_bfloat16*>(x_bf16),
                          static_cast<const __nv_bfloat16*>(gate_bf16), nullptr, 0.f, sign, q, qd,
                          qs);
    return true;
}

bool launch_gemv_ptq1_i8_bf16(const signed char* xq, const float* xd, const int* xs,
                              const void* w0, const void* w1, void* y0, void* y1,
                              int n_rows, int k, cudaStream_t st) {
    return launch_gemv_i8<__nv_bfloat16>(xq, xd, xs, w0, w1, static_cast<__nv_bfloat16*>(y0),
                                         static_cast<__nv_bfloat16*>(y1), n_rows, k, st);
}

bool launch_gemv_ptq1_i8_f32(const signed char* xq, const float* xd, const int* xs,
                             const void* w, float* y, int n_rows, int k, cudaStream_t st) {
    return launch_gemv_i8<float>(xq, xd, xs, w, nullptr, y, nullptr, n_rows, k, st);
}

bool launch_gemm_ptq1_i8_rows_f32(const signed char* xq, const float* xd, const int* xs,
                                  const void* w, float* y, int m, int n_rows, int k,
                                  cudaStream_t st, float* part, size_t part_cap) {
    return launch_rows_i8<float>(xq, xd, xs, w, nullptr, y, nullptr, m, n_rows, k, st, part,
                                 part_cap);
}

bool launch_gemm_ptq1_i8_rows_bf16(const signed char* xq, const float* xd, const int* xs,
                                   const void* w0, const void* w1, void* y0, void* y1, int m,
                                   int n_rows, int k, cudaStream_t st, float* part,
                                   size_t part_cap) {
    return launch_rows_i8<__nv_bfloat16>(xq, xd, xs, w0, w1, static_cast<__nv_bfloat16*>(y0),
                                         static_cast<__nv_bfloat16*>(y1), m, n_rows, k, st, part,
                                         part_cap);
}

// launch_gemm_ptq1_i8_row_partials' kernel: launch_rows_i8's for one split row, without the
// reduce launch after it.
template <int WARPS>
void launch_row_partials_t(const signed char* xq, const float* xd, const int* xs, const void* w,
                           int n_rows, int nblk, int S, float* part, cudaStream_t st) {
    constexpr int NT = 1, KB = kStepBlocks, ST = 2;
    auto kern = ptq1_mma_rows_kernel<NT, WARPS, KB, ST, __nv_bfloat16, true>;
    constexpr size_t shm = (size_t)ST * (WARPS * 16 * KB * kBlkBytes + NT * 8 * (KB * kBlk + 16) +
                                         NT * 8 * KB * 8);
    static const bool attr = [&] {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm);
        return true;
    }();
    (void)attr;
    const int ctas = (n_rows + 16 * WARPS - 1) / (16 * WARPS);
    const int nsteps = nblk / KB, sps = (nsteps + S - 1) / S;
    launch_rows_pdl(true, kern, dim3(ctas, S), dim3(WARPS * 32), shm, st, xq, xd, xs,
                    static_cast<const unsigned char*>(w), (const unsigned char*)nullptr,
                    (__nv_bfloat16*)nullptr, (__nv_bfloat16*)nullptr, 1, n_rows, nblk, ctas, part,
                    sps, (unsigned*)nullptr);   // no arrival counters: the partials are the output
}

int launch_gemm_ptq1_i8_row_partials(const signed char* xq, const float* xd, const int* xs,
                                     const void* w, int n_rows, int k, cudaStream_t st,
                                     float* part, size_t part_cap) {
    if (!part || n_rows <= 0 || n_rows % 128 != 0 || k <= 0 || k % (kBlk * kStepBlocks) != 0)
        return 0;
    const int nblk = k / kBlk;
    const int S = row_splits(n_rows, nblk, 1);
    if (S < 2 || (size_t)S * n_rows > part_cap) return 0;
    if (row1_tiles_on()) launch_row_partials_t<4>(xq, xd, xs, w, n_rows, nblk, S, part, st);
    else                 launch_row_partials_t<8>(xq, xd, xs, w, n_rows, nblk, S, part, st);
    return S;
}

bool launch_ptq1_rows_i8(const void* w_ptq1, signed char* q, float* scale, int rows, int k,
                         cudaStream_t st) {
    if (rows <= 0 || k <= 0 || k % kBlk != 0) return false;
    constexpr int WPC = 4;
    const size_t shm = (size_t)WPC * k;
    if (shm > 96 * 1024) return false;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(ptq1_rows_i8_kernel<WPC>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             96 * 1024);
        attr = true;
    }
    ptq1_rows_i8_kernel<WPC><<<(rows + WPC - 1) / WPC, WPC * 32, shm, st>>>(
        static_cast<const unsigned char*>(w_ptq1), q, scale, rows, k / kBlk);
    return true;
}

bool launch_ptq1_swiglu_rotq_rows_i8(const void* gate_bf16, const void* up_bf16,
                                     const signed char* sign, signed char* q, float* scale,
                                     signed char* qp, int rows, int k, int block, cudaStream_t st) {
    if (rows <= 0 || block != kSpan || k % kSpan != 0 || k > 17 * kSpan) return false;
    const auto* g = static_cast<const __nv_bfloat16*>(gate_bf16);
    const auto* u = static_cast<const __nv_bfloat16*>(up_bf16);
    if (k <= 8 * kSpan)
        ptq1_rotq_rows_i8_kernel<8, true><<<rows, 256, 0, st>>>(g, sign, q, scale, qp, rows, k, u);
    else
        ptq1_rotq_rows_i8_kernel<17, true><<<rows, 256, 0, st>>>(g, sign, q, scale, qp, rows, k, u);
    return true;
}

bool launch_ptq1_rotq_rows_nvfp4(const void* x_bf16, const void* up_bf16, const signed char* sign,
                                 void* q, void* sf_rowmajor, int rows, int k, int block,
                                 cudaStream_t st, void* sf_cutlass) {
    if (rows <= 0 || block != kSpan || k % kSpan != 0 || k > 17 * kSpan) return false;
    if (!sf_cutlass_clear(sf_cutlass, rows, k, st)) return false;
    auto* sfl = static_cast<unsigned char*>(sf_cutlass);
    const auto* x = static_cast<const __nv_bfloat16*>(x_bf16);
    const auto* u = static_cast<const __nv_bfloat16*>(up_bf16);
    auto* qq = static_cast<signed char*>(q);
    auto* sf = static_cast<signed char*>(sf_rowmajor);
    if (rotq_fp4_warp_on()) {
        if (u) rotq_fp4_warp_launch<kRotSwiglu>(x, sign, q, sf, rows, k, u, sfl, nullptr, 0.f, 0, nullptr, st);
        else   rotq_fp4_warp_launch<kRotPlain>(x, sign, q, sf, rows, k, nullptr, sfl, nullptr, 0.f, 0, nullptr, st);
        return true;
    }
    if (u) {
        if (k <= 8 * kSpan)
            ptq1_rotq_rows_i8_kernel<8, true, true><<<rows, 256, 0, st>>>(x, sign, qq, nullptr, sf, rows, k, u, sfl);
        else
            ptq1_rotq_rows_i8_kernel<17, true, true><<<rows, 256, 0, st>>>(x, sign, qq, nullptr, sf, rows, k, u, sfl);
    } else {
        if (k <= 5 * kSpan)
            ptq1_rotq_rows_i8_kernel<5, false, true><<<rows, 256, 0, st>>>(x, sign, qq, nullptr, sf, rows, k, nullptr, sfl);
        else if (k <= 8 * kSpan)
            ptq1_rotq_rows_i8_kernel<8, false, true><<<rows, 256, 0, st>>>(x, sign, qq, nullptr, sf, rows, k, nullptr, sfl);
        else   // SwiGLU's output already formed (launch_prefill_nvfp4_gate_up_swiglu_bf16)
            ptq1_rotq_rows_i8_kernel<17, false, true><<<rows, 256, 0, st>>>(x, sign, qq, nullptr, sf, rows, k, nullptr, sfl);
    }
    return true;
}

// launch_ptq1_rotq_rows_nvfp4 with the kernel that produced its input folded in (see the MODE
// list above ptq1_rotq_rows_i8_kernel): that kernel's bf16 row is never written and read back.
template <int MODE>
bool rotq_rows_nvfp4_fused(const __nv_bfloat16* x, const __nv_bfloat16* u, int u_ld,
                           const __nv_bfloat16* nw, float eps, __nv_bfloat16* out_norm,
                           const signed char* sign, void* q, int rows, int k, int block,
                           cudaStream_t st, void* sf_cutlass) {
    if (rows <= 0 || block != kSpan || k % kSpan != 0 || k > 8 * kSpan || !sf_cutlass) return false;
    if (!sf_cutlass_clear(sf_cutlass, rows, k, st)) return false;
    auto* qq = static_cast<signed char*>(q);
    auto* sfl = static_cast<unsigned char*>(sf_cutlass);
    // The gated norm and the gate stream two operands and run at the DRAM's rate on either kernel.
    if (MODE == kRotNorm && rotq_fp4_warp_on()) {
        rotq_fp4_warp_launch<MODE>(x, sign, q, nullptr, rows, k, u, sfl, nw, eps, u_ld, out_norm, st);
        return true;
    }
    if (k <= 5 * kSpan)
        ptq1_rotq_rows_i8_kernel<5, MODE, true><<<rows, 256, 0, st>>>(
            x, sign, qq, nullptr, nullptr, rows, k, u, sfl, nw, eps, u_ld, out_norm);
    else
        ptq1_rotq_rows_i8_kernel<8, MODE, true><<<rows, 256, 0, st>>>(
            x, sign, qq, nullptr, nullptr, rows, k, u, sfl, nw, eps, u_ld, out_norm);
    return true;
}

bool launch_ptq1_norm_rotq_rows_nvfp4(const void* x_bf16, const void* weight_bf16, float eps,
                                      void* out_norm, const signed char* sign, void* q, int rows,
                                      int k, int block, cudaStream_t st, void* sf_cutlass) {
    if (!weight_bf16) return false;
    return rotq_rows_nvfp4_fused<kRotNorm>(static_cast<const __nv_bfloat16*>(x_bf16), nullptr, 0,
                                           static_cast<const __nv_bfloat16*>(weight_bf16), eps,
                                           static_cast<__nv_bfloat16*>(out_norm), sign, q, rows,
                                           k, block, st, sf_cutlass);
}

bool launch_ptq1_gnorm_rotq_rows_nvfp4(const void* x_bf16, const void* z_bf16,
                                       const void* weight_bf16, float eps, int head_dim,
                                       const signed char* sign, void* q, int rows, int k,
                                       int block, cudaStream_t st, void* sf_cutlass) {
    if (!z_bf16 || !weight_bf16 || head_dim != kBlk) return false;
    return rotq_rows_nvfp4_fused<kRotGnorm>(static_cast<const __nv_bfloat16*>(x_bf16),
                                            static_cast<const __nv_bfloat16*>(z_bf16), 0,
                                            static_cast<const __nv_bfloat16*>(weight_bf16), eps,
                                            nullptr, sign, q, rows, k, block, st, sf_cutlass);
}

bool launch_ptq1_gate_rotq_rows_nvfp4(const void* x_bf16, const void* gate_bf16, int gate_ld,
                                      const signed char* sign, void* q, int rows, int k,
                                      int block, cudaStream_t st, void* sf_cutlass) {
    if (!gate_bf16) return false;
    return rotq_rows_nvfp4_fused<kRotGate>(static_cast<const __nv_bfloat16*>(x_bf16),
                                           static_cast<const __nv_bfloat16*>(gate_bf16), gate_ld,
                                           nullptr, 0.f, nullptr, sign, q, rows, k, block, st,
                                           sf_cutlass);
}

bool launch_ptq1_rotq_rows_i8(const void* x_bf16, const signed char* sign, signed char* q,
                              float* scale, signed char* qp, int rows, int k, int block,
                              cudaStream_t st) {
    if (rows <= 0 || block != kSpan || k % kSpan != 0 || k > 8 * kSpan) return false;
    const auto* x = static_cast<const __nv_bfloat16*>(x_bf16);
    if (k <= 5 * kSpan)
        ptq1_rotq_rows_i8_kernel<5><<<rows, 256, 0, st>>>(x, sign, q, scale, qp, rows, k);
    else
        ptq1_rotq_rows_i8_kernel<8><<<rows, 256, 0, st>>>(x, sign, q, scale, qp, rows, k);
    return true;
}

}}  // namespace sparkinfer::kernels
