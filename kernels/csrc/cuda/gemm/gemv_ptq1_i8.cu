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

#include <cstdlib>

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
enum : int { kRotPlain = 0, kRotSwiglu = 1, kRotGnorm = 2, kRotGate = 3, kRotAddNorm = 4 };
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
// part[((mat * gridDim.y + y) * M + token) * N + row]; ptq1_split_reduce_kernel sums them in split
// order. For the down projection, whose 5120 rows are only 40 CTAs on a 170-SM part.
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
                     float* __restrict__ part = nullptr, int sps = 0) {
    constexpr int TOK = NT * 8;
    constexpr int ROWB = KB * kBlk + 16;       // +16: a B-fragment load's 8 tokens hit 8 banks
    constexpr int WSEG = KB * kBlkBytes;       // one weight row's bytes per step
    constexpr int WCH = WSEG / 16;
    constexpr int WROWS = WARPS * 16;
    extern __shared__ __align__(16) unsigned char smem_mma[];
    unsigned char* sw = smem_mma;                                              // [ST][WROWS][WSEG]
    signed char* sx = reinterpret_cast<signed char*>(sw + ST * WROWS * WSEG);   // [ST][TOK][ROWB]
    float* sdx = reinterpret_cast<float*>(sx + ST * TOK * ROWB);          // [ST][TOK][KB]
    int* ssx = reinterpret_cast<int*>(sdx + ST * TOK * KB);               // [ST][TOK][KB]
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
        // A token's KB scales (and sums) are one aligned 16-byte run, since nblk and b0 are
        // multiples of KB: copied like the rest, so no warp stalls on a load before its MMAs.
        if (threadIdx.x < 2 * TOK) {
            const int tok = threadIdx.x >> 1;
            const bool ok = tok < M;
            const size_t src = (size_t)(ok ? tok : 0) * nblk + b0;
            const size_t dst = ((size_t)buf * TOK + tok) * KB;
            if (threadIdx.x & 1) __pipeline_memcpy_async(ssx + dst, xs + src, 16, ok ? 0 : 16);
            else                 __pipeline_memcpy_async(sdx + dst, xd + src, 16, ok ? 0 : 16);
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
                const signed char* xr = sx + ((size_t)buf * TOK + n * 8 + g) * ROWB + bb * kBlk;
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const int b0v = *reinterpret_cast<const int*>(xr + 32 * s + 4 * t);
                    const int b1v = *reinterpret_cast<const int*>(xr + 32 * s + 16 + 4 * t);
                    mma_u8s8(cc, fa[s][0], fb[s][0], fa[s][1], fb[s][1], b0v, b1v);
                }
                const int t0 = n * 8 + 2 * t;
                const size_t o0 = ((size_t)buf * TOK + t0) * KB + bb, o1 = o0 + KB;
                const float d0 = sdx[o0], d1 = sdx[o1];
                const int s0 = ssx[o0], s1 = ssx[o1];
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
    launch_rows_pdl(NT == 1, ptq1_mma_rows_kernel<NT, WARPS, KB, ST, OutT, SPLIT>,
                    dim3(ctas * nmat, SPLIT ? S : 1), dim3(WARPS * 32), shm, st, xq, xd, xs,
                    static_cast<const unsigned char*>(w0), static_cast<const unsigned char*>(w1),
                    y0, y1, m, n_rows, nblk, ctas, part, sps);
    if (SPLIT) {
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
// SWIGLU: x is the gate and u the up projection; the row rotated is SwiGLU's output rounded to
// bf16 exactly as launch_prefill_swiglu_quant_i8 forms it, bf16(g / (1 + exp(-g)) * u), which is
// also what the decode shadow's down reads (launch_ptq1_swiglu_rotq_bf16).
template <int NS, bool SWIGLU = false>
__global__ void __launch_bounds__(256)
ptq1_rotq_rows_i8_kernel(const __nv_bfloat16* __restrict__ x, const signed char* __restrict__ sign,
                         signed char* __restrict__ q, float* __restrict__ scale,
                         signed char* __restrict__ qp, int rows, int k,
                         const __nv_bfloat16* __restrict__ u = nullptr) {
    __shared__ float sh[kSpan];
    __shared__ float sred[8];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int row = blockIdx.x;
    const int ns = k / kSpan;
    float v[NS][4];
    float am = 0.f;
#pragma unroll
    for (int sp = 0; sp < NS; ++sp) {
        if (sp < ns) {
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
                    o[j] = __float2bfloat16(g / (1.f + __expf(-g)) * __bfloat162float(uh[j]));
                }
                raw = *reinterpret_cast<const uint2*>(o);
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
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[sp][i] = sh[t * 4 + i] * 0.03125f;
                am = fmaxf(am, fabsf(v[sp][i]));
            }
        }
    }
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

}  // namespace

bool launch_ptq1_rotq_bf16(const void* x_bf16, const signed char* sign, signed char* q,
                           float* qd, int* qs, int rows, int k, int block, cudaStream_t st) {
    if (rows <= 0 || block != kSpan || k % kSpan != 0) return false;
    ptq1_rotq_kernel<kRotPlain><<<dim3((unsigned)(k / kSpan), (unsigned)rows), 256, 0, st>>>(
        static_cast<const __nv_bfloat16*>(x_bf16), nullptr, nullptr, 0.f, sign, q, qd, qs, k);
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
    ptq1_rotq_kernel<kRotAddNorm><<<dim3((unsigned)(k / kSpan), (unsigned)rows), 256, 0, st>>>(
        static_cast<const __nv_bfloat16*>(x_bf16), static_cast<const __nv_bfloat16*>(residual_bf16),
        static_cast<const __nv_bfloat16*>(weight_bf16), eps, sign, q, qd, qs, k,
        static_cast<__nv_bfloat16*>(out_sum), static_cast<__nv_bfloat16*>(out_norm),
        static_cast<i8_blk_q8_1*>(out_q8));
    return true;
}

bool launch_ptq1_swiglu_rotq_bf16(const void* gate_bf16, const void* up_bf16,
                                  const signed char* sign, signed char* q, float* qd, int* qs,
                                  int rows, int k, int block, cudaStream_t st) {
    if (rows <= 0 || block != kSpan || k % kSpan != 0) return false;
    ptq1_rotq_kernel<kRotSwiglu><<<dim3((unsigned)(k / kSpan), (unsigned)rows), 256, 0, st>>>(
        static_cast<const __nv_bfloat16*>(gate_bf16), static_cast<const __nv_bfloat16*>(up_bf16),
        nullptr, 0.f, sign, q, qd, qs, k);
    return true;
}

bool launch_ptq1_gnorm_rotq_bf16(const void* x_bf16, const void* z_bf16, const void* norm_bf16,
                                 float eps, const signed char* sign, signed char* q, float* qd,
                                 int* qs, int rows, int k, int head_dim, int block,
                                 cudaStream_t st) {
    if (rows <= 0 || block != kSpan || k % kSpan != 0 || head_dim != kBlk) return false;
    ptq1_rotq_kernel<kRotGnorm><<<dim3((unsigned)(k / kSpan), (unsigned)rows), 256, 0, st>>>(
        static_cast<const __nv_bfloat16*>(x_bf16), static_cast<const __nv_bfloat16*>(z_bf16),
        static_cast<const __nv_bfloat16*>(norm_bf16), eps, sign, q, qd, qs, k);
    return true;
}

bool launch_ptq1_gate_rotq_bf16(const void* x_bf16, const void* gate_bf16,
                                const signed char* sign, signed char* q, float* qd, int* qs,
                                int rows, int k, int block, cudaStream_t st) {
    if (rows <= 0 || block != kSpan || k % kSpan != 0) return false;
    ptq1_rotq_kernel<kRotGate><<<dim3((unsigned)(k / kSpan), (unsigned)rows), 256, 0, st>>>(
        static_cast<const __nv_bfloat16*>(x_bf16), static_cast<const __nv_bfloat16*>(gate_bf16),
        nullptr, 0.f, sign, q, qd, qs, k);
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
