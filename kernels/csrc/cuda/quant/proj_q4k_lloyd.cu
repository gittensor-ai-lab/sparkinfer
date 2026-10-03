// Load-time Q4_K requantizer for Qwen3.6 attention/GDN projection weights, fit by
// Lloyd-max coordinate descent, then REFINED against the quantized super-block levels
// (pql_refine_group) -- measured +0.7 to +1.1 points of top-1 agreement with llama.cpp and a
// ~30% lower mean KL for zero decode cost; see that function and the launcher.
//
// Lloyd-max coordinate descent. Each 32-value group is quantized to an affine code
// y = s*l + o (l in [0,15], o <= 0) by alternating (a) nearest-code assignment and
// (b) a joint weighted least-squares solve for (s, o) to convergence — a genuinely
// iterative fit, distinct from a single min/max + fixed-offset scale pass and from a
// one-shot iscale grid search. After the 8 group scales/mins are quantized to the
// shared 6-bit super-block resolution, the 4-bit codes are RE-FIT against the actual
// reconstruction levels (d*sc, dmin*m) so the packed nibbles match what si_vec_dot_q4_K
// will read back (y = d*sc*l - dmin*m). Load-time only; n_values must be a multiple of 256.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdlib.h>

namespace sparkinfer { namespace kernels {

// ggml Q4_K super-block: {d, dmin : fp16}{8x6-bit scale + 8x6-bit min, packed}{256x4-bit}.
struct pql_q4k_block { __half2 dm; unsigned char sc[12]; unsigned char nib[128]; };

__device__ __forceinline__ int pql_iclamp(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
__device__ __forceinline__ int pql_rint(float v) { return (int)floorf(v + 0.5f); }

// Fit 32 values by Lloyd coordinate descent. Emits the per-group scale (return value),
// the non-negative packed min (*out_min = -offset), and the 4-bit codes. The importance
// weight w_i = 1 + |x_i| mildly favors the large-magnitude weights that dominate the
// GEMV dot product without letting outliers own the whole scale.
__device__ float pql_fit_group(const float* __restrict__ x, float* out_min,
                               unsigned char* __restrict__ code) {
    float lo = x[0], hi = x[0];
    #pragma unroll
    for (int i = 1; i < 32; ++i) { lo = fminf(lo, x[i]); hi = fmaxf(hi, x[i]); }
    if (lo > 0.f) lo = 0.f;                       // additive offset o <= 0
    if (hi <= lo) {                               // constant / all-zero group
        #pragma unroll
        for (int i = 0; i < 32; ++i) code[i] = 0;
        *out_min = -lo; return 0.f;
    }
    float w[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) w[i] = 1.f + fabsf(x[i]);

    float scale = (hi - lo) / 15.f;               // min/max seed
    float off   = lo;                             // additive offset (<= 0)
    #pragma unroll
    for (int it = 0; it < 8; ++it) {
        const float inv = scale > 0.f ? 1.f / scale : 0.f;
        float sw = 0.f, swl = 0.f, swl2 = 0.f, swx = 0.f, swlx = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {            // (a) assign each value to its nearest code
            int l = pql_iclamp(pql_rint((x[i] - off) * inv), 0, 15);
            code[i] = (unsigned char)l;
            const float wi = w[i], fl = (float)l;
            sw += wi; swl += wi * fl; swl2 += wi * fl * fl;
            swx += wi * x[i]; swlx += wi * fl * x[i];
        }
        const float D = sw * swl2 - swl * swl;    // (b) joint weighted LS for (scale, off)
        if (D > 1e-12f) {
            float ns = (sw * swlx - swx * swl) / D;
            float no = (swl2 * swx - swl * swlx) / D;
            if (no > 0.f) {                        // clamp offset to <= 0, refit scale alone
                no = 0.f;
                ns = swl2 > 0.f ? swlx / swl2 : ns;
            }
            if (ns > 0.f) { scale = ns; off = no; }
        }
    }
    *out_min = -off;
    return scale;
}

// Weighted reconstruction error of one 32-value group under a CONCRETE pair of quantized
// super-block levels (dg = d*s6, og = dmin*m6), with the 4-bit codes re-fit against them.
// This is the quantity the group's packed bytes will actually incur at read-back time --
// y = dg*l - og -- which is what makes the refinement below meaningful.
__device__ __forceinline__ float pql_group_err(const float* __restrict__ x, float dg, float og) {
    float err = 0.f;
    for (int i = 0; i < 32; ++i) {
        const int l = dg > 0.f ? pql_iclamp(pql_rint((x[i] + og) / dg), 0, 15) : 0;
        const float r = dg * (float)l - og - x[i];
        err += (1.f + fabsf(x[i])) * r * r;       // same importance weight as pql_fit_group
    }
    return err;
}

// Joint refinement of the 6-bit (scale, min) pair for one group.
//
// pql_fit_group optimizes (scale, off) in CONTINUOUS space, and only afterwards are both
// rounded onto the shared 6-bit super-block grid (s6 = rint(scale/d), m6 = rint(min/dmin)).
// Nothing in the fit accounts for that rounding, so the continuous optimum's nearest grid
// point is frequently not the best grid point -- the two errors interact, and the codes are
// re-fit against whichever pair is chosen. Rounding is worst exactly where a group's scale is
// small relative to the super-block max, since d is pinned to that max: a group at topS/13
// lands on a grid whose relative step there is ~4%.
//
// So evaluate the 3x3 neighbourhood of the rounded pair against the real read-back error and
// keep the best. Load-time only -- the output is the same Q4_K byte layout, read by the same
// si_vec_dot_q4_K, so decode speed and weight traffic are bit-for-bit unchanged. This buys
// accuracy with load time, not with throughput.
__device__ __forceinline__ void pql_refine_group(const float* __restrict__ x, float d, float dmin,
                                                 int& s6, int& m6) {
    float best = pql_group_err(x, d * (float)s6, dmin * (float)m6);
    int bs = s6, bm = m6;
    for (int ds = -1; ds <= 1; ++ds) {
        const int cs = pql_iclamp(s6 + ds, 0, 63);
        for (int dm = -1; dm <= 1; ++dm) {
            const int cm = pql_iclamp(m6 + dm, 0, 63);
            if (cs == s6 && cm == m6) continue;
            const float e = pql_group_err(x, d * (float)cs, dmin * (float)cm);
            if (e < best) { best = e; bs = cs; bm = cm; }
        }
    }
    s6 = bs; m6 = bm;
}

// Fit + pack one 256-value super-block into a Q4_K block.
__device__ void pql_pack_superblock(const float* __restrict__ x, pql_q4k_block* __restrict__ dst,
                                    bool refine) {
    float grpS[8], grpM[8];
    unsigned char code[256];
    #pragma unroll
    for (int g = 0; g < 8; ++g)
        grpS[g] = pql_fit_group(x + g * 32, &grpM[g], code + g * 32);

    float topS = 0.f, topM = 0.f;
    #pragma unroll
    for (int g = 0; g < 8; ++g) { topS = fmaxf(topS, grpS[g]); topM = fmaxf(topM, grpM[g]); }
    const float d = topS / 63.f, dmin = topM / 63.f;

    unsigned char s6[8], m6[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        int si = pql_iclamp(d    > 0.f ? pql_rint(grpS[g] / d)    : 0, 0, 63);
        int mi = pql_iclamp(dmin > 0.f ? pql_rint(grpM[g] / dmin) : 0, 0, 63);
        if (refine) pql_refine_group(x + g * 32, d, dmin, si, mi);
        s6[g] = (unsigned char)si;
        m6[g] = (unsigned char)mi;
    }

    pql_q4k_block blk;
    blk.dm = __floats2half2_rn(d, dmin);
    // ggml get_scale_min_k4 inverse packing (the only layout si_vec_dot_q4_K accepts).
    #pragma unroll
    for (int i = 0; i < 12; ++i) blk.sc[i] = 0;
    #pragma unroll
    for (int g = 0; g < 4; ++g) { blk.sc[g] = s6[g]; blk.sc[g + 4] = m6[g]; }
    #pragma unroll
    for (int g = 4; g < 8; ++g) {
        blk.sc[g + 4] = (unsigned char)((s6[g] & 0xF) | ((m6[g] & 0xF) << 4));
        blk.sc[g - 4] |= (unsigned char)(((s6[g] >> 4) & 3) << 6);
        blk.sc[g]     |= (unsigned char)(((m6[g] >> 4) & 3) << 6);
    }
    // Re-fit the 4-bit codes against the *quantized* reconstruction levels d*sc, dmin*m.
    #pragma unroll
    for (int i = 0; i < 128; ++i) blk.nib[i] = 0;
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        const float dg = d * (float)s6[g];
        const float og = dmin * (float)m6[g];      // reconstruction: y = dg*l - og
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int l = dg > 0.f ? pql_iclamp(pql_rint((x[g * 32 + i] + og) / dg), 0, 15)
                             : (int)code[g * 32 + i];
            const int byte = (g >> 1) * 32 + i;
            if (g & 1) blk.nib[byte] |= (unsigned char)(l << 4);
            else       blk.nib[byte] |= (unsigned char)(l & 0xF);
        }
    }
    *dst = blk;
}

__global__ void pql_requant_kernel(const __nv_bfloat16* __restrict__ src,
                                   pql_q4k_block* __restrict__ dst, long n_super, bool refine) {
    const long sb = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= n_super) return;
    const __nv_bfloat16* base = src + sb * 256;
    float x[256];
    #pragma unroll
    for (int i = 0; i < 256; ++i) x[i] = __bfloat162float(base[i]);
    pql_pack_superblock(x, dst + sb, refine);
}

// bf16 -> Q4_K (ggml super-block layout consumed by si_vec_dot_q4_K), Lloyd fit.
// n_values must be a multiple of 256.
void launch_proj_requant_q4k_lloyd(const void* src_bf16, void* dst_q4k, long n_values,
                                   cudaStream_t stream) {
    // SPARKINFER_PROJ_REQUANT_REFINE=0 restores the plain round-to-nearest 6-bit scale/min
    // (both arms out of one binary, for A/B). Default ON: it is strictly a load-time cost --
    // the emitted bytes are the same Q4_K layout, so decode reads the same weights at the
    // same speed either way.
    static const bool refine = [] {
        const char* e = getenv("SPARKINFER_PROJ_REQUANT_REFINE");
        return !(e && e[0] == '0');
    }();
    const long n_super = n_values / 256;
    const int threads = 128;
    const long blocks = (n_super + threads - 1) / threads;
    pql_requant_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src_bf16),
        reinterpret_cast<pql_q4k_block*>(dst_q4k), n_super, refine);
}

__global__ void requant_q8_0_kernel(const __nv_bfloat16* __restrict__ src,
                                    unsigned char* __restrict__ dst, long n_blocks) {
    const long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const __nv_bfloat16* x = src + b * 32;
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(__bfloat162float(x[i])));
    const float d = amax / 127.f;
    const float id = d ? 1.f / d : 0.f;
    unsigned char* o = dst + b * 34;
    const __half dh = __float2half(d);
    *reinterpret_cast<unsigned short*>(o) = *reinterpret_cast<const unsigned short*>(&dh);
    #pragma unroll
    for (int i = 0; i < 32; ++i)
        o[2 + i] = (unsigned char)(signed char)roundf(__bfloat162float(x[i]) * id);
}

void launch_requant_q8_0(const void* src_bf16, void* dst_q8_0, long n_values, cudaStream_t stream) {
    const long n_blocks = n_values / 32;
    const int threads = 128;
    requant_q8_0_kernel<<<(unsigned)((n_blocks + threads - 1) / threads), threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src_bf16), reinterpret_cast<unsigned char*>(dst_q8_0),
        n_blocks);
}

}} // namespace sparkinfer::kernels
