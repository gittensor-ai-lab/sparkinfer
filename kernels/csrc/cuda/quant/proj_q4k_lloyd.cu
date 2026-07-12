// Load-time Q4_K requantizer for Qwen3.6 attention/GDN projection weights, fit by
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

// Fit + pack one 256-value super-block into a Q4_K block.
__device__ void pql_pack_superblock(const float* __restrict__ x, pql_q4k_block* __restrict__ dst) {
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
        s6[g] = (unsigned char)pql_iclamp(d    > 0.f ? pql_rint(grpS[g] / d)    : 0, 0, 63);
        m6[g] = (unsigned char)pql_iclamp(dmin > 0.f ? pql_rint(grpM[g] / dmin) : 0, 0, 63);
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
                                   pql_q4k_block* __restrict__ dst, long n_super) {
    const long sb = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= n_super) return;
    const __nv_bfloat16* base = src + sb * 256;
    float x[256];
    #pragma unroll
    for (int i = 0; i < 256; ++i) x[i] = __bfloat162float(base[i]);
    pql_pack_superblock(x, dst + sb);
}

// bf16 -> Q4_K (ggml super-block layout consumed by si_vec_dot_q4_K), Lloyd fit.
// n_values must be a multiple of 256.
void launch_proj_requant_q4k_lloyd(const void* src_bf16, void* dst_q4k, long n_values,
                                   cudaStream_t stream) {
    const long n_super = n_values / 256;
    const int threads = 128;
    const long blocks = (n_super + threads - 1) / threads;
    pql_requant_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src_bf16),
        reinterpret_cast<pql_q4k_block*>(dst_q4k), n_super);
}

}} // namespace sparkinfer::kernels
