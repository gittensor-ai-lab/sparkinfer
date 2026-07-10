// Requantize a dense-FFN down projection from Q6_K to Q4_K once at model load, then
// decode it on-read via the existing int8 dp4a Q4_K MMVQ. On Qwythos (32 dense layers,
// down = [4096,12288] Q6_K) the down weights are ~1.3 GB of the per-token weight read;
// dropping them from 6.5625 to 4.5 bits/weight trims ~31% of that read (a load-time-only
// cost). Gated by SPARKINFER_DOWN_REQUANT_Q4K in the runtime; the down projection is
// accuracy-sensitive, so the fit below is validated against the KL/top-1 gate.
//
// The per-group fit is a min/max seed followed by a fixed-offset least-squares scale
// refinement — a standard affine-quantizer construction, packed into the ggml Q4_K
// super-block layout that si_vec_dot_q4_K consumes (y = d*sc*q - dmin*m).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace sparkinfer { namespace kernels {

// Matches si_block_q4_K: {d,dmin fp16}{8x 6-bit scale + 8x 6-bit min, packed}{256x 4-bit}.
struct sdq_q4k_block { __half2 dm; unsigned char sc[12]; unsigned char nib[128]; };

__device__ __forceinline__ int sdq_round(float v) { return (int)floorf(v + 0.5f); }

// Fit one 32-value group to y ~= s*q + o with q in [0,15]. The Q4_K reconstruction is
// y = d*sc*q - dmin*m, so the group offset o = -negmin is constrained <= 0. Seeds from
// min/max, then refines the scale in closed form against the fixed offset (two passes).
__device__ void sdq_fit_group(const float* v, float* out_scale, float* out_negmin,
                              unsigned char* code) {
    float lo = v[0], hi = v[0];
    #pragma unroll
    for (int i = 1; i < 32; ++i) { lo = fminf(lo, v[i]); hi = fmaxf(hi, v[i]); }
    if (lo > 0.f) lo = 0.f;                       // offset must be <= 0
    if (hi <= lo) {                               // degenerate (constant / all-zero) group
        #pragma unroll
        for (int i = 0; i < 32; ++i) code[i] = 0;
        *out_scale = 0.f; *out_negmin = -lo; return;
    }
    float s = (hi - lo) / 15.f;
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const float inv = 1.f / s;
        float num = 0.f, den = 0.f;               // least-squares: s = sum(q*(v-lo)) / sum(q*q)
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int q = sdq_round(inv * (v[i] - lo));
            q = q < 0 ? 0 : (q > 15 ? 15 : q);
            code[i] = (unsigned char)q;
            num += (float)q * (v[i] - lo);
            den += (float)q * (float)q;
        }
        if (den > 0.f) s = num / den;             // refine scale, keep offset (lo) fixed
    }
    *out_scale = s; *out_negmin = -lo;
}

// One thread per 256-value super-block (8 groups of 32).
__global__ void sdq_requant_kernel(const __nv_bfloat16* __restrict__ src,
                                   sdq_q4k_block* __restrict__ dst, long n_super) {
    const long sb = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= n_super) return;
    const __nv_bfloat16* base = src + sb * 256;
    float grpS[8], grpM[8];
    unsigned char q[256];
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        float buf[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) buf[i] = __bfloat162float(base[g * 32 + i]);
        sdq_fit_group(buf, &grpS[g], &grpM[g], q + g * 32);
    }
    float topS = 0.f, topM = 0.f;
    #pragma unroll
    for (int g = 0; g < 8; ++g) { topS = fmaxf(topS, grpS[g]); topM = fmaxf(topM, grpM[g]); }
    const float d = topS / 63.f, dmin = topM / 63.f;
    unsigned char s6[8], m6[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        int a = d > 0.f ? sdq_round(grpS[g] / d) : 0;    a = a < 0 ? 0 : (a > 63 ? 63 : a);
        int b = dmin > 0.f ? sdq_round(grpM[g] / dmin) : 0; b = b < 0 ? 0 : (b > 63 ? 63 : b);
        s6[g] = (unsigned char)a; m6[g] = (unsigned char)b;
    }
    sdq_q4k_block blk;
    blk.dm = __floats2half2_rn(d, dmin);
    // ggml get_scale_min_k4 inverse packing (the single valid layout for si_vec_dot_q4_K).
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
    #pragma unroll
    for (int i = 0; i < 128; ++i) blk.nib[i] = 0;
    #pragma unroll
    for (int g = 0; g < 8; ++g)
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int byte = (g >> 1) * 32 + i;
            if (g & 1) blk.nib[byte] |= (unsigned char)(q[g * 32 + i] << 4);
            else       blk.nib[byte] |= (unsigned char)(q[g * 32 + i] & 0xF);
        }
    dst[sb] = blk;
}

// n_values must be a multiple of 256.
void launch_ffn_down_requant_q4k(const void* src_bf16, void* dst_q4k, long n_values,
                                 cudaStream_t stream) {
    const long n_super = n_values / 256;
    const int threads = 128;
    const long blocks = (n_super + threads - 1) / threads;
    sdq_requant_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src_bf16),
        reinterpret_cast<sdq_q4k_block*>(dst_q4k), n_super);
}

}} // namespace sparkinfer::kernels
