// Load-time requant to Q4_K (ggml super-block layout consumed by si_vec_dot_q4_K).
//
// Used for:
//   - dense-FFN down Q6_K -> Q4_K (SPARKINFER_DOWN_REQUANT_Q4K)
//   - Qwen3.6 attn/GDN Q8_0 -> Q4_K (SPARKINFER_Q80_REQUANT_Q4K)
//
// The fitter mirrors llama.cpp's quantize_row_q4_K_ref / make_qkx2_quants: weighted
// RMSE search over (scale, min), then re-encode quants against the packed 6-bit
// scales. The prior min/max + fixed-offset LS fit was too lossy for attn/GDN
// (qkvz KL≈0.37 vs main). Q8 path dequants to fp32 in-registers (no bf16 hop).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>

namespace sparkinfer { namespace kernels {

struct sdq_q4k_block { __half2 dm; unsigned char sc[12]; unsigned char nib[128]; };

__device__ __forceinline__ int sdq_nearest(float v) {
    return (int)roundf(v);
}

// Port of llama.cpp make_qkx2_quants (n=32, nmax=15). Searches iscale steps and
// jointly fits scale+min under weighted SSE. weights may be nullptr -> |x|+rms.
__device__ float sdq_make_qkx2(
    const float* __restrict__ x, const float* __restrict__ weights,
    unsigned char* __restrict__ L, float* __restrict__ the_min,
    unsigned char* __restrict__ Laux,
    float rmin, float rdelta, int nstep
) {
    constexpr int n = 32, nmax = 15;
    float minv = x[0], maxv = x[0], sum_w = 0.f, sum_x = 0.f;
    float wloc[32];
    #pragma unroll
    for (int i = 0; i < n; ++i) {
        float xi = x[i];
        minv = fminf(minv, xi);
        maxv = fmaxf(maxv, xi);
        float w = weights ? weights[i] : fabsf(xi);
        wloc[i] = w;
        sum_w += w;
        sum_x += w * xi;
    }
    if (minv > 0.f) minv = 0.f;
    if (maxv <= minv) {
        #pragma unroll
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -minv;
        return 0.f;
    }
    float iscale = (float)nmax / (maxv - minv);
    float scale = 1.f / iscale;
    float best_error = 0.f;
    #pragma unroll
    for (int i = 0; i < n; ++i) {
        int l = sdq_nearest(iscale * (x[i] - minv));
        l = l < 0 ? 0 : (l > nmax ? nmax : l);
        L[i] = (unsigned char)l;
        float diff = scale * (float)l + minv - x[i];
        best_error += wloc[i] * diff * diff;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * (float)is + (float)nmax) / (maxv - minv);
        float sum_l = 0.f, sum_l2 = 0.f, sum_xl = 0.f;
        #pragma unroll
        for (int i = 0; i < n; ++i) {
            int l = sdq_nearest(iscale * (x[i] - minv));
            l = l < 0 ? 0 : (l > nmax ? nmax : l);
            Laux[i] = (unsigned char)l;
            float w = wloc[i];
            sum_l  += w * (float)l;
            sum_l2 += w * (float)l * (float)l;
            sum_xl += w * (float)l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0.f) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0.f) {
                this_min = 0.f;
                this_scale = (sum_l2 > 0.f) ? (sum_xl / sum_l2) : 0.f;
            }
            float cur_error = 0.f;
            #pragma unroll
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * (float)Laux[i] + this_min - x[i];
                cur_error += wloc[i] * diff * diff;
            }
            if (cur_error < best_error) {
                #pragma unroll
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_error = cur_error;
                scale = this_scale;
                minv = this_min;
            }
        }
    }
    *the_min = -minv;
    return scale;
}

// Fit one 256-value super-block from fp32 into a Q4_K block (llama q4_K_ref).
__device__ void sdq_pack_q4k_from_f32(const float* __restrict__ x, sdq_q4k_block* __restrict__ dst) {
    unsigned char L[256], Laux[32];
    float mins[8], scales[8], weights[32];
    float max_scale = 0.f, max_min = 0.f;
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        float sum_x2 = 0.f;
        #pragma unroll
        for (int l = 0; l < 32; ++l) sum_x2 += x[g * 32 + l] * x[g * 32 + l];
        float av_x = sqrtf(sum_x2 / 32.f);
        #pragma unroll
        for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[g * 32 + l]);
        scales[g] = sdq_make_qkx2(x + g * 32, weights, L + g * 32, &mins[g], Laux, -0.9f, 0.05f, 36);
        max_scale = fmaxf(max_scale, scales[g]);
        max_min   = fmaxf(max_min, mins[g]);
    }
    float inv_scale = max_scale > 0.f ? 63.f / max_scale : 0.f;
    float inv_min   = max_min   > 0.f ? 63.f / max_min   : 0.f;
    unsigned char s6[8], m6[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        int ls = sdq_nearest(inv_scale * scales[g]); ls = ls < 0 ? 0 : (ls > 63 ? 63 : ls);
        int lm = sdq_nearest(inv_min   * mins[g]);   lm = lm < 0 ? 0 : (lm > 63 ? 63 : lm);
        s6[g] = (unsigned char)ls; m6[g] = (unsigned char)lm;
    }
    sdq_q4k_block blk;
    blk.dm = __floats2half2_rn(max_scale / 63.f, max_min / 63.f);
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
    // Re-encode quants against the packed (d*sc, dmin*m) — matches llama.cpp.
    float d = __low2float(blk.dm), dmin = __high2float(blk.dm);
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        int sc = s6[g], mn = m6[g];
        float dg = d * (float)sc;
        if (dg == 0.f) {
            #pragma unroll
            for (int i = 0; i < 32; ++i) L[g * 32 + i] = 0;
            continue;
        }
        float dm = dmin * (float)mn;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int l = sdq_nearest((x[g * 32 + i] + dm) / dg);
            l = l < 0 ? 0 : (l > 15 ? 15 : l);
            L[g * 32 + i] = (unsigned char)l;
        }
    }
    #pragma unroll
    for (int i = 0; i < 128; ++i) blk.nib[i] = 0;
    // ggml Q4_K nibble layout: pairs of groups (g,g+1) share 32 bytes.
    #pragma unroll
    for (int g = 0; g < 8; g += 2) {
        #pragma unroll
        for (int i = 0; i < 32; ++i)
            blk.nib[(g >> 1) * 32 + i] =
                (unsigned char)((L[g * 32 + i] & 0xF) | (L[(g + 1) * 32 + i] << 4));
    }
    *dst = blk;
}

__global__ void sdq_requant_kernel(const __nv_bfloat16* __restrict__ src,
                                   sdq_q4k_block* __restrict__ dst, long n_super) {
    const long sb = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= n_super) return;
    float x[256];
    const __nv_bfloat16* base = src + sb * 256;
    #pragma unroll
    for (int i = 0; i < 256; ++i) x[i] = __bfloat162float(base[i]);
    sdq_pack_q4k_from_f32(x, dst + sb);
}

// Direct Q8_0 -> Q4_K: dequant each Q8_0 block to fp32 (no bf16 hop), then fit.
__global__ void sdq_q80_requant_kernel(const unsigned char* __restrict__ src_q80,
                                       sdq_q4k_block* __restrict__ dst, long n_super) {
    const long sb = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= n_super) return;
    float x[256];
    const unsigned char* base = src_q80 + sb * 8 * 34;   // 8 Q8_0 blocks / Q4_K superblock
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        const unsigned char* blk = base + b * 34;
        __half h; *((unsigned short*)&h) = *(const unsigned short*)blk;
        const float d = __half2float(h);
        const signed char* qs = reinterpret_cast<const signed char*>(blk + 2);
        #pragma unroll
        for (int i = 0; i < 32; ++i) x[b * 32 + i] = d * (float)qs[i];
    }
    sdq_pack_q4k_from_f32(x, dst + sb);
}

void launch_ffn_down_requant_q4k(const void* src_bf16, void* dst_q4k, long n_values,
                                 cudaStream_t stream) {
    const long n_super = n_values / 256;
    const int threads = 128;
    const long blocks = (n_super + threads - 1) / threads;
    sdq_requant_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src_bf16),
        reinterpret_cast<sdq_q4k_block*>(dst_q4k), n_super);
}

void launch_q80_requant_q4k(const void* src_q80, void* dst_q4k, long n_values,
                            cudaStream_t stream) {
    const long n_super = n_values / 256;
    const int threads = 128;
    const long blocks = (n_super + threads - 1) / threads;
    sdq_q80_requant_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        reinterpret_cast<const unsigned char*>(src_q80),
        reinterpret_cast<sdq_q4k_block*>(dst_q4k), n_super);
}

}} // namespace sparkinfer::kernels
