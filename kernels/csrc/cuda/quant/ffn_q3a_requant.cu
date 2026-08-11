// Requantize a weight matrix from bf16 to Q3_A once at model load, then decode it on-read via
// the int8 dp4a Q3_A MMVQ (si_vec_dot_q3_A in moe/expert_ffn_q4k.cu).
//
// Q3_A is a sparkinfer-internal 3.5 bits/weight super-block: Q4_K with the 4-bit quant plane
// replaced by a 3-bit one (a 2-bit plane + a 1-bit plane), keeping Q4_K's asymmetric per-32 scale
// AND min and Q4_K's exact 6-bit scale/min packing. 112 B per 256 weights against Q4_K's 144 B,
// so a converted tensor is read 22.2% smaller on every decode token. It is never read from or
// written to a GGUF file -- it exists only between this load-time fitter and the matching
// vec-dot, so the layout is chosen to make that vec-dot cheap rather than to match any ggml type.
//
// Layout (per 256-weight super-block; s = sub-block 0..7, p = position 0..31 inside it):
//   dm        : {d, dmin} fp16, exactly Q4_K's
//   scales[12]: 8x 6-bit scale + 8x 6-bit min, exactly Q4_K's get_scale_min_k4 packing
//   qs[64]    : low 2 bits. Indexed by (j = s/2, m = p/8) as one int at byte 16*j + 4*m;
//               inside it byte lane b = p%4 holds four 2-bit fields, field
//               f = (s%2) | ((p%8)/4 << 1) at bit 2*f.
//   qh[32]    : high 1 bit. Byte p holds the high bits of all 8 sub-blocks at position p,
//               bit s -- so one int load covers four positions for every sub-block at once.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

namespace sparkinfer { namespace kernels {

struct sq3a_block { __half2 dm; unsigned char sc[12]; unsigned char qs[64]; unsigned char qh[32]; };  // 112 B

__device__ __forceinline__ int sq3a_round(float v) { return (int)floorf(v + 0.5f); }

// Fit one 32-value group to y ~= s*q + o with q in [0,7]. Q3_A reconstructs as
// y = d*sc*q - dmin*m, so the offset o = -negmin is constrained <= 0.
//
// This is ggml's make_qkx2_quants search (the one quantize_row_q4_K_impl uses), at nmax=7
// instead of 15 and with the same importance weights w = av_x + |x|: seed from min/max, then
// sweep 21 candidate scales around that seed, and for each one refit BOTH scale and offset by
// weighted least squares, keeping whichever candidate minimises the weighted squared error.
//
// A cheaper min/max-plus-two-LS-passes fit (what the Q4_K down requantizer uses) is not good
// enough at 8 levels: measured top1 0.758 / KL 0.497 against this fit's 0.798 / 0.376 on the
// same arm. At 16 levels one outlier stretching the group range wastes one level; at 8 it
// wastes a quarter of the resolution.
__device__ void sq3a_fit_group(const float* v, float* out_scale, float* out_negmin,
                               unsigned char* code) {
    constexpr int NMAX = 7;
    float sum_x2 = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) sum_x2 += v[i] * v[i];
    const float av_x = sqrtf(sum_x2 / 32.f);

    float lo = v[0], hi = v[0], sum_w = 0.f, sum_x = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        lo = fminf(lo, v[i]); hi = fmaxf(hi, v[i]);
        const float w = av_x + fabsf(v[i]);
        sum_w += w; sum_x += w * v[i];
    }
    if (lo > 0.f) lo = 0.f;                       // offset must be <= 0
    if (hi == lo) {                               // degenerate (constant / all-zero) group
        #pragma unroll
        for (int i = 0; i < 32; ++i) code[i] = 0;
        *out_scale = 0.f; *out_negmin = -lo; return;
    }

    float iscale = (float)NMAX / (hi - lo);
    float scale = 1.f / iscale, best_min = lo, best_mad = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        int l = sq3a_round(iscale * (v[i] - lo));
        l = l < 0 ? 0 : (l > NMAX ? NMAX : l);
        code[i] = (unsigned char)l;
        const float diff = scale * (float)l + lo - v[i];
        best_mad += (av_x + fabsf(v[i])) * diff * diff;
    }

    unsigned char aux[32];
    for (int is = 0; is <= 20; ++is) {
        const float isc = (-1.f + 0.1f * (float)is + (float)NMAX) / (hi - lo);
        float sum_l = 0.f, sum_l2 = 0.f, sum_xl = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            int l = sq3a_round(isc * (v[i] - lo));
            l = l < 0 ? 0 : (l > NMAX ? NMAX : l);
            aux[i] = (unsigned char)l;
            const float w = av_x + fabsf(v[i]), lf = (float)l;
            sum_l += w * lf; sum_l2 += w * lf * lf; sum_xl += w * lf * v[i];
        }
        const float D = sum_w * sum_l2 - sum_l * sum_l;
        if (!(D > 0.f)) continue;
        float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
        float this_min   = (sum_l2 * sum_x - sum_l * sum_xl) / D;
        if (this_min > 0.f) {                     // offset clamped to 0: refit the scale alone
            this_min = 0.f;
            this_scale = (sum_l2 > 0.f) ? (sum_xl / sum_l2) : this_scale;
        }
        float mad = 0.f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const float diff = this_scale * (float)aux[i] + this_min - v[i];
            mad += (av_x + fabsf(v[i])) * diff * diff;
        }
        if (mad < best_mad) {
            #pragma unroll
            for (int i = 0; i < 32; ++i) code[i] = aux[i];
            best_mad = mad; scale = this_scale; best_min = this_min;
        }
    }
    if (!(scale > 0.f)) scale = 0.f;              // 6-bit scales are unsigned; never encode < 0
    *out_scale = scale; *out_negmin = -best_min;
}

// Re-derive the codes with error feedback: quantize (v[i] + carried residual) instead of v[i],
// then carry (target - reconstruction) forward. This does not shrink the per-weight error, it
// shifts it so that a RUN of consecutive weights sums to roughly the right total -- which is what
// a dot product against a smooth activation actually integrates. Scale/offset are already fixed
// by the fit above, so this only changes which of the 8 levels each weight lands on.
// OFF by default: measured, it makes things WORSE here -- relative RMS reconstruction error
// against the Q4_K source goes 13.88% -> 23.18%, and the scored top1/KL follow it down
// (0.929/0.179 -> 0.727/0.705). Kept behind SPARKINFER_Q3A_EFB=1 only so the result is
// reproducible; do not turn it on.
__device__ void sq3a_error_feedback(const float* v, float scale, float negmin,
                                    unsigned char* code) {
    if (!(scale > 0.f)) return;
    const float inv = 1.f / scale, o = -negmin;
    float e = 0.f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const float target = v[i] + e;
        int l = sq3a_round(inv * (target - o));
        l = l < 0 ? 0 : (l > 7 ? 7 : l);
        code[i] = (unsigned char)l;
        e = target - (scale * (float)l + o);
        // Do not let the residual run away on a clipped outlier.
        e = fmaxf(-scale, fminf(scale, e));
    }
}

// One thread per 256-value super-block (8 groups of 32).
__global__ void sq3a_requant_kernel(const __nv_bfloat16* __restrict__ src,
                                    sq3a_block* __restrict__ dst, long n_super,
                                    int efb, float* __restrict__ err_acc) {
    const long sb = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= n_super) return;
    const __nv_bfloat16* base = src + sb * 256;
    float grpS[8], grpM[8];
    unsigned char q[256];
    float vbuf[256];
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        float buf[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) { buf[i] = __bfloat162float(base[g * 32 + i]); vbuf[g * 32 + i] = buf[i]; }
        sq3a_fit_group(buf, &grpS[g], &grpM[g], q + g * 32);
        if (efb) sq3a_error_feedback(buf, grpS[g], grpM[g], q + g * 32);
    }
    float topS = 0.f, topM = 0.f;
    #pragma unroll
    for (int g = 0; g < 8; ++g) { topS = fmaxf(topS, grpS[g]); topM = fmaxf(topM, grpM[g]); }
    const float d = topS / 63.f, dmin = topM / 63.f;
    unsigned char s6[8], m6[8];
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        int a = d > 0.f ? sq3a_round(grpS[g] / d) : 0;       a = a < 0 ? 0 : (a > 63 ? 63 : a);
        int b = dmin > 0.f ? sq3a_round(grpM[g] / dmin) : 0; b = b < 0 ? 0 : (b > 63 ? 63 : b);
        s6[g] = (unsigned char)a; m6[g] = (unsigned char)b;
    }
    sq3a_block blk;
    blk.dm = __floats2half2_rn(d, dmin);
    // ggml get_scale_min_k4 inverse packing -- byte-identical to the Q4_K requantizer's, because
    // si_vec_dot_q3_A reuses si_vec_dot_q4_K's scale/min unpack verbatim.
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
    for (int i = 0; i < 64; ++i) blk.qs[i] = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) blk.qh[i] = 0;
    #pragma unroll
    for (int s = 0; s < 8; ++s)
        #pragma unroll
        for (int p = 0; p < 32; ++p) {
            const int val = q[s * 32 + p];
            const int j = s >> 1, m = p >> 3, r = p & 7;
            const int f = (s & 1) | ((r >> 2) << 1);       // 2-bit field inside the byte
            blk.qs[16 * j + 4 * m + (r & 3)] |= (unsigned char)((val & 3) << (2 * f));
            blk.qh[p] |= (unsigned char)(((val >> 2) & 1) << s);
        }
    dst[sb] = blk;

    // Self-check: reconstruct straight out of the PACKED block (not out of q[]), so a
    // pack/unpack mismatch shows up here rather than hiding until the vec-dot. Accumulates
    // sum((w-w_hat)^2) and sum(w^2) so the caller can report a relative RMS and compare it
    // against the theoretical 3.5-bit figure. Diagnostic only.
    if (err_acc) {
        float se = 0.f, sx = 0.f;
        const float d_f = __low2float(blk.dm), dmin_f = __high2float(blk.dm);
        for (int s = 0; s < 8; ++s) {
            // unpack the 6-bit scale/min pair exactly as si_vec_dot_q4_K does
            int sc6, m6u;
            if (s < 4) { sc6 = blk.sc[s] & 63; m6u = blk.sc[s + 4] & 63; }
            else {
                sc6 = (blk.sc[s + 4] & 0xF) | (((blk.sc[s - 4] >> 6) & 3) << 4);
                m6u = (blk.sc[s + 4] >> 4)  | (((blk.sc[s]     >> 6) & 3) << 4);
            }
            for (int p = 0; p < 32; ++p) {
                const int j = s >> 1, m = p >> 3, r = p & 7;
                const int f = (s & 1) | ((r >> 2) << 1);
                const int lo2 = (blk.qs[16 * j + 4 * m + (r & 3)] >> (2 * f)) & 3;
                const int hi1 = (blk.qh[p] >> s) & 1;
                const float w_hat = d_f * (float)sc6 * (float)(lo2 | (hi1 << 2)) - dmin_f * (float)m6u;
                const float w = vbuf[s * 32 + p];
                se += (w - w_hat) * (w - w_hat); sx += w * w;
            }
        }
        atomicAdd(err_acc + 0, se);
        atomicAdd(err_acc + 1, sx);
    }
}

// n_values must be a multiple of 256. dst must hold n_values/256 * 112 bytes.
void launch_ffn_requant_q3a(const void* src_bf16, void* dst_q3a, long n_values,
                            cudaStream_t stream) {
    static int efb = -1;
    if (efb < 0) { const char* e = getenv("SPARKINFER_Q3A_EFB"); efb = (e && e[0] == '1') ? 1 : 0; }
    static int selfcheck = -1;
    if (selfcheck < 0) { const char* e = getenv("SPARKINFER_Q3A_SELFCHECK"); selfcheck = (e && e[0] == '1') ? 1 : 0; }

    const long n_super = n_values / 256;
    const int threads = 128;
    const long blocks = (n_super + threads - 1) / threads;
    float* err = nullptr;
    if (selfcheck && cudaMalloc(&err, 2 * sizeof(float)) == cudaSuccess)
        cudaMemsetAsync(err, 0, 2 * sizeof(float), stream);
    sq3a_requant_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src_bf16),
        reinterpret_cast<sq3a_block*>(dst_q3a), n_super, efb, err);
    if (err) {
        float h[2] = {0.f, 0.f};
        cudaStreamSynchronize(stream);
        cudaMemcpy(h, err, sizeof(h), cudaMemcpyDeviceToHost);
        cudaFree(err);
        if (h[1] > 0.f)
            fprintf(stderr, "[q3a] n=%ld relative RMS reconstruction error = %.4f%% (efb=%d)\n",
                    n_values, 100.0 * sqrt((double)h[0] / (double)h[1]), efb);
    }
}

}} // namespace sparkinfer::kernels
