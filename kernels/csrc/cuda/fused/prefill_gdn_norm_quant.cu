// Fused gated RMSNorm + int8 row-quantize for the Gated-DeltaNet prefill o_proj input.
//
// The batched GDN prefill layer writes att = gdn_scan(...), runs a standalone gated RMSNorm
// (out = (att/rms(att)) * weight * silu(z), pf_gated_norm_kernel) into a bf16 scratch buffer
// lnrm, and then -- whenever the o_proj GEMM takes the residual-fused int8 path -- a separate
// per-row int8 quantizer re-reads all of lnrm before the GEMM. lnrm exists only to ferry values
// between those two kernels; fusing them removes its store and its reload, the same DRAM
// round-trip elimination prefill_swiglu_quant.cu already applies to the FFN down-proj input,
// applied here to the GDN o_proj input instead.
//
// One block per token, HPW heads per warp (v_heads = 8*HPW). Each warp keeps its heads' raw
// x/z values in registers, computes the per-head RMS norm and the bf16-rounded gated output
// (same math and rounding order as pf_gated_norm_kernel -> numerically identical) without a
// second DRAM pass, contributes its local max|.| to a block-wide reduction for the per-row int8
// scale (same amax/127 scale and roundf as the standalone row quantizer -- max is associative
// and exact in fp, so this reduction order gives the identical amax), then writes int8 straight
// from the registers it already holds.
#include "sparkinfer/kernels/prefill.h"
#include "sparkinfer/kernels/prefill_fp8.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdlib>

namespace sparkinfer {
namespace kernels {

namespace {

// Matches FP8_TGT in prefill_gemm_fp8.cu -- the fp16-accumulate GEMM's overflow bound depends on
// activations being scaled to this same target, so it cannot drift from that file's constant.
constexpr float GNQ_FP8_TGT = 2.0f;

__device__ __forceinline__ float gnq_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float gnq_silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float gnq_wsum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

template <int HEAD_DIM, int HPW>
__global__ __launch_bounds__(256) void pf_gdn_norm_quant_i8_kernel(
        const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ z,
        const __nv_bfloat16* __restrict__ weight,
        signed char* __restrict__ q, float* __restrict__ scale,
        int n_tokens, int v_heads, float eps) {
    constexpr int NROW = HEAD_DIM / 32;
    constexpr int N_WARPS = 256 / 32;
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // weight[hd] is the SAME vector for every head and every token -- load once per thread.
    float wv[NROW];
    #pragma unroll
    for (int r = 0; r < NROW; r++) wv[r] = gnq_to_f(weight[lane + r * 32]);

    // Each warp owns HPW contiguous heads; keep the bf16-rounded gated output in registers so
    // the quantize pass below never re-reads x/z or a materialized norm output from DRAM.
    float ov[HPW][NROW];
    float amax = 0.f;
    #pragma unroll
    for (int hh = 0; hh < HPW; hh++) {
        const int h = warp * HPW + hh;
        const size_t base = ((size_t)t * v_heads + h) * HEAD_DIM;
        float xv[NROW], zv[NROW], ss = 0.f;
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int d = lane + r * 32;
            xv[r] = gnq_to_f(x[base + d]);
            zv[r] = gnq_to_f(z[base + d]);
            ss += xv[r] * xv[r];
        }
        const float inv = rsqrtf(gnq_wsum(ss) / HEAD_DIM + eps);
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const float o = xv[r] * inv * wv[r] * gnq_silu(zv[r]);
            const float ob = __bfloat162float(__float2bfloat16(o));   // matches the bf16 store/reload it replaces
            ov[hh][r] = ob;
            amax = fmaxf(amax, fabsf(ob));
        }
    }

    // Block-wide (all v_heads of this token) max reduction for the int8 row scale.
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    __shared__ float s_warp[N_WARPS];
    if (lane == 0) s_warp[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < N_WARPS) ? s_warp[lane] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (lane == 0) s_warp[0] = v;
    }
    __syncthreads();
    const float row_amax = s_warp[0];
    const float d = row_amax / 127.0f;
    if (threadIdx.x == 0) scale[t] = d;

    #pragma unroll
    for (int hh = 0; hh < HPW; hh++) {
        const int h = warp * HPW + hh;
        const size_t base = ((size_t)t * v_heads + h) * HEAD_DIM;
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int dd = lane + r * 32;
            q[base + dd] = (signed char)((row_amax == 0.f) ? 0 : (int)roundf(ov[hh][r] / d));
        }
    }
}

// Same fusion, e4m3 output: for MoE (Qwen3.6) GDN layers, which always run the o_proj GEMM on the
// fp8 tensor cores (moe_fp8, default on) rather than int8 (use_i8 defaults off for MoE -- the
// discrete top-k router amplifies int8 activation-quant error). Numerically identical to
// pf_gated_norm_kernel followed by pf_quantize_rows_fp8_kernel (same amax/FP8_TGT scale, same
// e4m3 rounding), just without the bf16 lnrm round trip.
template <int HEAD_DIM, int HPW>
__global__ __launch_bounds__(256) void pf_gdn_norm_quant_fp8_kernel(
        const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ z,
        const __nv_bfloat16* __restrict__ weight,
        __nv_fp8_e4m3* __restrict__ q, float* __restrict__ scale,
        int n_tokens, int v_heads, float eps) {
    constexpr int NROW = HEAD_DIM / 32;
    constexpr int N_WARPS = 256 / 32;
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    float wv[NROW];
    #pragma unroll
    for (int r = 0; r < NROW; r++) wv[r] = gnq_to_f(weight[lane + r * 32]);

    float ov[HPW][NROW];
    float amax = 0.f;
    #pragma unroll
    for (int hh = 0; hh < HPW; hh++) {
        const int h = warp * HPW + hh;
        const size_t base = ((size_t)t * v_heads + h) * HEAD_DIM;
        float xv[NROW], zv[NROW], ss = 0.f;
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int d = lane + r * 32;
            xv[r] = gnq_to_f(x[base + d]);
            zv[r] = gnq_to_f(z[base + d]);
            ss += xv[r] * xv[r];
        }
        const float inv = rsqrtf(gnq_wsum(ss) / HEAD_DIM + eps);
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const float o = xv[r] * inv * wv[r] * gnq_silu(zv[r]);
            const float ob = __bfloat162float(__float2bfloat16(o));
            ov[hh][r] = ob;
            amax = fmaxf(amax, fabsf(ob));
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    __shared__ float s_warp[N_WARPS];
    if (lane == 0) s_warp[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < N_WARPS) ? s_warp[lane] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (lane == 0) s_warp[0] = v;
    }
    __syncthreads();
    const float row_amax = s_warp[0];
    const float d = (row_amax == 0.f) ? 1.f : (row_amax / GNQ_FP8_TGT);
    if (threadIdx.x == 0) scale[t] = d;

    #pragma unroll
    for (int hh = 0; hh < HPW; hh++) {
        const int h = warp * HPW + hh;
        const size_t base = ((size_t)t * v_heads + h) * HEAD_DIM;
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int dd = lane + r * 32;
            q[base + dd] = __nv_fp8_e4m3(ov[hh][r] / d);
        }
    }
}

}  // namespace

bool launch_prefill_gated_norm_quant_i8(const void* x, const void* z, const void* weight,
                                        signed char* q, float* scale,
                                        int n_tokens, int v_heads, int head_dim, float eps,
                                        cudaStream_t stream) {
    constexpr int BLOCK = 256, N_WARPS = BLOCK / 32;

    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_PREFILL_GDN_NORM_QUANT");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    if (!enabled || head_dim != 128 || n_tokens <= 0 || v_heads <= 0 || (v_heads % N_WARPS) != 0)
        return false;

    const auto xb = reinterpret_cast<const __nv_bfloat16*>(x);
    const auto zb = reinterpret_cast<const __nv_bfloat16*>(z);
    const auto wb = reinterpret_cast<const __nv_bfloat16*>(weight);
    const int hpw = v_heads / N_WARPS;
    switch (hpw) {
        case 1: pf_gdn_norm_quant_i8_kernel<128, 1><<<n_tokens, BLOCK, 0, stream>>>(xb, zb, wb, q, scale, n_tokens, v_heads, eps); break;
        case 2: pf_gdn_norm_quant_i8_kernel<128, 2><<<n_tokens, BLOCK, 0, stream>>>(xb, zb, wb, q, scale, n_tokens, v_heads, eps); break;
        case 4: pf_gdn_norm_quant_i8_kernel<128, 4><<<n_tokens, BLOCK, 0, stream>>>(xb, zb, wb, q, scale, n_tokens, v_heads, eps); break;
        case 8: pf_gdn_norm_quant_i8_kernel<128, 8><<<n_tokens, BLOCK, 0, stream>>>(xb, zb, wb, q, scale, n_tokens, v_heads, eps); break;
        default: return false;
    }
    return true;
}

bool launch_prefill_gated_norm_quant_fp8(const void* x, const void* z, const void* weight,
                                         void* q, float* scale,
                                         int n_tokens, int v_heads, int head_dim, float eps,
                                         cudaStream_t stream) {
    constexpr int BLOCK = 256, N_WARPS = BLOCK / 32;

    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_PREFILL_GDN_NORM_QUANT");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    if (!enabled || head_dim != 128 || n_tokens <= 0 || v_heads <= 0 || (v_heads % N_WARPS) != 0)
        return false;

    const auto xb = reinterpret_cast<const __nv_bfloat16*>(x);
    const auto zb = reinterpret_cast<const __nv_bfloat16*>(z);
    const auto wb = reinterpret_cast<const __nv_bfloat16*>(weight);
    const auto qb = reinterpret_cast<__nv_fp8_e4m3*>(q);
    const int hpw = v_heads / N_WARPS;
    switch (hpw) {
        case 1: pf_gdn_norm_quant_fp8_kernel<128, 1><<<n_tokens, BLOCK, 0, stream>>>(xb, zb, wb, qb, scale, n_tokens, v_heads, eps); break;
        case 2: pf_gdn_norm_quant_fp8_kernel<128, 2><<<n_tokens, BLOCK, 0, stream>>>(xb, zb, wb, qb, scale, n_tokens, v_heads, eps); break;
        case 4: pf_gdn_norm_quant_fp8_kernel<128, 4><<<n_tokens, BLOCK, 0, stream>>>(xb, zb, wb, qb, scale, n_tokens, v_heads, eps); break;
        case 8: pf_gdn_norm_quant_fp8_kernel<128, 8><<<n_tokens, BLOCK, 0, stream>>>(xb, zb, wb, qb, scale, n_tokens, v_heads, eps); break;
        default: return false;
    }
    return true;
}

}  // namespace kernels
}  // namespace sparkinfer
