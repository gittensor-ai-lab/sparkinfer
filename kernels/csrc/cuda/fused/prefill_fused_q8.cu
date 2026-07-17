// Fused elementwise->int8 producers for the Qwythos batched prefill.
//
// The prefill activation path materializes a bf16 tensor, then re-reads it twice in
// pf_quantize_rows_i8 (once for the row amax, once to quantize). Measured on RTX 5090 @32k
// (nsys, main c9e20d1), that elementwise+quantize path is 9.4% of prefill:
//   pf_quantize_rows_i8 4.2% | pf_swiglu 2.5% | pf_add 1.6% | rmsnorm 1.1%
// while the DECODE path already fuses the same work (add_rmsnorm2_q8_kernel emits its Q8_1
// inline). These kernels port that pattern to prefill, in the per-row int8 + per-row float
// scale format pf_gemm_i8 consumes.
//
// The big one is SwiGLU: ffh feeds only proj(ffh, w.down_q, ..., H, ffn) with n_out = H = 4096
// >= 128, i.e. the int8 path exclusively -- nothing reads ffh in bf16 -- so the fused kernel
// never writes it. Per element: 11B (2 read + 2B write ffh, then 2x2B read + 1B write q)
// becomes 5B (2B + 2B read, 1B write q). The row (ffn=12288 bf16 = 24KB) stages in shared, so
// the amax pass and the quantize pass both hit shared instead of HBM.
//
// Numerics are bit-identical to the unfused pair: same silu/mul math, same amax/127 scale, same
// roundf. SPARKINFER_PREFILL_FUSED_Q8=0 restores the separate kernels.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdlib>

namespace sparkinfer { namespace kernels {
namespace {

__device__ __forceinline__ float fq_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float fq_silu(float x) { return x / (1.f + __expf(-x)); }

// Warp-max then block-max over the block's warps.
template <int NT>
__device__ __forceinline__ float fq_block_amax(float v, float* red) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = 0.f;
        #pragma unroll
        for (int w = 0; w < NT / 32; ++w) m = fmaxf(m, red[w]);
        red[0] = m;
    }
    __syncthreads();
    return red[0];
}

// out_q[r][c] = round( silu(gate[r][c]) * up[r][c] / d ),  d = amax(row)/127,  scale[r] = d.
// One block per row; the row is staged in shared so amax + quantize never re-read HBM.
// ffh (bf16) is NOT written -- nothing consumes it outside the int8 GEMM.
template <int NT>
__global__ __launch_bounds__(NT, 1)
void pf_swiglu_q8_kernel(const __nv_bfloat16* __restrict__ gate,
                         const __nv_bfloat16* __restrict__ up,
                         signed char* __restrict__ q, float* __restrict__ scale,
                         int rows, int cols) {
    extern __shared__ char sfq[];
    float* red = reinterpret_cast<float*>(sfq);          // [NT/32] warp-max staging
    __nv_bfloat16* row = reinterpret_cast<__nv_bfloat16*>(red + (NT / 32));  // [cols]

    const int r = blockIdx.x;
    if (r >= rows) return;
    const size_t base = (size_t)r * cols;

    float amax = 0.f;
    for (int c = threadIdx.x; c < cols; c += NT) {
        const float v = fq_silu(fq_f(gate[base + c])) * fq_f(up[base + c]);
        row[c] = __float2bfloat16(v);                    // stage bf16 -- matches the unfused
        amax = fmaxf(amax, fabsf(fq_f(row[c])));         // ffh write, so amax sees the same bits
    }
    amax = fq_block_amax<NT>(amax, red);

    const float d = amax / 127.0f;
    if (threadIdx.x == 0) scale[r] = d;
    for (int c = threadIdx.x; c < cols; c += NT)
        q[base + c] = (signed char)((amax == 0.f) ? 0 : (int)roundf(fq_f(row[c]) / d));
}

// out_bf16[r][c] = rmsnorm(x)[r][c] * w[c]   AND   out_q/scale = row-quantized(out_bf16).
// The bf16 output is still written: the tiny per-v-head gate projections (n_out < 128) consume
// it on the bf16 path. The saving is the two HBM re-reads pf_quantize_rows_i8 would do.
template <int NT>
__global__ __launch_bounds__(NT, 1)
void pf_rmsnorm_q8_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                          __nv_bfloat16* __restrict__ out, signed char* __restrict__ q,
                          float* __restrict__ scale, int rows, int cols, float eps) {
    extern __shared__ char sfq[];
    float* red = reinterpret_cast<float*>(sfq);
    __nv_bfloat16* row = reinterpret_cast<__nv_bfloat16*>(red + (NT / 32));

    const int r = blockIdx.x;
    if (r >= rows) return;
    const size_t base = (size_t)r * cols;

    float ss = 0.f;
    for (int c = threadIdx.x; c < cols; c += NT) { const float v = fq_f(x[base + c]); ss += v * v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    { const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
      if (lane == 0) red[warp] = ss;
      __syncthreads();
      if (threadIdx.x == 0) { float s = 0.f;
          #pragma unroll
          for (int wi = 0; wi < NT / 32; ++wi) s += red[wi];
          red[0] = s; }
      __syncthreads(); ss = red[0]; }
    const float inv = rsqrtf(ss / (float)cols + eps);

    float amax = 0.f;
    for (int c = threadIdx.x; c < cols; c += NT) {
        const __nv_bfloat16 v = __float2bfloat16(fq_f(x[base + c]) * inv * fq_f(w[c]));
        row[c] = v; out[base + c] = v;
        amax = fmaxf(amax, fabsf(fq_f(v)));
    }
    __syncthreads();
    amax = fq_block_amax<NT>(amax, red);

    const float d = amax / 127.0f;
    if (threadIdx.x == 0) scale[r] = d;
    for (int c = threadIdx.x; c < cols; c += NT)
        q[base + c] = (signed char)((amax == 0.f) ? 0 : (int)roundf(fq_f(row[c]) / d));
}
} // anon namespace

// cols must fit the staged row: cols * 2B + (NT/32)*4B <= smem cap.
bool launch_prefill_swiglu_q8(const void* gate, const void* up, signed char* q, float* scale,
                              int rows, int cols, cudaStream_t stream) {
    constexpr int NT = 256;
    const size_t sh = (size_t)(NT / 32) * sizeof(float) + (size_t)cols * sizeof(__nv_bfloat16);
    if (sh > 96 * 1024) return false;                    // fall back to the unfused pair
    static bool once = false;
    if (!once) { cudaFuncSetAttribute(pf_swiglu_q8_kernel<NT>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); once = true; }
    pf_swiglu_q8_kernel<NT><<<rows, NT, sh, stream>>>(
        (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up, q, scale, rows, cols);
    return true;
}

bool launch_prefill_rmsnorm_q8(const void* x, const void* w, void* out, signed char* q, float* scale,
                               int rows, int cols, float eps, cudaStream_t stream) {
    constexpr int NT = 256;
    const size_t sh = (size_t)(NT / 32) * sizeof(float) + (size_t)cols * sizeof(__nv_bfloat16);
    if (sh > 96 * 1024) return false;
    static bool once = false;
    if (!once) { cudaFuncSetAttribute(pf_rmsnorm_q8_kernel<NT>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024); once = true; }
    pf_rmsnorm_q8_kernel<NT><<<rows, NT, sh, stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (__nv_bfloat16*)out, q, scale, rows, cols, eps);
    return true;
}

bool prefill_fused_q8_enabled() {
    const char* e = getenv("SPARKINFER_PREFILL_FUSED_Q8");
    return !(e && e[0] == '0');
}

}} // namespace
