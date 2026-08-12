// Fused SwiGLU + per-row int8 quantize for the long-context dense FFN down-projection input.
//
// The chunked int8 FFN computes gate/up GEMMs into ffg/ffu, then runs a standalone SwiGLU
// (silu(gate)*up -> ffg) and a standalone per-row int8 quantize (ffg -> A_i8, sx) before the down
// GEMM. Both are memory-bound passes over the ffn-wide (12288) intermediate; fusing them removes the
// ffg store (SwiGLU) and the ffg reload (quantize) -- ~2 x rows*ffn bf16 of DRAM traffic per chunk.
//
// One block per row: each thread strides the row computing h = silu(ffg)*ffu, tracking the row amax;
// a block reduction yields the per-row scale, then a second strided pass writes int8. Numerically
// identical to launch_prefill_swiglu followed by launch_prefill_quantize_rows_i8 (both round the
// SwiGLU result to bf16 first, so the int8 quant sees the same bf16 values).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include "sparkinfer/kernels/prefill_fp8.h"

namespace sparkinfer { namespace kernels {

namespace {
__device__ __forceinline__ float sq_silu(float x) { return x / (1.f + __expf(-x)); }

__global__ void pf_swiglu_quant_i8_kernel(const __nv_bfloat16* __restrict__ gate,
                                          const __nv_bfloat16* __restrict__ up,
                                          signed char* __restrict__ q, float* __restrict__ scale,
                                          int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const size_t base = (size_t)row * cols;
    __shared__ float s_warp[32];
    // pass 1: compute h (bf16-rounded, matching the standalone SwiGLU) + row amax
    float amax = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float h = __bfloat162float(__float2bfloat16(sq_silu(__bfloat162float(gate[base + c]))
                                                           * __bfloat162float(up[base + c])));
        amax = fmaxf(amax, fabsf(h));
    }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
        if (threadIdx.x == 0) s_warp[0] = v;
    }
    __syncthreads();
    const float d = (s_warp[0] == 0.f) ? 1.f : (s_warp[0] / 127.0f);
    if (threadIdx.x == 0) scale[row] = d;
    // pass 2: recompute h (cheap ALU) and store int8
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float h = __bfloat162float(__float2bfloat16(sq_silu(__bfloat162float(gate[base + c]))
                                                           * __bfloat162float(up[base + c])));
        q[base + c] = (signed char)((s_warp[0] == 0.f) ? 0 : (int)roundf(h / d));
    }
}
// Register-resident variant. The kernel above reads the whole gate and up row TWICE -- once for the
// amax, once to quantize -- which is the dominant cost of a pass whose useful output is one int8
// byte per value: measured on an RTX 5090 at Muse Glimmer's 128x19968, 37.3 us for 23.0 MB of
// traffic, of which 10.2 MB is the redundant second read. Widening the block to 1024 threads makes
// a row of at most MAXV*1024 columns fit MAXV values per thread, so pass 2 reuses what pass 1 left
// in registers and the row is read once. The wider block also gives the SM 32 warps instead of 8 to
// hide the load latency with, on a launch that only has `rows` blocks to fill the device.
//
// Bit-identical to the strided kernel: same h (bf16-rounded the same way), same fmaxf reduction
// (exact and associative, so the changed thread->value map cannot move the max), same
// `s_warp[0] == 0.f` guard and the same `h / d` -- NOT h * (1/d), which differs in the last ulp.
// The static trip count is what keeps hv[] in registers; a `c += blockDim.x` loop would spill it.
template <int MAXV>
__global__ __launch_bounds__(1024) void pf_swiglu_quant_i8_reg_kernel(
        const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
        signed char* __restrict__ q, float* __restrict__ scale, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const size_t base = (size_t)row * cols;
    __shared__ float s_warp[32];
    float hv[MAXV];
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < MAXV; i++) {
        const int c = threadIdx.x + i * 1024;
        if (c < cols) {
            hv[i] = __bfloat162float(__float2bfloat16(sq_silu(__bfloat162float(gate[base + c]))
                                                       * __bfloat162float(up[base + c])));
            amax = fmaxf(amax, fabsf(hv[i]));
        }
    }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
    if ((threadIdx.x & 31) == 0) s_warp[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x + 31) / 32) ? s_warp[threadIdx.x] : 0.f;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m));
        if (threadIdx.x == 0) s_warp[0] = v;
    }
    __syncthreads();
    const float d = (s_warp[0] == 0.f) ? 1.f : (s_warp[0] / 127.0f);
    if (threadIdx.x == 0) scale[row] = d;
    #pragma unroll
    for (int i = 0; i < MAXV; i++) {
        const int c = threadIdx.x + i * 1024;
        if (c < cols)
            q[base + c] = (signed char)((s_warp[0] == 0.f) ? 0 : (int)roundf(hv[i] / d));
    }
}
} // namespace

void launch_prefill_swiglu_quant_i8(const void* gate, const void* up, signed char* q, float* scale,
                                    int rows, int cols, cudaStream_t stream) {
    // Narrow rows (the MoE per-expert ffn) do not fill a 1024-wide block, and rows wider than
    // 20*1024 exceed the register budget; both keep the strided kernel.
    // SPARKINFER_PREFILL_SWIGLU_REG=0 restores it everywhere (A/B).
    static const bool reg_ok = [] {
        const char* e = getenv("SPARKINFER_PREFILL_SWIGLU_REG");
        return !(e && e[0] == '0');
    }();
    const int nv = (cols + 1023) / 1024;
    if (reg_ok && cols >= 2048 && nv <= 20) {
        auto* g = reinterpret_cast<const __nv_bfloat16*>(gate);
        auto* u = reinterpret_cast<const __nv_bfloat16*>(up);
        if (nv <= 8)       pf_swiglu_quant_i8_reg_kernel<8><<<rows, 1024, 0, stream>>>(g, u, q, scale, rows, cols);
        else if (nv <= 12) pf_swiglu_quant_i8_reg_kernel<12><<<rows, 1024, 0, stream>>>(g, u, q, scale, rows, cols);
        else if (nv <= 16) pf_swiglu_quant_i8_reg_kernel<16><<<rows, 1024, 0, stream>>>(g, u, q, scale, rows, cols);
        else               pf_swiglu_quant_i8_reg_kernel<20><<<rows, 1024, 0, stream>>>(g, u, q, scale, rows, cols);
        return;
    }
    pf_swiglu_quant_i8_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate), reinterpret_cast<const __nv_bfloat16*>(up),
        q, scale, rows, cols);
}

}} // namespace sparkinfer::kernels
