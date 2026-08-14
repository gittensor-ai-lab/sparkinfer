// ============================================================================
// Block-parallel, single-pass int8 row quantization for prefill activations.
//
// WHY THIS EXISTS
// ---------------
// pf_quantize_rows_i8 (prefill_gemm_i8.cu, #422) is launched <<<rows, 32>>> — ONE WARP per row —
// and streams the row twice:
//
//     for (c = lane; c < cols; c += 32) amax = fmaxf(amax, fabsf(x[r*cols + c]));   // pass 1
//     ... warp reduce ...
//     for (c = lane; c < cols; c += 32) q[r*cols + c] = round(x[r*cols + c] / d);   // pass 2
//
// Two costs follow. The row is read from DRAM twice (the projections are large enough that a row
// does not survive in cache between the passes), and a 32-thread block is a poor unit of work: one
// warp per block wastes most of each SM's issue slots and leaves the memory pipe underfed.
//
// Measured on an RTX 5090 (nsys, main @ cb34dc1, ctx=4096): 17.1 ms per prefill over 200 calls.
// After #464 fused the WEIGHT dequant into int8, these 200 calls are the activation side only, and
// they are pure streaming: sum over projections of rows*cols is ~4.43 G elements, so the ideal is
// 2 B read + 1 B write = ~13.3 GB = 7.4 ms at 1792 GB/s. The kernel sits at ~43% of that.
//
// THE FIX. One BLOCK per row (not one warp), and stage the row in registers so it is read once:
//   * BLOCK threads cover the row; each thread keeps its slice in registers across both phases,
//     so DRAM sees exactly one read and one write. The amax reduction runs over the registers.
//   * 16-byte vector loads/stores: each thread handles VEC=8 contiguous bf16, so a warp moves 512 B.
//   * grid = rows, block = BLOCK threads -> the SM gets whole blocks of work instead of lone warps.
//
// NUMERICS ARE UNCHANGED and that is deliberate: same amax over the row, same d = amax/127.0f, same
// roundf, same amax==0 -> 0 rule, and the reduction is over the same value set (max is associative
// and exact in fp, so reduction order cannot change the result). The int8 output and per-row scale
// are bit-identical to #422's kernel, which is checked by A/B in the eval and by the guard model.
// ============================================================================
#include "sparkinfer/kernels/prefill_quant_rows.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdlib>

namespace sparkinfer {
namespace kernels {

namespace {

__device__ __forceinline__ float qr_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }

// K-TILED ACTIVATION COPY. Muse Glimmer's dense prefill GEMM stages a 32-byte K slice of all 128
// rows per pipeline step. In this [row][k] layout those 128 chunks sit K bytes apart, so a warp's
// 16 rows issue 16 separate sector requests for 512 B; the same stage out of a [k/32][row][32]
// buffer is one contiguous 4 KB block, i.e. 4 line requests. Same bytes, same cp.async count, a
// quarter of the transactions -- and the GEMM re-reads the whole activation once per output tile
// (312 times for the 19968-wide FFN pair), so the read side is worth far more than the scattered
// store it costs here. Written ALONGSIDE the row-major output, never instead of it: proj()'s
// materialize fallback and every non-Muse consumer still read [row][k].
__device__ __forceinline__ size_t qr_pack_off(int r, int c, int rows) {
    return (size_t)(c >> 5) * (size_t)rows * 32 + (size_t)r * 32 + (size_t)(c & 31);
}

// One block per row. VEC bf16 per thread per slot; SLOTS slots keeps the row in registers.
template <int BLOCK, int VEC, int SLOTS>
__global__ __launch_bounds__(BLOCK) void pf_quant_rows_fast_kernel(
        const __nv_bfloat16* __restrict__ x, signed char* __restrict__ q,
        float* __restrict__ scale, int rows, int cols, signed char* __restrict__ qp) {
    const int r   = blockIdx.x;
    const int tid = threadIdx.x;
    if (r >= rows) return;
    const size_t base = (size_t)r * cols;

    __nv_bfloat16 reg[SLOTS][VEC];
    float amax = 0.f;

    // ---- single read: pull the row into registers, computing amax on the way ----
    #pragma unroll
    for (int s = 0; s < SLOTS; s++) {
        const int c = (tid + s * BLOCK) * VEC;
        if (c < cols) {
            *reinterpret_cast<uint4*>(reg[s]) = *reinterpret_cast<const uint4*>(&x[base + c]);
            #pragma unroll
            for (int v = 0; v < VEC; v++) amax = fmaxf(amax, fabsf(qr_to_f(reg[s][v])));
        }
    }

    // ---- block reduce max (exact in fp: max is associative, so order cannot change the result) ----
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    __shared__ float sred[BLOCK / 32];
    if ((tid & 31) == 0) sred[tid >> 5] = amax;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < BLOCK / 32) ? sred[tid] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (tid == 0) sred[0] = v;
    }
    __syncthreads();
    const float d = sred[0] / 127.0f;
    if (tid == 0) scale[r] = d;

    // ---- write from registers: the row is never re-read ----
    #pragma unroll
    for (int s = 0; s < SLOTS; s++) {
        const int c = (tid + s * BLOCK) * VEC;
        if (c < cols) {
            signed char out[VEC];
            #pragma unroll
            for (int v = 0; v < VEC; v++)
                out[v] = (signed char)((sred[0] == 0.f) ? 0 : (int)roundf(qr_to_f(reg[s][v]) / d));
            *reinterpret_cast<uint2*>(&q[base + c]) = *reinterpret_cast<const uint2*>(out);
            if (qp)
                *reinterpret_cast<uint2*>(&qp[qr_pack_off(r, c, rows)]) =
                    *reinterpret_cast<const uint2*>(out);
        }
    }
}


// Muse Glimmer's attention output is gated (attn *= sigmoid(gate)) and then immediately row-quantized
// to int8 for the o projection -- two kernels over the same qdim-wide tensor, with a full bf16 write
// and re-read of `attn` in between that nothing else ever looks at. Folding the gate into the load
// phase of the quantizer removes that round trip and one launch per layer.
//
// BIT-IDENTICAL to pf_mul_sigmoid_kernel followed by pf_quant_rows_fast_kernel: the gated value is
// rounded to bf16 exactly as the separate kernel stored it, the amax is over that same value set
// (max is associative and exact in fp), and d = amax/127.0f with the same roundf and amax==0 rule.
template <int BLOCK, int VEC, int SLOTS>
__global__ __launch_bounds__(BLOCK) void pf_gate_quant_rows_kernel(
        const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gate,
        signed char* __restrict__ q, float* __restrict__ scale, int rows, int cols,
        signed char* __restrict__ qp) {
    const int r   = blockIdx.x;
    const int tid = threadIdx.x;
    if (r >= rows) return;
    const size_t base = (size_t)r * cols;

    __nv_bfloat16 reg[SLOTS][VEC];
    float amax = 0.f;
    #pragma unroll
    for (int s = 0; s < SLOTS; s++) {
        const int c = (tid + s * BLOCK) * VEC;
        if (c < cols) {
            __nv_bfloat16 xv[VEC], gv[VEC];
            *reinterpret_cast<uint4*>(xv) = *reinterpret_cast<const uint4*>(&x[base + c]);
            *reinterpret_cast<uint4*>(gv) = *reinterpret_cast<const uint4*>(&gate[base + c]);
            #pragma unroll
            for (int v = 0; v < VEC; v++) {
                const float g = qr_to_f(gv[v]);
                reg[s][v] = __float2bfloat16(qr_to_f(xv[v]) * (1.f / (1.f + __expf(-g))));
                amax = fmaxf(amax, fabsf(qr_to_f(reg[s][v])));
            }
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    __shared__ float sred[BLOCK / 32];
    if ((tid & 31) == 0) sred[tid >> 5] = amax;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < BLOCK / 32) ? sred[tid] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (tid == 0) sred[0] = v;
    }
    __syncthreads();
    const float d = sred[0] / 127.0f;
    if (tid == 0) scale[r] = d;

    #pragma unroll
    for (int s = 0; s < SLOTS; s++) {
        const int c = (tid + s * BLOCK) * VEC;
        if (c < cols) {
            signed char out[VEC];
            #pragma unroll
            for (int v = 0; v < VEC; v++)
                out[v] = (signed char)((sred[0] == 0.f) ? 0 : (int)roundf(qr_to_f(reg[s][v]) / d));
            *reinterpret_cast<uint2*>(&q[base + c]) = *reinterpret_cast<const uint2*>(out);
            if (qp)
                *reinterpret_cast<uint2*>(&qp[qr_pack_off(r, c, rows)]) =
                    *reinterpret_cast<const uint2*>(out);
        }
    }
}

}  // namespace

bool launch_prefill_gate_quant_rows_i8(const void* x, const void* gate, signed char* q, float* scale,
                                       int rows, int cols, cudaStream_t stream,
                                       signed char* qp) {
    constexpr int BLOCK = 256, VEC = 8;
    // The k-tiled copy is only defined on whole 32-column groups; K into the dense GEMM is always a
    // whole number of 256-value super-blocks, so this only ever declines shapes it never serves.
    if (qp && (cols % 32) != 0) qp = nullptr;
    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_PREFILL_GATE_QUANT");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    if (!enabled || rows <= 0 || cols <= 0 || (cols % VEC) != 0) return false;
    auto xb = reinterpret_cast<const __nv_bfloat16*>(x);
    auto gb = reinterpret_cast<const __nv_bfloat16*>(gate);
    const int vecs = cols / VEC;
    if (vecs <= BLOCK * 1)      pf_gate_quant_rows_kernel<BLOCK, VEC, 1><<<rows, BLOCK, 0, stream>>>(xb, gb, q, scale, rows, cols, qp);
    else if (vecs <= BLOCK * 2) pf_gate_quant_rows_kernel<BLOCK, VEC, 2><<<rows, BLOCK, 0, stream>>>(xb, gb, q, scale, rows, cols, qp);
    else if (vecs <= BLOCK * 4) pf_gate_quant_rows_kernel<BLOCK, VEC, 4><<<rows, BLOCK, 0, stream>>>(xb, gb, q, scale, rows, cols, qp);
    else return false;
    return true;
}

bool launch_prefill_quant_rows_fast(const void* x, signed char* q, float* scale,
                                    int rows, int cols, cudaStream_t stream, signed char* qp) {
    constexpr int BLOCK = 256, VEC = 8;
    if (qp && (cols % 32) != 0) qp = nullptr;

    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_PREFILL_QUANT_ROWS");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    if (!enabled || rows <= 0 || cols <= 0) return false;
    if ((cols % VEC) != 0) return false;                 // 16-byte vector path only

    auto xb = reinterpret_cast<const __nv_bfloat16*>(x);
    const int vecs = cols / VEC;                          // vector slots in a row
    // SLOTS is a compile-time register budget; dispatch the smallest that covers the row.
    if (vecs <= BLOCK * 1)      pf_quant_rows_fast_kernel<BLOCK, VEC, 1><<<rows, BLOCK, 0, stream>>>(xb, q, scale, rows, cols, qp);
    else if (vecs <= BLOCK * 2) pf_quant_rows_fast_kernel<BLOCK, VEC, 2><<<rows, BLOCK, 0, stream>>>(xb, q, scale, rows, cols, qp);
    else if (vecs <= BLOCK * 4) pf_quant_rows_fast_kernel<BLOCK, VEC, 4><<<rows, BLOCK, 0, stream>>>(xb, q, scale, rows, cols, qp);
    else if (vecs <= BLOCK * 6) pf_quant_rows_fast_kernel<BLOCK, VEC, 6><<<rows, BLOCK, 0, stream>>>(xb, q, scale, rows, cols, qp);
    else if (vecs <= BLOCK * 8) pf_quant_rows_fast_kernel<BLOCK, VEC, 8><<<rows, BLOCK, 0, stream>>>(xb, q, scale, rows, cols, qp);
    // Muse's ffn width (19968 -> 2496 vectors) sat just past SLOTS=6 and fell to the warp-per-row
    // scalar kernel: 128 blocks of 32 threads, the row read twice, 18.9 us a call against 2.0 for
    // this one. SLOTS=10 covers it in registers (10 * VEC bf16 = 40 registers of a 256-budget).
    else if (vecs <= BLOCK * 10) pf_quant_rows_fast_kernel<BLOCK, VEC, 10><<<rows, BLOCK, 0, stream>>>(xb, q, scale, rows, cols, qp);
    else return false;                                    // very wide rows: keep the scalar path
    return true;
}

}  // namespace kernels
}  // namespace sparkinfer
