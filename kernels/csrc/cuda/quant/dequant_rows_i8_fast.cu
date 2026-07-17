// ============================================================================
// Vector-store fused GGUF -> int8 row dequantization.
//
// WHY THIS EXISTS
// ---------------
// deq_rows_i8_kernel (dequant_gguf.cu, #464) fuses Q4_K/Q6_K -> int8 and skips the bf16 scratch round
// trip, which is what carried #464 to XL. What it leaves on the table is the store shape:
//
//     const int row = blockIdx.x, t = threadIdx.x;      // <<<rows, 256>>>
//     for (sb = 0; sb < nsb; sb++) qrow[sb * 256 + t] = (signed char)(int)roundf(v * inv);
//
// Thread t owns value t of a 256-value super-block, so every thread stores exactly ONE byte and a
// warp emits 32 consecutive bytes — a 32-byte transaction where the memory system moves 128. The
// kernel's whole job is streaming (one prefill decodes ~6.9 G weights: ~3.9 GB of Q4_K in, 6.9 GB of
// int8 out = ~6 ms at 1792 GB/s), and it measures 19.5 ms — ~31% of DRAM peak.
//
// It also walks the row twice (once for the amax, once to scale and store), because the int8 scale is
// a property of the whole row. That looks like the obvious defect and is NOT: caching the decoded row
// in registers to make it single-pass was tried first and bought 0.5 ms (19.75 -> 19.26), because the
// second pass hits L2, not DRAM. The stores were always the cost.
//
// THE FIX. Give each thread VEC=4 CONSECUTIVE values instead of one, so a thread stores a 4-byte
// word and a warp writes 128 contiguous bytes. 256 threads x 4 = 1024 values per group; the row is
// walked in cols/1024 groups. Since a thread's 4 values are 4-aligned, they always fall inside one
// super-block, so the decode still reads a single block header per value-quad. The decoded row is
// also kept in registers (VEC * NG floats: 16 for a 4096-wide row, 48 for a 12288-wide one — 48*256
// = 12288 regs/block, still 5 blocks/SM), which costs nothing now and removes the second decode.
//
// BIT-IDENTICAL, deliberately: same deq_q4k_val/deq_q6k_val, same amax over the same value set, same
// d = amax/127.f, same inv, same roundf(v * inv). Only the thread->value map and the store width move.
// ============================================================================
#include "sparkinfer/kernels/dequant_rows_i8_fast.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdlib>

namespace sparkinfer {
namespace kernels {

namespace {

enum { DQR_Q4_K = 12, DQR_Q6_K = 14 };

// --- decode helpers: byte-for-byte the ones in dequant_gguf.cu ---
__device__ __forceinline__ float dqr_h2f(const unsigned char* p) {
    __half h; *((unsigned short*)&h) = *(const unsigned short*)p; return __half2float(h);
}
__device__ __forceinline__ void dqr_scale_min_k4(int j, const unsigned char* q, int* d, int* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}
__device__ __forceinline__ float dqr_q4k_val(const unsigned char* blk, int t) {
    const float d = dqr_h2f(blk), dmin = dqr_h2f(blk + 2);
    const unsigned char* sc = blk + 4; const unsigned char* qs = blk + 16;
    const int j64 = t >> 6, r = t & 63, l = r & 31, hi = r >> 5;
    const unsigned char byte = qs[j64 * 32 + l];
    const int nib = hi ? (byte >> 4) : (byte & 0xF);
    int s, m; dqr_scale_min_k4(2 * j64 + hi, sc, &s, &m);
    return d * s * nib - dmin * m;
}
__device__ __forceinline__ float dqr_q6k_val(const unsigned char* blk, int t) {
    const int half = t >> 7, r = t & 127, quad = r >> 5, l = r & 31;
    const unsigned char* ql = blk + half * 64;
    const unsigned char* qh = blk + 128 + half * 32;
    const signed char* sc = (const signed char*)(blk + 192) + half * 8;
    const float d = dqr_h2f(blk + 208);
    const int is = l / 16;
    int qv;
    if (quad == 0)      qv = (int)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
    else if (quad == 1) qv = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
    else if (quad == 2) qv = (int)((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
    else                qv = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
    return d * sc[is + 2 * quad] * qv;
}

constexpr int DQR_BLOCK = 256, DQR_VEC = 4;   // 4 consecutive values/thread => 4B store, 128B/warp

// One block per row; NG groups of DQR_BLOCK*DQR_VEC values each.
template <int QT, int NG>
__global__ __launch_bounds__(DQR_BLOCK) void deq_rows_i8_vec_kernel(
        const unsigned char* __restrict__ src, signed char* __restrict__ q,
        float* __restrict__ scale, int cols) {
    constexpr int BS = (QT == DQR_Q4_K) ? 144 : 210;
    constexpr int GSPAN = DQR_BLOCK * DQR_VEC;               // values per group
    const int row = blockIdx.x, t = threadIdx.x;
    const int nsb = cols >> 8;
    const unsigned char* rbase = src + (size_t)row * nsb * BS;

    float v[NG][DQR_VEC];
    float amax = 0.f;
    #pragma unroll
    for (int g = 0; g < NG; g++) {
        const int base = g * GSPAN + t * DQR_VEC;            // 4-aligned => one super-block
        if (base < cols) {
            const unsigned char* blk = rbase + (size_t)(base >> 8) * BS;
            const int off = base & 255;
            #pragma unroll
            for (int i = 0; i < DQR_VEC; i++) {
                v[g][i] = (QT == DQR_Q4_K) ? dqr_q4k_val(blk, off + i) : dqr_q6k_val(blk, off + i);
                amax = fmaxf(amax, fabsf(v[g][i]));
            }
        }
    }

    __shared__ float swarp[DQR_BLOCK / 32];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((t & 31) == 0) swarp[t >> 5] = amax;
    __syncthreads();
    if (t < 32) {
        float w = (t < DQR_BLOCK / 32) ? swarp[t] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) w = fmaxf(w, __shfl_xor_sync(0xffffffffu, w, o));
        if (t == 0) swarp[0] = w;
    }
    __syncthreads();
    const float d = swarp[0] / 127.f;
    if (t == 0) scale[row] = d;
    const float inv = (d > 0.f) ? (1.f / d) : 0.f;

    signed char* qrow = q + (size_t)row * cols;
    #pragma unroll
    for (int g = 0; g < NG; g++) {
        const int base = g * GSPAN + t * DQR_VEC;
        if (base < cols) {
            signed char o[DQR_VEC];
            #pragma unroll
            for (int i = 0; i < DQR_VEC; i++) o[i] = (signed char)(int)roundf(v[g][i] * inv);
            *reinterpret_cast<unsigned*>(&qrow[base]) = *reinterpret_cast<const unsigned*>(o);
        }
    }
}

template <int QT>
bool dispatch(const unsigned char* s, signed char* q, float* scale, int rows, int cols,
              cudaStream_t stream) {
    const int ng = (cols + DQR_BLOCK * DQR_VEC - 1) / (DQR_BLOCK * DQR_VEC);
    if (ng <= 4)       deq_rows_i8_vec_kernel<QT, 4><<<rows, DQR_BLOCK, 0, stream>>>(s, q, scale, cols);
    else if (ng <= 8)  deq_rows_i8_vec_kernel<QT, 8><<<rows, DQR_BLOCK, 0, stream>>>(s, q, scale, cols);
    else if (ng <= 12) deq_rows_i8_vec_kernel<QT, 12><<<rows, DQR_BLOCK, 0, stream>>>(s, q, scale, cols);
    else return false;                                   // wider than the register budget
    return true;
}

}  // namespace

bool launch_gguf_dequant_rows_i8_fast(int ggml_type, const void* src, signed char* q, float* scale,
                                      int rows, int cols, cudaStream_t stream) {
    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_DEQUANT_ROWS_I8_FAST");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    // cols % 1024 keeps every thread's 4-value quad inside one super-block and 4-byte aligned.
    if (!enabled || rows <= 0 || cols <= 0 || (cols % (DQR_BLOCK * DQR_VEC)) != 0) return false;

    auto s = reinterpret_cast<const unsigned char*>(src);
    if (ggml_type == DQR_Q4_K) return dispatch<DQR_Q4_K>(s, q, scale, rows, cols, stream);
    if (ggml_type == DQR_Q6_K) return dispatch<DQR_Q6_K>(s, q, scale, rows, cols, stream);
    return false;
}

}  // namespace kernels
}  // namespace sparkinfer
