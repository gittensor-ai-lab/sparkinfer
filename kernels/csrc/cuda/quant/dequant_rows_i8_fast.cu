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
// walked in ceil(cols/1024) groups (cols=512 uses one partial group). Since a thread's 4 values are 4-aligned, they always fall inside one
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

enum { DQR_Q4_K = 12, DQR_Q5_K = 13, DQR_Q6_K = 14 };

// --- decode helpers: byte-for-byte the ones in dequant_gguf.cu ---
__device__ __forceinline__ float dqr_h2f(const unsigned char* p) {
    __half h; *((unsigned short*)&h) = *(const unsigned short*)p; return __half2float(h);
}
__device__ __forceinline__ float dqr_q4k_val(const unsigned char* blk, int t) {
    const float d = dqr_h2f(blk), dmin = dqr_h2f(blk + 2);
    const unsigned char* sc = blk + 4; const unsigned char* qs = blk + 16;
    const int j64 = t >> 6, r = t & 63, l = r & 31, hi = r >> 5;
    const unsigned char byte = qs[j64 * 32 + l];
    const int nib = hi ? (byte >> 4) : (byte & 0xF);
    // 6-bit packed scale/min unpack (same layout gg_scale_min_k4 decodes in dequant_gguf.cu).
    const int j = 2 * j64 + hi;
    int s, m;
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
    else {
        s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        m = (sc[j + 4] >> 4)  | ((sc[j]     >> 6) << 4);
    }
    return d * s * nib - dmin * m;
}

__device__ __forceinline__ float dqr_q5k_val(const unsigned char* blk, int t) {
    const float d = dqr_h2f(blk), dmin = dqr_h2f(blk + 2);
    const unsigned char* sc = blk + 4; const unsigned char* qh = blk + 16;
    const unsigned char* ql = blk + 48;
    const int j64 = t >> 6, r = t & 63, l = r & 31, hi = r >> 5;
    const unsigned char byte = ql[j64 * 32 + l];
    const int nib = hi ? (byte >> 4) : (byte & 0xF);
    const int hbit = (qh[l] >> (2 * j64 + hi)) & 1;
    // 6-bit packed scale/min unpack (same as Q4_K / deq_q5k_val in dequant_gguf.cu).
    const int j = 2 * j64 + hi;
    int s, m;
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
    else {
        s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        m = (sc[j + 4] >> 4)  | ((sc[j]     >> 6) << 4);
    }
    return d * s * (nib + (hbit ? 16 : 0)) - dmin * m;
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


template <int QT>
__device__ __forceinline__ constexpr int dqr_bs() {
    if constexpr (QT == DQR_Q4_K) return 144;
    else if constexpr (QT == DQR_Q5_K) return 176;
    else return 210;
}
template <int QT>
__device__ __forceinline__ float dqr_val(const unsigned char* blk, int t) {
    if constexpr (QT == DQR_Q4_K) return dqr_q4k_val(blk, t);
    else if constexpr (QT == DQR_Q5_K) return dqr_q5k_val(blk, t);
    else return dqr_q6k_val(blk, t);
}

constexpr int DQR_BLOCK = 256, DQR_VEC = 4;   // 4 consecutive values/thread => 4B store, 128B/warp

// One block per row; NG groups of BLK*DQR_VEC values each. BLK is a template parameter because
// v[NG][DQR_VEC] lives in registers: a wide row at BLK=256 needs a large NG (Muse Glimmer's
// ffn_down, cols=19968, needs NG=20 => 80 floats/thread) and the resulting register pressure
// costs more occupancy than the extra threads cost. Widening the block instead keeps NG small
// (BLK=1024 => NG=5 for the same row). Value->thread mapping is unchanged, so the decode, the
// amax and the stores are all bit-identical whatever BLK is.
template <int QT, int NG, int BLK = DQR_BLOCK>
__global__ __launch_bounds__(BLK) void deq_rows_i8_vec_kernel(
        const unsigned char* __restrict__ src, signed char* __restrict__ q,
        float* __restrict__ scale, int cols) {
    constexpr int BS = dqr_bs<QT>();
    constexpr int GSPAN = BLK * DQR_VEC;                     // values per group
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
                v[g][i] = dqr_val<QT>(blk, off + i);
                amax = fmaxf(amax, fabsf(v[g][i]));
            }
        }
    }

    __shared__ float swarp[BLK / 32];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((t & 31) == 0) swarp[t >> 5] = amax;
    __syncthreads();
    if (t < 32) {
        float w = (t < BLK / 32) ? swarp[t] : 0.f;
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
    if (ng <= 1)       deq_rows_i8_vec_kernel<QT, 1><<<rows, DQR_BLOCK, 0, stream>>>(s, q, scale, cols);
    else if (ng <= 4)  deq_rows_i8_vec_kernel<QT, 4><<<rows, DQR_BLOCK, 0, stream>>>(s, q, scale, cols);
    else if (ng <= 8)  deq_rows_i8_vec_kernel<QT, 8><<<rows, DQR_BLOCK, 0, stream>>>(s, q, scale, cols);
    else if (ng <= 12) deq_rows_i8_vec_kernel<QT, 12><<<rows, DQR_BLOCK, 0, stream>>>(s, q, scale, cols);
    else {
        // Beyond 12 groups the register array is the binding constraint, not the thread count:
        // Muse Glimmer's ffn_down (cols=19968) would need NG=20 at 256 threads = 80 floats/thread.
        // A 1024-wide block covers the same row in NG=5, so widen the block instead of growing NG.
        constexpr int WBLK = 1024, WSPAN = WBLK * DQR_VEC;
        const int ngw = (cols + WSPAN - 1) / WSPAN;
        if (ngw <= 5)      deq_rows_i8_vec_kernel<QT, 5, WBLK><<<rows, WBLK, 0, stream>>>(s, q, scale, cols);
        else if (ngw <= 8) deq_rows_i8_vec_kernel<QT, 8, WBLK><<<rows, WBLK, 0, stream>>>(s, q, scale, cols);
        else return false;                               // wider than the register budget
    }
    return true;
}

// Dual-tensor: grid = 2*rows. which = blockIdx.x / rows selects src/dst.
template <int QT, int NG>
__global__ __launch_bounds__(DQR_BLOCK) void deq_rows_i8_vec_pair_kernel(
        const unsigned char* __restrict__ src0, signed char* __restrict__ q0, float* __restrict__ scale0,
        const unsigned char* __restrict__ src1, signed char* __restrict__ q1, float* __restrict__ scale1,
        int rows, int cols) {
    constexpr int BS = dqr_bs<QT>();
    constexpr int GSPAN = DQR_BLOCK * DQR_VEC;
    const int which = (int)(blockIdx.x / (unsigned)rows);
    const int row = (int)(blockIdx.x - (unsigned)which * (unsigned)rows);
    const unsigned char* src = which ? src1 : src0;
    signed char* q = which ? q1 : q0;
    float* scale = which ? scale1 : scale0;

    const int t = threadIdx.x;
    const int nsb = cols >> 8;
    const unsigned char* rbase = src + (size_t)row * nsb * BS;

    float v[NG][DQR_VEC];
    float amax = 0.f;
    #pragma unroll
    for (int g = 0; g < NG; g++) {
        const int base = g * GSPAN + t * DQR_VEC;
        if (base < cols) {
            const unsigned char* blk = rbase + (size_t)(base >> 8) * BS;
            const int off = base & 255;
            #pragma unroll
            for (int i = 0; i < DQR_VEC; i++) {
                v[g][i] = dqr_val<QT>(blk, off + i);
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
bool dispatch_pair(const unsigned char* s0, signed char* q0, float* sc0,
                   const unsigned char* s1, signed char* q1, float* sc1,
                   int rows, int cols, cudaStream_t stream) {
    const int ng = (cols + DQR_BLOCK * DQR_VEC - 1) / (DQR_BLOCK * DQR_VEC);
    const int grid = 2 * rows;
    if (ng <= 4)
        deq_rows_i8_vec_pair_kernel<QT, 4><<<grid, DQR_BLOCK, 0, stream>>>(s0, q0, sc0, s1, q1, sc1, rows, cols);
    else if (ng <= 8)
        deq_rows_i8_vec_pair_kernel<QT, 8><<<grid, DQR_BLOCK, 0, stream>>>(s0, q0, sc0, s1, q1, sc1, rows, cols);
    else if (ng <= 12)
        deq_rows_i8_vec_pair_kernel<QT, 12><<<grid, DQR_BLOCK, 0, stream>>>(s0, q0, sc0, s1, q1, sc1, rows, cols);
    else return false;
    return true;
}

// Sparse gather: blockIdx.x = live_i * rows_per_expert + row_in_expert.
// BLOCK defaults to 256; cols=512 uses BLOCK=128 so 128*4 covers the row with no idle threads.
template <int QT, int NG, int BLOCK = DQR_BLOCK>
__global__ __launch_bounds__(BLOCK) void deq_rows_i8_vec_gather_kernel(
        const unsigned char* __restrict__ src0, signed char* __restrict__ q0,
        float* __restrict__ scale0, const int* __restrict__ live_le,
        int rows_per_expert, int cols, size_t expert_bytes) {
    constexpr int BS = dqr_bs<QT>();
    constexpr int GSPAN = BLOCK * DQR_VEC;
    const int live_i = (int)(blockIdx.x / (unsigned)rows_per_expert);
    const int row_in = (int)(blockIdx.x - (unsigned)live_i * (unsigned)rows_per_expert);
    const int le = live_le[live_i];
    const int t = threadIdx.x;
    const int nsb = cols >> 8;
    const size_t row_bytes = (size_t)nsb * (size_t)BS;
    const unsigned char* rbase = src0 + (size_t)le * expert_bytes + (size_t)row_in * row_bytes;
    float* scale = scale0 + (size_t)le * (size_t)rows_per_expert + (size_t)row_in;
    signed char* qrow = q0 + ((size_t)le * (size_t)rows_per_expert + (size_t)row_in) * (size_t)cols;

    float v[NG][DQR_VEC];
    float amax = 0.f;
    #pragma unroll
    for (int g = 0; g < NG; g++) {
        const int base = g * GSPAN + t * DQR_VEC;
        if (base < cols) {
            const unsigned char* blk = rbase + (size_t)(base >> 8) * BS;
            const int off = base & 255;
            #pragma unroll
            for (int i = 0; i < DQR_VEC; i++) {
                v[g][i] = dqr_val<QT>(blk, off + i);
                amax = fmaxf(amax, fabsf(v[g][i]));
            }
        }
    }

    __shared__ float swarp[BLOCK / 32];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((t & 31) == 0) swarp[t >> 5] = amax;
    __syncthreads();
    if (t < 32) {
        float w = (t < BLOCK / 32) ? swarp[t] : 0.f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) w = fmaxf(w, __shfl_xor_sync(0xffffffffu, w, o));
        if (t == 0) swarp[0] = w;
    }
    __syncthreads();
    const float d = swarp[0] / 127.f;
    if (t == 0) *scale = d;
    const float inv = (d > 0.f) ? (1.f / d) : 0.f;

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

template <int QT, int NG>
__global__ __launch_bounds__(DQR_BLOCK) void deq_rows_i8_vec_gather_pair_kernel(
        const unsigned char* __restrict__ src0, signed char* __restrict__ q0, float* __restrict__ scale0,
        const unsigned char* __restrict__ src1, signed char* __restrict__ q1, float* __restrict__ scale1,
        const int* __restrict__ live_le, int n_rows, int rows_per_expert, int cols,
        size_t expert_bytes0, size_t expert_bytes1) {
    constexpr int BS = dqr_bs<QT>();
    constexpr int GSPAN = DQR_BLOCK * DQR_VEC;
    const int which = (int)(blockIdx.x / (unsigned)n_rows);
    const int flat = (int)(blockIdx.x - (unsigned)which * (unsigned)n_rows);
    const int live_i = flat / rows_per_expert;
    const int row_in = flat - live_i * rows_per_expert;
    const int le = live_le[live_i];
    const int t = threadIdx.x;
    const int nsb = cols >> 8;
    const size_t row_bytes = (size_t)nsb * (size_t)BS;
    const unsigned char* src = which ? src1 : src0;
    const size_t expert_bytes = which ? expert_bytes1 : expert_bytes0;
    signed char* q0b = which ? q1 : q0;
    float* scale0b = which ? scale1 : scale0;
    const unsigned char* rbase = src + (size_t)le * expert_bytes + (size_t)row_in * row_bytes;
    float* scale = scale0b + (size_t)le * (size_t)rows_per_expert + (size_t)row_in;
    signed char* qrow = q0b + ((size_t)le * (size_t)rows_per_expert + (size_t)row_in) * (size_t)cols;

    float v[NG][DQR_VEC];
    float amax = 0.f;
    #pragma unroll
    for (int g = 0; g < NG; g++) {
        const int base = g * GSPAN + t * DQR_VEC;
        if (base < cols) {
            const unsigned char* blk = rbase + (size_t)(base >> 8) * BS;
            const int off = base & 255;
            #pragma unroll
            for (int i = 0; i < DQR_VEC; i++) {
                v[g][i] = dqr_val<QT>(blk, off + i);
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
    if (t == 0) *scale = d;
    const float inv = (d > 0.f) ? (1.f / d) : 0.f;

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
bool dispatch_gather(const unsigned char* src0, signed char* q0, float* scale0,
                     const int* live_le, int n_live, int rows_per_expert, int cols,
                     size_t expert_bytes, cudaStream_t stream) {
    const int grid = n_live * rows_per_expert;
    // MoE down (cols=512): 128 threads x 4 = full row, no idle half-block.
    if (cols == 512) {
        deq_rows_i8_vec_gather_kernel<QT, 1, 128><<<grid, 128, 0, stream>>>(
            src0, q0, scale0, live_le, rows_per_expert, cols, expert_bytes);
        return true;
    }
    const int ng = (cols + DQR_BLOCK * DQR_VEC - 1) / (DQR_BLOCK * DQR_VEC);
    if (ng <= 1)
        deq_rows_i8_vec_gather_kernel<QT, 1><<<grid, DQR_BLOCK, 0, stream>>>(
            src0, q0, scale0, live_le, rows_per_expert, cols, expert_bytes);
    else if (ng <= 4)
        deq_rows_i8_vec_gather_kernel<QT, 4><<<grid, DQR_BLOCK, 0, stream>>>(
            src0, q0, scale0, live_le, rows_per_expert, cols, expert_bytes);
    else if (ng <= 8)
        deq_rows_i8_vec_gather_kernel<QT, 8><<<grid, DQR_BLOCK, 0, stream>>>(
            src0, q0, scale0, live_le, rows_per_expert, cols, expert_bytes);
    else if (ng <= 12)
        deq_rows_i8_vec_gather_kernel<QT, 12><<<grid, DQR_BLOCK, 0, stream>>>(
            src0, q0, scale0, live_le, rows_per_expert, cols, expert_bytes);
    else return false;
    return true;
}

template <int QT>
bool dispatch_gather_pair(const unsigned char* src0, signed char* q0, float* scale0,
                          const unsigned char* src1, signed char* q1, float* scale1,
                          const int* live_le, int n_live, int rows_per_expert, int cols,
                          size_t expert_bytes0, size_t expert_bytes1, cudaStream_t stream) {
    const int ng = (cols + DQR_BLOCK * DQR_VEC - 1) / (DQR_BLOCK * DQR_VEC);
    const int n_rows = n_live * rows_per_expert;
    const int grid = 2 * n_rows;
    if (ng <= 4)
        deq_rows_i8_vec_gather_pair_kernel<QT, 4><<<grid, DQR_BLOCK, 0, stream>>>(
            src0, q0, scale0, src1, q1, scale1, live_le, n_rows, rows_per_expert, cols,
            expert_bytes0, expert_bytes1);
    else if (ng <= 8)
        deq_rows_i8_vec_gather_pair_kernel<QT, 8><<<grid, DQR_BLOCK, 0, stream>>>(
            src0, q0, scale0, src1, q1, scale1, live_le, n_rows, rows_per_expert, cols,
            expert_bytes0, expert_bytes1);
    else if (ng <= 12)
        deq_rows_i8_vec_gather_pair_kernel<QT, 12><<<grid, DQR_BLOCK, 0, stream>>>(
            src0, q0, scale0, src1, q1, scale1, live_le, n_rows, rows_per_expert, cols,
            expert_bytes0, expert_bytes1);
    else return false;
    return true;
}

// MoE group mask: blockIdx maps (le, row_in); skip when counts[e_base+le]==0.
template <int QT, int NG>
__global__ __launch_bounds__(DQR_BLOCK) void deq_rows_i8_vec_mask_kernel(
        const unsigned char* __restrict__ src0, signed char* __restrict__ q0,
        float* __restrict__ scale0, const int* __restrict__ counts,
        int e_base, int n_in, int rows_per_expert, int cols, size_t expert_bytes) {
    constexpr int BS = dqr_bs<QT>();
    constexpr int GSPAN = DQR_BLOCK * DQR_VEC;
    const int flat = (int)blockIdx.x;
    const int le = flat / rows_per_expert;
    const int row_in = flat - le * rows_per_expert;
    if (le >= n_in || counts[e_base + le] <= 0) return;
    const int t = threadIdx.x;
    const int nsb = cols >> 8;
    const size_t row_bytes = (size_t)nsb * (size_t)BS;
    const unsigned char* rbase = src0 + (size_t)le * expert_bytes + (size_t)row_in * row_bytes;
    float* scale = scale0 + (size_t)le * (size_t)rows_per_expert + (size_t)row_in;
    signed char* qrow = q0 + ((size_t)le * (size_t)rows_per_expert + (size_t)row_in) * (size_t)cols;

    float v[NG][DQR_VEC];
    float amax = 0.f;
    #pragma unroll
    for (int g = 0; g < NG; g++) {
        const int base = g * GSPAN + t * DQR_VEC;
        if (base < cols) {
            const unsigned char* blk = rbase + (size_t)(base >> 8) * BS;
            const int off = base & 255;
            #pragma unroll
            for (int i = 0; i < DQR_VEC; i++) {
                v[g][i] = dqr_val<QT>(blk, off + i);
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
    if (t == 0) *scale = d;
    const float inv = (d > 0.f) ? (1.f / d) : 0.f;

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

template <int QT, int NG>
__global__ __launch_bounds__(DQR_BLOCK) void deq_rows_i8_vec_mask_pair_kernel(
        const unsigned char* __restrict__ src0, signed char* __restrict__ q0, float* __restrict__ scale0,
        const unsigned char* __restrict__ src1, signed char* __restrict__ q1, float* __restrict__ scale1,
        const int* __restrict__ counts, int e_base, int n_in, int rows_per_expert, int cols,
        size_t expert_bytes0, size_t expert_bytes1) {
    constexpr int BS = dqr_bs<QT>();
    constexpr int GSPAN = DQR_BLOCK * DQR_VEC;
    const int n_flat = n_in * rows_per_expert;
    const int which = (int)(blockIdx.x / (unsigned)n_flat);
    const int flat = (int)(blockIdx.x - (unsigned)which * (unsigned)n_flat);
    const int le = flat / rows_per_expert;
    const int row_in = flat - le * rows_per_expert;
    if (le >= n_in || counts[e_base + le] <= 0) return;
    const unsigned char* src = which ? src1 : src0;
    const size_t expert_bytes = which ? expert_bytes1 : expert_bytes0;
    signed char* q = which ? q1 : q0;
    float* scale = which ? scale1 : scale0;
    const int t = threadIdx.x;
    const int nsb = cols >> 8;
    const size_t row_bytes = (size_t)nsb * (size_t)BS;
    const unsigned char* rbase = src + (size_t)le * expert_bytes + (size_t)row_in * row_bytes;
    float* srow = scale + (size_t)le * (size_t)rows_per_expert + (size_t)row_in;
    signed char* qrow = q + ((size_t)le * (size_t)rows_per_expert + (size_t)row_in) * (size_t)cols;

    float v[NG][DQR_VEC];
    float amax = 0.f;
    #pragma unroll
    for (int g = 0; g < NG; g++) {
        const int base = g * GSPAN + t * DQR_VEC;
        if (base < cols) {
            const unsigned char* blk = rbase + (size_t)(base >> 8) * BS;
            const int off = base & 255;
            #pragma unroll
            for (int i = 0; i < DQR_VEC; i++) {
                v[g][i] = dqr_val<QT>(blk, off + i);
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
    if (t == 0) *srow = d;
    const float inv = (d > 0.f) ? (1.f / d) : 0.f;

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
bool dispatch_mask(const unsigned char* src0, signed char* q0, float* scale0,
                   const int* counts, int e_base, int n_in, int rows_per_expert, int cols,
                   size_t expert_bytes, cudaStream_t stream) {
    const int ng = (cols + DQR_BLOCK * DQR_VEC - 1) / (DQR_BLOCK * DQR_VEC);
    const int grid = n_in * rows_per_expert;
    if (ng <= 1)
        deq_rows_i8_vec_mask_kernel<QT, 1><<<grid, DQR_BLOCK, 0, stream>>>(
            src0, q0, scale0, counts, e_base, n_in, rows_per_expert, cols, expert_bytes);
    else if (ng <= 4)
        deq_rows_i8_vec_mask_kernel<QT, 4><<<grid, DQR_BLOCK, 0, stream>>>(
            src0, q0, scale0, counts, e_base, n_in, rows_per_expert, cols, expert_bytes);
    else if (ng <= 8)
        deq_rows_i8_vec_mask_kernel<QT, 8><<<grid, DQR_BLOCK, 0, stream>>>(
            src0, q0, scale0, counts, e_base, n_in, rows_per_expert, cols, expert_bytes);
    else if (ng <= 12)
        deq_rows_i8_vec_mask_kernel<QT, 12><<<grid, DQR_BLOCK, 0, stream>>>(
            src0, q0, scale0, counts, e_base, n_in, rows_per_expert, cols, expert_bytes);
    else return false;
    return true;
}

template <int QT>
bool dispatch_mask_pair(const unsigned char* src0, signed char* q0, float* scale0,
                        const unsigned char* src1, signed char* q1, float* scale1,
                        const int* counts, int e_base, int n_in, int rows_per_expert, int cols,
                        size_t expert_bytes0, size_t expert_bytes1, cudaStream_t stream) {
    const int ng = (cols + DQR_BLOCK * DQR_VEC - 1) / (DQR_BLOCK * DQR_VEC);
    const int grid = 2 * n_in * rows_per_expert;
    if (ng <= 4)
        deq_rows_i8_vec_mask_pair_kernel<QT, 4><<<grid, DQR_BLOCK, 0, stream>>>(
            src0, q0, scale0, src1, q1, scale1, counts, e_base, n_in, rows_per_expert, cols,
            expert_bytes0, expert_bytes1);
    else if (ng <= 8)
        deq_rows_i8_vec_mask_pair_kernel<QT, 8><<<grid, DQR_BLOCK, 0, stream>>>(
            src0, q0, scale0, src1, q1, scale1, counts, e_base, n_in, rows_per_expert, cols,
            expert_bytes0, expert_bytes1);
    else if (ng <= 12)
        deq_rows_i8_vec_mask_pair_kernel<QT, 12><<<grid, DQR_BLOCK, 0, stream>>>(
            src0, q0, scale0, src1, q1, scale1, counts, e_base, n_in, rows_per_expert, cols,
            expert_bytes0, expert_bytes1);
    else return false;
    return true;
}

}  // namespace

bool launch_gguf_dequant_rows_i8_fast(int ggml_type, const void* src, signed char* q, float* scale,
                                      int rows, int cols, cudaStream_t stream) {
    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_DEQUANT_ROWS_I8_FAST");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    // The kernel only needs a thread's 4 values to sit inside one super-block, which 4-alignment
    // already guarantees, and nsb = cols>>8 to be exact -- i.e. cols % 256 == 0. Every group is
    // guarded by `base < cols`, so a trailing partial group is already handled. Requiring a full
    // 1024-wide group on top of that was a throughput guard (no idle threads), but it excluded
    // every width that is super-block- but not 1024-aligned: Muse Glimmer is 6656 (ng=7, last
    // group half-used) and 19968 (ng=20), so all of its attention and FFN weights fell back to
    // the two-pass kernel, which decodes the whole row TWICE (once for amax, once to quantize)
    // and costs more than the i8 GEMM that consumes it. A half-idle final group is far cheaper
    // than a second decode pass. Still bit-identical to the two-pass kernel -- same decode, same
    // amax (max is associative and exact in fp), same d = amax/127.f, same roundf(v * inv).
    // SPARKINFER_DEQUANT_ROWS_I8_ANYCOL=0 restores the 1024-aligned-only gate.
    static const int anycol = [] {
        const char* e = getenv("SPARKINFER_DEQUANT_ROWS_I8_ANYCOL");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    // Only widths with at least one full group are admitted, so a row is never dominated by its
    // partial tail. That deliberately leaves every sub-1024 width (Qwen3.6's MoE down is cols=512,
    // which the gather/mask launchers already serve) on exactly the path it takes today.
    const bool wide_ok = anycol && (cols % 256) == 0 && cols >= (DQR_BLOCK * DQR_VEC);
    if (!enabled || rows <= 0 || cols <= 0 ||
        ((cols % (DQR_BLOCK * DQR_VEC)) != 0 && !wide_ok)) return false;

    auto s = reinterpret_cast<const unsigned char*>(src);
    if (ggml_type == DQR_Q4_K) return dispatch<DQR_Q4_K>(s, q, scale, rows, cols, stream);
    if (ggml_type == DQR_Q5_K) return dispatch<DQR_Q5_K>(s, q, scale, rows, cols, stream);
    if (ggml_type == DQR_Q6_K) return dispatch<DQR_Q6_K>(s, q, scale, rows, cols, stream);
    return false;
}

bool launch_gguf_dequant_rows_i8_fast_pair(int ggml_type,
                                           const void* src0, signed char* q0, float* scale0,
                                           const void* src1, signed char* q1, float* scale1,
                                           int rows, int cols, cudaStream_t stream) {
    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_DEQUANT_ROWS_I8_FAST");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    if (!enabled || rows <= 0 || cols <= 0 || (cols % (DQR_BLOCK * DQR_VEC)) != 0) return false;
    auto s0 = reinterpret_cast<const unsigned char*>(src0);
    auto s1 = reinterpret_cast<const unsigned char*>(src1);
    if (ggml_type == DQR_Q4_K)
        return dispatch_pair<DQR_Q4_K>(s0, q0, scale0, s1, q1, scale1, rows, cols, stream);
    if (ggml_type == DQR_Q5_K)
        return dispatch_pair<DQR_Q5_K>(s0, q0, scale0, s1, q1, scale1, rows, cols, stream);
    if (ggml_type == DQR_Q6_K)
        return dispatch_pair<DQR_Q6_K>(s0, q0, scale0, s1, q1, scale1, rows, cols, stream);
    return false;
}

bool launch_gguf_dequant_rows_i8_fast_gather(
    int ggml_type, const void* src0, signed char* q0, float* scale0,
    const int* live_le, int n_live, int rows_per_expert, int cols,
    size_t expert_bytes, cudaStream_t stream) {
    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_DEQUANT_ROWS_I8_FAST");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    // Live-expert gather: allow cols=512 (Qwen3.6 MoE down Q5_K). Partial GSPAN is
    // base<cols-guarded; idle threads only hurt when dequanting the full expert stack.
    if (!enabled || !live_le || n_live <= 0 || rows_per_expert <= 0 || cols <= 0 ||
        (cols & 255) != 0)
        return false;
    auto s = reinterpret_cast<const unsigned char*>(src0);
    if (ggml_type == DQR_Q4_K)
        return dispatch_gather<DQR_Q4_K>(s, q0, scale0, live_le, n_live, rows_per_expert, cols,
                                         expert_bytes, stream);
    if (ggml_type == DQR_Q5_K)
        return dispatch_gather<DQR_Q5_K>(s, q0, scale0, live_le, n_live, rows_per_expert, cols,
                                         expert_bytes, stream);
    if (ggml_type == DQR_Q6_K)
        return dispatch_gather<DQR_Q6_K>(s, q0, scale0, live_le, n_live, rows_per_expert, cols,
                                         expert_bytes, stream);
    return false;
}

bool launch_gguf_dequant_rows_i8_fast_gather_pair(
    int ggml_type,
    const void* src0, signed char* q0, float* scale0,
    const void* src1, signed char* q1, float* scale1,
    const int* live_le, int n_live, int rows_per_expert, int cols,
    size_t expert_bytes0, size_t expert_bytes1, cudaStream_t stream) {
    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_DEQUANT_ROWS_I8_FAST");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    if (!enabled || !live_le || n_live <= 0 || rows_per_expert <= 0 || cols <= 0 ||
        (cols & 255) != 0)
        return false;
    auto s0 = reinterpret_cast<const unsigned char*>(src0);
    auto s1 = reinterpret_cast<const unsigned char*>(src1);
    if (ggml_type == DQR_Q4_K)
        return dispatch_gather_pair<DQR_Q4_K>(s0, q0, scale0, s1, q1, scale1, live_le, n_live,
                                              rows_per_expert, cols, expert_bytes0, expert_bytes1,
                                              stream);
    if (ggml_type == DQR_Q5_K)
        return dispatch_gather_pair<DQR_Q5_K>(s0, q0, scale0, s1, q1, scale1, live_le, n_live,
                                              rows_per_expert, cols, expert_bytes0, expert_bytes1,
                                              stream);
    if (ggml_type == DQR_Q6_K)
        return dispatch_gather_pair<DQR_Q6_K>(s0, q0, scale0, s1, q1, scale1, live_le, n_live,
                                              rows_per_expert, cols, expert_bytes0, expert_bytes1,
                                              stream);
    return false;
}

bool launch_gguf_dequant_rows_i8_fast_mask(
    int ggml_type, const void* src0, signed char* q0, float* scale0,
    const int* counts, int e_base, int n_in, int rows_per_expert, int cols,
    size_t expert_bytes, cudaStream_t stream) {
    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_DEQUANT_ROWS_I8_FAST");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    if (!enabled || !counts || n_in <= 0 || rows_per_expert <= 0 || cols <= 0 ||
        (cols & 255) != 0)
        return false;
    auto s = reinterpret_cast<const unsigned char*>(src0);
    if (ggml_type == DQR_Q4_K)
        return dispatch_mask<DQR_Q4_K>(s, q0, scale0, counts, e_base, n_in, rows_per_expert, cols,
                                       expert_bytes, stream);
    if (ggml_type == DQR_Q5_K)
        return dispatch_mask<DQR_Q5_K>(s, q0, scale0, counts, e_base, n_in, rows_per_expert, cols,
                                       expert_bytes, stream);
    if (ggml_type == DQR_Q6_K)
        return dispatch_mask<DQR_Q6_K>(s, q0, scale0, counts, e_base, n_in, rows_per_expert, cols,
                                       expert_bytes, stream);
    return false;
}

bool launch_gguf_dequant_rows_i8_fast_mask_pair(
    int ggml_type,
    const void* src0, signed char* q0, float* scale0,
    const void* src1, signed char* q1, float* scale1,
    const int* counts, int e_base, int n_in, int rows_per_expert, int cols,
    size_t expert_bytes0, size_t expert_bytes1, cudaStream_t stream) {
    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_DEQUANT_ROWS_I8_FAST");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    if (!enabled || !counts || n_in <= 0 || rows_per_expert <= 0 || cols <= 0 ||
        (cols & 255) != 0)
        return false;
    auto s0 = reinterpret_cast<const unsigned char*>(src0);
    auto s1 = reinterpret_cast<const unsigned char*>(src1);
    if (ggml_type == DQR_Q4_K)
        return dispatch_mask_pair<DQR_Q4_K>(s0, q0, scale0, s1, q1, scale1, counts, e_base, n_in,
                                            rows_per_expert, cols, expert_bytes0, expert_bytes1,
                                            stream);
    if (ggml_type == DQR_Q5_K)
        return dispatch_mask_pair<DQR_Q5_K>(s0, q0, scale0, s1, q1, scale1, counts, e_base, n_in,
                                            rows_per_expert, cols, expert_bytes0, expert_bytes1,
                                            stream);
    if (ggml_type == DQR_Q6_K)
        return dispatch_mask_pair<DQR_Q6_K>(s0, q0, scale0, s1, q1, scale1, counts, e_base, n_in,
                                            rows_per_expert, cols, expert_bytes0, expert_bytes1,
                                            stream);
    return false;
}

}  // namespace kernels
}  // namespace sparkinfer
