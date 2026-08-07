// Row-batched Q4_K / Q8_0 MMVQ for the DFlash verify window.
//
// The single-row mmvq kernels in gemm/gemv.cu are pure weight bandwidth: one CTA per
// output row, four warps striding the K super-blocks, ~2 dp4a per weight byte. Decode
// therefore costs one full read of the projection weights per forward, and the verify
// loop runs one forward per proposal token — a window of R proposals reads the entire
// weight stream R times even though every row hits the SAME weights.
//
// These kernels keep the one-CTA-per-output-row structure and hold R accumulators
// instead of one: the CTA loads each weight super-block once and dots it against all R
// activation rows, so a window costs one weight stream regardless of R. Only the
// activation side scales with R, and that is L2-resident (rows * K/32 * 36 B).
//
// EXACTNESS: the DFlash gate asserts token-for-token equality with the autoregressive
// path, so nothing about the arithmetic may move. Row r accumulates over the same kbx
// sequence in the same order, through the same vec_dot, and reduces through the same
// shared-memory + shuffle tree as the single-row kernel would on row r alone. There is
// no cross-row reduction anywhere and no reassociation: tmp[r] sees exactly the operands
// tmp saw before. Batching only decides which rows share a CTA.
//
// Self-contained per house convention: the block structs and the vec_dot helpers are
// copied from gemm/gemv.cu instead of shared through a header, so this TU compiles (and
// NVRTC-compiles) alone. Those copies must stay byte-identical to the originals —
// divergence is an accuracy-gate failure, not a style problem.
//
// Portable CUDA — sm_89 .. sm_120/sm_121.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#include "sparkinfer/kernels/dflash_rows.h"   // at file scope so kDFlashMaxRows is visible
#endif

namespace sparkinfer {
namespace kernels {

__device__ __forceinline__ void dfr_write(float* p, float v) { *p = v; }
__device__ __forceinline__ void dfr_write(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

struct dfr_block_q8_1 { __half2 ds; signed char qs[32]; };               // 36 B / 32 values
struct dfr_block_q4_K { __half2 dm; unsigned char scales[12]; unsigned char qs[128]; };  // 144 B / 256

__device__ __forceinline__ float dfr_vec_dot_q4_K(const dfr_block_q4_K* bq4,
                                                 const dfr_block_q8_1* bq8_1, int iqs) {
    int v[2], u[4]; float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* q4 = (const int*)(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0]; v[1] = q4[4];
    const unsigned short* scales = (const unsigned short*)bq4->scales;
    unsigned short aux[2]; const int j = bq8_offset / 2;
    if (j < 2) { aux[0] = scales[j] & 0x3f3f; aux[1] = scales[j + 2] & 0x3f3f; }
    else { aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
           aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j]     & 0xc0c0) >> 2); }
    const unsigned char* sc = (const unsigned char*)aux; const unsigned char* m = sc + 2;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const dfr_block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i] = q8[0]; u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.0f, sumf_m = 0.0f;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F, v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    float2 dm4f = __half22float2(bq4->dm);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

// Same dot, evaluated for R activation rows against one weight super-block. The two int4
// weight pulls, the 6-bit scale/min unpack and dm depend only on the weight, so they are
// done once and reused by every row -- that reuse is the whole point of the window, and the
// compiler does not find it on its own across R inlined calls (the reason gemv.cu's Q6_K
// multi-row kernel hoists the unpack by hand too). Every float operation and its order is
// unchanged from dfr_vec_dot_q4_K, so acc[r] lands bit-identical to calling that helper once
// per row. bq8_1 points at row 0's block for this super-block; rows are q81_per_row apart.
template <int R>
__device__ __forceinline__ void dfr_vec_dot_q4_K_rows(const dfr_block_q4_K* bq4,
                                                      const dfr_block_q8_1* bq8_1,
                                                      int q81_per_row, int iqs, float* acc) {
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* q4 = (const int*)(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int v0 = q4[0], v1 = q4[4];
    const unsigned short* scales = (const unsigned short*)bq4->scales;
    unsigned short aux[2]; const int j = bq8_offset / 2;
    if (j < 2) { aux[0] = scales[j] & 0x3f3f; aux[1] = scales[j + 2] & 0x3f3f; }
    else { aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
           aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j]     & 0xc0c0) >> 2); }
    const unsigned char* sc = (const unsigned char*)aux; const unsigned char* m = sc + 2;
    const float2 dm4f = __half22float2(bq4->dm);
    #pragma unroll
    for (int r = 0; r < R; r++) {
        const dfr_block_q8_1* base = bq8_1 + (size_t)r * q81_per_row;
        int u[4]; float d8[2];
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const dfr_block_q8_1* bq8i = base + bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = (const int*)bq8i->qs + ((iqs / 2) % 4);
            u[2 * i] = q8[0]; u[2 * i + 1] = q8[4];
        }
        float sumf_d = 0.0f, sumf_m = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int v0i = (v0 >> (4 * i)) & 0x0F0F0F0F, v1i = (v1 >> (4 * i)) & 0x0F0F0F0F;
            const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i], 0));
            const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i], 0));
            sumf_d += d8[i] * (dot1 * sc[i]);
            sumf_m += d8[i] * (dot2 * m[i]);
        }
        acc[r] += dm4f.x * sumf_d - dm4f.y * sumf_m;
    }
}

// Q8_0 blocks are 34 B (2-byte aligned only); read via explicit byte offsets like Q6_K.
__device__ __forceinline__ float dfr_q80_h2f(const unsigned char* p) {
    __half h; *reinterpret_cast<unsigned short*>(&h) = *reinterpret_cast<const unsigned short*>(p);
    return __half2float(h);
}
__device__ __forceinline__ int dfr_q80_get_int_b2(const unsigned char* p, int i32) {
    const unsigned short* u = reinterpret_cast<const unsigned short*>(p);
    return (int)u[2 * i32] | ((int)u[2 * i32 + 1] << 16);
}
__device__ __forceinline__ float dfr_vec_dot_q8_0_mmvq(const unsigned char* bw, const dfr_block_q8_1* ba) {
    const float dw = dfr_q80_h2f(bw);
    const int* a = reinterpret_cast<const int*>(ba->qs);
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) sumi = __dp4a(dfr_q80_get_int_b2(bw + 2, i), a[i], sumi);
    return dw * __low2float(ba->ds) * (float)sumi;
}

// Activation-row stride, in dfr_block_q8_1 units. The caller lays the window out as `rows`
// contiguous llama_q8_1_bytes(K) = (K >> 5) * sizeof(dfr_block_q8_1) byte rows, so row r
// begins exactly (K >> 5) BLOCKS in — the stride is only ever expressed in blocks here,
// never in bytes, because dfr_block_q8_1 is 36 B and a byte/element mix-up would leave row
// 0 correct while silently corrupting every later row of the window.
__device__ __forceinline__ int si_q81_blocks_per_row(int K) { return K >> 5; }

// ---- Q4_K MMVQ, R activation rows against one weight matrix ----------------------
// Mirrors si_mmvq_q4k_kernel: same kbx striding, same kqs, same 4-warp reduction, with the
// R rows of the verify window sharing one pass over the weight. Each super-block is read and
// unpacked once (dfr_vec_dot_q4_K_rows) and dotted against all R rows, so the window costs
// one weight stream instead of R.
template <typename OutT, int R>
__global__ void si_dflash_rows_mmvq_q4k_kernel(const dfr_block_q8_1* __restrict__ vy,
                                               const unsigned char* __restrict__ W,
                                               OutT* __restrict__ y, int N, int K) {
    constexpr int NW = 4, WS = 32, vdr = 2, qi = 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
    const int row = blockIdx.x;
    const dfr_block_q4_K* x_row = (const dfr_block_q4_K*)(W + (size_t)row * (K >> 8) * 144);
    const int blocks_per_row = K >> 8;                       // 256-weight superblocks
    const int blocks_per_iter = vdr * NW * WS / qi;          // = 8
    const int q81_per_row = si_q81_blocks_per_row(K);
    float tmp[R];
    #pragma unroll
    for (int r = 0; r < R; r++) tmp[r] = 0.0f;
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * 8;                             // q8_1 blocks per superblock = 8
        const int kqs = vdr * (tid % (qi / vdr));
        dfr_vec_dot_q4_K_rows<R>(x_row + kbx, vy + kby, q81_per_row, kqs, tmp);
    }
    // Each row owns its own slice, so the R writes are independent and one barrier still
    // separates every write from every read.
    __shared__ float tmp_shared[R][NW - 1][WS];
    if (warp > 0) {
        #pragma unroll
        for (int r = 0; r < R; r++) tmp_shared[r][warp - 1][lane] = tmp[r];
    }
    __syncthreads();
    if (warp > 0) return;
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int l = 0; l < NW - 1; l++) tmp[r] += tmp_shared[r][l][lane];
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) tmp[r] += __shfl_xor_sync(0xffffffff, tmp[r], m);
        if (lane == 0) dfr_write(y + (size_t)r * N + row, tmp[r]);
    }
}

// The row count is the only template parameter: K stays runtime so one instantiation
// serves every K % 256 == 0 shape (H=2048 -> 8 superblocks, the GDN 4096 -> 16), and the
// loop bound is not what this kernel is bound by.
#ifndef _MSC_VER
template __global__ void si_dflash_rows_mmvq_q4k_kernel<__nv_bfloat16, 1>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_dflash_rows_mmvq_q4k_kernel<__nv_bfloat16, 2>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_dflash_rows_mmvq_q4k_kernel<__nv_bfloat16, 3>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_dflash_rows_mmvq_q4k_kernel<__nv_bfloat16, 4>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_dflash_rows_mmvq_q4k_kernel<__nv_bfloat16, 5>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_dflash_rows_mmvq_q4k_kernel<__nv_bfloat16, 6>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_dflash_rows_mmvq_q4k_kernel<__nv_bfloat16, 7>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_dflash_rows_mmvq_q4k_kernel<__nv_bfloat16, 8>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
#endif
#ifndef _MSC_VER
template __global__ void si_dflash_rows_mmvq_q4k_kernel<float, 1>(const dfr_block_q8_1*, const unsigned char*, float*, int, int);
template __global__ void si_dflash_rows_mmvq_q4k_kernel<float, 2>(const dfr_block_q8_1*, const unsigned char*, float*, int, int);
template __global__ void si_dflash_rows_mmvq_q4k_kernel<float, 3>(const dfr_block_q8_1*, const unsigned char*, float*, int, int);
template __global__ void si_dflash_rows_mmvq_q4k_kernel<float, 4>(const dfr_block_q8_1*, const unsigned char*, float*, int, int);
template __global__ void si_dflash_rows_mmvq_q4k_kernel<float, 5>(const dfr_block_q8_1*, const unsigned char*, float*, int, int);
template __global__ void si_dflash_rows_mmvq_q4k_kernel<float, 6>(const dfr_block_q8_1*, const unsigned char*, float*, int, int);
template __global__ void si_dflash_rows_mmvq_q4k_kernel<float, 7>(const dfr_block_q8_1*, const unsigned char*, float*, int, int);
template __global__ void si_dflash_rows_mmvq_q4k_kernel<float, 8>(const dfr_block_q8_1*, const unsigned char*, float*, int, int);
#endif

// ---- Q8_0 MMVQ, R activation rows against one weight matrix ----------------------
// Mirrors si_mmvq_q80_kernel: the same flat `kb` stride over the K/32 blocks that the
// kfixed form fixes at compile time, kept runtime here. Q8_0 blocks are 34 B and only
// 2-byte aligned, so the weight side stays on the byte-offset helpers; those eight
// dfr_q80_get_int_b2 loads and the fp16 scale are invariant across the row loop.
template <typename OutT, int R>
__global__ void si_dflash_rows_mmvq_q80_kernel(const dfr_block_q8_1* __restrict__ vy,
                                               const unsigned char* __restrict__ W,
                                               OutT* __restrict__ y, int N, int K) {
    constexpr int NW = 4, WS = 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
    const int row = blockIdx.x;
    const int nb = K >> 5;
    const unsigned char* w_row = W + (size_t)row * nb * 34;
    // Q8_0 and Q8_1 both hold 32 values per block, so the activation stride is the same
    // block count the weight walks.
    const int q81_per_row = si_q81_blocks_per_row(K);
    float tmp[R];
    #pragma unroll
    for (int r = 0; r < R; r++) tmp[r] = 0.0f;
    for (int kb = tid; kb < nb; kb += NW * WS) {
        #pragma unroll
        for (int r = 0; r < R; r++)
            tmp[r] += dfr_vec_dot_q8_0_mmvq(w_row + (size_t)kb * 34,
                                           vy + (size_t)r * q81_per_row + kb);
    }
    __shared__ float tmp_shared[R][NW - 1][WS];
    if (warp > 0) {
        #pragma unroll
        for (int r = 0; r < R; r++) tmp_shared[r][warp - 1][lane] = tmp[r];
    }
    __syncthreads();
    if (warp > 0) return;
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int l = 0; l < NW - 1; l++) tmp[r] += tmp_shared[r][l][lane];
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) tmp[r] += __shfl_xor_sync(0xffffffff, tmp[r], m);
        if (lane == 0) dfr_write(y + (size_t)r * N + row, tmp[r]);
    }
}

#ifndef _MSC_VER
template __global__ void si_dflash_rows_mmvq_q80_kernel<__nv_bfloat16, 1>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_dflash_rows_mmvq_q80_kernel<__nv_bfloat16, 2>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_dflash_rows_mmvq_q80_kernel<__nv_bfloat16, 3>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_dflash_rows_mmvq_q80_kernel<__nv_bfloat16, 4>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_dflash_rows_mmvq_q80_kernel<__nv_bfloat16, 5>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_dflash_rows_mmvq_q80_kernel<__nv_bfloat16, 6>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_dflash_rows_mmvq_q80_kernel<__nv_bfloat16, 7>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_dflash_rows_mmvq_q80_kernel<__nv_bfloat16, 8>(const dfr_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
#endif
// OutT stays generic to match the single-row pair, but only the bf16 half is instantiated:
// the verify window has no fp32 Q8_0 consumer, and each row count is a separate kernel body.

// ---- fused GDN qkv + z, R activation rows ---------------------------------------
// One block per row index: warps 0-3 -> qkv[row], warps 4-7 -> z[row], grid =
// max(n_qkv, n_z), keeping vy hot in L2 across both when row < min(n_qkv, n_z). The two
// halves keep separate shared arrays and separate early-outs, exactly as the single-row
// pack2 kernel: a half whose row index is past its own N leaves before its barrier and
// the other half completes on its own.
template <int R>
__global__ void si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel(const dfr_block_q8_1* __restrict__ vy,
                                                           const unsigned char* __restrict__ qkv_w,
                                                           const unsigned char* __restrict__ z_w,
                                                           __nv_bfloat16* __restrict__ qkv_out,
                                                           __nv_bfloat16* __restrict__ z_out,
                                                           int n_qkv, int n_z, int K) {
    constexpr int NW = 4, WS = 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int sub = warp & 3;
    const int row = blockIdx.x;
    const int tid4 = sub * WS + lane;
    const int kbx0 = tid4 >> 4;
    const int kqs = 2 * (tid4 & 15);
    const int nsuper = K >> 8;
    const int q81_per_row = si_q81_blocks_per_row(K);
    float tmp[R];
    #pragma unroll
    for (int r = 0; r < R; r++) tmp[r] = 0.f;
    if (warp < 4) {
        if (row >= n_qkv) return;
        const dfr_block_q4_K* x_row = (const dfr_block_q4_K*)(qkv_w + (size_t)row * nsuper * 144);
        for (int kbx = kbx0; kbx < nsuper; kbx += 8) {
            dfr_vec_dot_q4_K_rows<R>(x_row + kbx, vy + (size_t)kbx * 8, q81_per_row, kqs, tmp);
        }
        __shared__ float tq[R][NW - 1][WS];
        if (sub > 0) {
            #pragma unroll
            for (int r = 0; r < R; r++) tq[r][sub - 1][lane] = tmp[r];
        }
        __syncthreads();
        if (sub > 0) return;
        #pragma unroll
        for (int r = 0; r < R; r++) {
            #pragma unroll
            for (int l = 0; l < NW - 1; l++) tmp[r] += tq[r][l][lane];
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) tmp[r] += __shfl_xor_sync(0xffffffff, tmp[r], m);
            if (lane == 0) dfr_write(qkv_out + (size_t)r * n_qkv + row, tmp[r]);
        }
    } else {
        if (row >= n_z) return;
        const dfr_block_q4_K* x_row = (const dfr_block_q4_K*)(z_w + (size_t)row * nsuper * 144);
        for (int kbx = kbx0; kbx < nsuper; kbx += 8) {
            dfr_vec_dot_q4_K_rows<R>(x_row + kbx, vy + (size_t)kbx * 8, q81_per_row, kqs, tmp);
        }
        __shared__ float tz[R][NW - 1][WS];
        if (sub > 0) {
            #pragma unroll
            for (int r = 0; r < R; r++) tz[r][sub - 1][lane] = tmp[r];
        }
        __syncthreads();
        if (sub > 0) return;
        #pragma unroll
        for (int r = 0; r < R; r++) {
            #pragma unroll
            for (int l = 0; l < NW - 1; l++) tmp[r] += tz[r][l][lane];
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) tmp[r] += __shfl_xor_sync(0xffffffff, tmp[r], m);
            if (lane == 0) dfr_write(z_out + (size_t)r * n_z + row, tmp[r]);
        }
    }
}

#ifndef _MSC_VER
template __global__ void si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<1>(const dfr_block_q8_1*, const unsigned char*, const unsigned char*, __nv_bfloat16*, __nv_bfloat16*, int, int, int);
template __global__ void si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<2>(const dfr_block_q8_1*, const unsigned char*, const unsigned char*, __nv_bfloat16*, __nv_bfloat16*, int, int, int);
template __global__ void si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<3>(const dfr_block_q8_1*, const unsigned char*, const unsigned char*, __nv_bfloat16*, __nv_bfloat16*, int, int, int);
template __global__ void si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<4>(const dfr_block_q8_1*, const unsigned char*, const unsigned char*, __nv_bfloat16*, __nv_bfloat16*, int, int, int);
template __global__ void si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<5>(const dfr_block_q8_1*, const unsigned char*, const unsigned char*, __nv_bfloat16*, __nv_bfloat16*, int, int, int);
template __global__ void si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<6>(const dfr_block_q8_1*, const unsigned char*, const unsigned char*, __nv_bfloat16*, __nv_bfloat16*, int, int, int);
template __global__ void si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<7>(const dfr_block_q8_1*, const unsigned char*, const unsigned char*, __nv_bfloat16*, __nv_bfloat16*, int, int, int);
template __global__ void si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<8>(const dfr_block_q8_1*, const unsigned char*, const unsigned char*, __nv_bfloat16*, __nv_bfloat16*, int, int, int);
#endif

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY

namespace {

// Every launcher below screens the window shape once, here. K must split into whole
// 256-value super-blocks (the Q4_K weight block and the 8-q8_1-blocks-per-super-block
// activation walk both assume it) and `rows` must be one of the instantiated row counts.
// A rejected window is a silent no-op: the caller keeps its single-token verify path.
inline bool dflash_rows_shape_ok(int K, int rows) {
    return rows >= 1 && rows <= kDFlashMaxRows && K > 0 && (K & 255) == 0;
}

template <typename OutT>
inline void dflash_rows_mmvq_q4k_dispatch(const dfr_block_q8_1* q, const unsigned char* w,
                                          OutT* y, int N, int K, int rows, cudaStream_t stream) {
    switch (rows) {
        case 1: si_dflash_rows_mmvq_q4k_kernel<OutT, 1><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        case 2: si_dflash_rows_mmvq_q4k_kernel<OutT, 2><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        case 3: si_dflash_rows_mmvq_q4k_kernel<OutT, 3><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        case 4: si_dflash_rows_mmvq_q4k_kernel<OutT, 4><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        case 5: si_dflash_rows_mmvq_q4k_kernel<OutT, 5><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        case 6: si_dflash_rows_mmvq_q4k_kernel<OutT, 6><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        case 7: si_dflash_rows_mmvq_q4k_kernel<OutT, 7><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        case 8: si_dflash_rows_mmvq_q4k_kernel<OutT, 8><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        default: break;
    }
}

template <typename OutT>
inline void dflash_rows_mmvq_q80_dispatch(const dfr_block_q8_1* q, const unsigned char* w,
                                          OutT* y, int N, int K, int rows, cudaStream_t stream) {
    switch (rows) {
        case 1: si_dflash_rows_mmvq_q80_kernel<OutT, 1><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        case 2: si_dflash_rows_mmvq_q80_kernel<OutT, 2><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        case 3: si_dflash_rows_mmvq_q80_kernel<OutT, 3><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        case 4: si_dflash_rows_mmvq_q80_kernel<OutT, 4><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        case 5: si_dflash_rows_mmvq_q80_kernel<OutT, 5><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        case 6: si_dflash_rows_mmvq_q80_kernel<OutT, 6><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        case 7: si_dflash_rows_mmvq_q80_kernel<OutT, 7><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        case 8: si_dflash_rows_mmvq_q80_kernel<OutT, 8><<<N, 4 * 32, 0, stream>>>(q, w, y, N, K); break;
        default: break;
    }
}

} // namespace

void launch_dflash_rows_mmvq_q4k(const void* q81, const void* W, void* y,
                                 int N, int K, int rows, cudaStream_t stream) {
    if (N <= 0 || !dflash_rows_shape_ok(K, rows)) return;
    dflash_rows_mmvq_q4k_dispatch(reinterpret_cast<const dfr_block_q8_1*>(q81),
                                  reinterpret_cast<const unsigned char*>(W),
                                  reinterpret_cast<__nv_bfloat16*>(y), N, K, rows, stream);
}

void launch_dflash_rows_mmvq_q4k_f32(const void* q81, const void* W, float* y,
                                     int N, int K, int rows, cudaStream_t stream) {
    if (N <= 0 || !dflash_rows_shape_ok(K, rows)) return;
    dflash_rows_mmvq_q4k_dispatch(reinterpret_cast<const dfr_block_q8_1*>(q81),
                                  reinterpret_cast<const unsigned char*>(W),
                                  y, N, K, rows, stream);
}

void launch_dflash_rows_mmvq_q80(const void* q81, const void* W, void* y,
                                 int N, int K, int rows, cudaStream_t stream) {
    if (N <= 0 || !dflash_rows_shape_ok(K, rows)) return;
    dflash_rows_mmvq_q80_dispatch(reinterpret_cast<const dfr_block_q8_1*>(q81),
                                  reinterpret_cast<const unsigned char*>(W),
                                  reinterpret_cast<__nv_bfloat16*>(y), N, K, rows, stream);
}

void launch_dflash_rows_mmvq_gdn_qkv_z(const void* q81, const void* qkv_w, const void* z_w,
                                       void* qkv_out, void* z_out,
                                       int n_qkv, int n_z, int K, int rows,
                                       cudaStream_t stream) {
    const int grid = n_qkv > n_z ? n_qkv : n_z;
    if (grid <= 0 || !dflash_rows_shape_ok(K, rows)) return;
    const dfr_block_q8_1* q = reinterpret_cast<const dfr_block_q8_1*>(q81);
    const unsigned char* qw = reinterpret_cast<const unsigned char*>(qkv_w);
    const unsigned char* zw = reinterpret_cast<const unsigned char*>(z_w);
    __nv_bfloat16* qo = reinterpret_cast<__nv_bfloat16*>(qkv_out);
    __nv_bfloat16* zo = reinterpret_cast<__nv_bfloat16*>(z_out);
    switch (rows) {
        case 1: si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<1><<<grid, 8 * 32, 0, stream>>>(q, qw, zw, qo, zo, n_qkv, n_z, K); break;
        case 2: si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<2><<<grid, 8 * 32, 0, stream>>>(q, qw, zw, qo, zo, n_qkv, n_z, K); break;
        case 3: si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<3><<<grid, 8 * 32, 0, stream>>>(q, qw, zw, qo, zo, n_qkv, n_z, K); break;
        case 4: si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<4><<<grid, 8 * 32, 0, stream>>>(q, qw, zw, qo, zo, n_qkv, n_z, K); break;
        case 5: si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<5><<<grid, 8 * 32, 0, stream>>>(q, qw, zw, qo, zo, n_qkv, n_z, K); break;
        case 6: si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<6><<<grid, 8 * 32, 0, stream>>>(q, qw, zw, qo, zo, n_qkv, n_z, K); break;
        case 7: si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<7><<<grid, 8 * 32, 0, stream>>>(q, qw, zw, qo, zo, n_qkv, n_z, K); break;
        case 8: si_dflash_rows_mmvq_gdn_qkv_z_pack2_kernel<8><<<grid, 8 * 32, 0, stream>>>(q, qw, zw, qo, zo, n_qkv, n_z, K); break;
        default: break;
    }
}
#endif

} // namespace kernels
} // namespace sparkinfer
