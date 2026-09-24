// GEMV against PTQ1_0 ternary weights, read in their stored 28-byte blocks.
// See sparkinfer/kernels/ternary.h for the format and for the basis the activation must be in.
#include "sparkinfer/kernels/ternary.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace sparkinfer { namespace kernels {

namespace {

constexpr int kBlockElems = 128;
constexpr int kBlockBytes = 28;   // 24 five-trit carriers + 2 four-trit carriers + fp16 scale
constexpr int kWarpsPerCta = 4;

// Which carrier byte and trit position a weight sits in. The 24 five-trit bytes are walked in two
// runs -- 16 then 8 -- each emitting its trits position-major, then the two four-trit bytes. This
// is ggml's TQ1_0 walk; reading it carrier-major instead gets every value right and every one in
// the wrong place, which is invisible until you compare against the unquantized checkpoint.
__device__ __forceinline__ int ptq1_trit(const unsigned char* __restrict__ qs, int idx) {
    int byte, m;
    if (idx < 80) {
        m = idx >> 4;
        byte = idx & 15;
    } else if (idx < 120) {
        const int r = idx - 80;
        m = r >> 3;
        byte = 16 + (r & 7);
    } else {
        const int r = idx - 120;
        m = r >> 1;
        byte = 24 + (r & 1);
    }
    // Carriers are scaled into the whole byte rather than packed as plain base 3, so the digit
    // comes back out by multiplying up and taking the high bits -- ggml's own extraction.
    //
    // The multiplier is a select chain, not a table. `m` is a runtime value that differs across
    // the lanes of a warp, so an array indexed by it -- however it is declared -- costs either a
    // local-memory load per trit or a serialized constant-bank access per distinct m. Every
    // weight in the model goes through this line.
    const unsigned int p3 = m == 0 ? 1u : m == 1 ? 3u : m == 2 ? 9u : m == 3 ? 27u : 81u;
    const unsigned int q = (unsigned char)(qs[byte] * p3);
    return (int)((q * 3u) >> 8) - 1;
}

template <typename OutT>
__device__ __forceinline__ void store_out(OutT* y, int row, float v);
template <>
__device__ __forceinline__ void store_out<__nv_bfloat16>(__nv_bfloat16* y, int row, float v) {
    y[row] = __float2bfloat16(v);
}
template <>
__device__ __forceinline__ void store_out<float>(float* y, int row, float v) {
    y[row] = v;
}

// blockIdx.y selects the activation, so one launch covers a batch of them -- which is what
// prefill needs, since it projects N tokens at once rather than one. The weight row a warp walks
// is the same for every activation, so the batch reuses those loads within the CTA's L1.
template <typename OutT>
__global__ void gemv_ptq1_kernel(const __nv_bfloat16* __restrict__ x,
                                 const unsigned char* __restrict__ w,
                                 OutT* __restrict__ y, int n_rows, int k) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarpsPerCta + warp;
    if (row >= n_rows) return;
    const int batch = blockIdx.y;
    x += (size_t)batch * k;
    y += (size_t)batch * n_rows;

    const int n_blocks = k / kBlockElems;
    const unsigned char* wrow = w + (size_t)row * n_blocks * kBlockBytes;

    // The block is staged in shared memory once and read four times from there. Each lane's
    // carrier byte is a data-dependent index into the same 28 bytes, so straight off global every
    // one of the four passes re-issued 32 scattered byte loads that only L1 was saving.
    __shared__ unsigned char sblk[kWarpsPerCta][kBlockBytes];
    unsigned char* myblk = sblk[warp];

    float acc = 0.0f;
    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* qs = wrow + (size_t)b * kBlockBytes;
        if (lane < kBlockBytes) myblk[lane] = qs[lane];
        __syncwarp();
        // The scale sits in the last two bytes. Blocks are 28 bytes and rows start block-aligned,
        // so this is 2-byte aligned.
        const __half scale_h = *reinterpret_cast<const __half*>(myblk + kBlockBytes - 2);
        const float scale = __half2float(scale_h);

        const __nv_bfloat16* xb = x + (size_t)b * kBlockElems;
        float part = 0.0f;
#pragma unroll
        for (int t = 0; t < kBlockElems / 32; ++t) {
            const int idx = lane + t * 32;
            part += (float)ptq1_trit(myblk, idx) * __bfloat162float(xb[idx]);
        }
        acc += scale * part;
        // The next iteration overwrites the staging this one is still reading.
        __syncwarp();
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) store_out<OutT>(y, row, acc);
}

// A batch of activations against ONE weight read. The previous batched launch put the batch on
// blockIdx.y, which made every activation an independent block: it re-read and re-decoded the
// whole weight matrix per row, so it was N GEMVs with fewer launches and no sharing at all. Here
// a warp owns a weight row for the whole batch, so each trit is fetched and decoded once and then
// multiplied into every activation -- the weight traffic and the unpacking are paid once rather
// than BATCH times, which is the entire reason a packed decode step is cheaper than N single ones.
//
// Per row the accumulation is unchanged: the same four per-lane terms in the same order, the same
// `acc += scale * part`, the same shuffle reduction. That is what keeps a packed step bit-identical
// to the N separate steps it stands in for -- gemv_ptq1_gpu_test asserts exactly that.
template <typename OutT, int BMAX>
__global__ void gemm_ptq1_kernel(const __nv_bfloat16* __restrict__ x,
                                 const unsigned char* __restrict__ w,
                                 OutT* __restrict__ y, int n_rows, int k, int batch) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarpsPerCta + warp;
    if (row >= n_rows) return;

    const int n_blocks = k / kBlockElems;
    const unsigned char* wrow = w + (size_t)row * n_blocks * kBlockBytes;

    __shared__ unsigned char sblk[kWarpsPerCta][kBlockBytes];
    unsigned char* myblk = sblk[warp];

    float acc[BMAX];
#pragma unroll
    for (int j = 0; j < BMAX; ++j) acc[j] = 0.0f;

    for (int b = 0; b < n_blocks; ++b) {
        const unsigned char* qs = wrow + (size_t)b * kBlockBytes;
        if (lane < kBlockBytes) myblk[lane] = qs[lane];
        __syncwarp();
        const __half scale_h = *reinterpret_cast<const __half*>(myblk + kBlockBytes - 2);
        const float scale = __half2float(scale_h);

        float part[BMAX];
#pragma unroll
        for (int j = 0; j < BMAX; ++j) part[j] = 0.0f;
#pragma unroll
        for (int t = 0; t < kBlockElems / 32; ++t) {
            const int idx = lane + t * 32;
            // Decoded ONCE, then applied to every activation in the batch.
            const float tv = (float)ptq1_trit(myblk, idx);
            const __nv_bfloat16* xt = x + (size_t)b * kBlockElems + idx;
#pragma unroll
            for (int j = 0; j < BMAX; ++j)
                if (j < batch) part[j] += tv * __bfloat162float(xt[(size_t)j * k]);
        }
#pragma unroll
        for (int j = 0; j < BMAX; ++j) acc[j] += scale * part[j];
        __syncwarp();
    }

#pragma unroll
    for (int j = 0; j < BMAX; ++j) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc[j] += __shfl_down_sync(0xffffffffu, acc[j], off);
        if (lane == 0 && j < batch) store_out<OutT>(y + (size_t)j * n_rows, row, acc[j]);
    }
}

// ----- dp4a path: int8 activation, trits decoded through a shared-memory table -----
//
// The kernels above are issue-bound, not load-bound: every weight costs a byte load, a select
// chain, two multiplies, a shift, an int->float and a bf16 load before its FMA -- ~12 instructions
// a weight, ~6 ms of pure issue per token at 27B weights, so they run at ~10% of DRAM bandwidth.
// Here a carrier byte costs one table load for all of its trits and four weights cost one dp4a.
//
// The activation is quantized to int8 per 128-block (one f32 scale, absmax/127) and PERMUTED into
// carrier order, so the trits one byte carries meet four contiguous activation bytes:
//   words 5g+i (g<6, i<4): the activation at carrier 4g+i's trits m=0..3
//   word  5g+4           : the activation at trit m=4 of carriers 4g..4g+3
//   words 30, 31         : the four-trit carriers 24 and 25, m=0..3
// The table maps a carrier byte to base-3 digit CODES d = trit+1 in {0,1,2}: bits 0-1 of byte m
// hold d_m for m<4 and bits 2-3 of byte 0 hold d_4. sum(trit*x) = sum(d*x) - sum(x) exactly, so
// the block's int8 sum is subtracted once. Per row the result does not depend on the batch width,
// so a packed step stays bit-identical to the single-row steps it stands in for.
constexpr int kDpThreads = 256;
constexpr int kDpMaxBatch = 32;   // a c32 packed step reads the weights once, not four times
constexpr int kDpMaxK = 17408;    // Ternary-Bonsai-2-27B's widest input (the FFN down leg)
// Quantized activations for in-flight launches. A launch takes the next slot round-robin: the
// decode layer runs ternary projections on a side stream concurrently with the main one, so a
// single scratch would be overwritten under a running kernel. Static device memory, so nothing is
// allocated inside a graph capture.
constexpr int kDpSlots = 16;
__device__ int4 g_dp_xq[kDpSlots][kDpMaxBatch * kDpMaxK / 16];
__device__ float g_dp_xs[kDpSlots][kDpMaxBatch * kDpMaxK / kBlockElems];
__device__ int g_dp_xsum[kDpSlots][kDpMaxBatch * kDpMaxK / kBlockElems];

__device__ __forceinline__ int dp_perm_src(int p) {
    const int wd = p >> 2, i = p & 3;
    int j, m;
    if (wd < 30) {
        const int g = wd / 5, r = wd - g * 5;
        j = r < 4 ? 4 * g + r : 4 * g + i;
        m = r < 4 ? i : 4;
    } else {
        j = 24 + (wd - 30);
        m = i;
    }
    return j < 16 ? m * 16 + j : j < 24 ? 80 + m * 8 + (j - 16) : 120 + m * 2 + (j - 24);
}

// grid (n_blocks, batch), kBlockElems threads: one thread per permuted position.
__global__ void ptq1_dp_quant_kernel(const __nv_bfloat16* __restrict__ x, int k, int slot) {
    const int b = blockIdx.x, j = blockIdx.y, p = threadIdx.x;
    const int nb = k / kBlockElems;
    const float v = __bfloat162float(x[(size_t)j * k + (size_t)b * kBlockElems + dp_perm_src(p)]);
    __shared__ float s_max[kBlockElems / 32];
    __shared__ int s_sum[kBlockElems / 32];
    float a = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    if ((p & 31) == 0) s_max[p >> 5] = a;
    __syncthreads();
    float amax = s_max[0];
#pragma unroll
    for (int i = 1; i < kBlockElems / 32; ++i) amax = fmaxf(amax, s_max[i]);
    const float inv = amax > 0.f ? 127.f / amax : 0.f;
    const int q = max(-127, min(127, __float2int_rn(v * inv)));
    reinterpret_cast<signed char*>(g_dp_xq[slot])[((size_t)j * nb + b) * kBlockElems + p] =
        (signed char)q;
    int s = q;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    if ((p & 31) == 0) s_sum[p >> 5] = s;
    __syncthreads();
    if (p == 0) {
        int t = 0;
#pragma unroll
        for (int i = 0; i < kBlockElems / 32; ++i) t += s_sum[i];
        g_dp_xsum[slot][j * nb + b] = t;
        g_dp_xs[slot][j * nb + b] = amax / 127.f;
    }
}

__device__ __forceinline__ unsigned dp_lut_entry(int q) {
    unsigned e = 0;
#pragma unroll
    for (int m = 0; m < 5; ++m) {
        const unsigned p3 = m == 0 ? 1u : m == 1 ? 3u : m == 2 ? 9u : m == 3 ? 27u : 81u;
        const unsigned d = (((unsigned)(unsigned char)(q * p3)) * 3u) >> 8;
        e |= m < 4 ? d << (8 * m) : d << 2;
    }
    return e;
}

// G lanes per weight row, each taking whole 28-byte blocks b = sub, sub+G, ...
template <typename OutT, int G, int BMAX>
__global__ void __launch_bounds__(kDpThreads)
gemm_ptq1_dp4a_kernel(const unsigned char* __restrict__ w, OutT* __restrict__ y,
                      int n_rows, int k, int batch, int slot) {
    // 32 copies of the 256-entry table, entry e of copy c at word e*32+c: lane c always hits bank
    // c, so the data-dependent lookups never conflict.
    __shared__ unsigned s_lut[256 * 32];
    for (int i = threadIdx.x; i < 256 * 32; i += kDpThreads) s_lut[i] = dp_lut_entry(i >> 5);
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const unsigned* lut = s_lut + lane;
    const int row = blockIdx.x * (kDpThreads / G) + threadIdx.x / G;
    const int sub = threadIdx.x % G;
    const bool live = row < n_rows;
    const int nb = k / kBlockElems;
    const int4* xq = g_dp_xq[slot];
    const float* xs = g_dp_xs[slot];
    const int* xsum = g_dp_xsum[slot];

    float acc[BMAX];
#pragma unroll
    for (int j = 0; j < BMAX; ++j) acc[j] = 0.0f;

    if (live) {
        const unsigned* wrow =
            reinterpret_cast<const unsigned*>(w + (size_t)row * nb * kBlockBytes);
        for (int b = sub; b < nb; b += G) {
            unsigned wq[7];
#pragma unroll
            for (int i = 0; i < 7; ++i) wq[i] = __ldg(wrow + (size_t)b * 7 + i);
            unsigned C[26], E[6];
#pragma unroll
            for (int g = 0; g < 6; ++g) {
                unsigned L[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    L[i] = lut[((wq[g] >> (8 * i)) & 0xffu) * 32];
                    C[4 * g + i] = L[i] & 0x03030303u;
                }
                const unsigned lo = __byte_perm(L[0], L[1], 0x0040);
                const unsigned hi = __byte_perm(L[2], L[3], 0x0040);
                E[g] = (__byte_perm(lo, hi, 0x5410) >> 2) & 0x03030303u;
            }
            C[24] = lut[(wq[6] & 0xffu) * 32] & 0x03030303u;
            C[25] = lut[((wq[6] >> 8) & 0xffu) * 32] & 0x03030303u;
            const float ws = __half2float(__ushort_as_half((unsigned short)(wq[6] >> 16)));
#pragma unroll
            for (int j = 0; j < BMAX; ++j) {
                if (j >= batch) break;
                const int4* xb = xq + ((size_t)j * nb + b) * (kBlockElems / 16);
                int X[32];
#pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int4 t = xb[v];
                    X[4 * v] = t.x; X[4 * v + 1] = t.y; X[4 * v + 2] = t.z; X[4 * v + 3] = t.w;
                }
                int dot = 0;
#pragma unroll
                for (int g = 0; g < 6; ++g) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) dot = __dp4a((int)C[4 * g + i], X[5 * g + i], dot);
                    dot = __dp4a((int)E[g], X[5 * g + 4], dot);
                }
                dot = __dp4a((int)C[24], X[30], dot);
                dot = __dp4a((int)C[25], X[31], dot);
                acc[j] += ws * xs[j * nb + b] * (float)(dot - xsum[j * nb + b]);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < BMAX; ++j) {
#pragma unroll
        for (int off = G / 2; off > 0; off >>= 1)
            acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], off);
        if (live && sub == 0 && j < batch) store_out<OutT>(y + (size_t)j * n_rows, row, acc[j]);
    }
}

// SPARKINFER_PTQ1_DP4A=0 restores the float kernels above, for an A/B out of one binary.
bool ptq1_dp4a_on() {
    static const bool v = [] {
        const char* e = getenv("SPARKINFER_PTQ1_DP4A");
        return !(e && e[0] == '0');
    }();
    return v;
}

template <typename OutT, int BMAX>
void launch_dp4a_g(const unsigned char* w, OutT* y, int n_rows, int k, int batch, int slot,
                   int g, cudaStream_t stream) {
    const unsigned grid = (unsigned)((n_rows + kDpThreads / g - 1) / (kDpThreads / g));
    if (g == 8)       gemm_ptq1_dp4a_kernel<OutT, 8, BMAX><<<grid, kDpThreads, 0, stream>>>(w, y, n_rows, k, batch, slot);
    else if (g == 16) gemm_ptq1_dp4a_kernel<OutT, 16, BMAX><<<grid, kDpThreads, 0, stream>>>(w, y, n_rows, k, batch, slot);
    else              gemm_ptq1_dp4a_kernel<OutT, 32, BMAX><<<grid, kDpThreads, 0, stream>>>(w, y, n_rows, k, batch, slot);
}

template <typename OutT>
bool launch_dp4a(const __nv_bfloat16* x, const unsigned char* w, OutT* y, int n_rows, int k,
                 int batch, cudaStream_t stream) {
    if (!ptq1_dp4a_on() || k > kDpMaxK || (reinterpret_cast<uintptr_t>(w) & 3) != 0) return false;
    static std::atomic<unsigned> next_slot{0};
    const int nb = k / kBlockElems;
    // Enough CTAs to cover the SMs twice over, with no more lanes per row than it has blocks.
    int g = 8;
    while (g < 32 && (long)n_rows * g / kDpThreads < 340 && nb >= 2 * g) g *= 2;
    for (int b0 = 0; b0 < batch; b0 += kDpMaxBatch) {
        const int m = batch - b0 < kDpMaxBatch ? batch - b0 : kDpMaxBatch;
        const int slot = (int)(next_slot.fetch_add(1, std::memory_order_relaxed) % kDpSlots);
        ptq1_dp_quant_kernel<<<dim3((unsigned)nb, (unsigned)m), kBlockElems, 0, stream>>>(
            x + (size_t)b0 * k, k, slot);
        OutT* yc = y + (size_t)b0 * n_rows;
        if (m == 1)      launch_dp4a_g<OutT, 1>(w, yc, n_rows, k, m, slot, g, stream);
        else if (m <= 2) launch_dp4a_g<OutT, 2>(w, yc, n_rows, k, m, slot, g, stream);
        else if (m <= 4) launch_dp4a_g<OutT, 4>(w, yc, n_rows, k, m, slot, g, stream);
        else if (m <= 8) launch_dp4a_g<OutT, 8>(w, yc, n_rows, k, m, slot, g, stream);
        else if (m <= 16) launch_dp4a_g<OutT, 16>(w, yc, n_rows, k, m, slot, g, stream);
        else             launch_dp4a_g<OutT, 32>(w, yc, n_rows, k, m, slot, g, stream);
    }
    return true;
}

// Embedding lookup and un-rotation in one pass, keeping float across the transform. Decoding to
// bf16 first and rotating afterwards costs real accuracy -- the Hadamard sums 1024 values, so it
// sums 1024 already-rounded ones -- which showed up as PPL 8.11 against the host path's 8.07.
__global__ void embedding_ptq1_unrotate_kernel(const int* __restrict__ tok,
                                               const unsigned char* __restrict__ table,
                                               const signed char* __restrict__ sign,
                                               __nv_bfloat16* __restrict__ out, int k, int block) {
    extern __shared__ float sh[];
    const int r = blockIdx.y;
    const int base = blockIdx.x * block;
    const int n_blocks = k / kBlockElems;
    const unsigned char* wrow = table + (size_t)tok[r] * n_blocks * kBlockBytes;

    for (int i = threadIdx.x; i < block; i += blockDim.x) {
        const int e = base + i;
        const unsigned char* qs = wrow + (size_t)(e / kBlockElems) * kBlockBytes;
        const __half scale_h = *reinterpret_cast<const __half*>(qs + kBlockBytes - 2);
        sh[i] = (float)ptq1_trit(qs, e % kBlockElems) * __half2float(scale_h);
    }
    __syncthreads();

    for (int len = 1; len < block; len <<= 1) {
        for (int i = threadIdx.x; i < block / 2; i += blockDim.x) {
            const int lo = ((i / len) * 2 * len) + (i % len);
            const int hi = lo + len;
            const float a = sh[lo], b = sh[hi];
            sh[lo] = a + b;
            sh[hi] = a - b;
        }
        __syncthreads();
    }

    // R^-1 = diag(s) . H: the transform, then the signs.
    const float norm = rsqrtf((float)block);
    for (int i = threadIdx.x; i < block; i += blockDim.x)
        out[(size_t)r * k + base + i] =
            __float2bfloat16(sh[i] * norm * (float)sign[base + i]);
}

// A whole weight matrix out of its ternary blocks and back into the architecture's basis: decode,
// then take the stored rotation off each row. This is what lets prefill keep its existing
// projection branches -- they ask dq() for bf16 weights and get ordinary ones, while the resident
// copy stays ternary. Same body as the embedding lookup, with the row chosen directly.
__global__ void ptq1_rows_unrotate_kernel(const unsigned char* __restrict__ w,
                                          const signed char* __restrict__ sign,
                                          __nv_bfloat16* __restrict__ out, int k, int block) {
    extern __shared__ float sh[];
    const int row = blockIdx.y;
    const int base = blockIdx.x * block;
    const int n_blocks = k / kBlockElems;
    const unsigned char* wrow = w + (size_t)row * n_blocks * kBlockBytes;

    for (int i = threadIdx.x; i < block; i += blockDim.x) {
        const int e = base + i;
        const unsigned char* qs = wrow + (size_t)(e / kBlockElems) * kBlockBytes;
        const __half scale_h = *reinterpret_cast<const __half*>(qs + kBlockBytes - 2);
        sh[i] = (float)ptq1_trit(qs, e % kBlockElems) * __half2float(scale_h);
    }
    __syncthreads();

    for (int len = 1; len < block; len <<= 1) {
        for (int i = threadIdx.x; i < block / 2; i += blockDim.x) {
            const int lo = ((i / len) * 2 * len) + (i % len);
            const int hi = lo + len;
            const float a = sh[lo], b = sh[hi];
            sh[lo] = a + b;
            sh[hi] = a - b;
        }
        __syncthreads();
    }

    const float norm = rsqrtf((float)block);
    for (int i = threadIdx.x; i < block; i += blockDim.x)
        out[(size_t)row * k + base + i] = __float2bfloat16(sh[i] * norm * (float)sign[base + i]);
}

// Lookup without the rotation, for a table that does not carry one.

__global__ void embedding_ptq1_kernel(const int* __restrict__ tok,
                                      const unsigned char* __restrict__ table,
                                      __nv_bfloat16* __restrict__ out, int k) {
    const int r = blockIdx.y;
    const int row = tok[r];
    const int n_blocks = k / kBlockElems;
    const unsigned char* wrow = table + (size_t)row * n_blocks * kBlockBytes;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += gridDim.x * blockDim.x) {
        const unsigned char* qs = wrow + (size_t)(i / kBlockElems) * kBlockBytes;
        const __half scale_h = *reinterpret_cast<const __half*>(qs + kBlockBytes - 2);
        out[(size_t)r * k + i] =
            __float2bfloat16((float)ptq1_trit(qs, i % kBlockElems) * __half2float(scale_h));
    }
}

template <typename OutT>
void launch_typed(const void* x, const void* w, OutT* y, int n_rows, int k, int batch,
                  cudaStream_t stream) {
    if (n_rows <= 0 || k <= 0 || batch <= 0 || k % kBlockElems != 0) return;
    const auto* xb = reinterpret_cast<const __nv_bfloat16*>(x);
    const auto* wb = reinterpret_cast<const unsigned char*>(w);
    if (launch_dp4a<OutT>(xb, wb, y, n_rows, k, batch, stream)) return;
    const dim3 grid((unsigned)((n_rows + kWarpsPerCta - 1) / kWarpsPerCta), 1u);
    if (batch == 1) {
        gemv_ptq1_kernel<OutT><<<grid, kWarpsPerCta * 32, 0, stream>>>(xb, wb, y, n_rows, k);
        return;
    }
    // Chunked by the widest instantiation rather than templated on every batch: a chunk computes
    // exactly the rows it holds, in the same order, so chunking changes nothing a caller can see.
    // Registers bound the chunk -- acc[] and part[] are both BMAX floats per lane.
    constexpr int kBatchChunk = 8;
    for (int b0 = 0; b0 < batch; b0 += kBatchChunk) {
        const int m = batch - b0 < kBatchChunk ? batch - b0 : kBatchChunk;
        const __nv_bfloat16* xc = xb + (size_t)b0 * k;
        OutT* yc = y + (size_t)b0 * n_rows;
        if (m <= 2)
            gemm_ptq1_kernel<OutT, 2><<<grid, kWarpsPerCta * 32, 0, stream>>>(xc, wb, yc, n_rows, k, m);
        else if (m <= 4)
            gemm_ptq1_kernel<OutT, 4><<<grid, kWarpsPerCta * 32, 0, stream>>>(xc, wb, yc, n_rows, k, m);
        else
            gemm_ptq1_kernel<OutT, 8><<<grid, kWarpsPerCta * 32, 0, stream>>>(xc, wb, yc, n_rows, k, m);
    }
}

}  // namespace

void launch_gemv_ptq1(const void* x_bf16, const void* w_ptq1, void* y_bf16,
                      int n_rows, int k, cudaStream_t stream) {
    launch_typed<__nv_bfloat16>(x_bf16, w_ptq1, reinterpret_cast<__nv_bfloat16*>(y_bf16),
                                n_rows, k, 1, stream);
}

void launch_gemv_ptq1_f32(const void* x_bf16, const void* w_ptq1, float* y_f32,
                          int n_rows, int k, cudaStream_t stream) {
    launch_typed<float>(x_bf16, w_ptq1, y_f32, n_rows, k, 1, stream);
}

void launch_gemm_ptq1_f32(const void* x_bf16, const void* w_ptq1, float* y_f32,
                          int n_rows, int k, int batch, cudaStream_t stream) {
    launch_typed<float>(x_bf16, w_ptq1, y_f32, n_rows, k, batch, stream);
}

void launch_gemm_ptq1(const void* x_bf16, const void* w_ptq1, void* y_bf16,
                      int n_rows, int k, int batch, cudaStream_t stream) {
    launch_typed<__nv_bfloat16>(x_bf16, w_ptq1, reinterpret_cast<__nv_bfloat16*>(y_bf16),
                                n_rows, k, batch, stream);
}

void launch_embedding_ptq1(const int* tokens, const void* table_ptq1, void* out_bf16,
                           int n_tokens, int k, cudaStream_t stream) {
    if (n_tokens <= 0 || k <= 0 || k % kBlockElems != 0) return;
    const int threads = 256;
    const dim3 grid((unsigned)((k + threads - 1) / threads), (unsigned)n_tokens);
    embedding_ptq1_kernel<<<grid, threads, 0, stream>>>(
        tokens, reinterpret_cast<const unsigned char*>(table_ptq1),
        reinterpret_cast<__nv_bfloat16*>(out_bf16), k);
}

void launch_ptq1_rows_unrotate_bf16(const void* w_ptq1, const signed char* sign, void* out_bf16,
                                    int n_rows, int k, int block, cudaStream_t stream) {
    if (n_rows <= 0 || k <= 0 || block <= 0 || k % kBlockElems != 0 || k % block != 0) return;
    const dim3 grid((unsigned)(k / block), (unsigned)n_rows);
    ptq1_rows_unrotate_kernel<<<grid, 256, (size_t)block * sizeof(float), stream>>>(
        reinterpret_cast<const unsigned char*>(w_ptq1), sign,
        reinterpret_cast<__nv_bfloat16*>(out_bf16), k, block);
}

void launch_embedding_ptq1_unrotate(const int* tokens, const void* table_ptq1,
                                    const signed char* sign, void* out_bf16,
                                    int n_tokens, int k, int block, cudaStream_t stream) {
    if (n_tokens <= 0 || k <= 0 || block <= 0 || k % kBlockElems != 0 || k % block != 0) return;
    const dim3 grid((unsigned)(k / block), (unsigned)n_tokens);
    embedding_ptq1_unrotate_kernel<<<grid, 256, (size_t)block * sizeof(float), stream>>>(
        tokens, reinterpret_cast<const unsigned char*>(table_ptq1), sign,
        reinterpret_cast<__nv_bfloat16*>(out_bf16), k, block);
}

}}  // namespace sparkinfer::kernels
