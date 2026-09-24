// GEMV against PTQ1_0 ternary weights, read in their stored 28-byte blocks.
// See sparkinfer/kernels/ternary.h for the format and for the basis the activation must be in.
#include "sparkinfer/kernels/ternary.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

// ----- Tensor-core GEMV/GEMM -----
//
// The two kernels above give each weight row to one warp and walk its blocks one at a time: 28
// bytes in flight per warp, a staging barrier per block, and a trit decoded per lane per multiply
// through a byte load, two multiplies and an int->float convert. That reads the weights at
// 150-280 GB/s, a sixth of the bus -- slower than the folded Q4_K copy of the same tensor despite
// reading 2.6x fewer bytes, which is why the native head measured SLOWER than folded. The batched
// form re-decodes the block per activation, so eight activations cost eight times one.
//
// Here the trits go into mma.m16n8k16 as the bf16 A operand -- exactly -1, 0 or +1 -- and the
// activations are the B operand, so a batch of up to eight costs one MMA per step and the multiply
// is exact. A quad of lanes owns a 16-row tile's block: lane c takes carrier bytes 6c..6c+5 (three
// pairs, five digits each) and digit c of the four-trit pair 24/25, sixteen k-pairs, two per step.
// The trits come out two at a time: a carrier pair sits in the two 16-bit halves of one word, so
// ONE multiply by 3^m digit-shifts both (255*81 fits a half), (q*3)>>8 leaves each digit alone in
// a byte, a byte permute drops it into the mantissa of bf16 128.0, and subtracting 129 leaves the
// trit. Both trits of a pair are ADJACENT positions in the block, so each B register is one 32-bit
// load of the activation as it lies -- no staging, no permuted copy.
//
// One kernel serves one activation and many, and a row's result does not depend on which: an MMA
// output element reads only its own row and column, the blocks fold in ascending order, and the
// split-K parts are summed in warp order with the split chosen from the shape alone. So AR decode
// (one activation, the other seven columns zero) and a packed step are bit-identical per row --
// which is what lets both read the same ternary weights (gemv_ptq1_gpu_test checks it).
constexpr int kMmaSplit = 4;          // warps per CTA, each a quarter of the blocks
constexpr int kMmaMaxBatch = 16;      // activations per launch: two n-tiles

__device__ __forceinline__ unsigned int ptq1_trit_pair_bf16(unsigned int pair, unsigned int p3) {
    const unsigned int e = ((pair * p3) & 0x00FF00FFu) * 3u;          // digits in bytes 1 and 3
    unsigned int v = __byte_perm(e, 0x43u, 0x4341u);                    // bf16 0x43dd = 128 + d
    __nv_bfloat162 h = *reinterpret_cast<__nv_bfloat162*>(&v);
    h = __hsub2(h, __floats2bfloat162_rn(129.f, 129.f));                // d - 1, exact
    return *reinterpret_cast<unsigned int*>(&h);
}

__host__ __device__ constexpr unsigned int ptq1_pow3(int m) {
    return m == 0 ? 1u : m == 1 ? 3u : m == 2 ? 9u : m == 3 ? 27u : 81u;
}

// Lane c's k-pair e (0..15) as a position in the 128-block: e < 15 is carrier pair e / 5 of
// bytes 6c..6c+5 at digit e % 5, e == 15 the four-trit pair at digit c.
__device__ __forceinline__ int ptq1_pair_pos(int c, int e) {
    if (e == 15) return 120 + 2 * c;
    const int beta = 6 * c + 2 * (e / 5), m = e % 5;
    return beta < 16 ? 16 * m + beta : 80 + 8 * m + (beta - 16);
}

// RT 16-row tiles per warp (more tiles share each activation load; it does not change a row's
// arithmetic), NT n-tiles of eight activations. `split` warps per CTA divide the blocks.
template <typename OutT, int NT, int RT>
__global__ void __launch_bounds__(32 * kMmaSplit)
gemm_ptq1_mma_kernel(const __nv_bfloat16* __restrict__ x, const unsigned char* __restrict__ w,
                     OutT* __restrict__ y, int n_rows, int k, int batch, int split) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, c = lane & 3;
    const int nb = k / kBlockElems;
    const int r0 = blockIdx.x * 16 * RT;
    const int b_lo = (int)((long)nb * warp / split), b_hi = (int)((long)nb * (warp + 1) / split);
    const int wlo = (6 * c) >> 2;                   // first word holding bytes 6c..6c+5
    const unsigned int* x32 = reinterpret_cast<const unsigned int*>(x);
    const int k2 = k >> 1;

    int qpos[16];
#pragma unroll
    for (int e = 0; e < 16; ++e) qpos[e] = ptq1_pair_pos(c, e) >> 1;
    const unsigned int p3c = ptq1_pow3(c);
    unsigned int sel[3];
    int wsel[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int beta = 6 * c + 2 * i;             // even, so a pair never straddles a word
        wsel[i] = (beta >> 2) - wlo;
        sel[i] = (beta & 3) ? 0x4342u : 0x4140u;    // its two bytes into the two halves
    }
    int rows[2 * RT];
    bool live[2 * RT];
#pragma unroll
    for (int i = 0; i < 2 * RT; ++i) { rows[i] = r0 + 8 * i + g; live[i] = rows[i] < n_rows; }

    float acc[RT][NT][4];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[r][t][i] = 0.f;

    // Two blocks of weight words in flight ahead of the one being decoded.
    constexpr int kAhead = 2;
    unsigned int ahead[kAhead][2 * RT][3];
    auto fetch = [&](int b, unsigned int (*dst)[3]) {
#pragma unroll
        for (int i = 0; i < 2 * RT; ++i) {
            dst[i][0] = dst[i][1] = dst[i][2] = 0;
            if (b < b_hi && live[i]) {
                const unsigned int* p =
                    reinterpret_cast<const unsigned int*>(w + ((size_t)rows[i] * nb + b) * kBlockBytes);
                dst[i][0] = __ldg(p + wlo);
                dst[i][1] = __ldg(p + wlo + 1);
                dst[i][2] = __ldg(p + 6);           // carriers 24/25 and the scale
            }
        }
    };
#pragma unroll
    for (int a = 0; a < kAhead; ++a) fetch(b_lo + a, ahead[a]);

    for (int b = b_lo; b < b_hi; ++b) {
        unsigned int W[2 * RT][3];
#pragma unroll
        for (int i = 0; i < 2 * RT; ++i)
#pragma unroll
            for (int j = 0; j < 3; ++j) W[i][j] = ahead[0][i][j];
#pragma unroll
        for (int a = 0; a + 1 < kAhead; ++a)
#pragma unroll
            for (int i = 0; i < 2 * RT; ++i)
#pragma unroll
                for (int j = 0; j < 3; ++j) ahead[a][i][j] = ahead[a + 1][i][j];
        fetch(b + kAhead, ahead[kAhead - 1]);

        unsigned int pairs[2 * RT][4];
#pragma unroll
        for (int i = 0; i < 2 * RT; ++i) {
#pragma unroll
            for (int j = 0; j < 3; ++j) pairs[i][j] = __byte_perm(wsel[j] ? W[i][1] : W[i][0], 0u, sel[j]);
            pairs[i][3] = __byte_perm(W[i][2], 0u, 0x4140u);
        }
        float cb[RT][NT][4];
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int t = 0; t < NT; ++t)
#pragma unroll
                for (int i = 0; i < 4; ++i) cb[r][t][i] = 0.f;
        const unsigned int* xb = x32 + (size_t)b * (kBlockElems / 2);
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int e0 = 2 * s, e1 = 2 * s + 1;   // k-pairs c and c+4 of this step
            const int pi0 = e0 == 15 ? 3 : e0 / 5, pi1 = e1 == 15 ? 3 : e1 / 5;
            const unsigned int m0 = e0 == 15 ? p3c : ptq1_pow3(e0 % 5);
            const unsigned int m1 = e1 == 15 ? p3c : ptq1_pow3(e1 % 5);
            unsigned int bf[NT][2];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const int n = 8 * t + g;
                bf[t][0] = bf[t][1] = 0;
                if (n < batch) {
                    bf[t][0] = __ldg(xb + (size_t)n * k2 + qpos[e0]);
                    bf[t][1] = __ldg(xb + (size_t)n * k2 + qpos[e1]);
                }
            }
#pragma unroll
            for (int r = 0; r < RT; ++r) {
                const unsigned int a0 = ptq1_trit_pair_bf16(pairs[2 * r][pi0], m0);
                const unsigned int a1 = ptq1_trit_pair_bf16(pairs[2 * r + 1][pi0], m0);
                const unsigned int a2 = ptq1_trit_pair_bf16(pairs[2 * r][pi1], m1);
                const unsigned int a3 = ptq1_trit_pair_bf16(pairs[2 * r + 1][pi1], m1);
#pragma unroll
                for (int t = 0; t < NT; ++t)
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(cb[r][t][0]), "+f"(cb[r][t][1]), "+f"(cb[r][t][2]), "+f"(cb[r][t][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bf[t][0]), "r"(bf[t][1]));
            }
        }
        // The block's scale, once per row per block.
#pragma unroll
        for (int r = 0; r < RT; ++r) {
            const float sg = __half2float(__ushort_as_half((unsigned short)(W[2 * r][2] >> 16)));
            const float sh = __half2float(__ushort_as_half((unsigned short)(W[2 * r + 1][2] >> 16)));
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                acc[r][t][0] = fmaf(sg, cb[r][t][0], acc[r][t][0]);
                acc[r][t][1] = fmaf(sg, cb[r][t][1], acc[r][t][1]);
                acc[r][t][2] = fmaf(sh, cb[r][t][2], acc[r][t][2]);
                acc[r][t][3] = fmaf(sh, cb[r][t][3], acc[r][t][3]);
            }
        }
    }

    __shared__ float part[kMmaSplit][RT * NT * 4][32];
    if (split > 1) {
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int t = 0; t < NT; ++t)
#pragma unroll
                for (int i = 0; i < 4; ++i) part[warp][(r * NT + t) * 4 + i][lane] = acc[r][t][i];
        __syncthreads();
        if (warp != 0) return;
        for (int s2 = 1; s2 < split; ++s2)
#pragma unroll
            for (int r = 0; r < RT; ++r)
#pragma unroll
                for (int t = 0; t < NT; ++t)
#pragma unroll
                    for (int i = 0; i < 4; ++i) acc[r][t][i] += part[s2][(r * NT + t) * 4 + i][lane];
    }
    // c0/c1 are row g, activations 2c and 2c+1; c2/c3 the same for row g+8.
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int n0 = 8 * t + 2 * c;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                if (!live[2 * r + h]) continue;
                const int row = rows[2 * r + h];
                if (n0 < batch)     store_out<OutT>(y + (size_t)n0 * n_rows, row, acc[r][t][2 * h]);
                if (n0 + 1 < batch) store_out<OutT>(y + (size_t)(n0 + 1) * n_rows, row, acc[r][t][2 * h + 1]);
            }
        }
}

bool ptq1_legacy() {
    static const bool v = [] {
        const char* e = getenv("SPARKINFER_PTQ1_GEMV_LEGACY");
        return e && e[0] == '1';
    }();
    return v;
}

template <typename OutT>
void launch_mma(const __nv_bfloat16* x, const unsigned char* w, OutT* y, int n_rows, int k,
                int batch, cudaStream_t stream) {
    // From the shape alone: the split sets the order the parts are summed in.
    const int split = (k / kBlockElems) >= 4 * kMmaSplit ? kMmaSplit : 1;
    for (int b0 = 0; b0 < batch; b0 += kMmaMaxBatch) {
        const int m = batch - b0 < kMmaMaxBatch ? batch - b0 : kMmaMaxBatch;
        const __nv_bfloat16* xc = x + (size_t)b0 * k;
        OutT* yc = y + (size_t)b0 * n_rows;
        if (m <= 8) {
            const dim3 grid((unsigned)((n_rows + 15) / 16));
            gemm_ptq1_mma_kernel<OutT, 1, 1><<<grid, 32 * split, 0, stream>>>(xc, w, yc, n_rows, k, m, split);
        } else {
            // Two row tiles share each activation load once there are two n-tiles to feed.
            const dim3 grid((unsigned)((n_rows + 31) / 32));
            gemm_ptq1_mma_kernel<OutT, 2, 2><<<grid, 32 * split, 0, stream>>>(xc, w, yc, n_rows, k, m, split);
        }
    }
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
    if (!ptq1_legacy()) {
        launch_mma<OutT>(xb, wb, y, n_rows, k, batch, stream);
        return;
    }
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
