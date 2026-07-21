// Direct-4-bit ("MMQ") grouped MoE expert GEMM for batched prefill — see prefill_moe_mmq.h.
//
// Replaces the "dequant whole expert to int8, then int8 tensor-core GEMM" pair with a single
// dp4a matmul that reads the GGUF Q4_K/Q5_K super-blocks directly, the way decode's MMVQ does.
// At the small token counts of the 128/512 prefill contexts each expert's GEMM is tiny and the
// layer is bound by the weight-DRAM traffic of the dequant materialize; folding the weight read
// into the matmul removes the int8 round-trip (~5x less weight traffic).
//
// The Q4_K/Q5_K per-super-block dp4a dots and the Q8_1 quantizer below are the standard
// llama.cpp-faithful routines (byte-identical to the ones already in expert_ffn_q4k.cu /
// gemv.cu that the decode MMVQ path uses); they are duplicated here so this stays a single
// self-contained translation unit. Weight blocks are staged once into shared memory per
// super-block and reused across the tile's tokens, so each 4-bit block is read from DRAM once.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "sparkinfer/kernels/prefill_moe_mmq.h"

namespace sparkinfer {
namespace kernels {
namespace {

// ---- GGUF super-block layouts (llama.cpp) --------------------------------------------------
struct mmq_q8_1 { __half2 ds; signed char qs[32]; };                                     // 36 B / 32
struct mmq_q4_K { __half2 dm; unsigned char scales[12]; unsigned char qs[128]; };        // 144 B / 256
struct mmq_q5_K { __half2 dm; unsigned char scales[12]; unsigned char qh[32];
                              unsigned char qs[128]; };                                   // 176 B / 256

// Q4_K x Q8_1 dp4a for one 256-super-block at position iqs (0,2,..,30). Faithful to
// llama.cpp vec_dot_q4_K_q8_1 (matches the decode si_vec_dot_q4_K).
__device__ __forceinline__ float mmq_dot_q4_K(const mmq_q4_K* bq4, const mmq_q8_1* bq8_1, int iqs) {
    int v[2], u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* q4 = (const int*)(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];
    const unsigned short* scales = (const unsigned short*)bq4->scales;
    unsigned short aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j] & 0xc0c0) >> 2);
    }
    const unsigned char* sc = (const unsigned char*)aux;
    const unsigned char* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const mmq_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
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

// Q5_K x Q8_1 dp4a — Q4_K plus one high bit per quant (matches decode si_vec_dot_q5_K).
__device__ __forceinline__ float mmq_dot_q5_K(const mmq_q5_K* bq5, const mmq_q8_1* bq8_1, int iqs) {
    int v[2], u[4];
    float d8[2];
    const int L = iqs >> 1;
    const int bq8_offset = 2 * (L / 4);
    const int* q4 = (const int*)(bq5->qs + 16 * bq8_offset + 4 * (L % 4));
    v[0] = q4[0];
    v[1] = q4[4];
    const int* qhp = (const int*)(bq5->qh + 4 * (L % 4));
    const int qh0 = qhp[0], qh1 = qhp[4];
    const unsigned short* scales = (const unsigned short*)bq5->scales;
    unsigned short aux[2];
    const int j = bq8_offset / 2;
    if (j < 2) {
        aux[0] = scales[j] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j] & 0xc0c0) >> 2);
    }
    const unsigned char* sc = (const unsigned char*)aux;
    const unsigned char* m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const mmq_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*)bq8i->qs + (L % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.f, sumf_m = 0.f;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int hs = bq8_offset + i;
        const int v0i = ((v[0] >> (4 * i)) & 0x0F0F0F0F) | (((qh0 >> hs) & 0x01010101) << 4);
        const int v1i = ((v[1] >> (4 * i)) & 0x0F0F0F0F) | (((qh1 >> hs) & 0x01010101) << 4);
        const int dot1 = __dp4a(v0i, u[2 * i], __dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i], __dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    float2 dm5f = __half22float2(bq5->dm);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

template <int QT> struct mmq_wtraits;
template <> struct mmq_wtraits<12> {
    using B = mmq_q4_K;
    static constexpr int BS = 144;
    static __device__ __forceinline__ float dot(const B* w, const mmq_q8_1* a, int iqs) {
        return mmq_dot_q4_K(w, a, iqs);
    }
};
template <> struct mmq_wtraits<13> {
    using B = mmq_q5_K;
    static constexpr int BS = 176;
    static __device__ __forceinline__ float dot(const B* w, const mmq_q8_1* a, int iqs) {
        return mmq_dot_q5_K(w, a, iqs);
    }
};

// bf16 -> Q8_1 (one 32-value block per warp), interleaved {ds, qs[32]}.
__global__ void mmq_quantize_rows_q8_1_kernel(const __nv_bfloat16* __restrict__ x,
                                              mmq_q8_1* __restrict__ y, long total_blocks) {
    const int warpsPB = blockDim.x >> 5;
    const long ib = (long)blockIdx.x * warpsPB + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (ib >= total_blocks) return;
    float xv = __bfloat162float(x[ib * 32 + lane]), a = fabsf(xv);
#pragma unroll
    for (int mm = 16; mm > 0; mm >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, mm));
    float d = a / 127.0f;
    int qi = (a == 0.0f) ? 0 : (int)roundf(xv / d);
    y[ib].qs[lane] = (signed char)qi;
    int s = qi;
#pragma unroll
    for (int mm = 16; mm > 0; mm >>= 1) s += __shfl_xor_sync(0xffffffffu, s, mm);
    if (lane == 0) y[ib].ds = __floats2half2_rn(d, d * (float)s);
}

// Grouped MoE GEMM. One block == one (expert, m-tile) x BN output columns. The expert's
// weight super-block is staged into shared memory once and dp4a'd against all BM tokens'
// Q8_1 activations, so each 4-bit block is read from DRAM once (weight-amortized).
constexpr int MMQ_BM = 16;
constexpr int MMQ_BN = 32;

template <int QT, bool A_INDIRECT, bool C_SCATTER>
__global__ void __launch_bounds__(MMQ_BM* MMQ_BN)
mmq_moe_gemm_kernel(const mmq_q8_1* __restrict__ A_q8, const unsigned char* __restrict__ W_q,
                    const int* __restrict__ pair_tok, const float* __restrict__ pair_w,
                    const int* __restrict__ offsets, const int* __restrict__ tilemap,
                    const int* __restrict__ d_ntiles, __nv_bfloat16* __restrict__ C,
                    float* __restrict__ out_f32, int N, int K, int e_base) {
    using WT = mmq_wtraits<QT>;
    using WB = typename WT::B;
    constexpr int BS = WT::BS;
    constexpr int BSW = BS / 4;  // ints per weight super-block

    const int tile = blockIdx.y;
    if (tile >= d_ntiles[0]) return;
    const int e = tilemap[2 * tile];
    const int mt = tilemap[2 * tile + 1];
    const int p0 = offsets[e] + mt * MMQ_BM;
    const int cnt = offsets[e + 1] - offsets[e];
    const int M = min(MMQ_BM, cnt - mt * MMQ_BM);
    const int n0 = blockIdx.x * MMQ_BN;
    const int nsb = K >> 8;   // 256-super-blocks along K
    const int kblk = K >> 5;  // Q8_1 blocks along K

    __shared__ int s_tok[MMQ_BM];
    __shared__ WB Ws[MMQ_BN];
    __shared__ mmq_q8_1 As[MMQ_BM][8];

    const int tid = threadIdx.x;
    const int m = tid / MMQ_BN;
    const int n = tid % MMQ_BN;
    const int gn = n0 + n;

    if (tid < MMQ_BM)
        s_tok[tid] = (tid < M) ? (A_INDIRECT ? pair_tok[p0 + tid] : (p0 + tid)) : -1;
    __syncthreads();

    const size_t we_base = (size_t)(e - e_base) * (size_t)N;
    float acc = 0.f;
    for (int sb = 0; sb < nsb; sb++) {
        // stage BN weight super-blocks (one per output column of this tile)
        for (int i = tid; i < MMQ_BN * BSW; i += blockDim.x) {
            const int col = i / BSW, w = i % BSW;
            const int gncol = n0 + col;
            const int* src = (const int*)(W_q + ((we_base + gncol) * (size_t)nsb + sb) * BS);
            ((int*)&Ws[col])[w] = (gncol < N) ? src[w] : 0;
        }
        // stage BM tokens' 8 Q8_1 blocks for this super-block (9 ints per 36 B block)
        for (int i = tid; i < MMQ_BM * 8 * 9; i += blockDim.x) {
            const int r = i / 72, rem = i % 72, blk = rem / 9, w = rem % 9;
            const int arow = s_tok[r];
            const int* src = (const int*)(A_q8 + ((size_t)max(arow, 0) * kblk + (size_t)sb * 8 + blk));
            ((int*)&As[r][blk])[w] = (arow >= 0) ? src[w] : 0;
        }
        __syncthreads();
        if (m < M && gn < N) {
            const WB* wblk = &Ws[n];
            const mmq_q8_1* a = &As[m][0];
#pragma unroll
            for (int iqs = 0; iqs < 32; iqs += 2)
                acc += WT::dot(wblk, a, iqs);
        }
        __syncthreads();
    }

    if (m < M && gn < N) {
        const int p = p0 + m;
        if (C_SCATTER)
            atomicAdd(&out_f32[(size_t)pair_tok[p] * N + gn], acc * pair_w[p]);
        else
            C[(size_t)p * N + gn] = __float2bfloat16(acc);
    }
}

template <int QT>
inline void dispatch_mmq(const mmq_q8_1* A, const unsigned char* W, const int* pair_tok,
                         const float* pair_w, const int* offsets, const int* tilemap,
                         const int* d_ntiles, __nv_bfloat16* C, float* out_f32, int N, int K,
                         int max_tiles, int e_base, bool a_indirect, bool c_scatter,
                         cudaStream_t st) {
    dim3 grid((N + MMQ_BN - 1) / MMQ_BN, max_tiles);
    const int block = MMQ_BM * MMQ_BN;
    if (a_indirect && !c_scatter)
        mmq_moe_gemm_kernel<QT, true, false><<<grid, block, 0, st>>>(
            A, W, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N, K, e_base);
    else if (!a_indirect && c_scatter)
        mmq_moe_gemm_kernel<QT, false, true><<<grid, block, 0, st>>>(
            A, W, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N, K, e_base);
    else if (a_indirect && c_scatter)
        mmq_moe_gemm_kernel<QT, true, true><<<grid, block, 0, st>>>(
            A, W, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N, K, e_base);
    else
        mmq_moe_gemm_kernel<QT, false, false><<<grid, block, 0, st>>>(
            A, W, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N, K, e_base);
}

}  // namespace

void launch_prefill_quantize_rows_q8_1(const void* x_bf16, void* y_q8_1, int rows, int K,
                                       cudaStream_t stream) {
    const long total = (long)rows * (K >> 5);
    if (total <= 0) return;
    const int WPB = 8;
    const long blocks = (total + WPB - 1) / WPB;
    mmq_quantize_rows_q8_1_kernel<<<(unsigned)blocks, WPB * 32, 0, stream>>>(
        (const __nv_bfloat16*)x_bf16, (mmq_q8_1*)y_q8_1, total);
}

bool launch_pfm_moe_mmq(const void* A_q8_1, const void* W_q, int qtype, const int* pair_tok,
                        const float* pair_w, const int* offsets, const int* tilemap,
                        const int* d_ntiles, void* C_bf16, float* out_f32, int N_out, int K,
                        int max_tiles, int e_base, bool a_indirect, bool c_scatter,
                        cudaStream_t stream) {
    const mmq_q8_1* A = (const mmq_q8_1*)A_q8_1;
    const unsigned char* W = (const unsigned char*)W_q;
    __nv_bfloat16* C = (__nv_bfloat16*)C_bf16;
    if (qtype == 12)
        dispatch_mmq<12>(A, W, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K,
                         max_tiles, e_base, a_indirect, c_scatter, stream);
    else if (qtype == 13)
        dispatch_mmq<13>(A, W, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K,
                         max_tiles, e_base, a_indirect, c_scatter, stream);
    else
        return false;
    return true;
}

}  // namespace kernels
}  // namespace sparkinfer
