// fp8 (e4m3) tensor-core GEMM for Qwythos batched prefill at long context (see prefill_fp8.h).
//
// C[M,N] = A[M,K] @ W^T, W dequantized bf16 [N,K]. fp8 x fp8 -> fp16 accumulate, with a periodic
// flush of the fp16 partials into an fp32 accumulator, then the dequant (per-row sx[m] * per-channel
// sw[n]) folded into the bf16 store.
//
// Why fp8 here: above ~96k the Gated-DeltaNet recurrence (near-1 decay) amplifies per-row int8
// activation-quant error across the sequence, so the int8 projection path diverges (128k top1 ~0.31).
// The dense long-ctx fallback therefore runs the GDN projections in bf16 -- ~half the int8 MAC rate.
// e4m3 keeps a floating range (uniform *relative* error, unlike int8's uniform absolute step), so it
// holds the recurrence far closer to bf16 fidelity than int8 while running on the fp8 tensor cores.
//
// Rate note (GeForce Blackwell / sm_120): fp8 with *fp32* accumulate is throttled to ~half, but fp8
// with *fp16* accumulate runs at ~2x bf16 (the same op-bandwidth-bound rate the int8 projections
// hit). K=4096 overflows a raw fp16 accumulator, so the operands are scaled to amax->FP8_TGT and the
// fp16 partials are flushed to fp32 every FP8_FLUSH BK-tiles (see below).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cstdlib>
#include "sparkinfer/kernels/prefill_fp8.h"

namespace sparkinfer { namespace kernels {

namespace {
constexpr int FP8_BM = 128;
constexpr int FP8_BN = 128;
constexpr int FP8_BK = 64;          // 4 x 16B chunks per row
constexpr int FP8_MFRAG = 2;        // 32 rows per warp / 16
constexpr int FP8_NFRAG = 8;        // 64 cols per warp / 8
// The fp16 partials are flushed to the fp32 accumulator every FP8_FLUSH BK-tiles (not every tile),
// which keeps the fp32-accumulate precision while bounding how large the fp16 running sum can grow.
// Overflow bound: 2*FP8_FLUSH k32 mma steps accumulate at most 2*FP8_FLUSH*32*FP8_TGT^2 in fp16,
// which must stay < 65504. FP8_TGT=2, FP8_FLUSH=8 -> 2*8*32*4 = 2048, huge headroom. (Empirically a
// larger FP8_TGT loses fidelity here well before that bound; +-2 with periodic fp32 flush tracks the
// bf16 GDN path most closely.)
constexpr int FP8_FLUSH = 8;
// Operand target amax. e4m3 with values in +-2 keeps a 2/2^-9 ~= 1024:1 dynamic range (still ~8x
// finer than int8's 127:1 for the small activations the recurrence is sensitive to; e4m3's 3-bit
// mantissa gives the same relative step at any scale, so the target sets range, not per-value error).
constexpr float FP8_TGT = 2.0f;

__device__ __forceinline__ void fp8_cp16(void* dst, const void* src, bool pred) {
    if (pred) __pipeline_memcpy_async(dst, src, 16);
    else      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
}

// XOR swizzle at 16B granularity: chunk c of row r lives at chunk (c ^ (r & 3)) -- rows 0..3 (the
// stride the 4B operand loads walk) land on disjoint banks. Same scheme as the int8 GEMM.
// XOR swizzle at 16B granularity: chunk c of row r lives at chunk (c ^ key(r)).
//
// SWZ8=false is the legacy key (row & 3). The smem row stride is FP8_BK = 64 B = 16 banks, so
// row bit 0 ALREADY flips the 4-bank group by 16; keying the XOR on row bits 0-1 therefore
// yields only 4 distinct (parity, chunk) states across the 8 rows one ldmatrix.m8n8 phase
// reads, and rows d and d+4 land on the SAME 4 banks -- every ldmatrix.x4 phase is 2-way
// conflicted. SWZ8=true keys on ((row >> 1) & 3) instead, which composes with the stride's own
// parity flip so the 8 rows of a phase cover 8 disjoint 4-bank groups: conflict-free.
//
// Both are bijections on the chunk index for a fixed row, and writer and reader apply the same
// function of (k, row), so the bytes staged and the registers loaded are IDENTICAL either way.
template <bool SWZ8>
__device__ __forceinline__ int fp8_swz(int k, int row) {
    return (((k >> 4) ^ (SWZ8 ? ((row >> 1) & 3) : (row & 3))) << 4) | (k & 15);
}

// Default ON; SPARKINFER_PREFILL_FP8_SWZ8=0 restores the legacy key for A/B.
static bool fp8_use_swz8() {
    static const bool v = [] {
        const char* e = getenv("SPARKINFER_PREFILL_FP8_SWZ8");
        return !e || e[0] != '0';
    }();
    return v;
}

// ldmatrix.x4 operand staging, same as the int8 GEMM: one instruction per four 8x8 tiles instead
// of four scalar lds.32, so fragment loads stop competing with the mma issue rate. e4m3 and s8 are
// both 1-byte k32 operand types, so the m16n8k32 fragment layout (and this mapping) is identical.
__device__ __forceinline__ void fp8_ldm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                           const __nv_fp8_e4m3* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

// Per-row symmetric fp8 quantize (amax -> FP8_TGT), one warp per row. Input bf16.
__global__ void pf_quantize_rows_fp8_kernel(const __nv_bfloat16* __restrict__ x,
                                            __nv_fp8_e4m3* __restrict__ q,
                                            float* __restrict__ scale, int rows, int cols) {
    const int r = blockIdx.x, lane = threadIdx.x;
    if (r >= rows) return;
    float amax = 0.f;
    for (int c = lane; c < cols; c += 32) amax = fmaxf(amax, fabsf(__bfloat162float(x[(size_t)r * cols + c])));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float d = (amax == 0.f) ? 1.f : (amax / FP8_TGT);
    if (lane == 0) scale[r] = d;
    for (int c = lane; c < cols; c += 32)
        q[(size_t)r * cols + c] = __nv_fp8_e4m3(__bfloat162float(x[(size_t)r * cols + c]) / d);
}

// The 2 in __launch_bounds__ mirrors the int8 kernel: unbounded, nvcc picks a register count that
// keeps only one block per SM resident.
// SPLITK partitions the K loop across blockIdx.z (same occupancy fix as launch_prefill_gemm_i8_splitk)
// and atomicAdds the unscaled fp32 tile into P[M,N]; a separate epilogue applies sx*sw.
template <bool SPLITK, bool SWZ8>
__global__ __launch_bounds__(256, 2) void pf_gemm_fp8_kernel(
        const __nv_fp8_e4m3* __restrict__ A, const __nv_fp8_e4m3* __restrict__ W,
        const float* __restrict__ sx, const float* __restrict__ sw,
        __nv_bfloat16* __restrict__ C, float* __restrict__ P,
        int M, int N, int K, int ktiles) {
    __shared__ __nv_fp8_e4m3 As[2][FP8_BM][FP8_BK];
    __shared__ __nv_fp8_e4m3 Bs[2][FP8_BN][FP8_BK];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 2;                       // 0..7
    const int tig  = lane & 3;                        // thread-in-group
    const int sub  = lane >> 3;                       // ldmatrix tile this thread addresses (0..3)
    const int lrow = lane & 7;                        // row within that tile
    const int wm   = warp & 3;                        // rows [wm*32, +32)
    const int wn   = warp >> 2;                       // cols [wn*64, +64)
    const int m0   = blockIdx.y * FP8_BM;
    const int n0   = blockIdx.x * FP8_BN;
    const int nk   = (K + FP8_BK - 1) / FP8_BK;
    int t0 = 0, t1 = nk;
    if (SPLITK) {
        t0 = blockIdx.z * ktiles;
        t1 = t0 + ktiles;
        if (t1 > nk) t1 = nk;
        if (t0 >= t1) return;
    }

    float acc[FP8_MFRAG][FP8_NFRAG][4];
    #pragma unroll
    for (int i = 0; i < FP8_MFRAG; i++)
        #pragma unroll
        for (int j = 0; j < FP8_NFRAG; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

    // 128 rows x 64B = 512 16B chunks per tile; 256 threads stage 2 A-chunks + 2 B-chunks each.
    auto stage = [&](int buf, int k0) {
        #pragma unroll
        for (int s = tid; s < 512; s += 256) {
            const int r = s >> 2, c = s & 3, k = c << 4;
            const int gm = m0 + r, gn = n0 + r, gk = k0 + k;
            fp8_cp16(&As[buf][r][fp8_swz<SWZ8>(k, r)], &A[(size_t)gm * K + gk], gm < M && gk < K);
            fp8_cp16(&Bs[buf][r][fp8_swz<SWZ8>(k, r)], &W[(size_t)gn * K + gk], gn < N && gk < K);
        }
        __pipeline_commit();
    };

    // fp16 partials, reset after each flush (every FP8_FLUSH BK-tiles) to bound the running sum.
    unsigned h[FP8_MFRAG][FP8_NFRAG][2];
    #pragma unroll
    for (int i = 0; i < FP8_MFRAG; i++)
        #pragma unroll
        for (int j = 0; j < FP8_NFRAG; j++) { h[i][j][0] = 0u; h[i][j][1] = 0u; }

    stage(0, t0 * FP8_BK);
    int buf = 0;
    for (int t = t0; t < t1; t++) {
        if (t + 1 < t1) stage(buf ^ 1, (t + 1) * FP8_BK);
        __pipeline_wait_prior(t + 1 < t1 ? 1 : 0);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < FP8_BK; kk += 32) {
            unsigned af[FP8_MFRAG][4], bf[FP8_NFRAG][2];
            // A fragment i: tiles {rows lo,k0} {rows hi,k0} {rows lo,k16} {rows hi,k16} -> af[i][0..3]
            #pragma unroll
            for (int i = 0; i < FP8_MFRAG; i++) {
                const int row = wm * 32 + i * 16 + (sub & 1) * 8 + lrow;
                fp8_ldm_x4(af[i][0], af[i][1], af[i][2], af[i][3],
                           &As[buf][row][fp8_swz<SWZ8>(kk + (sub >> 1) * 16, row)]);
            }
            // B pair (j, j+1): tiles {cols j,k0} {cols j,k16} {cols j+1,k0} {cols j+1,k16}
            #pragma unroll
            for (int jp = 0; jp < FP8_NFRAG; jp += 2) {
                const int col = wn * 64 + (jp + (sub >> 1)) * 8 + lrow;
                fp8_ldm_x4(bf[jp][0], bf[jp][1], bf[jp + 1][0], bf[jp + 1][1],
                           &Bs[buf][col][fp8_swz<SWZ8>(kk + (sub & 1) * 16, col)]);
            }
            #pragma unroll
            for (int i = 0; i < FP8_MFRAG; i++)
                #pragma unroll
                for (int j = 0; j < FP8_NFRAG; j++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                        : "+r"(h[i][j][0]), "+r"(h[i][j][1])
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]),
                          "r"(bf[j][0]), "r"(bf[j][1]));
        }
        // flush fp16 partials into the fp32 accumulator every FP8_FLUSH tiles (and on the last one)
        if ((t % FP8_FLUSH) == FP8_FLUSH - 1 || t == t1 - 1) {
            #pragma unroll
            for (int i = 0; i < FP8_MFRAG; i++)
                #pragma unroll
                for (int j = 0; j < FP8_NFRAG; j++) {
                    const __half2 p0 = *reinterpret_cast<__half2*>(&h[i][j][0]);
                    const __half2 p1 = *reinterpret_cast<__half2*>(&h[i][j][1]);
                    acc[i][j][0] += __half2float(p0.x);
                    acc[i][j][1] += __half2float(p0.y);
                    acc[i][j][2] += __half2float(p1.x);
                    acc[i][j][3] += __half2float(p1.y);
                    h[i][j][0] = 0u; h[i][j][1] = 0u;
                }
        }
        __syncthreads();
        buf ^= 1;
    }

    if (SPLITK) {
        #pragma unroll
        for (int i = 0; i < FP8_MFRAG; i++) {
            #pragma unroll
            for (int j = 0; j < FP8_NFRAG; j++) {
                const int gn = n0 + wn * 64 + j * 8 + tig * 2;
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int gm = m0 + wm * 32 + i * 16 + grp + (e >> 1) * 8;
                    const int cn = gn + (e & 1);
                    if (gm < M && cn < N) atomicAdd(&P[(size_t)gm * N + cn], acc[i][j][e]);
                }
            }
        }
        return;
    }

    // Registers straight to global: c0/c1 (and c2/c3) are adjacent columns -> one bf16x2 store each.
    #pragma unroll
    for (int i = 0; i < FP8_MFRAG; i++) {
        #pragma unroll
        for (int j = 0; j < FP8_NFRAG; j++) {
            const int gn = n0 + wn * 64 + j * 8 + tig * 2;
            if (gn + 1 >= N) {                        // tail: scalar path
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int gm = m0 + wm * 32 + i * 16 + grp + (e >> 1) * 8;
                    const int cn = gn + (e & 1);
                    if (gm < M && cn < N)
                        C[(size_t)gm * N + cn] = __float2bfloat16(acc[i][j][e] * sx[gm] * sw[cn]);
                }
                continue;
            }
            const float w0 = sw[gn], w1 = sw[gn + 1];
            #pragma unroll
            for (int h2 = 0; h2 < 2; h2++) {
                const int gm = m0 + wm * 32 + i * 16 + grp + h2 * 8;
                if (gm >= M) continue;
                const float s = sx[gm];
                const __nv_bfloat162 v = __floats2bfloat162_rn(acc[i][j][h2 * 2] * s * w0,
                                                               acc[i][j][h2 * 2 + 1] * s * w1);
                *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)gm * N + gn]) = v;
            }
        }
    }
}
} // namespace

__global__ void pf_fp8_wscales_bf16_kernel(const __nv_bfloat16* __restrict__ s,
                                           float* __restrict__ sw, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sw[i] = __bfloat162float(s[i]);
}

void launch_prefill_fp8_wscales_bf16(const void* scale_bf16, float* sw, int n,
                                     cudaStream_t stream) {
    if (n <= 0) return;
    pf_fp8_wscales_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(scale_bf16), sw, n);
}

void launch_prefill_quantize_rows_fp8(const void* x_bf16, void* q, float* scale,
                                      int rows, int cols, cudaStream_t stream) {
    pf_quantize_rows_fp8_kernel<<<rows, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<__nv_fp8_e4m3*>(q), scale, rows, cols);
}

void launch_prefill_gemm_fp8(const void* A, const void* W,
                             const float* sx, const float* sw, void* C,
                             int M, int N, int K, cudaStream_t stream) {
    dim3 grid((N + FP8_BN - 1) / FP8_BN, (M + FP8_BM - 1) / FP8_BM);
    if (fp8_use_swz8())
        pf_gemm_fp8_kernel<false, true><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(A), reinterpret_cast<const __nv_fp8_e4m3*>(W),
            sx, sw, reinterpret_cast<__nv_bfloat16*>(C), nullptr, M, N, K, 0);
    else
        pf_gemm_fp8_kernel<false, false><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(A), reinterpret_cast<const __nv_fp8_e4m3*>(W),
            sx, sw, reinterpret_cast<__nv_bfloat16*>(C), nullptr, M, N, K, 0);
}

// Same occupancy knee as the int8 split-K launcher: one 128x128 tile per block, so GDN
// qkv/z/out at M=128 are 64/48/40 blocks on a 170-SM 5090.
constexpr int FP8_SK_TILES_MAX = 96;
constexpr int FP8_SK_TARGET    = 170;
constexpr int FP8_SK_MIN_KT    = 2;
constexpr int FP8_SK_MAX       = 32;

static int fp8_sk_splits(int M, int N, int K) {
    const int tiles = ((N + FP8_BN - 1) / FP8_BN) * ((M + FP8_BM - 1) / FP8_BM);
    if (tiles <= 0 || tiles >= FP8_SK_TILES_MAX) return 1;
    int s = (FP8_SK_TARGET + tiles - 1) / tiles;
    if (s > FP8_SK_MAX) s = FP8_SK_MAX;
    const int smax = ((K + FP8_BK - 1) / FP8_BK) / FP8_SK_MIN_KT;
    if (s > smax) s = smax;
    return s > 1 ? s : 1;
}

__global__ void pf_gemm_fp8_sk_epi_kernel(const float* __restrict__ P, const float* __restrict__ sx,
                                          const float* __restrict__ sw, __nv_bfloat16* __restrict__ C,
                                          int M, int N) {
    const int m = blockIdx.y;
    if (m >= M) return;
    const float s = sx[m];
    const size_t row = (size_t)m * N;
    int n = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (n + 1 < N) {
        const float2 p = *reinterpret_cast<const float2*>(&P[row + n]);
        *reinterpret_cast<__nv_bfloat162*>(&C[row + n]) =
            __floats2bfloat162_rn(p.x * s * sw[n], p.y * s * sw[n + 1]);
    } else if (n < N) {
        C[row + n] = __float2bfloat16(P[row + n] * s * sw[n]);
    }
}

bool launch_prefill_gemm_fp8_splitk(const void* A, const void* W,
                                    const float* sx, const float* sw, void* C,
                                    int M, int N, int K, float* partials,
                                    cudaStream_t stream) {
    static const bool on = [] {
        const char* e = getenv("SPARKINFER_PREFILL_GEMM_SPLITK");
        return !(e && e[0] == '0');
    }();
    if (!on || !partials || M <= 0 || M > FP8_BM || N <= 0 || K <= 0) return false;
    const int splits = fp8_sk_splits(M, N, K);
    if (splits <= 1) return false;
    const int nk = (K + FP8_BK - 1) / FP8_BK;
    const int ktiles = (nk + splits - 1) / splits;
    const int nz = (nk + ktiles - 1) / ktiles;
    if (cudaMemsetAsync(partials, 0, (size_t)M * N * sizeof(float), stream) != cudaSuccess)
        return false;
    dim3 grid((N + FP8_BN - 1) / FP8_BN, (M + FP8_BM - 1) / FP8_BM, nz);
    if (fp8_use_swz8())
        pf_gemm_fp8_kernel<true, true><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(A), reinterpret_cast<const __nv_fp8_e4m3*>(W),
            sx, sw, nullptr, partials, M, N, K, ktiles);
    else
        pf_gemm_fp8_kernel<true, false><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(A), reinterpret_cast<const __nv_fp8_e4m3*>(W),
            sx, sw, nullptr, partials, M, N, K, ktiles);
    dim3 eg(((N + 1) / 2 + 255) / 256, M);
    pf_gemm_fp8_sk_epi_kernel<<<eg, 256, 0, stream>>>(
        partials, sx, sw, reinterpret_cast<__nv_bfloat16*>(C), M, N);
    return true;
}

}} // namespace sparkinfer::kernels
