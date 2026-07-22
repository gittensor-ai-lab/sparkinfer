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
__device__ __forceinline__ int fp8_swz(int k, int row) {
    return (((k >> 4) ^ (row & 3)) << 4) | (k & 15);
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
__global__ __launch_bounds__(256, 2) void pf_gemm_fp8_kernel(
        const __nv_fp8_e4m3* __restrict__ A, const __nv_fp8_e4m3* __restrict__ W,
        const float* __restrict__ sx, const float* __restrict__ sw,
        __nv_bfloat16* __restrict__ C, int M, int N, int K) {
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
            fp8_cp16(&As[buf][r][fp8_swz(k, r)], &A[(size_t)gm * K + gk], gm < M && gk < K);
            fp8_cp16(&Bs[buf][r][fp8_swz(k, r)], &W[(size_t)gn * K + gk], gn < N && gk < K);
        }
        __pipeline_commit();
    };

    // fp16 partials, reset after each flush (every FP8_FLUSH BK-tiles) to bound the running sum.
    unsigned h[FP8_MFRAG][FP8_NFRAG][2];
    #pragma unroll
    for (int i = 0; i < FP8_MFRAG; i++)
        #pragma unroll
        for (int j = 0; j < FP8_NFRAG; j++) { h[i][j][0] = 0u; h[i][j][1] = 0u; }

    stage(0, 0);
    int buf = 0;
    for (int t = 0; t < nk; t++) {
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * FP8_BK);
        __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < FP8_BK; kk += 32) {
            unsigned af[FP8_MFRAG][4], bf[FP8_NFRAG][2];
            // A fragment i: tiles {rows lo,k0} {rows hi,k0} {rows lo,k16} {rows hi,k16} -> af[i][0..3]
            #pragma unroll
            for (int i = 0; i < FP8_MFRAG; i++) {
                const int row = wm * 32 + i * 16 + (sub & 1) * 8 + lrow;
                fp8_ldm_x4(af[i][0], af[i][1], af[i][2], af[i][3],
                           &As[buf][row][fp8_swz(kk + (sub >> 1) * 16, row)]);
            }
            // B pair (j, j+1): tiles {cols j,k0} {cols j,k16} {cols j+1,k0} {cols j+1,k16}
            #pragma unroll
            for (int jp = 0; jp < FP8_NFRAG; jp += 2) {
                const int col = wn * 64 + (jp + (sub >> 1)) * 8 + lrow;
                fp8_ldm_x4(bf[jp][0], bf[jp][1], bf[jp + 1][0], bf[jp + 1][1],
                           &Bs[buf][col][fp8_swz(kk + (sub & 1) * 16, col)]);
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
        if ((t % FP8_FLUSH) == FP8_FLUSH - 1 || t == nk - 1) {
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
    pf_gemm_fp8_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(A), reinterpret_cast<const __nv_fp8_e4m3*>(W),
        sx, sw, reinterpret_cast<__nv_bfloat16*>(C), M, N, K);
}

}} // namespace sparkinfer::kernels
