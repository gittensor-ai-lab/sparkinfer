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
constexpr int FP8_BK = 64;          // default K tile: 4 x 16B chunks per row
constexpr int FP8_MFRAG = 2;        // 32 rows per warp / 16
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

// XOR swizzle at 16B granularity: chunk c of row r lives at chunk (c ^ (r & (BK/16-1))) -- the
// rows the 4B operand loads walk land on disjoint banks. Same scheme as the int8 GEMM. The mask
// is the tile's chunk count so a WIDER K tile spreads over all of its chunks; at BK=64 it is
// (row & 3), byte for byte what this always was.
template <int BK>
__device__ __forceinline__ int fp8_swz(int k, int row) {
    return (((k >> 4) ^ (row & (BK / 16 - 1))) << 4) | (k & 15);
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
template <bool SPLITK, int BM = FP8_BM, int STAGES = 2, int BK = FP8_BK>
__global__ __launch_bounds__(256, 2) void pf_gemm_fp8_kernel(
        const __nv_fp8_e4m3* __restrict__ A, const __nv_fp8_e4m3* __restrict__ W,
        const float* __restrict__ sx, const float* __restrict__ sw,
        __nv_bfloat16* __restrict__ C, float* __restrict__ P,
        int M, int N, int K, int ktiles) {
    // The block's eight warps split WM ways down the tile's rows and WN ways across its columns.
    // At BM=128 that is the 4x2 map this kernel has always used, 32 rows and 64 columns each. A
    // SHORTER tile spends the freed warps on columns instead: every warp still owns 32 rows, the
    // block still covers all 128 columns, and the mma count falls with the rows that are gone.
    static_assert(BM % 32 == 0 && 8 % (BM / 32) == 0, "BM must be 32, 64 or 128");
    constexpr int WM    = BM / 32;             // warp rows
    constexpr int WN    = 8 / WM;              // warp columns
    constexpr int NPW   = FP8_BN / WN;         // columns per warp
    constexpr int NFRAG = NPW / 8;
    // How many 16B chunks one tile row holds, and how many BK-tiles the fp16 partials run for.
    // A weight row is read BK bytes at a time and the rows of a tile are K bytes apart in DRAM,
    // so BK is the length of each of the BN row-streams a CTA interleaves -- 64 B of every row,
    // then the next 64 B of every row, which is why this GEMM streams at 0.93 TB/s where the
    // row-at-a-time FP8 GEMV reads the SAME weights at 1.40. Doubling BK doubles the run.
    // FLUSH_T keeps the fp16 partials running over the same 512 K-elements at any BK, so the
    // accumulation grouping within a split is unchanged.
    constexpr int CPR     = BK / 16;
    constexpr int FLUSH_T = FP8_FLUSH * FP8_BK / BK;
    static_assert(BK % 32 == 0 && FLUSH_T >= 1, "BK must be a multiple of 32 and at most 512");
    __shared__ __nv_fp8_e4m3 As[STAGES][BM][BK];
    __shared__ __nv_fp8_e4m3 Bs[STAGES][FP8_BN][BK];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 2;                       // 0..7
    const int tig  = lane & 3;                        // thread-in-group
    const int sub  = lane >> 3;                       // ldmatrix tile this thread addresses (0..3)
    const int lrow = lane & 7;                        // row within that tile
    const int wm   = warp % WM;                       // rows [wm*32, +32)
    const int wn   = warp / WM;                       // cols [wn*NPW, +NPW)
    const int m0   = blockIdx.y * BM;
    const int n0   = blockIdx.x * FP8_BN;
    const int nk   = (K + BK - 1) / BK;
    int t0 = 0, t1 = nk;
    if (SPLITK) {
        t0 = blockIdx.z * ktiles;
        t1 = t0 + ktiles;
        if (t1 > nk) t1 = nk;
        if (t0 >= t1) return;
    }

    float acc[FP8_MFRAG][NFRAG][4];
    #pragma unroll
    for (int i = 0; i < FP8_MFRAG; i++)
        #pragma unroll
        for (int j = 0; j < NFRAG; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

    // A tile row is BK bytes = CPR 16B chunks, so the A tile is BM*CPR chunks and the 128-column
    // B tile is FP8_BN*CPR. They were ONE loop while BM was 128 and the extents agreed -- and
    // the rows past M were staged as explicit zeros, then multiplied by a weight column for
    // nothing. Separate loops are what let A be short; every chunk, predicate and address is the
    // one the single loop issued. At BM == FP8_BN the single loop is kept verbatim, so the wide
    // tile is untouched.
    auto stage = [&](int buf, int k0) {
        if constexpr (BM == FP8_BN) {
            // Equal extents: ONE loop, an A chunk and a B chunk per step, which is the cp.async
            // issue order this kernel has always had. Keeping it is not cosmetic -- staging the
            // two tiles in separate loops at this tile size measured c16 -1.9% on its own, so
            // the knob-off arm would not have been main.
            #pragma unroll
            for (int s = tid; s < FP8_BN * CPR; s += 256) {
                const int r = s / CPR, k = (s % CPR) << 4;
                const int gm = m0 + r, gn = n0 + r, gk = k0 + k;
                fp8_cp16(&As[buf][r][fp8_swz<BK>(k, r)], &A[(size_t)gm * K + gk], gm < M && gk < K);
                fp8_cp16(&Bs[buf][r][fp8_swz<BK>(k, r)], &W[(size_t)gn * K + gk], gn < N && gk < K);
            }
        } else {
            #pragma unroll
            for (int s = tid; s < BM * CPR; s += 256) {
                const int r = s / CPR, k = (s % CPR) << 4;
                const int gm = m0 + r, gk = k0 + k;
                fp8_cp16(&As[buf][r][fp8_swz<BK>(k, r)], &A[(size_t)gm * K + gk], gm < M && gk < K);
            }
            #pragma unroll
            for (int s = tid; s < FP8_BN * CPR; s += 256) {
                const int r = s / CPR, k = (s % CPR) << 4;
                const int gn = n0 + r, gk = k0 + k;
                fp8_cp16(&Bs[buf][r][fp8_swz<BK>(k, r)], &W[(size_t)gn * K + gk], gn < N && gk < K);
            }
        }
        __pipeline_commit();
    };

    // fp16 partials, reset after each flush (every FLUSH_T BK-tiles) to bound the running sum.
    unsigned h[FP8_MFRAG][NFRAG][2];
    #pragma unroll
    for (int i = 0; i < FP8_MFRAG; i++)
        #pragma unroll
        for (int j = 0; j < NFRAG; j++) { h[i][j][0] = 0u; h[i][j][1] = 0u; }

    // STAGES-1 K-tiles are in flight while the mainloop computes on the oldest. At STAGES=2 the
    // prologue, the issue point and the wait count below are exactly the double buffer this
    // kernel has always run; a short A tile leaves shared memory for a deeper one.
    int inflight = 0;
    #pragma unroll
    for (int p = 0; p < STAGES - 1; p++)
        if (t0 + p < t1) { stage(p, (t0 + p) * BK); inflight++; }
    for (int t = t0; t < t1; t++) {
        const int buf = (t - t0) % STAGES;
        if (t + STAGES - 1 < t1) {
            stage((t - t0 + STAGES - 1) % STAGES, (t + STAGES - 1) * BK);
            inflight++;
        }
        __pipeline_wait_prior(inflight - 1);
        inflight--;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 32) {
            unsigned af[FP8_MFRAG][4], bf[NFRAG][2];
            // A fragment i: tiles {rows lo,k0} {rows hi,k0} {rows lo,k16} {rows hi,k16} -> af[i][0..3]
            #pragma unroll
            for (int i = 0; i < FP8_MFRAG; i++) {
                const int row = wm * 32 + i * 16 + (sub & 1) * 8 + lrow;
                fp8_ldm_x4(af[i][0], af[i][1], af[i][2], af[i][3],
                           &As[buf][row][fp8_swz<BK>(kk + (sub >> 1) * 16, row)]);
            }
            // B pair (j, j+1): tiles {cols j,k0} {cols j,k16} {cols j+1,k0} {cols j+1,k16}
            #pragma unroll
            for (int jp = 0; jp < NFRAG; jp += 2) {
                const int col = wn * NPW + (jp + (sub >> 1)) * 8 + lrow;
                fp8_ldm_x4(bf[jp][0], bf[jp][1], bf[jp + 1][0], bf[jp + 1][1],
                           &Bs[buf][col][fp8_swz<BK>(kk + (sub & 1) * 16, col)]);
            }
            #pragma unroll
            for (int i = 0; i < FP8_MFRAG; i++)
                #pragma unroll
                for (int j = 0; j < NFRAG; j++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                        : "+r"(h[i][j][0]), "+r"(h[i][j][1])
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]),
                          "r"(bf[j][0]), "r"(bf[j][1]));
        }
        // flush fp16 partials into the fp32 accumulator every FLUSH_T tiles (and on the last one)
        if ((t % FLUSH_T) == FLUSH_T - 1 || t == t1 - 1) {
            #pragma unroll
            for (int i = 0; i < FP8_MFRAG; i++)
                #pragma unroll
                for (int j = 0; j < NFRAG; j++) {
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
    }

    if (SPLITK) {
        #pragma unroll
        for (int i = 0; i < FP8_MFRAG; i++) {
            #pragma unroll
            for (int j = 0; j < NFRAG; j++) {
                const int gn = n0 + wn * NPW + j * 8 + tig * 2;
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
        for (int j = 0; j < NFRAG; j++) {
            const int gn = n0 + wn * NPW + j * 8 + tig * 2;
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

// The same quantize as pf_quantize_rows_fp8_kernel, shaped like pf_quant_rows_fast_kernel
// (prefill_quant_rows.cu), which fixed exactly this for the int8 activations. The warp-per-row
// kernel gives each row 32 lanes that each stride every 32nd element, so a 5120-wide row is a
// 160-deep chain of dependent loads, walked twice -- and at the widths a packed decode step runs
// that chain is the whole cost: 5.4 us a launch at cols 5120 and 6.2 at 6144, the same at 1 row
// as at 32, for the 96 launches a step Qwen3.8's GDN projections issue. Here one 256-thread block
// covers a row, each thread holds SLOTS x 8 bf16 in registers, the row is read once and written
// once, and the row max is reduced over the registers in two levels. max is exact in any order and
// d = amax / FP8_TGT with the same amax == 0 rule, so every e4m3 byte and every scale match the
// kernel above. Measured 2.8-3.0 us at 1-32 rows, 4.1 at 256 rows (6.2 before) and 11.6 / 13.6 at
// 4096 rows x 5120 / 6144 (19.3 / 22.8). SPARKINFER_FP8_QUANT_FAST=0 restores the warp-per-row kernel.
template <int BLOCK, int VEC, int SLOTS>
__global__ __launch_bounds__(BLOCK) void pf_quantize_rows_fp8_fast_kernel(
        const __nv_bfloat16* __restrict__ x, __nv_fp8_e4m3* __restrict__ q,
        float* __restrict__ scale, int rows, int cols) {
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
            *reinterpret_cast<uint4*>(reg[s]) = *reinterpret_cast<const uint4*>(&x[base + c]);
            #pragma unroll
            for (int v = 0; v < VEC; v++) amax = fmaxf(amax, fabsf(__bfloat162float(reg[s][v])));
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
    const float d = (sred[0] == 0.f) ? 1.f : (sred[0] / FP8_TGT);
    if (tid == 0) scale[r] = d;
    #pragma unroll
    for (int s = 0; s < SLOTS; s++) {
        const int c = (tid + s * BLOCK) * VEC;
        if (c < cols) {
            __nv_fp8_e4m3 out[VEC];
            #pragma unroll
            for (int v = 0; v < VEC; v++) out[v] = __nv_fp8_e4m3(__bfloat162float(reg[s][v]) / d);
            *reinterpret_cast<uint2*>(&q[base + c]) = *reinterpret_cast<const uint2*>(out);
        }
    }
}

void launch_prefill_quantize_rows_fp8(const void* x_bf16, void* q, float* scale,
                                      int rows, int cols, cudaStream_t stream) {
    static const bool fast = [] {
        const char* e = getenv("SPARKINFER_FP8_QUANT_FAST");
        return !(e && e[0] == '0');
    }();
    if (fast && rows > 0 && cols > 0 && (cols % 8) == 0) {
        constexpr int BLOCK = 256, VEC = 8;
        const int vecs = cols / VEC;
        const auto* xb = reinterpret_cast<const __nv_bfloat16*>(x_bf16);
        auto* qb = reinterpret_cast<__nv_fp8_e4m3*>(q);
        bool done = true;
        if (vecs <= BLOCK * 1)       pf_quantize_rows_fp8_fast_kernel<BLOCK, VEC, 1><<<rows, BLOCK, 0, stream>>>(xb, qb, scale, rows, cols);
        else if (vecs <= BLOCK * 2)  pf_quantize_rows_fp8_fast_kernel<BLOCK, VEC, 2><<<rows, BLOCK, 0, stream>>>(xb, qb, scale, rows, cols);
        else if (vecs <= BLOCK * 3)  pf_quantize_rows_fp8_fast_kernel<BLOCK, VEC, 3><<<rows, BLOCK, 0, stream>>>(xb, qb, scale, rows, cols);
        else if (vecs <= BLOCK * 4)  pf_quantize_rows_fp8_fast_kernel<BLOCK, VEC, 4><<<rows, BLOCK, 0, stream>>>(xb, qb, scale, rows, cols);
        else if (vecs <= BLOCK * 8)  pf_quantize_rows_fp8_fast_kernel<BLOCK, VEC, 8><<<rows, BLOCK, 0, stream>>>(xb, qb, scale, rows, cols);
        else if (vecs <= BLOCK * 10) pf_quantize_rows_fp8_fast_kernel<BLOCK, VEC, 10><<<rows, BLOCK, 0, stream>>>(xb, qb, scale, rows, cols);
        else done = false;
        if (done) return;
    }
    pf_quantize_rows_fp8_kernel<<<rows, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<__nv_fp8_e4m3*>(q), scale, rows, cols);
}

// A packed decode step hands this GEMM eight to thirty-two rows and the tile is 128 tall, so
// three quarters of every mma -- and of the A tile in shared memory -- are rows that do not
// exist. The 32-row tile computes only the rows that do, which on its own is worth nothing
// measurable (c32 1362 -> 1364 tok/s): this kernel was never short of math. What the short tile
// BUYS is the shared memory, and that goes into the K tile, which is the axis it IS short on --
// see fp8_narrow_bk below.
//
// The 32-row tile alone is bit-identical: it walks K in the same order with the same flush
// cadence and only re-assigns which warp owns which columns. The wider K tile is NOT, because
// the split-K partition is counted in K-tiles and its boundaries land elsewhere, so the fp32
// partials a split contributes are summed in a different grouping. Nothing else about the
// arithmetic moves -- see FLUSH_T in the kernel.
// SPARKINFER_FP8_GEMM_NARROW_M=0 restores the 128-row 64-byte tile at every width, which is
// main byte for byte (A/B in ONE binary).
namespace {
constexpr int FP8_NARROW_BM = 32;
// Lower bound, not just the 32-row upper one. The tile was measured across the packed decode
// widths the bots score -- 8, 16 and 32 rows -- and nothing below that was ever benchmarked: a
// prefill tail chunk of a handful of rows lands in the same launcher, and at BK=128 it has half
// as many K-tiles to split across, so the split-K rule can hand it fewer blocks than the 64-byte
// tile would. Below eight rows the old tile stays, which is also what every caller on the packed
// path already gates on (kProjGemmMinRows / SPARKINFER_FFN_GEMM_MIN_ROWS default to 8).
constexpr int FP8_NARROW_MIN_M = 8;
bool fp8_narrow_m() {
    static const bool v = [] {
        const char* e = getenv("SPARKINFER_FP8_GEMM_NARROW_M");
        return !(e && atoi(e) == 0);
    }();
    return v;
}
// K width of the narrow tile, and the pipeline depth that fits beside it. The static shared
// allocation may take 48 KB, and one stage is (BM + FP8_BN) * BK bytes, so at BM=32 the two
// reachable points carry the SAME 40 KB in flight: BK=64 four stages deep, or BK=128 two. They
// differ only in how the weight is READ -- 64 B of each of the tile's 128 rows before returning
// to the first, or 128 B -- and that is the whole measured difference between them: c32 1364.2
// tok/s at BK=64 against 1411.1 at BK=128, same tile, same binary, same bytes resident.
int fp8_narrow_bk() {
    static const int v = [] {
        const char* e = getenv("SPARKINFER_FP8_GEMM_NARROW_BK");
        const int x = e ? atoi(e) : 128;
        return (x == 64 || x == 128) ? x : 128;
    }();
    return v;
}
int fp8_narrow_stages() {
    static const int v = [] {
        const char* e = getenv("SPARKINFER_FP8_GEMM_NARROW_STAGES");
        const int x = e ? atoi(e) : 0;
        return (x >= 2 && x <= 4) ? x : 0;
    }();
    return v ? v : (fp8_narrow_bk() == 128 ? 2 : 4);
}
} // namespace

#define SI_FP8_NARROW(SK, GRID, ...)                                                        \
    do {                                                                                    \
        const int bk_ = fp8_narrow_bk(), st_ = fp8_narrow_stages();                         \
        if (bk_ == 128) {                                                                   \
            pf_gemm_fp8_kernel<SK, FP8_NARROW_BM, 2, 128>                                   \
                <<<GRID, 256, 0, stream>>>(__VA_ARGS__);                                    \
        } else if (st_ == 2) {                                                              \
            pf_gemm_fp8_kernel<SK, FP8_NARROW_BM, 2, 64>                                    \
                <<<GRID, 256, 0, stream>>>(__VA_ARGS__);                                    \
        } else if (st_ == 3) {                                                              \
            pf_gemm_fp8_kernel<SK, FP8_NARROW_BM, 3, 64>                                    \
                <<<GRID, 256, 0, stream>>>(__VA_ARGS__);                                    \
        } else {                                                                            \
            pf_gemm_fp8_kernel<SK, FP8_NARROW_BM, 4, 64>                                    \
                <<<GRID, 256, 0, stream>>>(__VA_ARGS__);                                    \
        }                                                                                   \
    } while (0)
// K tile the dispatch below actually uses, so the split-K partition is counted in the same
// units the kernel walks.
static inline int fp8_tile_bk(bool narrow) { return narrow ? fp8_narrow_bk() : FP8_BK; }

void launch_prefill_gemm_fp8(const void* A, const void* W,
                             const float* sx, const float* sw, void* C,
                             int M, int N, int K, cudaStream_t stream) {
    const bool narrow = M >= FP8_NARROW_MIN_M && M <= FP8_NARROW_BM && fp8_narrow_m();
    const int bm = narrow ? FP8_NARROW_BM : FP8_BM;
    dim3 grid((N + FP8_BN - 1) / FP8_BN, (M + bm - 1) / bm);
    if (narrow) {
        SI_FP8_NARROW(false, grid,
            reinterpret_cast<const __nv_fp8_e4m3*>(A), reinterpret_cast<const __nv_fp8_e4m3*>(W),
            sx, sw, reinterpret_cast<__nv_bfloat16*>(C), nullptr, M, N, K, 0);
        return;
    }
    pf_gemm_fp8_kernel<false><<<grid, 256, 0, stream>>>(
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
    // The split is over K, so the M tiling is the one the launcher above picks and the number of
    // blocks -- which is what fp8_sk_splits balanced -- is the same either way at these widths.
    const bool narrow = M >= FP8_NARROW_MIN_M && M <= FP8_NARROW_BM && fp8_narrow_m();
    const int bm = narrow ? FP8_NARROW_BM : FP8_BM;
    const int bk = fp8_tile_bk(narrow);
    // fp8_sk_splits balances BLOCKS, which BK does not change, but the K partition is counted in
    // BK-tiles -- so a wider tile has fewer of them and the same split has to stay above the
    // minimum tiles-per-split the split-K rule keeps.
    const int nk = (K + bk - 1) / bk;
    int sp = splits;
    const int spmax = nk / FP8_SK_MIN_KT;
    if (sp > spmax) sp = spmax;
    if (sp <= 1) return false;
    const int ktiles = (nk + sp - 1) / sp;
    const int nz = (nk + ktiles - 1) / ktiles;
    if (cudaMemsetAsync(partials, 0, (size_t)M * N * sizeof(float), stream) != cudaSuccess)
        return false;
    dim3 grid((N + FP8_BN - 1) / FP8_BN, (M + bm - 1) / bm, nz);
    if (narrow) {
        SI_FP8_NARROW(true, grid,
            reinterpret_cast<const __nv_fp8_e4m3*>(A), reinterpret_cast<const __nv_fp8_e4m3*>(W),
            sx, sw, nullptr, partials, M, N, K, ktiles);
    } else {
        pf_gemm_fp8_kernel<true><<<grid, 256, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(A), reinterpret_cast<const __nv_fp8_e4m3*>(W),
            sx, sw, nullptr, partials, M, N, K, ktiles);
    }
    dim3 eg(((N + 1) / 2 + 255) / 256, M);
    pf_gemm_fp8_sk_epi_kernel<<<eg, 256, 0, stream>>>(
        partials, sx, sw, reinterpret_cast<__nv_bfloat16*>(C), M, N);
    return true;
}

}} // namespace sparkinfer::kernels
