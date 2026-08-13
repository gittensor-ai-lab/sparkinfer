// int8 tensor-core GEMM for Qwythos batched prefill (see prefill_i8.h).
//
// C[M,N] = A[M,K] @ W^T, W native GGUF [N,K] row-major. int8 x int8 -> int32 with the dequant
// (per-token sx[m] * per-channel sw[n]) folded into the store, emitting bf16 C.
//
// Shaped for int8 rather than mirroring the bf16 GEMM: mma.sync m16n8k32 (int8's native shape --
// wmma m16n16k16 can only emit the k16 shape, which caps at half the int8 MAC rate), BK=64 to halve
// the main-loop barrier count, an XOR-swizzled smem layout so the 4B operand loads spread across
// banks, and a register->global epilogue that keeps the int32 accumulators out of shared memory.
// Same int8 quantization scheme and accumulation order as before, so C is bit-identical.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdlib>
#include "sparkinfer/kernels/prefill_i8.h"

#include "sparkinfer/kernels/prefill_quant_rows.h"

namespace sparkinfer { namespace kernels {

namespace {
constexpr int PF_BM = 128;
constexpr int PF_BN = 128;
constexpr int PF_BK = 64;          // 4 x 16B chunks per row
constexpr int PF_MFRAG = 2;        // 32 rows per warp / 16
constexpr int PF_NFRAG = 8;        // 64 cols per warp / 8

__device__ __forceinline__ void pf_cp16(void* dst, const void* src, bool pred) {
    if (pred) __pipeline_memcpy_async(dst, src, 16);
    else      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
}

// XOR swizzle at 16B granularity: chunk c of row r lives at chunk (c ^ (r & 3)). Rows 4 apart still
// collide (2-way); rows 0..3 -- the stride the 4B operand loads walk -- land on disjoint banks.
__device__ __forceinline__ int pf_swz(int k, int row) {
    return (((k >> 4) ^ (row & 3)) << 4) | (k & 15);
}

// ldmatrix.x4: one instruction moves all four 8x8 operand tiles of an mma fragment through the
// LDS pipe (the four scalar lds.32 it replaces issued 1:1 against the mma pipe and were the
// staging bottleneck). Thread t supplies the shared-memory address of row (t&7) of tile (t>>3);
// the XOR swizzle still applies per row address, so the smem layout is unchanged and the loaded
// registers are identical to the lds.32 path.
__device__ __forceinline__ void pf_ldm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                          const signed char* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

// Per-row symmetric int8 quantize, one warp per row.
__global__ void pf_quantize_rows_i8(const __nv_bfloat16* __restrict__ x, signed char* __restrict__ q,
                                    float* __restrict__ scale, int rows, int cols) {
    const int r = blockIdx.x, lane = threadIdx.x;
    if (r >= rows) return;
    float amax = 0.f;
    for (int c = lane; c < cols; c += 32) amax = fmaxf(amax, fabsf(__bfloat162float(x[(size_t)r * cols + c])));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    const float d = amax / 127.0f;
    if (lane == 0) scale[r] = d;
    for (int c = lane; c < cols; c += 32)
        q[(size_t)r * cols + c] = (signed char)((amax == 0.f) ? 0 : (int)roundf(__bfloat162float(x[(size_t)r * cols + c]) / d));
}

// The 2 in __launch_bounds__ is required, not decorative: left to itself nvcc picks 131 registers,
// and 2 * 256 * 131 exceeds the 65536-register file, so only one block per SM would be resident.
// RESID folds the residual add into the store: C[m,n] = bf16(C[m,n] + bf16(acc*sx*sw)) -- the same
// two-step rounding as the pf_add kernel it replaces, reading and writing through the ONE C
// pointer (no second aliased argument), so the fused path stays bit-identical to GEMM-then-add.
//
// SPLITK partitions the K loop across blockIdx.z (ktiles BK-tiles each) and accumulates the int32
// tile into P[M,N] with atomicAdd instead of storing C; a separate epilogue applies sx/sw. The
// output is bit-identical because the accumulator is int32: integer addition is exact and
// associative, so the reordered partial sums land on the same value the single-block loop produces.
template <bool RESID, bool SPLITK>
__global__ __launch_bounds__(256, 2) void pf_gemm_i8_kernel(
        const signed char* __restrict__ A, const signed char* __restrict__ W,
        const float* __restrict__ sx, const float* __restrict__ sw,
        __nv_bfloat16* C, int* __restrict__ P, int M, int N, int K, int ktiles) {
    __shared__ signed char As[2][PF_BM][PF_BK];
    __shared__ signed char Bs[2][PF_BN][PF_BK];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 2;                       // 0..7
    const int tig  = lane & 3;                        // thread-in-group
    const int sub  = lane >> 3;                       // ldmatrix tile this thread addresses (0..3)
    const int lrow = lane & 7;                        // row within that tile
    const int wm   = warp & 3;                        // rows [wm*32, +32)
    const int wn   = warp >> 2;                       // cols [wn*64, +64)
    const int m0   = blockIdx.y * PF_BM;
    const int n0   = blockIdx.x * PF_BN;
    const int nk   = (K + PF_BK - 1) / PF_BK;
    // K-tile range this block owns. Without SPLITK that is the whole K loop, exactly as before.
    int t0 = 0, t1 = nk;
    if (SPLITK) {
        t0 = blockIdx.z * ktiles;
        t1 = t0 + ktiles;
        if (t1 > nk) t1 = nk;
        if (t0 >= t1) return;
    }

    int acc[PF_MFRAG][PF_NFRAG][4];
    #pragma unroll
    for (int i = 0; i < PF_MFRAG; i++)
        #pragma unroll
        for (int j = 0; j < PF_NFRAG; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0;

    // 128 rows x 64B = 512 16B chunks per tile; 256 threads stage 2 A-chunks + 2 B-chunks each.
    auto stage = [&](int buf, int k0) {
        #pragma unroll
        for (int s = tid; s < 512; s += 256) {
            const int r = s >> 2, c = s & 3, k = c << 4;
            const int gm = m0 + r, gn = n0 + r, gk = k0 + k;
            pf_cp16(&As[buf][r][pf_swz(k, r)], &A[(size_t)gm * K + gk], gm < M && gk < K);
            pf_cp16(&Bs[buf][r][pf_swz(k, r)], &W[(size_t)gn * K + gk], gn < N && gk < K);
        }
        __pipeline_commit();
    };

    stage(0, t0 * PF_BK);
    int buf = 0;
    for (int t = t0; t < t1; t++) {
        if (t + 1 < t1) stage(buf ^ 1, (t + 1) * PF_BK);
        __pipeline_wait_prior(t + 1 < t1 ? 1 : 0);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < PF_BK; kk += 32) {
            unsigned af[PF_MFRAG][4], bf[PF_NFRAG][2];
            // A fragment i: tiles {rows lo,k0} {rows hi,k0} {rows lo,k16} {rows hi,k16} -> af[i][0..3]
            #pragma unroll
            for (int i = 0; i < PF_MFRAG; i++) {
                const int row = wm * 32 + i * 16 + (sub & 1) * 8 + lrow;
                pf_ldm_x4(af[i][0], af[i][1], af[i][2], af[i][3],
                          &As[buf][row][pf_swz(kk + (sub >> 1) * 16, row)]);
            }
            // B pair (j, j+1): tiles {cols j,k0} {cols j,k16} {cols j+1,k0} {cols j+1,k16}
            #pragma unroll
            for (int jp = 0; jp < PF_NFRAG; jp += 2) {
                const int col = wn * 64 + (jp + (sub >> 1)) * 8 + lrow;
                pf_ldm_x4(bf[jp][0], bf[jp][1], bf[jp + 1][0], bf[jp + 1][1],
                          &Bs[buf][col][pf_swz(kk + (sub & 1) * 16, col)]);
            }
            #pragma unroll
            for (int i = 0; i < PF_MFRAG; i++)
                #pragma unroll
                for (int j = 0; j < PF_NFRAG; j++)
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+r"(acc[i][j][0]), "+r"(acc[i][j][1]), "+r"(acc[i][j][2]), "+r"(acc[i][j][3])
                        : "r"(af[i][0]), "r"(af[i][1]), "r"(af[i][2]), "r"(af[i][3]),
                          "r"(bf[j][0]), "r"(bf[j][1]));
        }
        __syncthreads();
        buf ^= 1;
    }

    // Split-K: hand the int32 tile to the partial buffer and let the epilogue scale it. Same
    // (row, col) map as the scalar tail below.
    if (SPLITK) {
        #pragma unroll
        for (int i = 0; i < PF_MFRAG; i++) {
            #pragma unroll
            for (int j = 0; j < PF_NFRAG; j++) {
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

    // Registers straight to global: c0/c1 (and c2/c3) are adjacent columns, so each pair packs into
    // one 4B bf16x2 store.
    #pragma unroll
    for (int i = 0; i < PF_MFRAG; i++) {
        #pragma unroll
        for (int j = 0; j < PF_NFRAG; j++) {
            const int gn = n0 + wn * 64 + j * 8 + tig * 2;
            if (gn + 1 >= N) {                        // tail: scalar path
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int gm = m0 + wm * 32 + i * 16 + grp + (e >> 1) * 8;
                    const int cn = gn + (e & 1);
                    if (gm < M && cn < N) {
                        __nv_bfloat16 v = __float2bfloat16((float)acc[i][j][e] * sx[gm] * sw[cn]);
                        if (RESID)
                            v = __float2bfloat16(__bfloat162float(C[(size_t)gm * N + cn]) +
                                                 __bfloat162float(v));
                        C[(size_t)gm * N + cn] = v;
                    }
                }
                continue;
            }
            const float w0 = sw[gn], w1 = sw[gn + 1];
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int gm = m0 + wm * 32 + i * 16 + grp + h * 8;
                if (gm >= M) continue;
                const float s = sx[gm];
                __nv_bfloat162 v = __floats2bfloat162_rn((float)acc[i][j][h * 2] * s * w0,
                                                         (float)acc[i][j][h * 2 + 1] * s * w1);
                __nv_bfloat162* cp = reinterpret_cast<__nv_bfloat162*>(&C[(size_t)gm * N + gn]);
                if (RESID) {
                    const __nv_bfloat162 r = *cp;
                    v = __floats2bfloat162_rn(__bfloat162float(r.x) + __bfloat162float(v.x),
                                              __bfloat162float(r.y) + __bfloat162float(v.y));
                }
                *cp = v;
            }
        }
    }
}
// Split-K epilogue: apply the per-token / per-channel scales to the int32 partial sum. The
// arithmetic is copied from the in-GEMM epilogue above -- ((float)acc * sx[m]) * sw[n] rounded
// round-to-nearest, and for RESID the same round-then-add-then-round the fused store does -- so
// the split path and the single-block path emit the same bits.
template <bool RESID>
__global__ void pf_gemm_i8_sk_epi_kernel(const int* __restrict__ P, const float* __restrict__ sx,
                                         const float* __restrict__ sw,
                                         __nv_bfloat16* __restrict__ C, int M, int N) {
    const int m = blockIdx.y;
    if (m >= M) return;
    const float s = sx[m];
    const size_t row = (size_t)m * N;
    int n = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (n + 1 < N) {                                     // pair: one int2 load, one bf16x2 store
        const int2 p = *reinterpret_cast<const int2*>(&P[row + n]);
        __nv_bfloat162 v = __floats2bfloat162_rn((float)p.x * s * sw[n], (float)p.y * s * sw[n + 1]);
        __nv_bfloat162* cp = reinterpret_cast<__nv_bfloat162*>(&C[row + n]);
        if (RESID) {
            const __nv_bfloat162 r = *cp;
            v = __floats2bfloat162_rn(__bfloat162float(r.x) + __bfloat162float(v.x),
                                      __bfloat162float(r.y) + __bfloat162float(v.y));
        }
        *cp = v;
    } else if (n < N) {                                  // odd tail
        __nv_bfloat16 v = __float2bfloat16((float)P[row + n] * s * sw[n]);
        if (RESID)
            v = __float2bfloat16(__bfloat162float(C[row + n]) + __bfloat162float(v));
        C[row + n] = v;
    }
}

// How many ways to split K. The launch is one 128x128 output tile per block, so a projection with a
// narrow n_out gets a grid far smaller than the device: Muse Glimmer's attn k/v (n_out=256) run TWO
// blocks, and measured on an RTX 5090 that launch costs the same 69 us as the 32-block q/gate one --
// per-block streaming rate is ~12.3 GB/s and the device only saturates (~1.0 TB/s of weight) past
// ~80 blocks. Splitting K until the grid reaches that knee is what converts the idle SMs into
// throughput. Above it, extra blocks buy nothing and the partial-buffer traffic is a pure cost, so
// wide projections (Muse's ffn gate/up at 156 tiles) are deliberately left alone.
constexpr int PF_SK_TILES_MAX = 96;    // tile counts at/above this already fill the device
constexpr int PF_SK_TARGET    = 170;   // blocks to aim for (one per SM)
constexpr int PF_SK_MIN_KT    = 2;     // never leave a block fewer than this many BK-tiles
constexpr int PF_SK_MAX       = 32;    // cap: partial-buffer atomics scale with the split count

static int pf_sk_splits(int M, int N, int K) {
    const int tiles = ((N + PF_BN - 1) / PF_BN) * ((M + PF_BM - 1) / PF_BM);
    if (tiles <= 0 || tiles >= PF_SK_TILES_MAX) return 1;
    int s = (PF_SK_TARGET + tiles - 1) / tiles;
    if (s > PF_SK_MAX) s = PF_SK_MAX;
    const int smax = ((K + PF_BK - 1) / PF_BK) / PF_SK_MIN_KT;
    if (s > smax) s = smax;
    return s > 1 ? s : 1;
}
} // namespace

bool launch_prefill_gemm_i8_splitk(const signed char* A, const signed char* W,
                                   const float* sx, const float* sw, void* C,
                                   int M, int N, int K, int* partials, bool resid,
                                   cudaStream_t stream) {
    static const bool on = [] {
        const char* e = getenv("SPARKINFER_PREFILL_GEMM_SPLITK");
        return !(e && e[0] == '0');
    }();
    // One M tile only: the partial buffer is M*N int32 and the caller sizes it for that.
    if (!on || !partials || M <= 0 || M > PF_BM || N <= 0 || K <= 0) return false;
    const int splits = pf_sk_splits(M, N, K);
    if (splits <= 1) return false;
    const int nk = (K + PF_BK - 1) / PF_BK;
    const int ktiles = (nk + splits - 1) / splits;
    const int nz = (nk + ktiles - 1) / ktiles;
    if (cudaMemsetAsync(partials, 0, (size_t)M * N * sizeof(int), stream) != cudaSuccess) return false;
    dim3 grid((N + PF_BN - 1) / PF_BN, (M + PF_BM - 1) / PF_BM, nz);
    pf_gemm_i8_kernel<false, true><<<grid, 256, 0, stream>>>(
        A, W, sx, sw, nullptr, partials, M, N, K, ktiles);
    dim3 eg(((N + 1) / 2 + 255) / 256, M);
    if (resid)
        pf_gemm_i8_sk_epi_kernel<true><<<eg, 256, 0, stream>>>(
            partials, sx, sw, reinterpret_cast<__nv_bfloat16*>(C), M, N);
    else
        pf_gemm_i8_sk_epi_kernel<false><<<eg, 256, 0, stream>>>(
            partials, sx, sw, reinterpret_cast<__nv_bfloat16*>(C), M, N);
    return true;
}

bool launch_prefill_quantize_rows_i8(const void* x_bf16, signed char* q, float* scale,
                                     int rows, int cols, cudaStream_t stream, signed char* qp) {
    // Block-parallel single-pass path (one block per row, row held in registers; bit-identical).
    // SPARKINFER_PREFILL_QUANT_ROWS=0 restores the warp-per-row kernel below.
    if (launch_prefill_quant_rows_fast(x_bf16, q, scale, rows, cols, stream, qp)) return true;
    pf_quantize_rows_i8<<<rows, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16), q, scale, rows, cols);
    // The warp-per-row fallback has no k-tiled output, so tell the caller its packed copy is stale.
    return qp == nullptr;
}

void launch_prefill_gemm_i8(const signed char* A, const signed char* W,
                            const float* sx, const float* sw, void* C,
                            int M, int N, int K, cudaStream_t stream) {
    dim3 grid((N + PF_BN - 1) / PF_BN, (M + PF_BM - 1) / PF_BM);
    pf_gemm_i8_kernel<false, false><<<grid, 256, 0, stream>>>(
        A, W, sx, sw, reinterpret_cast<__nv_bfloat16*>(C), nullptr, M, N, K, 0);
}

// Residual-fused variant: C[m,n] += bf16(acc*sx*sw) with pf_add's rounding. Passing the residual
// tensor AS C removes the ao scratch round-trip and the separate full-tensor add launch.
void launch_prefill_gemm_i8_resid(const signed char* A, const signed char* W,
                                  const float* sx, const float* sw, void* C,
                                  int M, int N, int K, cudaStream_t stream) {
    dim3 grid((N + PF_BN - 1) / PF_BN, (M + PF_BM - 1) / PF_BM);
    pf_gemm_i8_kernel<true, false><<<grid, 256, 0, stream>>>(
        A, W, sx, sw, reinterpret_cast<__nv_bfloat16*>(C), nullptr, M, N, K, 0);
}

}} // namespace sparkinfer::kernels
