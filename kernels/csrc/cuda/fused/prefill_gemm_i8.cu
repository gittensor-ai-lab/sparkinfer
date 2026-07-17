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

__device__ __forceinline__ unsigned pf_lds32(const signed char* p) {
    return *reinterpret_cast<const unsigned*>(p);
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
__global__ __launch_bounds__(256, 2) void pf_gemm_i8_kernel(
        const signed char* __restrict__ A, const signed char* __restrict__ W,
        const float* __restrict__ sx, const float* __restrict__ sw,
        __nv_bfloat16* __restrict__ C, int M, int N, int K) {
    __shared__ signed char As[2][PF_BM][PF_BK];
    __shared__ signed char Bs[2][PF_BN][PF_BK];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 2;                       // 0..7
    const int tig  = lane & 3;                        // thread-in-group
    const int wm   = warp & 3;                        // rows [wm*32, +32)
    const int wn   = warp >> 2;                       // cols [wn*64, +64)
    const int m0   = blockIdx.y * PF_BM;
    const int n0   = blockIdx.x * PF_BN;
    const int nk   = (K + PF_BK - 1) / PF_BK;

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

    stage(0, 0);
    int buf = 0;
    for (int t = 0; t < nk; t++) {
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * PF_BK);
        __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < PF_BK; kk += 32) {
            const int ka = kk + tig * 4;
            unsigned af[PF_MFRAG][4], bf[PF_NFRAG][2];
            #pragma unroll
            for (int i = 0; i < PF_MFRAG; i++) {
                const int rlo = wm * 32 + i * 16 + grp, rhi = rlo + 8;
                af[i][0] = pf_lds32(&As[buf][rlo][pf_swz(ka,      rlo)]);
                af[i][1] = pf_lds32(&As[buf][rhi][pf_swz(ka,      rhi)]);
                af[i][2] = pf_lds32(&As[buf][rlo][pf_swz(ka + 16, rlo)]);
                af[i][3] = pf_lds32(&As[buf][rhi][pf_swz(ka + 16, rhi)]);
            }
            #pragma unroll
            for (int j = 0; j < PF_NFRAG; j++) {
                const int col = wn * 64 + j * 8 + grp;
                bf[j][0] = pf_lds32(&Bs[buf][col][pf_swz(ka,      col)]);
                bf[j][1] = pf_lds32(&Bs[buf][col][pf_swz(ka + 16, col)]);
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
                    if (gm < M && cn < N)
                        C[(size_t)gm * N + cn] = __float2bfloat16((float)acc[i][j][e] * sx[gm] * sw[cn]);
                }
                continue;
            }
            const float w0 = sw[gn], w1 = sw[gn + 1];
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int gm = m0 + wm * 32 + i * 16 + grp + h * 8;
                if (gm >= M) continue;
                const float s = sx[gm];
                const __nv_bfloat162 v = __floats2bfloat162_rn((float)acc[i][j][h * 2] * s * w0,
                                                               (float)acc[i][j][h * 2 + 1] * s * w1);
                *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)gm * N + gn]) = v;
            }
        }
    }
}
} // namespace

void launch_prefill_quantize_rows_i8(const void* x_bf16, signed char* q, float* scale,
                                     int rows, int cols, cudaStream_t stream) {
    // Block-parallel single-pass path (one block per row, row held in registers; bit-identical).
    // SPARKINFER_PREFILL_QUANT_ROWS=0 restores the warp-per-row kernel below.
    if (launch_prefill_quant_rows_fast(x_bf16, q, scale, rows, cols, stream)) return;
    pf_quantize_rows_i8<<<rows, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16), q, scale, rows, cols);
}

void launch_prefill_gemm_i8(const signed char* A, const signed char* W,
                            const float* sx, const float* sw, void* C,
                            int M, int N, int K, cudaStream_t stream) {
    dim3 grid((N + PF_BN - 1) / PF_BN, (M + PF_BM - 1) / PF_BM);
    pf_gemm_i8_kernel<<<grid, 256, 0, stream>>>(
        A, W, sx, sw, reinterpret_cast<__nv_bfloat16*>(C), M, N, K);
}

}} // namespace sparkinfer::kernels
