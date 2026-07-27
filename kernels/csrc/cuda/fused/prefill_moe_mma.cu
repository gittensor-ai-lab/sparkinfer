// Native-int8 mma.sync variant of the grouped MoE expert GEMM (see prefill_moe.cu).
//
// The stock pfm_moe_gemm_i8_kernel accumulates through the wmma m16n16k16 path, which for int8
// operands can only issue the k16 tensor-core shape — half the native int8 MAC rate. This kernel
// keeps the exact tiling, expert indirection and scatter epilogue of the wmma version but drives the
// K loop with mma.sync.aligned.m16n8k32.s32.s8.s8.s32 (int8's native shape), ldmatrix.x4 fragment
// staging and an XOR-swizzled smem layout, mirroring the pf_gemm_i8 dense kernel. int8 x int8 -> int32
// accumulation is exact, so the accumulators — and therefore C (or the scatter contributions) — are
// bit-identical to the wmma path; only the tensor-core instruction shape changes.
//
// Opt-in via SPARKINFER_PREFILL_MOE_MMA (dispatched from launch_pfm_moe_gemm_i8_bm_base).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include "sparkinfer/kernels/prefill_moe.h"

namespace sparkinfer { namespace kernels {

namespace {
constexpr int MM_BM = 128;
constexpr int MM_BN = 128;
constexpr int MM_BK = 64;          // 4 x 16B chunks per row (mma.sync k32 pairs per iter)
constexpr int MM_MFRAG = 2;        // 32 rows per warp / 16
constexpr int MM_NFRAG = 8;        // 64 cols per warp / 8

__device__ __forceinline__ void mm_cp16(void* dst, const void* src, bool pred) {
    if (pred) __pipeline_memcpy_async(dst, src, 16);
    else      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
}

// XOR swizzle at 16B granularity — identical layout to pf_gemm_i8 so the 4B operand loads spread
// across banks.
__device__ __forceinline__ int mm_swz(int k, int row) {
    return (((k >> 4) ^ (row & 3)) << 4) | (k & 15);
}

__device__ __forceinline__ void mm_ldm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                          const signed char* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}

// Grouped int8 GEMM over expert-partitioned pair tiles. Same signature/semantics as
// pfm_moe_gemm_i8_kernel; A_INDIRECT gathers A rows through pair_tok, C_SCATTER scatter-adds the
// weighted result into out_f32 (else emits bf16 C).
template <bool A_INDIRECT, bool C_SCATTER>
__global__ __launch_bounds__(256, 2) void pfm_moe_gemm_i8_mma_kernel(
        const signed char* __restrict__ A_i8, const float* __restrict__ sx,
        const signed char* __restrict__ W_i8, const float* __restrict__ sw,
        const int* __restrict__ pair_tok, const float* __restrict__ pair_w,
        const int* __restrict__ offsets, const int* __restrict__ tilemap,
        const int* __restrict__ d_ntiles,
        __nv_bfloat16* __restrict__ C, float* __restrict__ out_f32,
        int N, int K, int e_base) {
    const int tile = blockIdx.y;
    if (tile >= d_ntiles[0]) return;
    const int e   = tilemap[2 * tile];
    const int mt  = tilemap[2 * tile + 1];
    const int p0  = offsets[e] + mt * MM_BM;         // first pair row of this tile
    const int cnt = offsets[e + 1] - offsets[e];     // pairs for this expert
    const int M   = min(MM_BM, cnt - mt * MM_BM);    // valid rows in tile

    __shared__ signed char As[2][MM_BM][MM_BK];
    __shared__ signed char Bs[2][MM_BN][MM_BK];
    __shared__ int s_tok[MM_BM];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 2;                       // 0..7 (accumulator row group)
    const int tig  = lane & 3;                        // thread-in-group (accumulator col)
    const int sub  = lane >> 3;                       // ldmatrix tile addressed (0..3)
    const int lrow = lane & 7;                        // row within that tile
    const int wm   = warp & 3;                        // rows [wm*32, +32)
    const int wn   = warp >> 2;                       // cols [wn*64, +64)
    const int n0   = blockIdx.x * MM_BN;
    const int nk   = (K + MM_BK - 1) / MM_BK;
    const signed char* We  = W_i8 + (size_t)(e - e_base) * N * K;
    const float*       swe = sw + (size_t)(e - e_base) * N;

    for (int r = tid; r < MM_BM; r += blockDim.x)
        s_tok[r] = (r < M) ? (A_INDIRECT ? pair_tok[p0 + r] : (p0 + r)) : -1;
    __syncthreads();

    int acc[MM_MFRAG][MM_NFRAG][4];
    #pragma unroll
    for (int i = 0; i < MM_MFRAG; i++)
        #pragma unroll
        for (int j = 0; j < MM_NFRAG; j++)
            #pragma unroll
            for (int e2 = 0; e2 < 4; e2++) acc[i][j][e2] = 0;

    // 128 rows x 64B = 512 16B chunks per tile; 256 threads stage 2 A-chunks + 2 B-chunks each.
    auto stage = [&](int buf, int k0) {
        #pragma unroll
        for (int s = tid; s < 512; s += 256) {
            const int r = s >> 2, c = s & 3, k = c << 4;
            const int gk = k0 + k;
            const int arow = s_tok[r];
            mm_cp16(&As[buf][r][mm_swz(k, r)], &A_i8[(size_t)max(arow, 0) * K + gk],
                    arow >= 0 && gk < K);
            const int gn = n0 + r;
            mm_cp16(&Bs[buf][r][mm_swz(k, r)], &We[(size_t)gn * K + gk], gn < N && gk < K);
        }
        __pipeline_commit();
    };

    stage(0, 0);
    int buf = 0;
    for (int t = 0; t < nk; t++) {
        if (t + 1 < nk) stage(buf ^ 1, (t + 1) * MM_BK);
        __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < MM_BK; kk += 32) {
            unsigned af[MM_MFRAG][4], bf[MM_NFRAG][2];
            #pragma unroll
            for (int i = 0; i < MM_MFRAG; i++) {
                const int row = wm * 32 + i * 16 + (sub & 1) * 8 + lrow;
                mm_ldm_x4(af[i][0], af[i][1], af[i][2], af[i][3],
                          &As[buf][row][mm_swz(kk + (sub >> 1) * 16, row)]);
            }
            #pragma unroll
            for (int jp = 0; jp < MM_NFRAG; jp += 2) {
                const int col = wn * 64 + (jp + (sub >> 1)) * 8 + lrow;
                mm_ldm_x4(bf[jp][0], bf[jp][1], bf[jp + 1][0], bf[jp + 1][1],
                          &Bs[buf][col][mm_swz(kk + (sub & 1) * 16, col)]);
            }
            #pragma unroll
            for (int i = 0; i < MM_MFRAG; i++)
                #pragma unroll
                for (int j = 0; j < MM_NFRAG; j++)
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

    // Register epilogue: acc[i][j][e2] holds output (rm, gn) with rm = wm*32+i*16+grp+(e2>>1)*8,
    // gn = n0+wn*64+j*8+tig*2+(e2&1). Fold per-token sx and per-channel swe like the wmma path.
    #pragma unroll
    for (int i = 0; i < MM_MFRAG; i++) {
        #pragma unroll
        for (int j = 0; j < MM_NFRAG; j++) {
            #pragma unroll
            for (int e2 = 0; e2 < 4; e2++) {
                const int rm = wm * 32 + i * 16 + grp + (e2 >> 1) * 8;
                const int gn = n0 + wn * 64 + j * 8 + tig * 2 + (e2 & 1);
                if (rm < M && gn < N) {
                    const int p = p0 + rm;
                    const int srow = A_INDIRECT ? s_tok[rm] : p;
                    const float v = (float)acc[i][j][e2] * sx[srow] * swe[gn];
                    if (C_SCATTER) atomicAdd(&out_f32[(size_t)pair_tok[p] * N + gn], v * pair_w[p]);
                    else           C[(size_t)p * N + gn] = __float2bfloat16(v);
                }
            }
        }
    }
}
}  // namespace

void launch_pfm_moe_gemm_i8_mma(const signed char* A_i8, const float* sx,
                                const signed char* W_i8, const float* sw,
                                const int* pair_tok, const float* pair_w,
                                const int* offsets, const int* tilemap, const int* d_ntiles,
                                void* C_bf16, float* out_f32,
                                int N_out, int K, int max_tiles, int e_base,
                                bool a_indirect, bool c_scatter, cudaStream_t stream) {
    dim3 grid((N_out + MM_BN - 1) / MM_BN, max_tiles);
    auto* C = reinterpret_cast<__nv_bfloat16*>(C_bf16);
    if (a_indirect && !c_scatter)
        pfm_moe_gemm_i8_mma_kernel<true, false><<<grid, 256, 0, stream>>>(
            A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
    else if (!a_indirect && c_scatter)
        pfm_moe_gemm_i8_mma_kernel<false, true><<<grid, 256, 0, stream>>>(
            A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
    else if (a_indirect && c_scatter)
        pfm_moe_gemm_i8_mma_kernel<true, true><<<grid, 256, 0, stream>>>(
            A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
    else
        pfm_moe_gemm_i8_mma_kernel<false, false><<<grid, 256, 0, stream>>>(
            A_i8, sx, W_i8, sw, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, N_out, K, e_base);
}

}}  // namespace sparkinfer::kernels
