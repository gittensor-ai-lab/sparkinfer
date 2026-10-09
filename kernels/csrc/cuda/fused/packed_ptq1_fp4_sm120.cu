// Ternary-Bonsai-2's packed decode rows (17..32 at once) on the FP4 tensor cores (sm_120a only:
// the block-scaled FP4 MMA does not assemble for plain sm_120, so this file builds in si_nvfp4).
//
// ptq1_mma_rows_kernel (gemv_ptq1_i8.cu) runs these rows on int8 m16n8k32 MMAs, and past 16 rows
// it is bound by its operand traffic and MMA chains, not the weight stream: every CTA stages the
// whole int8 activation of each step, and each 128-trit block takes four MMAs per 8 tokens. Here
// the activation is NVFP4 (e2m1 with a ue4m3 scale per 16 values, written by the rotation that
// produces it -- ptq1_rotq_kernel's FP4 copy), and a block is two m16n8k64 kind::mxf4nvf4 MMAs
// with both scales applied in the MMA. The weight operand is the checkpoint's trit digits d (0..2)
// unchanged: a digit is itself the e2m1 code of d / 2, so with A's scale at 2 the MMA sums d * x,
// and subtracting the block's activation sum (written beside the codes) leaves the trits' (d - 1)
// * x -- the int8 kernel's own identity. Half the activation bytes, half the MMAs.
//
// The k order inside a block is the int8 kernel's: its k-step s is register s & 1 of MMA s >> 1,
// whose eight codes per word are the step's two halves (lo in the low nibbles). The rotation
// writes the activation in that order, so both operands agree and no data is permuted here.
//
// Numerics: the activation's NVFP4 rounding replaces the int8 one (the long-prompt prefill already
// runs its GEMMs on NVFP4 activations); the weights are exact. It is coarser: teacher-forced over
// eval_corpus at 32 packed rows, perplexity against one forward per row is 1.009x (int8: 1.000x).
#include "sparkinfer/kernels/prefill_ptq1_fp4.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cstdlib>
#include <mutex>

namespace sparkinfer { namespace kernels {

namespace {

constexpr int kBlk = 128;         // weights per PTQ1_0 block
constexpr int kBlkBytes = 28;
constexpr int kStepBlocks = 4;

template <typename OutT> __device__ __forceinline__ void put(OutT* y, size_t i, float v);
template <> __device__ __forceinline__ void put<__nv_bfloat16>(__nv_bfloat16* y, size_t i, float v) {
    y[i] = __float2bfloat16(v);
}


// A trit digit d (0..2, one a byte) is itself an e2m1 code, of value d / 2: two words of four
// digits make one operand register as they are, lo's in the low nibbles. A's scale of 2 makes the
// MMA's sum over the digits d, and the block's activation sum turns that into the trits' d - 1.
__device__ __forceinline__ unsigned fp4_pair(unsigned lo, unsigned hi) {
    return lo | (hi << 4);
}
__device__ __forceinline__ void ldsm4(const void* p, unsigned (&r)[4]) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
// A's per-16 scales are all 2.0 (see fp4_pair); B's are the activation's ue4m3s, lane
// t == 0 of each quad supplying its token's four.
__device__ __forceinline__ void mma_fp4(float (&d)[4], const unsigned (&a)[4], unsigned b0,
                                        unsigned b1, unsigned sfb) {
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32"
                 ".e2m1.e2m1.f32.ue4m3 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, "
                 "{0,0}, %11, {0,0};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "r"(0x40404040u), "r"(sfb));
}
int num_sms() {
    static const int sms = [] {
        int dev = 0, n = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n > 0 ? n : 170;
    }();
    return sms;
}
__device__ __forceinline__ void digits5(unsigned w, unsigned (&d)[5]) {
    unsigned ql = w & 0x00FF00FFu, qh = (w >> 8) & 0x00FF00FFu;
#pragma unroll
    for (int m = 0; m < 5; ++m) {
        const unsigned ul = ql * 3u, uh = qh * 3u;
        d[m] = __byte_perm(ul, uh, 0x7351);
        ql = ul & 0x00FF00FFu;
        qh = uh & 0x00FF00FFu;
    }
}


__device__ __forceinline__ void row_frags(unsigned wt, unsigned w45, unsigned w6, int t,
                                          unsigned (&f)[4][2]) {
    unsigned a[5], b[5];
    digits5(wt, a);
    digits5(w45, b);
    f[0][0] = a[0]; f[0][1] = a[1]; f[1][0] = a[2]; f[1][1] = a[3]; f[2][0] = a[4];
    const bool hm = t >> 1;
    f[2][1] = hm ? b[1] : b[0];
    f[3][0] = hm ? b[3] : b[2];
    const unsigned x = (w6 & 0xFFu) | ((w6 & 0xFF00u) << 8);
    const unsigned xm = (x * (t == 3 ? 9u : 1u)) & 0x00FF00FFu;
    const unsigned u0 = xm * 3u, u1 = (u0 & 0x00FF00FFu) * 3u;
    f[3][1] = __byte_perm(b[4], __byte_perm(u0, u1, 0x7531), t < 2 ? 0x3210 : 0x7654);
}


int row_splits(int n_rows, int nblk, int nmat) {
    const int sms = num_sms();
    const int ctas = (n_rows + 127) / 128 * nmat, nsteps = nblk / kStepBlocks;
    // Four waves and more (the LM head's 1940) already fill the device; a split there only
    // trades a wave-rounding sliver for a partial plane of vocab floats per row.
    if (ctas >= 4 * sms) return 1;
    int best = 1;
    long best_cost = (long)((ctas + sms - 1) / sms) * nsteps;
    for (int S = 2; S <= 8; ++S) {
        const int sps = (nsteps + S - 1) / S;
        if (sps < 4 || (S - 1) * sps >= nsteps) continue;
        const long cost = (long)((ctas * S + sms - 1) / sms) * sps;
        if (cost < best_cost) { best_cost = cost; best = S; }
    }
    return best;
}


constexpr int kCntSlots = 8, kCntTiles = 1024;
__device__ unsigned g_split_cnt[kCntSlots][kCntTiles];

unsigned* split_cnt_for(cudaStream_t st, int tiles) {
    static const bool on = [] {
        const char* e = getenv("SPARKINFER_ROWS_SPLIT_FUSED");
        return !(e && e[0] == '0');
    }();
    if (!on || tiles > kCntTiles) return nullptr;
    static std::mutex mu;
    static cudaStream_t owners[kCntSlots] = {};
    static int used = 0;
    static unsigned* base = nullptr;
    std::lock_guard<std::mutex> lk(mu);
    if (!base && cudaGetSymbolAddress(reinterpret_cast<void**>(&base), g_split_cnt) != cudaSuccess) {
        base = nullptr;
        return nullptr;
    }
    for (int i = 0; i < used; ++i)
        if (owners[i] == st) return base + (size_t)i * kCntTiles;
    if (used >= kCntSlots) return nullptr;
    owners[used] = st;
    return base + (size_t)(used++) * kCntTiles;
}


template <int NT, int WARPS, int KB, int ST, typename OutT, bool SPLIT = false>
__global__ void __launch_bounds__(WARPS * 32)
ptq1_fp4_rows_kernel(const unsigned char* __restrict__ xf, const unsigned char* __restrict__ xsf,
                     const float* __restrict__ xsum, const unsigned char* __restrict__ w0,
                     const unsigned char* __restrict__ w1, OutT* __restrict__ y0,
                     OutT* __restrict__ y1, int M, int N, int nblk, int ctas_per_mat,
                     float* __restrict__ part = nullptr, int sps = 0,
                     unsigned* __restrict__ cnt = nullptr) {
    constexpr int TOK = NT * 8;
    constexpr int ROWB = KB * 64 + 16;         // a token's FP4 bytes for one step, +16 (banks)
    constexpr int SROW = KB * 8;               // its ue4m3 scales for one step
    constexpr int WSEG = KB * kBlkBytes;       // one weight row's bytes per step
    constexpr int WCH = WSEG / 16;
    constexpr int WROWS = WARPS * 16;
    extern __shared__ __align__(16) unsigned char smem_mma[];
    unsigned char* sw = smem_mma;                                              // [ST][WROWS][WSEG]
    unsigned char* sx = sw + ST * WROWS * WSEG;                           // [ST][TOK][ROWB]
    unsigned char* ssf = sx + ST * TOK * ROWB;                             // [ST][TOK][SROW]
    float* ssum = reinterpret_cast<float*>(ssf + ST * TOK * SROW);         // [ST][KB][32]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int cta = blockIdx.x, mat = 0;
    const unsigned char* W = w0;
    OutT* y = y0;
    if (cta >= ctas_per_mat) { cta -= ctas_per_mat; W = w1; y = y1; mat = 1; }
    const int row0 = cta * WROWS;
    const int s_beg = SPLIT ? blockIdx.y * sps : 0;
    const int s_end = SPLIT ? min(nblk / KB, s_beg + sps) : nblk / KB;
    // A thread copies the same chunks every step; only b0 moves. Weight chunk i is row i / WCH,
    // unit i % WCH, and lands at i * 16 in its stage (WSEG == WCH * 16); activation chunk j is
    // token warp + WARPS * j, unit lane. So the offsets are formed once, not divided out per step.
    static_assert(KB * 8 == 32 && KB == 4, "issue map");
    constexpr bool FIT = WARPS == 8;   // the tile never runs past the matrix
    constexpr int NTH = WARPS * 32;
    constexpr int WJ = (WROWS * WCH + NTH - 1) / NTH;
    unsigned wsrc[WJ];
    unsigned wok = 0;   // !FIT: bit j set when chunk j is a row of this matrix
#pragma unroll
    for (int j = 0; j < WJ; ++j) {
        const int i = threadIdx.x + j * NTH, r = i / WCH, c = i - r * WCH;
        wsrc[j] = (unsigned)(r * nblk * kBlkBytes + c * 16);
        if (!FIT && i < WROWS * WCH && row0 + r < N) wok |= 1u << j;
    }
    const bool live = FIT || row0 + warp * 16 < N;
    const unsigned char* wbase = W + (size_t)row0 * nblk * kBlkBytes;
    // Tokens past M are zero in every stage from the start (their e2m1 codes and scales both),
    // and no copy ever lands on them.
    if (M < TOK) {
        for (int i = threadIdx.x; i < ST * TOK * (ROWB / 16); i += NTH) {
            const int row = i / (ROWB / 16), tok = row % TOK;
            if (tok >= M) reinterpret_cast<uint4*>(sx)[i] = make_uint4(0, 0, 0, 0);
        }
        for (int i = threadIdx.x; i < ST * TOK * (SROW / 16); i += NTH) {
            const int row = i / (SROW / 16), tok = row % TOK;
            if (tok >= M) reinterpret_cast<uint4*>(ssf)[i] = make_uint4(0, 0, 0, 0);
        }
    }
    auto issue_w = [&](int stp, int buf) {
        unsigned char* sws = sw + (size_t)buf * WROWS * WSEG;
        const unsigned char* wstep = wbase + (size_t)stp * KB * kBlkBytes;
#pragma unroll
        for (int j = 0; j < WJ; ++j) {
            const int i = threadIdx.x + j * NTH;
            if (FIT ? (WROWS * WCH % NTH == 0 || i < WROWS * WCH) : ((wok >> j) & 1u))
                __pipeline_memcpy_async(sws + i * 16, wstep + wsrc[j], 16);
        }
    };
    auto issue_x = [&](int stp, int buf) {
        const int b0 = stp * KB;
        unsigned char* sxs = sx + (size_t)buf * TOK * ROWB;
        for (int i = threadIdx.x; i < TOK * (KB * 4); i += NTH) {      // 16-byte chunks
            const int tok = i / (KB * 4), c = i - tok * (KB * 4);
            if (tok < M)
                __pipeline_memcpy_async(sxs + tok * ROWB + c * 16,
                                        xf + ((size_t)tok * nblk + b0) * 64 + c * 16, 16);
        }
        unsigned char* sss = ssf + (size_t)buf * TOK * SROW;
        for (int i = threadIdx.x; i < TOK * (SROW / 16); i += NTH) {
            const int tok = i / (SROW / 16), c = i - tok * (SROW / 16);
            if (tok < M)
                __pipeline_memcpy_async(sss + tok * SROW + c * 16,
                                        xsf + ((size_t)tok * nblk + b0) * 8 + c * 16, 16);
        }
        // The block sums sit [block][32 tokens], so a step's are one contiguous run.
        float* sms = ssum + (size_t)buf * KB * 32;
        for (int i = threadIdx.x; i < KB * 8; i += NTH)
            __pipeline_memcpy_async(sms + i * 4, xsum + (size_t)b0 * 32 + i * 4, 16);
    };
    auto issue = [&](int stp, int buf) {
        issue_w(stp, buf);
        issue_x(stp, buf);
        __pipeline_commit();
    };
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
    // The weights of the first stages are the kernel's own; only the activation comes from the
    // launch before it. So they are in flight before the wait (see pdl_wait), and each stage's
    // group commits once its activation is issued too.
#pragma unroll
    for (int s0 = 0; s0 < ST - 1; ++s0)
        if (s_beg + s0 < s_end) issue_w(s_beg + s0, s0);
#pragma unroll
    for (int s0 = 0; s0 < ST - 1; ++s0) {
        if (s_beg + s0 < s_end) issue_x(s_beg + s0, s0);
        __pipeline_commit();
    }
    const int o45 = 4 + (t & 1);
    // ldmatrix: matrix lane / 8 is (j, r) of the block's FP4 words, row lane % 8 the tile's token.
    const unsigned char* xl = sx + (size_t)(lane & 7) * ROWB + (lane >> 3) * 16;
    for (int stp = s_beg; stp < s_end; ++stp) {
        const int buf = (stp - s_beg) % ST;
        __pipeline_wait_prior(ST - 2);
        __syncthreads();   // this step's data is visible, and the buffer refilled next is idle
        {
            const int nx = stp + ST - 1;
            if (nx < s_end) issue(nx, (nx - s_beg) % ST);
            else __pipeline_commit();
        }
        if (!live) continue;
        const unsigned char* wa = sw + ((size_t)buf * WROWS + warp * 16 + g) * WSEG;
        const unsigned char* wbr = wa + 8 * WSEG;
        const unsigned* sss = reinterpret_cast<const unsigned*>(ssf + (size_t)buf * TOK * SROW);
        const float* smb = ssum + (size_t)buf * KB * 32 + 2 * t;
#pragma unroll
        for (int bb = 0; bb < KB; ++bb) {
            const unsigned* A = reinterpret_cast<const unsigned*>(wa + bb * kBlkBytes);
            const unsigned* B = reinterpret_cast<const unsigned*>(wbr + bb * kBlkBytes);
            const unsigned a6 = A[6], b6 = B[6];
            unsigned fa[4][2], fb[4][2];
            row_frags(A[t], A[o45], a6, t, fa);
            row_frags(B[t], B[o45], b6, t, fb);
            const float swA = __half2float(__ushort_as_half((unsigned short)(a6 >> 16)));
            const float swB = __half2float(__ushort_as_half((unsigned short)(b6 >> 16)));
            // k-step s of the int8 operand (its halves lo / hi) is register r = s & 1 of MMA
            // j = s >> 1 here, each word eight e2m1 codes: lo's four in the low nibbles.
            unsigned af[2][4];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                af[j][0] = fp4_pair(fa[2 * j][0], fa[2 * j][1]);
                af[j][1] = fp4_pair(fb[2 * j][0], fb[2 * j][1]);
                af[j][2] = fp4_pair(fa[2 * j + 1][0], fa[2 * j + 1][1]);
                af[j][3] = fp4_pair(fb[2 * j + 1][0], fb[2 * j + 1][1]);
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                unsigned bv[4];
                ldsm4(xl + ((size_t)buf * TOK + n * 8) * ROWB + bb * 64, bv);
                const unsigned* sc = sss + (size_t)(n * 8 + g) * (SROW / 4) + bb * 2;
                const unsigned sb0 = sc[0], sb1 = sc[1];
                float c[4] = {0.f, 0.f, 0.f, 0.f};
                mma_fp4(c, af[0], bv[0], bv[1], sb0);
                mma_fp4(c, af[1], bv[2], bv[3], sb1);
                const float2 xs = *reinterpret_cast<const float2*>(smb + bb * 32 + n * 8);
                acc[n][0] = fmaf(swA, c[0] - xs.x, acc[n][0]);
                acc[n][1] = fmaf(swA, c[1] - xs.y, acc[n][1]);
                acc[n][2] = fmaf(swB, c[2] - xs.x, acc[n][2]);
                acc[n][3] = fmaf(swB, c[3] - xs.y, acc[n][3]);
            }
        }
    }
    if (!live) return;
    const int rA = row0 + warp * 16 + g, rB = rA + 8;
    if (SPLIT) {
        float* pp = part + ((size_t)mat * gridDim.y + blockIdx.y) * M * N;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int t0 = n * 8 + 2 * t, t1 = t0 + 1;
            if (t0 < M) { pp[(size_t)t0 * N + rA] = acc[n][0]; pp[(size_t)t0 * N + rB] = acc[n][2]; }
            if (t1 < M) { pp[(size_t)t1 * N + rA] = acc[n][1]; pp[(size_t)t1 * N + rB] = acc[n][3]; }
        }
        // Every warp is live on a split launch (FIT), so all of them reach the barriers below.
        if (!FIT || !cnt) return;
        __shared__ unsigned s_last;
        __threadfence();   // this CTA's partials are visible before its arrival is counted
        __syncthreads();
        unsigned* c = cnt + (size_t)mat * ctas_per_mat + cta;
        if (threadIdx.x == 0) s_last = atomicAdd(c, 1u) == gridDim.y - 1;
        __syncthreads();
        if (!s_last) return;
        __threadfence();
        // Four consecutive rows a thread (N % 4 == 0 on a split launch), every split's load issued
        // before the first add.
        const size_t plane = (size_t)M * N;
        const float* pm = part + (size_t)mat * gridDim.y * plane;
        const int S = gridDim.y;
        for (int i = threadIdx.x; i < M * (WROWS / 4); i += NTH) {
            const int tok = i / (WROWS / 4);
            const size_t o = (size_t)tok * N + row0 + 4 * (i - tok * (WROWS / 4));
            float4 v[8];
#pragma unroll
            for (int q = 0; q < 8; ++q)
                if (q < S) v[q] = __ldcg(reinterpret_cast<const float4*>(pm + q * plane + o));
            float4 a = v[0];
#pragma unroll
            for (int q = 1; q < 8; ++q)
                if (q < S) {
                    a.x = __fadd_rn(a.x, v[q].x); a.y = __fadd_rn(a.y, v[q].y);
                    a.z = __fadd_rn(a.z, v[q].z); a.w = __fadd_rn(a.w, v[q].w);
                }
            put<OutT>(y, o, a.x); put<OutT>(y, o + 1, a.y);
            put<OutT>(y, o + 2, a.z); put<OutT>(y, o + 3, a.w);
        }
        if (threadIdx.x == 0) *c = 0u;
        return;
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int t0 = n * 8 + 2 * t, t1 = t0 + 1;
        if (t0 < M) {
            put<OutT>(y, (size_t)t0 * N + rA, acc[n][0]);
            put<OutT>(y, (size_t)t0 * N + rB, acc[n][2]);
        }
        if (t1 < M) {
            put<OutT>(y, (size_t)t1 * N + rA, acc[n][1]);
            put<OutT>(y, (size_t)t1 * N + rB, acc[n][3]);
        }
    }
}


template <int NT, int ST, typename OutT, bool SPLIT, int WARPS>
bool launch_fp4_rows_t(const unsigned char* xf, const unsigned char* xsf, const float* xsum,
                       const void* w0,
                       const void* w1, OutT* y0, OutT* y1, int m, int n_rows, int nblk, int S,
                       float* part, cudaStream_t st) {
    constexpr int KB = kStepBlocks;
    constexpr size_t shm = (size_t)ST * (WARPS * 16 * KB * kBlkBytes + NT * 8 * (KB * 64 + 16) +
                                         NT * 8 * KB * 8 + KB * 32 * 4);
    auto kern = ptq1_fp4_rows_kernel<NT, WARPS, KB, ST, OutT, SPLIT>;
    static const bool attr = [&] {
        return cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm) ==
               cudaSuccess;
    }();
    if (!attr) { cudaGetLastError(); return false; }
    const int ctas = (n_rows + 16 * WARPS - 1) / (16 * WARPS), nmat = w1 ? 2 : 1;
    const int nsteps = nblk / KB, sps = (nsteps + S - 1) / S;
    unsigned* cnt = SPLIT ? split_cnt_for(st, ctas * nmat) : nullptr;
    if (SPLIT && !cnt) return false;
    kern<<<dim3(ctas * nmat, SPLIT ? S : 1), dim3(WARPS * 32), shm, st>>>(
        xf, xsf, xsum, static_cast<const unsigned char*>(w0), static_cast<const unsigned char*>(w1),
        y0, y1, m, n_rows, nblk, ctas, part, sps, cnt);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <typename OutT>
bool launch_fp4_rows(const void* xf, const void* xsf, const float* xsum, const void* w0,
                     const void* w1, OutT* y0, OutT* y1, int m, int n_rows, int k, cudaStream_t st,
                     float* part, size_t part_cap) {
    if (m <= 16 || m > 32 || n_rows <= 0 || n_rows % 128 != 0 || k <= 0 ||
        k % (kBlk * kStepBlocks) != 0)
        return false;
    const int nblk = k / kBlk;
    const int nmat = w1 ? 2 : 1;
    const int S = row_splits(n_rows, nblk, nmat);
    if (S > 1 && (!part || (size_t)S * nmat * m * n_rows > part_cap || n_rows % 4 != 0))
        return false;
    const auto* X = static_cast<const unsigned char*>(xf);
    const auto* XS = static_cast<const unsigned char*>(xsf);
    // The int8 kernel's tiling: 13-warp CTAs where 128-row tiles need between one and two
    // waves (gate/up), else 128-row tiles split S ways.
    constexpr int kBalWarps = 13;
    const int groups = n_rows / 16 * nmat;
    const bool bal = S == 1 && groups > 8 * num_sms() &&
                     (groups + num_sms() - 1) / num_sms() <= kBalWarps;
    if (bal) return launch_fp4_rows_t<4, 2, OutT, false, kBalWarps>(X, XS, xsum, w0, w1, y0, y1, m, n_rows, nblk, 1, nullptr, st);
    if (S > 1) return launch_fp4_rows_t<4, 2, OutT, true, 8>(X, XS, xsum, w0, w1, y0, y1, m, n_rows, nblk, S, part, st);
    return launch_fp4_rows_t<4, 2, OutT, false, 8>(X, XS, xsum, w0, w1, y0, y1, m, n_rows, nblk, 1, nullptr, st);
}

}  // namespace

bool launch_ptq1_fp4_rows_bf16(const void* xf, const void* xsf, const float* xsum,
                               const void* w0, const void* w1, void* y0, void* y1, int m,
                               int n_rows, int k, cudaStream_t st, float* part, size_t part_cap) {
    if (!ptq1_fp4_gemm_supported(m, k)) return false;
    return launch_fp4_rows<__nv_bfloat16>(xf, xsf, xsum, w0, w1, static_cast<__nv_bfloat16*>(y0),
                                          static_cast<__nv_bfloat16*>(y1), m, n_rows, k, st, part,
                                          part_cap);
}

}}  // namespace sparkinfer::kernels
