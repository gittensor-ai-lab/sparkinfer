// ============================================================================
// Routed-MoE grouped int8 GEMM straight off the native GGUF expert weights.
//
// WHY THIS EXISTS
// ---------------
// The batched MoE prefill runs the routed experts as: dequantize the whole expert pool to int8
// rows (deq_rows_i8*), then a grouped int8 WMMA GEMM over pair tiles (pfm_moe_gemm_i8*). For
// Qwen3.6-35B-A3B (256 experts, per-expert ffn 512, H 2048) the dequant is a FIXED per-pass cost
// -- it materializes all 256 experts for every one of the 40 layers whatever the prompt length:
//
//     read  486 MB of Q4_K/Q5_K   (gate+up Q4_K, down Q5_K)
//     write 805 MB of int8        <- pure overhead
//     read  805 MB of int8 back   <- pure overhead, in the GEMM
//
// Measured on an RTX 5090 (nsys --cuda-graph-trace=node, main @ 37b3bb5): the dequant is 65.3 ms
// of the 237.8 ms 4k prefill (27.5%) and 32.4 ms of the 85.4 ms 512 prefill (38%), while at 32k it
// is only 4.9% -- it does not scale with N, so it dominates exactly where the prompt is short.
// Both dequant kernels already run at 0.8-1.6 TB/s, i.e. near DRAM peak: the traffic is the cost,
// not the code. The only way to win is to not move those bytes at all.
//
// THIS KERNEL decodes the expert weight to int8 inside the B-tile stage, so the 1.6 GB/layer
// round trip disappears and the GEMM reads only the 486 MB it fundamentally needs. It reads one
// whole 256-value super-block per row per stage, which is what makes the quantized read cheaper
// than the int8 one: at a 32-value granularity a Q4_K row costs 16 B of block header for every
// 16 B of nibbles, so it would be no smaller than int8. At 256 values the header amortizes to
// 4.5 bits/value.
//
// WHEN IT WINS. The materialize path decodes each weight element ONCE PER LAYER; this kernel
// decodes it once per M-tile that touches it. Tiles per expert = ceil(N*top_k/E/BM), so at
// BM=128 that is 1 at N=4k, 2 at 8k, but 8 at 32k -- where re-decoding loses to just
// materializing. The caller gates on N; above the gate nothing changes.
//
// BIT-IDENTICAL to the materialize path, deliberately:
//   * the per-row int8 scale is the one deq_rows_i8 itself produced (precomputed once at load and
//     handed in as row_scale), so it is identical by construction rather than by re-derivation;
//   * the value decode keeps the same evaluation order. Hoisting ds = d*s and dm = dmin*m to the
//     32-value sub-block is exact: the reference computes ((d*s)*nib) - (dmin*m) left to right, so
//     the hoisted form multiplies and subtracts the same floats. This is also what makes the
//     decode cheap (~5 ops/value instead of re-deriving d/dmin/s/m per value);
//   * same roundf(v * inv) and same (signed char)(int) cast, so every int8 byte matches;
//   * int32 WMMA accumulation is exact, and the epilogue applies sx * row_scale in the same order.
// ============================================================================
#include "sparkinfer/kernels/prefill_moe_q.h"
#include "sparkinfer/kernels/prefill_fp8.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include <cmath>
#include <cuda_pipeline.h>
#include <mma.h>

namespace sparkinfer {
namespace kernels {

namespace {

enum { QMQ_Q4_K = 12, QMQ_Q5_K = 13, QMQ_Q6_K = 14 };

constexpr int PF_QGROUP_MAX = 4;   // projections fusable into one grouped launch
// Block-slot budget the split-K fan-out aims at: ~3 passes over a 5090's 170 SMs x 3 blocks/SM.
// Measured: aiming at ONE pass picks 2 splits for the 312-tile FFN and loses to 4 (2211 vs 2550
// pp) -- oversubscribing hides latency and shortens the tail. A target, never a cap.
// Re-tuned with the m16n8k32 inner loop and the `red` epilogue below: each slice now retires
// faster and each slice's epilogue is cheaper, so the fan-out that best hides the tail moved up.
// Re-measured with Q6_K in the fused path (median of 5, prefill@128): 1536 -> 4182.6 pp, then a
// flat plateau -- 2048 -> 4215.6, 2560 -> 4223.6, 3072 -> 4221.9, i.e. anything from 2048 up is
// within run-to-run noise of anything else. 1536 is the only value that is clearly off it.
// A target, never a cap -- the per-GEMM slice count is still clamped by the partials plane
// budget and by K depth.
constexpr int QM_TARGET_BLOCKS = 2560;
constexpr int QM_BM = 128;    // pair rows per tile (must match the caller's tilemap)
constexpr int QM_BN = 64;     // weight rows (output channels) per block
constexpr int QM_SB = 256;    // values per GGUF super-block == the B-stage K depth
constexpr int QM_BK = 32;     // WMMA K step
constexpr int QM_LD = QM_SB + 16;   // Bs row stride: 16B-aligned for WMMA, +16 breaks bank conflicts
// Weight rows per block for the DENSE (non-routed) fused GEMM only -- the routed MoE kernels keep
// QM_BN. MEASURED, do not "optimize" this to 128: every block re-stages the whole A tile for its
// own N slice, so A traffic is (n_out / BND) * M * K and at muse's shapes that is 968 MB per layer
// against 272 MB of actual Q4_K weight -- 78% of everything the GEMM reads. Doubling BND halves
// that and still LOSES 3570 -> 3288 pp, because 43008 B of Bs only leaves two resident blocks.
// This kernel alternates a decode phase with an MMA phase and lives entirely on having a third
// block to run the other phase; bytes are not what it is short of.
constexpr int QM_BND = 64;

template <int QT>
__device__ __forceinline__ constexpr int qm_bs() {
    return (QT == QMQ_Q4_K) ? 144 : (QT == QMQ_Q5_K) ? 176 : 210;
}

__device__ __forceinline__ float qm_h2f(const unsigned char* p) {
    __half h; *((unsigned short*)&h) = *(const unsigned short*)p; return __half2float(h);
}

// 6-bit packed (scale, min) pair j of a K-quant super-block -- the unpack deq_rows_i8 uses.
__device__ __forceinline__ void qm_scale_min(const unsigned char* sc, int j, int* s, int* m) {
    if (j < 4) { *s = sc[j] & 63; *m = sc[j + 4] & 63; }
    else {
        *s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
        *m = (sc[j + 4] >> 4)  | ((sc[j]     >> 6) << 4);
    }
}

__device__ __forceinline__ void qm_cp16(void* dst, const void* src, bool pred) {
    if (pred) __pipeline_memcpy_async(dst, src, 16);
    else      *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
}

// A Q4_K value's int8 depends only on its 4-bit nibble and its 32-value sub-block's (ds, dm), so a
// sub-block has just SIXTEEN distinct results -- and the per-value decode was computing 32 of them.
// Tabulate the 16 once, packed one per byte across four registers.
//
// Why this is the lever: cuobjdump -sass on the per-value form counts 64 I2FP + 65 F2I per 64
// values decoded, and format conversion is quarter-rate on this part, so those two instructions
// alone cost more than every other instruction in the decode combined. Here `n` is a compile-time
// constant of the unrolled loop, so (float)n folds away and the I2F disappears entirely; the F2I
// survives but is paid 16 times per 32 values instead of 32.
//
// BIT-IDENTICAL by construction: entry n is the exact expression the per-value path evaluated for
// nibble n -- same ds * (float)n - dm, same * inv, same roundf, same truncating cast to a byte.
__device__ __forceinline__ void qm_lut16(float ds, float dm, float inv, unsigned* __restrict__ t) {
#pragma unroll
    for (int g = 0; g < 4; g++) {
        unsigned wv = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int q = (int)roundf((ds * (float)(g * 4 + j) - dm) * inv);
            wv |= ((unsigned)q & 0xFFu) << (8 * j);
        }
        t[g] = wv;
    }
}

// Look up FOUR table entries at once and return them already packed as four int8, which is the
// layout the shared-memory store wants -- so this also absorbs the per-value shift-and-or the old
// loop used to assemble its output word. `nib` carries the four 4-bit indices one per byte.
//
//   w    : | .. |n3| .. |n2| .. |n1| .. |n0|  ->  byte0 = n1:n0, byte2 = n3:n2
//   s    : the four indices packed one per NIBBLE, which is what prmt's selector wants
//   p01  : entries 0-7 gathered by (n & 7);  p23: the same selector against entries 8-15
//   last : picks p01 or p23 per byte by adding 4 to the selector nibble wherever n >= 8
// Ten integer instructions for four values, none of them float and none a conversion.
__device__ __forceinline__ unsigned qm_gather4(const unsigned* __restrict__ t, unsigned nib) {
    const unsigned w = nib | (nib >> 4);
    const unsigned s = __byte_perm(w, 0u, 0x4420u);
    const unsigned sl = s & 0x7777u;
    const unsigned p01 = __byte_perm(t[0], t[1], sl);
    const unsigned p23 = __byte_perm(t[2], t[3], sl);
    return __byte_perm(p01, p23, 0x3210u | ((s >> 1) & 0x4444u));
}

// Decode the 64 values of sub-block pair `j64` of one super-block into int8.
// Emits t = j64*64 + l (low nibbles, sub-block 2*j64) and t = j64*64 + 32 + l (high nibbles,
// sub-block 2*j64+1), i.e. 64 CONSECUTIVE int8 at dst + j64*64.
//
// Everything stays in registers: the 32 quant bytes come in as two 16-byte loads and the int8
// results are packed into words before being stored, so no byte array is ever addressed
// dynamically (a dynamically indexed local array would land in local memory).
template <int QT>
__device__ __forceinline__ void qm_decode_j64(const unsigned char* __restrict__ blk, int j64,
                                              float inv, signed char* __restrict__ dst) {
    const float bd = qm_h2f(blk), bdmin = qm_h2f(blk + 2);
    const unsigned char* sc = blk + 4;
    const unsigned char* qs = (QT == QMQ_Q4_K) ? (blk + 16) : (blk + 48);

    int s0, m0, s1, m1;
    qm_scale_min(sc, 2 * j64,     &s0, &m0);
    qm_scale_min(sc, 2 * j64 + 1, &s1, &m1);
    // Same left-to-right products the reference evaluates, hoisted out of the value loop:
    // reference is ((bd*s)*nib) - (bdmin*m), so these are the identical intermediate floats.
    const float ds0 = bd * s0, dm0 = bdmin * m0;
    const float ds1 = bd * s1, dm1 = bdmin * m1;

    const unsigned char* qb = qs + j64 * 32;
    const unsigned char* qh = blk + 16;               // Q5_K high-bit plane (unused for Q4_K)
    const int shl = 2 * j64, shh = 2 * j64 + 1;

    // Q6_K carries no per-sub-block min and packs its 6 bits as a nibble plane plus a 2-bit
    // plane, so it shares nothing with the K-quant path above and gets its own decode.
    //
    // ALIGNMENT is what kept this type out of the fused GEMM: a Q6_K super-block is 210 B, which
    // is even but is not a multiple of 4 or 16, so a block's base is only ever 2-byte aligned and
    // the uint4 loads the other types use would fault on every odd super-block. Every load here is
    // a ushort, which 210 always satisfies.
    //
    // Thread j64 owns half = j64>>1 and, within it, the sub = j64&1 half of ql. Those 32 ql bytes
    // carry quads (sub) and (sub+2) -- their low and high nibbles -- so each ql byte is read
    // exactly twice per super-block instead of the four times a naive split would cost, and each
    // thread still writes two contiguous 16-byte runs.
    //
    // BIT-IDENTICAL to dqr_q6k_val + the deq_rows_i8 quantize that produced `row_scale`:
    // the reference evaluates d * sc[is + 2*quad] * qv left to right, so hoisting ds = d * sc to
    // the 16-value group multiplies the identical floats, and the same roundf and truncating cast
    // follow.
    if (QT == QMQ_Q6_K) {
        const int hh = j64 >> 1, sub = j64 & 1;
        const unsigned char* q6 = blk + hh * 64 + sub * 32;
        const unsigned char* h6 = blk + 128 + hh * 32;
        const signed char*   s6 = reinterpret_cast<const signed char*>(blk + 192) + hh * 8;
        const float d6 = qm_h2f(blk + 208);
        const int qa = sub, qb = sub + 2;                    // this thread's two quads
        const float dsa[2] = { d6 * s6[0 + 2 * qa], d6 * s6[1 + 2 * qa] };
        const float dsb[2] = { d6 * s6[0 + 2 * qb], d6 * s6[1 + 2 * qb] };
        const int sha = 2 * qa, shb = 2 * qb;
        signed char* dA = dst + hh * 128 + sub * 32;         // quad `qa` output run
        signed char* dB = dst + hh * 128 + 64 + sub * 32;    // quad `qb` output run
#pragma unroll
        for (int c = 0; c < 2; c++) {                        // c IS the reference's `is` (l / 16)
            uint4 va, vb;
#pragma unroll
            for (int w = 0; w < 4; w++) {
                const unsigned char* qp = q6 + c * 16 + w * 4;
                const unsigned char* hp = h6 + c * 16 + w * 4;
                const unsigned qwd = (unsigned)(*(const unsigned short*)qp)
                                   | ((unsigned)(*(const unsigned short*)(qp + 2)) << 16);
                const unsigned hwd = (unsigned)(*(const unsigned short*)hp)
                                   | ((unsigned)(*(const unsigned short*)(hp + 2)) << 16);
                unsigned aw = 0, bw = 0;
#pragma unroll
                for (int b = 0; b < 4; b++) {
                    const int sb8 = 8 * b;
                    const unsigned by = (qwd >> sb8) & 0xFFu;
                    const unsigned hb = (hwd >> sb8) & 0xFFu;
                    const int va_q = (int)((by & 0xFu) | (((hb >> sha) & 3u) << 4)) - 32;
                    const int vb_q = (int)((by >> 4)   | (((hb >> shb) & 3u) << 4)) - 32;
                    const int ia = (int)roundf((dsa[c] * (float)va_q) * inv);
                    const int ib = (int)roundf((dsb[c] * (float)vb_q) * inv);
                    aw |= ((unsigned)(ia & 0xFF)) << sb8;
                    bw |= ((unsigned)(ib & 0xFF)) << sb8;
                }
                if (w == 0)      { va.x = aw; vb.x = bw; }
                else if (w == 1) { va.y = aw; vb.y = bw; }
                else if (w == 2) { va.z = aw; vb.z = bw; }
                else             { va.w = aw; vb.w = bw; }
            }
            *reinterpret_cast<uint4*>(dA + c * 16) = va;
            *reinterpret_cast<uint4*>(dB + c * 16) = vb;
        }
        return;
    }
    // Q4_K: 4 bits per value, so the whole sub-block collapses to a 16-entry table (see qm_lut16).
    // Q5_K's 5 bits would need 32 entries for 32 values -- no amortization -- so it keeps the
    // per-value decode below, unchanged.
    if (QT == QMQ_Q4_K) {
        unsigned ta[4], tb[4];
        qm_lut16(ds0, dm0, inv, ta);
        qm_lut16(ds1, dm1, inv, tb);
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const uint4 qv = *reinterpret_cast<const uint4*>(qb + c * 16);
            uint4 vlo, vhi;
#pragma unroll
            for (int w = 0; w < 4; w++) {
                const unsigned qwd = (w == 0) ? qv.x : (w == 1) ? qv.y : (w == 2) ? qv.z : qv.w;
                const unsigned alo = qm_gather4(ta, qwd & 0x0F0F0F0Fu);
                const unsigned ahi = qm_gather4(tb, (qwd >> 4) & 0x0F0F0F0Fu);
                if (w == 0)      { vlo.x = alo; vhi.x = ahi; }
                else if (w == 1) { vlo.y = alo; vhi.y = ahi; }
                else if (w == 2) { vlo.z = alo; vhi.z = ahi; }
                else             { vlo.w = alo; vhi.w = ahi; }
            }
            *reinterpret_cast<uint4*>(dst + j64 * 64 + c * 16) = vlo;
            *reinterpret_cast<uint4*>(dst + j64 * 64 + 32 + c * 16) = vhi;
        }
        return;
    }
    // One 16-byte load per 16 values and one 16-byte store, for both nibble halves. 16 B is the
    // widest load the block strides permit and they are all aligned: the Q4_K/Q5_K block sizes
    // (144/176) and the qs/qh offsets (16/48) and j64*32 are every one a multiple of 16, on a
    // cudaMalloc'd base. Loading a uint4 instead of 4 separate words cuts the decode's load
    // instruction count 4x, which matters because this kernel is decode-bound, not MMA-bound.
#pragma unroll
    for (int c = 0; c < 2; c++) {
        const uint4 qv = *reinterpret_cast<const uint4*>(qb + c * 16);
        uint4 hv = make_uint4(0u, 0u, 0u, 0u);
        if (QT == QMQ_Q5_K) hv = *reinterpret_cast<const uint4*>(qh + c * 16);
        uint4 vlo, vhi;
#pragma unroll
        for (int w = 0; w < 4; w++) {
            const unsigned qwd = (w == 0) ? qv.x : (w == 1) ? qv.y : (w == 2) ? qv.z : qv.w;
            const unsigned hwd = (w == 0) ? hv.x : (w == 1) ? hv.y : (w == 2) ? hv.z : hv.w;
            unsigned alo = 0, ahi = 0;
#pragma unroll
            for (int b = 0; b < 4; b++) {
                const int sb8 = 8 * b;
                const unsigned byte = (qwd >> sb8) & 0xFFu;
                int nlo = (int)(byte & 0xFu), nhi = (int)(byte >> 4);
                if (QT == QMQ_Q5_K) {
                    const unsigned hb = (hwd >> sb8) & 0xFFu;
                    nlo += ((hb >> shl) & 1u) ? 16 : 0;
                    nhi += ((hb >> shh) & 1u) ? 16 : 0;
                }
                const int qlo = (int)roundf((ds0 * nlo - dm0) * inv);
                const int qhi = (int)roundf((ds1 * nhi - dm1) * inv);
                alo |= ((unsigned)(qlo & 0xFF)) << sb8;
                ahi |= ((unsigned)(qhi & 0xFF)) << sb8;
            }
            if (w == 0)      { vlo.x = alo; vhi.x = ahi; }
            else if (w == 1) { vlo.y = alo; vhi.y = ahi; }
            else if (w == 2) { vlo.z = alo; vhi.z = ahi; }
            else             { vlo.w = alo; vhi.w = ahi; }
        }
        *reinterpret_cast<uint4*>(dst + j64 * 64 + c * 16) = vlo;
        *reinterpret_cast<uint4*>(dst + j64 * 64 + 32 + c * 16) = vhi;
    }
}

// ---------------------------------------------------------------------------------------------
// Staged weight fetch. The dense kernel below runs the B decode one super-block AHEAD of the MMA
// that consumes it, so the global read has a whole MMA phase to retire in. Splitting the decode
// into a load half and an arithmetic half is what makes that possible: the load half is issued
// before the K loop and only its 12 registers stay live across it, and the arithmetic half runs
// after. Q4_K needs exactly three uint4 -- the 16 B header and the thread's 32 quant bytes --
// and both are 16 B aligned because the super-block is 144 B. Q5_K/Q6_K keep the fused
// load-and-decode below; they are a small minority of this GGUF's tensors.
__device__ __forceinline__ float qm_h2f_u(unsigned u) {
    __half h; *((unsigned short*)&h) = (unsigned short)u; return __half2float(h);
}
__device__ __forceinline__ int qm_b(unsigned w, int i) { return (int)((w >> (8 * i)) & 0xFFu); }

// The same unpack as qm_scale_min, reading the 12 packed bytes out of the header registers.
// sc[0..3] are hdr.y, sc[4..7] are hdr.z, sc[8..11] are hdr.w.
__device__ __forceinline__ void qm_scale_min_r(uint4 hdr, int j, int* s, int* m) {
    if (j < 4) { *s = qm_b(hdr.y, j) & 63; *m = qm_b(hdr.z, j) & 63; }
    else {
        const int k = j - 4;
        *s = (qm_b(hdr.w, k) & 0xF) | ((qm_b(hdr.y, k) >> 6) << 4);
        *m = (qm_b(hdr.w, k) >> 4)  | ((qm_b(hdr.z, k) >> 6) << 4);
    }
}

template <int QT>
struct QmStage {
    const unsigned char* blk;      // Q5_K / Q6_K: decoded straight from global, as before
    uint4 hdr, q0, q1;             // Q4_K: the header and this thread's 32 quant bytes
};

template <int QT>
__device__ __forceinline__ void qm_stage_fetch(const unsigned char* __restrict__ blk, int j64,
                                               QmStage<QT>& s) {
    if constexpr (QT == QMQ_Q4_K) {
        // __ldcs = evict-first. The weight stream is read exactly once and never revisited, but it
        // is 14.2 GB per prefill pushed through a 96 MB L2 that other things genuinely want: the
        // k-tiled activation tile every N-tile re-reads, and the int32 split-K accumulator the
        // fused SwiGLU reads microseconds after the GEMM's red.add wrote it. Marking the weights
        // streaming stops them evicting either. Same bytes, same values -- only which cache line
        // survives changes.
        s.hdr = __ldcs(reinterpret_cast<const uint4*>(blk));
        const unsigned char* qb = blk + 16 + j64 * 32;
        s.q0 = __ldcs(reinterpret_cast<const uint4*>(qb));
        s.q1 = __ldcs(reinterpret_cast<const uint4*>(qb + 16));
    } else {
        s.blk = blk;
    }
}

// Bit-identical to qm_decode_j64: the same ds/dm products, the same 16-entry table and the same
// gather, only the bytes arrive in registers instead of being re-read here.
template <int QT>
__device__ __forceinline__ void qm_stage_decode(const QmStage<QT>& s, int j64, float inv,
                                                signed char* __restrict__ dst) {
    if constexpr (QT == QMQ_Q4_K) {
        const float bd = qm_h2f_u(s.hdr.x), bdmin = qm_h2f_u(s.hdr.x >> 16);
        int s0, m0, s1, m1;
        qm_scale_min_r(s.hdr, 2 * j64,     &s0, &m0);
        qm_scale_min_r(s.hdr, 2 * j64 + 1, &s1, &m1);
        const float ds0 = bd * s0, dm0 = bdmin * m0;
        const float ds1 = bd * s1, dm1 = bdmin * m1;
        unsigned ta[4], tb[4];
        qm_lut16(ds0, dm0, inv, ta);
        qm_lut16(ds1, dm1, inv, tb);
#pragma unroll
        for (int c = 0; c < 2; c++) {
            const uint4 qv = (c == 0) ? s.q0 : s.q1;
            uint4 vlo, vhi;
#pragma unroll
            for (int w = 0; w < 4; w++) {
                const unsigned qwd = (w == 0) ? qv.x : (w == 1) ? qv.y : (w == 2) ? qv.z : qv.w;
                const unsigned alo = qm_gather4(ta, qwd & 0x0F0F0F0Fu);
                const unsigned ahi = qm_gather4(tb, (qwd >> 4) & 0x0F0F0F0Fu);
                if (w == 0)      { vlo.x = alo; vhi.x = ahi; }
                else if (w == 1) { vlo.y = alo; vhi.y = ahi; }
                else if (w == 2) { vlo.z = alo; vhi.z = ahi; }
                else             { vlo.w = alo; vhi.w = ahi; }
            }
            *reinterpret_cast<uint4*>(dst + j64 * 64 + c * 16) = vlo;
            *reinterpret_cast<uint4*>(dst + j64 * 64 + 32 + c * 16) = vhi;
        }
    } else {
        qm_decode_j64<QT>(s.blk, j64, inv, dst);
    }
}

// 3 resident blocks/SM. Occupancy is what this kernel lives on: it alternates a decode phase
// (global reads + ALU) with an MMA phase, so the other resident blocks are what keep the tensor
// cores fed while one block is decoding. BN=64 rather than 128 is chosen for exactly this -- it puts
// Bs at 17408 B so three blocks fit the 100 KB an sm_120 SM has, and it halves the accumulator
// fragments so ptxas can reach the matching register budget. The cost is that the A tile is re-read
// by twice as many blocks, which is cheap because A is small enough to stay L2-resident.
template <int QT, bool A_INDIRECT, bool C_SCATTER>
__global__ __launch_bounds__(256, 3) void pfm_moe_gemm_qi8_kernel(
        const signed char* __restrict__ A_i8, const float* __restrict__ sx,
        const unsigned char* __restrict__ W_q, const float* __restrict__ row_scale,
        const int* __restrict__ pair_tok, const float* __restrict__ pair_w,
        const int* __restrict__ offsets, const int* __restrict__ tilemap,
        const int* __restrict__ d_ntiles,
        __nv_bfloat16* __restrict__ C, float* __restrict__ out_f32,
        int N, int K) {
    using namespace nvcuda;
    constexpr int BS = qm_bs<QT>();
    const int tile = blockIdx.y;
    if (tile >= d_ntiles[0]) return;
    const int e   = tilemap[2 * tile];
    const int mt  = tilemap[2 * tile + 1];
    const int p0  = offsets[e] + mt * QM_BM;
    const int cnt = offsets[e + 1] - offsets[e];
    const int M   = min(QM_BM, cnt - mt * QM_BM);
    const int n0  = blockIdx.x * QM_BN;
    const int nsb = K >> 8;

    __shared__ __align__(16) signed char Bs[QM_BN][QM_LD];
    __shared__ __align__(16) signed char As[2][QM_BM][QM_BK];
    __shared__ int s_tok[QM_BM];

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 3, wn = warp >> 2;

    // This block's 128 weight rows and their int8 row scales.
    const float* swe = row_scale + (size_t)e * N;

    for (int r = tid; r < QM_BM; r += blockDim.x)
        s_tok[r] = (r < M) ? (A_INDIRECT ? pair_tok[p0 + r] : (p0 + r)) : -1;

    // Decode assignment: 4 threads per weight row, one 64-value sub-block pair (j64) each.
    const int dr = tid >> 2, dj = tid & 3;
    const int dgn = n0 + dr;
    const bool drow_ok = dgn < N;
    const unsigned char* drow = W_q + ((size_t)e * N + (drow_ok ? dgn : 0)) * (size_t)nsb * BS;
    const float dscale = drow_ok ? swe[dgn] : 0.f;
    const float dinv = (dscale > 0.f) ? (1.f / dscale) : 0.f;

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> cf[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) wmma::fill_fragment(cf[i][j], 0);

    auto stageA = [&](int buf, int k0) {
        for (int idx = tid; idx < QM_BM * 2; idx += blockDim.x) {
            const int r = idx >> 1, c16 = (idx & 1) * 16;
            const int arow = s_tok[r];
            qm_cp16(&As[buf][r][c16], &A_i8[(size_t)max(arow, 0) * K + k0 + c16],
                    arow >= 0 && (k0 + c16) < K);
        }
        __pipeline_commit();
    };

    __syncthreads();          // s_tok published before stageA reads it
    stageA(0, 0);
    int abuf = 0;

    for (int sb = 0; sb < nsb; sb++) {
        __syncthreads();      // previous super-block's MMA finished reading Bs
        if (drow_ok) {
            const unsigned char* blk = drow + (size_t)sb * BS;
            qm_decode_j64<QT>(blk, dj, dinv, &Bs[dr][0]);
        } else if (dj == 0) {
#pragma unroll
            for (int i = 0; i < QM_SB / 16; i++)
                *reinterpret_cast<uint4*>(&Bs[dr][i * 16]) = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();      // Bs ready

        for (int kk = 0; kk < QM_SB; kk += QM_BK) {
            const int knext = sb * QM_SB + kk + QM_BK;
            if (knext < K) stageA(abuf ^ 1, knext);
            __pipeline_wait_prior(knext < K ? 1 : 0);
            __syncthreads();
#pragma unroll
            for (int k16 = 0; k16 < QM_BK; k16 += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af[2];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf[2];
#pragma unroll
                for (int i = 0; i < 2; i++)
                    wmma::load_matrix_sync(af[i], &As[abuf][wm * 32 + i * 16][k16], QM_BK);
#pragma unroll
                for (int j = 0; j < 2; j++)
                    wmma::load_matrix_sync(bf[j], &Bs[wn * 32 + j * 16][kk + k16], QM_LD);
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 2; j++) wmma::mma_sync(cf[i][j], af[i], bf[j], cf[i][j]);
            }
            __syncthreads();
            abuf ^= 1;
        }
    }

    // Epilogue staging reuses Bs (the K loop is done with it).
    __syncthreads();
    int* Cs = reinterpret_cast<int*>(&Bs[0][0]);
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int rm0 = wm * 32 + i * 16, gn0 = n0 + wn * 32 + j * 16;
            wmma::store_matrix_sync(&Cs[warp * 256], cf[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            for (int el = lane; el < 256; el += 32) {
                const int r = el >> 4, cc = el & 15;
                const int rm = rm0 + r, rn = gn0 + cc;
                if (rm < M && rn < N) {
                    const int p = p0 + rm;
                    const float v = (float)Cs[warp * 256 + el]
                                    * sx[A_INDIRECT ? s_tok[rm] : p] * swe[rn];
                    if (C_SCATTER) atomicAdd(&out_f32[(size_t)pair_tok[p] * N + rn], v * pair_w[p]);
                    else           C[(size_t)p * N + rn] = __float2bfloat16(v);
                }
            }
            __syncwarp();
        }
    }
}

template <int QT>
void dispatch_qi8(const signed char* A_i8, const float* sx, const void* W_q, const float* row_scale,
                  const int* pair_tok, const float* pair_w, const int* offsets,
                  const int* tilemap, const int* d_ntiles,
                  __nv_bfloat16* C, float* out_f32, int n_out, int K, int max_tiles,
                  bool a_indirect, bool c_scatter, cudaStream_t stream) {
    dim3 grid((n_out + QM_BN - 1) / QM_BN, max_tiles);
    const auto* W = reinterpret_cast<const unsigned char*>(W_q);
    if (a_indirect && !c_scatter)
        pfm_moe_gemm_qi8_kernel<QT, true, false><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
    else if (!a_indirect && c_scatter)
        pfm_moe_gemm_qi8_kernel<QT, false, true><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
    else if (a_indirect && c_scatter)
        pfm_moe_gemm_qi8_kernel<QT, true, true><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
    else
        pfm_moe_gemm_qi8_kernel<QT, false, false><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
}

// ============================================================================
// BM=16 variant: the same native-quant fused-decode B-tile technique, tiled for the short-N
// path (moe_serial's regime, N<=512) where pairs/expert average N*top_k/E — far below a
// BM=128 tile's fill (prefill_moe.cu's own pfm_moe_gemm_i8_bm16_kernel exists for exactly this
// reason). Before this, N<=512 could only reach the routed GEMM via deq_rows_i8_vec_gather* ->
// materialize -> pfm_moe_gemm_i8_bm16, paying the SAME fixed per-layer dequant this file's BM=128
// kernel already eliminates for N>512 -- measured 40%+ of the ctx=512 prefill wall (nsys,
// --cuda-graph-trace=node, frontier c66e87f). BM=16 has no M-subdivision (one 16-row wmma tile
// covers the whole M dimension), so the warp/epilogue layout collapses to prefill_moe.cu's bm16
// kernel's shape (8 warps, each owns one 16-col N-tile, BN=128) rather than the BM=128 kernel's
// 2x2 warp grid. The decode primitive (qm_decode_j64) is UNCHANGED and reused verbatim: BN doubling
// (64->128) is covered by assigning 2 threads per weight row instead of 4 (dr = tid>>1 spans all
// 128 rows; each thread decodes 2 of the super-block's 4 j64 groups instead of 1), so the same
// bit-identical per-value decode still runs, just distributed differently across threads.
constexpr int QM_BM16 = 16;
constexpr int QM_BN16 = 128;

template <int QT, bool A_INDIRECT, bool C_SCATTER>
__global__ __launch_bounds__(256, 2) void pfm_moe_gemm_qi8_bm16_kernel(
        const signed char* __restrict__ A_i8, const float* __restrict__ sx,
        const unsigned char* __restrict__ W_q, const float* __restrict__ row_scale,
        const int* __restrict__ pair_tok, const float* __restrict__ pair_w,
        const int* __restrict__ offsets, const int* __restrict__ tilemap,
        const int* __restrict__ d_ntiles,
        __nv_bfloat16* __restrict__ C, float* __restrict__ out_f32,
        int N, int K) {
    using namespace nvcuda;
    constexpr int BS = qm_bs<QT>();
    const int tile = blockIdx.y;
    if (tile >= d_ntiles[0]) return;
    const int e   = tilemap[2 * tile];
    const int mt  = tilemap[2 * tile + 1];
    const int p0  = offsets[e] + mt * QM_BM16;
    const int cnt = offsets[e + 1] - offsets[e];
    const int M   = min(QM_BM16, cnt - mt * QM_BM16);
    const int n0  = blockIdx.x * QM_BN16;
    const int nsb = K >> 8;

    __shared__ __align__(16) signed char Bs[QM_BN16][QM_LD];
    __shared__ __align__(16) signed char As[2][QM_BM16][QM_BK];
    __shared__ int s_tok[QM_BM16];

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wn = warp;                    // 0..7 -> 16-col N tile (single M tile, no wm)

    const float* swe = row_scale + (size_t)e * N;

    for (int r = tid; r < QM_BM16; r += blockDim.x)
        s_tok[r] = (r < M) ? (A_INDIRECT ? pair_tok[p0 + r] : (p0 + r)) : -1;

    // 2 threads per weight row (covers all QM_BN16=128 rows); each decodes 2 of the 4 j64 groups
    // of the super-block (dj, dj+2) instead of the BM=128 kernel's 1-of-4 (4 threads/row). Same
    // qm_decode_j64 primitive, same bit-identical per-value math -- only the thread->work mapping
    // changes to cover twice the rows with the same 256 threads.
    const int dr = tid >> 1, dj = tid & 1;
    const int dgn = n0 + dr;
    const bool drow_ok = dgn < N;
    const unsigned char* drow = W_q + ((size_t)e * N + (drow_ok ? dgn : 0)) * (size_t)nsb * BS;
    const float dscale = drow_ok ? swe[dgn] : 0.f;
    const float dinv = (dscale > 0.f) ? (1.f / dscale) : 0.f;

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> cf;
    wmma::fill_fragment(cf, 0);

    auto stageA = [&](int buf, int k0) {
        for (int idx = tid; idx < QM_BM16 * 2; idx += blockDim.x) {
            const int r = idx >> 1, c16 = (idx & 1) * 16;
            const int arow = s_tok[r];
            qm_cp16(&As[buf][r][c16], &A_i8[(size_t)max(arow, 0) * K + k0 + c16],
                    arow >= 0 && (k0 + c16) < K);
        }
        __pipeline_commit();
    };

    __syncthreads();          // s_tok published before stageA reads it
    stageA(0, 0);
    int abuf = 0;

    for (int sb = 0; sb < nsb; sb++) {
        __syncthreads();      // previous super-block's MMA finished reading Bs
        if (drow_ok) {
            const unsigned char* blk = drow + (size_t)sb * BS;
            qm_decode_j64<QT>(blk, dj,     dinv, &Bs[dr][0]);
            qm_decode_j64<QT>(blk, dj + 2, dinv, &Bs[dr][0]);
        } else if (dj == 0) {
#pragma unroll
            for (int i = 0; i < QM_SB / 16; i++)
                *reinterpret_cast<uint4*>(&Bs[dr][i * 16]) = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();      // Bs ready

        for (int kk = 0; kk < QM_SB; kk += QM_BK) {
            const int knext = sb * QM_SB + kk + QM_BK;
            if (knext < K) stageA(abuf ^ 1, knext);
            __pipeline_wait_prior(knext < K ? 1 : 0);
            __syncthreads();
#pragma unroll
            for (int k16 = 0; k16 < QM_BK; k16 += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf;
                wmma::load_matrix_sync(af, &As[abuf][0][k16], QM_BK);
                wmma::load_matrix_sync(bf, &Bs[wn * 16][kk + k16], QM_LD);
                wmma::mma_sync(cf, af, bf, cf);
            }
            __syncthreads();
            abuf ^= 1;
        }
    }

    // Epilogue staging reuses Bs (the K loop is done with it).
    __syncthreads();
    int* Cs = reinterpret_cast<int*>(&Bs[0][0]);
    {
        const int gn0 = n0 + wn * 16;
        wmma::store_matrix_sync(&Cs[warp * 256], cf, 16, wmma::mem_row_major);
        __syncwarp();
        for (int el = lane; el < 256; el += 32) {
            const int r = el >> 4, cc = el & 15;
            const int rm = r, rn = gn0 + cc;
            if (rm < M && rn < N) {
                const int p = p0 + rm;
                const float v = (float)Cs[warp * 256 + el]
                                * sx[A_INDIRECT ? s_tok[rm] : p] * swe[rn];
                if (C_SCATTER) atomicAdd(&out_f32[(size_t)pair_tok[p] * N + rn], v * pair_w[p]);
                else           C[(size_t)p * N + rn] = __float2bfloat16(v);
            }
        }
        __syncwarp();
    }
}

template <int QT>
void dispatch_qi8_bm16(const signed char* A_i8, const float* sx, const void* W_q, const float* row_scale,
                       const int* pair_tok, const float* pair_w, const int* offsets,
                       const int* tilemap, const int* d_ntiles,
                       __nv_bfloat16* C, float* out_f32, int n_out, int K, int max_tiles,
                       bool a_indirect, bool c_scatter, cudaStream_t stream) {
    dim3 grid((n_out + QM_BN16 - 1) / QM_BN16, max_tiles);
    const auto* W = reinterpret_cast<const unsigned char*>(W_q);
    if (a_indirect && !c_scatter)
        pfm_moe_gemm_qi8_bm16_kernel<QT, true, false><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
    else if (!a_indirect && c_scatter)
        pfm_moe_gemm_qi8_bm16_kernel<QT, false, true><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
    else if (a_indirect && c_scatter)
        pfm_moe_gemm_qi8_bm16_kernel<QT, true, true><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
    else
        pfm_moe_gemm_qi8_bm16_kernel<QT, false, false><<<grid, 256, 0, stream>>>(
            A_i8, sx, W, row_scale, pair_tok, pair_w, offsets, tilemap, d_ntiles, C, out_f32, n_out, K);
}

// ---------------------------------------------------------------------------
// Dense (non-routed) fused-decode int8 GEMM: C[M,N] = A_i8[M,K] @ dequant(W_q[N,K])^T.
// The same qm_decode_j64 B-stage + int8 WMMA as pfm_moe_gemm_qi8_kernel, with the expert /
// pair-routing indirection stripped away: one dense weight [N,K], M contiguous activation rows,
// C written directly (no gather, no scatter). Muse Glimmer's dense prefill uses this to consume
// its native Q4_K/Q5_K attn + FFN gate/up weights straight from VRAM, skipping the per-layer int8
// materialize (dequant -> write W_i8 -> read W_i8 back in the GEMM) that launch_prefill_gemm_i8
// pays. The per-output-row scale is the one launch_gguf_dequant_rows_i8 itself produced,
// precomputed once at load (Qwen35LayerWeights::*_rs), so every int8 byte -- and thus the whole
// accumulation -- matches the materialize path by construction. BM=128 so each weight row is
// decoded once per M-tile: at prefill's M=128 that is exactly once (one M-tile).
// Grouped form: several projections sharing the activation (same Mtot, same K), fused into ONE
// launch. At prefill's M=128 the grid is (ceil(N/QM_BN), 1), so a projection gets ceil(N/64) CTAs
// -- 64 for a 4096-wide q or gate, but only 4 for a 256-wide k or v. All are far under a 5090's
// 170 SMs, so each launch costs a full CTA-duration however little work it carries: k and v each
// burn a whole wave for four CTAs. Muse Glimmer's q/gate/k/v all read the same xn and are
// mutually independent, so co-scheduling their tiles in one grid turns four waves into one.
// blockIdx.x becomes a global tile id resolved against the descriptor; every other line of the
// kernel is untouched, so each output tile computes exactly what its own launch would have.
struct PfQGroup {
    const unsigned char* W[PF_QGROUP_MAX];
    const float*         rs[PF_QGROUP_MAX];
    __nv_bfloat16*       C[PF_QGROUP_MAX];
    int                  N[PF_QGROUP_MAX];
    int                  first[PF_QGROUP_MAX];   // prefix sum of each group's tile count
    int                  poff[PF_QGROUP_MAX];    // split-K: base of each group in `partials`
    int                  ngroup;
};

// ---------------------------------------------------------------------------
// mma.m16n8k32.s8 path.
//
// wmma's 16x16x16 int8 fragment can only reach a K=16 hardware shape: cuobjdump on this kernel
// shows every mma_sync lowering to IMMA.16816.S8.S8. sm_120 also has IMMA.16832 -- the same
// tensor-core throughput per instruction-pair, but twice the K per instruction -- so issuing
// mma.m16n8k32 directly halves the MMA instruction stream for identical arithmetic.
//
// Why that is the lever here: deleting the MMA phase outright saves 6.5 ms of a 33.8 ms prefill,
// but DOUBLING the MMA instruction count costs 10.8 ms. The stream is issue-bound, not
// throughput-bound, so instruction count is the thing that matters and halving it is worth far
// more than the delete-probe alone suggests.
//
// BIT-IDENTICAL: int32 accumulation is exact and order-independent, and the int8 operands are the
// same bytes out of the same As/Bs tiles, so only the grouping of the adds changes.
//
// Costs nothing that the occupancy law taxes: the accumulator is 8 m16n8 tiles x 4 s32 = 32
// registers per lane, exactly what the four 16x16 wmma accumulators used, and B is loaded one
// 8-column tile at a time so at most 4+2 operand registers are live. Shared memory is untouched.
// The split-K epilogue never reads the accumulated value back, but atomicAdd() is an
// exchange: cuobjdump shows it lowering to ATOMG.E.ADD.STRONG.GPU, which allocates a destination
// register and carries strong-GPU ordering. `red` is the fire-and-forget form -- no return, and
// relaxed ordering is all this needs, since the split-K reduce that consumes the plane is a
// separate kernel launch on the same stream and the launch boundary already orders it.
// Same int32 adds in the same (order-independent) sum, so the result is unchanged.
__device__ __forceinline__ void qm_red_add(int* p, int v) {
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

// Takes the shared-window address rather than a generic pointer: converting one per use is what
// the K loop cannot afford -- see the a_sm/b_sm bases below.
__device__ __forceinline__ void qm_ldsm_x4(unsigned (&r)[4], unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void qm_ldsm_x2(unsigned (&r)[2], const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void qm_mma_16832(int (&d)[4], const unsigned (&a)[4],
                                             unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// SPARKINFER_MUSE_MMA_K32=0 restores the wmma 16x16x16 inner loop (A/B; bit-identical either way).
static int qm_mma_k32() {
    static int e = -1;
    if (e < 0) { const char* v = getenv("SPARKINFER_MUSE_MMA_K32"); e = (v && v[0] == '0') ? 0 : 1; }
    return e;
}

// SPLIT=true: one blockIdx dimension owns a slice of the K super-blocks and contributes its raw
// int32 tile to `partials`; pf_dense_splitk_reduce_kernel applies sx*row_scale once. int32
// accumulation is exact and associative, so the total is BIT-IDENTICAL to the unsplit kernel --
// only the block count and the summation order change.
//
// At prefill's M=128 grid.y collapses to 1, leaving a grid of just N/QM_BN blocks: 104 for the
// 6656-wide o/down, 312 for the 19968-wide FFN pair, against ~510 block slots on a 5090. ncu on
// the unsplit kernel: 0.13 waves/SM, 16.7% achieved occupancy, DRAM 6.0%, SM throughput 13.1% --
// the kernel is neither bandwidth- nor compute-bound, it simply is not on the machine.
//
// HOW THE SLICES COMBINE. `atomic_acc=0` is the original scheme: each slice STORES its own tile
// into a private [split][m][n] plane and the reduce pass sums the planes. That plane array is
// `splits` times the output, and at muse's shapes it is the single largest DRAM stream in the
// whole prefill -- 294 MB per layer written and read back, against 272 MB of actual Q4_K weights.
// `atomic_acc=1` (the default) instead red.global.add's into ONE [m][n] accumulator, so the
// footprint drops by `splits` and the reduce degenerates to a scale-and-store. Combined with the
// K slice living on blockIdx.x -- so all slices of a tile are dispatched back-to-back rather than
// a whole grid apart -- a tile's 32 KB of accumulator is written, re-hit and consumed inside L2.
// Integer adds are exact and order-independent, so the accumulated value is the same either way.
//
// WHY TWO RESIDENT BLOCKS AND NOT THREE. The routed kernel above takes three, on the reasoning
// that one block's MMA phase covers another block's decode phase. That reasoning does not survive
// measurement here: all resident blocks march through the same barrier at the same rate, so they
// tend to sit in the SAME phase, and the weight read that was supposed to be hidden is instead
// serialised against every block's MMA at once. Giving the block a second Bs plane and letting it
// hide its OWN weight fetch -- issue super-block sb+1's loads, run sb's eight MMA steps, then do
// the decode arithmetic -- is worth more than the third block: 3 blocks 4373 pp vs 2 blocks
// pipelined 4482 pp at prefill@128. The freed budget is what pays for it (43008 B of smem and 128
// registers against 25600 B and 80), and the decode split that makes it possible is qm_stage_fetch
// / qm_stage_decode above. Also measured, and NOT kept: a third As plane so the K loop carries one
// barrier instead of two (4452 pp -- barriers are not the bottleneck).
template <int QT, bool GROUPED, bool SPLIT = false, bool K32 = true>
__global__ __launch_bounds__(256, 2) void pf_dense_gemm_qi8_kernel_g(
        const signed char* __restrict__ A_i8, const float* __restrict__ sx,
        const unsigned char* __restrict__ W_q_arg, const float* __restrict__ row_scale_arg,
        __nv_bfloat16* __restrict__ C_arg, int Mtot, int N_arg, int K, PfQGroup gd,
        int* __restrict__ partials = nullptr, int sb_per_split = 0, int atomic_acc = 0,
        const signed char* __restrict__ A_pack = nullptr) {
    using namespace nvcuda;
    constexpr int BS = qm_bs<QT>();
    // Split-K takes blockIdx.x for the K slice and shifts the tile/M indices up one dimension.
    // Blocks are dispatched x-fastest, so the slices that share an output tile now issue together
    // and their accumulator lines stay resident between them.
    const int zsl  = SPLIT ? (int)blockIdx.x : 0;
    const int btil = SPLIT ? (int)blockIdx.y : (int)blockIdx.x;
    const int bmt  = SPLIT ? (int)blockIdx.z : (int)blockIdx.y;
    const int p0 = bmt * QM_BM;
    const int M  = min(QM_BM, Mtot - p0);
    if (M <= 0) return;
    const unsigned char* W_q = W_q_arg;
    const float* row_scale = row_scale_arg;
    __nv_bfloat16* C = C_arg;
    int N = N_arg;
    int xtile = btil;
    int grp = 0;
    if (GROUPED) {
        #pragma unroll
        for (int i = 1; i < PF_QGROUP_MAX; i++)
            if (i < gd.ngroup && btil >= gd.first[i]) grp = i;
        W_q = gd.W[grp]; row_scale = gd.rs[grp]; C = gd.C[grp]; N = gd.N[grp];
        xtile = btil - gd.first[grp];
    }
    // Every group owns a disjoint [split][m][n] region of `partials`, so a grouped split-K launch
    // needs its base here -- the one thing that kept the grouped launch from splitting.
    const size_t pbase = GROUPED ? (size_t)gd.poff[grp] : 0;
    const int n0  = xtile * QM_BND;
    const int nsb = K >> 8;

    __shared__ __align__(16) signed char Bs[2][QM_BND][QM_LD];
    __shared__ __align__(16) signed char As[2][QM_BM][QM_BK];

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 3, wn = warp >> 2;

    // Direct activation rows: row r of this M-tile IS global token p0 + r, so it is computed in
    // stageA rather than staged through smem (the pair_tok gather this kernel was cloned from is
    // what needed a table). Saves 512 B of smem and the barrier that published it.

    // Decode assignment: 4 threads per weight row, one 64-value sub-block pair (j64) each.
    const int dr = tid >> 2, dj = tid & 3;
    const int dgn = n0 + dr;
    const bool drow_ok = dgn < N;
    const unsigned char* drow = W_q + (size_t)(drow_ok ? dgn : 0) * (size_t)nsb * BS;
    const float dscale = drow_ok ? row_scale[dgn] : 0.f;
    const float dinv = (dscale > 0.f) ? (1.f / dscale) : 0.f;

    // K32: eight m16n8 int32 tiles (2 M-subtiles x 4 N-subtiles), 4 regs each = the same 32
    // accumulator registers per lane the four 16x16 wmma fragments occupied.
    int acc[2][4][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> cf[2][2];
    if constexpr (K32) {
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0;
    } else {
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) wmma::fill_fragment(cf[i][j], 0);
    }

    // A staging, straight-line. QM_BM*2 == the 256 threads this kernel is always launched with, so
    // every thread owns exactly one 16 B chunk and the strided loop has exactly one iteration --
    // but blockDim.x is not a compile-time constant, so ptxas could not see that and kept a real
    // loop, with its own BSSY/BSYNC pair, induction variable and bound test, wrapped around a
    // single cp.async. That ran 8 times per super-block. The row, the column and the source row
    // base are all loop-invariant too, so only `+ k0` and the buffer offset move.
    static_assert(QM_BM * 2 == 256, "stageA assumes one 16 B chunk per thread at 256 threads");
    const int a_r   = tid >> 1;
    const int a_c16 = (tid & 1) * 16;
    const int a_row = (a_r < M) ? (p0 + a_r) : -1;
    // A tile source. In the row-major activation the 128 chunks one stage wants are K bytes apart,
    // so a warp's 16 rows issue 16 sector requests for 512 B. `A_pack` is the same int8 bytes in
    // the k-tiled [k/32][row][32] layout the quantizer also emits (qr_pack_off, prefill_quant_rows.cu):
    // there a stage is one contiguous 4 KB block, i.e. 4 line requests for the same 512 B per warp.
    // Nothing else changes -- same cp.async count, same bytes, same destination -- so the int8 the
    // MMA consumes is bit-identical and A_pack == nullptr keeps the original addressing.
    const bool apk = A_pack != nullptr;
    const signed char* const a_src0 =
        apk ? (A_pack + (size_t)p0 * 32 + (size_t)tid * 16)
            : (A_i8 + (size_t)max(a_row, 0) * K + a_c16);
    // Bytes between consecutive QM_BK-wide stages. Both layouts advance by a constant, so the K
    // loop carries a running pointer and one add either way.
    const size_t a_step = apk ? (size_t)Mtot * 32 : (size_t)QM_BK;
    // XOR swizzle on the 16 B chunk index. As rows are QM_BK = 32 B apart, so the eight rows one
    // ldmatrix phase gathers land on only 16 of the 32 banks and every A-side phase pays a 2-way
    // conflict. Flipping the chunk order on rows whose bit 2 is set spreads those eight rows over
    // all 32 banks. It costs nothing -- no padding, no extra smem -- and it is a pure permutation
    // of where a byte lives, undone by reading with the same XOR (see a_sm below). Only the K32
    // path reads As through ldmatrix; the wmma fallback walks the row linearly, so it keeps the
    // natural layout.
    const int a_swz = K32 ? (16 * ((a_r >> 2) & 1)) : 0;
    signed char* const a_dst0 = &As[0][a_r][a_c16 ^ a_swz];

    // The column bound test this predicate used to carry can never fail: the launcher rejects any
    // K that is not a whole number of super-blocks, and every k0 staged here is a multiple of
    // QM_BK that the caller has already checked against kend <= K, so k0 + a_c16 is at most
    // K - QM_BK + 16 < K. What is left is `a_row >= 0`, which is loop-invariant -- so the
    // predicated branch and its BSSY/BSYNC pair lift out of the unrolled K loop entirely.
    static_assert(QM_BK > 16, "stageA drops the column bound test on QM_BK > a_c16");
    const bool a_ok = a_row >= 0;
    // Rows past M hold zero for the whole kernel, so clear both A buffers once here rather than
    // re-storing zeros on every K step. That leaves the loop with a bare cp.async under a uniform
    // `if`, dropping the else-arm ptxas was wrapping in a BSSY/BSYNC pair -- along with the three
    // dead predicated-off LDS it pads each async copy with -- eight times per super-block.
    if (!a_ok) {
        *reinterpret_cast<uint4*>(a_dst0) = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(a_dst0 + QM_BM * QM_BK) = make_uint4(0u, 0u, 0u, 0u);
    }

    auto stageA = [&](int buf, const signed char* src) {
        if (a_ok) __pipeline_memcpy_async(a_dst0 + buf * (QM_BM * QM_BK), src, 16);
        __pipeline_commit();
    };

    // ldmatrix takes a shared-window address, and deriving one from a generic pointer inside the K
    // loop is expensive on this part: ptxas rebuilt the whole conversion every step -- an
    // S2UR SR_CgaCtaId, a UMOV/UIADD3/ULEA chain and the lane row math, ~26 instructions per
    // iteration to feed 4 LDSM. The lane's row is fixed for the life of the block and everything
    // that moves is a plain byte offset (A: the buffer parity; B: kk), so take the two bases once.
    unsigned a_sm = 0, b_sm = 0;
    if constexpr (K32) {
        // Row within the 16-row subtile. wm*32 + i*16 is a multiple of 16, so the swizzle bit
        // (row >> 2) & 1 depends only on the lane -- it is the same for both i, and the i and
        // buffer offsets below stay plain byte adds.
        const int arr = (lane & 7) + 8 * ((lane >> 3) & 1);
        a_sm = (unsigned)__cvta_generic_to_shared(
            &As[0][wm * 32 + arr][16 * (((lane >> 4) & 1) ^ ((arr >> 2) & 1))]);
        b_sm = (unsigned)__cvta_generic_to_shared(
            &Bs[0][wn * 32 + ((lane >> 4) & 1) * 8 + (lane & 7)][16 * ((lane >> 3) & 1)]);
    }

    const int sb_lo = SPLIT ? zsl * sb_per_split : 0;
    const int sb_hi = SPLIT ? min(nsb, sb_lo + sb_per_split) : nsb;
    if (sb_lo >= sb_hi) return;
    const int kend = sb_hi << 8;          // first K past this block's slice
    const signed char* a_ptr = a_src0 + (apk ? (size_t)(sb_lo << 3) * (size_t)Mtot * 32
                                             : (size_t)(sb_lo << 8));
    stageA(0, a_ptr);
    int abuf = 0, bbuf = 0;

    // Zero-fill only ever applies to a tile past N, and those rows never change, so both planes
    // can be cleared once instead of once per super-block.
    auto zero_plane = [&](int buf) {
#pragma unroll
        for (int i = 0; i < QM_SB / 16; i++)
            *reinterpret_cast<uint4*>(&Bs[buf][dr][i * 16]) = make_uint4(0u, 0u, 0u, 0u);
    };

    // Prologue: stage the first super-block into plane 0. From here on the loop body always holds
    // a decoded plane for `sb` and fills the other one for `sb + 1`.
    QmStage<QT> stg;
    if (drow_ok) {
        qm_stage_fetch<QT>(drow + (size_t)sb_lo * BS, dj, stg);
        qm_stage_decode<QT>(stg, dj, dinv, &Bs[0][dr][0]);
    } else if (dj == 0) {
        zero_plane(0);
        zero_plane(1);
    }

    for (int sb = sb_lo; sb < sb_hi; sb++) {
        __syncthreads();      // Bs[bbuf] is decoded, and the MMA that read Bs[bbuf^1] is done
        const bool more = (sb + 1) < sb_hi;
        // Issue the NEXT super-block's weight read here, ahead of the MMA that follows. This is
        // the whole point of the second Bs plane: the decode used to sit between two barriers with
        // the tensor cores idle behind it, so a DRAM read whose latency the 3-blocks/SM occupancy
        // was supposed to hide was in fact serialised against the MMA phase of every block on the
        // SM at once. Only the 12 registers of the fetched bytes stay live across the K loop.
        if (more && drow_ok) qm_stage_fetch<QT>(drow + (size_t)(sb + 1) * BS, dj, stg);

        for (int kk = 0; kk < QM_SB; kk += QM_BK) {
            const int knext = sb * QM_SB + kk + QM_BK;
            if (knext < kend) { a_ptr += a_step; stageA(abuf ^ 1, a_ptr); }
            __pipeline_wait_prior(knext < kend ? 1 : 0);
            __syncthreads();
            if constexpr (K32) {
                // One K=32 step covers the whole QM_BK slice, so the k16 loop disappears.
                // ldmatrix lane->address mapping for m16n8k32: .x4 wants matrices
                // (rows 0-7 | rows 8-15) x (bytes 0-15 | bytes 16-31) of a 16x32 A tile, and .x2
                // wants (bytes 0-15 | bytes 16-31) of an 8-row B tile -- B lives in Bs as
                // [n][k], which is already the col-major operand the instruction expects.
                // Both operand rows sit at a fixed byte offset from the bases taken above:
                // the A subtile is +16 rows (i * 16 * QM_BK) and the buffer is +QM_BM*QM_BK,
                // the B subtile is +16 rows (j2 * 16 * QM_LD) and the K step is +kk bytes.
                const unsigned ab = a_sm + (unsigned)abuf * (QM_BM * QM_BK);
                unsigned af32[2][4];
#pragma unroll
                for (int i = 0; i < 2; i++)
                    qm_ldsm_x4(af32[i], ab + (unsigned)(i * 16 * QM_BK));
#pragma unroll
                for (int j2 = 0; j2 < 2; j2++) {
                    // One .x4 covers two 8-column B subtiles: its four matrices are
                    // (subtile 0 | subtile 1) x (k 0-15 | k 16-31), which is exactly two
                    // m16n8k32 B operands. Halves the B load instructions for the same bytes.
                    unsigned bb[4];
                    qm_ldsm_x4(bb, b_sm + (unsigned)bbuf * (QM_BND * QM_LD)
                                        + (unsigned)(j2 * 16 * QM_LD) + (unsigned)kk);
#pragma unroll
                    for (int i = 0; i < 2; i++) {
                        qm_mma_16832(acc[i][2 * j2],     af32[i], bb[0], bb[1]);
                        qm_mma_16832(acc[i][2 * j2 + 1], af32[i], bb[2], bb[3]);
                    }
                }
            } else {
#pragma unroll
            for (int k16 = 0; k16 < QM_BK; k16 += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> af[2];
                wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> bf[2];
#pragma unroll
                for (int i = 0; i < 2; i++)
                    wmma::load_matrix_sync(af[i], &As[abuf][wm * 32 + i * 16][k16], QM_BK);
#pragma unroll
                for (int j = 0; j < 2; j++)
                    wmma::load_matrix_sync(bf[j], &Bs[bbuf][wn * 32 + j * 16][kk + k16], QM_LD);
#pragma unroll
                for (int i = 0; i < 2; i++)
#pragma unroll
                    for (int j = 0; j < 2; j++) wmma::mma_sync(cf[i][j], af[i], bf[j], cf[i][j]);
            }
            }
            __syncthreads();
            abuf ^= 1;
        }

        // Now turn the bytes fetched before the K loop into the other plane. Its consumer is the
        // MMA of the next iteration, which the barrier at the top of the loop separates from this
        // store; the plane being written was last read by the MMA of iteration sb-1, which that
        // same barrier has already retired.
        if (more && drow_ok) qm_stage_decode<QT>(stg, dj, dinv, &Bs[bbuf ^ 1][dr][0]);
        bbuf ^= 1;
    }

    // Epilogue staging reuses Bs (the K loop is done with it).
    __syncthreads();
    int* Cs = reinterpret_cast<int*>(&Bs[0][0][0]);
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int rm0 = wm * 32 + i * 16, gn0 = n0 + wn * 32 + j * 16;
            if constexpr (K32) {
                // m16n8k32's D layout: lane holds rows (lane>>2) and (lane>>2)+8, columns
                // 2*(lane&3) and +1, of each 16x8 tile. Two of those tiles side by side rebuild
                // exactly the 16x16 row-major tile the shared staging below already expects.
#pragma unroll
                for (int hh = 0; hh < 2; hh++) {
                    const int* d = acc[i][2 * j + hh];
                    const int rb = lane >> 2, cb = hh * 8 + 2 * (lane & 3);
                    Cs[warp * 256 + rb * 16 + cb]            = d[0];
                    Cs[warp * 256 + rb * 16 + cb + 1]        = d[1];
                    Cs[warp * 256 + (rb + 8) * 16 + cb]      = d[2];
                    Cs[warp * 256 + (rb + 8) * 16 + cb + 1]  = d[3];
                }
            } else {
                wmma::store_matrix_sync(&Cs[warp * 256], cf[i][j], 16, wmma::mem_row_major);
            }
            __syncwarp();
            // `atomic_acc` is a kernel argument, so this test was being re-evaluated for every one
            // of the 32 elements a lane writes: a BSSY/BSYNC pair, an ISETP and two branches per
            // element, around the three instructions that do the work. It is uniform over the whole
            // grid, so lift it (and the SPLIT arm) out of the element loop. Each element is
            // independent, so the values and their order are unchanged.
            if (SPLIT && atomic_acc) {
                for (int el = lane; el < 256; el += 32) {
                    const int rm = rm0 + (el >> 4), rn = gn0 + (el & 15);
                    if (rm < M && rn < N)
                        qm_red_add(&partials[pbase + (size_t)(p0 + rm) * N + rn], Cs[warp * 256 + el]);
                }
            } else if (SPLIT) {
                for (int el = lane; el < 256; el += 32) {
                    const int rm = rm0 + (el >> 4), rn = gn0 + (el & 15);
                    if (rm < M && rn < N)
                        partials[pbase + (((size_t)zsl * Mtot) + (p0 + rm)) * N + rn] =
                            Cs[warp * 256 + el];
                }
            } else {
                for (int el = lane; el < 256; el += 32) {
                    const int rm = rm0 + (el >> 4), rn = gn0 + (el & 15);
                    if (rm < M && rn < N) {
                        const int p = p0 + rm;
                        const float v = (float)Cs[warp * 256 + el] * sx[p] * row_scale[rn];
                        C[(size_t)p * N + rn] = __float2bfloat16(v);
                    }
                }
            }
            __syncwarp();
        }
    }
}

} // namespace

bool pfm_moe_gemm_qi8_supported(int ggml_type) {
    return ggml_type == QMQ_Q4_K || ggml_type == QMQ_Q5_K;
}

// The DENSE fused GEMM additionally decodes Q6_K (see qm_decode_j64). Kept separate from the
// routed predicate above on purpose: the routed MoE launcher gates every other model's expert
// GEMMs on that one, and this widening is only wanted for the dense Muse Glimmer projections.
// This GGUF gives 26 of its 52 layers a Q6_K attn_v, and those were the layers whose v fell out
// of the fused path into a full materialize round trip (measured: 0.47 ms of a 30.4 ms prefill).
bool pf_dense_gemm_qi8_supported(int ggml_type) {
    return ggml_type == QMQ_Q4_K || ggml_type == QMQ_Q5_K || ggml_type == QMQ_Q6_K;
}

bool launch_pfm_moe_gemm_qi8(int ggml_type, const signed char* A_i8, const float* sx,
                             const void* W_q, const float* row_scale,
                             const int* pair_tok, const float* pair_w,
                             const int* offsets, const int* tilemap, const int* d_ntiles,
                             void* C_bf16, float* out_f32,
                             int n_out, int K, int max_tiles, int bm,
                             bool a_indirect, bool c_scatter, cudaStream_t stream) {
    if (!pfm_moe_gemm_qi8_supported(ggml_type)) return false;
    if (bm != QM_BM && bm != QM_BM16) return false;       // tilemap must match one of these
    if (K <= 0 || (K & (QM_SB - 1)) != 0) return false;   // whole super-blocks only
    if (!row_scale || max_tiles <= 0) return false;
    auto* C = reinterpret_cast<__nv_bfloat16*>(C_bf16);
    if (bm == QM_BM16) {
        if (n_out <= 0 || (n_out % QM_BN16) != 0) return false;
        if (ggml_type == QMQ_Q4_K)
            dispatch_qi8_bm16<QMQ_Q4_K>(A_i8, sx, W_q, row_scale, pair_tok, pair_w, offsets, tilemap,
                                        d_ntiles, C, out_f32, n_out, K, max_tiles, a_indirect, c_scatter, stream);
        else
            dispatch_qi8_bm16<QMQ_Q5_K>(A_i8, sx, W_q, row_scale, pair_tok, pair_w, offsets, tilemap,
                                        d_ntiles, C, out_f32, n_out, K, max_tiles, a_indirect, c_scatter, stream);
        return true;
    }
    if (n_out <= 0 || (n_out % QM_BN) != 0) return false;
    if (ggml_type == QMQ_Q4_K)
        dispatch_qi8<QMQ_Q4_K>(A_i8, sx, W_q, row_scale, pair_tok, pair_w, offsets, tilemap,
                               d_ntiles, C, out_f32, n_out, K, max_tiles, a_indirect, c_scatter, stream);
    else
        dispatch_qi8<QMQ_Q5_K>(A_i8, sx, W_q, row_scale, pair_tok, pair_w, offsets, tilemap,
                               d_ntiles, C, out_f32, n_out, K, max_tiles, a_indirect, c_scatter, stream);
    return true;
}

// Sum the split-K int32 partials and apply sx*row_scale once. The sum is over int32, so the total
// is exactly what the unsplit kernel accumulated and the bf16 store sees an identical value.
__global__ void pf_dense_splitk_reduce_kernel(const int* __restrict__ partials,
                                              const float* __restrict__ sx,
                                              const float* __restrict__ row_scale,
                                              __nv_bfloat16* __restrict__ C,
                                              int Mtot, int N, int splits) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)Mtot * (size_t)N) return;
    const int m = (int)(idx / (size_t)N);
    const int n = (int)(idx - (size_t)m * (size_t)N);
    int acc = 0;
    for (int s = 0; s < splits; s++) acc += partials[((size_t)s * Mtot + m) * N + n];
    C[idx] = __float2bfloat16((float)acc * sx[m] * row_scale[n]);
}

// Grouped split-K reduce. One launch covers every group: the flattened column picks the group
// from the same tile prefix sum the GEMM prologue uses, so no per-group reduce launches are needed.
__global__ void pf_dense_splitk_reduce_group_kernel(const int* __restrict__ partials,
                                                    const float* __restrict__ sx,
                                                    PfQGroup gd, int Mtot, int ncol, int splits) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)Mtot * (size_t)ncol) return;
    const int m    = (int)(idx / (size_t)ncol);
    const int rest = (int)(idx - (size_t)m * (size_t)ncol);
    int g = 0;
#pragma unroll
    for (int i = 1; i < PF_QGROUP_MAX; i++)
        if (i < gd.ngroup && rest >= gd.first[i] * QM_BND) g = i;
    const int n = rest - gd.first[g] * QM_BND;
    const int N = gd.N[g];
    const int* base = partials + (size_t)gd.poff[g];
    int acc = 0;
    for (int s = 0; s < splits; s++) acc += base[((size_t)s * Mtot + m) * N + n];
    gd.C[g][(size_t)m * N + n] = __float2bfloat16((float)acc * sx[m] * gd.rs[g][n]);
}

// Accumulate the K slices with int32 atomics into ONE [m][n] plane instead of storing `splits`
// private planes and summing them. SPARKINFER_MUSE_QB_ATOMIC=0 restores the plane array (A/B).
static int qm_atomic_acc() {
    static int e = -1;
    if (e < 0) { const char* v = getenv("SPARKINFER_MUSE_QB_ATOMIC"); e = (v && v[0] == '0') ? 0 : 1; }
    return e;
}

// Resident block slots on this device for a __launch_bounds__(256, 2) kernel.
static int qm_block_slots() {
    static int slots = 0;
    if (!slots) {
        int dev = 0, sm = 170;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
        slots = sm * 2;
    }
    return slots;
}

// Among the slice counts NOT LARGER than the one the block-target heuristic asked for, take the one
// that leaves the fewest idle block slots in the final wave.
//
// Why this matters here: the grid is `splits * ntiles` blocks against `slots` of them, and at
// Muse's prefill@128 those land badly. The q/gate/k/v group is 136 tiles, so 13 slices put 1768
// blocks on 340 slots = 5.20 waves -- six waves of occupancy doing 5.20 waves of work -- while 5
// slices give exactly 2.00. The o projection is 104 tiles: 8 slices = 2.45 waves, 6 slices = 1.84.
// Only ever moving DOWN is deliberate: a larger slice count would also multiply the split-K atomic
// traffic, and that is what made a plain "raise the block target" retune lose (swept 2560 -> 5600).
// Ties break toward more slices, since shorter blocks balance the tail better.
// SPARKINFER_MUSE_QB_WAVE=0 restores the plain heuristic (A/B; bit-identical either way).
static int qm_wave_pick(int ntiles, int nsb, int want) {
    static int on = -1;
    if (on < 0) { const char* e = getenv("SPARKINFER_MUSE_QB_WAVE"); on = (e && e[0] == '0') ? 0 : 1; }
    if (!on || ntiles <= 0 || want <= 1) return want;
    const double slots = (double)qm_block_slots();
    int best = want; double best_eff = -1.0;
    for (int sbp = 1; sbp <= nsb; sbp++) {
        const int s = (nsb + sbp - 1) / sbp;
        if (s > want) continue;
        const double w = (double)s * (double)ntiles / slots;
        const double eff = w / ceil(w);
        if (eff > best_eff + 1e-9 || (eff > best_eff - 1e-9 && s > best)) { best_eff = eff; best = s; }
    }
    return best;
}

// K slices per launch. Shared by the single and grouped launchers so both fan out the same way.
static int qm_pick_splits(int ntiles, int K, int mtiles, bool have_partials, int partials_splits) {
    static int sk_env = -1;
    if (sk_env < 0) { const char* e = getenv("SPARKINFER_MUSE_QB_SPLITK"); sk_env = e ? atoi(e) : -2; }
    int splits = 1;
    if (have_partials && partials_splits > 1 && mtiles == 1 && sk_env != 0) {
        const int nsb = K >> 8;
        splits = (sk_env > 0) ? sk_env : (QM_TARGET_BLOCKS + ntiles - 1) / ntiles;
        if (splits > partials_splits) splits = partials_splits;
        if (splits > nsb / 2) splits = nsb / 2;
        if (splits < 1) splits = 1;
        splits = qm_wave_pick(ntiles, nsb, splits);
    }
    return splits;
}

// Dispatch on weight type and on the mma shape (K32 -> mma.m16n8k32, else the wmma inner loop).
#define QM_LAUNCH_ONE(TY, GRP, SPL, ...)                                                           \
    do {                                                                                           \
        if (qm_mma_k32()) pf_dense_gemm_qi8_kernel_g<TY, GRP, SPL, true>                           \
                              <<<grid, 256, 0, stream>>>(__VA_ARGS__);                             \
        else              pf_dense_gemm_qi8_kernel_g<TY, GRP, SPL, false>                          \
                              <<<grid, 256, 0, stream>>>(__VA_ARGS__);                             \
    } while (0)

#define QM_LAUNCH_DENSE(GRP, SPL, ...)                                                             \
    do {                                                                                           \
        if (ggml_type == QMQ_Q4_K)      QM_LAUNCH_ONE(QMQ_Q4_K, GRP, SPL, __VA_ARGS__);            \
        else if (ggml_type == QMQ_Q5_K) QM_LAUNCH_ONE(QMQ_Q5_K, GRP, SPL, __VA_ARGS__);            \
        else                            QM_LAUNCH_ONE(QMQ_Q6_K, GRP, SPL, __VA_ARGS__);            \
    } while (0)

// Dense fused-decode int8 GEMM (single weight, no routing). See pf_dense_gemm_qi8_kernel_g.
// `out_acc` (optional): when the split-K atomic path is taken, skip the reduce pass and report
// that `partials[0 .. M*N)` holds the raw int32 accumulator instead. The caller's next kernel then
// applies sx*row_scale itself -- see launch_norm_then_add_acc -- which saves the reduce launch and
// the bf16 tensor it would have materialized for a single consumer.
bool launch_prefill_gemm_qi8_dense(int ggml_type, const signed char* A_i8, const float* sx,
                                   const void* W_q, const float* row_scale, void* C_bf16,
                                   int M, int N, int K, cudaStream_t stream,
                                   int* partials, int partials_splits, int* out_acc,
                                   const signed char* A_pack) {
    if (!pf_dense_gemm_qi8_supported(ggml_type)) return false;   // Q4_K / Q5_K / Q6_K
    if (!row_scale || M <= 0) return false;
    if (K <= 0 || (K & (QM_SB - 1)) != 0) return false;         // whole super-blocks only
    if (N <= 0 || (N % QM_BND) != 0) return false;               // tile-aligned output width
    auto* C = reinterpret_cast<__nv_bfloat16*>(C_bf16);
    const auto* W = reinterpret_cast<const unsigned char*>(W_q);
    const int ntiles = (N + QM_BND - 1) / QM_BND;
    const int mtiles = (M + QM_BM - 1) / QM_BM;

    // Split K only when the plain grid leaves the device idle. With more than one M-tile the grid
    // already fans out over M and a split would only add the reduce pass. Each slice must own whole
    // super-blocks, at least 2, or the per-block prologue dominates it.
    int splits = qm_pick_splits(ntiles, K, mtiles, partials != nullptr, partials_splits);
    if (splits > 1) {
        const int nsb_all = K >> 8;
        const int sb_per_split = (nsb_all + splits - 1) / splits;
        // Re-derive the launch count from the slice size. Going the other way leaves trailing z
        // slices with sb_lo >= nsb: those blocks return before writing, the reduce then sums
        // never-written partials and the output is garbage (measured: TOP1 0.92 -> 0.67,
        // KL 0.036 -> 0.306, while the throughput still looked fine). Every launched slice must
        // own at least one super-block.
        splits = (nsb_all + sb_per_split - 1) / sb_per_split;
        if (splits > 1) {
            const int atomic = qm_atomic_acc();
            const size_t total = (size_t)M * (size_t)N;
            // The atomic accumulator must start at zero; one [m][n] plane, not `splits` of them.
            if (atomic) cudaMemsetAsync(partials, 0, total * sizeof(int), stream);
            dim3 grid(splits, ntiles, mtiles);
            QM_LAUNCH_DENSE(false, true,
                    A_i8, sx, W, row_scale, C, M, N, K, PfQGroup{}, partials, sb_per_split, atomic,
                    A_pack);
            // With the atomic accumulator every slice has already been summed, so the reduce pass
            // reads the single plane (splits=1 indexes exactly the [m][n] the atomics wrote).
            if (atomic && out_acc) { *out_acc = 1; return true; }   // consumer reads the int32
            pf_dense_splitk_reduce_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
                partials, sx, row_scale, C, M, N, atomic ? 1 : splits);
            return true;
        }
    }
    dim3 grid(ntiles, mtiles);
    QM_LAUNCH_DENSE(false, false, A_i8, sx, W, row_scale, C, M, N, K, PfQGroup{}, nullptr, 0, 0,
                    A_pack);
    return true;
}

// Same kernel, one launch for up to PF_QGROUP_MAX projections sharing A_i8/sx (same M, same K).
// Each output tile runs the identical decode, accumulation order and scales its own launch would
// have, so the result is bit-identical; only the scheduling changes.
bool launch_prefill_gemm_qi8_dense_group(int ggml_type, const signed char* A_i8, const float* sx,
                                         const void* const* W_q, const float* const* row_scale,
                                         void* const* C_bf16, const int* N, int ngroup,
                                         int M, int K, cudaStream_t stream,
                                         int* partials, int partials_splits, size_t partials_cap,
                                         signed char* fuse_q, float* fuse_sx, int* out_fused,
                                         const signed char* A_pack, signed char* fuse_qp) {
    if (!pf_dense_gemm_qi8_supported(ggml_type)) return false;
    if (ngroup <= 0 || ngroup > PF_QGROUP_MAX || M <= 0) return false;
    if (K <= 0 || (K & (QM_SB - 1)) != 0) return false;
    PfQGroup d{};
    d.ngroup = ngroup;
    int tiles = 0;
    for (int i = 0; i < ngroup; i++) {
        if (!row_scale[i] || N[i] <= 0 || (N[i] % QM_BND) != 0) return false;
        d.W[i]  = reinterpret_cast<const unsigned char*>(W_q[i]);
        d.rs[i] = row_scale[i];
        d.C[i]  = reinterpret_cast<__nv_bfloat16*>(C_bf16[i]);
        d.N[i]  = N[i];
        d.first[i] = tiles;
        tiles += N[i] / QM_BND;
    }
    const int mtiles = (M + QM_BM - 1) / QM_BM;

    // The grouped launch is the one the K split never reached, and at M=128 it is also the emptiest:
    // q+gate+k+v is 136 tiles of a 510-block device, a quarter of one wave for the full K. Splitting
    // it needs only the per-group base above; the arithmetic is the same exact int32 sum, so the
    // result stays bit-identical. SPARKINFER_MUSE_GROUP_SPLITK=0 restores the single-slice launch.
    static int gsk_env = -1;
    if (gsk_env < 0) { const char* e = getenv("SPARKINFER_MUSE_GROUP_SPLITK"); gsk_env = e ? atoi(e) : 1; }
    int splits = gsk_env == 0 ? 1
                              : qm_pick_splits(tiles, K, mtiles, partials != nullptr, partials_splits);
    // The caller sizes `partials` for one projection at a time; a group needs the sum of its
    // widths, so clamp the slice count to what actually fits rather than trusting it. The atomic
    // accumulator needs one plane whatever the slice count, so there it is a yes/no test.
    const int atomic = qm_atomic_acc();
    int ncol_all = 0;
    for (int i = 0; i < ngroup; i++) ncol_all += N[i];
    if (atomic) {
        if ((size_t)M * (size_t)ncol_all > partials_cap) splits = 1;
    } else {
        while (splits > 1 && (size_t)splits * (size_t)M * (size_t)ncol_all > partials_cap) splits--;
    }
    if (splits > 1) {
        const int nsb_all = K >> 8;
        const int sb_per_split = (nsb_all + splits - 1) / splits;
        // Same guard the single launcher documents: derive the launch count back from the slice
        // size so every launched slice owns at least one super-block, or the reduce sums memory
        // that was never written.
        splits = (nsb_all + sb_per_split - 1) / sb_per_split;
        if (splits > 1) {
            int off = 0;
            const int planes = atomic ? 1 : splits;
            for (int i = 0; i < ngroup; i++) { d.poff[i] = off; off += planes * M * N[i]; }
            // Every group's plane is packed back to back from `partials`, so one memset covers them.
            if (atomic) cudaMemsetAsync(partials, 0, (size_t)off * sizeof(int), stream);
            dim3 grid(splits, tiles, mtiles);
            QM_LAUNCH_DENSE(true, true,
                    A_i8, sx, nullptr, nullptr, nullptr, M, 0, K, d, partials, sb_per_split, atomic,
                    A_pack);
            // The FFN's gate/up pair is consumed by nothing except the SwiGLU + int8 quantize that
            // follows it, so when the caller asks for that, apply the scale inside it and never
            // materialize gate/up as bf16 at all: 20.4 MB of DRAM and one launch per layer gone.
            if (atomic && out_fused && fuse_q && ngroup == 2 && N[0] == N[1] &&
                kernels::launch_prefill_swiglu_quant_i8_acc(
                    partials + d.poff[0], partials + d.poff[1], sx, d.rs[0], d.rs[1],
                    fuse_q, fuse_sx, M, N[0], stream, fuse_qp)) {
                *out_fused = 1;
                return true;
            }
            const size_t total = (size_t)M * (size_t)tiles * QM_BND;
            pf_dense_splitk_reduce_group_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
                partials, sx, d, M, tiles * QM_BND, atomic ? 1 : splits);
            return true;
        }
    }
    dim3 grid(tiles, mtiles);
    QM_LAUNCH_DENSE(true, false, A_i8, sx, nullptr, nullptr, nullptr, M, 0, K, d, nullptr, 0, 0,
                    A_pack);
    return true;
}

} // namespace kernels
} // namespace sparkinfer
