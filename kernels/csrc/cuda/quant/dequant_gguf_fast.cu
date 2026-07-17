// ============================================================================
// Coalesced Q4_K -> bf16 dequantization.
//
// WHY THIS EXISTS
// ---------------
// deq_q4k_kernel (dequant_gguf.cu) assigns ONE THREAD per 256-value Q4_K super-block, and that
// thread writes all 256 bf16 outputs with scalar stores:
//
//     long b = blockIdx.x * blockDim.x + threadIdx.x;   // one thread == one 256-value block
//     __nv_bfloat16* yy = y + b * 256;
//     for (int l = 0; l < 32; l++) yy[j + l] = __float2bfloat16(d1 * (q[l] & 0xF) - m1);
//
// Consecutive lanes therefore write addresses 512 B apart. Every store instruction in a warp
// touches 32 DISTINCT 32-byte sectors and uses 2 bytes of each => ~16x write amplification.
//
// This costs real time in two places. At LOAD, every dense GGUF tensor goes through here: measured
// 19.05 ms in one Q4_K call on an RTX 5090 (main @ cb34dc1). In PREFILL, the Qwythos batched path
// (#398) dequantizes projection weights to bf16 scratch on EVERY call (qwen35_prefill.cpp `dq()` ->
// launch_gguf_dequant). Since #464 fused Q4K/Q6K -> int8 for every projection with n_out >= 128,
// what still lands here is the 48 ssm_alpha/ssm_beta projections per prefill (n_out == 32, kept in
// bf16 on purpose because per-row int8 quant of a 32-wide weight costs more accuracy than it saves)
// = 4.42 ms per prefill. Both numbers come from a reps=1-vs-2 instance diff (49 -> 97 instances,
// 23.47 -> 27.89 ms), which is the only way to separate the one-time load cost from the per-prefill
// cost — the raw nsys total mixes them and reads as a single 23.5 ms line.
//
// The arithmetic says the pattern, not the work, is the cost: dequantizing W weights moves
// 2*W bytes of bf16 writes + ~0.5625*W bytes of Q4_K reads, which at 1792 GB/s should be
// bandwidth-bound; 16x write amplification is what puts the kernel at ~7% of DRAM peak instead.
//
// THE FIX. One WARP per super-block instead of one thread. Each lane owns 8 consecutive outputs
// and emits them as a single 16-byte store, so a warp writes 512 contiguous bytes — fully
// coalesced, no amplification. Each lane's 8 outputs always fall inside one 64-value sub-block
// and one nibble half (lane*8 is a multiple of 8, so lane*8 .. lane*8+7 never straddles a 32- or
// 64-boundary), so the scale/min pair is computed once per lane rather than per value.
//
// BYTE-EXACT. The per-value math is unchanged — same d/dmin, same 6-bit scale/min unpacking, same
// `d*s * nibble - dmin*m` in the same order, same __float2bfloat16. Only the thread->output
// mapping changes. This is deliberate: dequant_gguf.cu is SHARED with Qwen3.6 (it is the GGUF
// load path for every model), and the closed #464 scored eval-qwen35:XL but was rejected
// eval-qwen36:REJECT after changing shared dequant SEMANTICS (fusing Q4K/Q6K -> int8). Changing
// only the memory access pattern cannot move any model's numerics.
//
// Q4_K only. Every one of the 248 per-prefill dequants is Q4_K (Q6_K tensors are requantized to
// Q4_K at load, which is why deq_q6k's instance count does NOT scale with prefill reps); other
// types fall through to the existing kernels.
// ============================================================================
#include "sparkinfer/kernels/dequant_gguf_fast.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>

namespace sparkinfer {
namespace kernels {

namespace {

constexpr int DQF_Q4_K = 12;   // ggml type id, matches dequant_gguf.cu

__device__ __forceinline__ float dqf_h2f(const unsigned char* p) {
    __half h; *((unsigned short*)&h) = *(const unsigned short*)p; return __half2float(h);
}

// 6-bit packed scale/min unpack — identical to gg_scale_min_k4 in dequant_gguf.cu.
__device__ __forceinline__ void dqf_scale_min_k4(int j, const unsigned char* q, int* d, int* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

// One warp per 256-value super-block; one 16-byte coalesced store per lane.
__global__ void deq_q4k_coalesced_kernel(const unsigned char* __restrict__ src,
                                         __nv_bfloat16* __restrict__ y, long nblocks) {
    const long gtid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long b    = gtid >> 5;                 // warp id == super-block index
    if (b >= nblocks) return;
    const int lane = (int)(gtid & 31);

    const unsigned char* blk = src + b * 144;
    const float d    = dqf_h2f(blk);
    const float dmin = dqf_h2f(blk + 2);
    const unsigned char* sc = blk + 4;
    const unsigned char* q  = blk + 16;

    // This lane owns outputs [n0, n0+8). n0 is a multiple of 8, so all 8 share one 64-value
    // sub-block (jj) and one nibble half — the scale/min pair is uniform across the lane.
    const int n0   = lane * 8;
    const int jj   = n0 >> 6;             // sub-block 0..3   (scales are consumed in pairs)
    const int half = (n0 & 63) >> 5;      // 0 = low nibbles, 1 = high nibbles
    const int l0   = n0 & 31;

    int s, m;
    dqf_scale_min_k4(jj * 2 + half, sc, &s, &m);
    const float dd = d * s, mm = dmin * m;

    const unsigned char* qp = q + jj * 32 + l0;
    __nv_bfloat16 out[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) {
        const unsigned char qb = qp[t];
        const int nib = half ? (qb >> 4) : (qb & 0xF);
        out[t] = __float2bfloat16(dd * nib - mm);
    }
    // 16-byte aligned: y is a cudaMalloc'd base, block stride is 256 bf16 (512 B), lane stride 16 B.
    *reinterpret_cast<uint4*>(y + b * 256 + n0) = *reinterpret_cast<const uint4*>(out);
}

}  // namespace

bool launch_gguf_dequant_fast(int ggml_type, const void* src, void* dst_bf16, long n_values,
                              cudaStream_t stream) {
    static const int enabled = [] {
        const char* e = getenv("SPARKINFER_DEQUANT_COALESCED");
        return (e && e[0] == '0') ? 0 : 1;
    }();
    if (!enabled || ggml_type != DQF_Q4_K) return false;
    if (n_values <= 0 || (n_values % 256) != 0) return false;

    const long nb = n_values / 256;
    const int  T  = 256;                       // 8 warps per CUDA block
    const long threads = nb * 32;              // one warp per super-block
    deq_q4k_coalesced_kernel<<<(threads + T - 1) / T, T, 0, stream>>>(
        reinterpret_cast<const unsigned char*>(src),
        reinterpret_cast<__nv_bfloat16*>(dst_bf16), nb);
    return true;
}

}  // namespace kernels
}  // namespace sparkinfer
