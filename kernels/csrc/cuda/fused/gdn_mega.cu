// Qwen3.6-35B-A3B Gated-DeltaNet MEGAKERNEL.
//
// Collapses the per-layer main-stream serial chain
//     conv_split (depthwise causal conv + SiLU + q/k/v split)
//   -> l2_norm_qk (per-head L2 normalize of q and k)
//   -> gdn_ar_fast (DeltaNet recurrent state update)
//   -> (conv_state advance)
// into ONE persistent CUDA-graph node. In the decode graph these were 3 host launch
// wrappers firing 4 tiny, GPU-underfilling kernels on the MAIN stream (~30 GDN layers/
// token) whose q/k/v/gdn intermediates round-tripped through global memory between each
// launch. This kernel keeps every per-token transient (normalized q/k, conv'd v) in
// shared memory / registers, does the conv at the GPU-filling warp-per-column occupancy
// of the AR (instead of the conv's own 32-block launch), and never re-materializes the
// intermediates in gmem. gated_norm stays a separate kernel: its post-AR per-v-head
// reduction spans this grid's column tiles (cross-block), so it cannot fold in without a
// grid sync that would cost what it saves.
//
// BIT-IDENTICAL to conv_split_kernel + l2_norm_qk_kernel + gdn_ar_fast_kernel<8,128>
// composed, for the Qwen3.6 shape (head_dim=128, conv_kernel=4). The device math below is
// copied op-for-op from those kernels (same __expf/__logf/rsqrtf intrinsics, same warp
// butterfly reduction order, same two bf16 round-trips) so top1/KL vs the current main is
// exactly 1.0 / 0.0.
//
// conv_state is treated as a size-K position ring (ring[p % K] = raw input at position p)
// instead of the fallback's tap-major shift. That makes the single per-channel owner write
// exactly ring[pos % K] -- a slot no block reads this token (the taps read pos-1,pos-2,
// pos-3) -- so the redundant per-block conv reads never race the owner's write, and the
// state advance needs neither a separate launch nor a side stream. The caller sizes and
// zero-inits conv_state to K slots/layer; the fallback path uses the first K-1 slots
// (tap-major) unchanged.
//
// Qwen3.6-only: dispatched behind SPARKINFER_GDN_MEGA and the hybrid shape guard; the
// Qwen3-30B (non-hybrid) decode path never allocates GDN state nor reaches this kernel.

#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

// --- numeric primitives, identical to qwen36.cu's q36_* (load-bearing for bit-identity) ---
__device__ __forceinline__ float gm_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float gm_silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float gm_sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float gm_softplus(float x) { return x > 20.f ? x : __logf(1.f + __expf(x)); }
__device__ __forceinline__ float gm_wsum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

// Depthwise causal conv1d + SiLU for one channel d, read-only over the position ring.
// Reproduces conv_split_kernel's accumulation (taps t=0..K-2 ascending, then the newest
// input, in fp32) exactly. `write_owner` designates the single block that advances this
// channel's ring for the next token (ring[pos%K] = raw input) -- a slot disjoint from the
// K-1 taps this token reads, so it never races the redundant readers.
template <int CONV_K>
__device__ __forceinline__ float gm_conv_silu(const __nv_bfloat16* __restrict__ lin_qkv,
                                              const __nv_bfloat16* __restrict__ conv_w,
                                              __nv_bfloat16* __restrict__ conv_state,
                                              int d, int qkv_dim, int pos, bool write_owner) {
    const __nv_bfloat16 xnew = lin_qkv[d];
    float y = 0.f;
    #pragma unroll
    for (int t = 0; t < CONV_K - 1; t++) {
        const int age  = (CONV_K - 1) - t;                          // tap t is the input `age` tokens back
        const int slot = (unsigned)(pos - age) & (CONV_K - 1);      // CONV_K is a power of two
        y += gm_to_f(conv_state[(size_t)slot * qkv_dim + d]) *
             gm_to_f(conv_w[(size_t)d * CONV_K + t]);
    }
    y += gm_to_f(xnew) * gm_to_f(conv_w[(size_t)d * CONV_K + (CONV_K - 1)]);
    if (write_owner)
        conv_state[(size_t)((unsigned)pos & (CONV_K - 1)) * qkv_dim + d] = xnew;
    return gm_silu(y);
}

// One block owns v-head vh, column tile ct (COLS columns). COLS warps -> one warp/column
// (mirrors gdn_ar_fast). blockDim.x == COLS*32 == 2*HEAD_DIM so the block's 2*HEAD_DIM
// threads exactly cover the head's q(HEAD_DIM)+k(HEAD_DIM) channels for the conv+L2 stage.
template <int COLS, int HEAD_DIM, int CONV_K>
__global__ void gdn_mega_kernel(const __nv_bfloat16* __restrict__ lin_qkv,   // [qkv_dim]
                                const __nv_bfloat16* __restrict__ conv_w,     // [qkv_dim*CONV_K]
                                __nv_bfloat16* __restrict__ conv_state,       // ring [CONV_K][qkv_dim]
                                const __nv_bfloat16* __restrict__ alpha,
                                const __nv_bfloat16* __restrict__ beta,
                                const __nv_bfloat16* __restrict__ dt,
                                const __nv_bfloat16* __restrict__ a,
                                float* __restrict__ state,                    // TRANSPOSED [vh][col][row]
                                __nv_bfloat16* __restrict__ out,              // lin_gdn [v_heads*HEAD_DIM]
                                const int* __restrict__ dpos,                 // device position scalar
                                int q_heads, int v_heads, int qkv_dim, float eps) {
    static_assert(COLS * 32 == 2 * HEAD_DIM, "block must cover q+k channels of one head");
    static_assert((CONV_K & (CONV_K - 1)) == 0, "CONV_K must be a power of two for the ring mask");
    constexpr int NROW = HEAD_DIM / 32;     // rows per lane in the AR (compile-time -> unrolls)
    constexpr int NW   = HEAD_DIM / 32;      // warps per head in the L2 reduction (=4 for hd128)

    const int vh = blockIdx.x;
    const int ct = blockIdx.y;               // column tile
    if (vh >= v_heads) return;
    const int qh    = vh % q_heads;
    const int q_dim = q_heads * HEAD_DIM;
    const int pos   = dpos[0];
    const int tid   = threadIdx.x;
    const int warp  = tid >> 5;
    const int lane  = tid & 31;

    __shared__ __nv_bfloat16 sh_q[HEAD_DIM]; // normalized q[qh]
    __shared__ __nv_bfloat16 sh_k[HEAD_DIM]; // normalized k[qh]
    __shared__ __nv_bfloat16 sh_v[COLS];     // conv'd v for this block's columns
    __shared__ float red[COLS];              // per-warp L2 partials
    __shared__ float rq, rk;                 // L2 rsqrt scales for q and k

    // ---- Phase A1: conv + SiLU for q[qh] (tid<HEAD_DIM) and k[qh] (HEAD_DIM<=tid<2*HEAD_DIM) ----
    const bool own_qk = (vh < q_heads) && (ct == 0);   // single owner advances this head's q/k ring
    {
        const int isk = (tid >= HEAD_DIM) ? 1 : 0;
        const int t   = tid - isk * HEAD_DIM;                                  // 0..HEAD_DIM-1
        const int d   = (isk ? q_dim + qh * HEAD_DIM : qh * HEAD_DIM) + t;     // global qkv channel
        const float o = gm_conv_silu<CONV_K>(lin_qkv, conv_w, conv_state, d, qkv_dim, pos, own_qk);
        if (isk) sh_k[t] = __float2bfloat16(o); else sh_q[t] = __float2bfloat16(o);
    }
    __syncthreads();

    // ---- Phase A2: per-head L2 normalize of q and k (matches l2_norm_qk_kernel exactly) ----
    {
        const int isk = (tid >= HEAD_DIM) ? 1 : 0;
        const int t   = tid - isk * HEAD_DIM;
        const float xv = isk ? gm_to_f(sh_k[t]) : gm_to_f(sh_q[t]);
        const float ss = gm_wsum(xv * xv);
        if (lane == 0) red[warp] = ss;               // warps 0..NW-1 -> q, NW..2NW-1 -> k
    }
    __syncthreads();
    if (warp == 0) {                                  // reduce q partials red[0..NW-1]
        float v = (lane < NW) ? red[lane] : 0.f;
        v = gm_wsum(v);
        if (lane == 0) rq = rsqrtf(v + eps);
    } else if (warp == NW) {                          // reduce k partials red[NW..2NW-1]
        float v = (lane < NW) ? red[NW + lane] : 0.f;
        v = gm_wsum(v);
        if (lane == 0) rk = rsqrtf(v + eps);
    }
    __syncthreads();
    {
        const int isk = (tid >= HEAD_DIM) ? 1 : 0;
        const int t   = tid - isk * HEAD_DIM;
        if (isk) sh_k[t] = __float2bfloat16(gm_to_f(sh_k[t]) * rk);
        else     sh_q[t] = __float2bfloat16(gm_to_f(sh_q[t]) * rq);
    }

    // ---- Phase A3: conv + SiLU for this block's COLS v columns (v is not L2-normalized) ----
    if (tid < COLS) {
        const int j = ct * COLS + tid;                                   // column
        const int d = 2 * q_dim + vh * HEAD_DIM + j;                     // global v channel
        const float o = gm_conv_silu<CONV_K>(lin_qkv, conv_w, conv_state, d, qkv_dim, pos, /*write_owner=*/true);
        sh_v[tid] = __float2bfloat16(o);
    }
    __syncthreads();

    // ---- Phase B: DeltaNet AR update, warp w owns column j = ct*COLS + w (gdn_ar_fast math) ----
    const int j = ct * COLS + warp;
    if (j < HEAD_DIM) {
        const float scale = rsqrtf((float)HEAD_DIM);
        const float bb = gm_sigmoid(gm_to_f(beta[vh]));
        const float g  = __expf(gm_softplus(gm_to_f(alpha[vh]) + gm_to_f(dt[vh])) * gm_to_f(a[vh]));
        float* col = state + ((size_t)vh * HEAD_DIM + j) * HEAD_DIM;      // contiguous rows of column j
        float sloc[NROW];
        float part_sk = 0.f;
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int i = lane + r * 32;
            const float s = col[i];
            sloc[r] = s;
            part_sk += s * gm_to_f(sh_k[i]);
        }
        const float sk = g * gm_wsum(part_sk);
        const float delta = (gm_to_f(sh_v[warp]) - sk) * bb;
        float part_y = 0.f;
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int i = lane + r * 32;
            const float s_new = sloc[r] * g + gm_to_f(sh_k[i]) * delta;
            col[i] = s_new;
            part_y += s_new * gm_to_f(sh_q[i]) * scale;
        }
        const float y = gm_wsum(part_y);
        if (lane == 0) out[(size_t)vh * HEAD_DIM + j] = __float2bfloat16(y);
    }
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/fused.h"

// Fused conv+L2+AR for the Qwen3.6 GDN shape (head_dim=128, conv_kernel=4). conv_state is a
// CONV_K-slot position ring; `dpos` is the device position scalar the ring indexes by.
void launch_qwen36_gdn_mega(const void* lin_qkv_bf16, const void* conv_w_bf16,
                            void* conv_state_bf16, const void* alpha_bf16, const void* beta_bf16,
                            const void* dt_bf16, const void* a_bf16, float* state_f32,
                            void* out_bf16, const int* dpos,
                            int q_heads, int v_heads, int head_dim, int conv_kernel,
                            float eps, cudaStream_t stream) {
    const int q_dim = q_heads * head_dim;
    const int v_dim = v_heads * head_dim;
    const int qkv_dim = 2 * q_dim + v_dim;
    constexpr int COLS = 8;                         // 8 warps (columns) per block -> 256 threads
    dim3 grid(v_heads, head_dim / COLS);
    // head_dim==128 && conv_kernel==4 is enforced by the caller's shape guard.
    gdn_mega_kernel<COLS, 128, 4><<<grid, COLS * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(lin_qkv_bf16),
        reinterpret_cast<const __nv_bfloat16*>(conv_w_bf16),
        reinterpret_cast<__nv_bfloat16*>(conv_state_bf16),
        reinterpret_cast<const __nv_bfloat16*>(alpha_bf16),
        reinterpret_cast<const __nv_bfloat16*>(beta_bf16),
        reinterpret_cast<const __nv_bfloat16*>(dt_bf16),
        reinterpret_cast<const __nv_bfloat16*>(a_bf16),
        state_f32, reinterpret_cast<__nv_bfloat16*>(out_bf16), dpos,
        q_heads, v_heads, qkv_dim, eps);
}
#endif

} // namespace kernels
} // namespace sparkinfer
