// Qwen3.5/Qwen3.6 hybrid-layer helpers.
//
// These kernels implement the single-token decode path for the Gated DeltaNet
// recurrent layers used by Qwen3.6-35B-A3B. They favor clear, graph-capturable
// device-side state updates over aggressive specialization.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

__device__ __forceinline__ float q36_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float q36_silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float q36_sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float q36_softplus(float x) {
    return x > 20.f ? x : __logf(1.f + __expf(x));
}
__device__ __forceinline__ float q36_wsum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__global__ void split_q_gate_kernel(const __nv_bfloat16* __restrict__ qg,
                                    __nv_bfloat16* __restrict__ q,
                                    __nv_bfloat16* __restrict__ gate,
                                    int n_heads, int head_dim) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = n_heads * head_dim;
    if (gid >= n) return;
    const int h = gid / head_dim;
    const int d = gid - h * head_dim;
    const size_t src = (size_t)h * 2 * head_dim + d;
    q[gid] = qg[src];
    gate[gid] = qg[src + head_dim];
}

__global__ void mul_sigmoid_kernel(__nv_bfloat16* __restrict__ x,
                                   const __nv_bfloat16* __restrict__ gate,
                                   int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float y = q36_to_f(x[i]) * q36_sigmoid(q36_to_f(gate[i]));
    x[i] = __float2bfloat16(y);
}

__global__ void sigmoid_scalar_kernel(const __nv_bfloat16* __restrict__ x,
                                      float* __restrict__ out) {
    out[0] = q36_sigmoid(q36_to_f(x[0]));
}

__global__ void sigmoid_rows_kernel(const __nv_bfloat16* __restrict__ x,
                                    float* __restrict__ out, int rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) out[r] = q36_sigmoid(q36_to_f(x[r]));
}

// Shared-expert SwiGLU: out[i] = dw * SiLU(gate[i]) * up[i]. The shared-expert
// gate scalar dw folds in here (down is linear, so scaling the intermediate is
// identical to scaling the output), letting the caller finish with a plain add.
__global__ void shared_swiglu_kernel(const __nv_bfloat16* __restrict__ gate,
                                     const __nv_bfloat16* __restrict__ up,
                                     const float* __restrict__ dw,
                                     __nv_bfloat16* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16((*dw) * q36_silu(q36_to_f(gate[i])) * q36_to_f(up[i]));
}


__global__ void shared_swiglu_rows_kernel(const __nv_bfloat16* __restrict__ gate,
                                          const __nv_bfloat16* __restrict__ up,
                                          const float* __restrict__ dw,
                                          __nv_bfloat16* __restrict__ out,
                                          int rows, int ffn) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = rows * ffn;
    if (i >= n) return;
    const int row = i / ffn;
    out[i] = __float2bfloat16(dw[row] * q36_silu(q36_to_f(gate[i])) * q36_to_f(up[i]));
}

__global__ void conv_split_kernel(const __nv_bfloat16* __restrict__ qkv,
                                  const __nv_bfloat16* __restrict__ conv_w,
                                  __nv_bfloat16* __restrict__ conv_state,
                                  __nv_bfloat16* __restrict__ q,
                                  __nv_bfloat16* __restrict__ k,
                                  __nv_bfloat16* __restrict__ v,
                                  int q_dim, int v_dim, int qkv_dim,
                                  int conv_kernel) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= qkv_dim) return;

    float y = 0.f;
    for (int t = 0; t < conv_kernel - 1; t++)
        y += q36_to_f(conv_state[(size_t)t * qkv_dim + d]) *
             q36_to_f(conv_w[(size_t)d * conv_kernel + t]);
    y += q36_to_f(qkv[d]) * q36_to_f(conv_w[(size_t)d * conv_kernel + (conv_kernel - 1)]);

    for (int t = 0; t < conv_kernel - 2; t++)
        conv_state[(size_t)t * qkv_dim + d] = conv_state[(size_t)(t + 1) * qkv_dim + d];
    if (conv_kernel > 1)
        conv_state[(size_t)(conv_kernel - 2) * qkv_dim + d] = qkv[d];

    const __nv_bfloat16 oy = __float2bfloat16(q36_silu(y));
    if (d < q_dim) q[d] = oy;
    else if (d < 2 * q_dim) k[d - q_dim] = oy;
    else if (d < 2 * q_dim + v_dim) v[d - 2 * q_dim] = oy;
}

__global__ void l2_norm_heads_kernel(__nv_bfloat16* __restrict__ x,
                                     int n_heads, int head_dim, float eps) {
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int t = threadIdx.x;
    const size_t base = (size_t)h * head_dim;
    const float xv = (t < head_dim) ? q36_to_f(x[base + t]) : 0.f;
    __shared__ float sw[32];
    float ss = q36_wsum(xv * xv);
    if ((t & 31) == 0) sw[t >> 5] = ss;
    __syncthreads();
    if (t < 32) {
        float v = (t < (blockDim.x + 31) / 32) ? sw[t] : 0.f;
        v = q36_wsum(v);
        if (t == 0) sw[0] = rsqrtf(v + eps);
    }
    __syncthreads();
    if (t < head_dim) x[base + t] = __float2bfloat16(xv * sw[0]);
}

// Fused q/k L2 norm after conv (one launch for both head stacks).
__global__ void l2_norm_qk_kernel(__nv_bfloat16* __restrict__ q,
                                    __nv_bfloat16* __restrict__ k,
                                    int q_heads, int head_dim, float eps) {
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    __nv_bfloat16* x = (h < q_heads) ? q : k;
    const int hh = (h < q_heads) ? h : (h - q_heads);
    const size_t base = (size_t)hh * head_dim;
    const float xv = (t < head_dim) ? q36_to_f(x[base + t]) : 0.f;
    __shared__ float sw[32];
    float ss = q36_wsum(xv * xv);
    if ((t & 31) == 0) sw[t >> 5] = ss;
    __syncthreads();
    if (t < 32) {
        float v = (t < (blockDim.x + 31) / 32) ? sw[t] : 0.f;
        v = q36_wsum(v);
        if (t == 0) sw[0] = rsqrtf(v + eps);
    }
    __syncthreads();
    if (t < head_dim) x[base + t] = __float2bfloat16(xv * sw[0]);
}

// Fused causal-conv + split + L2(q,k) in ONE kernel (one block per output head), vs the
// conv_split_kernel + l2_norm_qk_kernel pair. Block-per-head lets the q/k heads finish their
// L2 reduction in shared/warp registers, so the normalized q/k never round-trip through HBM
// (the separate l2 kernel re-read everything it just wrote). v heads skip the norm. Runs on
// all 30 GDN layers per token, deleting one launch + one q/k HBM read+write each.
//   grid.x = q_heads (q) + q_heads (k) + v_heads (v);  block = head_dim (<=1024)
__global__ void conv_split_l2_kernel(const __nv_bfloat16* __restrict__ qkv,
                                     const __nv_bfloat16* __restrict__ conv_w,
                                     __nv_bfloat16* __restrict__ conv_state,
                                     __nv_bfloat16* __restrict__ q,
                                     __nv_bfloat16* __restrict__ k,
                                     __nv_bfloat16* __restrict__ v,
                                     int q_heads, int q_dim, int v_dim, int qkv_dim,
                                     int head_dim, int conv_kernel, float eps) {
    const int blk = blockIdx.x;
    const int t = threadIdx.x;              // channel within head
    if (t >= head_dim) return;
    // Map block -> (region, global channel d, output ptr).
    int d; __nv_bfloat16* out; int hh; bool do_norm;
    if (blk < q_heads)            { hh = blk;                d = hh * head_dim + t;             out = q; do_norm = true;  }
    else if (blk < 2 * q_heads)   { hh = blk - q_heads;      d = q_dim + hh * head_dim + t;     out = k; do_norm = true;  }
    else                          { hh = blk - 2 * q_heads;  d = 2 * q_dim + hh * head_dim + t; out = v; do_norm = false; }

    float y = 0.f;
    for (int c = 0; c < conv_kernel - 1; c++)
        y += q36_to_f(conv_state[(size_t)c * qkv_dim + d]) * q36_to_f(conv_w[(size_t)d * conv_kernel + c]);
    y += q36_to_f(qkv[d]) * q36_to_f(conv_w[(size_t)d * conv_kernel + (conv_kernel - 1)]);

    for (int c = 0; c < conv_kernel - 2; c++)
        conv_state[(size_t)c * qkv_dim + d] = conv_state[(size_t)(c + 1) * qkv_dim + d];
    if (conv_kernel > 1)
        conv_state[(size_t)(conv_kernel - 2) * qkv_dim + d] = qkv[d];

    float c = q36_silu(y);
    if (do_norm) {
        __shared__ float sw[32];
        float ss = q36_wsum(c * c);
        if ((t & 31) == 0) sw[t >> 5] = ss;
        __syncthreads();
        if (t < 32) {
            float vv = (t < (blockDim.x + 31) / 32) ? sw[t] : 0.f;
            vv = q36_wsum(vv);
            if (t == 0) sw[0] = rsqrtf(vv + eps);
        }
        __syncthreads();
        c *= sw[0];
    }
    const int local = (out == v) ? (d - 2 * q_dim) : (out == k) ? (d - q_dim) : d;
    out[local] = __float2bfloat16(c);
}

__global__ void gdn_ar_kernel(const __nv_bfloat16* __restrict__ q,
                              const __nv_bfloat16* __restrict__ k,
                              const __nv_bfloat16* __restrict__ v,
                              const __nv_bfloat16* __restrict__ alpha,
                              const __nv_bfloat16* __restrict__ beta,
                              const __nv_bfloat16* __restrict__ dt,
                              const __nv_bfloat16* __restrict__ a,
                              float* __restrict__ state,
                              __nv_bfloat16* __restrict__ out,
                              int q_heads, int v_heads, int head_dim) {
    const int vh = blockIdx.x;
    if (vh >= v_heads) return;
    const int j = threadIdx.x;
    if (j >= head_dim) return;
    const int qh = vh % q_heads;
    const float scale = rsqrtf((float)head_dim);
    const float b = q36_sigmoid(q36_to_f(beta[vh]));
    const float g = __expf(q36_softplus(q36_to_f(alpha[vh]) + q36_to_f(dt[vh])) * q36_to_f(a[vh]));
    const __nv_bfloat16* qhptr = q + (size_t)qh * head_dim;
    const __nv_bfloat16* khptr = k + (size_t)qh * head_dim;
    const __nv_bfloat16* vhptr = v + (size_t)vh * head_dim;
    float* sptr = state + (size_t)vh * head_dim * head_dim;

    float sk = 0.f;
    for (int i = 0; i < head_dim; i++) {
        float s = sptr[(size_t)i * head_dim + j] * g;
        sptr[(size_t)i * head_dim + j] = s;
        sk += s * q36_to_f(khptr[i]);
    }
    const float delta = (q36_to_f(vhptr[j]) - sk) * b;
    float y = 0.f;
    for (int i = 0; i < head_dim; i++) {
        float s = sptr[(size_t)i * head_dim + j] + q36_to_f(khptr[i]) * delta;
        sptr[(size_t)i * head_dim + j] = s;
        y += s * q36_to_f(qhptr[i]) * scale;
    }
    out[(size_t)vh * head_dim + j] = __float2bfloat16(y);
}

// Optimized Gated-DeltaNet AR state update (SPARKINFER_GDN_FAST). Same math as gdn_ar_kernel, but:
//  (1) ONE WARP PER STATE COLUMN: the 32 lanes split the head_dim rows, so the grid is
//      v_heads*(head_dim/COLS) blocks * COLS warps = it FILLS the GPU. The naive kernel launched only
//      <<<v_heads=32, head_dim>>> = 32 blocks on ~170 SMs (~81% idle) every token, 30x/token, and was
//      latency-bound on two serial 128-iter dependent-load loops it couldn't hide at 32 blocks.
//  (2) the column's state slice is cached in REGISTERS (sloc[]) -> ONE global read + ONE global write
//      of the 2 MB/layer state. The naive kernel wrote the decayed state then re-read it (4x traffic).
//  (3) TRANSPOSED state layout [vh][col][row]: warp-per-column then reads a contiguous 32-row run
//      (COALESCED). The naive [vh][row][col] layout would be 32-way scattered under warp-per-column.
// The recurrent state is zero-init (memset) and touched ONLY by the GDN AR kernel, so this internal
// layout is self-consistent PROVIDED SPARKINFER_GDN_FAST is all-or-nothing for the whole run (the
// dispatch flag is static, so it is). NOT byte-identical to the naive kernel — the warp-tree reduction
// reorders the fp32 sum — so gate by self-consistency (top1/KL) over a long sequence, not cmp.
// HEAD_DIM is a template param so NROW is compile-time -> the r-loops fully unroll and sloc[] stays in
// registers (a runtime head_dim would force sloc[] to local memory and defeat the register cache).
// ---- compacted bf16 recurrent state (continuous-batch decode only) -------------------------
// The GDN state is [v_heads][HEAD_DIM][HEAD_DIM] fp32 per layer PER SEQUENCE -- 3.1 MB at this
// model's shape -- and every element of it is read and written on every decode step. At
// concurrency 32 that is ~9.6 GB of traffic per step: the second-largest item in the step after
// the weight GEMMs, 15.3% of it. Halving the state halves that traffic -- measured by touching
// only half the state, aggregate goes 1162.8 -> 1301.4 tok/s.
//
// SCOPED to the packed path, and the scope is load-bearing: rounding the state to bf16 costs
// 10.1% of acceptance at ctx=16384 (tau 1.7297 -> 1.5542) against a 0.95x floor, a hard reject.
// A continuous-batch step is a plain argmax emit with no acceptance to lose; the speculative
// decode path has everything to lose. Only a session that has been through decode_packed is
// converted, and DSpark's verify never is, so every dspark-* path keeps the fp32 state
// bit-for-bit.
//
// The compacted array occupies the FIRST half of the same fp32 allocation, so the conversion is a
// one-off per session and costs no VRAM.
template <int COLS, int HEAD_DIM, bool SB16>
__global__ void gdn_ar_fast_kernel(const __nv_bfloat16* __restrict__ q,
                                   const __nv_bfloat16* __restrict__ k,
                                   const __nv_bfloat16* __restrict__ v,
                                   const __nv_bfloat16* __restrict__ alpha,
                                   const __nv_bfloat16* __restrict__ beta,
                                   const __nv_bfloat16* __restrict__ dt,
                                   const __nv_bfloat16* __restrict__ a,
                                   float* __restrict__ state,   // TRANSPOSED [vh][col][row]
                                   __nv_bfloat16* __restrict__ out,
                                   int q_heads, int v_heads, bool qh_block, bool state_bf16) {
    constexpr int NROW = HEAD_DIM / 32;                        // rows per lane (compile-time -> unrolls)
    const int vh   = blockIdx.x;
    const int j    = blockIdx.y * COLS + (threadIdx.x >> 5);   // state column (all lanes in a warp share j)
    const int lane = threadIdx.x & 31;
    if (vh >= v_heads || j >= HEAD_DIM) return;                // whole-warp guard (j is warp-uniform)
    // v-head -> q/k-head broadcast convention; see launch_qwen36_gdn_ar's own comment.
    const int qh   = qh_block ? (vh / (v_heads / q_heads)) : (vh % q_heads);
    const float scale = rsqrtf((float)HEAD_DIM);
    const float bb = q36_sigmoid(q36_to_f(beta[vh]));
    const float g  = __expf(q36_softplus(q36_to_f(alpha[vh]) + q36_to_f(dt[vh])) * q36_to_f(a[vh]));
    const __nv_bfloat16* qhptr = q + (size_t)qh * HEAD_DIM;
    const __nv_bfloat16* khptr = k + (size_t)qh * HEAD_DIM;
    const __nv_bfloat16* vhptr = v + (size_t)vh * HEAD_DIM;
    const size_t col_off = ((size_t)vh * HEAD_DIM + j) * HEAD_DIM;
    float* col = state + col_off;                             // contiguous [HEAD_DIM] rows of column j
    __nv_bfloat16* colb = reinterpret_cast<__nv_bfloat16*>(state) + col_off;

    float sloc[NROW];
    float part_sk = 0.f;
    #pragma unroll
    for (int r = 0; r < NROW; r++) {
        const int i = lane + r * 32;
        const float s = SB16 ? q36_to_f(colb[i]) : col[i];    // coalesced read
        sloc[r] = s;
        part_sk += s * q36_to_f(khptr[i]);
    }
    const float sk = g * q36_wsum(part_sk);                   // sk = g * sum_i S[i][j]*k[i]
    const float delta = (q36_to_f(vhptr[j]) - sk) * bb;
    float part_y = 0.f;
    #pragma unroll
    for (int r = 0; r < NROW; r++) {
        const int i = lane + r * 32;
        float s_new = sloc[r] * g + q36_to_f(khptr[i]) * delta;   // S[i][j]*g + k[i]*delta
        if (SB16) { colb[i] = __float2bfloat16(s_new); s_new = q36_to_f(colb[i]); }
        else { if (state_bf16) s_new = q36_to_f(__float2bfloat16(s_new)); col[i] = s_new; }
        part_y += s_new * q36_to_f(qhptr[i]) * scale;
    }
    const float y = q36_wsum(part_y);
    if (lane == 0) out[(size_t)vh * HEAD_DIM + j] = __float2bfloat16(y);
}

// ---------------------------------------------------------------------------
// BATCHED decode step: B INDEPENDENT sequences, one token each.
//
// Not the same shape as the chunked/compact scans next door. Those walk N CONSECUTIVE positions
// of ONE sequence and therefore carry a serial dependence between rows. Here each row is a
// different request advancing its own recurrent state by exactly one step, so the rows are fully
// independent and the batch is just another grid axis.
//
// Why it exists: the continuous-batch worker runs one full 64-layer forward PER SEQUENCE, and
// decode is bandwidth-bound on weight reads, so N concurrent requests read every weight N times
// and aggregate throughput does not scale with concurrency. Packing the rows into one forward
// fixes that, and the 48 Gated-DeltaNet layers are the only stage with no batched decode form --
// the projections/FFN already take R rows through one weight read, and the paged attention
// already takes num_seqs.
//
// The per-row state pointers arrive as a DEVICE ARRAY rather than a base + stride, because each
// session's lin_state / lin_conv_state is its own cudaMalloc. That also keeps this CUDA-graph
// safe: the graph bakes the ADDRESS of the pointer array, and the packer refreshes its CONTENTS
// before each replay, the same trick d_scalars already uses for token id / position.
//
// Row b reads activation row b and state states[b]; with batch == 1 and states[0] equal to what
// the unbatched launcher is handed, every lane does bit-identical arithmetic in the same order,
// so a packed step and a sequential step agree exactly rather than approximately.
template <int COLS, int HEAD_DIM, bool SB16>
__global__ void gdn_ar_fast_batched_kernel(const __nv_bfloat16* __restrict__ q,
                                           const __nv_bfloat16* __restrict__ k,
                                           const __nv_bfloat16* __restrict__ v,
                                           const __nv_bfloat16* __restrict__ alpha,
                                           const __nv_bfloat16* __restrict__ beta,
                                           const __nv_bfloat16* __restrict__ dt,
                                           const __nv_bfloat16* __restrict__ a,
                                           float* const* __restrict__ states,
                                           size_t state_off,
                                           __nv_bfloat16* __restrict__ out,
                                           int q_heads, int v_heads, bool qh_block,
                                           bool state_bf16) {
    constexpr int NROW = HEAD_DIM / 32;
    const int vh   = blockIdx.x;
    const int j    = blockIdx.y * COLS + (threadIdx.x >> 5);
    const int b    = blockIdx.z;
    const int lane = threadIdx.x & 31;
    if (vh >= v_heads || j >= HEAD_DIM) return;
    const int qh   = qh_block ? (vh / (v_heads / q_heads)) : (vh % q_heads);
    const float scale = rsqrtf((float)HEAD_DIM);
    // Per-row activation strides. dt/a are WEIGHTS (one value per v-head, shared by every row)
    // and are deliberately not strided; alpha/beta are per-token projections and are.
    const int qdim = q_heads * HEAD_DIM, vdim = v_heads * HEAD_DIM;
    const __nv_bfloat16* qrow = q + (size_t)b * qdim;
    const __nv_bfloat16* krow = k + (size_t)b * qdim;
    const __nv_bfloat16* vrow = v + (size_t)b * vdim;
    const __nv_bfloat16* arow = alpha + (size_t)b * v_heads;
    const __nv_bfloat16* brow = beta  + (size_t)b * v_heads;
    const float bb = q36_sigmoid(q36_to_f(brow[vh]));
    const float g  = __expf(q36_softplus(q36_to_f(arow[vh]) + q36_to_f(dt[vh])) * q36_to_f(a[vh]));
    const __nv_bfloat16* qhptr = qrow + (size_t)qh * HEAD_DIM;
    const __nv_bfloat16* khptr = krow + (size_t)qh * HEAD_DIM;
    const __nv_bfloat16* vhptr = vrow + (size_t)vh * HEAD_DIM;
    const size_t col_off = state_off + ((size_t)vh * HEAD_DIM + j) * HEAD_DIM;
    float* col = states[b] + col_off;
    __nv_bfloat16* colb = reinterpret_cast<__nv_bfloat16*>(states[b]) + col_off;

    float sloc[NROW];
    float part_sk = 0.f;
    #pragma unroll
    for (int r = 0; r < NROW; r++) {
        const int i = lane + r * 32;
        const float sv = SB16 ? q36_to_f(colb[i]) : col[i];
        sloc[r] = sv;
        part_sk += sv * q36_to_f(khptr[i]);
    }
    const float sk = g * q36_wsum(part_sk);
    const float delta = (q36_to_f(vhptr[j]) - sk) * bb;
    float part_y = 0.f;
    #pragma unroll
    for (int r = 0; r < NROW; r++) {
        const int i = lane + r * 32;
        float s_new = sloc[r] * g + q36_to_f(khptr[i]) * delta;
        if (SB16) { colb[i] = __float2bfloat16(s_new); s_new = q36_to_f(colb[i]); }
        else { if (state_bf16) s_new = q36_to_f(__float2bfloat16(s_new)); col[i] = s_new; }
        part_y += s_new * q36_to_f(qhptr[i]) * scale;
    }
    const float y = q36_wsum(part_y);
    if (lane == 0) out[(size_t)b * vdim + (size_t)vh * HEAD_DIM + j] = __float2bfloat16(y);
}

// Default GDN path: warp-per-state-column with native [vh][row][col] layout (no transposed
// state). One warp owns column j; row slices live in registers; grid fills the GPU.
template <int WARPS_PER_BLK, int HEAD_DIM>
__global__ void gdn_ar_warpgrid_kernel(const __nv_bfloat16* __restrict__ q,
                                       const __nv_bfloat16* __restrict__ k,
                                       const __nv_bfloat16* __restrict__ v,
                                       const __nv_bfloat16* __restrict__ alpha,
                                       const __nv_bfloat16* __restrict__ beta,
                                       const __nv_bfloat16* __restrict__ dt,
                                       const __nv_bfloat16* __restrict__ a,
                                       float* __restrict__ state,
                                       __nv_bfloat16* __restrict__ out,
                                       int q_heads, int v_heads) {
    constexpr int NROW = HEAD_DIM / 32;
    const int vh   = blockIdx.x;
    const int j    = blockIdx.y * WARPS_PER_BLK + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (vh >= v_heads || j >= HEAD_DIM) return;

    const int qh = vh % q_heads;
    const float scale = rsqrtf((float)HEAD_DIM);
    const float bb = q36_sigmoid(q36_to_f(beta[vh]));
    const float g = __expf(q36_softplus(q36_to_f(alpha[vh]) + q36_to_f(dt[vh])) * q36_to_f(a[vh]));
    const __nv_bfloat16* qhptr = q + (size_t)qh * HEAD_DIM;
    const __nv_bfloat16* khptr = k + (size_t)qh * HEAD_DIM;
    const __nv_bfloat16* vhptr = v + (size_t)vh * HEAD_DIM;
    float* sptr = state + (size_t)vh * HEAD_DIM * HEAD_DIM;

    float sloc[NROW];
    float part_sk = 0.f;
    #pragma unroll
    for (int r = 0; r < NROW; r++) {
        const int i = lane + r * 32;
        const float s = sptr[(size_t)i * HEAD_DIM + j];
        sloc[r] = s;
        part_sk += s * q36_to_f(khptr[i]);
    }
    const float sk = g * q36_wsum(part_sk);
    const float delta = (q36_to_f(vhptr[j]) - sk) * bb;
    float part_y = 0.f;
    #pragma unroll
    for (int r = 0; r < NROW; r++) {
        const int i = lane + r * 32;
        const float s = sloc[r] * g + q36_to_f(khptr[i]) * delta;
        sptr[(size_t)i * HEAD_DIM + j] = s;
        part_y += s * q36_to_f(qhptr[i]) * scale;
    }
    const float y = q36_wsum(part_y);
    if (lane == 0) out[(size_t)vh * HEAD_DIM + j] = __float2bfloat16(y);
}

__global__ void gated_norm_kernel(const __nv_bfloat16* __restrict__ x,
                                  const __nv_bfloat16* __restrict__ z,
                                  const __nv_bfloat16* __restrict__ weight,
                                  __nv_bfloat16* __restrict__ out,
                                  int v_heads, int head_dim, float eps) {
    const int h = blockIdx.x;
    if (h >= v_heads) return;
    const int t = threadIdx.x;
    const size_t base = (size_t)h * head_dim;
    const float xv = (t < head_dim) ? q36_to_f(x[base + t]) : 0.f;
    __shared__ float sw[32];
    float ss = q36_wsum(xv * xv);
    if ((t & 31) == 0) sw[t >> 5] = ss;
    __syncthreads();
    if (t < 32) {
        float v = (t < (blockDim.x + 31) / 32) ? sw[t] : 0.f;
        v = q36_wsum(v);
        if (t == 0) sw[0] = rsqrtf(v / head_dim + eps);
    }
    __syncthreads();
    if (t < head_dim) {
        const float y = xv * sw[0] * q36_to_f(weight[t]) * q36_silu(q36_to_f(z[base + t]));
        out[base + t] = __float2bfloat16(y);
    }
}

// One warp per v-head (HEAD_DIM compile-time): 32 threads vs 128, same 32 CTAs but 4x less
// block pressure — the naive gated_norm idles most SMs at <<<32,128>>> on Qwen3.6 decode.
template <int HEAD_DIM>
__global__ void gated_norm_warp_kernel(const __nv_bfloat16* __restrict__ x,
                                       const __nv_bfloat16* __restrict__ z,
                                       const __nv_bfloat16* __restrict__ weight,
                                       __nv_bfloat16* __restrict__ out,
                                       int v_heads, float eps) {
    constexpr int NROW = HEAD_DIM / 32;
    const int h = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (h >= v_heads) return;
    const size_t base = (size_t)h * HEAD_DIM;
    float ss = 0.f;
    float xv[NROW], zv[NROW], wv[NROW];
    #pragma unroll
    for (int r = 0; r < NROW; r++) {
        const int t = lane + r * 32;
        xv[r] = q36_to_f(x[base + t]);
        zv[r] = q36_to_f(z[base + t]);
        wv[r] = q36_to_f(weight[t]);
        ss += xv[r] * xv[r];
    }
    const float inv = rsqrtf(q36_wsum(ss) / HEAD_DIM + eps);
    #pragma unroll
    for (int r = 0; r < NROW; r++) {
        const int t = lane + r * 32;
        out[base + t] = __float2bfloat16(xv[r] * inv * wv[r] * q36_silu(zv[r]));
    }
}
#ifndef _MSC_VER
template __global__ void gated_norm_warp_kernel<128>(const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, __nv_bfloat16*, int, float);
#endif
struct si_blk_q8_1 { __half2 ds; signed char qs[32]; };

// Gated norm then Q8_1 from bf16-rounded values — bit-identical to gated_norm + quantize_q8_1_blocks.
template <int HEAD_DIM>
__global__ void gated_norm_q8_warp_kernel(const __nv_bfloat16* __restrict__ x,
                                          const __nv_bfloat16* __restrict__ z,
                                          const __nv_bfloat16* __restrict__ weight,
                                          si_blk_q8_1* __restrict__ out_q8,
                                          int v_heads, float eps) {
    constexpr int NROW = HEAD_DIM / 32;
    const int h = blockIdx.x;
    const int lane = threadIdx.x & 31;
    if (h >= v_heads) return;
    const size_t base = (size_t)h * HEAD_DIM;
    const int qbase = h * NROW;
    float ss = 0.f;
    float xv[NROW], zv[NROW], wv[NROW];
    #pragma unroll
    for (int r = 0; r < NROW; r++) {
        const int t = lane + r * 32;
        xv[r] = q36_to_f(x[base + t]);
        zv[r] = q36_to_f(z[base + t]);
        wv[r] = q36_to_f(weight[t]);
        ss += xv[r] * xv[r];
    }
    const float inv = rsqrtf(q36_wsum(ss) / HEAD_DIM + eps);
    #pragma unroll
    for (int r = 0; r < NROW; r++) {
        const float y = xv[r] * inv * wv[r] * q36_silu(zv[r]);
        const float bv = __bfloat162float(__float2bfloat16(y));
        float amax = fabsf(bv);
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, m));
        const float d = amax / 127.0f;
        const int qi = (amax == 0.0f) ? 0 : (int)roundf(bv / d);
        out_q8[qbase + r].qs[lane] = (signed char)qi;
        int s = qi;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
        if (lane == 0) out_q8[qbase + r].ds = __floats2half2_rn(d, d * (float)s);
    }
}
#ifndef _MSC_VER
template __global__ void gated_norm_q8_warp_kernel<128>(const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, si_blk_q8_1*, int, float);
#endif
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/fused.h"

void launch_qwen36_shared_swiglu(const void* gate_bf16, const void* up_bf16,
                                 const float* dw_f32, void* out_bf16, int n,
                                 cudaStream_t stream) {
    shared_swiglu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate_bf16),
        reinterpret_cast<const __nv_bfloat16*>(up_bf16),
        dw_f32, reinterpret_cast<__nv_bfloat16*>(out_bf16), n);
}

void launch_qwen36_shared_swiglu_rows(const void* gate_bf16, const void* up_bf16,
                                      const float* dw_f32, void* out_bf16,
                                      int rows, int ffn, cudaStream_t stream) {
    const int n = rows * ffn;
    shared_swiglu_rows_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate_bf16),
        reinterpret_cast<const __nv_bfloat16*>(up_bf16), dw_f32,
        reinterpret_cast<__nv_bfloat16*>(out_bf16), rows, ffn);
}

void launch_qwen36_split_q_gate(const void* qg_bf16, void* q_bf16, void* gate_bf16,
                                int n_heads, int head_dim, cudaStream_t stream) {
    const int n = n_heads * head_dim;
    split_q_gate_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qg_bf16),
        reinterpret_cast<__nv_bfloat16*>(q_bf16),
        reinterpret_cast<__nv_bfloat16*>(gate_bf16), n_heads, head_dim);
}

void launch_qwen36_mul_sigmoid(void* x_bf16, const void* gate_bf16, int n,
                               cudaStream_t stream) {
    mul_sigmoid_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x_bf16),
        reinterpret_cast<const __nv_bfloat16*>(gate_bf16), n);
}

void launch_qwen36_sigmoid_scalar(const void* x_bf16, float* out_f32,
                                  cudaStream_t stream) {
    sigmoid_scalar_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16), out_f32);
}


void launch_qwen36_sigmoid_rows(const void* x_bf16, float* out_f32, int rows,
                                cudaStream_t stream) {
    if (rows <= 0) return;
    sigmoid_rows_kernel<<<(rows + 31) / 32, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16), out_f32, rows);
}

void launch_qwen36_conv_split_l2(const void* qkv_bf16, const void* conv_w_bf16,
                                 void* conv_state_bf16, void* q_bf16, void* k_bf16,
                                 void* v_bf16, int q_heads, int v_heads, int head_dim,
                                 int conv_kernel, float eps, cudaStream_t stream) {
    const int q_dim = q_heads * head_dim;
    const int v_dim = v_heads * head_dim;
    const int qkv_dim = 2 * q_dim + v_dim;
    static int fused = -1;
    if (fused < 0) { const char* e = getenv("SPARKINFER_GDN_CONVL2"); fused = (e && e[0] == '0') ? 0 : 1; }
    if (fused && head_dim <= 1024) {
        conv_split_l2_kernel<<<2 * q_heads + v_heads, head_dim, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(qkv_bf16),
            reinterpret_cast<const __nv_bfloat16*>(conv_w_bf16),
            reinterpret_cast<__nv_bfloat16*>(conv_state_bf16),
            reinterpret_cast<__nv_bfloat16*>(q_bf16),
            reinterpret_cast<__nv_bfloat16*>(k_bf16),
            reinterpret_cast<__nv_bfloat16*>(v_bf16),
            q_heads, q_dim, v_dim, qkv_dim, head_dim, conv_kernel, eps);
        return;
    }
    conv_split_kernel<<<(qkv_dim + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv_bf16),
        reinterpret_cast<const __nv_bfloat16*>(conv_w_bf16),
        reinterpret_cast<__nv_bfloat16*>(conv_state_bf16),
        reinterpret_cast<__nv_bfloat16*>(q_bf16),
        reinterpret_cast<__nv_bfloat16*>(k_bf16),
        reinterpret_cast<__nv_bfloat16*>(v_bf16),
        q_dim, v_dim, qkv_dim, conv_kernel);
    l2_norm_qk_kernel<<<q_heads * 2, head_dim, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q_bf16),
        reinterpret_cast<__nv_bfloat16*>(k_bf16),
        q_heads, head_dim, eps);
}

// Batched decode step over B independent sequences. Mirrors launch_qwen36_gdn_ar's dispatch so
// the two stay on the same kernel family; only the grid gains a batch axis and the state arrives
// as a per-row pointer array. Restricted to the fast head_dim==128 path, which is the only one
// the packed decode path uses (Qwen3.8 linear_head_dim = 128); anything else must keep running
// the unbatched launcher per row.
// One-off fp32 -> compacted-bf16 conversion of a session's whole GDN state. Staged through a
// scratch buffer rather than written in place: the destination (the first half of the same
// allocation) overlaps the source, and a grid-wide shrink has no ordering between the block that
// writes an element and the block that has yet to read the float underneath it.
__global__ void gdn_state_f32_to_b16_kernel(const float* __restrict__ src,
                                            __nv_bfloat16* __restrict__ dst, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        dst[i] = __float2bfloat16(src[i]);
}
__global__ void gdn_state_b16_copy_kernel(const __nv_bfloat16* __restrict__ src,
                                          __nv_bfloat16* __restrict__ dst, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        dst[i] = src[i];
}
bool launch_qwen36_gdn_state_to_b16(float* state, void* staging, size_t n, cudaStream_t stream) {
    if (!state || !staging || n == 0) return false;
    const int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 8192) blocks = 8192;
    auto* stg = reinterpret_cast<__nv_bfloat16*>(staging);
    gdn_state_f32_to_b16_kernel<<<blocks, threads, 0, stream>>>(state, stg, n);
    gdn_state_b16_copy_kernel<<<blocks, threads, 0, stream>>>(
        stg, reinterpret_cast<__nv_bfloat16*>(state), n);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool launch_qwen36_gdn_ar_batched(const void* q_bf16, const void* k_bf16, const void* v_bf16,
                                  const void* alpha_bf16, const void* beta_bf16,
                                  const void* dt_bf16, const void* a_bf16,
                                  float* const* states, size_t state_off, void* out_bf16,
                                  int batch, int q_heads, int v_heads, int head_dim,
                                  bool qh_block, cudaStream_t stream, bool state_compact_b16) {
    if (batch < 1 || head_dim != 128) return false;
    static const bool state_bf16 = [] {
        const char* e = getenv("SPARKINFER_GDN_STATE_BF16");
        return e && e[0] == '1';
    }();
    static int cols = -1;
    if (cols < 0) {
        const char* e = getenv("SPARKINFER_GDN_FAST_COLS");
        if (e) cols = atoi(e);
        else cols = (v_heads >= 32) ? 4 : 8;
        if (!(cols == 4 || cols == 8 || cols == 16)) cols = 8;
    }
    constexpr int HD = 128;
    const int c = cols;
    dim3 grid(v_heads, (HD + c - 1) / c, batch);
#define SI_GDN_AR_B(C_, B_)                                                                \
    gdn_ar_fast_batched_kernel<C_, HD, B_><<<grid, (C_) * 32, 0, stream>>>(                    \
        reinterpret_cast<const __nv_bfloat16*>(q_bf16),                                    \
        reinterpret_cast<const __nv_bfloat16*>(k_bf16),                                    \
        reinterpret_cast<const __nv_bfloat16*>(v_bf16),                                    \
        reinterpret_cast<const __nv_bfloat16*>(alpha_bf16),                                \
        reinterpret_cast<const __nv_bfloat16*>(beta_bf16),                                 \
        reinterpret_cast<const __nv_bfloat16*>(dt_bf16),                                   \
        reinterpret_cast<const __nv_bfloat16*>(a_bf16),                                    \
        states, state_off, reinterpret_cast<__nv_bfloat16*>(out_bf16),                     \
        q_heads, v_heads, qh_block, state_bf16)
#define SI_GDN_AR_B_SEL(C_)  do { if (state_compact_b16) SI_GDN_AR_B(C_, true); \
                                  else                   SI_GDN_AR_B(C_, false); } while (0)
    if (c == 4)       SI_GDN_AR_B_SEL(4);
    else if (c == 16) SI_GDN_AR_B_SEL(16);
    else              SI_GDN_AR_B_SEL(8);
#undef SI_GDN_AR_B_SEL
#undef SI_GDN_AR_B
    return true;
}

void launch_qwen36_gdn_ar(const void* q_bf16, const void* k_bf16, const void* v_bf16,
                          const void* alpha_bf16, const void* beta_bf16,
                          const void* dt_bf16, const void* a_bf16,
                          float* state_f32, void* out_bf16,
                          int q_heads, int v_heads, int head_dim, bool qh_block,
                          cudaStream_t stream, bool state_compact_b16) {
    static const bool state_bf16 = [] {
        const char* e = getenv("SPARKINFER_GDN_STATE_BF16");
        return e && e[0] == '1';
    }();
    // SPARKINFER_GDN_FAST: warp-per-column, register-cached, transposed-state kernel (fills the GPU +
    // 2x state traffic). Uses a transposed internal state layout, so it MUST be all-or-nothing for the
    // run — the static flag guarantees that. Requires head_dim a multiple of 32 (128 -> NROW=4).
    static int fast = -1;
    if (fast < 0) { const char* e = getenv("SPARKINFER_GDN_FAST"); fast = (e && e[0] == '0') ? 0 : 1; }
    // `state_compact_b16` forces the fast path: it is the only single-row kernel that understands
    // the compacted form, and a session that holds one must never reach a kernel that does not.
    if ((fast || state_compact_b16) && head_dim == 128) {      // Qwen3.6/Qwythos linear_head_dim=128
        static int cols = -1;
        if (cols < 0) {
            const char* e = getenv("SPARKINFER_GDN_FAST_COLS");
            if (e) cols = atoi(e);
            else cols = (v_heads >= 32) ? 4 : 8;              // 32 v-heads -> 1024 blocks @ COLS=4
            if (!(cols == 4 || cols == 8 || cols == 16)) cols = 8;
        }
        constexpr int HD = 128;
        const int c = cols;
        dim3 grid(v_heads, (HD + c - 1) / c);
#define SI_GDN_AR_ONE(C_, B_)                                                              \
        gdn_ar_fast_kernel<C_, HD, B_><<<grid, (C_) * 32, 0, stream>>>(                    \
            reinterpret_cast<const __nv_bfloat16*>(q_bf16),                                \
            reinterpret_cast<const __nv_bfloat16*>(k_bf16),                                \
            reinterpret_cast<const __nv_bfloat16*>(v_bf16),                                \
            reinterpret_cast<const __nv_bfloat16*>(alpha_bf16),                            \
            reinterpret_cast<const __nv_bfloat16*>(beta_bf16),                              \
            reinterpret_cast<const __nv_bfloat16*>(dt_bf16),                                \
            reinterpret_cast<const __nv_bfloat16*>(a_bf16),                                 \
            state_f32, reinterpret_cast<__nv_bfloat16*>(out_bf16),                          \
            q_heads, v_heads, (bool)qh_block, state_bf16)
#define SI_GDN_AR_ONE_SEL(C_) do { if (state_compact_b16) SI_GDN_AR_ONE(C_, true);         \
                                   else                   SI_GDN_AR_ONE(C_, false); } while (0)
        if (c == 4)       SI_GDN_AR_ONE_SEL(4);
        else if (c == 16) SI_GDN_AR_ONE_SEL(16);
        else              SI_GDN_AR_ONE_SEL(8);
#undef SI_GDN_AR_ONE_SEL
#undef SI_GDN_AR_ONE
        return;
    }
    static int warpgrid = -1;
    if (warpgrid < 0) { const char* e = getenv("SPARKINFER_GDN_WARPGRID"); warpgrid = (e && e[0] == '0') ? 0 : 1; }
    if (warpgrid && head_dim == 128) {
        constexpr int WPB = 8, HD = 128;
        dim3 grid(v_heads, (HD + WPB - 1) / WPB);
        gdn_ar_warpgrid_kernel<WPB, HD><<<grid, WPB * 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_bf16),
            reinterpret_cast<const __nv_bfloat16*>(k_bf16),
            reinterpret_cast<const __nv_bfloat16*>(v_bf16),
            reinterpret_cast<const __nv_bfloat16*>(alpha_bf16),
            reinterpret_cast<const __nv_bfloat16*>(beta_bf16),
            reinterpret_cast<const __nv_bfloat16*>(dt_bf16),
            reinterpret_cast<const __nv_bfloat16*>(a_bf16),
            state_f32, reinterpret_cast<__nv_bfloat16*>(out_bf16),
            q_heads, v_heads);
        return;
    }
    gdn_ar_kernel<<<v_heads, head_dim, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_bf16),
        reinterpret_cast<const __nv_bfloat16*>(k_bf16),
        reinterpret_cast<const __nv_bfloat16*>(v_bf16),
        reinterpret_cast<const __nv_bfloat16*>(alpha_bf16),
        reinterpret_cast<const __nv_bfloat16*>(beta_bf16),
        reinterpret_cast<const __nv_bfloat16*>(dt_bf16),
        reinterpret_cast<const __nv_bfloat16*>(a_bf16),
        state_f32, reinterpret_cast<__nv_bfloat16*>(out_bf16),
        q_heads, v_heads, head_dim);
}

void launch_qwen36_gated_norm(const void* x_bf16, const void* z_bf16,
                              const void* weight_bf16, void* out_bf16,
                              int v_heads, int head_dim, float eps,
                              cudaStream_t stream) {
    static int warp = -1;
    if (warp < 0) {
        const char* e = getenv("SPARKINFER_GDN_GNORM");
        warp = (e && e[0] == '0') ? 0 : 1;
    }
    if (warp && head_dim == 128) {
        gated_norm_warp_kernel<128><<<v_heads, 32, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x_bf16),
            reinterpret_cast<const __nv_bfloat16*>(z_bf16),
            reinterpret_cast<const __nv_bfloat16*>(weight_bf16),
            reinterpret_cast<__nv_bfloat16*>(out_bf16),
            v_heads, eps);
        return;
    }
    gated_norm_kernel<<<v_heads, head_dim, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<const __nv_bfloat16*>(z_bf16),
        reinterpret_cast<const __nv_bfloat16*>(weight_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        v_heads, head_dim, eps);
}

void launch_qwen36_gated_norm_q8(const void* x_bf16, const void* z_bf16,
                                 const void* weight_bf16, void* out_q8,
                                 int v_heads, int head_dim, float eps,
                                 cudaStream_t stream) {
    (void)head_dim;
    gated_norm_q8_warp_kernel<128><<<v_heads, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<const __nv_bfloat16*>(z_bf16),
        reinterpret_cast<const __nv_bfloat16*>(weight_bf16),
        reinterpret_cast<si_blk_q8_1*>(out_q8),
        v_heads, eps);
}
#endif

// ---- fused conv_split + l2_norm (per-head) ----------------------------------
// Replaces conv_split_kernel + 2× l2_norm_heads_kernel with a single per-head
// kernel. One block per head (64 blocks = 16q + 16k + 32v) with head_dim
// threads. Each thread processes one dimension: 1D depthwise conv, SiLU,
// then l2_norm (q/k heads only). Eliminates two kernel launches and the
// intermediate non-normalized q/k global writes. SKIP_FUSED_CONV_L2NORM=1
// restores the split path for A/B.
__global__ void conv_split_l2norm_fused_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ conv_w,
    __nv_bfloat16* __restrict__ conv_state,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    int q_heads, int v_heads, int head_dim, int conv_kernel, float eps)
{
    const int h  = blockIdx.x;
    const int t  = threadIdx.x;
    const int q_dim = q_heads * head_dim;
    const int v_dim = v_heads * head_dim;
    const int qkv_dim = 2 * q_dim + v_dim;

    bool do_norm = false;
    int d;
    __nv_bfloat16* out;
    if (h < q_heads) {
        d = h * head_dim + t;  out = q + d;           do_norm = true;
    } else if (h < 2 * q_heads) {
        d = q_dim + (h - q_heads) * head_dim + t;  out = k + d - q_dim;  do_norm = true;
    } else {
        d = 2 * q_dim + (h - 2 * q_heads) * head_dim + t;  out = v + d - 2 * q_dim;
    }
    if (d >= qkv_dim) return;

    // 1D conv + SiLU
    float y = 0.f;
    for (int p = 0; p < conv_kernel - 1; p++)
        y += q36_to_f(conv_state[(size_t)p * qkv_dim + d]) *
             q36_to_f(conv_w[(size_t)d * conv_kernel + p]);
    y += q36_to_f(qkv[d]) * q36_to_f(conv_w[(size_t)d * conv_kernel + (conv_kernel - 1)]);

    for (int p = 0; p < conv_kernel - 2; p++)
        conv_state[(size_t)p * qkv_dim + d] = conv_state[(size_t)(p + 1) * qkv_dim + d];
    if (conv_kernel > 1)
        conv_state[(size_t)(conv_kernel - 2) * qkv_dim + d] = qkv[d];

    const float oy = q36_silu(y);

    if (do_norm) {
        const float ss = q36_wsum(oy * oy);
        __shared__ float sw[32];
        if ((t & 31) == 0) sw[t >> 5] = ss;
        __syncthreads();
        if (t < 32) {
            float vv = (t < (head_dim + 31) / 32) ? sw[t] : 0.f;
            vv = q36_wsum(vv);
            if (t == 0) sw[0] = rsqrtf(vv + eps);
        }
        __syncthreads();
        out[0] = __float2bfloat16(oy * sw[0]);
    } else {
        out[0] = __float2bfloat16(oy);
    }
}

// Batched twin of conv_split_l2norm_fused_kernel: B independent rows, each with its OWN conv
// state (a device array of pointers, same reasoning as gdn_ar_fast_batched_kernel above).
// blockIdx.y is the row; everything else is the unbatched kernel unchanged, so batch == 1 with
// states[0] equal to the unbatched conv_state argument reproduces it exactly.
__global__ void conv_split_l2norm_fused_batched_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ conv_w,
    __nv_bfloat16* const* __restrict__ conv_states,
    size_t conv_off,
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    __nv_bfloat16* __restrict__ v,
    int q_heads, int v_heads, int head_dim, int conv_kernel, float eps)
{
    const int h  = blockIdx.x;
    const int b  = blockIdx.y;
    const int t  = threadIdx.x;
    const int q_dim = q_heads * head_dim;
    const int v_dim = v_heads * head_dim;
    const int qkv_dim = 2 * q_dim + v_dim;

    const __nv_bfloat16* qkv_row = qkv + (size_t)b * qkv_dim;
    __nv_bfloat16* conv_state = conv_states[b] + conv_off;

    bool do_norm = false;
    int d;
    __nv_bfloat16* out;
    if (h < q_heads) {
        d = h * head_dim + t;  out = q + (size_t)b * q_dim + d;  do_norm = true;
    } else if (h < 2 * q_heads) {
        d = q_dim + (h - q_heads) * head_dim + t;
        out = k + (size_t)b * q_dim + d - q_dim;  do_norm = true;
    } else {
        d = 2 * q_dim + (h - 2 * q_heads) * head_dim + t;
        out = v + (size_t)b * v_dim + d - 2 * q_dim;
    }
    if (d >= qkv_dim) return;

    float y = 0.f;
    for (int p = 0; p < conv_kernel - 1; p++)
        y += q36_to_f(conv_state[(size_t)p * qkv_dim + d]) *
             q36_to_f(conv_w[(size_t)d * conv_kernel + p]);
    y += q36_to_f(qkv_row[d]) * q36_to_f(conv_w[(size_t)d * conv_kernel + (conv_kernel - 1)]);

    for (int p = 0; p < conv_kernel - 2; p++)
        conv_state[(size_t)p * qkv_dim + d] = conv_state[(size_t)(p + 1) * qkv_dim + d];
    if (conv_kernel > 1)
        conv_state[(size_t)(conv_kernel - 2) * qkv_dim + d] = qkv_row[d];

    const float oy = q36_silu(y);

    if (do_norm) {
        const float ss = q36_wsum(oy * oy);
        __shared__ float sw[32];
        if ((t & 31) == 0) sw[t >> 5] = ss;
        __syncthreads();
        if (t < 32) {
            float vv = (t < (head_dim + 31) / 32) ? sw[t] : 0.f;
            vv = q36_wsum(vv);
            if (t == 0) sw[0] = rsqrtf(vv + eps);
        }
        __syncthreads();
        out[0] = __float2bfloat16(oy * sw[0]);
    } else {
        out[0] = __float2bfloat16(oy);
    }
}

void launch_qwen36_conv_split_l2norm_fused_batched(
    const void* qkv_bf16, const void* conv_w_bf16,
    void* const* conv_states_bf16, size_t conv_off, void* q_bf16, void* k_bf16,
    void* v_bf16, int batch, int q_heads, int v_heads, int head_dim,
    int conv_kernel, float eps, cudaStream_t stream)
{
    if (batch < 1) return;
    conv_split_l2norm_fused_batched_kernel<<<dim3(2 * q_heads + v_heads, batch), head_dim, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv_bf16),
        reinterpret_cast<const __nv_bfloat16*>(conv_w_bf16),
        reinterpret_cast<__nv_bfloat16* const*>(conv_states_bf16), conv_off,
        reinterpret_cast<__nv_bfloat16*>(q_bf16),
        reinterpret_cast<__nv_bfloat16*>(k_bf16),
        reinterpret_cast<__nv_bfloat16*>(v_bf16),
        q_heads, v_heads, head_dim, conv_kernel, eps);
}

void launch_qwen36_conv_split_l2norm_fused(
    const void* qkv_bf16, const void* conv_w_bf16,
    void* conv_state_bf16, void* q_bf16, void* k_bf16,
    void* v_bf16, int q_heads, int v_heads, int head_dim,
    int conv_kernel, float eps, cudaStream_t stream)
{
    conv_split_l2norm_fused_kernel<<<2 * q_heads + v_heads, head_dim, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv_bf16),
        reinterpret_cast<const __nv_bfloat16*>(conv_w_bf16),
        reinterpret_cast<__nv_bfloat16*>(conv_state_bf16),
        reinterpret_cast<__nv_bfloat16*>(q_bf16),
        reinterpret_cast<__nv_bfloat16*>(k_bf16),
        reinterpret_cast<__nv_bfloat16*>(v_bf16),
        q_heads, v_heads, head_dim, conv_kernel, eps);
}

} // namespace kernels
} // namespace sparkinfer
