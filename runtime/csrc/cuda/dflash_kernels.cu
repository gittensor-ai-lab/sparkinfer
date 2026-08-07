// DFlash draft kernels: small-seq GQA attention, RoPE, SwiGLU, RMSNorm.
#include "sparkinfer/models/dflash_kernels.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

namespace sparkinfer {
namespace dflash_kernels {
namespace {

using bf16 = __nv_bfloat16;

__device__ inline float b2f(bf16 x) { return __bfloat162float(x); }
__device__ inline bf16 f2b(float x) { return __float2bfloat16(x); }

__global__ void k_add(const bf16* a, const bf16* b, bf16* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = f2b(b2f(a[i]) + b2f(b[i]));
}

__global__ void k_swiglu(const bf16* gate, const bf16* up, bf16* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = b2f(gate[i]);
        float u = b2f(up[i]);
        float s = g / (1.f + expf(-g));
        out[i] = f2b(s * u);
    }
}

__global__ void k_rms(const bf16* x, const bf16* w, bf16* out, int rows, int cols, float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const bf16* xr = x + (size_t)r * cols;
    bf16* or_ = out + (size_t)r * cols;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    // uint4 (8 bf16) loads plus a shuffle reduction. The tree reduction this replaces cost
    // log2(blockDim) block-wide barriers and issued 2-byte scalar loads, which on a 16x2048 block
    // left these norms latency-bound at ~7 us per launch -- and a draft block runs 14 of them.
    float ss = 0.f;
    if ((cols & 7) == 0) {
        const uint4* x4 = (const uint4*)xr;
        for (int i = threadIdx.x; i < (cols >> 3); i += blockDim.x) {
            const uint4 p = x4[i];
            const __nv_bfloat162* h = (const __nv_bfloat162*)&p;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 f = __bfloat1622float2(h[j]);
                ss += f.x * f.x + f.y * f.y;
            }
        }
    } else {
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            float v = b2f(xr[i]);
            ss += v * v;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);
    __shared__ float ws[32];
    if (lane == 0) ws[warp] = ss;
    __syncthreads();
    float tot = 0.f;
    for (int i = 0; i < nwarps; i++) tot += ws[i];
    const float inv = rsqrtf(tot / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        or_[i] = f2b(b2f(xr[i]) * inv * b2f(w[i]));
}

// Residual add fused with the RMSNorm that always follows it: sum = a + b (kept for the next
// residual) and out = rms(sum) * w. Identical arithmetic to launch_add followed by launch_rms, but
// one eager launch instead of two and `sum` stays in registers for the norm's first pass.
__global__ void k_add_rms(const bf16* a, const bf16* b, bf16* sum, const bf16* w, bf16* out,
                          int rows, int cols, float eps) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const bf16* ar = a + (size_t)r * cols;
    const bf16* br = b + (size_t)r * cols;
    bf16* sr = sum + (size_t)r * cols;
    bf16* orow = out + (size_t)r * cols;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = b2f(ar[i]) + b2f(br[i]);
        sr[i] = f2b(v);
        const float vq = b2f(sr[i]);          // round-trip through bf16, as launch_add then rms does
        ss += vq * vq;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);
    __shared__ float ws[32];
    if (lane == 0) ws[warp] = ss;
    __syncthreads();
    float tot = 0.f;
    for (int i = 0; i < nwarps; i++) tot += ws[i];
    const float inv = rsqrtf(tot / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        orow[i] = f2b(b2f(sr[i]) * inv * b2f(w[i]));
}

__global__ void k_rms_heads(bf16* x, const bf16* w, int seq, int n_heads, int d, float eps) {
    // One warp per (token, head): d is 128 here, so a head fits a single shuffle reduction and
    // the per-head block barriers disappear entirely.
    const int idx = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (idx >= seq * n_heads) return;
    bf16* h = x + (size_t)idx * d;
    const int lane = threadIdx.x & 31;
    float ss = 0.f;
    for (int i = lane; i < d; i += 32) {
        float v = b2f(h[i]);
        ss += v * v;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    const float inv = rsqrtf(ss / (float)d + eps);
    for (int i = lane; i < d; i += 32)
        h[i] = f2b(b2f(h[i]) * inv * b2f(w[i]));
}

// Per-head RMSNorm immediately followed by RoPE on the same buffer. Two launches per tensor and
// two per layer for q and k is 24 launches a draft block spends on ~37 us of work; the draft is
// launched eagerly (no graph), so each one also pays a full launch gap. One warp per (token, head)
// as in k_rms_heads, then the same rotation k_rope applies, in the same order per element.
__global__ void k_rms_heads_rope(bf16* x, const bf16* w, int seq, int n_heads, int d,
                                 float eps, int pos0, float theta) {
    const int idx = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (idx >= seq * n_heads) return;
    bf16* h = x + (size_t)idx * d;
    const int lane = threadIdx.x & 31;
    float ss = 0.f;
    for (int i = lane; i < d; i += 32) {
        float v = b2f(h[i]);
        ss += v * v;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    const float inv = rsqrtf(ss / (float)d + eps);
    for (int i = lane; i < d; i += 32)
        h[i] = f2b(b2f(h[i]) * inv * b2f(w[i]));
    __syncwarp();
    const int pos = pos0 + (idx / n_heads);
    const int half = d / 2;
    for (int i = lane; i < half; i += 32) {
        const float freq = 1.f / powf(theta, (float)(2 * i) / (float)d);
        const float ang = (float)pos * freq;
        const float c = cosf(ang), sn = sinf(ang);
        const float x0 = b2f(h[i]), x1 = b2f(h[i + half]);
        h[i] = f2b(x0 * c - x1 * sn);
        h[i + half] = f2b(x0 * sn + x1 * c);
    }
}

__global__ void k_rope(bf16* x, int seq, int n_heads, int d, int pos0, float theta) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= seq || h >= n_heads) return;
    bf16* v = x + ((size_t)t * n_heads + h) * d;
    int pos = pos0 + t;
    int half = d / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = 1.f / powf(theta, (float)(2 * i) / (float)d);
        float ang = (float)pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = b2f(v[i]), x1 = b2f(v[i + half]);
        v[i] = f2b(x0 * c - x1 * s);
        v[i + half] = f2b(x0 * s + x1 * c);
    }
}

// One block per (q_token, q_head). Online softmax over kv_len.
__global__ void k_attn(const bf16* q, const bf16* k, const bf16* v, bf16* out,
                       int q_len, int kv_len, int n_q, int n_kv, int d,
                       int q_pos0, int k_pos0, int window, bool causal, float scale) {
    int qt = blockIdx.x;
    int qh = blockIdx.y;
    if (qt >= q_len || qh >= n_q) return;
    const int kv_h = qh / (n_q / n_kv);
    const bf16* qv = q + ((size_t)qt * n_q + qh) * d;
    bf16* ov = out + ((size_t)qt * n_q + qh) * d;
    const int q_pos = q_pos0 + qt;

    extern __shared__ float sm[];
    float* q_s = sm;               // d
    float* acc = sm + d;           // d
    float* red = sm + 2 * d;       // blockDim.x
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        q_s[i] = b2f(qv[i]);
        acc[i] = 0.f;
    }
    __syncthreads();

    float max_s = -1e30f;
    float sum = 0.f;

    for (int t = 0; t < kv_len; t++) {
        const int k_pos = k_pos0 + t;
        if (causal && k_pos > q_pos) continue;
        if (window > 0 && (q_pos - k_pos) >= window) continue;
        const bf16* kv = k + ((size_t)t * n_kv + kv_h) * d;
        float dot = 0.f;
        for (int i = threadIdx.x; i < d; i += blockDim.x)
            dot += q_s[i] * b2f(kv[i]);
        red[threadIdx.x] = dot;
        __syncthreads();
        // Preserve the original 128-way reduction tree, but stop using block-wide
        // barriers once only warp 0 remains. The +64 and +32 stages still go through
        // shared memory; +16..+1 use the equivalent shuffle-down tree. This removes
        // four barriers per KV token without perturbing draft logits / acceptance.
        if (threadIdx.x < 64) red[threadIdx.x] += red[threadIdx.x + 64];
        __syncthreads();
        if (threadIdx.x < 32) red[threadIdx.x] += red[threadIdx.x + 32];
        __syncthreads();
        if (threadIdx.x < 32) {
            float warp_sum = red[threadIdx.x];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                warp_sum += __shfl_down_sync(0xffffffffu, warp_sum, off);
            if (threadIdx.x == 0) red[0] = warp_sum;
        }
        __syncthreads();
        float score = red[0] * scale;
        float new_max = fmaxf(max_s, score);
        float e1 = expf(max_s - new_max);
        float e2 = expf(score - new_max);
        float new_sum = sum * e1 + e2;
        const bf16* vv = v + ((size_t)t * n_kv + kv_h) * d;
        for (int i = threadIdx.x; i < d; i += blockDim.x)
            acc[i] = acc[i] * e1 + e2 * b2f(vv[i]);
        __syncthreads();
        max_s = new_max;
        sum = new_sum;
    }
    float inv = (sum > 0.f) ? (1.f / sum) : 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        ov[i] = f2b(acc[i] * inv);
}

// Split the key sequence across the warps of one CTA, then merge their online-
// softmax states once. This keeps the hd128 Q/output state in registers and exposes
// parallelism across KV tokens as the sequence grows.
__global__ void k_attn_multiwarp_hd128(
    const bf16* __restrict__ q, const bf16* __restrict__ k,
    const bf16* __restrict__ v, bf16* __restrict__ out,
    int q_len, int kv_len, int n_q, int n_kv,
    int q_pos0, int k_pos0, int window, bool causal, float scale) {
    const int qt = blockIdx.x;
    const int qh = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    if (qt >= q_len || qh >= n_q) return;

    constexpr int D = 128;
    constexpr int E = D / 32;
    const int base = lane * E;
    const int kv_h = qh / (n_q / n_kv);
    const bf16* qv = q + ((size_t)qt * n_q + qh) * D + base;
    bf16* ov = out + ((size_t)qt * n_q + qh) * D + base;
    float qr[E], acc[E];
#pragma unroll
    for (int e = 0; e < E; e++) {
        qr[e] = b2f(qv[e]);
        acc[e] = 0.f;
    }

    float max_s = -1e30f;
    float sum = 0.f;
    const int q_pos = q_pos0 + qt;
    for (int t = warp; t < kv_len; t += nwarps) {
        const int k_pos = k_pos0 + t;
        if (causal && k_pos > q_pos) continue;
        if (window > 0 && (q_pos - k_pos) >= window) continue;
        const bf16* kv = k + ((size_t)t * n_kv + kv_h) * D + base;
        float dot = 0.f;
#pragma unroll
        for (int e = 0; e < E; e++) dot += qr[e] * b2f(kv[e]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_down_sync(0xffffffffu, dot, off);
        const float score = __shfl_sync(0xffffffffu, dot, 0) * scale;
        const float new_max = fmaxf(max_s, score);
        const float e1 = expf(max_s - new_max);
        const float e2 = expf(score - new_max);
        sum = sum * e1 + e2;
        const bf16* vv = v + ((size_t)t * n_kv + kv_h) * D + base;
#pragma unroll
        for (int e = 0; e < E; e++) acc[e] = acc[e] * e1 + e2 * b2f(vv[e]);
        max_s = new_max;
    }

    extern __shared__ float sm[];
    float* s_max = sm;
    float* s_sum = sm + nwarps;
    float* s_acc = sm + 2 * nwarps;
    if (lane == 0) {
        s_max[warp] = max_s;
        s_sum[warp] = sum;
    }
#pragma unroll
    for (int e = 0; e < E; e++) s_acc[(size_t)warp * D + base + e] = acc[e];
    __syncthreads();
    if (warp != 0) return;

    float global_max = -1e30f;
    for (int w = 0; w < nwarps; w++) global_max = fmaxf(global_max, s_max[w]);
    float global_sum = 0.f;
    float result[E] = {0.f, 0.f, 0.f, 0.f};
    for (int w = 0; w < nwarps; w++) {
        const float correction = expf(s_max[w] - global_max);
        global_sum += s_sum[w] * correction;
#pragma unroll
        for (int e = 0; e < E; e++)
            result[e] += s_acc[(size_t)w * D + base + e] * correction;
    }
    const float inv = global_sum > 0.f ? 1.f / global_sum : 0.f;
#pragma unroll
    for (int e = 0; e < E; e++) ov[e] = f2b(result[e] * inv);
}

// One warp per output row `n`, computing y[b][n] = sum_k x[b][k] * W[n][k] for all b in
// [0,BATCH) at once. W stays in its native [N,K] "out,in" row-major layout (the same layout
// the single-row GEMV path reads) -- unlike a tiled A@B GEMM, this needs no relayout. Each
// warp reads its weight row from DRAM once and reuses it across the whole BATCH instead of
// once per row, so BATCH separate GEMV launches collapse into one launch with ~BATCH-times
// less weight traffic. Register cost is O(BATCH), so this is only used for the draft's
// fixed block_size batch (16), never for the variable, potentially-large context length.
// K is always a multiple of 8 for every weight this is used on (H=2048, qdim=4096,
// kvdim=1024, I=6144), and cudaMalloc'd row bases land on >=16-byte boundaries, so every
// lane's 8-element (16-byte) chunk is safely loadable as one uint4 -- this is the kernel's
// dominant cost (each lane otherwise issues K/32 separate 2-byte loads per weight row), same
// lesson as the fused prefill dequant-GEMM kernels: load width, not FLOPs, is the limiter here.
template <int BATCH>
__global__ void k_gemv_batched(const bf16* __restrict__ x, const bf16* __restrict__ W,
                               bf16* __restrict__ y, int N, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int n = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    if (n >= N) return;
    const int lane = threadIdx.x & 31;
    float acc[BATCH];
#pragma unroll
    for (int b = 0; b < BATCH; b++) acc[b] = 0.f;
    const uint4* wrow4 = (const uint4*)(W + (size_t)n * K);
    const int K8 = K / 8;
    for (int k8 = lane; k8 < K8; k8 += 32) {
        // Materialize the wide packets in registers. Merely taking a bf16 pointer
        // into wrow4 lets ptxas scalarize every unrolled element access.
        const uint4 wp = wrow4[k8];
        const bf16* wv = (const bf16*)&wp;
#pragma unroll
        for (int b = 0; b < BATCH; b++) {
            const uint4 xp = ((const uint4*)(x + (size_t)b * K))[k8];
            const bf16* xv = (const bf16*)&xp;
#pragma unroll
            for (int j = 0; j < 8; j++) acc[b] += b2f(wv[j]) * b2f(xv[j]);
        }
    }
#pragma unroll
    for (int b = 0; b < BATCH; b++) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[b] += __shfl_down_sync(0xffffffffu, acc[b], off);
    }
    if (lane == 0) {
#pragma unroll
        for (int b = 0; b < BATCH; b++) y[(size_t)b * N + n] = f2b(acc[b]);
    }
}

// Route several projections that share x through one grid, ROWS output rows per warp.
// One row per warp re-reads the whole BATCH-row activation block for EVERY output row: per lane
// and K-chunk that is BATCH uint4 activation loads and BATCH*8 bf16->float converts to feed only
// BATCH*8 MACs, and across a layer the activation re-reads (~98 MB) outweigh the weight stream
// (~50 MB) that the kernel exists to do once. Giving a warp ROWS rows amortizes those loads AND
// their converts over ROWS weight rows: loads per chunk go from BATCH+1 per BATCH*8 MACs to
// BATCH+ROWS per ROWS*BATCH*8. The weight rows still stream exactly once. Measured monotone over
// a 1/2/4/8 sweep on an RTX 5090.
template <int BATCH, int ROWS>
__global__ void k_gemv_batched_fused3(
        const bf16* __restrict__ x,
        const bf16* __restrict__ W0, const bf16* __restrict__ W1,
        const bf16* __restrict__ W2,
        bf16* __restrict__ y0, bf16* __restrict__ y1, bf16* __restrict__ y2,
        int N0, int N1, int N2, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int global_n = (blockIdx.x * warps_per_block + (threadIdx.x / 32)) * ROWS;
    const int total = N0 + N1 + N2;
    if (global_n >= total) return;
    const bf16* W;
    bf16* y;
    int N, n0;
    if (global_n < N0) {
        W = W0; y = y0; N = N0; n0 = global_n;
    } else if (global_n < N0 + N1) {
        W = W1; y = y1; N = N1; n0 = global_n - N0;
    } else {
        W = W2; y = y2; N = N2; n0 = global_n - N0 - N1;
    }
    // A warp's ROWS rows never straddle two projections: every N here is a multiple of ROWS.
    const int nr = (N - n0 < ROWS) ? (N - n0) : ROWS;
    const int lane = threadIdx.x & 31;
    float acc[ROWS][BATCH];
#pragma unroll
    for (int r = 0; r < ROWS; r++)
#pragma unroll
        for (int b = 0; b < BATCH; b++) acc[r][b] = 0.f;
    const int K8 = K / 8;
    for (int k8 = lane; k8 < K8; k8 += 32) {
        float wf[ROWS][8];
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            // Clamp the tail rows onto row 0; their results are simply not stored below.
            const int rr = (r < nr) ? r : 0;
            const uint4 wp = reinterpret_cast<const uint4*>(W + (size_t)(n0 + rr) * K)[k8];
            const bf16* wv = reinterpret_cast<const bf16*>(&wp);
#pragma unroll
            for (int j = 0; j < 8; j++) wf[r][j] = b2f(wv[j]);
        }
#pragma unroll
        for (int b = 0; b < BATCH; b++) {
            const uint4 xp = reinterpret_cast<const uint4*>(x + (size_t)b * K)[k8];
            const bf16* xv = reinterpret_cast<const bf16*>(&xp);
            float xf[8];
#pragma unroll
            for (int j = 0; j < 8; j++) xf[j] = b2f(xv[j]);
#pragma unroll
            for (int r = 0; r < ROWS; r++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[r][b] += wf[r][j] * xf[j];
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
#pragma unroll
        for (int b = 0; b < BATCH; b++) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[r][b] += __shfl_down_sync(0xffffffffu, acc[r][b], off);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            if (r >= nr) continue;
#pragma unroll
            for (int b = 0; b < BATCH; b++) y[(size_t)b * N + n0 + r] = f2b(acc[r][b]);
        }
    }
}

// Same batched-GEMV shape as k_gemv_batched, but for the LM head: fp32 accumulate/output
// (matching the precision the original per-row launch_gemv_q_f32/launch_gemv_f32 path used
// for logits) and no BATCH-sized register cap on N (vocab is large, only grid.x scales).
template <int BATCH>
__global__ void k_gemv_batched_f32(const bf16* __restrict__ x, const bf16* __restrict__ W,
                                   float* __restrict__ y, int N, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int n = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    if (n >= N) return;
    const int lane = threadIdx.x & 31;
    float acc[BATCH];
#pragma unroll
    for (int b = 0; b < BATCH; b++) acc[b] = 0.f;
    const uint4* wrow4 = (const uint4*)(W + (size_t)n * K);
    const int K8 = K / 8;
    for (int k8 = lane; k8 < K8; k8 += 32) {
        const uint4 wp = wrow4[k8];
        const bf16* wv = (const bf16*)&wp;
#pragma unroll
        for (int b = 0; b < BATCH; b++) {
            const uint4 xp = ((const uint4*)(x + (size_t)b * K))[k8];
            const bf16* xv = (const bf16*)&xp;
#pragma unroll
            for (int j = 0; j < 8; j++) acc[b] += b2f(wv[j]) * b2f(xv[j]);
        }
    }
#pragma unroll
    for (int b = 0; b < BATCH; b++) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[b] += __shfl_down_sync(0xffffffffu, acc[b], off);
    }
    if (lane == 0) {
#pragma unroll
        for (int b = 0; b < BATCH; b++) y[(size_t)b * N + n] = acc[b];
    }
}

// Arithmetic-identical to kernels::gemv_f32_sk_kernel<bf16,8> for N<4096,
// extended with grid.y for independent activation rows. Each CTA still maps to
// one (batch, output) pair and uses the same per-lane K traversal, XOR warp
// reduction, and ordered eight-way split sum as the individual launches.
__global__ void k_gemv_rows_exact_s8(const bf16* __restrict__ x,
                                     const bf16* __restrict__ W,
                                     bf16* __restrict__ y, int rows, int N, int K) {
    const int n = blockIdx.x;
    const int batch = blockIdx.y;
    const int split = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (n >= N || batch >= rows) return;
    const uint4* w4 = reinterpret_cast<const uint4*>(W + (size_t)n * K);
    const uint4* x4 = reinterpret_cast<const uint4*>(x + (size_t)batch * K);
    const int K8 = K / 8;
    float acc = 0.f;
    for (int i = split * 32 + lane; i < K8; i += 8 * 32) {
        const uint4 wp = w4[i];
        const uint4 xp = x4[i];
        const __nv_bfloat162* wh = reinterpret_cast<const __nv_bfloat162*>(&wp);
        const __nv_bfloat162* xh = reinterpret_cast<const __nv_bfloat162*>(&xp);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 wf = __bfloat1622float2(wh[j]);
            const float2 xf = __bfloat1622float2(xh[j]);
            acc += wf.x * xf.x + wf.y * xf.y;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    __shared__ float partial[8];
    if (lane == 0) partial[split] = acc;
    __syncthreads();
    if (split == 0 && lane == 0) {
        float result = 0.f;
#pragma unroll
        for (int s = 0; s < 8; s++) result += partial[s];
        y[(size_t)batch * N + n] = f2b(result);
    }
}

// Row-batched form of the context projection. The exact split-K kernel maps one CTA to each
// (context row, output) pair, so it re-reads the whole weight once PER CONTEXT ROW. Here one warp
// owns an output row and accumulates a tile of RMAX context rows from a SINGLE weight pass, so the
// weight streams ceil(rows/RMAX) times instead of `rows` times. Draft-only: these feed the
// proposals, and every emitted token is still a target argmax, so accumulation order is free.
template <int RMAX>
__global__ void k_gemv_rows_batched_fused2(
        const bf16* __restrict__ x,
        const bf16* __restrict__ W0, const bf16* __restrict__ W1,
        bf16* __restrict__ y0, bf16* __restrict__ y1,
        int rows, int N0, int N1, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int global_n = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    if (global_n >= N0 + N1) return;
    const bool second = global_n >= N0;
    const int n = second ? global_n - N0 : global_n;
    const int N = second ? N1 : N0;
    const bf16* __restrict__ W = second ? W1 : W0;
    bf16* __restrict__ y = second ? y1 : y0;
    const int r0 = blockIdx.y * RMAX;
    const int nr = (rows - r0 < RMAX) ? (rows - r0) : RMAX;
    if (nr <= 0) return;
    const int lane = threadIdx.x & 31;
    float acc[RMAX];
#pragma unroll
    for (int r = 0; r < RMAX; r++) acc[r] = 0.f;
    const uint4* w4 = reinterpret_cast<const uint4*>(W + (size_t)n * K);
    const int K8 = K / 8;
    for (int k8 = lane; k8 < K8; k8 += 32) {
        const uint4 wp = w4[k8];
        const __nv_bfloat162* wh = reinterpret_cast<const __nv_bfloat162*>(&wp);
#pragma unroll
        for (int r = 0; r < RMAX; r++) {
            const int rr = r0 + ((r < nr) ? r : 0);
            const uint4 xp = reinterpret_cast<const uint4*>(x + (size_t)rr * K)[k8];
            const __nv_bfloat162* xh = reinterpret_cast<const __nv_bfloat162*>(&xp);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 wf = __bfloat1622float2(wh[j]);
                const float2 xf = __bfloat1622float2(xh[j]);
                acc[r] += wf.x * xf.x + wf.y * xf.y;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RMAX; r++) {
        float a = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0 && r < nr) y[(size_t)(r0 + r) * N + n] = f2b(a);
    }
}

// bf16 -> Q8_0-style int8 with one fp32 scale per 32 weights, one warp per 32-weight block.
// Done once at load: the draft's per-block weight stream is ~705 MB of bf16, and after the
// diffusion width narrowed to 4 rows these projections are squarely DRAM-bound, so halving the
// bytes is the only remaining lever. Draft-only, and every emitted token is still a target
// argmax, so the quantization can only move the accept length, never the output.
__global__ void k_quantize_w_q8(const bf16* __restrict__ w, signed char* __restrict__ q,
                                float* __restrict__ sc, long nblocks) {
    const long blk = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (blk >= nblocks) return;
    const int lane = threadIdx.x & 31;
    const float v = b2f(w[blk * 32 + lane]);
    float a = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    const float d = a / 127.0f;
    q[blk * 32 + lane] = (signed char)(a == 0.f ? 0 : __float2int_rn(v / d));
    if (lane == 0) sc[blk] = d;
}

// Q8_0 weights, bf16 activations. Same tiling as k_gemv_batched_fused3: ROWS output rows per
// warp so the BATCH activation loads amortize; the weight side now reads 8 bytes instead of 16
// per K-chunk, plus one fp32 block scale per 4 chunks.
template <int BATCH, int ROWS>
__global__ void k_gemv_batched_fused3_q8(
        const bf16* __restrict__ x,
        const signed char* __restrict__ Q0, const signed char* __restrict__ Q1,
        const signed char* __restrict__ Q2,
        const float* __restrict__ S0, const float* __restrict__ S1, const float* __restrict__ S2,
        bf16* __restrict__ y0, bf16* __restrict__ y1, bf16* __restrict__ y2,
        int N0, int N1, int N2, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int total = N0 + N1 + N2;
    // Grid-stride over row tiles: the launcher caps the grid so this leaves SM slots for the
    // concurrent target verify forward (see launch_gemv_batched_q8_fused3).
    for (int global_n = (blockIdx.x * warps_per_block + (threadIdx.x / 32)) * ROWS;
         global_n < total; global_n += gridDim.x * warps_per_block * ROWS) {
    const signed char* Q; const float* S; bf16* y; int N, n0;
    if (global_n < N0)            { Q = Q0; S = S0; y = y0; N = N0; n0 = global_n; }
    else if (global_n < N0 + N1)  { Q = Q1; S = S1; y = y1; N = N1; n0 = global_n - N0; }
    else                          { Q = Q2; S = S2; y = y2; N = N2; n0 = global_n - N0 - N1; }
    const int nr = (N - n0 < ROWS) ? (N - n0) : ROWS;
    const int lane = threadIdx.x & 31;
    const int K8 = K / 8, KS = K / 32;
    float acc[ROWS][BATCH];
#pragma unroll
    for (int r = 0; r < ROWS; r++)
#pragma unroll
        for (int b = 0; b < BATCH; b++) acc[r][b] = 0.f;
    for (int k8 = lane; k8 < K8; k8 += 32) {
        float wf[ROWS][8];
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int rr = (r < nr) ? r : 0;
            const uint2 wp = reinterpret_cast<const uint2*>(Q + (size_t)(n0 + rr) * K)[k8];
            const signed char* wv = reinterpret_cast<const signed char*>(&wp);
            const float d = S[(size_t)(n0 + rr) * KS + (k8 >> 2)];
#pragma unroll
            for (int j = 0; j < 8; j++) wf[r][j] = (float)wv[j] * d;
        }
#pragma unroll
        for (int b = 0; b < BATCH; b++) {
            const uint4 xp = reinterpret_cast<const uint4*>(x + (size_t)b * K)[k8];
            const bf16* xv = reinterpret_cast<const bf16*>(&xp);
            float xf[8];
#pragma unroll
            for (int j = 0; j < 8; j++) xf[j] = b2f(xv[j]);
#pragma unroll
            for (int r = 0; r < ROWS; r++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[r][b] += wf[r][j] * xf[j];
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; r++)
#pragma unroll
        for (int b = 0; b < BATCH; b++) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[r][b] += __shfl_down_sync(0xffffffffu, acc[r][b], off);
        }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            if (r >= nr) continue;
#pragma unroll
            for (int b = 0; b < BATCH; b++) y[(size_t)b * N + n0 + r] = f2b(acc[r][b]);
        }
    }
    }
}

// Copy one captured hidden row into dflash_hidden[row][slot]. `row` is read from DEVICE memory,
// which is the whole point: the destination row changes on every verify token, but a captured
// CUDA graph bakes its node arguments, so a host-side destination forced the capture into a
// row-independent staging buffer that then had to be flushed by a second, out-of-graph memcpy on
// every verify token. Reading the row on the device lets the capture write its final location
// from inside the graph, and the per-token flush disappears.
__global__ void k_capture_row(const bf16* __restrict__ x, bf16* __restrict__ hidden,
                              const int* __restrict__ cap_row, int slot, int H,
                              int row_elems, int max_rows) {
    const int r = *cap_row;
    if (r < 0 || r >= max_rows) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H / 8) return;
    uint4* dst = (uint4*)(hidden + (size_t)r * row_elems + (size_t)slot * H);
    dst[i] = ((const uint4*)x)[i];
}

// bf16 -> asymmetric int4: 32 weights per block as packed nibbles plus an fp16 (scale, min) pair.
// ~5 bits/weight against Q8_0's 9, so the draft's per-block projection stream drops from ~353 MB
// to ~196 MB. Asymmetric (scale+min, like Q4_K) rather than symmetric: at 4 bits the extra
// degree of freedom is what keeps the proposals good enough that the accept length holds.
__global__ void k_quantize_w_q4(const bf16* __restrict__ w, unsigned char* __restrict__ q,
                                __half2* __restrict__ dm, long nblocks) {
    const long blk = (long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (blk >= nblocks) return;
    const int lane = threadIdx.x & 31;
    const float v = b2f(w[blk * 32 + lane]);
    float mn = v, mx = v;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, off));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
    }
    const float d = (mx - mn) / 15.0f;
    const int qi = (d > 0.f) ? __float2int_rn((v - mn) / d) : 0;
    const unsigned int nib = (unsigned)(qi < 0 ? 0 : (qi > 15 ? 15 : qi));
    // Two lanes share a byte: even lane -> low nibble, odd lane -> high nibble.
    const unsigned int other = __shfl_xor_sync(0xffffffffu, nib, 1);
    if ((lane & 1) == 0) q[blk * 16 + (lane >> 1)] = (unsigned char)(nib | (other << 4));
    if (lane == 0) dm[blk] = __floats2half2_rn(d, mn);
}

// int4 weights, bf16 activations; same tiling as the Q8 variant.
template <int BATCH, int ROWS>
__global__ void k_gemv_batched_fused3_q4(
        const bf16* __restrict__ x,
        const unsigned char* __restrict__ Q0, const unsigned char* __restrict__ Q1,
        const unsigned char* __restrict__ Q2,
        const __half2* __restrict__ D0, const __half2* __restrict__ D1, const __half2* __restrict__ D2,
        bf16* __restrict__ y0, bf16* __restrict__ y1, bf16* __restrict__ y2,
        int N0, int N1, int N2, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int total = N0 + N1 + N2;
    for (int global_n = (blockIdx.x * warps_per_block + (threadIdx.x / 32)) * ROWS;
         global_n < total; global_n += gridDim.x * warps_per_block * ROWS) {
    const unsigned char* Q; const __half2* D; bf16* y; int N, n0;
    if (global_n < N0)           { Q = Q0; D = D0; y = y0; N = N0; n0 = global_n; }
    else if (global_n < N0 + N1) { Q = Q1; D = D1; y = y1; N = N1; n0 = global_n - N0; }
    else                         { Q = Q2; D = D2; y = y2; N = N2; n0 = global_n - N0 - N1; }
    const int nr = (N - n0 < ROWS) ? (N - n0) : ROWS;
    const int lane = threadIdx.x & 31;
    const int K8 = K / 8, KS = K / 32;
    float acc[ROWS][BATCH];
#pragma unroll
    for (int r = 0; r < ROWS; r++)
#pragma unroll
        for (int b = 0; b < BATCH; b++) acc[r][b] = 0.f;
    for (int k8 = lane; k8 < K8; k8 += 32) {
        float wf[ROWS][8];
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int rr = (r < nr) ? r : 0;
            const unsigned int packed =
                reinterpret_cast<const unsigned int*>(Q + (size_t)(n0 + rr) * (K / 2))[k8];
            const float2 dmf = __half22float2(D[(size_t)(n0 + rr) * KS + (k8 >> 2)]);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const unsigned int byte = (packed >> (8 * j)) & 0xFFu;
                wf[r][2 * j]     = (float)(byte & 0xFu) * dmf.x + dmf.y;
                wf[r][2 * j + 1] = (float)(byte >> 4)   * dmf.x + dmf.y;
            }
        }
#pragma unroll
        for (int b = 0; b < BATCH; b++) {
            const uint4 xp = reinterpret_cast<const uint4*>(x + (size_t)b * K)[k8];
            const bf16* xv = reinterpret_cast<const bf16*>(&xp);
            float xf[8];
#pragma unroll
            for (int j = 0; j < 8; j++) xf[j] = b2f(xv[j]);
#pragma unroll
            for (int r = 0; r < ROWS; r++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[r][b] += wf[r][j] * xf[j];
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; r++)
#pragma unroll
        for (int b = 0; b < BATCH; b++) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[r][b] += __shfl_down_sync(0xffffffffu, acc[r][b], off);
        }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            if (r >= nr) continue;
#pragma unroll
            for (int b = 0; b < BATCH; b++) y[(size_t)b * N + n0 + r] = f2b(acc[r][b]);
        }
    }
    }
}

// Same math as k_gemv_batched_fused3_q4, but the KSPLIT warps of a CTA cooperate on ONE group of
// ROWS weight rows, each sweeping a 1/KSPLIT stripe of K, instead of each owning a different group.
// That multiplies the CTA count by KSPLIT without touching the activation traffic (a CTA still
// reads BATCH*K activations exactly once, just spread over its warps). The draft's projections run
// at 0.75-4.5 CTAs/SM on a 170-SM device -- far too few to cover DRAM latency -- so parallelism,
// not bytes, is what they are short of. Cutting ROWS instead would also add CTAs, but it multiplies
// the activation re-reads, which is why 4 -> 2 -> 1 measured progressively worse.
template <int BATCH, int ROWS, int KSPLIT>
__global__ void k_gemv_batched_fused3_q4_ks(
        const bf16* __restrict__ x,
        const unsigned char* __restrict__ Q0, const unsigned char* __restrict__ Q1,
        const unsigned char* __restrict__ Q2,
        const __half2* __restrict__ D0, const __half2* __restrict__ D1, const __half2* __restrict__ D2,
        bf16* __restrict__ y0, bf16* __restrict__ y1, bf16* __restrict__ y2,
        int N0, int N1, int N2, int K) {
    __shared__ float red[KSPLIT][ROWS][BATCH];
    const int total = N0 + N1 + N2;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    for (int global_n = blockIdx.x * ROWS; global_n < total; global_n += gridDim.x * ROWS) {
        const unsigned char* Q; const __half2* D; bf16* y; int N, n0;
        if (global_n < N0)           { Q = Q0; D = D0; y = y0; N = N0; n0 = global_n; }
        else if (global_n < N0 + N1) { Q = Q1; D = D1; y = y1; N = N1; n0 = global_n - N0; }
        else                         { Q = Q2; D = D2; y = y2; N = N2; n0 = global_n - N0 - N1; }
        const int nr = (N - n0 < ROWS) ? (N - n0) : ROWS;
        const int K8 = K / 8, KS = K / 32;
        float acc[ROWS][BATCH];
#pragma unroll
        for (int r = 0; r < ROWS; r++)
#pragma unroll
            for (int b = 0; b < BATCH; b++) acc[r][b] = 0.f;
        for (int k8 = warp * 32 + lane; k8 < K8; k8 += KSPLIT * 32) {
            float wf[ROWS][8];
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int rr = (r < nr) ? r : 0;
                const unsigned int packed =
                    reinterpret_cast<const unsigned int*>(Q + (size_t)(n0 + rr) * (K / 2))[k8];
                const float2 dmf = __half22float2(D[(size_t)(n0 + rr) * KS + (k8 >> 2)]);
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const unsigned int byte = (packed >> (8 * j)) & 0xFFu;
                    wf[r][2 * j]     = (float)(byte & 0xFu) * dmf.x + dmf.y;
                    wf[r][2 * j + 1] = (float)(byte >> 4)   * dmf.x + dmf.y;
                }
            }
#pragma unroll
            for (int b = 0; b < BATCH; b++) {
                const uint4 xp = reinterpret_cast<const uint4*>(x + (size_t)b * K)[k8];
                const bf16* xv = reinterpret_cast<const bf16*>(&xp);
                float xf[8];
#pragma unroll
                for (int j = 0; j < 8; j++) xf[j] = b2f(xv[j]);
#pragma unroll
                for (int r = 0; r < ROWS; r++)
#pragma unroll
                    for (int j = 0; j < 8; j++) acc[r][b] += wf[r][j] * xf[j];
            }
        }
#pragma unroll
        for (int r = 0; r < ROWS; r++)
#pragma unroll
            for (int b = 0; b < BATCH; b++) {
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    acc[r][b] += __shfl_down_sync(0xffffffffu, acc[r][b], off);
            }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < ROWS; r++)
#pragma unroll
                for (int b = 0; b < BATCH; b++) red[warp][r][b] = acc[r][b];
        }
        __syncthreads();
        // Fold the KSPLIT partials, one thread per (row, batch) output.
        for (int i = threadIdx.x; i < ROWS * BATCH; i += blockDim.x) {
            const int r = i / BATCH, b = i - r * BATCH;
            if (r >= nr) continue;
            float sum = 0.f;
#pragma unroll
            for (int s = 0; s < KSPLIT; s++) sum += red[s][r][b];
            y[(size_t)b * N + n0 + r] = f2b(sum);
        }
        __syncthreads();
    }
}

} // namespace

void launch_quantize_w_q4(const void* w, void* q, void* dm, int N, int K, cudaStream_t stream) {
    if (N <= 0 || (K & 31) != 0) return;
    const long nblocks = (long)N * (K / 32);
    constexpr int WPB = 8;
    k_quantize_w_q4<<<(nblocks + WPB - 1) / WPB, WPB * 32, 0, stream>>>(
        (const bf16*)w, (unsigned char*)q, (__half2*)dm, nblocks);
}

void launch_gemv_batched_q4_fused3(const void* x,
                                   const void* Q0, const void* Q1, const void* Q2,
                                   const void* D0, const void* D1, const void* D2,
                                   void* y0, void* y1, void* y2,
                                   int N0, int N1, int N2, int K, cudaStream_t stream,
                                   int batch) {
    const int total = N0 + N1 + N2;
    if (total <= 0) return;
    // ROWS is the number of WEIGHT rows a warp owns, which amortizes the activation re-reads --- but
    // acc[ROWS][BATCH] plus wf[ROWS][8] are both live across the K loop, so at ROWS=8, BATCH=8 the
    // kernel compiled to 167 registers and only 3 CTAs fit an SM (~19% occupancy), leaving it ~3.7x
    // off HBM roofline. 4 halves both arrays and roughly doubles resident CTAs; measured 861.9 tok/s
    // vs 855.4 at 8, with 3 and 5 within noise of 4.
    constexpr int WPB = 4, ROWS = 4;
    // K-split factor: how many warps of a CTA cooperate on one group of ROWS weight rows.
    // 2 measured best (1008.5 -> 1072.6 tok/s against the legacy kernel on the eval prompt); 4 is
    // within noise and 8 is worse, and selecting it per launch from the grid size was worse still
    // -- these projections are not latency-starved, they just want a second warp on the same rows.
    // 0 restores the legacy one-group-per-warp kernel.
    static const int ksplit = []{
        const char* e = getenv("SPARKINFER_DFLASH_KSPLIT");
        int v = e ? atoi(e) : 2;
        return (v == 2 || v == 4 || v == 8) ? v : 0;
    }();
    const int nblk = (total + ROWS - 1) / ROWS;
    if (ksplit) {
        dim3 grid(nblk);
        const dim3 blk(ksplit * 32);
#define SI_Q4KS(BW_, KS_) k_gemv_batched_fused3_q4_ks<BW_, ROWS, KS_><<<grid, blk, 0, stream>>>(  \
        (const bf16*)x, (const unsigned char*)Q0, (const unsigned char*)Q1,                       \
        (const unsigned char*)Q2, (const __half2*)D0, (const __half2*)D1, (const __half2*)D2,     \
        (bf16*)y0, (bf16*)y1, (bf16*)y2, N0, N1, N2, K)
#define SI_Q4KS_B(BW_) do { if (ksplit == 2) SI_Q4KS(BW_, 2);                                     \
                            else if (ksplit == 4) SI_Q4KS(BW_, 4);                                \
                            else SI_Q4KS(BW_, 8); } while (0)
        if (batch == 4)      SI_Q4KS_B(4);
        else if (batch == 8) SI_Q4KS_B(8);
        else                 SI_Q4KS_B(16);
#undef SI_Q4KS_B
#undef SI_Q4KS
        return;
    }
    const int warps = nblk;
    dim3 grid((warps + WPB - 1) / WPB);
    const dim3 blk(WPB * 32);
#define SI_Q4F3(BW_) k_gemv_batched_fused3_q4<BW_, ROWS><<<grid, blk, 0, stream>>>(              \
        (const bf16*)x, (const unsigned char*)Q0, (const unsigned char*)Q1,                     \
        (const unsigned char*)Q2, (const __half2*)D0, (const __half2*)D1, (const __half2*)D2,   \
        (bf16*)y0, (bf16*)y1, (bf16*)y2, N0, N1, N2, K)
    if (batch == 4)      SI_Q4F3(4);
    else if (batch == 8) SI_Q4F3(8);
    else                 SI_Q4F3(16);
#undef SI_Q4F3
}

void launch_capture_row(const void* x, void* hidden, const int* cap_row, int slot, int H,
                        int row_elems, int max_rows, cudaStream_t stream) {
    if (H <= 0 || (H & 7) != 0) return;
    const int n4 = H / 8;
    k_capture_row<<<(n4 + 255) / 256, 256, 0, stream>>>(
        (const bf16*)x, (bf16*)hidden, cap_row, slot, H, row_elems, max_rows);
}

void launch_quantize_w_q8(const void* w, void* q, float* sc, int N, int K, cudaStream_t stream) {
    if (N <= 0 || (K & 31) != 0) return;
    const long nblocks = (long)N * (K / 32);
    constexpr int WPB = 8;
    k_quantize_w_q8<<<(nblocks + WPB - 1) / WPB, WPB * 32, 0, stream>>>(
        (const bf16*)w, (signed char*)q, sc, nblocks);
}

void launch_gemv_batched_q8_fused3(const void* x,
                                   const void* Q0, const void* Q1, const void* Q2,
                                   const float* S0, const float* S1, const float* S2,
                                   void* y0, void* y1, void* y2,
                                   int N0, int N1, int N2, int K, cudaStream_t stream,
                                   int batch) {
    const int total = N0 + N1 + N2;
    if (total <= 0) return;
    constexpr int WPB = 4, ROWS = 8;
    static const int cap = []{ const char* e = getenv("SPARKINFER_DFLASH_PROJ_CTAS");
                               return e ? atoi(e) : 0; }();
    const int warps = (total + ROWS - 1) / ROWS;
    int nblk = (warps + WPB - 1) / WPB;
    if (cap > 0 && nblk > cap) nblk = cap;
    dim3 grid(nblk);
    const dim3 blk(WPB * 32);
#define SI_Q8F3(BW_) k_gemv_batched_fused3_q8<BW_, ROWS><<<grid, blk, 0, stream>>>(               \
        (const bf16*)x, (const signed char*)Q0, (const signed char*)Q1, (const signed char*)Q2,   \
        S0, S1, S2, (bf16*)y0, (bf16*)y1, (bf16*)y2, N0, N1, N2, K)
    if (batch == 4)      SI_Q8F3(4);
    else if (batch == 8) SI_Q8F3(8);
    else                 SI_Q8F3(16);
#undef SI_Q8F3
}

void launch_add(const void* x, const void* y, void* out, int n, cudaStream_t stream) {
    if (n <= 0) return;
    k_add<<<(n + 255) / 256, 256, 0, stream>>>((const bf16*)x, (const bf16*)y, (bf16*)out, n);
}

void launch_swiglu(const void* gate, const void* up, void* out, int n, cudaStream_t stream) {
    if (n <= 0) return;
    k_swiglu<<<(n + 255) / 256, 256, 0, stream>>>((const bf16*)gate, (const bf16*)up, (bf16*)out, n);
}

void launch_rms(const void* x, const void* w, void* out, int rows, int cols, float eps,
                cudaStream_t stream) {
    if (rows <= 0) return;
    k_rms<<<rows, 256, 0, stream>>>((const bf16*)x, (const bf16*)w, (bf16*)out, rows, cols, eps);
}

void launch_add_rms(const void* a, const void* b, void* sum, const void* w, void* out,
                    int rows, int cols, float eps, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    k_add_rms<<<rows, 256, 0, stream>>>((const bf16*)a, (const bf16*)b, (bf16*)sum,
                                        (const bf16*)w, (bf16*)out, rows, cols, eps);
}

void launch_rms_heads_rope(void* x, const void* w, int seq, int n_heads, int d, float eps,
                           int pos0, float theta, cudaStream_t stream) {
    if (seq <= 0 || n_heads <= 0) return;
    constexpr int WPB = 4;
    const int total = seq * n_heads;
    k_rms_heads_rope<<<(total + WPB - 1) / WPB, WPB * 32, 0, stream>>>(
        (bf16*)x, (const bf16*)w, seq, n_heads, d, eps, pos0, theta);
}

void launch_rms_heads(void* x, const void* w, int seq, int n_heads, int d, float eps,
                      cudaStream_t stream) {
    int n = seq * n_heads;
    if (n <= 0) return;
    constexpr int WPB = 4;                       // one warp per (token, head)
    k_rms_heads<<<(n + WPB - 1) / WPB, WPB * 32, 0, stream>>>(
        (bf16*)x, (const bf16*)w, seq, n_heads, d, eps);
}

void launch_rope_seq(void* x, int seq, int n_heads, int d, int pos0, float theta,
                     cudaStream_t stream) {
    if (seq <= 0) return;
    dim3 grid(seq, n_heads);
    k_rope<<<grid, 64, 0, stream>>>((bf16*)x, seq, n_heads, d, pos0, theta);
}

void launch_attn_gqa(const void* q, const void* k, const void* v, void* out,
                     int q_len, int kv_len, int n_q, int n_kv, int d,
                     int q_pos0, int k_pos0, int window, bool causal, float scale,
                     cudaStream_t stream) {
    if (q_len <= 0 || kv_len <= 0) return;
    dim3 grid(q_len, n_q);
    if (d == 128) {
        constexpr int THREADS = 512;
        constexpr int NWARPS = THREADS / 32;
        constexpr int SMEM = (2 * NWARPS + NWARPS * 128) * sizeof(float);
        k_attn_multiwarp_hd128<<<grid, THREADS, SMEM, stream>>>(
            (const bf16*)q, (const bf16*)k, (const bf16*)v, (bf16*)out,
            q_len, kv_len, n_q, n_kv, q_pos0, k_pos0, window, causal, scale);
        return;
    }
    int smem = (2 * d + 128) * (int)sizeof(float);
    k_attn<<<grid, 128, smem, stream>>>((const bf16*)q, (const bf16*)k, (const bf16*)v,
                                        (bf16*)out, q_len, kv_len, n_q, n_kv, d,
                                        q_pos0, k_pos0, window, causal, scale);
}

void launch_gemv_batched16(const void* x, const void* W, void* y, int N, int K,
                           cudaStream_t stream, int batch) {
    if (N <= 0) return;
    // Single projection = the fused path with one entry, so wo/down get the same
    // ROWS-per-warp activation amortization as Q/K/V and gate/up.
    launch_gemv_batched16_fused3(x, W, nullptr, nullptr, y, nullptr, nullptr, N, 0, 0, K, stream, batch);
}

void launch_gemv_batched16_fused3(const void* x,
                                  const void* W0, const void* W1, const void* W2,
                                  void* y0, void* y1, void* y2,
                                  int N0, int N1, int N2, int K, cudaStream_t stream,
                                  int batch) {
    const int total = N0 + N1 + N2;
    if (total <= 0) return;
    constexpr int WARPS_PER_BLOCK = 4, ROWS = 8;
    const int warps = (total + ROWS - 1) / ROWS;
    dim3 grid((warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    const dim3 blk(WARPS_PER_BLOCK * 32);
#define SI_FUSED3(BW_) k_gemv_batched_fused3<BW_, ROWS><<<grid, blk, 0, stream>>>(              \
        (const bf16*)x, (const bf16*)W0, (const bf16*)W1, (const bf16*)W2,                     \
        (bf16*)y0, (bf16*)y1, (bf16*)y2, N0, N1, N2, K)
    if (batch == 4)       SI_FUSED3(4);
    else if (batch == 8)  SI_FUSED3(8);
    else                  SI_FUSED3(16);
#undef SI_FUSED3
}

void launch_gemv_batched16_fused2(const void* x,
                                  const void* W0, const void* W1,
                                  void* y0, void* y1,
                                  int N0, int N1, int K, cudaStream_t stream, int batch) {
    launch_gemv_batched16_fused3(x, W0, W1, nullptr, y0, y1, nullptr,
                                 N0, N1, 0, K, stream, batch);
}

void launch_gemv_rows_batched(const void* x, const void* W, void* y,
                              int rows, int N, int K, cudaStream_t stream) {
    if (rows <= 0 || N <= 0) return;
    launch_gemv_rows_exact_fused2(x, W, nullptr, y, nullptr, rows, N, 0, K, stream);
}

void launch_gemv_rows_exact(const void* x, const void* W, void* y,
                            int rows, int N, int K, cudaStream_t stream) {
    if (rows <= 0 || N <= 0) return;
    dim3 grid(N, rows);
    k_gemv_rows_exact_s8<<<grid, 8 * 32, 0, stream>>>(
        (const bf16*)x, (const bf16*)W, (bf16*)y, rows, N, K);
}

void launch_gemv_rows_exact_fused2(const void* x,
                                   const void* W0, const void* W1,
                                   void* y0, void* y1,
                                   int rows, int N0, int N1, int K,
                                   cudaStream_t stream) {
    if (rows <= 0 || N0 + N1 <= 0) return;
    // RMAX is the activation-row tile. The r loop runs the full RMAX and folds r >= nr back onto
    // row 0, so any slack is redundant work: with kProposalDepth=5 the accepted prefix this
    // projects is at most 6 rows, and 8 spent a quarter of the kernel recomputing row 0.
    constexpr int RMAX = 6, WPB = 4;
    dim3 grid((N0 + N1 + WPB - 1) / WPB, (rows + RMAX - 1) / RMAX);
    k_gemv_rows_batched_fused2<RMAX><<<grid, WPB * 32, 0, stream>>>(
        (const bf16*)x, (const bf16*)W0, (const bf16*)W1,
        (bf16*)y0, (bf16*)y1, rows, N0, N1, K);
}

void launch_gemv_batched16_f32(const void* x, const void* W, float* y, int N, int K,
                               cudaStream_t stream) {
    if (N <= 0) return;
    constexpr int WARPS_PER_BLOCK = 4;
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    k_gemv_batched_f32<16><<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(
        (const bf16*)x, (const bf16*)W, y, N, K);
}

// ---- all-layer accepted-prefix GDN commit -------------------------------------------------
// Same expressions, same reduction order and the same intrinsics as the single-layer commits in
// fused/batched_prefill.cu, so each layer's result is bit-identical; only the layer loop moves
// from the host stream to a grid dimension.
namespace {

__device__ __forceinline__ float gc_sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float gc_softplus(float x) { return x > 20.f ? x : __logf(1.f + __expf(x)); }
__device__ __forceinline__ float gc_wsum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__global__ void k_gdn_conv_commit_layers(const bf16* __restrict__ qkv_base, size_t qkv_stride,
                                         bf16* __restrict__ live_base, size_t live_stride,
                                         const int* __restrict__ layer_ids,
                                         int n_tokens, int qkv_dim, int conv_kernel) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= qkv_dim) return;
    const size_t L = (size_t)layer_ids[blockIdx.y];
    const bf16* qkv = qkv_base + qkv_stride * L;
    bf16* live_state = live_base + live_stride * L;
    const int history = conv_kernel - 1;
    for (int c = 0; c < history; c++) {
        const int src_tok = n_tokens - history + c;
        live_state[(size_t)c * qkv_dim + d] = src_tok >= 0
            ? qkv[(size_t)src_tok * qkv_dim + d]
            : live_state[(size_t)(c + n_tokens) * qkv_dim + d];
    }
}

template <int COLS, int HEAD_DIM>
__global__ void k_gdn_scan_commit_layers(
    const bf16* __restrict__ k_base, size_t k_stride,
    const bf16* __restrict__ v_base, size_t v_stride,
    const bf16* __restrict__ alpha_base, size_t ab_stride,
    const bf16* __restrict__ beta_base,
    const GdnCommitLayer* __restrict__ layers,
    float* __restrict__ live_base, size_t live_stride,
    int n_tokens, int q_heads, int v_heads) {
    constexpr int NROW = HEAD_DIM / 32;
    const int li = blockIdx.z;
    const int vh = blockIdx.x;
    const int j = blockIdx.y * COLS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (vh >= v_heads || j >= HEAD_DIM) return;
    const size_t L = (size_t)layers[li].layer;
    const bf16* k = k_base + k_stride * L;
    const bf16* v = v_base + v_stride * L;
    const bf16* alpha = alpha_base + ab_stride * L;
    const bf16* beta = beta_base + ab_stride * L;
    const bf16* dt = reinterpret_cast<const bf16*>(layers[li].dt);
    const bf16* a = reinterpret_cast<const bf16*>(layers[li].a);
    float* live_state = live_base + live_stride * L;

    const int qh = vh % q_heads;
    const int q_dim = q_heads * HEAD_DIM;
    const int v_dim = v_heads * HEAD_DIM;
    const size_t col_off = ((size_t)vh * HEAD_DIM + j) * HEAD_DIM;
    const float a_h = b2f(a[vh]);
    const float dt_h = b2f(dt[vh]);

    float sloc[NROW];
    #pragma unroll
    for (int r = 0; r < NROW; r++) sloc[r] = live_state[col_off + lane + r * 32];
    for (int t = 0; t < n_tokens; t++) {
        const float bb = gc_sigmoid(b2f(beta[(size_t)t * v_heads + vh]));
        const float g = __expf(gc_softplus(b2f(alpha[(size_t)t * v_heads + vh]) + dt_h) * a_h);
        const bf16* kp = k + (size_t)t * q_dim + qh * HEAD_DIM;
        const bf16* vp = v + (size_t)t * v_dim + vh * HEAD_DIM;
        float part_sk = 0.f;
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int i = lane + r * 32;
            part_sk += sloc[r] * b2f(kp[i]);
        }
        const float sk = g * gc_wsum(part_sk);
        const float delta = (b2f(vp[j]) - sk) * bb;
        #pragma unroll
        for (int r = 0; r < NROW; r++) {
            const int i = lane + r * 32;
            sloc[r] = sloc[r] * g + b2f(kp[i]) * delta;
        }
    }
    #pragma unroll
    for (int r = 0; r < NROW; r++) live_state[col_off + lane + r * 32] = sloc[r];
}

} // namespace

void launch_gdn_conv_commit_layers(const void* qkv_base, size_t qkv_layer_stride,
                                   void* live_base, size_t live_layer_stride,
                                   const int* layer_ids, int n_layers, int n_tokens,
                                   int q_heads, int v_heads, int head_dim, int conv_kernel,
                                   cudaStream_t stream) {
    if (n_tokens <= 0 || head_dim <= 0 || conv_kernel < 2 || conv_kernel > 8 || n_layers <= 0) return;
    const int qkv_dim = 2 * q_heads * head_dim + v_heads * head_dim;
    constexpr int BLOCK = 256;
    dim3 grid((qkv_dim + BLOCK - 1) / BLOCK, n_layers);
    k_gdn_conv_commit_layers<<<grid, BLOCK, 0, stream>>>(
        (const bf16*)qkv_base, qkv_layer_stride, (bf16*)live_base, live_layer_stride,
        layer_ids, n_tokens, qkv_dim, conv_kernel);
}

void launch_gdn_scan_commit_layers(const void* k_base, size_t k_layer_stride,
                                   const void* v_base, size_t v_layer_stride,
                                   const void* alpha_base, size_t ab_layer_stride,
                                   const void* beta_base, const GdnCommitLayer* layers,
                                   float* live_base, size_t live_layer_stride,
                                   int n_layers, int n_tokens, int q_heads, int v_heads,
                                   int head_dim, cudaStream_t stream) {
    if (n_tokens <= 0 || head_dim != 128 || n_layers <= 0) return;
    constexpr int COLS = 4;
    dim3 grid(v_heads, (head_dim + COLS - 1) / COLS, n_layers);
    k_gdn_scan_commit_layers<COLS, 128><<<grid, COLS * 32, 0, stream>>>(
        (const bf16*)k_base, k_layer_stride, (const bf16*)v_base, v_layer_stride,
        (const bf16*)alpha_base, ab_layer_stride, (const bf16*)beta_base, layers,
        live_base, live_layer_stride, n_tokens, q_heads, v_heads);
}

} // namespace dflash_kernels
} // namespace sparkinfer
