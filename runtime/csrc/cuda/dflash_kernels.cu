// DFlash draft kernels: small-seq GQA attention, RoPE, SwiGLU, RMSNorm.
#include "sparkinfer/models/dflash_kernels.h"
#include <cuda_bf16.h>
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
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = b2f(xr[i]);
        ss += v * v;
    }
    __shared__ float buf[256];
    buf[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(buf[0] / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        or_[i] = f2b(b2f(xr[i]) * inv * b2f(w[i]));
}

__global__ void k_rms_heads(bf16* x, const bf16* w, int seq, int n_heads, int d, float eps) {
    int idx = blockIdx.x; // seq * n_heads
    if (idx >= seq * n_heads) return;
    bf16* h = x + (size_t)idx * d;
    float ss = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = b2f(h[i]);
        ss += v * v;
    }
    __shared__ float buf[128];
    buf[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(buf[0] / (float)d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        h[i] = f2b(b2f(h[i]) * inv * b2f(w[i]));
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

// Route several projections that share x through one grid. Each warp still owns exactly one
// output row and executes the same accumulation loop as k_gemv_batched; only launch boundaries
// change, so Q/K/V and gate/up keep bit-identical arithmetic.
template <int BATCH>
__global__ void k_gemv_batched_fused3(
        const bf16* __restrict__ x,
        const bf16* __restrict__ W0, const bf16* __restrict__ W1,
        const bf16* __restrict__ W2,
        bf16* __restrict__ y0, bf16* __restrict__ y1, bf16* __restrict__ y2,
        int N0, int N1, int N2, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int global_n = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    const int total = N0 + N1 + N2;
    if (global_n >= total) return;
    const bf16* W;
    bf16* y;
    int N, n;
    if (global_n < N0) {
        W = W0; y = y0; N = N0; n = global_n;
    } else if (global_n < N0 + N1) {
        W = W1; y = y1; N = N1; n = global_n - N0;
    } else {
        W = W2; y = y2; N = N2; n = global_n - N0 - N1;
    }
    const int lane = threadIdx.x & 31;
    float acc[BATCH];
#pragma unroll
    for (int b = 0; b < BATCH; b++) acc[b] = 0.f;
    const uint4* wrow4 = reinterpret_cast<const uint4*>(W + (size_t)n * K);
    const int K8 = K / 8;
    for (int k8 = lane; k8 < K8; k8 += 32) {
        const uint4 wp = wrow4[k8];
        const bf16* wv = reinterpret_cast<const bf16*>(&wp);
#pragma unroll
        for (int b = 0; b < BATCH; b++) {
            const uint4 xp = reinterpret_cast<const uint4*>(x + (size_t)b * K)[k8];
            const bf16* xv = reinterpret_cast<const bf16*>(&xp);
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

__global__ void k_gemv_rows_exact_s8_fused2(
        const bf16* __restrict__ x,
        const bf16* __restrict__ W0, const bf16* __restrict__ W1,
        bf16* __restrict__ y0, bf16* __restrict__ y1,
        int rows, int N0, int N1, int K) {
    const int global_n = blockIdx.x;
    const int batch = blockIdx.y;
    if (global_n >= N0 + N1 || batch >= rows) return;
    const bool second = global_n >= N0;
    const int n = second ? global_n - N0 : global_n;
    const int N = second ? N1 : N0;
    const bf16* W = second ? W1 : W0;
    bf16* y = second ? y1 : y0;
    const int split = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
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

} // namespace

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

void launch_rms_heads(void* x, const void* w, int seq, int n_heads, int d, float eps,
                      cudaStream_t stream) {
    int n = seq * n_heads;
    if (n <= 0) return;
    k_rms_heads<<<n, 128, 0, stream>>>((bf16*)x, (const bf16*)w, seq, n_heads, d, eps);
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
                           cudaStream_t stream) {
    if (N <= 0) return;
    constexpr int WARPS_PER_BLOCK = 4;
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    k_gemv_batched<16><<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(
        (const bf16*)x, (const bf16*)W, (bf16*)y, N, K);
}

void launch_gemv_batched16_fused3(const void* x,
                                  const void* W0, const void* W1, const void* W2,
                                  void* y0, void* y1, void* y2,
                                  int N0, int N1, int N2, int K, cudaStream_t stream) {
    const int total = N0 + N1 + N2;
    if (total <= 0) return;
    constexpr int WARPS_PER_BLOCK = 4;
    dim3 grid((total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    k_gemv_batched_fused3<16><<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(
        (const bf16*)x, (const bf16*)W0, (const bf16*)W1, (const bf16*)W2,
        (bf16*)y0, (bf16*)y1, (bf16*)y2, N0, N1, N2, K);
}

void launch_gemv_batched16_fused2(const void* x,
                                  const void* W0, const void* W1,
                                  void* y0, void* y1,
                                  int N0, int N1, int K, cudaStream_t stream) {
    launch_gemv_batched16_fused3(x, W0, W1, nullptr, y0, y1, nullptr,
                                 N0, N1, 0, K, stream);
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
    dim3 grid(N0 + N1, rows);
    k_gemv_rows_exact_s8_fused2<<<grid, 8 * 32, 0, stream>>>(
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

} // namespace dflash_kernels
} // namespace sparkinfer
