// DFlash draft kernels: small-seq GQA attention, RoPE, SwiGLU, RMSNorm.
#include "sparkinfer/models/dflash_kernels.h"
#include "sparkinfer/kernels/gemm.h"
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
                       int q_pos0, int k_pos0, int window, float scale) {
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
        if (window > 0 && (q_pos - k_pos) >= window) continue;
        const bf16* kv = k + ((size_t)t * n_kv + kv_h) * d;
        float dot = 0.f;
        for (int i = threadIdx.x; i < d; i += blockDim.x)
            dot += q_s[i] * b2f(kv[i]);
        red[threadIdx.x] = dot;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
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
        const bf16* wv = (const bf16*)&wrow4[k8];
#pragma unroll
        for (int b = 0; b < BATCH; b++) {
            const bf16* xv = (const bf16*)&(((const uint4*)(x + (size_t)b * K))[k8]);
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
        const bf16* wv = (const bf16*)&wrow4[k8];
#pragma unroll
        for (int b = 0; b < BATCH; b++) {
            const bf16* xv = (const bf16*)&(((const uint4*)(x + (size_t)b * K))[k8]);
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
                     int q_pos0, int k_pos0, int window, float scale,
                     cudaStream_t stream) {
    if (q_len <= 0 || kv_len <= 0) return;
    dim3 grid(q_len, n_q);
    int smem = (2 * d + 128) * (int)sizeof(float);
    k_attn<<<grid, 128, smem, stream>>>((const bf16*)q, (const bf16*)k, (const bf16*)v,
                                        (bf16*)out, q_len, kv_len, n_q, n_kv, d,
                                        q_pos0, k_pos0, window, scale);
}

// Runtime batch (2..16): same weight-reuse win as k_gemv_batched<16> for ctx projections
// where ctx_len is variable (prompt prefill, accepted-prefix handoff).
__global__ void k_gemv_batched_dyn(const bf16* __restrict__ x, const bf16* __restrict__ W,
                                   bf16* __restrict__ y, int batch, int N, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int n = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    if (n >= N || batch <= 0) return;
    const int lane = threadIdx.x & 31;
    float acc[16];
    for (int b = 0; b < batch; b++) acc[b] = 0.f;
    const uint4* wrow4 = (const uint4*)(W + (size_t)n * K);
    const int K8 = K / 8;
    for (int k8 = lane; k8 < K8; k8 += 32) {
        const bf16* wv = (const bf16*)&wrow4[k8];
        for (int b = 0; b < batch; b++) {
            const bf16* xv = (const bf16*)&(((const uint4*)(x + (size_t)b * K))[k8]);
            for (int j = 0; j < 8; j++) acc[b] += b2f(wv[j]) * b2f(xv[j]);
        }
    }
    for (int b = 0; b < batch; b++) {
        for (int off = 16; off > 0; off >>= 1)
            acc[b] += __shfl_down_sync(0xffffffffu, acc[b], off);
    }
    if (lane == 0) {
        for (int b = 0; b < batch; b++) y[(size_t)b * N + n] = f2b(acc[b]);
    }
}

void launch_gemv_batched(const void* x, const void* W, void* y, int batch, int N, int K,
                         cudaStream_t stream) {
    if (batch <= 0 || N <= 0) return;
    if (batch == 1) {
        kernels::launch_gemv(x, W, y, N, K, stream);
        return;
    }
    if (batch > 16) {
        const bf16* xp = (const bf16*)x;
        bf16* yp = (bf16*)y;
        for (int t0 = 0; t0 < batch; t0 += 16) {
            const int b = (batch - t0 < 16) ? (batch - t0) : 16;
            launch_gemv_batched(xp + (size_t)t0 * K, W, yp + (size_t)t0 * N, b, N, K, stream);
        }
        return;
    }
    if (batch == 16) {
        launch_gemv_batched16(x, W, y, N, K, stream);
        return;
    }
    constexpr int WARPS_PER_BLOCK = 4;
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    k_gemv_batched_dyn<<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(
        (const bf16*)x, (const bf16*)W, (bf16*)y, batch, N, K);
}

void launch_gemv_batched16(const void* x, const void* W, void* y, int N, int K,
                           cudaStream_t stream) {
    if (N <= 0) return;
    constexpr int WARPS_PER_BLOCK = 4;
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    k_gemv_batched<16><<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(
        (const bf16*)x, (const bf16*)W, (bf16*)y, N, K);
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
