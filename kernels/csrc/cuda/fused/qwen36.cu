// Qwen3.5/Qwen3.6 hybrid-layer helpers.
//
// These kernels implement the single-token decode path for the Gated DeltaNet
// recurrent layers used by Qwen3.6-35B-A3B. They favor clear, graph-capturable
// device-side state updates over aggressive specialization.

#include <cuda_bf16.h>
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

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/fused.h"

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

void launch_qwen36_conv_split_l2(const void* qkv_bf16, const void* conv_w_bf16,
                                 void* conv_state_bf16, void* q_bf16, void* k_bf16,
                                 void* v_bf16, int q_heads, int v_heads, int head_dim,
                                 int conv_kernel, float eps, cudaStream_t stream) {
    const int q_dim = q_heads * head_dim;
    const int v_dim = v_heads * head_dim;
    const int qkv_dim = 2 * q_dim + v_dim;
    conv_split_kernel<<<(qkv_dim + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv_bf16),
        reinterpret_cast<const __nv_bfloat16*>(conv_w_bf16),
        reinterpret_cast<__nv_bfloat16*>(conv_state_bf16),
        reinterpret_cast<__nv_bfloat16*>(q_bf16),
        reinterpret_cast<__nv_bfloat16*>(k_bf16),
        reinterpret_cast<__nv_bfloat16*>(v_bf16),
        q_dim, v_dim, qkv_dim, conv_kernel);
    l2_norm_heads_kernel<<<q_heads, head_dim, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q_bf16), q_heads, head_dim, eps);
    l2_norm_heads_kernel<<<q_heads, head_dim, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(k_bf16), q_heads, head_dim, eps);
}

void launch_qwen36_gdn_ar(const void* q_bf16, const void* k_bf16, const void* v_bf16,
                          const void* alpha_bf16, const void* beta_bf16,
                          const void* dt_bf16, const void* a_bf16,
                          float* state_f32, void* out_bf16,
                          int q_heads, int v_heads, int head_dim, cudaStream_t stream) {
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
    gated_norm_kernel<<<v_heads, head_dim, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<const __nv_bfloat16*>(z_bf16),
        reinterpret_cast<const __nv_bfloat16*>(weight_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        v_heads, head_dim, eps);
}
#endif

} // namespace kernels
} // namespace sparkinfer
