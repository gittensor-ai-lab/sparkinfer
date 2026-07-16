// Flash prefill — full attention for prompt processing.
//   q/k/v: [batch, seqlen, num_heads/num_kv_heads, head_dim] (contiguous, bf16)
// One warp computes one (batch, q-head, query-position) with online softmax over
// the KV positions (causal-masked when requested). Coalesced lane+e*32 layout.
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

__device__ __forceinline__ float pf_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float pf_warp_sum(float v) {
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

template <int HEAD_DIM>
__global__ void flash_prefill_kernel(
    const __nv_bfloat16* __restrict__ q,   // [batch, seqlen_q,  num_heads,    HEAD_DIM]
    const __nv_bfloat16* __restrict__ k,   // [batch, seqlen_kv, num_kv_heads, HEAD_DIM]
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,       // [batch, seqlen_q,  num_heads,    HEAD_DIM]
    const float scale,
    const int seqlen_q, const int seqlen_kv,
    const int num_heads, const int num_kv_heads,
    const int causal
) {
    constexpr int ELEMS = HEAD_DIM / 32;
    const int qpos = blockIdx.x;
    const int head = blockIdx.y;
    const int b    = blockIdx.z;
    const int lane = threadIdx.x;
    const int kv_head = head / (num_heads / num_kv_heads);

    const size_t q_off = (((size_t)b * seqlen_q + qpos) * num_heads + head) * HEAD_DIM;
    float q_reg[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) q_reg[e] = pf_to_f(q[q_off + lane + e * 32]);

    float m = -1e30f, l = 0.f, acc[ELEMS];
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) acc[e] = 0.f;

    // Causal: query position qpos maps to absolute KV position qpos + (seqlen_kv - seqlen_q)
    // and may attend through it inclusive. Reduces to qpos+1 when seqlen_q == seqlen_kv.
    const int kv_end = causal ? min(seqlen_kv, qpos + 1 + (seqlen_kv - seqlen_q)) : seqlen_kv;
    for (int t = 0; t < kv_end; t++) {
        const size_t kv_off = (((size_t)b * seqlen_kv + t) * num_kv_heads + kv_head) * HEAD_DIM;
        float partial = 0.f;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) partial += q_reg[e] * pf_to_f(k[kv_off + lane + e * 32]);
        const float score = pf_warp_sum(partial) * scale;

        const float m_new = fmaxf(m, score);
        const float corr  = __expf(m - m_new);
        const float p     = __expf(score - m_new);
        l = l * corr + p;
        #pragma unroll
        for (int e = 0; e < ELEMS; e++) acc[e] = acc[e] * corr + p * pf_to_f(v[kv_off + lane + e * 32]);
        m = m_new;
    }

    const float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
    #pragma unroll
    for (int e = 0; e < ELEMS; e++) out[q_off + lane + e * 32] = __float2bfloat16(acc[e] * inv_l);
}

#ifndef _MSC_VER
template __global__ void flash_prefill_kernel<64> (const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float, int, int, int, int, int);
#endif
#ifndef _MSC_VER
template __global__ void flash_prefill_kernel<128>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float, int, int, int, int, int);
#endif
#ifndef _MSC_VER
template __global__ void flash_prefill_kernel<256>(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, float, int, int, int, int, int);
#endif
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
template <int HEAD_DIM>
static void pf_dispatch(const void* q, const void* k, const void* v, void* out,
                        int batch, int seqlen_q, int seqlen_kv, int num_heads,
                        int num_kv_heads, float scale, bool causal, cudaStream_t stream) {
    dim3 grid(seqlen_q, num_heads, batch);
    flash_prefill_kernel<HEAD_DIM><<<grid, 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<__nv_bfloat16*>(out),
        scale, seqlen_q, seqlen_kv, num_heads, num_kv_heads, causal ? 1 : 0);
}

void launch_flash_prefill(
    const void* q, const void* k, const void* v, void* out,
    int batch, int seqlen_q, int seqlen_kv,
    int num_heads, int num_kv_heads, int head_dim,
    float scale, bool causal, cudaStream_t stream
) {
    switch (head_dim) {
        case 64:  pf_dispatch<64> (q,k,v,out,batch,seqlen_q,seqlen_kv,num_heads,num_kv_heads,scale,causal,stream); break;
        case 128: pf_dispatch<128>(q,k,v,out,batch,seqlen_q,seqlen_kv,num_heads,num_kv_heads,scale,causal,stream); break;
        case 256: pf_dispatch<256>(q,k,v,out,batch,seqlen_q,seqlen_kv,num_heads,num_kv_heads,scale,causal,stream); break;
        default: break;
    }
}
#endif

} // namespace kernels
} // namespace sparkinfer
