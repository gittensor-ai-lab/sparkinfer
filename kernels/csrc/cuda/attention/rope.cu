// Rotary position embedding (RoPE), HF "rotate-half" convention (GPT-NeoX style)
// as used by Qwen/Llama. Applied to Q and K after projection, before attention.
//
// For a head vector x[head_dim] at position p, with half = head_dim/2:
//   freq_i  = theta^(-2i/head_dim),  angle = p * freq_i
//   out[i]      = x[i]*cos - x[i+half]*sin
//   out[i+half] = x[i+half]*cos + x[i]*sin     for i in [0, half)
//
// Portable CUDA — runs on sm_89 .. sm_120 (RTX 5090).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

// grid = (n_tokens, n_heads); blockDim = head_dim/2 threads (one per rotated pair).
__global__ void rope_kernel(
    __nv_bfloat16* __restrict__ x,        // [n_tokens, n_heads, head_dim]
    const int* __restrict__ positions,    // [n_tokens]
    int n_heads, int head_dim, float theta
) {
    const int tok  = blockIdx.x;
    const int head = blockIdx.y;
    const int i    = threadIdx.x;
    const int half = head_dim / 2;
    if (i >= half) return;

    const float p    = (float)positions[tok];
    const float freq = __powf(theta, -2.f * (float)i / (float)head_dim);
    const float ang  = p * freq;
    const float c = __cosf(ang), s = __sinf(ang);

    const size_t base = ((size_t)tok * n_heads + head) * head_dim;
    const float x0 = __bfloat162float(x[base + i]);
    const float x1 = __bfloat162float(x[base + i + half]);
    x[base + i]        = __float2bfloat16(x0 * c - x1 * s);
    x[base + i + half] = __float2bfloat16(x1 * c + x0 * s);
}

// Fused Q+K rope: ONE kernel over all (n_q_heads + n_kv_heads) heads with a flat
// 256-thread layout — 1 graph node instead of 2, and better occupancy than the
// head_dim/2-thread blocks. Mirrors llama's single rope_neox launch.
__global__ void rope_qk_kernel(
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k,
    const int* __restrict__ positions, int n_q_heads, int n_kv_heads, int head_dim, float theta
) {
    const int tok  = blockIdx.y;
    const int half = head_dim >> 1;
    const int total = (n_q_heads + n_kv_heads) * half;     // rotated pairs across Q|K
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    const int hh = gid / half, i = gid - hh * half;
    __nv_bfloat16* x; int head, nh;
    if (hh < n_q_heads) { x = q; head = hh;             nh = n_q_heads; }
    else                { x = k; head = hh - n_q_heads; nh = n_kv_heads; }
    const float p = (float)positions[tok];
    const float freq = __powf(theta, -2.f * (float)i / (float)head_dim);
    const float ang = p * freq;
    const float c = __cosf(ang), s = __sinf(ang);
    const size_t base = ((size_t)(tok * nh + head)) * head_dim;
    const float x0 = __bfloat162float(x[base + i]);
    const float x1 = __bfloat162float(x[base + i + half]);
    x[base + i]        = __float2bfloat16(x0 * c - x1 * s);
    x[base + i + half] = __float2bfloat16(x1 * c + x0 * s);
}

// Fused RoPE + KV-append: ropes Q in place, ropes K and writes it STRAIGHT into the paged
// KV cache, and copies V into the cache — one kernel replacing rope_qk + kv_append (one graph
// node instead of two, and no s.k round-trip). The roped Q/K are bit-identical to rope_qk and
// the cached V is identical to kv_append. On the decode path positions == write_pos (the
// token's absolute slot), so one pointer drives both the rope angle and the cache slot.
__global__ void rope_kv_append_kernel(
    __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ k_pool, __nv_bfloat16* __restrict__ v_pool,
    const int* __restrict__ block_table, const int* __restrict__ positions,
    int n_q_heads, int n_kv_heads, int head_dim, float theta,
    int block_size, int max_blocks_per_seq
) {
    const int tok  = blockIdx.y;
    const int half = head_dim >> 1;
    const int nq = n_q_heads  * half;        // Q rotated pairs
    const int nk = n_kv_heads * half;        // K rotated pairs
    const int nv = n_kv_heads * head_dim;    // V elements (no rope)
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nq + nk + nv) return;

    const int pos    = positions[tok];
    const int blk    = pos / block_size;
    const int within = pos % block_size;
    const int phys   = block_table[tok * max_blocks_per_seq + blk];
    const size_t ctok = (size_t)(phys * block_size + within);   // cache token slot

    if (gid < nq) {                          // Q: rope in place
        const int hh = gid / half, i = gid - hh * half;
        const float freq = __powf(theta, -2.f * (float)i / (float)head_dim);
        const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
        const size_t base = ((size_t)(tok * n_q_heads + hh)) * head_dim;
        const float x0 = __bfloat162float(q[base + i]), x1 = __bfloat162float(q[base + i + half]);
        q[base + i]        = __float2bfloat16(x0 * c - x1 * s);
        q[base + i + half] = __float2bfloat16(x1 * c + x0 * s);
    } else if (gid < nq + nk) {              // K: rope, write straight to the cache
        const int g = gid - nq, hh = g / half, i = g - hh * half;
        const float freq = __powf(theta, -2.f * (float)i / (float)head_dim);
        const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
        const size_t base = ((size_t)(tok * n_kv_heads + hh)) * head_dim;
        const size_t dst  = (ctok * n_kv_heads + hh) * head_dim;
        const float x0 = __bfloat162float(k[base + i]), x1 = __bfloat162float(k[base + i + half]);
        k_pool[dst + i]        = __float2bfloat16(x0 * c - x1 * s);
        k_pool[dst + i + half] = __float2bfloat16(x1 * c + x0 * s);
    } else {                                 // V: copy to the cache (no rope)
        const int g = gid - nq - nk, hh = g / head_dim, d = g - hh * head_dim;
        const size_t base = ((size_t)(tok * n_kv_heads + hh)) * head_dim;
        const size_t dst  = (ctok * n_kv_heads + hh) * head_dim;
        v_pool[dst + d] = v[base + d];
    }
}

// Fused per-head QK-norm + RoPE + KV-append: ONE kernel replacing rmsnorm_qk + rope_kv_append.
// grid = (n_q_heads + 2*n_kv_heads, n_tokens); blockDim = head_dim (one block per head).
//   blocks [0, n_q_heads)                 : Q head -> RMSNorm(q_w) + RoPE in place
//   blocks [n_q_heads, +n_kv_heads)       : K head -> RMSNorm(k_w) + RoPE + write k_pool
//   blocks [.., +n_kv_heads)              : V head -> copy to v_pool (no norm/rope)
// The normed head is staged in shared so RoPE can pair element i with i+half. The roped/normed
// q,k are value-identical to rmsnorm_qk followed by rope_kv_append (same per-head RMS, same
// bf16 rounding, same rope angle); the cached V is identical to kv_append. positions == write_pos.
// int8_kv: k_pool/v_pool hold signed int8 and k_scale/v_scale one __half per (token slot, kv_head)
// head vector (per-token max-abs quant). int8_kv==0 is byte-identical bf16.
template <bool INT8>
__global__ void qknorm_rope_kv_kernel(
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_w, const __nv_bfloat16* __restrict__ k_w,
    void* __restrict__ k_pool, void* __restrict__ v_pool,
    __half* __restrict__ k_scale, __half* __restrict__ v_scale,
    const int* __restrict__ block_table, const int* __restrict__ positions,
    int n_q_heads, int n_kv_heads, int head_dim, float theta,
    int block_size, int max_blocks_per_seq, float eps
) {
    const int tok  = blockIdx.y;
    const int b    = blockIdx.x;
    const int t    = threadIdx.x;                 // 0 .. head_dim-1
    const int half = head_dim >> 1;
    const int pos  = positions[tok];
    const int blk = pos / block_size, within = pos % block_size;
    const size_t ctok = (size_t)((size_t)block_table[tok * max_blocks_per_seq + blk] * block_size + within);
    __shared__ float s_red[32];                    // warp-partials for block reductions

    const bool is_v = (b >= n_q_heads + n_kv_heads);
    if (is_v) {                                    // V: copy head straight to the cache (no norm/rope)
        const int hh = b - n_q_heads - n_kv_heads;
        const size_t base = ((size_t)(tok * n_kv_heads + hh)) * head_dim;
        const size_t dst  = (ctok * n_kv_heads + hh) * head_dim;
        if constexpr (!INT8) {                         // bf16: straight copy — byte-identical to main
            __nv_bfloat16* __restrict__ vp = reinterpret_cast<__nv_bfloat16*>(v_pool);
            vp[dst + t] = v[base + t];
        } else {
            const float val = __bfloat162float(v[base + t]);   // per-token max-abs int8 quant over head_dim
            float amax = fabsf(val);
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m));
            if ((t & 31) == 0) s_red[t >> 5] = amax;
            __syncthreads();
            if (t < 32) { float a = (t < (head_dim + 31) / 32) ? s_red[t] : 0.f;
                #pragma unroll
                for (int m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, m));
                if (t == 0) s_red[0] = a; }
            __syncthreads();
            const float d = s_red[0] / 127.0f;
            reinterpret_cast<signed char*>(v_pool)[dst + t] = (signed char)((s_red[0] == 0.f) ? 0 : (int)roundf(val / d));
            if (t == 0) v_scale[ctok * n_kv_heads + hh] = __float2half(d);
        }
        return;
    }

    const bool is_q = (b < n_q_heads);
    __nv_bfloat16* x        = is_q ? q : k;
    const __nv_bfloat16* w  = is_q ? q_w : k_w;
    const int head          = is_q ? b : (b - n_q_heads);
    const int nh            = is_q ? n_q_heads : n_kv_heads;
    const size_t base       = ((size_t)(tok * nh + head)) * head_dim;

    extern __shared__ float s_h[];                 // normed head (head_dim floats)
    __shared__ float s_warp[32];
    const float xv = __bfloat162float(x[base + t]);
    float ss = xv * xv;                            // per-head RMS over head_dim
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
    if ((t & 31) == 0) s_warp[t >> 5] = ss;
    __syncthreads();
    if (t < 32) {
        float vv = (t < (head_dim + 31) / 32) ? s_warp[t] : 0.f;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) vv += __shfl_xor_sync(0xffffffff, vv, m);
        if (t == 0) s_warp[0] = rsqrtf(vv / head_dim + eps);
    }
    __syncthreads();
    // Normed value, bf16-rounded exactly as rmsnorm_qk writes it (so RoPE sees identical inputs).
    s_h[t] = __bfloat162float(__float2bfloat16(xv * s_warp[0] * __bfloat162float(w[t])));
    __syncthreads();

    // RoPE (HF rotate-half): pair (i, i+half). Threads [0,half) own a pair and write both halves.
    float o0 = 0.f, o1 = 0.f;
    if (t < half) {
        const float freq = __powf(theta, -2.f * (float)t / (float)head_dim);
        const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
        const float x0 = s_h[t], x1 = s_h[t + half];
        o0 = __bfloat162float(__float2bfloat16(x0 * c - x1 * s));   // bf16-rounded (matches bf16 path)
        o1 = __bfloat162float(__float2bfloat16(x1 * c + x0 * s));
        if (is_q) { q[base + t] = __float2bfloat16(o0); q[base + t + half] = __float2bfloat16(o1); }
        else if constexpr (!INT8) {                    // bf16 K write — byte-identical to main
            const size_t dst = (ctok * n_kv_heads + head) * head_dim;
            __nv_bfloat16* __restrict__ kp = reinterpret_cast<__nv_bfloat16*>(k_pool);
            kp[dst + t] = __float2bfloat16(o0);
            kp[dst + t + half] = __float2bfloat16(o1);
        }
    }
    if constexpr (INT8) if (!is_q) {               // K int8: per-token max-abs over all head_dim
        float amax = fmaxf(fabsf(o0), fabsf(o1));  // thread t<half holds dims t and t+half
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m));
        if ((t & 31) == 0 && t < half) s_red[t >> 5] = amax;
        __syncthreads();
        if (t == 0) { float a = 0.f; const int nw = half / 32;
            for (int i = 0; i < nw; i++) a = fmaxf(a, s_red[i]); s_red[0] = a; }
        __syncthreads();
        const float d = s_red[0] / 127.0f;
        if (t < half) {
            const size_t dst = (ctok * n_kv_heads + head) * head_dim;
            signed char* kp = reinterpret_cast<signed char*>(k_pool);
            kp[dst + t]        = (signed char)((s_red[0] == 0.f) ? 0 : (int)roundf(o0 / d));
            kp[dst + t + half] = (signed char)((s_red[0] == 0.f) ? 0 : (int)roundf(o1 / d));
        }
        if (t == 0) k_scale[ctok * n_kv_heads + head] = __float2half(d);
    }
}
#ifndef _MSC_VER
template __global__ void qknorm_rope_kv_kernel<false>(
    __nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    void*, void*, __half*, __half*, const int*, const int*, int, int, int, float, int, int, float);
#endif
#ifndef _MSC_VER
template __global__ void qknorm_rope_kv_kernel<true>(
    __nv_bfloat16*, __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    void*, void*, __half*, __half*, const int*, const int*, int, int, int, float, int, int, float);
#endif
// Fused QK-norm + partial-RoPE + KV-append for Qwen3.6 (rope_dim < head_dim). One kernel per head
// replaces launch_rmsnorm_qk + launch_rope_kv_append_partial on the 10 full-attn layers.
__global__ void qknorm_rope_kv_partial_kernel(
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_w, const __nv_bfloat16* __restrict__ k_w,
    __nv_bfloat16* __restrict__ k_pool, __nv_bfloat16* __restrict__ v_pool,
    const int* __restrict__ block_table, const int* __restrict__ positions,
    int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim, float theta,
    int block_size, int max_blocks_per_seq, float eps
) {
    const int tok  = blockIdx.y;
    const int b    = blockIdx.x;
    const int t    = threadIdx.x;
    const int rhalf = rotary_dim >> 1;
    const int pos  = positions[tok];
    const int blk = pos / block_size, within = pos % block_size;
    const size_t ctok = (size_t)((size_t)block_table[tok * max_blocks_per_seq + blk] * block_size + within);

    const bool is_v = (b >= n_q_heads + n_kv_heads);
    if (is_v) {
        const int hh = b - n_q_heads - n_kv_heads;
        const size_t base = ((size_t)(tok * n_kv_heads + hh)) * head_dim;
        const size_t dst  = (ctok * n_kv_heads + hh) * head_dim;
        if (t < head_dim) v_pool[dst + t] = v[base + t];
        return;
    }

    const bool is_q = (b < n_q_heads);
    __nv_bfloat16* x        = is_q ? q : k;
    const __nv_bfloat16* w  = is_q ? q_w : k_w;
    const int head          = is_q ? b : (b - n_q_heads);
    const int nh            = is_q ? n_q_heads : n_kv_heads;
    const size_t base       = ((size_t)(tok * nh + head)) * head_dim;

    extern __shared__ float s_h[];
    __shared__ float s_warp[32];
    const float xv = __bfloat162float(x[base + t]);
    float ss = xv * xv;
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
    if ((t & 31) == 0) s_warp[t >> 5] = ss;
    __syncthreads();
    if (t < 32) {
        float vv = (t < (head_dim + 31) / 32) ? s_warp[t] : 0.f;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) vv += __shfl_xor_sync(0xffffffff, vv, m);
        if (t == 0) s_warp[0] = rsqrtf(vv / head_dim + eps);
    }
    __syncthreads();
    s_h[t] = __bfloat162float(__float2bfloat16(xv * s_warp[0] * __bfloat162float(w[t])));
    __syncthreads();

    if (t < rhalf) {
        const float freq = __powf(theta, -2.f * (float)t / (float)rotary_dim);
        const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
        const float x0 = s_h[t], x1 = s_h[t + rhalf];
        s_h[t] = __bfloat162float(__float2bfloat16(x0 * c - x1 * s));
        s_h[t + rhalf] = __bfloat162float(__float2bfloat16(x1 * c + x0 * s));
    }
    __syncthreads();

    if (is_q) {
        if (t < head_dim) q[base + t] = __float2bfloat16(s_h[t]);
    } else {
        const size_t dst = (ctok * n_kv_heads + head) * head_dim;
        if (t < head_dim) k_pool[dst + t] = __float2bfloat16(s_h[t]);
    }
}

__global__ void rope_kv_append_partial_kernel(
    __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ k_pool, __nv_bfloat16* __restrict__ v_pool,
    const int* __restrict__ block_table, const int* __restrict__ positions,
    int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim, float theta,
    int block_size, int max_blocks_per_seq
) {
    const int tok  = blockIdx.y;
    const int rhalf = rotary_dim >> 1;
    const int nq = n_q_heads  * rhalf;
    const int nk = n_kv_heads * rhalf;
    const int ktail = n_kv_heads * (head_dim - rotary_dim);
    const int nv = n_kv_heads * head_dim;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nq + nk + ktail + nv) return;

    const int pos    = positions[tok];
    const int blk    = pos / block_size;
    const int within = pos % block_size;
    const int phys   = block_table[tok * max_blocks_per_seq + blk];
    const size_t ctok = (size_t)(phys * block_size + within);

    if (gid < nq) {
        const int hh = gid / rhalf, i = gid - hh * rhalf;
        const float freq = __powf(theta, -2.f * (float)i / (float)rotary_dim);
        const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
        const size_t base = ((size_t)(tok * n_q_heads + hh)) * head_dim;
        const float x0 = __bfloat162float(q[base + i]);
        const float x1 = __bfloat162float(q[base + i + rhalf]);
        q[base + i]         = __float2bfloat16(x0 * c - x1 * s);
        q[base + i + rhalf] = __float2bfloat16(x1 * c + x0 * s);
    } else if (gid < nq + nk) {
        const int g = gid - nq, hh = g / rhalf, i = g - hh * rhalf;
        const float freq = __powf(theta, -2.f * (float)i / (float)rotary_dim);
        const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
        const size_t base = ((size_t)(tok * n_kv_heads + hh)) * head_dim;
        const size_t dst  = (ctok * n_kv_heads + hh) * head_dim;
        const float x0 = __bfloat162float(k[base + i]);
        const float x1 = __bfloat162float(k[base + i + rhalf]);
        k_pool[dst + i]         = __float2bfloat16(x0 * c - x1 * s);
        k_pool[dst + i + rhalf] = __float2bfloat16(x1 * c + x0 * s);
    } else if (gid < nq + nk + ktail) {
        const int g = gid - nq - nk;
        const int hh = g / (head_dim - rotary_dim);
        const int d = rotary_dim + (g - hh * (head_dim - rotary_dim));
        const size_t base = ((size_t)(tok * n_kv_heads + hh)) * head_dim;
        const size_t dst  = (ctok * n_kv_heads + hh) * head_dim;
        k_pool[dst + d] = k[base + d];
    } else {
        const int g = gid - nq - nk - ktail, hh = g / head_dim, d = g - hh * head_dim;
        const size_t base = ((size_t)(tok * n_kv_heads + hh)) * head_dim;
        const size_t dst  = (ctok * n_kv_heads + hh) * head_dim;
        v_pool[dst + d] = v[base + d];
    }
}

// int8-KV variant of the partial-RoPE append (Qwen3.6 hd256 full-attn). The bf16 kernel above is
// element-parallel, which can't compute the per-(token,kv_head) max-abs an int8 scale needs; this one
// is organized one block per (token, head-unit) with blockDim==head_dim so a full head vector reduces
// in-block. Q heads get RoPE (bf16, in-place, no quant); K/V heads are per-head-vector max-abs
// quantized to int8 with one fp16 scale each — the same scheme launch_qknorm_rope_kv_append uses for
// hd128, so the int8 tensor-core flash-decode reads a consistent cache.
__global__ void rope_kv_append_partial_int8_kernel(
    __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    signed char* __restrict__ k_pool, signed char* __restrict__ v_pool,
    __half* __restrict__ k_scale, __half* __restrict__ v_scale,
    const int* __restrict__ block_table, const int* __restrict__ positions,
    int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim, float theta,
    int block_size, int max_blocks_per_seq
) {
    const int tok  = blockIdx.y;
    const int unit = blockIdx.x;          // [0,nq)=Q rope ; [nq,nq+nkv)=K ; [nq+nkv,nq+2nkv)=V
    const int t    = threadIdx.x;         // 0..head_dim-1
    const int rhalf = rotary_dim >> 1;
    const int pos    = positions[tok];
    const int blk    = pos / block_size, within = pos % block_size;
    const int phys   = block_table[tok * max_blocks_per_seq + blk];
    const size_t ctok = (size_t)(phys * block_size + within);

    if (unit < n_q_heads) {               // Q RoPE, bf16 in-place (no quant)
        const size_t base = ((size_t)(tok * n_q_heads + unit)) * head_dim;
        if (t < rhalf) {
            const float freq = __powf(theta, -2.f * (float)t / (float)rotary_dim);
            const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
            const float x0 = __bfloat162float(q[base + t]);
            const float x1 = __bfloat162float(q[base + t + rhalf]);
            q[base + t]         = __float2bfloat16(x0 * c - x1 * s);
            q[base + t + rhalf] = __float2bfloat16(x1 * c + x0 * s);
        }
        return;
    }

    const bool is_k = unit < n_q_heads + n_kv_heads;
    const int  hh   = is_k ? (unit - n_q_heads) : (unit - n_q_heads - n_kv_heads);
    const size_t base = ((size_t)(tok * n_kv_heads + hh)) * head_dim;
    const size_t dst  = (ctok * n_kv_heads + hh) * head_dim;

    // per-dim value: K rotates the first rotary_dim (paired i,i+rhalf) and copies the tail; V is raw.
    float val;
    if (is_k && t < rotary_dim) {
        const int i = (t < rhalf) ? t : (t - rhalf);
        const float freq = __powf(theta, -2.f * (float)i / (float)rotary_dim);
        const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
        const float x0 = __bfloat162float(k[base + i]);
        const float x1 = __bfloat162float(k[base + i + rhalf]);
        val = (t < rhalf) ? (x0 * c - x1 * s) : (x1 * c + x0 * s);
    } else {
        val = __bfloat162float((is_k ? k : v)[base + t]);
    }

    __shared__ float s_red[8];            // head_dim/32 warp partials (<=256 -> <=8)
    float amax = fabsf(val);
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m));
    if ((t & 31) == 0) s_red[t >> 5] = amax;
    __syncthreads();
    if (t == 0) {
        float a = 0.f;
        for (int w = 0; w < (head_dim >> 5); w++) a = fmaxf(a, s_red[w]);
        s_red[0] = a;
    }
    __syncthreads();
    const float d  = s_red[0] / 127.0f;
    const int   qi = (s_red[0] == 0.f) ? 0 : (int)roundf(val / d);
    if (is_k) {
        k_pool[dst + t] = (signed char)qi;
        if (t == 0) k_scale[ctok * n_kv_heads + hh] = __float2half(d);
    } else {
        v_pool[dst + t] = (signed char)qi;
        if (t == 0) v_scale[ctok * n_kv_heads + hh] = __float2half(d);
    }
}

// Fused QK-norm + partial-RoPE + int8 KV-append for Qwen3.6 hd256 full-attn layers.
// Replaces launch_rmsnorm_qk + launch_rope_kv_append_partial_int8 (two graph nodes).
__global__ void qknorm_rope_kv_partial_int8_kernel(
    __nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_w, const __nv_bfloat16* __restrict__ k_w,
    signed char* __restrict__ k_pool, signed char* __restrict__ v_pool,
    __half* __restrict__ k_scale, __half* __restrict__ v_scale,
    const int* __restrict__ block_table, const int* __restrict__ positions,
    int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim, float theta,
    int block_size, int max_blocks_per_seq, float eps
) {
    const int tok  = blockIdx.y;
    const int unit = blockIdx.x;
    const int t    = threadIdx.x;
    const int rhalf = rotary_dim >> 1;
    const int pos    = positions[tok];
    const int blk    = pos / block_size, within = pos % block_size;
    const int phys   = block_table[tok * max_blocks_per_seq + blk];
    const size_t ctok = (size_t)(phys * block_size + within);

    extern __shared__ float s_h[];
    __shared__ float s_warp[32];
    __shared__ float s_red[8];

    if (unit < n_q_heads) {               // Q: RMSNorm + partial RoPE (bf16 in-place)
        const size_t base = ((size_t)(tok * n_q_heads + unit)) * head_dim;
        const float xv = __bfloat162float(q[base + t]);
        float ss = xv * xv;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
        if ((t & 31) == 0) s_warp[t >> 5] = ss;
        __syncthreads();
        if (t < 32) {
            float vv = (t < (head_dim + 31) / 32) ? s_warp[t] : 0.f;
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) vv += __shfl_xor_sync(0xffffffff, vv, m);
            if (t == 0) s_warp[0] = rsqrtf(vv / head_dim + eps);
        }
        __syncthreads();
        s_h[t] = __bfloat162float(__float2bfloat16(xv * s_warp[0] * __bfloat162float(q_w[t])));
        __syncthreads();
        // Rotate in place in s_h, then write back the WHOLE head vector. Writing only the
        // rotated dims (t < rhalf, i.e. 0..rope_dim-1) left dims rope_dim..head_dim-1 holding
        // the raw wq projection output — missing both the RMS scale and the q_norm weight —
        // because s_h above norms all head_dim dims but only rope_dim of them were stored.
        // At hd256/rope_dim=64 that is 192 of 256 dims (75% of Q) un-normalized on every
        // int8-KV decode step. Mirrors qknorm_rope_kv_partial_kernel (the bf16 path), which
        // rotates into s_h and then writes all t < head_dim.
        if (t < rhalf) {
            const float freq = __powf(theta, -2.f * (float)t / (float)rotary_dim);
            const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
            const float x0 = s_h[t], x1 = s_h[t + rhalf];
            s_h[t]         = __bfloat162float(__float2bfloat16(x0 * c - x1 * s));
            s_h[t + rhalf] = __bfloat162float(__float2bfloat16(x1 * c + x0 * s));
        }
        __syncthreads();
        if (t < head_dim) q[base + t] = __float2bfloat16(s_h[t]);
        return;
    }

    const bool is_k = unit < n_q_heads + n_kv_heads;
    const int  hh   = is_k ? (unit - n_q_heads) : (unit - n_q_heads - n_kv_heads);
    const size_t base = ((size_t)(tok * n_kv_heads + hh)) * head_dim;
    const size_t dst  = (ctok * n_kv_heads + hh) * head_dim;

    float val;
    if (is_k) {
        const float xv = __bfloat162float(k[base + t]);
        float ss = xv * xv;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
        if ((t & 31) == 0) s_warp[t >> 5] = ss;
        __syncthreads();
        if (t < 32) {
            float vv = (t < (head_dim + 31) / 32) ? s_warp[t] : 0.f;
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) vv += __shfl_xor_sync(0xffffffff, vv, m);
            if (t == 0) s_warp[0] = rsqrtf(vv / head_dim + eps);
        }
        __syncthreads();
        s_h[t] = __bfloat162float(__float2bfloat16(xv * s_warp[0] * __bfloat162float(k_w[t])));
        __syncthreads();
        if (t < rotary_dim) {
            const int i = (t < rhalf) ? t : (t - rhalf);
            const float freq = __powf(theta, -2.f * (float)i / (float)rotary_dim);
            const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
            const float x0 = s_h[i], x1 = s_h[i + rhalf];
            val = (t < rhalf) ? (x0 * c - x1 * s) : (x1 * c + x0 * s);
        } else {
            val = s_h[t];
        }
    } else {
        val = __bfloat162float(v[base + t]);
    }

    float amax = fabsf(val);
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m));
    if ((t & 31) == 0) s_red[t >> 5] = amax;
    __syncthreads();
    if (t == 0) {
        float a = 0.f;
        for (int w = 0; w < (head_dim >> 5); w++) a = fmaxf(a, s_red[w]);
        s_red[0] = a;
    }
    __syncthreads();
    const float d  = s_red[0] / 127.0f;
    const int   qi = (s_red[0] == 0.f) ? 0 : (int)roundf(val / d);
    if (is_k) {
        k_pool[dst + t] = (signed char)qi;
        if (t == 0) k_scale[ctok * n_kv_heads + hh] = __float2half(d);
    } else {
        v_pool[dst + t] = (signed char)qi;
        if (t == 0) v_scale[ctok * n_kv_heads + hh] = __float2half(d);
    }
}

// Gated Qwen3.6 full-attn: read Q from qraw (2*head_dim interleaved), extract gate,
// RMSNorm + partial RoPE in-place to q, then K/V int8 KV-append (same as int8 variant).
__global__ void qknorm_rope_kv_partial_int8_gated_kernel(
    const __nv_bfloat16* __restrict__ qraw, __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ qgate,
    __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_w, const __nv_bfloat16* __restrict__ k_w,
    signed char* __restrict__ k_pool, signed char* __restrict__ v_pool,
    __half* __restrict__ k_scale, __half* __restrict__ v_scale,
    const int* __restrict__ block_table, const int* __restrict__ positions,
    int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim, float theta,
    int block_size, int max_blocks_per_seq, float eps
) {
    const int tok  = blockIdx.y;
    const int unit = blockIdx.x;
    const int t    = threadIdx.x;
    const int rhalf = rotary_dim >> 1;
    const int pos    = positions[tok];
    const int blk    = pos / block_size, within = pos % block_size;
    const int phys   = block_table[tok * max_blocks_per_seq + blk];
    const size_t ctok = (size_t)(phys * block_size + within);

    extern __shared__ float s_h[];
    __shared__ float s_warp[32];
    __shared__ float s_red[8];

    if (unit < n_q_heads) {
        const size_t qbase = ((size_t)(tok * n_q_heads + unit)) * head_dim;
        const size_t rawbase = qbase * 2;
        if (t < head_dim) qgate[qbase + t] = qraw[rawbase + head_dim + t];
        const float xv = __bfloat162float(qraw[rawbase + t]);
        float ss = xv * xv;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
        if ((t & 31) == 0) s_warp[t >> 5] = ss;
        __syncthreads();
        if (t < 32) {
            float vv = (t < (head_dim + 31) / 32) ? s_warp[t] : 0.f;
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) vv += __shfl_xor_sync(0xffffffff, vv, m);
            if (t == 0) s_warp[0] = rsqrtf(vv / head_dim + eps);
        }
        __syncthreads();
        s_h[t] = __bfloat162float(__float2bfloat16(xv * s_warp[0] * __bfloat162float(q_w[t])));
        __syncthreads();
        // Same defect as qknorm_rope_kv_partial_int8_kernel, and worse here: this variant reads
        // qraw[] and writes a DISTINCT q[] buffer, so storing only t < rhalf left q dims
        // rope_dim..head_dim-1 not merely un-normalized but STALE — never written by this
        // kernel, retaining whatever the q arena last held (the previous layer's Q, in
        // forward_token's launch order). Rotate in s_h, then write the whole head vector.
        if (t < rhalf) {
            const float freq = __powf(theta, -2.f * (float)t / (float)rotary_dim);
            const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
            const float x0 = s_h[t], x1 = s_h[t + rhalf];
            s_h[t]         = __bfloat162float(__float2bfloat16(x0 * c - x1 * s));
            s_h[t + rhalf] = __bfloat162float(__float2bfloat16(x1 * c + x0 * s));
        }
        __syncthreads();
        if (t < head_dim) q[qbase + t] = __float2bfloat16(s_h[t]);
        return;
    }

    const bool is_k = unit < n_q_heads + n_kv_heads;
    const int  hh   = is_k ? (unit - n_q_heads) : (unit - n_q_heads - n_kv_heads);
    const size_t base = ((size_t)(tok * n_kv_heads + hh)) * head_dim;
    const size_t dst  = (ctok * n_kv_heads + hh) * head_dim;

    float val;
    if (is_k) {
        const float xv = __bfloat162float(k[base + t]);
        float ss = xv * xv;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) ss += __shfl_xor_sync(0xffffffff, ss, m);
        if ((t & 31) == 0) s_warp[t >> 5] = ss;
        __syncthreads();
        if (t < 32) {
            float vv = (t < (head_dim + 31) / 32) ? s_warp[t] : 0.f;
            #pragma unroll
            for (int m = 16; m > 0; m >>= 1) vv += __shfl_xor_sync(0xffffffff, vv, m);
            if (t == 0) s_warp[0] = rsqrtf(vv / head_dim + eps);
        }
        __syncthreads();
        s_h[t] = __bfloat162float(__float2bfloat16(xv * s_warp[0] * __bfloat162float(k_w[t])));
        __syncthreads();
        if (t < rotary_dim) {
            const int i = (t < rhalf) ? t : (t - rhalf);
            const float freq = __powf(theta, -2.f * (float)i / (float)rotary_dim);
            const float ang = (float)pos * freq, c = __cosf(ang), s = __sinf(ang);
            const float x0 = s_h[i], x1 = s_h[i + rhalf];
            val = (t < rhalf) ? (x0 * c - x1 * s) : (x1 * c + x0 * s);
        } else {
            val = s_h[t];
        }
    } else {
        val = __bfloat162float(v[base + t]);
    }

    float amax = fabsf(val);
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, m));
    if ((t & 31) == 0) s_red[t >> 5] = amax;
    __syncthreads();
    if (t == 0) {
        float a = 0.f;
        for (int w = 0; w < (head_dim >> 5); w++) a = fmaxf(a, s_red[w]);
        s_red[0] = a;
    }
    __syncthreads();
    const float d  = s_red[0] / 127.0f;
    const int   qi = (s_red[0] == 0.f) ? 0 : (int)roundf(val / d);
    if (is_k) {
        k_pool[dst + t] = (signed char)qi;
        if (t == 0) k_scale[ctok * n_kv_heads + hh] = __float2half(d);
    } else {
        v_pool[dst + t] = (signed char)qi;
        if (t == 0) v_scale[ctok * n_kv_heads + hh] = __float2half(d);
    }
}

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/attention.h"
#include "sparkinfer/kernels/fused.h"
#include <cstdlib>

// Fused QK-norm + RoPE + KV-append (SPARKINFER_ATTNIN, default on). One kernel replaces
// launch_rmsnorm_qk + launch_rope_kv_append, deleting a graph node and the intermediate
// normed-q/k global round-trip. Falls back (caller keeps the two kernels) when disabled.
void launch_qknorm_rope_kv_append(
    void* q, void* k, const void* v, const void* q_w, const void* k_w,
    void* k_pool, void* v_pool, const int* block_table, const int* positions,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim, float theta,
    float eps, int block_size, int max_blocks_per_seq, cudaStream_t stream,
    void* k_scale, void* v_scale, int int8_kv
) {
    dim3 grid(n_q_heads + 2 * n_kv_heads, n_tokens);
    const int smem = head_dim * sizeof(float);
    // int8 off (bf16) instantiates with no int8 code -> byte-identical to the pre-int8 (main) kernel.
    if (int8_kv)
        qknorm_rope_kv_kernel<true><<<grid, head_dim, smem, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(q), reinterpret_cast<__nv_bfloat16*>(k),
            reinterpret_cast<const __nv_bfloat16*>(v),
            reinterpret_cast<const __nv_bfloat16*>(q_w), reinterpret_cast<const __nv_bfloat16*>(k_w),
            k_pool, v_pool, reinterpret_cast<__half*>(k_scale), reinterpret_cast<__half*>(v_scale),
            block_table, positions, n_q_heads, n_kv_heads, head_dim, theta,
            block_size, max_blocks_per_seq, eps);
    else
        qknorm_rope_kv_kernel<false><<<grid, head_dim, smem, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(q), reinterpret_cast<__nv_bfloat16*>(k),
            reinterpret_cast<const __nv_bfloat16*>(v),
            reinterpret_cast<const __nv_bfloat16*>(q_w), reinterpret_cast<const __nv_bfloat16*>(k_w),
            k_pool, v_pool, reinterpret_cast<__half*>(k_scale), reinterpret_cast<__half*>(v_scale),
            block_table, positions, n_q_heads, n_kv_heads, head_dim, theta,
            block_size, max_blocks_per_seq, eps);
}

void launch_qknorm_rope_kv_partial(
    void* q, void* k, const void* v, const void* q_w, const void* k_w,
    void* k_pool, void* v_pool, const int* block_table, const int* positions,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim,
    float theta, float eps, int block_size, int max_blocks_per_seq, cudaStream_t stream
) {
    dim3 grid(n_q_heads + 2 * n_kv_heads, n_tokens);
    const int smem = head_dim * sizeof(float);
    qknorm_rope_kv_partial_kernel<<<grid, head_dim, smem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q), reinterpret_cast<__nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<const __nv_bfloat16*>(q_w), reinterpret_cast<const __nv_bfloat16*>(k_w),
        reinterpret_cast<__nv_bfloat16*>(k_pool), reinterpret_cast<__nv_bfloat16*>(v_pool),
        block_table, positions, n_q_heads, n_kv_heads, head_dim, rotary_dim, theta,
        block_size, max_blocks_per_seq, eps);
}

void launch_rope_kv_append(void* q, const void* k, const void* v, void* k_pool, void* v_pool,
                           const int* block_table, const int* positions,
                           int n_tokens, int n_q_heads, int n_kv_heads, int head_dim, float theta,
                           int block_size, int max_blocks_per_seq, cudaStream_t stream) {
    const int half = head_dim >> 1;
    const int total = n_q_heads * half + n_kv_heads * half + n_kv_heads * head_dim;
    dim3 grid((total + 255) / 256, n_tokens);
    rope_kv_append_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<__nv_bfloat16*>(k_pool), reinterpret_cast<__nv_bfloat16*>(v_pool),
        block_table, positions, n_q_heads, n_kv_heads, head_dim, theta, block_size, max_blocks_per_seq);
}

void launch_rope_kv_append_partial(void* q, const void* k, const void* v, void* k_pool, void* v_pool,
                                   const int* block_table, const int* positions,
                                   int n_tokens, int n_q_heads, int n_kv_heads,
                                   int head_dim, int rotary_dim, float theta,
                                   int block_size, int max_blocks_per_seq, cudaStream_t stream) {
    if (rotary_dim <= 0 || rotary_dim >= head_dim) {
        launch_rope_kv_append(q, k, v, k_pool, v_pool, block_table, positions,
                              n_tokens, n_q_heads, n_kv_heads, head_dim, theta,
                              block_size, max_blocks_per_seq, stream);
        return;
    }
    const int rhalf = rotary_dim >> 1;
    const int total = n_q_heads * rhalf + n_kv_heads * rhalf +
                      n_kv_heads * (head_dim - rotary_dim) +
                      n_kv_heads * head_dim;
    dim3 grid((total + 255) / 256, n_tokens);
    rope_kv_append_partial_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<__nv_bfloat16*>(k_pool), reinterpret_cast<__nv_bfloat16*>(v_pool),
        block_table, positions, n_q_heads, n_kv_heads, head_dim, rotary_dim, theta,
        block_size, max_blocks_per_seq);
}

void launch_rope_kv_append_partial_int8(void* q, const void* k, const void* v,
                                        void* k_pool, void* v_pool, void* k_scale, void* v_scale,
                                        const int* block_table, const int* positions,
                                        int n_tokens, int n_q_heads, int n_kv_heads,
                                        int head_dim, int rotary_dim, float theta,
                                        int block_size, int max_blocks_per_seq, cudaStream_t stream) {
    // one block per (token, head-unit); blockDim == head_dim so a full K/V head vector reduces in-block.
    dim3 grid(n_q_heads + 2 * n_kv_heads, n_tokens);
    rope_kv_append_partial_int8_kernel<<<grid, head_dim, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<signed char*>(k_pool), reinterpret_cast<signed char*>(v_pool),
        reinterpret_cast<__half*>(k_scale), reinterpret_cast<__half*>(v_scale),
        block_table, positions, n_q_heads, n_kv_heads, head_dim, rotary_dim, theta,
        block_size, max_blocks_per_seq);
}

void launch_qknorm_rope_kv_partial_int8(void* q, void* k, const void* v, const void* q_w, const void* k_w,
                                        void* k_pool, void* v_pool, void* k_scale, void* v_scale,
                                        const int* block_table, const int* positions,
                                        int n_tokens, int n_q_heads, int n_kv_heads,
                                        int head_dim, int rotary_dim, float theta, float eps,
                                        int block_size, int max_blocks_per_seq, cudaStream_t stream) {
    dim3 grid(n_q_heads + 2 * n_kv_heads, n_tokens);
    const int smem = head_dim * (int)sizeof(float);
    qknorm_rope_kv_partial_int8_kernel<<<grid, head_dim, smem, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q), reinterpret_cast<__nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<const __nv_bfloat16*>(q_w), reinterpret_cast<const __nv_bfloat16*>(k_w),
        reinterpret_cast<signed char*>(k_pool), reinterpret_cast<signed char*>(v_pool),
        reinterpret_cast<__half*>(k_scale), reinterpret_cast<__half*>(v_scale),
        block_table, positions, n_q_heads, n_kv_heads, head_dim, rotary_dim, theta,
        block_size, max_blocks_per_seq, eps);
}

void launch_qknorm_rope_kv_partial_int8_gated(
    const void* qraw, void* q, void* qgate, void* k, const void* v, const void* q_w, const void* k_w,
    void* k_pool, void* v_pool, void* k_scale, void* v_scale,
    const int* block_table, const int* positions,
    int n_tokens, int n_q_heads, int n_kv_heads, int head_dim, int rotary_dim,
    float theta, float eps, int block_size, int max_blocks_per_seq, cudaStream_t stream) {
    dim3 grid(n_q_heads + 2 * n_kv_heads, n_tokens);
    const int smem = head_dim * (int)sizeof(float);
    qknorm_rope_kv_partial_int8_gated_kernel<<<grid, head_dim, smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qraw), reinterpret_cast<__nv_bfloat16*>(q),
        reinterpret_cast<__nv_bfloat16*>(qgate),
        reinterpret_cast<__nv_bfloat16*>(k), reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<const __nv_bfloat16*>(q_w), reinterpret_cast<const __nv_bfloat16*>(k_w),
        reinterpret_cast<signed char*>(k_pool), reinterpret_cast<signed char*>(v_pool),
        reinterpret_cast<__half*>(k_scale), reinterpret_cast<__half*>(v_scale),
        block_table, positions, n_q_heads, n_kv_heads, head_dim, rotary_dim, theta,
        block_size, max_blocks_per_seq, eps);
}

void launch_rope(void* q, void* k, const int* positions,
                 int n_tokens, int n_q_heads, int n_kv_heads, int head_dim,
                 float theta, cudaStream_t stream) {
    static int fuse = -1;   // default ON: fused Q+K rope (1 kernel). SPARKINFER_ROPEFUSE=0 disables
    if (fuse < 0) { const char* e = getenv("SPARKINFER_ROPEFUSE"); fuse = (e && e[0] == '0') ? 0 : 1; }
    if (fuse) {
        const int total = (n_q_heads + n_kv_heads) * (head_dim >> 1);
        dim3 grid((total + 255) / 256, n_tokens);
        rope_qk_kernel<<<grid, 256, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(q), reinterpret_cast<__nv_bfloat16*>(k),
            positions, n_q_heads, n_kv_heads, head_dim, theta);
        return;
    }
    const int half = head_dim / 2;
    dim3 gq(n_tokens, n_q_heads);
    rope_kernel<<<gq, half, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(q), positions, n_q_heads, head_dim, theta);
    dim3 gk(n_tokens, n_kv_heads);
    rope_kernel<<<gk, half, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(k), positions, n_kv_heads, head_dim, theta);
}
#endif

} // namespace kernels
} // namespace sparkinfer
