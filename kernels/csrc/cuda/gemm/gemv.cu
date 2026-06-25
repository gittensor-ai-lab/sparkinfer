// Decode GEMV: y[N] = x[K] @ W^T, where W is [N, K] row-major (i.e. [out, in] —
// the GGUF-native linear layout). One warp computes one output row n: the warp
// streams W[n, :] (K contiguous bf16 → fully coalesced across lanes) and dots it
// with x (staged in shared memory). This replaces the M=1 tiled GEMM, which
// wasted ~16x of its threads on the empty batch dimension at decode time.
//
// Output is bf16 (projections) or fp32 (router / LM-head logits) via the OutT
// template. Portable CUDA — sm_89 .. sm_120/sm_121.

#include <cuda_bf16.h>
#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include <cuda_runtime.h>
#endif

namespace sparkinfer {
namespace kernels {

static constexpr int GEMV_WPB = 8;   // warps (output rows) per block

__device__ __forceinline__ void gemv_write(float* p, float v) { *p = v; }
__device__ __forceinline__ void gemv_write(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename OutT>
__global__ void gemv_kernel(const __nv_bfloat16* __restrict__ x,
                            const __nv_bfloat16* __restrict__ W,
                            OutT* __restrict__ y, int N, int K) {
    extern __shared__ float s_x[];                 // K floats
    for (int i = threadIdx.x; i < K; i += blockDim.x) s_x[i] = __bfloat162float(x[i]);
    __syncthreads();

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n = blockIdx.x * GEMV_WPB + warp;
    if (n >= N) return;
    // 128-bit coalesced loads: each lane pulls a uint4 = 8 bf16 of the weight row.
    const uint4* row4 = reinterpret_cast<const uint4*>(W + (size_t)n * K);
    const int n4 = K / 8;
    float acc = 0.f;
    for (int i = lane; i < n4; i += 32) {
        uint4 v = row4[i];
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&v);
        const int base = i * 8;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f = __bfloat1622float2(h2[j]);
            acc += f.x * s_x[base + 2*j] + f.y * s_x[base + 2*j + 1];
        }
    }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, m);
    if (lane == 0) gemv_write(y + n, acc);
}

template __global__ void gemv_kernel<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int);
template __global__ void gemv_kernel<float>(const __nv_bfloat16*, const __nv_bfloat16*, float*, int, int);

// ---- quantized on-read GEMV (W = GGUF-native Q4_K/Q6_K [N,K]) -----------------
// Dequantizes each 256-block in registers and dots with a full-precision (fp32)
// activation — reads the quantized weight bytes (~4x less than bf16) with NO int8
// activation, so the result matches the bf16-weight GEMV up to dequant order and
// token-match is preserved. k-quant decoders are the byte-exact ones validated in
// dequant_gguf.cu / expert_ffn_q4k.cu. One warp per output row. K % 256 == 0.
__device__ __forceinline__ float gq_h2f(const unsigned char* p) {
    __half h; *((unsigned short*)&h) = *(const unsigned short*)p; return __half2float(h);
}
__device__ __forceinline__ void gq_scale_min(int j, const unsigned char* q, int* d, int* m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
           *m = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4); }
}
__device__ __forceinline__ int gq_block_bytes(int t) { return t == 14 ? 210 : 144; }

template <typename OutT>
__global__ void gemv_q_kernel(const __nv_bfloat16* __restrict__ x,
                              const unsigned char* __restrict__ W,
                              OutT* __restrict__ y, int N, int K, int wtype) {
    extern __shared__ float s_x[];                 // K floats
    for (int i = threadIdx.x; i < K; i += blockDim.x) s_x[i] = __bfloat162float(x[i]);
    __syncthreads();

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int n = blockIdx.x * GEMV_WPB + warp;
    if (n >= N) return;
    const int nblk = K / 256, bb = gq_block_bytes(wtype);
    const unsigned char* base = W + (size_t)n * nblk * bb;
    float acc = 0.f;
    // dequant in registers and FMA straight against the activation — no shared
    // round-trip, one warp-reduce at the end. Reads the quantized row coalesced.
    for (int blk = 0; blk < nblk; blk++) {
        const unsigned char* b = base + (size_t)blk * bb;
        const float* sx = s_x + blk * 256;
        if (wtype == 14) {   // Q6_K
            const unsigned char* ql = b; const unsigned char* qh = b + 128;
            const signed char* sc = (const signed char*)(b + 192); float d = gq_h2f(b + 208);
            #pragma unroll
            for (int nn = 0; nn < 2; nn++) {
                const unsigned char* qln = ql + nn*64; const unsigned char* qhn = qh + nn*32; const signed char* scn = sc + nn*8;
                int is = lane / 16;
                int q1 = (int)((qln[lane]    & 0xF) | (((qhn[lane] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((qln[lane+32] & 0xF) | (((qhn[lane] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((qln[lane]    >> 4)  | (((qhn[lane] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((qln[lane+32] >> 4)  | (((qhn[lane] >> 6) & 3) << 4)) - 32;
                acc += d * scn[is+0] * q1 * sx[nn*128 + lane];
                acc += d * scn[is+2] * q2 * sx[nn*128 + lane + 32];
                acc += d * scn[is+4] * q3 * sx[nn*128 + lane + 64];
                acc += d * scn[is+6] * q4 * sx[nn*128 + lane + 96];
            }
        } else {             // Q4_K
            float d = gq_h2f(b), dmin = gq_h2f(b + 2);
            const unsigned char* sc = b + 4; const unsigned char* qs = b + 16;
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                int s1, m1, s2, m2;
                gq_scale_min(2*g, sc, &s1, &m1); gq_scale_min(2*g+1, sc, &s2, &m2);
                float d1 = d*s1, mm1 = dmin*m1, d2 = d*s2, mm2 = dmin*m2;
                unsigned char qb = qs[g*32 + lane];
                acc += (d1 * (qb & 0xF) - mm1) * sx[g*64 + lane];
                acc += (d2 * (qb >> 4)  - mm2) * sx[g*64 + 32 + lane];
            }
        }
    }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, m);
    if (lane == 0) gemv_write(y + n, acc);
}

template __global__ void gemv_q_kernel<__nv_bfloat16>(const __nv_bfloat16*, const unsigned char*, __nv_bfloat16*, int, int, int);
template __global__ void gemv_q_kernel<float>(const __nv_bfloat16*, const unsigned char*, float*, int, int, int);

// ---- faithful llama.cpp int8 MMVQ for a dense Q4_K [N,K] GEMV --------------------
// Quantizes the activation to Q8_1 (int8 + per-32 scale + sum) once per token, then
// dp4a's the Q4_K weight nibbles against it — the same vec_dot_q4_K_q8_1 math llama.cpp
// uses, so the output converges to llama's (no top-1 regression vs the int8 reference).
// Q4_K only (ggml type 12); the launcher keeps Q6_K on the fp path. One warp per row.
template <typename OutT>
__global__ void gemv_q_dp4a_kernel(const __nv_bfloat16* __restrict__ x,
                                   const unsigned char* __restrict__ W,
                                   OutT* __restrict__ y, int N, int K) {
    extern __shared__ char smemq[];
    float* s_xd = reinterpret_cast<float*>(smemq);        // [K/32]
    float* s_xs = s_xd + (K >> 5);                         // [K/32]
    signed char* s_xq8 = reinterpret_cast<signed char*>(s_xs + (K >> 5));  // [K]
    const int warpId = threadIdx.x >> 5, lane = threadIdx.x & 31, nsb = K >> 5;

    for (int b = warpId; b < nsb; b += GEMV_WPB) {        // activation -> Q8_1
        float xv = __bfloat162float(x[b * 32 + lane]);
        float a = fabsf(xv);
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
        float d = a / 127.0f;                                  // faithful to llama Q8_1:
        int qi = (a == 0.0f) ? 0 : (int)roundf(xv / d);        // roundf(xi/d), not rn(xi*inv)
        s_xq8[b * 32 + lane] = (signed char)qi;
        int sm = qi;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, m);
        if (lane == 0) { s_xd[b] = d; s_xs[b] = d * (float)sm; }
    }
    __syncthreads();

    const int n = blockIdx.x * GEMV_WPB + warpId;
    if (n >= N) return;
    const unsigned char* base = W + (size_t)n * (K >> 8) * 144;   // Q4_K: K/256 blocks * 144 B
    float acc = 0.f;
    for (int sb = lane; sb < nsb; sb += 32) {
        const int super = sb >> 3, sib = sb & 7;
        const int* aint = reinterpret_cast<const int*>(s_xq8 + (sb << 5));
        const float xd = s_xd[sb], xs = s_xs[sb];
        const unsigned char* blk = base + (size_t)super * 144;
        float d = gq_h2f(blk), dmin = gq_h2f(blk + 2);
        int scd, scm; gq_scale_min(sib, blk + 4, &scd, &scm);
        const int* q = reinterpret_cast<const int*>(blk + 16 + (sib >> 1) * 32);
        const bool hi = sib & 1;
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int w = hi ? ((q[k] >> 4) & 0x0F0F0F0F) : (q[k] & 0x0F0F0F0F);
            sumi = __dp4a(w, aint[k], sumi);
        }
        acc += d * (float)scd * xd * (float)sumi - dmin * (float)scm * xs;
    }
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, m);
    if (lane == 0) gemv_write(y + n, acc);
}

template __global__ void gemv_q_dp4a_kernel<__nv_bfloat16>(const __nv_bfloat16*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void gemv_q_dp4a_kernel<float>(const __nv_bfloat16*, const unsigned char*, float*, int, int);

#ifndef SPARKINFER_NVRTC_DEVICE_ONLY
#include "sparkinfer/kernels/gemm.h"
#include <cstdlib>

// int8 dp4a for Q4_K GEMVs (faithful to llama.cpp's mul_mat_vec_q). Default ON —
// ~27% faster decode than the fp32-dequant path and still clears the accuracy gate
// (top1 0.97, KL 0.15 vs llama.cpp). Set SPARKINFER_MMVQ=0 to fall back to fp32.
static bool gemv_mmvq() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("SPARKINFER_MMVQ"); v = (e && e[0] == '0') ? 0 : 1; }
    return v;
}

void launch_gemv(const void* x, const void* W, void* y, int N, int K, cudaStream_t stream) {
    dim3 grid((N + GEMV_WPB - 1) / GEMV_WPB);
    gemv_kernel<__nv_bfloat16><<<grid, GEMV_WPB * 32, (size_t)K * sizeof(float), stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(W),
        reinterpret_cast<__nv_bfloat16*>(y), N, K);
}

void launch_gemv_f32(const void* x, const void* W, float* y, int N, int K, cudaStream_t stream) {
    dim3 grid((N + GEMV_WPB - 1) / GEMV_WPB);
    gemv_kernel<float><<<grid, GEMV_WPB * 32, (size_t)K * sizeof(float), stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(W), y, N, K);
}

void launch_gemv_q(const void* x, const void* W, int wtype, void* y, int N, int K, cudaStream_t stream) {
    dim3 grid((N + GEMV_WPB - 1) / GEMV_WPB);
    if (gemv_mmvq() && wtype == 12) {   // faithful int8 dp4a (Q4_K)
        size_t sm = 2 * (size_t)(K >> 5) * sizeof(float) + (size_t)K;
        gemv_q_dp4a_kernel<__nv_bfloat16><<<grid, GEMV_WPB * 32, sm, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const unsigned char*>(W),
            reinterpret_cast<__nv_bfloat16*>(y), N, K);
    } else {
        gemv_q_kernel<__nv_bfloat16><<<grid, GEMV_WPB * 32, (size_t)K * sizeof(float), stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const unsigned char*>(W),
            reinterpret_cast<__nv_bfloat16*>(y), N, K, wtype);
    }
}
void launch_gemv_q_f32(const void* x, const void* W, int wtype, float* y, int N, int K, cudaStream_t stream) {
    dim3 grid((N + GEMV_WPB - 1) / GEMV_WPB);
    if (gemv_mmvq() && wtype == 12) {
        size_t sm = 2 * (size_t)(K >> 5) * sizeof(float) + (size_t)K;
        gemv_q_dp4a_kernel<float><<<grid, GEMV_WPB * 32, sm, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const unsigned char*>(W), y, N, K);
    } else {
        gemv_q_kernel<float><<<grid, GEMV_WPB * 32, (size_t)K * sizeof(float), stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const unsigned char*>(W), y, N, K, wtype);
    }
}
#endif

} // namespace kernels
} // namespace sparkinfer
