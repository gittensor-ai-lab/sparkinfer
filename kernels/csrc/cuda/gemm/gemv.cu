// Decode GEMV: y[N] = x[K] @ W^T, where W is [N, K] row-major (i.e. [out, in] —
// the GGUF-native linear layout). One warp computes one output row n: the warp
// streams W[n, :] (K contiguous bf16 → fully coalesced across lanes) and dots it
// with x (staged in shared memory). This replaces the M=1 tiled GEMM, which
// wasted ~16x of its threads on the empty batch dimension at decode time.
//
// Output is bf16 (projections) or fp32 (router / LM-head logits) via the OutT
// template. Portable CUDA — sm_89 .. sm_120/sm_121.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

// ---- pre-quantized activation Q8_1 + dp4a GEMV (kills per-block re-quantization) --
// gemv_q_dp4a_kernel re-quantizes the SAME activation to Q8_1 in EVERY block (256x for
// a 2048-row projection). When several GEMVs share an activation (Q/K/V all read xn) it
// is also re-done per GEMV. quantize_q8_1_kernel does it ONCE to a small global buffer;
// gemv_q4k_dp4a_pq_kernel then reads the pre-quantized int8 (L2-resident) and runs the
// IDENTICAL dp4a — same Q8_1 values, so the output is BIT-EXACT vs the in-kernel path.
__global__ void quantize_q8_1_kernel(const __nv_bfloat16* __restrict__ x,
                                     signed char* __restrict__ q8, float* __restrict__ ad,
                                     float* __restrict__ as, int K) {
    const int warpId = threadIdx.x >> 5, lane = threadIdx.x & 31, nsb = K >> 5;
    const int nwarp = blockDim.x >> 5;
    for (int b = warpId; b < nsb; b += nwarp) {
        float xv = __bfloat162float(x[b * 32 + lane]);
        float a = fabsf(xv);
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
        float d = a / 127.0f;
        int qi = (a == 0.0f) ? 0 : (int)roundf(xv / d);
        q8[b * 32 + lane] = (signed char)qi;
        int sm = qi;
        #pragma unroll
        for (int m = 16; m > 0; m >>= 1) sm += __shfl_xor_sync(0xffffffffu, sm, m);
        if (lane == 0) { ad[b] = d; as[b] = d * (float)sm; }
    }
}

template <typename OutT>
__global__ void gemv_q4k_dp4a_pq_kernel(const signed char* __restrict__ q8,
                                        const float* __restrict__ ad, const float* __restrict__ as,
                                        const unsigned char* __restrict__ W,
                                        OutT* __restrict__ y, int N, int K) {
    const int warpId = threadIdx.x >> 5, lane = threadIdx.x & 31, nsb = K >> 5;
    const int n = blockIdx.x * GEMV_WPB + warpId;
    if (n >= N) return;
    const unsigned char* base = W + (size_t)n * (K >> 8) * 144;   // Q4_K: K/256 blocks * 144 B
    float acc = 0.f;
    for (int sb = lane; sb < nsb; sb += 32) {
        const int super = sb >> 3, sib = sb & 7;
        const int* aint = reinterpret_cast<const int*>(q8 + (sb << 5));   // pre-quantized (global, L2)
        const float xd = ad[sb], xs = as[sb];
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

template __global__ void gemv_q4k_dp4a_pq_kernel<__nv_bfloat16>(const signed char*, const float*, const float*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void gemv_q4k_dp4a_pq_kernel<float>(const signed char*, const float*, const float*, const unsigned char*, float*, int, int);

// ---- split-K variant of the pre-quantized dp4a GEMV (occupancy lever) -------------
// ncu: the one-warp-per-row dp4a GEMV is occupancy-bound (~47%) — a 4096-row projection
// is only 4096 warps, under-filling the GPU. S warps cooperate per output row (each does
// 1/S of the K-blocks), then an S-way shared reduce. S=2 doubles the warps in flight ->
// fills the SMs. Bit-exact (same dp4a, only the partial-sum split changes). One block =
// RPB rows x S warps.
template <typename OutT>
__global__ void gemv_q4k_dp4a_sk_kernel(const signed char* __restrict__ q8,
                                        const float* __restrict__ ad, const float* __restrict__ as,
                                        const unsigned char* __restrict__ W,
                                        OutT* __restrict__ y, int N, int K) {
    constexpr int S = 2, RPB = GEMV_WPB / S;          // splits/row, rows/block
    __shared__ float s_part[RPB][S];
    const int lane = threadIdx.x & 31, warpId = threadIdx.x >> 5;
    const int row_local = warpId / S, split = warpId % S;
    const int n = blockIdx.x * RPB + row_local;
    const int nsb = K >> 5;
    float acc = 0.f;
    if (n < N) {
        const unsigned char* base = W + (size_t)n * (K >> 8) * 144;
        for (int sb = split * 32 + lane; sb < nsb; sb += S * 32) {     // this warp's K-slice
            const int super = sb >> 3, sib = sb & 7;
            const int* aint = reinterpret_cast<const int*>(q8 + (sb << 5));
            const float xd = ad[sb], xs = as[sb];
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
        if (lane == 0) s_part[row_local][split] = acc;
    }
    __syncthreads();
    if (n < N && split == 0 && lane == 0) {
        float o = 0.f;
        #pragma unroll
        for (int s = 0; s < S; s++) o += s_part[row_local][s];
        gemv_write(y + n, o);
    }
}

template __global__ void gemv_q4k_dp4a_sk_kernel<__nv_bfloat16>(const signed char*, const float*, const float*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void gemv_q4k_dp4a_sk_kernel<float>(const signed char*, const float*, const float*, const unsigned char*, float*, int, int);

// ===== faithful llama.cpp Q4_K mul_mat_vec_q port (block_q8_1 activation + vec_dot) =====
// Replicates ggml-cuda's mmvq exactly for decode (ncols=1): nwarps=4 cooperate on one row,
// vdr=2 ints/thread (16 threads/superblock), block_q8_1 interleaved activation, and llama's
// per-lane cross-warp reduction. Tests whether llama's holistic kernel beats our split-K.
struct si_block_q8_1 { __half2 ds; signed char qs[32]; };               // 36 B / 32 values
struct si_block_q4_K { __half2 dm; unsigned char scales[12]; unsigned char qs[128]; };  // 144 B / 256

__global__ void si_quantize_q8_1_blocks(const __nv_bfloat16* __restrict__ x,
                                        si_block_q8_1* __restrict__ y, int K) {
    const int warpsPB = blockDim.x >> 5, ib = blockIdx.x * warpsPB + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (ib >= (K >> 5)) return;
    float xv = __bfloat162float(x[ib * 32 + lane]), a = fabsf(xv);
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, m));
    float d = a / 127.0f;
    int qi = (a == 0.0f) ? 0 : (int)roundf(xv / d);
    y[ib].qs[lane] = (signed char)qi;
    int s = qi;
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
    if (lane == 0) y[ib].ds = __floats2half2_rn(d, d * (float)s);
}

__device__ __forceinline__ float si_vec_dot_q4_K(const si_block_q4_K* bq4,
                                                 const si_block_q8_1* bq8_1, int iqs) {
    int v[2], u[4]; float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int* q4 = (const int*)(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0]; v[1] = q4[4];
    const unsigned short* scales = (const unsigned short*)bq4->scales;
    unsigned short aux[2]; const int j = bq8_offset / 2;
    if (j < 2) { aux[0] = scales[j] & 0x3f3f; aux[1] = scales[j + 2] & 0x3f3f; }
    else { aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
           aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j]     & 0xc0c0) >> 2); }
    const unsigned char* sc = (const unsigned char*)aux; const unsigned char* m = sc + 2;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const si_block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*)bq8i->qs + ((iqs / 2) % 4);
        u[2 * i] = q8[0]; u[2 * i + 1] = q8[4];
    }
    float sumf_d = 0.0f, sumf_m = 0.0f;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F, v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    float2 dm4f = __half22float2(bq4->dm);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

template <typename OutT>
__global__ void si_mmvq_q4k_kernel(const si_block_q8_1* __restrict__ vy, const unsigned char* __restrict__ W,
                                   OutT* __restrict__ y, int N, int K) {
    constexpr int NW = 4, WS = 32, vdr = 2, qi = 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
    const int row = blockIdx.x;
    const si_block_q4_K* x_row = (const si_block_q4_K*)(W + (size_t)row * (K >> 8) * 144);
    const int blocks_per_row = K >> 8;                       // 256-weight superblocks
    const int blocks_per_iter = vdr * NW * WS / qi;          // = 8
    float tmp = 0.0f;
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * 8;                             // q8_1 blocks per superblock = 8
        const int kqs = vdr * (tid % (qi / vdr));
        tmp += si_vec_dot_q4_K(x_row + kbx, vy + kby, kqs);
    }
    __shared__ float tmp_shared[NW - 1][WS];
    if (warp > 0) tmp_shared[warp - 1][lane] = tmp;
    __syncthreads();
    if (warp > 0) return;
    #pragma unroll
    for (int l = 0; l < NW - 1; l++) tmp += tmp_shared[l][lane];
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) tmp += __shfl_xor_sync(0xffffffff, tmp, m);
    if (lane == 0) gemv_write(y + row, tmp);
}

template __global__ void si_mmvq_q4k_kernel<__nv_bfloat16>(const si_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_mmvq_q4k_kernel<float>(const si_block_q8_1*, const unsigned char*, float*, int, int);

// ===== faithful llama Q6_K mmvq for the fp32-path GEMVs (attn-V upgrades + LM head) =====
// Same 4-warp-per-row structure as the Q4_K mmvq, with vec_dot_q6_K_q8_1 (coalesced
// ql/qh int loads + __vsubss4 reconstruct + dp4a). Mirrors the #65 MoE-down dot.
__device__ __forceinline__ int si_get_int_b2(const void* x, int i32) {
    const unsigned short* x16 = reinterpret_cast<const unsigned short*>(x);
    return (int)x16[2 * i32] | ((int)x16[2 * i32 + 1] << 16);
}
__device__ __forceinline__ float si_vec_dot_q6_K(const unsigned char* __restrict__ bq6,
                                                 const si_block_q8_1* __restrict__ bq8, int iqs) {
    const signed char* scales = reinterpret_cast<const signed char*>(bq6 + 192);
    const float d = gq_h2f(bq6 + 208);
    const int bq8_offset   = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift     = 2 * ((iqs % 16) / 8);
    const int vl = si_get_int_b2(bq6, iqs);
    const int vh = si_get_int_b2(bq6 + 128, 8 * (iqs / 16) + (iqs % 8)) >> vh_shift;
    const signed char* sc = scales + scale_offset;
    float sumf = 0.f;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const si_block_q8_1* b8 = bq8 + bq8_offset + 2 * i;
        const int u = reinterpret_cast<const int*>(b8->qs)[iqs % 8];
        const float d8 = __low2float(b8->ds);
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi  = __vsubss4((vil | vih), 0x20202020);
        sumf += d8 * (__dp4a(vi, u, 0) * (int)sc[4 * i]);
    }
    return d * sumf;
}

template <typename OutT>
__global__ void si_mmvq_q6k_kernel(const si_block_q8_1* __restrict__ vy, const unsigned char* __restrict__ W,
                                   OutT* __restrict__ y, int N, int K) {
    constexpr int NW = 4, WS = 32, vdr = 1, qi = 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
    const int row = blockIdx.x;
    const unsigned char* x_row = W + (size_t)row * (K >> 8) * 210;   // Q6_K: 210 B / 256-superblock
    const int blocks_per_row = K >> 8;
    const int blocks_per_iter = vdr * NW * WS / qi;                  // = 4
    float tmp = 0.0f;
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * 8;                                    // q8_1 blocks per superblock
        const int kqs = vdr * (tid % (qi / vdr));                   // = lane
        tmp += si_vec_dot_q6_K(x_row + (size_t)kbx * 210, vy + kby, kqs);
    }
    __shared__ float tmp_shared[NW - 1][WS];
    if (warp > 0) tmp_shared[warp - 1][lane] = tmp;
    __syncthreads();
    if (warp > 0) return;
    #pragma unroll
    for (int l = 0; l < NW - 1; l++) tmp += tmp_shared[l][lane];
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) tmp += __shfl_xor_sync(0xffffffff, tmp, m);
    if (lane == 0) gemv_write(y + row, tmp);
}
template __global__ void si_mmvq_q6k_kernel<__nv_bfloat16>(const si_block_q8_1*, const unsigned char*, __nv_bfloat16*, int, int);
template __global__ void si_mmvq_q6k_kernel<float>(const si_block_q8_1*, const unsigned char*, float*, int, int);

// 1-warp-per-row Q6_K dp4a GEMV: keeps the fp32 gemv_q block structure (GEMV_WPB rows/block,
// well-occupied for large N like the LM head's 151936 rows) but dp4a instead of fp32 dequant.
// The 4-warp si_mmvq is right for small-N rows (attn-V); this is right for the huge LM head.
template <typename OutT>
__global__ void gemv_q6k_dp4a_kernel(const si_block_q8_1* __restrict__ vy, const unsigned char* __restrict__ W,
                                     OutT* __restrict__ y, int N, int K) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GEMV_WPB + warp;
    if (row >= N) return;
    const unsigned char* x_row = W + (size_t)row * (K >> 8) * 210;
    const int nsuper = K >> 8;
    float acc = 0.f;
    for (int kbx = 0; kbx < nsuper; kbx++)
        acc += si_vec_dot_q6_K(x_row + (size_t)kbx * 210, vy + (size_t)kbx * 8, lane);
    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, m);
    if (lane == 0) gemv_write(y + row, acc);
}
template __global__ void gemv_q6k_dp4a_kernel<float>(const si_block_q8_1*, const unsigned char*, float*, int, int);

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

// Quantize an activation x[K] to Q8_1 once (q8[K] int8, ad[K/32] scales, as[K/32] = d*sum).
void launch_quantize_q8_1(const void* x, void* q8, float* ad, float* as, int K, cudaStream_t stream) {
    quantize_q8_1_kernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<signed char*>(q8), ad, as, K);
}
// SPARKINFER_GEMVSK=0 -> plain one-warp-per-row pre-quantized GEMV (default uses split-K
// for occupancy: S=2 warps/row fills the GPU on the small attn projections).
static bool gemv_sk() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("SPARKINFER_GEMVSK"); v = (e && e[0] == '0') ? 0 : 1; }
    return v;
}
// Q4_K dp4a GEMV against a pre-quantized activation (no per-block re-quant). bf16/f32 out.
void launch_gemv_q_dp4a_pq(const void* q8, const float* ad, const float* as, const void* W,
                           void* y, int N, int K, cudaStream_t stream) {
    if (gemv_sk()) {   // split-K: S=2 warps/row (measured optimum; 4-warp/fine-grained was slower)
        constexpr int RPB = GEMV_WPB / 2;
        dim3 grid((N + RPB - 1) / RPB);
        gemv_q4k_dp4a_sk_kernel<__nv_bfloat16><<<grid, GEMV_WPB * 32, 0, stream>>>(
            reinterpret_cast<const signed char*>(q8), ad, as, reinterpret_cast<const unsigned char*>(W),
            reinterpret_cast<__nv_bfloat16*>(y), N, K);
        return;
    }
    dim3 grid((N + GEMV_WPB - 1) / GEMV_WPB);
    gemv_q4k_dp4a_pq_kernel<__nv_bfloat16><<<grid, GEMV_WPB * 32, 0, stream>>>(
        reinterpret_cast<const signed char*>(q8), ad, as, reinterpret_cast<const unsigned char*>(W),
        reinterpret_cast<__nv_bfloat16*>(y), N, K);
}
void launch_gemv_q_dp4a_pq_f32(const void* q8, const float* ad, const float* as, const void* W,
                               float* y, int N, int K, cudaStream_t stream) {
    if (gemv_sk()) {   // split-K: S=2 warps/row (measured optimum)
        constexpr int RPB = GEMV_WPB / 2;
        dim3 grid((N + RPB - 1) / RPB);
        gemv_q4k_dp4a_sk_kernel<float><<<grid, GEMV_WPB * 32, 0, stream>>>(
            reinterpret_cast<const signed char*>(q8), ad, as, reinterpret_cast<const unsigned char*>(W), y, N, K);
        return;
    }
    dim3 grid((N + GEMV_WPB - 1) / GEMV_WPB);
    gemv_q4k_dp4a_pq_kernel<float><<<grid, GEMV_WPB * 32, 0, stream>>>(
        reinterpret_cast<const signed char*>(q8), ad, as, reinterpret_cast<const unsigned char*>(W), y, N, K);
}

// ---- faithful llama.cpp Q4_K mmvq launchers ----
size_t llama_q8_1_bytes(int K) { return (size_t)(K >> 5) * sizeof(si_block_q8_1); }  // 36 B / 32 vals
void launch_quantize_q8_1_blocks(const void* x, void* y, int K, cudaStream_t stream) {
    const int nb = K >> 5, warpsPB = 8;
    dim3 grid((nb + warpsPB - 1) / warpsPB);
    si_quantize_q8_1_blocks<<<grid, warpsPB * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<si_block_q8_1*>(y), K);
}
void launch_mmvq_q4k(const void* q81, const void* W, void* y, int N, int K, cudaStream_t stream) {
    si_mmvq_q4k_kernel<__nv_bfloat16><<<N, 4 * 32, 0, stream>>>(
        reinterpret_cast<const si_block_q8_1*>(q81), reinterpret_cast<const unsigned char*>(W),
        reinterpret_cast<__nv_bfloat16*>(y), N, K);
}
void launch_mmvq_q4k_f32(const void* q81, const void* W, float* y, int N, int K, cudaStream_t stream) {
    si_mmvq_q4k_kernel<float><<<N, 4 * 32, 0, stream>>>(
        reinterpret_cast<const si_block_q8_1*>(q81), reinterpret_cast<const unsigned char*>(W), y, N, K);
}
void launch_mmvq_q6k(const void* q81, const void* W, void* y, int N, int K, cudaStream_t stream) {
    si_mmvq_q6k_kernel<__nv_bfloat16><<<N, 4 * 32, 0, stream>>>(
        reinterpret_cast<const si_block_q8_1*>(q81), reinterpret_cast<const unsigned char*>(W),
        reinterpret_cast<__nv_bfloat16*>(y), N, K);
}
void launch_mmvq_q6k_f32(const void* q81, const void* W, float* y, int N, int K, cudaStream_t stream) {
    si_mmvq_q6k_kernel<float><<<N, 4 * 32, 0, stream>>>(
        reinterpret_cast<const si_block_q8_1*>(q81), reinterpret_cast<const unsigned char*>(W), y, N, K);
}
void launch_gemv_q6k_dp4a_f32(const void* q81, const void* W, float* y, int N, int K, cudaStream_t stream) {
    dim3 grid((N + GEMV_WPB - 1) / GEMV_WPB);
    gemv_q6k_dp4a_kernel<float><<<grid, GEMV_WPB * 32, 0, stream>>>(
        reinterpret_cast<const si_block_q8_1*>(q81), reinterpret_cast<const unsigned char*>(W), y, N, K);
}
#endif

} // namespace kernels
} // namespace sparkinfer
