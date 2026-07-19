// Weight-amortized (grouped-by-expert) MoE expert FFN for Qwen3.6 batched prefill.
// See sparkinfer/kernels/moe_prefill.h for the rationale. This is the correctness-first version:
// a plain shared-memory-tiled bf16 grouped GEMM (float accumulate) plus the permute/gather/scatter
// plumbing. The GEMM's inner loop is swapped for the int8 tensor-core path once the pipeline is
// verified numerically against the per-token forward_token MoE FFN.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "sparkinfer/kernels/moe_prefill.h"

namespace sparkinfer { namespace kernels {

namespace {
using bf16 = __nv_bfloat16;
__device__ __forceinline__ float b2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16  f2b(float x) { return __float2bfloat16(x); }

// ---- permute plumbing ----
// histogram: counts[e] = number of (token,slot) pairs routed to expert e.
__global__ void moe_hist_kernel(const int* __restrict__ ids, int* __restrict__ counts, int P) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    atomicAdd(&counts[ids[p]], 1);
}

// exclusive prefix sum of counts[E] -> offsets[E+1] (offsets[E] = P). One thread, E small (256).
__global__ void moe_offsets_kernel(const int* __restrict__ counts, int* __restrict__ offsets, int E) {
    if (threadIdx.x != 0) return;
    int acc = 0;
    for (int e = 0; e < E; e++) { offsets[e] = acc; acc += counts[e]; }
    offsets[E] = acc;
}

// scatter: place each (token,slot) pair p into its expert's contiguous group.
// fill[e] is a running per-expert counter (caller zeroes it).
__global__ void moe_scatter_kernel(const int* __restrict__ ids, const float* __restrict__ w,
                                   const int* __restrict__ offsets, int* __restrict__ fill,
                                   int* __restrict__ perm_src, float* __restrict__ perm_w,
                                   int P, int top_k) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    const int e = ids[p];
    const int pos = offsets[e] + atomicAdd(&fill[e], 1);
    perm_src[pos] = p / top_k;    // token index
    perm_w[pos]   = w[p];
}

// gather: x_perm[p,:] = src[perm_src[p],:].  P on grid.x (up to 2^31) — grid.y max is only 65535 and
// P = N*top_k exceeds it for N >= 8192.
__global__ void moe_gather_kernel(const bf16* __restrict__ src, const int* __restrict__ perm_src,
                                  bf16* __restrict__ x_perm, int P, int dim) {
    const int p = blockIdx.x;
    if (p >= P) return;
    const int t = perm_src[p];
    for (int d = blockIdx.y * blockDim.x + threadIdx.x; d < dim; d += blockDim.x * gridDim.y)
        x_perm[(size_t)p * dim + d] = src[(size_t)t * dim + d];
}

// build the per-tile schedule: expert + starting row for each BM-row tile. One block, E threads.
__global__ void moe_sched_kernel(const int* __restrict__ offsets, int* __restrict__ tile_expert,
                                 int* __restrict__ tile_row0, int* __restrict__ d_T, int E, int BM) {
    // exclusive scan of per-expert tile counts (serial; E small), then fill schedule.
    if (threadIdx.x != 0) return;
    int t = 0;
    for (int e = 0; e < E; e++) {
        const int cnt = offsets[e + 1] - offsets[e];
        const int nt  = (cnt + BM - 1) / BM;
        for (int i = 0; i < nt; i++) { tile_expert[t] = e; tile_row0[t] = offsets[e] + i * BM; t++; }
    }
    *d_T = t;
}

// weighted un-permute: out[perm_src[p],:] += perm_w[p] * y[p,:].  out is fp32 (a token's top_k slots
// land in different expert groups, so they collide on out[t] -> real atomics; accumulate in fp32 for
// both correctness of the atomic and precision of the top_k-way sum). Caller casts fp32 -> bf16 after.
__global__ void moe_scatter_weighted_kernel(const bf16* __restrict__ y, const int* __restrict__ perm_src,
                                            const float* __restrict__ perm_w, float* __restrict__ out,
                                            int P, int H) {
    const int p = blockIdx.x;                 // P on grid.x (grid.y max 65535 < P for N >= 8192)
    if (p >= P) return;
    const int t = perm_src[p];
    const float w = perm_w[p];
    for (int d = blockIdx.y * blockDim.x + threadIdx.x; d < H; d += blockDim.x * gridDim.y)
        atomicAdd(&out[(size_t)t * H + d], w * b2f(y[(size_t)p * H + d]));
}

// SwiGLU: h = silu(gate)*up
__global__ void moe_swiglu_kernel(const bf16* __restrict__ gate, const bf16* __restrict__ up,
                                  bf16* __restrict__ h, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = b2f(gate[i]);
    h[i] = f2b((g / (1.f + __expf(-g))) * b2f(up[i]));
}

// ---- grouped GEMM: C[p,:] = A[p,:] @ W[e]^T for each row p in expert e's group ----
constexpr int GM = 32, GN = 32, GK = 32;   // tile; block = 256 threads (16x16), 2x2 per thread
__global__ void moe_grouped_gemm_kernel(
        const bf16* __restrict__ A, const bf16* __restrict__ W, const int* __restrict__ offsets,
        const int* __restrict__ tile_expert, const int* __restrict__ tile_row0,
        const int* __restrict__ d_T, bf16* __restrict__ C, int Nout, int K) {
    const int tile = blockIdx.y;
    if (tile >= *d_T) return;
    const int e    = tile_expert[tile];
    const int row0 = tile_row0[tile];
    const int rowend = offsets[e + 1];
    const int col0 = blockIdx.x * GN;
    const bf16* We = W + (size_t)e * Nout * K;

    __shared__ float As[GM][GK];
    __shared__ float Bs[GN][GK];

    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;   // 16x16
    float acc[2][2] = {{0,0},{0,0}};

    for (int k0 = 0; k0 < K; k0 += GK) {
        // cooperative load A[GM,GK] and W-tile[GN,GK] (256 threads, 1024 elems each -> 4 per thread)
        for (int s = threadIdx.x; s < GM * GK; s += 256) {
            const int r = s / GK, c = s % GK;
            const int gr = row0 + r, gk = k0 + c;
            As[r][c] = (gr < rowend && gk < K) ? b2f(A[(size_t)gr * K + gk]) : 0.f;
        }
        for (int s = threadIdx.x; s < GN * GK; s += 256) {
            const int r = s / GK, c = s % GK;
            const int gn = col0 + r, gk = k0 + c;
            Bs[r][c] = (gn < Nout && gk < K) ? b2f(We[(size_t)gn * K + gk]) : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < GK; kk++) {
            const float a0 = As[ty * 2 + 0][kk], a1 = As[ty * 2 + 1][kk];
            const float b0 = Bs[tx * 2 + 0][kk], b1 = Bs[tx * 2 + 1][kk];
            acc[0][0] += a0 * b0; acc[0][1] += a0 * b1;
            acc[1][0] += a1 * b0; acc[1][1] += a1 * b1;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int gr = row0 + ty * 2 + i;
        if (gr >= rowend) continue;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            const int gc = col0 + tx * 2 + j;
            if (gc < Nout) C[(size_t)gr * Nout + gc] = f2b(acc[i][j]);
        }
    }
}
// router logits (fp32): C[t,e] = sum_h hn[t,h] * router_w[e,h].  simple tiled GEMM, float output.
__global__ void moe_router_logits_kernel(const bf16* __restrict__ hn, const bf16* __restrict__ rw,
                                         float* __restrict__ C, int N, int E, int H) {
    __shared__ float As[GM][GK];
    __shared__ float Bs[GN][GK];
    const int row0 = blockIdx.y * GM, col0 = blockIdx.x * GN;
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    float acc[2][2] = {{0,0},{0,0}};
    for (int k0 = 0; k0 < H; k0 += GK) {
        for (int s = threadIdx.x; s < GM * GK; s += 256) {
            const int r = s / GK, c = s % GK, gr = row0 + r, gk = k0 + c;
            As[r][c] = (gr < N && gk < H) ? b2f(hn[(size_t)gr * H + gk]) : 0.f;
        }
        for (int s = threadIdx.x; s < GN * GK; s += 256) {
            const int r = s / GK, c = s % GK, gn = col0 + r, gk = k0 + c;
            Bs[r][c] = (gn < E && gk < H) ? b2f(rw[(size_t)gn * H + gk]) : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < GK; kk++) {
            const float a0 = As[ty*2][kk], a1 = As[ty*2+1][kk], b0 = Bs[tx*2][kk], b1 = Bs[tx*2+1][kk];
            acc[0][0]+=a0*b0; acc[0][1]+=a0*b1; acc[1][0]+=a1*b0; acc[1][1]+=a1*b1;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 2; i++){ const int gr = row0+ty*2+i; if(gr>=N) continue;
        #pragma unroll
        for (int j = 0; j < 2; j++){ const int gc = col0+tx*2+j; if(gc<E) C[(size_t)gr*E+gc]=acc[i][j]; } }
}

// shared-expert scalar gate: dsw[t] = sigmoid(hn[t] . gate_inp).  one block per token, warp-reduce.
__global__ void moe_shared_gate_kernel(const bf16* __restrict__ hn, const bf16* __restrict__ gi,
                                       float* __restrict__ dsw, int N, int H) {
    const int t = blockIdx.x, lane = threadIdx.x;   // 32 threads
    float acc = 0.f;
    for (int h = lane; h < H; h += 32) acc += b2f(hn[(size_t)t * H + h]) * b2f(gi[h]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane == 0) dsw[t] = 1.f / (1.f + __expf(-acc));
}

// finalize: out[t,h] = routed_f32[t,h] + gate_t * shared[t,h]  (bf16 out). shared/dsw may be null.
__global__ void moe_finalize_kernel(const float* __restrict__ routed, const bf16* __restrict__ shared,
                                    const float* __restrict__ dsw, bf16* __restrict__ out, int N, int H) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)N * H) return;
    float v = routed[i];
    if (shared) { const float g = dsw ? dsw[i / H] : 1.f; v += g * b2f(shared[i]); }
    out[i] = f2b(v);
}
} // namespace

void launch_moe_prefill_router_logits(const void* hn, const void* router_w, float* logits,
                                      int N, int E, int H, cudaStream_t stream) {
    dim3 grid((E + GN - 1) / GN, (N + GM - 1) / GM);
    moe_router_logits_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const bf16*>(hn), reinterpret_cast<const bf16*>(router_w), logits, N, E, H);
}

void launch_moe_shared_gate(const void* hn, const void* gate_inp, float* dsw,
                            int N, int H, cudaStream_t stream) {
    moe_shared_gate_kernel<<<N, 32, 0, stream>>>(
        reinterpret_cast<const bf16*>(hn), reinterpret_cast<const bf16*>(gate_inp), dsw, N, H);
}

void launch_moe_prefill_finalize(const float* routed_f32, const void* shared, const float* dsw,
                                 void* out, int N, int H, cudaStream_t stream) {
    const size_t n = (size_t)N * H;
    moe_finalize_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        routed_f32, reinterpret_cast<const bf16*>(shared), dsw, reinterpret_cast<bf16*>(out), N, H);
}

void launch_moe_prefill_permute(const int* expert_ids, const float* expert_weights,
                                int* counts, int* offsets, int* perm_src, float* perm_w,
                                int N, int E, int top_k, cudaStream_t stream) {
    const int P = N * top_k;
    cudaMemsetAsync(counts, 0, (size_t)E * sizeof(int), stream);
    moe_hist_kernel<<<(P + 255) / 256, 256, 0, stream>>>(expert_ids, counts, P);
    moe_offsets_kernel<<<1, 32, 0, stream>>>(counts, offsets, E);
    cudaMemsetAsync(counts, 0, (size_t)E * sizeof(int), stream);   // reuse counts as fill counter
    moe_scatter_kernel<<<(P + 255) / 256, 256, 0, stream>>>(
        expert_ids, expert_weights, offsets, counts, perm_src, perm_w, P, top_k);
}

void launch_moe_prefill_gather(const void* src, const int* perm_src, void* x_perm,
                               int P, int dim, cudaStream_t stream) {
    dim3 grid(P, (dim + 255) / 256);
    moe_gather_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const bf16*>(src), perm_src, reinterpret_cast<bf16*>(x_perm), P, dim);
}

int moe_prefill_grouped_maxtiles(int P, int E) { return (P + GM - 1) / GM + E; }

void launch_moe_prefill_build_sched(const int* offsets, int* tile_expert, int* tile_row0,
                                    int* d_ntiles, int E, cudaStream_t stream) {
    moe_sched_kernel<<<1, 32, 0, stream>>>(offsets, tile_expert, tile_row0, d_ntiles, E, GM);
}

void launch_moe_prefill_grouped_gemm(const void* A, const void* W, const int* offsets,
                                     const int* tile_expert, const int* tile_row0, const int* d_ntiles,
                                     void* C, int P, int E, int Nout, int K, cudaStream_t stream) {
    const int maxT = moe_prefill_grouped_maxtiles(P, E);
    dim3 grid((Nout + GN - 1) / GN, maxT);
    moe_grouped_gemm_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const bf16*>(A), reinterpret_cast<const bf16*>(W), offsets,
        tile_expert, tile_row0, d_ntiles, reinterpret_cast<bf16*>(C), Nout, K);
}

void launch_moe_prefill_swiglu(const void* gate, const void* up, void* h, long n, cudaStream_t stream) {
    moe_swiglu_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<const bf16*>(gate), reinterpret_cast<const bf16*>(up),
        reinterpret_cast<bf16*>(h), n);
}

void launch_moe_prefill_scatter_weighted(const void* y, const int* perm_src, const float* perm_w,
                                         void* out, int P, int H, cudaStream_t stream) {
    dim3 grid(P, (H + 255) / 256);
    moe_scatter_weighted_kernel<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const bf16*>(y), perm_src, perm_w, reinterpret_cast<float*>(out), P, H);
}

}} // namespace sparkinfer::kernels
