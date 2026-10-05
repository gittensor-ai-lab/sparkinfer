// A packed batch of more than eight rows runs its whole 8-row chunks of the bf16 row GEMV, and of
// the Q8_0 row MMVQ, as one launch (chunk on grid.y) instead of one launch per chunk. Every row must come out bit-identical
// to what an 8-row-or-narrower launch of it computes: the packed MoE router and the DFlash verify
// both rely on a row's sum not depending on how many rows share the launch.
#include "sparkinfer/kernels/gemm.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

unsigned int rng = 0x2468aceu;
float next_unit() {
    rng = rng * 1664525u + 1013904223u;
    return ((int)(rng >> 9) - (1 << 22)) / (float)(1 << 22);
}

template <typename T>
bool check(const char* label, int M, int N, int K) {
    std::vector<__nv_bfloat16> hx((size_t)M * K), hw((size_t)N * K);
    for (auto& v : hx) v = __float2bfloat16(next_unit());
    for (auto& v : hw) v = __float2bfloat16(next_unit() * 0.05f);
    void *x = nullptr, *w = nullptr;
    T *fused = nullptr, *chunked = nullptr;
    bool ok = cudaMalloc(&x, hx.size() * 2) == cudaSuccess && cudaMalloc(&w, hw.size() * 2) == cudaSuccess &&
              cudaMalloc(&fused, (size_t)M * N * sizeof(T)) == cudaSuccess &&
              cudaMalloc(&chunked, (size_t)M * N * sizeof(T)) == cudaSuccess &&
              cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess;
    auto launch = [&](const void* xp, T* yp, int m) {
        if constexpr (sizeof(T) == 4)
            return sparkinfer::kernels::launch_gemv_rows_f32(xp, w, reinterpret_cast<float*>(yp), m, N, K, nullptr);
        else
            return sparkinfer::kernels::launch_gemv_rows(xp, w, yp, m, N, K, nullptr);
    };
    ok = ok && launch(x, fused, M);
    // the reference: the same rows, at most eight to a launch, as a lone call of each chunk computes
    for (int r0 = 0; ok && r0 < M; r0 += 8)
        ok = launch(static_cast<const __nv_bfloat16*>(x) + (size_t)r0 * K, chunked + (size_t)r0 * N,
                    M - r0 < 8 ? M - r0 : 8);
    std::vector<T> a((size_t)M * N), b((size_t)M * N);
    ok = ok && cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(a.data(), fused, a.size() * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess &&
         cudaMemcpy(b.data(), chunked, b.size() * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
    if (ok && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) != 0) {
        std::printf("[FAIL] %s M=%d N=%d K=%d: fused launch differs from per-chunk launches\n", label, M, N, K);
        ok = false;
    } else if (ok) {
        std::printf("[PASS] %s M=%d N=%d K=%d bit-identical\n", label, M, N, K);
    } else {
        std::printf("[FAIL] %s M=%d: CUDA error\n", label, M);
    }
    cudaFree(x); cudaFree(w); cudaFree(fused); cudaFree(chunked);
    return ok;
}

// Q8_0 weights (34 B blocks: fp16 d + 32 int8) against Q8_1 activations (36 B: fp16 d, fp16 sum,
// 32 int8), through launch_mmvq_rows.
bool check_q80(int M, int N, int K) {
    const int nb = K / 32;
    std::vector<unsigned char> hw((size_t)N * nb * 34), hq((size_t)M * nb * 36);
    auto fill = [](std::vector<unsigned char>& v, size_t block, size_t qs_off) {
        for (size_t b = 0; b + block <= v.size(); b += block) {
            const __half d = __float2half(0.01f + 0.01f * std::fabs(next_unit()));
            std::memcpy(&v[b], &d, 2);
            if (qs_off == 4) { const __half z = __float2half(0.f); std::memcpy(&v[b + 2], &z, 2); }
            for (size_t i = 0; i < 32; ++i) v[b + qs_off + i] = (unsigned char)(int)(next_unit() * 127.f);
        }
    };
    fill(hw, 34, 2);
    fill(hq, 36, 4);
    void *w = nullptr, *q = nullptr, *fused = nullptr, *chunked = nullptr;
    const size_t ybytes = (size_t)M * N * sizeof(__nv_bfloat16);
    bool ok = cudaMalloc(&w, hw.size()) == cudaSuccess && cudaMalloc(&q, hq.size()) == cudaSuccess &&
              cudaMalloc(&fused, ybytes) == cudaSuccess && cudaMalloc(&chunked, ybytes) == cudaSuccess &&
              cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(q, hq.data(), hq.size(), cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && sparkinfer::kernels::launch_mmvq_rows(8, q, w, fused, M, N, K, nullptr);
    for (int r0 = 0; ok && r0 < M; r0 += 8)
        ok = sparkinfer::kernels::launch_mmvq_rows(8, static_cast<const unsigned char*>(q) + (size_t)r0 * nb * 36, w,
                                                   static_cast<__nv_bfloat16*>(chunked) + (size_t)r0 * N,
                                                   M - r0 < 8 ? M - r0 : 8, N, K, nullptr);
    std::vector<unsigned char> a(ybytes), b(ybytes);
    ok = ok && cudaDeviceSynchronize() == cudaSuccess &&
         cudaMemcpy(a.data(), fused, ybytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
         cudaMemcpy(b.data(), chunked, ybytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    if (ok && std::memcmp(a.data(), b.data(), ybytes) != 0) {
        std::printf("[FAIL] Q8_0 rows M=%d N=%d K=%d: fused launch differs from per-chunk launches\n", M, N, K);
        ok = false;
    } else if (ok) {
        std::printf("[PASS] Q8_0 rows M=%d N=%d K=%d bit-identical\n", M, N, K);
    } else {
        std::printf("[FAIL] Q8_0 rows M=%d: CUDA error or declined\n", M);
    }
    cudaFree(w); cudaFree(q); cudaFree(fused); cudaFree(chunked);
    return ok;
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("[SKIP] no GPU\n");
        return 77;
    }
    bool ok = true;
    ok &= check<float>("router fp32", 32, 256, 2048);          // Qwen3.6's router at 32 rows
    ok &= check<float>("router fp32", 17, 256, 2048);          // two fused chunks + a 1-row tail
    ok &= check<__nv_bfloat16>("bf16 out", 32, 1, 2048);       // the shared-expert gate scalar
    ok &= check<__nv_bfloat16>("bf16 out", 24, 64, 4096);
    ok &= check_q80(32, 512, 2048);                            // Qwen3.6's attention k / v at 32 rows
    ok &= check_q80(19, 512, 4096);
    if (!ok) return 1;
    std::printf("[PASS] gemv_rows_fuse_gpu_test\n");
    return 0;
}
