// Timing for the packed-decode Q4_K row projections (launch_mmvq_rows, qtype 12) at the shapes a
// 16-32 row continuous-batch step hands them, weights DRAM-cold.
//
// Model-free: synthetic Q4_K blocks (small fp16 d / dmin so nothing overflows) and synthetic Q8_1
// activations. The weight is replicated into enough copies to overflow L2 and the copies are cycled,
// so every launch streams its operand from DRAM as a decode step does. Prints us per launch and the
// effective weight bandwidth. Whatever arm launch_mmvq_rows picks (the SPARKINFER_MMVQ_MMA_* knobs
// apply) is what is timed.
//
// Usage: q4k_rows_bench [M ...]   (default: 16 32)

#include "sparkinfer/kernels/gemm.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>

namespace {
struct Shape { const char* name; int n, k; };

// One Q4_K super-block: half2 dm (d, dmin), 12 scale bytes, 128 quant bytes.
void fill_q4k(std::vector<unsigned char>& w, int n, int k, std::mt19937& rng) {
    const int nb = k / 256;
    w.assign((size_t)n * nb * 144, 0);
    std::uniform_int_distribution<int> byte(0, 255);
    for (size_t b = 0; b < (size_t)n * nb; ++b) {
        unsigned char* p = w.data() + b * 144;
        const __half d = __float2half(0.002f), dm = __float2half(0.001f);
        memcpy(p, &d, 2); memcpy(p + 2, &dm, 2);
        for (int i = 4; i < 144; ++i) p[i] = (unsigned char)byte(rng);
    }
}
// Q8_1 blocks: half2 ds (d, sum) then 32 int8.
void fill_q81(std::vector<unsigned char>& a, int m, int k, std::mt19937& rng) {
    const int nb = k / 32;
    a.assign((size_t)m * nb * 36, 0);
    std::uniform_int_distribution<int> q(-127, 127);
    for (size_t b = 0; b < (size_t)m * nb; ++b) {
        unsigned char* p = a.data() + b * 36;
        int s = 0;
        for (int i = 0; i < 32; ++i) { const int v = q(rng); p[4 + i] = (unsigned char)(signed char)v; s += v; }
        const __half d = __float2half(0.01f), sum = __float2half(0.01f * s);
        memcpy(p, &d, 2); memcpy(p + 2, &sum, 2);
    }
}
}  // namespace

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }
    std::vector<int> ms;
    for (int i = 1; i < argc; ++i) ms.push_back(atoi(argv[i]));
    if (ms.empty()) ms = {16, 32};
    // Qwen3.6-35B-A3B's dense projections: GDN qkv, z / out, attention q|gate and o.
    const Shape q36[] = {
        {"gdn_qkv", 8192, 2048}, {"gdn_z", 4096, 2048}, {"gdn_out", 2048, 4096},
        {"attn_qg", 8192, 2048}, {"attn_o", 2048, 4096}, {"n4096k2048", 4096, 2048},
    };
    // Q4K_BENCH_SHAPES=q38: Qwen3.8-27B's (FFN gate / up, down, GDN qkv / z / out, attention q|gate).
    const Shape q38[] = {
        {"ffn_gate", 17408, 5120}, {"ffn_down", 5120, 17408}, {"gdn_qkv", 10240, 5120},
        {"gdn_z", 6144, 5120}, {"gdn_out", 5120, 6144}, {"attn_qg", 12288, 5120},
    };
    const char* set = getenv("Q4K_BENCH_SHAPES");
    const bool use38 = set && strcmp(set, "q38") == 0;
    const Shape* shapes = use38 ? q38 : q36;
    const int nshapes = 6;
    std::mt19937 rng(42);
    cudaStream_t st; cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    for (int m : ms) {
        for (int si = 0; si < nshapes; ++si) {
            const Shape& s = shapes[si];
            std::vector<unsigned char> hw, ha;
            fill_q4k(hw, s.n, s.k, rng);
            fill_q81(ha, m, s.k, rng);
            const size_t wb = hw.size();
            const int copies = (int)((512ull << 20) / wb) + 1;
            std::vector<void*> dw(copies);
            for (int i = 0; i < copies; ++i) {
                cudaMalloc(&dw[i], wb);
                cudaMemcpy(dw[i], hw.data(), wb, cudaMemcpyHostToDevice);
            }
            void *da = nullptr, *dy = nullptr;
            cudaMalloc(&da, ha.size()); cudaMemcpy(da, ha.data(), ha.size(), cudaMemcpyHostToDevice);
            cudaMalloc(&dy, (size_t)m * s.n * 2);
            bool ok = true;
            // Q4K_BENCH_CHECK=1: the timed arm's output against the exact MMVQ rows kernel (rows
            // issued four at a time, under the tensor-core floor), as max |diff| / max |ref|.
            if (getenv("Q4K_BENCH_CHECK")) {
                void* dref = nullptr; cudaMalloc(&dref, (size_t)m * s.n * 2);
                const size_t row_q81 = (size_t)(s.k / 32) * 36;
                for (int r0 = 0; r0 < m; r0 += 4)
                    ok = ok && sparkinfer::kernels::launch_mmvq_rows(
                                   12, static_cast<char*>(da) + r0 * row_q81, dw[0],
                                   static_cast<char*>(dref) + (size_t)r0 * s.n * 2,
                                   m - r0 < 4 ? m - r0 : 4, s.n, s.k, st);
                ok = ok && sparkinfer::kernels::launch_mmvq_rows(12, da, dw[0], dy, m, s.n, s.k, st);
                cudaStreamSynchronize(st);
                std::vector<__half> hr((size_t)m * s.n), hy((size_t)m * s.n);
                std::vector<unsigned short> br((size_t)m * s.n), by((size_t)m * s.n);
                cudaMemcpy(br.data(), dref, br.size() * 2, cudaMemcpyDeviceToHost);
                cudaMemcpy(by.data(), dy, by.size() * 2, cudaMemcpyDeviceToHost);
                auto bf = [](unsigned short v) { unsigned u = (unsigned)v << 16; float f; memcpy(&f, &u, 4); return f; };
                double md = 0, mr = 0;
                for (size_t i = 0; i < br.size(); ++i) {
                    md = std::max(md, (double)fabsf(bf(br[i]) - bf(by[i])));
                    mr = std::max(mr, (double)fabsf(bf(br[i])));
                }
                printf("  %-11s M=%-3d check: max|diff| %.4g  max|ref| %.4g  rel %.2e\n", s.name, m, md, mr, md / (mr > 0 ? mr : 1));
                cudaFree(dref);
            }
            for (int i = 0; i < copies; ++i)
                ok = ok && sparkinfer::kernels::launch_mmvq_rows(12, da, dw[i], dy, m, s.n, s.k, st);
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            const int iters = copies * 8;
            cudaEventRecord(e0, st);
            for (int i = 0; i < iters; ++i)
                ok = ok && sparkinfer::kernels::launch_mmvq_rows(12, da, dw[i % copies], dy, m, s.n, s.k, st);
            cudaEventRecord(e1, st); cudaEventSynchronize(e1);
            float ms_ = 0; cudaEventElapsedTime(&ms_, e0, e1);
            const double us = 1e3 * ms_ / iters;
            printf("  %-11s M=%-3d N=%-5d K=%-5d %8.2f us  %6.0f GB/s %s\n", s.name, m, s.n, s.k, us,
                   wb / (us * 1e3), ok && cudaGetLastError() == cudaSuccess ? "" : "FAILED");
            for (void* p : dw) cudaFree(p);
            cudaFree(da); cudaFree(dy);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
    }
    cudaStreamDestroy(st);
    return 0;
}
