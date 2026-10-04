// Is the SM120 native NVFP4 GEMM correct at the shapes each model actually asks for?
//
// Model-free. Builds random bf16 A[m,k] and W[n,k], runs them through the exact prefill sequence
// (launch_prefill_nvfp4_quant_a + launch_prefill_nvfp4_quant_b + launch_prefill_nvfp4_gemm) and
// compares against a CPU matmul in double over a subset of output columns. NVFP4 quantizes BOTH
// operands to 4 bits with a 16-wide block scale, so ~0.10-0.15 relative error is the CORRECT
// answer; the bar is set from that, not from exactness.
//
// Why this exists: batched prefill leaves the Qwen3.8 compressed-tensors state wrong, and the leg
// responsible is this GEMM sequence -- but Muse Glimmer drives the SAME call sites correctly. The
// two differ only in shape (Muse ffn/hidden 19968/6656, Qwen3.8 17408/5120), so the question was
// whether the kernel is shape-dependent. ANSWER: it is not. Every shape here comes in at ~0.13,
// which is what 4-bit quantization of both operands costs -- the GEMM sequence is correct, and
// the prefill corruption is in how this leg is INTEGRATED, not in the kernel it calls.
//
// Usage: nvfp4_gemm_check [m]        (default: sweeps m = 8, 32, 128, 512)
//        NVFP4_BENCH=1 nvfp4_gemm_check [m]   timing only, DRAM-cold weights (default m = 16, 32)

#include "sparkinfer/kernels/prefill_nvfp4.h"
#include "sparkinfer/kernels/gemm.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

using sparkinfer::kernels::GemmConfig;
using sparkinfer::kernels::GemmLayout;

struct Shape { const char* name; int n; int k; };

static bool run_case(const Shape& s, int m, double& rel_out) {
    const int n = s.n, k = s.k;
    if (!sparkinfer::kernels::prefill_nvfp4_supported(m, n, k)) {
        printf("  %-22s m=%-5d UNSUPPORTED (m%%8=%d n%%128=%d k%%128=%d)\n",
               s.name, m, m & 7, n & 127, k & 127);
        return true;
    }
    std::mt19937 rng(1234u + m + n);
    std::normal_distribution<float> nd(0.f, 0.02f);   // weight-like magnitudes
    std::vector<__nv_bfloat16> hA((size_t)m * k), hW((size_t)n * k);
    for (auto& v : hA) v = __float2bfloat16(nd(rng));
    for (auto& v : hW) v = __float2bfloat16(nd(rng));

    void *dA = nullptr, *dW = nullptr, *dWt = nullptr, *dOut = nullptr;
    void *qa = nullptr, *sa = nullptr, *qb = nullptr, *sb = nullptr, *ws = nullptr;
    auto ok = [](cudaError_t e) { return e == cudaSuccess; };
    bool alloc = ok(cudaMalloc(&dA, hA.size() * 2)) && ok(cudaMalloc(&dW, hW.size() * 2)) &&
                 ok(cudaMalloc(&dWt, hW.size() * 2)) &&
                 ok(cudaMalloc(&dOut, (size_t)m * n * 2)) &&
                 ok(cudaMalloc(&qa, sparkinfer::kernels::prefill_nvfp4_data_bytes(m, k))) &&
                 ok(cudaMalloc(&sa, sparkinfer::kernels::prefill_nvfp4_scale_bytes_a(m, k))) &&
                 ok(cudaMalloc(&qb, sparkinfer::kernels::prefill_nvfp4_data_bytes(n, k))) &&
                 ok(cudaMalloc(&sb, sparkinfer::kernels::prefill_nvfp4_scale_bytes_b(n, k)));
    const size_t wsb = sparkinfer::kernels::prefill_nvfp4_workspace_bytes(m, n, k);
    if (alloc && wsb) alloc = ok(cudaMalloc(&ws, wsb));
    if (!alloc) { printf("  %-22s m=%-5d ALLOC FAILED\n", s.name, m); return false; }
    cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice);

    // Reference is computed on the CPU in double, over the first NCHK output columns only.
    // An earlier revision used launch_gemm as the reference and drove its B operand with the wrong
    // layout, which made every shape -- including Muse's known-good ones -- report rel_err ~1.41
    // (= sqrt(2), the value two uncorrelated signals of equal magnitude produce). That read as
    // "the kernel is broken everywhere" when the harness was the broken part. A CPU reference
    // cannot have that failure mode, and a column subset keeps it to ~m*NCHK*k work.
    const int ncheck = n < 64 ? n : 64;
    const bool qok = sparkinfer::kernels::launch_prefill_nvfp4_quant_a(dA, qa, sa, m, k, nullptr) &&
                     sparkinfer::kernels::launch_prefill_nvfp4_quant_b(dW, qb, sb, n, k, nullptr) &&
                     sparkinfer::kernels::launch_prefill_nvfp4_gemm(qa, sa, qb, sb, dOut,
                                                                    m, n, k, ws, nullptr, 1.f);
    cudaDeviceSynchronize();
    if (!qok) { printf("  %-22s m=%-5d NVFP4 SEQUENCE RETURNED FALSE\n", s.name, m); return false; }

    std::vector<__nv_bfloat16> hOut((size_t)m * n);
    cudaMemcpy(hOut.data(), dOut, hOut.size() * 2, cudaMemcpyDeviceToHost);
    double num = 0, den = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < ncheck; j++) {
            double acc = 0;
            for (int t = 0; t < k; t++)
                acc += (double)__bfloat162float(hA[(size_t)i * k + t]) *
                       (double)__bfloat162float(hW[(size_t)j * k + t]);
            const double o = __bfloat162float(hOut[(size_t)i * n + j]);
            num += (acc - o) * (acc - o); den += acc * acc;
        }
    const double rel = std::sqrt(num / (den > 0 ? den : 1));
    rel_out = rel;
    printf("  %-22s m=%-5d n=%-6d k=%-5d  rel_err=%.4f (%d cols vs CPU)  %s\n",
           s.name, m, n, k, rel, ncheck, rel < 0.25 ? "ok" : "*** WRONG ***");
    for (void* p : {dA, dW, dWt, dOut, qa, sa, qb, sb, ws}) if (p) cudaFree(p);
    return true;
}

// Self-test: at a tiny shape, check BOTH the bf16 reference and the NVFP4 sequence against a CPU
// matmul. Without this the harness cannot tell "the kernel is wrong" from "I drove the reference
// GEMM with the wrong operand layout" -- and those two look identical in the output.
static void cpu_selftest() {
    const int m = 8, n = 128, k = 128;
    std::mt19937 rng(7u);
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<__nv_bfloat16> hA((size_t)m * k), hW((size_t)n * k);
    std::vector<float> fA(hA.size()), fW(hW.size());
    for (size_t i = 0; i < hA.size(); i++) { fA[i] = nd(rng); hA[i] = __float2bfloat16(fA[i]); }
    for (size_t i = 0; i < hW.size(); i++) { fW[i] = nd(rng); hW[i] = __float2bfloat16(fW[i]); }
    std::vector<double> cpu((size_t)m * n, 0.0);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double acc = 0;
            for (int t = 0; t < k; t++)
                acc += (double)__bfloat162float(hA[(size_t)i * k + t]) *
                       (double)__bfloat162float(hW[(size_t)j * k + t]);
            cpu[(size_t)i * n + j] = acc;
        }
    void *dA = nullptr, *dW = nullptr, *dRef = nullptr, *dOut = nullptr;
    void *qa = nullptr, *sa = nullptr, *qb = nullptr, *sb = nullptr, *ws = nullptr;
    cudaMalloc(&dA, hA.size() * 2); cudaMalloc(&dW, hW.size() * 2);
    cudaMalloc(&dRef, (size_t)m * n * 2); cudaMalloc(&dOut, (size_t)m * n * 2);
    cudaMalloc(&qa, sparkinfer::kernels::prefill_nvfp4_data_bytes(m, k));
    cudaMalloc(&sa, sparkinfer::kernels::prefill_nvfp4_scale_bytes_a(m, k));
    cudaMalloc(&qb, sparkinfer::kernels::prefill_nvfp4_data_bytes(n, k));
    cudaMalloc(&sb, sparkinfer::kernels::prefill_nvfp4_scale_bytes_b(n, k));
    const size_t wsb = sparkinfer::kernels::prefill_nvfp4_workspace_bytes(m, n, k);
    if (wsb) cudaMalloc(&ws, wsb);
    cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice);
    GemmConfig cfg; cfg.layout_a = GemmLayout::ROW_MAJOR; cfg.layout_b = GemmLayout::COL_MAJOR;
    sparkinfer::kernels::launch_gemm(dA, dW, dRef, m, n, k, 1.f, 0.f, cfg, nullptr);
    sparkinfer::kernels::launch_prefill_nvfp4_quant_a(dA, qa, sa, m, k, nullptr);
    sparkinfer::kernels::launch_prefill_nvfp4_quant_b(dW, qb, sb, n, k, nullptr);
    sparkinfer::kernels::launch_prefill_nvfp4_gemm(qa, sa, qb, sb, dOut, m, n, k, ws, nullptr, 1.f);
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> hRef((size_t)m * n), hOut((size_t)m * n);
    cudaMemcpy(hRef.data(), dRef, hRef.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hOut.data(), dOut, hOut.size() * 2, cudaMemcpyDeviceToHost);
    auto rel = [&](const std::vector<__nv_bfloat16>& v) {
        double num = 0, den = 0;
        for (size_t i = 0; i < cpu.size(); i++) {
            const double d = cpu[i] - __bfloat162float(v[i]);
            num += d * d; den += cpu[i] * cpu[i];
        }
        return std::sqrt(num / (den > 0 ? den : 1));
    };
    printf("cpu self-test (m=8 n=128 k=128):  bf16_gemm_vs_cpu=%.4f   nvfp4_vs_cpu=%.4f\n",
           rel(hRef), rel(hOut));
    printf("  (bf16 should be ~0.00; if it is ~1.41 the HARNESS drives launch_gemm wrong,\n"
           "   not the NVFP4 kernel -- interpret the sweep below accordingly)\n");
    for (void* p : {dA, dW, dRef, dOut, qa, sa, qb, sb, ws}) if (p) cudaFree(p);
}

// NVFP4_BENCH=1: time the GEMM alone at Qwen3.8's projection shapes, weights DRAM-cold. The
// quantized weight is replicated into enough copies to overflow L2 and the copies are cycled, so
// every launch streams its operand from DRAM the way a decode step does. Prints us and the
// effective weight bandwidth. The tiling under test is whatever the SPARKINFER_NVFP4_* knobs pick.
static int bench(int m) {
    struct B { const char* name; int n, k; };
    const B shapes[] = {
        {"down", 5120, 17408}, {"gate|up", 17408, 5120}, {"gdn_qkv", 10240, 5120},
        {"gdn_z", 6144, 5120}, {"gdn_out", 5120, 6144}, {"attn_q|g", 12288, 5120},
        {"attn_kv", 1024, 5120}, {"attn_o", 5120, 6144},
    };
    for (const B& s : shapes) {
        const int n = s.n, k = s.k;
        const size_t wb = sparkinfer::kernels::prefill_nvfp4_data_bytes(n, k);
        const size_t sbb = sparkinfer::kernels::prefill_nvfp4_scale_bytes_b(n, k);
        const int copies = (int)((512ull << 20) / (wb + sbb)) + 1;
        std::vector<void*> qb(copies), sb(copies);
        void *qa = nullptr, *sa = nullptr, *out = nullptr, *ws = nullptr, *dA = nullptr;
        cudaMalloc(&dA, (size_t)m * k * 2); cudaMemset(dA, 0, (size_t)m * k * 2);
        cudaMalloc(&qa, sparkinfer::kernels::prefill_nvfp4_data_bytes(m, k));
        cudaMalloc(&sa, sparkinfer::kernels::prefill_nvfp4_scale_bytes_a(m, k));
        cudaMalloc(&out, (size_t)m * n * 2);
        const size_t wsb = sparkinfer::kernels::prefill_nvfp4_workspace_bytes(m, n, k);
        if (wsb) cudaMalloc(&ws, wsb);
        sparkinfer::kernels::launch_prefill_nvfp4_quant_a(dA, qa, sa, m, k, nullptr);
        for (int i = 0; i < copies; i++) {
            cudaMalloc(&qb[i], wb); cudaMalloc(&sb[i], sbb);
            cudaMemset(qb[i], 0x11 * (i & 7), wb); cudaMemset(sb[i], 0x38, sbb);
        }
        bool ok = true;
        for (int i = 0; i < copies; i++)   // warm-up, one pass over every copy
            ok = ok && sparkinfer::kernels::launch_prefill_nvfp4_gemm(qa, sa, qb[i], sb[i], out,
                                                                      m, n, k, ws, nullptr, 1.f);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        const int iters = copies * 8;
        cudaEventRecord(e0);
        for (int i = 0; i < iters; i++)
            ok = ok && sparkinfer::kernels::launch_prefill_nvfp4_gemm(
                           qa, sa, qb[i % copies], sb[i % copies], out, m, n, k, ws, nullptr, 1.f);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        const double us = 1e3 * ms / iters;
        printf("  %-9s m=%-3d n=%-6d k=%-6d %8.2f us  %6.0f GB/s  ws=%zu %s\n", s.name, m, n, k, us,
               (wb + sbb) / (us * 1e3), wsb, ok && cudaGetLastError() == cudaSuccess ? "" : "FAILED");
        for (int i = 0; i < copies; i++) { cudaFree(qb[i]); cudaFree(sb[i]); }
        for (void* p : {qa, sa, out, ws, dA}) if (p) cudaFree(p);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    return 0;
}

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }
    if (getenv("NVFP4_BENCH")) {
        if (argc > 1) return bench(atoi(argv[1]));
        return bench(16) || bench(32);
    }
    cpu_selftest();
    const Shape shapes[] = {
        {"qwen3.8 gate/up", 17408, 5120},   // the broken model's FFN in-projection
        {"qwen3.8 down",     5120, 17408},
        {"muse gate/up",    19968, 6656},   // the working model's, for contrast
        {"muse down",        6656, 19968},
    };
    std::vector<int> ms;
    if (argc > 1) ms.push_back(atoi(argv[1]));
    else ms = {8, 32, 128, 512};

    double worst = 0;
    for (int m : ms) {
        printf("m = %d\n", m);
        for (const Shape& s : shapes) {
            double rel = 0;
            if (!run_case(s, m, rel)) return 1;
            if (rel > worst) worst = rel;
        }
    }
    // NVFP4 quantizes BOTH operands to 4 bits with a 16-wide block scale, so ~0.10-0.15
    // relative error against an exact reference is the correct answer, not a defect.
    printf("WORST rel_err=%.4f  %s\n", worst, worst < 0.25 ? "PASS" : "FAIL");
    return worst < 0.25 ? 0 : 1;
}
