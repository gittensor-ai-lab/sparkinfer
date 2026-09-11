// The row-batched dense gate/up kernels must be BIT-IDENTICAL, per row, to the per-token kernels
// they replace. That is the whole basis on which a packed continuous-batch step is allowed to
// share one weight read across its rows: the batch changes how the weights are fetched, never
// what any row computes.
//
// Both of Muse Glimmer's block formats are covered, because its 52 dense layers use both -- the
// Q3_A requantizer takes 42 of them and the calibration leaves 10 native Q4_K.
//
// SPARKINFER_MUSE_Q3A_SPEC=0 is set before the first launch so the num_tokens == 1 reference runs
// the GENERIC per-token kernel rather than the one-token specialisation. The comparison is then
// exactly the claim being made -- rows kernel vs per-token kernel -- and does not also depend on
// the specialisation matching, which is a separate (and separately documented) equality.

#include "sparkinfer/kernels/moe.h"
#include "sparkinfer/kernels/qtype.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int H = 6656, F = 19968;     // Muse Glimmer's dense FFN
constexpr int MAXROWS = 16;

// Deterministic byte fill. Any byte pattern is a structurally valid Q3_A / Q4_K super-block -- the
// formats have no reserved encodings -- so random bytes exercise the decode paths without needing
// a quantizer here.
unsigned int rng = 0x1234567u;
unsigned char next_byte() {
    rng = rng * 1664525u + 1013904223u;
    return (unsigned char)(rng >> 24);
}

bool fill_dev(void* p, size_t bytes) {
    std::vector<unsigned char> h(bytes);
    for (size_t i = 0; i < bytes; ++i) h[i] = next_byte();
    return cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

// One FFN call, writing h_scratch for `rows` tokens.
bool run(const void* in, const void* g, const void* u, const void* d, int qt,
         const int* ids, const float* wts, void* out, float* hs, float* os, int rows) {
    sparkinfer::kernels::launch_moe_expert_ffn_q4k(in, g, u, d, qt, qt, 12, ids, wts, out,
                                                   hs, os, rows, 1, H, F);
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool check(int qt, const char* label) {
    const size_t gu_bytes = (qt == sparkinfer::kernels::SI_QTYPE_Q3A) ? (size_t)F * (H / 256) * 112
                                                 : (size_t)F * (H / 256) * 144;
    const size_t dn_bytes = (size_t)H * (F / 256) * 144;          // down stays Q4_K
    void *g = nullptr, *u = nullptr, *d = nullptr, *in = nullptr, *out = nullptr;
    float *hs = nullptr, *os = nullptr;
    int* ids = nullptr; float* wts = nullptr;
    bool ok = cudaMalloc(&g, gu_bytes) == cudaSuccess &&
              cudaMalloc(&u, gu_bytes) == cudaSuccess &&
              cudaMalloc(&d, dn_bytes) == cudaSuccess &&
              cudaMalloc(&in, (size_t)MAXROWS * H * sizeof(__nv_bfloat16)) == cudaSuccess &&
              cudaMalloc(&out, (size_t)MAXROWS * H * sizeof(__nv_bfloat16)) == cudaSuccess &&
              cudaMalloc(&hs, (size_t)MAXROWS * F * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&os, (size_t)MAXROWS * F * sizeof(float)) == cudaSuccess &&
              cudaMalloc(&ids, MAXROWS * sizeof(int)) == cudaSuccess &&
              cudaMalloc(&wts, MAXROWS * sizeof(float)) == cudaSuccess;
    if (ok) ok = fill_dev(g, gu_bytes) && fill_dev(u, gu_bytes) && fill_dev(d, dn_bytes) &&
                 fill_dev(in, (size_t)MAXROWS * H * sizeof(__nv_bfloat16));
    if (ok) {
        std::vector<int> hid(MAXROWS, 0);
        std::vector<float> hw(MAXROWS, 1.f);
        ok = cudaMemcpy(ids, hid.data(), MAXROWS * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemcpy(wts, hw.data(), MAXROWS * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    }
    // Per-token reference: one call per row, each reading that row's slice of `in`.
    std::vector<float> ref((size_t)MAXROWS * F), got((size_t)MAXROWS * F);
    for (int r = 0; ok && r < MAXROWS; ++r) {
        ok = run((const __nv_bfloat16*)in + (size_t)r * H, g, u, d, qt, ids, wts, out, hs, os, 1) &&
             cudaMemcpy(ref.data() + (size_t)r * F, hs, (size_t)F * sizeof(float),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    // Every batch width the dispatch instantiates, against the same reference.
    for (int m = 2; ok && m <= MAXROWS; ++m) {
        ok = run(in, g, u, d, qt, ids, wts, out, hs, os, m) &&
             cudaMemcpy(got.data(), hs, (size_t)m * F * sizeof(float),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
        if (!ok) break;
        for (int r = 0; r < m; ++r) {
            if (std::memcmp(ref.data() + (size_t)r * F, got.data() + (size_t)r * F,
                            (size_t)F * sizeof(float)) != 0) {
                std::printf("[FAIL] %s m=%d row=%d differs from the per-token kernel\n",
                            label, m, r);
                ok = false;
                break;
            }
        }
    }
    if (ok) std::printf("[PASS] %s: rows kernel bit-identical at m=2..%d\n", label, MAXROWS);
    cudaFree(g); cudaFree(u); cudaFree(d); cudaFree(in); cudaFree(out);
    cudaFree(hs); cudaFree(os); cudaFree(ids); cudaFree(wts);
    return ok;
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("[SKIP] no GPU\n");
        return 77;
    }
    setenv("SPARKINFER_MUSE_Q3A_SPEC", "0", 1);   // before the launcher caches its statics
    const bool ok = check(sparkinfer::kernels::SI_QTYPE_Q3A, "Q3_A gate/up") & check(12, "Q4_K gate/up");
    if (!ok) return 1;
    std::printf("[PASS] gate_up_rows_gpu_test\n");
    return 0;
}
