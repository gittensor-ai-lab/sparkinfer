// The row-batched dense down projection must be BIT-IDENTICAL, per row, to the per-token kernel it
// replaces. That is the whole basis on which a packed continuous-batch step is allowed to share one
// weight read across its rows: the batch changes how the weights are fetched, never what any row
// computes. This walks every batch width the dispatch instantiates -- 2..16, including the 9..16
// the cap used to decline -- against a per-token reference built one row at a time.
//
// Muse Glimmer's dense FFN shape (6656 x 19968, top_k == 1) is the one the arm is gated to, and the
// down projection stays Q4_K there whichever block format the gate/up calibration left behind, so
// both gate/up quants are run: the down result has to match under either.

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
unsigned int rng = 0x9e3779b9u;
unsigned char next_byte() {
    rng = rng * 1664525u + 1013904223u;
    return (unsigned char)(rng >> 24);
}

bool fill_dev(void* p, size_t bytes) {
    std::vector<unsigned char> h(bytes);
    for (size_t i = 0; i < bytes; ++i) h[i] = next_byte();
    return cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

// One FFN call, writing `rows` rows of the bf16 down output.
bool run(const void* in, const void* g, const void* u, const void* d, int qt,
         const int* ids, const float* wts, void* out, float* hs, float* os, int rows) {
    sparkinfer::kernels::launch_moe_expert_ffn_q4k(in, g, u, d, qt, qt, 12, ids, wts, out,
                                                   hs, os, rows, 1, H, F);
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool check(int qt, const char* label) {
    const size_t gu_bytes = (qt == sparkinfer::kernels::SI_QTYPE_Q3A) ? (size_t)F * (H / 256) * 112
                                                                     : (size_t)F * (H / 256) * 144;
    const size_t dn_bytes = (size_t)H * (F / 256) * 144;          // down is Q4_K
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
    // Per-token reference: one call per row, each reading that row's slice of `in`. A single-row
    // call cannot reach the rows kernel (it is gated at num_tokens >= 2), so this is the per-token
    // split-K down by construction.
    std::vector<__nv_bfloat16> ref((size_t)MAXROWS * H), got((size_t)MAXROWS * H);
    for (int r = 0; ok && r < MAXROWS; ++r) {
        ok = run((const __nv_bfloat16*)in + (size_t)r * H, g, u, d, qt, ids, wts, out, hs, os, 1) &&
             cudaMemcpy(ref.data() + (size_t)r * H, out, (size_t)H * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    for (int m = 2; ok && m <= MAXROWS; ++m) {
        ok = run(in, g, u, d, qt, ids, wts, out, hs, os, m) &&
             cudaMemcpy(got.data(), out, (size_t)m * H * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
        if (!ok) break;
        for (int r = 0; r < m; ++r) {
            if (std::memcmp(ref.data() + (size_t)r * H, got.data() + (size_t)r * H,
                            (size_t)H * sizeof(__nv_bfloat16)) != 0) {
                std::printf("[FAIL] %s m=%d row=%d differs from the per-token kernel\n",
                            label, m, r);
                ok = false;
                break;
            }
        }
    }
    if (ok) std::printf("[PASS] %s: down rows kernel bit-identical at m=2..%d\n", label, MAXROWS);
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
    const bool ok = check(sparkinfer::kernels::SI_QTYPE_Q3A, "down Q4_K, fed by Q3_A gate/up") & check(12, "down Q4_K, fed by Q4_K gate/up");
    if (!ok) return 1;
    std::printf("[PASS] down_rows_gpu_test\n");
    return 0;
}
