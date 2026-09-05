// Runs the REAL batched-prefill QK-norm+RoPE kernel with MRoPE positions and dumps its rotated Q,
// so bench/scripts/mrope_kernel_check.py can diff it against transformers' own rotary embedding.
//
// The position-id check (mrope_ref_check.py) proves we COMPUTE what the reference computes. It
// says nothing about whether the kernel APPLIES those positions the way the reference does --
// which is the half that contains the logic written from scratch here (pf_mrope_axis, the
// interleaved axis selection). This dump exists to close exactly that gap.
#include "sparkinfer/kernels/prefill.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

bool cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); return false; }
    return true;
}

float bf16_to_f(__nv_bfloat16 x) { return __bfloat162float(x); }
__nv_bfloat16 f_to_bf16(float x) { return __float2bfloat16(x); }

}  // namespace

int main(int argc, char** argv) {
    const char* out_path = argc > 1 ? argv[1] : "mrope_kernel.json";

    // Qwen3.8-27B's real attention geometry, so the exercised shapes are the shipped ones.
    const int N = 6;               // tokens
    const int n_q = 2, n_kv = 1;   // enough heads to catch a head-indexing error, small enough to print
    const int head_dim = 256;
    const int rotary_dim = 64;     // head_dim * partial_rotary_factor (0.25)
    const int sec_h = 11, sec_w = 10;
    const float theta = 10000000.f, eps = 1e-6f;
    const int block_size = 16, mbs = 8;

    // Deterministic, non-degenerate inputs. A constant Q would hide an axis-selection bug because
    // every frequency would rotate identical values.
    std::vector<__nv_bfloat16> hq((size_t)N * n_q * head_dim), hk((size_t)N * n_kv * head_dim),
                               hv((size_t)N * n_kv * head_dim);
    for (size_t i = 0; i < hq.size(); i++) hq[i] = f_to_bf16(std::sin(0.017f * (float)i) * 1.3f);
    for (size_t i = 0; i < hk.size(); i++) hk[i] = f_to_bf16(std::cos(0.023f * (float)i) * 0.9f);
    for (size_t i = 0; i < hv.size(); i++) hv[i] = f_to_bf16(0.01f * (float)i);
    std::vector<__nv_bfloat16> hqw(head_dim), hkw(head_dim);
    for (int i = 0; i < head_dim; i++) {
        hqw[i] = f_to_bf16(1.0f + 0.001f * (float)i);   // non-unit, so a dropped norm weight shows
        hkw[i] = f_to_bf16(1.0f - 0.001f * (float)i);
    }

    // Positions where all three axes DIFFER -- the only regime in which MRoPE is distinguishable
    // from 1D RoPE at all. Token 0 and 5 are text (t=h=w); 1..4 are a 2x2 vision span.
    const std::vector<int> mpos = {
        3, 3, 3,      // text
        4, 4, 4,      // vision (t, h, w) for a 2x2 merged grid starting at 4
        4, 4, 5,
        4, 5, 4,
        4, 5, 5,
        6, 6, 6,      // text resuming after max(h,w)=2
    };

    std::vector<int> hbt(mbs);
    for (int i = 0; i < mbs; i++) hbt[i] = i;

    void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dqw = nullptr, *dkw = nullptr;
    void *dkp = nullptr, *dvp = nullptr;
    int *dbt = nullptr, *dmp = nullptr;
    const size_t pool = (size_t)mbs * block_size * n_kv * head_dim * sizeof(__nv_bfloat16);
    if (!cu(cudaMalloc(&dq, hq.size() * 2), "q") || !cu(cudaMalloc(&dk, hk.size() * 2), "k") ||
        !cu(cudaMalloc(&dv, hv.size() * 2), "v") || !cu(cudaMalloc(&dqw, head_dim * 2), "qw") ||
        !cu(cudaMalloc(&dkw, head_dim * 2), "kw") || !cu(cudaMalloc(&dkp, pool), "kp") ||
        !cu(cudaMalloc(&dvp, pool), "vp") ||
        !cu(cudaMalloc((void**)&dbt, hbt.size() * sizeof(int)), "bt") ||
        !cu(cudaMalloc((void**)&dmp, mpos.size() * sizeof(int)), "mp")) return 1;

    if (!cu(cudaMemcpy(dq, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice), "cp q") ||
        !cu(cudaMemcpy(dk, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice), "cp k") ||
        !cu(cudaMemcpy(dv, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice), "cp v") ||
        !cu(cudaMemcpy(dqw, hqw.data(), head_dim * 2, cudaMemcpyHostToDevice), "cp qw") ||
        !cu(cudaMemcpy(dkw, hkw.data(), head_dim * 2, cudaMemcpyHostToDevice), "cp kw") ||
        !cu(cudaMemcpy(dbt, hbt.data(), hbt.size() * sizeof(int), cudaMemcpyHostToDevice), "cp bt") ||
        !cu(cudaMemcpy(dmp, mpos.data(), mpos.size() * sizeof(int), cudaMemcpyHostToDevice), "cp mp"))
        return 1;

    sparkinfer::kernels::launch_prefill_qknorm_rope_kv_bf16(
        dq, dk, dv, dqw, dkw, dkp, dvp, dbt, N, n_q, n_kv, head_dim,
        rotary_dim, theta, eps, block_size, mbs, nullptr, /*pos0=*/0,
        dmp, sec_h, sec_w);
    if (!cu(cudaDeviceSynchronize(), "sync")) return 1;

    std::vector<__nv_bfloat16> gq(hq.size());
    if (!cu(cudaMemcpy(gq.data(), dq, gq.size() * 2, cudaMemcpyDeviceToHost), "read q")) return 1;

    FILE* f = std::fopen(out_path, "w");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", out_path); return 1; }
    std::fprintf(f, "{\n  \"n_tokens\": %d, \"n_q_heads\": %d, \"head_dim\": %d,\n", N, n_q, head_dim);
    std::fprintf(f, "  \"rotary_dim\": %d, \"theta\": %.1f, \"eps\": %g,\n", rotary_dim, theta, eps);
    std::fprintf(f, "  \"mrope_section\": [%d, %d, %d],\n", 32 - sec_h - sec_w, sec_h, sec_w);
    auto arr = [&](const char* name, const std::vector<__nv_bfloat16>& v, bool last) {
        std::fprintf(f, "  \"%s\": [", name);
        for (size_t i = 0; i < v.size(); i++) std::fprintf(f, "%s%.9g", i ? "," : "", bf16_to_f(v[i]));
        std::fprintf(f, "]%s\n", last ? "" : ",");
    };
    std::fprintf(f, "  \"positions\": [");
    for (size_t i = 0; i < mpos.size(); i++) std::fprintf(f, "%s%d", i ? "," : "", mpos[i]);
    std::fprintf(f, "],\n");
    arr("q_in", hq, false);
    arr("q_norm_w", hqw, false);
    arr("q_out", gq, true);
    std::fprintf(f, "}\n");
    std::fclose(f);
    std::printf("wrote %s (%d tokens x %d heads x %d dims)\n", out_path, N, n_q, head_dim);
    return 0;
}
