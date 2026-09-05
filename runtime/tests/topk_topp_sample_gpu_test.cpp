// GPU statistical + ordering correctness test for launch_topk_topp_mask (top_k/top_p nucleus
// truncation) and its interaction with the existing launch_temperature_sample/launch_argmax
// (Gumbel-max sampling). Calls the kernels directly against small synthetic logits buffers -- no
// model needed. Mirrors temperature_sample_gpu_test.cpp's structure and self-skip-on-no-device
// pattern.
#include "sparkinfer/kernels/fused.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

using sparkinfer::kernels::launch_topk_topp_mask;
using sparkinfer::kernels::launch_temperature_sample;
using sparkinfer::kernels::launch_argmax;
using sparkinfer::kernels::launch_vocab_iota_init;
using sparkinfer::kernels::topk_sort_temp_storage_bytes;
using sparkinfer::kernels::topk_scan_temp_storage_bytes;

// Owns the fixed-address, vocab-sized scratch launch_topk_topp_mask needs -- mirrors the
// load-time-only allocation qwen35.cpp's Impl does once per model load.
struct Scratch {
    int vocab = 0;
    int* vocab_iota = nullptr;
    float* sorted_logits = nullptr;
    int* sorted_idx = nullptr;
    float* topk_exp = nullptr;
    float* topk_cumsum = nullptr;
    void* sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    void* scan_temp = nullptr;
    size_t scan_temp_bytes = 0;
    int* d_top_k = nullptr;
    float* d_top_p = nullptr;
    int* rank_by_id = nullptr;

    explicit Scratch(int v) : vocab(v) {
        cudaMalloc(&vocab_iota, vocab * sizeof(int));
        cudaMalloc(&sorted_logits, vocab * sizeof(float));
        cudaMalloc(&sorted_idx, vocab * sizeof(int));
        cudaMalloc(&topk_exp, vocab * sizeof(float));
        cudaMalloc(&topk_cumsum, vocab * sizeof(float));
        cudaMalloc(&d_top_k, sizeof(int));
        cudaMalloc(&d_top_p, sizeof(float));
        cudaMalloc(&rank_by_id, vocab * sizeof(int));
        launch_vocab_iota_init(vocab_iota, vocab);
        sort_temp_bytes = topk_sort_temp_storage_bytes(vocab);
        scan_temp_bytes = topk_scan_temp_storage_bytes(vocab);
        cudaMalloc(&sort_temp, sort_temp_bytes);
        cudaMalloc(&scan_temp, scan_temp_bytes);
        cudaDeviceSynchronize();
    }
    ~Scratch() {
        cudaFree(vocab_iota); cudaFree(sorted_logits); cudaFree(sorted_idx);
        cudaFree(topk_exp); cudaFree(topk_cumsum);
        cudaFree(sort_temp); cudaFree(scan_temp);
        cudaFree(d_top_k); cudaFree(d_top_p);
        cudaFree(rank_by_id);
    }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
};

// Runs the mask in place on a host logits vector; returns the masked result (host copy).
std::vector<float> masked_logits(Scratch& s, std::vector<float> logits, int top_k, float top_p) {
    const int vocab = (int)logits.size();
    float* d_logits = nullptr;
    cudaMalloc(&d_logits, vocab * sizeof(float));
    cudaMemcpy(d_logits, logits.data(), vocab * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_top_k, &top_k, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_top_p, &top_p, sizeof(float), cudaMemcpyHostToDevice);
    launch_topk_topp_mask(d_logits, vocab, s.vocab_iota, s.sorted_logits, s.sorted_idx,
                          s.topk_exp, s.topk_cumsum, s.sort_temp, s.sort_temp_bytes,
                          s.scan_temp, s.scan_temp_bytes, s.d_top_k, s.d_top_p, s.rank_by_id);
    cudaMemcpy(logits.data(), d_logits, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_logits);
    return logits;
}

// Full pipeline (mask -> Gumbel draw -> argmax), matching the real decode call-site order.
int sample_one(Scratch& s, const std::vector<float>& logits, int top_k, float top_p,
               float temperature, unsigned long long seed, unsigned long long step) {
    const int vocab = (int)logits.size();
    float* d_logits = nullptr; int* d_out = nullptr;
    float* d_temp = nullptr; unsigned long long *d_seed = nullptr, *d_step = nullptr;
    cudaMalloc(&d_logits, vocab * sizeof(float));
    cudaMalloc(&d_out, sizeof(int));
    cudaMalloc(&d_temp, sizeof(float));
    cudaMalloc(&d_seed, sizeof(unsigned long long));
    cudaMalloc(&d_step, sizeof(unsigned long long));
    cudaMemcpy(d_logits, logits.data(), vocab * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_top_k, &top_k, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_top_p, &top_p, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp, &temperature, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seed, &seed, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_step, &step, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    launch_topk_topp_mask(d_logits, vocab, s.vocab_iota, s.sorted_logits, s.sorted_idx,
                          s.topk_exp, s.topk_cumsum, s.sort_temp, s.sort_temp_bytes,
                          s.scan_temp, s.scan_temp_bytes, s.d_top_k, s.d_top_p, s.rank_by_id);
    launch_temperature_sample(d_logits, 1, vocab, d_temp, d_seed, d_step);
    launch_argmax(d_logits, d_out, 1, vocab);
    int out = -1;
    cudaMemcpy(&out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_logits); cudaFree(d_out); cudaFree(d_temp); cudaFree(d_seed); cudaFree(d_step);
    return out;
}

bool is_neg_inf(float v) { return v == -std::numeric_limits<float>::infinity(); }

// The indices that SURVIVED masking, in order.
//
// Stronger than spelling the survivors out elementwise (!is_neg_inf(out[0]) && is_neg_inf(out[1])
// && ...): that form asserts what the listed entries are, but not that nothing ELSE survived, so a
// mask that kept an extra token beyond the ones written down would still pass. Comparing the whole
// surviving SET catches that.
std::vector<int> finite_indices(const std::vector<float>& v) {
    std::vector<int> idx;
    for (size_t i = 0; i < v.size(); i++)
        if (!is_neg_inf(v[i])) idx.push_back((int)i);
    return idx;
}

bool test_top_k_1_always_greedy() {
    const std::vector<float> logits = {1.0f, 5.0f, 3.0f, 2.0f, -1.0f, 4.0f};
    Scratch s((int)logits.size());
    bool ok = true;
    for (unsigned long long seed : {1ull, 2ull, 99ull}) {
        for (float T : {0.0f, 0.5f, 1.0f, 1.5f}) {
            for (unsigned long long step = 0; step < 20; ++step) {
                const int out = sample_one(s, logits, /*top_k=*/1, /*top_p=*/1.0f, T, seed, step);
                if (out != 1) {  // index 1 (value 5.0) is the sole greedy winner
                    printf("FAIL: top_k=1 returned %d (expected 1) at seed=%llu T=%.2f step=%llu\n",
                           out, seed, T, step);
                    ok = false;
                }
            }
        }
    }
    if (ok) printf("[OK] top_k=1 always returns the greedy winner\n");
    return ok;
}

bool test_top_k_3_excludes_rest() {
    // Ranks by value: idx5=8 idx2=7 idx4=6 | idx1=5 idx0=4 idx3=3 idx6=2 idx7=1 (excluded, k=3)
    const std::vector<float> logits = {4.0f, 5.0f, 7.0f, 3.0f, 6.0f, 8.0f, 2.0f, 1.0f};
    Scratch s((int)logits.size());
    std::vector<int> counts(logits.size(), 0);
    const int N = 4000;
    for (int i = 0; i < N; ++i) {
        const int out = sample_one(s, logits, /*top_k=*/3, /*top_p=*/1.0f, /*T=*/1.0f,
                                   /*seed=*/777, (unsigned long long)i);
        counts[out]++;
    }
    bool ok = true;
    const std::vector<int> excluded = {0, 1, 3, 6, 7};
    for (int idx : excluded) {
        if (counts[idx] != 0) {
            printf("FAIL: excluded index %d got %d hits (expected 0)\n", idx, counts[idx]);
            ok = false;
        }
    }
    if (counts[2] + counts[4] + counts[5] != N) {
        printf("FAIL: survivors' counts don't sum to N\n");
        ok = false;
    }
    if (ok) printf("[OK] top_k=3 -- the 5 excluded indices get exactly 0 hits across %d draws\n", N);
    return ok;
}

bool test_top_p_hand_computed_cutoff() {
    // softmax(logits) with T=1 is exactly [0.4, 0.3, 0.2, 0.1] since sum(p_i)=1 and
    // logits_i = ln(p_i) => exp(logits_i)/sum(exp(logits_j)) = p_i/1 = p_i.
    const std::vector<float> logits = {(float)std::log(0.4), (float)std::log(0.3),
                                       (float)std::log(0.2), (float)std::log(0.1)};
    Scratch s((int)logits.size());

    // top_p=0.75: cumsum after sorting is [0.4, 0.7, 0.9, 1.0]. i=0 always kept. i=1:
    // cumsum[0]=0.4 < 0.75 -> keep. i=2: cumsum[1]=0.7 < 0.75 -> keep. i=3: cumsum[2]=0.9 < 0.75
    // -> false, excluded. Expected surviving set: {0,1,2}, excluded: {3}.
    {
        const auto out = masked_logits(s, logits, /*top_k=*/0, /*top_p=*/0.75f);
        bool ok = !is_neg_inf(out[0]) && !is_neg_inf(out[1]) && !is_neg_inf(out[2]) && is_neg_inf(out[3]);
        if (!ok) { printf("FAIL: top_p=0.75 surviving set mismatch\n"); return false; }
    }
    // top_p=0.35: i=0 always kept regardless (invariant). i=1: cumsum[0]=0.4 < 0.35 -> false,
    // excluded. Expected surviving set: {0} only.
    {
        const auto out = masked_logits(s, logits, /*top_k=*/0, /*top_p=*/0.35f);
        bool ok = !is_neg_inf(out[0]) && is_neg_inf(out[1]) && is_neg_inf(out[2]) && is_neg_inf(out[3]);
        if (!ok) { printf("FAIL: top_p=0.35 surviving set mismatch (rank-0 invariant)\n"); return false; }
    }
    // OpenAI permits top_p=0: it means the rank-0 token is the only survivor.
    {
        const auto out = masked_logits(s, logits, /*top_k=*/0, /*top_p=*/0.0f);
        const bool ok = finite_indices(out) == std::vector<int>({0});
        if (!ok) { printf("FAIL: top_p=0 must keep only rank 0\n"); return false; }
    }
    printf("[OK] top_p hand-computed cutoffs match exactly\n");
    return true;
}

bool test_combined_top_k_then_top_p_ordering() {
    // Probabilities (T=1, logits=ln(p)): [0.5, 0.2, 0.15, 0.1, 0.05]. top_k=3, top_p=0.8.
    //
    // This implementation (top_k first, top_p renormalized against the top-k survivors' own
    // mass): restrict to {0,1,2} (masses 0.5,0.2,0.15), total_topk = 0.85. Threshold =
    // 0.8*0.85 = 0.68. i=1: cumsum[0]=0.5 < 0.68 -> keep. i=2: cumsum[1]=0.7 < 0.68 -> false,
    // excluded. Expected survivors: {0,1}.
    //
    // The alternate (wrong) ordering -- top_p first against the FULL-vocab mass, then top_k --
    // would instead use threshold 0.8*1.0 = 0.8: i=1: 0.5<0.8 keep; i=2: 0.7<0.8 keep; then
    // top_k=3 keeps all three -> survivors {0,1,2}. This genuinely discriminates the two
    // orderings for this input, so this test would fail under the wrong ordering.
    const std::vector<float> logits = {(float)std::log(0.5), (float)std::log(0.2), (float)std::log(0.15),
                                       (float)std::log(0.1), (float)std::log(0.05)};
    Scratch s((int)logits.size());
    const auto out = masked_logits(s, logits, /*top_k=*/3, /*top_p=*/0.8f);
    const bool ok = !is_neg_inf(out[0]) && !is_neg_inf(out[1]) &&
                    is_neg_inf(out[2]) && is_neg_inf(out[3]) && is_neg_inf(out[4]);
    if (!ok) {
        printf("FAIL: combined top_k=3,top_p=0.8 surviving set mismatch (ordering not top_k-then-top_p)\n");
        return false;
    }
    printf("[OK] combined top_k-then-top_p matches the confirmed HF/vLLM ordering\n");
    return true;
}

bool test_temperature_zero_inertness() {
    const std::vector<float> logits = {1.0f, 5.0f, 3.0f, 2.0f, -1.0f, 4.0f};  // greedy winner: idx1
    Scratch s((int)logits.size());
    bool ok = true;
    const std::vector<std::pair<int, float>> combos = {
        {0, 1.0f}, {1, 1.0f}, {2, 1.0f}, {3, 0.9f}, {0, 0.5f}, {0, 0.01f}, {6, 1.0f},
    };
    for (auto [top_k, top_p] : combos) {
        for (unsigned long long seed : {1ull, 42ull}) {
            const int out = sample_one(s, logits, top_k, top_p, /*T=*/0.f, seed, /*step=*/0);
            if (out != 1) {
                printf("FAIL: temperature<=0 not inert for top_k=%d top_p=%.3f (got %d, expected 1)\n",
                       top_k, top_p, out);
                ok = false;
            }
        }
    }
    if (ok) printf("[OK] temperature<=0 is inert to top_k/top_p -- always matches plain greedy\n");
    return ok;
}

bool test_determinism() {
    const std::vector<float> logits = {1.0f, 3.0f, 2.0f, 0.5f, -1.0f, 2.5f};
    Scratch s((int)logits.size());
    const int a = sample_one(s, logits, 4, 0.8f, 0.7f, 555, 3);
    const int b = sample_one(s, logits, 4, 0.8f, 0.7f, 555, 3);
    const int c = sample_one(s, logits, 4, 0.8f, 0.7f, 555, 3);
    if (a != b || b != c) {
        printf("FAIL: same (seed,step,temperature,top_k,top_p,logits) gave different outcomes\n");
        return false;
    }
    printf("[OK] same (seed,step,temperature,top_k,top_p,logits) -> identical output, repeated\n");
    return true;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no CUDA device\n");
        return 0;
    }
    bool ok = true;
    ok = test_top_k_1_always_greedy() && ok;
    ok = test_top_k_3_excludes_rest() && ok;
    ok = test_top_p_hand_computed_cutoff() && ok;
    ok = test_combined_top_k_then_top_p_ordering() && ok;
    ok = test_temperature_zero_inertness() && ok;
    ok = test_determinism() && ok;
    if (!ok) return 1;
    printf("topk_topp_sample_gpu_test: OK\n");
    return 0;
}
