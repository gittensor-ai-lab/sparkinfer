// GPU statistical/correctness test for the logprobs/top_logprobs extraction primitives: the
// rank_by_id inverse permutation topk_topp_exp_kernel scatters, and extract_chosen_logit_kernel's
// chosen-token lookup. Mirrors topk_topp_sample_gpu_test.cpp's Scratch-struct-plus-direct-kernel-
// call structure, no model needed.
#include "sparkinfer/kernels/fused.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

using sparkinfer::kernels::launch_topk_topp_mask;
using sparkinfer::kernels::launch_extract_chosen_logit;
using sparkinfer::kernels::launch_vocab_iota_init;
using sparkinfer::kernels::topk_sort_temp_storage_bytes;
using sparkinfer::kernels::topk_scan_temp_storage_bytes;

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
    int* rank_by_id = nullptr;
    float* chosen_logit = nullptr;
    int* d_top_k = nullptr;
    float* d_top_p = nullptr;
    int* d_out_id = nullptr;

    explicit Scratch(int v) : vocab(v) {
        cudaMalloc(&vocab_iota, vocab * sizeof(int));
        cudaMalloc(&sorted_logits, vocab * sizeof(float));
        cudaMalloc(&sorted_idx, vocab * sizeof(int));
        cudaMalloc(&topk_exp, vocab * sizeof(float));
        cudaMalloc(&topk_cumsum, vocab * sizeof(float));
        cudaMalloc(&rank_by_id, vocab * sizeof(int));
        cudaMalloc(&chosen_logit, sizeof(float));
        cudaMalloc(&d_top_k, sizeof(int));
        cudaMalloc(&d_top_p, sizeof(float));
        cudaMalloc(&d_out_id, sizeof(int));
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
        cudaFree(rank_by_id); cudaFree(chosen_logit);
        cudaFree(d_top_k); cudaFree(d_top_p); cudaFree(d_out_id);
    }
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
};

struct PipelineResult {
    std::vector<int> sorted_idx;
    std::vector<float> sorted_logits;
    std::vector<int> rank_by_id;
    float chosen_logit = 0.f;
    float denom = 0.f;   // topk_cumsum[vocab-1], the full-vocab softmax denominator
};

// Runs the sort/scan/mask pipeline (top_k/top_p both disabled -- this test is about the sort/
// rank/extraction machinery, not truncation) + the chosen-logit extraction for a given "chosen"
// vocab id, matching the real decode call-site order (launch_topk_topp_mask then
// launch_extract_chosen_logit).
PipelineResult run_pipeline(Scratch& s, const std::vector<float>& logits, int chosen_id) {
    const int vocab = (int)logits.size();
    float* d_logits = nullptr;
    cudaMalloc(&d_logits, vocab * sizeof(float));
    cudaMemcpy(d_logits, logits.data(), vocab * sizeof(float), cudaMemcpyHostToDevice);
    int top_k = 0;
    float top_p = 1.f;
    cudaMemcpy(s.d_top_k, &top_k, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_top_p, &top_p, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(s.d_out_id, &chosen_id, sizeof(int), cudaMemcpyHostToDevice);

    launch_topk_topp_mask(d_logits, vocab, s.vocab_iota, s.sorted_logits, s.sorted_idx,
                          s.topk_exp, s.topk_cumsum, s.sort_temp, s.sort_temp_bytes,
                          s.scan_temp, s.scan_temp_bytes, s.d_top_k, s.d_top_p, s.rank_by_id);
    launch_extract_chosen_logit(s.d_out_id, s.rank_by_id, s.sorted_logits, s.chosen_logit);

    PipelineResult r;
    r.sorted_idx.resize(vocab);
    r.sorted_logits.resize(vocab);
    r.rank_by_id.resize(vocab);
    cudaMemcpy(r.sorted_idx.data(), s.sorted_idx, vocab * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(r.sorted_logits.data(), s.sorted_logits, vocab * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(r.rank_by_id.data(), s.rank_by_id, vocab * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&r.chosen_logit, s.chosen_logit, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&r.denom, s.topk_cumsum + (vocab - 1), sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_logits);
    return r;
}

bool test_rank_by_id_is_exact_inverse_permutation() {
    const int vocab = 500;
    std::vector<float> logits(vocab);
    for (int i = 0; i < vocab; i++) logits[i] = (float)((i * 131 + 7) % vocab);
    Scratch s(vocab);
    auto r = run_pipeline(s, logits, /*chosen_id=*/0);
    bool ok = true;
    for (int i = 0; i < vocab; i++) {
        if (r.rank_by_id[r.sorted_idx[i]] != i) {
            printf("FAIL: rank_by_id[sorted_idx[%d]=%d] = %d, expected %d\n",
                   i, r.sorted_idx[i], r.rank_by_id[r.sorted_idx[i]], i);
            ok = false;
            break;
        }
    }
    if (ok) printf("[OK] rank_by_id is an exact inverse permutation of sorted_idx\n");
    return ok;
}

bool test_top_n_matches_cpu_sort_reference() {
    const int vocab = 300;
    std::vector<float> logits(vocab);
    for (int i = 0; i < vocab; i++) logits[i] = (float)((i * 977 + 13) % vocab) - 150.0f;
    Scratch s(vocab);
    auto r = run_pipeline(s, logits, 0);

    std::vector<int> idx(vocab);
    for (int i = 0; i < vocab; i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });

    bool ok = true;
    for (int top_n : {0, 1, 5, 20}) {
        for (int i = 0; i < top_n; i++) {
            if (r.sorted_idx[i] != idx[i] || r.sorted_logits[i] != logits[idx[i]]) {
                printf("FAIL: top_n=%d rank=%d: got idx=%d logit=%.3f, expected idx=%d logit=%.3f\n",
                       top_n, i, r.sorted_idx[i], r.sorted_logits[i], idx[i], logits[idx[i]]);
                ok = false;
            }
        }
    }
    if (ok) printf("[OK] top-N sorted output matches CPU std::sort reference\n");
    return ok;
}

bool test_chosen_logit_matches_rank0_at_greedy() {
    const int vocab = 50;
    std::vector<float> logits(vocab);
    for (int i = 0; i < vocab; i++) logits[i] = (float)(vocab - i);
    logits[7] = 1000.f;   // make id 7 the unambiguous greedy winner
    Scratch s(vocab);
    auto r = run_pipeline(s, logits, /*chosen_id=*/7);
    const bool ok = (r.sorted_idx[0] == 7) && (r.chosen_logit == r.sorted_logits[0]);
    if (!ok) printf("FAIL: chosen logit %.3f != rank0 logit %.3f (sorted_idx[0]=%d)\n",
                     r.chosen_logit, r.sorted_logits[0], r.sorted_idx[0]);
    else printf("[OK] chosen logit matches sorted_logits[0] when the chosen id is the greedy winner\n");
    return ok;
}

// The test that actually exercises why launch_extract_chosen_logit's rank lookup is needed at
// all: the sampled token is not necessarily rank 0 once real temperature sampling is active, so
// its raw (pre-noise) logit must be found via rank_by_id, not assumed to be sorted_logits[0].
bool test_chosen_logit_matches_arbitrary_nonzero_rank() {
    const int vocab = 50;
    std::vector<float> logits(vocab);
    for (int i = 0; i < vocab; i++) logits[i] = (float)(vocab - i);   // strictly descending by id
    Scratch s(vocab);
    auto r = run_pipeline(s, logits, /*chosen_id=*/10);

    std::vector<int> idx(vocab);
    for (int i = 0; i < vocab; i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    int rank10 = -1;
    for (int i = 0; i < vocab; i++) if (idx[i] == 10) { rank10 = i; break; }

    bool ok = (r.chosen_logit == logits[10]) && (rank10 != 0);
    if (!ok) printf("FAIL: chosen_logit=%.3f expected=%.3f rank10=%d\n", r.chosen_logit, logits[10], rank10);
    else printf("[OK] chosen logit correctly recovered for a non-rank-0 chosen token (rank=%d)\n", rank10);
    return ok;
}

bool test_logsumexp_hand_computed() {
    // Clean softmax probabilities [0.4, 0.3, 0.2, 0.1] sum to 1.0, so logits=ln(p) gives
    // sum(exp(logits))=sum(p)=1 exactly -- the TRUE logsumexp(logits) = ln(1) = 0.
    //
    // The kernel computes denom = sum(exp(logit_i - row_max)) (numerically-stable, row_max
    // subtracted first), NOT sum(exp(logit_i)) directly -- so denom itself is NOT expected to be
    // 1.0. Here row_max = ln(0.4), denom = sum(p_i/0.4) = 1/0.4 = 2.5 exactly; logsumexp =
    // row_max + log(denom) = ln(0.4) + ln(2.5) = ln(0.4*2.5) = ln(1) = 0, recovering the TRUE
    // logsumexp despite the intermediate row_max-shifted denom not being 1.
    const std::vector<float> logits = {(float)std::log(0.4), (float)std::log(0.3),
                                       (float)std::log(0.2), (float)std::log(0.1)};
    Scratch s((int)logits.size());
    auto r = run_pipeline(s, logits, 0);
    const float logsumexp = r.sorted_logits[0] + std::log(r.denom);
    const bool ok = std::fabs(logsumexp - 0.0f) < 1e-3f;
    if (!ok) printf("FAIL: logsumexp=%.6f (expected ~0.0) row_max=%.6f denom=%.6f\n",
                     logsumexp, r.sorted_logits[0], r.denom);
    else printf("[OK] hand-computed logsumexp matches (denom=%.6f, row_max=%.6f)\n", r.denom, r.sorted_logits[0]);
    return ok;
}

// Teacher-forced scoring (/v1/score) path: launch_extract_chosen_logit_id takes the token id as a
// by-value kernel ARGUMENT rather than through a device buffer, and must agree exactly with the
// device-buffer form for every id.
//
// Regression guard for #1001. token_logprob_for() used to stage the id with a cudaMemcpy on the
// legacy default stream and then launch the kernel on Impl::stream (cudaStreamNonBlocking), which
// does not serialize against it -- so the kernel could read the PREVIOUS call's id and return a
// confident, wrong logprob while the argmax and top_logprobs (which come from the sort, not from
// this lookup) stayed correct. The by-value form has no second stream to race against.
//
// The loop below is the part that matters: it scores a SEQUENCE of different ids back to back on
// one non-blocking stream, exactly as scoring a completion does. Under the old pattern that is
// the shape that lets iteration i observe iteration i-1's id; here every value must be exact.
bool test_chosen_logit_by_value_matches_device_buffer_form() {
    const int vocab = 4096;
    std::vector<float> logits(vocab);
    for (int i = 0; i < vocab; i++) logits[i] = std::sin((float)i * 0.37f) * 40.0f;

    Scratch s(vocab);
    // Populate the sort/rank scratch once, the way a forward pass leaves it.
    auto r = run_pipeline(s, logits, /*chosen_id=*/0);

    cudaStream_t st = nullptr;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);

    bool ok = true;
    int checked = 0;
    // Walk a run of distinct ids back to back, no sync between the id changing and the launch --
    // the exact call shape /v1/score produces, one lookup per completion token.
    for (int id = 1; id < vocab; id += 7) {
        float got = 0.f;
        sparkinfer::kernels::launch_extract_chosen_logit_id(id, s.rank_by_id, s.sorted_logits,
                                                            s.chosen_logit, st);
        cudaStreamSynchronize(st);
        cudaMemcpy(&got, s.chosen_logit, sizeof(float), cudaMemcpyDeviceToHost);

        const float want = r.sorted_logits[r.rank_by_id[id]];
        if (got != want) {
            printf("FAIL: by-value id=%d got %.6f, expected %.6f (rank %d)%s\n",
                   id, got, want, r.rank_by_id[id],
                   /* the signature of the old race: we saw the PREVIOUS id's logit */
                   (id >= 8 && got == r.sorted_logits[r.rank_by_id[id - 7]]) ? "  <-- PREVIOUS id"
                                                                            : "");
            ok = false;
            break;
        }
        checked++;
    }
    cudaStreamDestroy(st);
    if (ok) printf("[OK] by-value chosen-logit matches the sorted/rank reference for %d "
                   "back-to-back ids\n", checked);
    return ok;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no CUDA device\n");
        return 0;
    }
    bool ok = true;
    ok = test_rank_by_id_is_exact_inverse_permutation() && ok;
    ok = test_top_n_matches_cpu_sort_reference() && ok;
    ok = test_chosen_logit_matches_rank0_at_greedy() && ok;
    ok = test_chosen_logit_matches_arbitrary_nonzero_rank() && ok;
    ok = test_logsumexp_hand_computed() && ok;
    ok = test_chosen_logit_by_value_matches_device_buffer_form() && ok;
    if (!ok) return 1;
    printf("logprob_extract_gpu_test: OK\n");
    return 0;
}
