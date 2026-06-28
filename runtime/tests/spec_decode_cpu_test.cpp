// CPU test for prompt-lookup speculative decoding (sparkinfer/spec_decode.h).
// Pure host logic, no GPU needed.
//
// Two checks:
//  1. propose_draft on a few concrete histories.
//  2. Equivalence: a speculative draft+verify+accept loop, driven by a fake
//     deterministic "model" function, must produce byte-identical output to
//     plain sequential greedy decoding using the same function. This is the
//     correctness property speculative decoding depends on -- it must never
//     change *what* is generated, only how many forward passes it takes.
//
// Build: g++ -O2 -std=c++17 spec_decode_cpu_test.cpp ../src/spec_decode.cpp \
//        -I ../include -o spec_decode_cpu_test

#include "sparkinfer/spec_decode.h"

#include <cstdio>
#include <vector>
#include <random>

using namespace sparkinfer;
using std::vector;

static bool check(bool cond, const char* what, bool& all_ok) {
    printf("%-60s -> %s\n", what, cond ? "PASS" : "FAIL");
    all_ok = all_ok && cond;
    return cond;
}

// Deterministic fake "model": next token is a function of the full prefix
// (stands in for a real autoregressive transformer for this test -- the
// point is to exercise the accept/reject bookkeeping, not real weights).
static int fake_model_next(const vector<int>& prefix, int vocab) {
    unsigned h = 2166136261u;
    for (int t : prefix) { h ^= (unsigned)t; h *= 16777619u; }
    return (int)(h % (unsigned)vocab);
}

// Plain sequential greedy decode using the fake model.
static vector<int> sequential_generate(const vector<int>& prompt, int n_new, int vocab) {
    vector<int> hist = prompt, out;
    for (int i = 0; i < n_new; i++) {
        int tok = fake_model_next(hist, vocab);
        out.push_back(tok);
        hist.push_back(tok);
    }
    return out;
}

// Speculative draft+verify+accept loop using the SAME fake model as the
// "verifier", mirroring the real generate_speculative() control flow:
//   - draft = propose_draft(history so far, up to draft_k-1 tokens)
//   - batch_in = [cur] + draft
//   - verified[i] = fake_model_next(history-so-far + batch_in[0..i])
//   - accept the longest matching prefix of draft against verified[0..],
//     plus the bonus token at the first mismatch (or the end).
static vector<int> speculative_generate(const vector<int>& prompt, int n_new, int vocab, int draft_k) {
    vector<int> committed = prompt, out;
    int cur = fake_model_next(committed, vocab);   // bootstrap, like the prefill's last token
    while ((int)out.size() < n_new) {
        vector<int> draft = propose_draft(committed, draft_k > 0 ? draft_k - 1 : 0);
        vector<int> batch_in; batch_in.push_back(cur);
        for (int d : draft) batch_in.push_back(d);

        vector<int> verified;
        vector<int> prefix = committed;
        for (int t : batch_in) { prefix.push_back(t); verified.push_back(fake_model_next(prefix, vocab)); }

        AcceptResult r = accept_draft(draft, verified);
        out.push_back(cur);
        committed.push_back(cur);
        for (int i = 0; i < r.accepted; i++) {
            out.push_back(draft[i]);
            committed.push_back(draft[i]);
            if ((int)out.size() >= n_new) break;
        }
        cur = r.bonus_token;
    }
    out.resize(n_new);
    return out;
}

int main() {
    bool all_ok = true;

    // --- propose_draft on concrete cases ---
    {
        // trailing 3-gram {1,2,3} last occurred at index 0, followed by {9,9}.
        vector<int> h = {1, 2, 3, 9, 9, 9, 1, 2, 3};
        auto d = propose_draft(h, 2);
        check(d.size() == 2 && d[0] == 9 && d[1] == 9, "propose_draft finds repeated n-gram continuation", all_ok);
    }
    {
        vector<int> h = {1, 2, 3, 4, 5, 6, 7};
        auto d = propose_draft(h, 3);
        check(d.empty(), "propose_draft returns empty when nothing repeats", all_ok);
    }

    // --- accept_draft bookkeeping ---
    {
        AcceptResult r = accept_draft({5, 6, 7}, {5, 6, 9, 2});
        check(r.accepted == 2 && r.bonus_token == 9, "accept_draft stops at first mismatch", all_ok);
    }
    {
        AcceptResult r = accept_draft({5, 6, 7}, {5, 6, 7, 2});
        check(r.accepted == 3 && r.bonus_token == 2, "accept_draft accepts a fully-matching draft", all_ok);
    }
    {
        AcceptResult r = accept_draft({}, {3});
        check(r.accepted == 0 && r.bonus_token == 3, "accept_draft handles an empty draft", all_ok);
    }

    // --- equivalence: speculative loop must match sequential greedy decode ---
    std::mt19937 rng(42);
    int trials_ok = 0, trials = 200;
    for (int t = 0; t < trials; t++) {
        int vocab = 5 + (int)(rng() % 20);          // small vocab -> frequent repeats -> exercises drafting
        int prompt_len = 3 + (int)(rng() % 6);
        int n_new = 10 + (int)(rng() % 30);
        int draft_k = 2 + (int)(rng() % 5);          // K in [2,6]
        vector<int> prompt;
        for (int i = 0; i < prompt_len; i++) prompt.push_back((int)(rng() % vocab));

        auto gt = sequential_generate(prompt, n_new, vocab);
        auto sp = speculative_generate(prompt, n_new, vocab, draft_k);
        if (gt == sp) trials_ok++;
        else if (trials_ok == t) {   // print first failure for debugging
            printf("MISMATCH trial %d (vocab=%d, draft_k=%d)\n", t, vocab, draft_k);
        }
    }
    char buf[96];
    snprintf(buf, sizeof(buf), "speculative loop matches sequential greedy (%d/%d trials)", trials_ok, trials);
    check(trials_ok == trials, buf, all_ok);

    printf("\n%s\n", all_ok ? "ALL PASS" : "SOME FAILED");
    return all_ok ? 0 : 1;
}
