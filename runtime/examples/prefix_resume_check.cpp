// Numerics check for resuming batched prefill from a cached prefix (the server's automatic prefix
// cache), on a compressed-tensors checkpoint -- what the server serves.
//
// A request with a cache checkpoint is prefilled in two batched passes, [0, c) and then [c, P) at
// pos0 = c, instead of one pass over [0, P). The second pass is a different shape (often only a
// few tokens), so its quantized kernels can land on different numbers. This measures how
// different, at the distribution that picks the first generated token, four ways:
//
//   single  one batched pass over [0, P)                     -- an uncached server
//   loop    batched [0, c), then the token loop for [c, P)   -- the decode path, one token at a time
//   split   batched [0, c), then batched resume [c, P)       -- a cache miss that takes a checkpoint
//   cached  [0, c) in one session and snapshot its recurrent state; a second session shares those
//           KV blocks, restores the snapshot, then batched resume [c, P)   -- a cache hit
//
// `loop` is the reference: every decode step already runs it. If split sits as close to loop as
// single does, resuming is as sound as the production pass.
//
// cached must reproduce split exactly -- they are the same computation from the same prefix, so any
// difference is the sharing or the restore. That is only checkable under SPARKINFER_DETERMINISTIC=1:
// by default a short batched pass is not bit-reproducible run to run (see kernels/deterministic.h),
// so two identical resumes can differ by a few tenths of a nat and the check reports, not judges.
//
// usage: prefix_resume_check <model_dir> <ids_file> [tail ...]   (tail = P - c before block alignment)

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "sparkinfer/kernels/deterministic.h"
#include "qwen38_hf_config.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using TL = sparkinfer::Qwen35Model::TokenLogprob;

namespace {

constexpr int kForced = 32;   // teacher-forced continuation tokens compared after the prompt

struct Cmp {
    int top1_same = 0;   // positions whose top-1 token agrees
    int positions = 0;
    double max_dlp = 0;  // worst |logprob of reference top-1 under ref - under x|
    double mean_kl = 0;  // KL(ref || x) over ref's top-20, averaged over positions; a token x does not
                         // list gets x's floor
};

void compare_one(const TL& ref, const TL& x, Cmp& c) {
    if (ref.top_alternatives.empty() || x.top_alternatives.empty()) return;
    c.positions++;
    if (ref.top_alternatives[0].first == x.top_alternatives[0].first) c.top1_same++;
    const float floor_lp = x.top_alternatives.back().second;
    auto lp_in_x = [&](int id) {
        for (const auto& p : x.top_alternatives) if (p.first == id) return p.second;
        return floor_lp;
    };
    c.max_dlp = std::max(c.max_dlp, (double)std::fabs(ref.top_alternatives[0].second -
                                                     lp_in_x(ref.top_alternatives[0].first)));
    double kl = 0;
    for (const auto& p : ref.top_alternatives)
        kl += std::exp((double)p.second) * ((double)p.second - (double)lp_in_x(p.first));
    c.mean_kl += kl;
}

Cmp compare(const std::vector<TL>& ref, const std::vector<TL>& x) {
    Cmp c;
    for (size_t i = 0; i < std::min(ref.size(), x.size()); i++) compare_one(ref[i], x[i], c);
    if (c.positions) c.mean_kl /= c.positions;
    return c;
}

bool exact(const std::vector<TL>& a, const std::vector<TL>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i].top_alternatives.size() != b[i].top_alternatives.size()) return false;
        for (size_t k = 0; k < a[i].top_alternatives.size(); k++)
            if (a[i].top_alternatives[k] != b[i].top_alternatives[k]) return false;
    }
    return true;
}

std::string fmt(const Cmp& c) {
    char buf[96];
    snprintf(buf, sizeof(buf), "top1 %2d/%2d KL %.1e max|dlp| %.1e", c.top1_same, c.positions, c.mean_kl, c.max_dlp);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <model_dir> <ids_file> [tail ...]\n", argv[0]);
        return 2;
    }
    const std::string model_dir = argv[1];
    std::vector<int> ids;
    {
        std::ifstream f(argv[2]);
        int v;
        while (f >> v) ids.push_back(v);
    }
    std::vector<int> tails;
    for (int i = 3; i < argc; i++) tails.push_back(atoi(argv[i]));
    if (tails.empty()) tails = {15, 64, 512, 4096};
    const int P = (int)ids.size();
    if (P < 64) { printf("[FAIL] need a longer prompt (%d ids)\n", P); return 1; }

    sparkinfer::Qwen35Config cfg;
    std::string err;
    if (!qwen38_config_from_hf_json(model_dir, cfg, err)) { printf("[FAIL] config: %s\n", err.c_str()); return 1; }
    cfg.max_seq = P + 256;

    auto rt = sparkinfer::Runtime::create({});
    rt->initialize();

    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers;
    kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim;
    kvc.block_size = 16;
    kvc.int8_kv = true;   // what the server runs at this context
    kvc.layer_slot = sparkinfer::hybrid_kv_layer_slots(cfg.n_layers, cfg.hybrid, cfg.full_attn_interval);
    const int kvL = sparkinfer::kv_slot_count(kvc.layer_slot, cfg.n_layers);
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = 2 * ((size_t)(cfg.max_seq + 15) / 16) + 16;   // two live sessions at once
    sparkinfer::KVCacheManager kv(kvc, (size_t)kvL * 2 * epb * 2 * blocks);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts;
    mc.top_k = cfg.top_k;
    mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn;
    mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);
    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    if (!model.load_compressed_tensors(model_dir)) { printf("[FAIL] load_compressed_tensors\n"); return 1; }

    auto open = [&](const std::vector<int>* shared) -> uint64_t {
        const uint64_t sid = model.open_session(P + 64, nullptr, shared);
        if (sid) {
            model.activate_session(sid);
            model.reset_mrope_offset();
        }
        return sid;
    };
    auto ingest = [&](int start, int end, bool want, bool resume) {
        int out = start;
        const int seed = model.ingest_prompt_range(ids.data(), start, end, 0, &out, want, resume);
        return (out == end) ? seed : -1;
    };

    // The first token's distribution, then kForced teacher-forced steps along one fixed continuation.
    // Decode (forward_token) is the same arithmetic in every arm, so everything that differs between
    // arms is what the prefill left in the KV and the recurrent state -- measured over 33 positions
    // rather than one, which on a single distribution is too noisy to rank arms by.
    std::vector<int> cont;   // the continuation every arm is forced along (single pass #0's greedy)
    auto follow = [&](int P0) {
        std::vector<TL> d;
        d.push_back(model.last_token_logprobs(20));
        for (size_t t = 0; t < cont.size(); t++) {
            model.forward_token(cont[t], P0 + (int)t, true);
            d.push_back(model.last_token_logprobs(20));
        }
        return d;
    };

    std::vector<TL> single[2];
    for (int r = 0; r < 2; r++) {
        const uint64_t sid = open(nullptr);
        const int seed = sid ? ingest(0, P, true, false) : -1;
        if (seed < 0) { printf("[FAIL] single pass\n"); return 1; }
        if (r == 0) {
            int tok = seed;
            for (int t = 0; t < kForced && tok >= 0; t++) {
                cont.push_back(tok);
                tok = model.forward_token(tok, P + t, true);
            }
            model.close_session(sid);
            const uint64_t again = open(nullptr);   // redo it to measure along the fixed continuation
            if (!again || ingest(0, P, true, false) < 0) { printf("[FAIL] single pass\n"); return 1; }
            single[0] = follow(P);
            model.close_session(again);
            continue;
        }
        single[r] = follow(P);
        model.close_session(sid);
    }

    printf("\nP = %d tokens, int8 KV, %d positions each (first token + %d forced). Cells: top-1 agreement, "
           "mean KL over top-20, worst |dlogprob| of the reference top-1\n", P, kForced + 1, kForced);
    printf("noise (single vs single): %s%s\n", fmt(compare(single[0], single[1])).c_str(),
           exact(single[0], single[1]) ? "  [bit-identical]" : "");
    printf("%6s %6s | %-40s | %-40s | %-40s | %-40s\n", "tail", "c", "split vs loop (same prefix)",
           "cached vs split (same prefix)", "split vs single", "single vs loop");
    int failures = 0;
    for (int tail : tails) {
        const int c = std::max(16, (P - tail) / 16 * 16);
        if (c >= P) continue;

        // One prefix computation: session A ingests [0, c), then its state and blocks are kept.
        const uint64_t a = open(nullptr);
        sparkinfer::Qwen35Model::RecurrentStateSnapshot snap;
        if (!a || ingest(0, c, false, false) < 0 || !model.snapshot_recurrent_state(a, snap)) { printf("[FAIL] prefix c=%d\n", c); return 1; }
        std::vector<int> shared = kv.retain_prefix_blocks(a, c / 16);

        // split: A itself continues with a batched resume.
        if (ingest(c, P, true, true) < 0) { printf("[FAIL] split c=%d\n", c); return 1; }
        const std::vector<TL> split = follow(P);
        model.close_session(a);

        // cached: a new session on A's blocks and snapshot, batched resume.
        const uint64_t b = open(&shared);
        if (!b || !model.restore_recurrent_state(b, snap) || ingest(c, P, true, true) < 0) { printf("[FAIL] cached c=%d\n", c); return 1; }
        const std::vector<TL> cached = follow(P);
        model.close_session(b);

        // loop: the same restored prefix, then the token loop -- the decode path's arithmetic.
        const uint64_t l = open(&shared);
        if (!l || !model.restore_recurrent_state(l, snap) || ingest(c, P, true, false) < 0) { printf("[FAIL] loop c=%d\n", c); return 1; }
        const std::vector<TL> loop = follow(P);
        model.close_session(l);
        kv.release_blocks(shared);

        const bool same = exact(split, cached);
        if (sparkinfer::deterministic_mode() && !same) failures++;
        printf("%6d %6d | %-40s | %-40s | %-40s | %-40s\n", P - c, c, fmt(compare(loop, split)).c_str(),
               same ? "bit-identical" : fmt(compare(split, cached)).c_str(),
               fmt(compare(single[0], split)).c_str(), fmt(compare(loop, single[0])).c_str());
    }
    printf("free blocks at exit: %d of %d\n", kv.num_free_blocks(), kv.num_total_blocks());
    if (!sparkinfer::deterministic_mode())
        printf("[INFO] cached vs split not judged: run with SPARKINFER_DETERMINISTIC=1 for an exact check\n");
    else
        printf("%s\n", failures ? "[FAIL] cached does not reproduce split" : "[OK] cached reproduces split exactly");
    return failures ? 1 : 0;
}
