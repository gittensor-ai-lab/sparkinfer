// How often would the checkpoint's own MTP head propose the token the target actually emits?
//
// That single number decides whether MTP is worth using as DSpark's replacement drafter. The
// economics: DSpark costs ~2.12 ms of draft per decode step against MTP's ~0.64 ms requantised,
// so at the measured verify cost (c0 ~9.7 ms, c1 ~0.74 ms/row) MTP breaks even at an acceptance
// of ~0.47 and reaches a 10% end-to-end win at ~0.55. DSpark's own position-1 acceptance on this
// target is 0.486.
//
// Teacher-forces the eval corpus through the target one position at a time. That is what makes
// the probe cheap: MTP's KV cache fills incrementally as a side effect, so no batched MTP prefill
// is needed to get a faithful measurement.
//
// At position p the target's forward gives hidden h_p and its greedy next token x_{p+1}. The head
// is defined as MTP(h_p, emb(x_{p+1})) -> x_{p+2}, so the proposal is scored against the target's
// OWN argmax at p+1 -- acceptance in speculative decoding is agreement with the target, not with
// the gold text.
//
// Usage: mtp_probe <qwen38_checkpoint_dir> <id0> [id1 ...]
#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "qwen_checkpoint.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s <qwen38_dir> <id0> [id1 ...]\n", argv[0]); return 2; }
    setenv("SPARKINFER_PREFILL_I8", "0", 0);
    const std::string tpath = argv[1];
    std::vector<int> ids;
    for (int i = 2; i < argc; i++) ids.push_back(atoi(argv[i]));
    if ((int)ids.size() < 8) { printf("[FAIL] need >= 8 ids\n"); return 2; }

    sparkinfer::Qwen35Config cfg; sparkinfer::GGUF g;
    QwenCheckpointKind kind; std::string err;
    if (!qwen_checkpoint_open(tpath, cfg, g, kind, err)) { printf("[FAIL] %s\n", err.c_str()); return 1; }
    cfg.max_seq = std::max(2048, (int)ids.size() + 64);

    auto rt = sparkinfer::Runtime::create({}); rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers; kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim; kvc.block_size = 16; kvc.int8_kv = false;
    kvc.layer_slot = sparkinfer::hybrid_kv_layer_slots(cfg.n_layers, cfg.hybrid, cfg.full_attn_interval);
    const int kvL = sparkinfer::kv_slot_count(kvc.layer_slot, cfg.n_layers);
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = (cfg.max_seq + 15) / 16 + 8;
    sparkinfer::KVCacheManager kv(kvc, (size_t)kvL * 2 * epb * 2 * blocks);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts; mc.top_k = cfg.top_k; mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn; mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    printf("loading target (%s) ...\n", qwen_checkpoint_kind_label(kind));
    if (!qwen_checkpoint_load(model, tpath, kind)) { printf("[FAIL] target load\n"); return 1; }
    if (!model.mtp_available()) { printf("[FAIL] this checkpoint has no MTP head\n"); return 1; }

    // Capture the LAST layer's residual stream -- the pre-final-norm hidden the MTP head consumes.
    model.set_dflash_capture(true, {cfg.n_layers - 1}, 1);
    const uint64_t sid = model.open_session(cfg.max_seq);
    if (!sid) { printf("[FAIL] session\n"); return 1; }
    model.activate_session(sid);

    const int H = cfg.hidden;
    std::vector<uint16_t> hh((size_t)H);
    void* h_dev = nullptr;
    if (cudaMalloc(&h_dev, (size_t)H * 2) != cudaSuccess) { printf("[FAIL] alloc\n"); return 1; }

    // Prompt is teacher-forced to establish context, then the probe FREE-RUNS on the target's own
    // argmax chain. That matters: speculative decoding only ever sees the argmax chain, so scoring
    // a proposal conditioned on argmax against a continuation that was teacher-forced with gold
    // compares two different contexts and understates acceptance wherever the two diverge.
    const size_t n_prompt = std::min<size_t>(ids.size() / 2, 512);
    long hits = 0, seen = 0;
    int prev_prop = -1;           // MTP's proposal for THIS position, made one step earlier
    int cur = ids[0];
    for (size_t p = 0; p + 1 < ids.size(); p++) {
        const int fed = (p < n_prompt) ? ids[p] : cur;
        const int nxt = model.forward_token(fed, (int)p, true);
        if (nxt < 0) { printf("[FAIL] forward at %zu\n", p); return 1; }
        cur = nxt;
        const bool scoring = p >= n_prompt;
        // Score the proposal made at p-1, which predicted the target's argmax here.
        if (prev_prop >= 0 && scoring) { seen++; if (prev_prop == nxt) hits++; }
        // Snapshot the captured hidden (the capture buffer is overwritten by the next forward).
        cudaMemcpy(h_dev, model.dflash_hidden_ptr(), (size_t)H * 2, cudaMemcpyDeviceToDevice);
        { int pr[4] = {-1,-1,-1,-1};
            model.mtp_propose(h_dev, nxt, (int)p + 1, 1, pr);
            prev_prop = pr[0]; }
        if ((p + 1) % 256 == 0)
            fprintf(stderr, "  ... %zu positions, running p1=%.4f\n", p + 1,
                    seen ? (double)hits / seen : 0.0);
    }
    printf("\nMTP positions scored : %ld\n", seen);
    printf("METRIC mtp_p1 %.4f\n", seen ? (double)hits / seen : 0.0);
    printf("(DSpark's own position-1 acceptance on this target is 0.486; MTP break-even ~0.47,\n"
           " +10%% end-to-end at ~0.55)\n");
    model.close_session(sid);
    return 0;
}
