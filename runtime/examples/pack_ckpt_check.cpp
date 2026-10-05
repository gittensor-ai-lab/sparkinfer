// Check for packed prefill with prefix-cache checkpoints (Qwen35Model::ingest_prompts_packed with
// ckpt_rows), on a compressed-tensors checkpoint -- what the server serves.
//
// Three chat-length prompts, each with one block-aligned checkpoint near its end, prefilled two
// ways:
//   alone   one prompt per pass through ingest_prompt_checkpointed (the one-prompt path)
//   packed  all three in one pass through ingest_prompts_packed with their checkpoint rows
// The pack's total is not a multiple of 8, so one prompt's last tokens also take the decode-step
// tail. Reports, per prompt, whether the seeds (argmax first tokens) agree and how far the packed
// snapshot is from the one-prompt one. The two are different pass shapes, so their quantized
// kernels may land on slightly different numbers; the check fails only on a wrong layout -- a
// snapshot that is not close, or a missing one.
//
// usage: pack_ckpt_check <model_dir or .gguf> <ids_file>
#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "qwen38_hf_config.h"
#include "sparkinfer/gguf.h"
#include "qwen3_gguf_config.h"
#include "qwen_checkpoint.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <model_dir> <ids_file>\n", argv[0]);
        return 2;
    }
    const std::string model_dir = argv[1];
    std::vector<int> ids;
    {
        std::ifstream f(argv[2]);
        int v;
        while (f >> v) ids.push_back(v);
    }
    const int lens[3] = {1076, 1077, 777};
    const int offs[3] = {0, 40, 300};
    for (int i = 0; i < 3; i++)
        if (offs[i] + lens[i] > (int)ids.size()) { printf("[FAIL] need %d ids\n", offs[i] + lens[i]); return 1; }
    int ckpt[3];
    for (int i = 0; i < 3; i++) ckpt[i] = ((lens[i] - 24) / 16) * 16;   // block-aligned, >= 16 before the end

    // Any checkpoint qwen3_gguf_bench opens: a compressed-tensors directory or a GGUF (an MoE pack,
    // Qwen3.6-35B-A3B, takes the same path).
    sparkinfer::Qwen35Config cfg;
    sparkinfer::GGUF g;
    QwenCheckpointKind kind{};
    std::string err;
    if (!qwen_checkpoint_open(model_dir, cfg, g, kind, err)) { printf("[FAIL] open: %s\n", err.c_str()); return 1; }
    cfg.max_seq = 1200;

    auto rt = sparkinfer::Runtime::create({});
    rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers;
    kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim;
    kvc.block_size = 16;
    kvc.int8_kv = true;
    kvc.layer_slot = sparkinfer::hybrid_kv_layer_slots(cfg.n_layers, cfg.hybrid, cfg.full_attn_interval);
    const int kvL = sparkinfer::kv_slot_count(kvc.layer_slot, cfg.n_layers);
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = 6 * ((size_t)(cfg.max_seq + 15) / 16) + 16;
    sparkinfer::KVCacheManager kv(kvc, (size_t)kvL * 2 * epb * 2 * blocks);
    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts;
    mc.top_k = cfg.top_k;
    mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn;
    mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);
    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    if (!qwen_checkpoint_load(model, model_dir, kind)) { printf("[FAIL] load %s\n", qwen_checkpoint_kind_label(kind)); return 1; }

    using Snap = sparkinfer::Qwen35Model::RecurrentStateSnapshot;
    // alone
    int seed_a[3];
    Snap snap_a[3];
    for (int i = 0; i < 3; i++) {
        const uint64_t sid = model.open_session(lens[i] + 64);
        if (!sid) {
            size_t f = 0, t = 0;
            cudaMemGetInfo(&f, &t);
            printf("[FAIL] open alone %d (free %.2f GB, free KV blocks %d)\n", i, f / 1e9, kv.num_free_blocks());
            return 1;
        }
        model.activate_session(sid);
        model.reset_mrope_offset();
        int done = 0;
        seed_a[i] = model.ingest_prompt_checkpointed(ids.data() + offs[i], 0, lens[i], &ckpt[i], 1,
                                                     &snap_a[i], &done, false);
        if (seed_a[i] < 0 || done != lens[i]) { printf("[FAIL] alone %d\n", i); return 1; }
        model.close_session(sid);
    }
    // packed
    uint64_t sids[3];
    const int* prompts[3];
    for (int i = 0; i < 3; i++) {
        sids[i] = model.open_session(lens[i] + 64);
        if (!sids[i]) {
            size_t f = 0, t = 0;
            cudaMemGetInfo(&f, &t);
            printf("[FAIL] open packed %d (free %.2f GB, free KV blocks %d)\n", i, f / 1e9, kv.num_free_blocks());
            return 1;
        }
        prompts[i] = ids.data() + offs[i];
    }
    model.reset_mrope_offset();
    int seed_p[3] = {-1, -1, -1};
    Snap snap_p[3];
    if (!model.ingest_prompts_packed(sids, prompts, lens, 3, seed_p, nullptr, ckpt, snap_p)) {
        printf("[FAIL] ingest_prompts_packed declined\n");
        return 1;
    }
    bool ok = true;
    printf("prompt  len  ckpt | seed alone packed | snapshot state max|d| (max|x|)   conv max|d|\n");
    for (int i = 0; i < 3; i++) {
        const std::vector<char> a = sparkinfer::Qwen35Model::snapshot_bytes(snap_a[i]);
        const std::vector<char> b = sparkinfer::Qwen35Model::snapshot_bytes(snap_p[i]);
        if (a.empty() || a.size() != b.size()) { printf("[FAIL] snapshot %d missing or mis-sized\n", i); ok = false; continue; }
        double dmax = 0, ref = 0, cmax = 0;
        const float* fa = reinterpret_cast<const float*>(a.data());
        const float* fb = reinterpret_cast<const float*>(b.data());
        for (size_t k = 0; k < snap_a[i].state_bytes / sizeof(float); k++) {
            dmax = std::max(dmax, (double)std::fabs(fa[k] - fb[k]));
            ref = std::max(ref, (double)std::fabs(fa[k]));
        }
        const uint16_t* ca = reinterpret_cast<const uint16_t*>(a.data() + snap_a[i].state_bytes);
        const uint16_t* cb = reinterpret_cast<const uint16_t*>(b.data() + snap_a[i].state_bytes);
        auto bf = [](uint16_t h) { uint32_t u = (uint32_t)h << 16; float v; std::memcpy(&v, &u, 4); return v; };
        for (size_t k = 0; k < snap_a[i].conv_bytes / 2; k++) cmax = std::max(cmax, (double)std::fabs(bf(ca[k]) - bf(cb[k])));
        printf("%6d %5d %5d | %6d %6d %s | %.2e (%.2e)   %.2e\n", i, lens[i], ckpt[i], seed_a[i], seed_p[i],
               seed_a[i] == seed_p[i] ? "same" : "DIFF", dmax, ref, cmax);
        // Close means rounding-level: a layout slip (a layer's state in another's slot, a missing
        // copy) shows up as differences of the order of the values themselves.
        if (!(dmax <= 0.05 * ref)) ok = false;
    }
    for (int i = 0; i < 3; i++) model.close_session(sids[i]);
    printf(ok ? "[OK] packed checkpoints match the one-prompt path\n" : "[FAIL] packed checkpoints differ\n");
    return ok ? 0 : 1;
}
