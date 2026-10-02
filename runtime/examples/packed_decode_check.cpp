// Check for Qwen35Model::decode_packed: does a packed step decode each row as one forward per row
// would?
//
// Three sets of sessions get the same prompts. Set A decodes one forward per row (forward_token,
// greedy) and is the reference; its tokens are fed to all three sets (teacher forcing), so the
// sets stay on the same text however their own picks go. Set B decodes the rows in decode_packed
// steps; set C decodes one forward per row again, as a control. Reports how often B's and C's
// argmax agree with A's. Some models are not bitwise repeatable (Qwen3.6's MoE among them), so C
// does not agree 100% either; packed decode passes when B agrees about as often as C does. A wrong
// row layout or state (a row reading another's KV or recurrent state) agrees near chance.
//
// With switch_after = K, set B decodes packed for K steps and then one forward per row, as the
// engine does for a batch's last row: a packed step compacts each row's recurrent state to bf16,
// and the per-row forward must read it so.
//
// usage: packed_decode_check <model> <ids_file> [rows] [steps] [prompt_len] [switch_after]
#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "qwen3_gguf_config.h"
#include "qwen_checkpoint.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <model> <ids_file> [rows] [steps] [prompt_len] [switch_after]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int rows = argc > 3 ? std::max(1, atoi(argv[3])) : 16;
    const int steps = argc > 4 ? std::max(1, atoi(argv[4])) : 48;
    const int plen = argc > 5 ? std::max(8, atoi(argv[5])) : 300;
    const int switch_after = argc > 6 ? atoi(argv[6]) : -1;
    std::vector<int> ids;
    {
        std::ifstream f(argv[2]);
        int v;
        while (f >> v) ids.push_back(v);
    }
    const int stride = 61;
    if ((int)ids.size() < (rows - 1) * stride + plen + 8 * rows) {
        printf("[FAIL] need more ids\n");
        return 1;
    }
    if (rows > sparkinfer::kQwen35MaxPackedRows) {
        printf("[FAIL] at most %d rows\n", sparkinfer::kQwen35MaxPackedRows);
        return 1;
    }

    sparkinfer::GGUF g;
    sparkinfer::Qwen35Config cfg;
    QwenCheckpointKind kind{};
    std::string err;
    if (!qwen_checkpoint_open(path, cfg, g, kind, err)) {
        printf("[FAIL] cannot open %s: %s\n", path.c_str(), err.c_str());
        return 1;
    }
    cfg.max_seq = std::max(cfg.max_seq, plen + 8 * rows + steps + 64);
    cfg.eos_id = -1;
    auto rt = sparkinfer::Runtime::create({});
    rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers;
    kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim;
    kvc.block_size = 16;
    kvc.int8_kv = cfg.muse_glimmer ? false : true;
    kvc.layer_slot = sparkinfer::hybrid_kv_layer_slots(cfg.n_layers, cfg.hybrid, cfg.full_attn_interval);
    const int kvL = sparkinfer::kv_slot_count(kvc.layer_slot, cfg.n_layers);
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = (size_t)3 * rows * ((plen + 8 * rows + steps + 64 + 15) / 16 + 4) + 16;
    sparkinfer::KVCacheManager kv(kvc, (size_t)kvL * 2 * epb * 2 * blocks);
    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts;
    mc.top_k = cfg.top_k;
    mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn;
    mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);
    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    if (!qwen_checkpoint_load(model, path, kind)) {
        printf("[FAIL] load %s\n", path.c_str());
        return 1;
    }

    // Prompts of slightly different lengths, so the rows sit at different positions.
    std::vector<int> len(rows), pos(rows);
    std::vector<uint64_t> sa(rows), sb(rows), sc(rows);
    std::vector<int> tok(rows), outa(rows), outb(rows), outc(rows);
    int seed_agree_b = 0, seed_agree_c = 0;
    for (int i = 0; i < rows; ++i) {
        len[i] = plen + 7 * i;
        const int* p = ids.data() + (size_t)i * stride;
        uint64_t* sets[3] = {&sa[i], &sb[i], &sc[i]};
        int seed[3];
        for (int k = 0; k < 3; ++k) {
            *sets[k] = model.open_session(len[i] + steps + 16);
            if (!*sets[k]) { printf("[FAIL] open session\n"); return 1; }
            model.activate_session(*sets[k]);
            model.reset_mrope_offset();
            int done = 0;
            seed[k] = model.ingest_prompt_range(p, 0, len[i], 0, &done);
            if (done != len[i] || seed[k] < 0) { printf("[FAIL] prefill row %d\n", i); return 1; }
        }
        tok[i] = seed[0];
        pos[i] = len[i];
        seed_agree_b += seed[1] == seed[0];
        seed_agree_c += seed[2] == seed[0];
    }
    long agree_b = 0, agree_c = 0, total = 0;
    int packed_steps = 0;
    for (int t = 0; t < steps; ++t) {
        for (int i = 0; i < rows; ++i) {
            model.activate_session(sa[i]);
            outa[i] = model.forward_token(tok[i], pos[i], true, 0.f);
            model.activate_session(sc[i]);
            outc[i] = model.forward_token(tok[i], pos[i], true, 0.f);
        }
        const bool packed_now = switch_after < 0 || t < switch_after;
        if (packed_now && model.decode_packed(tok.data(), pos.data(), sb.data(), rows, outb.data())) {
            ++packed_steps;
        } else {
            for (int i = 0; i < rows; ++i) {
                model.activate_session(sb[i]);
                outb[i] = model.forward_token(tok[i], pos[i], true, 0.f);
            }
        }
        for (int i = 0; i < rows; ++i) {
            agree_b += outb[i] == outa[i];
            agree_c += outc[i] == outa[i];
            ++total;
            tok[i] = outa[i];
            ++pos[i];
        }
    }
    cudaDeviceSynchronize();
    const double rb = (double)agree_b / total, rc = (double)agree_c / total;
    printf("rows %d, steps %d (%d of them packed): seed agreement packed-set %d/%d, control %d/%d\n",
           rows, steps, packed_steps, seed_agree_b, rows, seed_agree_c, rows);
    printf("argmax agreement with one forward per row: packed %.4f, control (one forward per row "
           "again) %.4f\n", rb, rc);
    const int want_packed = switch_after < 0 ? steps : std::min(steps, switch_after);
    const bool ok = packed_steps == want_packed && rb >= rc - 0.02 && rb >= 0.9;
    printf(ok ? "[OK] packed decode matches one forward per row\n"
              : "[FAIL] packed decode disagrees with one forward per row\n");
    return ok ? 0 : 1;
}
