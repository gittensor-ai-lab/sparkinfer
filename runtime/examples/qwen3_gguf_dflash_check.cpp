// DFlash correctness check: greedy AR vs DFlash must be byte-identical (SPEC_AGREE).
//
// Usage:
//   qwen3_gguf_dflash_check <target.gguf> <draft_dir> <max_new> <id0> [id1 ...]
//
// Prints METRIC lines and VERDICT PASS/FAIL.

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/models/dflash_draft.h"
#include "sparkinfer/moe/engine.h"
#include "qwen3_gguf_config.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("usage: %s <target.gguf> <draft_dir> <max_new> <id0> [id1 ...]\n", argv[0]);
        return 2;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no GPU\n");
        return 0;
    }

    const std::string gguf = argv[1];
    const std::string draft_dir = argv[2];
    const int max_new = atoi(argv[3]);
    std::vector<int> prompt;
    for (int i = 4; i < argc; i++) prompt.push_back(atoi(argv[i]));

    sparkinfer::GGUF g;
    if (!g.open(gguf)) { printf("[FAIL] cannot open %s\n", gguf.c_str()); return 1; }
    sparkinfer::Qwen35Config cfg;
    qwen3_config_from_gguf(g, cfg);
    cfg.max_seq = std::max(2048, (int)prompt.size() + max_new + 64);

    auto rt = sparkinfer::Runtime::create({});
    rt->initialize();

    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers;
    kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim;
    kvc.block_size = 16;
    kvc.int8_kv = false;
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = (cfg.max_seq + 15) / 16 + 8;
    sparkinfer::KVCacheManager kv(kvc, (size_t)cfg.n_layers * 2 * epb * 2 * blocks);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts;
    mc.top_k = cfg.top_k;
    mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn;
    mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    if (!model.load_gguf(gguf)) { printf("[FAIL] load_gguf\n"); return 1; }

    sparkinfer::DFlashDraftConfig dcfg;
    sparkinfer::DFlashDraftModel draft(dcfg);
    if (!draft.load(draft_dir)) { printf("[FAIL] load draft %s\n", draft_dir.c_str()); return 1; }
    model.set_dflash_draft(&draft);

    // AR baseline (DFlash disabled)
    setenv("SPARKINFER_DFLASH", "0", 1);
    std::vector<int> ar = model.generate(prompt, max_new);
    printf("AR tokens (%zu):", ar.size());
    for (int t : ar) printf(" %d", t);
    printf("\n");

    // DFlash
    setenv("SPARKINFER_DFLASH", "1", 1);
    sparkinfer::Qwen35Model::DFlashStats st{};
    std::vector<int> df = model.dflash_generate(prompt, max_new, &st);
    printf("DFlash tokens (%zu):", df.size());
    for (int t : df) printf(" %d", t);
    printf("\n");

    const size_t n = std::min(ar.size(), df.size());
    size_t agree = 0;
    for (size_t i = 0; i < n; i++) if (ar[i] == df[i]) agree++;
    const double rate = n ? (double)agree / (double)n : 0.0;
    printf("METRIC SPEC_AGREE %zu/%zu = %.4f\n", agree, n, rate);
    printf("METRIC MEAN_ACCEPT %.3f steps=%d\n", st.mean_accept, st.steps);
    printf("METRIC TTFT_S %.4f DECODE_S %.4f\n", st.ttft_s, st.decode_s);

    const bool pass = (ar.size() == df.size()) && (agree == n) && n > 0;
    printf("VERDICT %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
