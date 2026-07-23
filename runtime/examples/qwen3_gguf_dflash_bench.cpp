// DFlash throughput bench vs AR for Qwen3.6 + z-lab draft.
//
// Usage:
//   qwen3_gguf_dflash_bench <target.gguf> <draft_dir> [n_tokens] [id0 id1 ...]
//
// Reports tok/s, mean accept τ, TTFT for DFlash and AR.

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
#include <chrono>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <target.gguf> <draft_dir> [n_tokens] [id0 ...]\n", argv[0]);
        return 2;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no GPU\n");
        return 0;
    }

    const std::string gguf = argv[1];
    const std::string draft_dir = argv[2];
    int n_tokens = 64;
    std::vector<int> prompt = {1, 2, 3, 4, 5, 6, 7, 8};  // fallback tiny prompt
    if (argc >= 4) n_tokens = atoi(argv[3]);
    if (argc >= 5) {
        prompt.clear();
        for (int i = 4; i < argc; i++) prompt.push_back(atoi(argv[i]));
    }

    sparkinfer::GGUF g;
    if (!g.open(gguf)) { printf("[FAIL] open %s\n", gguf.c_str()); return 1; }
    sparkinfer::Qwen35Config cfg;
    qwen3_config_from_gguf(g, cfg);
    cfg.max_seq = std::max(2048, (int)prompt.size() + n_tokens + 64);

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
    if (!draft.load(draft_dir)) { printf("[FAIL] draft load\n"); return 1; }
    model.set_dflash_draft(&draft);

    // AR
    setenv("SPARKINFER_DFLASH", "0", 1);
    auto a0 = std::chrono::steady_clock::now();
    auto ar = model.generate(prompt, n_tokens);
    auto a1 = std::chrono::steady_clock::now();
    const double ar_s = std::chrono::duration<double>(a1 - a0).count();
    const double ar_tps = ar.empty() ? 0.0 : (double)ar.size() / ar_s;

    // DFlash
    setenv("SPARKINFER_DFLASH", "1", 1);
    sparkinfer::Qwen35Model::DFlashStats st{};
    auto d0 = std::chrono::steady_clock::now();
    auto df = model.dflash_generate(prompt, n_tokens, &st);
    auto d1 = std::chrono::steady_clock::now();
    const double df_wall = std::chrono::duration<double>(d1 - d0).count();
    const double df_tps = (st.decode_s > 0 && !df.empty())
                              ? (double)df.size() / st.decode_s
                              : (df.empty() ? 0.0 : (double)df.size() / df_wall);

    printf("\n=== sparkinfer DFlash bench ===\n");
    printf("prompt_tokens : %zu\n", prompt.size());
    printf("gen_tokens    : AR=%zu DFlash=%zu\n", ar.size(), df.size());
    printf("AR            : %.2f tok/s  (wall %.3fs)\n", ar_tps, ar_s);
    printf("DFlash        : %.2f tok/s  (decode %.3fs, ttft %.3fs, wall %.3fs)\n",
           df_tps, st.decode_s, st.ttft_s, df_wall);
    printf("mean_accept τ : %.3f  (steps=%d)\n", st.mean_accept, st.steps);
    printf("METRIC AR_TPS %.4f\n", ar_tps);
    printf("METRIC DFLASH_TPS %.4f\n", df_tps);
    printf("METRIC MEAN_ACCEPT %.4f\n", st.mean_accept);
    printf("METRIC TTFT_S %.4f\n", st.ttft_s);
    return 0;
}
