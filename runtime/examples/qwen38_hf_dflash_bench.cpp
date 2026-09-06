// DFlash (DSpark) throughput bench vs AR for a Qwen3.8 HF/compressed-tensors checkpoint.
//
// Same measurement as qwen3_gguf_dflash_bench, but the target is an NVFP4
// safetensors directory (config.json + model-*.safetensors) instead of a GGUF,
// so it benches exactly what the server serves.
//
//   qwen38_hf_dflash_bench <target_dir> <draft_dir> [n_tokens] [id0 id1 ...]

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/models/dflash_draft.h"
#include "sparkinfer/moe/engine.h"
#include "qwen38_hf_config.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <target_dir> <draft_dir> [n_tokens] [id0 ...]\n", argv[0]);
        return 2;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no GPU\n");
        return 0;
    }

    const std::string model_dir = argv[1];
    const std::string draft_dir = argv[2];
    int n_tokens = 64;
    std::vector<int> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    if (argc >= 4) n_tokens = atoi(argv[3]);
    if (argc >= 5) {
        prompt.clear();
        for (int i = 4; i < argc; i++) prompt.push_back(atoi(argv[i]));
    }

    sparkinfer::Qwen35Config cfg;
    std::string err;
    if (!qwen38_config_from_hf_json(model_dir, cfg, err)) {
        printf("[FAIL] config: %s\n", err.c_str());
        return 1;
    }
    cfg.max_seq = std::max(2048, (int)prompt.size() + n_tokens + 64);

    auto rt = sparkinfer::Runtime::create({});
    rt->initialize();

    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers;
    kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim;
    kvc.block_size = 16;
    kvc.int8_kv = false;
    kvc.layer_slot = sparkinfer::hybrid_kv_layer_slots(cfg.n_layers, cfg.hybrid, cfg.full_attn_interval);
    const int kvL = sparkinfer::kv_slot_count(kvc.layer_slot, cfg.n_layers);
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = (cfg.max_seq + 15) / 16 + 8;
    sparkinfer::KVCacheManager kv(kvc, (size_t)kvL * 2 * epb * 2 * blocks);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts;
    mc.top_k = cfg.top_k;
    mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn;
    mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    if (!model.load_compressed_tensors(model_dir)) {
        printf("[FAIL] load_compressed_tensors\n");
        return 1;
    }

    sparkinfer::DFlashDraftConfig dcfg;
    dcfg.max_seq = cfg.max_seq;
    sparkinfer::DFlashDraftModel draft(dcfg);
    if (!draft.load(draft_dir)) { printf("[FAIL] draft load\n"); return 1; }
    model.set_dflash_draft(&draft);

    // AR
    setenv("SPARKINFER_DFLASH", "0", 1);
    double ar_ttft_s = 0, ar_decode_s = 0;
    auto a0 = std::chrono::steady_clock::now();
    auto ar = model.generate(prompt, n_tokens, nullptr, &ar_ttft_s, &ar_decode_s);
    auto a1 = std::chrono::steady_clock::now();
    const double ar_s = std::chrono::duration<double>(a1 - a0).count();
    const double ar_tps = (ar_decode_s > 0 && !ar.empty()) ? (double)ar.size() / ar_decode_s
                                                           : (ar.empty() ? 0.0 : (double)ar.size() / ar_s);

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

    const size_t ncmp = std::min(ar.size(), df.size());
    size_t agree = 0; long first_diff = -1;
    for (size_t i = 0; i < ncmp; i++) {
        if (ar[i] == df[i]) agree++;
        else if (first_diff < 0) first_diff = (long)i;
    }
    bool identical = (ar.size() == df.size()) && (first_diff < 0);

    printf("\n=== sparkinfer DFlash bench (compressed-tensors target) ===\n");
    printf("prompt_tokens : %zu\n", prompt.size());
    printf("gen_tokens    : AR=%zu DFlash=%zu  identical=%s\n",
           ar.size(), df.size(), identical ? "yes" : "NO");
    printf("AR            : %.2f tok/s  (decode %.3fs, ttft %.3fs, wall %.3fs)\n",
           ar_tps, ar_decode_s, ar_ttft_s, ar_s);
    printf("DFlash        : %.2f tok/s  (decode %.3fs, ttft %.3fs, wall %.3fs)\n",
           df_tps, st.decode_s, st.ttft_s, df_wall);
    printf("mean_accept t : %.3f  (steps=%d)\n", st.mean_accept, st.steps);
    printf("METRIC AR_TPS %.4f\n", ar_tps);
    printf("METRIC DFLASH_TPS %.4f\n", df_tps);
    printf("METRIC SPEEDUP %.4f\n", ar_tps > 0 ? df_tps / ar_tps : 0.0);
    printf("METRIC MEAN_ACCEPT %.4f\n", st.mean_accept);
    printf("METRIC IDENTICAL %d\n", identical ? 1 : 0);
    printf("METRIC AGREE_PCT %.2f\n", ncmp ? 100.0 * (double)agree / (double)ncmp : 0.0);
    printf("METRIC FIRST_DIFF %ld\n", first_diff);
    return 0;
}
