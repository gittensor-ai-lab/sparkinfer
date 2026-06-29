// Qwen3.5 greedy generation demo.
//
// Loads converted weights (see tools/convert_qwen35.py) and greedily generates
// from a prompt given as token IDs. Tokenization is left to the HF tokenizer
// (run it in Python to get IDs and to decode the output IDs); this binary is the
// pure on-device model runner.
//
// Usage: qwen35_generate <weight_dir> <max_new_tokens> <id0> <id1> ...
// Requires an RTX 5090 (sm_120) and the ~20 GB converted weights.

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s <weight_dir> <max_new> <id0> [id1 ...]\n", argv[0]); return 2; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no CUDA device — qwen35_generate needs an RTX 5090\n"); return 0;
    }

    const std::string dir = argv[1];
    const int max_new = atoi(argv[2]);
    std::vector<int> prompt;
    for (int i = 3; i < argc; i++) prompt.push_back(atoi(argv[i]));

    auto rt = sparkinfer::Runtime::create({});
    rt->initialize();

    sparkinfer::RuntimeConfig rtcfg;
    rtcfg.device_id = 0;
    // Optional: SPARKINFER_EXPERT_CACHE_MB limits GPU-resident MoE layer expert tensors.
    if (const char* mb = std::getenv("SPARKINFER_EXPERT_CACHE_MB")) {
        const size_t n = (size_t)std::max(0, atoi(mb));
        rtcfg.expert_cache_bytes = n * 1024u * 1024u;
    }

    sparkinfer::Qwen35Config cfg;          // full Qwen3.5-35B-A3B defaults
    cfg.max_seq = 2048;

    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers; kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim; kvc.block_size = 16;
    sparkinfer::KVCacheManager kv(kvc, 512ull * 1024 * 1024);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts; mc.top_k = cfg.top_k; mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn; mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc, rtcfg.expert_cache_bytes);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    if (!model.load_weights(dir)) { printf("[FAIL] could not load weights from %s\n", dir.c_str()); return 1; }

    auto out = model.generate(prompt, max_new);

    printf("generated %zu tokens:", out.size());
    for (int id : out) printf(" %d", id);
    printf("\n");
    return 0;
}
