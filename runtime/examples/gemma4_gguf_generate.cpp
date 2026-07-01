// Greedy generation for Gemma 4 26B-A4B from a GGUF file.
//
// Usage: gemma4_gguf_generate <model.gguf> <prompt_tokens_csv> [max_new]

#include "sparkinfer/models/gemma4.h"
#include "sparkinfer/moe/engine.h"
#include "sparkinfer/gguf.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <model.gguf> <tok1,tok2,...> [max_new]\n", argv[0]);
        return 2;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no GPU\n"); return 0;
    }

    const std::string path = argv[1];
    const int max_new = argc > 3 ? atoi(argv[3]) : 32;

    sparkinfer::Gemma4Config cfg;
    sparkinfer::GGUF g;
    if (!g.open(path)) { printf("[FAIL] open gguf\n"); return 1; }
    std::string arch = g.meta_str("general.architecture", "gemma4");
    if (arch.back() != '.') arch += '.';
    auto mi = [&](const std::string& k, long d) { return (int)g.meta_int(arch + k, d); };
    cfg.n_layers = mi("block_count", 30);
    cfg.hidden = mi("embedding_length", 2112);
    cfg.n_experts = mi("expert_count", 128);
    cfg.top_k = mi("expert_used_count", 8);
    cfg.moe_ffn = mi("expert_feed_forward_length", 704);
    cfg.n_shared = g.tensor("blk.0.ffn_gate_shexp.weight") ? 1 : 0;
    const sparkinfer::GGUFTensor* e = g.tensor("token_embd.weight");
    cfg.vocab = e ? (int)e->dims[1] : 262144;
    cfg.max_seq = 2048;

    std::vector<int> prompt;
    {
        std::stringstream ss(argv[2]);
        std::string tok;
        while (std::getline(ss, tok, ',')) prompt.push_back(atoi(tok.c_str()));
    }

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts; mc.top_k = cfg.top_k;
    mc.hidden_dim = cfg.hidden; mc.ffn_dim = cfg.moe_ffn; mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Gemma4Model model(cfg, engine.get());
    if (!model.load_gguf(path)) { printf("[FAIL] load\n"); return 1; }

    auto out = model.generate(prompt, max_new);
    printf("prompt (%zu tokens) -> generated %zu tokens:\n", prompt.size(), out.size());
    for (size_t i = 0; i < out.size(); i++) {
        if (i) printf(",");
        printf("%d", out[i]);
    }
    printf("\n");
    return 0;
}
