// Decode-throughput benchmark for Gemma 4 26B-A4B.
//
// Usage: gemma4_gguf_bench <model.gguf | weight_dir> [n_tokens]

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/gemma4.h"
#include "sparkinfer/moe/engine.h"
#include "sparkinfer/gpu_stats.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>
#include <unordered_map>

static bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

static sparkinfer::Gemma4Config cfg_from_gguf(const std::string& path) {
    sparkinfer::Gemma4Config cfg;
    sparkinfer::GGUF g;
    if (!g.open(path)) return cfg;
    std::string arch = g.meta_str("general.architecture", "gemma4");
    if (arch.back() != '.') arch += '.';
    auto mi = [&](const std::string& k, long d) { return (int)g.meta_int(arch + k, d); };
    cfg.n_layers          = mi("block_count", 30);
    cfg.hidden            = mi("embedding_length", 2112);
    cfg.n_q_heads         = mi("attention.head_count", 16);
    cfg.local_n_kv_heads  = mi("attention.head_count_kv", 8);
    cfg.global_n_kv_heads = mi("attention.head_count_kv_global", 2);
    cfg.local_head_dim    = mi("attention.key_length", 256);
    cfg.global_head_dim   = mi("attention.key_length_global", 512);
    cfg.n_experts         = mi("expert_count", 128);
    cfg.top_k             = mi("expert_used_count", 8);
    cfg.moe_ffn           = mi("expert_feed_forward_length", 704);
    cfg.local_rope_theta  = (float)g.meta_float(arch + "rope.freq_base", 1e4);
    cfg.global_rope_theta = (float)g.meta_float(arch + "rope.freq_base_global", 1e6);
    cfg.rms_eps           = (float)g.meta_float(arch + "attention.layer_norm_rms_epsilon", 1e-6);
    const sparkinfer::GGUFTensor* e = g.tensor("token_embd.weight");
    cfg.vocab = e ? (int)e->dims[1] : 262144;
    cfg.n_shared = g.tensor("blk.0.ffn_gate_shexp.weight") ? 1 : 0;
    return cfg;
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <model.gguf|weight_dir> [n_tokens]\n", argv[0]); return 2; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }

    const std::string path = argv[1];
    const int n_tokens = argc > 2 ? atoi(argv[2]) : 64;
    const bool gguf_mode = ends_with(path, ".gguf");

    sparkinfer::Gemma4Config cfg;
    if (gguf_mode) cfg = cfg_from_gguf(path);
    else {
        std::ifstream f(path + "/config.txt");
        std::string line;
        std::unordered_map<std::string, std::string> m;
        while (std::getline(f, line)) {
            auto p = line.find('=');
            if (p != std::string::npos) m[line.substr(0, p)] = line.substr(p + 1);
        }
        auto gi = [&](const char* k, int d) {
            auto it = m.find(k); return it == m.end() ? d : atoi(it->second.c_str());
        };
        cfg.vocab = gi("vocab", 262144); cfg.hidden = gi("hidden", 2112);
        cfg.n_layers = gi("n_layers", 30); cfg.n_experts = gi("n_experts", 128);
        cfg.top_k = gi("top_k", 8); cfg.moe_ffn = gi("moe_ffn", 704);
        cfg.n_shared = gi("n_shared", 1);
    }
    cfg.max_seq = 2048;

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts; mc.top_k = cfg.top_k;
    mc.hidden_dim = cfg.hidden; mc.ffn_dim = cfg.moe_ffn; mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Gemma4Model model(cfg, engine.get());
    printf("loading %s (%s) ...\n", path.c_str(), gguf_mode ? "native GGUF" : "bf16");
    bool ok = gguf_mode ? model.load_gguf(path) : model.load_weights(path);
    if (!ok) { printf("[FAIL] load\n"); return 1; }

    size_t freeb = 0, totb = 0;
    cudaMemGetInfo(&freeb, &totb);
    double toks = model.bench_decode(8, n_tokens);
    auto gpu = sparkinfer::query_gpu_stats();

    printf("\n=== sparkinfer Gemma4 bench (%s) ===\n", gguf_mode ? "Q4_K_M native" : "bf16");
    printf("model        : Gemma4-26B-A4B  (%d layers, 5L1G, %d experts top-%d)\n",
           cfg.n_layers, cfg.n_experts, cfg.top_k);
    printf("VRAM used    : %.1f GB\n", (totb - freeb) / 1e9);
    printf("decode tg    : %.2f tok/s  (%.1f ms/token, n=%d, bs=1)\n",
           toks, toks > 0 ? 1000.0 / toks : 0.0, n_tokens);
    if (gpu.valid && gpu.temp_c >= 0) {
        printf("GPU          : %d°C", gpu.temp_c);
        if (gpu.power_w >= 0) printf(" · %d W", gpu.power_w);
        if (gpu.sm_clock_mhz >= 0) printf(" · %d MHz", gpu.sm_clock_mhz);
        printf(" (peak under load)\n");
    }
    return 0;
}
