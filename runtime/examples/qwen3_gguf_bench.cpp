// Decode-throughput benchmark for the sparkinfer Qwen3 runtime.
// Reports steady-state single-stream generation tokens/sec, to compare against
// llama.cpp's `llama-bench` tg number on the same model + GPU.
//
// Usage: qwen3_gguf_bench <model.gguf | weight_dir> [n_tokens] [context_tokens]
//   *.gguf  -> native load (experts kept quantized, Q4_K_M-sized)
//   dir     -> bf16 weights from convert_gguf.py (reads config.txt)

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>
#include <unordered_map>
#include <algorithm>

static bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <model.gguf|weight_dir> [n_tokens] [context_tokens]\n", argv[0]); return 2; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }
    const std::string path = argv[1];
    const int n_tokens = argc > 2 ? atoi(argv[2]) : 64;
    const int context_tokens = argc > 3 ? atoi(argv[3]) : 0;
    const bool gguf_mode = ends_with(path, ".gguf");

    sparkinfer::Qwen35Config cfg;
    if (gguf_mode) {
        sparkinfer::GGUF g; if (!g.open(path)) { printf("[FAIL] open gguf\n"); return 1; }
        const char* A = "qwen3moe.";
        auto mi = [&](const std::string& k, long d){ return (int)g.meta_int(A + k, d); };
        cfg.n_layers=mi("block_count",48); cfg.hidden=mi("embedding_length",2048);
        cfg.n_q_heads=mi("attention.head_count",32); cfg.n_kv_heads=mi("attention.head_count_kv",4);
        cfg.head_dim=mi("attention.key_length",128); cfg.n_experts=mi("expert_count",128);
        cfg.top_k=mi("expert_used_count",8); cfg.moe_ffn=mi("expert_feed_forward_length",768);
        cfg.rope_theta=(float)g.meta_float(std::string(A)+"rope.freq_base",1e6);
        cfg.rms_eps=(float)g.meta_float(std::string(A)+"attention.layer_norm_rms_epsilon",1e-6);
        cfg.n_shared=0; const sparkinfer::GGUFTensor* e=g.tensor("token_embd.weight");
        cfg.vocab = e ? (int)e->dims[1] : 151936;
    } else {
        std::ifstream f(path + "/config.txt"); std::string line;
        std::unordered_map<std::string,std::string> m;
        while (std::getline(f, line)) { auto p=line.find('='); if(p!=std::string::npos) m[line.substr(0,p)]=line.substr(p+1); }
        auto gi=[&](const char*k,int d){auto it=m.find(k);return it==m.end()?d:atoi(it->second.c_str());};
        auto gf=[&](const char*k,float d){auto it=m.find(k);return it==m.end()?d:(float)atof(it->second.c_str());};
        cfg.vocab=gi("vocab",151936); cfg.hidden=gi("hidden",2048); cfg.n_layers=gi("n_layers",48);
        cfg.n_q_heads=gi("n_q_heads",32); cfg.n_kv_heads=gi("n_kv_heads",4); cfg.head_dim=gi("head_dim",128);
        cfg.n_experts=gi("n_experts",128); cfg.top_k=gi("top_k",8); cfg.n_shared=gi("n_shared",0);
        cfg.moe_ffn=gi("moe_ffn",768); cfg.rope_theta=gf("rope_theta",1e6f); cfg.rms_eps=gf("rms_eps",1e-6f);
    }
    cfg.max_seq = std::max(2048, context_tokens + n_tokens + 16);
    if (const char* e = getenv("SPARKINFER_BENCH_MAX_SEQ")) {
        int v = atoi(e);
        if (v > cfg.max_seq) cfg.max_seq = v;
    }

    auto rt = sparkinfer::Runtime::create({}); rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers=cfg.n_layers; kvc.num_kv_heads=cfg.n_kv_heads; kvc.head_dim=cfg.head_dim; kvc.block_size=16;
    // int8 KV pays off only once the halved long-context read outweighs its fixed per-token write cost,
    // so enable it context-adaptively (>= 8k) by default; short contexts stay bf16 (no regression).
    // SPARKINFER_KV_INT8=1/0 forces it on/off regardless.
    { const char* e8 = getenv("SPARKINFER_KV_INT8"); kvc.int8_kv = e8 ? (e8[0] != '0') : (context_tokens >= 8192); }
    const size_t epb=(size_t)16*cfg.n_kv_heads*cfg.head_dim, blocks=(cfg.max_seq+15)/16+8;
    sparkinfer::KVCacheManager kv(kvc, (size_t)cfg.n_layers*2*epb*2*blocks);
    sparkinfer::moe::MoEConfig mc;
    mc.num_experts=cfg.n_experts; mc.top_k=cfg.top_k; mc.hidden_dim=cfg.hidden; mc.ffn_dim=cfg.moe_ffn; mc.num_layers=cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    printf("loading %s (%s) ...\n", path.c_str(), gguf_mode ? "native GGUF, experts quantized" : "bf16");
    bool ok = gguf_mode ? model.load_gguf(path) : model.load_weights(path);
    if (!ok) { printf("[FAIL] load\n"); return 1; }
    size_t freeb=0, totb=0; cudaMemGetInfo(&freeb,&totb);

    double toks = model.bench_decode(8, n_tokens, context_tokens);
    auto gpu = sparkinfer::query_gpu_stats();   // sampled right after the decode loop — near peak heat
    printf("\n=== sparkinfer bench (%s) ===\n", gguf_mode ? "Q4_K_M native" : "bf16");
    printf("model        : Qwen3-30B-A3B  (%d layers, %d experts top-%d)\n", cfg.n_layers, cfg.n_experts, cfg.top_k);
    printf("VRAM used    : %.1f GB\n", (totb - freeb) / 1e9);
    printf("max seq      : %d\n", cfg.max_seq);
    printf("decode tg    : %.2f tok/s  (%.1f ms/token, n=%d, ctx=%d, bs=1)\n",
           toks, 1000.0 / toks, n_tokens, context_tokens);
    if (gpu.valid && gpu.temp_c >= 0) {
        printf("GPU          : %d°C", gpu.temp_c);
        if (gpu.power_w      >= 0) printf(" · %d W", gpu.power_w);
        if (gpu.sm_clock_mhz >= 0) printf(" · %d MHz", gpu.sm_clock_mhz);
        printf(" (peak under load)\n");
    }
    return 0;
}
