// Teacher-forced scoring for accuracy checks (perplexity / token-match / KL).
// Feeds a fixed token sequence; at each position reports the model's next-token
// argmax, the logprob of the actual next token, and the top-K logprobs.
//
// Usage: qwen3_gguf_score <model.gguf> <topk> <id0> <id1> ...
// Output:  per-position  "S i=<i> tgt=<t> am=<argmax> lp=<logprob_tgt> top=id:lp,..."
//          summary       "PPL <perplexity> over <N> positions"
//          summary       "ARGMATCH <m>/<N> <fraction>"   (am == fed next token)

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "qwen3_gguf_config.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: %s <model.gguf> <topk> <id0> <id1> ...\n", argv[0]); return 2; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }

    const std::string path = argv[1];
    const int topk = atoi(argv[2]);
    std::vector<int> toks;
    for (int i = 3; i < argc; i++) toks.push_back(atoi(argv[i]));
    if ((int)toks.size() < 2) { printf("[FAIL] need >= 2 tokens\n"); return 1; }

    sparkinfer::GGUF g;
    if (!g.open(path)) { printf("[FAIL] cannot open %s\n", path.c_str()); return 1; }
    sparkinfer::Qwen35Config cfg;
    qwen3_config_from_gguf(g, cfg);
    cfg.max_seq    = 2048;
    if (const char* e = getenv("SPARKINFER_SCORE_MAX_SEQ")) {
        int v = atoi(e);
        if (v > cfg.max_seq) cfg.max_seq = v;
    }

    auto rt = sparkinfer::Runtime::create({}); rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers; kvc.num_kv_heads = cfg.n_kv_heads; kvc.head_dim = cfg.head_dim; kvc.block_size = 16;
    // int8 KV is the Qwen3-MoE head_dim=128 tensor-core path; Qwen3.6 (hybrid, gated head_dim=256)
    // writes bf16 KV. Scoring with int8 KV on the hybrid model corrupts attention -> false accuracy
    // divergence vs llama.cpp. Mirror generate.cpp: bf16 KV for hybrid.
    // Context-adaptive int8 KV for hybrid (>= 4k fed length) so accuracy gate matches bench at 4k.
    { const char* e = getenv("SPARKINFER_KV_INT8");
      kvc.int8_kv = e ? (e[0] != '0') : (cfg.hybrid ? ((argc - 3) >= 4096) : true); }
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = (cfg.max_seq + 15) / 16 + 8;
    sparkinfer::KVCacheManager kv(kvc, (size_t)cfg.n_layers * 2 * epb * 2 * blocks);
    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts; mc.top_k = cfg.top_k; mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn; mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);
    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    if (!model.load_gguf(path)) { printf("[FAIL] load_gguf\n"); return 1; }
    // forward_token() reads the KV block table for seq 0 (the model's default
    // seq_id); generate() allocates it — do the same here before scoring.
    if (!kv.allocate(0, cfg.max_seq)) { printf("[FAIL] KV allocate\n"); return 1; }

    int prefix_len = 0;
    if (const char* pl = getenv("SPARKINFER_PREFIX_CACHE_LEN")) {
        prefix_len = atoi(pl);
        if (prefix_len > 0 && prefix_len < (int)toks.size()) {
            std::vector<int> prefix(toks.begin(), toks.begin() + prefix_len);
            if (!model.cache_prefix(prefix)) { printf("[FAIL] cache_prefix(%d)\n", prefix_len); return 1; }
        } else prefix_len = 0;
    }

    const int V = cfg.vocab;
    std::vector<float> lg(V);
    std::vector<int> idx(V);
    double nll = 0.0; int scored = 0, ammatch = 0;
    const int K = std::min(topk, V);
    const size_t i0 = prefix_len > 0 ? (size_t)prefix_len : 0;

    for (size_t i = i0; i + 1 < toks.size(); i++) {
        int am = model.forward_token(toks[i], (int)i);   // logits predict token i+1
        model.copy_logits(lg.data());
        float mx = lg[0];
        for (int v = 1; v < V; v++) mx = std::max(mx, lg[v]);
        double se = 0.0;
        for (int v = 0; v < V; v++) se += std::exp((double)lg[v] - mx);
        const double lse = mx + std::log(se);            // log-sum-exp normalizer
        const int tgt = toks[i + 1];
        const double lp_tgt = (double)lg[tgt] - lse;
        nll += -lp_tgt; scored++;
        if (am == tgt) ammatch++;

        for (int v = 0; v < V; v++) idx[v] = v;
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b){ return lg[a] > lg[b]; });
        printf("S i=%zu tgt=%d am=%d lp=%.6f top=", i, tgt, am, lp_tgt);
        for (int k = 0; k < K; k++) printf("%s%d:%.6f", k ? "," : "", idx[k], (double)lg[idx[k]] - lse);
        printf("\n");
    }
    kv.free(0);
    printf("PPL %.5f over %d positions\n", std::exp(nll / std::max(1, scored)), scored);
    printf("ARGMATCH %d/%d %.4f\n", ammatch, scored, (double)ammatch / std::max(1, scored));
    return 0;
}
