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
    const char* A = "qwen3moe.";
    auto mi = [&](const std::string& k, long d){ return g.meta_int(A + k, d); };
    sparkinfer::Qwen35Config cfg;
    cfg.n_layers   = (int)mi("block_count", 48);
    cfg.hidden     = (int)mi("embedding_length", 2048);
    cfg.n_q_heads  = (int)mi("attention.head_count", 32);
    cfg.n_kv_heads = (int)mi("attention.head_count_kv", 4);
    cfg.head_dim   = (int)mi("attention.key_length", 128);
    cfg.n_experts  = (int)mi("expert_count", 128);
    cfg.top_k      = (int)mi("expert_used_count", 8);
    cfg.moe_ffn    = (int)mi("expert_feed_forward_length", 768);
    cfg.rope_theta = (float)g.meta_float(std::string(A) + "rope.freq_base", 1e6);
    cfg.rms_eps    = (float)g.meta_float(std::string(A) + "attention.layer_norm_rms_epsilon", 1e-6);
    cfg.eos_id     = (int)g.meta_int("tokenizer.ggml.eos_token_id", 151645);
    cfg.n_shared   = 0;
    const sparkinfer::GGUFTensor* emb = g.tensor("token_embd.weight");
    cfg.vocab      = emb ? (int)emb->dims[1] : 151936;
    cfg.max_seq    = 2048;
    if (const char* e = getenv("SPARKINFER_SCORE_MAX_SEQ")) {
        int v = atoi(e);
        if (v > cfg.max_seq) cfg.max_seq = v;
    }

    auto rt = sparkinfer::Runtime::create({}); rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers; kvc.num_kv_heads = cfg.n_kv_heads; kvc.head_dim = cfg.head_dim; kvc.block_size = 16;
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

    const int V = cfg.vocab;
    std::vector<float> lg(V);
    std::vector<int> idx(V);
    double nll = 0.0; int scored = 0, ammatch = 0;
    const int K = std::min(topk, V);

    for (size_t i = 0; i + 1 < toks.size(); i++) {
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
