// Correctness A/B for batched prefill: does prefill_batched() leave the KV cache and
// Gated-DeltaNet state such that a subsequent decode matches the token-by-token fill?
//
// Fills the cache two ways over the SAME prompt, then teacher-forces the SAME continuation
// tokens and compares the per-position next-token logits:
//   TOKEN   : forward_token() over the whole prompt (the reference path)
//   BATCHED : prefill_batched() over the prompt
// Reports top-1 agreement and mean KL(token || batched) over the continuation positions.
//
// Usage: qwen3_gguf_prefill_check <model.gguf> [prefix_len=4096] [cont_len=16]

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

static int argmax(const std::vector<float>& v) {
    int b = 0; for (int i = 1; i < (int)v.size(); i++) if (v[i] > v[b]) b = i; return b;
}
// KL(p||q) where p,q are softmax(la), softmax(lb).
static double kl(const std::vector<float>& la, const std::vector<float>& lb) {
    const int V = (int)la.size();
    float ma = la[0], mb = lb[0];
    for (int i = 1; i < V; i++) { ma = std::max(ma, la[i]); mb = std::max(mb, lb[i]); }
    double za = 0, zb = 0;
    for (int i = 0; i < V; i++) { za += std::exp((double)la[i] - ma); zb += std::exp((double)lb[i] - mb); }
    const double lza = ma + std::log(za), lzb = mb + std::log(zb);
    double d = 0;
    for (int i = 0; i < V; i++) {
        const double lp = (double)la[i] - lza;
        const double lq = (double)lb[i] - lzb;
        const double p = std::exp(lp);
        if (p > 1e-12) d += p * (lp - lq);
    }
    return d;
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: %s <model.gguf> [prefix_len] [cont_len]\n", argv[0]); return 2; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }
    const std::string path = argv[1];
    const int P = argc > 2 ? atoi(argv[2]) : 4096;
    const int C = argc > 3 ? atoi(argv[3]) : 16;

    sparkinfer::GGUF g;
    if (!g.open(path)) { printf("[FAIL] open %s\n", path.c_str()); return 1; }
    sparkinfer::Qwen35Config cfg;
    qwen3_config_from_gguf(g, cfg);
    cfg.max_seq = P + C + 16;

    auto rt = sparkinfer::Runtime::create({}); rt->initialize();
    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers; kvc.num_kv_heads = cfg.n_kv_heads; kvc.head_dim = cfg.head_dim; kvc.block_size = 16;
    { const char* e = getenv("SPARKINFER_KV_INT8"); kvc.int8_kv = e ? (e[0] != '0') : true; }  // batched prefill uses int8 KV
    const size_t epb = (size_t)16 * cfg.n_kv_heads * cfg.head_dim;
    const size_t blocks = (cfg.max_seq + 15) / 16 + 8;
    sparkinfer::KVCacheManager kv(kvc, (size_t)cfg.n_layers * 2 * epb * 2 * blocks);
    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts; mc.top_k = cfg.top_k; mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn; mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);
    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    if (!model.load_gguf(path)) { printf("[FAIL] load_gguf\n"); return 1; }

    std::vector<int> prompt(P), cont(C);
    for (int i = 0; i < P; i++) prompt[i] = 100 + (i % 20000);
    for (int j = 0; j < C; j++) cont[j] = 100 + ((P + j) % 20000);

    const int V = cfg.vocab;
    std::vector<std::vector<float>> LA(C, std::vector<float>(V)), LB(C, std::vector<float>(V));
    std::vector<int> amA(C), amB(C);

    // ---- TOKEN path (reference) ----
    if (!kv.allocate(0, cfg.max_seq)) { printf("[FAIL] kv allocate\n"); return 1; }
    for (int i = 0; i < P; i++) model.forward_token(prompt[i], i);
    for (int j = 0; j < C; j++) { amA[j] = model.forward_token(cont[j], P + j); model.copy_logits(LA[j].data()); }
    kv.free(0);

    // ---- BATCHED path ----
    if (!kv.allocate(0, cfg.max_seq)) { printf("[FAIL] kv allocate\n"); return 1; }
    int seed = model.prefill_batched(prompt.data(), P);
    if (seed < 0) { printf("[FAIL] prefill_batched unsupported (seed=%d)\n", seed); return 1; }
    for (int j = 0; j < C; j++) { amB[j] = model.forward_token(cont[j], P + j); model.copy_logits(LB[j].data()); }
    kv.free(0);

    int top1 = 0; double sumkl = 0;
    for (int j = 0; j < C; j++) {
        if (argmax(LA[j]) == argmax(LB[j])) top1++;
        sumkl += kl(LA[j], LB[j]);
    }
    printf("prefix=%d cont=%d int8_kv=%d\n", P, C, kvc.int8_kv ? 1 : 0);
    printf("TOP1  %d/%d %.4f\n", top1, C, (double)top1 / C);
    printf("KL    %.5f (mean over %d positions)\n", sumkl / C, C);
    printf("seed(batched)=%d\n", seed);
    return 0;
}
