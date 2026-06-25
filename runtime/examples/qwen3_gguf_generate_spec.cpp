// Same as qwen3_gguf_generate.cpp but drives the prompt-lookup speculative
// decode path (generate_speculative) instead of the plain single-token loop,
// so the two can be diffed for output equality and compared for round count.
//
// Usage: qwen3_gguf_generate_spec <model.gguf> <max_new> <draft_k> <id0> [id1 ...]
// (tokenize/detokenize with tools/run_qwen3.py)

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 5) { printf("usage: %s <model.gguf> <max_new> <draft_k> <id0> [id1 ...]\n", argv[0]); return 2; }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { printf("[SKIP] no GPU\n"); return 0; }

    const std::string path = argv[1];
    const int max_new = atoi(argv[2]);
    const int draft_k = atoi(argv[3]);
    std::vector<int> prompt;
    for (int i = 4; i < argc; i++) prompt.push_back(atoi(argv[i]));

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
    printf("arch: %d layers, hidden %d, %dQ/%dKV hd%d, %d experts top-%d, ffn %d, vocab %d\n",
           cfg.n_layers, cfg.hidden, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim,
           cfg.n_experts, cfg.top_k, cfg.moe_ffn, cfg.vocab);

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
    printf("loading GGUF (dense->bf16, experts kept quantized) ...\n");
    if (!model.load_gguf(path)) { printf("[FAIL] load_gguf\n"); return 1; }
    size_t freeb=0, totb=0; cudaMemGetInfo(&freeb, &totb);
    printf("loaded. VRAM used ~%.1f GB. speculative-generating %d tokens (draft_k=%d) from %zu prompt tokens\n",
           (totb - freeb) / 1e9, max_new, draft_k, prompt.size());

    auto out = model.generate_speculative(prompt, max_new, draft_k);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("[FAIL] cuda: %s\n", cudaGetErrorString(e)); return 1; }

    printf("OUTPUT_IDS:");
    for (int id : out) printf(" %d", id);
    printf("\n");
    return 0;
}
