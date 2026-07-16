// GPU smoke test for the full Qwen3.5 model path on real hardware.
//
// Builds a small Qwen3.5-shaped model with random device weights and runs the
// real generate() loop — embedding, per-head QK-norm, RoPE, paged GQA decode,
// routed top-k MoE + shared expert, LM head, greedy argmax — entirely on the
// GPU. Validates that the whole Qwen35Model device path executes and produces
// in-range token ids. Skips cleanly without a CUDA device.
//
// Uses the gqa8 attention dims (16 Q / 2 KV heads, head_dim 128) so the real
// specialized kernel is exercised.

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

static void* rand_bf16(size_t n, float s) {
    std::vector<uint16_t> h(n);
    for (size_t i = 0; i < n; i++) {
        float f = s * (2.f * ((i * 2654435761u + 40503u) % 1000) / 1000.f - 1.f);
        uint32_t b; __builtin_memcpy(&b, &f, 4); h[i] = (uint16_t)(b >> 16);
    }
    void* d = nullptr; cudaMalloc(&d, n * sizeof(uint16_t));
    cudaMemcpy(d, h.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    return d;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no CUDA device — qwen35_gpu_test needs a GPU\n"); return 0;
    }
    auto rt = sparkinfer::Runtime::create({}); rt->initialize();

    sparkinfer::Qwen35Config cfg;
    cfg.vocab = 2000; cfg.hidden = 2048; cfg.n_layers = 2;
    cfg.n_q_heads = 16; cfg.n_kv_heads = 2; cfg.head_dim = 128;   // gqa8 kernel
    cfg.n_experts = 8; cfg.top_k = 2; cfg.n_shared = 1; cfg.moe_ffn = 64;
    cfg.max_seq = 128; cfg.eos_id = -1;                            // don't early-stop

    const int H = cfg.hidden, Q = cfg.n_q_heads*cfg.head_dim, KV = cfg.n_kv_heads*cfg.head_dim;
    const int E = cfg.n_experts, F = cfg.moe_ffn;

    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers; kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim; kvc.block_size = 16;
    sparkinfer::KVCacheManager kv(kvc, 128ull*1024*1024);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = E; mc.top_k = cfg.top_k; mc.hidden_dim = H; mc.ffn_dim = F; mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());

    sparkinfer::Qwen35Weights w;
    w.embed_tokens = rand_bf16((size_t)cfg.vocab*H, 1.f);
    w.final_norm   = rand_bf16(H, 0.5f);
    w.lm_head      = rand_bf16((size_t)H*cfg.vocab, 0.05f);
    w.layers.resize(cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; l++) {
        auto& lw = w.layers[l];
        lw.input_norm = rand_bf16(H,0.5f);
        lw.wq = rand_bf16((size_t)H*Q,0.04f); lw.wk = rand_bf16((size_t)H*KV,0.04f);
        lw.wv = rand_bf16((size_t)H*KV,0.04f); lw.wo = rand_bf16((size_t)Q*H,0.04f);
        lw.q_norm = rand_bf16(cfg.head_dim,0.5f); lw.k_norm = rand_bf16(cfg.head_dim,0.5f);
        lw.post_attn_norm = rand_bf16(H,0.5f);
        lw.router_w = rand_bf16((size_t)H*E,0.1f);
        lw.gate = rand_bf16((size_t)E*H*F,0.04f); lw.up = rand_bf16((size_t)E*H*F,0.04f);
        lw.down = rand_bf16((size_t)E*F*H,0.04f);
        lw.shared_gate = rand_bf16((size_t)H*F,0.04f); lw.shared_up = rand_bf16((size_t)H*F,0.04f);
        lw.shared_down = rand_bf16((size_t)F*H,0.04f);
    }
    model.set_weights(w);

    std::vector<int> prompt = {1, 5, 9, 13};
    auto out = model.generate(prompt, 8);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("[FAIL] cuda error: %s\n", cudaGetErrorString(err)); return 1; }
    if ((int)out.size() != 8) { printf("[FAIL] expected 8 tokens, got %zu\n", out.size()); return 1; }
    for (int id : out) if (id < 0 || id >= cfg.vocab) { printf("[FAIL] token %d out of range\n", id); return 1; }

    printf("[PASS] qwen35_gpu_test: generated 8 tokens:");
    for (int id : out) printf(" %d", id);
    printf("\n");
    return 0;
}
