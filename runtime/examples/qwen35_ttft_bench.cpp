// TTFT + decode micro-benchmark on a tiny Qwen3.5-shaped model (random weights).
// Fits on 8 GB GPUs — exercises the real device forward path without a 17 GB GGUF.
//
// Usage: qwen35_ttft_bench [prompt_len] [decode_n]
//   prompt_len  teacher-forced prompt tokens (default 64)
//   decode_n    steady-state decode tokens timed (default 32)

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static void* rand_bf16(size_t n, float s) {
    std::vector<uint16_t> h(n);
    for (size_t i = 0; i < n; i++) {
        float f = s * (2.f * ((i * 2654435761u + 40503u) % 1000) / 1000.f - 1.f);
        uint32_t b; std::memcpy(&b, &f, 4); h[i] = (uint16_t)(b >> 16);
    }
    void* d = nullptr; cudaMalloc(&d, n * sizeof(uint16_t));
    cudaMemcpy(d, h.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    return d;
}

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no CUDA device\n"); return 0;
    }
    const int prompt_len = argc > 1 ? atoi(argv[1]) : 64;
    const int decode_n   = argc > 2 ? atoi(argv[2]) : 32;

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    auto rt = sparkinfer::Runtime::create({}); rt->initialize();

    sparkinfer::Qwen35Config cfg;
    cfg.vocab = 2000; cfg.hidden = 2048; cfg.n_layers = 2;
    cfg.n_q_heads = 16; cfg.n_kv_heads = 2; cfg.head_dim = 128;
    cfg.n_experts = 8; cfg.top_k = 2; cfg.n_shared = 1; cfg.moe_ffn = 64;
    cfg.max_seq = 512; cfg.eos_id = -1;

    const int H = cfg.hidden, Q = cfg.n_q_heads * cfg.head_dim, KV = cfg.n_kv_heads * cfg.head_dim;
    const int E = cfg.n_experts, F = cfg.moe_ffn;

    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers; kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim; kvc.block_size = 16;
    sparkinfer::KVCacheManager kv(kvc, 128ull * 1024 * 1024);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = E; mc.top_k = cfg.top_k; mc.hidden_dim = H; mc.ffn_dim = F; mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());

    sparkinfer::Qwen35Weights w;
    w.embed_tokens = rand_bf16((size_t)cfg.vocab * H, 1.f);
    w.final_norm   = rand_bf16(H, 0.5f);
    w.lm_head      = rand_bf16((size_t)H * cfg.vocab, 0.05f);
    w.layers.resize(cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; l++) {
        auto& lw = w.layers[l];
        lw.input_norm = rand_bf16(H, 0.5f);
        lw.wq = rand_bf16((size_t)H * Q, 0.04f); lw.wk = rand_bf16((size_t)H * KV, 0.04f);
        lw.wv = rand_bf16((size_t)H * KV, 0.04f); lw.wo = rand_bf16((size_t)Q * H, 0.04f);
        lw.q_norm = rand_bf16(cfg.head_dim, 0.5f); lw.k_norm = rand_bf16(cfg.head_dim, 0.5f);
        lw.post_attn_norm = rand_bf16(H, 0.5f);
        lw.router_w = rand_bf16((size_t)H * E, 0.1f);
        lw.gate = rand_bf16((size_t)E * H * F, 0.04f); lw.up = rand_bf16((size_t)E * H * F, 0.04f);
        lw.down = rand_bf16((size_t)E * F * H, 0.04f);
        lw.shared_gate = rand_bf16((size_t)H * F, 0.04f); lw.shared_up = rand_bf16((size_t)H * F, 0.04f);
        lw.shared_down = rand_bf16((size_t)F * H, 0.04f);
    }
    model.set_weights(w);

    std::vector<int> prompt((size_t)prompt_len);
    for (int i = 0; i < prompt_len; i++) prompt[i] = 1 + (i % 100);

    double ttft_ms = model.bench_ttft(prompt);
    double decode_tps = model.bench_decode(4, decode_n);

    printf("\n=== sparkinfer qwen35_ttft_bench ===\n");
    printf("GPU          : %s (sm_%d.%d)\n", prop.name, prop.major, prop.minor);
    printf("prompt_len   : %d\n", prompt_len);
    printf("TTFT         : %.2f ms  (prompt + first sample)\n", ttft_ms);
    printf("decode tg    : %.2f tok/s  (n=%d, bs=1)\n", decode_tps, decode_n);
    return 0;
}
