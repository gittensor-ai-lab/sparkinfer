// GPU smoke test: ContinuousBatchEngine + per-request sessions on real hardware.
#include "sparkinfer/inference_engine.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/moe/engine.h"
#include "sparkinfer/runtime.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>

static void* rand_bf16(size_t n, float s) {
    std::vector<uint16_t> h(n);
    for (size_t i = 0; i < n; i++) {
        float f = s * (2.f * ((i * 2654435761u + 40503u) % 1000) / 1000.f - 1.f);
        uint32_t b;
        __builtin_memcpy(&b, &f, 4);
        h[i] = (uint16_t)(b >> 16);
    }
    void* d = nullptr;
    cudaMalloc(&d, n * sizeof(uint16_t));
    cudaMemcpy(d, h.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    return d;
}

static void fill_weights(sparkinfer::Qwen35Weights& w, const sparkinfer::Qwen35Config& cfg) {
    const int H = cfg.hidden, Q = cfg.n_q_heads * cfg.head_dim, KV = cfg.n_kv_heads * cfg.head_dim;
    const int E = cfg.n_experts, F = cfg.moe_ffn;
    w.embed_tokens = rand_bf16((size_t)cfg.vocab * H, 1.f);
    w.final_norm = rand_bf16(H, 0.5f);
    w.lm_head = rand_bf16((size_t)H * cfg.vocab, 0.05f);
    w.layers.resize(cfg.n_layers);
    for (int l = 0; l < cfg.n_layers; l++) {
        auto& lw = w.layers[l];
        lw.input_norm = rand_bf16(H, 0.5f);
        lw.wq = rand_bf16((size_t)H * Q, 0.04f);
        lw.wk = rand_bf16((size_t)H * KV, 0.04f);
        lw.wv = rand_bf16((size_t)H * KV, 0.04f);
        lw.wo = rand_bf16((size_t)Q * H, 0.04f);
        lw.q_norm = rand_bf16(cfg.head_dim, 0.5f);
        lw.k_norm = rand_bf16(cfg.head_dim, 0.5f);
        lw.post_attn_norm = rand_bf16(H, 0.5f);
        lw.router_w = rand_bf16((size_t)H * E, 0.1f);
        lw.gate = rand_bf16((size_t)E * H * F, 0.04f);
        lw.up = rand_bf16((size_t)E * H * F, 0.04f);
        lw.down = rand_bf16((size_t)E * F * H, 0.04f);
        lw.shared_gate = rand_bf16((size_t)H * F, 0.04f);
        lw.shared_up = rand_bf16((size_t)H * F, 0.04f);
        lw.shared_down = rand_bf16((size_t)F * H, 0.04f);
    }
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no CUDA device\n");
        return 0;
    }

    auto rt = sparkinfer::Runtime::create({});
    rt->initialize();

    sparkinfer::Qwen35Config cfg;
    cfg.vocab = 2000;
    cfg.hidden = 2048;
    cfg.n_layers = 2;
    cfg.n_q_heads = 16;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 128;
    cfg.n_experts = 8;
    cfg.top_k = 2;
    cfg.n_shared = 1;
    cfg.moe_ffn = 64;
    cfg.max_seq = 128;
    cfg.eos_id = -1;

    sparkinfer::KVCacheConfig kvc;
    kvc.num_layers = cfg.n_layers;
    kvc.num_kv_heads = cfg.n_kv_heads;
    kvc.head_dim = cfg.head_dim;
    kvc.block_size = 16;
    sparkinfer::KVCacheManager kv(kvc, 256ull * 1024 * 1024);

    sparkinfer::moe::MoEConfig mc;
    mc.num_experts = cfg.n_experts;
    mc.top_k = cfg.top_k;
    mc.hidden_dim = cfg.hidden;
    mc.ffn_dim = cfg.moe_ffn;
    mc.num_layers = cfg.n_layers;
    auto engine = sparkinfer::moe::MoEEngine::create(mc);

    sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
    sparkinfer::Qwen35Weights w;
    fill_weights(w, cfg);
    model.set_weights(w);

    sparkinfer::ContinuousBatchEngine batch(&model, &kv, 8);

    sparkinfer::ContinuousBatchEngine::Request a;
    a.prompt = {1, 5, 9};
    a.max_new_tokens = 4;
    auto ra = batch.complete(a);
    if (!ra.error.empty()) {
        printf("[FAIL] request A: %s\n", ra.error.c_str());
        return 1;
    }
    if ((int)ra.tokens.size() != 4) {
        printf("[FAIL] request A: expected 4 tokens, got %zu\n", ra.tokens.size());
        return 1;
    }

    sparkinfer::ContinuousBatchEngine::Request b;
    b.prompt = {2, 6, 10, 14};
    b.max_new_tokens = 6;
    b.priority = 1;
    auto rb = batch.complete(b);
    if (!rb.error.empty()) {
        printf("[FAIL] request B: %s\n", rb.error.c_str());
        return 1;
    }
    if ((int)rb.tokens.size() != 6) {
        printf("[FAIL] request B: expected 6 tokens, got %zu\n", rb.tokens.size());
        return 1;
    }

    for (int id : ra.tokens)
        if (id < 0 || id >= cfg.vocab) {
            printf("[FAIL] token %d out of range\n", id);
            return 1;
        }
    for (int id : rb.tokens)
        if (id < 0 || id >= cfg.vocab) {
            printf("[FAIL] token %d out of range\n", id);
            return 1;
        }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[FAIL] cuda error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("[PASS] batch_engine_gpu_test: A=%zu tokens B=%zu tokens free_kv_blocks=%d\n",
           ra.tokens.size(), rb.tokens.size(), kv.num_free_blocks());
    return 0;
}
