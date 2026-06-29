// GPU smoke test — MoE forward with a tight expert-cache byte budget.

#include "sparkinfer/moe/engine.h"
#include "sparkinfer/kernels/moe.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

using sparkinfer::moe::expert_layer_bytes;
using sparkinfer::moe::MoEConfig;
using sparkinfer::moe::LayerWeights;

static uint16_t f2bf16(float f) {
    uint32_t b;
    __builtin_memcpy(&b, &f, 4);
    return (uint16_t)(b >> 16);
}

static void* dev_rand_bf16(size_t n, float s) {
    std::vector<uint16_t> h(n);
    for (size_t i = 0; i < n; i++)
        h[i] = f2bf16(s * (2.f * ((i * 1103515245u + 12345u) % 1000) / 1000.f - 1.f));
    void* d = nullptr;
    cudaMalloc(&d, n * sizeof(uint16_t));
    cudaMemcpy(d, h.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    return d;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no CUDA device — expert_cache_gpu_test requires a GPU\n");
        return 0;
    }

    const int E = 4, K = 2, H = 64, F = 32, layers = 3;
    MoEConfig mc{};
    mc.num_experts = E;
    mc.top_k = K;
    mc.hidden_dim = H;
    mc.ffn_dim = F;
    mc.num_layers = layers;
    mc.expert_cache_slots = E;
    mc.async_expert_prefetch = true;
    mc.async_prefetch_depth = 2;

    // Only one layer's expert tensors fit in the cache budget.
    const size_t budget = expert_layer_bytes(E, H, F);
    auto engine = sparkinfer::moe::MoEEngine::create(mc, budget);

    std::vector<LayerWeights> w(layers);
    for (int l = 0; l < layers; l++) {
        w[l].router_w = dev_rand_bf16((size_t)H * E, 0.1f);
        w[l].gate_w   = dev_rand_bf16((size_t)E * H * F, 0.05f);
        w[l].up_w     = dev_rand_bf16((size_t)E * H * F, 0.05f);
        w[l].down_w   = dev_rand_bf16((size_t)E * F * H, 0.05f);
        engine->set_layer_weights(l, w[l]);
    }

    void* input = dev_rand_bf16(H, 1.f);
    void* output = dev_rand_bf16(H, 0.f);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int l = 0; l < layers; l++)
        engine->forward(input, output, 1, l, stream);
    cudaStreamSynchronize(stream);

    auto st = engine->cache_stats();
    if (st.layer_misses == 0) {
        printf("[FAIL] expected layer cache misses with budget=%zu bytes\n", budget);
        return 1;
    }
    if (st.prefetches == 0) {
        printf("[FAIL] expected prefetches with async_expert_prefetch enabled\n");
        return 1;
    }

    printf("[PASS] expert_cache_gpu_test: misses=%llu prefetches=%llu max_resident=%d\n",
           (unsigned long long)st.layer_misses, (unsigned long long)st.prefetches,
           st.max_resident_layers);
    return 0;
}
