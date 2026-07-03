// GPU integration test — exercises the real DecodeRunner (kernels + MoE engine
// + KV cache) on device with Qwen3.5-style attention dims (16 Q / 2 KV heads,
// head_dim=128). Skips cleanly when no CUDA device is present, so it is a no-op
// in CI without a GPU and a real end-to-end check on an RTX 5090.

#include "sparkinfer/runtime.h"
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/decode.h"
#include "sparkinfer/moe/engine.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>

using sparkinfer::AttnConfig;
using sparkinfer::DecodeRunner;
using sparkinfer::KVCacheConfig;
using sparkinfer::KVCacheManager;
using sparkinfer::TransformerLayerWeights;
namespace moe = sparkinfer::moe;

static uint16_t f2bf16(float f) { uint32_t b; __builtin_memcpy(&b, &f, 4); return (uint16_t)(b >> 16); }
static float bf162f(uint16_t h) { uint32_t b = (uint32_t)h << 16; float f; __builtin_memcpy(&f, &b, 4); return f; }

static void* dev_rand_bf16(size_t n, float s) {
    std::vector<uint16_t> h(n);
    for (size_t i = 0; i < n; i++) h[i] = f2bf16(s * (2.f * ((i * 1103515245u + 12345u) % 1000) / 1000.f - 1.f));
    void* d = nullptr; cudaMalloc(&d, n * sizeof(uint16_t));
    cudaMemcpy(d, h.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice);
    return d;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no CUDA device — decode_runner_gpu_test requires a GPU (e.g. RTX 5090)\n");
        return 0;
    }

    auto rt = sparkinfer::Runtime::create({});
    rt->initialize();

    const int H = 2048, nkv = 2, nq = 16, hd = 128;        // gqa8 / Qwen3.5 attention
    const int E = 8, K = 2, F = 64, layers = 2, seqs = 2;
    AttnConfig ac{nq, nkv, hd, 1.f / std::sqrt((float)hd), 1e6f, 1e-6f};

    KVCacheConfig kvc; kvc.num_layers = layers; kvc.num_kv_heads = nkv; kvc.head_dim = hd; kvc.block_size = 16;
    KVCacheManager kv(kvc, 64ull * 1024 * 1024);
    for (int i = 0; i < seqs; i++) if (!kv.allocate(i, 64)) { printf("[FAIL] kv allocate\n"); return 1; }

    moe::MoEConfig mc; mc.num_experts = E; mc.top_k = K; mc.hidden_dim = H; mc.ffn_dim = F; mc.num_layers = layers;
    auto engine = moe::MoEEngine::create(mc);

    const int Q = nq * hd, KVd = nkv * hd;
    std::vector<TransformerLayerWeights> w(layers);
    for (int l = 0; l < layers; l++) {
        w[l].attn_norm = dev_rand_bf16(H, 0.5f);
        w[l].wq = dev_rand_bf16((size_t)H * Q, 0.05f);
        w[l].wk = dev_rand_bf16((size_t)H * KVd, 0.05f);
        w[l].wv = dev_rand_bf16((size_t)H * KVd, 0.05f);
        w[l].wo = dev_rand_bf16((size_t)Q * H, 0.05f);
        w[l].q_norm = dev_rand_bf16(hd, 0.5f);
        w[l].k_norm = dev_rand_bf16(hd, 0.5f);
        w[l].ffn_norm = dev_rand_bf16(H, 0.5f);
        w[l].moe.router_w = dev_rand_bf16((size_t)H * E, 0.1f);
        w[l].moe.gate_w = dev_rand_bf16((size_t)E * H * F, 0.05f);
        w[l].moe.up_w   = dev_rand_bf16((size_t)E * H * F, 0.05f);
        w[l].moe.down_w = dev_rand_bf16((size_t)E * F * H, 0.05f);
    }

    DecodeRunner runner(H, ac, &kv, engine.get(), seqs);
    void* x = dev_rand_bf16((size_t)seqs * H, 1.f);

    cudaStream_t stream; cudaStreamCreate(&stream);
    std::vector<int> lens(seqs, 7);                 // 7 tokens already cached
    runner.begin_step(lens);
    for (int l = 0; l < layers; l++) runner.decode_layer(l, x, seqs, w[l], stream);
    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("[FAIL] cuda error: %s\n", cudaGetErrorString(err)); return 1; }

    std::vector<uint16_t> hx((size_t)seqs * H);
    cudaMemcpy(hx.data(), x, hx.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < hx.size(); i++) if (!std::isfinite(bf162f(hx[i]))) { printf("[FAIL] non-finite output at %zu\n", i); return 1; }

    printf("[PASS] decode_runner_gpu_test: %d layers x %d seqs, output finite\n", layers, seqs);
    return 0;
}
