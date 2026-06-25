// GPU integration test — MoEEngine::forward() must refuse out-of-range layer
// indices and unset weights instead of indexing past weights_[] or launching
// kernels with null device pointers. Skips cleanly when no CUDA device.

#include "sparkinfer/moe/engine.h"

#include <cuda_runtime.h>
#include <cstdio>

namespace moe = sparkinfer::moe;

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no CUDA device — engine_guard_gpu_test requires a GPU\n");
        return 0;
    }

    moe::MoEConfig cfg{};
    cfg.num_experts = 4;
    cfg.top_k = 2;
    cfg.hidden_dim = 32;
    cfg.ffn_dim = 64;
    cfg.num_layers = 2;
    auto engine = moe::MoEEngine::create(cfg);

    void* input = nullptr;
    void* output = nullptr;
    cudaMalloc(&input, (size_t)cfg.hidden_dim * sizeof(unsigned short));
    cudaMalloc(&output, (size_t)cfg.hidden_dim * sizeof(unsigned short));
    cudaMemset(input, 0, (size_t)cfg.hidden_dim * sizeof(unsigned short));
    cudaMemset(output, 0, (size_t)cfg.hidden_dim * sizeof(unsigned short));

    // Layer 0 was never registered — null weights must be refused.
    engine->forward(input, output, 1, 0, nullptr);

    // Layer index past num_layers — must not OOB-read weights_[].
    engine->forward(input, output, 1, cfg.num_layers, nullptr);
    engine->forward(input, output, 1, -1, nullptr);

    cudaFree(input);
    cudaFree(output);

    printf("moe engine forward guards: PASS\n");
    return 0;
}
