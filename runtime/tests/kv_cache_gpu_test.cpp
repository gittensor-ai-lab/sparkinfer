// GPU test for KVCacheManager block growth — exercises seq_id re-allocation.

#include "sparkinfer/kv_cache.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

using sparkinfer::KVCacheConfig;
using sparkinfer::KVCacheManager;

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no CUDA device — kv_cache_gpu_test requires a GPU\n");
        return 0;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    constexpr int block_size = 16;
    KVCacheConfig cfg{};
    cfg.num_layers = 2;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 128;
    cfg.block_size = block_size;

    KVCacheManager kv(cfg, 32ull * 1024 * 1024);
    const int total = kv.num_total_blocks();
    const int free0 = kv.num_free_blocks();
    if (free0 != total) {
        printf("[FAIL] expected all blocks free at start (got %d, total %d)\n", free0, total);
        return 1;
    }

    if (!kv.allocate(0, 32)) { printf("[FAIL] first allocate(0, 32)\n"); return 1; }
    const int free1 = kv.num_free_blocks();
    if (free1 != free0 - 2) {
        printf("[FAIL] first allocate should take 2 blocks (free %d -> %d)\n", free0, free1);
        return 1;
    }

    std::vector<int> seq0_blocks(2);
    cudaMemcpy(seq0_blocks.data(), kv.block_table(0), 2 * sizeof(int), cudaMemcpyDeviceToHost);

    if (!kv.allocate(0, 64)) { printf("[FAIL] re-allocate(0, 64)\n"); return 1; }
    const int free2 = kv.num_free_blocks();
    if (free2 != free1 - 2) {
        printf("[FAIL] re-allocate should add 2 blocks (free %d -> %d, want %d)\n",
               free1, free2, free1 - 2);
        return 1;
    }

    std::vector<int> seq0_grown(4);
    cudaMemcpy(seq0_grown.data(), kv.block_table(0), 4 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 2; i++) {
        if (seq0_grown[i] != seq0_blocks[i]) {
            printf("[FAIL] re-allocate changed existing block id at %d (%d != %d)\n",
                   i, seq0_grown[i], seq0_blocks[i]);
            return 1;
        }
    }

    if (!kv.allocate(1, 32)) { printf("[FAIL] allocate(1, 32)\n"); return 1; }
    std::vector<int> seq1_blocks(2);
    cudaMemcpy(seq1_blocks.data(), kv.block_table(1), 2 * sizeof(int), cudaMemcpyDeviceToHost);
    if (seq1_blocks[0] == seq0_grown[0] && seq1_blocks[1] == seq0_grown[1]) {
        printf("[FAIL] seq 1 block table aliases seq 0 (table row corruption)\n");
        return 1;
    }

    const int free3 = kv.num_free_blocks();
    if (!kv.allocate(0, 64)) { printf("[FAIL] idempotent re-allocate(0, 64)\n"); return 1; }
    if (kv.num_free_blocks() != free3) {
        printf("[FAIL] idempotent re-allocate consumed blocks (%d -> %d)\n",
               free3, kv.num_free_blocks());
        return 1;
    }

    printf("[PASS] kv_cache_gpu_test on %s: fresh allocate, delta re-grow, no leak, "
           "seq isolation (%d blocks total)\n", prop.name, total);
    return 0;
}
