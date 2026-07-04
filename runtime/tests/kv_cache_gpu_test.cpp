// GPU test for grow-aware KV block allocation.
// Verifies that re-allocating the same seq_id tops up by delta only (no block leak
// or block-table overrun). Skips when no CUDA device is present.

#include "sparkinfer/kv_cache.h"

#include <cuda_runtime.h>
#include <cstdio>

using sparkinfer::KVCacheConfig;
using sparkinfer::KVCacheManager;

static int blocks_for(int tokens, int block_size) {
    return (tokens + block_size - 1) / block_size;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no CUDA device — kv_cache_gpu_test requires a GPU\n");
        return 0;
    }

    KVCacheConfig cfg;
    cfg.num_layers = 2;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 128;
    cfg.block_size = 16;
    KVCacheManager kv(cfg, 64ull * 1024 * 1024);

    const int free0 = kv.num_free_blocks();
    const uint64_t seq = 42;

    if (!kv.allocate(seq, 32)) { printf("[FAIL] initial allocate(32)\n"); return 1; }
    const int b32 = blocks_for(32, cfg.block_size);
    if (free0 - kv.num_free_blocks() != b32) {
        printf("[FAIL] expected %d blocks for 32 tokens, got %d\n", b32, free0 - kv.num_free_blocks());
        return 1;
    }

    // Same token count — must not consume additional blocks.
    const int free1 = kv.num_free_blocks();
    if (!kv.allocate(seq, 32)) { printf("[FAIL] re-allocate(32)\n"); return 1; }
    if (kv.num_free_blocks() != free1) {
        printf("[FAIL] re-allocate(32) leaked blocks: free %d -> %d\n", free1, kv.num_free_blocks());
        return 1;
    }

    // Grow to 48 tokens — only the delta should be charged.
    if (!kv.allocate(seq, 48)) { printf("[FAIL] grow allocate(48)\n"); return 1; }
    const int b48 = blocks_for(48, cfg.block_size);
    if (free1 - kv.num_free_blocks() != b48 - b32) {
        printf("[FAIL] grow to 48: expected delta %d, got %d\n", b48 - b32, free1 - kv.num_free_blocks());
        return 1;
    }

    kv.free(seq);
    if (kv.num_free_blocks() != free0) {
        printf("[FAIL] free did not return all blocks: %d vs %d\n", kv.num_free_blocks(), free0);
        return 1;
    }

    printf("[PASS] kv_cache grow-aware allocate\n");
    return 0;
}
