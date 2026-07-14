// GPU test for grow-aware KV block allocation (issue #110).
// Re-allocating an existing seq_id — the natural way to grow a paged sequence —
// must top up by the delta only: no block leak, and the cumulative block count
// must stay bounded by max_blocks_per_seq so the device block-table row never
// overruns. Skips when no CUDA device is present.

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

    // Small per-block footprint so the pool holds more blocks than max_blocks_per_seq,
    // making the cumulative-bound path below reachable on a modest device allocation.
    KVCacheConfig cfg;
    cfg.num_layers = 1;
    cfg.num_kv_heads = 1;
    cfg.head_dim = 16;
    cfg.block_size = 16;
    KVCacheManager kv(cfg, 64ull * 1024 * 1024);

    const int cap = kv.max_blocks_per_seq();
    if (kv.num_total_blocks() <= cap) {
        printf("[FAIL] test needs a pool larger than max_blocks_per_seq (%d blocks)\n", cap);
        return 1;
    }

    const int free0 = kv.num_free_blocks();
    const uint64_t seq = 42;

    // Initial allocate charges exactly the blocks for 32 tokens.
    if (!kv.allocate(seq, 32)) { printf("[FAIL] initial allocate(32)\n"); return 1; }
    const int b32 = blocks_for(32, cfg.block_size);
    if (free0 - kv.num_free_blocks() != b32) {
        printf("[FAIL] expected %d blocks for 32 tokens, got %d\n", b32, free0 - kv.num_free_blocks());
        return 1;
    }

    // Re-allocate the same token count — must not consume additional blocks (no leak).
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

    // Cumulative bound: two calls that each individually pass the per-seq cap but together
    // would exceed it must be rejected on the total — never allowed to grow the row past cap
    // (the buggy code appended blindly and overran the device block-table row).
    const uint64_t big = 7;
    const int near = (cap - 1) * cfg.block_size;               // ceil(near/bs) == cap-1 <= cap
    if (!kv.allocate(big, near)) { printf("[FAIL] allocate(near-cap)\n"); return 1; }
    if (kv.allocate(big, near + 2 * cfg.block_size)) {         // total would be cap+1 blocks
        printf("[FAIL] grow past max_blocks_per_seq was not rejected\n");
        return 1;
    }

    kv.free(seq);
    kv.free(big);
    if (kv.num_free_blocks() != free0) {
        printf("[FAIL] free did not return all blocks: %d vs %d\n", kv.num_free_blocks(), free0);
        return 1;
    }

    printf("[PASS] kv_cache grow-aware allocate\n");
    return 0;
}
