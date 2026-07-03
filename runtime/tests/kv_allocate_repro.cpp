// Standalone repro: allocate(1,100) -> grow allocate(1,200) on live seq_id.
// Matches the bug report scenario (block_size=16 -> 7 blocks, grow to 13).

#include "sparkinfer/kv_cache.h"

#include <cuda_runtime.h>
#include <cstdio>

using sparkinfer::KVCacheConfig;
using sparkinfer::KVCacheManager;

static int used(KVCacheManager& kv) {
    return kv.num_total_blocks() - kv.num_free_blocks();
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("[SKIP] no CUDA device\n");
        return 0;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    KVCacheConfig cfg{};
    cfg.num_layers = 2;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 128;
    cfg.block_size = 16;

    KVCacheManager kv(cfg, 32ull * 1024 * 1024);
    const int total = kv.num_total_blocks();

    if (!kv.allocate(1, 100)) { printf("RESULT: FAIL (allocate 100)\n"); return 1; }
    const int u100 = used(kv);
    printf("after allocate(1,100):  used=%d  (expect 7)\n", u100);
    if (u100 != 7) { printf("RESULT: FAIL\n"); return 1; }

    if (!kv.allocate(1, 200)) { printf("RESULT: FAIL (allocate 200)\n"); return 1; }
    const int u200 = used(kv);
    printf("after allocate(1,200):  used=%d  (correct=13)\n", u200);
    if (u200 != 13) { printf("RESULT: FAIL (expected 13, got %d)\n", u200); return 1; }

    if (!kv.allocate(1, 200)) { printf("RESULT: FAIL (idempotent)\n"); return 1; }
    const int u200b = used(kv);
    printf("re-allocate same size:  used=%d  (expect 13)\n", u200b);
    if (u200b != 13) { printf("RESULT: FAIL\n"); return 1; }

    kv.free(1);
    const int free_after = kv.num_free_blocks();
    printf("free(1):                free=%d  (total=%d)\n", free_after, total);
    if (free_after != total) { printf("RESULT: FAIL (leak: %d blocks missing)\n", total - free_after); return 1; }

    printf("GPU: %s (sm_%d.%d)\n", prop.name, prop.major, prop.minor);
    printf("RESULT: PASS\n");
    return 0;
}
