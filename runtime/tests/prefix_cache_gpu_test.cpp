// GPU test for prefix sharing in the paged KV pool, and for the PrefixCache built on it. No model:
// a 64-block KVCacheManager with a real device pool, so allocate() and the block-table uploads run
// for real. Every free-block count below is worked out by hand -- a refcount bug shows up as a
// block that never returns to the pool, or one that returns while something still reads it.

#include "sparkinfer/kv_cache.h"
#include "sparkinfer/prefix_cache.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

using namespace sparkinfer;

#define CHECK(x) do { if (!(x)) { std::printf("FAIL: %s line %d\n", #x, __LINE__); return 1; } } while (0)

int main() {
    KVCacheConfig cfg;
    cfg.num_layers = 2;
    cfg.num_kv_heads = 1;
    cfg.head_dim = 8;
    cfg.block_size = 16;
    // bf16 budget per block = 16 tokens * 1 head * 8 dim * 2 bytes = 256; K and V over 2 slots.
    const int kBlocks = 64;
    KVCacheManager kv(cfg, (size_t)kBlocks * 2 * 2 * 256);
    CHECK(kv.num_total_blocks() == kBlocks);
    CHECK(kv.num_free_blocks() == kBlocks);

    // ---- retain outlives the sequence ----
    CHECK(kv.allocate(1, 160));                              // 10 blocks
    CHECK(kv.num_free_blocks() == kBlocks - 10);
    const std::vector<int> ret = kv.retain_prefix_blocks(1, 4);
    CHECK(ret.size() == 4);
    CHECK(ret == std::vector<int>(kv.physical_block_ids(1).begin(), kv.physical_block_ids(1).begin() + 4));
    CHECK(kv.retain_prefix_blocks(1, 11).empty());           // more than it has: nothing retained
    CHECK(kv.retain_prefix_blocks(99, 1).empty());           // unknown sequence
    kv.free(1);
    CHECK(kv.num_free_blocks() == kBlocks - 4);              // the 4 retained stay out of the pool

    // ---- a sequence built on the retained prefix shares it and takes only the rest ----
    CHECK(kv.allocate_with_prefix(2, ret, 100));             // 7 blocks, 4 of them shared
    CHECK(kv.num_free_blocks() == kBlocks - 7);
    const std::vector<int> b2 = kv.physical_block_ids(2);
    CHECK(b2.size() == 7);
    for (int i = 0; i < 4; ++i) CHECK(b2[i] == ret[i]);
    std::vector<int> dev(7, -1);
    cudaMemcpy(dev.data(), kv.block_table(2), 7 * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK(dev == b2);                                         // the device table row says the same

    // ---- refusals change nothing ----
    CHECK(!kv.allocate_with_prefix(2, ret, 100));            // already has blocks
    CHECK(!kv.allocate_with_prefix(3, ret, 32));             // prefix longer than the request needs
    const int free_before = kv.num_free_blocks();
    CHECK(!kv.allocate_with_prefix(4, ret, 16 * (4 + free_before + 1)));   // pool can't cover the rest
    CHECK(kv.num_free_blocks() == free_before);

    kv.release_blocks(ret);                                  // seq 2 still holds them
    CHECK(kv.num_free_blocks() == free_before);
    kv.free(2);
    CHECK(kv.num_free_blocks() == kBlocks);
    CHECK(!kv.allocate_with_prefix(5, ret, 100));            // nobody holds those blocks any more

    // ---- truncate on a sharing sequence drops only its own references ----
    CHECK(kv.allocate(6, 64));                               // a b c d
    const std::vector<int> r6 = kv.retain_prefix_blocks(6, 2);   // a b
    CHECK(kv.allocate_with_prefix(7, r6, 48));               // a b e
    CHECK(kv.truncate_blocks(7, 1));                         // e freed, b loses seq 7
    kv.free(6);                                              // c d freed; a b still retained
    CHECK(kv.num_free_blocks() == kBlocks - 2);
    kv.release_blocks(r6);                                   // b freed; a still in seq 7
    CHECK(kv.num_free_blocks() == kBlocks - 1);
    kv.free(7);
    CHECK(kv.num_free_blocks() == kBlocks);

    // ---- PrefixCache ----
    {
        PrefixCache::Limits lim;
        lim.max_entries = 2;
        PrefixCache cache(&kv, lim);
        std::vector<int> base(200);
        for (int i = 0; i < 200; ++i) base[i] = (i * 7 + 1) % 1000;
        auto head = [&](int n) { return std::vector<int>(base.begin(), base.begin() + n); };

        CHECK(kv.allocate(10, 200));
        cache.insert(head(48), kv.retain_prefix_blocks(10, 3), {});
        kv.free(10);
        CHECK(kv.num_free_blocks() == kBlocks - 3);
        CHECK(kv.allocate(11, 200));
        cache.insert(head(96), kv.retain_prefix_blocks(11, 6), {});
        kv.free(11);
        CHECK(cache.stats().entries == 2);

        CHECK(cache.lookup(base).tokens == 96);              // longest proper prefix
        CHECK(cache.lookup(base).blocks.size() == 6);
        CHECK(cache.lookup(head(96)).tokens == 48);          // an entry must be strictly shorter
        std::vector<int> other = base;
        other[10] = 999999;
        CHECK(cache.lookup(other).tokens == 0);              // diverges inside every entry

        CHECK(kv.allocate(12, 96));                          // duplicate prefix: dropped, blocks released
        const int f = kv.num_free_blocks();
        cache.insert(head(48), kv.retain_prefix_blocks(12, 3), {});
        kv.free(12);
        CHECK(kv.num_free_blocks() == f + 6);
        CHECK(cache.stats().entries == 2);

        CHECK(kv.allocate(13, 32));                          // not block-aligned: dropped
        cache.insert(head(20), kv.retain_prefix_blocks(13, 2), {});
        kv.free(13);
        CHECK(cache.stats().entries == 2);

        // LRU is now the 96-token entry (last used by the first lookup; 48 was refreshed since).
        CHECK(kv.allocate(14, 200));
        cache.insert(head(144), kv.retain_prefix_blocks(14, 9), {});
        kv.free(14);
        PrefixCache::Stats s = cache.stats();
        CHECK(s.entries == 2 && s.evictions == 1 && s.blocks == 3 + 9);
        CHECK(kv.num_free_blocks() == kBlocks - 3 - 9);
        CHECK(cache.lookup(base).tokens == 144);
        CHECK(cache.lookup(head(144)).tokens == 48);         // 96 is gone

        CHECK(cache.evict_for(kBlocks - 3));                 // evicts the LRU 144-token entry, then stops
        CHECK(kv.num_free_blocks() == kBlocks - 3);
        CHECK(cache.stats().entries == 1);
        CHECK(!cache.evict_for(0));                          // already enough free

        // A sequence built on a hit keeps its blocks alive after the entry itself is evicted.
        const PrefixCache::Hit h = cache.lookup(base);
        CHECK(h.tokens == 48);
        CHECK(kv.allocate_with_prefix(15, h.blocks, 100));
        CHECK(cache.evict_for(kBlocks));                     // empties the cache, cannot reach 64
        CHECK(cache.stats().entries == 0);
        CHECK(kv.num_free_blocks() == kBlocks - 7);
        kv.free(15);
        CHECK(kv.num_free_blocks() == kBlocks);

        CHECK(kv.allocate(16, 32));                          // left in the cache for the destructor
        cache.insert(head(32), kv.retain_prefix_blocks(16, 2), {});
        kv.free(16);
        CHECK(kv.num_free_blocks() == kBlocks - 2);
        CHECK(cache.stats().hits == 6 && cache.stats().lookups == 7);
    }
    CHECK(kv.num_free_blocks() == kBlocks);                  // destructor released the last entry

    std::printf("prefix_cache_gpu_test: OK\n");
    return 0;
}
