// GPU test for the windowed (ring) KV slices. No model: a real device pool, so the slice layout
// and the ring block-table uploads run for real.
//
// What it pins down, because each one is a way a ring can be silently wrong:
//   - capacity is unchanged (total_blocks / max_blocks_per_seq still come from pool_bytes as if
//     every slice were full-context), so every caller reading num_total_blocks() is unaffected;
//   - a windowed slice's base sits in its own, SMALLER region and never overlaps a full slice;
//   - the ring table repeats: logical block i maps to the same physical block as i + ring_blocks,
//     which is what lets the unchanged append/attention kernels carry the wrap;
//   - rings are per sequence (no two sequences share a ring block) and come back on free();
//   - prefix sharing is refused rather than served from a stranger's ring;
//   - window_tokens = 0 and SPARKINFER_KV_SWA_CAP=0 both keep the old single-group layout.

#include "sparkinfer/kv_cache.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

using namespace sparkinfer;

#define CHECK(x) do { if (!(x)) { std::printf("FAIL: %s line %d\n", #x, __LINE__); return 1; } } while (0)

namespace {
KVCacheConfig base_cfg(int layers) {
    KVCacheConfig cfg;
    cfg.num_layers = layers;
    cfg.num_kv_heads = 1;
    cfg.head_dim = 8;
    cfg.block_size = 16;
    return cfg;
}
// One block of bf16 budget: block_size * heads * dim * 2 bytes.
constexpr size_t kBlockBytes = 16 * 1 * 8 * 2;

std::vector<int> read_table(const int* d, int n) {
    std::vector<int> h((size_t)n, -1);
    cudaMemcpy(h.data(), d, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost);
    return h;
}
}  // namespace

int main() {
    if (cudaSetDevice(0) != cudaSuccess) { std::printf("[SKIP] no GPU\n"); return 0; }

    // 4 layers, every 4th full-causal: slots 0,1,2 windowed, slot 3 full -- Muse's rule in
    // miniature. Window 64 tokens + a 32-token pass = 6 blocks + 1 spare = 7, against 32 blocks
    // of full context (512 tokens).
    const int kLayers = 4;
    KVCacheConfig cfg = base_cfg(kLayers);
    cfg.window_tokens = 64;
    cfg.window_pass_tokens = 32;
    cfg.max_seq_tokens = 512;
    // layers 0,1,2 slide; layer 3 is full-causal -- Muse's rule in miniature.
    const std::vector<bool> swa_layers = {true, true, true, false};
    cfg.slot_windowed = swa_slot_flags(kLayers, cfg.layer_slot, swa_layers);
    CHECK(cfg.slot_windowed.size() == 4);
    CHECK(cfg.slot_windowed[0] == 1 && cfg.slot_windowed[1] == 1 && cfg.slot_windowed[2] == 1);
    CHECK(cfg.slot_windowed[3] == 0);              // layer 3 is the global one

    const int kBlocks = 128;                       // 4 sessions of 512 tokens
    KVCacheManager kv(cfg, (size_t)kBlocks * kLayers * 2 * kBlockBytes);

    CHECK(kv.windowed());
    CHECK(kv.num_total_blocks() == kBlocks);       // capacity unchanged by capping
    CHECK(kv.max_blocks_per_seq() == kBlocks);
    CHECK(kv.window_pass_limit() == 32);
    CHECK(!kv.prefix_sharing_supported());

    // ---- slice layout: windowed slices are smaller, and disjoint from the full ones ----
    const size_t full_stride = kv.layer_stride_elems();
    const size_t ring_blocks = 7;                  // (64+32)/16 + 1
    const size_t sessions = kBlocks / (512 / 16);  // 4
    const size_t win_stride = ring_blocks * sessions * (size_t)16 * 1 * 8;
    // Layer 3 (full) is group index 0 of the full region; layers 0..2 follow the full region.
    CHECK(kv.layer_base_elems(3) == 0);
    CHECK(kv.layer_base_elems(0) >= full_stride);                       // past every full slice
    CHECK(kv.layer_base_elems(1) == kv.layer_base_elems(0) + win_stride);
    CHECK(kv.layer_base_elems(2) == kv.layer_base_elems(1) + win_stride);
    CHECK(win_stride < full_stride);                                    // the whole point

    // ---- the ring table repeats, the full table does not ----
    CHECK(kv.allocate(1, 512));                    // 32 full blocks + one ring
    const std::vector<int> full1 = read_table(kv.block_table(1), 32);
    const std::vector<int> ring1 = read_table(kv.block_table_win(1), (int)(3 * ring_blocks));
    for (size_t i = 0; i < ring_blocks; ++i) {
        CHECK(ring1[i] >= 0);
        CHECK(ring1[i + ring_blocks] == ring1[i]);          // wraps
        CHECK(ring1[i + 2 * ring_blocks] == ring1[i]);
    }
    std::set<int> ring1_set(ring1.begin(), ring1.begin() + (long)ring_blocks);
    CHECK(ring1_set.size() == ring_blocks);                 // a ring has no repeats inside itself
    std::set<int> full1_set(full1.begin(), full1.end());
    CHECK(full1_set.size() == 32);

    // ---- rings are private: sequence 2 shares no ring block with sequence 1 ----
    CHECK(kv.allocate(2, 512));
    const std::vector<int> ring2 = read_table(kv.block_table_win(2), (int)ring_blocks);
    for (int b : ring2) CHECK(ring1_set.count(b) == 0);

    // ---- prefix sharing is refused, not mis-served ----
    const std::vector<int> pre = kv.retain_prefix_blocks(1, 4);
    CHECK(pre.size() == 4);
    CHECK(!kv.allocate_with_prefix(3, pre, 512));
    kv.release_blocks(pre);

    // ---- free returns both the full blocks and the ring ----
    const int free_before = kv.num_free_blocks();
    kv.free(2);
    CHECK(kv.num_free_blocks() == free_before + 32);
    CHECK(kv.allocate(4, 512));                             // the freed ring is reusable
    const std::vector<int> ring4 = read_table(kv.block_table_win(4), (int)ring_blocks);
    std::set<int> ring4_set(ring4.begin(), ring4.end());
    CHECK(ring4_set.size() == ring_blocks);
    for (int b : ring4) CHECK(ring1_set.count(b) == 0);      // still not sequence 1's

    // ---- rings run out before the pool does, and that is a clean refusal ----
    int opened = 2;                                          // sequences 1 and 4
    for (uint64_t sid = 10; sid < 40; ++sid) {
        if (!kv.allocate(sid, 16)) break;                    // one block each, but a whole ring
        ++opened;
    }
    CHECK(opened >= (int)sessions);                          // at least what the pool was sized for
    CHECK(kv.num_free_blocks() >= 0);

    // ---- window_tokens = 0 keeps the old layout, byte for byte ----
    {
        KVCacheConfig plain = base_cfg(kLayers);
        KVCacheManager k2(plain, (size_t)kBlocks * kLayers * 2 * kBlockBytes);
        CHECK(!k2.windowed());
        CHECK(k2.prefix_sharing_supported());
        CHECK(k2.num_total_blocks() == kBlocks);
        for (int L = 0; L < kLayers; ++L)
            CHECK(k2.layer_base_elems(L) == (size_t)L * k2.layer_stride_elems());
        CHECK(k2.block_table_win(1) == nullptr);             // unknown sequence, either layout
    }

    // ---- the env kill switch restores it too ----
    {
        setenv("SPARKINFER_KV_SWA_CAP", "0", 1);
        KVCacheManager k3(cfg, (size_t)kBlocks * kLayers * 2 * kBlockBytes);
        CHECK(!k3.windowed());
        CHECK(k3.prefix_sharing_supported());
        for (int L = 0; L < kLayers; ++L)
            CHECK(k3.layer_base_elems(L) == (size_t)L * k3.layer_stride_elems());
        CHECK(k3.allocate(1, 512));
        CHECK(k3.block_table_win(1) == k3.block_table(1));   // one table when uncapped
        unsetenv("SPARKINFER_KV_SWA_CAP");
    }

    std::printf("[PASS] kv_swa_ring_gpu_test\n");
    return 0;
}
