#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace sparkinfer {

enum class KVLayout {
    PAGED,       // PagedAttention-style block allocation
    CONTIGUOUS,  // flat contiguous (single sequence)
    COMPRESSED,  // quantized / compressed KV (future)
};

struct KVCacheConfig {
    int num_layers;
    int num_kv_heads;
    int head_dim;
    int block_size = 16;        // tokens per page block
    KVLayout layout = KVLayout::PAGED;
    bool fp8_kv = false;        // FP8 KV cache compression
    bool int8_kv = false;       // int8 (Q8-style) KV cache; halves the long-context KV read
    // Hybrid stacks (Qwen3.5/3.6/3.8) give paged KV to the FULL-ATTENTION layers only -- the
    // Gated-DeltaNet layers carry a recurrent state instead and never touch these pools. Sizing
    // the pool for every layer therefore over-allocates by n_layers/n_attn_layers (4x on
    // Qwen3.8-27B: 16 attention layers of 64). layer_slot maps a layer index to its pool slot,
    // -1 for layers with no KV; empty = identity (every layer has a slot, the old behaviour).
    std::vector<int> layer_slot;
};

// Layer -> pool-slot map for the hybrid interval rule these models share: with
// full_attn_interval = k, layer L is a linear (Gated-DeltaNet) layer -- which carries a recurrent
// state and never touches paged KV -- unless (L+1) % k == 0. Returns an empty map for non-hybrid
// models, which KVCacheManager reads as the identity (every layer gets a slot).
inline std::vector<int> hybrid_kv_layer_slots(int num_layers, bool hybrid, int full_attn_interval) {
    if (!hybrid || full_attn_interval <= 0 || num_layers <= 0) return {};
    std::vector<int> slot((size_t)num_layers, -1);
    int n = 0;
    for (int L = 0; L < num_layers; ++L)
        if (((L + 1) % full_attn_interval) == 0) slot[(size_t)L] = n++;
    if (n == 0 || n == num_layers) return {};   // nothing to compact
    return slot;
}
inline int kv_slot_count(const std::vector<int>& slot, int num_layers) {
    if (slot.empty()) return num_layers;
    int n = 0;
    for (int v : slot) if (v + 1 > n) n = v + 1;
    return n;
}

// GPU-side KV block pool.
// Manages a fixed-size pool of blocks and maps sequence positions
// to physical block indices via a per-sequence block table.
class KVCacheManager {
public:
    explicit KVCacheManager(const KVCacheConfig& cfg, size_t pool_bytes);
    ~KVCacheManager();

    // Allocate physical blocks for a sequence (grows if already allocated).
    // Returns false if OOM. Idempotent when num_tokens fits existing allocation.
    bool allocate(uint64_t seq_id, int num_tokens);

    // Tokens already covered by the current block allocation for seq_id (0 if none).
    int allocated_tokens(uint64_t seq_id) const;

    // Free all blocks owned by a sequence
    void free(uint64_t seq_id);

    // Speculative decode: truncate a sequence to its first keep_blocks logical blocks
    // (returns false if seq_id is unknown). Freed physical blocks go back to the pool.
    bool truncate_blocks(uint64_t seq_id, int keep_blocks);

    // Number of logical KV blocks currently allocated for seq_id (0 if none).
    int num_blocks(uint64_t seq_id) const;

    // Returns device pointer to the block table for seq_id
    // Shape: [num_layers, max_blocks_per_seq]
    int* block_table(uint64_t seq_id) const;

    // Physical block ids owned by seq_id, in logical order (index 0 = the sequence's first
    // block). Empty if seq_id is unknown. Physical blocks are free-list allocated and not
    // guaranteed contiguous, so a caller walking (layer, physical_block) byte ranges for bulk
    // copy-out/copy-in (e.g. an external KV cache tier) needs this rather than assuming a
    // contiguous span. Returns a reference into internal state -- valid only until the next
    // allocate()/free()/truncate_blocks() call for this seq_id.
    const std::vector<int>& physical_block_ids(uint64_t seq_id) const;

    // Device pointers to K and V storage pools (base = the first slot).
    // Per-layer pointer = (bf16*)k_pool() + layer_base_elems(layer) -- NOT layer * layer_stride,
    // which is only the same thing while the pool is uncompacted.
    void* k_pool() const;
    void* v_pool() const;
    size_t layer_stride_elems() const;   // elements between consecutive slots' sub-pools
    int kv_slots() const;                // slots actually allocated (== num_layers uncompacted)
    // Element offset of `layer`'s slice. A layer with no KV lands on a trap slice (and warns once)
    // rather than aliasing a real layer's slot.
    size_t layer_base_elems(int layer) const;
    size_t scale_layer_base_elems(int layer) const;

    // int8 KV (Q8-style int8 + per-(token,kv_head) fp16 scale). When int8_kv(), k_pool/v_pool hold
    // int8 and k_scale_pool/v_scale_pool hold one __half scale per head vector.
    // Per-layer scale pointer = (__half*)k_scale_pool() + scale_layer_base_elems(layer).
    bool int8_kv() const;
    void* k_scale_pool() const;
    void* v_scale_pool() const;
    size_t scale_layer_stride_elems() const;

    int block_size() const;
    int max_blocks_per_seq() const;
    int num_free_blocks() const;
    int num_total_blocks() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sparkinfer
