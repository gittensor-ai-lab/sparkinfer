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
    // Hybrid (Gated-DeltaNet) models: only every Nth layer is a full-attention layer that owns any
    // KV; the linear-attention layers keep their state in the recurrent/conv buffers instead. When
    // >0, the pool is sized for num_layers/attn_every sub-pools and layer_offset_elems() maps an
    // absolute layer index onto its compact slot. 0 = every layer owns KV (the default, and what
    // every non-hybrid model wants).
    //
    // Qwen3.8-27B is 64 layers with full_attn_interval=4, so 48 of its 64 sub-pools were allocated
    // and never written: 4.34 GB of the pool at ctx=16k, on a part with ~2.9 GB free after weights.
    // That is what pushed the batched prefill's scratch arena over the edge -- it declined and fell
    // back to the per-token loop (79 pp).
    int attn_every = 0;
};

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

    // Device pointers to K and V storage pools (base = layer 0).
    // Per-layer pointer = (bf16*)k_pool() + layer * layer_stride_elems().
    void* k_pool() const;
    void* v_pool() const;
    size_t layer_stride_elems() const;   // elements between consecutive layers' sub-pools

    // Element offset of `layer`'s sub-pool. Use this instead of layer * layer_stride_elems():
    // with attn_every > 0 the sub-pools are compacted and the absolute layer index is not the
    // slot index. Identity when attn_every == 0.
    size_t layer_offset_elems(int layer) const;
    size_t scale_layer_offset_elems(int layer) const;
    // Number of sub-pools actually allocated (num_layers when attn_every == 0).
    int kv_layer_count() const;

    // int8 KV (Q8-style int8 + per-(token,kv_head) fp16 scale). When int8_kv(), k_pool/v_pool hold
    // int8 and k_scale_pool/v_scale_pool hold one __half scale per head vector.
    // Per-layer scale pointer = (__half*)k_scale_pool() + layer * scale_layer_stride_elems().
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
