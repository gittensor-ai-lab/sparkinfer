#pragma once

#include <cstdint>
#include <memory>
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
};

// GPU-side KV block pool.
// Manages a fixed-size pool of blocks and maps sequence positions
// to physical block indices via a per-sequence block table.
class KVCacheManager {
public:
    explicit KVCacheManager(const KVCacheConfig& cfg, size_t pool_bytes);
    ~KVCacheManager();

    // Allocate physical blocks for a new sequence; returns false if OOM
    bool allocate(uint64_t seq_id, int num_tokens);

    // Free all blocks owned by a sequence
    void free(uint64_t seq_id);

    // Returns device pointer to the block table for seq_id
    // Shape: [num_layers, max_blocks_per_seq]
    int* block_table(uint64_t seq_id) const;

    // Device pointers to K and V storage pools (base = layer 0).
    // Per-layer pointer = (bf16*)k_pool() + layer * layer_stride_elems().
    void* k_pool() const;
    void* v_pool() const;
    size_t layer_stride_elems() const;   // elements between consecutive layers' sub-pools

    int block_size() const;
    int max_blocks_per_seq() const;
    int num_free_blocks() const;
    int num_total_blocks() const;

    // Returns the first physical block when seq_id owns one linear block run
    // (base, base+1, ...), otherwise -1. Kernels can use this to skip page-table
    // lookups without changing correctness for fragmented allocations.
    int contiguous_block_base(uint64_t seq_id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sparkinfer
