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
    bool int8_kv = false;       // int8 (Q8-style) KV cache; halves the long-context KV read
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

    // Returns device pointer to the block table for seq_id
    // Shape: [num_layers, max_blocks_per_seq]
    int* block_table(uint64_t seq_id) const;

    // Device pointers to K and V storage pools (base = layer 0).
    // Per-layer pointer = (bf16*)k_pool() + layer * layer_stride_elems().
    void* k_pool() const;
    void* v_pool() const;
    size_t layer_stride_elems() const;   // elements between consecutive layers' sub-pools

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
