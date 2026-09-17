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

    // SLIDING-WINDOW SLOTS. A layer whose attention only ever reads the last `window_tokens`
    // tokens does not need a full-context slice: it needs the window, plus whatever one prefill
    // pass appends before that pass's attention runs (the pass writes every token of the pass
    // first, and its EARLIEST query still reads `window_tokens` back from itself). Those slots get
    // a private per-sequence RING of window_tokens + window_pass_tokens instead, which on Muse
    // Glimmer (39 of 52 layers windowed at 2048) is where most of the pool went.
    //
    // Capacity is deliberately unchanged: total_blocks/max_blocks_per_seq still come out of
    // pool_bytes as if every slot were full-context, so num_total_blocks() keeps meaning what
    // every caller already reads it as. The windowed slices simply allocate less of the pool, and
    // the difference shows up as free VRAM.
    //
    // Two things a ring cannot do, so both are refused rather than silently served:
    //   - CROSS-SEQUENCE PREFIX SHARING. A ring slot is private to one sequence; a shared prefix
    //     block would alias. prefix_sharing_supported() says so and allocate_with_prefix()
    //     refuses, so the engine recomputes instead of reading someone else's window.
    //   - A PREFILL PASS LONGER THAN window_pass_tokens. window_pass_limit() publishes the bound;
    //     a caller past it must chunk or decline. (Re-using ONE sequence's own session across
    //     requests is fine: its ring holds that sequence's last window, which is all a windowed
    //     layer ever reads.)
    // window_tokens = 0 (the default) keeps every slot full-context -- the old layout, byte for
    // byte. SPARKINFER_KV_SWA_CAP=0 also forces it off, for an A/B out of one binary.
    int window_tokens = 0;        // sliding window, in tokens (0 = no windowed slots)
    int window_pass_tokens = 0;   // longest prefill pass a windowed slot must survive
    int max_seq_tokens = 0;       // per-sequence context the pool was sized for (required to cap)
    std::vector<char> slot_windowed;   // per POOL SLOT: 1 = windowed ring, 0 = full context
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

// Per-slot windowed flags from the model's own per-layer contract (Qwen35Config::swa_layers:
// true = slides over `sliding_window` tokens, false = full/global attention, which must keep a
// full-context slice). `layer_slot` is the same map KVCacheConfig carries (empty = identity).
// Returns an empty vector when nothing would be capped -- no windowed layer, or every one of them
// windowed, since a single group is the cheaper layout either way -- which KVCacheManager reads as
// "no windowed slots".
inline std::vector<char> swa_slot_flags(int num_layers, const std::vector<int>& layer_slot,
                                        const std::vector<bool>& swa_layers) {
    if (num_layers <= 0 || (int)swa_layers.size() != num_layers) return {};
    std::vector<char> win((size_t)kv_slot_count(layer_slot, num_layers), 0);
    int n = 0;
    for (int L = 0; L < num_layers; ++L) {
        const int s = layer_slot.empty() ? L : (L < (int)layer_slot.size() ? layer_slot[(size_t)L] : -1);
        if (s < 0 || s >= (int)win.size()) continue;
        if (swa_layers[(size_t)L]) { win[(size_t)s] = 1; ++n; }
    }
    if (n == 0 || n == (int)win.size()) return {};   // all or nothing to cap: keep one layout
    return win;
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

    // PREFIX SHARING. Physical blocks are reference-counted, so one block can have several holders
    // at once: the sequence that wrote it, a prefix-cache entry that retained it, and any later
    // sequence that starts from that prefix. A block returns to the pool only when its last holder
    // lets go. Shared blocks are READ-ONLY by contract -- a sequence built on a prefix starts
    // writing at prefix.size() * block_size() -- so no copy-on-write is needed.
    //
    // Retain seq_id's first n_blocks physical blocks (+1 each) and return them in logical order, for
    // a holder that is not a sequence (it takes no block-table row). Empty, retaining nothing, when
    // seq_id is unknown or has fewer blocks.
    std::vector<int> retain_prefix_blocks(uint64_t seq_id, int n_blocks);
    // Drop one reference from each block; any that reach zero go back to the pool.
    void release_blocks(const std::vector<int>& physical_ids);
    // allocate() for a seq_id that has no blocks yet, whose first logical blocks ARE `prefix`
    // (shared, +1 each); only the remainder of num_tokens comes from the pool. False, changing
    // nothing, when seq_id already has blocks, a prefix block is not live, prefix is longer than
    // num_tokens needs, or the pool cannot cover the rest.
    bool allocate_with_prefix(uint64_t seq_id, const std::vector<int>& prefix, int num_tokens);

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

    // WINDOWED SLOTS (see KVCacheConfig::window_tokens). windowed() is false when this pool has
    // none, and then every accessor below behaves exactly as the full-context ones.
    bool windowed() const;
    // The ring table for seq_id: [max_blocks_per_seq] entries, logical block i -> the physical
    // block holding i's window slot, i.e. ring[i % ring_blocks]. A windowed layer passes THIS to
    // the same append/attention kernels the full layers drive with block_table(), so no kernel
    // learns about rings -- the repeated rows carry the wrap. Null for an unknown sequence, and
    // block_table() itself when this pool has no windowed slots.
    int* block_table_win(uint64_t seq_id) const;
    // Longest prefill pass a windowed slot can serve (window_pass_tokens), or INT_MAX uncapped.
    int window_pass_limit() const;
    // False when a ring is in play: cross-sequence prefix blocks would alias (see the header note).
    bool prefix_sharing_supported() const;

    int block_size() const;
    int max_blocks_per_seq() const;
    int num_free_blocks() const;
    int num_total_blocks() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sparkinfer
