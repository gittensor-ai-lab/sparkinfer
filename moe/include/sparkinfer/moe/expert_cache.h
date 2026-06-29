#pragma once

#include "sparkinfer/moe/engine.h"

#include <cstddef>
#include <cstdint>

namespace sparkinfer {
namespace moe {

// Bytes for one layer's bf16 expert tensors: gate + up + down  [E,H,F] each (down is [E,F,H]).
size_t expert_layer_bytes(int num_experts, int hidden_dim, int ffn_dim);

// How many layer expert sets fit in a VRAM budget (at least 1, capped at num_layers).
int max_resident_layers_from_bytes(size_t expert_cache_bytes, int num_experts,
                                   int hidden_dim, int ffn_dim, int num_layers);

// Apply defaults / derive residency from a VRAM byte budget (0 = unlimited).
MoEConfig normalize_moe_config(MoEConfig cfg, size_t expert_cache_bytes = 0);

// LRU layer residency + async expert-slice prefetch for MoE weights.
//
// When inactive (all layers fit the budget and expert_cache_slots == num_experts),
// registered device pointers are returned unchanged. When active, layer expert
// tensors are mirrored in pinned host memory and a bounded pool of GPU-resident
// layers is maintained; on decode, only the routed top-k expert slices are
// refreshed when expert_cache_slots < num_experts.
class ExpertCache {
public:
    explicit ExpertCache(const MoEConfig& cfg, size_t expert_cache_bytes = 0);
    ~ExpertCache();

    ExpertCache(const ExpertCache&) = delete;
    ExpertCache& operator=(const ExpertCache&) = delete;

    bool active() const { return active_; }

    // Register weights for a layer. Router stays on the supplied device pointer.
    // When the cache is active, expert tensors are copied to pinned host on first use.
    void register_layer(int layer, const LayerWeights& w, cudaStream_t stream);

    // Ensure layer expert weights are GPU-resident for FFN; may evict another layer.
    LayerWeights weights_for_forward(int layer, cudaStream_t stream);

    // After routing: prefetch expert slices for layers [layer+1 .. layer+depth) using
    // the routed expert ids (host-side, top_k * num_tokens ints).
    void prefetch_after_route(int layer, const int* host_expert_ids, int count,
                              cudaStream_t compute_stream);

    // Copy routed expert slices into the layer's GPU buffers (partial residency mode).
    void refresh_routed_experts(int layer, const int* expert_ids, int count, cudaStream_t stream);

    void sync_prefetch(cudaStream_t compute_stream);

    const ExpertCacheStats& stats() const { return stats_; }

private:
    struct LayerStore;

    MoEConfig cfg_;
    size_t    layer_bytes_ = 0;
    int       max_resident_layers_ = 0;
    bool      active_ = false;

    std::vector<LayerStore> layers_;
    uint64_t lru_tick_ = 0;

    cudaStream_t prefetch_stream_ = nullptr;
    cudaEvent_t  prefetch_done_   = nullptr;

    ExpertCacheStats stats_;

    void touch_layer(int layer);
    void evict_one_layer();
    void load_layer_to_gpu(int layer, cudaStream_t stream);
    void ensure_experts_on_gpu(int layer, const int* expert_ids, int count, cudaStream_t stream);
    void prefetch_experts(int layer, const int* expert_ids, int count);
};

}} // namespace sparkinfer::moe
