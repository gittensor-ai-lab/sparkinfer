// Expert weight cache — pinned host backing, bounded GPU layer residency, and
// async prefetch of next-layer expert slices on a side stream.

#include "sparkinfer/moe/expert_cache.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace sparkinfer {
namespace moe {

namespace {
inline void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fprintf(stderr, "[expert_cache] %s: %s\n", what, cudaGetErrorString(e));
}

constexpr size_t kBf16 = 2;

size_t tensor_bytes(int a, int b, int c) {
    return (size_t)a * (size_t)b * (size_t)c * kBf16;
}
} // namespace

size_t expert_layer_bytes(int num_experts, int hidden_dim, int ffn_dim) {
    const size_t gate = tensor_bytes(num_experts, hidden_dim, ffn_dim);
    const size_t up   = gate;
    const size_t down = tensor_bytes(num_experts, ffn_dim, hidden_dim);
    return gate + up + down;
}

int max_resident_layers_from_bytes(size_t expert_cache_bytes, int num_experts,
                                   int hidden_dim, int ffn_dim, int num_layers) {
    if (expert_cache_bytes == 0) return num_layers;
    const size_t layer_bytes = expert_layer_bytes(num_experts, hidden_dim, ffn_dim);
    const int n = (int)std::max<size_t>(1, expert_cache_bytes / std::max<size_t>(1, layer_bytes));
    return std::min(n, num_layers);
}

MoEConfig normalize_moe_config(MoEConfig cfg, size_t expert_cache_bytes) {
    if (cfg.expert_cache_slots <= 0) cfg.expert_cache_slots = cfg.num_experts;
    if (cfg.async_prefetch_depth <= 0) cfg.async_prefetch_depth = 1;
    (void)expert_cache_bytes;
    return cfg;
}

struct ExpertCache::LayerStore {
    LayerWeights source{};
    void* host_gate = nullptr;
    void* host_up   = nullptr;
    void* host_down = nullptr;
    void* gpu_gate  = nullptr;
    void* gpu_up    = nullptr;
    void* gpu_down  = nullptr;
    bool  host_ready = false;
    bool  gpu_resident = false;
    bool  owns_host = false;
    bool  owns_gpu  = false;
    uint64_t lru_tick = 0;
    // Per-expert LRU within a resident layer (meaningful when slots < num_experts).
    std::vector<uint64_t> expert_tick;
};

ExpertCache::ExpertCache(const MoEConfig& cfg, size_t expert_cache_bytes)
    : cfg_(normalize_moe_config(cfg, expert_cache_bytes)) {
    layer_bytes_  = expert_layer_bytes(cfg_.num_experts, cfg_.hidden_dim, cfg_.ffn_dim);
    layers_.resize((size_t)cfg_.num_layers);

    const bool partial_experts = cfg_.expert_cache_slots < cfg_.num_experts;
    if (expert_cache_bytes > 0) {
        max_resident_layers_ = max_resident_layers_from_bytes(
            expert_cache_bytes, cfg_.num_experts, cfg_.hidden_dim, cfg_.ffn_dim, cfg_.num_layers);
    } else {
        max_resident_layers_ = cfg_.num_layers;
    }

    active_ = partial_experts || max_resident_layers_ < cfg_.num_layers;
    stats_.max_resident_layers = max_resident_layers_;

    if (active_) {
        cu(cudaStreamCreateWithFlags(&prefetch_stream_, cudaStreamNonBlocking), "prefetch stream");
        cu(cudaEventCreateWithFlags(&prefetch_done_, cudaEventDisableTiming), "prefetch event");
    }
}

ExpertCache::~ExpertCache() {
    for (auto& L : layers_) {
        if (L.owns_host) {
            if (L.host_gate) cudaFreeHost(L.host_gate);
            if (L.host_up)   cudaFreeHost(L.host_up);
            if (L.host_down) cudaFreeHost(L.host_down);
        }
        if (L.owns_gpu) {
            if (L.gpu_gate) cudaFree(L.gpu_gate);
            if (L.gpu_up)   cudaFree(L.gpu_up);
            if (L.gpu_down) cudaFree(L.gpu_down);
        }
    }
    if (prefetch_stream_) cudaStreamDestroy(prefetch_stream_);
    if (prefetch_done_)   cudaEventDestroy(prefetch_done_);
}

void ExpertCache::register_layer(int layer, const LayerWeights& w, cudaStream_t stream) {
    if (layer < 0 || layer >= cfg_.num_layers) return;
    if (!stream) stream = cudaStreamLegacy;
    LayerStore& L = layers_[(size_t)layer];
    L.source = w;
    if (!active_) return;
    if (L.host_ready) return;
    if (!w.gate_w || !w.up_w || !w.down_w) return;

    const int E = cfg_.num_experts, H = cfg_.hidden_dim, F = cfg_.ffn_dim;
    const size_t gate_sz = tensor_bytes(E, H, F);
    const size_t up_sz   = gate_sz;
    const size_t down_sz = tensor_bytes(E, F, H);

    cu(cudaHostAlloc(&L.host_gate, gate_sz, cudaHostAllocPortable), "host gate");
    cu(cudaHostAlloc(&L.host_up,   up_sz,   cudaHostAllocPortable), "host up");
    cu(cudaHostAlloc(&L.host_down, down_sz, cudaHostAllocPortable), "host down");
    L.owns_host = true;

    cu(cudaMemcpyAsync(L.host_gate, w.gate_w, gate_sz, cudaMemcpyDeviceToHost, stream), "pin gate");
    cu(cudaMemcpyAsync(L.host_up,   w.up_w,   up_sz,   cudaMemcpyDeviceToHost, stream), "pin up");
    cu(cudaMemcpyAsync(L.host_down, w.down_w, down_sz, cudaMemcpyDeviceToHost, stream), "pin down");
    cu(cudaStreamSynchronize(stream), "sync pin");
    L.host_ready = true;
    L.expert_tick.assign((size_t)E, 0);
}

void ExpertCache::touch_layer(int layer) {
    layers_[(size_t)layer].lru_tick = ++lru_tick_;
}

void ExpertCache::evict_one_layer() {
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < cfg_.num_layers; i++) {
        LayerStore& L = layers_[(size_t)i];
        if (!L.gpu_resident) continue;
        if (L.lru_tick < oldest) { oldest = L.lru_tick; victim = i; }
    }
    if (victim < 0) return;
    LayerStore& L = layers_[(size_t)victim];
    if (L.owns_gpu) {
        cudaFree(L.gpu_gate); L.gpu_gate = nullptr;
        cudaFree(L.gpu_up);   L.gpu_up   = nullptr;
        cudaFree(L.gpu_down); L.gpu_down = nullptr;
        L.owns_gpu = false;
    }
    L.gpu_resident = false;
}

void ExpertCache::load_layer_to_gpu(int layer, cudaStream_t stream) {
    if (!stream) stream = cudaStreamLegacy;
    LayerStore& L = layers_[(size_t)layer];
    if (L.gpu_resident) return;
    if (!L.host_ready) {
        register_layer(layer, L.source, stream);
        if (!L.host_ready) return;
    }

    int resident = 0;
    for (const auto& s : layers_) if (s.gpu_resident) resident++;
    while (resident >= max_resident_layers_) {
        evict_one_layer();
        resident--;
    }

    const int E = cfg_.num_experts, H = cfg_.hidden_dim, F = cfg_.ffn_dim;
    const size_t gate_sz = tensor_bytes(E, H, F);
    const size_t up_sz   = gate_sz;
    const size_t down_sz = tensor_bytes(E, F, H);

    if (!L.gpu_gate) cu(cudaMalloc(&L.gpu_gate, gate_sz), "gpu gate");
    if (!L.gpu_up)   cu(cudaMalloc(&L.gpu_up,   up_sz),   "gpu up");
    if (!L.gpu_down) cu(cudaMalloc(&L.gpu_down, down_sz), "gpu down");
    L.owns_gpu = true;

    const bool full_layer = cfg_.expert_cache_slots >= cfg_.num_experts;
    if (full_layer) {
        cu(cudaMemcpyAsync(L.gpu_gate, L.host_gate, gate_sz, cudaMemcpyHostToDevice, stream), "load gate");
        cu(cudaMemcpyAsync(L.gpu_up,   L.host_up,   up_sz,   cudaMemcpyHostToDevice, stream), "load up");
        cu(cudaMemcpyAsync(L.gpu_down, L.host_down, down_sz, cudaMemcpyHostToDevice, stream), "load down");
    }
    L.gpu_resident = true;
    stats_.layer_misses++;
    touch_layer(layer);
}

void ExpertCache::refresh_routed_experts(int layer, const int* expert_ids, int count,
                                          cudaStream_t stream) {
    if (!active_) return;
    if (!stream) stream = cudaStreamLegacy;
    ensure_experts_on_gpu(layer, expert_ids, count, stream);
}

void ExpertCache::ensure_experts_on_gpu(int layer, const int* expert_ids, int count,
                                        cudaStream_t stream) {
    LayerStore& L = layers_[(size_t)layer];
    if (!L.gpu_resident) return;

    const int H = cfg_.hidden_dim, F = cfg_.ffn_dim;
    const size_t gate_stride = tensor_bytes(1, H, F);
    const size_t up_stride   = gate_stride;
    const size_t down_stride = tensor_bytes(1, F, H);

    std::unordered_set<int> uniq;
    for (int i = 0; i < count; i++) {
        if (expert_ids[i] >= 0 && expert_ids[i] < cfg_.num_experts) uniq.insert(expert_ids[i]);
    }

    for (int e : uniq) {
        L.expert_tick[(size_t)e] = ++lru_tick_;
        const size_t off_gate = (size_t)e * gate_stride;
        const size_t off_up   = (size_t)e * up_stride;
        const size_t off_down = (size_t)e * down_stride;
        cu(cudaMemcpyAsync((char*)L.gpu_gate + off_gate, (char*)L.host_gate + off_gate,
                           gate_stride, cudaMemcpyHostToDevice, stream), "expert gate");
        cu(cudaMemcpyAsync((char*)L.gpu_up + off_up, (char*)L.host_up + off_up,
                           up_stride, cudaMemcpyHostToDevice, stream), "expert up");
        cu(cudaMemcpyAsync((char*)L.gpu_down + off_down, (char*)L.host_down + off_down,
                           down_stride, cudaMemcpyHostToDevice, stream), "expert down");
        stats_.expert_loads++;
    }
}

LayerWeights ExpertCache::weights_for_forward(int layer, cudaStream_t stream) {
    if (!active_) return layers_[(size_t)layer].source;

    LayerStore& L = layers_[(size_t)layer];
    if (L.gpu_resident) {
        stats_.layer_hits++;
        touch_layer(layer);
    } else {
        load_layer_to_gpu(layer, stream);
    }

    LayerWeights out{};
    out.router_w = L.source.router_w;
    if (L.gpu_resident) {
        out.gate_w = L.gpu_gate;
        out.up_w   = L.gpu_up;
        out.down_w = L.gpu_down;
    } else {
        out = L.source;
    }
    return out;
}

void ExpertCache::prefetch_experts(int layer, const int* expert_ids, int count) {
    if (layer < 0 || layer >= cfg_.num_layers) return;
    LayerStore& L = layers_[(size_t)layer];
    if (!L.host_ready) return;

    // Bring the layer's GPU buffers resident (may evict another layer).
    if (!L.gpu_resident) {
        int resident = 0;
        for (const auto& s : layers_) if (s.gpu_resident) resident++;
        while (resident >= max_resident_layers_) {
            evict_one_layer();
            resident--;
        }
        const int E = cfg_.num_experts, H = cfg_.hidden_dim, F = cfg_.ffn_dim;
        const size_t gate_sz = tensor_bytes(E, H, F);
        const size_t up_sz   = gate_sz;
        const size_t down_sz = tensor_bytes(E, F, H);
        if (!L.gpu_gate) cu(cudaMalloc(&L.gpu_gate, gate_sz), "prefetch gpu gate");
        if (!L.gpu_up)   cu(cudaMalloc(&L.gpu_up,   up_sz),   "prefetch gpu up");
        if (!L.gpu_down) cu(cudaMalloc(&L.gpu_down, down_sz), "prefetch gpu down");
        L.owns_gpu = true;
        L.gpu_resident = true;
        touch_layer(layer);
    }

    const int H = cfg_.hidden_dim, F = cfg_.ffn_dim;
    const size_t gate_stride = tensor_bytes(1, H, F);
    const size_t up_stride   = gate_stride;
    const size_t down_stride = tensor_bytes(1, F, H);

    std::unordered_set<int> uniq;
    for (int i = 0; i < count; i++) {
        if (expert_ids[i] >= 0 && expert_ids[i] < cfg_.num_experts) uniq.insert(expert_ids[i]);
    }

    for (int e : uniq) {
        const size_t off_gate = (size_t)e * gate_stride;
        const size_t off_up   = (size_t)e * up_stride;
        const size_t off_down = (size_t)e * down_stride;
        cu(cudaMemcpyAsync((char*)L.gpu_gate + off_gate, (char*)L.host_gate + off_gate,
                           gate_stride, cudaMemcpyHostToDevice, prefetch_stream_), "prefetch gate");
        cu(cudaMemcpyAsync((char*)L.gpu_up + off_up, (char*)L.host_up + off_up,
                           up_stride, cudaMemcpyHostToDevice, prefetch_stream_), "prefetch up");
        cu(cudaMemcpyAsync((char*)L.gpu_down + off_down, (char*)L.host_down + off_down,
                           down_stride, cudaMemcpyHostToDevice, prefetch_stream_), "prefetch down");
        stats_.prefetches++;
    }
}

void ExpertCache::prefetch_after_route(int layer, const int* host_expert_ids, int count,
                                       cudaStream_t compute_stream) {
    if (!active_ || !cfg_.async_expert_prefetch || !prefetch_stream_) return;
    cu(cudaEventRecord(prefetch_done_, compute_stream), "route event");
    cu(cudaStreamWaitEvent(prefetch_stream_, prefetch_done_, 0), "prefetch wait route");

    for (int d = 1; d <= cfg_.async_prefetch_depth; d++) {
        const int next = layer + d;
        if (next >= cfg_.num_layers) break;
        prefetch_experts(next, host_expert_ids, count);
    }
}

void ExpertCache::sync_prefetch(cudaStream_t compute_stream) {
    if (!active_ || !prefetch_stream_) return;
    cu(cudaEventRecord(prefetch_done_, prefetch_stream_), "prefetch done");
    cu(cudaStreamWaitEvent(compute_stream, prefetch_done_, 0), "compute wait prefetch");
}

}} // namespace sparkinfer::moe
