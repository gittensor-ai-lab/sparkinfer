// MoE engine — sync-free forward pass for one transformer layer.
//
// Pipeline (all on-device, no host sync on the hot path when cache is inactive):
//   logits = input @ router_w                       (launch_moe_router_gemm)
//   top-k  selection + per-expert counts            (launch_moe_router)
//   out    = sum_k weight_k * SwiGLU_FFN_expert_k    (launch_moe_expert_ffn)
//
// When the expert cache is active, layer weights may be loaded from pinned host
// and next-layer expert slices are prefetched on a side stream after routing.

#include "sparkinfer/moe/engine.h"
#include "sparkinfer/moe/expert_cache.h"
#include "sparkinfer/kernels/moe.h"

#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

namespace sparkinfer {
namespace moe {

namespace {
inline void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fprintf(stderr, "[moe] %s: %s\n", what, cudaGetErrorString(e));
}
}

MoEConfig make_moe_config(int num_experts, int top_k, int hidden_dim, int ffn_dim,
                          int num_layers, size_t expert_cache_bytes) {
    MoEConfig cfg{};
    cfg.num_experts = num_experts;
    cfg.top_k = top_k;
    cfg.hidden_dim = hidden_dim;
    cfg.ffn_dim = ffn_dim;
    cfg.num_layers = num_layers;
    cfg.expert_cache_slots = num_experts;
    return normalize_moe_config(cfg, expert_cache_bytes);
}

class MoEEngineImpl : public MoEEngine {
public:
    MoEEngineImpl(const MoEConfig& cfg, size_t expert_cache_bytes)
        : cfg_(normalize_moe_config(cfg, expert_cache_bytes)),
          cache_(cfg_, expert_cache_bytes) {
        weights_.resize(cfg_.num_layers);
        max_tokens_ = 4096;
        cu(cudaMalloc(&d_logits_,  (size_t)max_tokens_ * cfg_.num_experts * sizeof(float)), "malloc logits");
        cu(cudaMalloc(&d_ids_,     (size_t)max_tokens_ * cfg_.top_k * sizeof(int)),         "malloc ids");
        cu(cudaMalloc(&d_weights_, (size_t)max_tokens_ * cfg_.top_k * sizeof(float)),       "malloc weights");
        cu(cudaMalloc(&d_counts_,  (size_t)cfg_.num_experts * sizeof(int)),                 "malloc counts");
        ids_host_.resize((size_t)max_tokens_ * cfg_.top_k);
    }
    ~MoEEngineImpl() override {
        cudaFree(d_logits_); cudaFree(d_ids_); cudaFree(d_weights_); cudaFree(d_counts_);
    }

    void set_layer_weights(int layer, const LayerWeights& w) override {
        if (layer >= 0 && layer < (int)weights_.size()) {
            weights_[layer] = w;
            if (cache_.active()) cache_.register_layer(layer, w, cudaStreamLegacy);
        }
    }

    void forward(const void* input, void* output, int num_tokens, int layer,
                 cudaStream_t stream) override {
        if (num_tokens <= 0) return;
        if (num_tokens > max_tokens_) {
            fprintf(stderr, "[moe] forward: num_tokens %d exceeds scratch capacity %d — skipping\n",
                    num_tokens, max_tokens_);
            return;
        }
        const int E = cfg_.num_experts, K = cfg_.top_k;
        const int H = cfg_.hidden_dim, F = cfg_.ffn_dim;

        LayerWeights w = cache_.active()
            ? cache_.weights_for_forward(layer, stream)
            : weights_[layer];

        kernels::launch_moe_router_gemm(input, w.router_w, d_logits_, num_tokens, H, E, stream);

        cu(cudaMemsetAsync(d_counts_, 0, (size_t)E * sizeof(int), stream), "memset counts");
        kernels::launch_moe_router(d_logits_, d_ids_, d_weights_, d_counts_,
                                   num_tokens, E, K, cfg_.normalize_expert_weights ? 1 : 0, stream);

        const int id_count = num_tokens * K;
        if (cache_.active() && cfg_.expert_cache_slots < E) {
            cu(cudaMemcpyAsync(ids_host_.data(), d_ids_, (size_t)id_count * sizeof(int),
                               cudaMemcpyDeviceToHost, stream), "ids d2h");
            cu(cudaStreamSynchronize(stream), "sync ids");
            cache_.refresh_routed_experts(layer, ids_host_.data(), id_count, stream);
        }

        cu(cudaMemsetAsync(output, 0, (size_t)num_tokens * H * sizeof(unsigned short), stream), "memset out");
        kernels::launch_moe_expert_ffn(input, w.gate_w, w.up_w, w.down_w,
                                       d_ids_, d_weights_, output,
                                       num_tokens, K, E, H, F, stream);

        // Prefetch next-layer expert slices only after this layer's FFN completes so
        // a tight residency pool cannot evict the weights we are still reading.
        if (cache_.active() && cfg_.async_expert_prefetch) {
            if (cfg_.expert_cache_slots >= E) {
                cu(cudaMemcpyAsync(ids_host_.data(), d_ids_, (size_t)id_count * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream), "ids d2h");
                cu(cudaStreamSynchronize(stream), "sync ids");
            }
            cache_.prefetch_after_route(layer, ids_host_.data(), id_count, stream);
        }
    }

    const int* tokens_per_expert() const override { return d_counts_; }
    const MoEConfig& config() const override { return cfg_; }
    ExpertCacheStats cache_stats() const override { return cache_.stats(); }

private:
    MoEConfig cfg_;
    ExpertCache cache_;
    std::vector<LayerWeights> weights_;
    std::vector<int> ids_host_;
    int max_tokens_ = 0;
    float* d_logits_  = nullptr;
    int*   d_ids_     = nullptr;
    float* d_weights_ = nullptr;
    int*   d_counts_  = nullptr;
};

std::unique_ptr<MoEEngine> MoEEngine::create(const MoEConfig& cfg) {
    return std::unique_ptr<MoEEngine>(new MoEEngineImpl(cfg, 0));
}

std::unique_ptr<MoEEngine> MoEEngine::create(const MoEConfig& cfg, size_t expert_cache_bytes) {
    return std::unique_ptr<MoEEngine>(new MoEEngineImpl(cfg, expert_cache_bytes));
}

}} // namespace sparkinfer::moe
