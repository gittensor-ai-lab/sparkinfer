// MoE engine — sync-free forward pass for one transformer layer.
//
// Pipeline (all on-device, no host sync, CUDA-graph capturable):
//   logits = input @ router_w                       (launch_moe_router_gemm)
//   top-k  selection + per-expert counts            (launch_moe_router)
//   out    = sum_k weight_k * SwiGLU_FFN_expert_k    (launch_moe_expert_ffn)
//
// The token-per-expert counter stays in device memory; we never copy it to the
// host, which is what keeps the whole pass inside a single CUDA graph.

#include "sparkinfer/moe/engine.h"
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

class MoEEngineImpl : public MoEEngine {
public:
    explicit MoEEngineImpl(const MoEConfig& cfg) : cfg_(cfg) {
        weights_.resize(cfg.num_layers);
        // Scratch sized for the largest batch we expect to route at once.
        max_tokens_ = 4096;
        cu(cudaMalloc(&d_logits_,  (size_t)max_tokens_ * cfg.num_experts * sizeof(float)), "malloc logits");
        cu(cudaMalloc(&d_ids_,     (size_t)max_tokens_ * cfg.top_k * sizeof(int)),         "malloc ids");
        cu(cudaMalloc(&d_weights_, (size_t)max_tokens_ * cfg.top_k * sizeof(float)),       "malloc weights");
        cu(cudaMalloc(&d_counts_,  (size_t)cfg.num_experts * sizeof(int)),                 "malloc counts");
    }
    ~MoEEngineImpl() override {
        cudaFree(d_logits_); cudaFree(d_ids_); cudaFree(d_weights_); cudaFree(d_counts_);
    }

    void set_layer_weights(int layer, const LayerWeights& w) override {
        if (layer >= 0 && layer < (int)weights_.size()) weights_[layer] = w;
    }

    void forward(const void* input, void* output, int num_tokens, int layer,
                 cudaStream_t stream) override {
        if (num_tokens <= 0) return;
        if (num_tokens > max_tokens_) {
            // Router/top-k scratch is sized for max_tokens_; a larger batch would
            // overflow d_logits_/d_ids_/d_weights_ (OOB device writes). Refuse instead.
            fprintf(stderr, "[moe] forward: num_tokens %d exceeds scratch capacity %d — skipping\n",
                    num_tokens, max_tokens_);
            return;
        }
        const LayerWeights& w = weights_[layer];
        const int E = cfg_.num_experts, K = cfg_.top_k;
        const int H = cfg_.hidden_dim, F = cfg_.ffn_dim;

        // 1. router projection -> logits [num_tokens, E]
        kernels::launch_moe_router_gemm(input, w.router_w, d_logits_, num_tokens, H, E, stream);

        // 2. top-k selection (+ sync-free per-expert counts)
        cu(cudaMemsetAsync(d_counts_, 0, (size_t)E * sizeof(int), stream), "memset counts");
        kernels::launch_moe_router(d_logits_, d_ids_, d_weights_, d_counts_,
                                   num_tokens, E, K, cfg_.normalize_expert_weights ? 1 : 0, stream);

        // 3. fused SwiGLU expert FFN, accumulated over top-k
        cu(cudaMemsetAsync(output, 0, (size_t)num_tokens * H * sizeof(unsigned short), stream), "memset out");
        kernels::launch_moe_expert_ffn(input, w.gate_w, w.up_w, w.down_w,
                                       d_ids_, d_weights_, output,
                                       num_tokens, K, E, H, F, stream);
    }

    const int* tokens_per_expert() const override { return d_counts_; }
    const MoEConfig& config() const override { return cfg_; }

private:
    MoEConfig cfg_;
    std::vector<LayerWeights> weights_;
    int max_tokens_ = 0;
    float* d_logits_  = nullptr;
    int*   d_ids_     = nullptr;
    float* d_weights_ = nullptr;
    int*   d_counts_  = nullptr;
};

std::unique_ptr<MoEEngine> MoEEngine::create(const MoEConfig& cfg) {
    return std::unique_ptr<MoEEngine>(new MoEEngineImpl(cfg));
}

}} // namespace sparkinfer::moe
