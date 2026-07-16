#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>

namespace sparkinfer { namespace moe {

enum class RouterType {
    SOFTMAX_TOP_K,     // Standard: softmax over all experts, pick top-k
    SIGMOID_TOP_K,     // Per-expert sigmoid (DeepSeek-V2 style)
    EXPERT_CHOICE,     // Each expert selects its tokens (capacity-based)
};

struct RouterConfig {
    RouterType type = RouterType::SOFTMAX_TOP_K;
    int num_experts;
    int top_k;
    float capacity_factor = 1.25f;  // for EXPERT_CHOICE
    bool drop_tokens = false;       // drop overflow tokens (lower latency, lower quality)
    bool normalize_weights = true;  // normalize top-k weights to sum=1
    float aux_loss_coeff = 1e-2f;   // load-balancing auxiliary loss weight

    // Keep token counts in GPU memory only — enables CUDA graph capture.
    // When true, last_expert_counts() is unavailable (returns empty vector).
    bool sync_free = true;
};

class Router {
public:
    explicit Router(const RouterConfig& cfg);
    ~Router();

    // Compute routing decisions on device.
    // router_logits: [num_tokens, num_experts]  (float, device ptr)
    // expert_ids:    [num_tokens, top_k]         (int32, device ptr, output)
    // expert_w:      [num_tokens, top_k]         (float, device ptr, output)
    // tokens_per_expert: [num_experts]           (int32, device ptr, output) — GPU only if sync_free
    void route(
        const float* router_logits,
        int* expert_ids, float* expert_weights,
        int* tokens_per_expert,
        int num_tokens,
        cudaStream_t stream
    );

    // Returns per-expert token counts from last route() call.
    // Returns empty vector if sync_free=true (no CPU readback).
    std::vector<int> last_expert_counts() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}} // namespace sparkinfer::moe
