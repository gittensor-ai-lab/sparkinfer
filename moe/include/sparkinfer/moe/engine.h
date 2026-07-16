#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace sparkinfer { namespace moe {

struct MoEConfig {
    int num_experts;
    int top_k;
    int hidden_dim;
    int ffn_dim;
    int num_layers;

    // Expert residency budget: how many expert weight matrices to keep in VRAM.
    // Remaining experts are evicted to CPU unified memory and fetched on demand.
    int expert_cache_slots;   // default: num_experts (all in VRAM)

    // Prefetch next-layer experts while current layer executes
    bool async_expert_prefetch = true;

    // Normalize routing weights after top-k selection
    bool normalize_expert_weights = true;

    // Sync-free mode: token counts stay on GPU, never read to CPU.
    // Required for CUDA graph capture of the full MoE forward pass.
    bool sync_free = true;
};

// Device pointers (bf16) for one layer's MoE weights. All live in VRAM (or
// unified memory). Supplied by the runtime, which owns weight loading.
//   router_w: [hidden_dim, num_experts]            (pre-transposed for the GEMM)
//   gate_w:   [num_experts, hidden_dim, ffn_dim]
//   up_w:     [num_experts, hidden_dim, ffn_dim]
//   down_w:   [num_experts, ffn_dim, hidden_dim]
struct LayerWeights {
    const void* router_w = nullptr;
    const void* gate_w   = nullptr;
    const void* up_w     = nullptr;
    const void* down_w   = nullptr;
};

class MoEEngine {
public:
    static std::unique_ptr<MoEEngine> create(const MoEConfig& cfg);
    virtual ~MoEEngine() = default;

    // Register device weight pointers for a layer.
    virtual void set_layer_weights(int layer_idx, const LayerWeights& w) = 0;

    // Route tokens and dispatch through expert FFNs (sync-free, CUDA-graph safe).
    //   input:  [num_tokens, hidden_dim]  (bf16, device ptr)
    //   output: [num_tokens, hidden_dim]  (bf16, device ptr; zeroed internally)
    virtual void forward(
        const void* input, void* output,
        int num_tokens, int layer_idx,
        cudaStream_t stream
    ) = 0;

    // Device pointer to the sync-free per-expert token counter from the last
    // forward() (length num_experts, int32). Never read back to host in
    // sync-free mode; exposed for diagnostics / load-balancing kernels.
    virtual const int* tokens_per_expert() const = 0;

    virtual const MoEConfig& config() const = 0;
};

}} // namespace sparkinfer::moe
