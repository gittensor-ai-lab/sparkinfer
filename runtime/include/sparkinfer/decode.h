#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/moe/engine.h"

namespace sparkinfer {

struct AttnConfig {
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    float scale;        // 1/sqrt(head_dim)
    float rope_theta = 1e6f;
    float rms_eps    = 1e-6f;
};

// Device weight pointers (bf16) for one MoE transformer layer.
struct TransformerLayerWeights {
    const void* attn_norm = nullptr;   // [hidden]
    const void* wq = nullptr;          // [hidden, num_q_heads*head_dim]
    const void* wk = nullptr;          // [hidden, num_kv_heads*head_dim]
    const void* wv = nullptr;          // [hidden, num_kv_heads*head_dim]
    const void* wo = nullptr;          // [num_q_heads*head_dim, hidden]
    const void* q_norm = nullptr;      // [head_dim] per-head Q RMSNorm (Qwen3; null = skip)
    const void* k_norm = nullptr;      // [head_dim] per-head K RMSNorm (Qwen3; null = skip)
    const void* ffn_norm = nullptr;    // [hidden]
    moe::LayerWeights moe;             // router_w, gate_w, up_w, down_w
};

// Drives a batch of single-token decode steps through the full stack:
//   RMSNorm -> Q/K/V proj -> QK-norm + RoPE + KV append -> GQA flash decode -> O proj
//   -> residual+RMSNorm -> sync-free MoE -> residual.
// Attention uses the gqa8 kernel (Qwen3.5-35B-A3B: 16 Q / 2 KV heads, hd=128).
class DecodeRunner {
public:
    DecodeRunner(int hidden, AttnConfig attn, KVCacheManager* kv, moe::MoEEngine* moe,
                 int max_batch = 256);
    ~DecodeRunner();

    // Begin a decode step: set the new-token position for each sequence
    // (host seq_lens = length BEFORE this token). Uploads positions to device.
    void begin_step(const std::vector<int>& seq_lens_before);

    // Run one layer in-place on x: [num_seqs, hidden] (bf16, device).
    void decode_layer(int layer, void* x, int num_seqs,
                      const TransformerLayerWeights& w, cudaStream_t stream);

private:
    struct Impl;
    Impl* p_;
};

} // namespace sparkinfer
