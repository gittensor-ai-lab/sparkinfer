#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/moe/engine.h"

namespace sparkinfer {

class ThermalGovernor;

// Gemma 4 26B-A4B architecture (5L1G interleaved local/global attention).
//   30 layers, hidden 2112, 128 routed experts (top-8) + 1 shared, moe ffn 704,
//   local: 16 Q / 8 KV, head_dim 256, sliding window 1024, RoPE θ=10k
//   global: 16 Q / 2 KV, head_dim 512, full context, RoPE θ=1M
struct Gemma4Config {
    int   vocab              = 262144;
    int   hidden             = 2112;
    int   n_layers           = 30;
    int   n_q_heads          = 16;
    int   local_head_dim     = 256;
    int   global_head_dim    = 512;
    int   local_n_kv_heads   = 8;
    int   global_n_kv_heads  = 2;
    int   local_window       = 1024;
    float local_rope_theta   = 10000.f;
    float global_rope_theta  = 1000000.f;
    int   n_experts          = 128;
    int   top_k              = 8;
    int   n_shared           = 1;
    int   moe_ffn            = 704;
    float rms_eps            = 1e-6f;
    int   max_seq            = 8192;
    int   eos_id             = 1;
};

// True for global-attention layers in the 5L1G pattern (layers 5, 11, 17, 23, 29).
inline bool gemma4_is_global_layer(int layer) { return layer % 6 == 5; }

struct Gemma4LayerWeights {
    const void* input_norm    = nullptr;
    const void* wq = nullptr;
    const void* wk = nullptr;
    const void* wv = nullptr;
    const void* wo = nullptr;
    const void* q_norm = nullptr;
    const void* k_norm = nullptr;
    const void* post_attn_norm = nullptr;
    const void* router_w = nullptr;
    const void* gate = nullptr;
    const void* up   = nullptr;
    const void* down = nullptr;
    const void* shared_gate = nullptr;
    const void* shared_up   = nullptr;
    const void* shared_down = nullptr;

    const void* gate_q = nullptr;
    const void* up_q   = nullptr;
    const void* down_q = nullptr;
    int gate_qtype = 0, up_qtype = 0, down_qtype = 0;
    int wq_type = 0, wk_type = 0, wv_type = 0, wo_type = 0;
};

struct Gemma4Weights {
    const void* embed_tokens = nullptr;
    const void* final_norm   = nullptr;
    const void* lm_head      = nullptr;
    int lm_head_type = 0;
    std::vector<Gemma4LayerWeights> layers;
};

// Per-layer attention geometry for the 5L1G pattern.
struct Gemma4LayerAttn {
    int   head_dim;
    int   num_kv_heads;
    int   qdim;
    int   kvdim;
    float rope_theta;
    bool  global;
};

inline Gemma4LayerAttn gemma4_layer_attn(int layer, const Gemma4Config& cfg) {
    const bool g = gemma4_is_global_layer(layer);
    const int hd = g ? cfg.global_head_dim : cfg.local_head_dim;
    const int nkv = g ? cfg.global_n_kv_heads : cfg.local_n_kv_heads;
    return {hd, nkv, cfg.n_q_heads * hd, nkv * hd,
            g ? cfg.global_rope_theta : cfg.local_rope_theta, g};
}

class Gemma4Model {
public:
    Gemma4Model(const Gemma4Config& cfg, moe::MoEEngine* engine);
    ~Gemma4Model();

    void set_weights(const Gemma4Weights& w);
    bool load_weights(const std::string& dir);
    bool load_gguf(const std::string& path);

    std::vector<int> generate(const std::vector<int>& prompt_ids, int max_new_tokens,
                              ThermalGovernor* gov = nullptr);
    int forward_token(int token_id, int position);
    void copy_logits(float* host_logits) const;
    double bench_decode(int warmup, int n_tokens);

    const Gemma4Config& config() const;

private:
    struct Impl;
    Impl* p_;
};

} // namespace sparkinfer
