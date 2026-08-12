#pragma once

#include <vector>

namespace sparkinfer {

// Qwen MoE decode configuration. Defaults match the original full-attention
// Qwen3.5-style target; GGUF metadata can switch this to the Qwen3.5/Qwen3.6
// 35B-A3B hybrid stack with Gated DeltaNet recurrent layers.
struct Qwen35Config {
    int   vocab       = 151936;
    int   hidden      = 2048;
    int   n_layers    = 40;
    int   n_q_heads   = 16;
    int   n_kv_heads  = 2;
    int   head_dim    = 128;
    int   n_experts   = 256;
    int   top_k       = 8;
    int   n_shared    = 1;
    int   moe_ffn     = 512;
    float rope_theta  = 1000000.f;
    float rms_eps     = 1e-6f;
    int   max_seq     = 4096;   // KV-cache cap for a sequence
    int   eos_id      = 151645;
    int   eos_id2     = -1;     // second stop token (e.g. Muse Glimmer's <|eot|> alongside
                                 // <|end_of_text|>); -1 = none, checked wherever eos_id is

    // Qwen3.5/Qwen3.6 35B-A3B GGUFs use a hybrid stack: 3 Gated DeltaNet
    // recurrent layers followed by 1 full-attention layer. The legacy Qwen3
    // MoE path leaves these fields at their defaults.
    bool  hybrid      = false;
    int   full_attn_interval = 4;
    int   rope_dim    = 0;      // 0 = rotate the full attention head
    int   linear_q_heads = 16;
    int   linear_v_heads = 32;
    int   linear_head_dim = 128;
    int   linear_conv_kernel = 4;

    // Qwythos / Qwen3.5-9B dense hybrid GGUFs use a single SwiGLU FFN per layer
    // (ffn_gate/up/down) instead of routed experts.
    bool  dense_ffn   = false;

    // Muse Glimmer: dense GQA transformer, no MoE/linear-attention layers. Per-layer
    // sliding-window (RoPE, window `sliding_window` tokens) vs full/global attention
    // (NoPE -- no RoPE at all), sandwich norm (post_attn_norm already existed; ffn also
    // gets a post-norm), a sigmoid gate on the attention output (reuses the existing
    // q_has_gate path -- see Qwen35LayerWeights::q_has_gate), and tanh logit softcapping.
    bool  muse_glimmer = false;
    std::vector<bool> swa_layers;      // per-layer: true = sliding-window, false = global/NoPE
    int   sliding_window = 0;          // token window for swa_layers entries (0 = disabled)
    float final_logit_softcapping = 0.f;  // 0 = disabled
    float logit_scale = 1.f;              // 1 = no-op

    // Multi-Token Prediction (MTP / NextN) speculative decode head count.
    // 0 = no MTP head in the GGUF; 1 = one NextN block (Qwythos-MTP GGUFs).
    // When > 0 and SPARKINFER_MTP=1, the decode loop runs the MTP head as a
    // speculative draft and accepts greedily (AR-identical output when correct).
    int   n_nextn_layers = 0;
};

} // namespace sparkinfer
