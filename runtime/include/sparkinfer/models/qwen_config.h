#pragma once

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
};

} // namespace sparkinfer
