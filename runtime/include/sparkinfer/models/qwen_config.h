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
    // Interleaved MRoPE (rope_parameters.mrope_section). Qwen3.5 gives a token THREE rotary
    // positions -- temporal, height, width -- and the frequency index selects which one it reads,
    // laid out [T H W T H W ...] rather than in contiguous chunks.
    //
    // Only the H and W section lengths are stored: the T section is whatever is left over, and
    // the axis rule (i%3, bounded by 3*sec_h / 3*sec_w) needs exactly these two bounds.
    // Zero means the checkpoint declares no mrope_section, i.e. ordinary 1D RoPE.
    //
    // For a TEXT token all three positions are equal, so MRoPE degenerates to 1D RoPE bit for
    // bit. That is why a text-only prompt is unaffected whether or not this is set.
    int   mrope_sec_h = 0;
    int   mrope_sec_w = 0;
    bool  mrope() const { return mrope_sec_h > 0 || mrope_sec_w > 0; }
    int   linear_q_heads = 16;
    int   linear_v_heads = 32;
    int   linear_head_dim = 128;
    int   linear_conv_kernel = 4;
    // "This is a Qwen3.8-27B-family checkpoint" -- drives server-side chat-template behaviour only
    // (ModelEngine::is_qwen38: reasoning-effort system message, enable_thinking default). Its
    // full-attention output gate is a plain sigmoid like every other model here, despite the HF
    // config's output_gate_type: "swish" -- see qwen38_hf_config.h for why that string is a trap.
    bool  qwen38 = false;
    // GDN's v-head -> q/k-head broadcast convention (kernels::launch_qwen36_gdn_ar). false =
    // cyclic (vh % linear_q_heads), validated for Qwythos/Qwen3.6-35B-A3B (v/q ratio 2). true =
    // block (vh / (linear_v_heads/linear_q_heads)), which Qwen3.8-27B's checkpoint needs instead
    // (ratio 3) -- the two checkpoints' own v-head layout conventions differ despite the shared
    // architecture family, confirmed empirically against a real reference implementation.
    bool  gdn_qh_block = false;

    // Dense hybrids use a single SwiGLU FFN per layer (ffn_gate/up/down) instead of routed
    // experts: Qwythos / Qwen3.5-9B, and Qwen3.8-27B (the scored checkpoint). Note that the
    // decode FFN still dispatches through the MoE kernels with n_experts==1/top_k==1, which is
    // why several of those kernels carry a dense special-case -- see qwen35.cpp's FFN branch.
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

    // Spark-X2.5 (iFlytek, model_type "spark2_5"): dense GQA transformer, 3 sliding-window
    // layers to every 1 full-attention layer, reusing swa_layers/sliding_window above. What it
    // does NOT share with Muse Glimmer, the other swa_layers consumer:
    //
    //   * BOTH layer kinds apply RoPE. Muse's non-SWA layers are NoPE; Spark's differ only in
    //     their rotary PARAMETERS, so a `!w.swa` layer here must still rotate.
    //   * Those parameters are per-layer-kind, which is why rope_theta/rope_dim alone cannot
    //     describe this model -- see rope_theta_swa/rope_dim_swa below.
    //   * No QK-norm at all (no attn_q_norm/attn_k_norm tensors; Spark2_5Attention normalizes
    //     neither), unlike every other architecture wired up here.
    //   * Two norms per layer (attn_norm, ffn_norm), not Muse's four-norm sandwich.
    //   * GeGLU rather than SwiGLU -- see ffn_gelu.
    //   * The attention output gate is one scalar PER HEAD, not per head-element -- see
    //     headwise_attn_gate.
    bool  spark25 = false;
    // Rotary parameters for the sliding-window layers. rope_theta/rope_dim above stay the
    // FULL-attention layers' values, so every model that has only one layer kind is unaffected.
    // Spark-X2.5-4B: full = (5e6, 64 of 256 dims); sliding = (1e4, all 256). Left at 0 by every
    // other checkpoint, which resolve_rope() then reads as "same as the full-attention layers".
    float rope_theta_swa = 0.f;
    int   rope_dim_swa   = 0;
    // FFN activation: down(act(gate(x)) * up(x)) with act = GELU instead of SiLU. Spark-X2.5's
    // config.json pins hidden_act to "gelu" and its reference MLP refuses to build with anything
    // else. transformers' ACT2FN["gelu"] is the ERF form, not the tanh approximation.
    bool  ffn_gelu = false;
    // Attention output gate shape. false (Qwen3.6/Qwen3.8/Muse): the gate projection is
    // [hidden, n_q_heads*head_dim] and multiplies the attention output elementwise. true
    // (Spark-X2.5, config.json's headwise_attn_output_gate): it is [hidden, n_q_heads] -- ONE
    // sigmoid scalar per head, broadcast across that head's head_dim elements.
    bool  headwise_attn_gate = false;
    // Some checkpoints ship no attn_q_norm/attn_k_norm (Spark-X2.5). Kept explicit rather than
    // inferred from a null q_norm so a genuinely failed norm load still fails loudly.
    bool  no_qk_norm = false;

    // Rotary parameters actually in force on layer `i`, honouring the per-layer-kind split
    // above. Every non-Spark model returns (rope_theta, rope_dim) for every layer.
    float rope_theta_for(int i) const {
        const bool sw = i >= 0 && i < (int)swa_layers.size() && swa_layers[i];
        return (sw && rope_theta_swa > 0.f) ? rope_theta_swa : rope_theta;
    }
    int rope_dim_for(int i) const {
        const bool sw = i >= 0 && i < (int)swa_layers.size() && swa_layers[i];
        return (sw && rope_dim_swa > 0) ? rope_dim_swa : rope_dim;
    }
};

} // namespace sparkinfer
