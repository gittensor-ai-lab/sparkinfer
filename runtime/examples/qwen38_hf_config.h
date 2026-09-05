#pragma once

// Qwen3.8-27B: dense hybrid Gated-DeltaNet/full-attention text model, same architecture
// family as the existing Qwythos/Qwen3.5-9B dense-hybrid GGUF path (hybrid=true,
// dense_ffn=true) but with its own dimensions and, crucially, an independent
// linear-attention head count (24 full-attention heads vs. 16 linear-attention heads --
// unlike Qwythos where both happen to be 16, see the note at cfg.linear_q_heads below).
// The vision tower and MTP head are intentionally not modeled here (text-only, no MTP --
// see the Qwen3.8-27B plan). No GGUF exists for this model; config comes straight from the
// HF checkout's config.json (`text_config` block) + generation_config.json (stop-token
// ids), which is a flat, standard-key JSON file -- no namespace-fallback-chain archaeology
// like qwen3_gguf_config.h's GGUF-metadata reads need.
//
// Only included by server/src/model_engine.cpp today, which already links nlohmann_json
// (see server/CMakeLists.txt) -- the runtime library itself links no JSON dependency (see
// runtime/src/safetensors.cpp's own hand-rolled parser for why), so this header is
// deliberately kept out of runtime/src/ and only pulled in by a translation unit that
// already has the dependency, mirroring where qwen3_gguf_config.h itself lives.

#include "sparkinfer/models/qwen_config.h"

#include <fstream>
#include <string>
#include <nlohmann/json.hpp>

static bool qwen38_config_from_hf_json(const std::string& model_dir,
                                       sparkinfer::Qwen35Config& cfg, std::string& err) {
    std::ifstream cf(model_dir + "/config.json");
    if (!cf) { err = "config.json not found in " + model_dir; return false; }
    nlohmann::json root;
    try {
        cf >> root;
    } catch (const std::exception& e) {
        err = std::string("config.json parse error: ") + e.what();
        return false;
    }
    if (!root.contains("text_config") || !root["text_config"].is_object()) {
        err = "config.json missing text_config block";
        return false;
    }
    const auto& tc = root["text_config"];

    auto geti = [&](const char* key, int def) {
        return tc.contains(key) && tc[key].is_number_integer() ? tc[key].get<int>() : def;
    };
    auto getf = [&](const char* key, float def) {
        return tc.contains(key) && tc[key].is_number() ? tc[key].get<float>() : def;
    };

    cfg.hidden     = geti("hidden_size", cfg.hidden);
    cfg.n_layers   = geti("num_hidden_layers", cfg.n_layers);
    cfg.vocab      = geti("vocab_size", cfg.vocab);
    cfg.n_q_heads  = geti("num_attention_heads", cfg.n_q_heads);
    cfg.n_kv_heads = geti("num_key_value_heads", cfg.n_kv_heads);
    cfg.head_dim   = geti("head_dim", cfg.head_dim);
    cfg.rms_eps    = getf("rms_norm_eps", cfg.rms_eps);

    // rope_theta lives nested under text_config.rope_parameters, not at text_config's own
    // top level (unlike partial_rotary_factor below, which happens to be duplicated at both
    // levels) -- confirmed against the actual released config.json, not assumed.
    float partial_rotary = 1.f;
    if (tc.contains("rope_parameters") && tc["rope_parameters"].is_object()) {
        const auto& rp = tc["rope_parameters"];
        if (rp.contains("rope_theta") && rp["rope_theta"].is_number())
            cfg.rope_theta = rp["rope_theta"].get<float>();
        if (rp.contains("partial_rotary_factor") && rp["partial_rotary_factor"].is_number())
            partial_rotary = rp["partial_rotary_factor"].get<float>();
        // mrope_section is [T, H, W] frequency-band lengths summing to rope_dim/2. Only read when
        // the checkpoint declares it; absent means ordinary 1D RoPE, which is the correct reading
        // for every non-multimodal Qwen in this codebase.
        if (rp.contains("mrope_section") && rp["mrope_section"].is_array() &&
            rp["mrope_section"].size() >= 3) {
            const auto& ms = rp["mrope_section"];
            if (ms[1].is_number_integer() && ms[2].is_number_integer()) {
                cfg.mrope_sec_h = ms[1].get<int>();
                cfg.mrope_sec_w = ms[2].get<int>();
            }
        }
    }
    partial_rotary = getf("partial_rotary_factor", partial_rotary);   // top-level override/fallback

    // Dense FFN (no MoE routing) -- single intermediate_size, reuses the dense-hybrid
    // ("Qwythos") code path via moe_ffn repurposed as the single dense FFN width, same as
    // museglimmer_config_from_gguf does for Muse Glimmer's own dense FFN.
    cfg.dense_ffn = true;
    cfg.n_experts = 1; cfg.top_k = 1; cfg.n_shared = 0;
    cfg.moe_ffn = geti("intermediate_size", cfg.moe_ffn);

    // Hybrid Gated-DeltaNet stack: 3 linear-attention layers per 1 full-attention layer.
    cfg.hybrid = true;
    cfg.full_attn_interval = geti("full_attention_interval", 4);
    cfg.rope_dim = (int)(partial_rotary * cfg.head_dim);
    // Deliberately NOT cfg.n_q_heads -- that shortcut (qwen3_gguf_config.h:136,157) only
    // holds for the existing Qwythos/Qwen3.5-9B model, where both head counts happen to be
    // 16. Here they differ (24 full-attention heads vs. 16 linear-attention heads), so the
    // linear-attention head count MUST be read from its own key, never derived from n_q_heads.
    cfg.linear_q_heads     = geti("linear_num_key_heads", cfg.linear_q_heads);
    cfg.linear_v_heads     = geti("linear_num_value_heads", cfg.linear_v_heads);
    cfg.linear_head_dim    = geti("linear_key_head_dim", cfg.linear_head_dim);
    cfg.linear_conv_kernel = geti("linear_conv_kernel_dim", cfg.linear_conv_kernel);

    // NOTE: the full-attention output gate is a plain SIGMOID (attn * sigmoid(gate)), the same as
    // every other model here -- despite this config's output_gate_type: "swish", which does NOT
    // describe that multiply. Confirmed against llama.cpp's qwen35 implementation (ggml_sigmoid on
    // the gate view) and its coherent reference output. Using SiLU sign-flips the gated attention
    // output (the gate is strongly negative, mean ~ -4.5) and corrupts every layer from the first
    // full-attention one onward, so do not "fix" this to match the config string.
    cfg.qwen38 = true;

    // See Qwen35Config::gdn_qh_block's own comment -- this checkpoint's GDN v-head layout needs
    // block broadcast, not the cyclic convention Qwythos/Qwen3.6-35B-A3B validated.
    cfg.gdn_qh_block = true;

    cfg.muse_glimmer = false;

    // Stop tokens: generation_config.json's eos_token_id is a LIST here (unlike GGUF's single
    // scalar metadata key) -- e.g. [248046, 248044]. Falls back to config.json's own
    // (singular) eos_token_id if generation_config.json is missing or malformed.
    cfg.eos_id = -1;
    cfg.eos_id2 = -1;
    std::ifstream gf(model_dir + "/generation_config.json");
    if (gf) {
        nlohmann::json groot;
        try { gf >> groot; } catch (const std::exception&) { /* fall through to config.json below */ }
        if (groot.contains("eos_token_id")) {
            if (groot["eos_token_id"].is_array()) {
                const auto& arr = groot["eos_token_id"];
                if (arr.size() > 0 && arr[0].is_number_integer()) cfg.eos_id = arr[0].get<int>();
                if (arr.size() > 1 && arr[1].is_number_integer()) cfg.eos_id2 = arr[1].get<int>();
            } else if (groot["eos_token_id"].is_number_integer()) {
                cfg.eos_id = groot["eos_token_id"].get<int>();
            }
        }
    }
    if (cfg.eos_id < 0) cfg.eos_id = geti("eos_token_id", 151645);

    return true;
}

static const char* qwen38_model_label() { return "Qwen3.8-27B dense hybrid (text-only)"; }
