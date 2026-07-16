#pragma once

#include "sparkinfer/gguf.h"
#include "sparkinfer/models/qwen_config.h"

#include <limits>
#include <string>

static long qwen3_meta_int(const sparkinfer::GGUF& g, const std::string& key, long def) {
    const long missing = std::numeric_limits<long>::min();
    long v = g.meta_int("qwen35." + key, missing);
    if (v != missing) return v;
    v = g.meta_int("qwen35moe." + key, missing);
    if (v != missing) return v;
    v = g.meta_int("qwen3moe." + key, missing);
    if (v != missing) return v;
    v = g.meta_int("qwen3_5_moe." + key, missing);
    return v != missing ? v : def;
}

static double qwen3_meta_float(const sparkinfer::GGUF& g, const std::string& key, double def) {
    const double missing = -std::numeric_limits<double>::infinity();
    double v = g.meta_float("qwen35." + key, missing);
    if (v != missing) return v;
    v = g.meta_float("qwen35moe." + key, missing);
    if (v != missing) return v;
    v = g.meta_float("qwen3moe." + key, missing);
    if (v != missing) return v;
    v = g.meta_float("qwen3_5_moe." + key, missing);
    return v != missing ? v : def;
}

static bool qwen3_is_dense_qwen35(const sparkinfer::GGUF& g) {
    const std::string arch = g.meta_str("general.architecture");
    if (arch != "qwen35") return false;
    return g.tensor("blk.0.ffn_gate.weight") != nullptr &&
           g.tensor("blk.0.ffn_gate_exps.weight") == nullptr;
}

static bool qwen3_is_hybrid_35b(const sparkinfer::GGUF& g) {
    const std::string name = g.meta_str("general.name");
    if (name.find("Qwen3.5-35B-A3B") != std::string::npos ||
        name.find("Qwen3.6-35B-A3B") != std::string::npos)
        return true;
    return g.tensor("blk.0.attn_qkv.weight") != nullptr &&
           g.tensor("blk.3.attn_q.weight") != nullptr;
}

static void qwen3_config_from_gguf(const sparkinfer::GGUF& g, sparkinfer::Qwen35Config& cfg) {
    cfg.n_layers   = (int)qwen3_meta_int(g, "block_count", cfg.n_layers);
    cfg.hidden     = (int)qwen3_meta_int(g, "embedding_length", cfg.hidden);
    cfg.n_q_heads  = (int)qwen3_meta_int(g, "attention.head_count", cfg.n_q_heads);
    cfg.n_kv_heads = (int)qwen3_meta_int(g, "attention.head_count_kv", cfg.n_kv_heads);
    cfg.head_dim   = (int)qwen3_meta_int(g, "attention.key_length", cfg.head_dim);
    cfg.n_experts  = (int)qwen3_meta_int(g, "expert_count", cfg.n_experts);
    cfg.top_k      = (int)qwen3_meta_int(g, "expert_used_count", cfg.top_k);
    cfg.moe_ffn    = (int)qwen3_meta_int(g, "expert_feed_forward_length", cfg.moe_ffn);
    cfg.rope_theta = (float)qwen3_meta_float(g, "rope.freq_base", cfg.rope_theta);
    cfg.rms_eps    = (float)qwen3_meta_float(g, "attention.layer_norm_rms_epsilon", cfg.rms_eps);
    cfg.eos_id     = (int)g.meta_int("tokenizer.ggml.eos_token_id", cfg.eos_id);
    cfg.n_shared   = g.tensor("blk.0.ffn_gate_shexp.weight") ? 1 : 0;
    cfg.vocab      = (int)qwen3_meta_int(g, "vocab_size", cfg.vocab);
    const sparkinfer::GGUFTensor* emb = g.tensor("token_embd.weight");
    if (emb && emb->n_dims >= 2) cfg.vocab = (int)emb->dims[1];

    cfg.hybrid = qwen3_is_hybrid_35b(g);
    if (qwen3_is_dense_qwen35(g)) {
        cfg.dense_ffn = true;
        cfg.hybrid = true;
        cfg.n_experts = 1;
        cfg.top_k = 1;
        cfg.n_shared = 0;
        cfg.moe_ffn = (int)qwen3_meta_int(g, "feed_forward_length", cfg.moe_ffn);
        cfg.full_attn_interval = (int)qwen3_meta_int(g, "full_attention_interval", 4);
        cfg.rope_dim = (int)qwen3_meta_int(g, "rope.dimension_count", 64);
        cfg.linear_q_heads = cfg.n_q_heads;
        cfg.linear_v_heads = (int)qwen3_meta_int(g, "ssm.group_count", cfg.linear_v_heads);
        cfg.linear_head_dim = (int)qwen3_meta_int(g, "ssm.state_size", cfg.linear_head_dim);
        cfg.linear_conv_kernel = (int)qwen3_meta_int(g, "ssm.conv_kernel", cfg.linear_conv_kernel);
        if (const sparkinfer::GGUFTensor* qkv = g.tensor("blk.0.attn_qkv.weight")) {
            const int qkv_out = qkv->n_dims >= 2 ? (int)qkv->dims[1] : 0;
            const int q_dim = cfg.linear_q_heads * cfg.linear_head_dim;
            const int v_dim = qkv_out - 2 * q_dim;
            if (v_dim > 0 && v_dim % cfg.linear_head_dim == 0)
                cfg.linear_v_heads = v_dim / cfg.linear_head_dim;
        }
        return;
    }
    if (!cfg.hybrid) {
        cfg.rope_dim = 0;
        return;
    }

    cfg.full_attn_interval = 4;
    cfg.rope_dim = (cfg.head_dim == 256) ? 64 : cfg.rope_dim;
    cfg.linear_head_dim = (int)qwen3_meta_int(g, "ssm.state_size", 128);
    cfg.linear_q_heads = cfg.n_q_heads;
    cfg.linear_v_heads = (int)qwen3_meta_int(g, "ssm.group_count", 32);
    if (const sparkinfer::GGUFTensor* qkv = g.tensor("blk.0.attn_qkv.weight")) {
        const int qkv_out = qkv->n_dims >= 2 ? (int)qkv->dims[1] : 0;
        const int q_dim = cfg.linear_q_heads * cfg.linear_head_dim;
        const int v_dim = qkv_out - 2 * q_dim;
        if (v_dim > 0 && v_dim % cfg.linear_head_dim == 0)
            cfg.linear_v_heads = v_dim / cfg.linear_head_dim;
    }
    cfg.linear_conv_kernel = (int)qwen3_meta_int(g, "ssm.conv_kernel", 4);
    if (const sparkinfer::GGUFTensor* conv = g.tensor("blk.0.ssm_conv1d.weight"))
        if (conv->n_dims >= 1) cfg.linear_conv_kernel = (int)conv->dims[0];
}

static const char* qwen3_model_label(const sparkinfer::Qwen35Config& cfg) {
    if (cfg.dense_ffn) return "Qwen3.5-9B dense hybrid";
    return cfg.hybrid ? "Qwen3.5/Qwen3.6-35B-A3B hybrid" : "Qwen3-MoE";
}
