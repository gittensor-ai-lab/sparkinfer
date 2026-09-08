// CPU-only test for Spark-X2.5 (general.architecture "spark2_5") GGUF metadata parsing.
//
// Writes a tiny GGUF carrying this architecture's metadata -- including the BOOL
// sliding_window_pattern array and the _swa-suffixed rotary keys that no other checkpoint here
// uses -- and checks spark25_config_from_gguf() derives Qwen35Config correctly.
//
// What is actually at stake in each group of assertions:
//   * the per-layer-kind rotary split. rope_theta/rope_dim describe the FULL-attention layers and
//     rope_theta_swa/rope_dim_swa the sliding ones; getting these crossed rotates 3 layers in 4 at
//     the wrong frequency, which produces fluent-looking but wrong text rather than a crash.
//   * hybrid=true with full_attn_interval=0. Together these mean "batched-prefill-capable, and
//     is_linear_layer() false for every layer". A nonzero interval would send layers looking for
//     Gated-DeltaNet ssm_* tensors this checkpoint does not have.
//   * the sliding_window_pattern being read as BOOL. GGUF writes it as an array of bools; if
//     GGUF::meta_int_array ever stopped capturing VT_BOOL the vector would come back empty and
//     every layer would silently become full-attention.

#include "../examples/qwen3_gguf_config.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {
enum { VT_U32 = 4, VT_F32 = 6, VT_BOOL = 7, VT_STR = 8, VT_ARR = 9 };

template <typename T>
void put(std::vector<uint8_t>& b, T v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put<uint64_t>(b, (uint64_t)s.size());
    b.insert(b.end(), s.begin(), s.end());
}

struct Meta {
    std::string key;
    int type;
    uint32_t u = 0;
    float f = 0.f;
    std::string s;
    std::vector<uint8_t> barr;   // VT_ARR of VT_BOOL
};

bool write_tiny_gguf(const std::string& path) {
    // 36 layers: sliding, sliding, sliding, full -- repeating, exactly config.json's layer_types.
    std::vector<uint8_t> pattern;
    for (int i = 0; i < 36; i++) pattern.push_back(((i + 1) % 4) != 0 ? 1 : 0);

    std::vector<Meta> meta = {
        {"general.architecture", VT_STR, 0, 0.f, "spark2_5"},
        {"general.name", VT_STR, 0, 0.f, "Hf_Format"},
        {"spark2_5.block_count", VT_U32, 36},
        {"spark2_5.embedding_length", VT_U32, 2560},
        {"spark2_5.feed_forward_length", VT_U32, 10240},
        {"spark2_5.attention.head_count", VT_U32, 16},
        {"spark2_5.attention.head_count_kv", VT_U32, 4},
        {"spark2_5.attention.key_length", VT_U32, 256},
        {"spark2_5.attention.value_length", VT_U32, 256},
        {"spark2_5.attention.layer_norm_rms_epsilon", VT_F32, 0, 1e-6f},
        {"spark2_5.attention.sliding_window", VT_U32, 512},
        {"spark2_5.vocab_size", VT_U32, 131072},
        {"spark2_5.rope.freq_base", VT_F32, 0, 5000000.f},
        {"spark2_5.rope.freq_base_swa", VT_F32, 0, 10000.f},
        {"spark2_5.rope.dimension_count", VT_U32, 64},
        {"spark2_5.rope.dimension_count_swa", VT_U32, 256},
        {"tokenizer.ggml.eos_token_id", VT_U32, 1},
        {"tokenizer.ggml.bos_token_id", VT_U32, 0},
        {"spark2_5.attention.sliding_window_pattern", VT_ARR, 0, 0.f, "", pattern},
    };

    std::vector<uint8_t> b;
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    put<uint32_t>(b, 3);
    put<uint64_t>(b, 0);            // n_tensors -- metadata-only is enough here
    put<uint64_t>(b, meta.size());
    for (const Meta& m : meta) {
        put_str(b, m.key);
        put<uint32_t>(b, (uint32_t)m.type);
        if (m.type == VT_STR) put_str(b, m.s);
        else if (m.type == VT_F32) put<float>(b, m.f);
        else if (m.type == VT_ARR) {
            put<uint32_t>(b, (uint32_t)VT_BOOL);
            put<uint64_t>(b, (uint64_t)m.barr.size());
            for (uint8_t v : m.barr) put<uint8_t>(b, v);
        } else put<uint32_t>(b, m.u);
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(b.data()), (std::streamsize)b.size());
    return out.good();
}

#define CHECK(x) do { if (!(x)) { std::printf("FAIL: %s line %d\n", #x, __LINE__); return 1; } } while (0)
} // namespace

int main() {
    const std::string path = "/tmp/sparkinfer_spark25_config_cpu_test.gguf";
    CHECK(write_tiny_gguf(path));

    sparkinfer::GGUF g;
    CHECK(g.open(path));
    CHECK(g.meta_str("general.architecture") == "spark2_5");

    sparkinfer::Qwen35Config cfg;
    qwen3_config_from_gguf(g, cfg);          // must dispatch to spark25_config_from_gguf

    CHECK(cfg.spark25 == true);
    CHECK(cfg.muse_glimmer == false);
    CHECK(cfg.n_layers == 36);
    CHECK(cfg.hidden == 2560);
    CHECK(cfg.n_q_heads == 16);
    CHECK(cfg.n_kv_heads == 4);
    CHECK(cfg.head_dim == 256);
    CHECK(cfg.moe_ffn == 10240);
    CHECK(cfg.vocab == 131072);
    CHECK(cfg.eos_id == 1);
    CHECK(cfg.rms_eps > 9e-7f && cfg.rms_eps < 1.1e-6f);

    // Dense GeGLU FFN with the head-wise gate and no QK-norm.
    CHECK(cfg.dense_ffn == true);
    CHECK(cfg.n_experts == 1 && cfg.top_k == 1 && cfg.n_shared == 0);
    CHECK(cfg.ffn_gelu == true);
    CHECK(cfg.headwise_attn_gate == true);
    CHECK(cfg.no_qk_norm == true);

    // hybrid unlocks the gate path; interval 0 keeps every layer off the linear-attention path.
    CHECK(cfg.hybrid == true);
    CHECK(cfg.full_attn_interval == 0);

    // Per-layer-kind rotary. The unsuffixed pair is the FULL-attention layers'.
    CHECK(cfg.rope_theta == 5000000.f);
    CHECK(cfg.rope_dim == 64);
    CHECK(cfg.rope_theta_swa == 10000.f);
    CHECK(cfg.rope_dim_swa == 256);
    CHECK(cfg.sliding_window == 512);

    // The bool array must have survived as 36 entries in [sliding,sliding,sliding,full] order.
    CHECK(cfg.swa_layers.size() == 36);
    for (int i = 0; i < 36; i++) CHECK(cfg.swa_layers[(size_t)i] == (((i + 1) % 4) != 0));

    // rope_*_for(i) is what the decode/prefill dispatch actually reads: sliding layers get the
    // _swa pair, full-attention layers the unsuffixed one. Layer 3 is the first full layer.
    CHECK(cfg.rope_theta_for(0) == 10000.f && cfg.rope_dim_for(0) == 256);
    CHECK(cfg.rope_theta_for(2) == 10000.f && cfg.rope_dim_for(2) == 256);
    CHECK(cfg.rope_theta_for(3) == 5000000.f && cfg.rope_dim_for(3) == 64);
    CHECK(cfg.rope_theta_for(35) == 5000000.f && cfg.rope_dim_for(35) == 64);

    // A config with no per-kind split (every other model here) must be unaffected: both
    // accessors fall back to the unsuffixed pair for every layer.
    sparkinfer::Qwen35Config plain;
    plain.rope_theta = 1000000.f;
    plain.rope_dim = 0;
    plain.swa_layers.assign(4, true);        // sliding flags set, but no _swa rotary given
    for (int i = 0; i < 4; i++) {
        CHECK(plain.rope_theta_for(i) == 1000000.f);
        CHECK(plain.rope_dim_for(i) == 0);
    }

    CHECK(qwen3_model_label(cfg) == std::string("Spark-X2.5 dense SWA hybrid"));

    std::printf("spark25_config_cpu_test: OK\n");
    return 0;
}
