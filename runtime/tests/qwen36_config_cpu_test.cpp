// CPU-only test for Qwen3.6 GGUF metadata parsing.
//
// Writes a tiny GGUF with Qwen3.6-style scalar metadata and hybrid tensor names,
// then verifies qwen3_config_from_gguf() derives the config needed before
// Qwen35Model construction sizes CUDA scratch buffers and KV cache.

#include "../examples/qwen3_gguf_config.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
enum { VT_U32 = 4, VT_F32 = 6, VT_STR = 8 };

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
};

struct Tensor {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0; // F32
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

uint64_t tensor_bytes(const Tensor& t) {
    uint64_t n = 1;
    for (uint64_t d : t.dims) n *= d;
    return n * 4;
}

bool write_tiny_gguf(const std::string& path) {
    std::vector<Meta> meta = {
        {"general.name", VT_STR, 0, 0.f, "Qwen3.6-35B-A3B"},
        {"general.alignment", VT_U32, 32},
        {"qwen35moe.block_count", VT_U32, 40},
        {"qwen35moe.embedding_length", VT_U32, 64},
        {"qwen35moe.vocab_size", VT_U32, 1234},
        {"qwen35moe.attention.head_count", VT_U32, 4},
        {"qwen35moe.attention.head_count_kv", VT_U32, 1},
        {"qwen35moe.attention.key_length", VT_U32, 16},
        {"qwen35moe.expert_count", VT_U32, 8},
        {"qwen35moe.expert_used_count", VT_U32, 2},
        {"qwen35moe.expert_feed_forward_length", VT_U32, 32},
        {"qwen35moe.rope.freq_base", VT_F32, 0, 10000000.f},
        {"qwen35moe.attention.layer_norm_rms_epsilon", VT_F32, 0, 1e-6f},
        {"qwen35moe.ssm.state_size", VT_U32, 32},
        {"qwen35moe.ssm.group_count", VT_U32, 8},
        {"qwen35moe.ssm.conv_kernel", VT_U32, 4},
        {"tokenizer.ggml.eos_token_id", VT_U32, 248044},
    };
    std::vector<Tensor> tensors = {
        {"token_embd.weight", {64, 1234}},
        {"blk.0.attn_qkv.weight", {64, 512}},
        {"blk.0.ssm_conv1d.weight", {4, 512}},
        {"blk.0.ffn_gate_shexp.weight", {64, 32}},
        {"blk.3.attn_q.weight", {64, 128}},
    };

    uint64_t off = 0;
    for (Tensor& t : tensors) {
        t.offset = off;
        t.bytes = tensor_bytes(t);
        off += t.bytes;
    }

    std::vector<uint8_t> b;
    b.insert(b.end(), {'G', 'G', 'U', 'F'});
    put<uint32_t>(b, 3);
    put<uint64_t>(b, tensors.size());
    put<uint64_t>(b, meta.size());

    for (const Meta& m : meta) {
        put_str(b, m.key);
        put<uint32_t>(b, (uint32_t)m.type);
        if (m.type == VT_STR) put_str(b, m.s);
        else if (m.type == VT_F32) put<float>(b, m.f);
        else put<uint32_t>(b, m.u);
    }

    for (const Tensor& t : tensors) {
        put_str(b, t.name);
        put<uint32_t>(b, (uint32_t)t.dims.size());
        for (uint64_t d : t.dims) put<uint64_t>(b, d);
        put<uint32_t>(b, t.type);
        put<uint64_t>(b, t.offset);
    }

    while (b.size() % 32) b.push_back(0);
    for (const Tensor& t : tensors) b.insert(b.end(), (size_t)t.bytes, 0);

    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(b.data()), (std::streamsize)b.size());
    return out.good();
}

#define CHECK(x) do { if (!(x)) { std::printf("FAIL: %s line %d\n", #x, __LINE__); return 1; } } while (0)
} // namespace

int main() {
    const std::string path = "/tmp/sparkinfer_qwen36_config_cpu_test.gguf";
    CHECK(write_tiny_gguf(path));

    sparkinfer::GGUF g;
    CHECK(g.open(path));

    sparkinfer::Qwen35Config cfg;
    qwen3_config_from_gguf(g, cfg);

    CHECK(cfg.hybrid);
    CHECK(cfg.n_layers == 40);
    CHECK(cfg.hidden == 64);
    CHECK(cfg.vocab == 1234);
    CHECK(cfg.n_q_heads == 4);
    CHECK(cfg.n_kv_heads == 1);
    CHECK(cfg.head_dim == 16);
    CHECK(cfg.n_experts == 8);
    CHECK(cfg.top_k == 2);
    CHECK(cfg.n_shared == 1);
    CHECK(cfg.moe_ffn == 32);
    CHECK(cfg.full_attn_interval == 4);
    CHECK(cfg.linear_head_dim == 32);
    CHECK(cfg.linear_q_heads == 4);
    CHECK(cfg.linear_v_heads == 8);
    CHECK(cfg.linear_conv_kernel == 4);
    CHECK(cfg.eos_id == 248044);

    std::printf("qwen36_config_cpu_test: OK\n");
    return 0;
}
