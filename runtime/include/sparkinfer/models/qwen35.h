#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/moe/engine.h"

namespace sparkinfer {

class ThermalGovernor;   // optional decode-time thermal pacing (thermal_governor.h)

// Qwen3.5-35B-A3B architecture.
//   40 layers, hidden 2048, 16 Q / 2 KV heads (8:1 GQA), head_dim 128,
//   256 routed experts (top-8) + 1 shared expert, moe ffn 512,
//   RoPE + per-head QK-norm, RMSNorm, SwiGLU.
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
};

// Device (bf16) weight pointers for one layer.
struct Qwen35LayerWeights {
    const void* input_norm   = nullptr;  // [hidden]
    const void* wq = nullptr;            // [hidden, n_q_heads*head_dim]
    const void* wk = nullptr;            // [hidden, n_kv_heads*head_dim]
    const void* wv = nullptr;            // [hidden, n_kv_heads*head_dim]
    const void* wo = nullptr;            // [n_q_heads*head_dim, hidden]
    const void* q_norm = nullptr;        // [head_dim]
    const void* k_norm = nullptr;        // [head_dim]
    const void* post_attn_norm = nullptr;// [hidden]
    const void* router_w = nullptr;      // [hidden, n_experts]
    const void* gate = nullptr;          // [n_experts, hidden, moe_ffn]
    const void* up   = nullptr;          // [n_experts, hidden, moe_ffn]
    const void* down = nullptr;          // [n_experts, moe_ffn, hidden]
    const void* shared_gate = nullptr;   // [hidden, moe_ffn]
    const void* shared_up   = nullptr;   // [hidden, moe_ffn]
    const void* shared_down = nullptr;   // [moe_ffn, hidden]

    // GGUF path: experts kept quantized in VRAM (gguf-native [E,out,in] layout).
    // When gate_q != nullptr the model dequantizes these per-layer into scratch
    // instead of using the bf16 gate/up/down above. *_qtype are ggml type ids.
    const void* gate_q = nullptr; const void* up_q = nullptr; const void* down_q = nullptr;
    int gate_qtype = 0, up_qtype = 0, down_qtype = 0;
    // attention projections: 0 = bf16 dense (default); else ggml type id (12=Q4_K,
    // 14=Q6_K) -> weights kept quantized in VRAM, decoded on-read by launch_gemv_q.
    int wq_type = 0, wk_type = 0, wv_type = 0, wo_type = 0;
};

struct Qwen35Weights {
    const void* embed_tokens = nullptr;  // [vocab, hidden]
    const void* final_norm   = nullptr;  // [hidden]
    const void* lm_head      = nullptr;  // [hidden, vocab]  (pre-transposed)
    int lm_head_type = 0;                 // 0 = bf16; else ggml type -> on-read quantized GEMV
    std::vector<Qwen35LayerWeights> layers;
};

// Single-sequence (batch=1) greedy decoder for Qwen3.5. Owns scratch buffers and
// drives embed -> N layers -> final norm -> LM head -> argmax per token.
class Qwen35Model {
public:
    Qwen35Model(const Qwen35Config& cfg, KVCacheManager* kv, moe::MoEEngine* engine);
    ~Qwen35Model();

    void set_weights(const Qwen35Weights& w);

    // Load weights from a sparkinfer weight directory (see tools/convert_qwen35.py).
    // Returns false on failure. Allocates device buffers it owns.
    bool load_weights(const std::string& dir);

    // Load weights directly from a GGUF file (native). Dense tensors are
    // dequantized to bf16; expert tensors are kept quantized in VRAM and
    // dequantized per-layer at decode time (Q4_K_M-sized resident footprint).
    bool load_gguf(const std::string& path);

    // Greedy generate: prompt token ids -> generated token ids (host). An optional ThermalGovernor
    // paces decode under thermal pressure (accuracy-preserving); nullptr = full speed, no overhead.
    std::vector<int> generate(const std::vector<int>& prompt_ids, int max_new_tokens,
                              ThermalGovernor* gov = nullptr);

    // Run one token at `position`, return the argmax next-token id.
    int forward_token(int token_id, int position);

    // Copy the most recent step's logits (vocab floats) to host. Valid after a
    // forward_token() call. Used for teacher-forced scoring (perplexity / KL).
    void copy_logits(float* host_logits) const;

    // Steady-state decode throughput benchmark at a target KV depth: runs untimed
    // prefill to `context_tokens`, then `warmup` untimed decode steps, then times
    // `n_tokens` more. Returns tokens/sec. Requires weights.
    double bench_decode(int warmup, int n_tokens, int context_tokens = 0);

    const Qwen35Config& config() const;

private:
    struct Impl;
    Impl* p_;
};

} // namespace sparkinfer
