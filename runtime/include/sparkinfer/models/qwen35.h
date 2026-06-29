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

    // Allocate the fixed-size scratch (and CUDA graph) used by
    // forward_tokens_batch() / generate_speculative(). Must be called before
    // either; `k` is the verification batch width and cannot change afterward
    // (the graph is captured once for this shape, like forward_token's).
    void enable_speculative(int k);

    // Run `n` tokens (n == the `k` passed to enable_speculative) starting at
    // `start_position` in one batched forward pass -- the speculative-decoding
    // verification step. `tokens[i]` is fed at position start_position+i;
    // out_argmax[i] receives the model's greedy next-token prediction after
    // having seen tokens[0..i]. KV for all n positions is appended whether or
    // not the caller ends up accepting them; positions beyond the last
    // accepted token are simply overwritten by the next call, so no rollback
    // is needed. Returns false if enable_speculative() was not called first.
    bool forward_tokens_batch(const int* tokens, int n, int start_position, int* out_argmax);

    // Prompt-lookup speculative greedy generate: drafts up to (draft_k - 1)
    // tokens per round from repeats in the token history (no second model),
    // verifies them with one forward_tokens_batch() call, and accepts the
    // longest matching prefix -- producing the identical token sequence
    // generate() would, in fewer forward passes when the draft hits. Calls
    // enable_speculative(draft_k) internally on first use.
    std::vector<int> generate_speculative(const std::vector<int>& prompt_ids, int max_new_tokens, int draft_k = 4);

    // Copy the most recent step's logits (vocab floats) to host. Valid after a
    // forward_token() call. Used for teacher-forced scoring (perplexity / KL).
    void copy_logits(float* host_logits) const;

    // Steady-state decode throughput benchmark: runs `warmup` untimed decode
    // steps then times `n_tokens` more. Returns tokens/sec. Requires weights.
    double bench_decode(int warmup, int n_tokens);

    // Same timing protocol as bench_decode(), but times prompt-lookup speculative
    // decode (generate_speculative's inner loop) instead of forward_token().
    double bench_decode_speculative(int warmup, int n_tokens, int draft_k = 4);

    const Qwen35Config& config() const;

private:
    struct Impl;
    Impl* p_;
};

} // namespace sparkinfer
