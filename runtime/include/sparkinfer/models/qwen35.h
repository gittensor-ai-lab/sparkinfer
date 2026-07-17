#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen_config.h"
#include "sparkinfer/moe/engine.h"

namespace sparkinfer {

class ThermalGovernor;   // optional decode-time thermal pacing (thermal_governor.h)

// Device (bf16) weight pointers for one layer.
struct Qwen35LayerWeights {
    bool linear_attn = false;
    bool q_has_gate = false;
    const void* input_norm   = nullptr;  // [hidden]
    const void* wq = nullptr;            // [hidden, n_q_heads*head_dim]
    const void* wk = nullptr;            // [hidden, n_kv_heads*head_dim]
    const void* wv = nullptr;            // [hidden, n_kv_heads*head_dim]
    const void* wo = nullptr;            // [n_q_heads*head_dim, hidden]
    const void* q_norm = nullptr;        // [head_dim]
    const void* k_norm = nullptr;        // [head_dim]
    const void* post_attn_norm = nullptr;// [hidden]
    const void* router_w = nullptr;      // [hidden, n_experts]
    int router_w_type = 0;              // ggml_type when router_w is kept quantized (0 = bf16 dense)
    const void* gate = nullptr;          // [n_experts, hidden, moe_ffn]
    const void* up   = nullptr;          // [n_experts, hidden, moe_ffn]
    const void* down = nullptr;          // [n_experts, moe_ffn, hidden]
    const void* shared_gate = nullptr;   // bf16 dense fallback
    const void* shared_up   = nullptr;
    const void* shared_down = nullptr;
    const void* shared_gate_inp = nullptr;// [hidden] -> scalar shared-expert gate
    // Qwen3.6 UD: shared expert stored as Q8_0 (on-read GEMV, ~2x less weight BW vs bf16).
    const void* shared_gate_q = nullptr; const void* shared_up_q = nullptr; const void* shared_down_q = nullptr;
    int shared_gate_qtype = 0, shared_up_qtype = 0, shared_down_qtype = 0;

    // Qwen3.5/Qwen3.6 Gated DeltaNet tensors (linear-attention layers only).
    const void* wqkv = nullptr;           // [hidden, q+k+v]
    const void* wqkv_gate = nullptr;      // [hidden, value_dim]
    const void* ssm_conv = nullptr;       // [conv_kernel, q+k+v]
    const void* ssm_dt = nullptr;         // [value_heads]
    const void* ssm_a = nullptr;          // [value_heads]
    const void* ssm_beta = nullptr;       // [hidden, value_heads]
    const void* ssm_alpha = nullptr;      // [hidden, value_heads]
    const void* ssm_norm = nullptr;       // [linear_head_dim]
    const void* ssm_out = nullptr;        // [value_dim, hidden]

    // GGUF path: experts kept quantized in VRAM (gguf-native [E,out,in] layout).
    // When gate_q != nullptr the model dequantizes these per-layer into scratch
    // instead of using the bf16 gate/up/down above. *_qtype are ggml type ids.
    const void* gate_q = nullptr; const void* up_q = nullptr; const void* down_q = nullptr;
    int gate_qtype = 0, up_qtype = 0, down_qtype = 0;
    // attention projections: 0 = bf16 dense (default); else ggml type id (12=Q4_K,
    // 14=Q6_K) -> weights kept quantized in VRAM, decoded on-read by launch_gemv_q.
    int wq_type = 0, wk_type = 0, wv_type = 0, wo_type = 0;
    int wqkv_type = 0, wqkv_gate_type = 0, ssm_beta_type = 0, ssm_alpha_type = 0, ssm_out_type = 0;
    int shared_gate_inp_type = 0;
};

struct Qwen35Weights {
    const void* embed_tokens = nullptr;  // [vocab, hidden]
    const void* final_norm   = nullptr;  // [hidden]
    const void* lm_head      = nullptr;  // [hidden, vocab]  (pre-transposed)
    int lm_head_type = 0;                 // 0 = bf16; else ggml type -> on-read quantized GEMV
    std::vector<Qwen35LayerWeights> layers;
};

// Single-sequence (batch=1) greedy decoder for Qwen MoE. Owns scratch buffers and
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
    // When a prefix cache is installed (cache_prefix), skips re-prefilling the matching prefix.
    std::vector<int> generate(const std::vector<int>& prompt_ids, int max_new_tokens,
                              ThermalGovernor* gov = nullptr);

    // Prefill `tokens` and retain KV + hybrid recurrent state for reuse on the next request
    // whose prompt starts with the same token sequence. Returns false on allocation failure.
    bool cache_prefix(const std::vector<int>& tokens);

    // Drop the installed prefix cache and free its KV blocks.
    void clear_prefix_cache();

    // Length of the currently cached prefix (0 if none).
    int prefix_cached_len() const;

    // Argmax seed after cache_prefix(); -1 if no active prefix cache.
    int prefix_seed_token() const;

    // True when `prompt` begins with the installed prefix token sequence (requires active cache).
    bool prompt_matches_prefix(const std::vector<int>& prompt) const;

    // Time-to-first-token: ingest `prompt` with prefill (no LM head on interior tokens when
    // not legacy), then one sampled forward. Reuses cache_prefix when the prompt starts with
    // the cached tokens (only the suffix is prefilled). Returns seconds.
    double bench_ttft(const std::vector<int>& prompt);

    // Run one token at `position`. When sample=false (prefill), runs embed→layers→final
    // norm without LM head/argmax and without CUDA-graph capture — teacher-forced ingestion.
    // When sample=true (decode / last prompt token), runs the full path and may capture/replay
    // the decode graph. Returns argmax next-token id when sample=true, else token_id.
    int forward_token(int token_id, int position, bool sample = true);

    // Copy the most recent step's logits (vocab floats) to host. Valid after a
    // forward_token() call. Used for teacher-forced scoring (perplexity / KL).
    void copy_logits(float* host_logits) const;

    struct BenchDecodeResult {
        double decode_tps = 0;
        double prefill_pp = 0;
    };
    // Benchmark at a target KV depth: timed prefill, untimed warmup decode, timed decode.
    BenchDecodeResult bench_decode(int warmup, int n_tokens, int context_tokens = 0);

    const Qwen35Config& config() const;

    // Batched prompt prefill (Qwen3.5 dense-hybrid only): process all `n` prompt tokens in one
    // pass, filling the paged KV cache and Gated-DeltaNet recurrent/conv state for positions
    // 0..n-1 so a subsequent decode is faithful to the forward_token loop. Returns the argmax at
    // the last prompt position (seed for the first decode step), or -1 if the batched path is
    // unsupported for this model/config. Implemented in qwen35_prefill.cpp.
    int prefill_batched(const int* prompt_ids, int n);

    // Per-request session lifecycle for continuous batching / serving.
    // open_session() allocates right-sized KV blocks (+ hybrid recurrent state when needed).
    // activate_session() binds forward_token / prefill to that seq_id. Returns 0 on OOM.
    uint64_t open_session(int num_tokens);
    void close_session(uint64_t seq_id);
    void activate_session(uint64_t seq_id);
    uint64_t active_session() const;

    // Token budget for KV allocation: prompt + decode headroom, capped at max_seq.
    static int session_token_budget(size_t prompt_len, int max_new, int max_seq);

private:
    void invalidate_decode_graph();
    // Prefill prompt tokens [start, end) with batched path when start==0, else token loop.
    // Returns argmax seed at end-1 for decode, or -1 on failure.
    int ingest_prompt_range(const int* ids, int start, int end);

    struct Impl;
    Impl* p_;
};

} // namespace sparkinfer
