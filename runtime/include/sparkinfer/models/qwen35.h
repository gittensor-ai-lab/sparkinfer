#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen_config.h"
#include "sparkinfer/moe/engine.h"

namespace sparkinfer {

class ThermalGovernor;   // optional decode-time thermal pacing (thermal_governor.h)
class BridgeClient;      // optional external KV cache tier (lmcache_bridge_client.h)

// Device (bf16) weight pointers for one layer.
struct Qwen35LayerWeights {
    bool linear_attn = false;
    bool q_has_gate = false;
    // Muse Glimmer: true = sliding-window attention (RoPE, windowed KV), false = global
    // attention (NoPE -- no RoPE at all, full KV). Mirrors Qwen35Config::swa_layers[i];
    // copied per-layer at weight-load time so the decode loop doesn't need the config.
    bool swa = false;
    const void* input_norm   = nullptr;  // [hidden]
    const void* wq = nullptr;            // [hidden, n_q_heads*head_dim]
    // Muse Glimmer only: attn_gate ships as its OWN [hidden, n_q_heads*head_dim] tensor rather
    // than pre-fused into wq the way Qwen3.6's GGUF writes it, so it is kept quantized and
    // projected separately instead of being dequantized and interleaved into wq at load.
    const void* wgate = nullptr;         // [hidden, n_q_heads*head_dim]
    const void* wk = nullptr;            // [hidden, n_kv_heads*head_dim]
    const void* wv = nullptr;            // [hidden, n_kv_heads*head_dim]
    const void* wo = nullptr;            // [n_q_heads*head_dim, hidden]
    const void* q_norm = nullptr;        // [head_dim]
    const void* k_norm = nullptr;        // [head_dim]
    const void* post_attn_norm = nullptr;// [hidden]
    // Muse Glimmer sandwich norm -- FOUR distinct norm weights per layer, vs the two
    // (input_norm, post_attn_norm double-duty as the FFN's pre-norm) every other
    // architecture here uses:
    //   input_norm       existing field -- pre-attention norm ("attn_norm" tensor)
    //   post_attn_norm   existing field, repurposed -- sandwich norm on the attention
    //                    output ("post_attention_norm" tensor), added to the residual via
    //                    launch_norm_then_add rather than being norm'd itself like every
    //                    other model's post_attn_norm usage
    //   ffn_norm         new -- genuine pre-FFN norm ("ffn_norm" tensor). Other
    //                    architectures reuse post_attn_norm for this (one norm serves as
    //                    both "post-attention" and "pre-FFN" since there's no sandwich
    //                    step between them); Muse Glimmer ships both as distinct tensors.
    //   post_ffn_norm    new -- sandwich norm on the FFN output ("post_ffw_norm" tensor),
    //                    added to the residual the same way post_attn_norm now is.
    const void* ffn_norm = nullptr;      // [hidden]
    const void* post_ffn_norm = nullptr; // [hidden]
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
    // Optional native gate/up copies retained for batched prefill when decode uses an
    // internal compact format. Decode continues to read gate_q/up_q.
    const void* prefill_gate_q = nullptr; const void* prefill_up_q = nullptr;
    int prefill_gate_qtype = 0, prefill_up_qtype = 0;
    // Optional SM120-native NVFP4 copies used only by Muse batched prefill. Decode and all
    // unsupported shapes continue to use the GGUF-native pointers above.
    const void* gate_fp4 = nullptr; const void* gate_fp4_sf = nullptr;
    const void* up_fp4 = nullptr;   const void* up_fp4_sf = nullptr;
    const void* down_fp4 = nullptr; const void* down_fp4_sf = nullptr;
    const void* wo_fp4 = nullptr;   const void* wo_fp4_sf = nullptr;
    // attention projections: 0 = bf16 dense (default); else ggml type id (12=Q4_K,
    // 14=Q6_K) -> weights kept quantized in VRAM, decoded on-read by launch_gemv_q.
    int wq_type = 0, wgate_type = 0, wk_type = 0, wv_type = 0, wo_type = 0;
    int wqkv_type = 0, wqkv_gate_type = 0, ssm_beta_type = 0, ssm_alpha_type = 0, ssm_out_type = 0;
    int shared_gate_inp_type = 0;

    // Muse Glimmer fused prefill: per-output-row int8 scales (amax/127) of the native Q4_K/Q5_K
    // attn + FFN gate/up weights, precomputed once at load. Let the batched prefill GEMM decode the
    // weight to int8 in-register (launch_prefill_gemm_qi8_dense) instead of materializing the whole
    // int8 weight first. Null => that weight stays on the materialize path (e.g. Q6_K down, or when
    // the precompute is disabled/unavailable). Owned by the model Impl, not freed per-layer.
    const float* wq_rs = nullptr; const float* wgate_rs = nullptr;
    const float* wk_rs = nullptr; const float* wv_rs = nullptr; const float* wo_rs = nullptr;
    const float* gate_rs = nullptr; const float* up_rs = nullptr;
    // ffn_down's per-row int8 scale. Left out of the original dense set on the assumption that
    // down is always Q6_K; in the shipped Muse Q4_K_M GGUF a large share of the down tensors are
    // Q4_K, i.e. fusable, and were falling back to the int8 materialize only because the scale
    // pool had no slot for them. Null when this layer's down really is unfusable (Q6_K).
    const float* down_rs = nullptr;
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
    // When SPARKINFER_DFLASH=1 and a draft is attached via set_dflash_draft(), uses DFlash
    // block-diffusion speculative decoding (greedy-equivalent to AR when correct).
    // Optional out_ttft_s/out_decode_s split prefill from decode wall-clock, same convention as
    // DFlashStats below -- without this, tok/s computed from total wall time collapses toward the
    // prefill rate at long context (32k prefill dwarfs a 128-token decode), not the decode rate.
    std::vector<int> generate(const std::vector<int>& prompt_ids, int max_new_tokens,
                              ThermalGovernor* gov = nullptr,
                              double* out_ttft_s = nullptr, double* out_decode_s = nullptr);

    // DFlash speculative generate (greedy). Requires set_dflash_draft(). Returns generated ids.
    // Optional stats: mean acceptance length τ and wall-clock seconds for decode (post-TTFT).
    struct DFlashStats {
        double mean_accept = 0;   // mean tokens committed per draft step (τ)
        double decode_s = 0;
        double ttft_s = 0;
        int    steps = 0;
    };
    std::vector<int> dflash_generate(const std::vector<int>& prompt_ids, int max_new_tokens,
                                     DFlashStats* stats = nullptr,
                                     ThermalGovernor* gov = nullptr);

    // Prefill `tokens` and retain KV + hybrid recurrent state for reuse on the next request
    // whose prompt starts with the same token sequence. Returns false on allocation failure.
    bool cache_prefix(const std::vector<int>& tokens);

    // Drop the installed prefix cache and free its KV blocks.
    void clear_prefix_cache();

    // Soft-invalidate after session 0's KV was freed (generate / ContinuousBatchEngine).
    // Keeps prefix_tokens/len so the next cache_prefix() can re-warm; clears prefix_active
    // so callers do not skip re-warm while the KV is empty.
    void release_prefix_session();

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
    //
    // temperature <= 0 (the default) is plain greedy argmax, byte-identical to the pre-sampling
    // behavior. temperature > 0 draws via Gumbel-max (kernels::launch_temperature_sample) before
    // the argmax reduction; seed/sample_step select the draw -- sample_step must be a fresh,
    // request-scoped, 0-based decode-step counter (ContinuousBatchEngine::Job::decode_emitted is
    // the intended source) so the SAME captured decode graph can be replayed across separate
    // requests/sessions with different temperature/seed without invalidating reproducibility.
    //
    // top_k (<=0 or >=vocab disables) and top_p (<=0 or >=1.0 disables) truncate the candidate
    // set BEFORE the Gumbel draw above -- top_k first, then top_p narrows within it (see
    // kernels::launch_topk_topp_mask). Provably inert whenever temperature<=0: the greedy winner
    // (highest logit) is always in both the top_k and top_p surviving set by construction, so
    // masking never changes a greedy result -- neither param needs temperature>0 to be accepted,
    // and neither needs its own DFlash-incompatibility check (temperature>0 is already rejected
    // under DFlash independent of top_k/top_p).
    //
    // presence_penalty/frequency_penalty (OpenAI's [-2.0, 2.0] range; 0 disables both) subtract
    // count-weighted terms from the FULL vocab-sized logits row BEFORE top_k/top_p truncation
    // above -- see kernels::launch_presence_frequency_penalty. UNLIKE top_k/top_p, this has NO
    // inertness proof at temperature<=0: a nonzero penalty can change WHICH token has the highest
    // logit even under pure greedy argmax, since it isn't restricted to non-winning ranks the way
    // truncation is. DFlash (which requires exact greedy-argmax determinism against its draft
    // model) must therefore reject presence_penalty!=0 || frequency_penalty!=0 independently of
    // temperature -- see should_reject_dflash_penalty in chat_tools.hpp, mirroring
    // should_reject_dflash_temperature. Counts accumulate in the CURRENT session's per-session
    // penalty_counts buffer (see SessionBuffers, activate_session()) across every decode step of
    // the SAME request -- reset once per request at submit time
    // (Qwen35Model::reset_penalty_counts), NOT per forward_token() call.
    //
    // logit_bias (OpenAI's [-100, 100] per-token range) adds a fixed per-vocab-id bias to the same
    // FULL vocab-sized logits row, also before top_k/top_p truncation -- see
    // kernels::launch_logit_bias. Unlike every other sampling control here, it takes NO parameter
    // in this function: the bias is STATIC for the whole request (not refreshed every decode step
    // like temperature/top_k/top_p/the penalties), so it is set ONCE per request, at submit time,
    // directly into the CURRENT session's per-session logit_bias buffer (see SessionBuffers,
    // activate_session(), Qwen35Model::set_logit_bias) -- forward_token() just reads whatever is
    // currently in that buffer, every decode step, unconditionally (same graph-replay-safety
    // discipline as everything else on this path). Same "no inertness proof at temperature<=0,
    // needs its own DFlash check" story as presence/frequency penalty -- see
    // should_reject_dflash_logit_bias in chat_tools.hpp.
    int forward_token(int token_id, int position, bool sample = true, float temperature = 0.f,
                      unsigned long long seed = 0, unsigned long long sample_step = 0,
                      int top_k = 0, float top_p = 1.f,
                      float presence_penalty = 0.f, float frequency_penalty = 0.f);

    // Copy the most recent step's logits (vocab floats) to host. Valid after a
    // forward_token() call. Used for teacher-forced scoring (perplexity / KL).
    void copy_logits(float* host_logits) const;

    // Minimal, string-free (tokenization is a server-layer concern) -- the chosen token's own
    // logprob plus its top `top_alternatives.size()` alternatives by raw logit, sorted descending.
    struct TokenLogprob {
        int token_id = -1;
        float logprob = 0.f;
        std::vector<std::pair<int, float>> top_alternatives;
    };
    static constexpr int kMaxTopLogprobs = 20;   // OpenAI's own top_logprobs ceiling

    // Reads the RAW (post-softcap, pre-truncation, pre-temperature-noise) model distribution
    // behind the token the IMMEDIATELY PRECEDING forward_token() call produced: that token's own
    // logprob plus its top `top_n` (clamped to [0, kMaxTopLogprobs]) alternatives. "Raw" here means
    // this reports the model's true confidence, not an artifact of the caller's own
    // temperature/top_k/top_p choices -- matches real-world OpenAI/vLLM logprobs semantics.
    //
    // Valid under the same "forward_token() syncs its stream before returning" contract as
    // copy_logits() above -- NOT valid on the DFlash deferred-collect early return
    // (s.defer_decode_sync == true, forward_token() returns kDFlashDeferred without syncing);
    // unexercised in practice since ContinuousBatchEngine (the sole caller behind
    // /v1/chat/completions) never sets dflash_cap.
    //
    // Deliberately NOT baked into forward_token()'s own D2H tail (unlike h_out_id): this does its
    // own on-demand cudaMemcpy(s), so calling (or never calling) it costs nothing extra on the
    // decode hot path for logprobs=false requests -- the only always-on cost logprobs adds to
    // every decode step is the two small device-side kernel changes documented on
    // kernels::launch_topk_topp_mask/launch_extract_chosen_logit (both correctness-critical for
    // graph replay safety regardless of whether THIS request wants logprobs).
    TokenLogprob last_token_logprobs(int top_n = kMaxTopLogprobs) const;

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

    // Prefill prompt tokens [start, end) with the batched path when start==0 and eligible, else
    // the token loop. chunk_limit > 0 caps the token-loop path to at most chunk_limit tokens per
    // call (the batched path never chunks — it always covers the full range in one pass, so a
    // chunk_limit smaller than end-start forces the token-loop fallback); pass 0 for unlimited
    // (single call covers the whole range). out_pos, if non-null, receives the position reached
    // (== end once the whole range is consumed, < end if chunk_limit stopped it early). Returns
    // the argmax seed for decode once out_pos == end, or -1 while there is remaining work (or on
    // failure). The single funnel both cache_prefix()'s exclusive-session path and
    // ContinuousBatchEngine::step_job()'s continuous-batch path dispatch prefill through, so
    // batched-vs-token-loop routing and external KV cache lookup/store (when a bridge is
    // attached via set_lmcache_bridge()) only need to be implemented once.
    int ingest_prompt_range(const int* ids, int start, int end, int chunk_limit = 0,
                            int* out_pos = nullptr);

    // Attaches an optional external KV cache tier (docs/lmcache_bridge_protocol.md). Null (the
    // default) leaves every lookup/store call site a no-op -- existing behavior is unchanged
    // unless a caller explicitly opts in. Does not take ownership; the caller (ModelEngine) is
    // responsible for the BridgeClient's lifetime, which must outlive this model.
    void set_lmcache_bridge(BridgeClient* bridge);

    // Per-request session lifecycle for continuous batching / serving.
    // open_session() allocates right-sized KV blocks (+ hybrid recurrent state when needed).
    // activate_session() binds forward_token / prefill to that seq_id. Returns 0 on OOM.
    // alloc_failed, when non-null, is set true only when a real device allocation failed (a
    // cudaMalloc for the hybrid recurrent-state buffers) -- distinct from the KV block pool
    // simply being full, which is a normal, transient "no capacity right now" and leaves
    // *alloc_failed untouched. Callers that care about this distinction (ContinuousBatchEngine,
    // to report 503 instead of 429 -- #779) should zero-init their bool before passing it in.
    uint64_t open_session(int num_tokens, bool* alloc_failed = nullptr);
    // store_tokens, when non-null and an LMCache bridge is attached, stores this session's KV
    // for [0, store_tokens->size()) to the bridge (chunk-aligned, see lmcache_maybe_store in
    // qwen35.cpp) before freeing it -- the "session close" eviction point. Most callers don't
    // have the original prompt at this call site and pass nullptr, which is a pure no-op.
    void close_session(uint64_t seq_id, const std::vector<int>* store_tokens = nullptr);
    void activate_session(uint64_t seq_id);
    uint64_t active_session() const;

    // Zeros seq_id's running presence/frequency-penalty count buffer. MUST be called once per
    // REQUEST that will use this seq_id, even for a freshly open_session()'d id (open_session
    // already zeros at allocation time, but this call is the single, unambiguous "this request's
    // counts start at zero" point -- see ContinuousBatchEngine::submit_locked's call sites).
    // CRITICALLY needed for seq_id == 0 (the shared prefix session): unlike lin_state/
    // lin_conv_state, which are DELIBERATELY persistent/shared across the many unrelated requests
    // that reuse session 0 via use_prefix_session (that sharing IS the prefix-cache
    // optimization), OpenAI's presence/frequency-penalty semantics are scoped to "the CURRENT
    // completion's generated tokens" -- a fresh request reusing session 0 must start with
    // all-zero counts, or one client's generation would incorrectly penalize a later, unrelated
    // client's request sharing the same prefix session. No-op if seq_id has no session entry.
    void reset_penalty_counts(uint64_t seq_id);

    // Sets seq_id's per-request logit_bias buffer: zeros it, then scatters the given sparse
    // (token_id, bias) pairs into it (each bias applied via kernels::launch_logit_bias every decode
    // step of this request, see forward_token's doc comment above). MUST be called once per REQUEST
    // that will use this seq_id, same "even for a freshly open_session()'d id" and "critically
    // needed for seq_id == 0" reasoning as reset_penalty_counts -- call it right alongside that
    // function at every call site (see ContinuousBatchEngine::submit_locked). An empty `bias` still
    // zeros the buffer (clears any prior request's bias when seq_id == 0 is reused) and returns.
    // token ids are NOT re-validated against vocab here (parse_request_controls already did that,
    // with the real vocab size in scope) -- the scatter kernel keeps a defensive bound check as a
    // backstop only. No-op if seq_id has no session entry.
    void set_logit_bias(uint64_t seq_id, const std::vector<std::pair<int, float>>& bias);

    // Token budget for KV allocation: prompt + decode headroom, capped at max_seq.
    static int session_token_budget(size_t prompt_len, int max_new, int max_seq);

    // Shared weights for DFlash draft (embed + lm_head come from target).
    const void* embed_weights() const;
    const void* lm_head_weights() const;
    int lm_head_quant_type() const;

    // Attach / detach a DFlash draft model (non-owning). nullptr clears.
    void set_dflash_draft(class DFlashDraftModel* draft);

    // DFlash: capture concat hidden states at target_layer_ids per forward step.
    // Disables CUDA-graph replay while enabled (capture needs eager layer outputs).
    void set_dflash_capture(bool on, const std::vector<int>& target_layer_ids, int max_rows = 16);
    void set_dflash_capture_row(int row);
    // Append the current capture-row into the growing context buffer at global_pos.
    void dflash_stash_capture(int global_pos);
    const void* dflash_hidden_buffer() const;   // scratch rows from last verify block
    const void* dflash_context_buffer() const;  // accumulated prefill/accept hiddens
    int dflash_hidden_row_stride() const;       // bf16 elems per row = n_capture * hidden
    int dflash_context_len() const;

    // Snapshot hybrid recurrent state for speculative rollback.
    void save_spec_snapshot();
    void restore_spec_snapshot();

    // Token-loop teacher-forced block verify with optional hidden capture.
    // Writes argmax at each position into out_argmax[0..n). Returns false on failure.
    bool verify_block(const int* token_ids, int n, int start_pos, int* out_argmax);

    // Build the DFlash verify replay graph without running it. Stream capture records kernels
    // instead of executing them, so this leaves model state untouched -- it exists purely to keep
    // ~4.9 ms of graph construction out of the decode loop, where it landed on decode step 2 and
    // made that token take twice as long as every other one.
    void dflash_warm_verify(int n, int start_pos);

    // Batched verify entry (may fall back to verify_block). Same contract as verify_block.
    bool batched_forward(const int* token_ids, int n, int start_pos, bool resume_gdn,
                         int* out_argmax, const void* dflash_capture_dst = nullptr);

private:
    void invalidate_decode_graph();
    void dflash_maybe_capture_layer(int layer);
    // Depth-adaptive KV-split count for a given seqlen (32/128/160/256 tiers, GQA-8/hd256
    // occupancy correction). Shared by forward_token()'s normal per-token adaptation and
    // dflash_generate()'s one-time pre-capture initialization (see qwen35.cpp).
    int adaptive_nsplits_for(int seqlen) const;

    struct Impl;
    Impl* p_;
};

} // namespace sparkinfer
