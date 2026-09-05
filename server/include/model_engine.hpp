#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace sparkinfer_server {

// Hand-mirrored server-layer copy of sparkinfer::Qwen35Model::TokenLogprob -- model_engine.hpp
// deliberately never re-exposes runtime types in its public API, matching CompletionResult's
// existing field-by-field-copy convention below.
struct TokenLogprob {
    int token_id = -1;
    float logprob = 0.f;
    std::vector<std::pair<int, float>> top_alternatives;
};

// Outcome of one complete()/complete_streaming() call. Returned by value so concurrent
// HTTP worker threads cannot observe or clear each other's failure state.
struct CompletionResult {
    std::vector<int> tokens;
    std::string error;  // empty on success
    bool overloaded = false;  // true => caller should return 429, not a generic 4xx
    bool alloc_failed = false;  // true => real device OOM, not capacity -- caller should return
                                 // 503 (permanent until restart), never 429 (implies transient/retry)
    bool timed_out = false;   // true => a per-request deadline was exceeded
    bool cancelled = false;   // true => on_token returned false; not an error
    bool reached_token_limit = false;  // true => max_new_tokens, not EOS, ended generation
    double ttft_ms = -1.0;
    double generation_ms = -1.0;
    double decode_tps = -1.0;
};

// Images already decoded and preprocessed, ready for the vision tower. Mirrored field-by-field
// into ContinuousBatchEngine::Request, keeping this header's "never re-expose runtime types"
// convention -- same reason TokenLogprob above is a hand-copied struct.
//
// Pixels, not embeddings: the tower has to run on the batch engine's worker thread (CUDA graph
// capture), so all this carries is the CPU-side preprocessing result.
struct PreparedImages {
    struct Image {
        // Shared so a copy of PreparedImages is cheap. Branch and retry paths each need their
        // OWN positions (see reexpand_images) while sharing one preprocessing result, and they
        // run concurrently, so mutating a single shared copy is not an option.
        std::shared_ptr<const std::vector<float>> pixels;
        int grid_h = 0;
        int grid_w = 0;
    };
    // One decoded clip. Each temporal group is an ordinary Image -- the tower call for a video
    // frame-pair is byte-for-byte the call for a still, which is why video needed no tower work.
    struct Video {
        std::vector<Image> groups;
        // Tokenized "<N.N seconds>" marker per group, tokenized by the server (the runtime has no
        // tokenizer). Kept here so a re-render can redo the expansion without re-decoding.
        std::vector<std::vector<int>> timestamp_tokens;
        int tokens_per_frame = 0;
    };

    // Inputs, in message order, kept so reexpand() can rebuild a prompt from scratch after a
    // tool-call retry re-renders it.
    std::vector<Image> src_images;
    std::vector<Video> src_videos;

    // FLATTENED tower units in PROMPT order, one per entry in `positions`. Images and video
    // groups interleave here exactly as their placeholders do in the prompt -- the engine pairs
    // images[i] with positions[i] positionally, so any other order silently splices each image's
    // embedding at another image's location.
    std::vector<Image> images;
    std::vector<int> positions;      // absolute prompt indices of the expanded placeholder run

    // Interleaved-MRoPE rotary positions for the whole prompt: [n_tokens*3], [t,h,w] per token.
    // Empty when the loaded checkpoint declares no mrope_section.
    std::vector<int> mrope_pos;
    // (rotary position after the prompt) - prompt length. Zero or negative, because a vision span
    // advances the rotary counter by max(h,w)/merge rather than by its token count. Every decode
    // step of the request adds this to its rotary position while its cache slot stays sequential.
    int mrope_decode_offset = 0;
};

// Thread-safe wrapper around sparkinfer::Qwen35Model + GGUF load.
class ModelEngine {
public:
    ModelEngine();
    ~ModelEngine();

    ModelEngine(const ModelEngine&) = delete;
    ModelEngine& operator=(const ModelEngine&) = delete;

    bool load(const std::string& gguf_path, int max_seq = 0);
    bool loaded() const;

    std::string model_path() const;
    int eos_id() const;
    // False once an unrecoverable CUDA error has killed the context. Permanent until restart --
    // /health reports it so an orchestrator replaces the process instead of routing to a server
    // that can only 503.
    bool device_healthy() const;
    // True for any token the runtime treats as a stop token -- eos_id AND the optional second
    // stop id (cfg.eos_id2, e.g. Muse Glimmer's <|eot|>). eos_id() alone is not sufficient:
    // step_job() stops on either, so either can be the final emitted token.
    bool is_stop_token(int token_id) const;
    int vocab() const;
    int max_seq() const;
    bool is_museglimmer() const;
    bool is_qwen38() const;
    // True when the loaded checkpoint shipped a vision tower and it loaded successfully. False
    // for every text-only model, and for a vision checkpoint whose tower failed to load -- in
    // both cases a request carrying images is refused rather than answered from its text.
    bool has_vision() const;

    // Decodes image_url values, preprocesses each to the tower's patch tensor, expands the
    // prompt's single <|image_pad|> per image to the token count that image's grid needs, and
    // reports where those tokens landed. prompt_ids is rewritten in place.
    //
    // The expansion belongs here rather than in the chat template because only the processor
    // knows the resized grid -- the template emits exactly one placeholder per image and the
    // reference processor expands it the same way.
    bool prepare_images(const std::vector<std::string>& urls, int image_token_id,
                        std::vector<int>& prompt_ids, PreparedImages& out,
                        std::string& err) const;

    // Sampling parameters for video decode. Frames become prompt tokens, so these are a context
    // budget as much as a memory one; 0 means "use the built-in default".
    struct VideoSampling { int max_frames = 0; double fps = 0.0; };

    // Same, for images and videos together. Videos decode through ffmpeg, sample to frames, and
    // preprocess into temporal groups; each group becomes an ordinary tower unit.
    //
    // tokenize renders the "<N.N seconds>" marker that precedes each temporal group's span. It is
    // a callback rather than a return-and-call-again because the timestamps depend on the sampled
    // frame indices, which only exist after decoding -- and decoding twice would mean two ffmpeg
    // subprocesses per request. The engine has no tokenizer of its own, hence the injection.
    //
    // Handles the interleaved case: a message carrying an image and a video in either order comes
    // back with images[] in PROMPT order, not images-then-videos, because the engine pairs
    // images[i] with positions[i] positionally.
    bool prepare_vision(const std::vector<std::string>& image_urls,
                        const std::vector<std::string>& video_urls,
                        int image_token_id, int video_token_id,
                        int vision_start_token_id, int vision_end_token_id,
                        const VideoSampling& sampling,
                        const std::function<std::vector<int>(const std::string&)>& tokenize,
                        std::vector<int>& prompt_ids, PreparedImages& out, std::string& err) const;

    // Re-expands placeholders and recomputes positions against a prompt that was rebuilt after
    // prepare_images already ran. The tool-call continuation path re-renders the entire prompt
    // from the message list, so both the earlier expansion and the offsets it recorded are stale
    // -- it comes back with one bare <|image_pad|> per image again. Reuses the preprocessing;
    // only the token bookkeeping is redone.
    //
    // Takes its own PreparedImages copy per branch by design: concurrent branches must not share
    // one positions vector.
    bool reexpand_images(int image_token_id, std::vector<int>& prompt_ids, PreparedImages& io,
                         std::string& err) const;

    // The general form: re-expands image AND video placeholders and rebuilds images[]/positions[]
    // in prompt order. reexpand_images() is this with the vision token ids taken from the loaded
    // checkpoint's vision_config.
    bool reexpand_vision(int image_token_id, int video_token_id,
                         int vision_start_token_id, int vision_end_token_id,
                         std::vector<int>& prompt_ids, PreparedImages& io, std::string& err) const;

    // Optional shared prompt prefix (e.g. system message tokens). When set, each request whose
    // prompt starts with these ids calls cache_prefix() (batched prefill) before generate().
    void set_prefix_tokens(const std::vector<int>& tokens);
    int prefix_token_len() const;

    // Greedy decode. Tokens are in .tokens; on failure .tokens is empty and .error is set.
    CompletionResult complete(const std::vector<int>& prompt_ids, int max_new_tokens);

    // Same, but invokes on_token after each generated token (for SSE streaming). on_token
    // returns false to cancel generation early (client disconnected); the result then comes
    // back with .cancelled = true, not an error.
    //
    // temperature <= 0 (default) is plain greedy argmax. > 0 samples via Gumbel-max -- see
    // ContinuousBatchEngine::Request's doc comment for the reproducibility contract and the
    // known "first token is always greedy" v1 scope limitation. top_k/top_p truncate the
    // candidate set before the Gumbel draw; neither requires temperature > 0 (see
    // ContinuousBatchEngine::Request's doc comment for the inertness proof).
    //
    // presence_penalty/frequency_penalty ([-2.0, 2.0], 0 disables both) -- see
    // ContinuousBatchEngine::Request's doc comment for the semantics and the "no inertness proof,
    // needs its own DFlash check" distinction from top_k/top_p.
    //
    // logit_bias: (token_id, bias in [-100,100]) pairs, empty disables it. Same tier as
    // presence_penalty/frequency_penalty (sampling control, not reporting) -- see
    // ContinuousBatchEngine::Request's doc comment. Unlike every other sampling control here, its
    // value is static for the whole request rather than refreshed per decode step.
    //
    // on_token_logprob (optional) fires once per token, immediately before on_token for that same
    // token, only when logprobs is true AND this callback is non-null -- pass nullptr (not a
    // no-op lambda) when logprobs aren't wanted; see ContinuousBatchEngine::complete_streaming's
    // doc comment for why an always-non-null callback would defeat the "costs nothing extra when
    // unused" property. Fires for EVERY emitted token including the first -- the old "first token
    // has no logprobs" gap is fixed in ContinuousBatchEngine::step_job()'s PREFILL branch.
    CompletionResult complete_streaming(const std::vector<int>& prompt_ids, int max_new_tokens,
                                        const std::function<bool(int)>& on_token,
                                        float temperature = 0.f, uint64_t seed = 0,
                                        int top_k = 0, float top_p = 1.0f,
                                        float presence_penalty = 0.f, float frequency_penalty = 0.f,
                                        const std::vector<std::pair<int, float>>& logit_bias = {},
                                        bool logprobs = false, int top_logprobs = 0,
                                        const std::function<void(const TokenLogprob&)>&
                                            on_token_logprob = nullptr,
                                        const std::vector<int>& forced_tokens = {},
                                        const PreparedImages* images = nullptr);

    // TEACHER-FORCED SCORING (POST /v1/score): non-empty `forced_tokens` turns the call into a
    // scoring pass instead of a generation. max_new_tokens must equal forced_tokens.size(); the
    // emitted tokens ARE forced_tokens; each on_token_logprob entry carries that token's logprob
    // under the distribution at its own position rather than the sampler's pick. Numerically
    // identical to what /v1/chat/completions reports for the same token at the same position --
    // same logits, same fp32 logsumexp (Qwen35Model::token_logprob_for). See
    // ContinuousBatchEngine::Request::forced_tokens for the rest of the semantics.


    // Live occupancy, for capacity-reporting endpoints. 0/0 if the model isn't loaded yet.
    int active_requests() const;
    int free_kv_blocks() const;
    int max_queue_depth() const;

    // LMCache bridge counters (docs/lmcache_bridge_protocol.md), for GET /metrics'
    // sparkinfer_lmcache_lookup_{hits,misses}_total. enabled=false (both counts 0) when
    // SPARKINFER_LMCACHE_ENABLE wasn't set or the sidecar never came up -- distinct from
    // "enabled but 0 lookups happened yet," which also reports zero counts but enabled=true.
    struct LMCacheStats {
        bool enabled = false;
        uint64_t lookup_hits = 0;
        uint64_t lookup_misses = 0;
    };
    LMCacheStats lmcache_stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    mutable std::mutex mu_;
};

}  // namespace sparkinfer_server
