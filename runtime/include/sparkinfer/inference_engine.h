#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/scheduler.h"

namespace sparkinfer {

// Continuous-batch serving engine: queues requests, assigns per-request seq_ids,
// right-sizes KV allocation, and interleaves decode steps via the Scheduler.
//
// DFlash/DSpark multi-token accept in step_job() is still NOT wired, but read why carefully --
// the original condition has since been met. This was deferred until the single-stream
// dflash_generate path proved SPEC_AGREE=100% and a tok/s win (bench/scripts/dflash_accuracy.sh);
// as of 2026-08-21 DSpark is byte-lossless against AR and measures 1.27x at ctx=4k, so that bar
// is cleared and the remaining blocker is the continuous-batch integration itself, not the
// speculation. Anyone picking this up owns the sampling guard: Qwen35Model::generate has no
// check against a temperature-sampling caller (see its comment), and step_job serves arbitrary
// per-request sampling params, so multi-accept here needs one.
// Held by pointer only, so a forward declaration keeps the vision tower out of every
// translation unit that merely wants to submit a request.
struct QwenVisionWeights;
struct QwenVisionConfig;

class ContinuousBatchEngine {
public:
    struct Request {
        std::vector<int> prompt;
        int max_new_tokens = 0;
        int priority = 0;
        int prefill_start = 0;          // skip tokens already in a shared prefix cache
        bool use_prefix_session = false; // bind to session 0 (cache_prefix KV)
        // <= 0 (default) is plain greedy argmax, byte-identical to pre-sampling behavior. > 0
        // samples via Gumbel-max (Qwen35Model::forward_token). Note: the prefill-phase seed
        // token (the very first token of the response) is always greedy regardless of this
        // value -- see step_job()'s PREFILL branch comment; every token from the second onward
        // respects it.
        float temperature = 0.f;
        uint64_t seed = 0;              // only meaningful when temperature > 0
        // top_k <= 0 or >= vocab disables top_k (no truncation). top_p <= 0 or >= 1.0 disables
        // top_p. Both truncate the candidate set before the Gumbel draw above; neither requires
        // temperature > 0 to be accepted -- see Qwen35Model::forward_token's doc comment for the
        // inertness proof. Same prefill-phase-seed-token caveat as temperature applies unchanged.
        int top_k = 0;
        float top_p = 1.0f;
        // [-2.0, 2.0]; 0 (default) disables both. OpenAI semantics: subtracted from every vocab
        // logit each decode step, weighted by this request's own running per-token generation
        // count (presence: binary "appeared at all"; frequency: linear in count) -- see
        // Qwen35Model::forward_token's doc comment. UNLIKE top_k/top_p above, this has no
        // inertness proof at temperature<=0 -- it can change the greedy winner -- so it needs its
        // own DFlash-incompatibility check (should_reject_dflash_penalty), independent of the
        // existing temperature>0 check.
        float presence_penalty = 0.f;
        float frequency_penalty = 0.f;
        // top_logprobs is only meaningful when logprobs is true. Delivery is via the separate
        // on_token_logprob callback (complete_streaming's new trailing param), not this struct --
        // pure data here, same as every other sampling control.
        //
        // One entry per emitted token, including the first. The prefill-phase seed token (this
        // response's very first token) is produced by ingest_prompt_range() rather than
        // forward_token(), so it used to get no last_token_logprobs() call at all and callers
        // saw one FEWER entry than emitted tokens; step_job()'s PREFILL branch now asks prefill
        // to leave the sampler distribution populated for the seed and stages its logprob the
        // same way decode stages every other token's.
        //
        // The temperature gap below is NOT fixed by that and still stands: the seed is still
        // always the greedy argmax. The logprob reported for it is the true logprob of the token
        // that was actually emitted either way, so the two are independent.
        bool logprobs = false;
        int top_logprobs = 0;   // 0-20; only meaningful when logprobs is true
        // OpenAI's logit_bias: (token_id, bias in [-100,100]) pairs, added to every vocab logit
        // each decode step -- see Qwen35Model::forward_token's doc comment. Empty (default)
        // disables it. UNLIKE every other sampling control above, this is NOT refreshed every
        // forward_token() call -- it's static for the whole request, set once at submit time
        // (Qwen35Model::set_logit_bias, called from submit_locked() alongside
        // reset_penalty_counts). Same "no inertness proof at temperature<=0, needs its own DFlash
        // check" story as presence/frequency penalty (should_reject_dflash_logit_bias). First
        // non-scalar sampling-control field here -- deep-copied into Job by submit_locked's
        // `job.req = req;`, same safe-across-the-async-worker-boundary mechanism `prompt` already
        // relies on.
        std::vector<std::pair<int, float>> logit_bias;
        // TEACHER-FORCED SCORING. Non-empty => this request does not generate: it replays exactly
        // these tokens as its output and reports what the model thought of each one. Every decode
        // step still runs a real forward pass (so KV/GDN state advances exactly as it would while
        // generating), but the token emitted is forced_tokens[i] instead of the sampler's pick,
        // and the logprob reported for it is that token's logprob under the distribution at its
        // own position -- see Qwen35Model::token_logprob_for().
        //
        // Deliberately routed through the ordinary Job/step_job machinery rather than an
        // exclusive direct-model path: scoring then inherits continuous batching, right-sized KV
        // allocation, admission control/429 and the request deadline, and shares one code path
        // (and therefore one set of numerics) with generation.
        //
        // Semantics when set:
        //   * max_new_tokens should equal forced_tokens.size(); scoring stops on that limit.
        //   * EOS does NOT stop generation -- a supplied completion may legitimately contain one,
        //     and truncating there would silently score fewer tokens than the caller asked about.
        //   * sampling controls (temperature/top_k/top_p/penalties/logit_bias) are irrelevant to
        //     the OUTPUT (the tokens are given) but are still applied to the logits, so leave
        //     them at their defaults for a faithful score.
        //   * the LAST forced token costs no forward pass: its logprob comes from the position
        //     before it, and nothing is predicted after it.
        std::vector<int> forced_tokens;
        // MULTIMODAL. One entry per image, in prompt order; vision_pos holds the absolute
        // prompt indices the resulting embedding rows overwrite (the expanded <|image_pad|> run),
        // concatenated across images. Empty for a text request, which is the no-op path.
        //
        // PIXELS, not embeddings, and deliberately so. Two reasons, and both are load-bearing:
        //
        // 1. The tower must run on the WORKER thread. qwen_vision_forward issues its work on the
        //    legacy default stream, and the decode path captures CUDA graphs; the legacy stream
        //    implicitly synchronises with capturing blocking streams, so running the tower from a
        //    submitting HTTP thread would break capture out from under an unrelated decoding job.
        //    Preprocessing (decode, resize, normalise) is pure CPU and stays with the caller.
        //
        // 2. The embeddings are staged on the SHARED model via set_pending_vision, a single slot
        //    consumed by the next prefill. Only the worker knows which job that is. Staging from
        //    the submitting thread would splice one request's image into another's prefill --
        //    the same shape as the clear_prefix_cache race in ModelEngine::complete_streaming,
        //    which corrupted the KV cache under concurrent load.
        //
        // Deep-copied into Job by submit_locked's `job.req = req;`, like prompt and logit_bias.
        struct VisionImage {
            std::vector<float> pixels;   // [grid_h*grid_w, in_ch*temporal*patch*patch], row-major
            int grid_h = 0;
            int grid_w = 0;
        };
        std::vector<VisionImage> vision_images;
        std::vector<int> vision_pos;
    };

    struct Result {
        std::vector<int> tokens;
        std::string error;
        // true => caller should surface 429 (no capacity right now), not a generic 4xx —
        // the request itself was fine, there was just nowhere to run it.
        bool overloaded = false;
        // true => a real device allocation failed (distinct from the KV pool being full, which
        // sets `overloaded` instead) -- caller should surface 503, not 429: this condition is
        // permanent until the process restarts, not a transient "try again shortly" (#779, where
        // a leak eventually exhausted VRAM and every subsequent request got a misleading 429
        // "overloaded" even though the queue was empty and the KV pool had free blocks).
        bool alloc_failed = false;
        // true => a per-request deadline (SPARKINFER_REQUEST_TIMEOUT_S) was exceeded.
        bool timed_out = false;
        // true => on_token returned false (client went away mid-stream); not an error.
        bool cancelled = false;
        // true => generation exhausted max_new_tokens instead of reaching an EOS token.
        // HTTP callers surface this as finish_reason="length"; in particular, a truncated
        // tool-call payload must never be reported as a successful "stop".
        bool reached_token_limit = false;
        // GPU-side timings (exclude SSE/on_token backpressure).
        double ttft_ms = -1.0;
        double generation_ms = -1.0;
        double decode_tps = -1.0;
    };

    ContinuousBatchEngine(Qwen35Model* model, KVCacheManager* kv,
                          int max_tokens_per_batch = 64,
                          SchedulePolicy policy = SchedulePolicy::CONTINUOUS_BATCHING);
    ~ContinuousBatchEngine();

    ContinuousBatchEngine(const ContinuousBatchEngine&) = delete;
    ContinuousBatchEngine& operator=(const ContinuousBatchEngine&) = delete;

    // Blocking completion (used by the HTTP server).
    Result complete(const Request& req);

    // Streaming completion: on_token is invoked on the worker thread as tokens are produced.
    // on_token returns false to cancel generation early (e.g. the client disconnected) --
    // the request then finishes with Result::cancelled = true, not an error.
    //
    // on_token_logprob (optional) is invoked once per token, immediately BEFORE on_token fires
    // for that same token, only when req.logprobs is true AND this callback is non-null -- pass
    // nullptr (not a no-op lambda) when logprobs aren't wanted, since step_job()'s cost gate is
    // `req.logprobs && on_token_logprob`; an always-non-null callback would defeat the
    // "logprobs=false costs nothing extra" property.
    Result complete_streaming(const Request& req, const std::function<bool(int)>& on_token,
                              const std::function<void(const Qwen35Model::TokenLogprob&)>&
                                  on_token_logprob = nullptr);

    int num_active() const;

    // Vision tower used by step_job when a request carries images. Both non-owning and expected
    // to outlive the engine (ModelEngine owns them). Never set => a request with images is
    // rejected rather than silently answered from its text alone, which would produce a fluent
    // description of an image the model never saw.
    void set_vision(const QwenVisionWeights* weights, const QwenVisionConfig* cfg);
    int num_free_kv_blocks() const;
    // Admission-time queue depth cap (SPARKINFER_MAX_QUEUE_DEPTH, 0 = unlimited). Requests
    // beyond this are rejected as overloaded before any KV allocation is attempted.
    int max_queue_depth() const;

private:
    struct Job;
    enum class EnqueueError { NONE, BAD_REQUEST, OVERLOADED, ALLOC_FAILED };
    uint64_t submit_locked(Job job, const std::function<bool(int)>& on_token,
                           const std::function<void(const Qwen35Model::TokenLogprob&)>& on_token_logprob,
                           EnqueueError* err_out);
    Result wait_locked(uint64_t request_id);
    void worker_loop();
    bool step_job(Job& job, bool chunked = false);

    Qwen35Model* model_;
    KVCacheManager* kv_;
    Scheduler scheduler_;
    SchedulePolicy policy_ = SchedulePolicy::CONTINUOUS_BATCHING;
    const QwenVisionWeights* vision_weights_ = nullptr;
    const QwenVisionConfig* vision_cfg_ = nullptr;
    std::thread worker_;
    std::atomic<bool> running_{false};
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::unordered_map<uint64_t, std::unique_ptr<Job>> jobs_;
    std::atomic<uint64_t> next_req_id_{1};
};

}  // namespace sparkinfer
