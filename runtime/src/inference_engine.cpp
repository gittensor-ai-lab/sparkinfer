#include "sparkinfer/inference_engine.h"
#include "sparkinfer/models/qwen_vision.h"

#include "sparkinfer/device_health.h"

#include <mutex>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace sparkinfer {

namespace {

// Chunked-prefill budget (vLLM-style): when decode requests are waiting, only
// advance this many prefill tokens before yielding. 0 = unlimited (full prompt).
int prefill_chunk_tokens() {
    static int chunk = []{
        const char* e = getenv("SPARKINFER_PREFILL_CHUNK_TOKENS");
        // Default 512 — large enough for batched GEMM amortization, small enough
        // that concurrent decode keeps receiving tokens every few ms.
        int c = e ? atoi(e) : 512;
        return c >= 0 ? c : 512;
    }();
    return chunk;
}

// Admission-time queue depth cap. 0 (default) = unlimited, matching prior behaviour --
// operators opt in to admission control rather than getting a surprise cap.
int max_queue_depth_config() {
    static int cap = []{
        const char* e = getenv("SPARKINFER_MAX_QUEUE_DEPTH");
        return e ? std::max(0, atoi(e)) : 0;
    }();
    return cap;
}

// Per-request wall-clock deadline from submit to finish. 0 (default) = disabled --
// long-context prefill alone can legitimately take well over a minute (measured: ~94s TTFT
// at 32k context), so an aggressive default would misfire on correct, expected-slow requests.
double request_timeout_s_config() {
    static double s = []{
        const char* e = getenv("SPARKINFER_REQUEST_TIMEOUT_S");
        return e ? std::max(0.0, atof(e)) : 0.0;
    }();
    return s;
}

}  // namespace

struct ContinuousBatchEngine::Job {
    uint64_t request_id = 0;
    Request req;
    uint64_t seq_id = 0;
    SeqPhase phase = SeqPhase::PREFILL;
    int prefill_pos = 0;
    int decode_emitted = 0;
    int next_token = -1;
    std::vector<int> output;
    std::string error;
    std::function<bool(int)> on_token;  // false return = cancel
    // Optional. Delivered one step_job() call AFTER forward_token() actually computed it -- see
    // step_job()'s implementation for why (worker_loop() interleaves step_job() across jobs
    // sharing one Qwen35Model instance, so the logprobs data must be read out of the model's
    // shared decode scratch synchronously, within the SAME step_job() call that produced it,
    // before any other job's forward_token() can overwrite that scratch).
    std::function<void(const Qwen35Model::TokenLogprob&)> on_token_logprob;
    Qwen35Model::TokenLogprob pending_logprob;
    bool have_pending_logprob = false;
    bool done = false;
    bool overloaded = false;
    bool timed_out = false;
    bool cancelled = false;
    bool reached_token_limit = false;

    std::chrono::steady_clock::time_point t_submit{};
    std::chrono::steady_clock::time_point t_first{};
    bool saw_first_tok = false;
    double ttft_ms = -1.0;
    double generation_ms = -1.0;
    double decode_tps = -1.0;
};

ContinuousBatchEngine::ContinuousBatchEngine(Qwen35Model* model, KVCacheManager* kv,
                                             int max_tokens_per_batch, SchedulePolicy policy)
    : model_(model), kv_(kv), scheduler_(policy, max_tokens_per_batch), policy_(policy) {
    running_ = true;
    worker_ = std::thread([this] { worker_loop(); });
}

ContinuousBatchEngine::~ContinuousBatchEngine() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        running_ = false;
        cv_.notify_all();
    }
    if (worker_.joinable()) worker_.join();
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& kv : jobs_) {
        if (!kv.second || kv.second->done) continue;
        // seq_id 0 is a valid prefix session (cannot use truthiness).
        if (kv.second->seq_id != 0) model_->close_session(kv.second->seq_id);
        else if (kv.second->req.use_prefix_session) {
            kv_->free(0);
            model_->release_prefix_session();
        }
    }
    jobs_.clear();
}

ContinuousBatchEngine::Result ContinuousBatchEngine::complete(const Request& req) {
    return complete_streaming(req, nullptr);
}

ContinuousBatchEngine::Result ContinuousBatchEngine::complete_streaming(
    const Request& req, const std::function<bool(int)>& on_token,
    const std::function<void(const Qwen35Model::TokenLogprob&)>& on_token_logprob) {
    uint64_t rid = 0;
    EnqueueError err = EnqueueError::NONE;
    {
        std::lock_guard<std::mutex> lock(mu_);
        Job job;
        job.req = req;
        job.prefill_pos = req.prefill_start;
        rid = submit_locked(std::move(job), on_token, on_token_logprob, &err);
    }
    if (!rid) {
        Result out;
        out.overloaded = (err == EnqueueError::OVERLOADED);
        out.alloc_failed = (err == EnqueueError::ALLOC_FAILED);
        out.error = out.overloaded ? "server overloaded: no capacity for this request right now"
                  : out.alloc_failed ? "device out of memory (not a capacity/queue condition -- "
                                        "requires operator attention)"
                                    : "failed to enqueue request";
        return out;
    }
    return wait_locked(rid);
}

void ContinuousBatchEngine::set_vision(const QwenVisionWeights* weights,
                                      const QwenVisionConfig* cfg) {
    std::lock_guard<std::mutex> lock(mu_);
    vision_weights_ = weights;
    vision_cfg_ = cfg;
}

int ContinuousBatchEngine::num_active() const {
    std::lock_guard<std::mutex> lock(mu_);
    int n = 0;
    for (const auto& kv : jobs_) if (!kv.second->done) n++;
    return n;
}

int ContinuousBatchEngine::num_free_kv_blocks() const { return kv_->num_free_blocks(); }

int ContinuousBatchEngine::max_queue_depth() const { return max_queue_depth_config(); }

uint64_t ContinuousBatchEngine::submit_locked(Job job, const std::function<bool(int)>& on_token,
                                              const std::function<void(const Qwen35Model::TokenLogprob&)>& on_token_logprob,
                                              EnqueueError* err_out) {
    EnqueueError err = EnqueueError::BAD_REQUEST;
    auto fail = [&](EnqueueError e) { if (err_out) *err_out = e; return uint64_t{0}; };

    if (!model_ || !kv_) return fail(err);
    // Context already dead (see device_health.h). Refuse immediately with the same code a real
    // device OOM uses -- ALLOC_FAILED maps to 503 "requires operator attention", which is exactly
    // right: this is permanent until the process restarts. Admitting the request instead would
    // launch more work against a dead context, and that is the path that ends in a corrupted host
    // heap rather than an error response.
    if (device_lost()) return fail(EnqueueError::ALLOC_FAILED);
    if (job.req.prompt.empty() || job.req.max_new_tokens <= 0) return fail(err);
    if ((int)job.req.prompt.size() + job.req.max_new_tokens > model_->config().max_seq)
        return fail(EnqueueError::BAD_REQUEST);

    const int cap = max_queue_depth_config();
    if (cap > 0) {
        int active = 0;
        for (const auto& kv : jobs_) if (!kv.second->done) active++;
        if (active >= cap) return fail(EnqueueError::OVERLOADED);
    }

    const int budget = Qwen35Model::session_token_budget(
        job.req.prompt.size(), job.req.max_new_tokens, model_->config().max_seq);

    uint64_t seq_id = 0;
    {
        // EVERY device call below must be excluded from the worker thread's CUDA-graph capture.
        // They all issue work on the legacy default stream -- allocate()'s block-table cudaMemcpy,
        // open_session()'s cudaMalloc, reset_penalty_counts()'s memset, set_logit_bias()'s copy --
        // and the decode stream is a blocking stream, so the legacy stream implicitly synchronizes
        // with it. Landing any of them inside a capture is a hard CUDA error that poisons the
        // graph and takes the process down a few instructions later. See
        // Qwen35Model::device_mutex().
        //
        // Scoped to just this block, NOT the whole function: submit_locked already holds mu_, and
        // holding a device lock across the queue bookkeeping below would widen the window a decode
        // step can be blocked for, for no benefit -- none of that bookkeeping touches the device.
        //
        // Ordering is mu_ (held by our caller) then device_mutex(). The worker takes only
        // device_mutex() and never mu_ while stepping, so there is no cycle.
        std::lock_guard<std::recursive_mutex> device_lock(model_->device_mutex());
        if (job.req.use_prefix_session) {
            seq_id = 0;
            if (!kv_->allocate(seq_id, budget)) return fail(EnqueueError::OVERLOADED);
            model_->activate_session(seq_id);
            model_->reset_penalty_counts(seq_id);   // session 0 is shared across unrelated requests
            model_->set_logit_bias(seq_id, job.req.logit_bias);   // same reason
        } else {
            bool alloc_failed = false;
            seq_id = model_->open_session(budget, &alloc_failed);
            if (!seq_id) return fail(alloc_failed ? EnqueueError::ALLOC_FAILED : EnqueueError::OVERLOADED);
            model_->reset_penalty_counts(seq_id);   // explicit, not relying on open_session's internal zero
            model_->set_logit_bias(seq_id, job.req.logit_bias);    // same reason
        }
    }

    job.request_id = next_req_id_.fetch_add(1);
    job.seq_id = seq_id;
    job.on_token = on_token;
    job.on_token_logprob = on_token_logprob;
    job.prefill_pos = job.req.prefill_start;
    job.t_submit = std::chrono::steady_clock::now();
    auto ptr = std::make_unique<Job>(std::move(job));
    const uint64_t rid = ptr->request_id;
    jobs_[rid] = std::move(ptr);
    cv_.notify_one();
    if (err_out) *err_out = EnqueueError::NONE;
    return rid;
}

ContinuousBatchEngine::Result ContinuousBatchEngine::wait_locked(uint64_t request_id) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] {
        auto it = jobs_.find(request_id);
        return it == jobs_.end() || it->second->done;
    });
    auto it = jobs_.find(request_id);
    if (it == jobs_.end()) return Result{{}, "request not found"};
    Result out;
    out.tokens = it->second->output;
    out.error = it->second->error;
    out.overloaded = it->second->overloaded;
    out.timed_out = it->second->timed_out;
    out.cancelled = it->second->cancelled;
    out.reached_token_limit = it->second->reached_token_limit;
    out.ttft_ms = it->second->ttft_ms;
    out.generation_ms = it->second->generation_ms;
    out.decode_tps = it->second->decode_tps;
    jobs_.erase(it);
    return out;
}

void ContinuousBatchEngine::worker_loop() {
    while (true) {
        std::vector<uint64_t> prefill_ids, decode_ids;
        {
            std::unique_lock<std::mutex> lock(mu_);
            if (!running_ && jobs_.empty()) return;

            std::vector<ScheduledSequence> active;
            active.reserve(jobs_.size());
            for (const auto& kv : jobs_) {
                if (kv.second->done) continue;
                ScheduledSequence s;
                s.request_id = kv.first;
                s.seq_id = kv.second->seq_id;
                s.phase = kv.second->phase;
                s.priority = kv.second->req.priority;
                s.tokens_in_phase = (kv.second->phase == SeqPhase::PREFILL)
                                        ? kv.second->prefill_pos
                                        : kv.second->decode_emitted;
                s.prefill_remaining = (kv.second->phase == SeqPhase::PREFILL)
                                          ? (int)kv.second->req.prompt.size() - kv.second->prefill_pos
                                          : 0;
                active.push_back(s);
            }

            ScheduleBatch batch = scheduler_.schedule(active);
            prefill_ids = batch.prefill_request_ids;
            decode_ids = batch.decode_request_ids;

            if (prefill_ids.empty() && decode_ids.empty()) {
                if (!running_) {
                    bool any = false;
                    for (const auto& kv : jobs_)
                        if (!kv.second->done) { any = true; break; }
                    if (!any) return;
                }
                cv_.wait_for(lock, std::chrono::milliseconds(2));
                continue;
            }
        }

        // vLLM V1 iteration: advance every packed decode token first (ITPS), then
        // one prefill chunk if scheduled. Re-enter the scheduler after the step.
        bool any_finished = false;
        const bool mix_decode = !decode_ids.empty();
        for (uint64_t id : decode_ids) {
            Job* job = nullptr;
            {
                std::lock_guard<std::mutex> lock(mu_);
                auto it = jobs_.find(id);
                if (it != jobs_.end() && !it->second->done) job = it->second.get();
            }
            if (job) any_finished = step_job(*job, /*chunked=*/false) || any_finished;
        }
        if (!prefill_ids.empty()) {
            Job* job = nullptr;
            {
                std::lock_guard<std::mutex> lock(mu_);
                auto it = jobs_.find(prefill_ids.front());
                if (it != jobs_.end() && !it->second->done) job = it->second.get();
            }
            if (job) any_finished = step_job(*job, /*chunked=*/mix_decode ||
                                             policy_ == SchedulePolicy::CHUNKED_PREFILL) || any_finished;
        }
        if (any_finished) cv_.notify_all();
    }
}

bool ContinuousBatchEngine::step_job(Job& job, bool chunked) {
    const Qwen35Config& cfg = model_->config();
    // Bail before touching the device. An in-flight job on a lost context would otherwise keep
    // stepping -- every launch failing, every readback leaving stale host memory -- for the rest
    // of its max_new_tokens budget, across every queued job. That grind is what turned one
    // illegal access into 21,535 error lines and, eventually, a corrupted host heap. Fail the
    // job with a clear reason instead; submit_locked() is already refusing new ones.
    if (device_lost()) {
        job.error = "CUDA context lost (unrecoverable device error) -- request aborted; "
                    "the server requires a restart";
        job.done = true;
        cv_.notify_all();
        return true;
    }
    model_->activate_session(job.seq_id);

    // Shared "finish this job" helper: closes/frees whatever KV/session it holds and marks
    // done. step_job's several early-exit paths (timeout, cancel, invalid seed, eos/max_tokens)
    // all need exactly this cleanup -- duplicating it inline four times is how one of those
    // paths quietly drifts out of sync with the others. A lambda (not a free function) because
    // Job is private to ContinuousBatchEngine; only code with this member function's access can
    // name it.
    auto finish_job = [this](Job& j) {
        j.done = true;
        if (j.seq_id != 0) {
            // Offer this session's KV to the external cache tier (docs/lmcache_bridge_protocol.md)
            // only once the full prompt has actually been ingested -- j.phase only advances past
            // PREFILL once prefill_pos reaches the prompt's end (see step_job). A job
            // cancelled/timed-out mid-prefill has KV for only part of its prompt range (possibly
            // garbage past prefill_pos), so store_tokens must stay null in that case; passing the
            // full prompt would tell close_session a longer range is valid than actually is.
            model_->close_session(j.seq_id, j.phase != SeqPhase::PREFILL ? &j.req.prompt : nullptr);
        } else {
            kv_->free(j.seq_id);
            if (j.req.use_prefix_session) model_->release_prefix_session();
        }
        j.seq_id = 0;
    };

    const double timeout_s = request_timeout_s_config();
    if (timeout_s > 0.0) {
        const double elapsed_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - job.t_submit).count();
        if (elapsed_s > timeout_s) {
            job.error = "request timeout after " + std::to_string(elapsed_s) + "s (limit " +
                        std::to_string(timeout_s) + "s)";
            job.timed_out = true;
            finish_job(job);
            return true;
        }
    }

    if (job.phase == SeqPhase::PREFILL) {
        const int n = (int)job.req.prompt.size();
        const int chunk = prefill_chunk_tokens();
        // Qwen35Model::ingest_prompt_range() is the single funnel both this continuous-batch
        // path and cache_prefix()'s exclusive-session path dispatch prefill through: it picks
        // batched GEMM prefill (every hybrid this runtime serves -- Qwythos, Qwen3.6, Qwen3.8,
        // Muse Glimmer; prefill_batched_run's guards are the authority, and it is only ever
        // eligible from position 0) vs. the token-loop fallback itself. The batched path is
        // worth one to two orders of magnitude over the token loop depending on model and
        // context -- ~28x at ctx=16384 on Qwen3.8, more on Qwythos -- so treat any single ratio
        // quoted in this tree as the shape it was measured at. The batched path
        // never chunks (no start_pos support in prefill_batched_run yet) regardless of the
        // chunk_limit passed here -- decode-first scheduling already advances waiting decodes
        // once before a full batched pass runs, so that never hurts ITPS under mixed load.
        const int chunk_limit = chunked ? chunk : 0;
        int out_pos = job.prefill_pos;
        // want_seed_logprob: the seed this returns IS the response's first emitted token, and it
        // is produced here rather than by forward_token(), so its logprob has to be collected
        // here too or the first token gets no entry at all. Asked for only when the request wants
        // logprobs -- it costs a full-vocab sort, and unlike the decode path (frozen graph
        // topology) this one is free to branch on the host.
        const bool want_seed_logprob = job.req.logprobs && job.on_token_logprob;
        // Run the vision tower for this job HERE, on the worker thread, and stage the result
        // only for the prefill immediately below. Both halves of that are required: the tower
        // issues work on the legacy default stream, which cannot overlap another job's CUDA
        // graph capture, and set_pending_vision is a single slot on the shared model, so leaving
        // it set would splice this request's image into whatever job prefills next. One worker
        // thread runs step_job, so nothing can prefill between the set and the clear.
        const bool has_vision = !job.req.vision_pos.empty();
        if (has_vision && (!vision_weights_ || !vision_cfg_)) {
            job.error = "this model has no vision tower; image input is not supported";
            finish_job(job);
            return true;
        }
        if (has_vision) {
            std::vector<float> emb;
            std::string verr;
            for (const auto& img : job.req.vision_images) {
                const int nblk = (img.grid_h / vision_cfg_->spatial_merge) *
                                 (img.grid_w / vision_cfg_->spatial_merge);
                const size_t off = emb.size();
                emb.resize(off + (size_t)nblk * vision_cfg_->out_hidden);
                if (!img.pixels ||
                    !qwen_vision_forward(*vision_weights_, *vision_cfg_, img.pixels->data(),
                                         img.grid_h, img.grid_w, emb.data() + off, verr)) {
                    job.error = "vision tower failed: " + verr;
                    finish_job(job);
                    return true;
                }
            }
            if (emb.size() != job.req.vision_pos.size() * (size_t)vision_cfg_->out_hidden ||
                !model_->set_pending_vision(emb.data(), job.req.vision_pos.data(),
                                            (int)job.req.vision_pos.size(),
                                            vision_cfg_->out_hidden)) {
                job.error = "failed to stage image embeddings for prefill";
                finish_job(job);
                return true;
            }
            // Rotary positions for the same prompt. Staged separately from the embeddings because
            // they describe every token, not just the spliced ones -- and because a checkpoint
            // without an mrope_section supplies none while still having images.
            if (!job.req.mrope_pos.empty() &&
                !model_->set_pending_mrope(job.req.mrope_pos.data(),
                                           (int)(job.req.mrope_pos.size() / 3),
                                           job.req.mrope_decode_offset)) {
                job.error = "failed to stage MRoPE positions for prefill";
                finish_job(job);
                return true;
            }
        }
        // The decode offset deliberately OUTLIVES the positions -- decode needs it for every token
        // after the prompt -- so a request that supplies none must clear it explicitly. Without
        // this, a text-only request arriving after an image request on the same model would
        // inherit the image's rotary shift and silently decode at the wrong positions.
        if (job.req.mrope_pos.empty()) model_->reset_mrope_offset();
        const int seed = model_->ingest_prompt_range(job.req.prompt.data(), job.prefill_pos, n,
                                                      chunk_limit, &out_pos, want_seed_logprob);
        if (has_vision) model_->clear_pending_vision();
        // Positions are consumed by the prefill they were staged for; the offset is not cleared
        // here, by design.
        if (!job.req.mrope_pos.empty()) model_->clear_pending_mrope();
        job.prefill_pos = out_pos;
        if (job.prefill_pos >= n) {
            // Known v1 scope limitation for temperature sampling (runtime/src/models/qwen35.cpp's
            // forward_token doc comment): ingest_prompt_range() is a single funnel shared by
            // cache_prefix()'s exclusive-session path and several other internal call sites, so
            // its own argmax seed pick is not made temperature-aware here -- threading it through
            // would mean changing a widely-shared function for the benefit of exactly one token.
            // Concretely: the FIRST emitted token of every response is always the greedy/argmax
            // token regardless of `temperature`; every token from the second one onward (all of
            // which flow through forward_token() below) correctly respects temperature/seed.
            job.next_token = seed;
            if (job.next_token < 0 && job.req.use_prefix_session)
                job.next_token = model_->prefix_seed_token();
            job.phase = SeqPhase::DECODE;
            // Stage the seed's logprob exactly the way the decode branch stages every subsequent
            // token's: the NEXT step_job() call emits job.next_token and flushes this pending
            // entry alongside it, so the first entry describes the first emitted token and
            // logprobs.content finally has one entry per generated token.
            //
            // Read here, synchronously, for the same reason the decode branch does it: the
            // sampler scratch it comes from is shared across every job on this model, so it must
            // be consumed before any other job's forward_token() can overwrite it.
            //
            // Teacher-forced scoring: the first token of the "response" is the caller's, not the
            // model's, so replace the argmax seed before anything reports on it. The distribution
            // just computed at the last prompt position is exactly the one that predicts it.
            const bool forcing = !job.req.forced_tokens.empty();
            if (forcing) job.next_token = job.req.forced_tokens[0];
            if (want_seed_logprob && job.next_token >= 0 && (forcing || job.next_token == seed)) {
                // job.next_token == seed guard (generation case): the use_prefix_session fallback
                // above can substitute a token from a DIFFERENT forward pass (the cached prefix's
                // own seed), which this scratch does not describe -- reporting it would be a wrong
                // number rather than a missing one, so that case keeps the old one-short
                // behaviour. Forcing is exempt: the forced token is scored against this
                // distribution by definition, whatever the seed was.
                job.pending_logprob =
                    forcing ? model_->token_logprob_for(job.next_token, job.req.top_logprobs)
                            : model_->last_token_logprobs(job.req.top_logprobs);
                job.have_pending_logprob = true;
            }
        }
        return false;
    }

    if (job.next_token < 0 || job.next_token >= cfg.vocab) {
        job.error = "prefill produced invalid seed token";
        // Match generate(): KV is gone — soft-invalidate so the server does not
        // skip cache_prefix() on the next exclusive prefix hit (finish_job's
        // use_prefix_session branch below handles that).
        finish_job(job);
        return true;
    }

    // Timestamp before on_token so SSE/network backpressure never enters GPU metrics.
    const auto t_emit = std::chrono::steady_clock::now();
    if (!job.saw_first_tok) {
        job.t_first = t_emit;
        job.saw_first_tok = true;
        job.ttft_ms = std::chrono::duration<double, std::milli>(job.t_first - job.t_submit).count();
    }
    job.output.push_back(job.next_token);
    job.decode_emitted++;
    // Delivered one step_job() call after forward_token() computed it (see Job::on_token_logprob's
    // doc comment) -- must fire BEFORE on_token below, since job.next_token is exactly the token
    // this pending_logprob describes.
    if (job.on_token_logprob && job.have_pending_logprob) {
        job.on_token_logprob(job.pending_logprob);
        job.have_pending_logprob = false;
    }
    // false => caller (e.g. the HTTP layer, when the client disconnected mid-stream) wants
    // generation stopped now. Not an error -- free resources same as a normal finish.
    if (job.on_token && !job.on_token(job.next_token)) {
        job.cancelled = true;
        job.generation_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - job.t_submit).count();
        finish_job(job);
        return true;
    }

    // EOS never ends a teacher-forced score: the caller asked about a specific token sequence and
    // is owed a logprob for every token in it, even if it contains an end marker.
    const bool hit_eos = job.req.forced_tokens.empty() &&
                         (job.next_token == cfg.eos_id ||
                          (cfg.eos_id2 >= 0 && job.next_token == cfg.eos_id2));
    const bool hit_token_limit = job.decode_emitted >= job.req.max_new_tokens;
    if (hit_eos || hit_token_limit) {
        job.reached_token_limit = hit_token_limit && !hit_eos;
        const auto t_end = std::chrono::steady_clock::now();
        job.generation_ms = std::chrono::duration<double, std::milli>(t_end - job.t_submit).count();
        if (job.saw_first_tok && job.generation_ms > job.ttft_ms && job.decode_emitted > 0) {
            const double decode_ms = std::max(job.generation_ms - job.ttft_ms, 1.0);
            job.decode_tps = (double)job.decode_emitted * 1000.0 / decode_ms;
        }
        finish_job(job);
        return true;
    }

    const int prompt_len = (int)job.req.prompt.size();
    const int sampled = model_->forward_token(job.next_token, prompt_len + job.decode_emitted - 1, true,
                                           job.req.temperature, job.req.seed,
                                           (uint64_t)job.decode_emitted,
                                           job.req.top_k, job.req.top_p,
                                           job.req.presence_penalty, job.req.frequency_penalty);
    // Teacher-forced scoring substitutes the caller's token for the sampler's pick. The forward
    // pass above still ran in full, so the KV/GDN state this leaves behind is exactly the state
    // the supplied sequence implies -- which is the whole point: position i+1 is scored under a
    // context that actually contains token i.
    //
    // decode_emitted has already been incremented for the token emitted at the top of this call,
    // so it indexes the NEXT forced token. Past the end (only reachable if max_new_tokens exceeds
    // the forced sequence) it falls back to the sampled token rather than reading out of bounds.
    const bool forcing = !job.req.forced_tokens.empty();
    const size_t next_forced = (size_t)job.decode_emitted;
    job.next_token = (forcing && next_forced < job.req.forced_tokens.size())
                         ? job.req.forced_tokens[next_forced]
                         : sampled;
    // Must run in THIS step_job() call, synchronously, before any OTHER job's forward_token()
    // (worker_loop() interleaves jobs sharing one Qwen35Model instance) can overwrite the shared
    // decode scratch last_token_logprobs() reads from. See Job::on_token_logprob's doc comment.
    if (job.req.logprobs && job.on_token_logprob) {
        job.pending_logprob = forcing ? model_->token_logprob_for(job.next_token, job.req.top_logprobs)
                                      : model_->last_token_logprobs(job.req.top_logprobs);
        job.have_pending_logprob = true;
    }
    return false;
}

}  // namespace sparkinfer
