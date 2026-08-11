#include "sparkinfer/inference_engine.h"

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
    bool done = false;
    bool overloaded = false;
    bool timed_out = false;
    bool cancelled = false;

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
    const Request& req, const std::function<bool(int)>& on_token) {
    uint64_t rid = 0;
    EnqueueError err = EnqueueError::NONE;
    {
        std::lock_guard<std::mutex> lock(mu_);
        Job job;
        job.req = req;
        job.prefill_pos = req.prefill_start;
        rid = submit_locked(std::move(job), on_token, &err);
    }
    if (!rid) {
        Result out;
        out.overloaded = (err == EnqueueError::OVERLOADED);
        out.error = out.overloaded ? "server overloaded: no capacity for this request right now"
                                    : "failed to enqueue request";
        return out;
    }
    return wait_locked(rid);
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
                                              EnqueueError* err_out) {
    EnqueueError err = EnqueueError::BAD_REQUEST;
    auto fail = [&](EnqueueError e) { if (err_out) *err_out = e; return uint64_t{0}; };

    if (!model_ || !kv_) return fail(err);
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
    if (job.req.use_prefix_session) {
        seq_id = 0;
        if (!kv_->allocate(seq_id, budget)) return fail(EnqueueError::OVERLOADED);
        model_->activate_session(seq_id);
    } else {
        seq_id = model_->open_session(budget);
        if (!seq_id) return fail(EnqueueError::OVERLOADED);
    }

    job.request_id = next_req_id_.fetch_add(1);
    job.seq_id = seq_id;
    job.on_token = on_token;
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
    model_->activate_session(job.seq_id);

    // Shared "finish this job" helper: closes/frees whatever KV/session it holds and marks
    // done. step_job's several early-exit paths (timeout, cancel, invalid seed, eos/max_tokens)
    // all need exactly this cleanup -- duplicating it inline four times is how one of those
    // paths quietly drifts out of sync with the others. A lambda (not a free function) because
    // Job is private to ContinuousBatchEngine; only code with this member function's access can
    // name it.
    auto finish_job = [this](Job& j) {
        j.done = true;
        if (j.seq_id != 0) model_->close_session(j.seq_id);
        else {
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
        // batched GEMM prefill (Qwythos / Qwen3.6 hybrid, ~100x faster than the token loop, only
        // eligible from position 0) vs. the token-loop fallback itself, and the batched path
        // never chunks (no start_pos support in prefill_batched_run yet) regardless of the
        // chunk_limit passed here -- decode-first scheduling already advances waiting decodes
        // once before a full batched pass runs, so that never hurts ITPS under mixed load.
        const int chunk_limit = chunked ? chunk : 0;
        int out_pos = job.prefill_pos;
        const int seed = model_->ingest_prompt_range(job.req.prompt.data(), job.prefill_pos, n,
                                                      chunk_limit, &out_pos);
        job.prefill_pos = out_pos;
        if (job.prefill_pos >= n) {
            job.next_token = seed;
            if (job.next_token < 0 && job.req.use_prefix_session)
                job.next_token = model_->prefix_seed_token();
            job.phase = SeqPhase::DECODE;
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
    // false => caller (e.g. the HTTP layer, when the client disconnected mid-stream) wants
    // generation stopped now. Not an error -- free resources same as a normal finish.
    if (job.on_token && !job.on_token(job.next_token)) {
        job.cancelled = true;
        job.generation_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - job.t_submit).count();
        finish_job(job);
        return true;
    }

    if (job.next_token == cfg.eos_id || (cfg.eos_id2 >= 0 && job.next_token == cfg.eos_id2) ||
        job.decode_emitted >= job.req.max_new_tokens) {
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
    job.next_token = model_->forward_token(job.next_token, prompt_len + job.decode_emitted - 1, true);
    return false;
}

}  // namespace sparkinfer
