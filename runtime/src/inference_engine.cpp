#include "sparkinfer/inference_engine.h"
#include "sparkinfer/models/qwen_vision.h"

#include "sparkinfer/device_health.h"

#include <mutex>
#include <deque>
#include <memory>

#include <algorithm>
#include <chrono>
#include <cmath>
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

// How long a request that finds no free KV capacity waits for it, first come first served, before
// it is rejected as overloaded (#1088). 0 restores the old immediate 429. The default matches the
// server's 300 s socket timeouts.
double admission_wait_s_config() {
    static double s = []{
        const char* e = getenv("SPARKINFER_ADMISSION_WAIT_S");
        return e ? std::max(0.0, atof(e)) : 300.0;
    }();
    return s;
}

}  // namespace

struct ContinuousBatchEngine::Job {
    // Constrained decoding: the mask last uploaded for this job and the dense bias built from it. A
    // step whose mask is unchanged -- most of free text -- uploads nothing.
    std::vector<uint32_t> mask_bits;
    std::vector<uint32_t> mask_next;
    std::vector<float> mask_bias;
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
    // ASYNC EMISSION (SPARKINFER_ASYNC_EMIT, on unless =0). The caller's callbacks -- for a
    // streamed chat, incremental detokenization, the stop filter, the SSE JSON and a socket write
    // -- ran on the worker thread, one row after another, while the device idled: ~4 ms of every
    // 19 ms step at 32 rows. With a queue, on_token/on_token_logprob only enqueue, and the
    // request's own thread (wait_locked) runs the callbacks, all requests in parallel. A callback
    // that returns false sets `stop`, which on_token reports to the worker on its next token; the
    // result is cut back to the tokens the callback took.
    struct EmitQueue {
        struct Event {
            bool is_logprob = false;
            int token = -1;
            Qwen35Model::TokenLogprob lp;
        };
        std::mutex mu;
        std::condition_variable cv;
        std::deque<Event> events;
        bool stop = false;     // the caller's on_token returned false
        bool closed = false;   // the job is done: nothing more will be queued
        std::function<bool(int)> user_token;
        std::function<void(const Qwen35Model::TokenLogprob&)> user_logprob;
    };
    std::shared_ptr<EmitQueue> emit;
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
    bool internal_error = false;
    bool reached_token_limit = false;

    std::chrono::steady_clock::time_point t_submit{};
    std::chrono::steady_clock::time_point t_first{};
    bool saw_first_tok = false;
    double ttft_ms = -1.0;
    double generation_ms = -1.0;
    double decode_tps = -1.0;

    // Prefix cache: tokens this job started from rather than prefilled, and the recurrent-state
    // snapshots taken at req.cache_checkpoints (offered to the cache in finish_job_impl).
    int cached_tokens = 0;
    int mixed_tokens = 0;      // prompt tokens mixed steps have ingested (step_jobs_packed)
    struct Checkpoint {
        int pos = 0;
        Qwen35Model::RecurrentStateSnapshot state;
    };
    std::vector<Checkpoint> checkpoints;
    bool spec_tried = false;   // run_speculative has had its one chance at this job
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
    bool gave_up = false, deadline_ran_out = false;
    // Counted from before mu_ is taken until the submission resolves, so the worker can tell a
    // request that is about to appear from one that is not coming (see worker_loop).
    submitting_.fetch_add(1, std::memory_order_acq_rel);
    struct SubmitDone {
        ContinuousBatchEngine* e;
        ~SubmitDone() {
            e->submitting_.fetch_sub(1, std::memory_order_acq_rel);
            e->cv_.notify_all();
        }
    };
    {
        SubmitDone submit_done{this};
        std::unique_lock<std::mutex> lock(mu_);
        // Every request reserves KV for its prompt plus max_tokens when it is admitted, so a few
        // agents asking for long outputs can hold the whole pool. Such a request used to be
        // rejected with 429 at once, which clients like prime-agent surface as a failed turn
        // (#1088). It now waits for capacity, oldest first, woken whenever a job finishes or
        // completes its prefill (the worker notifies cv_). A device allocation that fails while
        // other requests run waits too: see alloc_wait below. The queue-depth cap still rejects
        // new arrivals at once, and a bad request, or an allocation failure with nothing else
        // running, never waits.
        const double wait_s = admission_wait_s_config();
        const double timeout_s = request_timeout_s_config();
        const double limit_s = timeout_s > 0 ? std::min(wait_s, timeout_s) : wait_s;
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(limit_s));
        uint64_t ticket = 0;
        bool queued = false;
        bool alloc_wait = false;
        for (;;) {
            const bool my_turn = limit_s <= 0 || waiting_.empty() ||
                                 (queued && *waiting_.begin() == ticket);
            if (my_turn) {
                Job job;
                job.req = req;
                job.prefill_pos = req.prefill_start;
                rid = submit_locked(std::move(job), on_token, on_token_logprob, &err);
                if (rid || limit_s <= 0 || !running_) break;
                // A session allocation that fails while other requests run is capacity, not the
                // card: their prefill scratch and session state come back as they progress. With
                // concurrent 20K-token prompts on serve-dspark at --ctx 131072, every request
                // arriving while one prefilled was refused as "device out of memory ... requires
                // operator attention", and requests that would have fit a moment later failed
                // (#1088). With nothing else running, it is the card, and still a 503.
                alloc_wait = err == EnqueueError::ALLOC_FAILED && !device_lost() && active_jobs_locked() > 0;
                if (err != EnqueueError::OVERLOADED && !alloc_wait) break;
            }
            if (!queued) {
                if (queue_depth_full_locked()) { err = EnqueueError::OVERLOADED; break; }
                ticket = next_wait_ticket_++;
                waiting_.insert(ticket);
                queued = true;
                admission_waits_.fetch_add(1, std::memory_order_relaxed);
            }
            // Memory can also come back without a job finishing or completing its prefill (a
            // speculative run's teardown, say), so an allocation wait retries every 2 s as well.
            const auto wake = alloc_wait ? std::min(deadline, std::chrono::steady_clock::now() +
                                                                  std::chrono::seconds(2))
                                         : deadline;
            if (cv_.wait_until(lock, wake) == std::cv_status::timeout &&
                std::chrono::steady_clock::now() >= deadline) {
                gave_up = true;
                deadline_ran_out = timeout_s > 0 && timeout_s <= wait_s;
                break;
            }
        }
        if (queued) {
            waiting_.erase(ticket);
            cv_.notify_all();   // the next waiter's turn
        }
    }
    if (!rid && gave_up) {
        admission_timeouts_.fetch_add(1, std::memory_order_relaxed);
        Result out;
        if (deadline_ran_out) {
            out.timed_out = true;
            out.error = "request timed out waiting for capacity (SPARKINFER_REQUEST_TIMEOUT_S)";
        } else {
            out.overloaded = true;
            out.error = "server overloaded: no capacity for this request within SPARKINFER_ADMISSION_WAIT_S";
        }
        return out;
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

// Free blocks plus those only the prefix cache holds: admission evicts the cache for those
// (PrefixCache::evict_for), so for a caller sizing what it may send they are free. The refcounts
// behind the second term change under the device mutex, so they are read only while holding it --
// and only if it is free: /v1/capacity and /metrics must not wait out a long prefill. Otherwise the
// last count stands.
int ContinuousBatchEngine::num_free_kv_blocks() const {
    if (prefix_cache_) {
        std::unique_lock<std::recursive_mutex> lock(model_->device_mutex(), std::try_to_lock);
        if (lock.owns_lock()) evictable_last_.store(prefix_cache_->evictable_blocks(), std::memory_order_relaxed);
    }
    return kv_->num_free_blocks() + (prefix_cache_ ? evictable_last_.load(std::memory_order_relaxed) : 0);
}

int ContinuousBatchEngine::num_waiting() const {
    std::lock_guard<std::mutex> lock(mu_);
    return (int)waiting_.size();
}

uint64_t ContinuousBatchEngine::admission_waits() const {
    return admission_waits_.load(std::memory_order_relaxed);
}

uint64_t ContinuousBatchEngine::admission_timeouts() const {
    return admission_timeouts_.load(std::memory_order_relaxed);
}

int ContinuousBatchEngine::active_jobs_locked() const {
    int active = 0;
    for (const auto& kv : jobs_) if (!kv.second->done) active++;
    return active;
}

// New arrivals only: a request already waiting is never pushed out by the cap it was admitted under.
bool ContinuousBatchEngine::queue_depth_full_locked() const {
    const int cap = max_queue_depth_config();
    if (cap <= 0) return false;
    return active_jobs_locked() + (int)waiting_.size() >= cap;
}

bool ContinuousBatchEngine::apply_constraint_mask(Job& job) {
    // Far below any real logit, finite so temperature scaling and logsumexp stay finite too.
    static constexpr float kMasked = -1.0e9f;
    const int vocab = model_->config().vocab;
    const int words = (vocab + 31) / 32;
    job.mask_next.assign(words, 0xffffffffu);
    job.req.constraint->fill_next_mask(job.mask_next.data(), vocab);
    if (vocab % 32) job.mask_next[words - 1] &= (1u << (vocab % 32)) - 1;
    bool any = false;
    for (uint32_t w : job.mask_next)
        if (w) { any = true; break; }
    if (!any) return false;
    const bool first = job.mask_bias.empty();
    if (!first && job.mask_next == job.mask_bits) return true;   // already on the device
    if (first) {
        job.mask_bias.assign(vocab, 0.f);
        job.mask_bits.assign(words, 0u);
    }
    // Rebuild only the words that changed: the request's own logit_bias where allowed, kMasked where not.
    std::vector<float> user(0);
    for (int w = 0; w < words; ++w) {
        if (!first && job.mask_next[w] == job.mask_bits[w]) continue;
        const int end = std::min(vocab, (w + 1) * 32);
        for (int id = w * 32; id < end; ++id)
            job.mask_bias[id] = ((job.mask_next[w] >> (id - w * 32)) & 1) ? 0.f : kMasked;
    }
    for (const auto& [id, value] : job.req.logit_bias)
        if (id >= 0 && id < vocab && ((job.mask_next[id / 32] >> (id % 32)) & 1)) job.mask_bias[id] = value;
    job.mask_bits.swap(job.mask_next);
    model_->set_logit_bias_dense(job.seq_id, job.mask_bias.data());
    return true;
}

int ContinuousBatchEngine::max_queue_depth() const { return max_queue_depth_config(); }

void ContinuousBatchEngine::enable_speculative(bool on) {
    std::lock_guard<std::mutex> lock(mu_);
    speculative_ = on;
}

ContinuousBatchEngine::SpecStats ContinuousBatchEngine::speculative_stats() const {
    SpecStats s;
    s.runs = spec_runs_.load(std::memory_order_relaxed);
    s.tokens = spec_tokens_.load(std::memory_order_relaxed);
    s.handoffs = spec_handoffs_.load(std::memory_order_relaxed);
    s.tier_stops = spec_tier_stops_.load(std::memory_order_relaxed);
    return s;
}

namespace {
// SPARKINFER_SPEC_PREFIX_HIT=0: a prefix-cache hit decodes without speculation, and a speculated
// prompt takes no prefix-cache checkpoints (the behaviour before either did).
bool prefix_hit_spec_on() {
    static const bool on = [] {
        const char* e = getenv("SPARKINFER_SPEC_PREFIX_HIT");
        return !(e && e[0] == '0');
    }();
    return on;
}
}  // namespace

// A request already decoding that a speculation group can carry without a draft: its next token is
// ready to emit, and it would speculate if it were fresh (the verify has none of the other sampler
// extras either).
bool ContinuousBatchEngine::spec_adoptable(const Job& j) {
    return !j.done && j.phase == SeqPhase::DECODE && j.next_token >= 0 && spec_eligible(j.req);
}

bool ContinuousBatchEngine::spec_eligible(const Request& r) {
    // A sampled request speculates too: every verify row draws its token with the request's
    // sampler at that token's own step (SpecHooks), so the output is the one ordinary sampled
    // decode gives. SPARKINFER_SPEC_SAMPLED=0 keeps speculation greedy-only. The verify path has
    // none of the other sampler extras (penalties, logit bias, logprobs). A constraint must stay
    // on the per-token path where its mask is applied. Images need the vision splice ordinary
    // prefill does. A prefix-cache hit speculates too: prefill resumes past the cached prefix and
    // the draft reads the target's hidden states from there (SpecHooks::prefill_start).
    static const bool sampled_on = [] {
        const char* e = getenv("SPARKINFER_SPEC_SAMPLED");
        return !(e && e[0] == '0');
    }();
    return !r.constraint && !r.ignore_eos && (r.temperature <= 0.f || sampled_on) && r.presence_penalty == 0.f &&
           r.frequency_penalty == 0.f && r.logit_bias.empty() && !r.logprobs &&
           r.forced_tokens.empty() && r.vision_pos.empty() && !r.use_prefix_session &&
           (r.prefill_start == 0 || prefix_hit_spec_on());
}

namespace {
// SPARKINFER_SPEC_GROUP: how many live requests speculate together, at most 8. Their blocks share
// one verify of kQwen35MaxPackedRows rows, so past four each verifies a shorter block. 1 keeps the
// single-request path (dflash_generate), which stops speculating once a second request arrives.
// Measured on Qwen3.8-27B + DFlash2 (batched draft), real prompts at T=0.7, aggregate tok/s at
// c4/c6/c8: 646/422/650 with 4, 652/768/861 with 8.
int spec_group_max() {
    static const int v = [] {
        const char* e = getenv("SPARKINFER_SPEC_GROUP");
        const int x = e ? atoi(e) : 8;
        return std::max(1, std::min(x, 8));
    }();
    return v;
}
bool spec_group_trace() {
    static const bool v = [] {
        const char* e = getenv("SPARKINFER_SPEC_GROUP_TRACE");
        return e && e[0] == '1';
    }();
    return v;
}
}  // namespace

void ContinuousBatchEngine::run_spec_group() {
    const Qwen35Config& cfg = model_->config();
    const int G = spec_group_max();
    const int depth = model_->spec_group_depth();
    const double timeout_s = request_timeout_s_config();
    struct Member {
        Job* job = nullptr;
        int slot = 0;
        int pos = 0;                 // position of `next`
        int next = -1;               // the verified token at `pos`, not yet ingested
        bool next_emitted = false;   // ...but already handed out (a join's seed)
        std::vector<int> block;      // [next, proposals...]
        bool have_block = false;
        int feed_off = 0, feed_len = 0;   // this member's rows of the last verify's capture
        // A join whose prompt tail rides the next verify (spec_group_join_body): `block` holds the
        // tail, committed whole, and the last row's token is the seed; `first_block` then marks
        // that the slot's first draft block (over the whole prompt) is still to run.
        bool tail_pending = false, first_block = false;
        // Its draft's record, decayed: proposals accepted, and verifies that ended in a rejection.
        // q = (acc + 1) / (acc + rej + 2) estimates the chance its next proposal is accepted.
        float acc = 2.f, rej = 1.f;
        // Past the draft's context (spec_group_reach): it stays in the group without proposals, its
        // block just `next`, so a one-row verify decodes it while the others keep speculating.
        bool no_draft = false;
        bool adopted = false;   // joined while decoding ordinarily (never speculated)
        bool done = false;
    };
    std::vector<Member> members;
    std::vector<bool> slot_used((size_t)G, false);
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& kv : jobs_)
            if (!kv.second->done) kv.second->spec_tried = true;
    }
    if (!model_->spec_group_begin()) return;
    // Requests already decoding join as members without a draft: their next token, not yet emitted,
    // is the block, and each step's verify ingests it at prompt + decode_emitted and draws the one
    // after with step decode_emitted + 1 -- the position and sampler step of the ordinary decode
    // step (step_job). The arithmetic differs as a group's does: the grouped verify's kernels, and
    // the GDN state in fp32 where packed decode keeps rounding it to bf16. At the group's end,
    // handoff gives them back as they were. The fresh prompts join one a step below and speculate.
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto& kv : jobs_) {
            Job* j = kv.second.get();
            if (!spec_adoptable(*j)) continue;
            // Not adopted if its state cannot take the verify's form: it then counts as waiting and
            // not fresh, and the group ends at its first join check, as before adoption existed.
            if (!model_->spec_adopt_session(j->seq_id)) continue;
            Member m;
            m.job = j;
            m.slot = -1;
            m.pos = (int)j->req.prompt.size() + j->decode_emitted;
            m.next = j->next_token;
            m.next_emitted = false;
            m.no_draft = true;
            m.adopted = true;
            members.push_back(m);
        }
    }
    if (spec_group_trace())
        fprintf(stderr, "[spec-group] start (group %d, depth %d, %zu adopted)\n", G, depth, members.size());
    int trace_steps = 0, trace_rows = 0, trace_kept = 0;
    double trace_draft_ms = 0, trace_join_ms = 0, trace_verify_ms = 0;
    auto ms_since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    };
    auto idx_of_next = [](const Member& m) {
        return (unsigned long long)(m.job->decode_emitted - (m.next_emitted ? 1 : 0));
    };
    auto finish = [&](Member& m) {
        Job& job = *m.job;
        job.generation_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - job.t_submit).count();
        if (job.saw_first_tok && job.generation_ms > job.ttft_ms && job.decode_emitted > 0) {
            const double decode_ms = std::max(job.generation_ms - job.ttft_ms, 1.0);
            job.decode_tps = (double)job.decode_emitted * 1000.0 / decode_ms;
        }
        finish_job_impl(job);
        if (m.slot >= 0) slot_used[(size_t)m.slot] = false;   // an adopted member has none
        m.done = true;
    };
    // Hand one token to the caller as run_speculative does. False when the request stops here
    // (callback, timeout, EOS, its token limit) -- the member is finished.
    auto emit = [&](Member& m, int tok) -> bool {
        Job& job = *m.job;
        const auto t = std::chrono::steady_clock::now();
        if (!job.saw_first_tok) {
            job.t_first = t;
            job.saw_first_tok = true;
            job.ttft_ms = std::chrono::duration<double, std::milli>(job.t_first - job.t_submit).count();
        }
        job.output.push_back(tok);
        job.decode_emitted++;
        if (job.on_token && !job.on_token(tok)) { job.cancelled = true; finish(m); return false; }
        if (timeout_s > 0.0 &&
            std::chrono::duration<double>(t - job.t_submit).count() > timeout_s) {
            job.error = "request timeout";
            job.timed_out = true;
            finish(m);
            return false;
        }
        const bool eos = tok == cfg.eos_id || (cfg.eos_id2 >= 0 && tok == cfg.eos_id2);
        const bool limit = job.decode_emitted >= job.req.max_new_tokens;
        if (eos || limit) {
            job.reached_token_limit = limit && !eos;
            finish(m);
            return false;
        }
        return true;
    };
    // Back to ordinary decode at the committed position, as run_speculative hands over: the next
    // step_job emits next_token and ingests it at prompt_len + decode_emitted.
    auto handoff = [&](Member& m) {
        Job& job = *m.job;
        if (m.next_emitted) {
            // The seed went out but was never ingested: ingest it as the next decode step would.
            // (finish below takes mu_, so not under the device lock: admission takes them the
            // other way round.)
            int after = -1;
            {
                std::lock_guard<std::recursive_mutex> device_lock(model_->device_mutex());
                model_->activate_session(job.seq_id);
                after = model_->forward_token(m.next, m.pos, true, job.req.temperature, job.req.seed,
                                              idx_of_next(m) + 1, job.req.top_k, job.req.top_p);
            }
            if (after < 0 || after >= cfg.vocab) {
                job.error = "speculative decode failed; the request was aborted";
                job.internal_error = true;
                finish(m);
                return;
            }
            m.next = after;
            m.pos += 1;
        }
        job.prefill_pos = (int)job.req.prompt.size();
        job.phase = SeqPhase::DECODE;
        job.next_token = m.next;
        if (!m.adopted) spec_handoffs_.fetch_add(1, std::memory_order_relaxed);
    };
    auto live_members = [&] {
        int n = 0;
        for (const Member& m : members) n += !m.done;
        return n;
    };
    // A lone member's verify (8-row equivalent) and draft times, for its depth choice.
    float single_v8 = 14.f, single_draft = 1.8f;
    bool leave = false;
    const int reach = model_->spec_group_reach();
    for (;;) {
        // 0. A member that has reached the end of the draft's context stops drafting: its next block
        //    would draft and capture past it. It stays in the group and verifies one row a step
        //    (lossless, as every verify is), so the others keep speculating. Ending the group for it
        //    stopped speculation for every member, and for the load, as soon as one answer passed
        //    16K tokens (a thinking model without max_tokens).
        for (Member& m : members) {
            if (m.done || m.no_draft || m.pos <= reach) continue;
            m.no_draft = true;
            if (spec_group_trace())
                fprintf(stderr, "[spec-group] a member reached the draft's context (%d): it drafts no more\n", m.pos);
        }
        for (Member& m : members)
            if (!m.done && m.no_draft && !m.have_block) {
                m.block.assign(1, m.next);
                m.have_block = true;
            }
        // 1. Drafts, from the hidden rows each member's last verify captured (consumed before a
        //    join below re-arms the capture buffers).
        auto t_draft = std::chrono::steady_clock::now();
        for (Member& m : members) {
            if (m.done || !m.first_block || leave) continue;
            const Job& job = *m.job;
            std::vector<int> prop((size_t)depth, -1);
            const char* hid = static_cast<const char*>(model_->dflash_hidden_buffer()) +
                              (size_t)m.feed_off * model_->dflash_hidden_row_stride() * 2;
            if (!model_->spec_group_join_finish(m.slot, m.pos, m.feed_len, hid, m.next, job.req.temperature,
                                                (unsigned long long)job.req.seed, job.req.top_k, job.req.top_p,
                                                prop.data())) {
                leave = true;
                break;
            }
            m.first_block = false;
            m.block.assign(1, m.next);
            m.block.insert(m.block.end(), prop.begin(), prop.end());
            m.have_block = true;
        }
        if (!leave) {
            // Every member that needs a block, drafted in one pass where the draft can
            // (spec_group_draft_multi); the loop below takes whatever is left.
            std::vector<Member*> need;
            for (Member& m : members)
                if (!m.done && !m.have_block) need.push_back(&m);
            if (need.size() >= 2) {
                const size_t n = need.size();
                std::vector<int> slots(n), th(n), seeds(n), pos(n), topk(n), prop(n * (size_t)depth, -1);
                std::vector<const void*> hid(n);
                std::vector<float> temp(n), topp(n);
                std::vector<unsigned long long> srng(n), step(n);
                const char* base = static_cast<const char*>(model_->dflash_hidden_buffer());
                const size_t row_bytes = (size_t)model_->dflash_hidden_row_stride() * 2;
                for (size_t i = 0; i < n; ++i) {
                    const Member& m = *need[i];
                    slots[i] = m.slot;
                    hid[i] = base + (size_t)m.feed_off * row_bytes;
                    th[i] = m.feed_len;
                    seeds[i] = m.next;
                    pos[i] = m.pos;
                    temp[i] = m.job->req.temperature;
                    srng[i] = (unsigned long long)m.job->req.seed;
                    step[i] = idx_of_next(m) + 1;
                    topk[i] = m.job->req.top_k;
                    topp[i] = m.job->req.top_p;
                }
                if (model_->spec_group_draft_multi((int)n, slots.data(), hid.data(), th.data(), seeds.data(),
                                                   pos.data(), temp.data(), srng.data(), step.data(),
                                                   topk.data(), topp.data(), prop.data())) {
                    for (size_t i = 0; i < n; ++i) {
                        Member& m = *need[i];
                        m.block.assign(1, m.next);
                        m.block.insert(m.block.end(), prop.begin() + (long)(i * depth),
                                       prop.begin() + (long)((i + 1) * depth));
                        m.have_block = true;
                    }
                }
            }
        }
        for (Member& m : members) {
            if (leave) break;
            if (m.done || m.have_block) continue;
            const Job& job = *m.job;
            std::vector<int> prop((size_t)depth, -1);
            const char* hid = static_cast<const char*>(model_->dflash_hidden_buffer()) +
                              (size_t)m.feed_off * model_->dflash_hidden_row_stride() * 2;
            if (!model_->spec_group_draft(m.slot, hid, m.feed_len, m.next, m.pos, job.req.temperature,
                                          job.req.seed, idx_of_next(m) + 1, job.req.top_k,
                                          job.req.top_p, prop.data())) {
                leave = true;
                break;
            }
            m.block.assign(1, m.next);
            m.block.insert(m.block.end(), prop.begin(), prop.end());
            m.have_block = true;
        }
        trace_draft_ms += ms_since(t_draft);
        if (members.size() == 1 && !members[0].done) single_draft = 0.8f * single_draft + 0.2f * (float)ms_since(t_draft);
        // 2. One arrival joins per step, if the group can take it; anything else ends the group.
        Job* joiner = nullptr;
        auto t_join = std::chrono::steady_clock::now();
        if (!leave) {
            std::lock_guard<std::mutex> lock(mu_);
            // Count everything waiting first. Each check below only asks whether ONE more fits, so
            // 15 prompts arriving behind one member all "fit" and joined one at a time -- an 8K
            // join is a 0.6 s speculative prefill with its own draft slot and capture -- until the
            // group was full, and then the next one ended it anyway. Measured: AIPerf 8K prompts
            // at 16 concurrent, 4-5 joins per burst, the 5th out of memory and on the token loop
            // (83 s), 38 tok/s against 167 without a draft. When they cannot all join, none should.
            int waiting = 0;
            for (auto& kv : jobs_) {
                const Job* j = kv.second.get();
                if (j->done) continue;
                bool member = false;
                for (const Member& m : members) member = member || (!m.done && m.job == j);
                if (!member) ++waiting;
            }
            if (live_members() + waiting > G) {
                if (spec_group_trace())
                    fprintf(stderr, "[spec-group] leave: %d waiting, %d members, group of %d\n",
                            waiting, live_members(), G);
                leave = true;
            }
            for (auto& kv : jobs_) {
                if (leave) break;
                Job* j = kv.second.get();
                if (j->done) continue;
                bool member = false;
                for (const Member& m : members) member = member || (!m.done && m.job == j);
                if (member) continue;
                const bool fits = j->phase == SeqPhase::PREFILL && j->prefill_pos == j->req.prefill_start &&
                                  spec_eligible(j->req) && live_members() + (joiner ? 2 : 1) <= G;
                if (fits && joiner) continue;   // one join per step: it joins on the next one
                if (!fits) {
                    if (spec_group_trace())
                        fprintf(stderr, "[spec-group] leave: request %llu cannot join "
                                        "(phase %d, eligible %d, members %d)\n",
                                (unsigned long long)kv.first, (int)j->phase, (int)spec_eligible(j->req),
                                live_members());
                    leave = true;
                    break;
                }
                joiner = j;
            }
            if (joiner) joiner->spec_tried = true;
        }
        if (!leave && joiner) {
            // The join grows the session past its admission budget by a block or two (the verify's
            // lookahead): make sure a pool the prefix cache has filled has them.
            if (prefix_cache_) {
                std::lock_guard<std::recursive_mutex> device_lock(model_->device_mutex());
                prefix_cache_->evict_for(4);
            }
            int slot = 0;
            while (slot < G && slot_used[(size_t)slot]) ++slot;
            Qwen35Model::SpecHooks hooks;
            hooks.seq_id = joiner->seq_id;
            hooks.temperature = joiner->req.temperature;
            hooks.seed = joiner->req.seed;
            hooks.top_k = joiner->req.top_k;
            hooks.top_p = joiner->req.top_p;
            hooks.prefill_start = joiner->req.prefill_start;
            const int n = (int)joiner->req.prompt.size();
            std::vector<int> ckpts;
            std::vector<Qwen35Model::RecurrentStateSnapshot> snaps;
            if (prefix_hit_spec_on() && prefix_cache_ && joiner->req.prefix_cache) {
                for (int ck : joiner->req.cache_checkpoints)
                    if (ck > joiner->req.prefill_start && ck < n && ck % kv_->block_size() == 0)
                        ckpts.push_back(ck);
                std::sort(ckpts.begin(), ckpts.end());
                ckpts.erase(std::unique(ckpts.begin(), ckpts.end()), ckpts.end());
                snaps.resize(ckpts.size());
                hooks.ckpts = ckpts.data();
                hooks.n_ckpts = (int)ckpts.size();
                hooks.snaps = snaps.data();
            }
            // With others speculating, a prompt ending in a partial group of eight prefills only its
            // aligned body here and verifies the rest in the group's next verify: one forward
            // fewer than prefilling the tail on its own. SPARKINFER_SPEC_JOIN_TAIL=0 keeps the
            // whole prefill here.
            static const bool join_tail = [] {
                const char* e = getenv("SPARKINFER_SPEC_JOIN_TAIL");
                return !(e && e[0] == '0');
            }();
            int body = 0;
            if (join_tail && live_members() >= 1 && ckpts.empty() &&
                model_->spec_group_join_body(joiner->req.prompt, joiner->req.max_new_tokens, slot, hooks,
                                             &body) == 0) {
                Member tm;
                tm.job = joiner;
                tm.slot = slot;
                tm.pos = body;
                tm.block.assign(joiner->req.prompt.begin() + body, joiner->req.prompt.end());
                tm.have_block = true;
                tm.tail_pending = true;
                slot_used[(size_t)slot] = true;
                members.push_back(std::move(tm));
                if (spec_group_trace())
                    fprintf(stderr, "[spec-group] join: %d-token prompt, body %d, tail in the verify, slot %d\n", n,
                            body, slot);
            } else {
                Qwen35Model::SpecResume r;
                std::vector<int> prop((size_t)depth, -1);
                Member nm;
                nm.job = joiner;
                nm.slot = slot;
                nm.pos = n;
                slot_used[(size_t)slot] = true;
                bool seed_out = false;
                // The seed goes out the moment prefill has drawn it, ahead of the first draft block
                // (which reads the whole prompt context), as dflash_generate hands its seed over.
                hooks.on_tokens = [&](const int* toks, int) -> bool {
                    if (r.ckpts_taken)
                        for (size_t i = 0; i < ckpts.size(); ++i) {
                            Job::Checkpoint cp;
                            cp.pos = ckpts[i];
                            cp.state = std::move(snaps[i]);
                            joiner->checkpoints.push_back(std::move(cp));
                        }
                    joiner->prefill_pos = n;
                    joiner->phase = SeqPhase::DECODE;
                    spec_runs_.fetch_add(1, std::memory_order_relaxed);
                    seed_out = true;
                    nm.next = toks[0];
                    if (!emit(nm, toks[0])) return false;
                    nm.next_emitted = true;
                    return true;
                };
                const int seed = model_->spec_group_join(joiner->req.prompt, joiner->req.max_new_tokens,
                                                         slot, hooks, &r, prop.data());
                if (spec_group_trace())
                    fprintf(stderr, "[spec-group] join: %zu-token prompt, slot %d, seed %d\n",
                            joiner->req.prompt.size(), slot, seed);
                if (nm.done) {
                    // Finished at its seed (EOS, a one-token limit, a cancel): nothing more to do.
                } else if (seed_out && seed >= 0) {
                    nm.block.assign(1, nm.next);
                    nm.block.insert(nm.block.end(), prop.begin(), prop.end());
                    nm.have_block = true;
                    members.push_back(std::move(nm));
                } else if (seed_out) {
                    // Prefilled and its seed handed out, but the first draft block failed: the member
                    // goes to ordinary decode with the others below.
                    members.push_back(std::move(nm));
                    leave = true;
                } else {
                    slot_used[(size_t)slot] = false;
                    if (r.failed) {
                        joiner->error = "speculative prefill failed; the request was aborted";
                        joiner->internal_error = true;
                        finish_job_impl(*joiner);
                    }
                    leave = true;   // otherwise nothing ran: the ordinary path prefills it
                }
            }
        }
        if (joiner) trace_join_ms += ms_since(t_join);
        // Nothing left to speculate (every member drafts no more and nobody joined): ordinary packed
        // decode carries those requests faster than one-row verifies.
        if (!leave && !joiner && live_members() > 0) {
            bool any_draft = false;
            for (const Member& m : members) any_draft = any_draft || (!m.done && !m.no_draft);
            if (!any_draft) {
                if (spec_group_trace()) fprintf(stderr, "[spec-group] leave: no member drafts\n");
                leave = true;
            }
        }
        if (leave) break;
        // 3. One verify for every member's block.
        std::vector<Member*> act;
        for (Member& m : members) if (!m.done && m.have_block) act.push_back(&m);
        if (act.empty()) {
            if (live_members() == 0) {
                std::lock_guard<std::mutex> lock(mu_);
                bool more = false;
                for (auto& kv : jobs_) more = more || !kv.second->done;
                if (!more) break;    // nothing left to speculate
                continue;            // a new arrival joins on the next pass
            }
            continue;
        }
        const int n = (int)act.size();
        std::vector<uint64_t> seqs((size_t)n);
        std::vector<const int*> blocks((size_t)n);
        std::vector<int> lens((size_t)n), starts((size_t)n), keep((size_t)n, 0);
        std::vector<float> temp((size_t)n), top_p((size_t)n);
        std::vector<unsigned long long> seedv((size_t)n), step((size_t)n);
        std::vector<int> top_k((size_t)n);
        bool any_sampled = false;
        int rows = 0;
        // The verify takes kQwen35MaxPackedRows rows: past four members each verifies a shorter
        // block (the draft still proposes its full depth; the rest is not scored). A joiner's
        // prompt tail (at most 7 rows) is verified whole, committed, and the others share the rest.
        std::unique_ptr<bool[]> commit(new bool[(size_t)n]());
        bool any_commit = false;
        // Members without a draft take one row each and are not counted among the sharers: seven
        // adopted beside one drafter would otherwise have cut its block to four rows.
        int tail_rows = 0, n_spec = 0, one_rows = 0;
        for (int g = 0; g < n; ++g) {
            const Member& m = *act[(size_t)g];
            if (m.tail_pending) tail_rows += (int)m.block.size();
            else if (m.no_draft) ++one_rows;
            else ++n_spec;
        }
        const int row_cap = std::max(2, (kQwen35MaxPackedRows - tail_rows - one_rows) / std::max(n_spec, 1));
        // A request speculating alone verifies the depth that makes the most tokens per ms. Its
        // single-sequence verify costs, relative to 8 rows, about kRel[L] (measured at ~1K context:
        // 10.45 / 10.36 / 10.89 / 11.20 / 12.46 / 13.76 / 13.18 / 14.11 ms for 1..8 rows), scaled by
        // what this run's verifies take (v8, tracked), plus the draft (d, tracked); row L is kept with
        // probability ~q^L. A predictable stream keeps its whole block; a hard one stops paying for
        // rows it would reject (14.5K-token prose: 97 -> 119 tok/s). In a group, rows are nearly free
        // (a 4 x 8 verify costs 15.2 ms against 12.4 for 1 x 8), so members keep their blocks there.
        // SPARKINFER_SPEC_ADAPTIVE_ROWS=0 always verifies the whole block.
        static const bool adaptive_rows = [] {
            const char* e = getenv("SPARKINFER_SPEC_ADAPTIVE_ROWS");
            return !(e && e[0] == '0');
        }();
        static constexpr float kRel[9] = {0.f, 0.74f, 0.73f, 0.77f, 0.79f, 0.88f, 0.98f, 0.93f, 1.f};
        std::vector<int> want((size_t)n);
        for (int g = 0; g < n; ++g) {
            const Member& m = *act[(size_t)g];
            want[(size_t)g] = m.tail_pending ? (int)m.block.size() : std::min((int)m.block.size(), row_cap);
        }
        if (adaptive_rows && n == 1 && !act[0]->tail_pending && want[0] > 2 && want[0] <= 8) {
            const Member& m = *act[0];
            const float q = (m.acc + 1.f) / (m.acc + m.rej + 2.f);
            int best = want[0];
            float best_rate = 0.f, e = 0.f, qp = 1.f;
            for (int L = 1; L <= want[0]; ++L) {
                e += qp;   // expected tokens kept from L rows: 1 + q + ... + q^(L-1)
                qp *= q;
                if (L < 2) continue;
                const float rate = e / (single_v8 * kRel[L] + single_draft);
                if (rate > best_rate) { best_rate = rate; best = L; }
            }
            want[0] = best;
        }
        for (int g = 0; g < n; ++g) {
            Member& m = *act[(size_t)g];
            seqs[(size_t)g] = m.job->seq_id;
            blocks[(size_t)g] = m.block.data();
            lens[(size_t)g] = want[(size_t)g];
            commit[(size_t)g] = m.tail_pending;
            any_commit = any_commit || m.tail_pending;
            starts[(size_t)g] = m.pos;
            temp[(size_t)g] = m.job->req.temperature;
            seedv[(size_t)g] = (unsigned long long)m.job->req.seed;
            // A tail's last row draws the seed, which prefill samples at step 0 (row i is drawn at
            // step[g] + i; unsigned wrap is the intent).
            step[(size_t)g] = m.tail_pending ? (unsigned long long)(1 - lens[(size_t)g]) : idx_of_next(m) + 1;
            top_k[(size_t)g] = m.job->req.top_k;
            top_p[(size_t)g] = m.job->req.top_p;
            any_sampled = any_sampled || temp[(size_t)g] > 0.f;
            rows += lens[(size_t)g];
        }
        Qwen35Model::PackedSampling samp;
        samp.temperature = temp.data();
        samp.seed = seedv.data();
        samp.step = step.data();
        samp.top_k = top_k.data();
        samp.top_p = top_p.data();
        std::vector<int> out((size_t)rows, -1);
        auto t_verify = std::chrono::steady_clock::now();
        if (!model_->spec_group_verify(n, seqs.data(), blocks.data(), lens.data(), starts.data(),
                                       any_sampled ? &samp : nullptr, out.data(), keep.data(),
                                       any_commit ? commit.get() : nullptr)) {
            // A verify that fails has committed nothing. An adopted member is still exactly where its
            // ordinary decode left it, so it goes back to that decode with the rest (leave, below)
            // instead of failing a request that was never speculating.
            for (Member* pm : act) {
                if (pm->adopted && !pm->next_emitted) continue;
                pm->job->error = "speculative decode failed; the request was aborted";
                pm->job->internal_error = true;
                finish(*pm);
            }
            leave = true;
            break;
        }
        trace_verify_ms += ms_since(t_verify);
        if (n == 1 && lens[0] >= 1 && lens[0] <= 8)
            single_v8 = 0.8f * single_v8 + 0.2f * (float)ms_since(t_verify) / kRel[lens[0]];
        ++trace_steps;
        trace_rows += rows;
        for (int g = 0; g < n; ++g) trace_kept += keep[(size_t)g];
        // 4. Each member's accepted prefix, and its bonus token as the next seed.
        for (int g = 0, off = 0; g < n; off += lens[(size_t)g], ++g) {
            Member& m = *act[(size_t)g];
            const int k = keep[(size_t)g];
            if (m.tail_pending) {
                // The prompt is in: its last row's token is the seed, emitted now, and the slot's
                // first draft block runs at the next step's draft, from these rows.
                Job& job = *m.job;
                m.tail_pending = false;
                m.pos += k;
                m.next = out[(size_t)(off + k - 1)];
                m.feed_off = off;
                m.feed_len = k;
                m.have_block = false;
                job.prefill_pos = (int)job.req.prompt.size();
                job.phase = SeqPhase::DECODE;
                spec_runs_.fetch_add(1, std::memory_order_relaxed);
                if (emit(m, m.next)) {
                    m.next_emitted = true;
                    m.first_block = true;
                }
                continue;
            }
            // The draft's record: k - 1 of its lens - 1 scored proposals accepted, and a rejection
            // unless every one was. A one-row verify scored none and says nothing.
            if (lens[(size_t)g] > 1) {
                m.acc = 0.9f * m.acc + (float)(k - 1);
                m.rej = 0.9f * m.rej + (k < lens[(size_t)g] ? 1.f : 0.f);
            }
            bool going = true;
            for (int i = (m.next_emitted ? 1 : 0); i < k && going; ++i) going = emit(m, m.block[(size_t)i]);
            if (!going) continue;
            m.next_emitted = false;
            m.next = out[(size_t)(off + k - 1)];
            m.pos += k;
            m.have_block = false;
            m.feed_off = off;
            m.feed_len = k;
            if (!m.adopted) spec_tokens_.fetch_add((uint64_t)k, std::memory_order_relaxed);
            // A bonus EOS ends the request here, as dflash_generate does.
            if (m.next == cfg.eos_id || (cfg.eos_id2 >= 0 && m.next == cfg.eos_id2))
                if (emit(m, m.next)) m.next_emitted = true;
        }
    }
    int handed = 0;
    for (Member& m : members)
        if (!m.done) { handoff(m); ++handed; }
    model_->spec_group_end();
    if (spec_group_trace())
        fprintf(stderr, "[spec-group] end: %zu members, %d handed off, %d verifies, %d rows, %d kept; "
                        "draft %.1f ms, join %.1f ms, verify %.1f ms\n",
                members.size(), handed, trace_steps, trace_rows, trace_kept, trace_draft_ms, trace_join_ms,
                trace_verify_ms);
}

void ContinuousBatchEngine::run_speculative(Job& job) {
    job.spec_tried = true;
    const Qwen35Config& cfg = model_->config();
    const int prompt_len = (int)job.req.prompt.size();
    const double timeout_s = request_timeout_s_config();
    // spec_running_ was raised, and spec_interrupt_ cleared, under mu_ when this job was picked.
    {
        std::lock_guard<std::recursive_mutex> device_lock(model_->device_mutex());
        model_->activate_session(job.seq_id);
        model_->reset_mrope_offset();
    }

    Qwen35Model::SpecHooks hooks;
    hooks.seq_id = job.seq_id;
    hooks.temperature = job.req.temperature;
    hooks.seed = job.req.seed;
    hooks.top_k = job.req.top_k;
    hooks.top_p = job.req.top_p;
    hooks.on_tokens = [&](const int* tokens, int n) -> bool {
        for (int i = 0; i < n; i++) {
            const auto t_emit = std::chrono::steady_clock::now();
            if (!job.saw_first_tok) {
                job.t_first = t_emit;
                job.saw_first_tok = true;
                job.ttft_ms = std::chrono::duration<double, std::milli>(job.t_first - job.t_submit).count();
            }
            job.output.push_back(tokens[i]);
            job.decode_emitted++;
            if (job.on_token && !job.on_token(tokens[i])) {
                job.cancelled = true;
                return false;
            }
        }
        if (timeout_s > 0.0) {
            const double elapsed_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - job.t_submit).count();
            if (elapsed_s > timeout_s) {
                job.error = "request timeout after " + std::to_string(elapsed_s) + "s (limit " +
                            std::to_string(timeout_s) + "s)";
                job.timed_out = true;
                return false;
            }
        }
        return !spec_interrupt_.load(std::memory_order_relaxed);
    };
    hooks.prefill_start = job.req.prefill_start;
    // The prompt's prefix-cache checkpoints, taken during the speculative prefill exactly as the
    // ordinary prefill takes them, so the next turn of a conversation finds its prefix cached.
    std::vector<int> ckpts;
    std::vector<Qwen35Model::RecurrentStateSnapshot> snaps;
    if (prefix_hit_spec_on() && prefix_cache_ && job.req.prefix_cache) {
        for (int ckpt : job.req.cache_checkpoints)
            if (ckpt > job.req.prefill_start && ckpt < prompt_len && ckpt % kv_->block_size() == 0)
                ckpts.push_back(ckpt);
        std::sort(ckpts.begin(), ckpts.end());
        ckpts.erase(std::unique(ckpts.begin(), ckpts.end()), ckpts.end());
        snaps.resize(ckpts.size());
        hooks.ckpts = ckpts.data();
        hooks.n_ckpts = (int)ckpts.size();
        hooks.snaps = snaps.data();
    }
    Qwen35Model::SpecResume r;
    if (prefix_cache_) {   // its verify lookahead grows the session past the admission budget
        std::lock_guard<std::recursive_mutex> device_lock(model_->device_mutex());
        prefix_cache_->evict_for(4);
    }
    model_->dflash_generate(job.req.prompt, job.req.max_new_tokens, nullptr, nullptr, &hooks, &r);
    if (r.ckpts_taken) {
        for (size_t i = 0; i < ckpts.size(); ++i) {
            Job::Checkpoint cp;
            cp.pos = ckpts[i];
            cp.state = std::move(snaps[i]);
            job.checkpoints.push_back(std::move(cp));
        }
    }
    spec_running_.store(false, std::memory_order_relaxed);
    if (!r.engaged) return;   // nothing ran: ordinary prefill picks the job up on the next iteration
    spec_runs_.fetch_add(1, std::memory_order_relaxed);
    spec_tokens_.fetch_add((uint64_t)job.decode_emitted, std::memory_order_relaxed);

    job.prefill_pos = prompt_len;
    job.phase = SeqPhase::DECODE;
    auto finish = [&] {
        job.generation_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - job.t_submit).count();
        if (job.saw_first_tok && job.generation_ms > job.ttft_ms && job.decode_emitted > 0) {
            const double decode_ms = std::max(job.generation_ms - job.ttft_ms, 1.0);
            job.decode_tps = (double)job.decode_emitted * 1000.0 / decode_ms;
        }
        finish_job_impl(job);
    };
    if (job.cancelled || job.timed_out) { finish(); return; }
    const int last = job.output.empty() ? -1 : job.output.back();
    const bool hit_eos = last >= 0 && (last == cfg.eos_id || (cfg.eos_id2 >= 0 && last == cfg.eos_id2));
    const bool hit_limit = job.decode_emitted >= job.req.max_new_tokens;
    if (r.failed || r.emitted != job.decode_emitted ||
        (!r.finished && !hit_eos && !hit_limit && r.position != prompt_len + job.decode_emitted)) {
        job.error = "speculative decode failed; the request was aborted";
        job.internal_error = true;
        finish();
        return;
    }
    if (r.finished || hit_eos || hit_limit) {
        job.reached_token_limit = hit_limit && !hit_eos;
        finish();
        return;
    }
    // Another request arrived: continue as ordinary decode from the committed position. step_job
    // emits next_token and ingests it at prompt_len + decode_emitted, which is r.position.
    // Another request arrived, or the next step would have crossed a KV split tier: continue as
    // ordinary decode from the committed position.
    (r.tier_boundary ? spec_tier_stops_ : spec_handoffs_).fetch_add(1, std::memory_order_relaxed);
    job.next_token = r.next_token;
}

void ContinuousBatchEngine::enable_prefix_cache(const PrefixCache::Limits& limits) {
    std::lock_guard<std::recursive_mutex> device_lock(model_->device_mutex());
    if (!prefix_cache_) prefix_cache_ = std::make_unique<PrefixCache>(kv_, limits);
    model_->warm_snapshot_pool();   // a burst's checkpoint snapshots must not pin memory mid-step
}

void ContinuousBatchEngine::disable_prefix_cache() {
    std::lock_guard<std::recursive_mutex> device_lock(model_->device_mutex());
    prefix_cache_.reset();
    evictable_last_.store(0, std::memory_order_relaxed);
}

PrefixCache::Stats ContinuousBatchEngine::prefix_cache_stats() const {
    return prefix_cache_ ? prefix_cache_->stats() : PrefixCache::Stats{};
}

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
    // Every id the forward will embed has to be a row of the embedding table. One that is not
    // reads past it on the device -- an illegal address that loses the CUDA context and takes
    // the whole server down for every client, where it should be one 400. Reachable from a
    // server run with a tokenizer from another model (Qwen3.8's 248K ids against Muse Glimmer's
    // 202K rows crashed on the first request) and from any embedding caller passing raw ids.
    {
        const int vocab = model_->config().vocab;
        auto bad = [vocab](const std::vector<int>& ids) {
            for (int t : ids) if (t < 0 || t >= vocab) return true;
            return false;
        };
        if (vocab > 0 && (bad(job.req.prompt) || bad(job.req.forced_tokens)))
            return fail(EnqueueError::BAD_REQUEST);
    }

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
            // The prefix cache may be holding the room this turn needs: give it back first, as the
            // automatic-cache branch below does.
            if (!kv_->allocate(seq_id, budget) &&
                !(prefix_cache_ && prefix_cache_->evict_for((budget + kv_->block_size() - 1) / kv_->block_size()) &&
                  kv_->allocate(seq_id, budget)))
                return fail(EnqueueError::OVERLOADED);
            model_->activate_session(seq_id);
            // The KV blocks survived the previous request, but its decoding advanced the hybrid
            // recurrent state past the prefix. Replay the end-of-prefix snapshot so the 48
            // Gated-DeltaNet layers start where the prefix ended rather than carrying the last
            // request's history -- silently wrong output otherwise, not a crash.
            model_->restore_prefix_state();
            model_->reset_penalty_counts(seq_id);   // session 0 is shared across unrelated requests
            model_->set_logit_bias(seq_id, job.req.logit_bias);   // same reason
        } else {
            // Automatic prefix cache: start from the longest cached prefix of this prompt, sharing
            // its KV blocks and restoring its recurrent state, so prefill covers only the rest.
            // A pool with windowed (ring) KV slices cannot lend its blocks to another sequence:
            // a ring slot is private, so a shared prefix block does not name the borrower's
            // window. allocate_with_prefix() refuses, and a refusal here would read as
            // "pool full" and fail the request -- so skip the lookup instead and prefill the
            // prompt in full, the same trade the recurrent-state mismatch below already takes.
            const bool cache_eligible = prefix_cache_ && job.req.prefix_cache &&
                                        kv_->prefix_sharing_supported() &&
                                        job.req.forced_tokens.empty() && job.req.vision_pos.empty();
            PrefixCache::Hit hit;
            if (cache_eligible) hit = prefix_cache_->lookup(job.req.prompt);
            bool alloc_failed = false;
            // A hit whose KV is on the host tier opens a plain session; the cache copies the KV in.
            auto open = [&](const PrefixCache::Hit& h) {
                return model_->open_session(budget, &alloc_failed,
                                            h.tokens > 0 && !h.on_host ? &h.blocks : nullptr);
            };
            seq_id = open(hit);
            if (!seq_id && !alloc_failed && prefix_cache_) {
                // The pool is full, possibly with blocks only the cache still holds. Evict and try
                // once more. The lookup is redone: eviction may have released the hit itself.
                const int bs = kv_->block_size();
                if (prefix_cache_->evict_for((budget + bs - 1) / bs)) {
                    hit = cache_eligible ? prefix_cache_->lookup(job.req.prompt) : PrefixCache::Hit{};
                    seq_id = open(hit);
                }
            }
            if (!seq_id) return fail(alloc_failed ? EnqueueError::ALLOC_FAILED : EnqueueError::OVERLOADED);
            // The copy did not happen (the entry went, or the transfer failed): the session's blocks
            // are its own, and the whole prompt is prefilled into them.
            if (hit.tokens > 0 && hit.on_host && !prefix_cache_->restore_host_hit(hit, seq_id))
                hit = PrefixCache::Hit{};
            if (hit.tokens > 0) {
                if (model_->restore_recurrent_state(seq_id, hit.state)) {
                    job.req.prefill_start = hit.tokens;
                    job.cached_tokens = hit.tokens;
                } else {
                    // Shared KV with anything but its own recurrent state is wrong output, not a
                    // slow path. Fall back to a plain session and recompute the whole prompt.
                    model_->close_session(seq_id);
                    alloc_failed = false;
                    seq_id = model_->open_session(budget, &alloc_failed);
                    if (!seq_id) return fail(alloc_failed ? EnqueueError::ALLOC_FAILED : EnqueueError::OVERLOADED);
                }
            }
            model_->reset_penalty_counts(seq_id);   // explicit, not relying on open_session's internal zero
            model_->set_logit_bias(seq_id, job.req.logit_bias);    // same reason
        }
        // The first token comes out of prefill, so its mask must be in place before prefill runs.
        if (job.req.constraint) {
            job.seq_id = seq_id;
            if (!apply_constraint_mask(job)) {
                if (seq_id != 0) model_->close_session(seq_id);   // session 0 is the shared prefix
                else kv_->free(seq_id);
                return fail(EnqueueError::BAD_REQUEST);
            }
        }
    }

    job.request_id = next_req_id_.fetch_add(1);
    job.seq_id = seq_id;
    static const bool async_emit = [] {
        const char* e = getenv("SPARKINFER_ASYNC_EMIT");
        return !(e && e[0] == '0');
    }();
    if (async_emit && on_token) {
        auto q = std::make_shared<Job::EmitQueue>();
        q->user_token = on_token;
        q->user_logprob = on_token_logprob;
        job.emit = q;
        job.on_token = [q](int t) {
            std::lock_guard<std::mutex> g(q->mu);
            if (q->stop) return false;
            Job::EmitQueue::Event ev;
            ev.token = t;
            q->events.push_back(std::move(ev));
            q->cv.notify_one();
            return true;
        };
        if (on_token_logprob)
            job.on_token_logprob = [q](const Qwen35Model::TokenLogprob& lp) {
                std::lock_guard<std::mutex> g(q->mu);
                if (q->stop) return;
                Job::EmitQueue::Event ev;
                ev.is_logprob = true;
                ev.lp = lp;
                q->events.push_back(std::move(ev));
                q->cv.notify_one();
            };
    } else {
        job.on_token = on_token;
        job.on_token_logprob = on_token_logprob;
    }
    job.prefill_pos = job.req.prefill_start;
    job.t_submit = std::chrono::steady_clock::now();
    auto ptr = std::make_unique<Job>(std::move(job));
    const uint64_t rid = ptr->request_id;
    jobs_[rid] = std::move(ptr);
    // A request running speculatively yields to this one at its next step boundary.
    if (spec_running_.load(std::memory_order_relaxed)) spec_interrupt_.store(true, std::memory_order_relaxed);
    cv_.notify_one();
    if (err_out) *err_out = EnqueueError::NONE;
    return rid;
}

ContinuousBatchEngine::Result ContinuousBatchEngine::wait_locked(uint64_t request_id) {
    // Async emission: run the caller's callbacks here, on its own thread, as the worker queues
    // them, until the job closes its queue. Never under mu_ or the queue's lock.
    std::shared_ptr<Job::EmitQueue> q;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = jobs_.find(request_id);
        if (it != jobs_.end()) q = it->second->emit;
    }
    size_t taken = 0, cut = 0;
    if (q) {
        bool stopped = false;
        std::deque<Job::EmitQueue::Event> batch;
        for (;;) {
            bool closed = false;
            {
                std::unique_lock<std::mutex> lk(q->mu);
                q->cv.wait_for(lk, std::chrono::milliseconds(100),
                               [&] { return !q->events.empty() || q->closed; });
                batch.swap(q->events);
                closed = q->closed;
            }
            for (auto& ev : batch) {
                if (stopped) continue;
                if (ev.is_logprob) {
                    if (q->user_logprob) q->user_logprob(ev.lp);
                    continue;
                }
                ++taken;
                if (!q->user_token(ev.token)) {
                    stopped = true;
                    cut = taken;
                    std::lock_guard<std::mutex> g(q->mu);
                    q->stop = true;
                }
            }
            batch.clear();
            if (closed) break;
            // Belt and braces: a job finished by a path that did not close its queue.
            std::lock_guard<std::mutex> lock(mu_);
            auto it = jobs_.find(request_id);
            if (it == jobs_.end() || it->second->done) {
                std::lock_guard<std::mutex> g(q->mu);
                if (q->events.empty()) break;
            }
        }
    }
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
    out.internal_error = it->second->internal_error;
    out.reached_token_limit = it->second->reached_token_limit;
    out.ttft_ms = it->second->ttft_ms;
    out.generation_ms = it->second->generation_ms;
    out.decode_tps = it->second->decode_tps;
    out.cached_tokens = it->second->cached_tokens;
    jobs_.erase(it);
    // The worker may have produced a token or two past the one the callback stopped on.
    // The callback stopped the request, as it would have synchronously: whatever the worker reached
    // after that (EOS, the token limit) is not what ended it.
    if (cut > 0) {
        if (out.tokens.size() > cut) out.tokens.resize(cut);
        out.cancelled = true;
        out.reached_token_limit = false;
    }
    return out;
}

namespace {
// scheduler.cpp's packed_decode_width(): the width a packed decode step serves at a flat cost.
int packed_decode_width_cb() {
    static const int v = [] {
        const char* e = getenv("SPARKINFER_WIDE_DECODE_ROWS");
        const int x = e ? atoi(e) : 32;
        return x < 1 ? 1 : x;
    }();
    return v;
}
}  // namespace

void ContinuousBatchEngine::worker_loop() {
    while (true) {
        // A request that is alone and eligible decodes speculatively (see enable_speculative). It
        // is picked up before its prefill starts, because speculation prefills with hidden-state
        // capture on.
        {
            Job* spec_job = nullptr;
            bool spec_group = false;
            bool draft_offload = false;
            bool draft_restore = false;
            {
                std::lock_guard<std::mutex> lock(mu_);
                // SPARKINFER_SPECULATIVE=0 keeps the draft loaded (the same device memory, so the
                // same prefill and decode paths) but decodes every request token by token: the
                // reference a speculative launch must reproduce.
                static const bool spec_env_on = [] {
                    const char* e = getenv("SPARKINFER_SPECULATIVE");
                    return !(e && e[0] == '0');
                }();
                if (speculative_ && running_ && spec_env_on) {
                    int live = 0;
                    Job* only = nullptr;
                    for (const auto& kv : jobs_) {
                        if (kv.second->done) continue;
                        ++live;
                        only = kv.second.get();
                    }
                    // Concurrent speculation: no more live requests than a group takes, each a fresh,
                    // eligible prompt -- or one already decoding that the group can adopt (it rides
                    // the verify a row a step without a draft, see run_spec_group), as long as at
                    // least one is fresh to speculate. Requiring every request to be fresh meant a
                    // load that never drained (the next prompt arriving while others decode) never
                    // formed a group again once one ended. SPARKINFER_SPEC_ADOPT=0 requires it.
                    static const bool adopt_on = [] {
                        const char* e = getenv("SPARKINFER_SPEC_ADOPT");
                        return !(e && e[0] == '0');
                    }();
                    if (spec_group_max() > 1 && live >= 1 && live <= spec_group_max()) {
                        bool all = true;
                        int fresh = 0;
                        for (const auto& kv : jobs_) {
                            const Job& j = *kv.second;
                            if (j.done) continue;
                            // ...and one the join would take: it speculates at least 64 positions
                            // inside the draft's context (spec_group_join's own test). Anything else
                            // started a group only to be declined, and with requests decoding beside
                            // it that cost each of them a state conversion and packed decode its graphs.
                            const bool is_fresh = !j.spec_tried && j.phase == SeqPhase::PREFILL &&
                                                  j.prefill_pos == j.req.prefill_start && spec_eligible(j.req) &&
                                                  j.req.max_new_tokens >= 64 &&
                                                  (int)j.req.prompt.size() + 64 <= model_->spec_group_reach();
                            fresh += is_fresh;
                            all = all && (is_fresh || (adopt_on && spec_adoptable(j)));
                        }
                        spec_group = all && fresh > 0;
                    }
                    // The draft's memory is the headroom concurrent serving needs (a 32-request burst
                    // overran a 32 GB card by ~1 GB with it resident), and past a group's size
                    // nothing reads it. Once the load has stayed there for a while, it goes to the
                    // host; the first speculation after that brings it back (~0.1 s).
                    // SPARKINFER_DRAFT_OFFLOAD_MS=-1 keeps it resident.
                    static const long long offload_ms = [] {
                        const char* e = getenv("SPARKINFER_DRAFT_OFFLOAD_MS");
                        return e ? atoll(e) : 1000LL;
                    }();
                    if (live > std::max(1, spec_group_max())) {
                        const auto now = std::chrono::steady_clock::now();
                        if (!spec_over_) {
                            spec_over_ = true;
                            spec_over_since_ = now;
                        } else if (offload_ms >= 0 &&
                                   now - spec_over_since_ >= std::chrono::milliseconds(offload_ms)) {
                            draft_offload = true;
                            spec_over_since_ = now;   // a failed attempt waits another interval
                        }
                    } else {
                        spec_over_ = false;
                    }
                    const bool want_spec =
                        spec_group ||
                        (live == 1 && !only->spec_tried && only->phase == SeqPhase::PREFILL &&
                         only->prefill_pos == only->req.prefill_start && spec_eligible(only->req));
                    if (want_spec && model_->dflash_draft_offloaded()) {
                        // Bring it back outside mu_ (submits keep queueing jobs meanwhile; one that
                        // opens a session waits on the device mutex for the ~0.1 s copy), then decide again.
                        draft_restore = true;
                        spec_group = false;
                    }
                    if (!draft_restore && !spec_group && live == 1 && !only->spec_tried &&
                        only->phase == SeqPhase::PREFILL &&
                        only->prefill_pos == only->req.prefill_start && spec_eligible(only->req)) {
                        spec_job = only;
                        // Raised under mu_, which submit_locked also holds: a request submitted from
                        // here on sees it and interrupts; one submitted before made live == 2.
                        spec_interrupt_.store(false, std::memory_order_relaxed);
                        spec_running_.store(true, std::memory_order_relaxed);
                    }
                }
            }
            if (draft_restore) {
                const auto t0 = std::chrono::steady_clock::now();
                if (model_->dflash_draft_restore()) {
                    fprintf(stderr, "[spec] draft back on the device (%.0f ms)\n",
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0).count());
                } else {
                    // No room for it now: these requests decode token by token instead.
                    std::lock_guard<std::mutex> lock(mu_);
                    for (auto& kv : jobs_)
                        if (!kv.second->done) kv.second->spec_tried = true;
                    fprintf(stderr, "[spec] no room to bring the draft back yet; decoding without it\n");
                }
                continue;
            }
            if (draft_offload && !model_->dflash_draft_offloaded()) {
                const auto t0 = std::chrono::steady_clock::now();
                if (const size_t b = model_->dflash_draft_offload())
                    fprintf(stderr, "[spec] draft off the device: %.2f GB freed while more requests run "
                                    "than speculation takes (%.0f ms)\n", (double)b / 1e9,
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0).count());
            }
            if (spec_group) {
                run_spec_group();
                cv_.notify_all();
                continue;
            }
            if (spec_job) {
                run_speculative(*spec_job);
                cv_.notify_all();
                continue;
            }
        }
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

            // Nothing decodes yet and a small load is still being submitted: give the requests
            // already inside complete_streaming a moment (bounded) to land, so the scheduler sees
            // the whole load at once and they are prefilled together.
            int arriving = submitting_.load(std::memory_order_acquire);
            if (arriving > 0) {
                int pending = 0, decoding = 0;
                for (const auto& s : active) (s.phase == SeqPhase::PREFILL ? pending : decoding)++;
                if (decoding == 0 && pending >= 1 && (pending + arriving) * 4 <= packed_decode_width_cb()) {
                    cv_.wait_for(lock, std::chrono::milliseconds(2), [&] {
                        return submitting_.load(std::memory_order_acquire) == 0;
                    });
                    arriving = submitting_.load(std::memory_order_acquire);
                    active.clear();
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
                }
            }
            ScheduleBatch batch = scheduler_.schedule(active, arriving);
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
        // One packed forward for the whole decode batch when every row is eligible; otherwise the
        // original one-forward-per-sequence loop, unchanged.
        std::vector<MixChunk> mix;
        std::vector<uint64_t> unmixable;
        if (mix_decode) pick_mixed_chunks(prefill_ids, (int)decode_ids.size(), mix, unmixable);
        if (!step_jobs_packed(decode_ids, any_finished, mix.empty() ? nullptr : &mix)) {
            for (uint64_t id : decode_ids) {
                Job* job = nullptr;
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    auto it = jobs_.find(id);
                    if (it != jobs_.end() && !it->second->done) job = it->second.get();
                }
                if (job) any_finished = step_job(*job, /*chunked=*/false) || any_finished;
            }
        }
        // The scheduler may hand back more than one prefill while the decode batch is still
        // filling (see Scheduler::schedule). They run back to back on this thread, which is the
        // point: each one widens the next decode step, and a decode step's cost is almost all
        // fixed weight read.
        // A mixed step already advanced the waiting prompts it could carry, with the decode rows
        // riding along; the next steps carry them on (running them now would stall the decode
        // batch again). A chunk that ended on a prefix-cache checkpoint snapshots it here, as
        // step_job would have. Only the prefills a mixed step can never carry run now.
        int mixed = 0;
        for (const MixChunk& c : mix) mixed += c.done;
        if (mixed > 0) {
            for (const MixChunk& c : mix) {
                Job& J = *c.job;
                if (c.done <= 0 || J.phase != SeqPhase::PREFILL || !(prefix_cache_ && J.req.prefix_cache))
                    continue;
                const int n = (int)J.req.prompt.size();
                bool at_ck = false;
                for (int ck : J.req.cache_checkpoints)
                    at_ck = at_ck || (ck == J.prefill_pos && ck < n && ck % kv_->block_size() == 0);
                bool have = false;
                for (const Job::Checkpoint& cp : J.checkpoints) have = have || cp.pos == J.prefill_pos;
                if (at_ck && !have) {
                    Job::Checkpoint cp;
                    cp.pos = J.prefill_pos;
                    if (model_->snapshot_recurrent_state(J.seq_id, cp.state))
                        J.checkpoints.push_back(std::move(cp));
                }
            }
            prefill_ids.swap(unmixable);
        }
        step_prefills_packed(prefill_ids);
        for (uint64_t pid : prefill_ids) {
            Job* job = nullptr;
            {
                std::lock_guard<std::mutex> lock(mu_);
                auto it = jobs_.find(pid);
                if (it != jobs_.end() && !it->second->done) job = it->second.get();
            }
            if (job) any_finished = step_job(*job, /*chunked=*/mix_decode ||
                                             policy_ == SchedulePolicy::CHUNKED_PREFILL) || any_finished;
        }
        if (any_finished) cv_.notify_all();
    }
}

// Was a lambda inside step_job(); hoisted so the packed decode path retires a row through the
// SAME code rather than a second copy that could drift from it.
void ContinuousBatchEngine::finish_job_impl(Job& j) {
    if (j.seq_id != 0) {
        // Offer each checkpointed prefix before this session's own references to its blocks go.
        // Only once prefill has finished and nothing failed: the KV for [0, checkpoint) was
        // written before the snapshot was taken, but a job that errored may have left the device
        // in a state not worth caching.
        if (prefix_cache_ && !j.checkpoints.empty() && j.phase != SeqPhase::PREFILL && j.error.empty()) {
            std::lock_guard<std::recursive_mutex> device_lock(model_->device_mutex());
            for (Job::Checkpoint& cp : j.checkpoints) {
                std::vector<int> blocks = kv_->retain_prefix_blocks(j.seq_id, cp.pos / kv_->block_size());
                if (!blocks.empty())
                    prefix_cache_->insert(std::vector<int>(j.req.prompt.begin(), j.req.prompt.begin() + cp.pos),
                                          std::move(blocks), std::move(cp.state));
            }
        }
        j.checkpoints.clear();   // release the pinned copies now
        // Offer this session's KV to the external cache tier (docs/lmcache_bridge_protocol.md)
        // only once the full prompt has actually been ingested -- j.phase only advances past
        // PREFILL once prefill_pos reaches the prompt's end (see step_job). A job
        // cancelled/timed-out mid-prefill has KV for only part of its prompt range (possibly
        // garbage past prefill_pos), so store_tokens must stay null in that case; passing the
        // full prompt would tell close_session a longer range is valid than actually is.
        model_->close_session(j.seq_id, j.phase != SeqPhase::PREFILL ? &j.req.prompt : nullptr);
    } else {
        // Session 0 is the shared prefix session. Freeing it wholesale is what made the
        // prefix cache cache nothing: prefix_cached_len() then returned 0 and the next
        // matching request re-prefilled the entire prefix. Keep the prefix's own blocks and
        // drop only the suffix + generated tail, so the next request reuses them. The
        // recurrent state is NOT kept -- decoding mutated it -- and is replayed from
        // cache_prefix()'s snapshot by restore_prefix_state() on the next reuse.
        const int keep = j.req.use_prefix_session ? model_->prefix_block_count() : 0;
        if (keep > 0 && kv_->truncate_blocks(j.seq_id, keep)) {
            // prefix stays installed and active; nothing else to do
        } else {
            kv_->free(j.seq_id);
            if (j.req.use_prefix_session) model_->release_prefix_session();
        }
    }
    j.seq_id = 0;
    // Last, and under mu_. `done` is what lets the request's own thread (wait_locked) take the result
    // and destroy this Job. Set first, as it used to be, that thread could wake -- any other job's
    // finish notifies cv_ -- and erase the Job while the worker was still inside the prefix-cache
    // insert and close_session above: a use-after-free that segfaulted the server in finish_job_impl
    // under concurrent load with the cache on. Nothing may touch j after this line.
    std::shared_ptr<Job::EmitQueue> q = j.emit;   // j may be gone once done is set
    std::lock_guard<std::mutex> lock(mu_);
    j.done = true;
    if (q) {
        std::lock_guard<std::mutex> g(q->mu);
        q->closed = true;
        q->cv.notify_all();
    }
}

// Packed decode: one forward for the whole decode batch.
//
// worker_loop() below used to run `for (id : decode_ids) step_job(...)`, i.e. a full 64-layer
// forward PER SEQUENCE. Decode is bandwidth-bound on weight reads, so N concurrent requests read
// every weight N times and aggregate throughput does not scale with concurrency at all. The
// scheduler already hands us the batch; this executes it as one.
//
// Declines (returning false having changed nothing) whenever a row would not decode identically
// to what step_job would have produced: anything still prefilling, teacher-forced scoring,
// per-token logprobs, presence/frequency penalties, a logit bias or a constraint. Temperature,
// top_k and top_p are packed: decode_packed() samples such a row from its own logits with
// forward_token()'s kernels, seed and step, so it draws the token step_job would have drawn. They
// used to decline too, and since requests that set no sampler take the checkpoint's
// generation_config (temperature 1.0 on Qwen3.8), almost every server request then decoded one
// forward per sequence: aggregate throughput stayed at single-stream speed at any concurrency.
// A declined batch just falls back to the sequential loop.
// MIXED STEPS (SPARKINFER_MIXED_CHUNK=<tokens> per step, default 1024 and 4096 on an MoE model,
// 0 = off). While requests decode and prompts wait, the decode step carries chunks of those
// prompts in the same forward:
// the decode rows ride the chunks' weight reads instead of stalling behind prefill passes of their
// own. The step's prompt tokens are filled oldest prompt first, as vLLM fills its token budget --
// one prompt at a time left the rest queued behind it, which is what set the TTFT tail at c16/c32.
// A chunk that reaches its prompt's end takes the first token from the same pass. A prompt the
// pass cannot finish (it wants logprobs, or a constraint shapes its first token) stops one token
// short and step_job owns its seed; a chunk also stops at the next prefix-cache checkpoint, which
// the worker snapshots once a step lands on it.
void ContinuousBatchEngine::pick_mixed_chunks(const std::vector<uint64_t>& prefill_ids, int n_decode,
                                              std::vector<MixChunk>& chunks,
                                              std::vector<uint64_t>& unmixable) {
    // An MoE model takes 4096. Its step's chunk routes to nearly every expert, so each mixed step
    // streams the whole expert set whatever the chunk size: at 1024 tokens that was ~24 ms of
    // grouped MoE GEMM per 1K prompt tokens, and fewer, larger steps carry the same prompts for a
    // fraction of the weight traffic. Measured on Qwen3.6-35B-A3B UD-Q4_K_M, AIPerf chat (1024 /
    // 256, distinct prompts): c16 1,355 -> 1,476, c32 1,523 -> 1,648 output tok/s; 8K prompts c16
    // 556 -> 578; c4 (no mixing) unchanged. 8192 measured the same as 4096 with a slower first
    // token. Dense models keep 1024.
    static const int env_budget = [] {
        const char* e = getenv("SPARKINFER_MIXED_CHUNK");
        return e ? std::max(0, atoi(e)) : -1;
    }();
    const int budget = env_budget >= 0 ? env_budget
                     : (model_ && model_->config().n_experts > 1 ? 4096 : 1024);
    // At most this many prompts in one step (SPARKINFER_MIXED_PROMPTS).
    static const int max_prompts = [] {
        const char* e = getenv("SPARKINFER_MIXED_PROMPTS");
        const int v = e ? atoi(e) : 8;
        return v < 1 ? 1 : v;
    }();
    // A prompt with more than this left is a long prefill: one batched pass of its own runs it
    // faster than budget-sized chunks would (8192-token prompts at c16/c32: -12% output tok/s
    // chunked), and the scheduler already meters those one per iteration. Same knob and default
    // as the scheduler's prefill_mix_max_tokens(); 0 mixes every length.
    static const int long_prompt = [] {
        const char* e = getenv("SPARKINFER_PREFILL_MIX_MAX");
        const int v = e ? atoi(e) : 2048;
        return v >= 0 ? v : 2048;
    }();
    // A budget below this does not make a mixed step; and the smallest part of a prompt the budget
    // is split down to (a prompt that fits whole is taken whole, however small).
    static constexpr int kMinBudget = 128, kMinPart = 64;
    chunks.clear();
    unmixable.clear();
    if (budget < kMinBudget || !model_) return;
    std::lock_guard<std::mutex> lock(mu_);
    // Nothing is mixed while 8 or fewer requests are live. With a drafter that load is
    // speculation's (worker_loop forms a group from fresh prompts and the requests decoding beside
    // them); a prompt part-way through mixed steps is neither, so mixing there kept groups from
    // forming (Poisson arrivals at 2 requests/s: mean latency 4.56 -> 5.76 s). Without one, a
    // prompt's own pass beside a handful of decode rows is the faster first token: Qwen3.6 at c4
    // TTFT p50 80 -> 148 ms mixed, since its whole 1K prompt is one ~35 ms pass.
    {
        int live = 0;
        for (const auto& kv : jobs_) live += !kv.second->done;
        if (live <= std::max(8, spec_group_max())) return;
    }
    // BURSTS (opt-in, SPARKINFER_MIXED_ROW_TOKENS=<tokens per decode row>, e.g. 256): fresh prompt
    // tokens waiting beyond decode rows x that -- a load ramping up, or a wave arriving at once --
    // take passes of SPARKINFER_MIXED_BURST_CHUNK tokens (default 4096) and up to 16 prompts, with
    // the decode rows riding along, or with that at 0 the packed prefill. Budget-sized passes split
    // a burst where one packed pass is cheaper: the continuous-batching bench (C streams of 256
    // tokens arriving together) runs 3-9% fewer tok/s mixing every step than either rule (Bonsai-2
    // c32 2,074 against 2,263). Off by default all the same: in AIPerf chat, mixing the first wave
    // in budget-sized steps staggers when its prompts finish, so the waves that follow do not
    // arrive at once -- TTFT p50 at c16 / c32 343-348 / 364-367 ms, against 635-1,058 with either
    // rule, and request latency p50 at c32 6.0-6.1 s against 6.3-7.1.
    static const int row_tokens = [] {
        const char* e = getenv("SPARKINFER_MIXED_ROW_TOKENS");
        return e ? std::max(0, atoi(e)) : 0;
    }();
    static const int burst_budget = [] {
        const char* e = getenv("SPARKINFER_MIXED_BURST_CHUNK");
        return e ? std::max(0, atoi(e)) : 4096;
    }();
    int step_budget = budget, step_prompts = max_prompts;
    if (row_tokens > 0) {
        long fresh = 0;
        for (uint64_t id : prefill_ids) {
            auto it = jobs_.find(id);
            if (it == jobs_.end() || it->second->done) continue;
            const Job& j = *it->second;
            if (j.phase == SeqPhase::PREFILL && j.mixed_tokens == 0)
                fresh += (long)j.req.prompt.size() - j.prefill_pos;
        }
        if (fresh > (long)n_decode * row_tokens) {
            if (burst_budget < kMinBudget) return;
            step_budget = std::max(budget, burst_budget);
            step_prompts = std::max(max_prompts, 16);
        }
    }
    int left = step_budget;
    for (uint64_t id : prefill_ids) {
        auto it = jobs_.find(id);
        if (it == jobs_.end() || it->second->done) continue;
        Job* j = it->second.get();
        const Request& r = j->req;
        if (j->phase != SeqPhase::PREFILL || j->seq_id == 0 || r.use_prefix_session ||
            !r.vision_pos.empty() || !r.mrope_pos.empty() || !r.forced_tokens.empty() ||
            !r.logit_bias.empty()) {
            unmixable.push_back(id);
            continue;
        }
        const int n = (int)r.prompt.size();
        if (long_prompt > 0 && j->mixed_tokens == 0 && n - j->prefill_pos > long_prompt) {
            unmixable.push_back(id);
            continue;
        }
        const bool finishable = !r.logprobs && !r.constraint;
        // Up to the next prefix-cache checkpoint step_job would snapshot, else the prompt's end
        // (one token short of it for a prompt the pass cannot finish).
        int limit = finishable ? n : n - 1;
        if (prefix_cache_ && r.prefix_cache)
            for (int ck : r.cache_checkpoints)
                if (ck > j->prefill_pos && ck < limit && ck % kv_->block_size() == 0) limit = ck;
        const int avail = limit - j->prefill_pos;
        // Under 8 tokens to go: step_job's short-resume forward takes them (about one decode
        // step). As a chunk they could leave a pass no 8-row alignment can fix, and an unaligned
        // mixed pass runs every layer on the NVFP4 fallback (a 9-row step: 85 ms against 16-20
        // for aligned ones up to 88 rows).
        if (avail < 8) {
            unmixable.push_back(id);
            continue;
        }
        // Past the budget or the prompt count it waits a step.
        if ((int)chunks.size() >= step_prompts) continue;
        const int take = std::min(avail, left);
        if (take < 1 || (take < avail && take < kMinPart)) continue;
        MixChunk c;
        c.job = j;
        c.max = take;
        c.finish = finishable && limit == n && take == avail;
        chunks.push_back(c);
        left -= take;
    }
}

// One mixed step: the decode rows in `toks`.. plus `chunks`, rows + chunk tokens trimmed to a
// multiple of 8 (the NVFP4 GEMMs' row granule, else the pass takes the slow unaligned arm on every
// layer) off the chunks that do not finish their prompts where possible. False having changed
// nothing when the model declines.
bool ContinuousBatchEngine::run_mixed_chunks(const std::vector<int>& toks, const std::vector<int>& pos,
                                             const std::vector<uint64_t>& seqs, std::vector<int>& out,
                                             const Qwen35Model::PackedSampling* samp,
                                             std::vector<MixChunk>& chunks) {
    const int m = (int)toks.size();
    const int k = (int)chunks.size();
    std::vector<int> lens((size_t)k);
    int total = m;
    for (int c = 0; c < k; ++c) total += lens[(size_t)c] = chunks[(size_t)c].max;
    std::vector<unsigned char> want((size_t)k);
    for (int c = 0; c < k; ++c) want[(size_t)c] = chunks[(size_t)c].finish ? 1 : 0;
    // The remainder comes off one chunk, whole: the latest one that stops short anyway, else the
    // latest that finishes, which leaves its last few tokens to step_job (see pick_mixed_chunks:
    // every chunk is 8+ tokens, so one always can). An unaligned pass would run every layer on
    // the NVFP4 fallback, so a step that cannot align does not mix.
    const int r = total % 8;
    if (r > 0) {
        int at = -1;
        for (int pass = 0; pass < 2 && at < 0; ++pass)
            for (int c = k - 1; c >= 0 && at < 0; --c)
                if ((pass == 1 || !want[(size_t)c]) && lens[(size_t)c] > r) at = c;
        if (at < 0) return false;
        lens[(size_t)at] -= r;
        want[(size_t)at] = 0;
    }
    std::vector<uint64_t> cs((size_t)k);
    std::vector<const int*> ci((size_t)k);
    std::vector<int> p0((size_t)k), seeds((size_t)k, -1), top_k((size_t)k);
    std::vector<float> temp((size_t)k), top_p((size_t)k);
    std::vector<unsigned long long> seed((size_t)k), step((size_t)k);
    for (int c = 0; c < k; ++c) {
        Job& J = *chunks[(size_t)c].job;
        cs[(size_t)c] = J.seq_id;
        ci[(size_t)c] = J.req.prompt.data() + J.prefill_pos;
        p0[(size_t)c] = J.prefill_pos;
        // step_job's arguments to sample_seed_token.
        temp[(size_t)c] = J.req.temperature;
        seed[(size_t)c] = (unsigned long long)J.req.seed;
        step[(size_t)c] = (unsigned long long)J.decode_emitted;
        top_k[(size_t)c] = J.req.top_k;
        top_p[(size_t)c] = J.req.top_p;
    }
    Qwen35Model::PackedSampling cs_samp;
    cs_samp.temperature = temp.data();
    cs_samp.seed = seed.data();
    cs_samp.step = step.data();
    cs_samp.top_k = top_k.data();
    cs_samp.top_p = top_p.data();
    // Text-only prompts: clear the rotary decode offset, as step_job does before each.
    model_->reset_mrope_offset();
    // SPARKINFER_MIXED_TRACE=1: one line per mixed step (rows, each chunk's length and whether it
    // finishes, wall time).
    static const bool trace = [] {
        const char* e = getenv("SPARKINFER_MIXED_TRACE");
        return e && e[0] == '1';
    }();
    const auto t0 = std::chrono::steady_clock::now();
    const bool ran = model_->mixed_step_multi(toks.data(), pos.data(), seqs.data(), m, out.data(), samp,
                                              k, cs.data(), ci.data(), p0.data(), lens.data(),
                                              want.data(), seeds.data(), &cs_samp);
    if (trace) {
        int rows = m;
        std::string cl;
        for (int c = 0; c < k; ++c) {
            rows += lens[(size_t)c];
            cl += " " + std::to_string(lens[(size_t)c]) + (want[(size_t)c] ? "f" : "");
        }
        fprintf(stderr, "[mixed] %s %d rows (%d decode,%s) %.1f ms\n", ran ? "step" : "declined", rows, m,
                cl.c_str(), std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    if (!ran) return false;
    bool finished_any = false;
    for (int c = 0; c < k; ++c) {
        MixChunk& mc = chunks[(size_t)c];
        Job& J = *mc.job;
        J.prefill_pos += lens[(size_t)c];
        J.mixed_tokens += lens[(size_t)c];
        mc.done = lens[(size_t)c];
        mc.finish = want[(size_t)c] != 0;
        if (mc.finish) {
            // As a packed prefill leaves a prompt: DECODE with its first token pending. A seed
            // that did not come back fails the request on its next step_job, as step_job's own
            // prefill would.
            J.next_token = seeds[(size_t)c];
            J.phase = SeqPhase::DECODE;
            finished_any = true;
        }
    }
    // The prefill's scratch memory is back: a request waiting on a failed allocation retries.
    if (finished_any) cv_.notify_all();
    return true;
}

bool ContinuousBatchEngine::step_jobs_packed(const std::vector<uint64_t>& ids, bool& any_finished,
                                             std::vector<MixChunk>* chunks) {
    if (chunks)
        for (MixChunk& c : *chunks) c.done = 0;
    if (chunks && chunks->empty()) chunks = nullptr;
    static const bool enabled = [] {
        const char* e = getenv("SPARKINFER_PACKED_DECODE");
        return !(e && e[0] == '0');
    }();
    if (!enabled || !model_) return false;
    // A batch WIDER than the packed graph tiers is split into chunks of `cap`, not declined.
    // Declining it fell all the way back to one forward per sequence, so crossing the cap cost
    // more than never packing at all: measured on RTX 5090 / Qwen3.8-27B-NVFP4, aggregate went
    // 334.9 tok/s at concurrency 8 to 79.0 at 12 -- a 4.2x collapse one request past the cap.
    const int cap = Qwen35Model::max_packed_rows();
    if ((int)ids.size() < (chunks ? 1 : 2)) return false;
    const Qwen35Config& cfg = model_->config();

    std::vector<Job*> jobs;
    jobs.reserve(ids.size());
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (uint64_t id : ids) {
            auto it = jobs_.find(id);
            if (it == jobs_.end() || it->second->done) return false;
            jobs.push_back(it->second.get());
        }
    }
    for (Job* j : jobs) {
        if (j->phase != SeqPhase::DECODE) return false;
        if (j->next_token < 0 || j->next_token >= cfg.vocab) return false;
        if (!j->req.forced_tokens.empty()) return false;
        if (j->req.logprobs || j->on_token_logprob) return false;
        // Penalties read the session's running token counts, which the packed pass neither applies
        // nor advances; such a row decodes on its own.
        if (j->req.presence_penalty != 0.f || j->req.frequency_penalty != 0.f) return false;
        // decode_packed applies no logit bias: a request with logit_bias or a constraint decodes on
        // its own, where forward_token applies it.
        if (!j->req.logit_bias.empty() || j->req.constraint) return false;
    }

    // Emit each row's pending token and run the same termination checks step_job() does. A job
    // that finishes here simply drops out of the packed forward below.
    std::vector<Job*> live;
    live.reserve(jobs.size());
    for (Job* j : jobs) {
        const auto t_emit = std::chrono::steady_clock::now();
        if (!j->saw_first_tok) {
            j->t_first = t_emit;
            j->saw_first_tok = true;
            j->ttft_ms = std::chrono::duration<double, std::milli>(j->t_first - j->t_submit).count();
        }
        j->output.push_back(j->next_token);
        j->decode_emitted++;
        if (j->on_token && !j->on_token(j->next_token)) {
            j->cancelled = true;
            j->generation_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - j->t_submit).count();
            finish_job_impl(*j);
            any_finished = true;
            continue;
        }
        const bool hit_eos = !j->req.ignore_eos &&
                             (j->next_token == cfg.eos_id ||
                              (cfg.eos_id2 >= 0 && j->next_token == cfg.eos_id2));
        const bool hit_limit = j->decode_emitted >= j->req.max_new_tokens;
        if (hit_eos || hit_limit) {
            j->reached_token_limit = hit_limit && !hit_eos;
            const auto t_end = std::chrono::steady_clock::now();
            j->generation_ms = std::chrono::duration<double, std::milli>(t_end - j->t_submit).count();
            if (j->saw_first_tok && j->generation_ms > j->ttft_ms && j->decode_emitted > 0) {
                const double decode_ms = std::max(j->generation_ms - j->ttft_ms, 1.0);
                j->decode_tps = (double)j->decode_emitted * 1000.0 / decode_ms;
            }
            finish_job_impl(*j);
            any_finished = true;
            continue;
        }
        live.push_back(j);
    }
    if (live.empty()) return true;

    // Advance the survivors in chunks of `cap`. A chunk of one (the tail of an odd batch, or all
    // but one row having finished above) is not worth a packed forward, and decode_packed declines
    // a batch it cannot serve; either way those rows still have to advance, so they fall through
    // to the ordinary per-row forward. Tokens already emitted stay emitted -- this is the same
    // work by a different route, not a retry.
    std::vector<int> toks, pos, out, top_k;
    std::vector<uint64_t> seqs;
    std::vector<float> temp, top_p;
    std::vector<unsigned long long> seed, step;
    for (size_t off = 0; off < live.size(); off += (size_t)cap) {
        const size_t m = std::min((size_t)cap, live.size() - off);
        toks.clear(); pos.clear(); seqs.clear(); out.assign(m, -1);
        temp.clear(); seed.clear(); step.clear(); top_k.clear(); top_p.clear();
        bool any_sampled = false;
        for (size_t i = 0; i < m; i++) {
            Job* j = live[off + i];
            toks.push_back(j->next_token);
            pos.push_back((int)j->req.prompt.size() + j->decode_emitted - 1);
            seqs.push_back(j->seq_id);
            // step_job's arguments to forward_token, row by row.
            temp.push_back(j->req.temperature);
            seed.push_back((unsigned long long)j->req.seed);
            step.push_back((unsigned long long)j->decode_emitted);
            top_k.push_back(j->req.top_k);
            top_p.push_back(j->req.top_p);
            any_sampled = any_sampled || j->req.temperature > 0.f;
        }
        // An all-greedy chunk passes no sampling at all and takes exactly the argmax path it did.
        Qwen35Model::PackedSampling samp;
        samp.temperature = temp.data();
        samp.seed = seed.data();
        samp.step = step.data();
        samp.top_k = top_k.data();
        samp.top_p = top_p.data();
        bool ok = false;
        // The first group carries the prompt chunks.
        if (off == 0 && chunks) ok = run_mixed_chunks(toks, pos, seqs, out, any_sampled ? &samp : nullptr,
                                                      *chunks);
        if (!ok && m >= 2)
            ok = model_->decode_packed(toks.data(), pos.data(), seqs.data(), (int)m, out.data(),
                                       any_sampled ? &samp : nullptr);
        if (!ok) {
            for (size_t i = 0; i < m; i++) {
                Job* j = live[off + i];
                model_->activate_session(j->seq_id);
                out[i] = model_->forward_token(j->next_token, pos[i], true, j->req.temperature,
                                               j->req.seed, (uint64_t)j->decode_emitted,
                                               j->req.top_k, j->req.top_p,
                                               j->req.presence_penalty, j->req.frequency_penalty);
            }
        }
        for (size_t i = 0; i < m; i++) live[off + i]->next_token = out[i];
    }
    return true;
}

// PACKED PROMPT PREFILL. A burst of requests is a burst of prompts, and prefilling them one pass
// each leaves most of the batched pass's width unused: the same pass runs 5858 tok/s at 256 rows
// and 10972 at 4096 (RTX 5090, unsloth Qwen3.8), yet 32 arriving 256-token prompts went through
// it as 32 separate passes before the first decode token -- about a fifth of a c32 run. So the
// fresh, text-only prompts the scheduler hands over are prefilled together in passes of up to
// SPARKINFER_PREFILL_PACK_TOKENS rows (0 disables). Anything a pack cannot carry -- logprobs,
// logit_bias, a constraint, forced tokens, an image, prefix-cache work, or an exactly-512-token
// prompt (which the batched pass keeps on bf16 GDN) -- takes step_job unchanged, and so does
// every job in a pack the model declines.
// The prefix-cache checkpoint a packed prefill takes for `j`: 0 for none, the row for exactly one,
// -1 when the job's checkpoints need the one-prompt path (several of them, or one the pack cannot
// place: it needs 16 tokens each side, and a block-aligned row strictly inside the prompt).
int ContinuousBatchEngine::pack_checkpoint(const Job& j) const {
    if (!(prefix_cache_ && j.req.prefix_cache) || j.req.cache_checkpoints.empty()) return 0;
    static const bool pack_ckpt = [] {
        const char* e = getenv("SPARKINFER_PACK_CHECKPOINTS");
        return !(e && e[0] == '0');
    }();
    if (!pack_ckpt) return -1;
    const int n = (int)j.req.prompt.size();
    int row = 0, count = 0;
    for (int ckpt : j.req.cache_checkpoints) {
        if (ckpt <= 0 || ckpt >= n || ckpt % kv_->block_size() != 0) continue;
        if (ckpt == row) continue;
        row = ckpt;
        ++count;
    }
    if (count == 0) return 0;
    // A checkpoint under 16 tokens from the prompt's end -- a chat prompt's last turn boundary is
    // usually ~12 tokens before it -- is taken a block earlier instead of sending the prompt down
    // the one-prompt path: the next turn still matches the cached prefix up to it and recomputes
    // one more block, where refusing kept every prompt of a burst out of the pack (Qwen3.6, 32
    // concurrent 1K-token chat prompts: 19 against 33 requests/s packed).
    const int bs = kv_->block_size();
    while (n - row < 16 && row - bs >= 16) row -= bs;
    if (count > 1 || row < 16 || n - row < 16) return -1;
    return row;
}

void ContinuousBatchEngine::step_prefills_packed(std::vector<uint64_t>& prefill_ids) {
    static const int pack_tokens = [] {
        const char* e = getenv("SPARKINFER_PREFILL_PACK_TOKENS");
        const int v = e ? atoi(e) : 4096;
        return v < 0 ? 0 : v;
    }();
    if (pack_tokens <= 0 || prefill_ids.size() < 2 || device_lost()) return;
    std::vector<Job*> eligible;
    {
        std::lock_guard<std::mutex> lock(mu_);
        for (uint64_t pid : prefill_ids) {
            auto it = jobs_.find(pid);
            if (it == jobs_.end() || it->second->done) continue;
            Job& j = *it->second;
            const int n = (int)j.req.prompt.size();
            const bool plain =
                j.phase == SeqPhase::PREFILL && j.prefill_pos == 0 && j.req.prefill_start == 0 &&
                j.cached_tokens == 0 && j.seq_id != 0 && !j.req.use_prefix_session &&
                j.req.vision_pos.empty() && j.req.mrope_pos.empty() && j.req.forced_tokens.empty() &&
                !j.req.logprobs && j.req.logit_bias.empty() && !j.req.constraint &&
                pack_checkpoint(j) >= 0;
            if (plain && n >= 2 && n <= pack_tokens && n != 512) eligible.push_back(&j);
        }
    }
    if (eligible.size() < 2) return;
    std::vector<std::vector<Job*>> packs;
    int rows = 0;
    for (Job* j : eligible) {
        const int n = (int)j->req.prompt.size();
        if (packs.empty() || rows + n > pack_tokens) {
            packs.emplace_back();
            rows = 0;
        }
        packs.back().push_back(j);
        rows += n;
    }
    std::vector<uint64_t> packed;
    for (auto& pk : packs) {
        if (pk.size() < 2) continue;
        std::vector<uint64_t> sids;
        std::vector<const int*> prompts;
        std::vector<int> lens, seeds(pk.size(), -1), top_k;
        std::vector<float> temp, top_p;
        std::vector<unsigned long long> seed, step;
        std::vector<int> ckpt_rows;
        bool any_sampled = false, any_ckpt = false;
        for (Job* j : pk) {
            ckpt_rows.push_back(pack_checkpoint(*j));
            any_ckpt = any_ckpt || ckpt_rows.back() > 0;
            sids.push_back(j->seq_id);
            prompts.push_back(j->req.prompt.data());
            lens.push_back((int)j->req.prompt.size());
            // step_job's arguments to sample_seed_token, prompt by prompt.
            temp.push_back(j->req.temperature);
            seed.push_back((unsigned long long)j->req.seed);
            step.push_back((unsigned long long)j->decode_emitted);
            top_k.push_back(j->req.top_k);
            top_p.push_back(j->req.top_p);
            any_sampled = any_sampled || j->req.temperature > 0.f;
        }
        Qwen35Model::PackedSampling samp;
        samp.temperature = temp.data();
        samp.seed = seed.data();
        samp.step = step.data();
        samp.top_k = top_k.data();
        samp.top_p = top_p.data();
        // Text-only prompts: clear the rotary decode offset, exactly as step_job does before each.
        model_->reset_mrope_offset();
        std::vector<Qwen35Model::RecurrentStateSnapshot> snaps(pk.size());
        if (!model_->ingest_prompts_packed(sids.data(), prompts.data(), lens.data(),
                                           (int)pk.size(), seeds.data(),
                                           any_sampled ? &samp : nullptr,
                                           any_ckpt ? ckpt_rows.data() : nullptr,
                                           any_ckpt ? snaps.data() : nullptr))
            continue;
        for (size_t k = 0; k < pk.size(); ++k) {
            if (ckpt_rows[k] > 0 && snaps[k].host) {
                Job::Checkpoint cp;
                cp.pos = ckpt_rows[k];
                cp.state = std::move(snaps[k]);
                pk[k]->checkpoints.push_back(std::move(cp));
            }
            pk[k]->prefill_pos = lens[k];
            pk[k]->next_token = seeds[k];
            pk[k]->phase = SeqPhase::DECODE;
            packed.push_back(pk[k]->request_id);
        }
    }
    if (packed.empty()) return;
    prefill_ids.erase(std::remove_if(prefill_ids.begin(), prefill_ids.end(),
                                     [&](uint64_t id) {
                                         return std::find(packed.begin(), packed.end(), id) !=
                                                packed.end();
                                     }),
                      prefill_ids.end());
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
        {
            std::shared_ptr<Job::EmitQueue> q = job.emit;
            std::lock_guard<std::mutex> lock(mu_);   // done lets the waiting thread destroy the Job
            job.done = true;
            if (q) {
                std::lock_guard<std::mutex> g(q->mu);
                q->closed = true;
                q->cv.notify_all();
            }
        }
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
    auto finish_job = [this](Job& j) { finish_job_impl(j); };

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
        // Prefix-cache checkpoints: prefill up to each one past the cached prefix, snapshot the
        // recurrent state there, and continue. Entries are offered to the cache only when the job
        // retires (finish_job_impl), once the prompt's KV is known to be complete. Every range that
        // starts past zero continues KV and recurrent state already in place -- a restored prefix,
        // an earlier checkpoint segment, or an earlier chunk -- so it may take the batched path.
        int pos = job.prefill_pos;
        int seed = -1;
        bool ingested = false;
        if (prefix_cache_ && job.req.prefix_cache && !has_vision && job.req.forced_tokens.empty()) {
            std::vector<int> ckpts;
            for (int ckpt : job.req.cache_checkpoints)
                if (ckpt > pos && ckpt < n && ckpt % kv_->block_size() == 0) ckpts.push_back(ckpt);
            std::sort(ckpts.begin(), ckpts.end());
            ckpts.erase(std::unique(ckpts.begin(), ckpts.end()), ckpts.end());
            // One pass that snapshots the recurrent state at each checkpoint as it goes by, where the
            // model can take the range that way (Qwen35Model::ingest_prompt_checkpointed). A pass per
            // segment with a snapshot between cost a chat request a second, eager pass over its last
            // 10-25 tokens and a device-wide sync. Only from where a pass starts anyway -- position 0
            // or a restored prefix -- never from a token-loop chunk's continuation.
            if (!ckpts.empty() && (pos == 0 || pos == job.cached_tokens)) {
                std::vector<Qwen35Model::RecurrentStateSnapshot> snaps(ckpts.size());
                int done = pos;
                seed = model_->ingest_prompt_checkpointed(job.req.prompt.data(), pos, n, ckpts.data(),
                                                          (int)ckpts.size(), snaps.data(), &done,
                                                          want_seed_logprob);
                if (seed >= 0 && done == n) {
                    for (size_t i = 0; i < ckpts.size(); ++i) {
                        Job::Checkpoint cp;
                        cp.pos = ckpts[i];
                        cp.state = std::move(snaps[i]);
                        job.checkpoints.push_back(std::move(cp));
                    }
                    out_pos = n;
                    ingested = true;
                }
            }
            for (size_t i = 0; !ingested && i < ckpts.size(); ++i) {
                const int ckpt = ckpts[i];
                int mid = pos;
                model_->ingest_prompt_range(job.req.prompt.data(), pos, ckpt, 0, &mid, false,
                                            /*allow_batched_resume=*/pos > 0);
                pos = mid;
                if (mid != ckpt) break;
                Job::Checkpoint cp;
                cp.pos = ckpt;
                if (model_->snapshot_recurrent_state(job.seq_id, cp.state))
                    job.checkpoints.push_back(std::move(cp));
            }
        }
        if (!ingested) {
            out_pos = pos;
            seed = model_->ingest_prompt_range(job.req.prompt.data(), pos, n, chunk_limit, &out_pos,
                                               want_seed_logprob,
                                               /*allow_batched_resume=*/pos > 0 &&
                                                   (pos != job.prefill_pos || job.cached_tokens > 0 ||
                                                    job.mixed_tokens > 0));
        }
        if (has_vision) model_->clear_pending_vision();
        // Positions are consumed by the prefill they were staged for; the offset is not cleared
        // here, by design.
        if (!job.req.mrope_pos.empty()) model_->clear_pending_mrope();
        job.prefill_pos = out_pos;
        if (job.prefill_pos >= n) {
            // The prefill's scratch memory is back: a request waiting on a failed allocation retries.
            cv_.notify_all();
            job.next_token = seed;
            if (job.next_token < 0 && job.req.use_prefix_session)
                job.next_token = model_->prefix_seed_token();
            job.phase = SeqPhase::DECODE;
            const bool forcing = !job.req.forced_tokens.empty();
            // ingest_prompt_range() returns the argmax. A sampled request draws its first token
            // from those same logits the way forward_token() draws every later one, at sampler
            // step 0 (the first forward_token below is step 1). Not for a seed that came from the
            // cached prefix's own pass (seed < 0 above), whose logits are gone.
            if (!forcing && seed >= 0 && job.req.temperature > 0.f) {
                const int drawn = model_->sample_seed_token(
                    job.req.temperature, (unsigned long long)job.req.seed,
                    (unsigned long long)job.decode_emitted, job.req.top_k, job.req.top_p);
                if (drawn >= 0) job.next_token = drawn;
            }
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
            if (forcing) job.next_token = job.req.forced_tokens[0];
            if (want_seed_logprob && job.next_token >= 0 && (forcing || seed >= 0)) {
                // seed >= 0 guard (generation case): the use_prefix_session fallback
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
    const bool hit_eos = job.req.forced_tokens.empty() && !job.req.ignore_eos &&
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

    // Constrained decoding: the token just emitted advances the constraint, and the next sample is
    // drawn under the mask for what may follow it.
    if (job.req.constraint) {
        if (!job.req.constraint->accept(job.next_token)) {
            job.error = "constrained decoding: emitted a token the constraint does not allow";
            finish_job(job);
            return true;
        }
        if (!apply_constraint_mask(job)) {
            job.error = "constrained decoding: no token can continue the output";
            finish_job(job);
            return true;
        }
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
