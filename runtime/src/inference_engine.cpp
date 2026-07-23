#include "sparkinfer/inference_engine.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace sparkinfer {

namespace {

bool prefill_samples_lmhead() {
    static int legacy = -1;
    if (legacy < 0) {
        const char* e = getenv("SPARKINFER_PREFILL_LEGACY");
        legacy = (e && e[0] == '1') ? 1 : 0;
    }
    return legacy != 0;
}

// Match Qwen35Model::batched_prefill_enabled: dense hybrid (Qwythos) OR MoE hybrid
// (Qwen3.6). The old dense_ffn-only gate forced MoE CB onto the token loop (~300 pp),
// which is why scored Qwen3.6 CB mixed TTFT sat at ~17s on main.
bool batched_prefill_enabled(const Qwen35Config& cfg, bool gguf, int n_tokens) {
    static int want_batched = -1, batched_maxctx = -1;
    if (want_batched < 0) {
        const char* e = getenv("SPARKINFER_PREFILL_BATCHED");
        want_batched = (e && e[0] == '0') ? 0 : 1;
        const char* mc = getenv("SPARKINFER_PREFILL_BATCHED_MAXCTX");
        batched_maxctx = mc ? atoi(mc) : 131072;
    }
    const bool ffn_ok = cfg.dense_ffn || cfg.n_experts > 0;
    return want_batched && gguf && cfg.hybrid && ffn_ok && n_tokens > 0 &&
           n_tokens <= batched_maxctx;
}

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

}  // namespace

struct ContinuousBatchEngine::Job {
    uint64_t request_id = 0;
    Request req;
    uint64_t seq_id = 0;
    SeqPhase phase = SeqPhase::PREFILL;
    int prefill_pos = 0;
    int decode_emitted = 0;
    int next_token = -1;
    bool batched_prefill_done = false;
    std::vector<int> output;
    std::string error;
    std::function<void(int)> on_token;
    bool done = false;
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
        if (!kv.second->seq_id) continue;
        if (kv.second->seq_id != 0) model_->close_session(kv.second->seq_id);
        else kv_->free(0);
    }
    jobs_.clear();
}

ContinuousBatchEngine::Result ContinuousBatchEngine::complete(const Request& req) {
    return complete_streaming(req, nullptr);
}

ContinuousBatchEngine::Result ContinuousBatchEngine::complete_streaming(
    const Request& req, const std::function<void(int)>& on_token) {
    uint64_t rid = 0;
    {
        std::lock_guard<std::mutex> lock(mu_);
        rid = submit_locked(Job{0, req, 0, SeqPhase::PREFILL, req.prefill_start, 0, -1, false, {}, {}, on_token, false},
                            on_token);
    }
    if (!rid) return Result{{}, "failed to enqueue request"};
    return wait_locked(rid);
}

int ContinuousBatchEngine::num_active() const {
    std::lock_guard<std::mutex> lock(mu_);
    int n = 0;
    for (const auto& kv : jobs_) if (!kv.second->done) n++;
    return n;
}

int ContinuousBatchEngine::num_free_kv_blocks() const { return kv_->num_free_blocks(); }

uint64_t ContinuousBatchEngine::submit_locked(Job job, const std::function<void(int)>& on_token) {
    if (!model_ || !kv_) return 0;
    if (job.req.prompt.empty() || job.req.max_new_tokens <= 0) return 0;

    const int budget = Qwen35Model::session_token_budget(
        job.req.prompt.size(), job.req.max_new_tokens, model_->config().max_seq);
    if ((int)job.req.prompt.size() + job.req.max_new_tokens > model_->config().max_seq) return 0;

    uint64_t seq_id = 0;
    if (job.req.use_prefix_session) {
        seq_id = 0;
        if (!kv_->allocate(seq_id, budget)) return 0;
        model_->activate_session(seq_id);
    } else {
        seq_id = model_->open_session(budget);
        if (!seq_id) return 0;
    }

    job.request_id = next_req_id_.fetch_add(1);
    job.seq_id = seq_id;
    job.on_token = on_token;
    job.prefill_pos = job.req.prefill_start;
    auto ptr = std::make_unique<Job>(std::move(job));
    const uint64_t rid = ptr->request_id;
    jobs_[rid] = std::move(ptr);
    cv_.notify_one();
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
    Result out{it->second->output, it->second->error};
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

    if (job.phase == SeqPhase::PREFILL) {
        const int n = (int)job.req.prompt.size();
        const int chunk = prefill_chunk_tokens();
        const int remain = n - job.prefill_pos;
        // Batched GEMM prefill (Qwythos / Qwen3.6 hybrid) is ~100× faster than the
        // token loop. Never demote it to token-loop "chunks" — that destroys ITPS under
        // mixed load. True mid-prompt chunked batched prefill needs start_pos support
        // in prefill_batched_run; until then, decode-first scheduling already advances
        // waiting decodes once before this full batched pass runs.
        if (job.prefill_pos == job.req.prefill_start && job.req.prefill_start == 0 &&
            batched_prefill_enabled(cfg, true, remain)) {
            const int seed = model_->prefill_batched(job.req.prompt.data() + job.prefill_pos, remain);
            if (seed >= 0 && seed < cfg.vocab) {
                job.next_token = seed;
                job.prefill_pos = n;
                job.batched_prefill_done = true;
                job.phase = SeqPhase::DECODE;
                return false;
            }
        }
        // Token-loop (or chunked) prefill: used when batched is unavailable (non-hybrid /
        // disabled). Advance up to `limit` tokens then yield so decode can run.
        int limit = remain;
        if (chunked && chunk > 0 && remain > chunk) limit = chunk;
        int advanced = 0;
        while (advanced < limit && job.prefill_pos < n) {
            const bool sample = prefill_samples_lmhead() || job.prefill_pos + 1 == n;
            if (sample)
                job.next_token = model_->forward_token(job.req.prompt[(size_t)job.prefill_pos],
                                                       job.prefill_pos, true);
            else
                model_->forward_token(job.req.prompt[(size_t)job.prefill_pos], job.prefill_pos, false);
            job.prefill_pos++;
            advanced++;
        }
        if (job.prefill_pos >= n) {
            if (job.next_token < 0 && job.req.use_prefix_session)
                job.next_token = model_->prefix_seed_token();
            job.phase = SeqPhase::DECODE;
        }
        return false;
    }

    if (job.next_token < 0 || job.next_token >= cfg.vocab) {
        job.error = "prefill produced invalid seed token";
        job.done = true;
        if (job.seq_id != 0) model_->close_session(job.seq_id);
        else kv_->free(job.seq_id);
        job.seq_id = 0;
        return true;
    }

    job.output.push_back(job.next_token);
    if (job.on_token) job.on_token(job.next_token);
    job.decode_emitted++;

    if (job.next_token == cfg.eos_id || job.decode_emitted >= job.req.max_new_tokens) {
        job.done = true;
        if (job.seq_id != 0) model_->close_session(job.seq_id);
        else kv_->free(job.seq_id);
        job.seq_id = 0;
        return true;
    }

    const int prompt_len = (int)job.req.prompt.size();
    job.next_token = model_->forward_token(job.next_token, prompt_len + job.decode_emitted - 1, true);
    return false;
}

}  // namespace sparkinfer
