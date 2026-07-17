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
#include <vector>

#include "sparkinfer/kv_cache.h"
#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/scheduler.h"

namespace sparkinfer {

// Continuous-batch serving engine: queues requests, assigns per-request seq_ids,
// right-sizes KV allocation, and interleaves decode steps via the Scheduler.
class ContinuousBatchEngine {
public:
    struct Request {
        std::vector<int> prompt;
        int max_new_tokens = 0;
        int priority = 0;
        int prefill_start = 0;          // skip tokens already in a shared prefix cache
        bool use_prefix_session = false; // bind to session 0 (cache_prefix KV)
    };

    struct Result {
        std::vector<int> tokens;
        std::string error;
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
    Result complete_streaming(const Request& req, const std::function<void(int)>& on_token);

    int num_active() const;
    int num_free_kv_blocks() const;

private:
    struct Job;
    uint64_t submit_locked(Job job, const std::function<void(int)>& on_token);
    Result wait_locked(uint64_t request_id);
    void worker_loop();
    bool step_job(Job& job);

    Qwen35Model* model_;
    KVCacheManager* kv_;
    Scheduler scheduler_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::unordered_map<uint64_t, std::unique_ptr<Job>> jobs_;
    std::atomic<uint64_t> next_req_id_{1};
};

}  // namespace sparkinfer
