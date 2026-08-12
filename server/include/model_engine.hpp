#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace sparkinfer_server {

// Outcome of one complete()/complete_streaming() call. Returned by value so concurrent
// HTTP worker threads cannot observe or clear each other's failure state.
struct CompletionResult {
    std::vector<int> tokens;
    std::string error;  // empty on success
    bool overloaded = false;  // true => caller should return 429, not a generic 4xx
    bool timed_out = false;   // true => a per-request deadline was exceeded
    bool cancelled = false;   // true => on_token returned false; not an error
    double ttft_ms = -1.0;
    double generation_ms = -1.0;
    double decode_tps = -1.0;
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
    int vocab() const;
    int max_seq() const;
    bool is_museglimmer() const;

    // Optional shared prompt prefix (e.g. system message tokens). When set, each request whose
    // prompt starts with these ids calls cache_prefix() (batched prefill) before generate().
    void set_prefix_tokens(const std::vector<int>& tokens);
    int prefix_token_len() const;

    // Greedy decode. Tokens are in .tokens; on failure .tokens is empty and .error is set.
    CompletionResult complete(const std::vector<int>& prompt_ids, int max_new_tokens);

    // Same, but invokes on_token after each generated token (for SSE streaming). on_token
    // returns false to cancel generation early (client disconnected); the result then comes
    // back with .cancelled = true, not an error.
    CompletionResult complete_streaming(const std::vector<int>& prompt_ids, int max_new_tokens,
                                        const std::function<bool(int)>& on_token);

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
