#pragma once

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

    // Optional shared prompt prefix (e.g. system message tokens). When set, each request whose
    // prompt starts with these ids calls cache_prefix() (batched prefill) before generate().
    void set_prefix_tokens(const std::vector<int>& tokens);
    int prefix_token_len() const;

    // Greedy decode. Tokens are in .tokens; on failure .tokens is empty and .error is set.
    CompletionResult complete(const std::vector<int>& prompt_ids, int max_new_tokens);

    // Same, but invokes cb after each generated token (for SSE streaming).
    CompletionResult complete_streaming(const std::vector<int>& prompt_ids, int max_new_tokens,
                                        const std::function<void(int)>& on_token);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    mutable std::mutex mu_;
};

}  // namespace sparkinfer_server
