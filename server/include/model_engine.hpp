#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace sparkinfer_server {

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

    // Greedy decode. Returns generated token ids (not including prompt).
    // Sets last_error() on failure (empty prompt, context overflow, KV alloc).
    std::vector<int> complete(const std::vector<int>& prompt_ids, int max_new_tokens);

    // Same, but invokes cb after each generated token (for SSE streaming).
    std::vector<int> complete_streaming(const std::vector<int>& prompt_ids, int max_new_tokens,
                                        const std::function<void(int)>& on_token);

    const std::string& last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    mutable std::mutex mu_;
    std::string last_error_;
};

}  // namespace sparkinfer_server
