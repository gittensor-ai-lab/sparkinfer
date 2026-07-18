#pragma once

#include "sparkinfer/models/qwen35.h"  // sparkinfer::SamplingConfig

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

    // Optional shared prompt prefix (e.g. system message tokens). When set, each request whose
    // prompt starts with these ids calls cache_prefix() (batched prefill) before generate().
    void set_prefix_tokens(const std::vector<int>& tokens);
    int prefix_token_len() const;

    // Decode. Returns generated token ids (not including prompt). `sampling` is
    // optional — nullptr (or temperature<=0) is the original greedy-argmax path;
    // a non-null config with temperature>0 switches to sampled decode.
    // Sets last_error() on failure (empty prompt, context overflow, KV alloc).
    std::vector<int> complete(const std::vector<int>& prompt_ids, int max_new_tokens,
                              const sparkinfer::SamplingConfig* sampling = nullptr);

    // Same, but invokes cb after each generated token (for SSE streaming).
    std::vector<int> complete_streaming(const std::vector<int>& prompt_ids, int max_new_tokens,
                                        const std::function<void(int)>& on_token,
                                        const sparkinfer::SamplingConfig* sampling = nullptr);

    const std::string& last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    mutable std::mutex mu_;
    std::string last_error_;
};

}  // namespace sparkinfer_server
