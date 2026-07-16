#pragma once

#include <memory>
#include <string>
#include <vector>

namespace sparkinfer_server {

struct ChatMessage {
    std::string role;
    std::string content;
};

// HuggingFace tokenizer.json + Qwen3.6 chat template.
class ChatTokenizer {
public:
    ChatTokenizer();
    ~ChatTokenizer();

    bool load(const std::string& tokenizer_json_path, std::string& err);

    bool encode_chat_request(const std::string& request_json, std::vector<int>& ids, bool enable_thinking,
                             std::string& err) const;
    std::string decode(const std::vector<int>& ids) const;
    std::string decode_delta(std::vector<int>& acc, int new_id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct ParsedAssistantOutput {
    std::string reasoning_content;
    std::string content;
};

// Incrementally routes decoded text into reasoning vs answer for SSE streaming.
class ThinkingStreamSplitter {
public:
    explicit ThinkingStreamSplitter(bool enable_thinking);

    struct Delta {
        std::string reasoning_content;
        std::string content;
    };
    Delta feed(const std::string& piece);
    void finish(Delta& tail);

private:
    bool enable_thinking_ = false;
    enum class Phase { kBeforeThink, kInThink, kInAnswer } phase_ = Phase::kBeforeThink;
    std::string carry_;
    std::string prefix_buffer_;
};

bool parse_chat_messages(const std::string& request_json, std::vector<ChatMessage>& messages, std::string& err);
bool parse_enable_thinking(const std::string& request_json, bool default_value = false);
std::string apply_qwen36_chat_template(const std::vector<ChatMessage>& messages, bool enable_thinking = false);
ParsedAssistantOutput parse_assistant_output(const std::string& raw, bool enable_thinking);

}  // namespace sparkinfer_server
