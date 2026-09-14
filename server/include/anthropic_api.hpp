#pragma once

// Anthropic Messages API (/v1/messages, /v1/messages/count_tokens) request and response
// translation.
//
// Same structure as ollama_api.hpp: pure functions over JSON, no engine/HTTP/CUDA dependency, so
// the wire contract is unit-testable on a box with no GPU. The route rewrites the request into
// the OpenAI chat shape, hands it to the SAME handler /v1/chat/completions uses, and rewrites the
// response back. Nothing here generates, times, or samples anything.
//
// STREAMING IS STATEFUL, unlike Ollama. An OpenAI chunk maps to an Ollama line on its own, but an
// Anthropic stream is a sequence of numbered content blocks that must be opened and closed:
// message_start, then per block content_block_start / _delta... / _stop, then message_delta
// (stop_reason + usage) and message_stop. Whether a reasoning piece continues the open thinking
// block or opens a new one depends on what came before it, so StreamTranslator carries that state
// for the life of one response. The caller must feed it every chunk in order, under one lock.
//
// What does not survive the translation, on purpose:
//   - stop_reason "stop_sequence": the OpenAI finish_reason "stop" does not say whether EOS or a
//     stop sequence ended generation, so both report "end_turn".
//   - thinking signatures: this model has no signature to give. Thinking blocks carry an empty
//     `signature`, and a thinking block sent back in history is fed to the model as reasoning.
//   - server tools (web_search, bash, code_execution, ...): Anthropic runs those itself. Only
//     custom tools exist here, so the others are refused rather than silently offered to nothing.
//
// Schema sources: https://platform.claude.com/docs/en/api/messages and
// https://platform.claude.com/docs/en/build-with-claude/streaming

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace sparkinfer_server {
namespace anthropic {

struct SseEvent {
    std::string name;      // the SSE `event:` field, e.g. "content_block_delta"
    nlohmann::json data;   // the `data:` payload; always carries "type" == name
};

// "event: <name>\ndata: <json>\n\n" for each event, concatenated. Empty for no events.
std::string format_sse(const std::vector<SseEvent>& events);

// ---- request translation --------------------------------------------------------------------

// Anthropic /v1/messages body -> OpenAI /v1/chat/completions body. Returns false with `err` set
// for anything this server cannot honour: missing required fields, server tools, document or
// search_result blocks, images inside a tool_result. `stream` is never copied -- the route decides.
//
// require_max_tokens=false is for /v1/messages/count_tokens, whose body has no max_tokens.
bool request_to_openai(const nlohmann::json& in, nlohmann::json& out, std::string& err,
                       bool require_max_tokens = true);

// ---- response translation -------------------------------------------------------------------

// OpenAI finish_reason -> Anthropic stop_reason ("stop" -> "end_turn", "length" -> "max_tokens",
// "tool_calls" -> "tool_use").
std::string stop_reason_from_finish(const std::string& finish_reason);

// OpenAI chat.completion -> Anthropic message object.
nlohmann::json openai_to_message(const nlohmann::json& oai, const std::string& id,
                                 const std::string& model);

// ---- errors ---------------------------------------------------------------------------------

// HTTP status -> Anthropic error.type. Clients switch on this (Claude Code retries
// rate_limit_error and overloaded_error), so every status maps to a documented value.
std::string error_type_for_status(int status);

// {"type":"error","error":{"type":...,"message":...}}
nlohmann::json error_body(int status, const std::string& message);

// The human-readable message out of an OpenAI-shaped error body ({"error":{"message":...}} or
// {"error":"..."}); the raw body when it is neither, so nothing the handler said is lost.
std::string openai_error_message(const std::string& body);

// ---- streaming ------------------------------------------------------------------------------

class StreamTranslator {
public:
    // input_tokens is the prompt length, reported in message_start before any usage chunk exists.
    StreamTranslator(std::string id, std::string model, long long input_tokens);

    // Translate ONE OpenAI chat.completion.chunk (or the handler's {"error":...} chunk) into zero
    // or more Anthropic events. The usage chunk closes the message: message_delta + message_stop.
    // Anything fed after the message has closed yields nothing.
    std::vector<SseEvent> on_chunk(const nlohmann::json& oai);

    bool finished() const { return finished_; }

private:
    void ensure_started(std::vector<SseEvent>& out);
    void open_block(std::vector<SseEvent>& out, const std::string& kind, nlohmann::json block);
    void close_block(std::vector<SseEvent>& out);

    std::string id_;
    std::string model_;
    long long input_tokens_ = 0;
    bool started_ = false;
    bool finished_ = false;
    int next_index_ = 0;
    int open_index_ = -1;
    std::string open_kind_;              // "text" | "thinking" | "" when no block is open
    std::string stop_reason_ = "end_turn";
};

}  // namespace anthropic
}  // namespace sparkinfer_server
