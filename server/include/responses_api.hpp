#pragma once

// OpenAI Responses API (/v1/responses) request and response translation.
//
// Same structure as ollama_api.hpp and anthropic_api.hpp: pure functions over JSON, no
// engine/HTTP/CUDA dependency. The route rewrites the request into the chat-completions shape,
// hands it to the SAME handler /v1/chat/completions uses, and rewrites the response back.
//
// STATELESS BY CONSTRUCTION. The Responses API can store responses and chain them through
// previous_response_id or a conversation id. This server keeps no response store, so those are
// refused with a message that says so, rather than accepted and answered without the history they
// point at -- a model replying to a conversation it never saw is worse than an error. Clients that
// send the whole conversation in `input` (Codex CLI does, with store=false) work unchanged.
//
// STREAMING IS STATEFUL: output items (reasoning, message, function_call) are opened, filled and
// closed, every event carries a strictly increasing sequence_number, and the terminal
// response.completed repeats the whole response. StreamTranslator carries that state for the life
// of one response; feed it every chunk in order, under one lock.
//
// Reasoning is emitted as a `reasoning` item with `reasoning_text` content -- the open-weight
// convention (gpt-oss) -- and an empty `summary`, since this model produces no separate summary.
//
// Schema source: the openai-python types under src/openai/types/responses/ (ResponseStreamEvent,
// ResponseCreateParams, ResponseInputItemParam).

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace sparkinfer_server {
namespace responses {

struct SseEvent {
    std::string name;      // the SSE `event:` field, e.g. "response.output_text.delta"
    nlohmann::json data;   // carries "type" == name and "sequence_number"
};

std::string format_sse(const std::vector<SseEvent>& events);

// ---- request translation --------------------------------------------------------------------

// Responses body -> chat-completions body. Returns false with `err` set for anything this server
// cannot honour: previous_response_id / conversation / item_reference (nothing is stored),
// background mode, non-function tools, file inputs. `stream` is never copied.
bool request_to_openai(const nlohmann::json& in, nlohmann::json& out, std::string& err);

// The request parameters a Response object echoes back (instructions, tools, tool_choice, text,
// reasoning, temperature, ...), with the API's defaults filled in. store is always false.
nlohmann::json request_echo(const nlohmann::json& in);

// ---- response translation -------------------------------------------------------------------

// A Response object. `usage`, `error` and `incomplete_details` may be null.
nlohmann::json response_object(const std::string& id, long long created_at,
                               const std::string& model, const nlohmann::json& echo,
                               const std::string& status, const nlohmann::json& output,
                               const nlohmann::json& usage, const nlohmann::json& error,
                               const nlohmann::json& incomplete_details);

// Chat-completions usage -> Responses usage.
nlohmann::json usage_from_openai(const nlohmann::json& oai_usage);

// OpenAI chat.completion -> Response object. finish_reason "length" gives status "incomplete"
// with incomplete_details.reason "max_output_tokens".
nlohmann::json openai_to_response(const nlohmann::json& oai, const std::string& id,
                                  long long created_at, const std::string& model,
                                  const nlohmann::json& echo);

// ---- errors ---------------------------------------------------------------------------------

// {"error":{"message":...,"type":...,"param":null,"code":null}}
nlohmann::json error_body(int status, const std::string& message);
std::string openai_error_message(const std::string& body);

// ---- streaming ------------------------------------------------------------------------------

class StreamTranslator {
public:
    StreamTranslator(std::string id, long long created_at, std::string model,
                     nlohmann::json echo, long long input_tokens);

    // Translate ONE chat.completion.chunk (or the handler's {"error":...} chunk) into zero or more
    // Responses events. The usage chunk closes the response with response.completed (or
    // response.incomplete); an error chunk closes it with error + response.failed.
    std::vector<SseEvent> on_chunk(const nlohmann::json& oai);

    bool finished() const { return finished_; }

private:
    SseEvent event(const std::string& type, nlohmann::json payload);
    nlohmann::json snapshot(const std::string& status, const nlohmann::json& usage,
                            const nlohmann::json& error,
                            const nlohmann::json& incomplete_details) const;
    void ensure_started(std::vector<SseEvent>& out);
    void open_item(std::vector<SseEvent>& out, const std::string& kind);
    void close_item(std::vector<SseEvent>& out);
    std::string next_item_id(const char* prefix);

    std::string id_;
    long long created_at_ = 0;
    std::string model_;
    nlohmann::json echo_;
    long long input_tokens_ = 0;
    long long seq_ = 0;
    int item_counter_ = 0;
    bool started_ = false;
    bool finished_ = false;
    nlohmann::json output_ = nlohmann::json::array();   // completed items, in output order
    std::string open_kind_;                             // "message" | "reasoning" | ""
    std::string open_id_;
    std::string open_text_;
    std::string finish_reason_ = "stop";
};

}  // namespace responses
}  // namespace sparkinfer_server
