// Unit tests for Anthropic Messages API translation. Pure JSON, no engine or GPU.
//
// The assertions are about the wire contract an Anthropic client depends on: block types and
// indices, the order of stream events, stop_reason vocabulary, and where tool results land when
// they move from inside a user turn to OpenAI's separate tool messages.

#include "anthropic_api.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace sparkinfer_server::anthropic;
using json = nlohmann::json;

#define CHECK(x) do { if (!(x)) { std::printf("FAIL: %s line %d\n", #x, __LINE__); return 1; } } while (0)

static std::vector<std::string> names(const std::vector<SseEvent>& events) {
    std::vector<std::string> out;
    for (const auto& e : events) out.push_back(e.name);
    return out;
}

static json chunk_delta(const json& delta, const json& finish = nullptr) {
    return {{"object", "chat.completion.chunk"},
            {"choices", json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}})}};
}

int main() {
    std::string err;

    // ---- required fields ----
    {
        json out;
        CHECK(!request_to_openai({{"max_tokens", 10}, {"messages", json::array()}}, out, err));
        CHECK(err.find("model") != std::string::npos);
        CHECK(!request_to_openai({{"model", "m"}, {"messages", json::array()}}, out, err));
        CHECK(err.find("max_tokens") != std::string::npos);
        // count_tokens takes no max_tokens.
        CHECK(request_to_openai({{"model", "m"}, {"messages", json::array()}}, out, err, false));
        CHECK(!out.contains("max_tokens"));
        CHECK(!request_to_openai({{"model", "m"}, {"max_tokens", 0}, {"messages", json::array()}}, out, err));
        CHECK(!request_to_openai(json::array(), out, err));
        // null fields must not throw: a throw out of a handler kills the server.
        CHECK(!request_to_openai({{"model", nullptr}, {"max_tokens", nullptr}, {"messages", nullptr}}, out, err));
    }

    // ---- system + plain text; stream is never copied ----
    {
        const json in = {{"model", "m"}, {"max_tokens", 64}, {"stream", true},
                         {"system", json::array({{{"type", "text"}, {"text", "be brief"}},
                                                 {{"type", "text"}, {"text", "be kind"}}})},
                         {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
                         {"stop_sequences", json::array({"END"})}, {"temperature", 0.2}, {"top_k", 5}};
        json out;
        CHECK(request_to_openai(in, out, err));
        CHECK(!out.contains("stream"));
        CHECK(out["max_tokens"] == 64);
        CHECK(out["messages"].size() == 2);
        CHECK(out["messages"][0]["role"] == "system");
        CHECK(out["messages"][0]["content"] == "be brief\nbe kind");
        CHECK(out["messages"][1] == json({{"role", "user"}, {"content", "hi"}}));
        CHECK(out["stop"] == json::array({"END"}));
        CHECK(out["temperature"] == 0.2 && out["top_k"] == 5);
    }

    // ---- images, tool results, assistant history ----
    {
        const json in = {
            {"model", "m"}, {"max_tokens", 64},
            {"messages", json::array({
                {{"role", "user"}, {"content", json::array({
                    {{"type", "text"}, {"text", "what is this"}},
                    {{"type", "image"}, {"source", {{"type", "base64"}, {"media_type", "image/png"}, {"data", "QUJD"}}}}})}},
                {{"role", "assistant"}, {"content", json::array({
                    {{"type", "thinking"}, {"thinking", "look it up"}, {"signature", "sig"}},
                    {{"type", "text"}, {"text", "Checking."}},
                    {{"type", "tool_use"}, {"id", "toolu_1"}, {"name", "lookup"}, {"input", {{"q", "cat"}}}}})}},
                {{"role", "user"}, {"content", json::array({
                    {{"type", "tool_result"}, {"tool_use_id", "toolu_1"}, {"content", "a cat"}},
                    {{"type", "tool_result"}, {"tool_use_id", "toolu_2"}, {"is_error", true},
                     {"content", json::array({{{"type", "text"}, {"text", "timeout"}}})}},
                    {{"type", "text"}, {"text", "thanks"}}})}}})}};
        json out;
        CHECK(request_to_openai(in, out, err));
        const json& m = out["messages"];
        CHECK(m.size() == 5);
        CHECK(m[0]["content"][1]["type"] == "image_url");
        CHECK(m[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,QUJD");
        CHECK(m[1]["role"] == "assistant");
        CHECK(m[1]["reasoning_content"] == "look it up");
        CHECK(m[1]["content"] == "Checking.");
        CHECK(m[1]["tool_calls"][0]["id"] == "toolu_1");
        CHECK(m[1]["tool_calls"][0]["function"]["name"] == "lookup");
        CHECK(json::parse(m[1]["tool_calls"][0]["function"]["arguments"].get<std::string>()) == json({{"q", "cat"}}));
        // Tool results come out FIRST, as their own messages, ahead of the turn's text.
        CHECK(m[2] == json({{"role", "tool"}, {"tool_call_id", "toolu_1"}, {"content", "a cat"}}));
        CHECK(m[3]["tool_call_id"] == "toolu_2");
        CHECK(m[3]["content"] == "Error: timeout");
        CHECK(m[4]["role"] == "user" && m[4]["content"][0]["text"] == "thanks");
    }

    // An assistant turn that is only a tool call has null content, not "".
    {
        const json in = {{"model", "m"}, {"max_tokens", 8},
                         {"messages", json::array({{{"role", "assistant"}, {"content", json::array({
                             {{"type", "tool_use"}, {"id", "t"}, {"name", "f"}, {"input", json::object()}}})}}})}};
        json out;
        CHECK(request_to_openai(in, out, err));
        CHECK(out["messages"][0]["content"].is_null());
        CHECK(out["messages"][0]["tool_calls"][0]["function"]["arguments"] == "{}");
    }

    // ---- refusals ----
    {
        json out;
        const json doc = {{"model", "m"}, {"max_tokens", 8},
                          {"messages", json::array({{{"role", "user"}, {"content", json::array({
                              {{"type", "document"}, {"source", {{"type", "text"}, {"data", "x"}}}}})}}})}};
        CHECK(!request_to_openai(doc, out, err));
        CHECK(err.find("document") != std::string::npos);
        const json img_in_result = {{"model", "m"}, {"max_tokens", 8},
                                    {"messages", json::array({{{"role", "user"}, {"content", json::array({
                                        {{"type", "tool_result"}, {"tool_use_id", "t"},
                                         {"content", json::array({{{"type", "image"}}})}}})}}})}};
        CHECK(!request_to_openai(img_in_result, out, err));
        const json server_tool = {{"model", "m"}, {"max_tokens", 8}, {"messages", json::array()},
                                  {"tools", json::array({{{"type", "web_search_20250305"}, {"name", "web_search"}}})}};
        CHECK(!request_to_openai(server_tool, out, err));
        CHECK(err.find("server tool") != std::string::npos);
        const json bad_role = {{"model", "m"}, {"max_tokens", 8},
                               {"messages", json::array({{{"role", "system"}, {"content", "x"}}})}};
        CHECK(!request_to_openai(bad_role, out, err));
    }

    // ---- tools, tool_choice, thinking, output_config ----
    {
        const json in = {{"model", "m"}, {"max_tokens", 8}, {"messages", json::array()},
                         {"tools", json::array({{{"name", "get_weather"}, {"description", "weather"},
                                                 {"input_schema", {{"type", "object"},
                                                                   {"properties", {{"city", {{"type", "string"}}}}}}}},
                                                {{"type", "custom"}, {"name", "noop"}}})},
                         {"tool_choice", {{"type", "tool"}, {"name", "get_weather"}, {"disable_parallel_tool_use", true}}},
                         {"thinking", {{"type", "enabled"}, {"budget_tokens", 2048}}},
                         {"output_config", {{"effort", "low"}}}};
        json out;
        CHECK(request_to_openai(in, out, err));
        CHECK(out["tools"].size() == 2);
        CHECK(out["tools"][0]["type"] == "function");
        CHECK(out["tools"][0]["function"]["name"] == "get_weather");
        CHECK(out["tools"][0]["function"]["description"] == "weather");
        CHECK(out["tools"][0]["function"]["parameters"]["properties"].contains("city"));
        CHECK(out["tools"][1]["function"]["parameters"]["type"] == "object");
        CHECK(out["tool_choice"] == json({{"type", "function"}, {"function", {{"name", "get_weather"}}}}));
        CHECK(out["parallel_tool_calls"] == false);
        CHECK(out["enable_thinking"] == true);
        CHECK(out["reasoning_effort"] == "low");

        json o2;
        CHECK(request_to_openai({{"model", "m"}, {"max_tokens", 8}, {"messages", json::array()},
                                 {"tool_choice", {{"type", "any"}}}, {"thinking", {{"type", "disabled"}}}}, o2, err));
        CHECK(o2["tool_choice"] == "required");
        CHECK(o2["enable_thinking"] == false);
        CHECK(!o2.contains("parallel_tool_calls"));
        // Absent thinking leaves the model default alone.
        json o3;
        CHECK(request_to_openai({{"model", "m"}, {"max_tokens", 8}, {"messages", json::array()}}, o3, err));
        CHECK(!o3.contains("enable_thinking"));

        json o4;
        CHECK(request_to_openai({{"model", "m"}, {"max_tokens", 8}, {"messages", json::array()},
                                 {"output_config", {{"format", {{"type", "json_schema"},
                                                                {"schema", {{"type", "object"}}}}}}}}, o4, err));
        CHECK(o4["response_format"]["type"] == "json_schema");
        CHECK(o4["response_format"]["json_schema"]["schema"]["type"] == "object");
    }

    // ---- non-streaming response ----
    {
        const json oai = {{"choices", json::array({{{"index", 0}, {"finish_reason", "tool_calls"},
            {"message", {{"role", "assistant"}, {"content", "Let me check."}, {"reasoning_content", "think"},
                         {"tool_calls", json::array({{{"id", "call_9"}, {"type", "function"},
                             {"function", {{"name", "get_weather"}, {"arguments", "{\"city\":\"Paris\"}"}}}}})}}}}})},
            {"usage", {{"prompt_tokens", 11}, {"completion_tokens", 7}, {"total_tokens", 18}}}};
        const json m = openai_to_message(oai, "msg_1", "qwen38");
        CHECK(m["type"] == "message" && m["role"] == "assistant" && m["id"] == "msg_1" && m["model"] == "qwen38");
        CHECK(m["content"].size() == 3);
        CHECK(m["content"][0] == json({{"type", "thinking"}, {"thinking", "think"}, {"signature", ""}}));
        CHECK(m["content"][1] == json({{"type", "text"}, {"text", "Let me check."}}));
        CHECK(m["content"][2]["type"] == "tool_use");
        CHECK(m["content"][2]["id"] == "call_9");
        CHECK(m["content"][2]["input"] == json({{"city", "Paris"}}));
        CHECK(m["stop_reason"] == "tool_use");
        CHECK(m["stop_sequence"].is_null());
        CHECK(m["usage"]["input_tokens"] == 11 && m["usage"]["output_tokens"] == 7);
    }
    CHECK(stop_reason_from_finish("stop") == "end_turn");
    CHECK(stop_reason_from_finish("length") == "max_tokens");
    CHECK(stop_reason_from_finish("tool_calls") == "tool_use");

    // ---- errors ----
    CHECK(error_type_for_status(400) == "invalid_request_error");
    CHECK(error_type_for_status(401) == "authentication_error");
    CHECK(error_type_for_status(404) == "not_found_error");
    CHECK(error_type_for_status(429) == "rate_limit_error");
    CHECK(error_type_for_status(503) == "api_error");
    CHECK(error_body(400, "bad")["type"] == "error");
    CHECK(error_body(400, "bad")["error"]["message"] == "bad");
    CHECK(openai_error_message("{\"error\":{\"message\":\"context overflow\"}}") == "context overflow");
    CHECK(openai_error_message("{\"error\":\"unauthorized\"}") == "unauthorized");
    CHECK(openai_error_message("not json") == "not json");

    // ---- streaming: thinking, then text, then finish + usage ----
    {
        StreamTranslator t("msg_s", "qwen38", 42);
        std::vector<SseEvent> all;
        auto feed = [&](const json& c) { auto e = t.on_chunk(c); all.insert(all.end(), e.begin(), e.end()); return e; };

        auto e0 = feed(chunk_delta({{"role", "assistant"}, {"content", nullptr}}));
        CHECK(names(e0) == std::vector<std::string>({"message_start"}));
        CHECK(e0[0].data["message"]["usage"]["input_tokens"] == 42);
        CHECK(e0[0].data["message"]["content"].empty());

        feed(chunk_delta({{"reasoning", "a"}, {"reasoning_content", "a"}}));
        feed(chunk_delta({{"reasoning", "b"}, {"reasoning_content", "b"}}));
        feed(chunk_delta({{"content", "Hel"}}));
        feed(chunk_delta({{"content", "lo"}}));
        feed(chunk_delta({{"content", ""}}));   // logprobs-only delta: nothing to say
        feed(chunk_delta(json::object(), "stop"));
        CHECK(!t.finished());
        feed({{"choices", json::array()}, {"usage", {{"prompt_tokens", 42}, {"completion_tokens", 5}}}});
        CHECK(t.finished());
        CHECK(t.on_chunk(chunk_delta({{"content", "late"}})).empty());

        CHECK(names(all) == std::vector<std::string>({
            "message_start",
            "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
            "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
            "message_delta", "message_stop"}));
        CHECK(all[1].data["index"] == 0 && all[1].data["content_block"]["type"] == "thinking");
        CHECK(all[2].data["delta"] == json({{"type", "thinking_delta"}, {"thinking", "a"}}));
        CHECK(all[4].data["index"] == 0);
        CHECK(all[5].data["index"] == 1 && all[5].data["content_block"] == json({{"type", "text"}, {"text", ""}}));
        CHECK(all[6].data["delta"] == json({{"type", "text_delta"}, {"text", "Hel"}}));
        CHECK(all[9].data["delta"]["stop_reason"] == "end_turn");
        CHECK(all[9].data["usage"]["output_tokens"] == 5);
        for (const auto& e : all) CHECK(e.data["type"] == e.name);

        const std::string wire = format_sse({all[0]});
        CHECK(wire.rfind("event: message_start\ndata: {", 0) == 0);
        CHECK(wire.size() > 2 && wire.substr(wire.size() - 2) == "\n\n");
    }

    // ---- streaming: buffered tool call path ----
    {
        StreamTranslator t("msg_t", "qwen38", 3);
        std::vector<SseEvent> all;
        auto feed = [&](const json& c) { auto e = t.on_chunk(c); all.insert(all.end(), e.begin(), e.end()); };
        feed(chunk_delta({{"role", "assistant"}, {"content", nullptr}}));
        feed(chunk_delta({{"content", "Checking."}}));
        feed(chunk_delta({{"tool_calls", json::array({
            {{"index", 0}, {"id", "call_1"}, {"type", "function"}, {"function", {{"name", "a"}, {"arguments", "{\"x\":1}"}}}},
            {{"index", 1}, {"id", "call_2"}, {"type", "function"}, {"function", {{"name", "b"}, {"arguments", "{}"}}}}})}}));
        feed(chunk_delta(json::object(), "tool_calls"));
        feed({{"choices", json::array()}, {"usage", {{"prompt_tokens", 3}, {"completion_tokens", 9}}}});
        CHECK(names(all) == std::vector<std::string>({
            "message_start",
            "content_block_start", "content_block_delta", "content_block_stop",
            "content_block_start", "content_block_delta", "content_block_stop",
            "content_block_start", "content_block_delta", "content_block_stop",
            "message_delta", "message_stop"}));
        CHECK(all[4].data["index"] == 1);
        CHECK(all[4].data["content_block"] == json({{"type", "tool_use"}, {"id", "call_1"}, {"name", "a"},
                                                    {"input", json::object()}}));
        CHECK(all[5].data["delta"] == json({{"type", "input_json_delta"}, {"partial_json", "{\"x\":1}"}}));
        CHECK(all[7].data["index"] == 2 && all[7].data["content_block"]["name"] == "b");
        CHECK(all[10].data["delta"]["stop_reason"] == "tool_use");
    }

    // ---- streaming: length, and an error mid-stream ----
    {
        StreamTranslator t("msg_l", "m", 1);
        t.on_chunk(chunk_delta({{"content", "abc"}}));
        auto end = t.on_chunk(chunk_delta(json::object(), "length"));
        CHECK(names(end) == std::vector<std::string>({"content_block_stop"}));
        auto fin = t.on_chunk({{"choices", json::array()}, {"usage", {{"prompt_tokens", 1}, {"completion_tokens", 3}}}});
        CHECK(fin[0].data["delta"]["stop_reason"] == "max_tokens");
    }
    {
        StreamTranslator t("msg_e", "m", 1);
        t.on_chunk(chunk_delta({{"role", "assistant"}}));
        auto e = t.on_chunk({{"error", {{"message", "device lost"}}}});
        CHECK(names(e) == std::vector<std::string>({"error"}));
        CHECK(e[0].data["error"]["type"] == "api_error");
        CHECK(e[0].data["error"]["message"] == "device lost");
        CHECK(t.finished());
        CHECK(t.on_chunk({{"choices", json::array()}, {"usage", {{"prompt_tokens", 1}}}}).empty());
    }

    std::printf("anthropic_api_test: OK\n");
    return 0;
}
