// Unit tests for OpenAI Responses API translation. Pure JSON, no engine or GPU.
//
// The assertions are about what a Responses client depends on: item types and ids, event order,
// strictly increasing sequence numbers, and a terminal event that repeats the whole response.

#include "responses_api.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace sparkinfer_server::responses;
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

    // ---- string input + instructions ----
    {
        json out;
        CHECK(request_to_openai({{"model", "m"}, {"input", "hi"}, {"instructions", "be brief"},
                                 {"max_output_tokens", 99}, {"temperature", 0.5}, {"stream", true},
                                 {"store", true}, {"parallel_tool_calls", false}}, out, err));
        CHECK(!out.contains("stream"));
        CHECK(out["messages"] == json::array({{{"role", "system"}, {"content", "be brief"}},
                                             {{"role", "user"}, {"content", "hi"}}}));
        CHECK(out["max_tokens"] == 99);
        CHECK(out["temperature"] == 0.5);
        CHECK(out["parallel_tool_calls"] == false);
    }

    // ---- stateless refusals ----
    {
        json out;
        CHECK(!request_to_openai({{"model", "m"}, {"input", "hi"}, {"previous_response_id", "resp_1"}}, out, err));
        CHECK(err.find("does not store") != std::string::npos);
        CHECK(!request_to_openai({{"model", "m"}, {"input", "hi"}, {"conversation", "conv_1"}}, out, err));
        CHECK(!request_to_openai({{"model", "m"}, {"input", "hi"}, {"background", true}}, out, err));
        CHECK(!request_to_openai({{"model", "m"}, {"input", json::array({{{"type", "item_reference"}, {"id", "x"}}})}}, out, err));
        CHECK(!request_to_openai({{"input", "hi"}}, out, err));
        CHECK(!request_to_openai({{"model", "m"}}, out, err));
        // null previous_response_id is how SDKs spell "absent".
        CHECK(request_to_openai({{"model", "m"}, {"input", "hi"}, {"previous_response_id", nullptr}}, out, err));
        CHECK(!request_to_openai({{"model", "m"}, {"input", "hi"},
                                  {"tools", json::array({{{"type", "web_search"}}})}}, out, err));
        CHECK(err.find("web_search") != std::string::npos);
        CHECK(!request_to_openai({{"model", nullptr}, {"input", nullptr}}, out, err));
    }

    // ---- item list: messages, images, reasoning, function calls and outputs ----
    {
        const json in = {{"model", "m"}, {"input", json::array({
            {{"role", "developer"}, {"content", "rules"}},
            {{"type", "message"}, {"role", "user"}, {"content", json::array({
                {{"type", "input_text"}, {"text", "what is this"}},
                {{"type", "input_image"}, {"image_url", "data:image/png;base64,QUJD"}, {"detail", "auto"}}})}},
            {{"type", "reasoning"}, {"id", "rs_1"}, {"summary", json::array()},
             {"content", json::array({{{"type", "reasoning_text"}, {"text", "need a tool"}}})}},
            {{"type", "message"}, {"role", "assistant"}, {"id", "msg_1"}, {"status", "completed"},
             {"content", json::array({{{"type", "output_text"}, {"text", "Checking."}, {"annotations", json::array()}}})}},
            {{"type", "function_call"}, {"call_id", "call_1"}, {"name", "lookup"}, {"arguments", "{\"q\":1}"}},
            {{"type", "function_call"}, {"call_id", "call_2"}, {"name", "lookup"}, {"arguments", "{\"q\":2}"}},
            {{"type", "function_call_output"}, {"call_id", "call_1"}, {"output", "one"}},
            {{"type", "function_call_output"}, {"call_id", "call_2"},
             {"output", json::array({{{"type", "input_text"}, {"text", "two"}}})}},
            {{"type", "reasoning"}, {"id", "rs_2"}, {"summary", json::array({{{"type", "summary_text"}, {"text", "again"}}})}},
            {{"type", "function_call"}, {"call_id", "call_3"}, {"name", "lookup"}, {"arguments", "{}"}}})}};
        json out;
        CHECK(request_to_openai(in, out, err));
        const json& m = out["messages"];
        CHECK(m.size() == 6);
        CHECK(m[0] == json({{"role", "developer"}, {"content", "rules"}}));
        CHECK(m[1]["content"][0] == json({{"type", "text"}, {"text", "what is this"}}));
        CHECK(m[1]["content"][1]["image_url"]["url"] == "data:image/png;base64,QUJD");
        // Reasoning attaches to the assistant turn after it; both calls join that same turn.
        CHECK(m[2]["role"] == "assistant");
        CHECK(m[2]["reasoning_content"] == "need a tool");
        CHECK(m[2]["content"] == "Checking.");
        CHECK(m[2]["tool_calls"].size() == 2);
        CHECK(m[2]["tool_calls"][1]["id"] == "call_2");
        CHECK(m[2]["tool_calls"][1]["function"]["arguments"] == "{\"q\":2}");
        CHECK(m[3] == json({{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "one"}}));
        CHECK(m[4]["content"] == "two");
        // A function call after tool output opens a NEW assistant turn, carrying its summary.
        CHECK(m[5]["role"] == "assistant" && m[5]["content"].is_null());
        CHECK(m[5]["reasoning_content"] == "again");
        CHECK(m[5]["tool_calls"][0]["id"] == "call_3");
    }

    // ---- tools, tool_choice, text.format, reasoning ----
    {
        const json in = {{"model", "m"}, {"input", "x"},
                         {"tools", json::array({{{"type", "function"}, {"name", "f"}, {"description", "does f"},
                                                 {"parameters", {{"type", "object"}}}, {"strict", true}}})},
                         {"tool_choice", {{"type", "function"}, {"name", "f"}}},
                         {"reasoning", {{"effort", "high"}, {"summary", "auto"}}}};
        json out;
        CHECK(request_to_openai(in, out, err));
        CHECK(out["tools"][0] == json({{"type", "function"}, {"function", {{"name", "f"}, {"description", "does f"},
                                                                          {"parameters", {{"type", "object"}}},
                                                                          {"strict", true}}}}));
        CHECK(out["tool_choice"] == json({{"type", "function"}, {"function", {{"name", "f"}}}}));
        CHECK(out["reasoning_effort"] == "high");

        json o2;
        CHECK(request_to_openai({{"model", "m"}, {"input", "x"}, {"tool_choice", "required"},
                                 {"text", {{"format", {{"type", "json_schema"}, {"name", "out"},
                                                       {"schema", {{"type", "object"}}}, {"strict", true}}}}}}, o2, err));
        CHECK(o2["tool_choice"] == "required");
        CHECK(o2["response_format"] == json({{"type", "json_schema"},
                                             {"json_schema", {{"name", "out"}, {"schema", {{"type", "object"}}},
                                                              {"strict", true}}}}));
        json o3;
        CHECK(request_to_openai({{"model", "m"}, {"input", "x"}, {"text", {{"format", {{"type", "text"}}}}}}, o3, err));
        CHECK(!o3.contains("response_format"));
        CHECK(!request_to_openai({{"model", "m"}, {"input", "x"}, {"tool_choice", "sometimes"}}, o3, err));
    }

    // ---- echo defaults ----
    {
        const json e = request_echo({{"model", "m"}, {"input", "x"}, {"instructions", "i"}, {"store", true}});
        CHECK(e["instructions"] == "i");
        CHECK(e["store"] == false);
        CHECK(e["tool_choice"] == "auto");
        CHECK(e["tools"].is_array() && e["tools"].empty());
        CHECK(e["text"]["format"]["type"] == "text");
        CHECK(e["previous_response_id"].is_null());
        CHECK(e["parallel_tool_calls"] == true);
    }

    // ---- non-streaming ----
    {
        const json oai = {{"choices", json::array({{{"index", 0}, {"finish_reason", "tool_calls"},
            {"message", {{"role", "assistant"}, {"content", "Checking."}, {"reasoning_content", "hmm"},
                         {"tool_calls", json::array({{{"id", "call_7"}, {"type", "function"},
                             {"function", {{"name", "f"}, {"arguments", "{\"a\":1}"}}}}})}}}}})},
            {"usage", {{"prompt_tokens", 20}, {"completion_tokens", 5}}}};
        const json r = openai_to_response(oai, "resp_x", 1700000000, "qwen38", request_echo({{"model", "m"}}));
        CHECK(r["object"] == "response" && r["status"] == "completed" && r["id"] == "resp_x");
        CHECK(r["created_at"] == 1700000000 && r["model"] == "qwen38");
        CHECK(r["error"].is_null() && r["incomplete_details"].is_null());
        CHECK(r["output"].size() == 3);
        CHECK(r["output"][0]["type"] == "reasoning");
        CHECK(r["output"][0]["content"][0] == json({{"type", "reasoning_text"}, {"text", "hmm"}}));
        CHECK(r["output"][1]["type"] == "message");
        CHECK(r["output"][1]["content"][0]["type"] == "output_text");
        CHECK(r["output"][1]["content"][0]["text"] == "Checking.");
        CHECK(r["output"][2] == json({{"id", "fc_resp_x_2"}, {"type", "function_call"}, {"status", "completed"},
                                     {"call_id", "call_7"}, {"name", "f"}, {"arguments", "{\"a\":1}"}}));
        CHECK(r["usage"]["input_tokens"] == 20 && r["usage"]["output_tokens"] == 5 && r["usage"]["total_tokens"] == 25);
        CHECK(r["usage"]["input_tokens_details"]["cached_tokens"] == 0);
        CHECK(r["store"] == false);

        const json cut = {{"choices", json::array({{{"finish_reason", "length"},
                                                    {"message", {{"content", "trunc"}}}}})}};
        const json r2 = openai_to_response(cut, "resp_y", 1, "m", json::object());
        CHECK(r2["status"] == "incomplete");
        CHECK(r2["incomplete_details"]["reason"] == "max_output_tokens");

        const json empty = {{"choices", json::array({{{"finish_reason", "stop"}, {"message", {{"content", nullptr}}}}})}};
        const json r3 = openai_to_response(empty, "resp_z", 1, "m", json::object());
        CHECK(r3["output"].size() == 1 && r3["output"][0]["content"][0]["text"] == "");
    }

    // ---- errors ----
    CHECK(error_body(400, "bad")["error"]["type"] == "invalid_request_error");
    CHECK(error_body(500, "boom")["error"]["type"] == "server_error");
    CHECK(error_body(400, "bad")["error"]["message"] == "bad");
    CHECK(openai_error_message("{\"error\":{\"message\":\"ctx\"}}") == "ctx");

    // ---- streaming: reasoning, text, completed ----
    {
        StreamTranslator t("resp_s", 123, "qwen38", request_echo({{"model", "m"}, {"instructions", "i"}}), 10);
        std::vector<SseEvent> all;
        auto feed = [&](const json& c) { auto e = t.on_chunk(c); all.insert(all.end(), e.begin(), e.end()); };
        feed(chunk_delta({{"role", "assistant"}, {"content", nullptr}}));
        feed(chunk_delta({{"reasoning", "r1"}, {"reasoning_content", "r1"}}));
        feed(chunk_delta({{"reasoning_content", "r2"}}));
        feed(chunk_delta({{"content", "He"}}));
        feed(chunk_delta({{"content", "y"}}));
        feed(chunk_delta(json::object(), "stop"));
        feed({{"choices", json::array()}, {"usage", {{"prompt_tokens", 10}, {"completion_tokens", 4}}}});
        CHECK(t.finished());
        CHECK(t.on_chunk(chunk_delta({{"content", "late"}})).empty());

        CHECK(names(all) == std::vector<std::string>({
            "response.created", "response.in_progress",
            "response.output_item.added", "response.content_part.added",
            "response.reasoning_text.delta", "response.reasoning_text.delta",
            "response.reasoning_text.done", "response.content_part.done", "response.output_item.done",
            "response.output_item.added", "response.content_part.added",
            "response.output_text.delta", "response.output_text.delta",
            "response.output_text.done", "response.content_part.done", "response.output_item.done",
            "response.completed"}));
        for (size_t i = 0; i < all.size(); ++i) {
            CHECK(all[i].data["sequence_number"] == (long long)i);
            CHECK(all[i].data["type"] == all[i].name);
        }
        CHECK(all[0].data["response"]["status"] == "in_progress");
        CHECK(all[0].data["response"]["instructions"] == "i");
        CHECK(all[2].data["output_index"] == 0 && all[2].data["item"]["type"] == "reasoning");
        const std::string rs_id = all[2].data["item"]["id"];
        CHECK(all[4].data["item_id"] == rs_id);
        CHECK(all[6].data["text"] == "r1r2");
        CHECK(all[9].data["output_index"] == 1 && all[9].data["item"]["type"] == "message");
        CHECK(all[11].data["delta"] == "He" && all[11].data["logprobs"].is_array());
        CHECK(all[13].data["text"] == "Hey");
        const json& done = all.back().data["response"];
        CHECK(done["status"] == "completed");
        CHECK(done["output"].size() == 2);
        CHECK(done["output"][1]["content"][0]["text"] == "Hey");
        CHECK(done["usage"]["input_tokens"] == 10 && done["usage"]["output_tokens"] == 4);
    }

    // ---- streaming: function calls, incomplete, empty, failure ----
    {
        StreamTranslator t("resp_f", 1, "m", json::object(), 2);
        std::vector<SseEvent> all;
        auto feed = [&](const json& c) { auto e = t.on_chunk(c); all.insert(all.end(), e.begin(), e.end()); };
        feed(chunk_delta({{"role", "assistant"}}));
        feed(chunk_delta({{"tool_calls", json::array({{{"index", 0}, {"id", "call_1"}, {"type", "function"},
                                                       {"function", {{"name", "f"}, {"arguments", "{\"a\":1}"}}}}})}}));
        feed(chunk_delta(json::object(), "tool_calls"));
        feed({{"choices", json::array()}, {"usage", {{"prompt_tokens", 2}, {"completion_tokens", 3}}}});
        CHECK(names(all) == std::vector<std::string>({
            "response.created", "response.in_progress",
            "response.output_item.added", "response.function_call_arguments.delta",
            "response.function_call_arguments.done", "response.output_item.done", "response.completed"}));
        CHECK(all[2].data["item"]["call_id"] == "call_1" && all[2].data["item"]["arguments"] == "");
        CHECK(all[4].data["arguments"] == "{\"a\":1}");
        CHECK(all[5].data["item"]["status"] == "completed");
        CHECK(all[6].data["response"]["output"][0]["type"] == "function_call");
    }
    {
        StreamTranslator t("resp_i", 1, "m", json::object(), 2);
        t.on_chunk(chunk_delta({{"content", "abc"}}));
        t.on_chunk(chunk_delta(json::object(), "length"));
        auto fin = t.on_chunk({{"choices", json::array()}, {"usage", {{"prompt_tokens", 2}, {"completion_tokens", 3}}}});
        CHECK(names(fin) == std::vector<std::string>({"response.incomplete"}));
        CHECK(fin[0].data["response"]["incomplete_details"]["reason"] == "max_output_tokens");
    }
    {
        StreamTranslator t("resp_empty", 1, "m", json::object(), 2);
        t.on_chunk(chunk_delta({{"role", "assistant"}}));
        t.on_chunk(chunk_delta(json::object(), "stop"));
        auto fin = t.on_chunk({{"choices", json::array()}, {"usage", {{"prompt_tokens", 2}, {"completion_tokens", 0}}}});
        CHECK(fin.back().name == "response.completed");
        CHECK(fin.back().data["response"]["output"].size() == 1);
    }
    {
        StreamTranslator t("resp_e", 1, "m", json::object(), 2);
        t.on_chunk(chunk_delta({{"content", "par"}}));
        auto e = t.on_chunk({{"error", {{"message", "device lost"}}}});
        CHECK(names(e) == std::vector<std::string>({"response.output_text.done", "response.content_part.done",
                                                    "response.output_item.done", "error", "response.failed"}));
        CHECK(e[3].data["message"] == "device lost");
        CHECK(e[4].data["response"]["status"] == "failed");
        CHECK(e[4].data["response"]["error"]["message"] == "device lost");
        CHECK(t.finished());
    }

    std::printf("responses_api_test: OK\n");
    return 0;
}
