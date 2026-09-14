#include "responses_api.hpp"

#include <utility>

namespace sparkinfer_server {
namespace responses {

using json = nlohmann::json;

namespace {

// Null-safe: the body is client-controlled, and a throw out of a handler kills the server.
std::string str_field(const json& j, const char* key, const std::string& dflt = "") {
    if (!j.is_object()) return dflt;
    auto it = j.find(key);
    return (it != j.end() && it->is_string()) ? it->get<std::string>() : dflt;
}

bool present(const json& j, const char* key) {
    if (!j.is_object()) return false;
    auto it = j.find(key);
    return it != j.end() && !it->is_null();
}

bool set_error(std::string& err, const std::string& message) {
    err = message;
    return false;
}

std::string error_type_for_status(int status) {
    if (status == 401) return "authentication_error";
    if (status == 404) return "not_found_error";
    if (status == 429) return "rate_limit_error";
    if (status >= 400 && status < 500) return "invalid_request_error";
    return "server_error";
}

// Text of a reasoning input item: its reasoning_text content when present, else its summary.
std::string reasoning_item_text(const json& item) {
    std::string text;
    for (const char* key : {"content", "summary"}) {
        if (!item.contains(key) || !item[key].is_array()) continue;
        for (const json& part : item[key]) {
            if (!text.empty()) text += "\n";
            text += str_field(part, "text");
        }
        if (!text.empty()) break;
    }
    return text;
}

}  // namespace

std::string format_sse(const std::vector<SseEvent>& events) {
    std::string out;
    for (const SseEvent& e : events) {
        out += "event: ";
        out += e.name;
        out += "\ndata: ";
        out += e.data.dump();
        out += "\n\n";
    }
    return out;
}

bool request_to_openai(const json& in, json& out, std::string& err) {
    if (!in.is_object()) return set_error(err, "request body must be a JSON object");
    const std::string model = str_field(in, "model");
    if (model.empty()) return set_error(err, "model is required");
    if (present(in, "previous_response_id"))
        return set_error(err, "previous_response_id is not supported: this server does not store "
                              "responses. Send the whole conversation in `input` (with store=false).");
    if (present(in, "conversation"))
        return set_error(err, "conversation is not supported: this server does not store "
                              "conversations. Send the whole conversation in `input`.");
    if (in.contains("background") && in["background"].is_boolean() && in["background"].get<bool>())
        return set_error(err, "background responses are not supported");

    out = json::object();
    out["model"] = model;
    json messages = json::array();

    const std::string instructions = str_field(in, "instructions");
    if (!instructions.empty()) messages.push_back({{"role", "system"}, {"content", instructions}});

    auto input = in.find("input");
    if (input == in.end() || input->is_null()) return set_error(err, "input is required");
    if (input->is_string()) {
        messages.push_back({{"role", "user"}, {"content", input->get<std::string>()}});
    } else if (input->is_array()) {
        // A reasoning item precedes the assistant turn it belongs to. It is held here and attached
        // to whichever assistant message comes next -- a text reply or a function call.
        std::string pending_reasoning;
        // True while the last message is an assistant turn a following function_call may join.
        bool assistant_open = false;
        for (size_t i = 0; i < input->size(); ++i) {
            const json& item = (*input)[i];
            const std::string where = "input[" + std::to_string(i) + "]";
            if (!item.is_object()) return set_error(err, where + " must be an object");
            const std::string type = str_field(item, "type", "message");

            if (type == "message") {
                std::string role = str_field(item, "role");
                if (role != "user" && role != "assistant" && role != "system" && role != "developer")
                    return set_error(err, where + ".role must be user, assistant, system, or developer");
                auto content = item.find("content");
                if (content == item.end() || content->is_null())
                    return set_error(err, where + ".content is required");
                if (role == "assistant") {
                    std::string text;
                    if (content->is_string()) {
                        text = content->get<std::string>();
                    } else if (content->is_array()) {
                        for (const json& part : *content) {
                            const std::string ptype = str_field(part, "type");
                            if (ptype == "output_text" || ptype == "input_text") text += str_field(part, "text");
                            else if (ptype == "refusal") text += str_field(part, "refusal");
                            else return set_error(err, where + ": assistant content part '" + ptype + "' is not supported");
                        }
                    } else {
                        return set_error(err, where + ".content must be a string or an array");
                    }
                    json msg = {{"role", "assistant"}, {"content", text}};
                    if (!pending_reasoning.empty()) msg["reasoning_content"] = pending_reasoning;
                    pending_reasoning.clear();
                    messages.push_back(std::move(msg));
                    assistant_open = true;
                    continue;
                }
                assistant_open = false;
                if (content->is_string()) {
                    messages.push_back({{"role", role}, {"content", content->get<std::string>()}});
                    continue;
                }
                if (!content->is_array()) return set_error(err, where + ".content must be a string or an array");
                json parts = json::array();
                for (const json& part : *content) {
                    const std::string ptype = str_field(part, "type");
                    if (ptype == "input_text" || ptype == "output_text") {
                        parts.push_back({{"type", "text"}, {"text", str_field(part, "text")}});
                    } else if (ptype == "input_image") {
                        const std::string url = str_field(part, "image_url");
                        if (url.empty())
                            return set_error(err, where + ": input_image needs image_url (file_id is not supported)");
                        parts.push_back({{"type", "image_url"}, {"image_url", {{"url", url}}}});
                    } else {
                        return set_error(err, where + ": content part '" + ptype + "' is not supported");
                    }
                }
                messages.push_back({{"role", role}, {"content", parts}});
            } else if (type == "reasoning") {
                const std::string text = reasoning_item_text(item);
                if (!pending_reasoning.empty() && !text.empty()) pending_reasoning += "\n";
                pending_reasoning += text;
            } else if (type == "function_call") {
                const std::string call_id = str_field(item, "call_id");
                const std::string name = str_field(item, "name");
                if (call_id.empty() || name.empty())
                    return set_error(err, where + ": function_call needs call_id and name");
                json call = {{"id", call_id}, {"type", "function"},
                             {"function", {{"name", name}, {"arguments", str_field(item, "arguments", "{}")}}}};
                if (assistant_open && !messages.empty()) {
                    json& last = messages.back();
                    if (!last.contains("tool_calls")) last["tool_calls"] = json::array();
                    if (last["content"].is_string() && last["content"].get<std::string>().empty())
                        last["content"] = nullptr;
                    last["tool_calls"].push_back(std::move(call));
                    if (!pending_reasoning.empty()) {
                        last["reasoning_content"] = str_field(last, "reasoning_content") + pending_reasoning;
                        pending_reasoning.clear();
                    }
                } else {
                    json msg = {{"role", "assistant"}, {"content", nullptr},
                                {"tool_calls", json::array({std::move(call)})}};
                    if (!pending_reasoning.empty()) msg["reasoning_content"] = pending_reasoning;
                    pending_reasoning.clear();
                    messages.push_back(std::move(msg));
                    assistant_open = true;
                }
            } else if (type == "function_call_output") {
                const std::string call_id = str_field(item, "call_id");
                if (call_id.empty()) return set_error(err, where + ": function_call_output needs call_id");
                std::string text;
                auto output = item.find("output");
                if (output != item.end() && output->is_string()) {
                    text = output->get<std::string>();
                } else if (output != item.end() && output->is_array()) {
                    for (const json& part : *output) {
                        if (str_field(part, "type") != "input_text")
                            return set_error(err, where + ": function_call_output supports text output only");
                        if (!text.empty()) text += "\n";
                        text += str_field(part, "text");
                    }
                }
                messages.push_back({{"role", "tool"}, {"tool_call_id", call_id}, {"content", text}});
                assistant_open = false;
            } else if (type == "item_reference") {
                return set_error(err, where + ": item_reference points at a stored item, and this "
                                              "server does not store responses");
            } else {
                return set_error(err, where + ": input item type '" + type + "' is not supported");
            }
        }
    } else {
        return set_error(err, "input must be a string or an array");
    }
    out["messages"] = messages;

    if (auto mot = in.find("max_output_tokens"); mot != in.end() && mot->is_number_integer())
        out["max_tokens"] = *mot;
    for (const char* key : {"temperature", "top_p"}) {
        auto it = in.find(key);
        if (it != in.end() && it->is_number()) out[key] = *it;
    }
    if (auto ptc = in.find("parallel_tool_calls"); ptc != in.end() && ptc->is_boolean())
        out["parallel_tool_calls"] = *ptc;

    if (auto tools = in.find("tools"); tools != in.end() && !tools->is_null()) {
        if (!tools->is_array()) return set_error(err, "tools must be an array");
        json oai_tools = json::array();
        for (const json& tool : *tools) {
            const std::string type = str_field(tool, "type");
            if (type != "function")
                return set_error(err, "tool type '" + type + "' is not supported: only function tools are");
            const std::string name = str_field(tool, "name");
            if (name.empty()) return set_error(err, "every function tool needs a name");
            json fn = {{"name", name},
                       {"parameters", present(tool, "parameters") && tool["parameters"].is_object()
                                          ? tool["parameters"]
                                          : json{{"type", "object"}, {"properties", json::object()}}}};
            const std::string description = str_field(tool, "description");
            if (!description.empty()) fn["description"] = description;
            if (tool.contains("strict") && tool["strict"].is_boolean()) fn["strict"] = tool["strict"];
            oai_tools.push_back({{"type", "function"}, {"function", fn}});
        }
        out["tools"] = oai_tools;
    }

    if (auto tc = in.find("tool_choice"); tc != in.end() && !tc->is_null()) {
        if (tc->is_string()) {
            const std::string v = tc->get<std::string>();
            if (v != "auto" && v != "none" && v != "required")
                return set_error(err, "tool_choice must be auto, none, required, or a function");
            out["tool_choice"] = v;
        } else if (tc->is_object() && str_field(*tc, "type") == "function" && !str_field(*tc, "name").empty()) {
            out["tool_choice"] = {{"type", "function"}, {"function", {{"name", str_field(*tc, "name")}}}};
        } else {
            return set_error(err, "tool_choice must be auto, none, required, or {\"type\":\"function\",\"name\":...}");
        }
    }

    if (auto text = in.find("text"); text != in.end() && text->is_object()) {
        if (auto fmt = text->find("format"); fmt != text->end() && fmt->is_object()) {
            const std::string type = str_field(*fmt, "type");
            if (type == "json_object") {
                out["response_format"] = {{"type", "json_object"}};
            } else if (type == "json_schema") {
                if (!present(*fmt, "schema")) return set_error(err, "text.format.schema is required for json_schema");
                json js = {{"name", str_field(*fmt, "name", "output")}, {"schema", (*fmt)["schema"]}};
                if (fmt->contains("strict") && (*fmt)["strict"].is_boolean()) js["strict"] = (*fmt)["strict"];
                out["response_format"] = {{"type", "json_schema"}, {"json_schema", js}};
            } else if (type != "text" && !type.empty()) {
                return set_error(err, "text.format.type '" + type + "' is not supported");
            }
        }
    }

    if (auto reasoning = in.find("reasoning"); reasoning != in.end() && reasoning->is_object()) {
        const std::string effort = str_field(*reasoning, "effort");
        if (!effort.empty()) out["reasoning_effort"] = effort;   // validated by the chat parser
    }
    return true;
}

json request_echo(const json& in) {
    auto pick = [&](const char* key, json dflt) -> json {
        return present(in, key) ? in[key] : std::move(dflt);
    };
    return {{"instructions", pick("instructions", nullptr)},
            {"max_output_tokens", pick("max_output_tokens", nullptr)},
            {"metadata", pick("metadata", json::object())},
            {"parallel_tool_calls", pick("parallel_tool_calls", true)},
            {"previous_response_id", nullptr},
            {"reasoning", pick("reasoning", json{{"effort", nullptr}, {"summary", nullptr}})},
            {"store", false},
            {"temperature", pick("temperature", 1.0)},
            {"text", pick("text", json{{"format", {{"type", "text"}}}})},
            {"tool_choice", pick("tool_choice", "auto")},
            {"tools", pick("tools", json::array())},
            {"top_p", pick("top_p", 1.0)},
            {"truncation", "disabled"}};
}

json response_object(const std::string& id, long long created_at, const std::string& model,
                     const json& echo, const std::string& status, const json& output,
                     const json& usage, const json& error, const json& incomplete_details) {
    json r = {{"id", id},
              {"object", "response"},
              {"created_at", created_at},
              {"status", status},
              {"error", error},
              {"incomplete_details", incomplete_details},
              {"model", model},
              {"output", output},
              {"usage", usage}};
    if (echo.is_object())
        for (auto it = echo.begin(); it != echo.end(); ++it) r[it.key()] = it.value();
    return r;
}

json usage_from_openai(const json& u) {
    const long long in = u.is_object() ? u.value("prompt_tokens", 0LL) : 0LL;
    const long long out = u.is_object() ? u.value("completion_tokens", 0LL) : 0LL;
    return {{"input_tokens", in},
            {"input_tokens_details", {{"cached_tokens", 0}, {"cache_write_tokens", 0}}},
            {"output_tokens", out},
            {"output_tokens_details", {{"reasoning_tokens", 0}}},
            {"total_tokens", in + out}};
}

json openai_to_response(const json& oai, const std::string& id, long long created_at,
                        const std::string& model, const json& echo) {
    json output = json::array();
    std::string finish = "stop";
    int n = 0;
    auto item_id = [&](const char* prefix) { return std::string(prefix) + id + "_" + std::to_string(n++); };
    if (oai.contains("choices") && oai["choices"].is_array() && !oai["choices"].empty()) {
        const json& choice = oai["choices"][0];
        finish = str_field(choice, "finish_reason", "stop");
        const json msg = choice.contains("message") ? choice["message"] : json::object();
        std::string reasoning = str_field(msg, "reasoning_content");
        if (reasoning.empty()) reasoning = str_field(msg, "reasoning");
        if (!reasoning.empty())
            output.push_back({{"id", item_id("rs_")}, {"type", "reasoning"}, {"summary", json::array()},
                              {"content", json::array({{{"type", "reasoning_text"}, {"text", reasoning}}})},
                              {"status", "completed"}});
        const std::string text = str_field(msg, "content");
        const bool has_calls = msg.contains("tool_calls") && msg["tool_calls"].is_array() && !msg["tool_calls"].empty();
        // An empty reply still gets a message item, so `output_text` on the client is "" rather
        // than a response with nothing in it to read.
        if (!text.empty() || !has_calls)
            output.push_back({{"id", item_id("msg_")}, {"type", "message"}, {"status", "completed"},
                              {"role", "assistant"},
                              {"content", json::array({{{"type", "output_text"}, {"text", text},
                                                        {"annotations", json::array()},
                                                        {"logprobs", json::array()}}})}});
        if (has_calls) {
            for (const json& call : msg["tool_calls"]) {
                const json fn = call.contains("function") ? call["function"] : json::object();
                output.push_back({{"id", item_id("fc_")}, {"type", "function_call"}, {"status", "completed"},
                                  {"call_id", str_field(call, "id")}, {"name", str_field(fn, "name")},
                                  {"arguments", str_field(fn, "arguments", "{}")}});
            }
        }
    }
    const bool incomplete = finish == "length";
    return response_object(id, created_at, model, echo, incomplete ? "incomplete" : "completed", output,
                           usage_from_openai(oai.contains("usage") ? oai["usage"] : json::object()),
                           nullptr,
                           incomplete ? json{{"reason", "max_output_tokens"}} : json(nullptr));
}

json error_body(int status, const std::string& message) {
    return {{"error", {{"message", message}, {"type", error_type_for_status(status)},
                       {"param", nullptr}, {"code", nullptr}}}};
}

std::string openai_error_message(const std::string& body) {
    const json j = json::parse(body, nullptr, false);
    if (!j.is_discarded() && j.is_object() && j.contains("error")) {
        const json& e = j["error"];
        if (e.is_string()) return e.get<std::string>();
        const std::string m = str_field(e, "message");
        if (!m.empty()) return m;
    }
    return body;
}

StreamTranslator::StreamTranslator(std::string id, long long created_at, std::string model,
                                   json echo, long long input_tokens)
    : id_(std::move(id)), created_at_(created_at), model_(std::move(model)),
      echo_(std::move(echo)), input_tokens_(input_tokens) {}

SseEvent StreamTranslator::event(const std::string& type, json payload) {
    payload["type"] = type;
    payload["sequence_number"] = seq_++;
    return {type, std::move(payload)};
}

json StreamTranslator::snapshot(const std::string& status, const json& usage, const json& error,
                                const json& incomplete_details) const {
    return response_object(id_, created_at_, model_, echo_, status, output_, usage, error,
                           incomplete_details);
}

std::string StreamTranslator::next_item_id(const char* prefix) {
    // Derived from the response id, so ids are unique within the response and deterministic.
    return std::string(prefix) + id_ + "_" + std::to_string(item_counter_++);
}

void StreamTranslator::ensure_started(std::vector<SseEvent>& out) {
    if (started_) return;
    started_ = true;
    out.push_back(event("response.created", {{"response", snapshot("in_progress", nullptr, nullptr, nullptr)}}));
    out.push_back(event("response.in_progress", {{"response", snapshot("in_progress", nullptr, nullptr, nullptr)}}));
}

void StreamTranslator::open_item(std::vector<SseEvent>& out, const std::string& kind) {
    close_item(out);
    open_kind_ = kind;
    open_text_.clear();
    const int index = (int)output_.size();
    if (kind == "reasoning") {
        open_id_ = next_item_id("rs_");
        out.push_back(event("response.output_item.added",
                            {{"output_index", index},
                             {"item", {{"id", open_id_}, {"type", "reasoning"}, {"summary", json::array()},
                                       {"content", json::array()}, {"status", "in_progress"}}}}));
        out.push_back(event("response.content_part.added",
                            {{"item_id", open_id_}, {"output_index", index}, {"content_index", 0},
                             {"part", {{"type", "reasoning_text"}, {"text", ""}}}}));
    } else {
        open_id_ = next_item_id("msg_");
        out.push_back(event("response.output_item.added",
                            {{"output_index", index},
                             {"item", {{"id", open_id_}, {"type", "message"}, {"status", "in_progress"},
                                       {"role", "assistant"}, {"content", json::array()}}}}));
        out.push_back(event("response.content_part.added",
                            {{"item_id", open_id_}, {"output_index", index}, {"content_index", 0},
                             {"part", {{"type", "output_text"}, {"text", ""}, {"annotations", json::array()},
                                       {"logprobs", json::array()}}}}));
    }
}

void StreamTranslator::close_item(std::vector<SseEvent>& out) {
    if (open_kind_.empty()) return;
    const int index = (int)output_.size();
    json item;
    if (open_kind_ == "reasoning") {
        const json part = {{"type", "reasoning_text"}, {"text", open_text_}};
        out.push_back(event("response.reasoning_text.done",
                            {{"item_id", open_id_}, {"output_index", index}, {"content_index", 0},
                             {"text", open_text_}}));
        out.push_back(event("response.content_part.done",
                            {{"item_id", open_id_}, {"output_index", index}, {"content_index", 0}, {"part", part}}));
        item = {{"id", open_id_}, {"type", "reasoning"}, {"summary", json::array()},
                {"content", json::array({part})}, {"status", "completed"}};
    } else {
        const json part = {{"type", "output_text"}, {"text", open_text_}, {"annotations", json::array()},
                           {"logprobs", json::array()}};
        out.push_back(event("response.output_text.done",
                            {{"item_id", open_id_}, {"output_index", index}, {"content_index", 0},
                             {"text", open_text_}, {"logprobs", json::array()}}));
        out.push_back(event("response.content_part.done",
                            {{"item_id", open_id_}, {"output_index", index}, {"content_index", 0}, {"part", part}}));
        item = {{"id", open_id_}, {"type", "message"}, {"status", "completed"}, {"role", "assistant"},
                {"content", json::array({part})}};
    }
    out.push_back(event("response.output_item.done", {{"output_index", index}, {"item", item}}));
    output_.push_back(std::move(item));
    open_kind_.clear();
    open_id_.clear();
    open_text_.clear();
}

std::vector<SseEvent> StreamTranslator::on_chunk(const json& oai) {
    std::vector<SseEvent> out;
    if (finished_ || !oai.is_object()) return out;
    ensure_started(out);

    if (oai.contains("error")) {
        const std::string message = oai["error"].is_string()
                                        ? oai["error"].get<std::string>()
                                        : str_field(oai["error"], "message", "generation failed");
        close_item(out);
        out.push_back(event("error", {{"code", "server_error"}, {"message", message}, {"param", nullptr}}));
        out.push_back(event("response.failed",
                            {{"response", snapshot("failed", nullptr,
                                                   json{{"code", "server_error"}, {"message", message}},
                                                   nullptr)}}));
        finished_ = true;
        return out;
    }

    if (oai.contains("choices") && oai["choices"].is_array() && !oai["choices"].empty()) {
        const json& choice = oai["choices"][0];
        const json delta = choice.contains("delta") && choice["delta"].is_object() ? choice["delta"]
                                                                                   : json::object();
        std::string reasoning = str_field(delta, "reasoning_content");
        if (reasoning.empty()) reasoning = str_field(delta, "reasoning");
        if (!reasoning.empty()) {
            if (open_kind_ != "reasoning") open_item(out, "reasoning");
            open_text_ += reasoning;
            out.push_back(event("response.reasoning_text.delta",
                                {{"item_id", open_id_}, {"output_index", (int)output_.size()},
                                 {"content_index", 0}, {"delta", reasoning}}));
        }
        const std::string text = str_field(delta, "content");
        if (!text.empty()) {
            if (open_kind_ != "message") open_item(out, "message");
            open_text_ += text;
            out.push_back(event("response.output_text.delta",
                                {{"item_id", open_id_}, {"output_index", (int)output_.size()},
                                 {"content_index", 0}, {"delta", text}, {"logprobs", json::array()}}));
        }
        if (delta.contains("tool_calls") && delta["tool_calls"].is_array()) {
            close_item(out);
            for (const json& call : delta["tool_calls"]) {
                const json fn = call.contains("function") ? call["function"] : json::object();
                const std::string item_id = next_item_id("fc_");
                const std::string args = str_field(fn, "arguments", "{}");
                const int index = (int)output_.size();
                json item = {{"id", item_id}, {"type", "function_call"}, {"status", "in_progress"},
                             {"call_id", str_field(call, "id")}, {"name", str_field(fn, "name")},
                             {"arguments", ""}};
                out.push_back(event("response.output_item.added", {{"output_index", index}, {"item", item}}));
                out.push_back(event("response.function_call_arguments.delta",
                                    {{"item_id", item_id}, {"output_index", index}, {"delta", args}}));
                out.push_back(event("response.function_call_arguments.done",
                                    {{"item_id", item_id}, {"output_index", index}, {"arguments", args}}));
                item["arguments"] = args;
                item["status"] = "completed";
                out.push_back(event("response.output_item.done", {{"output_index", index}, {"item", item}}));
                output_.push_back(std::move(item));
            }
        }
        const std::string finish = str_field(choice, "finish_reason");
        if (!finish.empty()) {
            close_item(out);
            finish_reason_ = finish;
        }
    }

    if (oai.contains("usage") && oai["usage"].is_object()) {
        close_item(out);
        // Same rule as the non-streaming path: an empty reply still ends with a message item.
        if (output_.empty()) {
            open_item(out, "message");
            close_item(out);
        }
        json usage = usage_from_openai(oai["usage"]);
        if (usage["input_tokens"].get<long long>() == 0 && input_tokens_ > 0) {
            usage["input_tokens"] = input_tokens_;
            usage["total_tokens"] = input_tokens_ + usage["output_tokens"].get<long long>();
        }
        if (finish_reason_ == "length") {
            out.push_back(event("response.incomplete",
                                {{"response", snapshot("incomplete", usage, nullptr,
                                                       json{{"reason", "max_output_tokens"}})}}));
        } else {
            out.push_back(event("response.completed",
                                {{"response", snapshot("completed", usage, nullptr, nullptr)}}));
        }
        finished_ = true;
    }
    return out;
}

}  // namespace responses
}  // namespace sparkinfer_server
