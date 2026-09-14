#include "anthropic_api.hpp"

#include <utility>

namespace sparkinfer_server {
namespace anthropic {

using json = nlohmann::json;

namespace {

// Null-safe readers. The body is client-controlled: value(key, default) throws on a present-but-
// null key, and a throw out of a request handler takes the whole server down with it.
std::string str_field(const json& j, const char* key, const std::string& dflt = "") {
    if (!j.is_object()) return dflt;
    auto it = j.find(key);
    return (it != j.end() && it->is_string()) ? it->get<std::string>() : dflt;
}

bool set_error(std::string& err, const std::string& message) {
    err = message;
    return false;
}

// tool_result content is a string or an array of blocks. Text blocks are joined; anything else
// (an image, a document) cannot be carried by an OpenAI tool message, so it is refused.
bool tool_result_text(const json& block, std::string& text, std::string& err) {
    auto it = block.find("content");
    if (it == block.end() || it->is_null()) return true;
    if (it->is_string()) {
        text = it->get<std::string>();
        return true;
    }
    if (!it->is_array()) return set_error(err, "tool_result.content must be a string or an array");
    for (const json& part : *it) {
        const std::string type = str_field(part, "type");
        if (type != "text")
            return set_error(err, "tool_result content of type '" + type +
                                      "' is not supported; only text is");
        if (!text.empty()) text += "\n";
        text += str_field(part, "text");
    }
    return true;
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

bool request_to_openai(const json& in, json& out, std::string& err, bool require_max_tokens) {
    if (!in.is_object()) return set_error(err, "request body must be a JSON object");
    const std::string model = str_field(in, "model");
    if (model.empty()) return set_error(err, "model: field required");

    out = json::object();
    out["model"] = model;

    if (require_max_tokens) {
        auto mt = in.find("max_tokens");
        if (mt == in.end() || !mt->is_number_integer())
            return set_error(err, "max_tokens: field required (an integer)");
        if (mt->get<long long>() < 1) return set_error(err, "max_tokens: must be at least 1");
        out["max_tokens"] = *mt;
    }

    json messages = json::array();

    // system: a string, or an array of text blocks.
    if (auto sys = in.find("system"); sys != in.end() && !sys->is_null()) {
        std::string text;
        if (sys->is_string()) {
            text = sys->get<std::string>();
        } else if (sys->is_array()) {
            for (const json& block : *sys) {
                if (str_field(block, "type") != "text")
                    return set_error(err, "system: only text blocks are supported");
                if (!text.empty()) text += "\n";
                text += str_field(block, "text");
            }
        } else {
            return set_error(err, "system: must be a string or an array of text blocks");
        }
        if (!text.empty()) messages.push_back({{"role", "system"}, {"content", text}});
    }

    auto msgs = in.find("messages");
    if (msgs == in.end() || !msgs->is_array())
        return set_error(err, "messages: field required (an array)");

    for (size_t i = 0; i < msgs->size(); ++i) {
        const json& m = (*msgs)[i];
        const std::string where = "messages." + std::to_string(i);
        const std::string role = str_field(m, "role");
        if (role != "user" && role != "assistant")
            return set_error(err, where + ".role: must be 'user' or 'assistant'");
        auto content = m.find("content");
        if (content == m.end() || content->is_null())
            return set_error(err, where + ".content: field required");
        if (content->is_string()) {
            messages.push_back({{"role", role}, {"content", content->get<std::string>()}});
            continue;
        }
        if (!content->is_array())
            return set_error(err, where + ".content: must be a string or an array of blocks");

        if (role == "user") {
            // Anthropic puts tool results INSIDE the user turn; OpenAI needs each as its own
            // role:"tool" message directly after the assistant's tool_calls. Emit the tool
            // messages first, then whatever else the turn carried as one user message.
            json parts = json::array();
            json tool_messages = json::array();
            for (const json& block : *content) {
                const std::string type = str_field(block, "type");
                if (type == "text") {
                    parts.push_back({{"type", "text"}, {"text", str_field(block, "text")}});
                } else if (type == "image") {
                    const json source = block.contains("source") ? block["source"] : json();
                    const std::string stype = str_field(source, "type");
                    std::string url;
                    if (stype == "base64") {
                        url = "data:" + str_field(source, "media_type") + ";base64," +
                              str_field(source, "data");
                    } else if (stype == "url") {
                        // Passed through as given: the image loader refuses remote fetches
                        // itself, so the refusal and its message stay in one place.
                        url = str_field(source, "url");
                    } else {
                        return set_error(err, where + ": image source type '" + stype +
                                                  "' is not supported; use base64");
                    }
                    parts.push_back({{"type", "image_url"}, {"image_url", {{"url", url}}}});
                } else if (type == "tool_result") {
                    const std::string id = str_field(block, "tool_use_id");
                    if (id.empty()) return set_error(err, where + ": tool_result.tool_use_id: field required");
                    std::string text;
                    if (!tool_result_text(block, text, err)) {
                        err = where + ": " + err;
                        return false;
                    }
                    // is_error has no OpenAI counterpart. Say it in the content, where the model
                    // reads it, rather than drop the one signal that the call failed.
                    auto is_error = block.find("is_error");
                    if (is_error != block.end() && is_error->is_boolean() && is_error->get<bool>())
                        text = "Error: " + text;
                    tool_messages.push_back({{"role", "tool"}, {"tool_call_id", id}, {"content", text}});
                } else {
                    return set_error(err, where + ": content block type '" + type +
                                              "' is not supported in a user message");
                }
            }
            for (json& t : tool_messages) messages.push_back(std::move(t));
            if (!parts.empty()) messages.push_back({{"role", "user"}, {"content", parts}});
        } else {
            std::string text;
            std::string reasoning;
            json tool_calls = json::array();
            for (const json& block : *content) {
                const std::string type = str_field(block, "type");
                if (type == "text") {
                    text += str_field(block, "text");
                } else if (type == "thinking") {
                    reasoning += str_field(block, "thinking");
                } else if (type == "redacted_thinking") {
                    // Opaque ciphertext from another provider; nothing here can read it.
                } else if (type == "tool_use") {
                    const std::string id = str_field(block, "id");
                    const std::string name = str_field(block, "name");
                    if (id.empty() || name.empty())
                        return set_error(err, where + ": tool_use needs id and name");
                    const json input = block.contains("input") && block["input"].is_object()
                                           ? block["input"] : json::object();
                    tool_calls.push_back({{"id", id}, {"type", "function"},
                                          {"function", {{"name", name}, {"arguments", input.dump()}}}});
                } else {
                    return set_error(err, where + ": content block type '" + type +
                                              "' is not supported in an assistant message");
                }
            }
            json msg = {{"role", "assistant"}};
            if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
            if (!tool_calls.empty()) {
                msg["tool_calls"] = tool_calls;
                msg["content"] = text.empty() ? json(nullptr) : json(text);
            } else {
                msg["content"] = text;
            }
            messages.push_back(std::move(msg));
        }
    }
    out["messages"] = messages;

    if (auto tools = in.find("tools"); tools != in.end() && !tools->is_null()) {
        if (!tools->is_array()) return set_error(err, "tools: must be an array");
        json oai_tools = json::array();
        for (const json& tool : *tools) {
            const std::string type = str_field(tool, "type", "custom");
            if (type != "custom")
                return set_error(err, "tools: '" + type + "' is a server tool Anthropic runs itself; "
                                      "this server supports custom tools only");
            const std::string name = str_field(tool, "name");
            if (name.empty()) return set_error(err, "tools: every tool needs a name");
            json fn = {{"name", name},
                       {"parameters", tool.contains("input_schema") && tool["input_schema"].is_object()
                                          ? tool["input_schema"]
                                          : json{{"type", "object"}, {"properties", json::object()}}}};
            const std::string description = str_field(tool, "description");
            if (!description.empty()) fn["description"] = description;
            oai_tools.push_back({{"type", "function"}, {"function", fn}});
        }
        out["tools"] = oai_tools;
    }

    if (auto tc = in.find("tool_choice"); tc != in.end() && !tc->is_null()) {
        if (!tc->is_object()) return set_error(err, "tool_choice: must be an object");
        const std::string type = str_field(*tc, "type");
        if (type == "auto") {
            out["tool_choice"] = "auto";
        } else if (type == "any") {
            out["tool_choice"] = "required";
        } else if (type == "none") {
            out["tool_choice"] = "none";
        } else if (type == "tool") {
            const std::string name = str_field(*tc, "name");
            if (name.empty()) return set_error(err, "tool_choice: type 'tool' needs a name");
            out["tool_choice"] = {{"type", "function"}, {"function", {{"name", name}}}};
        } else {
            return set_error(err, "tool_choice.type: must be auto, any, tool, or none");
        }
        auto dp = tc->find("disable_parallel_tool_use");
        if (dp != tc->end() && dp->is_boolean() && dp->get<bool>()) out["parallel_tool_calls"] = false;
    }

    if (auto stop = in.find("stop_sequences"); stop != in.end() && !stop->is_null()) {
        if (!stop->is_array()) return set_error(err, "stop_sequences: must be an array of strings");
        out["stop"] = *stop;
    }
    for (const char* key : {"temperature", "top_p", "top_k"}) {
        auto it = in.find(key);
        if (it != in.end() && it->is_number()) out[key] = *it;
    }

    // thinking: enabled/adaptive turn reasoning on, disabled turns it off, absent leaves the
    // model's own default -- the same default /v1/chat/completions applies.
    if (auto th = in.find("thinking"); th != in.end() && !th->is_null()) {
        const std::string type = str_field(*th, "type");
        if (type == "enabled" || type == "adaptive") out["enable_thinking"] = true;
        else if (type == "disabled") out["enable_thinking"] = false;
        else return set_error(err, "thinking.type: must be enabled, adaptive, or disabled");
    }

    if (auto oc = in.find("output_config"); oc != in.end() && oc->is_object()) {
        const std::string effort = str_field(*oc, "effort");
        if (!effort.empty()) out["reasoning_effort"] = effort;   // validated by the chat parser
        if (auto fmt = oc->find("format"); fmt != oc->end() && fmt->is_object()) {
            if (str_field(*fmt, "type") != "json_schema" || !fmt->contains("schema"))
                return set_error(err, "output_config.format: only json_schema with a schema is supported");
            out["response_format"] = {{"type", "json_schema"},
                                      {"json_schema", {{"name", "output"}, {"schema", (*fmt)["schema"]}}}};
        }
    }
    return true;
}

std::string stop_reason_from_finish(const std::string& finish_reason) {
    if (finish_reason == "length") return "max_tokens";
    if (finish_reason == "tool_calls") return "tool_use";
    return "end_turn";
}

json openai_to_message(const json& oai, const std::string& id, const std::string& model) {
    json content = json::array();
    std::string finish = "stop";
    if (oai.contains("choices") && oai["choices"].is_array() && !oai["choices"].empty()) {
        const json& choice = oai["choices"][0];
        finish = str_field(choice, "finish_reason", "stop");
        const json msg = choice.contains("message") ? choice["message"] : json::object();
        std::string reasoning = str_field(msg, "reasoning_content");
        if (reasoning.empty()) reasoning = str_field(msg, "reasoning");
        if (!reasoning.empty())
            content.push_back({{"type", "thinking"}, {"thinking", reasoning}, {"signature", ""}});
        const std::string text = str_field(msg, "content");
        if (!text.empty()) content.push_back({{"type", "text"}, {"text", text}});
        if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
            for (const json& call : msg["tool_calls"]) {
                const json fn = call.contains("function") ? call["function"] : json::object();
                // The handler has already validated the arguments against the tool's schema, so
                // they parse; an object is the fallback only so a malformed one cannot throw.
                json input = json::parse(str_field(fn, "arguments", "{}"), nullptr, false);
                if (input.is_discarded() || !input.is_object()) input = json::object();
                content.push_back({{"type", "tool_use"}, {"id", str_field(call, "id")},
                                   {"name", str_field(fn, "name")}, {"input", input}});
            }
        }
    }
    const json usage = oai.contains("usage") && oai["usage"].is_object() ? oai["usage"] : json::object();
    return {{"id", id},
            {"type", "message"},
            {"role", "assistant"},
            {"model", model},
            {"content", content},
            {"stop_reason", stop_reason_from_finish(finish)},
            {"stop_sequence", nullptr},
            {"usage", {{"input_tokens", usage.value("prompt_tokens", 0)},
                       {"output_tokens", usage.value("completion_tokens", 0)},
                       {"cache_creation_input_tokens", 0},
                       {"cache_read_input_tokens", 0}}}};
}

std::string error_type_for_status(int status) {
    switch (status) {
        case 400: return "invalid_request_error";
        case 401: return "authentication_error";
        case 403: return "permission_error";
        case 404: return "not_found_error";
        case 413: return "request_too_large";
        case 429: return "rate_limit_error";
        case 529: return "overloaded_error";
        default:  return status >= 400 && status < 500 ? "invalid_request_error" : "api_error";
    }
}

json error_body(int status, const std::string& message) {
    return {{"type", "error"},
            {"error", {{"type", error_type_for_status(status)}, {"message", message}}}};
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

StreamTranslator::StreamTranslator(std::string id, std::string model, long long input_tokens)
    : id_(std::move(id)), model_(std::move(model)), input_tokens_(input_tokens) {}

void StreamTranslator::ensure_started(std::vector<SseEvent>& out) {
    if (started_) return;
    started_ = true;
    json message = {{"id", id_},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"model", model_},
                    {"content", json::array()},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {{"input_tokens", input_tokens_}, {"output_tokens", 0}}}};
    out.push_back({"message_start", {{"type", "message_start"}, {"message", message}}});
}

void StreamTranslator::open_block(std::vector<SseEvent>& out, const std::string& kind, json block) {
    close_block(out);
    open_index_ = next_index_++;
    open_kind_ = kind;
    out.push_back({"content_block_start",
                   {{"type", "content_block_start"}, {"index", open_index_}, {"content_block", std::move(block)}}});
}

void StreamTranslator::close_block(std::vector<SseEvent>& out) {
    if (open_index_ < 0) return;
    out.push_back({"content_block_stop", {{"type", "content_block_stop"}, {"index", open_index_}}});
    open_index_ = -1;
    open_kind_.clear();
}

std::vector<SseEvent> StreamTranslator::on_chunk(const json& oai) {
    std::vector<SseEvent> out;
    if (finished_ || !oai.is_object()) return out;

    // The handler's mid-stream failure chunk. Anthropic ends a stream with an error event and
    // nothing after it, so the message closes here and the usage chunk that follows is dropped.
    if (oai.contains("error")) {
        std::string message = oai["error"].is_string() ? oai["error"].get<std::string>()
                                                       : str_field(oai["error"], "message", "generation failed");
        out.push_back({"error", {{"type", "error"},
                                 {"error", {{"type", "api_error"}, {"message", message}}}}});
        finished_ = true;
        return out;
    }

    ensure_started(out);

    if (oai.contains("choices") && oai["choices"].is_array() && !oai["choices"].empty()) {
        const json& choice = oai["choices"][0];
        const json delta = choice.contains("delta") && choice["delta"].is_object() ? choice["delta"]
                                                                                   : json::object();
        std::string reasoning = str_field(delta, "reasoning_content");
        if (reasoning.empty()) reasoning = str_field(delta, "reasoning");
        if (!reasoning.empty()) {
            if (open_kind_ != "thinking")
                open_block(out, "thinking", {{"type", "thinking"}, {"thinking", ""}, {"signature", ""}});
            out.push_back({"content_block_delta",
                           {{"type", "content_block_delta"}, {"index", open_index_},
                            {"delta", {{"type", "thinking_delta"}, {"thinking", reasoning}}}}});
        }
        const std::string text = str_field(delta, "content");
        if (!text.empty()) {
            if (open_kind_ != "text") open_block(out, "text", {{"type", "text"}, {"text", ""}});
            out.push_back({"content_block_delta",
                           {{"type", "content_block_delta"}, {"index", open_index_},
                            {"delta", {{"type", "text_delta"}, {"text", text}}}}});
        }
        if (delta.contains("tool_calls") && delta["tool_calls"].is_array()) {
            // The handler emits each call whole (validated before release), so each becomes one
            // complete block: start, the full arguments as a single input_json_delta, stop.
            for (const json& call : delta["tool_calls"]) {
                const json fn = call.contains("function") ? call["function"] : json::object();
                open_block(out, "tool_use", {{"type", "tool_use"}, {"id", str_field(call, "id")},
                                             {"name", str_field(fn, "name")}, {"input", json::object()}});
                const std::string args = str_field(fn, "arguments");
                if (!args.empty())
                    out.push_back({"content_block_delta",
                                   {{"type", "content_block_delta"}, {"index", open_index_},
                                    {"delta", {{"type", "input_json_delta"}, {"partial_json", args}}}}});
                close_block(out);
            }
        }
        const std::string finish = str_field(choice, "finish_reason");
        if (!finish.empty()) {
            close_block(out);
            stop_reason_ = stop_reason_from_finish(finish);
        }
    }

    if (oai.contains("usage") && oai["usage"].is_object()) {
        const json& u = oai["usage"];
        close_block(out);
        const long long prompt = u.value("prompt_tokens", input_tokens_);
        out.push_back({"message_delta",
                       {{"type", "message_delta"},
                        {"delta", {{"stop_reason", stop_reason_}, {"stop_sequence", nullptr}}},
                        {"usage", {{"input_tokens", prompt},
                                   {"output_tokens", u.value("completion_tokens", 0)}}}}});
        out.push_back({"message_stop", {{"type", "message_stop"}}});
        finished_ = true;
    }
    return out;
}

}  // namespace anthropic
}  // namespace sparkinfer_server
