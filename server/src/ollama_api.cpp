#include "ollama_api.hpp"

#include <ctime>
#include <cmath>

namespace sparkinfer_server {
namespace ollama {

nlohmann::json details_object(const ModelDetails& d) {
    return nlohmann::json{
        {"parent_model", d.parent_model},
        {"format", d.format},
        {"family", d.family},
        {"families", d.families},
        {"parameter_size", d.parameter_size},
        {"quantization_level", d.quantization_level},
    };
}

nlohmann::json model_entry(const ModelEntry& m) {
    return nlohmann::json{
        {"name", m.name},
        {"model", m.model},
        {"modified_at", m.modified_at},
        {"size", m.size},
        {"digest", m.digest},
        {"details", details_object(m.details)},
    };
}

nlohmann::json tags_list(const std::vector<ModelEntry>& models) {
    nlohmann::json arr = nlohmann::json::array();
    for (const ModelEntry& m : models) arr.push_back(model_entry(m));
    return nlohmann::json{{"models", std::move(arr)}};
}

nlohmann::json ps_list(const std::vector<ModelEntry>& models) {
    // /api/ps carries the same entries as /api/tags in Ollama's own responses.
    return tags_list(models);
}

nlohmann::json show_object(const ModelEntry& m, const nlohmann::json& model_info,
                           const std::vector<std::string>& capabilities,
                           const std::string& tmpl) {
    return nlohmann::json{
        // No Modelfile exists here -- this server was pointed at a checkpoint with -m, it did not
        // build one from a Modelfile. Report the path it is actually serving rather than
        // fabricating Modelfile syntax that would not round-trip through `ollama create`.
        {"modelfile", "FROM " + m.name},
        {"parameters", ""},
        {"template", tmpl},
        {"details", details_object(m.details)},
        {"model_info", model_info},
        {"capabilities", capabilities},
    };
}

std::string with_latest_tag(const std::string& id) {
    return id.find(':') == std::string::npos ? id + ":latest" : id;
}

bool model_name_matches(const std::string& requested, const std::string& served_id) {
    if (requested.empty()) return true;   // Ollama treats an absent model as "the loaded one"
    if (requested == served_id) return true;
    if (with_latest_tag(requested) == with_latest_tag(served_id)) return true;
    // "<id>:anything" also addresses this model: only one checkpoint is served, and refusing a
    // tag we do not carry would break clients that pin a tag from `ollama list` output.
    const size_t colon = requested.find(':');
    return colon != std::string::npos && requested.substr(0, colon) == served_id;
}

namespace {

// Ollama nests sampling knobs under "options"; OpenAI puts them at the top level.
void apply_options(const nlohmann::json& in, nlohmann::json& out, const char* max_tokens_key) {
    const auto opts = in.value("options", nlohmann::json::object());
    if (opts.contains("temperature")) out["temperature"] = opts["temperature"];
    if (opts.contains("top_p")) out["top_p"] = opts["top_p"];
    if (opts.contains("seed")) out["seed"] = opts["seed"];
    if (opts.contains("stop")) out["stop"] = opts["stop"];
    if (opts.contains("presence_penalty")) out["presence_penalty"] = opts["presence_penalty"];
    if (opts.contains("frequency_penalty")) out["frequency_penalty"] = opts["frequency_penalty"];
    // num_predict is Ollama's max-new-tokens. -1 means "until the context is full" and has no
    // OpenAI equivalent, so it is dropped rather than translated into a bogus finite cap.
    if (opts.contains("num_predict") && opts["num_predict"].is_number_integer()) {
        const long long n = opts["num_predict"].get<long long>();
        if (n > 0) out[max_tokens_key] = n;
    }
}

}  // namespace

nlohmann::json chat_request_to_openai(const nlohmann::json& in) {
    nlohmann::json out;
    out["model"] = in.value("model", "");
    out["messages"] = in.value("messages", nlohmann::json::array());
    // Always false: the caller re-frames a COMPLETED response into NDJSON, so the inner handler
    // must produce a whole JSON document, not a stream. See ollama_api.hpp on streaming.
    out["stream"] = false;
    if (in.contains("format")) {
        // Ollama's "format":"json" is OpenAI's response_format. A JSON-schema object maps onto
        // json_schema; the bare string maps onto json_object.
        if (in["format"].is_string() && in["format"] == "json")
            out["response_format"] = {{"type", "json_object"}};
        else if (in["format"].is_object())
            out["response_format"] = {{"type", "json_schema"},
                                      {"json_schema", {{"schema", in["format"]}}}};
    }
    if (in.contains("tools")) out["tools"] = in["tools"];
    apply_options(in, out, "max_tokens");
    return out;
}

nlohmann::json generate_request_to_openai(const nlohmann::json& in) {
    nlohmann::json out;
    out["model"] = in.value("model", "");
    out["prompt"] = in.value("prompt", "");
    out["stream"] = false;
    if (in.contains("suffix")) out["suffix"] = in["suffix"];
    apply_options(in, out, "max_tokens");
    return out;
}

long long ms_to_ns(double ms) {
    if (!(ms > 0.0)) return 0;
    return (long long)std::llround(ms * 1e6);
}

namespace {

// The duration/count block shared by /api/chat and /api/generate responses.
void add_metrics(const nlohmann::json& oai, nlohmann::json& out) {
    const auto usage = oai.value("usage", nlohmann::json::object());
    const int prompt_tokens = usage.value("prompt_tokens", 0);
    const int completion_tokens = usage.value("completion_tokens", 0);
    const double ttft_ms = usage.value("ttft_ms", 0.0);
    const double generation_ms = usage.value("generation_ms", 0.0);

    out["total_duration"] = ms_to_ns(ttft_ms + generation_ms);
    // load_duration is model-load time. The model was loaded at startup, long before this
    // request, so the honest value here is 0 rather than an invented share of the request.
    out["load_duration"] = 0;
    out["prompt_eval_count"] = prompt_tokens;
    out["prompt_eval_duration"] = ms_to_ns(ttft_ms);
    out["eval_count"] = completion_tokens;
    out["eval_duration"] = ms_to_ns(generation_ms);
}

std::string done_reason_from_finish(const std::string& finish) {
    // Ollama's vocabulary: "stop" for a natural end, "length" when the token cap was hit.
    if (finish == "length") return "length";
    return "stop";
}

}  // namespace

nlohmann::json openai_to_chat_response(const nlohmann::json& oai, const std::string& model,
                                       const std::string& created_at) {
    nlohmann::json out;
    out["model"] = model;
    out["created_at"] = created_at;
    std::string content, finish;
    nlohmann::json tool_calls = nlohmann::json::array();
    if (oai.contains("choices") && oai["choices"].is_array() && !oai["choices"].empty()) {
        const auto& c = oai["choices"][0];
        finish = c.value("finish_reason", "");
        const auto msg = c.value("message", nlohmann::json::object());
        content = msg.value("content", "");
        if (msg.contains("tool_calls") && msg["tool_calls"].is_array())
            tool_calls = msg["tool_calls"];
    }
    nlohmann::json message = {{"role", "assistant"}, {"content", content}};
    if (!tool_calls.empty()) message["tool_calls"] = tool_calls;
    out["message"] = std::move(message);
    out["done"] = true;
    out["done_reason"] = done_reason_from_finish(finish);
    add_metrics(oai, out);
    return out;
}

nlohmann::json openai_to_generate_response(const nlohmann::json& oai, const std::string& model,
                                           const std::string& created_at) {
    nlohmann::json out;
    out["model"] = model;
    out["created_at"] = created_at;
    std::string text, finish;
    if (oai.contains("choices") && oai["choices"].is_array() && !oai["choices"].empty()) {
        const auto& c = oai["choices"][0];
        finish = c.value("finish_reason", "");
        text = c.value("text", "");
    }
    out["response"] = text;
    out["done"] = true;
    out["done_reason"] = done_reason_from_finish(finish);
    // "context" is Ollama's opaque conversation-state token array, used to continue a generation.
    // This server keeps no such state, and returning a fabricated array would invite a client to
    // send it back as if it meant something. Omitted entirely.
    add_metrics(oai, out);
    return out;
}

std::string rfc3339_now() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return std::string(buf);
}

}  // namespace ollama
}  // namespace sparkinfer_server
