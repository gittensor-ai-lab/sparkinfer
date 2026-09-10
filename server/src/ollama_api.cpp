#include "ollama_api.hpp"

#include <ctime>
#include <cmath>
#include <cstdio>

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
    // An EMPTY request name never matches. Ollama documents `model` as required on /api/chat and
    // /api/generate, so an absent or null one is a malformed request, not a request for the
    // default model. Treating it as a match made {"model": null} return 200 and generate from an
    // empty prompt -- hiding an obvious client bug behind a successful-looking response. The
    // caller rejects it with 400 before reaching here.
    if (requested.empty()) return false;
    if (requested == served_id) return true;
    if (with_latest_tag(requested) == with_latest_tag(served_id)) return true;
    // "<id>:anything" also addresses this model: only one checkpoint is served, and refusing a
    // tag we do not carry would break clients that pin a tag from `ollama list` output.
    const size_t colon = requested.find(':');
    return colon != std::string::npos && requested.substr(0, colon) == served_id;
}

namespace {
void add_metrics(const nlohmann::json& oai, nlohmann::json& out);   // defined below

// Null-safe string read.
//
// nlohmann's value() returns the default only when the key is ABSENT. A key that is present and
// JSON-null throws type_error.302 ("type must be string, but is null") -- and OpenAI stream chunks
// are full of exactly that: delta.content is null on the role and finish chunks, finish_reason is
// null on every non-final chunk. Using value() directly on those crashed the whole server mid
// stream (terminate called after throwing ... type_error.302), taking every other in-flight
// request with it. Every string read from an upstream body goes through this.
std::string jstr(const nlohmann::json& j, const char* key, const char* dflt = "") {
    if (!j.is_object()) return dflt;
    auto it = j.find(key);
    if (it == j.end() || !it->is_string()) return dflt;
    return it->get<std::string>();
}
}  // namespace

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
    out["model"] = jstr(in, "model");
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
    out["model"] = jstr(in, "model");
    out["prompt"] = jstr(in, "prompt");
    out["stream"] = false;
    // Only a NON-EMPTY suffix is forwarded. The ollama CLI sends "suffix":"" on an ordinary
    // `ollama run`, and passing that through made /v1/completions reject the whole request with
    // 400 "suffix is not supported" -- so every `ollama run` failed on a field the user never set.
    if (in.contains("suffix") && in["suffix"].is_string() && !in["suffix"].get<std::string>().empty())
        out["suffix"] = in["suffix"];
    apply_options(in, out, "max_tokens");
    return out;
}

bool generate_wants_raw(const nlohmann::json& in) {
    return in.value("raw", false);
}

nlohmann::json generate_request_to_chat(const nlohmann::json& in) {
    nlohmann::json msgs = nlohmann::json::array();
    const std::string sys = jstr(in, "system");
    if (!sys.empty()) msgs.push_back({{"role", "system"}, {"content", sys}});
    msgs.push_back({{"role", "user"}, {"content", jstr(in, "prompt")}});
    nlohmann::json out;
    out["model"] = jstr(in, "model");
    out["messages"] = std::move(msgs);
    out["stream"] = false;
    if (in.contains("format")) {
        if (in["format"].is_string() && in["format"] == "json")
            out["response_format"] = {{"type", "json_object"}};
        else if (in["format"].is_object())
            out["response_format"] = {{"type", "json_schema"},
                                      {"json_schema", {{"schema", in["format"]}}}};
    }
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
        finish = jstr(c, "finish_reason");
        const auto msg = c.value("message", nlohmann::json::object());
        content = jstr(msg, "content");
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
        finish = jstr(c, "finish_reason");
        // Either upstream shape: "text" from /v1/completions (raw path) or "message.content"
        // from /v1/chat/completions (the default, template-applied path).
        text = jstr(c, "text");
        if (text.empty() && c.contains("message"))
            text = jstr(c["message"], "content");
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

nlohmann::json stream_chunk_from_openai(const nlohmann::json& oai, const std::string& model,
                                        const std::string& created_at, bool generate) {
    nlohmann::json out;
    out["model"] = model;
    out["created_at"] = created_at;

    // The usage chunk (choices empty, usage present) becomes Ollama's single done=true terminator.
    const bool has_choices = oai.contains("choices") && oai["choices"].is_array()
                             && !oai["choices"].empty();
    if (!has_choices && oai.contains("usage")) {
        if (generate) out["response"] = "";
        else          out["message"] = {{"role", "assistant"}, {"content", ""}};
        out["done"] = true;
        out["done_reason"] = "stop";
        add_metrics(oai, out);
        return out;
    }
    if (!has_choices) return nlohmann::json();          // nothing to say

    const auto& c = oai["choices"][0];
    const auto delta = c.value("delta", nlohmann::json::object());
    const std::string content = jstr(delta, "content");
    if (content.empty()) {
        // Role-only opener and the finish chunk carry no text. Ollama has no equivalent for
        // either: an empty-content done=false chunk is legal but pure noise, and emitting
        // done=true here would terminate the stream before the metrics chunk.
        return nlohmann::json();
    }
    if (generate) out["response"] = content;
    else          out["message"] = {{"role", "assistant"}, {"content", content}};
    out["done"] = false;
    return out;
}

std::string synthetic_digest(const std::string& seed) {
    // FNV-1a over the seed, re-run with four different offset bases to fill 64 hex characters.
    // Not cryptographic and not claimed to be -- it only has to be stable, well-distributed
    // enough that two checkpoints differ, and exactly the shape Ollama clients slice.
    static const unsigned long long kBases[4] = {
        1469598103934665603ULL, 1099511628211ULL, 14695981039346656037ULL, 1231ULL};
    std::string out;
    out.reserve(64);
    for (int k = 0; k < 4; k++) {
        unsigned long long h = kBases[k];
        for (unsigned char c : seed) {
            h ^= (unsigned long long)c;
            h *= 1099511628211ULL;
        }
        char buf[17];
        std::snprintf(buf, sizeof(buf), "%016llx", h);
        out += buf;
    }
    return out;   // exactly 64 lowercase hex characters
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
