#include "lmstudio_api.hpp"

#include <algorithm>
#include <cctype>

namespace sparkinfer_server {
namespace lmstudio {

namespace {

bool is_sep(char c) { return c == '/' || c == '\\'; }

std::string basename_of(const std::string& path) {
    size_t last = std::string::npos;
    for (size_t i = 0; i < path.size(); i++)
        if (is_sep(path[i])) last = i;
    return last == std::string::npos ? path : path.substr(last + 1);
}

std::string upper(std::string s) {
    for (char& c : s) c = (char)std::toupper((unsigned char)c);
    return s;
}

}  // namespace

nlohmann::json model_object(const ModelDesc& m) {
    nlohmann::json j = {
        {"id", m.id},
        {"object", "model"},
        {"type", m.type},
        {"publisher", m.publisher},
        {"arch", m.arch},
        {"compatibility_type", m.compatibility_type},
        {"quantization", m.quantization},
        {"state", m.state},
        {"max_context_length", m.max_context_length},
    };
    // Present only while loaded — see ModelDesc::loaded_context_length for why this is omitted
    // rather than sent as 0.
    if (m.state == "loaded" && m.loaded_context_length > 0)
        j["loaded_context_length"] = m.loaded_context_length;
    return j;
}

nlohmann::json models_list(const std::vector<ModelDesc>& models) {
    nlohmann::json data = nlohmann::json::array();
    for (const ModelDesc& m : models) data.push_back(model_object(m));
    return nlohmann::json{{"object", "list"}, {"data", std::move(data)}};
}

nlohmann::json stats_object(const Stats& s) {
    return nlohmann::json{
        {"tokens_per_second", s.tokens_per_second},
        {"time_to_first_token", s.time_to_first_token},
        {"generation_time", s.generation_time},
        {"stop_reason", s.stop_reason},
    };
}

std::string stop_reason_from_finish(const std::string& finish_reason) {
    // LM Studio's vocabulary, from its v0 reference. "eosFound" is the natural end of a turn;
    // "maxPredictedTokensReached" is the length cap; "userStopped" covers a client-supplied stop
    // string. An unknown value maps to eosFound rather than passing OpenAI's word through, because
    // a client switching on this field would otherwise hit a case it has no branch for.
    if (finish_reason == "length") return "maxPredictedTokensReached";
    if (finish_reason == "stop") return "eosFound";
    if (finish_reason == "tool_calls") return "toolCalls";
    if (finish_reason == "content_filter") return "userStopped";
    return "eosFound";
}

nlohmann::json model_info_object(const ModelInfo& mi) {
    return nlohmann::json{
        {"arch", mi.arch},
        {"quant", mi.quant},
        {"format", mi.format},
        {"context_length", mi.context_length},
    };
}

nlohmann::json runtime_object(const RuntimeDesc& r) {
    return nlohmann::json{
        {"name", r.name},
        {"version", r.version},
        {"supported_formats", r.supported_formats},
    };
}

std::string quantization_from_path(const std::string& path) {
    const std::string base = upper(basename_of(path));
    // Longest-first so Q4_K_M is not truncated to Q4_K, and IQ* before Q* for the same reason.
    static const char* kTags[] = {
        "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
        "IQ4_XS", "IQ4_NL",
        "Q2_K_S", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q3_K",
        "Q4_K_S", "Q4_K_M", "Q4_K", "Q4_0", "Q4_1",
        "Q5_K_S", "Q5_K_M", "Q5_K", "Q5_0", "Q5_1",
        "Q6_K", "Q8_0", "BF16", "FP16", "F16", "FP8", "NVFP4", "MXFP4",
    };
    std::string best;
    for (const char* t : kTags) {
        const std::string tag(t);
        if (base.find(tag) != std::string::npos && tag.size() > best.size()) best = tag;
    }
    return best;
}

std::string compatibility_type_from_path(const std::string& path) {
    const std::string base = basename_of(path);
    if (base.size() >= 5) {
        const std::string tail = base.substr(base.size() - 5);
        std::string lower;
        for (char c : tail) lower += (char)std::tolower((unsigned char)c);
        if (lower == ".gguf") return "gguf";
    }
    return "";
}

std::string publisher_from_path(const std::string& path) {
    // A publisher is reported ONLY when the path actually carries one, i.e. it sits under a
    // directory named "models" in LM Studio's own layout:
    //
    //     ~/.lmstudio/models/<publisher>/<repo>/<file>.gguf
    //
    // The obvious alternative -- "take the segment two above the filename" -- reads a publisher
    // out of any path at all, and produces nonsense for the ordinary case: it turned
    // /root/spark25_models/Spark-X2.5-4B-Q8_0.gguf into publisher "root", which then shows up in
    // LM Studio's model list as if it meant something. Returning nothing lets the caller
    // substitute an honest default.
    std::vector<std::string> parts;
    std::string cur;
    for (char c : path) {
        if (is_sep(c)) { if (!cur.empty()) parts.push_back(cur); cur.clear(); }
        else cur += c;
    }
    if (!cur.empty()) parts.push_back(cur);
    // Need <models>/<publisher>/<repo>/<file>: the marker cannot be the last three segments.
    for (size_t i = 0; i + 3 < parts.size(); i++)
        if (parts[i] == "models") return parts[i + 1];
    return "";
}

}  // namespace lmstudio
}  // namespace sparkinfer_server
