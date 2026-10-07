// The Qwen3.8 prompt builder against the checkpoints' own Jinja templates, byte for byte.
//
// tests/fixtures/qwen38_templates/golden.json holds requests and the prompt each template renders
// for them (gen_golden.py, transformers/vLLM semantics). Every request is parsed by the server's
// own parse_chat_request_json and rendered by apply_qwen36_tools_template for each variant:
//   pinned    gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090's template
//   official  Qwen3.8-27B's own template (Swift-Qwen3.8-27B ships it unmodified)
// A mismatch prints the case, the variant, and where the two prompts first part.
#include "chat_tools.hpp"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

using nlohmann::json;
using sparkinfer_server::ChatRequest;
using sparkinfer_server::QwenTemplateVariant;
using sparkinfer_server::apply_qwen36_tools_template;
using sparkinfer_server::parse_chat_request_json;

namespace {

std::string show(const std::string& s, size_t at) {
    const size_t from = at > 40 ? at - 40 : 0;
    std::string out = json(s.substr(from, 120)).dump();
    return (from ? "..." : "") + out;
}

}  // namespace

int main() {
    std::ifstream in(std::string(SPARKINFER_SERVER_TEST_DIR) + "/fixtures/qwen38_templates/golden.json");
    if (!in) {
        std::fprintf(stderr, "FAIL: cannot open golden.json\n");
        return 1;
    }
    const json cases = json::parse(in);
    int failed = 0, run = 0;
    for (const json& c : cases) {
        ChatRequest request;
        std::string err;
        if (!parse_chat_request_json(c["body"].dump(), request, err)) {
            std::fprintf(stderr, "FAIL %s: request did not parse: %s\n", c["name"].get<std::string>().c_str(),
                         err.c_str());
            ++failed;
            continue;
        }
        const bool thinking = c["enable_thinking"].get<bool>();
        for (const auto& [name, variant] : {std::pair{"pinned", QwenTemplateVariant::kQwen38Pinned},
                                            std::pair{"official", QwenTemplateVariant::kQwen38Official}}) {
            ++run;
            const std::string want = c["expected"][name].get<std::string>();
            const std::string got = apply_qwen36_tools_template(request, thinking, variant);
            if (got == want) continue;
            size_t at = 0;
            while (at < got.size() && at < want.size() && got[at] == want[at]) ++at;
            std::fprintf(stderr, "FAIL %s [%s] differs at byte %zu\n  want %s\n  got  %s\n",
                         c["name"].get<std::string>().c_str(), name, at, show(want, at).c_str(),
                         show(got, at).c_str());
            ++failed;
        }
    }
    // Each fixture template is recognised as itself.
    for (const auto& [file, want] : {std::pair{"pinned.jinja", QwenTemplateVariant::kQwen38Pinned},
                                     std::pair{"official.jinja", QwenTemplateVariant::kQwen38Official}}) {
        std::ifstream t(std::string(SPARKINFER_SERVER_TEST_DIR) + "/fixtures/qwen38_templates/" + file);
        const std::string text((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        if (text.empty() || sparkinfer_server::qwen38_template_variant(text) != want) {
            std::fprintf(stderr, "FAIL: %s is not detected as its own variant\n", file);
            ++failed;
        }
    }
    if (sparkinfer_server::qwen38_template_variant("") != QwenTemplateVariant::kQwen38Pinned) {
        std::fprintf(stderr, "FAIL: an absent template must keep the pinned default\n");
        ++failed;
    }
    std::printf("%d of %d renders match their template\n", run - failed, run);
    return failed ? 1 : 0;
}
