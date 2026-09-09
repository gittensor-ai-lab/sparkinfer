// Unit tests for the LM Studio /api/v0 response shaping. Pure JSON over plain structs, so this
// runs anywhere -- no GPU, no model, no HTTP server.
//
// The assertions are deliberately about EXACT field names and value vocabularies. That is the
// whole contract: a client written against LM Studio switches on strings like "eosFound" and
// reads keys like "max_context_length", and getting one of them subtly wrong produces a response
// that parses fine and behaves wrongly.

#include "lmstudio_api.hpp"

#include <cstdio>
#include <string>

using namespace sparkinfer_server::lmstudio;

#define CHECK(x) do { if (!(x)) { std::printf("FAIL: %s line %d\n", #x, __LINE__); return 1; } } while (0)

int main() {
    // ---- model object: every documented field present, exact names ----
    {
        ModelDesc m;
        m.id = "muse-glimmer-30b";
        m.type = "llm";
        m.publisher = "sparkinfer";
        m.arch = "muse-glimmer";
        m.compatibility_type = "gguf";
        m.quantization = "Q4_K_M";
        m.state = "loaded";
        m.max_context_length = 131072;
        m.loaded_context_length = 4096;
        const nlohmann::json j = model_object(m);
        for (const char* k : {"id", "object", "type", "publisher", "arch", "compatibility_type",
                              "quantization", "state", "max_context_length"})
            CHECK(j.contains(k));
        CHECK(j["object"] == "model");
        CHECK(j["max_context_length"] == 131072);
        CHECK(j["loaded_context_length"] == 4096);
    }
    // loaded_context_length is OMITTED when not loaded — a 0 would read as "a context of zero"
    {
        ModelDesc m; m.id = "x"; m.state = "not-loaded"; m.loaded_context_length = 0;
        const nlohmann::json j = model_object(m);
        CHECK(!j.contains("loaded_context_length"));
        CHECK(j["state"] == "not-loaded");
    }
    // a loaded model that somehow reports 0 also omits it, rather than emitting a misleading 0
    {
        ModelDesc m; m.id = "x"; m.state = "loaded"; m.loaded_context_length = 0;
        CHECK(!model_object(m).contains("loaded_context_length"));
    }

    // ---- list envelope ----
    {
        ModelDesc a; a.id = "a"; ModelDesc b; b.id = "b";
        const nlohmann::json j = models_list({a, b});
        CHECK(j["object"] == "list");
        CHECK(j["data"].is_array() && j["data"].size() == 2);
        CHECK(j["data"][0]["id"] == "a" && j["data"][1]["id"] == "b");
        CHECK(models_list({})["data"].is_array() && models_list({})["data"].empty());
    }

    // ---- stats: seconds, and LM Studio's stop_reason vocabulary ----
    {
        Stats s; s.tokens_per_second = 51.4; s.time_to_first_token = 0.111;
        s.generation_time = 0.954; s.stop_reason = "eosFound";
        const nlohmann::json j = stats_object(s);
        for (const char* k : {"tokens_per_second", "time_to_first_token", "generation_time", "stop_reason"})
            CHECK(j.contains(k));
        CHECK(j["stop_reason"] == "eosFound");
    }
    CHECK(stop_reason_from_finish("stop") == "eosFound");
    CHECK(stop_reason_from_finish("length") == "maxPredictedTokensReached");
    CHECK(stop_reason_from_finish("tool_calls") == "toolCalls");
    // an unmapped/absent value must NOT leak OpenAI's word through
    CHECK(stop_reason_from_finish("") == "eosFound");
    CHECK(stop_reason_from_finish("something_new") == "eosFound");

    // ---- model_info / runtime ----
    {
        ModelInfo mi; mi.arch = "muse-glimmer"; mi.quant = "Q4_K_M"; mi.format = "gguf";
        mi.context_length = 4096;
        const nlohmann::json j = model_info_object(mi);
        for (const char* k : {"arch", "quant", "format", "context_length"}) CHECK(j.contains(k));
        RuntimeDesc r; r.name = "sparkinfer-linux-x86_64-nvidia-cuda12-sm120"; r.version = "0.4.4";
        const nlohmann::json rj = runtime_object(r);
        CHECK(rj["supported_formats"].is_array());
        CHECK(rj["supported_formats"][0] == "gguf");
    }

    // ---- quantization parsing: longest tag wins, so Q4_K_M never truncates to Q4_K ----
    CHECK(quantization_from_path("/m/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf") == "Q4_K_M");
    CHECK(quantization_from_path("/m/Spark-X2.5-4B-Q8_0.gguf") == "Q8_0");
    CHECK(quantization_from_path("/m/Model-q4_k_m.gguf") == "Q4_K_M");        // case-insensitive
    CHECK(quantization_from_path("/m/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf") == "Q4_K_M");
    CHECK(quantization_from_path("/root/workspace/models_q38_modelopt") == "");  // dir, no tag
    CHECK(quantization_from_path("") == "");

    // ---- compatibility_type ----
    CHECK(compatibility_type_from_path("/m/x.gguf") == "gguf");
    CHECK(compatibility_type_from_path("/m/x.GGUF") == "gguf");
    CHECK(compatibility_type_from_path("/root/workspace/models_q38_modelopt") == "");

    // ---- publisher ----
    // Only LM Studio's own layout carries a publisher: <...>/models/<publisher>/<repo>/<file>
    CHECK(publisher_from_path("/home/u/.lmstudio/models/lmstudio-community/Foo-GGUF/foo-Q4_K_M.gguf")
          == "lmstudio-community");
    CHECK(publisher_from_path("/models/lmstudio-community/Foo-GGUF/foo.gguf") == "lmstudio-community");
    // An ordinary path has NO publisher. Reading one out of the path shape alone reported "root"
    // for /root/spark25_models/... and displayed it as if it were real.
    CHECK(publisher_from_path("/root/spark25_models/Spark-X2.5-4B-Q8_0.gguf") == "");
    CHECK(publisher_from_path("/root/workspace/models_muse_glimmer/Muse-Glimmer-30B-Q4_K_M.gguf") == "");
    CHECK(publisher_from_path("/a/b/c.gguf") == "");
    CHECK(publisher_from_path("c.gguf") == "");
    CHECK(publisher_from_path("/x.gguf") == "");
    // "models" as the immediate parent of the FILE is not a publisher layout either
    CHECK(publisher_from_path("/opt/models/foo.gguf") == "");

    std::printf("lmstudio_api_test: OK\n");
    return 0;
}
