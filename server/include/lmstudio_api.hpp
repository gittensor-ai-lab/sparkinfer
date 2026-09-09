#pragma once

// LM Studio REST API (/api/v0/*) response shaping.
//
// WHY A SEPARATE MODULE: these are pure JSON-shaping functions over plain structs, with no
// dependency on the engine, the HTTP server or CUDA. That keeps them unit-testable on a box with
// no GPU -- the same reason chat_tools.cpp and chat_tokenizer.cpp are their own translation units.
// The route handlers in sparkinfer_server.cpp supply the values; everything about the WIRE FORMAT
// lives here, so a change to LM Studio's schema is a change to one file with tests behind it.
//
// WHICH VERSION, AND WHY THE DEPRECATED ONE: LM Studio 0.4.0 released a native /api/v1/* and
// recommends it over /api/v0/*. v1 is deliberately NOT implemented here. Its additions beyond v0
// are MCP support, stateful chats, auth, and model management -- /api/v1/models/load, /unload and
// /download. Those last three assume a runtime that swaps models on demand; sparkinfer loads one
// checkpoint at startup from -m and serves exactly that, so there is nothing to load or unload and
// a v1 server would have to answer half its own contract with errors. v0's five endpoints map onto
// what this server actually is. Revisit if sparkinfer ever grows a model manager.
//
// Schema source: https://lmstudio.ai/docs/developer/rest/endpoints (v0 reference).

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace sparkinfer_server {
namespace lmstudio {

// One entry of GET /api/v0/models. Field names and value vocabularies are LM Studio's, not ours:
//   type                "llm" | "vlm" | "embeddings"
//   compatibility_type  "gguf" | "mlx"   (LM Studio knows only these two)
//   state               "loaded" | "not-loaded"
struct ModelDesc {
    std::string id;
    std::string type = "llm";
    std::string publisher;
    std::string arch;
    std::string compatibility_type = "gguf";
    std::string quantization;
    std::string state = "loaded";
    int max_context_length = 0;
    // Only meaningful while loaded. LM Studio omits it on a not-loaded model rather than sending
    // 0, so a consumer can tell "no context in use" from "a context of zero".
    int loaded_context_length = 0;
};

nlohmann::json model_object(const ModelDesc& m);
nlohmann::json models_list(const std::vector<ModelDesc>& models);

// The "stats" object LM Studio appends to a completion. Times are SECONDS (the server measures
// milliseconds internally, so the caller converts) and tokens_per_second is decode throughput,
// not end-to-end.
struct Stats {
    double tokens_per_second = 0.0;
    double time_to_first_token = 0.0;
    double generation_time = 0.0;
    std::string stop_reason = "eosFound";
};
nlohmann::json stats_object(const Stats& s);

// OpenAI finish_reason -> LM Studio stop_reason. The two vocabularies differ, and a client that
// switches on stop_reason will silently mis-handle an unmapped value, so map explicitly.
std::string stop_reason_from_finish(const std::string& finish_reason);

struct ModelInfo {
    std::string arch;
    std::string quant;
    std::string format = "gguf";
    int context_length = 0;
};
nlohmann::json model_info_object(const ModelInfo& mi);

struct RuntimeDesc {
    std::string name;
    std::string version;
    std::vector<std::string> supported_formats{"gguf"};
};
nlohmann::json runtime_object(const RuntimeDesc& r);

// Best-effort quantization label from a checkpoint path, e.g.
//   ".../Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf" -> "Q4_K_M"
// Filename-derived on purpose: the engine does not retain the GGUF after load, and LM Studio's
// own field is a display string rather than something a client computes with. Returns "" when
// nothing recognisable is present, and the caller decides what to report instead of guessing.
std::string quantization_from_path(const std::string& path);

// "gguf" for a .gguf file. Everything else this server can load (compressed-tensors directories,
// legacy weight dirs) has no LM Studio vocabulary term; the caller passes what it wants reported.
std::string compatibility_type_from_path(const std::string& path);

// Publisher segment of a HuggingFace-style path (".../lmstudio-community/Foo-GGUF/x.gguf" ->
// "lmstudio-community"), "" when the layout does not carry one.
std::string publisher_from_path(const std::string& path);

}  // namespace lmstudio
}  // namespace sparkinfer_server
