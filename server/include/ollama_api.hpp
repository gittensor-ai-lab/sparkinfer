#pragma once

// Ollama REST API (/api/*) request and response translation.
//
// Same structure and rationale as lmstudio_api.hpp: pure functions over JSON and plain structs,
// no engine/HTTP/CUDA dependency, so the wire contract is unit-testable on a box with no GPU.
//
// WHY TRANSLATION RATHER THAN A SECOND GENERATOR: Ollama's shapes differ from OpenAI's throughout
// -- message.content instead of choices[0].message.content, a `done` boolean instead of
// finish_reason, and durations in NANOSECONDS -- but the generation itself is identical. So the
// /api/* routes rewrite the request into the OpenAI shape, hand it to the SAME handler /v1/* uses,
// and rewrite the response back. Nothing here generates, times, or samples anything.
//
// STREAMING, AND THE ONE HONEST COMPROMISE. Ollama streams by DEFAULT (`stream` is true when
// absent), and its stream is NDJSON -- one JSON object per line -- not SSE. This server's stream
// path is SSE-shaped end to end (write_sse_json emits "data: {...}\n\n"), and a wrapper cannot
// re-frame it because the handler writes straight to the socket.
//
// So a streaming Ollama request is answered as NDJSON containing a SINGLE terminal chunk:
// correct framing, correct fields, done=true, the whole message in one object. An Ollama client
// parses it correctly and works. What it does NOT get is token-by-token delivery -- the response
// arrives when generation finishes. That is a real limitation, stated here and in the response's
// own shape rather than hidden: incremental delivery needs the SSE path generalised, which is a
// change to the generation loop, not to this file.
//
// Schema source: https://github.com/ollama/ollama/blob/main/docs/api.md

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace sparkinfer_server {
namespace ollama {

// ---- model metadata -------------------------------------------------------------------------

// The "details" sub-object shared by /api/tags, /api/ps and /api/show.
struct ModelDetails {
    std::string parent_model;
    std::string format = "gguf";
    std::string family;
    std::vector<std::string> families;
    std::string parameter_size;      // display string, e.g. "7.6B"
    std::string quantization_level;  // e.g. "Q4_K_M"
};

struct ModelEntry {
    std::string name;       // Ollama uses "<model>:<tag>"; a bare id gets ":latest" appended
    std::string model;      // same value as name in every Ollama response observed
    std::string modified_at;
    long long size = 0;     // bytes
    std::string digest;
    ModelDetails details;
};

nlohmann::json details_object(const ModelDetails& d);
nlohmann::json model_entry(const ModelEntry& m);
nlohmann::json tags_list(const std::vector<ModelEntry>& models);   // GET /api/tags
nlohmann::json ps_list(const std::vector<ModelEntry>& models);     // GET /api/ps

// POST /api/show. `capabilities` is Ollama's feature list, e.g. {"completion"} or
// {"completion","vision"}.
nlohmann::json show_object(const ModelEntry& m, const nlohmann::json& model_info,
                           const std::vector<std::string>& capabilities,
                           const std::string& tmpl);

// Ollama names models "<name>:<tag>". A request may address the model with or without the tag,
// so comparison is tag-insensitive when the request omits one.
std::string with_latest_tag(const std::string& id);
bool model_name_matches(const std::string& requested, const std::string& served_id);

// ---- request translation --------------------------------------------------------------------

// Ollama /api/chat body -> OpenAI /v1/chat/completions body. Maps `options` (num_predict,
// temperature, top_p, seed, stop, ...) onto their OpenAI equivalents. `stream` is forced FALSE in
// the translated request regardless of what the client asked for: the caller re-frames the
// completed response, so the inner call must not stream.
nlohmann::json chat_request_to_openai(const nlohmann::json& in);

// Ollama /api/generate body -> OpenAI /v1/completions body. Same treatment of `options`/`stream`.
// ONLY correct for a raw request -- see generate_wants_raw.
nlohmann::json generate_request_to_openai(const nlohmann::json& in);

// True when /api/generate must bypass prompt templating, i.e. the request set "raw": true.
//
// This distinction is the whole reason /api/generate cannot simply map onto /v1/completions.
// Ollama's /api/generate applies the MODEL'S TEMPLATE to the prompt by default; OpenAI's
// /v1/completions is raw by definition. Mapping one onto the other sent `ollama run` an
// unformatted prompt, and the model answered "Reply with exactly: OK" with
// "OK: OK: OK: OK: ..." -- a working endpoint producing bad output, which is worse than an error
// because nothing reports it. Only driving the real CLI surfaced it.
bool generate_wants_raw(const nlohmann::json& in);

// Ollama /api/generate body -> OpenAI /v1/chat/completions body, as a single user turn (plus a
// system turn when the request carries one). This is the DEFAULT path: it is what makes the
// server apply its chat template, matching Ollama's own behaviour.
nlohmann::json generate_request_to_chat(const nlohmann::json& in);

// ---- response translation -------------------------------------------------------------------

// Durations Ollama reports are NANOSECONDS; this server measures milliseconds.
long long ms_to_ns(double ms);

// OpenAI chat.completion -> Ollama /api/chat response. `model` is the name to echo back.
nlohmann::json openai_to_chat_response(const nlohmann::json& oai, const std::string& model,
                                       const std::string& created_at);
// OpenAI text_completion -> Ollama /api/generate response.
nlohmann::json openai_to_generate_response(const nlohmann::json& oai, const std::string& model,
                                           const std::string& created_at);

// RFC3339 UTC timestamp, the format Ollama's created_at uses.
std::string rfc3339_now();

// A stable 64-hex-character identifier derived from `seed`, for ModelEntry::digest.
//
// This is NOT a content hash, and the distinction matters enough to say twice. Ollama's digest is
// a sha256 of the model blob, used for dedup when pulling; hashing a multi-GB checkpoint on every
// request is out of the question, and doing it once at startup would add a minute to boot for a
// field nothing here consumes.
//
// It cannot simply be left empty, which was the first attempt: `ollama list` and `ollama ps`
// render the short hash as digest[:12] with no length check, so an empty string panics the
// official CLI outright ("slice bounds out of range [:12] with length 0"). A client can only be
// misled by a non-content digest if it can fetch blobs or pull from this server, and both are
// refused with 501 -- so a stable synthetic value is strictly better than a crash.
//
// Stable across restarts and distinct across checkpoints, because the seed carries path, size and
// mtime.
std::string synthetic_digest(const std::string& seed);

}  // namespace ollama
}  // namespace sparkinfer_server
