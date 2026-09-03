#pragma once

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace sparkinfer_server {

struct ToolCall {
    std::string id;
    std::string name;
    // OpenAI wire representation: a compact JSON object encoded as a string.
    std::string arguments;
};

struct ToolDefinition {
    std::string name;
    // The complete top-level OpenAI tool object ({"type":"function","function":{...}}).
    nlohmann::json spec;
};

enum class ToolChoiceMode {
    kAuto,
    kNone,
    kRequired,
    kNamed,
};

struct ChatMessage {
    std::string role;
    std::string content;
    bool content_is_null = false;
    std::string reasoning_content;
    std::string name;
    std::string tool_call_id;
    std::vector<ToolCall> tool_calls;
    // image_url values from multimodal content parts, in the order they appeared. The rendered
    // content carries one <|vision_start|><|image_pad|><|vision_end|> per entry at the position
    // that part occupied, so this vector and those markers stay in lockstep -- the processor, not
    // the template, later expands each placeholder to the token count its grid needs.
    std::vector<std::string> images;
};

enum class ResponseFormatType {
    kText,
    kJsonObject,
    kJsonSchema,
};

struct ResponseFormat {
    ResponseFormatType type = ResponseFormatType::kText;
    std::string schema_name;   // json_schema.name -- steering text / logging only
    nlohmann::json schema;     // json_schema.schema -- empty/ignored unless type == kJsonSchema
    // json_schema.strict -- parsed and stored, but a documented v1 no-op: this backend has no
    // constrained-decoding mechanism, so it cannot honor a real conformance guarantee either way.
    bool strict = false;
};

struct ChatRequest {
    std::vector<ChatMessage> messages;
    std::vector<ToolDefinition> tools;
    ToolChoiceMode tool_choice = ToolChoiceMode::kAuto;
    std::string required_tool_name;  // populated for OpenAI's named/object tool_choice
    bool parallel_tool_calls = true;
    // OpenRouter/OpenAI reasoning controls. The tokenizer uses this when it constructs the
    // Qwen3.8 system instruction; empty means the model's default (xhigh).
    std::string reasoning_effort;
    bool reasoning_exclude = false;
    ResponseFormat response_format;
};

struct ParsedToolOutput {
    std::string reasoning_content;
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::string error;
};

bool parse_chat_request_json(const std::string& body, ChatRequest& request, std::string& err);

// Decode-control fields that don't influence prompt construction (unlike ChatRequest's
// tools/response_format, which do) -- extracted here rather than kept private to
// sparkinfer_server.cpp so parsing/validation has a unit-test seam without a running server/GPU.
struct RequestControls {
    bool stream = false;
    bool include_usage = false;
    int max_tokens = 256;
    std::vector<std::string> stop;
    // <= 0 (default) is plain greedy argmax, byte-identical to pre-sampling behavior.
    float temperature = 0.f;
    uint64_t seed = 0;       // only meaningful when seed_set
    bool seed_set = false;   // client explicitly supplied `seed`; false => caller should generate one
    // top_k <= 0 disables top_k (0 = no limit, matching llama.cpp/vLLM). top_p >= 1.0 disables
    // top_p (1.0 is OpenAI's own "disabled" default). Neither requires temperature > 0 -- see
    // ContinuousBatchEngine::Request's doc comment for the inertness proof.
    int top_k = 0;
    float top_p = 1.0f;
    // [-2.0, 2.0]; 0 (default, OpenAI's own default) disables both. Sampling controls, same tier
    // as temperature/top_k/top_p (NOT logprobs, which is pure output reporting) -- threaded
    // through every complete_streaming call site, including the json_mode/tool-calling retry
    // loops that top_k/top_p already reach but logprobs/top_logprobs do not.
    float presence_penalty = 0.f;
    float frequency_penalty = 0.f;
    // logprobs=false (default) attaches no logprobs field anywhere in the response. top_logprobs
    // is only meaningful when logprobs is true -- unlike top_k/top_p, this IS cross-validated
    // against a sibling field: parse_request_controls rejects top_logprobs supplied without
    // logprobs=true (matches OpenAI's own documented constraint).
    bool logprobs = false;
    int top_logprobs = 0;   // 0-20 when logprobs=true
    // OpenAI's logit_bias: (token_id, bias) pairs, bias in [-100.0, 100.0]. Empty (default)
    // disables it -- same self-describing-default convention as top_p/top_k, no `_set` bool
    // needed. Sampling-control tier, same as presence_penalty/frequency_penalty -- threaded
    // through every complete_streaming call site, including the json_mode/tool-calling retry
    // loops.
    std::vector<std::pair<int, float>> logit_bias;
    // OpenAI's n: number of independent completions ("choices") to return for this request.
    // Default 1 (today's only behavior). Validated against kMaxN (chat_tools.cpp) -- sparkinfer's
    // own bound, not an OpenAI-documented limit; see sparkinfer_server.cpp for the fan-out
    // machinery this drives (n>1 runs n fully independent prefill+decode sessions concurrently,
    // no shared-prefill optimization in v1).
    int n = 1;
};

// vocab, when > 0, additionally rejects any logit_bias token id >= vocab with a real validation
// error (400) instead of silently letting it through -- 0 (the default) skips that upper-bound
// check, so existing callers that don't have a vocab size handy are unaffected. Negative token
// ids and malformed keys are always rejected regardless of vocab.
//
// legacy_logprobs switches the `logprobs` field's wire shape: false (default, chat completions)
// requires a boolean, paired with a separate `top_logprobs` integer field; true (the legacy
// /v1/completions endpoint) requires `logprobs` itself to be an integer in [0,20] ("how many top
// logprobs per token"), mapped onto out.logprobs = (value > 0) and out.top_logprobs = value --
// there is no separate top_logprobs field in legacy mode. The two modes deliberately do not
// accept each other's shape (a boolean is rejected in legacy mode, an integer is rejected in
// chat mode).
bool parse_request_controls(const std::string& body, RequestControls& out, std::string& err,
                            int vocab = 0, bool legacy_logprobs = false);

// Parses the legacy-completions-only fields (prompt/echo/suffix/best_of) that have no
// chat-completions equivalent -- call this AND parse_request_controls against the same request
// body for the /v1/completions endpoint. prompt_out/echo_out are left at their default (empty
// string, false) on any validation failure.
//
// v1 scope: prompt must be a single string (an array of strings/pre-tokenized ids -- both valid
// in OpenAI's own API -- is rejected with a clear error rather than silently only handling one).
// suffix (fill-in-the-middle) is rejected if set -- unsupported. best_of is rejected if set to
// anything other than its default (1) -- ranking multiple server-side completions by cumulative
// logprob has no existing scoring primitive in this runtime, out of scope for v1.
bool parse_legacy_completion_request(const std::string& body, std::string& prompt_out,
                                     bool& echo_out, std::string& err);

// POST /v1/score (teacher-forced scoring) request fields. Split out of the handler so the
// validation has a unit-test seam, exactly like parse_chat_request_json/parse_request_controls.
//
// Exactly one of messages/prompt, and exactly one of completion/completion_token_ids, must be
// supplied; supplying both or neither of a pair is an error rather than a silent precedence rule,
// because a verifier that thinks it pinned token ids and actually got its text re-tokenised would
// compare the wrong thing and never know.
struct ScoreRequest {
    // True => the caller sent `messages` and the chat template must be applied to it. The messages
    // themselves are not parsed here: the caller hands the SAME body to the chat tokenizer, which
    // already owns that parse.
    bool use_messages = false;
    std::string prompt;                     // raw prompt text, when use_messages is false
    bool completion_is_ids = false;         // true => completion_token_ids, false => completion text
    std::string completion;                 // completion text, when completion_is_ids is false
    std::vector<int> completion_token_ids;  // when completion_is_ids is true
    int top_logprobs = 0;                   // 0-20
};

// vocab, when > 0, rejects any completion_token_ids entry outside [0, vocab). 0 skips that check
// (the ids are still required to be non-negative integers), so a caller without a vocab handy can
// still validate everything else.
bool parse_score_request(const std::string& body, ScoreRequest& out, std::string& err,
                         int vocab = 0);

// Pure decision function for the DFlash+temperature-sampling incompatibility (kept separate from
// getenv() so it's unit-testable without a process-wide env var): true => the request should be
// rejected with 400. temperature<=0 is always accepted regardless of dflash_env_on.
bool should_reject_dflash_temperature(bool dflash_env_on, float temperature);

// Pure decision function for the DFlash+presence/frequency-penalty incompatibility -- mirrors
// should_reject_dflash_temperature, but is a SEPARATE check (not folded into it) because penalty
// has no inertness proof at temperature<=0 the way top_k/top_p do: a nonzero penalty CAN change
// the greedy-argmax winner on its own. true => the request should be rejected with 400.
// presence_penalty==0 && frequency_penalty==0 is always accepted regardless of dflash_env_on.
bool should_reject_dflash_penalty(bool dflash_env_on, float presence_penalty, float frequency_penalty);

// Pure decision function for the DFlash+logit_bias incompatibility -- mirrors
// should_reject_dflash_penalty; logit_bias has the same "no inertness proof at temperature<=0" gap
// (an arbitrary per-vocab additive bias CAN change the greedy-argmax winner on its own). true =>
// the request should be rejected with 400. An empty logit_bias is always accepted regardless of
// dflash_env_on.
bool should_reject_dflash_logit_bias(bool dflash_env_on, bool has_logit_bias);

// Best-effort validation only -- there is no constrained decoding in this backend, so this
// cannot guarantee the model's output actually conforms; it just checks after the fact.
// format.type == kText always returns true (no-op).
bool validate_response_format(const std::string& content, const ResponseFormat& format,
                              std::string& err);

// inject_reasoning_effort: Qwen3.8-27B's pinned chat_template.jinja unconditionally prepends a
// "Reasoning effort is set to xhigh..." system message whenever thinking is enabled -- regardless
// of tools/response_format -- which Qwen3.6/Muse Glimmer's own templates never do. Defaults false
// so every existing caller (Qwen3.6, tests) is byte-for-byte unchanged.
std::string apply_qwen36_tools_template(const ChatRequest& request,
                                        bool enable_thinking = false,
                                        bool inject_reasoning_effort = false);

ParsedToolOutput parse_qwen36_tool_output(const std::string& raw,
                                          bool enable_thinking,
                                          const ChatRequest& request);

}  // namespace sparkinfer_server
