#pragma once

#include "chat_tools.hpp"
#include <mutex>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sparkinfer_server {

// One token's raw byte-level-BPE bytes + a UTF-8-safe display rendering of them, for
// logprobs/top_logprobs' `token`/`bytes` fields. See gpt2_bytelevel_decode's doc comment.
struct RawTokenPiece {
    std::vector<uint8_t> bytes;
    std::string display;
};

// Reverses the standard GPT-2/ByteLevel-BPE byte<->unicode mapping (the same fixed 256-entry
// table used by every ByteLevel BPE tokenizer -- both models this server supports use
// decoder.type: "ByteLevel") to recover a single vocab piece's true raw UTF-8 bytes, then builds
// a display string via a lossy UTF-8 re-encode of those bytes (U+FFFD substitution for any
// sequence that doesn't form valid UTF-8 on its own -- real OpenAI shows the same kind of
// replacement-character artifacts for tokens that split a multi-byte character). Pure function,
// no tokenizer object needed -- exposed directly for unit testing without a loaded tokenizer.json.
RawTokenPiece gpt2_bytelevel_decode(const std::string& raw_piece);

// HuggingFace tokenizer.json + Qwen3.6 chat template.
class ChatTokenizer {
public:
    ChatTokenizer();
    ~ChatTokenizer();

    bool load(const std::string& tokenizer_json_path, std::string& err);

    // Selects the chat-template/parsing dialect: Qwen3.6's <think> tags (default) vs Muse
    // Glimmer's harmony-style <|start|>/<|message|>/<|eot|>/<|eom|> segments. Call once after
    // the model's architecture is known (ModelEngine::is_museglimmer()), before the first request.
    void set_museglimmer(bool on);
    // Qwen3.8-27B's pinned chat_template.jinja unconditionally prepends a reasoning-effort
    // system message (see apply_qwen36_tools_template's inject_reasoning_effort param). Call
    // once after the model's architecture is known (ModelEngine::is_qwen38()), before the first
    // request. Independent of set_museglimmer -- the two are mutually exclusive model families.
    void set_qwen38(bool on);
    // Spark-X2.5 (model_type "spark2_5"). Set once after the model's architecture is known
    // (ModelEngine::is_spark25()), before the first request. Mutually exclusive with the two
    // above -- its template shares no markers with either.
    void set_spark25(bool on);

    bool encode_chat_request(const std::string& request_json, std::vector<int>& ids, bool enable_thinking,
                             std::string& err, ChatRequest* parsed_request = nullptr) const;
    // Directly renders + tokenizes an already-parsed/validated ChatRequest, skipping the JSON
    // body parse step encode_chat_request does. Used to build a retry prompt (e.g. a
    // response_format correction turn appended to the original request) without round-tripping
    // through a serialized JSON body. Returns an empty vector if the tokenizer isn't loaded.
    std::vector<int> encode_augmented(const ChatRequest& request, bool enable_thinking) const;
    // Plain string -> ids, no chat template applied -- for the legacy /v1/completions endpoint's
    // raw prompt (unlike encode_chat_request/encode_augmented, which both wrap the input in the
    // Qwen3.6/Muse Glimmer chat template first). Returns an empty vector if the tokenizer isn't
    // loaded.
    std::vector<int> encode_raw(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
    std::string decode_delta(std::vector<int>& acc, int new_id) const;

    // Raw single-id vocab lookup + ByteLevel decode, for logprobs/top_logprobs -- see
    // gpt2_bytelevel_decode. Deliberately NOT built on decode()/decode_delta() (which are
    // context-sensitive, designed around incremental multi-token diffing): this is a pure,
    // stable, single-id lookup. Ids past the tokenizer's real vocab range (GGUF embedding-table
    // padding -- e.g. 248,320 vs 248,070 real ids for Qwen3.6) return an empty RawTokenPiece
    // (empty bytes, empty display) -- essentially never realistically selected, not specially
    // handled beyond that.
    RawTokenPiece id_to_raw_piece(int id) const;

private:
    // Serializes every call that reaches the tokenizers-cpp handle.
    //
    // NOT defensive: that handle is genuinely not thread-safe. tokenizers-cpp's Rust FFI keeps
    // per-handle scratch on the wrapper struct --
    //
    //     pub struct TokenizerWrapper { decode_str: String, id_to_token_result: String, ... }
    //     extern "C" fn tokenizers_id_to_token(...) { (*handle).id_to_token_result = ...; }
    //
    // -- and returns a pointer INTO that field. Two threads calling id_to_token concurrently
    // reallocate the same String under each other, so one frees a buffer the other is reading or
    // writing. That is glibc heap corruption, and it presents exactly as reported from a
    // 16-concurrent burst: "malloc(): unaligned tcache chunk detected",
    // "corrupted size vs. prev_size", std::length_error, SIGSEGV -- with no CUDA error anywhere,
    // because the device was never involved.
    //
    // logprobs:true is what makes it reachable in practice: token_logprob_entry_json() calls
    // id_to_raw_piece() once per emitted token AND once per top_logprobs alternative, so a
    // logprobs request drives roughly (1 + top_logprobs) x tokens lookups per request, from one
    // httplib thread per request. Without logprobs the call rate is low enough that threads
    // rarely overlap inside the FFI.
    //
    // Held here rather than at the ~10 call sites so a new caller cannot forget. The critical
    // sections are microseconds of vocab lookup; the contended path is JSON assembly, not
    // generation, so this does not serialize decoding.
    // RECURSIVE: encode_chat_request() calls encode_augmented(), and decode_delta() calls
    // decode() twice -- both would self-deadlock on a plain mutex, which under load is a hung
    // server rather than a crashed one. Recursion only permits same-thread re-entry; concurrent
    // threads still serialize, which is the whole point.
    mutable std::recursive_mutex tok_mu_;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct ParsedAssistantOutput {
    std::string reasoning_content;
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::string error;
};

// Incrementally routes decoded text into reasoning vs answer for SSE streaming.
class ThinkingStreamSplitter {
public:
    explicit ThinkingStreamSplitter(bool enable_thinking, bool museglimmer = false);

    struct Delta {
        std::string reasoning_content;
        std::string content;
    };
    Delta feed(const std::string& piece);
    void finish(Delta& tail);

private:
    bool enable_thinking_ = false;
    enum class Phase { kBeforeThink, kInThink, kInAnswer } phase_ = Phase::kBeforeThink;
    std::string carry_;
    std::string prefix_buffer_;
    // Qwen (non-museglimmer) starts in Phase::kInThink directly (the prompt already primes
    // "<think>\n"), but a generation can still defensively repeat the opening marker -- see
    // parse_assistant_output's identical handling for the non-streaming path. Checked once,
    // on the first non-empty data seen in kInThink, so a literal "<think>" appearing later in
    // genuine reasoning text is never mistaken for the marker.
    bool think_open_checked_ = false;

    // Muse Glimmer (harmony-style) state: segments are <|start|>{header}<|message|>{body}
    // then <|eom|> (more segments follow, same assistant turn) or <|eot|> (turn done). The
    // header names the recipient -- "assistant to=self" routes body to reasoning_content,
    // anything else (typically "assistant to=user") routes it to content.
    bool museglimmer_ = false;
    enum class MgPhase { kHeader, kAwaitStart, kBody } mg_phase_ = MgPhase::kHeader;
    // Primed with "assistant" -- the prompt itself already emits "<|start|>assistant" before
    // handing off to the model, so the first segment's generated header text picks up right
    // after that word (e.g. " to=self"), unlike later segments which regenerate it in full.
    std::string mg_header_ = "assistant";
    bool mg_is_self_ = false;
    bool mg_done_ = false;
};

// Detects OpenAI-style client-supplied `stop` sequences in a live, incrementally-decoded token
// stream. Generalizes the single-compile-time-marker holdback pattern ThinkingStreamSplitter
// uses for <|im_end|>/<think> to an arbitrary list of up to 4 dynamic strings: text that could
// still be an in-progress prefix of any stop string is buffered (never returned) until a later
// feed() either completes it into a full match (matched() becomes true; the held-back bytes and
// the matched text itself are permanently dropped, never returned) or disproves it (the
// held-back bytes are released as ordinary text). An empty stop list makes feed() a cheap
// passthrough, so it's safe to always construct one per request.
class StopSequenceFilter {
public:
    explicit StopSequenceFilter(std::vector<std::string> stops);

    // Feeds the next raw decoded delta. Returns the subset of text now known not to be part of
    // a stop match, safe to forward downstream immediately. Once matched() is true, feed() must
    // not be called again.
    std::string feed(const std::string& piece);

    bool matched() const { return matched_; }

private:
    std::vector<std::string> stops_;
    std::string carry_;
    bool matched_ = false;
};

bool parse_chat_messages(const std::string& request_json, std::vector<ChatMessage>& messages, std::string& err);
bool parse_enable_thinking(const std::string& request_json, bool default_value = false);
// no_tool_support: this model has no tool-call renderer/parser wired up (Muse Glimmer,
// Spark-X2.5). Rejects tools, tool history and structured response_format for it.
bool validate_chat_request_model_support(const ChatRequest& request, bool no_tool_support,
                                         std::string& err);
std::string apply_qwen36_chat_template(const std::vector<ChatMessage>& messages, bool enable_thinking = false);
std::string apply_museglimmer_chat_template(const std::vector<ChatMessage>& messages,
                                            const std::string& reasoning_strength = "high");
// Spark-X2.5's own template (chat_template.jinja in XHToken/Spark-X2.5-4B). Distinct markers
// from every other model here: a per-turn <|start_of_sentence|>/<|end_of_sentence|> envelope with
// <|System|>/<|User|>/<|Bot|>/<|Tool|> role tags, and an explicit <think> or </think> opener on the
// generation prompt rather than a separate reasoning-effort message.
std::string apply_spark25_chat_template(const std::vector<ChatMessage>& messages,
                                        bool enable_thinking = true);
ParsedAssistantOutput parse_assistant_output(const std::string& raw, bool enable_thinking,
                                             bool museglimmer = false,
                                             const ChatRequest* request = nullptr);

}  // namespace sparkinfer_server
