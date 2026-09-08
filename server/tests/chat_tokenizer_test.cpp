#include "chat_tokenizer.hpp"

#include <cstdio>
#include <string>
#include <utility>

namespace {

#define CHECK(expr)                                                                            \
    do {                                                                                       \
        if (!(expr)) {                                                                         \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #expr);             \
            return false;                                                                      \
        }                                                                                      \
    } while (0)

bool test_thinking_prompt_and_nonstream_parser() {
    sparkinfer_server::ChatRequest request;
    sparkinfer_server::ChatMessage message;
    message.role = "user";
    message.content = "Explain briefly.";
    request.messages.push_back(std::move(message));
    const std::string prompt = sparkinfer_server::apply_qwen36_tools_template(request, true);
    const std::string suffix = "<|im_start|>assistant\n<think>\n";
    CHECK(prompt.size() >= suffix.size());
    CHECK(prompt.compare(prompt.size() - suffix.size(), suffix.size(), suffix) == 0);

    const auto parsed = sparkinfer_server::parse_assistant_output(
        "private reasoning\n</think>\n\nPublic answer.<|im_end|>", true, false);
    CHECK(parsed.error.empty());
    CHECK(parsed.reasoning_content == "private reasoning");
    CHECK(parsed.content == "Public answer.");
    CHECK(parsed.tool_calls.empty());

    // Defensively tolerate a model that repeats the already-prefilled opening marker.
    const auto repeated = sparkinfer_server::parse_assistant_output(
        "<think>\nreason\n</think>\nanswer", true, false);
    CHECK(repeated.reasoning_content == "reason");
    CHECK(repeated.content == "answer");

    // Same tolerance when the repeated marker isn't the very first byte (e.g. a stray leading
    // token before it) -- the marker itself must never leak into reasoning_content.
    const auto repeated_not_at_start = sparkinfer_server::parse_assistant_output(
        " <think>\nreason\n</think>\nanswer", true, false);
    CHECK(repeated_not_at_start.reasoning_content == "reason");
    CHECK(repeated_not_at_start.content == "answer");
    CHECK(repeated_not_at_start.reasoning_content.find("<think>") == std::string::npos);
    return true;
}

bool test_thinking_stream_boundaries() {
    sparkinfer_server::ThinkingStreamSplitter splitter(true, false);
    std::string reasoning;
    std::string content;
    for (const char* piece : {"private ", "reasoning\n</thi", "nk>\n\nPublic ",
                              "answer.<|im_", "end|>"}) {
        const auto delta = splitter.feed(piece);
        reasoning += delta.reasoning_content;
        content += delta.content;
    }
    sparkinfer_server::ThinkingStreamSplitter::Delta tail;
    splitter.finish(tail);
    reasoning += tail.reasoning_content;
    content += tail.content;
    CHECK(reasoning == "private reasoning\n");
    CHECK(content == "Public answer.");
    CHECK(content.find("<think>") == std::string::npos);
    CHECK(content.find("</think>") == std::string::npos);
    CHECK(content.find("<|im_end|>") == std::string::npos);
    return true;
}

bool test_thinking_stream_repeated_opening_marker() {
    // Streaming starts directly in the "inside <think>" phase (the prompt already primed
    // "<think>\n"), but must still tolerate -- and strip -- a defensively repeated opening
    // marker the same way the non-streaming parser does, rather than leaking it into
    // reasoning_content.
    sparkinfer_server::ThinkingStreamSplitter splitter(true, false);
    std::string reasoning;
    std::string content;
    for (const char* piece : {"<thi", "nk>\nreason\n</thi", "nk>\nanswer"}) {
        const auto delta = splitter.feed(piece);
        reasoning += delta.reasoning_content;
        content += delta.content;
    }
    sparkinfer_server::ThinkingStreamSplitter::Delta tail;
    splitter.finish(tail);
    reasoning += tail.reasoning_content;
    content += tail.content;
    // Streaming reasoning_content chunks aren't leading-whitespace-trimmed the way the
    // non-streaming parser's final string is (pre-existing, orthogonal to this fix) -- the
    // marker itself must simply never appear in the output.
    CHECK(reasoning == "\nreason\n");
    CHECK(content == "answer");
    CHECK(reasoning.find("<think>") == std::string::npos);
    return true;
}

bool test_nonthinking_stream_hides_terminal_marker() {
    sparkinfer_server::ThinkingStreamSplitter splitter(false, false);
    std::string content;
    for (const char* piece : {"hello", "<|im_", "end|>"})
        content += splitter.feed(piece).content;
    sparkinfer_server::ThinkingStreamSplitter::Delta tail;
    splitter.finish(tail);
    content += tail.content;
    CHECK(content == "hello");
    return true;
}

bool test_stop_filter_full_match_single_chunk() {
    sparkinfer_server::StopSequenceFilter filter({"STOP"});
    const std::string safe = filter.feed("hello STOP world");
    CHECK(filter.matched());
    CHECK(safe == "hello ");
    return true;
}

bool test_stop_filter_match_split_across_chunks() {
    sparkinfer_server::StopSequenceFilter filter({"STOP"});
    std::string safe = filter.feed("hello ST");
    CHECK(!filter.matched());
    CHECK(safe == "hello ");  // "ST" held back as a possible partial prefix of "STOP"
    safe += filter.feed("OP more");
    CHECK(filter.matched());
    CHECK(safe == "hello ");
    return true;
}

bool test_stop_filter_multiple_stops_earliest_wins() {
    sparkinfer_server::StopSequenceFilter filter({"world", "STOP"});
    const std::string safe = filter.feed("hello STOP world");
    CHECK(filter.matched());
    CHECK(safe == "hello ");  // "STOP" occurs first even though it's second in the list
    return true;
}

bool test_stop_filter_prefix_of_another_stop() {
    // Shorter stop fires as soon as it completes, regardless of list order -- matches real
    // OpenAI/vLLM/llama.cpp behavior, not a bug.
    sparkinfer_server::StopSequenceFilter filter({"a", "ab"});
    const std::string safe = filter.feed("xy a z");
    CHECK(filter.matched());
    CHECK(safe == "xy ");
    return true;
}

bool test_stop_filter_no_match_never_swallows_permanently() {
    sparkinfer_server::StopSequenceFilter filter({"NEVER_APPEARS"});
    std::string safe;
    for (const char* piece : {"hello ", "there ", "world"}) safe += filter.feed(piece);
    CHECK(!filter.matched());
    CHECK(safe == "hello there world");
    return true;
}

bool test_stop_filter_empty_stop_list_is_passthrough() {
    sparkinfer_server::StopSequenceFilter filter({});
    const std::string safe = filter.feed("anything at all");
    CHECK(!filter.matched());
    CHECK(safe == "anything at all");
    return true;
}

bool test_stop_filter_marker_with_embedded_nul() {
    // A JSON string can encode an embedded NUL byte; std::string preserves it but strlen()
    // would silently truncate there. Indirectly exercises the marker_prefix_len(std::string)
    // overload StopSequenceFilter relies on for holdback -- if it truncated at the NUL, this
    // stop string would falsely match on "A" alone instead of requiring the full "A\0B".
    const std::string stop_with_nul = std::string("A\0B", 3);
    sparkinfer_server::StopSequenceFilter filter({stop_with_nul});
    std::string safe = filter.feed(std::string("xA", 2));
    CHECK(!filter.matched());  // "A" alone must not be treated as a full match
    safe += filter.feed(std::string("\0By", 3));
    CHECK(filter.matched());
    CHECK(safe == "x");
    return true;
}

bool test_muse_rejects_all_tool_protocol_history() {
    sparkinfer_server::ChatRequest parsed;
    std::string error;
    const std::string request = R"JSON({
      "messages":[
        {"role":"assistant","content":null,"tool_calls":[{
          "id":"call_1","type":"function",
          "function":{"name":"lookup","arguments":"{\"query\":\"x\"}"}
        }]},
        {"role":"tool","tool_call_id":"call_1","content":"ok"},
        {"role":"user","content":"continue"}
      ],
      "tools":[{"type":"function","function":{
        "name":"lookup",
        "parameters":{"type":"object","properties":{"query":{"type":"string"}}}
      }}],
      "tool_choice":"none"
    })JSON";
    CHECK(sparkinfer_server::parse_chat_request_json(request, parsed, error));
    CHECK(!sparkinfer_server::validate_chat_request_model_support(parsed, true, error));
    // The second argument is "this model has no tool-call renderer/parser" -- it covers Muse
    // Glimmer and Spark-X2.5, so the message names neither model family.
    CHECK(error.find("tool calling is not supported for this model") != std::string::npos);
    error.clear();
    CHECK(sparkinfer_server::validate_chat_request_model_support(parsed, false, error));
    return true;
}

bool test_tool_choice_none_still_uses_strict_output_parser() {
    const std::string request_json = R"JSON({
      "messages":[{"role":"user","content":"Do not call tools."}],
      "tools":[{"type":"function","function":{
        "name":"terminal",
        "parameters":{"type":"object","properties":{"command":{"type":"string"}}}
      }}],
      "tool_choice":"none"
    })JSON";
    sparkinfer_server::ChatRequest request;
    std::string error;
    CHECK(sparkinfer_server::parse_chat_request_json(request_json, request, error));
    const auto forbidden = sparkinfer_server::parse_assistant_output(
        "<tool_call>\n<function=terminal>\n<parameter=command>\nid\n</parameter>\n"
        "</function>\n</tool_call>",
        false, false, &request);
    CHECK(!forbidden.error.empty());
    CHECK(forbidden.content.empty());
    CHECK(forbidden.tool_calls.empty());

    const auto answer = sparkinfer_server::parse_assistant_output(
        "No tool used.<|im_end|>", false, false, &request);
    CHECK(answer.error.empty());
    CHECK(answer.content == "No tool used.");
    return true;
}

bool test_openrouter_reasoning_enablement() {
    CHECK(sparkinfer_server::parse_enable_thinking(
        R"({"reasoning":{"enabled":true}})", false));
    CHECK(!sparkinfer_server::parse_enable_thinking(
        R"({"reasoning":{"enabled":false,"effort":"high"}})", true));
    CHECK(sparkinfer_server::parse_enable_thinking(
        R"({"reasoning":{"effort":"low"}})", false));
    CHECK(sparkinfer_server::parse_enable_thinking(
        R"({"reasoning_effort":"medium"})", false));
    CHECK(!sparkinfer_server::parse_enable_thinking(
        R"({"reasoning":{"effort":"none"}})", true));
    CHECK(sparkinfer_server::parse_enable_thinking(
        R"({"reasoning":{"max_tokens":512}})", false));
    return true;
}

}  // namespace

// GPT-2/ByteLevel-BPE byte-table round-trip -- pure function, no loaded tokenizer needed. "Ġ" is
// U+0120 (UTF-8 0xC4 0xA0), the standard byte-level-BPE stand-in for the space byte 0x20.
bool test_gpt2_bytelevel_decode_space_marker() {
    const auto piece = sparkinfer_server::gpt2_bytelevel_decode("\xC4\xA0hello");
    CHECK(!piece.bytes.empty());
    CHECK(piece.bytes[0] == 0x20);
    CHECK(piece.display == " hello");
    return true;
}

// U+0122 (UTF-8 0xC4 0xA2) is the byte-level-BPE stand-in for raw byte 0x80 -- a lone UTF-8
// continuation byte, not valid standalone text. display must show a replacement character
// (matches real OpenAI's behavior for tokens that split a multi-byte character).
bool test_gpt2_bytelevel_decode_invalid_utf8_shows_replacement() {
    const auto piece = sparkinfer_server::gpt2_bytelevel_decode("\xC4\xA2");
    CHECK(piece.bytes.size() == 1 && piece.bytes[0] == 0x80);
    CHECK(piece.display.find("\xEF\xBF\xBD") != std::string::npos);
    return true;
}

bool test_gpt2_bytelevel_decode_empty_piece() {
    const auto piece = sparkinfer_server::gpt2_bytelevel_decode("");
    CHECK(piece.bytes.empty());
    CHECK(piece.display.empty());
    return true;
}

// Printable ASCII maps to itself under the byte-level table -- a plain word round-trips exactly.
bool test_gpt2_bytelevel_decode_printable_ascii_roundtrip() {
    const auto piece = sparkinfer_server::gpt2_bytelevel_decode("hello");
    CHECK(piece.bytes.size() == 5);
    CHECK(piece.display == "hello");
    return true;
}

// Strict UTF-8 validator. The tokenizer's Rust side rejects invalid UTF-8 by panicking without
// unwinding -- it aborts the process rather than returning an error -- so the prompt has to be
// checked here, before it ever reaches Encode().
bool is_valid_utf8(const std::string& s) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(s.data());
    size_t i = 0, n = s.size();
    while (i < n) {
        const unsigned char c = p[i];
        size_t extra;
        unsigned int cp;
        if (c < 0x80) { i++; continue; }
        else if ((c & 0xE0) == 0xC0) { extra = 1; cp = c & 0x1Fu; }
        else if ((c & 0xF0) == 0xE0) { extra = 2; cp = c & 0x0Fu; }
        else if ((c & 0xF8) == 0xF0) { extra = 3; cp = c & 0x07u; }
        else return false;                       // continuation byte or 5+ byte lead
        if (i + extra >= n) return false;   // truncated sequence at end of string
        for (size_t k = 1; k <= extra; k++) {
            if ((p[i + k] & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (p[i + k] & 0x3Fu);
        }
        // Reject overlong encodings, surrogates and out-of-range code points.
        if (extra == 1 && cp < 0x80) return false;
        if (extra == 2 && cp < 0x800) return false;
        if (extra == 3 && cp < 0x10000) return false;
        if (cp > 0x10FFFF) return false;
        if (cp >= 0xD800 && cp <= 0xDFFF) return false;
        i += extra + 1;
    }
    return true;
}

// Spark-X2.5's rendered prompt, byte for byte against chat_template.jinja.
//
// The point of asserting the FULL string rather than spot-checking markers: the turn envelope is
// built from hex escapes, and a C++ hex escape swallows every hex digit after it -- "\x9cend"
// lexes as \x9ce, not \x9c followed by "end". That mistake yields a prompt that is invalid UTF-8
// from the first end-of-sentence marker on, which the tokenizer answers with a non-unwinding Rust
// panic (it aborts the process, so it cannot even be caught). Comparing whole strings is what
// catches it; comparing lengths or searching for a substring would not.
bool test_spark25_chat_template_renders_exact_markers() {
    const std::string sos = "<\xef\xbd\x9c" "start" "\xe2\x96\x81" "of" "\xe2\x96\x81"
                            "sentence" "\xef\xbd\x9c" ">";
    const std::string eos = "<\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81"
                            "sentence" "\xef\xbd\x9c" ">";
    // Both markers must be well-formed UTF-8; the bug above makes eos specifically malformed.
    CHECK(is_valid_utf8(sos));
    CHECK(is_valid_utf8(eos));
    CHECK(sos.size() == 29 && eos.size() == 27);

    std::vector<sparkinfer_server::ChatMessage> messages;
    { sparkinfer_server::ChatMessage m; m.role = "user"; m.content = "hi"; messages.push_back(m); }

    // enable_thinking=false closes the reasoning block immediately in the generation prompt.
    const std::string got = sparkinfer_server::apply_spark25_chat_template(messages, false);
    const std::string want = sos + "<|System|>\nyou are a helpful assistant." + eos +
                             sos + "<|User|>hi" + eos +
                             sos + "<|Bot|></think>";
    if (got != want) {
        std::printf("FAIL: spark25 template\n  got:  %s\n  want: %s\n", got.c_str(), want.c_str());
        return false;
    }
    CHECK(is_valid_utf8(got));

    // enable_thinking=true opens it instead.
    const std::string think = sparkinfer_server::apply_spark25_chat_template(messages, true);
    CHECK(think == sos + "<|System|>\nyou are a helpful assistant." + eos +
                   sos + "<|User|>hi" + eos + sos + "<|Bot|><think>");

    // A leading system message is folded into the initial block (appended after the default
    // system prompt), not emitted as its own turn -- matching the template's messages[0] handling.
    std::vector<sparkinfer_server::ChatMessage> with_sys;
    { sparkinfer_server::ChatMessage m; m.role = "system"; m.content = "Be terse."; with_sys.push_back(m); }
    { sparkinfer_server::ChatMessage m; m.role = "user"; m.content = "hi"; with_sys.push_back(m); }
    const std::string sys = sparkinfer_server::apply_spark25_chat_template(with_sys, false);
    CHECK(sys == sos + "<|System|>\nyou are a helpful assistant.\n\nBe terse." + eos +
                 sos + "<|User|>hi" + eos + sos + "<|Bot|></think>");

    // Consecutive tool results share ONE envelope: open on the first, close after the last.
    std::vector<sparkinfer_server::ChatMessage> tools;
    { sparkinfer_server::ChatMessage m; m.role = "user"; m.content = "q"; tools.push_back(m); }
    { sparkinfer_server::ChatMessage m; m.role = "tool"; m.content = "a1"; tools.push_back(m); }
    { sparkinfer_server::ChatMessage m; m.role = "tool"; m.content = "a2"; tools.push_back(m); }
    const std::string tl = sparkinfer_server::apply_spark25_chat_template(tools, false);
    CHECK(tl == sos + "<|System|>\nyou are a helpful assistant." + eos +
                sos + "<|User|>q" + eos +
                sos + "<|Tool|><tool_response>a1</tool_response><tool_response>a2</tool_response>" + eos +
                sos + "<|Bot|></think>");
    return true;
}

int main() {
    if (!test_thinking_prompt_and_nonstream_parser()) return 1;
    if (!test_thinking_stream_boundaries()) return 1;
    if (!test_thinking_stream_repeated_opening_marker()) return 1;
    if (!test_nonthinking_stream_hides_terminal_marker()) return 1;
    if (!test_stop_filter_full_match_single_chunk()) return 1;
    if (!test_stop_filter_match_split_across_chunks()) return 1;
    if (!test_stop_filter_multiple_stops_earliest_wins()) return 1;
    if (!test_stop_filter_prefix_of_another_stop()) return 1;
    if (!test_stop_filter_no_match_never_swallows_permanently()) return 1;
    if (!test_stop_filter_empty_stop_list_is_passthrough()) return 1;
    if (!test_stop_filter_marker_with_embedded_nul()) return 1;
    if (!test_muse_rejects_all_tool_protocol_history()) return 1;
    if (!test_tool_choice_none_still_uses_strict_output_parser()) return 1;
    if (!test_openrouter_reasoning_enablement()) return 1;
    if (!test_gpt2_bytelevel_decode_space_marker()) return 1;
    if (!test_gpt2_bytelevel_decode_invalid_utf8_shows_replacement()) return 1;
    if (!test_gpt2_bytelevel_decode_empty_piece()) return 1;
    if (!test_gpt2_bytelevel_decode_printable_ascii_roundtrip()) return 1;
    if (!test_spark25_chat_template_renders_exact_markers()) return 1;
    std::printf("chat_tokenizer_test: OK\n");
    return 0;
}
