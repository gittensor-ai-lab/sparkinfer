#include "chat_tokenizer.hpp"
#include <mutex>

#include <nlohmann/json.hpp>
#include <tokenizers_cpp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace sparkinfer_server {
namespace {

// Standard GPT-2/ByteLevel-BPE byte<->unicode table (the same fixed 256-entry mapping used by
// every ByteLevel BPE tokenizer, e.g. HuggingFace's bytes_to_unicode()): printable ASCII (! .. ~)
// and printable Latin-1 supplement ranges map to themselves; the remaining ~68 byte values
// (control chars, space, DEL, C1 controls -- ones that would collide with whitespace/invisible
// characters if used as literal vocab-piece text) get remapped to codepoints starting at 0x100,
// in ascending byte order.
std::array<uint32_t, 256> gpt2_byte_to_unicode_table() {
    std::array<uint32_t, 256> table{};
    std::array<bool, 256> nice{};
    auto mark_range = [&](int lo, int hi) {
        for (int b = lo; b <= hi; b++) { table[b] = (uint32_t)b; nice[b] = true; }
    };
    mark_range(0x21, 0x7E);   // '!' .. '~'
    mark_range(0xA1, 0xAC);   // inverted-! .. not-sign
    mark_range(0xAE, 0xFF);   // registered-sign .. y-diaeresis
    uint32_t n = 0;
    for (int b = 0; b < 256; b++) {
        if (!nice[b]) { table[b] = 256 + n; n++; }
    }
    return table;
}

const std::unordered_map<uint32_t, uint8_t>& gpt2_unicode_to_byte_table() {
    static const std::unordered_map<uint32_t, uint8_t> inv = [] {
        std::unordered_map<uint32_t, uint8_t> m;
        const auto b2u = gpt2_byte_to_unicode_table();
        m.reserve(256);
        for (int b = 0; b < 256; b++) m[b2u[b]] = (uint8_t)b;
        return m;
    }();
    return inv;
}

// Permissive UTF-8 decode of a (trusted, tokenizer-vocab-derived) string into codepoints.
// Malformed leads/continuations are simply skipped -- vocab pieces are always well-formed UTF-8
// in practice, this is defensive, not a general-purpose decoder.
std::vector<uint32_t> utf8_decode(const std::string& s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c0 = (unsigned char)s[i];
        uint32_t cp;
        int len;
        if (c0 < 0x80) { cp = c0; len = 1; }
        else if ((c0 & 0xE0) == 0xC0) { cp = c0 & 0x1F; len = 2; }
        else if ((c0 & 0xF0) == 0xE0) { cp = c0 & 0x0F; len = 3; }
        else if ((c0 & 0xF8) == 0xF0) { cp = c0 & 0x07; len = 4; }
        else { i++; continue; }
        if (i + (size_t)len > s.size()) break;
        bool ok = true;
        for (int k = 1; k < len; k++) {
            const unsigned char ck = (unsigned char)s[i + k];
            if ((ck & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (ck & 0x3F);
        }
        if (!ok) { i++; continue; }
        out.push_back(cp);
        i += (size_t)len;
    }
    return out;
}

// Interprets raw bytes AS UTF-8, substituting U+FFFD (its own UTF-8 encoding, 0xEF 0xBF 0xBD)
// for any byte(s) that don't form a valid standalone sequence -- matches real OpenAI's behavior
// for byte-level-BPE tokens that split a multi-byte character across two token ids.
std::string utf8_lossy_from_bytes(const std::vector<uint8_t>& bytes) {
    std::string out;
    out.reserve(bytes.size());
    size_t i = 0;
    const size_t n = bytes.size();
    auto emit_replacement = [&] { out += "\xEF\xBF\xBD"; };
    while (i < n) {
        const uint8_t c0 = bytes[i];
        if (c0 < 0x80) { out.push_back((char)c0); i++; continue; }
        int len;
        uint32_t cp, min_cp;
        if ((c0 & 0xE0) == 0xC0) { len = 2; cp = c0 & 0x1F; min_cp = 0x80; }
        else if ((c0 & 0xF0) == 0xE0) { len = 3; cp = c0 & 0x0F; min_cp = 0x800; }
        else if ((c0 & 0xF8) == 0xF0) { len = 4; cp = c0 & 0x07; min_cp = 0x10000; }
        else { emit_replacement(); i++; continue; }
        if (i + (size_t)len > n) { emit_replacement(); i++; continue; }
        bool ok = true;
        for (int k = 1; k < len; k++) {
            const uint8_t ck = bytes[i + k];
            if ((ck & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (ck & 0x3F);
        }
        if (!ok || cp < min_cp || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
            emit_replacement();
            i++;
            continue;
        }
        out.append((const char*)&bytes[i], (size_t)len);
        i += (size_t)len;
    }
    return out;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

constexpr const char* kImEnd = "<|" "im_end|>";
constexpr const char* kThinkOpen = "<" "think>";
constexpr const char* kThinkClose = "</" "think>";

// Muse Glimmer harmony-style segment markers (server/scripts equivalent of Qwen3.6's
// im_start/im_end/think tags). See tokenizer.chat_template in the model's GGUF for the
// canonical jinja this mirrors. Muse's ATEM tool protocol is deliberately rejected at request
// validation until it receives its own renderer/parser; Qwen3.6 tools must never be sent here.
constexpr const char* kMgStart = "<|start|>";
constexpr const char* kMgMessage = "<|message|>";
constexpr const char* kMgEot = "<|eot|>";
constexpr const char* kMgEom = "<|eom|>";

size_t marker_prefix_len(const std::string& data, const char* marker) {
    const size_t n = strlen(marker);
    const size_t max = std::min(data.size(), n > 0 ? n - 1 : 0);
    for (size_t len = max; len > 0; len--) {
        if (data.compare(data.size() - len, len, marker, len) == 0) return len;
    }
    return 0;
}

// Same algorithm, but for a caller-supplied string whose length may include an embedded NUL
// byte (client input via JSON can encode one) -- strlen() would silently truncate at it, unlike
// the compile-time literal markers above where that can't happen.
size_t marker_prefix_len(const std::string& data, const std::string& marker) {
    const size_t n = marker.size();
    const size_t max = std::min(data.size(), n > 0 ? n - 1 : 0);
    for (size_t len = max; len > 0; len--) {
        if (data.compare(data.size() - len, len, marker, 0, len) == 0) return len;
    }
    return 0;
}

void trim_leading_ws(std::string& s) {
    while (!s.empty() && (s[0] == '\n' || s[0] == '\r' || s[0] == ' ' || s[0] == '\t')) s.erase(0, 1);
}

void trim_trailing_ws(std::string& s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t'))
        s.pop_back();
}

void strip_trailing_im_end(std::string& s) {
    const size_t n = strlen(kImEnd);
    if (s.size() >= n && s.compare(s.size() - n, n, kImEnd) == 0) s.resize(s.size() - n);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
}

std::string strip_think_markers(std::string s) {
    for (;;) {
        const size_t o = s.find(kThinkOpen);
        if (o == std::string::npos) break;
        const size_t c = s.find(kThinkClose, o + strlen(kThinkOpen));
        if (c == std::string::npos) {
            s.erase(o, strlen(kThinkOpen));
            continue;
        }
        s.erase(o, c + strlen(kThinkClose) - o);
    }
    for (;;) {
        const size_t c = s.find(kThinkClose);
        if (c == std::string::npos) break;
        s.erase(c, strlen(kThinkClose));
    }
    return s;
}

// Emit answer text while stripping any think markers that leak into the answer stream.
std::string filter_answer_chunk(std::string& carry, const std::string& piece) {
    std::string data = carry + piece;
    carry.clear();
    data = strip_think_markers(data);
    const size_t keep_open = marker_prefix_len(data, kThinkOpen);
    const size_t keep_close = marker_prefix_len(data, kThinkClose);
    const size_t keep = std::max(keep_open, keep_close);
    const size_t emit_len = data.size() - keep;
    std::string out;
    if (emit_len > 0) out = data.substr(0, emit_len);
    if (keep > 0) carry = data.substr(emit_len);
    return out;
}

}  // namespace

RawTokenPiece gpt2_bytelevel_decode(const std::string& raw_piece) {
    RawTokenPiece out;
    if (raw_piece.empty()) return out;
    const auto& unicode_to_byte = gpt2_unicode_to_byte_table();
    const auto codepoints = utf8_decode(raw_piece);
    out.bytes.reserve(codepoints.size());
    for (uint32_t cp : codepoints) {
        auto it = unicode_to_byte.find(cp);
        // A codepoint absent from the table shouldn't happen for a well-formed ByteLevel BPE
        // vocab piece -- skip rather than corrupt output if it ever does.
        if (it != unicode_to_byte.end()) out.bytes.push_back(it->second);
    }
    out.display = utf8_lossy_from_bytes(out.bytes);
    return out;
}

struct ChatTokenizer::Impl {
    std::unique_ptr<tokenizers::Tokenizer> tok;
    bool museglimmer = false;
    bool qwen38 = false;
    // Muse Glimmer harmony-format marker token ids, resolved once in set_museglimmer() (single
    // -threaded server startup, after `tok` is loaded) -- see decode()'s comment for why these
    // need special handling. Deliberately NOT lazily resolved on first decode(): decode() runs
    // on the shared g_tokenizer from every /v1/chat/completions request, and cpp-httplib
    // dispatches those across a real thread pool by default -- a lazy first-call init here
    // would race across concurrently-arriving requests.
    int32_t mg_start_id = -1, mg_message_id = -1, mg_eom_id = -1, mg_eot_id = -1;
};

ChatTokenizer::ChatTokenizer() : impl_(std::make_unique<Impl>()) {}
ChatTokenizer::~ChatTokenizer() = default;

bool ChatTokenizer::load(const std::string& tokenizer_json_path, std::string& err) {
    std::lock_guard<std::recursive_mutex> tok_lock(tok_mu_);
    const std::string blob = read_file(tokenizer_json_path);
    if (blob.empty()) {
        err = "cannot read tokenizer: " + tokenizer_json_path;
        return false;
    }
    try {
        impl_->tok = tokenizers::Tokenizer::FromBlobJSON(blob);
    } catch (const std::exception& e) {
        err = std::string("tokenizer load failed: ") + e.what();
        return false;
    }
    if (!impl_->tok) {
        err = "tokenizer load returned null";
        return false;
    }
    fprintf(stderr, "[sparkinfer-server] tokenizer loaded: %s (vocab=%zu)\n",
            tokenizer_json_path.c_str(), impl_->tok->GetVocabSize());
    return true;
}

void ChatTokenizer::set_qwen38(bool on) { impl_->qwen38 = on; }

void ChatTokenizer::set_museglimmer(bool on) {
    impl_->museglimmer = on;
    // Resolve the harmony-format marker ids here (single-threaded startup, `tok` already
    // loaded by this point) rather than lazily in decode() -- see the Impl::mg_start_id
    // comment for why. TokenToId returns -1 for an unknown token, which decode()'s
    // marker_str() already treats as "never matches", so a missing marker degrades
    // gracefully instead of crashing.
    if (on && impl_->tok) {
        impl_->mg_start_id = impl_->tok->TokenToId(kMgStart);
        impl_->mg_message_id = impl_->tok->TokenToId(kMgMessage);
        impl_->mg_eom_id = impl_->tok->TokenToId(kMgEom);
        impl_->mg_eot_id = impl_->tok->TokenToId(kMgEot);
    }
}

bool parse_chat_messages(const std::string& request_json, std::vector<ChatMessage>& messages, std::string& err) {
    ChatRequest request;
    if (!parse_chat_request_json(request_json, request, err)) return false;
    messages = std::move(request.messages);
    return true;
}

bool parse_enable_thinking(const std::string& request_json, bool default_value) {
    const auto root = nlohmann::json::parse(request_json, nullptr, false);
    if (root.is_discarded() || !root.is_object()) return default_value;
    const auto top = root.find("enable_thinking");
    if (top != root.end() && top->is_boolean()) return top->get<bool>();
    const auto reasoning = root.find("reasoning");
    if (reasoning != root.end() && reasoning->is_object()) {
        const auto enabled = reasoning->find("enabled");
        if (enabled != reasoning->end() && enabled->is_boolean()) return enabled->get<bool>();
        // OpenRouter's effort form requests reasoning even when it omits an explicit enabled flag.
        const auto effort = reasoning->find("effort");
        if (effort != reasoning->end() && effort->is_string()) return effort->get<std::string>() != "none";
        const auto budget = reasoning->find("max_tokens");
        if (budget != reasoning->end() && budget->is_number_integer()) return budget->get<long long>() > 0;
    }
    if (root.contains("reasoning_effort") && root["reasoning_effort"].is_string())
        return root["reasoning_effort"].get<std::string>() != "none";
    const auto kwargs = root.find("chat_template_kwargs");
    if (kwargs != root.end() && kwargs->is_object()) {
        const auto nested = kwargs->find("enable_thinking");
        if (nested != kwargs->end() && nested->is_boolean()) return nested->get<bool>();
    }
    return default_value;
}

bool validate_chat_request_model_support(const ChatRequest& request, bool museglimmer,
                                         std::string& err) {
    if (!museglimmer) return true;
    bool has_tool_history = false;
    for (const ChatMessage& message : request.messages) {
        if (message.role == "tool" || !message.tool_calls.empty()) {
            has_tool_history = true;
            break;
        }
    }
    if (!request.tools.empty() || has_tool_history) {
        err = "tool calling is currently supported only for Qwen3.6 models";
        return false;
    }
    if (request.response_format.type != ResponseFormatType::kText) {
        err = "structured response_format is currently supported only for Qwen3.6 models";
        return false;
    }
    return true;
}

std::string apply_qwen36_chat_template(const std::vector<ChatMessage>& messages, bool enable_thinking) {
    // Matches server/scripts/chat_tokens.py (thinking disabled) and HF enable_thinking=true.
    std::ostringstream parts;
    for (const auto& m : messages) {
        std::string role = m.role;
        for (auto& c : role) c = (char)tolower((unsigned char)c);
        parts << "<|im_start|>" << role << '\n' << m.content << kImEnd << '\n';
    }
    parts << "<|im_start|>assistant\n";
    if (!enable_thinking) parts << kThinkOpen << "\n\n" << kThinkClose << "\n\n";
    return parts.str();
}

// Matches tokenizer.chat_template in the Muse Glimmer GGUF for the plain (no tools, no
// tool_calls, no reasoning_content history) case -- the only case reachable today, since
// ChatMessage/parse_chat_messages carry no tool/reasoning fields. `current_date` is omitted
// (the upstream template only emits it when a `current_date`/`strftime_now` template var is
// supplied, which this server doesn't provide); `knowledge_cutoff` uses the template's own
// literal default.
std::string apply_museglimmer_chat_template(const std::vector<ChatMessage>& messages,
                                             const std::string& reasoning_strength) {
    std::ostringstream parts;
    bool has_system = false;
    for (const auto& m : messages)
        if (m.role == "system") { has_system = true; break; }

    if (!has_system) {
        parts << kMgStart << "system" << kMgMessage
              << "You are a helpful AI assistant.\n"
                 "Knowledge cutoff: 2026-01-04.\n\n"
                 "Reasoning strength: " << reasoning_strength << ".\n\n"
                 "# Valid recipients: \"self\", \"user\"." << kMgEot;
    }
    for (const auto& m : messages) {
        std::string role = m.role;
        for (auto& c : role) c = (char)tolower((unsigned char)c);
        if (role == "system") {
            parts << kMgStart << "system" << kMgMessage << m.content
                  << "\n\nReasoning strength: " << reasoning_strength << ".\n\n"
                     "# Valid recipients: \"self\", \"user\"." << kMgEot;
        } else if (role == "user") {
            parts << kMgStart << "user" << kMgMessage << m.content << kMgEot;
        } else if (role == "tool") {
            parts << kMgStart << "tool" << kMgMessage << m.content << kMgEot;
        } else {  // assistant history: no recipient/tool_calls fields to reconstruct, so this
                  // always matches the template's plain-content branch with its default
                  // recipient='user' (which the template always renders explicitly).
            parts << kMgStart << "assistant to=user" << kMgMessage << m.content << kMgEot;
        }
    }
    parts << kMgStart << "assistant";
    return parts.str();
}

// raw is everything generated after the prompt's trailing "<|start|>assistant" (the prompt
// itself stops there; the model supplies " to=...<|message|>...<|eom|>" / "...<|eot|>" itself,
// and re-emits its own "<|start|>assistant to=..." header for each subsequent segment of the
// same turn -- so re-prepending that first header lets one scan handle every segment uniformly.
ParsedAssistantOutput parse_museglimmer_output(const std::string& raw, bool enable_thinking) {
    ParsedAssistantOutput out;
    const std::string text = std::string(kMgStart) + "assistant" + raw;
    size_t pos = 0;
    while (pos < text.size()) {
        const size_t start = text.find(kMgStart, pos);
        if (start == std::string::npos) break;
        const size_t msg = text.find(kMgMessage, start);
        if (msg == std::string::npos) break;  // incomplete trailing header -- drop it
        const size_t header_start = start + strlen(kMgStart);
        const bool is_self = text.compare(header_start, msg - header_start, "assistant to=self") == 0;
        const size_t body_start = msg + strlen(kMgMessage);
        const size_t eom = text.find(kMgEom, body_start);
        const size_t eot = text.find(kMgEot, body_start);
        size_t end = std::string::npos;
        bool hit_eot = false;
        if (eom != std::string::npos && (eot == std::string::npos || eom < eot)) {
            end = eom;
        } else if (eot != std::string::npos) {
            end = eot;
            hit_eot = true;
        }
        const std::string body = (end == std::string::npos) ? text.substr(body_start)
                                                              : text.substr(body_start, end - body_start);
        if (!is_self) out.content += body;
        else if (enable_thinking) out.reasoning_content += body;  // else: drop reasoning silently
        if (end == std::string::npos) break;  // truncated mid-segment (hit max_tokens)
        if (hit_eot) break;                   // end of assistant turn
        pos = end + strlen(kMgEom);
    }
    return out;
}

ParsedAssistantOutput parse_assistant_output(const std::string& raw, bool enable_thinking, bool museglimmer,
                                             const ChatRequest* request) {
    if (museglimmer) return parse_museglimmer_output(raw, enable_thinking);

    if (request && !request->tools.empty()) {
        const ParsedToolOutput parsed = parse_qwen36_tool_output(raw, enable_thinking, *request);
        ParsedAssistantOutput out;
        out.reasoning_content = parsed.reasoning_content;
        out.content = parsed.content;
        out.tool_calls = parsed.tool_calls;
        out.error = parsed.error;
        return out;
    }

    ParsedAssistantOutput out;
    if (!enable_thinking) {
        out.content = raw;
        strip_trailing_im_end(out.content);
        return out;
    }

    // The official Qwen3.6 generation prompt already ends in "<think>\n" when thinking is
    // enabled, so generated text normally starts inside that block and contains only the
    // closing marker. Accept a repeated opening marker defensively, but do not require one.
    const size_t open = raw.find(kThinkOpen);
    const size_t body_start = open == std::string::npos ? 0 : open + strlen(kThinkOpen);
    const size_t close = raw.find(kThinkClose, body_start);
    if (close != std::string::npos) {
        out.reasoning_content = raw.substr(body_start, close - body_start);
        out.content = raw.substr(close + strlen(kThinkClose));
    } else {
        out.reasoning_content = raw.substr(body_start);
    }
    trim_leading_ws(out.reasoning_content);
    trim_trailing_ws(out.reasoning_content);
    trim_leading_ws(out.content);
    out.content = strip_think_markers(std::move(out.content));
    strip_trailing_im_end(out.content);
    return out;
}

ThinkingStreamSplitter::ThinkingStreamSplitter(bool enable_thinking, bool museglimmer)
    : enable_thinking_(enable_thinking), museglimmer_(museglimmer) {
    if (!enable_thinking_) phase_ = Phase::kInAnswer;
    else if (!museglimmer_) {
        // Qwen3.6's prompt pre-fills "<think>\n". Streaming therefore begins in the body,
        // not before an opening marker; wait directly for </think> and never leak reasoning
        // into delta.content.
        phase_ = Phase::kInThink;
    }
}

ThinkingStreamSplitter::Delta ThinkingStreamSplitter::feed(const std::string& piece) {
    Delta out;

    if (museglimmer_) {
        if (mg_done_) return out;
        std::string data = carry_ + piece;
        carry_.clear();
        while (!data.empty()) {
            if (mg_phase_ == MgPhase::kAwaitStart) {
                const size_t pos = data.find(kMgStart);
                if (pos == std::string::npos) {
                    const size_t keep = marker_prefix_len(data, kMgStart);
                    if (keep > 0) carry_ = data.substr(data.size() - keep);
                    break;
                }
                data.erase(0, pos + strlen(kMgStart));
                mg_header_.clear();
                mg_phase_ = MgPhase::kHeader;
                continue;
            }
            if (mg_phase_ == MgPhase::kHeader) {
                const size_t pos = data.find(kMgMessage);
                if (pos == std::string::npos) {
                    const size_t keep = marker_prefix_len(data, kMgMessage);
                    mg_header_ += data.substr(0, data.size() - keep);
                    if (keep > 0) carry_ = data.substr(data.size() - keep);
                    break;
                }
                mg_header_ += data.substr(0, pos);
                mg_is_self_ = mg_header_ == "assistant to=self";
                data.erase(0, pos + strlen(kMgMessage));
                mg_phase_ = MgPhase::kBody;
                continue;
            }
            // mg_phase_ == kBody
            const size_t eom = data.find(kMgEom);
            const size_t eot = data.find(kMgEot);
            size_t end = std::string::npos;
            bool hit_eot = false;
            if (eom != std::string::npos && (eot == std::string::npos || eom < eot)) {
                end = eom;
            } else if (eot != std::string::npos) {
                end = eot;
                hit_eot = true;
            }
            std::string& sink = mg_is_self_ ? out.reasoning_content : out.content;
            const bool emit = !mg_is_self_ || enable_thinking_;
            if (end == std::string::npos) {
                const size_t keep = std::max(marker_prefix_len(data, kMgEom), marker_prefix_len(data, kMgEot));
                const size_t emit_len = data.size() - keep;
                if (emit_len > 0 && emit) sink += data.substr(0, emit_len);
                if (keep > 0) carry_ = data.substr(emit_len);
                break;
            }
            if (end > 0 && emit) sink += data.substr(0, end);
            data.erase(0, end + (hit_eot ? strlen(kMgEot) : strlen(kMgEom)));
            if (hit_eot) { mg_done_ = true; break; }
            mg_phase_ = MgPhase::kAwaitStart;
        }
        return out;
    }

    std::string data = carry_ + piece;
    carry_.clear();

    if (!enable_thinking_) {
        // marker_prefix_len only ever returns up to strlen(kImEnd)-1 (see its own comment): it
        // detects an INCOMPLETE marker straddling this chunk and the next, to hold back and
        // wait -- it structurally cannot recognize "the complete marker just arrived", since
        // that's a same-chunk match, not a partial-suffix one. <|im_end|> is usually decoded
        // whole in a single piece (it's the terminal token), so that "complete, same-chunk"
        // case is the COMMON one, not an edge case -- every time it hit, the whole 10-byte
        // marker sailed straight into out.content since nothing here ever searched for a full
        // match, only a partial one. strip_trailing_im_end() exists specifically for this
        // string but only ran in finish(), long after a streaming client had already rendered
        // it. Search for the complete marker first and drop it (and never emit anything after
        // it -- it's the terminal marker) before falling back to the partial-suffix holdback.
        const size_t im_end_pos = data.find(kImEnd);
        if (im_end_pos != std::string::npos) {
            if (im_end_pos > 0) out.content += data.substr(0, im_end_pos);
            return out;
        }
        const size_t keep = marker_prefix_len(data, kImEnd);
        const size_t emit_len = data.size() - keep;
        if (emit_len > 0) out.content += data.substr(0, emit_len);
        if (keep > 0) carry_ = data.substr(emit_len);
        return out;
    }

    while (!data.empty()) {
        if (phase_ == Phase::kBeforeThink) {
            const size_t pos = data.find(kThinkOpen);
            if (pos == std::string::npos) {
                const size_t keep = marker_prefix_len(data, kThinkOpen);
                const size_t prefix_len = data.size() - keep;
                if (prefix_len > 0) prefix_buffer_ += data.substr(0, prefix_len);
                if (keep > 0) carry_ = data.substr(prefix_len);
                break;
            }
            prefix_buffer_.clear();
            data.erase(0, pos + strlen(kThinkOpen));
            phase_ = Phase::kInThink;
            continue;
        }

        if (phase_ == Phase::kInThink) {
            if (!think_open_checked_ && !data.empty()) {
                if (data.compare(0, strlen(kThinkOpen), kThinkOpen) == 0) {
                    data.erase(0, strlen(kThinkOpen));
                    think_open_checked_ = true;
                } else {
                    const size_t keep = marker_prefix_len(data, kThinkOpen);
                    if (keep > 0 && keep == data.size()) {
                        // Whole chunk so far could still be a partial opening marker --
                        // hold it and re-decide once more data arrives.
                        carry_ = data;
                        break;
                    }
                    think_open_checked_ = true;
                }
            }
            const size_t pos = data.find(kThinkClose);
            if (pos == std::string::npos) {
                const size_t keep = marker_prefix_len(data, kThinkClose);
                const size_t emit_len = data.size() - keep;
                if (emit_len > 0) out.reasoning_content += data.substr(0, emit_len);
                if (keep > 0) carry_ = data.substr(emit_len);
                break;
            }
            if (pos > 0) out.reasoning_content += data.substr(0, pos);
            data.erase(0, pos + strlen(kThinkClose));
            phase_ = Phase::kInAnswer;
            trim_leading_ws(data);
            continue;
        }

        if (phase_ == Phase::kInAnswer) {
            std::string chunk = filter_answer_chunk(carry_, data);
            // Same fix as the !enable_thinking_ path above: a complete <|im_end|> arriving
            // whole in `chunk` (the common case) was never matched by marker_prefix_len's
            // partial-suffix-only check, so it sailed straight into out.content.
            const size_t im_end_pos = chunk.find(kImEnd);
            if (im_end_pos != std::string::npos) {
                if (im_end_pos > 0) out.content += chunk.substr(0, im_end_pos);
                break;
            }
            const size_t keep = marker_prefix_len(chunk, kImEnd);
            if (keep > 0 && keep == chunk.size()) {
                carry_ = chunk;
                break;
            }
            if (keep > 0) {
                out.content += chunk.substr(0, chunk.size() - keep);
                carry_ = chunk.substr(chunk.size() - keep);
            } else {
                out.content += chunk;
            }
            break;
        }
    }
    return out;
}

void ThinkingStreamSplitter::finish(Delta& tail) {
    tail = {};
    if (museglimmer_) {
        if (mg_phase_ == MgPhase::kBody && !carry_.empty()) {
            if (!mg_is_self_) tail.content = carry_;
            else if (enable_thinking_) tail.reasoning_content = carry_;
        }
        carry_.clear();
        return;
    }
    if (enable_thinking_ && phase_ == Phase::kBeforeThink) {
        tail.content = strip_think_markers(prefix_buffer_ + carry_);
        prefix_buffer_.clear();
        carry_.clear();
    } else if (!carry_.empty()) {
        if (enable_thinking_ && phase_ == Phase::kInThink)
            tail.reasoning_content = carry_;
        else if (phase_ == Phase::kInAnswer)
            tail.content = filter_answer_chunk(carry_, "");
        else
            tail.content = carry_;
        carry_.clear();
    }
    tail.content = strip_think_markers(std::move(tail.content));
    strip_trailing_im_end(tail.content);
    trim_leading_ws(tail.content);
}

StopSequenceFilter::StopSequenceFilter(std::vector<std::string> stops) : stops_(std::move(stops)) {}

std::string StopSequenceFilter::feed(const std::string& piece) {
    std::string data = carry_ + piece;
    carry_.clear();
    if (stops_.empty()) return data;

    // Full-match scan: earliest position across all stops wins (ties are inconsequential -- the
    // trim point is identical either way, since a tie only happens when one stop is a prefix of
    // another and both start at the same position).
    size_t match_pos = std::string::npos;
    for (const auto& stop : stops_) {
        const size_t pos = data.find(stop);
        if (pos != std::string::npos && pos < match_pos) match_pos = pos;
    }
    if (match_pos != std::string::npos) {
        matched_ = true;
        return data.substr(0, match_pos);  // text after this point is permanently dropped
    }

    // No full match yet: hold back the longest suffix that could still be an in-progress
    // partial prefix of any stop string.
    size_t keep = 0;
    for (const auto& stop : stops_) keep = std::max(keep, marker_prefix_len(data, stop));
    const size_t emit_len = data.size() - keep;
    if (keep > 0) carry_ = data.substr(emit_len);
    return data.substr(0, emit_len);
}

bool ChatTokenizer::encode_chat_request(const std::string& request_json, std::vector<int>& ids,
                                         bool enable_thinking, std::string& err,
                                         ChatRequest* parsed_request) const {
    std::lock_guard<std::recursive_mutex> tok_lock(tok_mu_);
    ids.clear();
    if (!impl_->tok) {
        err = "tokenizer not loaded";
        return false;
    }
    ChatRequest request;
    if (!parse_chat_request_json(request_json, request, err)) return false;
    if (!validate_chat_request_model_support(request, impl_->museglimmer, err)) return false;

    ids = encode_augmented(request, enable_thinking);
    if (ids.empty()) {
        err = "tokenize returned no ids";
        return false;
    }
    if (parsed_request) *parsed_request = std::move(request);
    return true;
}

std::vector<int> ChatTokenizer::encode_augmented(const ChatRequest& request, bool enable_thinking) const {
    std::lock_guard<std::recursive_mutex> tok_lock(tok_mu_);
    if (!impl_->tok) return {};
    const std::string prompt = impl_->museglimmer
        ? apply_museglimmer_chat_template(request.messages, enable_thinking ? "high" : "low")
        : apply_qwen36_tools_template(request, enable_thinking, impl_->qwen38);
    const std::vector<int32_t> enc = impl_->tok->Encode(prompt);
    return std::vector<int>(enc.begin(), enc.end());
}

std::vector<int> ChatTokenizer::encode_raw(const std::string& text) const {
    std::lock_guard<std::recursive_mutex> tok_lock(tok_mu_);
    if (!impl_->tok) return {};
    const std::vector<int32_t> enc = impl_->tok->Encode(text);
    return std::vector<int>(enc.begin(), enc.end());
}

std::string ChatTokenizer::decode(const std::vector<int>& ids) const {
    std::lock_guard<std::recursive_mutex> tok_lock(tok_mu_);
    if (!impl_->tok || ids.empty()) return {};
    std::vector<int32_t> v(ids.begin(), ids.end());

    if (impl_->museglimmer) {
        // Muse Glimmer's harmony-format structural tokens (<|start|>, <|message|>, <|eom|>,
        // <|eot|>) are registered as special tokens in its tokenizer.json, so the underlying
        // tokenizers-cpp Decode() silently drops them from the output text (standard
        // skip-special-tokens decode behavior, confirmed empirically: decoding any of these
        // four ids alone returns ""). But parse_museglimmer_output/ThinkingStreamSplitter
        // both scan the DECODED TEXT for those literal marker strings to find segment
        // boundaries -- with them stripped, the scan never finds a match and both content
        // and reasoning_content come back empty regardless of what the model generated.
        //
        // Fix: decode body/header text in non-marker runs (so BPE merging/spacing across
        // ordinary tokens stays correct -- ordinary Decode() semantics, unchanged) and
        // splice the literal marker string back in at each split point using the same
        // kMgStart/kMgMessage/kMgEom/kMgEot constants the parser already searches for,
        // rather than relying on the library's special-token handling. Non-museglimmer
        // models (Qwen3.6 et al, whose <think> tags are ordinary text tokens, not special
        // ones) are unaffected -- this whole branch is gated on impl_->museglimmer. Marker
        // ids are resolved once in set_museglimmer(), not lazily here -- see Impl::mg_start_id.
        auto marker_str = [&](int32_t id) -> const char* {
            if (id == impl_->mg_start_id) return kMgStart;
            if (id == impl_->mg_message_id) return kMgMessage;
            if (id == impl_->mg_eom_id) return kMgEom;
            if (id == impl_->mg_eot_id) return kMgEot;
            return nullptr;
        };
        std::string out;
        std::vector<int32_t> run;
        auto flush_run = [&]() {
            if (!run.empty()) {
                out += impl_->tok->Decode(run);
                run.clear();
            }
        };
        for (int32_t id : v) {
            if (const char* m = marker_str(id)) {
                flush_run();
                out += m;
            } else {
                run.push_back(id);
            }
        }
        flush_run();
        return out;
    }

    return impl_->tok->Decode(v);
}

std::string ChatTokenizer::decode_delta(std::vector<int>& acc, int new_id) const {
    std::lock_guard<std::recursive_mutex> tok_lock(tok_mu_);
    acc.push_back(new_id);
    const std::string full = decode(acc);
    if (acc.size() == 1) return full;
    const std::string prev = decode(std::vector<int>(acc.begin(), acc.end() - 1));
    // HF-style Decode is not always prefix-stable: incomplete UTF-8 / BPE merges can
    // rewrite the tail (often via U+FFFD). Returning `full` on mismatch re-sends the
    // entire so-far string into the SSE stream. Emit only the bytes after the longest
    // common prefix so clients append a true delta.
    size_t i = 0;
    const size_t n = std::min(full.size(), prev.size());
    while (i < n && full[i] == prev[i]) ++i;
    while (i > 0 && i < full.size() &&
           (static_cast<unsigned char>(full[i]) & 0xC0) == 0x80)
        --i;
    return full.substr(i);
}

RawTokenPiece ChatTokenizer::id_to_raw_piece(int id) const {
    std::lock_guard<std::recursive_mutex> tok_lock(tok_mu_);
    if (!impl_->tok) return {};
    const std::string raw = impl_->tok->IdToToken(static_cast<int32_t>(id));
    return gpt2_bytelevel_decode(raw);
}

}  // namespace sparkinfer_server
