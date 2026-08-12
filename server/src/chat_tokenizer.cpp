#include "chat_tokenizer.hpp"

#include <tokenizers_cpp.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace sparkinfer_server {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Unescape a minimal JSON string value (handles \\ \n \t \").
std::string json_unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] != '\\' || i + 1 >= s.size()) {
            out.push_back(s[i]);
            continue;
        }
        char c = s[++i];
        switch (c) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            default: out.push_back(c); break;
        }
    }
    return out;
}

bool extract_json_string(const std::string& body, size_t start, size_t& end, std::string& out) {
    size_t q = body.find('"', start);
    if (q == std::string::npos) return false;
    std::string raw;
    for (size_t i = q + 1; i < body.size(); i++) {
        if (body[i] == '\\' && i + 1 < body.size()) {
            raw.push_back(body[i++]);
            raw.push_back(body[i]);
            continue;
        }
        if (body[i] == '"') {
            end = i + 1;
            out = json_unescape(raw);
            return true;
        }
        raw.push_back(body[i]);
    }
    return false;
}

// Find closing brace of a JSON object, respecting quoted strings.
size_t find_json_object_end(const std::string& s, size_t start) {
    if (start >= s.size() || s[start] != '{') return std::string::npos;
    int depth = 0;
    bool in_string = false;
    for (size_t i = start; i < s.size(); i++) {
        const char c = s[i];
        if (in_string) {
            if (c == '\\' && i + 1 < s.size()) {
                i++;
                continue;
            }
            if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{')
            depth++;
        else if (c == '}') {
            depth--;
            if (depth == 0) return i;
        }
    }
    return std::string::npos;
}

constexpr const char* kImEnd = "<|" "im_end|>";
constexpr const char* kThinkOpen = "<" "think>";
constexpr const char* kThinkClose = "</" "think>";

// Muse Glimmer harmony-style segment markers (server/scripts equivalent of Qwen3.6's
// im_start/im_end/think tags). See tokenizer.chat_template in the model's GGUF for the
// canonical jinja this mirrors (system/user/assistant/tool rendering, tool calls and
// per-tool-namespace recipients are not implemented -- request parsing has no `tools` field
// yet, so those branches of the upstream template are never exercised here).
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

void trim_leading_ws(std::string& s) {
    while (!s.empty() && (s[0] == '\n' || s[0] == '\r' || s[0] == ' ' || s[0] == '\t')) s.erase(0, 1);
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

struct ChatTokenizer::Impl {
    std::unique_ptr<tokenizers::Tokenizer> tok;
    bool museglimmer = false;
    // Muse Glimmer harmony-format marker token ids, resolved lazily on first decode() once
    // `tok` is loaded and `museglimmer` is set -- see decode()'s comment for why these need
    // special handling.
    int32_t mg_start_id = -1, mg_message_id = -1, mg_eom_id = -1, mg_eot_id = -1;
};

ChatTokenizer::ChatTokenizer() : impl_(std::make_unique<Impl>()) {}
ChatTokenizer::~ChatTokenizer() = default;

bool ChatTokenizer::load(const std::string& tokenizer_json_path, std::string& err) {
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

void ChatTokenizer::set_museglimmer(bool on) { impl_->museglimmer = on; }

bool parse_chat_messages(const std::string& request_json, std::vector<ChatMessage>& messages, std::string& err) {
    messages.clear();
    const std::string key = "\"messages\"";
    size_t p = request_json.find(key);
    if (p == std::string::npos) {
        err = "missing messages in request";
        return false;
    }
    p = request_json.find('[', p);
    if (p == std::string::npos) {
        err = "malformed messages array";
        return false;
    }

    size_t i = p + 1;
    while (i < request_json.size()) {
        while (i < request_json.size() && (request_json[i] == ' ' || request_json[i] == ',')) i++;
        if (i >= request_json.size() || request_json[i] == ']') break;
        if (request_json[i] != '{') {
            err = "malformed message object";
            return false;
        }

        ChatMessage msg;
        const size_t obj_end = find_json_object_end(request_json, i);
        if (obj_end == std::string::npos) {
            err = "unterminated message object";
            return false;
        }
        const std::string obj = request_json.substr(i, obj_end - i + 1);

        size_t role_pos = obj.find("\"role\"");
        if (role_pos != std::string::npos) {
            size_t dummy = 0;
            extract_json_string(obj, role_pos + 6, dummy, msg.role);
        }
        size_t content_pos = obj.find("\"content\"");
        if (content_pos != std::string::npos) {
            size_t c = obj.find(':', content_pos);
            if (c != std::string::npos) {
                while (c < obj.size() && (obj[c] == ':' || obj[c] == ' ')) c++;
                if (c < obj.size() && obj[c] == '"') {
                    size_t dummy = 0;
                    extract_json_string(obj, c, dummy, msg.content);
                } else if (c < obj.size() && obj[c] == '[') {
                    // Multimodal content parts: only concatenate "text" KEY values.
                    // A naive find("\"text\"") also matches the type VALUE {"type":"text"},
                    // and when extract fails `j` never advances → infinite loop / OOM.
                    size_t j = c;
                    const size_t lim = obj.size();
                    while (j < lim) {
                        const size_t text_key = obj.find("\"text\"", j);
                        if (text_key == std::string::npos) break;
                        j = text_key + 6;  // always past this match — scan cannot rewind
                        size_t v = j;
                        while (v < lim && (obj[v] == ':' || obj[v] == ' ')) ++v;
                        if (v == j || v >= lim || obj[v] != '"') continue;  // type value, not a key
                        size_t part_end = 0;
                        std::string piece;
                        if (!extract_json_string(obj, v, part_end, piece)) break;
                        if (!msg.content.empty()) msg.content.push_back(' ');
                        msg.content += piece;
                        j = part_end;
                    }
                }
            }
        }

        if (msg.role.empty()) msg.role = "user";
        messages.push_back(std::move(msg));
        i = obj_end + 1;
    }

    if (messages.empty()) {
        err = "no messages in request";
        return false;
    }
    return true;
}

bool parse_enable_thinking(const std::string& request_json, bool default_value) {
    const std::string needle = "\"enable_thinking\"";
    auto parse_at = [&](size_t pos) -> bool {
        pos = request_json.find(':', pos);
        if (pos == std::string::npos) return default_value;
        const size_t t = request_json.find("true", pos);
        const size_t f = request_json.find("false", pos);
        const size_t comma = request_json.find(',', pos);
        const size_t end = comma == std::string::npos ? request_json.size() : comma;
        if (t != std::string::npos && t < end) return true;
        if (f != std::string::npos && f < end) return false;
        return default_value;
    };

    const size_t messages = request_json.find("\"messages\"");
    const size_t first = request_json.find(needle);
    if (first != std::string::npos && (messages == std::string::npos || first < messages))
        return parse_at(first);

    const size_t kwargs = request_json.find("chat_template_kwargs");
    if (kwargs != std::string::npos) {
        const size_t in_kwargs = request_json.find(needle, kwargs);
        if (in_kwargs != std::string::npos) return parse_at(in_kwargs);
    }

    if (first != std::string::npos) return parse_at(first);
    return default_value;
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

ParsedAssistantOutput parse_assistant_output(const std::string& raw, bool enable_thinking, bool museglimmer) {
    if (museglimmer) return parse_museglimmer_output(raw, enable_thinking);

    ParsedAssistantOutput out;
    if (!enable_thinking) {
        out.content = raw;
        strip_trailing_im_end(out.content);
        return out;
    }

    const size_t open = raw.find(kThinkOpen);
    if (open == std::string::npos) {
        out.content = strip_think_markers(raw);
        strip_trailing_im_end(out.content);
        return out;
    }

    const size_t body_start = open + strlen(kThinkOpen);
    const size_t close = raw.find(kThinkClose, body_start);
    if (close != std::string::npos) {
        out.reasoning_content = raw.substr(body_start, close - body_start);
        out.content = raw.substr(close + strlen(kThinkClose));
    } else {
        out.reasoning_content = raw.substr(body_start);
    }
    trim_leading_ws(out.reasoning_content);
    trim_leading_ws(out.content);
    out.content = strip_think_markers(std::move(out.content));
    strip_trailing_im_end(out.content);
    return out;
}

ThinkingStreamSplitter::ThinkingStreamSplitter(bool enable_thinking, bool museglimmer)
    : enable_thinking_(enable_thinking), museglimmer_(museglimmer) {
    if (!enable_thinking_) phase_ = Phase::kInAnswer;
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

bool ChatTokenizer::encode_chat_request(const std::string& request_json, std::vector<int>& ids, bool enable_thinking,
                                         std::string& err) const {
    ids.clear();
    if (!impl_->tok) {
        err = "tokenizer not loaded";
        return false;
    }
    std::vector<ChatMessage> messages;
    if (!parse_chat_messages(request_json, messages, err)) return false;

    const std::string prompt = impl_->museglimmer
        ? apply_museglimmer_chat_template(messages, enable_thinking ? "high" : "low")
        : apply_qwen36_chat_template(messages, enable_thinking);
    const std::vector<int32_t> enc = impl_->tok->Encode(prompt);
    ids.assign(enc.begin(), enc.end());
    if (ids.empty()) {
        err = "tokenize returned no ids";
        return false;
    }
    return true;
}

std::string ChatTokenizer::decode(const std::vector<int>& ids) const {
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
        // ones) are unaffected -- this whole branch is gated on impl_->museglimmer.
        if (impl_->mg_start_id < 0) {
            impl_->mg_start_id = impl_->tok->TokenToId(kMgStart);
            impl_->mg_message_id = impl_->tok->TokenToId(kMgMessage);
            impl_->mg_eom_id = impl_->tok->TokenToId(kMgEom);
            impl_->mg_eot_id = impl_->tok->TokenToId(kMgEot);
        }
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

}  // namespace sparkinfer_server
