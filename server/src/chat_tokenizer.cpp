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
                    // Multimodal: concatenate text parts only (images skipped for now).
                    size_t j = c;
                    while (j < obj.size()) {
                        size_t text_key = obj.find("\"text\"", j);
                        if (text_key == std::string::npos || text_key >= obj.size()) break;
                        size_t dummy = 0;
                        std::string piece;
                        if (extract_json_string(obj, text_key + 6, dummy, piece)) {
                            if (!msg.content.empty()) msg.content.push_back(' ');
                            msg.content += piece;
                        }
                        j = dummy;
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

ParsedAssistantOutput parse_assistant_output(const std::string& raw, bool enable_thinking) {
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

ThinkingStreamSplitter::ThinkingStreamSplitter(bool enable_thinking) : enable_thinking_(enable_thinking) {
    if (!enable_thinking_) phase_ = Phase::kInAnswer;
}

ThinkingStreamSplitter::Delta ThinkingStreamSplitter::feed(const std::string& piece) {
    Delta out;
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

    const std::string prompt = apply_qwen36_chat_template(messages, enable_thinking);
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
    return impl_->tok->Decode(v);
}

std::string ChatTokenizer::decode_delta(std::vector<int>& acc, int new_id) const {
    acc.push_back(new_id);
    const std::string full = decode(acc);
    if (acc.size() == 1) return full;
    const std::string prev = decode(std::vector<int>(acc.begin(), acc.end() - 1));
    if (full.size() >= prev.size() && full.compare(0, prev.size(), prev) == 0)
        return full.substr(prev.size());
    return full;
}

}  // namespace sparkinfer_server
