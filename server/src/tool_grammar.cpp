#include "tool_grammar.hpp"

#include <nlohmann/json.hpp>
#include <xgrammar/xgrammar.h>

#include <list>
#include <map>
#include <set>
#include <mutex>
#include <optional>
#include <regex>
#include <unordered_map>

namespace sparkinfer_server {

namespace {

using nlohmann::json;

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    for (size_t at = 0; (at = s.find(from, at)) != std::string::npos; at += to.size()) s.replace(at, from.size(), to);
    return s;
}

// UTF-8 validity as a token-level automaton. A byte-level vocabulary has tokens for lone lead and
// continuation bytes; a grammar over bytes admits them, and the detokenizer then turns them into
// U+FFFD -- changing the text, and its length, after the grammar has judged it. Masking them keeps
// every generated prefix a prefix of valid UTF-8, so the text the parser sees is the text the grammar
// accepted.
enum : uint8_t { kClean, kCont1, kCont2, kCont3, kAfterE0, kAfterED, kAfterF0, kAfterF4, kUtf8States, kInvalid = 255 };

uint8_t utf8_step(uint8_t state, unsigned char b) {
    const bool cont = b >= 0x80 && b <= 0xBF;
    switch (state) {
        case kClean:
            if (b < 0x80) return kClean;
            if (b >= 0xC2 && b <= 0xDF) return kCont1;
            if (b == 0xE0) return kAfterE0;
            if ((b >= 0xE1 && b <= 0xEC) || b == 0xEE || b == 0xEF) return kCont2;
            if (b == 0xED) return kAfterED;
            if (b == 0xF0) return kAfterF0;
            if (b >= 0xF1 && b <= 0xF3) return kCont3;
            if (b == 0xF4) return kAfterF4;
            return kInvalid;
        case kCont1: return cont ? kClean : kInvalid;
        case kCont2: return cont ? kCont1 : kInvalid;
        case kCont3: return cont ? kCont2 : kInvalid;
        case kAfterE0: return (b >= 0xA0 && b <= 0xBF) ? kCont1 : kInvalid;
        case kAfterED: return (b >= 0x80 && b <= 0x9F) ? kCont1 : kInvalid;
        case kAfterF0: return (b >= 0x90 && b <= 0xBF) ? kCont2 : kInvalid;
        case kAfterF4: return (b >= 0x80 && b <= 0x8F) ? kCont2 : kInvalid;
        default: return kInvalid;
    }
}

struct Utf8Tables {
    int vocab_size = 0;
    int words = 0;
    std::vector<uint8_t> next;                   // [state * vocab_size + id]
    std::vector<uint32_t> allowed[kUtf8States];  // bitmask per state
    std::vector<int> stop_ids;
};

std::shared_ptr<const Utf8Tables> build_utf8_tables(const std::vector<std::string>& tokens, int vocab_size,
                                                    const std::vector<int>& stop_ids) {
    auto t = std::make_shared<Utf8Tables>();
    t->vocab_size = vocab_size;
    t->words = (vocab_size + 31) / 32;
    t->stop_ids = stop_ids;
    t->next.assign((size_t)kUtf8States * vocab_size, kInvalid);
    for (int s = 0; s < kUtf8States; ++s) t->allowed[s].assign(t->words, 0);
    for (int id = 0; id < vocab_size; ++id) {
        const std::string empty;
        const std::string& bytes = id < (int)tokens.size() ? tokens[id] : empty;
        for (int s = 0; s < kUtf8States; ++s) {
            uint8_t state = (uint8_t)s;
            for (unsigned char b : bytes) {
                state = utf8_step(state, b);
                if (state == kInvalid) break;
            }
            // A token with no text (a stop or control token) changes nothing, but a turn may only end
            // on a character boundary.
            if (bytes.empty() && s != kClean) state = kInvalid;
            t->next[(size_t)s * vocab_size + id] = state;
            if (state != kInvalid) t->allowed[s][id / 32] |= (uint32_t)1 << (id % 32);
        }
    }
    return t;
}

class GrammarConstraint final : public sparkinfer::TokenConstraint {
public:
    GrammarConstraint(const xgrammar::CompiledGrammar& grammar, std::shared_ptr<const Utf8Tables> utf8)
        : matcher_(grammar, utf8->stop_ids), utf8_(std::move(utf8)),
          scratch_(xgrammar::GetBitmaskSize(utf8_->vocab_size)) {}

    bool fill_next_mask(uint32_t* bits, int vocab_size) override {
        if (vocab_size != utf8_->vocab_size) return false;
        int64_t shape = (int64_t)scratch_.size();
        DLTensor tensor{scratch_.data(), DLDevice{kDLCPU, 0}, 1, xgrammar::GetBitmaskDLType(), &shape, nullptr, 0};
        matcher_.FillNextTokenBitmask(&tensor);
        const std::vector<uint32_t>& utf8 = utf8_->allowed[state_];
        for (int w = 0; w < utf8_->words; ++w) bits[w] = (uint32_t)scratch_[w] & utf8[w];
        return true;
    }

    bool accept(int token_id) override {
        if (token_id < 0 || token_id >= utf8_->vocab_size) return false;
        const uint8_t next = utf8_->next[(size_t)state_ * utf8_->vocab_size + token_id];
        if (next == kInvalid || !matcher_.AcceptToken(token_id)) return false;
        state_ = next;
        return true;
    }

private:
    xgrammar::GrammarMatcher matcher_;
    std::shared_ptr<const Utf8Tables> utf8_;
    std::vector<int32_t> scratch_;
    uint8_t state_ = kClean;
};

// Replace every json_schema node with the narrowed EBNF of its schema.
bool narrow_json_nodes(json& node, bool& exact, std::string& err) {
    if (node.is_object()) {
        if (node.value("type", "") == "json_schema" && node.contains("json_schema")) {
            std::string ebnf;
            const JsonStringGuard guard = !node.value("sparkinfer_json_mode", false) ? JsonStringGuard::kProtocolMarkup
                                        : node.value("sparkinfer_forbid_think", false) ? JsonStringGuard::kThinkMarkers
                                                                                       : JsonStringGuard::kNone;
            if (!json_value_ebnf(node["json_schema"].dump(), ebnf, exact, err, guard, node.value("sparkinfer_strict", true)))
                return false;
            node = json{{"type", "grammar"}, {"grammar", std::move(ebnf)}};
            return true;
        }
        for (auto& item : node.items())
            if (!narrow_json_nodes(item.value(), exact, err)) return false;
    } else if (node.is_array()) {
        for (auto& item : node)
            if (!narrow_json_nodes(item, exact, err)) return false;
    }
    return true;
}

}  // namespace

namespace {

// EBNF rules for the rest of a JSON string after a raw '<', built from a trie of the guarded marker
// suffixes: each node is the state "the string so far ends in '<' + prefix". A character that would
// complete a marker is excluded; one that extends a marker prefix moves to that node; any other
// character returns to the ordinary string body.
std::string marker_guard_rules(const std::vector<std::string>& suffixes) {
    struct Node {
        std::string prefix;
        std::map<char, int> children;
        std::set<char> terminal;
    };
    std::vector<Node> nodes(1);
    for (const std::string& s : suffixes) {
        int at = 0;
        for (size_t i = 0; i < s.size(); ++i) {
            if (i + 1 == s.size()) {
                nodes[at].terminal.insert(s[i]);
                break;
            }
            auto it = nodes[at].children.find(s[i]);
            if (it == nodes[at].children.end()) {
                nodes.push_back(Node{nodes[at].prefix + s[i], {}, {}});
                it = nodes[at].children.emplace(s[i], (int)nodes.size() - 1).first;
            }
            at = it->second;
        }
    }
    auto name = [](int i) { return i == 0 ? std::string("sparkinfer_lt") : "sparkinfer_lt_" + std::to_string(i); };
    std::string out;
    for (size_t i = 0; i < nodes.size(); ++i) {
        std::string excluded = R"(\0-\x1f\"\\\r\n<)";
        for (const auto& [c, _] : nodes[i].children) excluded.push_back(c);
        for (char c : nodes[i].terminal) excluded.push_back(c);
        out += name((int)i) + " ::= ((\"\\\"\") | (\"\\\\\" basic_escape basic_string_sub) | (\"<\" sparkinfer_lt) | ([^" +
               excluded + "] basic_string_sub)";
        for (const auto& [c, child] : nodes[i].children) out += " | (\"" + std::string(1, c) + "\" " + name(child) + ")";
        out += ")\n";
    }
    return out;
}

}  // namespace

bool json_value_ebnf(const std::string& schema_json, std::string& ebnf, bool& exact, std::string& err,
                     JsonStringGuard guard, bool strict) {
    try {
        ebnf = xgrammar::Grammar::FromJSONSchema(schema_json, /*any_whitespace=*/true, /*indent=*/std::nullopt,
                                                 /*separators=*/std::nullopt, /*strict_mode=*/strict)
                   .ToString();
    } catch (const std::exception& e) {
        err = std::string("schema has no grammar: ") + e.what();
        return false;
    }
    // Numbers the strict reader can hold: a longer integer part or exponent overflows a double.
    ebnf = replace_all(std::move(ebnf), R"(basic_number_2 ::= (("0") | ([1-9] [0-9]*)))",
                       R"(basic_number_2 ::= (("0") | ([1-9] [0-9]{0,17})))");
    ebnf = replace_all(std::move(ebnf), R"(basic_number_5 ::= ("" | ([eE] basic_number_4 basic_number_digits{1, -1})))",
                       R"(basic_number_5 ::= ("" | ([eE] basic_number_4 basic_number_digits{1,2})))");
    ebnf = replace_all(std::move(ebnf), R"(basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]*)))",
                       R"(basic_integer ::= (("0") | (basic_integer_1 [1-9] [0-9]{0,17})))");
    // xgrammar v0.2.6's two string character classes: unconstrained strings (which also take escapes)
    // and length-limited ones (which take no escapes but do take raw control characters).
    if (guard == JsonStringGuard::kNone) {
        ebnf = replace_all(std::move(ebnf), R"([^\"\\\r\n])", R"([^\0-\x1f\"\\])");
        return true;
    }
    const std::vector<std::string> suffixes = guard == JsonStringGuard::kThinkMarkers
        ? std::vector<std::string>{"think>", "/think>"}
        : std::vector<std::string>{"tool", "/tool", "function", "/function", "parameter", "/parameter", "think",
                                   "/think", "|im_"};
    const std::string body_rule =
        R"(basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n] basic_string_sub) | ("\\" basic_escape basic_string_sub)))";
    const std::string guarded_rule =
        R"(basic_string_sub ::= (("\"") | ([^\0-\x1f\"\\\r\n<] basic_string_sub) | ("\\" basic_escape basic_string_sub) | ("<" sparkinfer_lt)))";
    if (ebnf.find(body_rule) == std::string::npos) {
        err = "unexpected JSON string rule in xgrammar's grammar";
        return false;
    }
    ebnf = replace_all(std::move(ebnf), body_rule, guarded_rule);
    ebnf += marker_guard_rules(suffixes);
    ebnf = replace_all(std::move(ebnf), R"([^\"\\\r\n])", R"([^\0-\x1f\"\\<])");
    // Any other negated class could still admit '<' or a control character (a pattern's, say).
    static const std::regex negated_class(R"(\[\^[^\]]*\])");
    for (std::sregex_iterator it(ebnf.begin(), ebnf.end(), negated_class), end; it != end; ++it)
        if (it->str().find('<') == std::string::npos) exact = false;
    return true;
}

struct GrammarEngine::Impl {
    std::unique_ptr<xgrammar::TokenizerInfo> info;
    std::unique_ptr<xgrammar::GrammarCompiler> compiler;
    std::shared_ptr<const Utf8Tables> utf8;
    int vocab_size = 0;

    // Agent loops resend the same tools every turn; compiling costs ~100 ms, a cached grammar nothing.
    struct Entry {
        std::shared_ptr<xgrammar::CompiledGrammar> grammar;
        bool exact = true;
    };
    std::mutex mu;
    std::list<std::string> lru;
    std::unordered_map<std::string, std::pair<Entry, std::list<std::string>::iterator>> cache;
    static constexpr size_t kMaxEntries = 64;
};

GrammarEngine::GrammarEngine(const std::vector<std::string>& tokens, int vocab_size, const std::vector<int>& stop_ids)
    : impl_(std::make_unique<Impl>()) {
    impl_->vocab_size = vocab_size;
    std::vector<int32_t> stops(stop_ids.begin(), stop_ids.end());
    impl_->info = std::make_unique<xgrammar::TokenizerInfo>(tokens, xgrammar::VocabType::RAW, vocab_size, stops, false);
    impl_->compiler = std::make_unique<xgrammar::GrammarCompiler>(*impl_->info, 8, /*cache_enabled=*/true);
    impl_->utf8 = build_utf8_tables(tokens, vocab_size, stop_ids);
}

GrammarEngine::~GrammarEngine() = default;

int GrammarEngine::vocab_size() const { return impl_->vocab_size; }

std::shared_ptr<sparkinfer::TokenConstraint> GrammarEngine::make_constraint(const std::string& structural_tag,
                                                                            bool& exact, std::string& err) {
    std::shared_ptr<xgrammar::CompiledGrammar> grammar;
    {
        std::lock_guard<std::mutex> lock(impl_->mu);
        auto it = impl_->cache.find(structural_tag);
        if (it != impl_->cache.end()) {
            impl_->lru.splice(impl_->lru.begin(), impl_->lru, it->second.second);
            grammar = it->second.first.grammar;
            if (!it->second.first.exact) exact = false;
        } else {
            json tag;
            try {
                tag = json::parse(structural_tag);
            } catch (const std::exception& e) {
                err = std::string("invalid structural tag: ") + e.what();
                return nullptr;
            }
            bool narrowed_exact = true;
            if (!narrow_json_nodes(tag, narrowed_exact, err)) return nullptr;
            try {
                auto parsed = xgrammar::Grammar::FromStructuralTag(tag.dump(), *impl_->info);
                if (!std::holds_alternative<xgrammar::Grammar>(parsed)) {
                    err = "structural tag rejected by xgrammar";
                    return nullptr;
                }
                grammar = std::make_shared<xgrammar::CompiledGrammar>(
                    impl_->compiler->CompileGrammar(std::get<xgrammar::Grammar>(parsed)));
            } catch (const std::exception& e) {
                err = std::string("grammar does not compile: ") + e.what();
                return nullptr;
            }
            impl_->lru.push_front(structural_tag);
            impl_->cache[structural_tag] = {Impl::Entry{grammar, narrowed_exact}, impl_->lru.begin()};
            if (impl_->cache.size() > Impl::kMaxEntries) {
                impl_->cache.erase(impl_->lru.back());
                impl_->lru.pop_back();
            }
            if (!narrowed_exact) exact = false;
        }
    }
    return std::make_shared<GrammarConstraint>(*grammar, impl_->utf8);
}

}  // namespace sparkinfer_server
