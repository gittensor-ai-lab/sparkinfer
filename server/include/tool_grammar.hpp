#pragma once

#include "sparkinfer/token_constraint.h"

#include <memory>
#include <string>
#include <vector>

namespace sparkinfer_server {

// Compiles the structural tags build_tool_call_grammar produces against one model vocabulary and
// hands out per-request constraints. Thread-safe; one instance per loaded model.
class GrammarEngine {
public:
    // tokens[id]: the exact bytes token id decodes to, "" for an id that never decodes to text.
    // vocab_size: the model's logits width (at least tokens.size()). stop_ids: tokens that end a turn.
    GrammarEngine(const std::vector<std::string>& tokens, int vocab_size, const std::vector<int>& stop_ids);
    ~GrammarEngine();
    GrammarEngine(const GrammarEngine&) = delete;
    GrammarEngine& operator=(const GrammarEngine&) = delete;

    // A fresh constraint for one generation. Null, with err set, when the structural tag does not
    // compile. `exact` is cleared when compilation had to approximate part of it (see
    // json_value_ebnf); it is never set, so pass in the builder's own verdict.
    std::shared_ptr<sparkinfer::TokenConstraint> make_constraint(const std::string& structural_tag,
                                                                 bool& exact, std::string& err);
    int vocab_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// What a JSON value's strings may not spell, beyond what JSON itself forbids.
enum class JsonStringGuard {
    kProtocolMarkup,   // a tool-call argument: nothing parse_qwen36_tool_output treats as markup
    kThinkMarkers,     // response_format output with thinking on: no <think> or </think>
    kNone,             // response_format output with thinking off
};

// The EBNF xgrammar generates for a JSON schema, narrowed so every output parses the way the server's
// strict JSON reading does:
//   * strings: no raw control character (xgrammar's length-limited strings allow them), and no
//     guarded marker -- an unconstrained string may still hold '<', just never '<' starting a marker;
//     a length-limited one, whose characters are counted, holds no '<' at all when anything is guarded;
//   * numbers: at most 18 integer digits and a two-digit exponent, inside what a double holds.
// `exact` is cleared when a string character class the narrowing does not recognise remains. False on
// a schema xgrammar cannot convert. strict: whether undeclared object properties are refused.
bool json_value_ebnf(const std::string& schema_json, std::string& ebnf, bool& exact, std::string& err,
                     JsonStringGuard guard = JsonStringGuard::kProtocolMarkup, bool strict = true);

}  // namespace sparkinfer_server
