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

// The EBNF xgrammar generates for a JSON schema, with every JSON string narrowed: no raw '<' (so a JSON
// value can never contain protocol markup; '<' is still expressible as <) and no raw control
// character (which strict JSON forbids but xgrammar's length-limited strings allow). `exact` is
// cleared when a string character class the narrowing does not recognise remains. False on a schema
// xgrammar cannot convert.
bool json_value_ebnf(const std::string& schema_json, std::string& ebnf, bool& exact, std::string& err);

}  // namespace sparkinfer_server
