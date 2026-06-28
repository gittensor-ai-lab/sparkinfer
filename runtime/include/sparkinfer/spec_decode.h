#pragma once
#include <vector>

namespace sparkinfer {

// Prompt-lookup speculative draft (no second model). Scans `history` for the
// most recent earlier occurrence of its own trailing n-gram and proposes the
// tokens that followed that occurrence last time as a guess for what comes
// next. Tries the longest n-gram first (more specific -> more reliable) and
// falls back to shorter ones. Returns an empty vector if no match is found.
std::vector<int> propose_draft(const std::vector<int>& history, int max_draft,
                               int max_ngram = 4, int min_ngram = 1);

// Result of verifying `draft` against one batched forward pass whose outputs
// (greedy argmax per input position) are `verified`. `verified[i]` is the
// model's true next-token prediction after having been fed `batch_in[i]`,
// where batch_in = [anchor] + draft (so verified.size() == draft.size() + 1).
struct AcceptResult {
    int accepted = 0;       // number of leading draft tokens confirmed (0..draft.size())
    int bonus_token = -1;   // verified[accepted] -- always a true model output, fed next round
};

// Walks `draft` against `verified` and returns how many leading tokens match
// plus the bonus token at the first mismatch (or at the end, if all matched).
// `verified` must have at least draft.size() + 1 entries (see AcceptResult
// above); any entries beyond that are ignored.
AcceptResult accept_draft(const std::vector<int>& draft, const std::vector<int>& verified);

} // namespace sparkinfer
