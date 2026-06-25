// Host-only prompt-lookup speculative decoding helpers (no second model, no
// CUDA). See sparkinfer/spec_decode.h.

#include "sparkinfer/spec_decode.h"

namespace sparkinfer {

std::vector<int> propose_draft(const std::vector<int>& history, int max_draft,
                               int max_ngram, int min_ngram) {
    const int n = (int)history.size();
    if (max_draft <= 0) return {};
    const int hi = max_ngram < n - 1 ? max_ngram : n - 1;
    for (int g = hi; g >= min_ngram; --g) {
        if (g <= 0) continue;
        const int suf_start = n - g;
        // Nearest earlier occurrence wins: scan backward from just before the
        // trailing window itself.
        for (int p = suf_start - 1; p >= 0; --p) {
            bool match = true;
            for (int j = 0; j < g; j++) {
                if (history[p + j] != history[suf_start + j]) { match = false; break; }
            }
            if (!match) continue;
            std::vector<int> draft;
            for (int k = 0; k < max_draft && p + g + k < n; k++) draft.push_back(history[p + g + k]);
            if (!draft.empty()) return draft;
        }
    }
    return {};
}

AcceptResult accept_draft(const std::vector<int>& draft, const std::vector<int>& verified) {
    AcceptResult r;
    int accepted = 0;
    while (accepted < (int)draft.size() && draft[accepted] == verified[accepted]) accepted++;
    r.accepted = accepted;
    r.bonus_token = verified[accepted];
    return r;
}

} // namespace sparkinfer
