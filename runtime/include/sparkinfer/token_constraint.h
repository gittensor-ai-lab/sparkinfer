#pragma once

#include <cstdint>

namespace sparkinfer {

// Constrained decoding: a per-request restriction on which token may be sampled next. The engine asks
// for a mask before every sample -- including the first token, which comes out of prefill -- and
// reports every token it emits. What the restriction means (a grammar, a schema) lives above the
// runtime; the runtime only applies the mask to the logits.
class TokenConstraint {
public:
    virtual ~TokenConstraint() = default;

    // Fill `bits` -- (vocab_size + 31) / 32 words, bit (id % 32) of word id / 32 set for every allowed
    // token id -- for the next token. False when every token is allowed and the mask can be skipped.
    virtual bool fill_next_mask(uint32_t* bits, int vocab_size) = 0;

    // The token the engine just emitted. False when the constraint cannot accept it, which can only
    // mean the mask was not applied: the engine then fails the request rather than return output
    // that breaks the constraint.
    virtual bool accept(int token_id) = 0;
};

}  // namespace sparkinfer
