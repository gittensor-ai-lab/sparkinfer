#!/usr/bin/env python3
"""H1 (held-out / fuzzed eval prompt): emit a teacher-forcing token stream that a submission
CANNOT overfit, derived deterministically from a seed.

The accuracy gate used a single fixed in-repo prompt (`eval_text.txt`) — a contributor could read
it and special-case those exact tokens. Instead the eval bot passes a fresh, unpredictable seed each
run; this picks a random window of a diverse multi-domain corpus and a fuzzed token length from it.
The PR author can't know which slice will be scored, so correctness must generalize. The seed is
recorded in the eval log, so any third party reproduces the exact token stream and verdict.

Usage:  gen_eval_prompt.py <seed> <tokenizer.json> <corpus.txt> [fixed_text.txt]
Prints a space-separated list of token ids (the same format accuracy.sh feeds qwen3_gguf_score).
If <fixed_text.txt> is given AND seed=="fixed", reproduces the legacy fixed prompt (for bench.sh).
"""
import sys, random
from tokenizers import Tokenizer


def main():
    seed, tok_path, corpus_path = sys.argv[1], sys.argv[2], sys.argv[3]
    fixed = sys.argv[4] if len(sys.argv) > 4 else ""
    tok = Tokenizer.from_file(tok_path)

    if seed == "fixed" and fixed:
        ids = tok.encode(open(fixed).read().strip()).ids
        print(" ".join(map(str, ids)))
        return

    rng = random.Random(seed)            # seed is a string; Random hashes it deterministically
    paras = [p.strip() for p in open(corpus_path).read().split("\n\n") if p.strip()]
    # pick a random run of 2..5 adjacent paragraphs (held-out: which slice is unknown to the PR)
    k = rng.randint(2, min(5, len(paras)))
    start = rng.randint(0, max(0, len(paras) - k))
    ids = tok.encode(" ".join(paras[start:start + k])).ids
    # fuzz the SHAPE: score a seed-chosen window of 200..360 tokens (no fixed length to assume)
    n = rng.randint(200, 360)
    if len(ids) > n:
        s = rng.randint(0, len(ids) - n)
        ids = ids[s:s + n]
    if len(ids) < 32:                    # degenerate slice -> fall back to the whole corpus head
        ids = tok.encode(" ".join(paras)).ids[:n]
    print(" ".join(map(str, ids)))


if __name__ == "__main__":
    main()
