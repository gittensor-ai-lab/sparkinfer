#!/usr/bin/env python3
"""Long-context greedy decode must be CORRECT and REPRODUCIBLE (issue #976).

The accuracy smoke this repo already runs works at short context -- which is precisely the regime
that still worked while #976 was live. The fused GQA prefill attention wrote its P' plane with row
stride GN while allocating and reading it with pld; below 2048 tokens pad==0 so the two agreed, and
at or above 2048 every row was read shifted by 16r with the last row reading past all initialised
shared memory. Output was simultaneously wrong and nondeterministic, and nothing in CI could see it.

So this asks the narrowest question that would have caught it: one arithmetic question with an
unambiguous answer, a growing natural-prose prefix, N greedy runs at each depth. Two assertions,
and the second is the one that matters -- greedy decode must be a function of its input:

  1. every run at every depth answers correctly
  2. the runs at a given depth are IDENTICAL to each other

A correctness-only check would pass on a build that is stably wrong; a determinism-only check
would pass on a build that is consistently garbage. Both are needed.

Usage:
  python3 bench/scripts/long_context_determinism_check.py --base-url http://127.0.0.1:8080
"""
import argparse, json, sys, urllib.request

# Deliberately trivial arithmetic: the answer is not in dispute, so any deviation is the engine's.
QUESTION = "What is 17 multiplied by 23? Reply with only the number."
ANSWER = "391"


def ask(base_url, prose, max_tokens, timeout):
    body = json.dumps({
        "model": "q", "max_tokens": max_tokens, "temperature": 0.0,
        "enable_thinking": False,
        "messages": [{"role": "user", "content": (prose + "\n\n" + QUESTION) if prose else QUESTION}],
    }).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    if "error" in d:
        raise RuntimeError(json.dumps(d["error"])[:200])
    msg = d["choices"][0]["message"]
    return (msg.get("content") or "").strip(), d.get("usage", {}).get("prompt_tokens", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--corpus", default="bench/scripts/bench_prompt_32k.txt")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=900.0)
    # Prefix lengths in CHARACTERS. The defaults straddle the 2048-token cliff #976 lived at:
    # one clearly below, then several above, since a check that only probes the working regime is
    # exactly the hole this closes.
    ap.add_argument("--chars", default="0,8000,30000,90000,200000")
    args = ap.parse_args()

    try:
        corpus = open(args.corpus).read()
    except OSError as e:
        print(f"cannot read corpus: {e}")
        return 2

    failures = 0
    for spec in [int(c) for c in args.chars.split(",")]:
        prose = corpus[:spec]
        outs = []
        toks = 0
        for _ in range(args.runs):
            try:
                text, toks = ask(args.base_url, prose, args.max_tokens, args.timeout)
            except Exception as e:                      # noqa: BLE001 -- report, do not abort the sweep
                print(f"  chars={spec:<7} REQUEST FAILED: {e}")
                failures += 1
                outs = None
                break
            outs.append(text)
        if outs is None:
            continue

        stable = len(set(outs)) == 1
        # Substring, not equality: a correct engine may legitimately answer "391" or "391." etc.
        correct = all(ANSWER in o for o in outs)
        status = "OK" if (stable and correct) else ("WRONG" if stable else "FLAKY")
        print(f"  chars={spec:<7} tokens={toks:<7} {status:<6} " +
              " | ".join(repr(o[:26]) for o in outs))
        if not (stable and correct):
            failures += 1

    print()
    if failures:
        print(f"FAILED — {failures} depth(s) wrong or nondeterministic")
        print("Greedy decode must be a function of its input; see issue #976.")
        return 1
    print("PASSED — every depth is correct and reproducible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
