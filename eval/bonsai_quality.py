#!/usr/bin/env python3
"""Quality of the Ternary-Bonsai-2 checkpoint against the parent it was quantized from.

This is NOT the regression guard (eval/bonsai_regression.py). That one answers "did anything
break since yesterday" and compares the model against itself. This answers "what does 1.75-bit
ternary actually cost", which needs an outside reference.

The obvious outside reference -- llama.cpp on the same GGUF -- is impossible here, and not for a
boring reason: PTQ1_0 is ggml type 143, outside upstream's range entirely, so llama.cpp refuses
the file with "invalid ggml type 143. should be in [0, 43)". bench/scripts/accuracy_compare.py
therefore cannot be pointed at this model at all.

So the reference is the checkpoint this model was derived FROM, served by the same runtime:
/root/workspace/models_qwen38, the NVFP4 Qwen3.8-27B. NVFP4, not FP16 -- worth stating, because
the model card's "98.2% of FP16 intelligence" is a claim about a reference we do not have on this
box, and quoting these numbers against it would overstate what was measured.

Two metrics, both over bench/scripts/eval_corpus.txt via /v1/score:
  perplexity          exp(-mean logprob of the actual next token). Absolute quality.
  top-1 agreement     how often the model's own argmax matches the reference's, position by
                      position. This is the metric bench/scripts/accuracy_compare.py calls
                      "implementation correctness": perplexity can look reasonable while the
                      distribution is subtly wrong, and disagreement shows up here first.
"""
import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port, path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def serve(server, model, tokenizer, native, ctx):
    """Start a server and wait for health; returns (proc, port)."""
    port = free_port()
    env = dict(os.environ)
    if native is not None:
        env["SPARKINFER_BONSAI_NATIVE"] = native
    cmd = [server, "-m", model, "--ctx", str(ctx), "--port", str(port), "--tokenizer", tokenizer]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env=env, start_new_session=True)
    for _ in range(300):
        if p.poll() is not None:
            raise RuntimeError(f"server exited while loading {model}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
            return p, port
        except Exception:                                  # noqa: BLE001 - still starting
            time.sleep(4)
    raise RuntimeError(f"server never became healthy for {model}")


def stop(p):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:                                      # noqa: BLE001 - already gone
        p.kill()
    p.wait(timeout=60)
    time.sleep(5)


def score_corpus(port, lines):
    """Per-line teacher-forced scoring. Returns (total_logprob, n_tokens, [argmax ids]).

    Each line is split at its first space: the first word conditions, the rest is scored. The
    endpoint refuses an empty prompt by design -- there is nothing to condition on -- and the
    split is identical for every model, so the comparison is fair even though the first word is
    not itself scored.

    Response shape is parallel ARRAYS, not a list of per-token objects: `logprobs[i]` and
    `top_logprobs[i]` line up with `token_ids[i]`.
    """
    total, n, argmax = 0.0, 0, []
    for line in lines:
        cut = line.find(" ")
        if cut <= 0 or cut == len(line) - 1:
            continue                      # nothing to condition on, or nothing left to score
        prompt, completion = line[:cut], line[cut:]
        # Raw text, no chat template: the corpus is prose, and a template would score the
        # template's own tokens as if they were the model's opinion of the text.
        d = post(port, "/v1/score", {"model": "m", "prompt": prompt, "completion": completion,
                                     "top_logprobs": 1})
        lps = d.get("logprobs") or []
        alts = d.get("top_logprobs") or []
        total += float(d.get("sum_logprob", sum(lps)))
        n += len(lps)
        for i in range(len(lps)):
            per = alts[i] if i < len(alts) else []
            if per:
                best = max(per, key=lambda a: a.get("logprob", -1e30))
                argmax.append(best.get("token_id"))
            else:
                argmax.append(None)
    return total, n, argmax


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/root/workspace/models_bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
    ap.add_argument("--reference", default="/root/workspace/models_qwen38",
                    help="the checkpoint this model was quantized from (NVFP4, not FP16)")
    ap.add_argument("--tokenizer", default="/root/workspace/models_qwen38/tokenizer.json")
    ap.add_argument("--server", default="/workspace/sparkinfer/build/server/sparkinfer_server")
    ap.add_argument("--corpus", default="/workspace/sparkinfer/bench/scripts/eval_corpus.txt")
    ap.add_argument("--ctx", type=int, default=16384)
    a = ap.parse_args()

    lines = [l.strip() for l in open(a.corpus, encoding="utf-8") if l.strip()]
    print(f"corpus: {len(lines)} lines from {os.path.basename(a.corpus)}")

    runs = [("reference (NVFP4 parent)", a.reference, None),
            ("bonsai folded", a.model, ""),
            ("bonsai native", a.model, "all")]

    results = {}
    for label, model, native in runs:
        print(f"\n--- {label}")
        p, port = serve(a.server, model, a.tokenizer, native, a.ctx)
        try:
            total, n, argmax = score_corpus(port, lines)
        finally:
            stop(p)
        if n == 0:
            print("  scored nothing -- endpoint returned no per-token logprobs")
            results[label] = None
            continue
        ppl = math.exp(-total / n)
        results[label] = (ppl, n, argmax)
        print(f"  {n} positions, perplexity {ppl:.4f}")

    ref = results.get("reference (NVFP4 parent)")
    if ref:
        print(f"\ntop-1 agreement against the parent, position by position")
        for label in ("bonsai folded", "bonsai native"):
            r = results.get(label)
            if not r:
                continue
            pairs = [(x, y) for x, y in zip(r[2], ref[2]) if x is not None and y is not None]
            if not pairs:
                print(f"  {label:16s} no comparable positions (no top_logprobs returned)")
                continue
            agree = sum(1 for x, y in pairs if x == y) / len(pairs)
            print(f"  {label:16s} {agree * 100:.2f}%  over {len(pairs)} positions "
                  f"(ppl {r[0]:.4f} against the parent's {ref[0]:.4f}, "
                  f"ratio {r[0] / ref[0]:.3f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
