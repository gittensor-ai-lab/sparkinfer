#!/usr/bin/env python3
"""Compare sparkinfer vs a running llama.cpp server on the same token sequence.

  accuracy_compare.py <spark_score.txt> <tokenizer.json> <text> [server_url] [topk]

Reads sparkinfer's teacher-forced score dump (from qwen3_gguf_score) and queries the
llama.cpp server (/completion, n_probs + cache_prompt) for the same per-position
distributions, then reports:
  - top-1 token agreement   (argmax_spark == argmax_llama)   -> implementation correctness
  - mean KL(llama || spark) over the top-k union             -> distribution closeness
  - perplexity for each engine (exp(-mean log p(actual next)))
"""
import sys, json, math, urllib.request
from tokenizers import Tokenizer

score_path, tok_path, text_path = sys.argv[1], sys.argv[2], sys.argv[3]
URL  = sys.argv[4] if len(sys.argv) > 4 else "http://localhost:8081"
# llama top-k to query per position. MUST be <= sparkinfer's dump depth (accuracy.sh dumps 128) so
# every token llama gives mass to is present in sparkinfer's distribution — otherwise it gets FLOOR'd
# and KL is inflated by a truncation artifact (the metric bug that read 0.14-0.33 instead of ~0.02).
TOPK = int(sys.argv[5]) if len(sys.argv) > 5 else 64
FLOOR = -20.0

# 3rd arg is either a file of space-separated token ids (the EXACT prompt scored — produced by
# gen_eval_prompt.py so sparkinfer and llama see the identical sequence) or, legacy, plain text.
_raw = open(text_path).read().strip()
_toks = _raw.split()
if _toks and all(t.lstrip("-").isdigit() for t in _toks):
    ids = [int(t) for t in _toks]
else:
    ids = Tokenizer.from_file(tok_path).encode(_raw).ids

def llama_dist(prefix):
    req = {"prompt": prefix, "n_predict": 1, "n_probs": TOPK, "temperature": 0, "cache_prompt": True}
    r = urllib.request.urlopen(urllib.request.Request(
        URL + "/completion", data=json.dumps(req).encode(),
        headers={"Content-Type": "application/json"}), timeout=120)
    tl = json.load(r)["completion_probabilities"][0]["top_logprobs"]
    return {e["id"]: e["logprob"] for e in tl}

spark = {}
for line in open(score_path):
    if not line.startswith("S "): continue
    p = line.split(); i = int(p[1][2:]); am = int(p[3][3:]); lp = float(p[4][3:])
    top = {int(x.split(":")[0]): float(x.split(":")[1]) for x in line.split("top=", 1)[1].split(",")}
    spark[i] = {"am": am, "lp": lp, "top": top}

match = n = 0; snll = lnll = 0.0; klsum = 0.0
for i in range(len(ids) - 1):
    if i not in spark: continue
    ld = llama_dist(ids[:i + 1]); lam = max(ld, key=ld.get); n += 1
    if spark[i]["am"] == lam: match += 1
    snll += -spark[i]["lp"]; lnll += -ld.get(ids[i + 1], FLOOR)
    sd = spark[i]["top"]; U = set(ld) | set(sd)
    P = {k: math.exp(ld.get(k, FLOOR)) for k in U}; Q = {k: math.exp(sd.get(k, FLOOR)) for k in U}
    ps = sum(P.values()); qs = sum(Q.values()); kl = 0.0
    for k in U:
        pp = P[k] / ps; qq = Q[k] / qs
        if pp > 0: kl += pp * math.log(pp / max(qq, 1e-12))
    klsum += kl

print(f"positions             : {n}")
print(f"token-match (top-1)   : {match}/{n} = {match/n:.3f}   (bar >= 0.90)")
print(f"mean KL(llama||spark) : {klsum/n:.4f} nats  (top-k approx)")
print(f"PPL sparkinfer        : {math.exp(snll/n):.3f}  (exact, full softmax)")
print(f"PPL llama.cpp         : {math.exp(lnll/n):.3f}  (top-{TOPK}+floor; inflated)")
# unambiguous machine-readable line for evaluate.sh (avoid parsing the human text above)
print(f"METRIC top1={match/n:.6f} kl={klsum/n:.6f} ppl_spark={math.exp(snll/n):.4f} ppl_llama={math.exp(lnll/n):.4f}")
