#!/usr/bin/env python3
"""Compare two sparkinfer teacher-forced score dumps against each other.

  accuracy_compare_pair.py <pr_score.txt> <main_score.txt> [--metric-label NAME]

Sibling of accuracy_compare.py, which compares sparkinfer against a live llama.cpp server. That
methodology needs llama.cpp to load *the same weights*, which is impossible for a HuggingFace
compressed-tensors (NVFP4/FP8) checkpoint -- llama.cpp cannot read one. Comparing against a GGUF
of the same model instead would compare two different quantizations and could never hold a tight
top1/KL bar.

So for the Qwen3.8-27B NVFP4 path the correctness gate is differential rather than absolute:
score the SAME token stream on the PR build and on origin/main, and require the two
distributions to agree. That catches any PR that changes the model's numerics -- which is what a
per-PR gate is for -- without needing an external reference. It deliberately cannot catch a bug
that is already present on main; only a newly introduced divergence.

`main` is the reference distribution (P) and the PR is the candidate (Q), so KL(main||pr) mirrors
accuracy_compare.py's KL(llama||spark) orientation: it penalises the PR putting little mass where
the reference puts a lot.

Dump line format (qwen3_gguf_score):
  S i=<pos> tgt=<id> am=<argmax id> lp=<logprob of tgt> top=<id>:<lp>,<id>:<lp>,...
"""
import math
import sys

argv = sys.argv[1:]
LABEL = "METRIC"
if "--metric-label" in argv:
    i = argv.index("--metric-label")
    LABEL = argv[i + 1]
    del argv[i:i + 2]

if len(argv) < 2:
    print(__doc__.strip().splitlines()[2].strip())
    sys.exit(2)

pr_path, main_path = argv[0], argv[1]
# Same floor accuracy_compare.py uses for tokens outside a dump's top-k: log-prob of an
# unlisted token is unknown, not zero, so clamp rather than drop it from the union.
FLOOR = -20.0


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("S "):
                continue
            try:
                p = line.split()
                pos = int(p[1][2:])
                am = int(p[3][3:])
                lp = float(p[4][3:])
                top = {
                    int(x.split(":")[0]): float(x.split(":")[1])
                    for x in line.split("top=", 1)[1].split(",")
                }
            except (IndexError, ValueError):
                continue   # a truncated final line (killed process) must not abort the compare
            out[pos] = {"am": am, "lp": lp, "top": top}
    return out


pr, ref = load(pr_path), load(main_path)
shared = sorted(set(pr) & set(ref))

if not shared:
    # Hostile defaults, matching accuracy_compare.py: a gate that cannot measure must FAIL,
    # never silently pass.
    print(f"{LABEL} top1=0 kl=99 ppl_pr=0 ppl_main=0 n=0 n_main={len(ref)}   (NO SHARED POSITIONS)")
    sys.exit(1)

match = 0
klsum = 0.0
pr_nll = 0.0
ref_nll = 0.0
for pos in shared:
    a, b = pr[pos], ref[pos]
    if a["am"] == b["am"]:
        match += 1
    pr_nll += -a["lp"]
    ref_nll += -b["lp"]
    U = set(a["top"]) | set(b["top"])
    P = {k: math.exp(b["top"].get(k, FLOOR)) for k in U}   # reference = main
    Q = {k: math.exp(a["top"].get(k, FLOOR)) for k in U}   # candidate = PR
    ps, qs = sum(P.values()), sum(Q.values())
    kl = 0.0
    for k in U:
        pp = P[k] / ps
        qq = Q[k] / qs
        if pp > 0:
            kl += pp * math.log(pp / max(qq, 1e-12))
    klsum += kl

n = len(shared)
skipped = (len(pr) - n) + (len(ref) - n)
print(f"positions             : {n}" + (f"  ({skipped} unpaired, skipped)" if skipped else ""))
print(f"top-1 agreement       : {match}/{n} = {match / n:.3f}   (PR vs main)")
print(f"mean KL(main||pr)     : {klsum / n:.4f} nats  (top-k union)")
print(f"PPL PR                : {math.exp(pr_nll / n):.3f}")
print(f"PPL main              : {math.exp(ref_nll / n):.3f}")
# n= / n_main=: only shared positions are compared, so a PR dump missing positions would be judged
# on the ones it has. The bots require n == n_main.
print(
    f"{LABEL} top1={match / n:.6f} kl={klsum / n:.6f} "
    f"ppl_pr={math.exp(pr_nll / n):.4f} ppl_main={math.exp(ref_nll / n):.4f} n={n} n_main={len(ref)}"
)
