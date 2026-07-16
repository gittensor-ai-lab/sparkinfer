#!/usr/bin/env python3
"""Self-consistency of two qwen3_gguf_score dumps on the SAME token sequence.

  self_consistency.py <score_a.txt> <score_b.txt>

Compares two teacher-forced score dumps from qwen3_gguf_score — e.g. the bf16
dequant-GEMV path (SPARKINFER_MMVQ=0) vs the int8 dp4a MMVQ path
(SPARKINFER_MMVQ=1). Because both run the *same* weights on the *same* engine,
this is the tightest possible correctness gate for a lossless kernel change:
a faithful optimization should be greedy-identical (argmax agreement = 1.0) with
a near-zero self-KL; any divergence beyond int8-activation rounding is a bug.
"""
import sys, math

def load(p):
    d = {}
    for line in open(p):
        if not line.startswith("S "): continue
        f = line.split()
        i = int(f[1][2:]); am = int(f[3][3:]); lp = float(f[4][3:])
        top = {int(x.split(":")[0]): float(x.split(":")[1])
               for x in line.split("top=", 1)[1].split(",")}
        d[i] = {"am": am, "lp": lp, "top": top}
    return d

a, b = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(a) & set(b))
if not keys:
    print("no overlapping positions"); sys.exit(1)

match = sum(1 for i in keys if a[i]["am"] == b[i]["am"])
mlp = max(abs(a[i]["lp"] - b[i]["lp"]) for i in keys)
kls = []
for i in keys:                              # KL(a||b) over union of the two top-sets
    A, B = a[i]["top"], b[i]["top"]; U = set(A) | set(B)
    P = {k: math.exp(A.get(k, -20.0)) for k in U}
    Q = {k: math.exp(B.get(k, -20.0)) for k in U}
    ps, qs, kl = sum(P.values()), sum(Q.values()), 0.0
    for k in U:
        pp = P[k] / ps; qq = Q[k] / qs
        if pp > 0: kl += pp * math.log(pp / max(qq, 1e-12))
    kls.append(kl)

print(f"positions            : {len(keys)}")
print(f"argmax agreement     : {match}/{len(keys)} = {match/len(keys):.4f}   (want 1.0000)")
print(f"max |logprob diff|   : {mlp:.5f}")
print(f"mean self-KL(a||b)   : {sum(kls)/len(kls):.6f} nats   (want ~0)")
print(f"METRIC agree={match/len(keys):.6f} maxlp={mlp:.6f} selfkl={sum(kls)/len(kls):.6f}")
