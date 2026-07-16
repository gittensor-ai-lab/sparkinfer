#!/usr/bin/env python3
# Compare per-position argmax between two qwen3_gguf_score dumps (sparse-off vs sparse-on).
# Score line format: "S i=<i> tgt=<t> am=<argmax> lp=<logprob_tgt> top=id:lp,..."
# Reports argmatch over positions >= MINPOS (where sparse KV is active) and overall.
import sys

def load(path):
    am = {}
    for ln in open(path):
        if not ln.startswith("S "):
            continue
        parts = ln.split()
        d = {}
        for p in parts[1:]:
            if "=" in p:
                k, _, v = p.partition("=")
                d[k] = v
        try:
            i = int(d["i"])
            am[i] = int(d["am"])
        except (KeyError, ValueError):
            continue
    return am

off = load(sys.argv[1])
on = load(sys.argv[2])
minpos = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
common = sorted(set(off) & set(on))
tot = long = tot_m = long_m = 0
for i in common:
    match = off[i] == on[i]
    tot += 1
    tot_m += match
    if i >= minpos:
        long += 1
        long_m += match
print(f"positions compared: {tot} (>= {minpos}: {long})")
if tot:
    print(f"argmatch ALL:        {tot_m}/{tot} = {tot_m/tot:.4f}")
if long:
    print(f"argmatch LONG(>= {minpos}): {long_m}/{long} = {long_m/long:.4f}")
else:
    print("no long positions — feed a longer sequence")
