#!/usr/bin/env python3
"""Eval-loop label = deterministic function of measurements (so validators converge).

  label.py <tps> <frontier_tps> <ceiling_tps> <top1> <kl> <commit>

Emits one line:  RESULT_JSON {...}

- Correctness gate first: top-1 token agreement >= 0.90 and KL <= 0.5, else REJECT (score 0).
- Significance gate: the gain must exceed SIG (2% of the frontier, a CI/noise proxy), else "none".
- Label = bucket of the **relative speedup over the frontier** (delta / frontier). Same denominator
  as the significance gate, so all five tiers stay reachable as decode speed grows — a
  fraction-of-headroom rule collapsed XS/S once the frontier neared the ceiling (the 2% noise floor
  alone already exceeded their headroom bands). The tiers are adaptive in that the *absolute* tok/s
  required for each grows with the frontier. Thresholds are governance-tunable.
"""
import sys, json

tps      = float(sys.argv[1])   # measured median tok/s of the submission
frontier = float(sys.argv[2])   # current best verified tok/s (0 = none yet)
ceiling  = float(sys.argv[3])   # roofline / strong-reference cap (display only)
top1     = float(sys.argv[4])   # token-match vs reference, 0..1
kl       = float(sys.argv[5])   # mean KL vs reference (nats)
commit   = sys.argv[6]

TOP1_BAR, KL_BAR = 0.90, 0.50
SIG = 0.02                                              # noise floor: gain must beat 2% of frontier
# min relative speedup (delta/frontier) for each tier; XS starts at the noise floor SIG.
BUCKETS = [(0.18, "XL"), (0.10, "L"), (0.06, "M"), (0.035, "S"), (SIG, "XS")]

res = {"commit": commit, "tps": round(tps, 2), "top1": round(top1, 4),
       "kl": round(kl, 4), "frontier_tps": round(frontier, 2)}

if top1 < TOP1_BAR or kl > KL_BAR:
    res.update(pass_=False, label="REJECT", reason=f"correctness (top1={top1}, kl={kl})")
elif frontier <= 0:
    res.update(pass_=True, label="BASELINE", note="no frontier set; this submission becomes it")
else:
    delta = tps - frontier
    g = delta / frontier                                # relative speedup over the frontier
    if g <= SIG:
        res.update(pass_=True, label="none", delta_tps=round(delta, 2),
                   pct_over_frontier=round(100 * g, 1),
                   note="within significance gate — not a verified improvement")
    else:
        label = next(l for thr, l in BUCKETS if g >= thr)
        res.update(pass_=True, label=label, delta_tps=round(delta, 2),
                   pct_over_frontier=round(100 * g, 1),
                   pct_of_ceiling=round(100 * tps / ceiling, 1) if ceiling > 0 else None)

# JSON keys can't be "pass" via kwarg; normalize
res["pass"] = res.pop("pass_", True)
print("RESULT_JSON " + json.dumps(res))
