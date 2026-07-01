#!/usr/bin/env python3
"""Eval-loop label = deterministic function of measurements (so validators converge).

  label.py <tps> <frontier_tps> <ceiling_tps> <top1> <kl> <commit>

Emits one line:  RESULT_JSON {...}

- Correctness gate first: top-1 token agreement >= 0.90 and KL <= 0.20 (preferred <= 0.15), else
  REJECT (score 0). A pass with KL above the preferred 0.15 is flagged `accuracy_warn` — accuracy is
  first; a speed gain that erodes parity with llama.cpp is not worth taking.
- Significance gate: the gain must exceed SIG (2% of the frontier, a CI/noise proxy), else "none".
- Label = bucket of the **relative speedup over the frontier** (delta / frontier). Same denominator
  as the significance gate, so all five tiers stay reachable as decode speed grows — a
  fraction-of-headroom rule collapsed XS/S once the frontier neared the ceiling (the 2% noise floor
  alone already exceeded their headroom bands). The tiers are adaptive in that the *absolute* tok/s
  required for each grows with the frontier. Thresholds are governance-tunable.
"""
import sys, json, os

tps      = float(sys.argv[1])   # measured median tok/s of the submission
frontier = float(sys.argv[2])   # current best verified tok/s (0 = none yet)
ceiling  = float(sys.argv[3])   # roofline / strong-reference cap (display only)
top1     = float(sys.argv[4])   # token-match vs reference, 0..1
kl       = float(sys.argv[5])   # mean KL vs reference (nats)
commit   = sys.argv[6]
# Optional 7th arg: M1/H1/C2 provenance (clocks_pinned, clock_mhz, eval_seed, llama_commit, ...)
# merged verbatim into the verdict so the immutable log is self-describing and a verifier can
# reproduce at the same clock + prompt seed. Does not affect the deterministic scoring above.
prov     = json.loads(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] else {}

# Correctness gate (governance-tunable). Accuracy parity with llama.cpp is the moat: a speedup that
# erodes it is REJECTed regardless of speed. KL_BAR is the HARD reject ceiling; KL_PREFER the soft
# target — a pass above it is flagged.
#
# These STRICT bars hold on the held-out prompts (H1) because the KL metric was fixed: it now dumps a
# deep sparkinfer top-k so llama's tail isn't floored (see accuracy.sh / accuracy_compare.py). Before
# the fix the gate read KL 0.14–0.33 on diverse prompts (a truncation artifact) and seemed to need
# loosening; with matched-depth measurement the TRUE divergence is ~0.01–0.03 (top-1 0.96–0.98), so
# the original 0.20 ceiling holds with large margin. Don't loosen these to paper over a metric bug.
TOP1_BAR  = float(os.environ.get("SPARKINFER_TOP1_BAR",  "0.90"))
KL_BAR    = float(os.environ.get("SPARKINFER_KL_BAR",    "0.20"))
KL_PREFER = float(os.environ.get("SPARKINFER_KL_PREFER", "0.15"))
SIG = 0.02                                              # noise floor: gain must beat 2% of frontier
# min relative speedup (delta/frontier) for each tier; XS starts at the noise floor SIG.
BUCKETS = [(0.18, "XL"), (0.10, "L"), (0.06, "M"), (0.035, "S"), (SIG, "XS")]

# ---- Optional difficulty compensation (Option B — opt-in, governance-tunable) ----
# As the frontier pulls past a mature reference (llama.cpp), each further % gain is harder; scale the
# LABEL up so a late-game hard PR scores like the effort it took. D = 1 for a frontier at/below the
# reference (the cold-start era is untouched — no retroactive inflation), grows with distance past it,
# and is bounded by DIFF_MAX so nothing runs away to XL. Crucially the boost multiplies the *label*
# only: the significance gate stays on the RAW delta (so noise is never boosted) and pct_over_frontier
# still reports the true measured speedup. OFF by default.
DIFF_BOOST = os.environ.get("SPARKINFER_DIFFICULTY_BOOST", "0") == "1"
DIFF_K     = float(os.environ.get("SPARKINFER_DIFFICULTY_K",   "8"))
DIFF_REF   = float(os.environ.get("SPARKINFER_DIFFICULTY_REF", "365.85"))  # llama.cpp 128-tok tok/s
DIFF_MAX   = float(os.environ.get("SPARKINFER_DIFFICULTY_MAX", "4"))

def difficulty_mult(frontier):
    if not DIFF_BOOST or DIFF_REF <= 0:
        return 1.0
    return min(1.0 + DIFF_K * max(0.0, frontier / DIFF_REF - 1.0), DIFF_MAX)

res = {"commit": commit, "tps": round(tps, 2), "top1": round(top1, 4),
       "kl": round(kl, 4), "frontier_tps": round(frontier, 2)}

if top1 < TOP1_BAR or kl > KL_BAR:
    res.update(pass_=False, label="REJECT",
               reason=f"correctness gate: top1={top1} (need >= {TOP1_BAR}), kl={kl} (need <= {KL_BAR})")
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
        D = difficulty_mult(frontier)
        g_eff = g * D                                   # boost the label tier, not the raw % or the gate
        label = next(l for thr, l in BUCKETS if g_eff >= thr)
        res.update(pass_=True, label=label, delta_tps=round(delta, 2),
                   pct_over_frontier=round(100 * g, 1),   # RAW measured speedup (honest reporting)
                   pct_of_ceiling=round(100 * tps / ceiling, 1) if ceiling > 0 else None)
        if D != 1.0:                                    # transparency: expose the boost in the verdict
            res["difficulty_mult"] = round(D, 2)
            res["effective_pct"] = round(100 * g_eff, 1)

# Soft accuracy flag: passed the gate but above the preferred KL ceiling — accepted, margin is thin.
if res.get("label") != "REJECT" and kl > KL_PREFER:
    res["accuracy_warn"] = f"KL {round(kl, 4)} above preferred {KL_PREFER} (hard reject at {KL_BAR})"

# JSON keys can't be "pass" via kwarg; normalize
res["pass"] = res.pop("pass_", True)
res.update(prov)                                       # M1/H1/C2 provenance (non-scoring)
print("RESULT_JSON " + json.dumps(res))
