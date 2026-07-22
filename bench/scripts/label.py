#!/usr/bin/env python3
"""Eval-loop label = deterministic function of measurements (so validators converge).

  label.py <value> <frontier> <ceiling> <top1> <kl> <commit> [prov_json]

Emits one line:  RESULT_JSON {...}

- Correctness gate first: top-1 token agreement >= 0.90 and KL <= 0.20 (preferred <= 0.15), else
  REJECT (score 0). A pass with KL above the preferred 0.15 is flagged `accuracy_warn` — accuracy is
  first; a speed gain that erodes parity with llama.cpp is not worth taking.
- Significance gate: the gain must exceed SIG (2% of the frontier, a CI/noise proxy), else "none".
  A gain that clears it floors at XS (verified but small); "none" always means "not verified".
- Label tier = bucket of the gain **sized against the llama.cpp reference** (delta / llama_ref), NOT
  the frontier. llama.cpp is a constant maturity anchor for every model, so the same tok/s of real
  work earns the same tier whether the model started at 23 or 493 tok/s. This fixes the unfairness of
  delta/frontier: over an un-optimized frontier (e.g. Qwen3.6 at 23 tok/s) a small absolute gain used
  to explode to XL, while a mature model (past llama) couldn't clear XS for equal effort. The
  past-llama difficulty boost (Option B) still multiplies the *tier* once the frontier is beyond the
  reference, then caps tier credit at 2× the measured %-over-frontier. Significance still gates on raw
  %-over-frontier (a gain must beat the current best); pct_over_frontier reports the honest measured
  speedup; pct_of_llama is the tier basis.
  llama_ref (SPARKINFER_DIFFICULTY_REF) <= 0 falls back to the legacy delta/frontier basis.
- SPARKINFER_LABEL_LOWER_IS_BETTER=1: value/frontier are latencies (e.g. TTFT seconds). Gain is the
  reduction fraction (frontier - value) / frontier; pct_over_frontier is % reduced. Prefill scoring
  uses this so tiers reflect time-to-first-token cuts vs same-box main.
  Thresholds are governance-tunable.
"""
import sys, json, os

tps      = float(sys.argv[1])   # measured median tok/s (or TTFT seconds when lower-is-better)
frontier = float(sys.argv[2])   # current best verified tok/s / TTFT (0 = none yet)
ceiling  = float(sys.argv[3])   # roofline / strong-reference cap (display only; unused if 0)
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
# Prefill / TTFT: score latency reduction instead of throughput increase.
LOWER_IS_BETTER = os.environ.get("SPARKINFER_LABEL_LOWER_IS_BETTER", "0") == "1"

# ---- Optional difficulty compensation (Option B — opt-in, governance-tunable) ----
# As the frontier pulls past a mature reference (llama.cpp), each further % gain is harder; scale the
# LABEL up so a late-game hard PR scores like the effort it took. D = 1 for a frontier at/below the
# reference (the cold-start era is untouched — no retroactive inflation), grows with distance past it,
# and is bounded by DIFF_MAX so nothing runs away to XL. Tier credit is then capped at 2× the measured
# %-over-frontier (g) so a low per-context llama ref cannot inflate the label past twice the real
# speedup. Crucially the boost multiplies the *label* only: the significance gate stays on the RAW
# delta (so noise is never boosted) and pct_over_frontier reports the true measured speedup. OFF by
# default.
DIFF_BOOST = os.environ.get("SPARKINFER_DIFFICULTY_BOOST", "0") == "1"
DIFF_K     = float(os.environ.get("SPARKINFER_DIFFICULTY_K",   "8"))
DIFF_REF   = float(os.environ.get("SPARKINFER_DIFFICULTY_REF", "365.85"))  # llama.cpp 128-tok tok/s
DIFF_MAX   = float(os.environ.get("SPARKINFER_DIFFICULTY_MAX", "1.5"))

def difficulty_mult(frontier):
    if not DIFF_BOOST or DIFF_REF <= 0 or frontier <= 0:
        return 1.0
    if LOWER_IS_BETTER:
        # Latency: past-llama when frontier TTFT is *below* llama TTFT (faster).
        return min(1.0 + DIFF_K * max(0.0, DIFF_REF / frontier - 1.0), DIFF_MAX)
    return min(1.0 + DIFF_K * max(0.0, frontier / DIFF_REF - 1.0), DIFF_MAX)

res = {"commit": commit, "tps": round(tps, 6 if LOWER_IS_BETTER else 2),
       "top1": round(top1, 4), "kl": round(kl, 4),
       "frontier_tps": round(frontier, 6 if LOWER_IS_BETTER else 2)}
if LOWER_IS_BETTER:
    res["lower_is_better"] = True


def _speed_tier_fields(tps, frontier, ceiling):
    """Compute speed-tier fields as if correctness already passed. Used for the headline label
    and, on correctness REJECT, for speed_label so real prefill/decode gains still get annotated."""
    out = {}
    if frontier <= 0:
        out["label"] = "BASELINE"
        return out
    if LOWER_IS_BETTER:
        delta = frontier - tps                          # seconds saved (positive when faster)
        g = delta / frontier                            # TTFT reduction fraction
    else:
        delta = tps - frontier
        g = delta / frontier                            # relative speedup — SIGNIFICANCE basis
    if g <= SIG:
        out.update(label="none", delta_tps=round(delta, 6 if LOWER_IS_BETTER else 2),
                   pct_over_frontier=round(100 * g, 1),
                   note=("within significance gate — not a verified TTFT reduction"
                         if LOWER_IS_BETTER else
                         "within significance gate — not a verified improvement"))
        return out
    # FAIR label tier: size the gain against the llama.cpp reference (DIFF_REF — a constant maturity
    # anchor for EVERY model), not the possibly-unoptimized frontier. So the same tok/s of real work
    # earns the same tier whether the model started at 23 or 493 tok/s — an un-optimized model can no
    # longer mint XLs from low-hanging fruit while a mature one (past llama) can't clear XS. Significance
    # still gates on raw %-over-frontier above (a gain must beat the current best); only the TIER is
    # llama-anchored. Past-llama difficulty boost (Option B) is unchanged. DIFF_REF<=0 -> legacy basis.
    ref = DIFF_REF if DIFF_REF > 0 else frontier
    if LOWER_IS_BETTER:
        # Latency: fair gain = how much of llama's TTFT we cut vs llama (ref - value) / ref.
        g_fair = (ref - tps) / ref if ref > 0 else g
    else:
        g_fair = delta / ref
    D = difficulty_mult(frontier)                       # hard-gain boost once past the reference
    g_eff = min(g_fair * D, 2 * g)                      # strict cap: tier credit ≤ 2× measured speedup
    # A verified improvement over the frontier floors at XS (real but small); the higher tiers
    # (S/M/L/XL) are earned by the llama-anchored size. So "none" always means "not a verified
    # improvement", never "real but tiny".
    label = next((l for thr, l in BUCKETS if g_eff >= thr), "XS")
    out.update(label=label, delta_tps=round(delta, 6 if LOWER_IS_BETTER else 2),
               pct_over_frontier=round(100 * g, 1),      # RAW measured speedup / TTFT reduction
               pct_of_llama=round(100 * g_fair, 1),      # gain vs llama — the label basis
               pct_of_ceiling=round(100 * tps / ceiling, 1) if ceiling > 0 and not LOWER_IS_BETTER else None)
    out["effective_pct"] = round(100 * g_eff, 1)
    if D != 1.0:                                        # transparency: expose the boost in the verdict
        out["difficulty_mult"] = round(D, 2)
    return out


if top1 < TOP1_BAR or kl > KL_BAR:
    res.update(pass_=False, label="REJECT",
               reason=f"correctness gate: top1={top1} (need >= {TOP1_BAR}), kl={kl} (need <= {KL_BAR})")
    # Keep reporting the speed tier that would have been earned — overall stays REJECT / not mergeable,
    # but prefill (or decode) progress is still labeled (e.g. eval-prefill:XL on PR #403).
    if frontier > 0 and tps > 0:
        speed = _speed_tier_fields(tps, frontier, ceiling)
        res["speed_label"] = speed["label"]
        for k in ("delta_tps", "pct_over_frontier", "pct_of_llama", "pct_of_ceiling",
                  "effective_pct", "difficulty_mult", "note"):
            if k in speed and speed[k] is not None:
                res[k] = speed[k]
elif frontier <= 0:
    res.update(pass_=True, label="BASELINE", note="no frontier set; this submission becomes it")
    res["speed_label"] = "BASELINE"
else:
    speed = _speed_tier_fields(tps, frontier, ceiling)
    res.update(pass_=True, **{k: v for k, v in speed.items() if k != "label"})
    res["label"] = speed["label"]
    res["speed_label"] = speed["label"]

# Soft accuracy flag: passed the gate but above the preferred KL ceiling — accepted, margin is thin.
if res.get("label") != "REJECT" and kl > KL_PREFER:
    res["accuracy_warn"] = f"KL {round(kl, 4)} above preferred {KL_PREFER} (hard reject at {KL_BAR})"

# JSON keys can't be "pass" via kwarg; normalize
res["pass"] = res.pop("pass_", True)
res.update(prov)                                       # M1/H1/C2 provenance (non-scoring)
print("RESULT_JSON " + json.dumps(res))
