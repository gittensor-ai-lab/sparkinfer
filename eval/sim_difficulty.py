#!/usr/bin/env python3
"""Replay the sparkinfer frontier history through the difficulty-compensated labeling (Option B), so a
governance change to the K / reference / cap can be judged on real data instead of vibes.

  eval/sim_difficulty.py [K ...]        # default K = 6 8 10

Mirrors bench/scripts/label.py: D = 1 + K*max(0, frontier/ref - 1), capped at DIFF_MAX; g_eff = g*D;
the significance gate stays on the RAW delta so noise is never boosted. Reads the landed frontier
journey from origin/main's dashboard/data.json.
"""
import json, subprocess, sys

REF = 365.85          # llama.cpp 128-tok reference (SPARKINFER_DIFFICULTY_REF)
DIFF_MAX = 4.0        # cap (SPARKINFER_DIFFICULTY_MAX)
SIG = 0.02
BUCKETS = [(0.18, "XL"), (0.10, "L"), (0.06, "M"), (0.035, "S"), (SIG, "XS")]


def bucket(g):
    return "none" if g <= SIG else next(l for thr, l in BUCKETS if g >= thr)


def sim(K, landed):
    print(f"\n===== K = {K}  (ref = {REF}, cap = {DIFF_MAX}) =====")
    print(f"{'PR':>5} {'base':>7} {'->tps':>7} {'g_raw':>6} {'cur':>5} {'D':>5} {'g_eff':>6} {'boosted':>7}  change")
    prev = None
    for m in landed:
        tps = m["tps"]
        if prev is None or tps <= prev:
            prev = tps; continue
        g = (tps - prev) / prev
        cur = bucket(g)
        D = min(1.0 + K * max(0.0, prev / REF - 1.0), DIFF_MAX)
        g_eff = g * D if g > SIG else g            # gate on raw, boost only real gains
        boosted = bucket(g_eff)
        chg = "" if boosted == cur else f"  {cur} -> {boosted}"
        print(f"#{m['pr']:>4} {prev:7.1f} {tps:7.1f} {100*g:5.1f}% {cur:>5} {D:5.2f} {100*g_eff:5.1f}% {boosted:>7}{chg}")
        prev = tps


def main():
    Ks = [float(x) for x in sys.argv[1:]] or [6, 8, 10]
    data = json.loads(subprocess.run(["git", "show", "origin/main:dashboard/data.json"],
                                     capture_output=True, text=True).stdout)
    landed = sorted(data["landed"], key=lambda m: m["tps"])
    for K in Ks:
        sim(K, landed)


if __name__ == "__main__":
    main()
