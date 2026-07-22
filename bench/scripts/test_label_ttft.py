#!/usr/bin/env python3
"""TTFT reduction scoring via SPARKINFER_LABEL_LOWER_IS_BETTER (prefill tiers).

Run from the repo root:
  python3 bench/scripts/test_label_ttft.py
"""
import json
import os
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABEL = HERE / "label.py"


def run_label(value, frontier, *, top1=0.95, kl=0.01, lower=True, diff_ref=0.0, boost="0"):
    env = os.environ.copy()
    env["SPARKINFER_LABEL_LOWER_IS_BETTER"] = "1" if lower else "0"
    env["SPARKINFER_DIFFICULTY_REF"] = str(diff_ref)
    env["SPARKINFER_DIFFICULTY_BOOST"] = boost
    out = subprocess.check_output(
        ["python3", str(LABEL), str(value), str(frontier), "0",
         str(top1), str(kl), "deadbeef", "{}"],
        env=env, text=True,
    ).strip()
    assert out.startswith("RESULT_JSON "), out
    return json.loads(out[len("RESULT_JSON "):])


class LabelTtftTest(unittest.TestCase):
    def test_20pct_pp_gain_is_ttft_reduction_L(self):
        """20% pp gain → TTFT reduction ≈ 16.7% → L (same mid-gain family as pp path)."""
        # ctx cancel: ttft_main=1.0, ttft_pr = 1/1.2 ≈ 0.833 → g = 16.67%
        res = run_label(1.0 / 1.2, 1.0, diff_ref=0.0)
        self.assertTrue(res["pass"])
        self.assertAlmostEqual(res["pct_over_frontier"], 16.7, places=0)
        self.assertEqual(res["label"], "L")
        self.assertTrue(res.get("lower_is_better"))

    def test_small_ttft_cut_below_sig_is_none(self):
        """g < 2% → none (noise / not verified)."""
        res = run_label(0.99, 1.0, diff_ref=0.0)  # 1% reduction
        self.assertEqual(res["label"], "none")
        self.assertAlmostEqual(res["pct_over_frontier"], 1.0, places=0)

    def test_correctness_reject_exposes_speed_label_from_ttft(self):
        """Correctness REJECT still annotates speed_label from the TTFT path."""
        # Huge TTFT cut (half latency) with failing top1
        res = run_label(0.5, 1.0, top1=0.5, kl=0.5, diff_ref=0.0)
        self.assertEqual(res["label"], "REJECT")
        self.assertFalse(res["pass"])
        self.assertEqual(res["speed_label"], "XL")
        self.assertGreater(res["pct_over_frontier"], 18.0)

    def test_pp_equivalence_formula(self):
        """g = 1 - pp_main/pp_PR matches TTFT reduction from ctx/pp."""
        ctx, pp_main, pp_pr = 4096.0, 1000.0, 1200.0
        ttft_main = ctx / pp_main
        ttft_pr = ctx / pp_pr
        g_pp = 1.0 - pp_main / pp_pr
        g_ttft = (ttft_main - ttft_pr) / ttft_main
        self.assertAlmostEqual(g_pp, g_ttft, places=9)
        res = run_label(ttft_pr, ttft_main, diff_ref=0.0)
        self.assertAlmostEqual(res["pct_over_frontier"] / 100.0, g_ttft, places=2)


if __name__ == "__main__":
    unittest.main()
