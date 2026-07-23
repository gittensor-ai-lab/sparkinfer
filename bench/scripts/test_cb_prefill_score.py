#!/usr/bin/env python3
"""CB mixed-load TTFT scoring + max-of with single-seq prefill.

Run from the repo root:
  python3 bench/scripts/test_cb_prefill_score.py
"""
import json
import os
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABEL = HERE / "label.py"

TIER_RANK = {"XL": 6, "L": 5, "M": 4, "S": 3, "XS": 2, "none": 1, "BASELINE": 0, "REJECT": -1}


def tier_rank(label):
    return TIER_RANK.get(label or "none", -1)


def run_label_ttft(value, frontier, *, top1=0.95, kl=0.01):
    env = os.environ.copy()
    env["SPARKINFER_LABEL_LOWER_IS_BETTER"] = "1"
    env["SPARKINFER_DIFFICULTY_REF"] = "0"
    env["SPARKINFER_DIFFICULTY_BOOST"] = "0"
    out = subprocess.check_output(
        ["python3", str(LABEL), str(value), str(frontier), "0",
         str(top1), str(kl), "deadbeef", "{}"],
        env=env, text=True,
    ).strip()
    assert out.startswith("RESULT_JSON "), out
    return json.loads(out[len("RESULT_JSON "):])


def max_of_prefill(single, cb):
    """Mirror evaluate.sh: promote CB when its tier beats single-seq."""
    pf = dict(single)
    pf["prefill_score_source"] = "single"
    if cb.get("cb_ttft_s") and cb.get("cb_frontier_ttft_s"):
        cb_tier = cb.get("speed_label") or cb.get("label")
        single_tier = pf.get("speed_label") or pf.get("label")
        if tier_rank(cb_tier) > tier_rank(single_tier):
            for k in ("label", "speed_label", "pct_over_frontier", "pct_of_llama",
                      "effective_pct"):
                if k in cb and cb[k] is not None:
                    pf[k] = cb[k]
            pf["prefill_ttft_s"] = cb["cb_ttft_s"]
            pf["prefill_frontier_ttft_s"] = cb["cb_frontier_ttft_s"]
            pf["prefill_score_source"] = "cb"
    return pf


class CbPrefillScoreTest(unittest.TestCase):
    def test_pr585_shaped_ttft_is_xl(self):
        """25.773s → 0.351s ≈ 98.6% reduction → XL."""
        res = run_label_ttft(0.351, 25.773)
        self.assertTrue(res["pass"])
        self.assertEqual(res["label"], "XL")
        self.assertAlmostEqual(res["pct_over_frontier"], 98.6, places=0)

    def test_flat_cb_is_none(self):
        res = run_label_ttft(25.0, 25.1)  # ~0.4% cut
        self.assertEqual(res["label"], "none")

    def test_cb_xl_beats_single_none(self):
        single = {"label": "none", "speed_label": "none", "pct_over_frontier": 0.5}
        cb = run_label_ttft(0.351, 25.773)
        cb["cb_ttft_s"] = 0.351
        cb["cb_frontier_ttft_s"] = 25.773
        out = max_of_prefill(single, cb)
        self.assertEqual(out["label"], "XL")
        self.assertEqual(out["prefill_score_source"], "cb")

    def test_single_L_beats_cb_none(self):
        single = {"label": "L", "speed_label": "L", "pct_over_frontier": 16.7}
        cb = run_label_ttft(25.0, 25.1)
        cb["cb_ttft_s"] = 25.0
        cb["cb_frontier_ttft_s"] = 25.1
        out = max_of_prefill(single, cb)
        self.assertEqual(out["label"], "L")
        self.assertEqual(out["prefill_score_source"], "single")


if __name__ == "__main__":
    unittest.main()
