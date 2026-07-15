#!/usr/bin/env python3
"""Tests for decode/prefill headline label merge in evaluate.sh.

Run from the repo root:
  python3 bench/scripts/test_prefill_label_merge.py
"""
import json
import unittest

TIER_RANK = {"XL": 6, "L": 5, "M": 4, "S": 3, "XS": 2, "none": 1, "BASELINE": 0, "REJECT": -1}


def tier_rank(label):
    return TIER_RANK.get(label or "none", -1)


def merge_decode_prefill(decode, prefill):
    res = dict(decode)
    pf = dict(prefill)
    res["prefill_label"] = pf.get("label")
    res["prefill_tps"] = pf.get("tps")
    if tier_rank(pf.get("label")) > tier_rank(res.get("label")):
        res["decode_label"] = res.get("label")
        for key in ("label", "tps", "frontier_tps", "delta_tps", "pct_over_frontier"):
            if key in pf:
                res[key] = pf[key]
        res["score_metric"] = "prefill"
    else:
        res["score_metric"] = "decode"
    return res


class PrefillLabelMergeTest(unittest.TestCase):
    def test_prefill_L_beats_decode_none(self):
        out = merge_decode_prefill(
            {"label": "none", "tps": 283.16, "frontier_tps": 283.28},
            {"label": "L", "tps": 320.28, "frontier_tps": 288.21,
             "delta_tps": 32.07, "pct_over_frontier": 11.1},
        )
        self.assertEqual(out["label"], "L")
        self.assertEqual(out["decode_label"], "none")
        self.assertEqual(out["score_metric"], "prefill")
        self.assertEqual(out["prefill_label"], "L")

    def test_decode_XS_beats_prefill_none(self):
        out = merge_decode_prefill(
            {"label": "XS", "tps": 300.0, "frontier_tps": 290.0},
            {"label": "none", "tps": 291.0, "frontier_tps": 290.0},
        )
        self.assertEqual(out["label"], "XS")
        self.assertEqual(out["score_metric"], "decode")


if __name__ == "__main__":
    unittest.main()
