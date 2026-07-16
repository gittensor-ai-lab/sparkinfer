#!/usr/bin/env python3
"""Bidir headline pick: REJECT must beat none when one optimize fails.

Run from repo root:
  python3 bench/scripts/test_bidir_headline.py
"""
import unittest

TIER_RANK = {"XL": 6, "L": 5, "M": 4, "S": 3, "XS": 2, "none": 1, "BASELINE": 0, "REJECT": -1}


def pick_best(a, b):
    ra, rb = TIER_RANK.get(a.get("label"), -1), TIER_RANK.get(b.get("label"), -1)
    if ra != rb:
        return a if ra > rb else b
    return a if float(a.get("tps") or 0) >= float(b.get("tps") or 0) else b


def pick_headline(a, b):
    if not a.get("pass") or not b.get("pass"):
        for s in (a, b):
            if not s.get("pass") and s.get("label") == "REJECT":
                return dict(s)
        for s in (a, b):
            if not s.get("pass"):
                return dict(s)
    return pick_best(a, b)


class BidirHeadlineTest(unittest.TestCase):
    def test_reject_beats_none(self):
        q35 = {"label": "REJECT", "pass": False, "tps": 298.1, "reason": "prefill regressed"}
        q36 = {"label": "none", "pass": True, "tps": 441.98}
        out = pick_headline(q35, q36)
        self.assertEqual(out["label"], "REJECT")
        self.assertFalse(out["pass"])

    def test_both_pass_picks_best_tier(self):
        q35 = {"label": "none", "pass": True, "tps": 283.0}
        q36 = {"label": "L", "pass": True, "tps": 480.0}
        out = pick_headline(q35, q36)
        self.assertEqual(out["label"], "L")

    def test_headline_reject_when_other_side_rejects(self):
        """Qwen3.5 keeps speed tier M but pass=false; headline follows Qwen3.6 REJECT."""
        q35 = {"label": "M", "pass": False, "tps": 5511.53,
               "reason": "no-regression guard: Qwen3.6 accuracy broke"}
        q36 = {"label": "REJECT", "pass": False, "tps": 482.53}
        out = pick_headline(q35, q36)
        self.assertEqual(out["label"], "REJECT")


if __name__ == "__main__":
    unittest.main()
