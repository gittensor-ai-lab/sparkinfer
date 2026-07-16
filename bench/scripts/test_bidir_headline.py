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


def guard_ok(guard):
    gctx = ["guard_128_pass", "guard_512_pass", "guard_4k_pass", "guard_16k_pass", "guard_32k_pass",
            "guard_64k_pass", "guard_128k_pass"]
    present = [k for k in gctx if k in guard]
    speed_ok = all(guard.get(k, True) for k in present)
    acc_ok = float(guard.get("top1", 0)) >= 0.90 and float(guard.get("kl", 99)) <= 0.20
    regressed = [k.replace("guard_", "").replace("_pass", "") for k in present if not guard.get(k, True)]
    return bool(guard) and speed_ok and acc_ok, regressed, speed_ok, acc_ok


def merge_primary(primary, guard, guard_model="Qwen3.6-35B-A3B"):
    _, regressed, speed_ok, acc_ok = guard_ok(guard)
    out = dict(primary)
    out["guard"] = {"speed_ok": speed_ok, "accuracy_ok": acc_ok}
    if not speed_ok:
        out["label"] = "REJECT"
        out["pass"] = False
        out["reason"] = "no-regression guard: speed"
    elif not acc_ok:
        out["pass"] = False
        out["reason"] = f"no-regression guard: {guard_model} accuracy broke"
    return out


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

    def test_merge_primary_guard_accuracy_fail_keeps_speed_tier(self):
        """PR #463: all decode guards pass but Qwen3.6 guard accuracy fails — keep M, pass=false."""
        primary = {"label": "M", "pass": True, "tps": 5511.53, "prefill_label": "M"}
        guard = {"top1": 0.8549, "kl": 0.0523,
                 "guard_128_pass": True, "guard_512_pass": True, "guard_4k_pass": True,
                 "guard_16k_pass": True, "guard_32k_pass": True, "guard_64k_pass": True,
                 "guard_128k_pass": True}
        out = merge_primary(primary, guard)
        self.assertEqual(out["label"], "M")
        self.assertFalse(out["pass"])
        self.assertTrue(out["guard"]["speed_ok"])
        self.assertFalse(out["guard"]["accuracy_ok"])


if __name__ == "__main__":
    unittest.main()
