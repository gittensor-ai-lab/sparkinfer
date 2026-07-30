#!/usr/bin/env python3
"""Unit tests for the enclave scoring script.

`scoring.py` is what the TDX enclave runs to turn a merged RESULT_JSON into the
signed verdict, so its gates decide every PR's reward. These pin the gate matrix
— correctness bar, no-regression guard, and the fail-closed posture on partial
or missing evidence.

Run:  python3 -m pytest eval/polaris/test_scoring.py -v
  or:  python3 eval/polaris/test_scoring.py
"""

import os
import sys
import unittest

# Ensure we can import from eval/polaris/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from eval.polaris.scoring import KL_BAR, TOP1_BAR, score


def _primary(**over):
    base = {"top1": 0.99, "kl": 0.01, "label": "M", "pass": True, "delta_pct": 3.2}
    base.update(over)
    return base


def _guard(**over):
    base = {
        "top1": 0.99,
        "kl": 0.01,
        "guard_128_pass": True,
        "guard_512_pass": True,
        "guard_4k_pass": True,
        "guard_16k_pass": True,
        "guard_32k_pass": True,
    }
    base.update(over)
    return base


class TestCorrectnessGate(unittest.TestCase):
    def test_accepts_a_clean_run(self):
        out = score({"primary": _primary(), "guard": _guard()})
        self.assertEqual(out["label"], "M")
        self.assertTrue(out["pass"])
        self.assertTrue(out["guard"]["speed_ok"])
        self.assertTrue(out["guard"]["accuracy_ok"])

    def test_empty_primary_fails_closed(self):
        out = score({"primary": {}, "guard": _guard()})
        self.assertEqual(out["label"], "REJECT")
        self.assertIn("infra error", out["reason"])

    def test_top1_below_bar_is_rejected(self):
        out = score({"primary": _primary(top1=TOP1_BAR - 0.01), "guard": _guard()})
        self.assertEqual(out["label"], "REJECT")
        self.assertIn("top1=", out["reason"])

    def test_kl_above_bar_is_rejected(self):
        out = score({"primary": _primary(kl=KL_BAR + 0.01), "guard": _guard()})
        self.assertEqual(out["label"], "REJECT")
        self.assertIn("kl=", out["reason"])


class TestNoRegressionGuard(unittest.TestCase):
    def test_regressed_context_is_rejected_and_named(self):
        out = score({"primary": _primary(), "guard": _guard(guard_4k_pass=False)})
        self.assertEqual(out["label"], "REJECT")
        self.assertIn("4k", out["reason"])
        self.assertIn("regression-qwen3-4k", out["guard_regression_labels"])

    def test_missing_guard_fails_closed(self):
        out = score({"primary": _primary(), "guard": {}})
        self.assertEqual(out["label"], "REJECT")

    def test_guard_accuracy_break_is_rejected(self):
        out = score({"primary": _primary(), "guard": _guard(top1=0.10)})
        self.assertEqual(out["label"], "REJECT")
        self.assertIn("accuracy broke", out["reason"])

    def test_zero_measured_contexts_fails_closed(self):
        """A guard with accuracy but no per-context speed flags proves nothing.

        `all([])` is vacuously true, so this used to report speed_ok=True and
        pass the no-regression guard on zero speed evidence — inside the enclave
        that signs the receipt.
        """
        guard = {"top1": 0.99, "kl": 0.01}  # accuracy pass ran, speed pass did not

        out = score({"primary": _primary(), "guard": guard})

        self.assertFalse(out["guard"]["speed_ok"])
        self.assertEqual(out["label"], "REJECT")
        self.assertFalse(out["pass"])
        self.assertIn("measured no decode context", out["reason"])

    def test_a_single_measured_context_is_enough(self):
        """The fix must not require all five contexts — only that one was measured."""
        guard = {"top1": 0.99, "kl": 0.01, "guard_4k_pass": True}

        out = score({"primary": _primary(), "guard": guard})

        self.assertTrue(out["guard"]["speed_ok"])
        self.assertEqual(out["label"], "M")


class TestVerdictShape(unittest.TestCase):
    def test_primary_metrics_are_carried_verbatim(self):
        out = score({"primary": _primary(delta_pct=7.5), "guard": _guard()})
        self.assertEqual(out["delta_pct"], 7.5)

    def test_flat_result_without_primary_key_is_accepted(self):
        # `primary = result.get("primary", result)` — a bare verdict still scores.
        out = score(dict(_primary(), guard=_guard()))
        self.assertEqual(out["label"], "M")


if __name__ == "__main__":
    unittest.main()
