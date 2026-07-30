#!/usr/bin/env python3
"""Unit tests for DFlash PR eval bot policy."""
import importlib
import os
import unittest
from unittest import mock

import pr_dflash_bot as dfb


class DFlashEvalBotTest(unittest.TestCase):
    def test_bench_tokens_default_512(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            mod = importlib.reload(dfb)
            self.assertEqual(mod.BENCH_TOKENS, 512)

    def test_bench_tokens_env_override(self):
        with mock.patch.dict(os.environ, {"DFLASH_BENCH_TOKENS": "256"}):
            mod = importlib.reload(dfb)
            self.assertEqual(mod.BENCH_TOKENS, 256)
        importlib.reload(dfb)

    def test_tier_from_gain_xs_at_two_percent(self):
        lab, pct, ok, _ = dfb.tier_from_gain(306.0, 300.0)
        self.assertEqual(lab, "XS")
        self.assertGreaterEqual(pct, 2.0)
        self.assertTrue(ok)

    def test_tier_from_gain_none_below_gate(self):
        lab, pct, ok, reason = dfb.tier_from_gain(306.0, 307.0)
        self.assertEqual(lab, "none")
        self.assertLess(pct, 2.0)
        self.assertTrue(ok)
        self.assertIn("significance", reason)

    def test_format_comment_includes_bench_tokens(self):
        body = dfb.format_comment("abc123", {
            "ok": True,
            "label": "XS",
            "delta_pct": 2.5,
            "pr_dflash_tps": 310.0,
            "main_dflash_tps": 302.0,
            "pr_ar_tps": 280.0,
            "speedup_vs_main": 1.025,
            "speedup_vs_ar": 1.1,
            "mean_accept": 4.5,
            "spec_agree": "METRIC SPEC_AGREE 32/32 = 1.0000",
            "reason": "ok",
            "bench_tokens": 512,
        })
        self.assertIn("bench gen tokens | 512", body)
        self.assertIn('"bench_tokens":512', body)


if __name__ == "__main__":
    unittest.main()
