#!/usr/bin/env python3
"""Unit tests for PR bot rendering/policy metadata.

Run from the repo root:
  python3 eval/test_pr_eval_bot.py
"""
import unittest
import json
import os
import tempfile
from unittest import mock

import pr_eval_bot as bot


class PrEvalBotPolicyTest(unittest.TestCase):
    def test_regression_labels_block_automerge(self):
        self.assertIn("regression-128", bot.AUTOMERGE_BLOCK_LABELS)
        self.assertIn("regression-512", bot.AUTOMERGE_BLOCK_LABELS)
        self.assertIn("regression-4k", bot.AUTOMERGE_BLOCK_LABELS)
        self.assertIn("regression-16k", bot.AUTOMERGE_BLOCK_LABELS)
        self.assertIn("regression-32k", bot.AUTOMERGE_BLOCK_LABELS)

    def test_mixed_win_render_keeps_eval_label_and_shows_regression(self):
        res = {
            "label": "S",
            "pass": True,
            "tps": 205.0,
            "frontier_tps": 195.0,
            "delta_tps": 10.0,
            "pct_over_frontier": 5.1,
            "top1": 0.97,
            "kl": 0.02,
            "eval_mode": "longctx",
            "score_context": 4096,
            "best_context_label": "4k-context",
            "ctx_128_tps": 470.0,
            "guard_128_baseline": 481.0,
            "guard_128_pass": False,
            "ctx_512_tps": 406.0,
            "guard_512_baseline": 405.0,
            "guard_512_pass": True,
            "ctx_4096_tps": 205.0,
            "guard_4k_baseline": 195.0,
            "guard_4k_pass": True,
            "ctx_16384_tps": 266.0,
            "guard_16k_baseline": 265.0,
            "guard_16k_pass": True,
            "ctx_32768_tps": 190.0,
            "guard_32k_baseline": 194.0,
            "guard_32k_pass": False,
            "regression_labels": ["regression-128"],
        }
        body = bot.render(res, "abc1234")
        self.assertIn("`eval:S`", body)
        self.assertIn("4096 ctx · 4k-context", body)
        self.assertIn("regression-128", body)
        self.assertIn("32k-context no-regression gate", body)
        self.assertNotIn("Auto-closing", body)

    def test_auto_close_reject_render_explains_regression_only_case(self):
        res = {
            "label": "REJECT",
            "pass": False,
            "auto_close": True,
            "reason": "512-context decode no-regression gate failed",
            "tps": 401.0,
            "frontier_tps": 405.0,
            "delta_tps": -4.0,
            "pct_over_frontier": -1.0,
            "top1": 0.97,
            "kl": 0.02,
            "eval_mode": "longctx",
            "score_context": 512,
            "best_context_label": "512-context",
            "ctx_512_tps": 401.0,
            "guard_512_baseline": 405.0,
            "guard_512_pass": False,
            "regression_labels": ["regression-512"],
        }
        body = bot.render(res, "def5678")
        self.assertIn("`eval:REJECT`", body)
        self.assertIn("regression-512", body)
        self.assertIn("Auto-closing this PR", body)

    def test_merged_4k_eval_updates_context_frontier_not_128_headline(self):
        data = {
            "updated": "2026-07-03",
            "status": {"frontier_tps": 481.24, "longctx_16k_tps": 265.17},
            "context_baselines": [
                {"ctx": 128, "label": "128", "sparkinfer_tps": 481.24, "llamacpp_decode_tps": 365.85},
                {"ctx": 512, "label": "512", "sparkinfer_tps": 405.27, "llamacpp_decode_tps": 342.59},
                {"ctx": 4096, "label": "4k", "sparkinfer_tps": 195.31, "llamacpp_decode_tps": 292.99},
                {"ctx": 16384, "label": "16k", "sparkinfer_tps": 265.17, "llamacpp_decode_tps": 245.53},
                {"ctx": 32768, "label": "32k", "sparkinfer_tps": 146.63, "llamacpp_decode_tps": 192.62},
            ],
            "prs": [{
                "num": 136,
                "title": "Enable GQA split path at 32 splits",
                "label": "XL",
                "eval_mode": "longctx",
                "score_context": 4096,
                "delta_pct": 78.53,
                "tps": 348.86,
                "ctx_128_tps": 487.45,
                "ctx_512_tps": 461.06,
                "ctx_4096_tps": 348.86,
                "ctx_16384_tps": 262.87,
                "ctx_32768_tps": 149.0,
                "guard_128_baseline": 481.59,
                "guard_512_baseline": 405.36,
                "guard_4k_baseline": 195.41,
                "guard_16k_baseline": 262.88,
                "guard_32k_baseline": 146.63,
            }],
            "landed": [],
            "landed_longctx": [],
        }
        with tempfile.TemporaryDirectory() as td:
            dash = os.path.join(td, "dashboard")
            os.mkdir(dash)
            path = os.path.join(dash, "data.json")
            with open(path, "w") as f:
                json.dump(data, f)
            with mock.patch.object(bot, "DASH", dash), \
                 mock.patch.object(bot, "DATA_JSON", path), \
                 mock.patch.object(bot, "push_dash"), \
                 mock.patch.object(bot, "append_frontier_ledger"):
                bot.record_merge("gittensor-ai-lab/sparkinfer", 136)
            with open(path) as f:
                out = json.load(f)
        rows = {r["ctx"]: r for r in out["context_baselines"]}
        self.assertEqual(out["status"]["frontier_tps"], 487.1)
        self.assertEqual(rows[4096]["sparkinfer_tps"], 348.68)
        self.assertEqual(rows[16384]["sparkinfer_tps"], 265.17)
        self.assertEqual(rows[32768]["sparkinfer_tps"], 149.0)
        self.assertEqual(out["status"]["longctx_4k_tps"], 348.68)
        self.assertEqual(out["landed_longctx"][0]["ctx"], 4096)
        self.assertFalse(out["landed"])

    def test_qwen35_ctx_uses_measured_tps_without_scaling(self):
        data = {
            "qwen35": {
                "frontier_tps": 281.63,
                "ctx": [
                    {"label": "128", "tps": 281.63, "ref_tps": 224.91},
                    {"label": "512", "tps": 277.69, "ref_tps": 225.1},
                    {"label": "4k", "tps": 264.06, "ref_tps": 224.68},
                ],
            }
        }
        sub = {
            "ctx_128_tps": 284.47,
            "ctx_512_tps": 281.34,
            "ctx_4096_tps": 267.66,
            "guard_128_baseline": 257.47,
            "guard_512_baseline": 254.27,
            "guard_4k_baseline": 242.49,
        }
        bot._upsert_qwen35_ctx(data, sub)
        by = {r["label"]: r["tps"] for r in data["qwen35"]["ctx"]}
        self.assertEqual(by["128"], 284.47)
        self.assertEqual(by["512"], 281.34)
        self.assertEqual(by["4k"], 267.66)
        # Second merge with same measured must not compound ratios.
        bot._upsert_qwen35_ctx(data, sub)
        by2 = {r["label"]: r["tps"] for r in data["qwen35"]["ctx"]}
        self.assertEqual(by2, by)


if __name__ == "__main__":
    unittest.main(verbosity=2)
