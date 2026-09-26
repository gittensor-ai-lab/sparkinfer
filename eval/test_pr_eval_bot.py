#!/usr/bin/env python3
"""Unit tests for PR bot rendering/policy metadata.

Run from the repo root:
  python3 eval/test_pr_eval_bot.py
"""
import unittest
import json
import os
import datetime
import tempfile
from unittest import mock

import pr_eval_bot as bot

# Nothing here may touch the controller's own state: every file the bot writes goes to a temp dir.
import atexit as _atexit
import shutil as _shutil
_STATE = tempfile.mkdtemp(prefix="sparkinfer-bot-tests-")
_atexit.register(_shutil.rmtree, _STATE, True)
for _n in ("INSTANCE_FILE", "PIN_FILE", "BOT_LOCK_FILE"):
    setattr(bot, _n, os.path.join(_STATE, _n))
bot.PINNED_INSTANCE = ""
import pr_museglimmer_bot as _muse   # imported by some tests below: its state is redirected too
for _n in ("STRIKES_FILE", "SCORES_FILE"):
    setattr(_muse, _n, os.path.join(_STATE, "muse." + _n))


class PrEvalBotPolicyTest(unittest.TestCase):
    def test_merge_conflict_blocks_eval(self):
        self.assertTrue(bot.pr_merge_conflict("CONFLICTING"))
        self.assertFalse(bot.pr_merge_conflict("MERGEABLE"))
        self.assertFalse(bot.pr_merge_conflict("UNKNOWN"))
        self.assertFalse(bot.pr_merge_conflict(None))

    _TEMPLATE_DECODE = """
- [x] Tested on **RTX 5090** (`sm_120`)
| | decode tok/s |
|---|--:|
| before (main) | {db} |
| after (this PR) | {da} |
"""
    _TEMPLATE_PREFILL = """
- [x] Tested on **RTX 5090** (`sm_120`)
| | prefill pp tok/s |
|---|--:|
| before prefill (main) | {pb} |
| after prefill (this PR) | {pa} |
"""
    _TEMPLATE_BOTH = """
- [x] Tested on **RTX 5090** (`sm_120`)
| | decode tok/s |
|---|--:|
| before (main) | {db} |
| after (this PR) | {da} |
| | prefill pp tok/s |
|---|--:|
| before prefill (main) | {pb} |
| after prefill (this PR) | {pa} |
"""

    def _greenlight(self, body):
        with mock.patch.object(bot, "gh", return_value=mock.Mock(returncode=0, stdout=json.dumps({"body": body}))):
            return bot.greenlight_status("gittensor-ai-lab/sparkinfer", 1, set())

    def test_greenlight_decode_only(self):
        status, reason = self._greenlight(self._TEMPLATE_DECODE.format(db=300, da=320))
        self.assertEqual(status, "ok")
        self.assertIn("decode 300.0→320.0", reason)

    def test_greenlight_prefill_only(self):
        status, reason = self._greenlight(self._TEMPLATE_PREFILL.format(pb=250, pa=280))
        self.assertEqual(status, "ok")
        self.assertIn("prefill 250.0→280.0", reason)
        self.assertIn("pp tok/s", reason)

    def test_greenlight_prefill_skips_decode_flat(self):
        body = self._TEMPLATE_BOTH.format(db=300, da=300, pb=250, pa=275)
        status, reason = self._greenlight(body)
        self.assertEqual(status, "ok")
        self.assertIn("prefill", reason)

    def test_greenlight_no_bench_when_flat(self):
        status, _ = self._greenlight(self._TEMPLATE_DECODE.format(db=300, da=300))
        self.assertEqual(status, "no-bench")

    def test_greenlight_unchecked_box(self):
        body = self._TEMPLATE_DECODE.format(db=300, da=320).replace("[x]", "[ ]")
        status, reason = self._greenlight(body)
        self.assertEqual(status, "unchecked")
        self.assertIn("unchecked", reason)

    def test_rtx5090_box_checked(self):
        ticked = self._TEMPLATE_DECODE.format(db=300, da=320)
        self.assertTrue(bot.rtx5090_box_checked(ticked))
        self.assertFalse(bot.rtx5090_box_checked(ticked.replace("[x]", "[ ]")))
        self.assertFalse(bot.rtx5090_box_checked("- [x] Tested on RTX 4090"))
        self.assertFalse(bot.rtx5090_box_checked(""))

    def test_rtx5090_has_checkbox_and_should_close(self):
        ticked = self._TEMPLATE_DECODE.format(db=300, da=320)
        unchecked = ticked.replace("[x]", "[ ]")
        self.assertTrue(bot.rtx5090_has_checkbox(unchecked))
        self.assertTrue(bot.rtx5090_has_checkbox(ticked))
        self.assertFalse(bot.rtx5090_has_checkbox("docs-only change, no proof section"))
        self.assertTrue(bot.rtx5090_should_close(unchecked))
        self.assertFalse(bot.rtx5090_should_close(ticked))
        self.assertFalse(bot.rtx5090_should_close("no template checkbox here"))

    def test_decode_val_ignores_prefill_rows(self):
        body = self._TEMPLATE_BOTH.format(db=301, da=310, pb=250, pa=260)
        self.assertEqual(bot._decode_val(body, "before"), 301.0)
        self.assertEqual(bot._prefill_val(body, "before"), 250.0)

    def test_regression_labels_block_automerge(self):
        self.assertIn("regression-128", bot.AUTOMERGE_BLOCK_LABELS)
        self.assertIn("regression-512", bot.AUTOMERGE_BLOCK_LABELS)
        self.assertIn("regression-4k", bot.AUTOMERGE_BLOCK_LABELS)
        self.assertIn("regression-16k", bot.AUTOMERGE_BLOCK_LABELS)
        self.assertIn("regression-32k", bot.AUTOMERGE_BLOCK_LABELS)
        self.assertIn("regression-4k-pp", bot.AUTOMERGE_BLOCK_LABELS)

    def test_bidir_qwen35_prefill_render(self):
        res = {
            "mode": "bidir",
            "label": "M",
            "pass": True,
            "eval_mode": "longctx",
            "label_qwen35": "M",
            "label_qwen36": "none",
            "pass_qwen35": True,
            "pass_qwen36": True,
            "score_qwen35": {
                "label": "M",
                "pass": True,
                "tps": 140.0,
                "frontier_tps": 126.0,
                "pct_over_frontier": 11.1,
                "delta_tps": 14.0,
                "top1": 0.97,
                "kl": 0.02,
                "score_context": 4096,
                "best_context_label": "4k-context",
                "eval_prefill": True,
                "prefill_label": "S",
                "prefill_tps": 4200.0,
                "score_prefill_context": 32768,
                "best_prefill_context_label": "32k-context",
                "ctx_4096_pp_tps": 4100.0,
                "ctx_32768_pp_tps": 4200.0,
                "guard_4k_pp_baseline": 4000.0,
                "guard_4k_pp_pass": True,
                "guard_32k_pp_baseline": 3900.0,
                "guard_32k_pp_pass": True,
            },
            "score_qwen36": {"label": "none", "pass": True, "tps": 300.0, "top1": 0.97, "kl": 0.02},
        }
        body = bot.render(res, "abc1234")
        self.assertIn("scored prefill", body)
        self.assertIn("`eval-prefill:S`", body)
        self.assertIn("4k prefill no-regression gate", body)

    def test_bidir_qwen35_zero_prefill_render(self):
        res = {
            "mode": "bidir",
            "label": "REJECT",
            "pass": False,
            "eval_mode": "longctx",
            "label_qwen35": "REJECT",
            "label_qwen36": "REJECT",
            "pass_qwen35": False,
            "pass_qwen36": False,
            "score_qwen35": {
                "label": "REJECT",
                "pass": False,
                "tps": 287.55,
                "frontier_tps": 285.7,
                "top1": 0.897,
                "kl": 0.0425,
                "score_context": 4096,
                "best_context_label": "4k-context",
                "eval_prefill": True,
                "ctx_4096_pp_tps": 0.0,
                "ctx_32768_pp_tps": 0.0,
                "ctx_65536_pp_tps": 0.0,
                "ctx_131072_pp_tps": 0.0,
                "guard_4k_pp_baseline": 289.26,
                "guard_4k_pp_pass": False,
                "guard_32k_pp_baseline": 285.42,
                "guard_32k_pp_pass": False,
            },
            "score_qwen36": {"label": "REJECT", "pass": False, "tps": 488.0, "top1": 0.92, "kl": 0.04},
        }
        body = bot.render(res, "943f58a")
        self.assertIn("not measured (0 pp tok/s on all contexts)", body)
        self.assertIn("4k prefill no-regression gate | 0.0 pp tok/s vs main 289.26 pp tok/s · fail", body)

    def test_bidir_prefill_render_without_label_shows_measured_pp(self):
        res = {
            "mode": "bidir",
            "label": "none",
            "pass": True,
            "eval_mode": "longctx",
            "label_qwen35": "none",
            "label_qwen36": "none",
            "pass_qwen35": True,
            "pass_qwen36": True,
            "score_qwen35": {
                "label": "none",
                "pass": True,
                "tps": 283.24,
                "frontier_tps": 283.16,
                "top1": 0.903,
                "kl": 0.0417,
                "score_context": 65536,
                "best_context_label": "64k-context",
                "eval_prefill": True,
                "score_prefill_context": 4096,
                "best_prefill_context_label": "4k-context",
                "ctx_4096_pp_tps": 4150.42,
                "ctx_32768_pp_tps": 2109.42,
                "guard_4k_pp_baseline": 320.45,
                "guard_4k_pp_pass": True,
            },
            "score_qwen36": {"label": "none", "pass": True, "tps": 473.14, "top1": 0.927, "kl": 0.0404},
        }
        body = bot.render(res, "9786172")
        self.assertIn("scored prefill (4096 ctx · 4k-context) | 4150.42 pp tok/s", body)
        self.assertNotIn("not measured (0 pp tok/s on all contexts)", body)

    def test_bidir_optimize_rows_use_scored_model_not_guard(self):
        q35 = "Qwythos-9B (Q4_K_M)"
        q36 = "Qwen3.6-35B-A3B"
        res = {
            "mode": "bidir",
            "label": "REJECT",
            "pass": False,
            "label_qwen35": "REJECT",
            "label_qwen36": "REJECT",
            "pass_qwen35": False,
            "pass_qwen36": False,
            "score_qwen35": {
                "label": "REJECT",
                "pass": False,
                "model": q35,
                "guard_model": q36,
                "tps": 283.6,
                "top1": 0.922,
                "kl": 0.0407,
                "ctx_128_tps": 295.0,
                "ctx_65536_tps": 283.45,
                "guard_128_pass": True,
                "guard_64k_pass": True,
                "guard": {
                    "top1": 0.927,
                    "kl": 0.0457,
                    "accuracy_ok": True,
                    "ctx_128_tps": 465.24,
                    "ctx_65536_tps": 403.7,
                    "guard_128_pass": True,
                    "guard_64k_pass": False,
                },
            },
            "score_qwen36": {
                "label": "REJECT",
                "pass": False,
                "model": q36,
                "guard_model": q35,
                "tps": 441.47,
                "top1": 0.927,
                "kl": 0.0457,
                "ctx_128_tps": 465.15,
                "ctx_32768_tps": 403.75,
                "guard_128_pass": True,
                "guard_32k_pass": False,
                "guard": {
                    "top1": 0.922,
                    "kl": 0.0407,
                    "accuracy_ok": True,
                    "ctx_128_tps": 294.79,
                    "ctx_65536_tps": 283.16,
                    "guard_128_pass": True,
                    "guard_64k_pass": True,
                },
            },
        }
        body = bot.render(res, "99ae7d5")
        self.assertIn(f"Qwen3.5 optimize — {q36} guard accuracy", body)
        self.assertIn(f"Qwen3.6 optimize — {q35} guard accuracy", body)
        self.assertIn(f"Qwen3.5 optimize — {q35} 128 | 295.0 tok/s", body)
        self.assertIn(f"Qwen3.5 optimize — {q35} 64k | 283.45 tok/s", body)
        self.assertIn(f"Qwen3.6 optimize — {q36} 128 | 465.15 tok/s", body)
        self.assertIn(f"Qwen3.6 optimize — {q36} 32k | 403.75 tok/s", body)
        self.assertNotIn(f"Qwen3.5 optimize — {q36} 128 | 465.24", body)
        self.assertNotIn(f"Qwen3.6 optimize — {q35} 64k | 283.16", body)

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
                    {"label": "4k", "tps": 264.06, "ref_tps": 224.68},
                    {"label": "32k", "tps": 200.0, "ref_tps": 0},
                ],
            }
        }
        sub = {
            "ctx_128_tps": 284.47,
            "ctx_4096_tps": 267.66,
            "ctx_32768_tps": 205.5,
            "guard_128_baseline": 257.47,
            "guard_4k_baseline": 242.49,
            "guard_32k_baseline": 198.0,
        }
        bot._upsert_qwen35_ctx(data, sub)
        by = {r["label"]: r["tps"] for r in data["qwen35"]["ctx"]}
        self.assertEqual(by["128"], 284.47)
        self.assertEqual(by["4k"], 267.66)
        self.assertEqual(by["32k"], 205.5)
        # Second merge with same measured must not compound ratios.
        bot._upsert_qwen35_ctx(data, sub)
        by2 = {r["label"]: r["tps"] for r in data["qwen35"]["ctx"]}
        self.assertEqual(by2, by)

    def test_qwen35_pp_uses_measured_pp_without_scaling(self):
        data = {
            "qwen35": {
                "prefill_frontier_pp": 290.57,
                "pp": [
                    {"label": "4k", "pp": 290.57, "ref_pp": 11104.62},
                    {"label": "32k", "pp": 272.07, "ref_pp": 9772.31},
                ],
            }
        }
        sub = {
            "ctx_4096_pp_tps": 295.0,
            "ctx_32768_pp_tps": 278.5,
            "ctx_65536_pp_tps": 260.0,
            "ctx_131072_pp_tps": 230.0,
            "prefill_tps": 295.0,
            "prefill_label": "M",
        }
        bot._upsert_qwen35_pp(data, sub)
        by = {r["label"]: r["pp"] for r in data["qwen35"]["pp"]}
        self.assertEqual(by["4k"], 295.0)
        self.assertEqual(by["32k"], 278.5)
        self.assertEqual(by["64k"], 260.0)
        self.assertEqual(by["128k"], 230.0)
        self.assertEqual(data["qwen35"]["prefill_frontier_pp"], 295.0)
        self.assertEqual(data["qwen35"]["prefill_label"], "M")
        bot._upsert_qwen35_pp(data, sub)
        by2 = {r["label"]: r["pp"] for r in data["qwen35"]["pp"]}
        self.assertEqual(by2, by)

    def test_qwen35_journey_tps_prefers_ctx_128(self):
        self.assertEqual(bot._qwen35_journey_tps({"tps": 283.28, "ctx_128_tps": 300.43}), 300.43)

    def test_rebuild_qwen35_journey(self):
        data = {
            "prs": [
                {"num": 323, "title": "perf(qwen35): first", "pass_qwen35": True, "label_qwen35": "S",
                 "score_qwen35": {"ctx_128_tps": 271.85, "guard_128_baseline": 256.95, "tps": 271.85}},
                {"num": 379, "title": "perf(attn): long", "pass_qwen35": True, "label_qwen35": "XL",
                 "score_qwen35": {"ctx_128_tps": 300.43, "tps": 283.28, "guard_128_baseline": 301.01}},
            ],
            "landed_qwen35": [
                {"pr": 323, "tps": 271.85, "name": "first", "date": "2026-07-10"},
                {"pr": 379, "tps": 298.27, "name": "wrong", "date": "2026-07-14"},
            ],
            "qwen35": {"frontier_tps": 298.27, "baseline_tps": 298.27},
        }
        bot._rebuild_qwen35_journey(data)
        self.assertEqual(data["qwen35"]["baseline_tps"], 256.95)
        self.assertEqual(data["qwen35"]["frontier_tps"], 300.43)
        self.assertEqual([m["tps"] for m in data["landed_qwen35"]], [271.85, 300.43])

    def test_rebuild_qwen35_journey_ratchet_monotonic(self):
        data = {
            "prs": [
                {"num": 324, "title": "perf(qwen35): b", "pass_qwen35": True, "label_qwen35": "M",
                 "score_qwen35": {"ctx_128_tps": 281.63, "guard_128_baseline": 257.47}},
                {"num": 326, "title": "perf(qwen35): c", "pass_qwen35": True, "label_qwen35": "XS",
                 "score_qwen35": {"ctx_128_tps": 272.63, "guard_128_baseline": 268.84}},
                {"num": 329, "title": "perf(qwen35): d", "pass_qwen35": True, "label_qwen35": "M",
                 "score_qwen35": {"ctx_128_tps": 303.18, "guard_128_baseline": 283.18}},
            ],
            "landed_qwen35": [],
            "qwen35": {},
        }
        bot._rebuild_qwen35_journey(data)
        self.assertEqual([m["tps"] for m in data["landed_qwen35"]], [281.63, 281.63, 303.18])
        self.assertEqual(data["landed_qwen35"][1].get("raw_tps"), 272.63)

    def test_rebuild_qwen35_pp_journey(self):
        data = {
            "prs": [
                {"num": 387, "title": "perf(qwen35): prefill graph", "pass_qwen35": True, "label_qwen35": "L",
                 "score_qwen35": {"prefill_tps": 320.33, "frontier_tps": 288.16, "eval_prefill": True}},
                {"num": 398, "title": "perf(qwen35): batched prefill", "pass_qwen35": True, "label_qwen35": "XL",
                 "score_qwen35": {"prefill_tps": 4150.42, "frontier_tps": 320.45, "eval_prefill": True}},
                {"num": 422, "title": "perf(qwen35): int8 GEMM", "pass_qwen35": True, "label_qwen35": "XL",
                 "score_qwen35": {"prefill_tps": 6096.4, "frontier_tps": 4179.68, "eval_prefill": True}},
            ],
            "landed_qwen35_pp": [],
            "qwen35": {},
        }
        bot._rebuild_qwen35_pp_journey(data)
        self.assertEqual(data["qwen35"]["baseline_pp"], 288.16)
        self.assertEqual(data["qwen35"]["prefill_frontier_pp"], 6096.4)
        self.assertEqual([m["tps"] for m in data["landed_qwen35_pp"]], [320.33, 4150.42, 6096.4])

    def test_qwen36_ctx_uses_measured_tps_without_scaling(self):
        data = {
            "qwen36": {
                "frontier_tps": 372.04,
                "ctx": [
                    {"label": "128", "tps": 423.77, "ref_tps": 275.81},
                    {"label": "512", "tps": 420.23, "ref_tps": 275.61},
                    {"label": "4k", "tps": 403.22, "ref_tps": 276.3},
                    {"label": "16k", "tps": 378.74, "ref_tps": 280.66},
                    {"label": "32k", "tps": 372.04, "ref_tps": 279.83},
                ],
            }
        }
        sub = {
            "ctx_128_tps": 411.95,
            "ctx_512_tps": 418.05,
            "ctx_4096_tps": 402.52,
            "ctx_16384_tps": 398.58,
            "ctx_32768_tps": 382.25,
        }
        bot._upsert_qwen36_ctx(data, sub)
        by = {r["label"]: r["tps"] for r in data["qwen36"]["ctx"]}
        self.assertEqual(by["128"], 423.77)
        self.assertEqual(by["512"], 420.23)
        self.assertEqual(by["4k"], 403.22)
        self.assertEqual(by["16k"], 398.58)
        self.assertEqual(by["32k"], 382.25)

    def test_polaris_tdx_falls_back_to_ed25519(self):
        from eval.polaris.receipt import generate_keypair, verify_attestation

        priv, _ = generate_keypair()
        att = {
            "code": {"commit": "abc1234"},
            "references": {"model_sha256": "deadbeef", "eval_seed": "seed1"},
            "measurements": {"tps": 100, "label": "S"},
        }
        with mock.patch("eval.polaris.client.PolarisClient") as mock_client_cls:
            mock_client_cls.return_value.attest_scoring.side_effect = RuntimeError("HTTP 404")
            receipt = bot.build_polaris_receipt_from_attestation(
                att, api_key="pi_sk_test", privkey=priv, pubkey="dGVzdA==")
        self.assertIsNotNone(receipt.get("signature"))
        self.assertNotIn("tdx", receipt)
        self.assertTrue(verify_attestation(att, receipt["signature"], receipt["public_key"]))

    def test_polaris_ed25519_only_when_no_api_key(self):
        from eval.polaris.receipt import generate_keypair

        priv, _ = generate_keypair()
        att = {
            "code": {"commit": "def5678"},
            "references": {"model_sha256": "cafebabe", "eval_seed": "seed2"},
            "measurements": {"tps": 200, "label": "M"},
        }
        receipt = bot.build_polaris_receipt_from_attestation(att, api_key="", privkey=priv)
        self.assertIsNotNone(receipt.get("signature"))
        self.assertNotIn("tdx", receipt)

    def test_merge_recorded_bidir_qwen36(self):
        data = {
            "prs": [{"num": 353, "mode": "bidir", "pass_qwen36": True, "label_qwen36": "XL"}],
            "landed_qwen36": [{"pr": 353, "tps": 427.54}],
            "landed_qwen35": [],
        }
        e = data["prs"][0]
        self.assertTrue(bot._merge_recorded(data, 353, e))
        self.assertFalse(bot._merge_recorded(data, 999, {"label": "XL"}))

    def test_sync_merged_dashboard_records_manual_merge(self):
        data = {
            "updated": "2026-07-12",
            "status": {"frontier_tps": 400.0},
            "qwen36": {"frontier_tps": 400.0, "baseline_tps": 23.0, "ctx": []},
            "prs": [{
                "num": 353,
                "title": "perf(qwen36): test",
                "mode": "bidir",
                "pass_qwen36": True,
                "label_qwen36": "XL",
                "label": "XL",
                "tps": 427.54,
                "score_qwen36": {
                    "tps": 427.54,
                    "top1": 0.97,
                    "kl": 0.02,
                    "ctx_128_tps": 427.54,
                    "ctx_512_tps": 420.0,
                    "ctx_4096_tps": 410.0,
                    "ctx_16384_tps": 390.0,
                    "ctx_32768_tps": 380.0,
                },
            }],
            "landed_qwen36": [],
            "landed_qwen35": [],
        }
        with tempfile.TemporaryDirectory() as td:
            dash = os.path.join(td, "dashboard")
            os.mkdir(dash)
            path = os.path.join(dash, "data.json")
            with open(path, "w") as f:
                json.dump(data, f)
            gh_out = json.dumps([{"number": 353}])
            pushes = []
            with mock.patch.object(bot, "DASH", dash), \
                 mock.patch.object(bot, "DATA_JSON", path), \
                 mock.patch.object(bot, "gh", return_value=mock.Mock(stdout=gh_out)), \
                 mock.patch.object(bot, "push_dash", side_effect=lambda m: pushes.append(m)):
                bot.sync_merged_dashboard("gittensor-ai-lab/sparkinfer")
            with open(path) as f:
                out = json.load(f)
        self.assertEqual(out["qwen36"]["frontier_tps"], 427.54)
        self.assertEqual(out["landed_qwen36"][0]["pr"], 353)
        self.assertTrue(any("merged" in m for m in pushes))

    def test_sync_merged_dashboard_skips_already_recorded(self):
        data = {
            "prs": [{"num": 353, "mode": "bidir", "pass_qwen36": True, "label_qwen36": "XL",
                     "score_qwen36": {"tps": 427.54}}],
            "landed_qwen36": [{"pr": 353, "tps": 427.54}],
        }
        with mock.patch.object(bot, "load_dash", return_value=data), \
             mock.patch.object(bot, "gh", return_value=mock.Mock(stdout=json.dumps([{"number": 353}]))), \
             mock.patch.object(bot, "record_merge") as rm:
            bot.sync_merged_dashboard("gittensor-ai-lab/sparkinfer")
        rm.assert_not_called()

    def test_qwen36_journey_tps_prefers_128_ctx(self):
        sub = {"tps": 456.42, "ctx_128_tps": 463.27}
        self.assertEqual(bot._qwen36_journey_tps(sub), 463.27)

    def test_pr_inactive_days_from_updated_at(self):
        now = datetime.datetime(2026, 7, 13, 12, 0, tzinfo=datetime.timezone.utc)
        pr = {"updatedAt": "2026-07-10T12:00:00Z"}
        self.assertAlmostEqual(bot.pr_inactive_days(pr, now), 3.0, places=5)

    def test_close_stale_prs_closes_inactive(self):
        stale = {
            "number": 42,
            "title": "old PR",
            "updatedAt": "2026-07-01T00:00:00Z",
            "labels": [{"name": "not-tested"}],
        }
        fresh = {
            "number": 43,
            "title": "active PR",
            "updatedAt": "2026-07-12T00:00:00Z",
            "labels": [],
        }
        gh_calls = []

        def fake_gh(args):
            gh_calls.append(args)
            if args[:3] == ["pr", "list", "-R"]:
                return mock.Mock(stdout=json.dumps([stale, fresh]))
            return mock.Mock(returncode=0)

        now = datetime.datetime(2026, 7, 13, 0, 0, tzinfo=datetime.timezone.utc)
        with mock.patch.object(bot, "gh", side_effect=fake_gh), \
             mock.patch.object(bot, "pr_inactive_days", side_effect=lambda pr, _now=None: 5.0 if pr["number"] == 42 else 1.0):
            closed = bot.close_stale_prs("gittensor-ai-lab/sparkinfer", days=2, dry_run=False)
        self.assertEqual(closed, {42})
        self.assertTrue(any(c[:3] == ["pr", "close", "42"] for c in gh_calls))

    def test_close_stale_prs_skips_hold_and_merge_first(self):
        prs = [
            {"number": 1, "updatedAt": "2026-01-01T00:00:00Z", "labels": [{"name": "hold"}]},
            {"number": 2, "updatedAt": "2026-01-01T00:00:00Z", "labels": [{"name": "merge-first"}]},
        ]
        with mock.patch.object(bot, "gh", return_value=mock.Mock(stdout=json.dumps(prs))), \
             mock.patch.object(bot, "pr_inactive_days", return_value=10.0):
            closed = bot.close_stale_prs("gittensor-ai-lab/sparkinfer", days=2)
        self.assertEqual(closed, set())

    def test_close_stale_prs_dry_run(self):
        prs = [{"number": 99, "updatedAt": "2026-01-01T00:00:00Z", "labels": [], "isDraft": False}]
        gh_mock = mock.Mock(return_value=mock.Mock(stdout=json.dumps(prs)))
        with mock.patch.object(bot, "gh", gh_mock), \
             mock.patch.object(bot, "pr_inactive_days", return_value=10.0):
            closed = bot.close_stale_prs("gittensor-ai-lab/sparkinfer", days=2, dry_run=True)
        self.assertEqual(closed, {99})
        gh_mock.assert_called_once()

    def test_close_stale_prs_skips_drafts_when_non_draft_only(self):
        prs = [
            {"number": 50, "updatedAt": "2026-01-01T00:00:00Z", "labels": [], "isDraft": True},
            {"number": 51, "updatedAt": "2026-01-01T00:00:00Z", "labels": [], "isDraft": False},
        ]
        with mock.patch.object(bot, "gh", return_value=mock.Mock(stdout=json.dumps(prs))), \
             mock.patch.object(bot, "pr_inactive_days", return_value=10.0):
            closed = bot.close_stale_prs("gittensor-ai-lab/sparkinfer", days=2,
                                         dry_run=True, drafts_only=False)
        self.assertEqual(closed, {51})

    def test_pr_draft_days_from_created_at(self):
        now = datetime.datetime(2026, 7, 16, 12, 0, tzinfo=datetime.timezone.utc)
        pr = {"number": 1, "createdAt": "2026-07-10T12:00:00Z", "isDraft": True}
        with mock.patch.object(bot, "pr_draft_since", return_value=bot._parse_github_time("2026-07-10T12:00:00Z")):
            self.assertAlmostEqual(bot.pr_draft_days("r/o", pr, now), 6.0, places=5)

    def test_pr_draft_since_uses_latest_convert(self):
        pr = {"number": 2, "createdAt": "2026-07-01T00:00:00Z", "isDraft": True}
        timeline = json.dumps([
            {"event": "converted_to_draft", "created_at": "2026-07-10T00:00:00Z"},
            {"event": "ready_for_review", "created_at": "2026-07-12T00:00:00Z"},
            {"event": "converted_to_draft", "created_at": "2026-07-14T00:00:00Z"},
        ])
        with mock.patch.object(bot, "gh", return_value=mock.Mock(returncode=0, stdout=timeline)):
            since = bot.pr_draft_since("gittensor-ai-lab/sparkinfer", pr)
        self.assertEqual(since.isoformat(), "2026-07-14T00:00:00+00:00")

    def test_close_stale_draft_prs_closes_old_drafts(self):
        prs = [{"number": 60, "createdAt": "2026-01-01T00:00:00Z", "labels": [], "isDraft": True}]
        gh_calls = []

        def fake_gh(args):
            gh_calls.append(args)
            if args[:3] == ["pr", "list", "-R"]:
                return mock.Mock(stdout=json.dumps(prs))
            return mock.Mock(returncode=0)

        with mock.patch.object(bot, "gh", side_effect=fake_gh), \
             mock.patch.object(bot, "pr_draft_days", return_value=10.0):
            closed = bot.close_stale_draft_prs("gittensor-ai-lab/sparkinfer", days=4, dry_run=False)
        self.assertEqual(closed, {60})
        self.assertTrue(any(c[:3] == ["pr", "close", "60"] for c in gh_calls))

    def test_close_stale_draft_prs_ignores_recent_activity(self):
        """Draft age is not reset by updatedAt — only time in draft status matters."""
        prs = [{"number": 61, "createdAt": "2026-01-01T00:00:00Z",
                "updatedAt": "2026-07-16T00:00:00Z", "labels": [], "isDraft": True}]
        with mock.patch.object(bot, "gh", return_value=mock.Mock(stdout=json.dumps(prs))), \
             mock.patch.object(bot, "pr_draft_days", return_value=10.0):
            closed = bot.close_stale_draft_prs("gittensor-ai-lab/sparkinfer", days=4, dry_run=True)
        self.assertEqual(closed, {61})

    def test_close_stale_draft_prs_skips_hold(self):
        prs = [{"number": 70, "createdAt": "2026-01-01T00:00:00Z",
                "labels": [{"name": "hold"}], "isDraft": True}]
        with mock.patch.object(bot, "gh", return_value=mock.Mock(stdout=json.dumps(prs))), \
             mock.patch.object(bot, "pr_draft_days", return_value=10.0):
            closed = bot.close_stale_draft_prs("gittensor-ai-lab/sparkinfer", days=4)
        self.assertEqual(closed, set())

    def _unchecked_pr(self, num, body=None, labels=None, **extra):
        body = body or self._TEMPLATE_DECODE.format(db=300, da=320).replace("[x]", "[ ]")
        pr = {
            "number": num,
            "title": f"PR {num}",
            "labels": [{"name": n} for n in (labels or [])],
            "isDraft": False,
            "author": {"login": "contrib"},
            "authorAssociation": "CONTRIBUTOR",
        }
        pr.update(extra)
        return pr, body

    def test_close_unchecked_closes_unticked_checkbox(self):
        pr, body = self._unchecked_pr(10)
        gh_calls = []

        def fake_gh(args):
            gh_calls.append(args)
            if args[:3] == ["pr", "list", "-R"]:
                return mock.Mock(stdout=json.dumps([pr]))
            if args[:4] == ["pr", "view", "10", "-R"]:
                return mock.Mock(returncode=0, stdout=json.dumps({"body": body}))
            return mock.Mock(returncode=0)

        with mock.patch.object(bot, "gh", side_effect=fake_gh), \
             mock.patch.object(bot, "areas_for_pr", return_value=set()), \
             mock.patch.object(bot, "close_rtx5090_unchecked_pr") as close_one:
            closed = bot.close_unchecked_rtx5090_prs("gittensor-ai-lab/sparkinfer")
        self.assertEqual(closed, {10})
        close_one.assert_called_once_with("gittensor-ai-lab/sparkinfer", 10, runtime=False)

    def test_close_unchecked_legacy_not_tested_label(self):
        ticked = self._TEMPLATE_DECODE.format(db=300, da=320)
        pr, body = self._unchecked_pr(11, body=ticked, labels=["not-tested"])

        def fake_gh(args):
            if args[:3] == ["pr", "list", "-R"]:
                return mock.Mock(stdout=json.dumps([pr]))
            if args[:4] == ["pr", "view", "11", "-R"]:
                return mock.Mock(returncode=0, stdout=json.dumps({"body": body}))
            return mock.Mock(returncode=0)

        with mock.patch.object(bot, "gh", side_effect=fake_gh), \
             mock.patch.object(bot, "areas_for_pr", return_value=set()), \
             mock.patch.object(bot, "close_rtx5090_unchecked_pr") as close_one:
            closed = bot.close_unchecked_rtx5090_prs("gittensor-ai-lab/sparkinfer")
        self.assertEqual(closed, {11})
        close_one.assert_called_once_with("gittensor-ai-lab/sparkinfer", 11, runtime=False)

    def test_close_unchecked_closes_runtime_without_checkbox(self):
        pr, body = self._unchecked_pr(30, body="runtime correctness fix — no proof section")

        def fake_gh(args):
            if args[:3] == ["pr", "list", "-R"]:
                return mock.Mock(stdout=json.dumps([pr]))
            if args[:4] == ["pr", "view", "30", "-R"]:
                return mock.Mock(returncode=0, stdout=json.dumps({"body": body}))
            return mock.Mock(returncode=0)

        with mock.patch.object(bot, "gh", side_effect=fake_gh), \
             mock.patch.object(bot, "areas_for_pr", return_value={"runtime"}), \
             mock.patch.object(bot, "close_rtx5090_unchecked_pr") as close_one:
            closed = bot.close_unchecked_rtx5090_prs("gittensor-ai-lab/sparkinfer")
        self.assertEqual(closed, {30})
        close_one.assert_called_once_with("gittensor-ai-lab/sparkinfer", 30, runtime=True)

    def test_rtx5090_should_close_runtime_without_checkbox(self):
        self.assertTrue(bot.rtx5090_should_close("no checkbox", {"runtime"}))
        self.assertFalse(bot.rtx5090_should_close("no checkbox", {"kernels"}))
        self.assertFalse(bot.rtx5090_should_close("docs only", set()))

    def test_close_unchecked_skips_exempt_and_no_checkbox(self):
        unchecked_body = self._TEMPLATE_DECODE.format(db=300, da=320).replace("[x]", "[ ]")
        prs = [
            self._unchecked_pr(20, isDraft=True)[0],
            self._unchecked_pr(21, labels=["hold"])[0],
            self._unchecked_pr(22, author={"login": "ai-hpc"})[0],
            self._unchecked_pr(23, authorAssociation="MEMBER")[0],
            self._unchecked_pr(24, body="docs-only, no checkbox")[0],
            self._unchecked_pr(25)[0],
        ]

        def fake_gh(args):
            if args[:3] == ["pr", "list", "-R"]:
                return mock.Mock(stdout=json.dumps(prs))
            num = args[2]
            if args[:4] == ["pr", "view", num, "-R"]:
                pr = next(p for p in prs if str(p["number"]) == num)
                body = unchecked_body if pr["number"] == 25 else "docs-only"
                return mock.Mock(returncode=0, stdout=json.dumps({"body": body}))
            return mock.Mock(returncode=0)

        with mock.patch.object(bot, "gh", side_effect=fake_gh), \
             mock.patch.object(bot, "areas_for_pr", return_value=set()), \
             mock.patch.object(bot, "close_rtx5090_unchecked_pr") as close_one:
            closed = bot.close_unchecked_rtx5090_prs("gittensor-ai-lab/sparkinfer", dry_run=True)
        self.assertEqual(closed, {25})
        close_one.assert_not_called()

    def test_not_tested_not_in_automerge_block(self):
        self.assertNotIn(bot.NOT_TESTED_LABEL, bot.AUTOMERGE_BLOCK_LABELS)

    def test_evaluated_commit_from_comment_accepts_verdict(self):
        body = bot.render({"label": "S", "pass": True, "tps": 200.0, "top1": 1.0, "kl": 0.0}, "df74674")
        self.assertEqual(bot._evaluated_commit_from_comment(body), "df74674")

    def test_infra_error_public_label_and_render(self):
        res = {
            "label": "REJECT", "pass": False, "infra_error": True, "mode": "bidir",
            "reason": "infra error: accuracy check produced no METRIC (ModuleNotFoundError: No module named 'tokenizers')",
            "score_qwen35": {"tps": 0, "top1": 0, "kl": 99, "label": "REJECT", "pass": False, "infra_error": True},
            "score_qwen36": {"tps": 0, "top1": 0, "kl": 99, "label": "REJECT", "pass": False, "infra_error": True},
            "label_qwen35": "REJECT", "label_qwen36": "REJECT",
        }
        self.assertEqual(bot._public_eval_label(res), "infra-error")
        body = bot.render(res, "e2d829a")
        self.assertIn("`eval:infra-error`", body)
        self.assertIn("infra error (not graded)", body)
        self.assertNotIn("eval-qwen35:REJECT", body)
        self.assertNotIn("Qwen3.5 optimize", body)
        self.assertIn("tokenizers", body)

    def test_bidir_public_label_reject_beats_passing_xl(self):
        """PR #555: Qwen3.5 REJECT must headline over Qwen3.6 XL."""
        res = {
            "mode": "bidir", "label": "XL", "pass": True,
            "label_qwen35": "REJECT", "pass_qwen35": False,
            "label_qwen36": "XL", "pass_qwen36": True,
            "score_qwen35": {"label": "REJECT", "pass": False, "tps": 283.06},
            "score_qwen36": {"label": "XL", "pass": True, "tps": 3388.88},
        }
        self.assertEqual(bot._public_eval_label(res), "REJECT")
        body = bot.render(res, "dc22645")
        self.assertIn("`eval:REJECT`", body)
        self.assertIn("eval-qwen35:REJECT", body)
        self.assertIn("eval-qwen36:XL", body)

    def test_none_reject_eval_count_ignores_infra_error(self):
        infra_body = bot.render({
            "label": "REJECT", "pass": False, "infra_error": True, "mode": "bidir",
            "reason": "infra error: missing tokenizers",
            "score_qwen35": {}, "score_qwen36": {},
        }, "abc1234")
        comments = [{"body": infra_body}]
        gh_mock = mock.Mock(return_value=mock.Mock(stdout=json.dumps({"comments": comments})))
        with mock.patch.object(bot, "gh", gh_mock):
            self.assertEqual(bot.none_reject_eval_count("gittensor-ai-lab/sparkinfer", 531), 0)

    def test_evaluated_commit_from_comment_rejects_error_marker(self):
        body = ("<!-- sparkinfer-eval:df74674 -->\n"
                "⚠️ **sparkinfer auto-eval errored** for `df74674` — re-run manually.")
        self.assertIsNone(bot._evaluated_commit_from_comment(body))

    def test_evaluated_commit_from_comment_rejects_error_marker_v2(self):
        body = ("<!-- sparkinfer-eval-error:df74674 -->\n"
                "⚠️ **sparkinfer auto-eval errored** for `df74674` — re-run manually.")
        self.assertIsNone(bot._evaluated_commit_from_comment(body))

    def test_evaluated_commits_ignores_errored_comments(self):
        comments = [
            {"body": "<!-- sparkinfer-eval:df74674 -->\n⚠️ **sparkinfer auto-eval errored**"},
            {"body": bot.render({"label": "REJECT", "pass": False, "reason": "x",
                                 "tps": 0, "top1": 0, "kl": 0}, "abc1234")},
        ]
        gh_mock = mock.Mock(return_value=mock.Mock(stdout=json.dumps({"comments": comments})))
        with mock.patch.object(bot, "gh", gh_mock):
            done = bot.evaluated_commits("gittensor-ai-lab/sparkinfer", 379)
        self.assertEqual(done, {"abc1234"})


class OnlyPrsAndBaselineCacheTest(unittest.TestCase):
    def test_parse_only_prs(self):
        self.assertEqual(bot._parse_only_prs(387, ""), {387})
        self.assertEqual(bot._parse_only_prs("531,511,530", ""), {531, 511, 530})
        self.assertEqual(bot._parse_only_prs(0, "387, 389"), {387, 389})
        self.assertEqual(bot._parse_only_prs(387, "389"), {387, 389})

    def test_fill_pp_allows_partial_q36_when_128_pp_zero(self):
        """Qwen3.6 baseline often has ctx_128_pp_tps=0; must still keep 512/4k/16k/32k pp."""
        q36 = {"128_pp": 0.0, "512_pp": 0.0, "4k_pp": 0.0, "16k_pp": 0.0, "32k_pp": 0.0,
               "128": 0, "512": 0, "4k": 0, "16k": 0, "32k": 0}
        q35 = {"128": 0, "4k": 0, "32k": 0, "64k": 0,
               "4k_pp": 0.0, "32k_pp": 0.0, "64k_pp": 0.0, "128k_pp": 0.0}
        bres = {
            "score_qwen36": {
                "ctx_128_tps": 475.97, "ctx_512_tps": 469.85, "ctx_4096_tps": 450.96,
                "ctx_16384_tps": 434.42, "ctx_32768_tps": 406.66,
                "ctx_128_pp_tps": 0.0, "ctx_512_pp_tps": 534.53,
                "ctx_4096_pp_tps": 521.72, "ctx_16384_pp_tps": 504.37,
                "ctx_32768_pp_tps": 484.09,
            },
            "score_qwen35": {
                "ctx_128_tps": 293.92, "ctx_4096_tps": 283.93,
                "ctx_32768_tps": 282.28, "ctx_65536_tps": 282.28,
                "ctx_4096_pp_tps": 15665.6, "ctx_32768_pp_tps": 17280.6,
                "ctx_65536_pp_tps": 17432.4, "ctx_131072_pp_tps": 287.3,
            },
        }
        self.assertTrue(bot._apply_bidir_ctx_from_bres(bres, q36, q35))
        self.assertEqual(q36["128_pp"], 0.0)
        self.assertAlmostEqual(q36["512_pp"], 534.53)
        self.assertAlmostEqual(q36["4k_pp"], 521.72)
        self.assertAlmostEqual(q36["32k_pp"], 484.09)
        self.assertAlmostEqual(q35["4k_pp"], 15665.6)

    def test_baseline_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            cache_path = os.path.join(td, ".baseline_cache.json")
            with mock.patch.object(bot, "BASELINE_CACHE_FILE", cache_path):
                q36 = {"128": 200.0, "512": 195.0}
                q35 = {"128": 150.0, "4k": 140.0}
                bres = {"pass": True, "score_qwen36": 200.0}
                bot._save_baseline_cache("ssh:host:22", q36, q35, bres)
                loaded = bot._load_baseline_cache("ssh:host:22")
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded["q36"], q36)
                self.assertEqual(loaded["q35"], q35)
                self.assertEqual(loaded["bres"], bres)
                self.assertIsNone(bot._load_baseline_cache("ssh:other:22"))

    def test_baseline_cache_valid_rejects_stale_main(self):
        cache = {"bres": {"commit": "abc1234", "pass": True, "tps": 300.0,
                          "score_qwen36": {"ctx_128_tps": 300, "ctx_512_tps": 290,
                                           "ctx_4096_tps": 280, "ctx_16384_tps": 270,
                                           "ctx_32768_tps": 260},
                          "score_qwen35": {"ctx_128_tps": 200, "ctx_4096_tps": 190,
                                           "ctx_32768_tps": 180, "ctx_65536_tps": 170}}}
        q36, q35 = {"128": 0}, {"128": 0}
        with mock.patch.object(bot, "_bench_harness_changed_between", return_value=True):
            self.assertFalse(bot._baseline_cache_valid(cache, True, q36, q35, "def5678"))

    def test_baseline_cache_valid_keeps_cache_when_only_eval_changed(self):
        cache = {"bres": {"commit": "abc1234", "pass": True, "tps": 300.0,
                          "score_qwen36": {"ctx_128_tps": 300, "ctx_512_tps": 290,
                                           "ctx_4096_tps": 280, "ctx_16384_tps": 270,
                                           "ctx_32768_tps": 260},
                          "score_qwen35": {"ctx_128_tps": 200, "ctx_4096_tps": 190,
                                           "ctx_32768_tps": 180, "ctx_65536_tps": 170,
                                           "ctx_4096_pp_tps": 100, "ctx_32768_pp_tps": 90,
                                           "ctx_65536_pp_tps": 80}}}
        q36, q35 = {"128": 0}, {"128": 0}
        with mock.patch.object(bot, "_bench_harness_changed_between", return_value=False):
            self.assertTrue(bot._baseline_cache_valid(cache, True, q36, q35, "def5678"))

    def test_baseline_cache_valid_accepts_matching_main(self):
        bres = {
            "commit": "abc1234", "pass": True,
            "score_qwen36": {"ctx_128_tps": 300, "ctx_512_tps": 290, "ctx_4096_tps": 280,
                             "ctx_16384_tps": 270, "ctx_32768_tps": 260},
            "score_qwen35": {"ctx_128_tps": 200, "ctx_4096_tps": 190, "ctx_32768_tps": 180,
                             "ctx_65536_tps": 170,
                             "ctx_4096_pp_tps": 100, "ctx_32768_pp_tps": 90, "ctx_65536_pp_tps": 80},
        }
        cache = {"bres": bres}
        q36, q35 = {"128": 0}, {"128": 0}
        self.assertTrue(bot._baseline_cache_valid(cache, True, q36, q35, "abc1234"))


class ExhaustedEvalCloseTest(unittest.TestCase):
    def test_eval_verdict_from_comment(self):
        none_body = bot.render({"label": "none", "pass": True, "tps": 300.0, "top1": 0.97, "kl": 0.02,
                                "frontier_tps": 298.0}, "abc1234")
        self.assertEqual(bot._eval_verdict_from_comment(none_body), "none")
        reject_body = bot.render({"label": "REJECT", "pass": False, "reason": "regression",
                                  "tps": 280.0, "top1": 0.97, "kl": 0.02, "frontier_tps": 300.0},
                                 "def5678")
        self.assertEqual(bot._eval_verdict_from_comment(reject_body), "REJECT")
        pass_body = bot.render({"label": "S", "pass": True, "tps": 320.0, "top1": 0.97, "kl": 0.02,
                                "frontier_tps": 300.0, "pct_over_frontier": 6.7, "delta_tps": 20.0},
                               "fed9012")
        self.assertEqual(bot._eval_verdict_from_comment(pass_body), "S")
        self.assertIsNone(bot._eval_verdict_from_comment("<!-- sparkinfer-eval:abc -->\nno verdict"))

    def test_none_reject_eval_count(self):
        comments = [
            {"body": bot.render({"label": "none", "pass": True, "tps": 300.0, "top1": 0.97, "kl": 0.02,
                                 "frontier_tps": 298.0}, "aaa1111")},
            {"body": bot.render({"label": "REJECT", "pass": False, "reason": "x", "tps": 0,
                                 "top1": 0, "kl": 0, "frontier_tps": 300.0}, "bbb2222")},
            {"body": bot.render({"label": "M", "pass": True, "tps": 330.0, "top1": 0.97, "kl": 0.02,
                                 "frontier_tps": 300.0, "pct_over_frontier": 10.0, "delta_tps": 30.0},
                                "ccc3333")},
        ]
        gh_mock = mock.Mock(return_value=mock.Mock(stdout=json.dumps({"comments": comments})))
        with mock.patch.object(bot, "gh", gh_mock):
            self.assertEqual(bot.none_reject_eval_count("gittensor-ai-lab/sparkinfer", 42), 2)

    def test_close_exhausted_eval_prs(self):
        prs = [
            {"number": 10, "title": "ok", "labels": [], "isDraft": False},
            {"number": 11, "title": "exhausted", "labels": [], "isDraft": False},
            {"number": 12, "title": "hold", "labels": [{"name": "hold"}], "isDraft": False},
        ]
        list_mock = mock.Mock(return_value=mock.Mock(stdout=json.dumps(prs)))
        close_mock = mock.Mock(return_value=mock.Mock(returncode=0))
        comment_mock = mock.Mock(return_value=mock.Mock(returncode=0))

        def gh_side_effect(args, *a, **kw):
            if len(args) >= 2 and args[0] == "pr" and args[1] == "list":
                return list_mock(*a, **kw)
            if len(args) >= 2 and args[0] == "pr" and args[1] == "close":
                return close_mock(*a, **kw)
            if len(args) >= 2 and args[0] == "pr" and args[1] == "comment":
                return comment_mock(*a, **kw)
            return mock.Mock(stdout=json.dumps({"comments": []}))

        with mock.patch.object(bot, "gh", side_effect=gh_side_effect), \
             mock.patch.object(bot, "none_reject_eval_count", side_effect=lambda _r, n: {10: 2, 11: 3, 12: 5}[n]):
            closed = bot.close_exhausted_eval_prs("gittensor-ai-lab/sparkinfer", max_none_reject=2)
        self.assertEqual(closed, {11})
        close_mock.assert_called_once()

    def test_run_poll_auto_closes(self):
        with mock.patch.object(bot, "close_stale_prs", return_value={1}), \
             mock.patch.object(bot, "close_stale_draft_prs", return_value={4}), \
             mock.patch.object(bot, "close_unchecked_rtx5090_prs", return_value={2}), \
             mock.patch.object(bot, "close_exhausted_eval_prs", return_value={3}):
            closed = bot.run_poll_auto_closes("gittensor-ai-lab/sparkinfer")
        self.assertEqual(closed, {1, 2, 3, 4})

    def test_maybe_close_exhausted_pr(self):
        view_mock = mock.Mock(return_value=mock.Mock(
            stdout=json.dumps({"labels": [], "isDraft": False})))
        close_mock = mock.Mock(return_value=mock.Mock(returncode=0))
        comment_mock = mock.Mock(return_value=mock.Mock(returncode=0))

        def gh_side_effect(args, *a, **kw):
            if len(args) >= 2 and args[0] == "pr" and args[1] == "view":
                return view_mock(*a, **kw)
            if len(args) >= 2 and args[0] == "pr" and args[1] == "close":
                return close_mock(*a, **kw)
            if len(args) >= 2 and args[0] == "pr" and args[1] == "comment":
                return comment_mock(*a, **kw)
            return mock.Mock(stdout="{}")

        with mock.patch.object(bot, "gh", side_effect=gh_side_effect), \
             mock.patch.object(bot, "none_reject_eval_count", return_value=3):
            self.assertTrue(bot.maybe_close_exhausted_pr("gittensor-ai-lab/sparkinfer", 99))
        close_mock.assert_called_once()



class GenericEvalLabelTest(unittest.TestCase):
    """sync_generic_eval_label(): the generic eval:* tier SN74 reads is derived from the per-bot
    eval-<model>:* labels, so staggered bots cannot clobber each other's verdicts."""

    def _run(self, labels):
        """Returns (chosen_tier, final_label_set) after running the sync against `labels`."""
        state = set(labels)
        with mock.patch.object(bot, "labels_on", return_value=set(state)), \
             mock.patch.object(bot, "add_label", side_effect=lambda r, n, l: state.add(l)), \
             mock.patch.object(bot, "remove_label", side_effect=lambda r, n, l: state.discard(l)):
            got = bot.sync_generic_eval_label("o/r", 1)
        return got, state

    def test_one_bots_none_cannot_erase_anothers_tier(self):
        # The #1018 case: a Muse PR scored L by the Muse bot, then benched by the DSpark bot
        # against a model it does not touch. Whichever cron ran last used to win.
        got, state = self._run({"eval-museglimmer:L", "eval-dspark:none", "eval:L"})
        self.assertEqual(got, "L")
        self.assertIn("eval:L", state)
        self.assertNotIn("eval:none", state)

    def test_order_independent(self):
        # Same two verdicts, whichever bot happens to write last -> same generic label.
        a, _ = self._run({"eval-museglimmer:L", "eval-dspark:none", "eval:none"})
        b, _ = self._run({"eval-dspark:none", "eval-museglimmer:L", "eval:L"})
        self.assertEqual(a, b)
        self.assertEqual(a, "L")

    def test_best_tier_wins_across_bots(self):
        got, _ = self._run({"eval-museglimmer:S", "eval-dspark:XL", "eval-qwen38:none"})
        self.assertEqual(got, "XL")

    def test_reject_beats_any_positive_tier(self):
        # A regression on ANY model must not advertise a positive tier from another.
        got, state = self._run({"eval-museglimmer:XL", "eval-dspark:REJECT"})
        self.assertEqual(got, "REJECT")
        self.assertIn("eval:REJECT", state)
        self.assertNotIn("eval:XL", state)

    def test_a_bot_can_still_lower_its_own_score(self):
        # Re-evaluation after a push: sole bot drops XL -> S, generic must follow it down.
        got, _ = self._run({"eval-museglimmer:S", "eval:XL"})
        self.assertEqual(got, "S")

    def test_all_none_stays_none(self):
        got, _ = self._run({"eval-museglimmer:none", "eval-dspark:none"})
        self.assertEqual(got, "none")

    def test_no_per_bot_labels_leaves_generic_alone(self):
        got, state = self._run({"eval:L", "hold"})
        self.assertIsNone(got)
        self.assertIn("eval:L", state)

    def test_generic_label_is_not_mistaken_for_a_per_bot_label(self):
        # `eval:XL` must not feed back into its own computation.
        got, _ = self._run({"eval:XL", "eval-dspark:none"})
        self.assertEqual(got, "none")

    def test_unknown_tier_ignored(self):
        got, _ = self._run({"eval-dspark:bogus", "eval-museglimmer:M"})
        self.assertEqual(got, "M")




class DeclaredTargetModelTest(unittest.TestCase):
    """declared_models()/model_skip_reason(): a bot may skip a PR the author says targets a
    different model, but ONLY on an explicit declaration -- everything else evaluates."""

    TICKED = ("- [x] Muse Glimmer\n"
              "- [ ] Qwen3.8-27B (ModelOpt NVFP4 / DSpark)\n"
              "- [ ] Shared / both\n")

    def test_parses_ticked_model(self):
        self.assertEqual(bot.declared_models(self.TICKED), {"muse"})

    def test_dspark_skips_a_muse_only_pr(self):
        # The #1025 case: perf(muse) burned a full DSpark round to score none at +0.1%.
        why = bot.model_skip_reason(self.TICKED, "qwen38")
        self.assertIsNotNone(why)
        self.assertIn("muse", why)

    def test_muse_bot_would_not_skip_its_own_pr(self):
        self.assertIsNone(bot.model_skip_reason(self.TICKED, "muse"))

    def test_shared_ticked_runs_everywhere(self):
        body = ("- [x] Muse Glimmer\n- [ ] Qwen3.8-27B (ModelOpt NVFP4 / DSpark)\n"
                "- [x] Shared / both\n")
        self.assertIsNone(bot.model_skip_reason(body, "qwen38"))

    def test_both_models_ticked_runs_everywhere(self):
        body = "- [x] Muse Glimmer\n- [x] Qwen3.8-27B (ModelOpt NVFP4 / DSpark)\n"
        self.assertIsNone(bot.model_skip_reason(body, "qwen38"))
        self.assertIsNone(bot.model_skip_reason(body, "muse"))

    # --- fail-open: none of these may ever cause a skip ---

    def test_no_declaration_evaluates(self):
        self.assertIsNone(bot.model_skip_reason("## Summary\nmakes it faster", "qwen38"))

    def test_empty_body_evaluates(self):
        self.assertIsNone(bot.model_skip_reason("", "qwen38"))
        self.assertIsNone(bot.model_skip_reason(None, "qwen38"))

    def test_unticked_boxes_evaluate(self):
        body = "- [ ] Muse Glimmer\n- [ ] Qwen3.8-27B (ModelOpt NVFP4 / DSpark)\n"
        self.assertIsNone(bot.model_skip_reason(body, "qwen38"))

    def test_unrecognised_model_name_evaluates(self):
        self.assertIsNone(bot.model_skip_reason("- [x] Some Future Model\n", "qwen38"))

    def test_rtx5090_attestation_box_is_not_a_model(self):
        # The attestation checkbox must never be read as a target declaration.
        body = "- [x] Tested on **RTX 5090** (`sm_120`)\n"
        self.assertEqual(bot.declared_models(body), set())
        self.assertIsNone(bot.model_skip_reason(body, "qwen38"))

    def test_dspark_and_modelopt_aliases_map_to_qwen38(self):
        for alias in ("- [x] DSpark\n", "- [x] ModelOpt NVFP4\n", "- [x] Qwen3.8-27B\n"):
            self.assertEqual(bot.declared_models(alias), {"qwen38"}, alias)
            self.assertIsNone(bot.model_skip_reason(alias, "qwen38"), alias)



class MuseBotHonoursDeclaredModelTests(unittest.TestCase):
    """#1082 declared Qwen3.8-27B only, and the Muse Glimmer bot evaluated and auto-closed it
    before the Qwen3.8 bot that scores its axis could poll it."""

    QWEN38_ONLY = (
        "**Target model(s)**\n\n"
        "- [ ] **Muse Glimmer**\n"
        "- [x] **Qwen3.8-27B** (ModelOpt NVFP4 / DSpark)\n"
        "- [ ] **Shared / both**\n"
    )

    def test_a_qwen38_only_pr_is_skipped_for_muse(self):
        self.assertTrue(bot.model_skip_reason(self.QWEN38_ONLY, "muse"))
        self.assertIsNone(bot.model_skip_reason(self.QWEN38_ONLY, "qwen38"))

    def test_the_muse_bot_checks_the_declaration_before_evaluating(self):
        import inspect
        import pr_museglimmer_bot as muse
        src = inspect.getsource(muse.main)
        self.assertIn('arb.model_skip_reason(pr.get("body") or "", "muse")', src)
        # Before the greenlight, i.e. before any PR is queued for GPU time.
        self.assertLess(src.index("model_skip_reason"), src.index("greenlight_status"))


class MuseBotUnslothGuardTests(unittest.TestCase):
    """pr_qwen38_bot.py skips PRs declared for Muse Glimmer alone, so the Muse bot must guard the
    unsloth Qwen3.8 checkpoint that bot scores."""

    def test_remote_script_guards_the_unsloth_checkpoint_before_the_end_marker(self):
        import pr_museglimmer_bot as muse
        script = muse._remote_script("main")
        self.assertIn('bench_sweep_run "$UNSLOTH_GUARD_MODEL_DIR" 128 32768 5', script)
        for marker in ("GUARDUN $ctx", "GUARDUN_FAILED", "GUARDUN_UNAVAILABLE"):
            self.assertIn(marker, script)
        self.assertLess(script.index("GUARDUN_UNAVAILABLE"), script.index('echo "GUARD_END"'))
        self.assertIn("unsloth", muse.EVAL_SCHEMA_VERSION)

    def test_unsloth_guard_parses_and_fails_on_regression(self):
        import pr_museglimmer_bot as muse
        main = muse._parse_remote("GUARDUN 32768 80.6 8727.0\n")
        self.assertEqual(main["guardun"], {32768: {"decode": 80.6, "prefill": 8727.0}})
        self.assertEqual(muse.check_unsloth_guard(main, main), (True, []))
        pr = muse._parse_remote("GUARDUN 32768 70.0 8727.0\n")
        ok, problems = muse.check_unsloth_guard(pr, main)
        self.assertFalse(ok)
        self.assertIn("unsloth qwen3.8 decode@32k", problems[0])
        self.assertTrue(muse._parse_remote("GUARDUN_UNAVAILABLE\n")["guardun_unavailable"])

class SharedBotHelperTests(unittest.TestCase):
    """Helpers the model bots share (pr_bonsai_bot, pr_qwen38_bot, pr_museglimmer_bot)."""

    def test_a_hung_gh_call_times_out_and_only_a_read_is_retried(self):
        import subprocess
        for args, calls in ((["pr", "view", "1"], 3), (["pr", "comment", "1", "--body", "x"], 1),
                            (["pr", "merge", "1", "--squash"], 1), (["pr", "close", "1"], 1),
                            (["api", "repos/o/r/commits/main", "--jq", ".sha"], 3),
                            (["api", "-X", "DELETE", "repos/o/r/issues/1/labels/x"], 1)):
            with self.subTest(args[:2]), \
                    mock.patch.object(bot.subprocess, "run", side_effect=subprocess.TimeoutExpired("gh", 1)) as r, \
                    mock.patch.object(bot.time, "sleep"):
                out = bot.gh(args, retries=3, quiet=True)
                self.assertEqual(out.returncode, 124)
                # A comment, close or merge that timed out may still have gone through: not repeated.
                self.assertEqual(r.call_count, calls)
                self.assertEqual(r.call_args.kwargs.get("timeout"), bot.GH_TIMEOUT_S)

    def test_current_main_sha_is_a_full_sha_or_nothing(self):
        R = lambda out: mock.Mock(returncode=0, stdout=out, stderr="")
        with mock.patch.object(bot, "gh", return_value=R("a" * 40 + "\n")):
            self.assertEqual(bot.current_main_sha("o/r"), "a" * 40)
        for junk in ("", "Not Found", "[]", "a" * 39):
            with mock.patch.object(bot, "gh", return_value=R(junk)):
                self.assertEqual(bot.current_main_sha("o/r"), "", junk)

    def test_scored_against_stale_main(self):
        self.assertFalse(bot.scored_against_stale_main({"onto": "a" * 40}, "a" * 40))
        self.assertTrue(bot.scored_against_stale_main({"onto": "a" * 40}, "b" * 40))
        self.assertTrue(bot.scored_against_stale_main({}, "a" * 40))            # predates "onto"
        self.assertTrue(bot.scored_against_stale_main({"onto": "a" * 40}, ""))   # main unknown

    def test_only_trusted_verdicts_count_toward_the_exhausted_close(self):
        verdict = self._ar_verdict("none")
        comments = {"comments": [{"body": verdict, "authorAssociation": "NONE"}] * 3
                    + [{"body": verdict, "authorAssociation": "MEMBER"}]}
        with mock.patch.object(bot, "gh", return_value=mock.Mock(returncode=0, stdout=json.dumps(comments))):
            self.assertEqual(bot.none_reject_eval_count("o/r", 1), 1)

    def test_no_bots_merge_first_is_closed_as_exhausted(self):
        prs = [{"number": n, "title": "t", "isDraft": False, "labels": [{"name": l}] if l else []}
               for n, l in ((1, None), (2, "bonsai-merge-first"), (3, "qwen38-merge-first"), (4, "hold"))]
        with mock.patch.object(bot, "gh", return_value=mock.Mock(returncode=0, stdout=json.dumps(prs))), \
                mock.patch.object(bot, "none_reject_eval_count", return_value=9):
            self.assertEqual(bot.close_exhausted_eval_prs("o/r", dry_run=True), {1})

    def _ar_verdict(self, tier):
        # The comment shape _eval_verdict_from_comment reads.
        body = (f"<!-- sparkinfer-eval:{'a' * 40} -->\n## sparkinfer auto-eval — PR\n\n"
                f"| **label** | `eval:{tier}` |")
        self.assertEqual(bot._eval_verdict_from_comment(body), tier)
        return body


class RoundGuardTests(unittest.TestCase):
    """arb.round_guard_sh, run for real against processes in a temp round directory."""

    def _script(self, d, bot_name="t"):
        return bot.round_guard_sh(bot_name).replace("/tmp/sparkinfer-bot-rounds", d)

    def _start(self, d, name, new_group):
        import subprocess
        p = subprocess.Popen(["bash", "-c", "sleep 300 & sleep 300; wait"], start_new_session=new_group)
        with open(f"/proc/{p.pid}/stat") as fh:
            start = fh.read().split()[21]
        with open(os.path.join(d, name + ".pid"), "w") as fh:
            fh.write(f"{p.pid} {start}\n")
        return p

    def test_an_orphaned_round_is_stopped_and_this_round_recorded(self):
        import subprocess
        import time as _t
        with tempfile.TemporaryDirectory() as d:
            orphan = self._start(d, "old", new_group=True)
            with open(os.path.join(d, "reused.pid"), "w") as fh:     # a pid reused by something else
                fh.write(f"{os.getpid()} 1\n")
            with open(os.path.join(d, "gone.pid"), "w") as fh:
                fh.write("999999999 1\n")
            r = subprocess.run(["bash", "-c", self._script(d)], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            for _ in range(50):
                if orphan.poll() is not None:
                    break
                _t.sleep(0.1)
            self.assertIsNotNone(orphan.poll(), "the orphaned round is still running")
            self.assertIn("reaping a previous round", r.stderr)
            self.assertEqual(sorted(os.listdir(d)), ["t.pid"])        # only this round's record
            os.kill(os.getpid(), 0)                                   # the reused pid was left alone


class Round2SharedHelperTests(unittest.TestCase):
    """Fixes from the second review of the 2026-09-26 change."""

    def test_dropping_the_last_per_bot_tier_drops_the_generic_one_it_fed(self):
        labels = {"eval-bonsai:XL", "eval:XL"}
        with mock.patch.object(bot, "labels_on", side_effect=lambda r, n: set(labels)), \
                mock.patch.object(bot, "remove_label", side_effect=lambda r, n, l: labels.discard(l)), \
                mock.patch.object(bot, "add_label", side_effect=lambda r, n, l: labels.add(l)):
            self.assertTrue(bot.strip_stale_verdict_labels(
                "o/r", 1, set(labels), "eval-bonsai:", "b" * 40, {"a" * 40}))
        self.assertEqual(labels, set())
        # Another bot's tier still there: the generic label is re-derived from it, not dropped.
        labels = {"eval-bonsai:XL", "eval-qwen38:S", "eval:XL"}
        with mock.patch.object(bot, "labels_on", side_effect=lambda r, n: set(labels)), \
                mock.patch.object(bot, "remove_label", side_effect=lambda r, n, l: labels.discard(l)), \
                mock.patch.object(bot, "add_label", side_effect=lambda r, n, l: labels.add(l)):
            bot.strip_stale_verdict_labels("o/r", 1, set(labels), "eval-bonsai:", "b" * 40, {"a" * 40})
        self.assertEqual(labels, {"eval-qwen38:S", "eval:S"})
        # Unknown verdicts (GitHub did not answer) never strip.
        self.assertFalse(bot.strip_stale_verdict_labels("o/r", 1, {"eval-bonsai:XL"}, "eval-bonsai:", "b" * 40, None))

    def test_a_failed_comments_read_is_unknown_not_empty(self):
        import re as _re
        rx = _re.compile(r"<!-- sparkinfer-t-eval:v1:([0-9a-f]+)(?:\s+(\{.*?\}))? -->")
        marker = '<!-- sparkinfer-t-eval:v1:' + "a" * 40 + ' {"label":"XL"} -->\n## sparkinfer t auto-eval'
        R = lambda out, rc=0: mock.Mock(returncode=rc, stdout=out)
        good = {"comments": [{"body": marker, "authorAssociation": "MEMBER"},
                             {"body": marker.replace("a" * 40, "b" * 40), "authorAssociation": "NONE"}]}
        with mock.patch.object(bot, "gh", return_value=R(json.dumps(good))):
            self.assertEqual(bot.evaluated_commits_from("o/r", 1, rx, "sparkinfer t auto-eval"), {"a" * 40})
        for bad in (R("", 1), R("not json"), R(json.dumps({"x": 1}))):
            with mock.patch.object(bot, "gh", return_value=bad):
                self.assertIsNone(bot.evaluated_commits_from("o/r", 1, rx, "sparkinfer t auto-eval"))

    def test_only_a_pr_the_bot_will_measure_is_waiting_on_it(self):
        pr = {"number": 1, "headRefOid": "a" * 40, "labels": [], "mergeable": "MERGEABLE",
              "files": [{"path": "kernels/x.cu"}]}
        with mock.patch.object(bot, "greenlight_status", return_value=("ok", "x")):
            self.assertTrue(bot.waiting_for_first_verdict("o/r", pr, set()))
            self.assertTrue(bot.waiting_for_first_verdict("o/r", pr, None))            # unknown: kept
            self.assertFalse(bot.waiting_for_first_verdict("o/r", pr, {"a" * 40}))
            self.assertFalse(bot.waiting_for_first_verdict("o/r", dict(pr, mergeable="CONFLICTING"), set()))
            self.assertFalse(bot.waiting_for_first_verdict("o/r", pr, set(), never_paths=("kernels/",)))

    def test_the_daily_stale_close_spares_bot_winners_and_queued_prs(self):
        old = "2026-01-01T00:00:00Z"
        prs = [{"number": n, "title": "t", "updatedAt": old, "isDraft": False, "headRefOid": "a" * 40,
                "mergeable": "MERGEABLE", "labels": [{"name": l} for l in labs]}
               for n, labs in ((1, []), (2, ["bonsai-merge-first"]), (3, ["qwen38-merge-first"]), (4, []))]
        verdict = {"comments": [{"authorAssociation": "MEMBER",
                                 "body": '<!-- sparkinfer-bonsai-eval:v1:' + "a" * 40 + ' {"label":"none"} -->'}]}

        def fake_gh(a):
            if a[:2] == ["pr", "list"]:
                return mock.Mock(returncode=0, stdout=json.dumps(prs))
            if a[:2] == ["pr", "view"]:
                return mock.Mock(returncode=0, stdout=json.dumps(verdict if a[2] == "1" else {"comments": []}))
            return mock.Mock(returncode=0, stdout="")
        with mock.patch.object(bot, "gh", side_effect=fake_gh), \
                mock.patch.object(bot, "greenlight_status", return_value=("ok", "x")):
            closed = bot.close_stale_prs("o/r", days=2, dry_run=True, drafts_only=False)
        # #1 has its verdict and waits on its author; #4 is queued; #2 and #3 are round winners.
        self.assertEqual(closed, {1})

    def test_a_merge_conflict_is_where_the_run_stopped_not_a_word_in_its_output(self):
        err = "MERGE_CONFLICT aaaaaaa does not merge cleanly onto ccccccc"
        self.assertEqual(bot.merge_conflict_line("", err), err)
        self.assertEqual(bot.merge_conflict_line("PR_TIP " + "a" * 40 + "\nbuilding\n",
                                                 "x.cu:3: error: MERGE_CONFLICT is not declared"), "")
        self.assertEqual(bot.merge_conflict_line("", "see MERGE_CONFLICT above"), "")

    def test_strikes_are_counted_per_key_and_reset_by_a_new_commit(self):
        p = os.path.join(_STATE, "strikes.json")
        self.assertEqual(bot.record_strike(p, 1, "a", "box"), 1)
        self.assertEqual(bot.record_strike(p, 1, "a", "cb"), 1)
        self.assertEqual(bot.record_strike(p, 1, "a", "box"), 2)
        self.assertEqual(bot.record_strikes(p, 1, "a", "cb+serve"), 2)
        self.assertEqual(bot.record_strike(p, 1, "b", "box"), 1)
        with open(p, "w") as f:                                  # the pre-2026-09-26 one-key form
            json.dump({"1": {"commit": "a", "key": "cb", "count": 1}}, f)
        self.assertEqual(bot.record_strike(p, 1, "a", "cb"), 2)
        bot.clear_strikes(p, 1)
        self.assertEqual(bot._load_strikes(p), {})

    def test_freshness_is_its_own_refusal(self):
        with mock.patch.object(bot, "current_main_sha", return_value="c" * 40):
            self.assertEqual(bot.fresh_against_main("o/r", {"onto": "c" * 40}), (True, "ok"))
            ok, why = bot.fresh_against_main("o/r", {"onto": "9" * 40})
        self.assertFalse(ok)
        self.assertTrue(bot.refused_only_for_stale_main(why))
        self.assertFalse(bot.refused_only_for_stale_main("carries a REJECT from another eval bot"))
        with mock.patch.object(bot, "current_main_sha", return_value=""):
            self.assertEqual(bot.fresh_against_main("o/r", {"onto": "c" * 40}), (False, bot.PR_UNREADABLE))

    def test_a_run_started_by_hand_waits_for_the_cron_rounds_lock(self):
        import fcntl
        with mock.patch.dict(os.environ, {"SPARKINFER_BOT_LOCK_HELD": ""}), \
                mock.patch.object(bot, "_bot_lock", None), mock.patch.object(bot.time, "sleep"):
            with open(bot.BOT_LOCK_FILE, "a") as held:
                fcntl.flock(held, fcntl.LOCK_EX)
                self.assertFalse(bot.hold_bot_lock(wait_s=0))
                fcntl.flock(held, fcntl.LOCK_UN)
            self.assertTrue(bot.hold_bot_lock(wait_s=0))
            bot._bot_lock.close()
        with mock.patch.dict(os.environ, {"SPARKINFER_BOT_LOCK_HELD": "1"}), mock.patch.object(bot, "_bot_lock", None):
            self.assertTrue(bot.hold_bot_lock(wait_s=0))          # a wrapper's run inherits the lock


class RoundGuardSessionTests(unittest.TestCase):
    def test_stages_timeout_moved_into_their_own_group_go_with_the_round(self):
        import subprocess
        import time as _t
        with tempfile.TemporaryDirectory() as d:
            # The orphan leads its session, like the `bash -s` of an ssh round; `timeout` then moves
            # its stage into a process group of its own.
            orphan = subprocess.Popen(["bash", "-c", "timeout 300 sleep 300; true"], start_new_session=True)
            _t.sleep(0.5)
            with open(f"/proc/{orphan.pid}/stat") as fh:
                start = fh.read().split()[21]
            with open(os.path.join(d, "old.pid"), "w") as fh:
                fh.write(f"{orphan.pid} {start}\n")
            script = bot.round_guard_sh("t").replace("/tmp/sparkinfer-bot-rounds", d)
            r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            orphan.wait(timeout=10)
            left = subprocess.run(["pgrep", "-s", str(orphan.pid)], capture_output=True, text=True).stdout
            self.assertEqual(left.strip(), "", "a stage of the orphaned round is still running")


class Round3SharedHelperTests(unittest.TestCase):
    """Fixes from the third review of the 2026-09-26 change."""

    R = staticmethod(lambda out, rc=0: mock.Mock(returncode=rc, stdout=out, stderr=""))

    def test_a_body_github_did_not_return_is_unknown_never_unticked(self):
        for bad in (self.R("", 1), self.R("not json")):
            with mock.patch.object(bot, "gh", return_value=bad):
                self.assertEqual(bot.greenlight_status("o/r", 1, set())[0], "unknown")
        with mock.patch.object(bot, "gh", return_value=self.R(json.dumps({"body": None}))):
            self.assertEqual(bot.greenlight_status("o/r", 1, set())[0], "unchecked")   # really empty

    def test_the_unticked_box_close_never_acts_on_a_failed_read(self):
        prs = [{"number": 1, "title": "t", "labels": [], "isDraft": False, "author": {"login": "dev"},
                "authorAssociation": "CONTRIBUTOR"}]

        def fake_gh(a):
            if a[:2] == ["pr", "list"]:
                return self.R(json.dumps(prs))
            if "body" in a:
                return self.R("", 1)                                  # GitHub hiccup
            return self.R(json.dumps({"files": [{"path": "runtime/x.cu"}]}))
        with mock.patch.object(bot, "gh", side_effect=fake_gh):
            self.assertEqual(bot.close_unchecked_rtx5090_prs("o/r", dry_run=True), set())

    def test_a_pr_a_bot_sent_to_rebase_is_waiting_on_its_author(self):
        pr = {"number": 1, "headRefOid": "a" * 40, "labels": [{"name": "qwen38-needs-rebase"}],
              "mergeable": "MERGEABLE", "files": [{"path": "kernels/x.cu"}]}
        with mock.patch.object(bot, "greenlight_status", return_value=("ok", "x")):
            self.assertFalse(bot.waiting_for_first_verdict("o/r", pr, set(), rebase_label="qwen38-needs-rebase"))
            # Another bot's (the paused Bonsai's, say) may be left from an older head: not this bot's call.
            self.assertTrue(bot.waiting_for_first_verdict("o/r", pr, set(), rebase_label="museglimmer-needs-rebase"))
        # The daily Action goes by GitHub's own conflict state and the verdicts: a needs-rebase may be
        # left from an older head (the bots drop their own once the head moves).
        with mock.patch.object(bot, "greenlight_status", return_value=("ok", "x")), \
                mock.patch.object(bot, "gh", return_value=self.R(json.dumps({"comments": []}))):
            self.assertTrue(bot.awaiting_any_model_verdict("o/r", pr))
            self.assertFalse(bot.awaiting_any_model_verdict("o/r", dict(pr, mergeable="CONFLICTING")))
            self.assertFalse(bot.awaiting_any_model_verdict(
                "o/r", dict(pr, labels=[], files=[{"path": "eval/pr_eval_bot.py"}])))   # measured by no bot

    def test_a_bot_started_by_a_wrapper_never_waits_on_its_parents_lock(self):
        import fcntl
        import subprocess
        import sys as _sys
        with open(bot.BOT_LOCK_FILE, "a") as held:                      # another round holds it
            fcntl.flock(held, fcntl.LOCK_EX)
            with mock.patch.dict(os.environ, {"SPARKINFER_BOT_LOCK_HELD": "1"}), \
                    mock.patch.object(bot, "_bot_lock", None):
                self.assertTrue(bot.hold_bot_lock(wait_s=0))
        # A wrapper from before SPARKINFER_BOT_LOCK_HELD: the bot re-locks the descriptor it inherited.
        env = {k: v for k, v in os.environ.items() if k != "SPARKINFER_BOT_LOCK_HELD"}
        env["SPARKINFER_LOCK_FILE"] = bot.BOT_LOCK_FILE
        code = "import pr_eval_bot as b; print(b.hold_bot_lock(wait_s=0))"
        r = subprocess.run(["bash", "-c", f'exec 9>"$SPARKINFER_LOCK_FILE"; flock 9; "{_sys.executable}" -c "{code}"'],
                           cwd=os.path.dirname(os.path.abspath(bot.__file__)), env=env,
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.stdout.strip().splitlines()[-1], "True", r.stderr)

    def test_one_round_is_one_strike_however_many_checks_it_lost(self):
        p = os.path.join(_STATE, "strikes-dedupe.json")
        self.assertEqual(bot.record_strikes(p, 1, "a", "guard-box+guard-box+guard-box+cb-box"), 1)
        self.assertEqual(bot.record_strikes(p, 1, "a", "guard-box"), 2)


class Round4SharedHelperTests(unittest.TestCase):
    R = staticmethod(lambda out, rc=0: mock.Mock(returncode=rc, stdout=out, stderr=""))

    def test_an_unknown_greenlight_keeps_a_pr_waiting(self):
        pr = {"number": 1, "headRefOid": "a" * 40, "labels": [], "mergeable": "MERGEABLE", "files": []}
        with mock.patch.object(bot, "greenlight_status", return_value=("unknown", "no answer")):
            self.assertTrue(bot.waiting_for_first_verdict("o/r", pr, set()))
        with mock.patch.object(bot, "greenlight_status", return_value=("no-bench", "x")):
            self.assertFalse(bot.waiting_for_first_verdict("o/r", pr, set()))

    def test_the_daily_close_spares_any_bots_merge_first_even_with_a_verdict(self):
        old = "2026-01-01T00:00:00Z"
        prs = [{"number": n, "title": "t", "updatedAt": old, "isDraft": False, "headRefOid": "a" * 40,
                "mergeable": "MERGEABLE", "labels": [{"name": l} for l in labs], "files": []}
               for n, labs in ((1, []), (2, ["bonsai-merge-first"]), (3, ["museglimmer-merge-first"]))]
        verdict = {"comments": [{"authorAssociation": "MEMBER",
                                 "body": '<!-- sparkinfer-qwen38-eval:v6:' + "a" * 40 + ' {"label":"none"} -->'}]}

        def fake_gh(a):
            if a[:2] == ["pr", "list"]:
                return self.R(json.dumps(prs))
            if a[:2] == ["pr", "view"]:
                return self.R(json.dumps(verdict))
            return self.R("")
        with mock.patch.object(bot, "gh", side_effect=fake_gh), \
                mock.patch.object(bot, "greenlight_status", return_value=("ok", "x")):
            self.assertEqual(bot.close_stale_prs("o/r", days=2, dry_run=True, drafts_only=False), {1})

    def test_the_retired_ar_bot_never_closes_on_an_unknown_read(self):
        src = open(bot.__file__).read()
        i = src.index('elif status == "unknown":')
        self.assertLess(i, src.index("else:  # unchecked", i))


class Iteration3SharedTests(unittest.TestCase):
    """Fixes from the post-merge review of 2026-09-26."""
    R = staticmethod(lambda out, rc=0: mock.Mock(returncode=rc, stdout=out, stderr=""))

    def _comments(self, *markers, assoc="MEMBER"):
        return json.dumps({"comments": [{"authorAssociation": assoc, "body": m} for m in markers]})

    @staticmethod
    def _marker(bot, sha, label="XL"):
        return f'<!-- sparkinfer-{bot}-eval:v1:{sha} {{"label":"{label}"}} -->'

    def test_other_bots_labels_count_only_for_the_head_they_measured(self):
        labels = {"eval-bonsai:XL", "bonsai-merge-first", "eval-qwen38:none", "eval-dspark:S",
                  "eval-museglimmer:M", "eval:XL"}
        comments = self._comments(self._marker("bonsai", "a" * 40), self._marker("qwen38", "b" * 40, "none"))
        removed = []
        with mock.patch.object(bot, "gh", return_value=self.R(comments)), \
                mock.patch.object(bot, "remove_label", side_effect=lambda r, n, l: removed.append(l)), \
                mock.patch.object(bot, "sync_generic_eval_label", return_value="none"):
            self.assertTrue(bot.strip_foreign_stale_labels("o/r", 1, labels, "b" * 40, "eval-museglimmer:"))
        # Bonsai measured a, not b: its tier and merge-first go. Qwen measured b: kept. DSpark left no
        # marker here (a format this code cannot read): left alone. The bot's own label: its own rules.
        self.assertEqual(sorted(removed), ["bonsai-merge-first", "eval-bonsai:XL"])

    def test_unread_comments_or_untrusted_markers_never_strip(self):
        with mock.patch.object(bot, "gh", return_value=self.R("", 1)), \
                mock.patch.object(bot, "remove_label") as rm:
            self.assertFalse(bot.strip_foreign_stale_labels("o/r", 1, {"eval-bonsai:XL"}, "b" * 40, "eval-qwen38:"))
            rm.assert_not_called()
        with mock.patch.object(bot, "gh", return_value=self.R(self._comments(self._marker("bonsai", "a" * 40),
                                                                            assoc="NONE"))):
            self.assertEqual(bot.bot_verdict_heads("o/r", 1), {})

    def test_a_failed_pr_list_is_none_not_empty(self):
        with mock.patch.object(bot, "gh", return_value=self.R("", 1)):
            self.assertIsNone(bot.open_prs_or_none("o/r", "number"))
        with mock.patch.object(bot, "gh", return_value=self.R("[]")):
            self.assertEqual(bot.open_prs_or_none("o/r", "number"), [])

    def test_the_bot_account_is_checked_when_configured(self):
        with mock.patch.dict(os.environ, {"SPARKINFER_BOT_LOGIN": "bot"}):
            with mock.patch.object(bot, "gh", return_value=self.R("someone\n")):
                self.assertEqual(bot.acting_account_ok(), (False, "someone"))
            with mock.patch.object(bot, "gh", return_value=self.R("bot\n")):
                self.assertEqual(bot.acting_account_ok(), (True, "bot"))
            with mock.patch.object(bot, "gh", return_value=self.R("", 1)):
                self.assertTrue(bot.acting_account_ok()[0])                  # GitHub silent: not a refusal
        with mock.patch.dict(os.environ, {"SPARKINFER_BOT_LOGIN": ""}):
            self.assertEqual(bot.acting_account_ok(), (True, ""))

    def test_a_run_killed_at_the_ssh_limit_keeps_the_tip_it_built(self):
        import subprocess
        e = subprocess.TimeoutExpired("ssh", 7200, output="REMOTE_HEAD x\nPR_TIP " + "c" * 40 + "\n")
        self.assertEqual(bot.exception_result(e)["pr_tip"], "c" * 40)
        self.assertIsNone(bot.exception_result(subprocess.TimeoutExpired("ssh", 7200))["pr_tip"])

    def test_the_daily_action_sees_every_open_pr(self):
        for fn in (bot.close_unchecked_rtx5090_prs, bot.close_exhausted_eval_prs, bot.close_stale_draft_prs):
            with mock.patch.object(bot, "gh", return_value=self.R("[]")) as g:
                fn("o/r", dry_run=True)
            args = g.call_args_list[0].args[0]
            self.assertIn("--limit", args, fn.__name__)


class MergeStepTests(unittest.TestCase):
    """arb.merged_checkout_script, run for real against a throwaway repository."""

    def setUp(self):
        import subprocess
        self.sp = subprocess
        self.t = tempfile.mkdtemp(prefix="sparkinfer-merge-")
        self.addCleanup(__import__("shutil").rmtree, self.t, True)
        g = lambda *a, cwd=None: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a],
                                                cwd=cwd, check=True, capture_output=True, text=True)
        self.g = g
        self.origin, self.work, self.box = (os.path.join(self.t, n) for n in ("origin.git", "work", "box"))
        g("init", "-q", "--bare", "-b", "main", self.origin)
        g("clone", "-q", self.origin, self.work)
        os.makedirs(os.path.join(self.work, "bench/scripts"))
        for p, c in (("kernel.cu", "a\n"), ("bench/scripts/_eval_speed.sh", "ruler\n")):
            with open(os.path.join(self.work, p), "w") as fh:
                fh.write(c)
        g("add", "-A", cwd=self.work); g("commit", "-qm", "base", cwd=self.work)
        g("push", "-q", "origin", "HEAD:main", cwd=self.work)
        self.main = g("rev-parse", "HEAD", cwd=self.work).stdout.strip()
        g("clone", "-q", self.origin, self.box)

    def _pr(self, path, content):
        self.g("checkout", "-q", "-B", "pr", self.main, cwd=self.work)
        with open(os.path.join(self.work, path), "w") as fh:
            fh.write(content)
        self.g("commit", "-qam", "pr", cwd=self.work)
        self.g("push", "-qf", "origin", "HEAD:refs/pull/1/head", cwd=self.work)
        return self.g("rev-parse", "HEAD", cwd=self.work).stdout.strip()

    def _run(self):
        script = "set -euo pipefail\n" + bot.merged_checkout_script("pull/1/head", self.main, ("bench/scripts/",))
        return self.sp.run(["bash", "-c", script], cwd=self.box, capture_output=True, text=True, timeout=60)

    def test_a_clean_merge_reports_the_tip_it_built(self):
        tip = self._pr("kernel.cu", "b\n")
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("PR_TIP " + tip, r.stdout)
        self.assertEqual(bot.merge_conflict_line(r.stdout, r.stderr), "")

    def test_a_tip_that_edits_the_harness_is_never_measured(self):
        self._pr("bench/scripts/_eval_speed.sh", "forged\n")
        r = self._run()
        self.assertEqual(r.returncode, 1)
        self.assertIn("bench/scripts/_eval_speed.sh", bot.harness_touched_line(r.stdout, r.stderr))

    def test_a_conflict_prints_no_tip(self):
        self.g("checkout", "-q", "main", cwd=self.work)
        with open(os.path.join(self.work, "kernel.cu"), "w") as fh:
            fh.write("main-side\n")
        self.g("commit", "-qam", "main moves", cwd=self.work)
        self.g("push", "-q", "origin", "HEAD:main", cwd=self.work)
        base, self.main = self.main, self.g("rev-parse", "HEAD", cwd=self.work).stdout.strip()
        self.g("checkout", "-q", "-B", "pr", base, cwd=self.work)
        with open(os.path.join(self.work, "kernel.cu"), "w") as fh:
            fh.write("pr-side\n")
        self.g("commit", "-qam", "pr", cwd=self.work)
        self.g("push", "-qf", "origin", "HEAD:refs/pull/1/head", cwd=self.work)
        r = self._run()
        self.assertEqual(r.returncode, 1)
        self.assertTrue(bot.merge_conflict_line(r.stdout, r.stderr).startswith("MERGE_CONFLICT "))
        self.assertNotIn("PR_TIP", r.stdout)

    def test_git_failing_on_the_box_is_the_boxs_and_an_old_lock_is_cleared(self):
        self._pr("kernel.cu", "b\n")
        lock = os.path.join(self.box, ".git", "index.lock")
        open(lock, "w").close()                                   # a checkout killed a moment ago
        r = self._run()
        self.assertEqual(r.returncode, 1)
        self.assertIn("RETRYABLE_INFRA_FAILURE git reset failed", r.stderr)
        old = __import__("time").time() - 3600
        os.utime(lock, (old, old))                                # ... or an hour ago: ours to clear
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stderr)


class BenchSweepTests(unittest.TestCase):
    def test_the_sweeps_exit_code_is_kept_for_the_caller(self):
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.abspath(bot.__file__)))
        for rc in (137, 1, 0):
            script = (f"set -uo pipefail\nsource {root}/bench/scripts/_eval_speed.sh\n"
                      f"si_run() {{ echo 'SWEEP_JSON {{}}'; return {rc}; }}\ngclks=()\n"
                      f"bench_sweep_run model 128 128 5 || true\necho RC=${{_BENCH_SWEEP_RC:-unset}}\n")
            r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
            self.assertIn(f"RC={rc}", r.stdout, r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
