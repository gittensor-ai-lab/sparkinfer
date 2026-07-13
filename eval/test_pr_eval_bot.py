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


class PrEvalBotPolicyTest(unittest.TestCase):
    def test_merge_conflict_blocks_eval(self):
        self.assertTrue(bot.pr_merge_conflict("CONFLICTING"))
        self.assertFalse(bot.pr_merge_conflict("MERGEABLE"))
        self.assertFalse(bot.pr_merge_conflict("UNKNOWN"))
        self.assertFalse(bot.pr_merge_conflict(None))

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
        prs = [{"number": 99, "updatedAt": "2026-01-01T00:00:00Z", "labels": []}]
        gh_mock = mock.Mock(return_value=mock.Mock(stdout=json.dumps(prs)))
        with mock.patch.object(bot, "gh", gh_mock), \
             mock.patch.object(bot, "pr_inactive_days", return_value=10.0):
            closed = bot.close_stale_prs("gittensor-ai-lab/sparkinfer", days=2, dry_run=True)
        self.assertEqual(closed, {99})
        gh_mock.assert_called_once()

    def test_evaluated_commit_from_comment_accepts_verdict(self):
        body = bot.render({"label": "S", "pass": True, "tps": 200.0, "top1": 1.0, "kl": 0.0}, "df74674")
        self.assertEqual(bot._evaluated_commit_from_comment(body), "df74674")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
