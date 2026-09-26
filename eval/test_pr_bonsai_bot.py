import os
import types
import json
import unittest
from unittest import mock

import pr_bonsai_bot as bot
import pr_eval_bot as arb
import pr_museglimmer_bot as muse_bot
import pr_qwen38_bot as qwen38_bot

# Nothing here may touch the controller's own state: every file the bots write goes to a temp dir.
import os as _os
import tempfile as _tempfile
import _bot_test_state
_STATE = _bot_test_state.new_state_dir()
_bot_test_state.isolate(_STATE, arb, bot, muse_bot, qwen38_bot)


def setUpModule():
    _bot_test_state.isolate(_STATE, arb, bot, muse_bot, qwen38_bot)


# The author clock (arb.AuthorWaitClock) of a PR that has waited on its author since the epoch: the
# stale-close tests below are about WHICH PRs may close; the clock has its own tests.
_LONG_WAITED = lambda self, num, head: 0.0

MAIN_BONSAI = {128: (99.2, 2028.6), 512: (98.9, 5200.0), 4096: (96.9, 8432.7),
               16384: (93.0, 7600.0), 32768: (89.0, 6500.0)}
MAIN_CB = {2: 190.0, 4: 360.0, 8: 640.0, 16: 900.0, 32: 1100.0}


def box_stdout(bonsai=None, cb=None, top1=1.0, kl=0.0, pf=None, reg_ok=True, reg_why=(),
               guards=None, role="pr", extra=""):
    """The lines _remote_script prints, for one ref."""
    bonsai = MAIN_BONSAI if bonsai is None else bonsai
    cb = MAIN_CB if cb is None else cb
    pf = {128: [(0.90, 0.018)] * 3, 1024: [(0.93, 0.012)] * 3} if pf is None else pf
    guards = {"GUARD36": (150.0, 9000.0), "GUARDMO": (60.0, 7000.0), "GUARDUN": (55.0, 6800.0),
              "GUARDMG": (80.0, 2000.0)} if guards is None else guards
    out = ["REMOTE_HEAD abc1234", "HARNESS_PINNED 5001fa4"]
    out += [f"BONSAI {c} {d} {p}" for c, (d, p) in bonsai.items()]
    out += [f"BONSAICB {c} {v} {v} {v} {v}" for c, v in cb.items()]
    out += ["RESULT_TOKEN_COUNT 1203", "SCORE_DONE 1202", "ACCURACY_STAGE_DONE"]
    if role == "main":
        out.append("REMOTE_SHA " + "b013fc9" + "0" * 33)
        out.append("SELFCHECK top1=1.0000 kl=0.000000 ppl_pr=9.56 ppl_main=9.56")
    else:
        out.append(f"METRIC top1={top1} kl={kl} ppl_pr=9.6 ppl_main=9.56")
    for p, runs in pf.items():
        out += [f"PFCHECK {p} {i + 1} {t} {k}" for i, (t, k) in enumerate(runs)]
    out.append("BONSAIREG_OK" if reg_ok else "BONSAIREG_FAILED")
    out += [f"BONSAIREG_WHY {w}" for w in reg_why]
    out.append("GUARD_START")
    for tag, v in guards.items():
        out.append(f"{tag}_FAILED" if v is None else f"{tag} 32768 {v[0]} {v[1]}")
    out.append("GUARD_END")
    return "\n".join(out) + "\n" + extra


def run(stdout, rc=0, stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


def main_baseline():
    with mock.patch.object(bot, "_ssh_run_resilient", return_value=run(box_stdout(role="main"))):
        m = bot.measure_main_baseline("h", 1)
    assert m["ok"], m
    return m


def evaluate(pr_stdout, main=None, rc=0, stderr=""):
    main = main or main_baseline()
    with mock.patch.object(bot, "_ssh_run_resilient", return_value=run(pr_stdout, rc, stderr)), \
            mock.patch.object(bot, "POLARIS_ENABLED", False):
        return bot.eval_bonsai_on_box("h", 1, "pull/1139/merge", main)


class ScopeTests(unittest.TestCase):
    def test_axes_are_the_ones_issue_1138_asked_for(self):
        for c in ("128", "512", "4k", "16k", "32k"):
            self.assertIn(f"bonsai-decode@{c}", bot.SCORING_DIMS)
            self.assertIn(f"bonsai-prefill@{c}", bot.SCORING_DIMS)
        for c in (2, 4, 8, 16, 32):
            self.assertIn(f"bonsai-cb-decode@c{c}", bot.SCORING_DIMS)
        self.assertEqual(len(bot.SCORING_DIMS), 15)

    def test_automerge_is_opt_in_and_autoclose_opt_out(self):
        # In the test process no env var is set. Auto-merge needs SPARKINFER_BONSAI_AUTOMERGE=1
        # (production sets it in .env.eval); closing is on, like the sibling bots', unless
        # SPARKINFER_BONSAI_AUTOCLOSE=0 (decision 2026-09-26).
        self.assertFalse(bot.AUTO_MERGE)
        self.assertTrue(bot.AUTO_CLOSE)
        src = open(bot.__file__).read()
        self.assertIn('os.environ.get("SPARKINFER_BONSAI_AUTOCLOSE", "1") != "0"', src)


class AutoMergeGateTests(unittest.TestCase):
    OK_INFO = {"state": "OPEN", "isDraft": False, "labels": [{"name": "eval-bonsai:XL"},
               {"name": "bonsai-merge-first"}], "author": {"login": "dev"}, "mergeable": "MERGEABLE",
               "files": [{"path": "kernels/foo.cu"}], "headRefOid": "a" * 40}

    MAIN = "c" * 40

    def _ok(self, info=None, scores=None, main_now=MAIN):
        info = info or self.OK_INFO
        scores = self.OK_INFO if scores is None else scores
        s = scores if isinstance(scores, dict) and "1139" in scores else {
            "1139": {"commit": "a" * 40, "label": "XL", "pass": True, "onto": self.MAIN}}
        with mock.patch.object(bot.arb, "gh", return_value=run(__import__("json").dumps(info))), \
                mock.patch.object(bot.arb, "current_main_sha", return_value=main_now), \
                mock.patch.object(bot.arb, "load_denylist", return_value=set()), \
                mock.patch.object(bot.arb, "author_penalty_until", return_value=None), \
                mock.patch.object(bot.arb, "AUTOMERGE_SENSITIVE", ("server/",)), \
                mock.patch.object(bot, "_load_scores", return_value=s):
            return bot.auto_merge_ok_bonsai("o/r", 1139)

    def test_scored_commit_passes(self):
        ok, why = self._ok()
        self.assertTrue(ok, why)

    def test_a_push_after_the_verdict_blocks_merge(self):
        # head moved to b..b, but the bot only scored a..a.
        info = dict(self.OK_INFO, headRefOid="b" * 40)
        ok, why = self._ok(info=info, scores={"1139": {"commit": "a" * 40, "label": "XL", "pass": True}})
        self.assertFalse(ok)
        self.assertIn("not the commit last scored", why)

    def test_a_non_passing_recorded_verdict_blocks_merge(self):
        ok, why = self._ok(scores={"1139": {"commit": "a" * 40, "label": "none", "pass": False}})
        self.assertFalse(ok)
        self.assertIn("recorded verdict", why)

    def test_a_verdict_against_an_older_main_is_not_merged(self):
        # Another bot merged since the verdict: the merge would ship a combination nobody measured.
        ok, why = self._ok(main_now="9" * 40)
        self.assertFalse(ok)
        self.assertIn("re-measured before it may merge", why)
        self.assertFalse(self._ok(scores={"1139": {"commit": "a" * 40, "label": "XL", "pass": True}})[0])
        self.assertEqual(self._ok(main_now=""), (False, bot.arb.PR_UNREADABLE))

    def test_a_reject_from_another_bot_blocks_merge(self):
        info = dict(self.OK_INFO, labels=self.OK_INFO["labels"] + [{"name": "eval-qwen38:REJECT"}])
        ok, why = self._ok(info=info)
        self.assertFalse(ok)
        self.assertIn("REJECT from another", why)

    def test_try_auto_merge_pins_the_scored_head(self):
        calls = []

        def fake_gh(a):
            calls.append(a)
            if a[:2] == ["pr", "view"]:
                return run('{"headRefOid": "' + "a" * 40 + '"}')
            return run("")
        # Pinned to the SCORED commit: a second head lookup would pin a push that landed between
        # the gate and the merge (here the live head has moved to b..b).
        fake_head = lambda a: calls.append(a) or (run('{"headRefOid": "' + "b" * 40 + '"}')
                                                  if a[:2] == ["pr", "view"] else run(""))
        with mock.patch.object(bot, "auto_merge_ok_bonsai", return_value=(True, "ok")), \
                mock.patch.object(bot, "_load_scores", return_value={"1139": {"commit": "a" * 40}}), \
                mock.patch.object(bot.arb, "gh", side_effect=fake_head):
            self.assertTrue(bot.try_auto_merge_bonsai("o/r", 1139))
        merge = next(a for a in calls if a[:2] == ["pr", "merge"])
        self.assertIn("--match-head-commit", merge)
        self.assertEqual(merge[merge.index("--match-head-commit") + 1], "a" * 40)
        calls.clear()
        with mock.patch.object(bot, "auto_merge_ok_bonsai", return_value=(True, "ok")), \
                mock.patch.object(bot, "_load_scores", return_value={}), \
                mock.patch.object(bot.arb, "gh", side_effect=fake_gh):
            self.assertFalse(bot.try_auto_merge_bonsai("o/r", 1139))     # nothing to pin: no merge
        self.assertFalse(any(a[:2] == ["pr", "merge"] for a in calls))

    def test_policy_note_follows_the_live_switches(self):
        with mock.patch.object(bot, "AUTO_MERGE", True), mock.patch.object(bot, "AUTO_CLOSE", False):
            note = bot._policy_note()
        self.assertIn("auto-merged", note)
        self.assertIn("exact commit scored", note)
        self.assertIn("does not close", note)
        with mock.patch.object(bot, "AUTO_MERGE", False):
            self.assertIn("does not auto-merge", bot._policy_note())

    def test_remote_script_measures_the_served_path_with_mains_harness(self):
        s = bot._remote_script("pull/1139/merge", role="pr")
        # The default (folded) loader: the bot must never select the native arm.
        self.assertNotIn("SPARKINFER_BONSAI_NATIVE", s)
        for f in ("qwen3_gguf_prefill_check.cpp", "qwen3_gguf_score.cpp", "bench/scripts",
                  "eval/bonsai_regression.py", "qwen3_gguf_bench.cpp"):
            self.assertIn(f, s.split("HARNESS_PIN_FAILED")[0])
        self.assertIn("-DBUILD_SERVER=ON", s)
        # A private nvcc TMPDIR, never a sweep of the shared /tmp.
        self.assertNotIn("rm -rf /tmp/tmpxft", s)
        self.assertIn('export TMPDIR="$REPO/.tmp-bonsai-bot"', s)
        self.assertIn(f"{bot.BONSAI_GGUF_BYTES}", s)
        for tag in ("GUARD36", "GUARDMO", "GUARDUN", "GUARDMG"):
            self.assertIn(f"guard {tag} ", s)

    def test_main_role_owns_the_reference_dump(self):
        m = bot._remote_script("main", role="main")
        self.assertIn('[ "$IS_PR" = "1" ] || rm -f "$DUMP_MAIN"', m)
        self.assertIn("IS_PR=0", m)
        self.assertIn("--metric-label SELFCHECK", m)
        self.assertIn("IS_PR=1", bot._remote_script("pull/1/head", role="pr"))

    def test_harness_edits_are_not_evaluated(self):
        for path in ("runtime/examples/qwen3_gguf_prefill_check.cpp", "eval/pr_bonsai_bot.py",
                     "bench/scripts/_eval_speed.sh"):
            self.assertTrue(any(path.startswith(h) for h in bot.HARNESS_PATHS), path)
        self.assertFalse(any("runtime/src/models/qwen35_prefill.cpp".startswith(h)
                             for h in bot.HARNESS_PATHS))


class ParseTests(unittest.TestCase):
    def test_every_wire_line(self):
        p = bot._parse_remote(box_stdout(extra="BONSAICB_FAILED 32\nPFCHECK_FAILED 128 4\n"
                                               "GUARDMO_UNAVAILABLE\n"))
        self.assertEqual(p["head"], "abc1234")
        self.assertEqual(p["bonsai"][128], {"decode": 99.2, "prefill": 2028.6})
        self.assertEqual(p["bonsai_cb"][2], 190.0)
        self.assertEqual(p["cb_runs"][2], [190.0, 190.0, 190.0])
        self.assertEqual(p["cb_failed"], [32])
        self.assertEqual((p["top1"], p["kl"]), (1.0, 0.0))
        self.assertEqual(p["score_positions"], 1202)
        self.assertEqual(len(p["pfcheck"][128]), 3)
        self.assertEqual(p["pfcheck_failed"], {128: 1})
        self.assertTrue(p["bonsaireg_ok"])
        self.assertEqual(p["guardmg"][32768], {"decode": 80.0, "prefill": 2000.0})
        self.assertTrue(p["guardmo_unavailable"])

    def test_failure_lines(self):
        p = bot._parse_remote("BONSAI_FAILED\nBONSAIREG_FAILED\nBONSAIREG_WHY generate: native and "
                              "folded disagree on greedy tokens\nGUARDUN_FAILED\nMETRIC top1=0 kl=99 "
                              "ppl_pr=0 ppl_main=0   (NO SHARED POSITIONS)\n")
        self.assertTrue(p["bonsai_failed"])
        self.assertFalse(p["bonsaireg_ok"])
        self.assertEqual(p["bonsaireg_why"], ["generate: native and folded disagree on greedy tokens"])
        self.assertTrue(p["guardun_failed"])
        self.assertEqual(p["top1"], 0.0)


class FailureClassTests(unittest.TestCase):
    def test_box_faults_are_retried_and_ref_faults_are_charged(self):
        infra = bot._is_infra_failure
        self.assertTrue(infra("", "RETRYABLE_INFRA_FAILURE GPU still holding 30000 MiB"))
        self.assertTrue(infra("", "MODEL_CHECK_FAILED Ternary-Bonsai-2-27B GGUF missing"))
        self.assertTrue(infra("", "TOKENIZE_FAILED (score corpus)"))
        self.assertTrue(infra("", "RETRYABLE_INFRA_FAILURE git fetch pull/1/merge failed\n"
                                  "REMOTE_SCRIPT_FAILED line=70 exit=1 reason="))
        self.assertTrue(infra("", "REMOTE_SCRIPT_FAILED line=90 exit=137 reason=likely OOM-killed"))
        self.assertTrue(infra("partial output", ""))  # no diagnostic: the session died
        self.assertFalse(infra("", "BUILD_FAILED — tail of the build log:\nfoo.cu(1): error: x"))
        self.assertFalse(infra("", "SCORE_FAILED -- tail of the score log:"))
        self.assertFalse(infra("", "REMOTE_SCRIPT_FAILED line=120 exit=139 reason=likely segfault"))


class PrefillPathGateTests(unittest.TestCase):
    MAIN = {"pfcheck": {128: [(0.875, 0.0178), (0.94, 0.015), (0.906, 0.020)],
                        1024: [(0.93, 0.012)] * 3}}

    def test_within_mains_spread_passes(self):
        pr = {"pfcheck": {128: [(0.92, 0.018), (0.89, 0.019), (0.95, 0.017)],
                          1024: [(0.94, 0.011)] * 3}}
        ok, problems, rows = bot.check_prefill_path(pr, self.MAIN)
        self.assertTrue(ok, problems)
        self.assertEqual([r["prefix"] for r in rows], [128, 1024])

    def test_a_broken_batched_prefill_fails(self):
        pr = {"pfcheck": {128: [(0.10, 3.2)] * 3, 1024: [(0.93, 0.012)] * 3}}
        ok, problems, _ = bot.check_prefill_path(pr, self.MAIN)
        self.assertFalse(ok)
        self.assertTrue(any("@128: mean KL" in p for p in problems))
        self.assertTrue(any("@128: mean top-1" in p for p in problems))

    def test_a_refused_or_crashed_run_fails(self):
        pr = {"pfcheck": {1024: [(0.93, 0.012)] * 3}, "pfcheck_failed": {128: 3}}
        ok, problems, _ = bot.check_prefill_path(pr, self.MAIN)
        self.assertFalse(ok)
        self.assertIn("every PR run failed", problems[0])

    def test_main_unmeasured_is_reported_as_unavailable(self):
        ok, problems, _ = bot.check_prefill_path({"pfcheck": {}}, {"pfcheck": {}})
        self.assertFalse(ok)
        self.assertTrue(all(p.endswith("measurement unavailable") for p in problems))


class ScoringTests(unittest.TestCase):
    def test_1139_shape_scores_its_prefill_win(self):
        pr = dict(MAIN_BONSAI)
        pr[128] = (99.2, 3941.2)          # +94.3%, #1139's claim
        pr[4096] = (96.9, 8378.5)         # -0.64%, inside the 2% band
        res = evaluate(box_stdout(bonsai=pr, top1=0.995, kl=0.004))
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["label"], "XL")
        self.assertEqual(res["best_dim"], "bonsai-prefill@128")
        self.assertTrue(res["pass"])

    def test_any_axis_below_the_floor_rejects(self):
        pr = dict(MAIN_BONSAI)
        pr[128] = (99.2, 3941.2)
        pr[16384] = (93.0, 7300.0)        # -3.9%
        res = evaluate(box_stdout(bonsai=pr))
        self.assertEqual(res["label"], "REJECT")
        self.assertEqual(res["best_dim"], "bonsai-prefill@16k")

    def test_log_matrix_carries_the_raw_readings_behind_every_scored_axis(self):
        pr = dict(MAIN_BONSAI)
        pr[16384] = (93.0, 7866.0)        # +3.5% prefill@16k
        cb = dict(MAIN_CB)
        del cb[32]                        # a dropped width must not appear
        res = evaluate(box_stdout(bonsai=pr, cb=cb))
        m = bot._axis_matrix(res)
        self.assertEqual(set(m), {d["dim"] for d in res["scored_dims"]})
        self.assertNotIn("bonsai-cb-decode@c32", m)
        self.assertEqual(m["bonsai-prefill@16k"], {"pr": 7866.0, "main": 7600.0, "delta": 3.5, "label": "S"})
        self.assertEqual(m["bonsai-decode@128"]["pr"], 99.2)
        self.assertEqual(m["bonsai-cb-decode@c8"], {"pr": 640.0, "main": 640.0, "delta": 0.0, "label": "none"})
        __import__("json").dumps(m)       # JSON-serialisable as written to result.json

    def test_a_width_only_the_pr_fails_is_a_strike_not_a_silent_drop(self):
        # c32 completes on main but not on the PR: no longer dropped unscored (it hid a PR that
        # crashes the engine at c32). Alone, it is a REJECT judged over two rounds (strike_key).
        cb = dict(MAIN_CB)
        del cb[32]
        res = evaluate(box_stdout(cb=cb, extra="BONSAICB_FAILED 32 run\n"))
        self.assertTrue(res["ok"])
        self.assertNotIn("bonsai-cb-decode@c32", {d["dim"] for d in res["scored_dims"]})
        self.assertEqual(res["label"], "REJECT")
        self.assertEqual(res["strike_key"], "cb")
        self.assertEqual(res["cb_pr_missing"], [32])
        self.assertIn("did not complete on the PR build", bot.format_comment("a" * 40, res))

    def test_a_width_lost_to_an_undrained_gpu_is_infra(self):
        cb = dict(MAIN_CB)
        del cb[16]
        res = evaluate(box_stdout(cb=cb, extra="BONSAICB_FAILED 16 gpu\n"))
        self.assertFalse(res["ok"])
        self.assertTrue(res["retry"])

    def test_beside_another_failure_a_missing_width_is_just_a_reason(self):
        pr = dict(MAIN_BONSAI)
        pr[512] = (90.0, 5200.0)          # a real decode regression
        cb = dict(MAIN_CB)
        del cb[32]
        res = evaluate(box_stdout(bonsai=pr, cb=cb, extra="BONSAICB_FAILED 32 run\n"))
        self.assertEqual(res["label"], "REJECT")
        self.assertIsNone(res["strike_key"])          # posted at once, not deferred
        self.assertIn("c32 did not complete", res["reason"])

    def test_a_width_main_cannot_measure_is_dropped_loudly_not_a_skipped_round(self):
        # Skipping the round would stall the bot for every PR if main itself broke at a width.
        cb = dict(MAIN_CB)
        del cb[8]
        with mock.patch.object(bot, "_ssh_run_resilient", return_value=run(box_stdout(role="main", cb=cb))):
            m = bot.measure_main_baseline("h", 1)
        self.assertTrue(m["ok"], m)
        res = evaluate(box_stdout(), main=m)
        self.assertNotIn("bonsai-cb-decode@c8", {d["dim"] for d in res["scored_dims"]})
        self.assertIsNone(res["strike_key"])
        self.assertIn("c8 not scored this round — main has no measurement", bot.format_comment("a" * 40, res))

    def test_main_must_measure_every_installed_guard(self):
        guards = {"GUARD36": (150.0, 9000.0), "GUARDMO": (60.0, 7000.0), "GUARDUN": None,
                  "GUARDMG": (80.0, 2000.0)}
        with mock.patch.object(bot, "_ssh_run_resilient", return_value=run(box_stdout(role="main", guards=guards))):
            m = bot.measure_main_baseline("h", 1)
        self.assertFalse(m["ok"])
        self.assertIn("unsloth", m["reason"])

    def test_accuracy_divergence_rejects(self):
        pr = dict(MAIN_BONSAI)
        pr[128] = (99.2, 3941.2)
        res = evaluate(box_stdout(bonsai=pr, top1=0.80, kl=0.3))
        self.assertEqual(res["label"], "REJECT")
        self.assertFalse(res["accuracy_ok"])
        self.assertEqual(res["speed_label"], "XL")

    def test_a_perplexity_loss_rejects_inside_top1_noise(self):
        # top-1 and KL inside the same-build spread, but PPL 3% worse than main's.
        stdout = box_stdout(top1=0.965, kl=0.010).replace("ppl_pr=9.6 ppl_main=9.56", "ppl_pr=9.85 ppl_main=9.56")
        res = evaluate(stdout)
        self.assertEqual(res["label"], "REJECT")
        self.assertFalse(res["accuracy_ok"])
        self.assertIn("ppl x1.03", res["reason"])

    def test_prefill_path_failure_rejects(self):
        res = evaluate(box_stdout(pf={128: [(0.1, 3.0)] * 3, 1024: [(0.93, 0.012)] * 3}))
        self.assertEqual(res["label"], "REJECT")
        self.assertFalse(res["prefill_path_ok"])

    def test_bonsai_regression_gates_only_when_main_passes_it(self):
        res = evaluate(box_stdout(reg_ok=False, reg_why=["serve: batched row differs"]))
        self.assertEqual(res["label"], "REJECT")
        self.assertIn("serve: batched row differs", res["reason"])
        with mock.patch.object(bot, "_ssh_run_resilient",
                               return_value=run(box_stdout(role="main", reg_ok=False))):
            main = bot.measure_main_baseline("h", 1)
        res = evaluate(box_stdout(reg_ok=False), main=main)
        self.assertEqual(res["label"], "none")
        self.assertIn("not gated", bot.format_comment("abc1234", res))

    def test_guard_regression_rejects_and_unmeasured_guard_retries(self):
        guards = {"GUARD36": (150.0, 9000.0), "GUARDMO": (60.0, 7000.0), "GUARDUN": (55.0, 6800.0),
                  "GUARDMG": (80.0, 1900.0)}
        res = evaluate(box_stdout(guards=guards))
        self.assertEqual(res["label"], "REJECT")
        self.assertFalse(res["guards"]["guardmg"]["ok"])
        main = main_baseline()
        main["guardun"] = {}
        main["guardun_failed"] = True
        res = evaluate(box_stdout(), main=main)
        self.assertFalse(res["ok"])
        self.assertTrue(res["retry"])

    def test_absent_checkpoint_is_skipped_and_said_so(self):
        guards = {"GUARD36": (150.0, 9000.0), "GUARDMO": (60.0, 7000.0), "GUARDUN": (55.0, 6800.0)}
        stdout = box_stdout(guards=guards, extra="GUARDMG_UNAVAILABLE\n")
        with mock.patch.object(bot, "_ssh_run_resilient", return_value=run(
                box_stdout(role="main", guards=guards, extra="GUARDMG_UNAVAILABLE\n"))):
            main = bot.measure_main_baseline("h", 1)
        res = evaluate(stdout, main=main)
        self.assertTrue(res["ok"])
        self.assertTrue(res["guards"]["guardmg"]["skipped"])
        self.assertIn("SKIPPED", bot.format_comment("abc1234", res))

    def test_baseline_refuses_an_unusable_reference(self):
        bad = box_stdout(role="main").replace("SELFCHECK top1=1.0000", "SELFCHECK top1=0.5000")
        with mock.patch.object(bot, "_ssh_run_resilient", return_value=run(bad)):
            self.assertFalse(bot.measure_main_baseline("h", 1)["ok"])
        short = box_stdout(role="main").replace("SCORE_DONE 1202", "SCORE_DONE 0")
        with mock.patch.object(bot, "_ssh_run_resilient", return_value=run(short)):
            self.assertFalse(bot.measure_main_baseline("h", 1)["ok"])
        nopf = "\n".join(l for l in box_stdout(role="main").splitlines() if not l.startswith("PFCHECK 1024"))
        with mock.patch.object(bot, "_ssh_run_resilient", return_value=run(nopf)):
            self.assertFalse(bot.measure_main_baseline("h", 1)["ok"])

    def test_a_ref_that_does_not_build_is_charged_once(self):
        res = evaluate("", rc=1, stderr="BUILD_FAILED — tail of the build log:\nx.cu(3): error: y")
        self.assertFalse(res["ok"])
        self.assertFalse(res["retry"])
        body = bot.format_comment("abc1234", res)
        m = bot.MARKER_RE.search(body)
        self.assertIsNotNone(m)
        self.assertIn('"label":"REJECT"', m.group(2))   # counted as evaluated for this commit
        self.assertIn("sparkinfer bonsai auto-eval", body)


TEMPLATE = open(bot.os.path.join(bot.ROOT, ".github", "PULL_REQUEST_TEMPLATE.md")).read()
BONSAI_ONLY_BODY = TEMPLATE.replace("- [ ] **Ternary-Bonsai-2-27B**", "- [x] **Ternary-Bonsai-2-27B**")


class TempStateMixin:
    """Point the strikes file at a temp path: tests must never touch the controller's real state."""
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        for name, fname in (("STRIKES_FILE", "strikes.json"), ("SCORES_FILE", "scores.json")):
            p = mock.patch.object(bot, name, bot.os.path.join(self._tmp.name, fname))
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)


class ApplyResultTests(TempStateMixin, unittest.TestCase):
    def _apply(self, res, autoclose=False, body="", commit="abc1234", head_now=None, labels=()):
        calls = []
        head_now = commit if head_now is None else head_now

        def fake_gh(a):
            calls.append(a)
            return run(json.dumps({"headRefOid": head_now})) if a[:2] == ["pr", "view"] else run("")
        with mock.patch.object(arb, "gh", side_effect=fake_gh), \
                mock.patch.object(arb, "labels_on_or_none", return_value=set(labels)), \
                mock.patch.object(arb, "add_label", side_effect=lambda r, n, l: calls.append(["add", l])), \
                mock.patch.object(arb, "remove_label"), \
                mock.patch.object(arb, "sync_generic_eval_label"), \
                mock.patch.object(bot, "upload_bonsai_eval_log"), \
                mock.patch.object(bot, "_save_scores"), \
                mock.patch.object(bot, "AUTO_CLOSE", autoclose):
            bot.apply_result("o/r", 1139, commit, res, body=body)
        return calls

    def test_infra_posts_nothing(self):
        self.assertEqual(self._apply({"ok": False, "retry": True, "reason": "GPU busy"}), [])

    def test_none_closes_only_a_pr_declared_for_this_model_alone(self):
        res = evaluate(box_stdout())
        self.assertEqual(res["label"], "none")
        undeclared = self._apply(res, autoclose=True, body=TEMPLATE)
        self.assertIn(["add", "eval-bonsai:none"], undeclared)
        self.assertFalse(any(c[:2] == ["pr", "close"] for c in undeclared))
        shared = TEMPLATE.replace("- [ ] **Shared", "- [x] **Shared").replace(
            "- [ ] **Ternary-Bonsai-2-27B**", "- [x] **Ternary-Bonsai-2-27B**")
        self.assertFalse(any(c[:2] == ["pr", "close"] for c in self._apply(res, autoclose=True, body=shared)))
        declared = self._apply(res, autoclose=True, body=BONSAI_ONLY_BODY)
        self.assertTrue(any(c[:2] == ["pr", "close"] for c in declared))
        close_comment = next(c for c in declared if "sparkinfer-bonsai-auto-close" in " ".join(c))
        self.assertIn("not a finding that anything is wrong", " ".join(close_comment))
        # Another bot scored it a speedup or made it merge-first: not this bot's `none` to close.
        for lab in ("eval-qwen38:M", "qwen38-merge-first"):
            calls = self._apply(res, autoclose=True, body=BONSAI_ONLY_BODY, labels={lab})
            self.assertFalse(any(c[:2] == ["pr", "close"] for c in calls), lab)

    def test_nothing_closes_over_a_commit_the_author_has_replaced(self):
        pr = dict(MAIN_BONSAI)
        pr[512] = (90.0, 5200.0)
        res = evaluate(box_stdout(bonsai=pr))
        self.assertEqual(res["label"], "REJECT")
        calls = self._apply(res, autoclose=True, head_now="f" * 40)
        self.assertFalse(any(c[:2] == ["pr", "close"] for c in calls))
        self.assertFalse(any("sparkinfer-bonsai-auto-close" in " ".join(c) for c in calls))

    def test_reject_closes_unless_switched_off_and_says_why(self):
        pr = dict(MAIN_BONSAI)
        pr[512] = (90.0, 5200.0)
        res = evaluate(box_stdout(bonsai=pr))
        self.assertEqual(res["label"], "REJECT")
        self.assertFalse(any(c[:2] == ["pr", "close"] for c in self._apply(res)))
        calls = self._apply(res, autoclose=True)
        self.assertTrue(any(c[:2] == ["pr", "close"] for c in calls))
        close_comment = " ".join(next(c for c in calls if "sparkinfer-bonsai-auto-close" in " ".join(c)))
        self.assertIn("bonsai-decode@512 regression", close_comment)

    def test_a_failed_run_never_closes(self):
        res = evaluate("", rc=1, stderr="BUILD_FAILED — errors in the build log:\nx.cu(3): error: y")
        calls = self._apply(res, autoclose=True, body=BONSAI_ONLY_BODY)
        self.assertIn(["add", "eval-bonsai:REJECT"], calls)
        self.assertFalse(any(c[:2] == ["pr", "close"] for c in calls))

    def test_a_pr_only_width_failure_rejects_on_the_second_round(self):
        cb = dict(MAIN_CB)
        del cb[32]
        res = evaluate(box_stdout(cb=cb, extra="BONSAICB_FAILED 32 run\n"))
        self.assertEqual(self._apply(res, commit="a" * 40), [])          # strike 1: nothing posted
        self.assertEqual(bot._load_strikes()["1139"]["counts"], {"cb": 1})
        self.assertEqual(self._apply(res, commit="b" * 40), [])          # new commit: starts again
        calls = self._apply(res, commit="b" * 40)                          # strike 2: posted
        self.assertIn(["add", "eval-bonsai:REJECT"], calls)
        self.assertNotIn("1139", bot._load_strikes())                      # cleared once posted


class MergeOntoBaselineTests(unittest.TestCase):
    """#1145 (2026-09-24): GitHub's pull/<n>/merge was built on a main from before #1143, so the PR
    was measured without #1143's prefill@16k gain and read as a -3.1% regression. PRs are now merged
    on the box onto the exact commit the round's baseline measured."""
    SHA = "b013fc9" + "0" * 33

    def test_pr_script_merges_onto_the_baseline_commit_and_pins_its_harness(self):
        s = bot._remote_script("pull/1145/head", role="pr", onto=self.SHA)
        self.assertIn(f"git checkout -qf {self.SHA}", s)
        self.assertIn('merge -q --no-ff --no-edit "$PR_TIP"', s)
        self.assertIn(f"git checkout -q {self.SHA} -- ", s)          # harness from the same commit
        self.assertNotIn("git checkout -q HEAD -- ", s)
        self.assertIn("MERGE_CONFLICT", s)
        self.assertNotIn("pull/1145/merge", s)

    def test_main_script_does_not_merge_and_reports_its_commit(self):
        m = bot._remote_script("main", role="main")
        self.assertNotIn("merge -q --no-ff", m)
        self.assertIn('echo "REMOTE_SHA $(git rev-parse HEAD)"', m)
        self.assertIn("git checkout -q HEAD -- ", m)

    def test_the_baseline_commit_is_what_the_pr_run_merges_onto(self):
        main = main_baseline()
        self.assertEqual(main["sha"], self.SHA)
        seen = {}

        def fake(h, p, script, label):
            seen["script"] = script
            return run(box_stdout() + f"PR_TIP bc31cc9\nMERGED_ONTO {self.SHA[:9]}\n")
        with mock.patch.object(bot, "_ssh_run_resilient", side_effect=fake), \
                mock.patch.object(bot, "POLARIS_ENABLED", False):
            res = bot.eval_bonsai_on_box("h", 1, "pull/1145/head", main)
        self.assertIn(f"git checkout -qf {self.SHA}", seen["script"])
        self.assertEqual(res["merged_onto"], self.SHA[:9])
        self.assertIn(f"measured merged onto `main` `{self.SHA[:9]}`", bot.format_comment("a" * 40, res))

    def test_a_merge_conflict_is_a_rebase_not_a_verdict(self):
        res = evaluate("", rc=1, stderr="MERGE_CONFLICT 4927c65 does not merge cleanly onto b013fc9")
        self.assertFalse(res["ok"])
        self.assertTrue(res["conflict"])
        calls = []
        with mock.patch.object(arb, "gh", side_effect=lambda a: calls.append(a) or run("")), \
                mock.patch.object(arb, "add_label", side_effect=lambda r, n, l: calls.append(["add", l])):
            bot.apply_result("o/r", 1145, "a" * 40, res)
        self.assertEqual(calls, [["add", bot.BONSAI_NEEDS_REBASE]])   # no comment, no REJECT
        self.assertFalse(bot._is_infra_failure("", "MERGE_CONFLICT x does not merge cleanly onto y"))

    def test_github_merge_refs_are_no_longer_used(self):
        src = open(bot.__file__).read()
        self.assertNotIn("_merge_ref_exists", src)
        self.assertNotIn('f"pull/{num}/merge"', src)
        self.assertIn('ref = f"pull/{num}/head"', src)


class ReconcileTests(unittest.TestCase):
    def _reconcile(self, prs, scores, refused=None):
        calls = []
        refused = refused or {}

        def fake_gh(a):
            if a[:2] == ["pr", "list"] and "open" in a:
                return run(__import__("json").dumps(
                    [{"number": n, "labels": [{"name": l} for l in labs]} for n, labs in prs.items()]))
            return run("[]")

        def fake_ok(repo, num, require_merge_first=True):
            self.assertFalse(require_merge_first)          # the reconcile asks without the label
            return (False, refused[num]) if num in refused else (True, "ok")
        with mock.patch.object(arb, "gh", side_effect=fake_gh), \
                mock.patch.object(arb, "add_label", side_effect=lambda r, n, l: calls.append(("add", n, l))), \
                mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: calls.append(("rm", n, l))), \
                mock.patch.object(bot, "_load_scores", return_value=scores), \
                mock.patch.object(bot, "auto_merge_ok_bonsai", side_effect=fake_ok), \
                mock.patch.object(bot, "AUTO_MERGE", False):
            bot.reconcile_bonsai_merge_labels("o/r")
        return calls

    def test_a_winner_auto_merge_would_refuse_does_not_take_merge_first(self):
        # #1167 (2026-09-25): its head moved past the scored commit, so auto-merge refused it; it
        # must neither win nor push the runner-up to needs-rebase for a merge that cannot happen.
        calls = self._reconcile(
            {1167: ["eval-bonsai:XL", "bonsai-merge-first"], 1168: ["eval-bonsai:L"]},
            {"1167": {"delta_pct": 19.8}, "1168": {"delta_pct": 10.8}},
            refused={1167: "head 84a574d3f is not the commit last scored (901cef64e)"})
        self.assertIn(("add", 1168, bot.BONSAI_MERGE_FIRST), calls)
        self.assertIn(("rm", 1167, bot.BONSAI_MERGE_FIRST), calls)
        self.assertNotIn(("add", 1168, bot.BONSAI_NEEDS_REBASE), calls)
        self.assertNotIn(("add", 1167, bot.BONSAI_NEEDS_REBASE), calls)

    def test_a_held_pr_neither_wins_nor_demotes_the_next_best(self):
        # #1154 (held, XL +35.4%) must not take merge-first from #1155 or push it to needs-rebase.
        calls = self._reconcile(
            {1154: ["eval-bonsai:XL", "hold", "bonsai-merge-first"], 1155: ["eval-bonsai:S"]},
            {"1154": {"delta_pct": 35.4}, "1155": {"delta_pct": 4.0}})
        self.assertIn(("add", 1155, bot.BONSAI_MERGE_FIRST), calls)
        self.assertIn(("rm", 1154, bot.BONSAI_MERGE_FIRST), calls)       # stale label cleared
        self.assertNotIn(("add", 1155, bot.BONSAI_NEEDS_REBASE), calls)
        self.assertNotIn(("add", 1154, bot.BONSAI_MERGE_FIRST), calls)

    def test_without_a_hold_the_best_still_wins(self):
        calls = self._reconcile({1154: ["eval-bonsai:XL"], 1155: ["eval-bonsai:S"]},
                                {"1154": {"delta_pct": 35.4}, "1155": {"delta_pct": 4.0}})
        self.assertIn(("add", 1154, bot.BONSAI_MERGE_FIRST), calls)
        self.assertIn(("add", 1155, bot.BONSAI_NEEDS_REBASE), calls)

    def test_a_merge_first_holder_with_no_speedup_tier_left_loses_it(self):
        # Its head moved (the tier was dropped) or a re-measure found none: the label used to stay,
        # exempting it from every close, next to the next winner's.
        calls = self._reconcile({1154: ["bonsai-merge-first"], 1155: ["eval-bonsai:none", "bonsai-merge-first"],
                                 1156: ["eval-bonsai:S"]},
                                {"1156": {"delta_pct": 4.0}})
        self.assertIn(("rm", 1154, bot.BONSAI_MERGE_FIRST), calls)
        self.assertIn(("rm", 1155, bot.BONSAI_MERGE_FIRST), calls)
        self.assertIn(("add", 1156, bot.BONSAI_MERGE_FIRST), calls)

    def test_a_stale_winner_the_selection_would_never_re_measure_loses_its_place(self):
        stale = arb.STALE_MAIN_PREFIX + "111111111, main is now 222222222 — re-measured before it may merge"
        with mock.patch.object(arb, "greenlight_status", return_value=("no-bench", "table edited")):
            calls = self._reconcile({1154: ["eval-bonsai:XL", "bonsai-merge-first"], 1155: ["eval-bonsai:S"]},
                                    {"1154": {"delta_pct": 35.4}, "1155": {"delta_pct": 4.0}},
                                    refused={1154: stale})
        self.assertIn(("rm", 1154, bot.BONSAI_MERGE_FIRST), calls)
        self.assertIn(("add", 1155, bot.BONSAI_MERGE_FIRST), calls)

    def test_a_lone_stale_winner_the_selection_would_skip_is_demoted(self):
        stale = arb.STALE_MAIN_PREFIX + "111111111, main is now 222222222 — re-measured before it may merge"
        with mock.patch.object(arb, "greenlight_status", return_value=("no-bench", "table edited")):
            calls = self._reconcile({1154: ["eval-bonsai:XL", "bonsai-merge-first"]},
                                    {"1154": {"delta_pct": 35.4}}, refused={1154: stale})
        self.assertIn(("rm", 1154, bot.BONSAI_MERGE_FIRST), calls)
        self.assertNotIn(("add", 1154, bot.BONSAI_MERGE_FIRST), calls)

    def test_a_winner_waiting_only_for_a_re_measure_keeps_its_place(self):
        # Demoted, it was stale-closed the next round although the bot itself owed it a measurement.
        stale = arb.STALE_MAIN_PREFIX + "111111111, main is now 222222222 — re-measured before it may merge"
        calls = self._reconcile({1154: ["eval-bonsai:XL", "bonsai-merge-first"]},
                                {"1154": {"delta_pct": 35.4}}, refused={1154: stale})
        self.assertNotIn(("rm", 1154, bot.BONSAI_MERGE_FIRST), calls)
        self.assertIn(("add", 1154, bot.BONSAI_MERGE_FIRST), calls)

    def test_a_pr_that_can_merge_now_outranks_one_waiting_for_its_re_measure(self):
        stale = arb.STALE_MAIN_PREFIX + "111111111, main is now 222222222 — re-measured before it may merge"
        calls = self._reconcile({1154: ["eval-bonsai:XL", "bonsai-merge-first"], 1155: ["eval-bonsai:S"],
                                 1156: ["eval-bonsai:M"]},
                                {"1154": {"delta_pct": 35.4}, "1155": {"delta_pct": 4.0}, "1156": {"delta_pct": 9.0}},
                                refused={1154: stale, 1156: stale})
        self.assertIn(("add", 1155, bot.BONSAI_MERGE_FIRST), calls)
        self.assertIn(("rm", 1154, bot.BONSAI_MERGE_FIRST), calls)
        # With the stale ones the only candidates, nobody is sent to rebase for a merge that waits.
        calls = self._reconcile({1154: ["eval-bonsai:XL"], 1156: ["eval-bonsai:M"]},
                                {"1154": {"delta_pct": 35.4}, "1156": {"delta_pct": 9.0}},
                                refused={1154: stale, 1156: stale})
        self.assertIn(("add", 1154, bot.BONSAI_MERGE_FIRST), calls)
        self.assertNotIn(("add", 1156, bot.BONSAI_NEEDS_REBASE), calls)

class DeclarationAndSiblingGuardTests(unittest.TestCase):
    TEMPLATE = open(bot.os.path.join(bot.ROOT, ".github", "PULL_REQUEST_TEMPLATE.md")).read()

    def test_bonsai_only_prs_are_left_to_this_bot(self):
        body = self.TEMPLATE.replace("- [ ] **Ternary-Bonsai-2-27B**", "- [x] **Ternary-Bonsai-2-27B**")
        self.assertEqual(arb.declared_models(body), {"bonsai"})
        self.assertIsNone(arb.model_skip_reason(body, "bonsai"))
        self.assertIsNotNone(arb.model_skip_reason(body, "muse"))
        self.assertIsNotNone(arb.model_skip_reason(body, "qwen38"))
        shared = self.TEMPLATE.replace("- [ ] **Shared", "- [x] **Shared")
        self.assertEqual(arb.declared_models(shared), {"shared"})
        self.assertIsNone(arb.model_skip_reason(self.TEMPLATE, "bonsai"))  # nothing ticked

    def test_sibling_bots_guard_ternary_bonsai(self):
        for mod, script in ((muse_bot, muse_bot._remote_script("pull/1/head")),
                            (qwen38_bot, qwen38_bot._remote_script("pull/1/merge", role="pr"))):
            self.assertIn('bench_sweep_run "$BONSAI_GUARD_GGUF" 128 128 5 32768 5', script, mod.__name__)
            self.assertIn("GUARDBN_UNAVAILABLE", script, mod.__name__)
            p = mod._parse_remote("GUARDBN 128 99.1 2028.0\nGUARDBN 32768 89.0 6500.0\n")
            self.assertEqual(p["guardbn"][128]["prefill"], 2028.0)
            regressed = mod._parse_remote("GUARDBN 128 99.1 1500.0\nGUARDBN 32768 89.0 6500.0\n")
            ok, problems = mod.check_bonsai_guard(regressed, p)
            self.assertFalse(ok, mod.__name__)
            self.assertIn("ternary-bonsai prefill@128", problems[0])


class ServeVerdictTests(unittest.TestCase):
    """bonsai_regression.py's serve check decides on 2 of up to 3 conclusive trials (2026-09-26):
    a build equal to main failed a single trial in 2 of 8 runs on the eval box."""
    def _verdict(self, outcomes):
        import bonsai_regression as reg
        seq = iter(outcomes)
        return reg.serve_verdict(lambda: next(seq))

    def test_a_first_pass_is_a_pass(self):
        self.assertEqual(self._verdict([("pass", "")])[0], "pass")

    def test_one_failure_is_retried_and_does_not_convict(self):
        out, detail = self._verdict([("fail", "row served differently"), ("pass", ""), ("pass", "")])
        self.assertEqual(out, "pass")
        self.assertIn("2 of 3", detail)

    def test_two_failures_convict(self):
        self.assertEqual(self._verdict([("fail", "x"), ("fail", "x")])[0], "fail")
        self.assertEqual(self._verdict([("fail", "x"), ("pass", ""), ("fail", "x")])[0], "fail")

    def test_non_deterministic_baselines_never_convict(self):
        import bonsai_regression as reg
        out, _ = self._verdict([("inconclusive", "baselines differ")] * reg.SERVE_MAX_TRIALS)
        self.assertEqual(out, "inconclusive")
        self.assertEqual(self._verdict([("inconclusive", "b"), ("pass", "")])[0], "pass")
        self.assertEqual(self._verdict([("fail", "x")] + [("inconclusive", "b")] * 4)[0], "inconclusive")

    def test_a_check_that_raises_is_a_named_failure_not_a_crash(self):
        import bonsai_regression as reg
        argv = ["bonsai_regression.py", "--build", "/nonexistent/build/runtime", "--skip", "score,generate,serve"]
        with mock.patch.object(reg.sys, "argv", argv), \
                mock.patch.object(reg, "check_tensors", side_effect=ModuleNotFoundError("No module named 'safetensors'")), \
                mock.patch("builtins.print") as p:
            self.assertEqual(reg.main(), 1)
        printed = "\n".join(" ".join(str(x) for x in c.args) for c in p.call_args_list)
        self.assertIn("tensors: raised ModuleNotFoundError", printed)
        self.assertIn("FAILED:", printed)


class RegressionGatingTests(unittest.TestCase):
    """Gate 2c per check: a check main also fails cannot reject; the other checks still gate."""
    def _main(self, reg_ok=True, reg_why=()):
        with mock.patch.object(bot, "_ssh_run_resilient",
                               return_value=run(box_stdout(role="main", reg_ok=reg_ok, reg_why=reg_why))):
            m = bot.measure_main_baseline("h", 1)
        self.assertTrue(m["ok"], m)
        return m

    def test_a_serve_failure_on_main_no_longer_ungates_the_other_checks(self):
        main = self._main(reg_ok=False, reg_why=["serve: folded row served differently alone and after a batch"])
        res = evaluate(box_stdout(reg_ok=False, reg_why=["tensors: blk.0.ffn_gate.weight cosine 0.1200 below 0.8"]),
                       main=main)
        self.assertEqual(res["label"], "REJECT")
        self.assertEqual(res["bonsaireg_gated_fail"], ["tensors"])
        self.assertIn("cosine", res["reason"])

    def test_a_check_main_also_fails_is_not_gated(self):
        main = self._main(reg_ok=False, reg_why=["serve: folded row served differently alone and after a batch"])
        res = evaluate(box_stdout(reg_ok=False, reg_why=["serve: folded row served differently alone and after a batch"]),
                       main=main)
        self.assertEqual(res["label"], "none")
        self.assertIn("not gated for serve", bot.format_comment("a" * 40, res))

    def test_the_prs_own_serve_failure_rejects_when_main_passes(self):
        res = evaluate(box_stdout(reg_ok=False, reg_why=["serve: folded row served differently alone and "
                                                         "after a batch (2 of 2 conclusive trials)"]))
        self.assertEqual(res["label"], "REJECT")
        self.assertIn("FAILED", bot.format_comment("a" * 40, res))
        # Judged over two rounds like the other checks that fail a few percent of sound builds.
        self.assertEqual(res["strike_key"], "serve")

    def test_the_serve_check_alone_is_a_strike_but_with_another_check_it_rejects(self):
        both = evaluate(box_stdout(reg_ok=False, reg_why=["serve: folded row served differently",
                                                          "tensors: blk.0.ffn_gate.weight cosine 0.1200"]))
        self.assertEqual(both["label"], "REJECT")
        self.assertIsNone(both.get("strike_key"))

    def test_a_serve_check_that_raised_stands_for_both_paths(self):
        # "serve: raised X" names no path; it must still cancel against main's per-path failure.
        self.assertEqual(bot._reg_failed_checks(False, ["serve: raised TimeoutError"]),
                         {"serve:folded", "serve:native"})
        main = self._main(reg_ok=False, reg_why=["serve: raised TimeoutError"])
        res = evaluate(box_stdout(reg_ok=False, reg_why=["serve: folded row served differently alone"]), main=main)
        self.assertNotEqual(res["label"], "REJECT")

    def test_a_regression_script_that_did_not_complete_on_main_gates_nothing(self):
        main = self._main(reg_ok=False)                   # no FAILED: lines -> "*"
        res = evaluate(box_stdout(reg_ok=False, reg_why=["tensors: x"]), main=main)
        self.assertNotEqual(res["label"], "REJECT")
        self.assertIn("did not complete on main", bot.format_comment("a" * 40, res))

    def test_notes_are_shown_not_gated(self):
        res = evaluate(box_stdout(extra="BONSAIREG_NOTE serve: folded inconclusive -- baselines differ\n"))
        self.assertEqual(res["label"], "none")
        self.assertIn("inconclusive", bot.format_comment("a" * 40, res))


class BuildFailureTests(unittest.TestCase):
    """#1163 (2026-09-25): its verdict showed 80 lines of `ptxas info` resource reports and none of
    the actual `error : Feature 'mbarrier.try_wait.parity' requires .target sm_90` lines."""
    LOG = "\n".join(
        ["ptxas info    : Compiling entry function 'k' for 'sm_89'",
         "ptxas info    : Used 38 registers, used 1 barriers",
         "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads",
         "ptxas /t/prefill_moe_q.compute_89.ptx, line 649; error   : Feature '.op_restrict' requires .target sm_90 or higher",
         "ptxas fatal   : Ptx assembly aborted due to errors",
         "gmake[3]: *** [x.make:227: prefill_moe_q.cu.o] Error 255"]
        + ["ptxas info    : Used 24 registers"] * 120)

    def _bash(self, script):
        import subprocess
        return subprocess.run(["bash", "-c", arb.BUILD_FAILURE_SH + script], capture_output=True, text=True)

    def test_report_shows_the_real_errors_first(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
            f.write(self.LOG)
        self.addCleanup(os.remove, f.name)
        r = self._bash(f'report_build_failure "{f.name}"')
        errs = r.stderr.split("--- end of the build log")[0]
        self.assertIn("requires .target sm_90", errs)
        self.assertIn("Error 255", errs)
        self.assertNotIn("ptxas info", r.stderr)
        self.assertEqual(bot._crash_reason("", r.stderr).split(": ", 1)[1][:4], "ptxa")
        self.assertIn("requires .target sm_90", bot._crash_reason("", r.stderr))

    def test_box_faults_are_recognised(self):
        import tempfile
        for line, fault in (("nvcc error   : 'cicc' died due to signal 9 (Kill signal)", True),
                            ("c++: fatal error: Killed signal terminated program cc1plus", True),
                            ("failed to build archive: Bad address (os error 14)", True),
                            ("/tmp/x: No space left on device", True),
                            ("x.cu(3): error: identifier \"y\" is undefined", False)):
            with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
                f.write(line + "\n")
            self.addCleanup(os.remove, f.name)
            self.assertEqual(self._bash(f'build_box_fault "{f.name}" >/dev/null').returncode == 0, fault, line)

    def test_first_build_error_prefers_the_compiler_over_make(self):
        self.assertIn("error   : Feature", arb.first_build_error(self.LOG.splitlines()))
        self.assertIn("Error 2", arb.first_build_error(["gmake: *** [all] Error 2", "ptxas info : x"]))
        self.assertIsNone(arb.first_build_error(["-Werror is set", "error_code = 3", "3 errors"]))

    def test_the_verdict_excerpt_starts_at_the_marker_not_the_end_of_the_log(self):
        stderr = "noise\n" * 50 + "BUILD_FAILED — errors in the build log:\nx.cu(1): error: y\n" + "tail line\n" * 400
        res = evaluate("", rc=1, stderr=stderr)
        self.assertTrue(res["log"].startswith("BUILD_FAILED"))
        self.assertIn("x.cu(1): error: y", bot.format_comment("a" * 40, res))

    def test_the_remote_script_retries_a_box_fault_before_blaming_the_pr(self):
        s = bot._remote_script("pull/1/head", role="pr", onto="b" * 40)
        self.assertIn("report_build_failure()", s)
        self.assertIn('build_targets 4', s)
        self.assertIn("RETRYABLE_INFRA_FAILURE build:", s)
        self.assertNotIn('tail -80 "$TMPDIR/build.log"', s)
        self.assertTrue(bot._is_infra_failure("", "RETRYABLE_INFRA_FAILURE build: died due to signal 9"))


class MeasuredCommitTests(unittest.TestCase):
    def test_the_built_tip_is_recorded_when_the_pr_moved(self):
        listed, built = "9" * 40, "8" * 40
        self.assertEqual(arb.measured_commit(listed, {"pr_tip": built}), (built, True))
        self.assertEqual(arb.measured_commit(listed, {"pr_tip": listed}), (listed, False))
        self.assertEqual(arb.measured_commit(listed, {"pr_tip": "8888888"}), (listed, False))   # short: unusable
        self.assertEqual(arb.measured_commit(listed, {}), (listed, False))

    def test_the_box_reports_the_full_tip(self):
        s = arb.merged_checkout_script("pull/1/head", "b" * 40)
        self.assertIn('echo "PR_TIP $(git rev-parse "$PR_TIP")"', s)
        self.assertNotIn('PR_TIP $(git rev-parse --short', s)

    def test_a_failed_run_carries_the_tip_it_fetched(self):
        res = evaluate(f"PR_TIP {'c' * 40}\n", rc=1, stderr="BUILD_FAILED — x\nx.cu(1): error: y")
        self.assertEqual(res["pr_tip"], "c" * 40)

    def test_polaris_attests_the_measured_tip(self):
        seen = []
        with mock.patch.object(bot, "POLARIS_ENABLED", True), \
                mock.patch.object(bot, "ssh_run", side_effect=lambda *a, **k: seen.append(a[2]) or run("", rc=1)):
            bot.collect_polaris_attestation("h", 1, {"pr_tip": "d" * 40}, "pull/1/head")
        self.assertIn(f"git checkout -qf {'d' * 40}", seen[0])


class MarkerTrustTests(unittest.TestCase):
    def test_a_marker_pasted_by_the_author_does_not_count(self):
        marker = f'<!-- sparkinfer-bonsai-eval:{bot.EVAL_SCHEMA_VERSION}:{"e" * 40} {{"label":"XL"}} -->\n' \
                 "## sparkinfer bonsai auto-eval"
        comments = {"comments": [{"body": marker, "authorAssociation": "NONE"},
                                 {"body": marker.replace("e" * 40, "f" * 40), "authorAssociation": "MEMBER"},
                                 {"body": marker.replace("e" * 40, "0" * 40)}]}
        with mock.patch.object(arb, "gh", return_value=run(__import__("json").dumps(comments))):
            done = bot.bonsai_evaluated_commits("o/r", 1)
        self.assertEqual(done, {"f" * 40, "0" * 40})


class StaleCloseTests(unittest.TestCase):
    def _pr(self, num, body="", labels=(), draft=False):
        return {"number": num, "body": body, "isDraft": draft, "labels": [{"name": l} for l in labels]}

    def test_only_idle_prs_routed_to_this_model_and_unprotected_close(self):
        muse_only = TEMPLATE.replace("- [ ] **Muse Glimmer**", "- [x] **Muse Glimmer**")
        prs = [self._pr(1, BONSAI_ONLY_BODY), self._pr(2, muse_only), self._pr(3, TEMPLATE),
               self._pr(4, BONSAI_ONLY_BODY, labels=["qwen38-merge-first"]),
               self._pr(5, BONSAI_ONLY_BODY, labels=["hold"]), self._pr(6, BONSAI_ONLY_BODY, draft=True)]
        calls = []
        with mock.patch.object(bot, "_pr_last_activity_ts", return_value=0.0), \
                mock.patch.object(arb.AuthorWaitClock, "since", _LONG_WAITED), \
                mock.patch.object(bot, "bonsai_evaluated_commits", return_value=set()), \
                mock.patch.object(arb, "gh", side_effect=lambda a: calls.append(a) or run("")):
            closed = bot.close_stale_bonsai_prs("o/r", prs)
        self.assertEqual(closed, {1, 3})
        self.assertTrue(any("Ternary-Bonsai-2-27B eval queue" in " ".join(c) for c in calls))

    def test_a_greenlit_pr_waiting_for_its_first_verdict_stays_open(self):
        prs = [dict(self._pr(1, BONSAI_ONLY_BODY), headRefOid="a" * 40),
               dict(self._pr(2, BONSAI_ONLY_BODY), headRefOid="b" * 40)]
        with mock.patch.object(bot, "_pr_last_activity_ts", return_value=0.0), \
                mock.patch.object(arb.AuthorWaitClock, "since", _LONG_WAITED), \
                mock.patch.object(bot, "bonsai_evaluated_commits", return_value={"b" * 40}), \
                mock.patch.object(arb, "greenlight_status", return_value=("ok", "claims a gain")), \
                mock.patch.object(arb, "gh", return_value=run("")):
            self.assertEqual(bot.close_stale_bonsai_prs("o/r", prs), {2})

    def test_the_shared_rule_protects_every_bots_merge_first(self):
        for lab in ("merge-first", "bonsai-merge-first", "qwen38-merge-first", "museglimmer-merge-first"):
            self.assertEqual(arb.stale_close_skip_reason(self._pr(1, labels=[lab]), "qwen38"), "merge-first")
        self.assertIsNotNone(arb.stale_close_skip_reason(self._pr(1, BONSAI_ONLY_BODY), "qwen38"))
        self.assertIsNone(arb.stale_close_skip_reason(self._pr(1, BONSAI_ONLY_BODY), "bonsai"))


class SelectionTests(unittest.TestCase):
    def _main(self, argv, labels=(), draft=False):
        prs = [{"number": 5, "title": "t", "labels": [{"name": l} for l in labels], "isDraft": draft,
                "headRefOid": "a" * 40, "headRefName": "b", "mergeable": "MERGEABLE",
                "author": {"login": "dev"}, "body": BONSAI_ONLY_BODY, "files": [{"path": "kernels/x.cu"}]}]
        with mock.patch.object(bot.sys, "argv", ["pr_bonsai_bot.py"] + argv), \
                mock.patch.object(arb, "gh", return_value=run(__import__("json").dumps(prs))), \
                mock.patch.object(arb, "load_denylist", return_value=set()), \
                mock.patch.object(arb, "pr_involved_logins", return_value=set()), \
                mock.patch.object(bot, "bonsai_evaluated_commits", return_value=set()), \
                mock.patch.object(bot, "close_stale_bonsai_prs", return_value=set()), \
                mock.patch.object(bot, "reconcile_bonsai_merge_labels"), \
                mock.patch("builtins.print") as p:
            bot.main()
        return "\n".join(str(c.args[0]) for c in p.call_args_list if c.args)

    def test_hold_and_drafts_are_honoured_by_any_run_that_posts(self):
        self.assertIn("hold — skip", self._main(["--only-prs", "5", "--dry-run"], labels=["hold"]))
        self.assertNotIn("would evaluate", self._main(["--only-prs", "5", "--dry-run"], draft=True))

    def test_a_report_only_run_may_still_measure_them_by_name(self):
        out = self._main(["--only-prs", "5", "--dry-run", "--no-post"], labels=["hold"])
        self.assertIn("would evaluate: #5", out)

    STALE = "scored against main 999999999, main is now ccccccccc — re-measured before it may merge"

    def _select(self, labels, scores, main_now, evaluated=("a" * 40,), argv=("--only-prs", "5"),
                gate=(False, STALE)):
        prs = [{"number": 5, "title": "t", "labels": [{"name": l} for l in labels], "isDraft": False,
                "headRefOid": "a" * 40, "headRefName": "b", "mergeable": "MERGEABLE",
                "author": {"login": "dev"}, "body": BONSAI_ONLY_BODY, "files": [{"path": "kernels/x.cu"}]}]
        removed = []
        with mock.patch.object(bot.sys, "argv", ["pr_bonsai_bot.py", *argv]), \
                mock.patch.object(arb, "gh", return_value=run(json.dumps(prs))), \
                mock.patch.object(arb, "current_main_sha", return_value=main_now), \
                mock.patch.object(arb, "load_denylist", return_value=set()), \
                mock.patch.object(arb, "pr_involved_logins", return_value=set()), \
                mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: removed.append(l)), \
                mock.patch.object(arb, "sync_generic_eval_label"), \
                mock.patch.object(arb, "greenlight_status", return_value=("ok", "x")), \
                mock.patch.object(bot, "bonsai_evaluated_commits",
                                  return_value=None if evaluated is None else set(evaluated)), \
                mock.patch.object(bot, "auto_merge_ok_bonsai", return_value=gate), \
                mock.patch.object(arb, "labels_on", return_value=set()), \
                mock.patch.object(bot, "_load_scores", return_value=scores), \
                mock.patch.object(bot, "close_stale_bonsai_prs", return_value=set()), \
                mock.patch.object(bot, "resolve_ssh", side_effect=RuntimeError("no box in tests")), \
                mock.patch.object(bot, "reconcile_bonsai_merge_labels"), \
                mock.patch("builtins.print") as p:
            bot.main()
        return "\n".join(str(c.args[0]) for c in p.call_args_list if c.args), removed

    def test_a_merge_candidate_scored_against_an_older_main_is_measured_again(self):
        entry = {"commit": "a" * 40, "label": "XL", "pass": True, "onto": "9" * 40}
        out, _ = self._select(["eval-bonsai:XL"], {"5": entry}, "c" * 40)
        self.assertIn("scored against an older main — re-measuring", out)
        out, _ = self._select(["eval-bonsai:XL"], {"5": dict(entry, onto="c" * 40)}, "c" * 40)
        self.assertIn("already bonsai-evaluated — skip", out)
        out, _ = self._select(["eval-bonsai:none"], {"5": dict(entry, label="none")}, "c" * 40)
        self.assertIn("already bonsai-evaluated — skip", out)
        # The merge gate refuses it for something a re-measure cannot change: not measured again.
        out, _ = self._select(["eval-bonsai:XL"], {"5": entry}, "c" * 40,
                              gate=(False, "touches protected paths: .github/x.yml"))
        self.assertIn("already bonsai-evaluated — skip", out)

    def test_a_pr_whose_comments_could_not_be_read_is_left_alone(self):
        out, removed = self._select(["eval-bonsai:XL"], {}, "c" * 40, evaluated=None)
        self.assertIn("GitHub did not return its comments", out)
        self.assertEqual(removed, [])

    def test_a_tier_from_an_older_head_is_dropped_before_anything_else(self):
        out, removed = self._select(["eval-bonsai:XL", "eval-qwen38:S"], {}, "c" * 40, evaluated=("f" * 40,))
        self.assertEqual(removed, ["eval-bonsai:XL"])
        self.assertIn("dropped the old eval-bonsai label", out)
        # A report-only run changes nothing on the PR.
        _, removed = self._select(["eval-bonsai:XL"], {}, "c" * 40, evaluated=("f" * 40,),
                                  argv=("--only-prs", "5", "--no-post"))
        self.assertEqual(removed, [])


class ReviewFixTests(TempStateMixin, unittest.TestCase):
    """Fixes from the pre-merge review of the 2026-09-26 change."""

    def _trial(self, answers, exited=False):
        """Run one real _serve_trial against a fake server: `answers` are what _chat receives."""
        import bonsai_regression as reg
        seq = iter(answers)
        srv = mock.Mock(pid=1)
        srv.poll.return_value = 1 if exited else None
        a = types.SimpleNamespace(server="s", model="m", tokenizer="t")
        with mock.patch.object(reg.subprocess, "Popen", return_value=srv), \
                mock.patch.object(reg.urllib.request, "urlopen"), \
                mock.patch.object(reg, "_chat", side_effect=lambda port, p, n, out: out.append(next(seq))), \
                mock.patch.object(reg.os, "killpg"), mock.patch.object(reg.os, "getpgid", return_value=1), \
                mock.patch.object(reg.time, "sleep"), mock.patch.object(reg, "_wait_gpu_clear"), \
                mock.patch("builtins.print"):
            # 2 baselines, then the decayed row and 3 short rows (thread start order).
            return reg._serve_trial(a, "folded", "")

    def test_a_server_that_dies_on_its_first_request_fails_the_trial(self):
        import bonsai_regression as reg
        # The four batched requests run on threads, so every answer after the two baselines is
        # the same value: which thread takes which one does not matter.
        dead = f"{reg.FAILED_PREFIX}Connection refused>"
        self.assertEqual(self._trial([dead] * 6)[0], "fail")                  # used to pass
        self.assertEqual(self._trial(["x"] * 6)[0], "pass")
        self.assertEqual(self._trial(["x"] * 6, exited=True)[0], "fail")
        self.assertEqual(self._trial(["x", "x"] + [dead] * 4)[0], "fail")

    def test_serve_is_gated_per_path(self):
        # main's native path flakes; the PR breaks the folded path: still gated.
        with mock.patch.object(bot, "_ssh_run_resilient", return_value=run(box_stdout(
                role="main", reg_ok=False, reg_why=["serve: native row served differently alone and after a batch"]))):
            main = bot.measure_main_baseline("h", 1)
        res = evaluate(box_stdout(reg_ok=False, reg_why=["serve: folded row served differently alone and after a batch"]),
                       main=main)
        self.assertEqual(res["label"], "REJECT")
        self.assertEqual(res["bonsaireg_gated_fail"], ["serve:folded"])

    def test_a_killed_regression_run_is_judged_over_two_rounds(self):
        stdout = box_stdout(reg_ok=False, reg_why=["did not complete (exit 137): Killed"],
                            extra="BONSAIREG_EXIT 137\n")
        res = evaluate(stdout)
        self.assertEqual(res["label"], "REJECT")
        self.assertEqual(res["strike_key"], "reg")
        self.assertEqual(ApplyResultTests._apply(self, res, commit="a" * 40), [])     # strike 1
        self.assertIn(["add", "eval-bonsai:REJECT"], ApplyResultTests._apply(self, res, commit="a" * 40))
        # A run that failed a NAMED check is gated at once, as before.
        res = evaluate(box_stdout(reg_ok=False, reg_why=["tensors: x cosine 0.1"], extra="BONSAIREG_EXIT 1\n"))
        self.assertIsNone(res["strike_key"])

    def test_a_recurring_box_fault_build_is_charged_after_three_rounds(self):
        res = evaluate("", rc=1, stderr="build hit a box-side fault (x) -- rebuilding with -j4\n"
                                        "RETRYABLE_INFRA_FAILURE build: died due to signal 9")
        self.assertTrue(res["retry"])
        self.assertEqual(res["strike_key"], "build-box")
        for _ in range(bot.BOX_FAULT_STRIKES - 1):
            self.assertEqual(ApplyResultTests._apply(self, res, commit="a" * 40), [])
        calls = ApplyResultTests._apply(self, res, commit="a" * 40, autoclose=True)
        self.assertIn(["add", "eval-bonsai:REJECT"], calls)
        self.assertFalse(any(c[:2] == ["pr", "close"] for c in calls))   # a failed run never closes

    def test_a_run_killed_at_the_ssh_limit_is_a_bounded_box_fault(self):
        import subprocess
        hang = arb.exception_result(subprocess.TimeoutExpired("ssh", 7200))
        self.assertEqual((hang["retry"], hang["strike_key"]), (True, "timeout"))
        self.addCleanup(bot.clear_strikes, 1139)
        for _ in range(arb.BOX_FAULT_STRIKES - 1):                             # a box step may hang too
            self.assertEqual(ApplyResultTests._apply(self, hang, commit="a" * 40, autoclose=True), [])
        calls = ApplyResultTests._apply(self, hang, commit="a" * 40, autoclose=True)
        self.assertIn(["add", "eval-bonsai:REJECT"], calls)                  # then it is the PR's
        self.assertFalse(any(c[:2] == ["pr", "close"] for c in calls))       # a failed run never closes
        self.assertTrue(arb.exception_result(ConnectionResetError("reset"))["retry"])

    def test_the_stale_clock_is_utc_and_counts_the_prs_opening(self):
        import calendar
        info = {"commits": [{"committedDate": "2026-09-20T00:00:00Z"}], "createdAt": "2026-09-25T12:00:00Z"}
        with mock.patch.object(arb, "gh", return_value=run(__import__("json").dumps(info))):
            ts = bot._pr_last_activity_ts("o/r", 1)
        self.assertEqual(ts, calendar.timegm((2026, 9, 25, 12, 0, 0)))

    def test_an_unreadable_pr_leaves_every_label_alone(self):
        calls = []

        def fake_gh(a):
            if a[:2] == ["pr", "list"] and "open" in a:
                return run('[{"number": 5, "labels": [{"name": "eval-bonsai:XL"}, {"name": "bonsai-merge-first"}]}]')
            return run("")                                          # gh pr view returned nothing
        with mock.patch.object(arb, "gh", side_effect=fake_gh), \
                mock.patch.object(arb, "add_label", side_effect=lambda *a: calls.append(a)), \
                mock.patch.object(arb, "remove_label", side_effect=lambda *a: calls.append(a)), \
                mock.patch.object(bot, "AUTO_MERGE", False):
            bot.reconcile_bonsai_merge_labels("o/r")
        self.assertEqual(calls, [])


class BoxFaultTests(TempStateMixin, unittest.TestCase):
    """A fault that is the box's more often than the PR's is retried with nothing posted -- and, like
    a build that exhausts the compiler, charged to the PR after BOX_FAULT_STRIKES rounds at one commit."""

    def test_a_killed_speed_sweep_is_retried_then_charged(self):
        stdout = box_stdout(bonsai={}, extra="BONSAI_FAILED 137\n")
        res = evaluate(stdout)
        self.assertTrue(res["retry"], res)
        self.assertEqual(res["strike_key"], "sweep-box")
        # Any other exit is the PR's own failure, posted at once.
        self.assertFalse(evaluate(box_stdout(bonsai={}, extra="BONSAI_FAILED 1\n")).get("retry"))
        calls = []
        with mock.patch.object(arb, "gh", side_effect=lambda a: calls.append(a) or run("")), \
                mock.patch.object(arb, "labels_on", return_value=set()), \
                mock.patch.object(arb, "add_label", side_effect=lambda r, n, l: calls.append(["add", l])), \
                mock.patch.object(arb, "remove_label"), mock.patch.object(arb, "sync_generic_eval_label"), \
                mock.patch.object(bot, "upload_bonsai_eval_log"), mock.patch.object(bot, "_save_scores"), \
                mock.patch.object(bot, "AUTO_CLOSE", False):
            for _ in range(bot.BOX_FAULT_STRIKES - 1):
                bot.apply_result("o/r", 1139, "a" * 40, res)
            self.assertEqual(calls, [])
            bot.apply_result("o/r", 1139, "a" * 40, res)
        self.assertIn(["add", "eval-bonsai:REJECT"], calls)

    def test_a_guard_the_gpu_never_drained_for_or_the_oom_killer_took_is_retried(self):
        guards = {"GUARD36": (150.0, 9000.0), "GUARDUN": (55.0, 6800.0), "GUARDMG": (80.0, 2000.0)}
        for why in ("gpu", "rc=137"):
            with self.subTest(why):
                res = evaluate(box_stdout(guards=guards, extra=f"GUARDMO_FAILED {why}\n"))
                self.assertTrue(res["retry"], res)
                self.assertEqual(res["strike_key"], "guard-box")

    def test_a_guard_only_the_pr_build_failed_is_judged_over_two_rounds(self):
        guards = {"GUARD36": (150.0, 9000.0), "GUARDUN": (55.0, 6800.0), "GUARDMG": (80.0, 2000.0)}
        res = evaluate(box_stdout(guards=guards, extra="GUARDMO_FAILED rc=1\n"))
        self.assertTrue(res["ok"])
        self.assertEqual(res["label"], "REJECT")
        self.assertEqual(res["strike_key"], "guard")
        # A guard the PR measured and regressed is a REJECT at once.
        slow = dict(guards, GUARDMO=(30.0, 7000.0))
        res = evaluate(box_stdout(guards=slow))
        self.assertEqual(res["label"], "REJECT")
        self.assertIsNone(res.get("strike_key"))

    def test_only_a_hard_gate_overrides_a_box_fault(self):
        guards = {"GUARD36": (150.0, 9000.0), "GUARDUN": (55.0, 6800.0), "GUARDMG": (80.0, 2000.0)}
        slow = dict(MAIN_BONSAI)
        slow[512] = (80.0, 5200.0)                                         # -19% decode@512
        # Throughput can be faked by the same contention that kept the guard from running: deferred.
        res = evaluate(box_stdout(bonsai=slow, guards=guards, extra="GUARDMO_FAILED gpu\n"))
        self.assertTrue(res["retry"], res)
        self.assertEqual(res["strike_key"], "guard-box")
        # A failed accuracy gate cannot be: the REJECT is posted, naming what did not run.
        res = evaluate(box_stdout(guards=guards, top1=0.2, kl=2.0, extra="GUARDMO_FAILED gpu\n"))
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["label"], "REJECT")
        self.assertIn("not run this round", res["reason"])
        self.assertIn("SKIPPED", bot.format_comment("a" * 40, res))      # the guard is not "FAILED"
        # Output already wrong: a killed sweep is the PR's, said at once, not retried.
        bad = evaluate(box_stdout(bonsai={}, top1=0.2, kl=2.0, extra="BONSAI_FAILED 137\n"))
        self.assertFalse(bad.get("retry"))
        self.assertIn("accuracy gate failed", bad["reason"])
        # So is one whose prefill path already failed.
        pf_bad = {128: [(0.10, 0.9)] * 3, 1024: [(0.93, 0.012)] * 3}
        bad = evaluate(box_stdout(bonsai={}, pf=pf_bad, extra="BONSAI_FAILED 137\n"))
        self.assertFalse(bad.get("retry"))
        self.assertIn("prefill-path", bad["reason"])

    def test_several_guards_the_box_kept_from_running_are_one_round(self):
        res = evaluate(box_stdout(guards={}, extra="".join(f"{t}_FAILED gpu\n" for t in
                                                            ("GUARD36", "GUARDMO", "GUARDUN", "GUARDMG"))))
        self.assertEqual(res["strike_key"], "guard-box")
        self.assertEqual(bot.record_strike(1139, "a" * 40, "guard-box+guard-box+guard-box"), 1)
        bot.clear_strikes(1139)

    def test_every_box_fault_of_the_pr_run_counts_toward_charging_it(self):
        for stderr, key in (("RETRYABLE_INFRA_FAILURE score step killed (exit 137)", "score-box"),
                            ("RETRYABLE_INFRA_FAILURE build: cc1plus killed", "build-box"),
                            ("RETRYABLE_INFRA_FAILURE git fetch pull/1/head failed", "box"),
                            ("", "box")):                                     # a hard kill
            with self.subTest(key):
                res = evaluate("REMOTE_HEAD abc\n", rc=1, stderr=stderr)
                self.assertTrue(res["retry"], res)
                self.assertEqual(res["strike_key"], key)

    def test_soft_strikes_add_up_per_check_instead_of_resetting(self):
        self.assertEqual(bot.record_strike(7, "a" * 40, "cb"), 1)
        self.assertEqual(bot.record_strike(7, "a" * 40, "serve"), 1)
        self.assertEqual(bot.record_strike(7, "a" * 40, "cb+reg"), 2)        # cb's second round
        self.assertEqual(bot.record_strike(7, "b" * 40, "cb"), 1)            # a new commit
        bot.clear_strikes(7)

    def test_a_nan_accuracy_run_is_a_failed_run_not_a_dropped_one(self):
        p = bot._parse_remote("PFCHECK 128 1 0.90 0.018\nPFCHECK 128 2 -nan 0.1\nPFCHECK 128 3 nan nan\n")
        self.assertEqual(p["pfcheck"][128], [(0.90, 0.018)])
        self.assertEqual(p["pfcheck_failed"][128], 2)

    def test_the_verdict_records_the_main_it_was_measured_against(self):
        self.assertEqual(evaluate(box_stdout())["onto"], "b013fc9" + "0" * 33)


class RemoteScriptRetryTests(unittest.TestCase):
    def test_every_round_first_reaps_what_an_earlier_round_left_on_the_box(self):
        s = bot._remote_script("pull/1/head", role="pr", onto="b" * 40)
        self.assertIn("/tmp/sparkinfer-bot-rounds", s)
        self.assertLess(s.index("sparkinfer-bot-rounds"), s.index("git fetch"))

    def test_the_accuracy_sed_never_captures_a_nan(self):
        import re as _re
        import subprocess
        s = bot._remote_script("pull/1/head", role="pr", onto="b" * 40)
        t_sed = _re.search(r"T=\$\(sed -n '([^']*)'", s).group(1)
        k_sed = _re.search(r"K=\$\(sed -n '([^']*)'", s).group(1)
        # The formats runtime/examples/qwen3_gguf_prefill_check.cpp prints.
        for sed, line, want in ((t_sed, "TOP1  9/10 0.9000", "0.9000"), (t_sed, "TOP1  9/10 -nan", ""),
                                (t_sed, "TOP1  9/10 nan", ""),
                                (k_sed, "KL    0.01230 (mean over 10 positions)", "0.01230"),
                                (k_sed, "KL    -nan (mean over 10 positions)", ""),
                                (k_sed, "KL    nan (mean over 10 positions)", "")):
            got = subprocess.run(["sed", "-n", sed], input=line + "\n", capture_output=True, text=True).stdout.strip()
            self.assertEqual(got, want, (sed, line))

    def test_retries_are_in_the_script(self):
        s = bot._remote_script("pull/1/head", role="pr", onto="b" * 40)
        self.assertIn('echo "CB_EXIT c=$cc attempt=$attempt exit=$rc"', s)
        self.assertIn('if [ "$rc" = 124 ]; then return 1; fi', s)
        self.assertIn("BONSAICB_FAILED $CC ${CB_WHY:-run}", s)
        self.assertIn(f"for TRY in $(seq 1 {bot.PF_TRIES})", s)
        self.assertIn("BONSAIREG_NOTE", s)
        self.assertIn("head -20", s)


if __name__ == "__main__":
    unittest.main()
