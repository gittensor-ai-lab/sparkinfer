import types
import unittest
from unittest import mock

import pr_bonsai_bot as bot
import pr_eval_bot as arb
import pr_museglimmer_bot as muse_bot
import pr_qwen38_bot as qwen38_bot

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

    def test_automerge_and_autoclose_default_off_and_are_env_gated(self):
        # In the test process no env var is set, so both are off: the switches are the only thing
        # that turns them on. Production sets SPARKINFER_BONSAI_AUTOMERGE=1 in .env.eval.
        self.assertFalse(bot.AUTO_MERGE)
        self.assertFalse(bot.AUTO_CLOSE)


class AutoMergeGateTests(unittest.TestCase):
    OK_INFO = {"state": "OPEN", "isDraft": False, "labels": [{"name": "eval-bonsai:XL"},
               {"name": "bonsai-merge-first"}], "author": {"login": "dev"}, "mergeable": "MERGEABLE",
               "files": [{"path": "kernels/foo.cu"}], "headRefOid": "a" * 40}

    def _ok(self, info=None, scores=None):
        info = info or self.OK_INFO
        scores = self.OK_INFO if scores is None else scores
        s = scores if isinstance(scores, dict) and "1139" in scores else {
            "1139": {"commit": "a" * 40, "label": "XL", "pass": True}}
        with mock.patch.object(bot.arb, "gh", return_value=run(__import__("json").dumps(info))), \
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
        with mock.patch.object(bot, "auto_merge_ok_bonsai", return_value=(True, "ok")), \
                mock.patch.object(bot.arb, "gh", side_effect=fake_gh):
            self.assertTrue(bot.try_auto_merge_bonsai("o/r", 1139))
        merge = next(a for a in calls if a[:2] == ["pr", "merge"])
        self.assertIn("--match-head-commit", merge)
        self.assertEqual(merge[merge.index("--match-head-commit") + 1], "a" * 40)

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

    def test_unmeasured_concurrency_drops_the_axis(self):
        cb = dict(MAIN_CB)
        del cb[32]
        res = evaluate(box_stdout(cb=cb))
        self.assertTrue(res["ok"])
        self.assertNotIn("bonsai-cb-decode@c32", {d["dim"] for d in res["scored_dims"]})
        self.assertEqual(res["label"], "none")

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


class ApplyResultTests(unittest.TestCase):
    def _apply(self, res, autoclose=False):
        calls = []
        with mock.patch.object(arb, "gh", side_effect=lambda a: calls.append(a) or run("")), \
                mock.patch.object(arb, "labels_on", return_value=set()), \
                mock.patch.object(arb, "add_label", side_effect=lambda r, n, l: calls.append(["add", l])), \
                mock.patch.object(arb, "remove_label"), \
                mock.patch.object(arb, "sync_generic_eval_label"), \
                mock.patch.object(bot, "upload_bonsai_eval_log"), \
                mock.patch.object(bot, "_save_scores"), \
                mock.patch.object(bot, "AUTO_CLOSE", autoclose):
            bot.apply_result("o/r", 1139, "abc1234", res)
        return calls

    def test_infra_posts_nothing(self):
        self.assertEqual(self._apply({"ok": False, "retry": True, "reason": "GPU busy"}), [])

    def test_none_never_closes(self):
        res = evaluate(box_stdout())
        self.assertEqual(res["label"], "none")
        calls = self._apply(res, autoclose=True)
        self.assertIn(["add", "eval-bonsai:none"], calls)
        self.assertFalse(any(c[:2] == ["pr", "close"] for c in calls))

    def test_reject_closes_only_when_switched_on(self):
        pr = dict(MAIN_BONSAI)
        pr[512] = (90.0, 5200.0)
        res = evaluate(box_stdout(bonsai=pr))
        self.assertEqual(res["label"], "REJECT")
        self.assertFalse(any(c[:2] == ["pr", "close"] for c in self._apply(res)))
        self.assertTrue(any(c[:2] == ["pr", "close"] for c in self._apply(res, autoclose=True)))


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
        self.assertNotIn("git checkout -q origin/main -- ", s)
        self.assertIn("MERGE_CONFLICT", s)
        self.assertNotIn("pull/1145/merge", s)

    def test_main_script_does_not_merge_and_reports_its_commit(self):
        m = bot._remote_script("main", role="main")
        self.assertNotIn("merge -q --no-ff", m)
        self.assertIn('echo "REMOTE_SHA $(git rev-parse HEAD)"', m)
        self.assertIn("git checkout -q origin/main -- ", m)

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


if __name__ == "__main__":
    unittest.main()
