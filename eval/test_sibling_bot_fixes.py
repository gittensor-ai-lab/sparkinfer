"""The 2026-09-26 fixes to the Muse Glimmer and Qwen3.8 bots, checked on both.

pr_museglimmer_bot.py had no tests of its own; these run every check against both sibling bots so
the two cannot drift apart again. Nothing here touches GitHub, the box or the controller's state
files: gh and the scores file are mocked.
"""
import json
import types
import unittest
from unittest import mock

import pr_eval_bot as arb
import pr_museglimmer_bot as muse
import pr_qwen38_bot as qwen

# Nothing here may touch the controller's own state: every file the bots write goes to a temp dir.
import atexit as _atexit
import os as _os
import shutil as _shutil
import tempfile as _tempfile
_STATE = _tempfile.mkdtemp(prefix="sparkinfer-bot-tests-")
_atexit.register(_shutil.rmtree, _STATE, True)
import pr_bonsai_bot as _bonsai   # imported by some tests below: its state is redirected too
for _mod, _names in ((arb, ("INSTANCE_FILE", "PIN_FILE", "BOT_LOCK_FILE")),
                     (muse, ("STRIKES_FILE", "SCORES_FILE")), (qwen, ("STRIKES_FILE", "SCORES_FILE")),
                     (_bonsai, ("STRIKES_FILE", "SCORES_FILE"))):
    for _n in _names:
        setattr(_mod, _n, _os.path.join(_STATE, f"{_mod.__name__}.{_n}"))
arb.PINNED_INSTANCE = ""

TEMPLATE = open(arb.os.path.join(arb.os.path.dirname(arb.os.path.dirname(arb.__file__)),
                                 ".github", "PULL_REQUEST_TEMPLATE.md")).read()


def run(stdout="", rc=0, stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


MAIN = "c" * 40          # the main commit a test round measured against


# (module, eval prefix, merge-first label, needs-rebase label, auto_merge_ok, try_auto_merge,
#  reconcile, evaluated_commits, close_stale, model key)
BOTS = (
    (muse, "eval-museglimmer:", muse.MUSEGLIMMER_MERGE_FIRST, muse.MUSEGLIMMER_NEEDS_REBASE,
     muse.auto_merge_ok_museglimmer, muse.try_auto_merge_museglimmer,
     muse.reconcile_museglimmer_merge_labels, muse.museglimmer_evaluated_commits,
     muse.close_stale_museglimmer_prs, "muse"),
    (qwen, "eval-qwen38:", qwen.QWEN38_MERGE_FIRST, qwen.QWEN38_NEEDS_REBASE,
     qwen.auto_merge_ok_qwen38, qwen.try_auto_merge_qwen38,
     qwen.reconcile_qwen38_merge_labels, qwen.qwen38_evaluated_commits,
     qwen.close_stale_qwen38_prs, "qwen38"),
)


class AutoMergeGateTests(unittest.TestCase):
    """Both bots now merge only the exact commit they scored, refuse a PR another bot rejected,
    and pin the merge to that commit (pr_bonsai_bot.py has done all three since it was written)."""

    def _ok(self, mod, prefix, first, info_over=None, scored=None, require_first=True, main_now=None):
        main_now = MAIN if main_now is None else main_now
        info = {"state": "OPEN", "isDraft": False, "labels": [{"name": prefix + "XL"}, {"name": first}],
                "author": {"login": "dev"}, "mergeable": "MERGEABLE", "files": [{"path": "kernels/x.cu"}],
                "headRefOid": "a" * 40}
        info.update(info_over or {})
        scored = ({"1": {"commit": "a" * 40, "label": "XL", "pass": True, "onto": MAIN}}
                  if scored is None else scored)
        with mock.patch.object(arb, "gh", return_value=run(json.dumps(info))), \
                mock.patch.object(arb, "current_main_sha", return_value=main_now), \
                mock.patch.object(arb, "load_denylist", return_value=set()), \
                mock.patch.object(arb, "author_penalty_until", return_value=None), \
                mock.patch.object(mod, "_load_scores", return_value=scored):
            fn = mod.auto_merge_ok_museglimmer if mod is muse else mod.auto_merge_ok_qwen38
            return fn("o/r", 1, require_merge_first=require_first)

    def test_the_scored_commit_merges_and_nothing_else_does(self):
        for mod, prefix, first, *_ in BOTS:
            with self.subTest(mod.__name__):
                self.assertTrue(self._ok(mod, prefix, first)[0])
                ok, why = self._ok(mod, prefix, first, info_over={"headRefOid": "b" * 40})
                self.assertFalse(ok)
                self.assertIn("not the commit last scored", why)
                ok, why = self._ok(mod, prefix, first, scored={})
                self.assertFalse(ok)
                ok, why = self._ok(mod, prefix, first,
                                   scored={"1": {"commit": "a" * 40, "label": "none", "pass": True, "onto": MAIN}})
                self.assertFalse(ok)
                self.assertIn("recorded verdict", why)

    def test_a_verdict_against_an_older_main_is_not_merged(self):
        # Another bot merged something since: merging now ships a combination nobody measured.
        for mod, prefix, first, *_ in BOTS:
            with self.subTest(mod.__name__):
                ok, why = self._ok(mod, prefix, first, main_now="9" * 40)
                self.assertFalse(ok)
                self.assertIn("re-measured before it may merge", why)
                old = {"1": {"commit": "a" * 40, "label": "XL", "pass": True}}      # predates "onto"
                self.assertFalse(self._ok(mod, prefix, first, scored=old)[0])
                ok, why = self._ok(mod, prefix, first, main_now="")                  # GitHub silent
                self.assertEqual((ok, why), (False, arb.PR_UNREADABLE))

    def test_a_reject_from_another_bot_blocks_the_merge(self):
        for mod, prefix, first, *_ in BOTS:
            with self.subTest(mod.__name__):
                labels = [{"name": prefix + "XL"}, {"name": first}, {"name": "eval-bonsai:REJECT"}]
                ok, why = self._ok(mod, prefix, first, info_over={"labels": labels})
                self.assertFalse(ok)
                self.assertIn("REJECT from another", why)

    def test_the_ranking_question_ignores_the_label_and_a_transient_unknown(self):
        for mod, prefix, first, *_ in BOTS:
            with self.subTest(mod.__name__):
                no_first = {"labels": [{"name": prefix + "XL"}], "mergeable": "UNKNOWN"}
                self.assertTrue(self._ok(mod, prefix, first, info_over=no_first, require_first=False)[0])
                self.assertFalse(self._ok(mod, prefix, first, info_over=no_first)[0])
                conflict = {"mergeable": "CONFLICTING"}
                self.assertFalse(self._ok(mod, prefix, first, info_over=conflict, require_first=False)[0])

    def test_unreadable_github_output_refuses_rather_than_crashes(self):
        for mod, *_ in BOTS:
            with self.subTest(mod.__name__), mock.patch.object(arb, "gh", return_value=run("[]")):
                fn = mod.auto_merge_ok_museglimmer if mod is muse else mod.auto_merge_ok_qwen38
                self.assertFalse(fn("o/r", 1)[0])

    def test_the_merge_is_pinned_to_the_scored_commit(self):
        # Not to a second head lookup: a push landing between the gate and the merge (the live head
        # is b..b here) must not be what gets merged.
        for mod, _p, _f, _n, ok_fn, try_fn, *_ in BOTS:
            calls = []

            def fake_gh(a):
                calls.append(a)
                return run('{"headRefOid": "' + "b" * 40 + '"}') if a[:2] == ["pr", "view"] else run("")
            name = ok_fn.__name__
            with self.subTest(mod.__name__), mock.patch.object(mod, name, return_value=(True, "ok")), \
                    mock.patch.object(mod, "_load_scores", return_value={"1": {"commit": "a" * 40}}), \
                    mock.patch.object(arb, "gh", side_effect=fake_gh):
                self.assertTrue(try_fn("o/r", 1))
                merge = next(a for a in calls if a[:2] == ["pr", "merge"])
                self.assertEqual(merge[merge.index("--match-head-commit") + 1], "a" * 40)


class ReconcileTests(unittest.TestCase):
    def _reconcile(self, mod, recon, ok_fn, prs, scores, refused=()):
        calls = []

        def fake_gh(a):
            if a[:2] == ["pr", "list"] and "open" in a:
                return run(json.dumps([{"number": n, "labels": [{"name": l} for l in labs]}
                                       for n, labs in prs.items()]))
            return run("[]")
        with mock.patch.object(arb, "gh", side_effect=fake_gh), \
                mock.patch.object(arb, "add_label", side_effect=lambda r, n, l: calls.append(("add", n, l))), \
                mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: calls.append(("rm", n, l))), \
                mock.patch.object(mod, "_load_scores", return_value=scores), \
                mock.patch.object(mod, ok_fn.__name__,
                                  side_effect=lambda r, n, require_merge_first=True:
                                  (False, "refused") if n in refused else (True, "ok")), \
                mock.patch.object(mod, "AUTO_MERGE", False):
            recon("o/r")
        return calls

    def test_a_held_or_refused_pr_neither_wins_nor_demotes_the_next_best(self):
        for mod, prefix, first, rebase, ok_fn, _t, recon, *_ in BOTS:
            for labs, refused in ((["hold"], ()), ([], (10,))):
                with self.subTest(mod.__name__, labs=labs, refused=refused):
                    calls = self._reconcile(
                        mod, recon, ok_fn,
                        {10: [prefix + "XL", first] + labs, 11: [prefix + "S"]},
                        {"10": {"label": "XL", "delta_pct": 30.0}, "11": {"label": "S", "delta_pct": 4.0}},
                        refused=refused)
                    self.assertIn(("add", 11, first), calls)
                    self.assertIn(("rm", 10, first), calls)
                    self.assertNotIn(("add", 11, rebase), calls)

    def test_without_one_the_best_still_wins(self):
        for mod, prefix, first, rebase, ok_fn, _t, recon, *_ in BOTS:
            with self.subTest(mod.__name__):
                calls = self._reconcile(
                    mod, recon, ok_fn, {10: [prefix + "XL"], 11: [prefix + "S"]},
                    {"10": {"label": "XL", "delta_pct": 30.0}, "11": {"label": "S", "delta_pct": 4.0}})
                self.assertIn(("add", 10, first), calls)
                self.assertIn(("add", 11, rebase), calls)


class InfraTests(unittest.TestCase):
    def test_a_box_fault_posts_nothing(self):
        for mod, *_ in BOTS:
            calls = []
            with self.subTest(mod.__name__), \
                    mock.patch.object(arb, "gh", side_effect=lambda a: calls.append(a) or run()), \
                    mock.patch.object(arb, "add_label", side_effect=lambda *a: calls.append(a)):
                mod.apply_result("o/r", 1, "a" * 40, {"ok": False, "retry": True, "reason": "GPU busy"})
                self.assertEqual(calls, [])

    def test_which_failures_are_the_box(self):
        q, m = qwen._is_box_fault, muse._is_box_fault
        for fn in (q, m):
            self.assertTrue(fn("", "RETRYABLE_INFRA_FAILURE git fetch pull/1/head failed"))
            self.assertTrue(fn("", "RETRYABLE_INFRA_FAILURE build: died due to signal 9"))
            self.assertTrue(fn("partial", ""))                     # the shell died, no diagnostic
            self.assertFalse(fn("", "BUILD_FAILED -- errors in the build log:\nx.cu(1): error: y"))
            self.assertFalse(fn("", "REMOTE_SCRIPT_FAILED line=9 exit=139 reason=likely segfault"))
        # Qwen3.8's own concurrency ladder failure is still reported on the PR, not retried for ever.
        self.assertFalse(q("", "RETRYABLE_INFRA_FAILURE concurrent decode failed at c=32 (see above)"))
        self.assertFalse(q("... GUARD_END", "REMOTE_SCRIPT_FAILED line=1 exit=1 reason="))

    def test_a_failed_pr_run_is_classified_and_shows_the_real_error(self):
        stderr = ("noise\n" * 40 + "BUILD_FAILED -- errors in the build log:\n"
                  "x.cu(3): error: identifier \"y\" is undefined\n" + "tail\n" * 300)
        for mod, *_ in BOTS:
            ev = mod.eval_museglimmer_on_box if mod is muse else mod.eval_qwen38_on_box
            with self.subTest(mod.__name__), \
                    mock.patch.object(mod, "_ssh_run_resilient", return_value=run(f"PR_TIP {'c' * 40}\n", 1, stderr)):
                res = ev("h", 1, "pull/1/head", {"sha": "b" * 40})
                self.assertFalse(res["retry"])
                self.assertTrue(res["log"].startswith("BUILD_FAILED"))
                self.assertIn("identifier", mod._crash_reason("", stderr))
            with self.subTest(mod.__name__, case="box"), \
                    mock.patch.object(mod, "_ssh_run_resilient",
                                      return_value=run("", 1, "RETRYABLE_INFRA_FAILURE git fetch x failed")):
                self.assertTrue(ev("h", 1, "pull/1/head", {"sha": "b" * 40})["retry"])

    def test_exceptions_go_through_the_shared_rule_and_the_baseline_is_guarded(self):
        import subprocess
        for mod, *_ in BOTS:
            with open(mod.__file__) as f:
                src = f.read()
            with self.subTest(mod.__name__):
                self.assertIn("res = arb.exception_result(e)", src)
                self.assertIn("main_result = measure_main_baseline(host, port)\n    except Exception", src)
        # A transport failure is retried; a run killed at the ssh limit is a hang, posted once with a
        # label in its marker so it is not re-run every round.
        self.assertTrue(arb.exception_result(ConnectionResetError("x"))["retry"])
        hang = arb.exception_result(subprocess.TimeoutExpired("ssh", 7200))
        self.assertFalse(hang["retry"])
        for mod in (muse, qwen):
            self.assertIn('"label":"REJECT"', mod.format_comment("a" * 40, hang), mod.__name__)

    def test_a_guard_that_measured_nothing_retries_on_muse(self):
        # Only the Ternary-Bonsai guard used to take the infra path; the ModelOpt, unsloth and
        # Qwen3.6 guards REJECTed -- and so closed -- a PR over a measurement that never happened.
        def stdout(drop=()):
            lines = ["PR_TIP " + "a" * 40, "REMOTE_SHA " + MAIN]
            lines += [f"MUSE {c} 100.0 5000.0" for c in muse.SCORED_CTXS]
            lines += [f"MUSECB {c} 500.0" for c in muse.CB_CONCS]
            lines += ["RESULT_DECODE_TPS 100.0", "RESULT_PREFILL128_PP 5000.0", "RESULT_TOP1 0.99",
                      "RESULT_KL 0.01", "ACCURACY_STAGE_DONE", "GUARD_START", "GUARD36 32768 50.0 900.0",
                      "GUARDMO 32768 60.0 7000.0", "GUARDUN 32768 55.0 6800.0",
                      "GUARDBN 128 99.0 2000.0", "GUARDBN 32768 89.0 6500.0", "GUARD_END"]
            return "\n".join(l for l in lines if not any(l.startswith(d) for d in drop)) + "\n"
        with mock.patch.object(muse, "_ssh_run_resilient", return_value=run(stdout())):
            full = muse.measure_main_baseline("h", 1)
        self.assertTrue(full["ok"], full)
        for tag, key in (("GUARDMO ", "guardmo"), ("GUARDUN ", "guardun"), ("GUARD36 ", "guard36")):
            # Main measured nothing for the guard: the round is skipped at the baseline ...
            with mock.patch.object(muse, "_ssh_run_resilient", return_value=run(stdout(drop=(tag,)))):
                self.assertFalse(muse.measure_main_baseline("h", 1)["ok"], tag)
            # ... and should such a baseline reach a PR anyway, the PR is deferred, not rejected.
            main = dict(full, **{key: {}})
            with self.subTest(tag), mock.patch.object(muse, "POLARIS_ENABLED", False), \
                    mock.patch.object(muse, "_ssh_run_resilient", return_value=run(stdout(drop=(tag,)))):
                res = muse.eval_museglimmer_on_box("h", 1, "pull/1/head", main)
                self.assertFalse(res["ok"])
                self.assertTrue(res["retry"], res.get("reason"))
        # Unchanged: main measured it and only the PR's run lost it -> fail closed (a regression).
        main = full
        with mock.patch.object(muse, "POLARIS_ENABLED", False), \
                mock.patch.object(muse, "_ssh_run_resilient", return_value=run(stdout(drop=("GUARDMO ",)))):
            res = muse.eval_museglimmer_on_box("h", 1, "pull/1/head", main)
        self.assertTrue(res["ok"])
        self.assertEqual(res["label"], "REJECT")

    def test_the_llama_cpp_reference_failing_to_build_is_the_box(self):
        s = muse._remote_script("pull/1/head")
        self.assertIn("RETRYABLE_INFRA_FAILURE llama.cpp reference build failed", s)
        self.assertIn("RETRYABLE_INFRA_FAILURE llama.cpp reference configure failed", s)
        self.assertTrue(muse._is_box_fault("", "LLAMACPP_BUILD_FAILED — errors, then tail:\nx\n"
                                               "RETRYABLE_INFRA_FAILURE llama.cpp reference build failed"))


class RemoteScriptTests(unittest.TestCase):
    def test_build_failures_report_errors_first_and_box_faults_retry(self):
        for name, s in (("muse", muse._remote_script("pull/1/head")),
                        ("qwen", qwen._remote_script("pull/1/head", role="pr", onto="b" * 40))):
            with self.subTest(name):
                self.assertIn("report_build_failure()", s)
                self.assertIn("build_targets 4", s)
                self.assertIn("RETRYABLE_INFRA_FAILURE build:", s)
                self.assertNotIn("tail -80 /tmp/mg_build.log", s)
                self.assertNotIn("tail -80 /tmp/q38_build.log", s)
                self.assertIn('echo "BUILD_FAILED', s)       # configure is guarded too

    def test_muse_measures_the_pr_merged_onto_the_baseline_and_reports_both_commits(self):
        # Like Qwen3.8 and Bonsai: the PR merged onto the exact main commit the round measured, so a
        # branch behind main is not charged with every speedup merged since it branched.
        s = muse._remote_script("pull/1/head", role="pr", onto=MAIN)
        self.assertIn(f"git checkout -qf {MAIN}", s)
        self.assertIn("MERGE_CONFLICT", s)
        self.assertIn('echo "PR_TIP $(git rev-parse "$PR_TIP")"', s)
        self.assertIn('RETRYABLE_INFRA_FAILURE git fetch pull/1/head failed', s)
        m = muse._remote_script("main", role="main")
        self.assertIn('echo "REMOTE_SHA $(git rev-parse HEAD)"', m)
        self.assertNotIn("MERGE_CONFLICT", m)
        p = muse._parse_remote(f"PR_TIP {'d' * 40}\nMERGED_ONTO cccccc\nREMOTE_SHA {'e' * 40}\n")
        self.assertEqual((p["pr_tip"], p["sha"], p["merged_onto"]), ("d" * 40, "e" * 40, "cccccc"))

    def test_a_pr_that_does_not_merge_onto_the_baseline_needs_a_rebase_not_a_verdict(self):
        for mod, prefix, first, rebase, *_ in BOTS:
            ev = mod.eval_museglimmer_on_box if mod is muse else mod.eval_qwen38_on_box
            err = "MERGE_CONFLICT aaaaaaa does not merge cleanly onto ccccccc"
            with self.subTest(mod.__name__), mock.patch.object(mod, "_ssh_run_resilient", return_value=run("", 1, err)):
                res = ev("h", 1, "pull/1/head", {"sha": MAIN})
            self.assertTrue(res.get("conflict"), res)
            calls = []
            with mock.patch.object(arb, "gh", side_effect=lambda a: calls.append(a) or run()), \
                    mock.patch.object(arb, "add_label", side_effect=lambda r, n, l: calls.append(("add", l))):
                mod.apply_result("o/r", 1, "a" * 40, res)
            self.assertEqual(calls, [("add", rebase)])

    def test_every_round_first_reaps_what_an_earlier_round_left_on_the_box(self):
        for name, s in (("muse", muse._remote_script("pull/1/head", role="pr", onto=MAIN)),
                        ("qwen", qwen._remote_script("pull/1/head", role="pr", onto=MAIN))):
            with self.subTest(name):
                self.assertIn("/tmp/sparkinfer-bot-rounds", s)
                self.assertLess(s.index("sparkinfer-bot-rounds"), s.index("git fetch"))

    def test_qwen38_gives_up_on_a_gpu_that_never_frees_as_infrastructure(self):
        s = qwen._remote_script("pull/1/head", role="pr", onto=MAIN)
        body = s[s.index("wait_gpu_clear() {"):]
        body = body[:body.index("\n}\n")]
        self.assertIn("RETRYABLE_INFRA_FAILURE GPU still holding", body)
        self.assertIn("exit 1", body)
        self.assertNotIn("-- proceeding anyway", body)

    def test_a_muse_llama_server_that_never_comes_up_is_the_box(self):
        s = muse._remote_script("pull/1/head", role="pr", onto=MAIN)
        self.assertIn("RETRYABLE_INFRA_FAILURE llama.cpp reference server never became healthy", s)

    def test_attestation_is_pinned_to_the_measured_tip(self):
        for mod, *_ in BOTS:
            seen = []
            with self.subTest(mod.__name__), mock.patch.object(mod, "POLARIS_ENABLED", True), \
                    mock.patch.object(mod, "ssh_run", side_effect=lambda *a, **k: seen.append(a[2]) or run(rc=1)):
                mod.collect_polaris_attestation("h", 1, {"pr_tip": "e" * 40}, "pull/1/head")
                self.assertIn(f"git checkout -qf {'e' * 40}", seen[0])


class SelectionTests(unittest.TestCase):
    def test_muse_skips_prs_that_change_its_ruler(self):
        # This bot builds the PR's own harness (no pin), so a PR editing it is not evaluated.
        for path in ("bench/scripts/_eval_speed.sh", "runtime/examples/qwen3_gguf_bench.cpp", "eval/x.py"):
            self.assertTrue(any(path.startswith(h) for h in muse.HARNESS_PATHS), path)
        prs = [{"number": 7, "title": "t", "labels": [], "isDraft": False, "headRefOid": "a" * 40,
                "headRefName": "b", "mergeable": "MERGEABLE", "author": {"login": "dev"}, "body": TEMPLATE,
                "files": [{"path": "bench/scripts/_eval_speed.sh"}, {"path": "kernels/x.cu"}]}]
        with mock.patch.object(muse.sys, "argv", ["pr_museglimmer_bot.py", "--only-prs", "7", "--dry-run"]), \
                mock.patch.object(arb, "gh", return_value=run(json.dumps(prs))), \
                mock.patch.object(arb, "load_denylist", return_value=set()), \
                mock.patch.object(arb, "pr_involved_logins", return_value=set()), \
                mock.patch.object(muse, "museglimmer_evaluated_commits", return_value=set()), \
                mock.patch.object(muse, "reconcile_museglimmer_merge_labels"), \
                mock.patch("builtins.print") as p:
            muse.main()
        out = "\n".join(" ".join(str(x) for x in c.args) for c in p.call_args_list)
        self.assertIn("touches the eval harness", out)
        self.assertNotIn("would evaluate", out)

    def test_markers_count_only_from_members(self):
        for mod, prefix, _f, _r, _o, _t, _rc, evaluated, *_ in BOTS:
            tag = "museglimmer" if mod is muse else "qwen38"
            marker = (f'<!-- sparkinfer-{tag}-eval:{mod.EVAL_SCHEMA_VERSION}:{"e" * 40} {{"label":"XL"}} -->\n'
                      f"## sparkinfer {tag} auto-eval")
            comments = {"comments": [{"body": marker, "authorAssociation": "NONE"},
                                     {"body": marker.replace("e" * 40, "f" * 40), "authorAssociation": "MEMBER"}]}
            with self.subTest(mod.__name__), mock.patch.object(arb, "gh", return_value=run(json.dumps(comments))):
                self.assertEqual(evaluated("o/r", 1), {"f" * 40})

    def test_stale_close_touches_only_this_models_unprotected_prs(self):
        bonsai_only = TEMPLATE.replace("- [ ] **Ternary-Bonsai-2-27B**", "- [x] **Ternary-Bonsai-2-27B**")
        for mod, _p, _f, _r, _o, _t, _rc, _e, close_stale, key in BOTS:
            prs = [{"number": 1, "body": TEMPLATE, "isDraft": False, "labels": []},
                   {"number": 2, "body": bonsai_only, "isDraft": False, "labels": []},       # #1157's case
                   {"number": 3, "body": TEMPLATE, "isDraft": False, "labels": [{"name": "bonsai-merge-first"}]},
                   {"number": 4, "body": TEMPLATE, "isDraft": True, "labels": []}]
            calls = []
            with self.subTest(mod.__name__), mock.patch.object(mod, "_pr_last_activity_ts", return_value=0.0), \
                    mock.patch.object(arb, "gh", side_effect=lambda a: calls.append(a) or run()):
                self.assertEqual(close_stale("o/r", prs), {1})
                self.assertFalse(any("push a new commit / open" in " ".join(c) for c in calls))


class ReviewFixTests(unittest.TestCase):
    """Fixes from the pre-merge review of the 2026-09-26 change."""

    def test_a_crash_in_the_prs_score_binary_is_the_prs(self):
        # An explicit `exit 1` after SCORE_FAILED fires no ERR trap: it used to read as a silent kill
        # and be retried every round with nothing posted.
        err = "SCORE_FAILED -- tail of /tmp/q38_score.err:\nCUDA error: an illegal memory access was encountered"
        self.assertFalse(qwen._is_box_fault("RESULT_DECODE128_TPS 80", err))
        self.assertIn("illegal memory access", qwen._crash_reason("", err))

    def test_qwen38_main_must_measure_every_installed_guard(self):
        stdout = "\n".join([
            "REMOTE_SHA " + "b" * 40, "RESULT_DECODE128_TPS 80", "RESULT_PREFILL128_PP 4000",
            "RESULT_PREFILL16K_PP 8000"] + [f"RESULT_CB{c}_AGG {100 * c}" for c in qwen.CB_CONCS] + [
            "GUARD36 32768 50 900", "GUARDMO 32768 60 7000", "GUARDCBMO 16 800", "GUARDCBMO 32 1000",
            "GUARDMG_UNAVAILABLE", "GUARDBN 128 99 2000", "GUARDBN 32768 89 6500", "GUARD_END"]) + "\n"
        with mock.patch.object(qwen, "_ssh_run_resilient", return_value=run(stdout)):
            self.assertTrue(qwen.measure_main_baseline("h", 1)["ok"])     # an absent checkpoint is fine
        without = stdout.replace("GUARDBN 128 99 2000\nGUARDBN 32768 89 6500\n", "GUARDBN_FAILED\n")
        with mock.patch.object(qwen, "_ssh_run_resilient", return_value=run(without)):
            m = qwen.measure_main_baseline("h", 1)
        self.assertFalse(m["ok"])
        self.assertIn("guardbn", m["reason"])

    def test_an_unreadable_pr_leaves_every_label_alone(self):
        for mod, prefix, first, _r, _o, _t, recon, *_ in BOTS:
            calls = []

            def fake_gh(a):
                if a[:2] == ["pr", "list"] and "open" in a:
                    return run(json.dumps([{"number": 5, "labels": [{"name": prefix + "XL"}, {"name": first}]}]))
                return run("")
            with self.subTest(mod.__name__), mock.patch.object(arb, "gh", side_effect=fake_gh), \
                    mock.patch.object(arb, "add_label", side_effect=lambda *a: calls.append(a)), \
                    mock.patch.object(arb, "remove_label", side_effect=lambda *a: calls.append(a)), \
                    mock.patch.object(mod, "AUTO_MERGE", False):
                recon("o/r")
                self.assertEqual(calls, [])

    def test_the_stale_clock_is_utc_and_counts_the_prs_opening(self):
        import calendar
        info = {"commits": [{"committedDate": "2026-09-20T00:00:00Z"}], "createdAt": "2026-09-25T12:00:00Z"}
        for mod in (muse, qwen):
            with self.subTest(mod.__name__), mock.patch.object(arb, "gh", return_value=run(json.dumps(info))):
                self.assertEqual(mod._pr_last_activity_ts("o/r", 1), calendar.timegm((2026, 9, 25, 12, 0, 0)))


class ClosePolicyTests(unittest.TestCase):
    """A REJECT closes. `none` closes only a PR declared for this bot's model alone that no other bot
    scored a speedup or made merge-first. Neither closes over a commit the author has replaced."""

    MUSE_ONLY = TEMPLATE.replace("- [ ] **Muse Glimmer**", "- [x] **Muse Glimmer**")
    QWEN_ONLY = TEMPLATE.replace("- [ ] **Qwen3.8-27B**", "- [x] **Qwen3.8-27B**")

    def _closed(self, mod, label, pr_body, labels=(), head_now="a" * 40):
        calls = []

        def fake_gh(a):
            calls.append(a)
            return run(json.dumps({"headRefOid": head_now})) if a[:2] == ["pr", "view"] else run()
        tag = "museglimmer" if mod is muse else "qwen38"
        res = {"ok": True, "label": label, "delta_pct": 0.4 if label == "none" else -9.0,
               "pass": label != "REJECT", "accuracy_ok": True}
        with mock.patch.object(arb, "gh", side_effect=fake_gh), \
                mock.patch.object(arb, "add_label"), mock.patch.object(arb, "remove_label"), \
                mock.patch.object(arb, "sync_generic_eval_label"), \
                mock.patch.object(arb, "labels_on_or_none", return_value=set(labels)), \
                mock.patch.object(mod, f"strip_{tag}_eval_labels"), \
                mock.patch.object(mod, f"upload_{tag}_eval_log"), \
                mock.patch.object(mod, "format_comment", return_value="the verdict comment"), \
                mock.patch.object(mod, "_load_scores", return_value={}), \
                mock.patch.object(mod, "_save_scores"):
            mod.apply_result("o/r", 1, "a" * 40, res, pr_body=pr_body)
        return any(a[:2] == ["pr", "close"] for a in calls)

    def test_the_template_names_the_models_these_tests_tick(self):
        self.assertEqual(arb.declared_models(self.MUSE_ONLY), {"muse"})
        self.assertEqual(arb.declared_models(self.QWEN_ONLY), {"qwen38"})

    def test_none_closes_only_a_pr_declared_for_this_model_alone(self):
        for mod, prefix, *_ in BOTS:
            mine = self.MUSE_ONLY if mod is muse else self.QWEN_ONLY
            other = self.QWEN_ONLY if mod is muse else self.MUSE_ONLY
            with self.subTest(mod.__name__):
                self.assertTrue(self._closed(mod, "none", mine))
                self.assertFalse(self._closed(mod, "none", TEMPLATE))           # undeclared
                self.assertFalse(self._closed(mod, "none", other))
                self.assertFalse(self._closed(mod, "none", ""))
                # Another bot measured a speedup, or made it merge-first: not this bot's to close.
                self.assertFalse(self._closed(mod, "none", mine, labels={"eval-bonsai:M"}))
                self.assertFalse(self._closed(mod, "none", mine, labels={"bonsai-merge-first"}))
                self.assertTrue(self._closed(mod, "none", mine, labels={"eval-bonsai:none"}))

    def test_a_reject_closes_whatever_was_declared(self):
        for mod, *_ in BOTS:
            with self.subTest(mod.__name__):
                self.assertTrue(self._closed(mod, "REJECT", TEMPLATE))

    def test_nothing_closes_over_a_commit_the_author_has_replaced(self):
        for mod, *_ in BOTS:
            with self.subTest(mod.__name__):
                self.assertFalse(self._closed(mod, "REJECT", TEMPLATE, head_now="b" * 40))

    def test_a_failed_run_is_recorded_for_its_commit(self):
        # Posted once with a label in the marker, so the same commit is not rebuilt and re-posted
        # every round; a new push is evaluated again.
        for mod, *_ in BOTS:
            with self.subTest(mod.__name__):
                c = mod.format_comment("a" * 40, {"ok": False, "reason": "PR run failed", "log": "x"})
                self.assertIn('"label":"REJECT"', c)
                self.assertIn("push a fix", c)


class StaleAndFreshnessTests(unittest.TestCase):
    def test_a_greenlit_pr_waiting_for_its_first_verdict_is_not_stale(self):
        for mod, _p, _f, _r, _o, _t, _rc, _e, close_stale, key in BOTS:
            prs = [{"number": 1, "body": TEMPLATE, "isDraft": False, "labels": [], "headRefOid": "a" * 40},
                   {"number": 2, "body": TEMPLATE, "isDraft": False, "labels": [], "headRefOid": "b" * 40}]
            ev = "museglimmer_evaluated_commits" if mod is muse else "qwen38_evaluated_commits"
            with self.subTest(mod.__name__), mock.patch.object(mod, "_pr_last_activity_ts", return_value=0.0), \
                    mock.patch.object(mod, ev, side_effect=lambda r, n: {"b" * 40}), \
                    mock.patch.object(arb, "greenlight_status", return_value=("ok", "claims a gain")), \
                    mock.patch.object(arb, "gh", return_value=run()):
                # #1 is queued behind the bot; #2 already has its verdict and waits on its author.
                self.assertEqual(close_stale("o/r", prs), {2})
            with self.subTest(mod.__name__, greenlit=False), \
                    mock.patch.object(mod, "_pr_last_activity_ts", return_value=0.0), \
                    mock.patch.object(mod, ev, return_value=set()), \
                    mock.patch.object(arb, "greenlight_status", return_value=("unchecked", "box not ticked")), \
                    mock.patch.object(arb, "gh", return_value=run()):
                self.assertEqual(close_stale("o/r", prs), {1, 2})

    def test_only_a_merge_candidate_scored_against_an_older_main_is_measured_again(self):
        good = {"commit": "a" * 40, "label": "XL", "pass": True, "onto": "9" * 40}
        stale = (False, arb.STALE_MAIN_PREFIX + "999999999, main is now ccccccccc — re-measured before it may merge")
        for mod, prefix, first, rebase, ok_fn, *_ in BOTS:
            with self.subTest(mod.__name__):
                def again(entry, labs=frozenset(), head="a" * 40, main_now=MAIN, gate=stale):
                    with mock.patch.object(mod, "_load_scores", return_value={"1": entry}), \
                            mock.patch.object(mod, ok_fn.__name__, return_value=gate):
                        return mod._remeasure_against_new_main("o/r", 1, head, set(labs), main_now)
                # The merge gate refuses it for something else: re-measuring cannot help.
                self.assertFalse(again(good, gate=(False, "touches protected paths: .github/x")))
                self.assertFalse(again(good, gate=(False, "author dev is under penalty")))
                self.assertTrue(again(good))
                self.assertFalse(again(dict(good, onto=MAIN)))                  # already current
                self.assertFalse(again(dict(good, label="none")))                # not a candidate
                self.assertFalse(again(dict(good, **{"pass": False})))
                self.assertFalse(again(good, head="b" * 40))                     # a new head re-runs anyway
                self.assertFalse(again(good, labs={"hold"}))
                self.assertFalse(again(good, labs={"eval-bonsai:REJECT"},
                                       gate=(False, "carries a REJECT from another eval bot")))
                self.assertFalse(again(good, main_now=""))                       # GitHub silent: no GPU spent

    def test_a_tier_from_an_older_head_is_dropped(self):
        removed = []
        with mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: removed.append(l)), \
                mock.patch.object(arb, "sync_generic_eval_label") as sync:
            self.assertTrue(arb.strip_stale_verdict_labels(
                "o/r", 1, {"eval-qwen38:XL", "eval-bonsai:S"}, "eval-qwen38:", "b" * 40, {"a" * 40}))
            self.assertEqual(removed, ["eval-qwen38:XL"])
            sync.assert_called_once()
            self.assertFalse(arb.strip_stale_verdict_labels(
                "o/r", 1, {"eval-qwen38:XL"}, "eval-qwen38:", "a" * 40, {"a" * 40}))


class BaselineTests(unittest.TestCase):
    def test_muse_skips_the_round_when_main_misses_its_own_accuracy_bar(self):
        # The gate is absolute (against llama.cpp): a main below it means the box or the reference
        # is off, and every PR would have been REJECTed -- and closed -- for it.
        lines = ["REMOTE_SHA " + MAIN] + [f"MUSE {c} 100.0 5000.0" for c in muse.SCORED_CTXS]
        lines += [f"MUSECB {c} 500.0" for c in muse.CB_CONCS]
        lines += ["RESULT_DECODE_TPS 100.0", "RESULT_PREFILL128_PP 5000.0", "RESULT_TOP1 0.40",
                  "RESULT_KL 0.90", "GUARD36 32768 50.0 900.0", "GUARDMO 32768 60.0 7000.0",
                  "GUARDUN 32768 55.0 6800.0", "GUARDBN 128 99.0 2000.0", "GUARD_END"]
        with mock.patch.object(muse, "_ssh_run_resilient", return_value=run("\n".join(lines) + "\n")):
            m = muse.measure_main_baseline("h", 1)
        self.assertFalse(m["ok"])
        self.assertIn("accuracy gate", m["reason"])

    def test_qwen38_the_optional_256k_row_is_not_a_guard(self):
        # #1113's 256k axis lives in the ModelOpt guard's dict; a PR whose 256k sweep failed near the
        # card's 32 GB used to be REJECTed -- and closed -- as a guard regression.
        main = {"guardmo": {32768: {"decode": 60.0, "prefill": 7000.0},
                            qwen.LONGCTX_CTX: {"decode": 30.0, "prefill": 3000.0}}}
        pr = {"guardmo": {32768: {"decode": 60.0, "prefill": 7000.0}}}
        self.assertTrue(qwen.check_modelopt_guard(pr, main)[0])
        pr = {"guardmo": {32768: {"decode": 40.0, "prefill": 7000.0}}}
        self.assertFalse(qwen.check_modelopt_guard(pr, main)[0])


class Round2ReviewTests(unittest.TestCase):
    """Fixes from the second review of the 2026-09-26 change."""

    def test_muse_charges_a_run_that_died_before_its_end_to_the_box(self):
        # The guards run after the accuracy stage: a shell killed there is not the PR's failure.
        self.assertTrue(muse._looks_like_hard_kill("ACCURACY_STAGE_DONE\nGUARD_START\nGUARD36 32768 5 9\n", ""))
        self.assertFalse(muse._looks_like_hard_kill("ACCURACY_STAGE_DONE\nGUARD_START\nGUARD_END\n", ""))
        self.assertTrue(muse._is_box_fault("ACCURACY_STAGE_DONE\nGUARD_START\n", ""))
        # A conflict is diagnosed, never a hard kill run a second time.
        self.assertFalse(muse._looks_like_hard_kill("", "MERGE_CONFLICT aaaaaaa does not merge cleanly onto ccccccc"))

    def test_a_box_fault_recurring_at_one_commit_is_charged_to_the_pr(self):
        for mod, *_ in BOTS:
            tag = "museglimmer" if mod is muse else "qwen38"
            calls = []
            res = {"ok": False, "retry": True, "strike_key": "box", "reason": "PR run died (hard kill)", "log": "x"}
            with self.subTest(mod.__name__), \
                    mock.patch.object(arb, "gh", side_effect=lambda a: calls.append(a) or run()), \
                    mock.patch.object(arb, "add_label", side_effect=lambda r, n, l: calls.append(("add", l))), \
                    mock.patch.object(arb, "remove_label"), mock.patch.object(arb, "sync_generic_eval_label"), \
                    mock.patch.object(mod, f"strip_{tag}_eval_labels"):
                for _ in range(arb.BOX_FAULT_STRIKES - 1):
                    mod.apply_result("o/r", 9, "a" * 40, res)
                self.assertEqual(calls, [])
                mod.apply_result("o/r", 9, "a" * 40, res)
                self.assertIn(("add", f"eval-{tag}:REJECT"), calls)
                posted = next(a for a in calls if isinstance(a, list) and a[:2] == ["pr", "comment"])
                self.assertIn("charged to the PR", " ".join(posted))
                self.assertNotIn("9", arb._load_strikes(mod.STRIKES_FILE))      # cleared once posted

    def test_qwen38_main_must_measure_the_32k_modelopt_guard_itself(self):
        stdout = "\n".join([
            "REMOTE_SHA " + "b" * 40, "RESULT_DECODE128_TPS 80", "RESULT_PREFILL128_PP 4000",
            "RESULT_PREFILL16K_PP 8000"] + [f"RESULT_CB{c}_AGG {100 * c}" for c in qwen.CB_CONCS] + [
            "GUARD36 32768 50 900", "GUARDMO_FAILED", f"GUARDMO {qwen.LONGCTX_CTX} 30 3000",
            "GUARDCBMO 16 800", "GUARDCBMO 32 1000", "GUARDMG_UNAVAILABLE",
            "GUARDBN 128 99 2000", "GUARDBN 32768 89 6500", "GUARD_END"]) + "\n"
        with mock.patch.object(qwen, "_ssh_run_resilient", return_value=run(stdout)):
            m = qwen.measure_main_baseline("h", 1)
        self.assertFalse(m["ok"])
        self.assertIn("guardmo", m["reason"])

    def test_a_merge_first_holder_without_a_speedup_tier_loses_it_and_a_stale_main_keeps_its_place(self):
        stale = arb.STALE_MAIN_PREFIX + "111111111, main is now 222222222 — re-measured before it may merge"
        for mod, prefix, first, rebase, ok_fn, _t, recon, *_ in BOTS:
            with self.subTest(mod.__name__):
                calls = ReconcileTests._reconcile(
                    self, mod, recon, ok_fn,
                    {10: [first], 11: [prefix + "none", first], 12: [prefix + "S"]},
                    {"12": {"label": "S", "delta_pct": 4.0}})
                self.assertIn(("rm", 10, first), calls)
                self.assertIn(("rm", 11, first), calls)
                self.assertIn(("add", 12, first), calls)
            with self.subTest(mod.__name__, case="stale main"):
                c2 = []
                def fake_gh(a):
                    if a[:2] == ["pr", "list"] and "open" in a:
                        return run(json.dumps([{"number": 10, "labels": [{"name": prefix + "XL"}, {"name": first}]}]))
                    return run("[]")
                with mock.patch.object(arb, "gh", side_effect=fake_gh), \
                        mock.patch.object(arb, "add_label", side_effect=lambda r, n, l: c2.append(("add", n, l))), \
                        mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: c2.append(("rm", n, l))), \
                        mock.patch.object(mod, "_load_scores", return_value={"10": {"label": "XL", "delta_pct": 30.0}}), \
                        mock.patch.object(mod, ok_fn.__name__, return_value=(False, stale)), \
                        mock.patch.object(mod, "AUTO_MERGE", False):
                    recon("o/r")
                self.assertNotIn(("rm", 10, first), c2)

    def _main(self, mod, labels, scores, evaluated=("a" * 40,), gate=(False, arb.STALE_MAIN_PREFIX + "x")):
        prs = [{"number": 5, "title": "t", "labels": [{"name": l} for l in labels], "isDraft": False,
                "headRefOid": "a" * 40, "headRefName": "b", "mergeable": "MERGEABLE",
                "author": {"login": "dev"}, "body": TEMPLATE, "files": [{"path": "kernels/x.cu"}]}]
        tag = "museglimmer" if mod is muse else "qwen38"
        ok_fn = muse.auto_merge_ok_museglimmer if mod is muse else qwen.auto_merge_ok_qwen38
        removed = []
        with mock.patch.object(mod.sys, "argv", [f"pr_{tag}_bot.py", "--only-prs", "5"]), \
                mock.patch.object(arb, "gh", return_value=run(json.dumps(prs))), \
                mock.patch.object(arb, "current_main_sha", return_value=MAIN), \
                mock.patch.object(arb, "load_denylist", return_value=set()), \
                mock.patch.object(arb, "pr_involved_logins", return_value=set()), \
                mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: removed.append(l)), \
                mock.patch.object(arb, "labels_on", return_value=set()), \
                mock.patch.object(arb, "sync_generic_eval_label"), \
                mock.patch.object(mod, f"{tag}_evaluated_commits",
                                  return_value=None if evaluated is None else set(evaluated)), \
                mock.patch.object(mod, ok_fn.__name__, return_value=gate), \
                mock.patch.object(mod, "_load_scores", return_value=scores), \
                mock.patch.object(mod, f"close_stale_{tag}_prs", return_value=set()), \
                mock.patch.object(mod, "resolve_ssh", side_effect=RuntimeError("no box in tests")), \
                mock.patch.object(mod, f"reconcile_{tag}_merge_labels"), \
                mock.patch("builtins.print") as p:
            mod.main()
        return "\n".join(" ".join(str(x) for x in c.args) for c in p.call_args_list), removed

    def test_main_re_measures_a_stale_merge_candidate_and_drops_an_old_heads_tier(self):
        for mod, prefix, *_ in BOTS:
            tag = "museglimmer" if mod is muse else "qwen38"
            entry = {"commit": "a" * 40, "label": "XL", "pass": True, "onto": "9" * 40}
            with self.subTest(mod.__name__):
                out, _ = self._main(mod, [prefix + "XL"], {"5": entry})
                self.assertIn("scored against an older main — re-measuring", out)
                out, _ = self._main(mod, [prefix + "XL"], {"5": dict(entry, onto=MAIN)})
                self.assertIn(f"already {tag}-evaluated — skip", out)
                out, removed = self._main(mod, [prefix + "XL", "eval-bonsai:S"], {}, evaluated=("f" * 40,))
                self.assertEqual(removed, [prefix + "XL"])
                out, removed = self._main(mod, [prefix + "XL"], {}, evaluated=None)
                self.assertIn("GitHub did not return its comments", out)
                self.assertEqual(removed, [])

    def test_none_is_not_blocked_by_the_bots_own_merge_first(self):
        muse_only = ClosePolicyTests.MUSE_ONLY
        self.assertTrue(arb.none_may_close(muse_only, {"museglimmer-merge-first"}, "muse", "eval-museglimmer:"))
        self.assertFalse(arb.none_may_close(muse_only, {"qwen38-merge-first"}, "muse", "eval-museglimmer:"))
        self.assertFalse(arb.none_may_close(muse_only, {"merge-first"}, "muse", "eval-museglimmer:"))


def muse_stdout(top1="0.99", kl="0.01", drop=(), sha=MAIN):
    """What a complete Muse Glimmer run prints (both roles; main also names its commit)."""
    lines = ["PR_TIP " + "a" * 40] + (["REMOTE_SHA " + sha] if sha else [])
    lines += [f"MUSE {c} 100.0 5000.0" for c in muse.SCORED_CTXS]
    lines += [f"MUSECB {c} 500.0" for c in muse.CB_CONCS]
    lines += ["RESULT_DECODE_TPS 100.0", "RESULT_PREFILL128_PP 5000.0", f"RESULT_TOP1 {top1}",
              f"RESULT_KL {kl}", "ACCURACY_STAGE_DONE", "GUARD_START", "GUARD36 32768 50.0 900.0",
              "GUARDMO 32768 60.0 7000.0", "GUARDUN 32768 55.0 6800.0",
              "GUARDBN 128 99.0 2000.0", "GUARDBN 32768 89.0 6500.0", "GUARD_END"]
    return "\n".join(l for l in lines if not any(l.startswith(d) for d in drop)) + "\n"


class WiringTests(unittest.TestCase):
    """The pieces that make a verdict fresh and a failure bounded, wired end to end."""

    def _main(self):
        with mock.patch.object(muse, "_ssh_run_resilient", return_value=run(muse_stdout())):
            m = muse.measure_main_baseline("h", 1)
        self.assertTrue(m["ok"], m)
        return m

    def test_a_verdict_is_measured_onto_mains_commit_and_records_it(self):
        main = self._main()
        seen = []
        with mock.patch.object(muse, "POLARIS_ENABLED", False), \
                mock.patch.object(muse, "_ssh_run_resilient",
                                  side_effect=lambda h, p, script, label: seen.append(script) or run(muse_stdout())):
            res = muse.eval_museglimmer_on_box("h", 1, "pull/1/head", main)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["onto"], MAIN)
        self.assertIn(f"git checkout -qf {MAIN}", seen[0])
        seen = []
        with mock.patch.object(qwen, "_ssh_run_resilient",
                               side_effect=lambda h, p, script, label: seen.append(script) or run("", 1, "x")):
            qwen.eval_qwen38_on_box("h", 1, "pull/1/head", {"sha": MAIN})
        self.assertIn(f"git checkout -qf {MAIN}", seen[0])
        self.assertIn('"onto": main.get("sha")', open(qwen.__file__).read())

    def test_the_scores_file_keeps_the_main_a_verdict_was_measured_against(self):
        for mod, *_ in BOTS:
            tag = "museglimmer" if mod is muse else "qwen38"
            saved = []
            res = {"ok": True, "label": "XL", "delta_pct": 25.0, "pass": True, "accuracy_ok": True, "onto": MAIN}
            with self.subTest(mod.__name__), mock.patch.object(arb, "gh", return_value=run()), \
                    mock.patch.object(arb, "add_label"), mock.patch.object(arb, "remove_label"), \
                    mock.patch.object(arb, "sync_generic_eval_label"), \
                    mock.patch.object(mod, f"strip_{tag}_eval_labels"), mock.patch.object(mod, f"upload_{tag}_eval_log"), \
                    mock.patch.object(mod, "format_comment", return_value="c"), \
                    mock.patch.object(mod, "_load_scores", return_value={}), \
                    mock.patch.object(mod, "_save_scores", side_effect=lambda d: saved.append(d)):
                mod.apply_result("o/r", 3, "a" * 40, res)
                self.assertEqual(saved[-1]["3"]["onto"], MAIN)

    def test_a_pr_run_that_died_without_a_word_carries_a_box_strike(self):
        for mod, *_ in BOTS:
            ev = mod.eval_museglimmer_on_box if mod is muse else mod.eval_qwen38_on_box
            with self.subTest(mod.__name__), mock.patch.object(mod, "_ssh_run_resilient", return_value=run("", 255, "")):
                res = ev("h", 1, "pull/1/head", {"sha": MAIN})
            self.assertTrue(res["retry"])
            self.assertEqual(res["strike_key"], "box")

    def test_muse_refuses_a_baseline_that_does_not_name_its_commit(self):
        with mock.patch.object(muse, "_ssh_run_resilient", return_value=run(muse_stdout(sha=""))):
            m = muse.measure_main_baseline("h", 1)
        self.assertFalse(m["ok"])
        self.assertIn("did not report its commit", m["reason"])

    def test_a_guard_gap_never_hides_an_accuracy_reject(self):
        main = dict(self._main(), guardmo={})               # main measured nothing for one guard
        with mock.patch.object(muse, "POLARIS_ENABLED", False), \
                mock.patch.object(muse, "_ssh_run_resilient", return_value=run(muse_stdout(top1="0.2", kl="2.0"))):
            res = muse.eval_museglimmer_on_box("h", 1, "pull/1/head", main)
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["label"], "REJECT")
        self.assertIn('label != "REJECT"', open(qwen.__file__).read())

    def test_no_bot_measures_anything_without_the_shared_lock(self):
        import pr_bonsai_bot as bonsai
        prs = [{"number": 5, "title": "t", "labels": [], "isDraft": False, "headRefOid": "a" * 40,
                "headRefName": "b", "mergeable": "MERGEABLE", "author": {"login": "dev"}, "body": TEMPLATE,
                "files": [{"path": "kernels/x.cu"}]}]
        for mod, tag in ((muse, "museglimmer"), (qwen, "qwen38"), (bonsai, "bonsai")):
            with self.subTest(tag), mock.patch.object(mod.sys, "argv", [f"pr_{tag}_bot.py", "--only-prs", "5"]), \
                    mock.patch.object(arb, "gh", return_value=run(json.dumps(prs))), \
                    mock.patch.object(arb, "current_main_sha", return_value=MAIN), \
                    mock.patch.object(arb, "load_denylist", return_value=set()), \
                    mock.patch.object(arb, "pr_involved_logins", return_value=set()), \
                    mock.patch.object(mod, f"{tag}_evaluated_commits", return_value=set()), \
                    mock.patch.object(mod, "resolve_ssh", return_value=("h", 1)), \
                    mock.patch.object(arb, "hold_bot_lock", return_value=False) as lock, \
                    mock.patch.object(mod, "measure_main_baseline") as baseline, \
                    mock.patch.object(mod, f"reconcile_{tag}_merge_labels"), \
                    mock.patch("builtins.print") as p:
                mod.main()
                lock.assert_called_once()
                baseline.assert_not_called()
                self.assertIn("stayed busy", "\n".join(" ".join(str(x) for x in c.args) for c in p.call_args_list))

    def test_every_bot_skips_what_no_bot_measures(self):
        import pr_bonsai_bot as bonsai
        for mod in (muse, qwen, bonsai):
            with self.subTest(mod.__name__):
                for p in arb.NEVER_MEASURED_PATHS:
                    self.assertIn(p, mod.HARNESS_PATHS)


class StaleWinnerTests(unittest.TestCase):
    """A winner refused only for a moved main keeps its place -- if, and only if, it will be
    re-measured -- and never costs a PR that can merge now its turn."""
    STALE = arb.STALE_MAIN_PREFIX + "111111111, main is now 222222222 — re-measured before it may merge"

    def _round(self, mod, recon, ok_fn, first, rebase, prefix, greenlight=("ok", "x"), fresh=True):
        prs = [{"number": 10, "labels": [{"name": prefix + "XL"}, {"name": first}], "isDraft": False,
                "body": TEMPLATE, "files": [{"path": "kernels/x.cu"}], "mergeable": "MERGEABLE"},
               {"number": 11, "labels": [{"name": prefix + "S"}], "isDraft": False, "body": TEMPLATE,
                "files": [{"path": "kernels/y.cu"}], "mergeable": "MERGEABLE"}]
        if not fresh:
            prs = prs[:1]
        calls = []

        def fake_gh(a):
            if a[:2] == ["pr", "list"] and "open" in a:
                return run(json.dumps(prs))
            return run("[]")
        with mock.patch.object(arb, "gh", side_effect=fake_gh), \
                mock.patch.object(arb, "greenlight_status", return_value=greenlight), \
                mock.patch.object(arb, "add_label", side_effect=lambda r, n, l: calls.append(("add", n, l))), \
                mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: calls.append(("rm", n, l))), \
                mock.patch.object(mod, "_load_scores", return_value={"10": {"label": "XL", "delta_pct": 30.0},
                                                                     "11": {"label": "S", "delta_pct": 4.0}}), \
                mock.patch.object(mod, ok_fn.__name__,
                                  side_effect=lambda r, n, require_merge_first=True:
                                  (False, self.STALE) if n == 10 else (True, "ok")), \
                mock.patch.object(mod, "AUTO_MERGE", False):
            recon("o/r")
        return calls

    def test_a_pr_that_can_merge_now_outranks_one_waiting_for_its_re_measure(self):
        for mod, prefix, first, rebase, ok_fn, _t, recon, *_ in BOTS:
            with self.subTest(mod.__name__):
                calls = self._round(mod, recon, ok_fn, first, rebase, prefix)
                self.assertIn(("add", 11, first), calls)
                self.assertNotIn(("add", 11, rebase), calls)
                self.assertIn(("rm", 10, first), calls)

    def test_alone_it_keeps_merge_first_and_nobody_is_sent_to_rebase(self):
        for mod, prefix, first, rebase, ok_fn, _t, recon, *_ in BOTS:
            with self.subTest(mod.__name__):
                calls = self._round(mod, recon, ok_fn, first, rebase, prefix, fresh=False)
                self.assertNotIn(("rm", 10, first), calls)
                self.assertFalse(any(c[0] == "add" and c[2] == rebase for c in calls))

    def test_a_stale_winner_sends_no_stale_rival_to_rebase(self):
        for mod, prefix, first, rebase, ok_fn, _t, recon, *_ in BOTS:
            calls = []
            prs = [{"number": n, "labels": [{"name": prefix + t}], "isDraft": False, "body": TEMPLATE,
                    "files": [{"path": "kernels/x.cu"}], "mergeable": "MERGEABLE"} for n, t in ((10, "XL"), (11, "S"))]
            with self.subTest(mod.__name__), \
                    mock.patch.object(arb, "gh", side_effect=lambda a: run(json.dumps(prs)) if a[:2] == ["pr", "list"]
                                      and "open" in a else run("[]")), \
                    mock.patch.object(arb, "greenlight_status", return_value=("ok", "x")), \
                    mock.patch.object(arb, "add_label", side_effect=lambda r, n, l: calls.append(("add", n, l))), \
                    mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: calls.append(("rm", n, l))), \
                    mock.patch.object(mod, "_load_scores", return_value={"10": {"label": "XL", "delta_pct": 30.0},
                                                                         "11": {"label": "S", "delta_pct": 4.0}}), \
                    mock.patch.object(mod, ok_fn.__name__, return_value=(False, self.STALE)), \
                    mock.patch.object(mod, "AUTO_MERGE", False):
                recon("o/r")
            self.assertIn(("add", 10, first), calls)
            self.assertNotIn(("add", 11, rebase), calls)

    def test_one_the_selection_would_never_re_measure_loses_it(self):
        for mod, prefix, first, rebase, ok_fn, _t, recon, *_ in BOTS:
            with self.subTest(mod.__name__):
                calls = self._round(mod, recon, ok_fn, first, rebase, prefix, greenlight=("no-bench", "edited"),
                                    fresh=False)
                self.assertIn(("rm", 10, first), calls)


class Round3Tests(unittest.TestCase):
    """Fixes from the third review of the 2026-09-26 change, on all three bots."""

    def _bots(self):
        import pr_bonsai_bot as bonsai
        return ((muse, "museglimmer", muse.MUSEGLIMMER_NEEDS_REBASE), (qwen, "qwen38", qwen.QWEN38_NEEDS_REBASE),
                (bonsai, "bonsai", bonsai.BONSAI_NEEDS_REBASE))

    def test_remote_output_that_is_not_utf8_cannot_raise(self):
        for mod, tag, _r in self._bots():
            with self.subTest(tag), mock.patch.object(mod.subprocess, "run", return_value=run()) as r:
                mod.ssh_run("h", 1, "true")
                self.assertEqual(r.call_args.kwargs.get("errors"), "replace")

    def test_the_bots_own_failure_is_bounded_and_never_charged(self):
        res = arb.exception_result(ValueError("boom"))
        self.assertEqual(res["strike_key"], "error")
        for mod, tag, _r in self._bots():
            self.addCleanup(mod.GAVE_UP.clear)          # per process in production; per test here
            calls = []
            with self.subTest(tag), mock.patch.object(arb, "gh", side_effect=lambda a: calls.append(a) or run()), \
                    mock.patch.object(arb, "add_label", side_effect=lambda *a: calls.append(a)):
                for _ in range(arb.BOX_FAULT_STRIKES + 1):
                    mod.apply_result("o/r", 4, "a" * 40, dict(res))
                self.assertEqual(calls, [])                                  # nothing ever posted
                self.assertTrue(arb.gave_up(mod.STRIKES_FILE, 4, "a" * 40))
                self.assertIn(4, mod.GAVE_UP)                                # so the run exits 3
                self.assertFalse(arb.gave_up(mod.STRIKES_FILE, 4, "b" * 40))  # a push starts again
                arb.clear_strikes(mod.STRIKES_FILE, 4)

    def test_a_needs_rebase_from_an_older_head_is_dropped_unless_github_says_it_conflicts(self):
        for mod, tag, rebase in self._bots():
            removed = []
            with self.subTest(tag), \
                    mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: removed.append(l)), \
                    mock.patch.object(arb, "sync_generic_eval_label"):
                self.assertTrue(arb.strip_stale_verdict_labels("o/r", 1, {rebase}, f"eval-{tag}:", "b" * 40,
                                                               {"a" * 40}, rebase, False))
                self.assertEqual(removed, [rebase])
                removed.clear()
                self.assertFalse(arb.strip_stale_verdict_labels("o/r", 1, {rebase}, f"eval-{tag}:", "b" * 40,
                                                                {"a" * 40}, rebase, True))
                self.assertFalse(arb.strip_stale_verdict_labels("o/r", 1, {rebase}, f"eval-{tag}:", "a" * 40,
                                                                {"a" * 40}, rebase, False))   # its own verdict
                self.assertEqual(removed, [])

    def test_a_none_close_needs_every_label_read(self):
        self.assertFalse(arb.none_may_close(ClosePolicyTests.MUSE_ONLY, None, "muse", "eval-museglimmer:"))
        for mod, *_ in BOTS:
            tag = "museglimmer" if mod is muse else "qwen38"
            body = ClosePolicyTests.MUSE_ONLY if mod is muse else ClosePolicyTests.QWEN_ONLY
            calls = []

            def fake_gh(a):
                calls.append(a)
                return run(json.dumps({"headRefOid": "a" * 40})) if a[:2] == ["pr", "view"] else run()
            res = {"ok": True, "label": "none", "delta_pct": 0.4, "pass": True, "accuracy_ok": True}
            with self.subTest(mod.__name__), mock.patch.object(arb, "gh", side_effect=fake_gh), \
                    mock.patch.object(arb, "labels_on_or_none", return_value=None), \
                    mock.patch.object(arb, "add_label"), mock.patch.object(arb, "remove_label"), \
                    mock.patch.object(arb, "sync_generic_eval_label"), \
                    mock.patch.object(mod, f"strip_{tag}_eval_labels"), mock.patch.object(mod, f"upload_{tag}_eval_log"), \
                    mock.patch.object(mod, "format_comment", return_value="c"), \
                    mock.patch.object(mod, "_load_scores", return_value={}), mock.patch.object(mod, "_save_scores"):
                mod.apply_result("o/r", 1, "a" * 40, res, pr_body=body)
                self.assertFalse(any(a[:2] == ["pr", "close"] for a in calls))

    def test_a_verified_pr_owed_a_re_measure_is_not_stale(self):
        for mod, tag, _r in self._bots():
            prs = [{"number": 7, "body": TEMPLATE, "isDraft": False, "labels": [{"name": f"eval-{tag}:XL"}],
                    "headRefOid": "a" * 40}]
            with self.subTest(tag), mock.patch.object(mod, "_pr_last_activity_ts", return_value=0.0), \
                    mock.patch.object(arb, "current_main_sha", return_value=MAIN), \
                    mock.patch.object(mod, "_remeasure_against_new_main", return_value=True), \
                    mock.patch.object(mod, "_unmeasurable_reason", return_value=None), \
                    mock.patch.object(arb, "gh", return_value=run()):
                self.assertEqual(getattr(mod, f"close_stale_{tag}_prs")("o/r", prs), set())
            with self.subTest(tag, case="not owed"), mock.patch.object(mod, "_pr_last_activity_ts", return_value=0.0), \
                    mock.patch.object(arb, "current_main_sha", return_value=MAIN), \
                    mock.patch.object(mod, "_remeasure_against_new_main", return_value=False), \
                    mock.patch.object(mod, f"{tag}_evaluated_commits", return_value={"a" * 40}), \
                    mock.patch.object(arb, "gh", return_value=run()):
                self.assertEqual(getattr(mod, f"close_stale_{tag}_prs")("o/r", prs), {7})

    def test_an_oom_kill_is_the_box_on_every_bot(self):
        err = "REMOTE_SCRIPT_FAILED line=40 exit=137 reason=likely OOM-killed (SIGKILL)"
        self.assertTrue(muse._is_box_fault("ACCURACY_STAGE_DONE\nGUARD_START\nGUARD_END\n", err))
        self.assertTrue(qwen._is_box_fault("GUARD_END\n", err))
        self.assertFalse(qwen._is_box_fault("GUARD_END\n", err.replace("137", "139")))   # a segfault is the PR's

    def test_main_names_its_commit_on_every_bot(self):
        import pr_bonsai_bot as bonsai
        for mod in (qwen, bonsai):
            self.assertIn('"main run did not report its commit"', open(mod.__file__).read())
        for mod, s in ((qwen, qwen._remote_script("main", role="main")),
                       (bonsai, bonsai._remote_script("main", role="main"))):
            self.assertIn("git checkout -q HEAD -- ", s, mod.__name__)       # main's own harness

    def test_scores_are_written_whole_or_not_at_all(self):
        p = _os.path.join(_STATE, "atomic.json")
        arb.write_json_atomic(p, {"1": {"commit": "a"}})
        self.assertEqual(json.load(open(p)), {"1": {"commit": "a"}})
        self.assertEqual([x for x in _os.listdir(_STATE) if x.startswith("atomic.json.tmp")], [])


class Round4Tests(unittest.TestCase):
    def test_the_box_script_markers_the_bots_classify_by(self):
        m = muse._remote_script("pull/1/head", role="pr", onto=MAIN)
        self.assertIn("ACCURACY_COMPARE_FAILED", m)
        self.assertIn("RETRYABLE_INFRA_FAILURE llama.cpp reference server went down", m)
        self.assertIsNotNone(muse._crash_reason("", "ACCURACY_COMPARE_FAILED -- x"))
        self.assertFalse(muse._is_box_fault("", "ACCURACY_COMPARE_FAILED -- x"))     # the PR's
        q = qwen._remote_script("main", role="main")
        self.assertEqual(q.count("RETRYABLE_INFRA_FAILURE git fetch"), 2)             # checkout + pin
        self.assertIn("git checkout -q HEAD -- ", q)

    def test_a_draft_or_held_pr_loses_an_older_heads_tier(self):
        for mod, prefix, first, rebase, *_ in BOTS:
            tag = "museglimmer" if mod is muse else "qwen38"
            for extra in ({"isDraft": True}, {"labels": [{"name": prefix + "XL"}, {"name": "hold"}]}):
                pr = {"number": 5, "title": "t", "labels": [{"name": prefix + "XL"}], "isDraft": False,
                      "headRefOid": "b" * 40, "headRefName": "x", "mergeable": "MERGEABLE",
                      "author": {"login": "dev"}, "body": TEMPLATE, "files": []}
                pr.update(extra)
                removed = []
                with self.subTest(mod.__name__, **{k: str(v)[:20] for k, v in extra.items()}), \
                        mock.patch.object(mod.sys, "argv", [f"pr_{tag}_bot.py", "--only-prs", "5"]), \
                        mock.patch.object(arb, "gh", return_value=run(json.dumps([pr]))), \
                        mock.patch.object(arb, "current_main_sha", return_value=MAIN), \
                        mock.patch.object(arb, "load_denylist", return_value=set()), \
                        mock.patch.object(arb, "pr_involved_logins", return_value=set()), \
                        mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: removed.append(l)), \
                        mock.patch.object(arb, "labels_on", return_value=set()), \
                        mock.patch.object(arb, "sync_generic_eval_label"), \
                        mock.patch.object(mod, f"{tag}_evaluated_commits", return_value={"a" * 40}), \
                        mock.patch.object(mod, f"reconcile_{tag}_merge_labels"), \
                        mock.patch("builtins.print"):
                    mod.main()
                    self.assertIn(prefix + "XL", removed)


class IsolationTests(unittest.TestCase):
    def test_no_state_file_points_outside_this_suites_temp_dir(self):
        # A test once recorded a strike in the controller's real ~/.sparkinfer_bonsai_strikes.json.
        for mod in (arb, muse, qwen, _bonsai):
            for name in ("STRIKES_FILE", "SCORES_FILE", "INSTANCE_FILE", "PIN_FILE", "BOT_LOCK_FILE"):
                if hasattr(mod, name):
                    # Any suite's temp dir: run together, the last suite imported sets the shared globals.
                    self.assertTrue(getattr(mod, name).startswith(_tempfile.gettempdir()), (mod.__name__, name))


class Round5Tests(unittest.TestCase):
    """Fixes from the fourth review of the 2026-09-26 change."""

    def _bots(self):
        return Round3Tests._bots(self)

    def _stale(self, mod, tag, **patches):
        prs = [{"number": 7, "body": TEMPLATE, "isDraft": False, "labels": [{"name": f"eval-{tag}:XL"}],
                "headRefOid": "a" * 40}]
        with mock.patch.object(mod, "_pr_last_activity_ts", return_value=0.0), \
                mock.patch.object(arb, "gh", return_value=run()), \
                mock.patch.object(mod, f"{tag}_evaluated_commits", return_value={"a" * 40}):
            ctx = [mock.patch.object(*p) for p in patches.get("extra", ())]
            for c in ctx:
                c.start()
            try:
                return getattr(mod, f"close_stale_{tag}_prs")("o/r", prs)
            finally:
                for c in ctx:
                    c.stop()

    def test_a_pr_the_bot_gave_up_on_is_never_stale_closed(self):
        for mod, tag, _r in self._bots():
            with self.subTest(tag):
                for _ in range(arb.BOX_FAULT_STRIKES):
                    arb.record_strike(mod.STRIKES_FILE, 7, "a" * 40, "error")
                self.addCleanup(arb.clear_strikes, mod.STRIKES_FILE, 7)
                self.assertEqual(self._stale(mod, tag), set())

    def test_an_unanswered_read_never_closes_a_pr_owed_a_re_measure(self):
        for mod, tag, _r in self._bots():
            with self.subTest(tag, case="main unknown"):
                closed = self._stale(mod, tag, extra=((arb, "current_main_sha", mock.Mock(return_value="")),
                                                      (mod, "_load_scores", mock.Mock(return_value={"7": {
                                                          "commit": "a" * 40, "label": "XL", "pass": True,
                                                          "onto": "9" * 40}}))))
                self.assertEqual(closed, set())
            with self.subTest(tag, case="pr unreadable"):
                okfn = {"museglimmer": "auto_merge_ok_museglimmer", "qwen38": "auto_merge_ok_qwen38",
                        "bonsai": "auto_merge_ok_bonsai"}[tag]
                closed = self._stale(mod, tag, extra=((arb, "current_main_sha", mock.Mock(return_value=MAIN)),
                                                      (mod, "_load_scores", mock.Mock(return_value={"7": {
                                                          "commit": "a" * 40, "label": "XL", "pass": True,
                                                          "onto": "9" * 40}})),
                                                      (mod, okfn, mock.Mock(return_value=(False, arb.PR_UNREADABLE)))))
                self.assertEqual(closed, set())

    def _main_exit(self, mod, tag, **extra):
        prs = [{"number": 5, "title": "t", "labels": [], "isDraft": False, "headRefOid": "a" * 40,
                "headRefName": "b", "mergeable": "MERGEABLE", "author": {"login": "dev"}, "body": TEMPLATE,
                "files": [{"path": "kernels/x.cu"}]}]
        with mock.patch.object(mod.sys, "argv", [f"pr_{tag}_bot.py", "--only-prs", "5"]), \
                mock.patch.object(arb, "gh", return_value=run(json.dumps(prs))), \
                mock.patch.object(arb, "current_main_sha", return_value=MAIN), \
                mock.patch.object(arb, "load_denylist", return_value=set()), \
                mock.patch.object(arb, "pr_involved_logins", return_value=set()), \
                mock.patch.object(mod, f"{tag}_evaluated_commits", return_value=set()), \
                mock.patch.object(mod, "resolve_ssh", return_value=("h", 1)), \
                mock.patch.object(arb, "hold_bot_lock", return_value=True), \
                mock.patch.object(mod, "measure_main_baseline",
                                  return_value=extra.get("baseline", {"ok": False, "reason": "main misses its bar"})), \
                mock.patch.object(mod, f"reconcile_{tag}_merge_labels"), \
                mock.patch("builtins.print"):
            try:
                mod.main()
            except SystemExit as e:
                return e.code
        return 0

    def test_an_unusable_baseline_is_loud(self):
        for mod, tag, _r in self._bots():
            with self.subTest(tag):
                self.assertEqual(self._main_exit(mod, tag), 3)

    def test_a_run_that_gave_up_on_a_pr_is_loud(self):
        for mod, tag, _r in self._bots():
            with self.subTest(tag):
                for _ in range(arb.BOX_FAULT_STRIKES):
                    arb.record_strike(mod.STRIKES_FILE, 5, "a" * 40, "error")
                self.addCleanup(arb.clear_strikes, mod.STRIKES_FILE, 5)
                self.addCleanup(mod.GAVE_UP.clear)
                self.assertEqual(self._main_exit(mod, tag), 3)             # nothing else to measure

    def test_an_oom_killed_muse_sweep_or_guard_is_the_box(self):
        main = WiringTests._main(self)
        for stdout, key in ((muse_stdout(drop=("MUSE ",)) + "MUSE_FAILED rc=137\n", "sweep-box"),
                            (muse_stdout(drop=("GUARDMO ",)) + "GUARDMO_FAILED rc=137\n", "guard-box")):
            with self.subTest(key), mock.patch.object(muse, "POLARIS_ENABLED", False), \
                    mock.patch.object(muse, "_ssh_run_resilient", return_value=run(stdout)):
                res = muse.eval_museglimmer_on_box("h", 1, "pull/1/head", main)
            self.assertTrue(res["retry"], res)
            self.assertEqual(res["strike_key"], key)
        # Beside wrong output the REJECT stands; any other exit is the PR's regression as before.
        for stdout in (muse_stdout(top1="0.2", kl="2.0", drop=("GUARDMO ",)) + "GUARDMO_FAILED rc=137\n",
                       muse_stdout(drop=("GUARDMO ",)) + "GUARDMO_FAILED rc=1\n"):
            with mock.patch.object(muse, "POLARIS_ENABLED", False), \
                    mock.patch.object(muse, "_ssh_run_resilient", return_value=run(stdout)):
                res = muse.eval_museglimmer_on_box("h", 1, "pull/1/head", main)
            self.assertEqual((res["ok"], res["label"]), (True, "REJECT"), res)

    def test_qwen38_reads_an_oom_kill_as_the_box(self):
        p = qwen._parse_remote("SWEEP_FAILED rc=137\nGUARDCBMO_FAILED 32 rc=137\nGUARDBN_FAILED rc=2\n")
        self.assertEqual((p.get("sweep_failed_box"), p.get("guardcbmo_failed_box"), p.get("guardbn_failed_box")),
                         (True, True, None))
        self.assertTrue(qwen._is_box_fault("", "RETRYABLE_INFRA_FAILURE concurrent decode killed at c=32 (exit 137)"))
        src = open(qwen.__file__).read()
        self.assertIn('"strike_key": "sweep-box"', src)
        self.assertIn('"strike_key": "guard-box"', src)


class Round6Tests(unittest.TestCase):
    """Fixes from the fifth review of the 2026-09-26 change."""

    def test_every_sweep_and_guard_failure_line_carries_its_exit_code(self):
        import re as _re
        for name, script in (("qwen", qwen._remote_script("pull/1/head", role="pr", onto=MAIN)),
                             ("muse", muse._remote_script("pull/1/head", role="pr", onto=MAIN))):
            lines = [l.strip() for l in script.splitlines()
                     if _re.search(r'echo "(GUARD[A-Z0-9]*|SWEEP|MUSE)_FAILED', l)]
            self.assertTrue(lines, name)
            for l in lines:
                self.assertIn("rc=", l, (name, l))

    def test_qwen38s_score_step_killed_is_the_box(self):
        s = qwen._remote_script("pull/1/head", role="pr", onto=MAIN)
        self.assertIn("RETRYABLE_INFRA_FAILURE score step killed (exit 137)", s)
        self.assertTrue(qwen._is_box_fault("", "RETRYABLE_INFRA_FAILURE score step killed (exit 137)"))

    def test_a_guard_that_measured_nothing_is_bounded_and_zeros_are_nothing(self):
        main = WiringTests._main(self)
        with mock.patch.object(muse, "POLARIS_ENABLED", False), \
                mock.patch.object(muse, "_ssh_run_resilient", return_value=run(muse_stdout())):
            res = muse.eval_museglimmer_on_box("h", 1, "pull/1/head", dict(main, guardmo={}))
        self.assertEqual((res["retry"], res["strike_key"]), (True, "guard-unmeasured"))
        zeros = muse_stdout(drop=("GUARDMO ",)) + "GUARDMO 32768 0 0\n"
        with mock.patch.object(muse, "_ssh_run_resilient", return_value=run(zeros)):
            m = muse.measure_main_baseline("h", 1)
        self.assertFalse(m["ok"])
        self.assertIn("guardmo", m["reason"])
        self.assertFalse(arb.guard_measured({32768: {"decode": 0, "prefill": 0}}))
        self.assertTrue(arb.guard_measured({16: {"cb-decode": 800.0}}))

    def test_a_pr_the_bot_gave_up_on_cannot_hold_merge_first(self):
        stale = arb.STALE_MAIN_PREFIX + "111111111, main is now 222222222 — re-measured before it may merge"
        for mod, prefix, first, rebase, ok_fn, _t, recon, *_ in BOTS:
            prs = [{"number": 10, "labels": [{"name": prefix + "XL"}, {"name": first}], "isDraft": False,
                    "body": TEMPLATE, "files": [], "mergeable": "MERGEABLE", "headRefOid": "a" * 40}]
            calls = []
            for _ in range(arb.BOX_FAULT_STRIKES):
                arb.record_strike(mod.STRIKES_FILE, 10, "a" * 40, "error")
            self.addCleanup(arb.clear_strikes, mod.STRIKES_FILE, 10)
            with self.subTest(mod.__name__), \
                    mock.patch.object(arb, "gh", side_effect=lambda a: run(json.dumps(prs)) if a[:2] == ["pr", "list"]
                                      and "open" in a else run("[]")), \
                    mock.patch.object(arb, "greenlight_status", return_value=("ok", "x")), \
                    mock.patch.object(arb, "add_label", side_effect=lambda r, n, l: calls.append(("add", n, l))), \
                    mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: calls.append(("rm", n, l))), \
                    mock.patch.object(mod, "_load_scores", return_value={"10": {"label": "XL", "delta_pct": 30.0}}), \
                    mock.patch.object(mod, ok_fn.__name__, return_value=(False, stale)), \
                    mock.patch.object(mod, "AUTO_MERGE", False):
                recon("o/r")
            self.assertIn(("rm", 10, first), calls)


if __name__ == "__main__":
    unittest.main()
