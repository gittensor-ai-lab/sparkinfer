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
import os as _os
import tempfile as _tempfile
import _bot_test_state
import pr_bonsai_bot as _bonsai   # imported by some tests below: its state is redirected too
_STATE = _bot_test_state.new_state_dir()
_bot_test_state.isolate(_STATE, arb, muse, qwen, _bonsai)


def setUpModule():
    _bot_test_state.isolate(_STATE, arb, muse, qwen, _bonsai)


# The author clock (arb.AuthorWaitClock) of a PR that has waited on its author since the epoch: the
# stale-close tests below are about WHICH PRs may close; the clock has its own tests.
_LONG_WAITED = lambda self, num, head: 0.0

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
                                  side_effect=lambda r, n, require_merge_first=True, ranking_loss_ok=False:
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
        # A transport failure is retried; so is a run killed at the ssh limit, charged to the PR only
        # once it recurs at one commit (a step of the box's own can hang too).
        self.assertTrue(arb.exception_result(ConnectionResetError("x"))["retry"])
        hang = arb.exception_result(subprocess.TimeoutExpired("ssh", 7200))
        self.assertEqual((hang["retry"], hang["strike_key"]), (True, "timeout"))
        for mod in (muse, qwen):
            charged = dict(hang, retry=False)
            self.assertIn('"label":"REJECT"', mod.format_comment("a" * 40, charged), mod.__name__)

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
                mock.patch.object(muse, "_verdict_heads", return_value=set()), \
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
                    mock.patch.object(arb.AuthorWaitClock, "since", _LONG_WAITED), \
                    mock.patch.object(mod, "_verdict_heads", return_value=set()), \
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
            "GUARDMG_UNAVAILABLE", "GUARDBN 128 99 2000", "GUARDBN 32768 89 6500", "GUARD_END",
            "SELFCHECK top1=1.000000 kl=0.000000 ppl_pr=1.0 ppl_main=1.0"]) + "\n"
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
            with self.subTest(mod.__name__), mock.patch.object(mod, "_pr_last_activity_ts", return_value=0.0), \
                    mock.patch.object(arb.AuthorWaitClock, "since", _LONG_WAITED), \
                    mock.patch.object(mod, "_verdict_heads", side_effect=lambda r, n: {"b" * 40}), \
                    mock.patch.object(arb, "greenlight_status", return_value=("ok", "claims a gain")), \
                    mock.patch.object(arb, "gh", return_value=run()):
                # #1 is queued behind the bot; #2 already has its verdict and waits on its author.
                self.assertEqual(close_stale("o/r", prs), {2})
            with self.subTest(mod.__name__, greenlit=False), \
                    mock.patch.object(mod, "_pr_last_activity_ts", return_value=0.0), \
                    mock.patch.object(arb.AuthorWaitClock, "since", _LONG_WAITED), \
                    mock.patch.object(mod, "_verdict_heads", return_value=set()), \
                    mock.patch.object(arb, "greenlight_status", return_value=("unchecked", "box not ticked")), \
                    mock.patch.object(arb, "gh", return_value=run()):
                # Never measured and not asking to be: not this bot's queue, left to the daily Action
                # ("not being measured is not grounds for closing", CONTRIBUTING).
                self.assertEqual(close_stale("o/r", prs), set())

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
                mock.patch.object(mod, "_verdict_heads",
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
                    mock.patch.object(mod, "_verdict_heads", return_value=set()), \
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
                                  side_effect=lambda r, n, require_merge_first=True, ranking_loss_ok=False:
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
                    mock.patch.object(arb.AuthorWaitClock, "since", _LONG_WAITED), \
                    mock.patch.object(arb, "current_main_sha", return_value=MAIN), \
                    mock.patch.object(mod, "_remeasure_against_new_main", return_value=True), \
                    mock.patch.object(mod, "_unmeasurable_reason", return_value=None), \
                    mock.patch.object(arb, "gh", return_value=run()):
                self.assertEqual(getattr(mod, f"close_stale_{tag}_prs")("o/r", prs), set())
            with self.subTest(tag, case="not owed"), mock.patch.object(mod, "_pr_last_activity_ts", return_value=0.0), \
                    mock.patch.object(arb.AuthorWaitClock, "since", _LONG_WAITED), \
                    mock.patch.object(arb, "current_main_sha", return_value=MAIN), \
                    mock.patch.object(mod, "_remeasure_against_new_main", return_value=False), \
                    mock.patch.object(mod, "_verdict_heads", return_value={"a" * 40}), \
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
                        mock.patch.object(mod, "_verdict_heads", return_value={"a" * 40}), \
                        mock.patch.object(mod, f"reconcile_{tag}_merge_labels"), \
                        mock.patch("builtins.print"):
                    mod.main()
                    self.assertIn(prefix + "XL", removed)


class IsolationTests(unittest.TestCase):
    def test_no_state_path_points_at_the_controllers_state(self):
        # A test once recorded a strike in the controller's real ~/.sparkinfer_bonsai_strikes.json,
        # and another wrote the stale clock's file there: every state path is covered by pattern.
        home = _os.path.expanduser("~/.sparkinfer_")
        for mod in (arb, muse, qwen, _bonsai):
            for name in dir(mod):
                value = getattr(mod, name)
                if name.isupper() and isinstance(value, str):
                    self.assertFalse(value.startswith(home), (mod.__name__, name, value))
                    self.assertNotIn(value, {"/tmp/sparkinfer_bot.lock", _os.environ.get("SPARKINFER_LOCK_FILE")},
                                     (mod.__name__, name))
        for mod, name in ((muse, "AUTHOR_WAIT_FILE"), (qwen, "AUTHOR_WAIT_FILE"), (_bonsai, "AUTHOR_WAIT_FILE"),
                          (arb, "LOG_DIR"), (arb, "BOT_LOCK_FILE")):
            self.assertTrue(getattr(mod, name).startswith(_STATE), (mod.__name__, name))   # this suite's own


class Round5Tests(unittest.TestCase):
    """Fixes from the fourth review of the 2026-09-26 change."""

    def _bots(self):
        return Round3Tests._bots(self)

    def _stale(self, mod, tag, **patches):
        prs = [{"number": 7, "body": TEMPLATE, "isDraft": False, "labels": [{"name": f"eval-{tag}:XL"}],
                "headRefOid": "a" * 40}]
        with mock.patch.object(mod, "_pr_last_activity_ts", return_value=0.0), \
                mock.patch.object(arb.AuthorWaitClock, "since", _LONG_WAITED), \
                mock.patch.object(arb, "gh", return_value=run()), \
                mock.patch.object(mod, "_verdict_heads", return_value={"a" * 40}):
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
                mock.patch.object(mod, "_verdict_heads", return_value=set()), \
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


class Iteration3Tests(unittest.TestCase):
    """Fixes from the post-merge review of 2026-09-26, on all three bots."""

    def _bots(self):
        return Round3Tests._bots(self)

    def _run_main(self, mod, tag, prs_out, extra=(), argv=("--only-prs", "5")):
        with mock.patch.object(mod.sys, "argv", [f"pr_{tag}_bot.py", *argv]), \
                mock.patch.dict(_os.environ, {"SPARKINFER_BOT_LOGIN": ""}), \
                mock.patch.object(arb, "gh", return_value=prs_out), \
                mock.patch.object(arb, "current_main_sha", return_value=MAIN), \
                mock.patch.object(arb, "load_denylist", return_value=set()), \
                mock.patch.object(arb, "pr_involved_logins", return_value=set()), \
                mock.patch.object(mod, "_verdict_heads", return_value=set()), \
                mock.patch.object(mod, "resolve_ssh", side_effect=RuntimeError("no box in tests")), \
                mock.patch.object(mod, f"reconcile_{tag}_merge_labels"), \
                mock.patch("builtins.print") as p:
            ctx = [mock.patch.object(*x) for x in extra]
            for c in ctx:
                c.start()
            try:
                mod.main()
                code = 0
            except SystemExit as e:
                code = e.code
            finally:
                for c in ctx:
                    c.stop()
        return code, "\n".join(" ".join(str(x) for x in c.args) for c in p.call_args_list)

    def test_a_failed_pr_list_or_a_wrong_account_is_loud(self):
        for mod, tag, _r in self._bots():
            with self.subTest(tag, case="list"):
                code, out = self._run_main(mod, tag, run("", 1))
                self.assertEqual(code, 3)
                self.assertIn("GitHub did not return the open PRs", out)
            with self.subTest(tag, case="account"):
                code, out = self._run_main(mod, tag, run("[]"),
                                           extra=((arb, "acting_account_ok", mock.Mock(return_value=(False, "someone"))),))
                self.assertEqual(code, 3)
                self.assertIn("gh acts as someone", out)

    def test_a_pr_into_another_branch_is_not_evaluated(self):
        pr = {"number": 5, "title": "t", "labels": [], "isDraft": False, "headRefOid": "a" * 40, "headRefName": "b",
              "baseRefName": "feat/x", "mergeable": "MERGEABLE", "author": {"login": "dev"}, "body": TEMPLATE,
              "files": [{"path": "kernels/x.cu"}]}
        for mod, tag, _r in self._bots():
            with self.subTest(tag):
                code, out = self._run_main(mod, tag, run(json.dumps([pr])))
                self.assertIn("based on feat/x, not main — not evaluated", out)
                self.assertEqual(mod._unmeasurable_reason("o/r", pr, set()), "based on feat/x, not main")

    def test_a_merge_conflict_the_box_found_stays_with_its_head(self):
        for mod, tag, rebase in self._bots():
            ev = {"museglimmer": "eval_museglimmer_on_box", "qwen38": "eval_qwen38_on_box",
                  "bonsai": "eval_bonsai_on_box"}[tag]
            res = {"ok": False, "retry": True, "conflict": True, "reason": "MERGE_CONFLICT a onto c"}
            with self.subTest(tag), mock.patch.object(arb, "add_label"):
                mod.apply_result("o/r", 5, "a" * 40, res)
            self.addCleanup(arb.clear_strikes, mod.STRIKES_FILE, 5)
            self.assertEqual(arb.strike_count(mod.STRIKES_FILE, 5, "a" * 40, "conflict"), 1)
            pr = {"number": 5, "title": "t", "labels": [{"name": rebase}], "isDraft": False, "headRefOid": "a" * 40,
                  "headRefName": "b", "baseRefName": "main", "mergeable": "MERGEABLE", "author": {"login": "dev"},
                  "body": TEMPLATE, "files": [{"path": "kernels/x.cu"}]}
            removed = []
            with self.subTest(tag, case="selection"):
                code, out = self._run_main(mod, tag, run(json.dumps([pr])),
                                           extra=((arb, "remove_label", mock.Mock(side_effect=lambda r, n, l: removed.append(l))),))
                self.assertIn("does not merge onto main on the box", out)
                self.assertNotIn(rebase, removed)

    def test_qwen38s_accuracy_tool_is_harness_on_every_bot(self):
        import pr_bonsai_bot as bonsai
        for mod in (qwen, muse, bonsai):
            self.assertIn("runtime/examples/qwen3_gguf_score.cpp", mod.HARNESS_PATHS, mod.__name__)
        s = qwen._remote_script("pull/1/head", role="pr", onto=MAIN)
        pin = s[s.index("git checkout -q " + MAIN + " -- "):s.index("HARNESS_PINNED")]
        self.assertIn("runtime/examples/qwen3_gguf_score.cpp", pin)

    def test_the_score_is_saved_before_the_slow_log_upload(self):
        for mod, *_ in BOTS:
            tag = "museglimmer" if mod is muse else "qwen38"
            order = []
            res = {"ok": True, "label": "XL", "delta_pct": 25.0, "pass": True, "accuracy_ok": True, "onto": MAIN}
            with self.subTest(mod.__name__), mock.patch.object(arb, "gh", return_value=run()), \
                    mock.patch.object(arb, "add_label"), mock.patch.object(arb, "remove_label"), \
                    mock.patch.object(arb, "sync_generic_eval_label"), mock.patch.object(mod, f"strip_{tag}_eval_labels"), \
                    mock.patch.object(mod, f"upload_{tag}_eval_log", side_effect=lambda *a: order.append("upload")), \
                    mock.patch.object(mod, "format_comment", return_value="c"), \
                    mock.patch.object(mod, "_load_scores", return_value={}), \
                    mock.patch.object(mod, "_save_scores", side_effect=lambda d: order.append("scores")):
                mod.apply_result("o/r", 3, "a" * 40, res)
            self.assertEqual(order, ["scores", "upload"])


class Iteration3bTests(unittest.TestCase):
    def _bots(self):
        return Round3Tests._bots(self)

    def test_a_harness_edit_at_the_fetched_tip_posts_nothing(self):
        err = "HARNESS_TOUCHED bench/scripts/_eval_speed.sh "
        for mod, tag, _r in self._bots():
            ev = {"museglimmer": "eval_museglimmer_on_box", "qwen38": "eval_qwen38_on_box",
                  "bonsai": "eval_bonsai_on_box"}[tag]
            calls = []
            with self.subTest(tag), mock.patch.object(mod, "_ssh_run_resilient", return_value=run("", 1, err)):
                res = getattr(mod, ev)("h", 1, "pull/1/head", {"sha": MAIN})
            self.assertTrue(res.get("harness"), res)
            with mock.patch.object(arb, "gh", side_effect=lambda a: calls.append(a) or run()), \
                    mock.patch.object(arb, "add_label", side_effect=lambda *a: calls.append(a)):
                mod.apply_result("o/r", 1, "a" * 40, res)
            self.assertEqual(calls, [])
            self.assertIn(f"git checkout -qf {MAIN}", mod._remote_script("pull/1/head", role="pr", onto=MAIN))
            self.assertIn("HARNESS_TOUCHED", mod._remote_script("pull/1/head", role="pr", onto=MAIN))

    def test_no_merge_gate_merges_a_harness_edit_or_files_it_cannot_see(self):
        import pr_bonsai_bot as bonsai
        for mod, okfn, prefix, first in ((muse, "auto_merge_ok_museglimmer", "eval-museglimmer:", muse.MUSEGLIMMER_MERGE_FIRST),
                                          (qwen, "auto_merge_ok_qwen38", "eval-qwen38:", qwen.QWEN38_MERGE_FIRST),
                                          (bonsai, "auto_merge_ok_bonsai", "eval-bonsai:", bonsai.BONSAI_MERGE_FIRST)):
            base = {"state": "OPEN", "isDraft": False, "labels": [{"name": prefix + "XL"}, {"name": first}],
                    "author": {"login": "dev"}, "mergeable": "MERGEABLE", "headRefOid": "a" * 40,
                    "files": [{"path": "kernels/x.cu"}], "changedFiles": 1}
            for over, why in (({"files": [{"path": "runtime/examples/qwen3_gguf_bench.cpp"}]}, "eval harness"),
                              ({"changedFiles": 250}, "more than GitHub lists")):
                with self.subTest(mod.__name__, why=why), \
                        mock.patch.object(arb, "gh", return_value=run(json.dumps(dict(base, **over)))), \
                        mock.patch.object(arb, "current_main_sha", return_value=MAIN), \
                        mock.patch.object(arb, "load_denylist", return_value=set()), \
                        mock.patch.object(arb, "author_penalty_until", return_value=None), \
                        mock.patch.object(mod, "_load_scores",
                                          return_value={"1": {"commit": "a" * 40, "label": "XL", "pass": True, "onto": MAIN}}):
                    ok, reason = getattr(mod, okfn)("o/r", 1)
                    self.assertFalse(ok)
                    self.assertIn(why, reason)

    def test_git_failing_on_the_box_is_the_boxs_on_every_bot(self):
        import pr_bonsai_bot as bonsai
        err = "RETRYABLE_INFRA_FAILURE git reset failed"
        self.assertTrue(muse._is_box_fault("", err))
        self.assertTrue(qwen._is_box_fault("", err))
        self.assertTrue(bonsai._is_infra_failure("", err))

    def test_muse_a_concurrent_width_only_the_pr_failed_is_a_two_round_reject(self):
        main = WiringTests._main(self)
        with mock.patch.object(muse, "POLARIS_ENABLED", False), \
                mock.patch.object(muse, "_ssh_run_resilient",
                                  return_value=run(muse_stdout(drop=("MUSECB ",)) + "MUSECB_FAILED 32 rc=139\n")):
            res = muse.eval_museglimmer_on_box("h", 1, "pull/1/head", main)
        self.assertEqual((res["ok"], res["label"], res["strike_key"]), (True, "REJECT", "cb"))
        calls = []
        with mock.patch.object(arb, "gh", side_effect=lambda a: calls.append(a) or run()), \
                mock.patch.object(arb, "add_label", side_effect=lambda *a: calls.append(a)):
            muse.apply_result("o/r", 6, "a" * 40, res)                   # round 1: nothing posted
        self.assertEqual(calls, [])
        self.addCleanup(arb.clear_strikes, muse.STRIKES_FILE, 6)
        # An OOM-killed width is the box's.
        with mock.patch.object(muse, "POLARIS_ENABLED", False), \
                mock.patch.object(muse, "_ssh_run_resilient",
                                  return_value=run(muse_stdout(drop=("MUSECB 32",)) + "MUSECB_FAILED 32 rc=137\n")):
            res = muse.eval_museglimmer_on_box("h", 1, "pull/1/head", main)
        self.assertTrue(res["retry"], res)

    def test_muses_baseline_needs_prefill_too(self):
        zero = muse_stdout().replace("MUSE 512 100.0 5000.0", "MUSE 512 100.0 0")
        with mock.patch.object(muse, "_ssh_run_resilient", return_value=run(zero)):
            m = muse.measure_main_baseline("h", 1)
        self.assertFalse(m["ok"])


class Iteration3cTests(unittest.TestCase):
    """Fixes from the pre-merge review of iteration 3."""

    def _bots(self):
        return Round3Tests._bots(self)

    def _pr(self, **over):
        pr = {"number": 5, "title": "t", "labels": [], "isDraft": False, "headRefOid": "a" * 40, "headRefName": "b",
              "baseRefName": "main", "mergeable": "MERGEABLE", "author": {"login": "dev"}, "body": TEMPLATE,
              "files": [{"path": "kernels/x.cu"}], "changedFiles": 1}
        pr.update(over)
        return pr

    def test_a_harness_edit_found_on_the_box_is_remembered_for_its_head(self):
        for mod, tag, _r in self._bots():
            res = {"ok": False, "harness": True, "reason": "HARNESS_TOUCHED eval/x.py", "pr_tip": "a" * 40}
            with self.subTest(tag), mock.patch.object(arb, "gh", return_value=run()):
                mod.apply_result("o/r", 5, "a" * 40, res)
            self.addCleanup(arb.clear_strikes, mod.STRIKES_FILE, 5)
            code, out = Iteration3Tests._run_main(self, mod, tag, run(json.dumps([self._pr()])))
            self.assertIn("edits the eval harness (found on the box)", out)
            self.assertEqual(mod._unmeasurable_reason("o/r", self._pr(), set()), "edits the eval harness (found on the box)")
            arb.clear_strikes(mod.STRIKES_FILE, 5)
            code, out = Iteration3Tests._run_main(self, mod, tag, run(json.dumps([self._pr(changedFiles=250)])))
            self.assertIn("more than GitHub lists", out)

    def test_a_held_pr_stays_held_after_other_bots_labels_are_dropped(self):
        for mod, tag, _r in self._bots():
            pr = self._pr(labels=[{"name": "hold"}, {"name": "eval-dspark:XL"}])
            with self.subTest(tag):
                code, out = Iteration3Tests._run_main(
                    self, mod, tag, run(json.dumps([pr])),
                    extra=((arb, "strip_foreign_stale_labels", mock.Mock(return_value={"eval-dspark:XL"})),
                           (arb, "labels_on", mock.Mock(return_value=set()))),       # a failed re-read
                    argv=("--only-prs", "5"))
                self.assertIn("hold", out)
                self.assertNotIn("greenlit", out)

    def test_labels_only_checks_the_account_first(self):
        for mod, tag, _r in self._bots():
            with self.subTest(tag), mock.patch.dict(_os.environ, {"SPARKINFER_BOT_LOGIN": "bot"}), \
                    mock.patch.object(mod.sys, "argv", [f"pr_{tag}_bot.py", "--labels-only"]), \
                    mock.patch.object(arb, "acting_account_ok", return_value=(False, "someone")), \
                    mock.patch.object(mod, f"reconcile_{tag}_merge_labels") as recon, \
                    mock.patch("builtins.print"):
                with self.assertRaises(SystemExit) as e:
                    mod.main()
                self.assertEqual(e.exception.code, 3)
                recon.assert_not_called()

    def test_no_merge_gate_merges_into_another_branch(self):
        import pr_bonsai_bot as bonsai
        for mod, okfn, prefix, first in ((muse, "auto_merge_ok_museglimmer", "eval-museglimmer:", muse.MUSEGLIMMER_MERGE_FIRST),
                                          (qwen, "auto_merge_ok_qwen38", "eval-qwen38:", qwen.QWEN38_MERGE_FIRST),
                                          (bonsai, "auto_merge_ok_bonsai", "eval-bonsai:", bonsai.BONSAI_MERGE_FIRST)):
            info = {"state": "OPEN", "isDraft": False, "labels": [{"name": prefix + "XL"}, {"name": first}],
                    "author": {"login": "dev"}, "mergeable": "MERGEABLE", "headRefOid": "a" * 40,
                    "files": [{"path": "kernels/x.cu"}], "changedFiles": 1, "baseRefName": "feat/x"}
            with self.subTest(mod.__name__), mock.patch.object(arb, "gh", return_value=run(json.dumps(info))), \
                    mock.patch.object(arb, "current_main_sha", return_value=MAIN), \
                    mock.patch.object(arb, "load_denylist", return_value=set()), \
                    mock.patch.object(arb, "author_penalty_until", return_value=None), \
                    mock.patch.object(mod, "_load_scores",
                                      return_value={"1": {"commit": "a" * 40, "label": "XL", "pass": True, "onto": MAIN}}):
                ok, why = getattr(mod, okfn)("o/r", 1)
            self.assertFalse(ok)
            self.assertIn("not main", why)

    def test_a_pr_no_bot_will_measure_is_not_waiting_on_one(self):
        with mock.patch.object(arb, "greenlight_status", return_value=("ok", "x")):
            for over in ({"baseRefName": "feat/x"}, {"changedFiles": 250}):
                with self.subTest(over):
                    self.assertFalse(arb.waiting_for_first_verdict("o/r", self._pr(**over), set()))
                    with mock.patch.object(arb, "gh", return_value=run(json.dumps({"comments": []}))):
                        self.assertFalse(arb.awaiting_any_model_verdict("o/r", self._pr(**over)))
            self.assertTrue(arb.waiting_for_first_verdict("o/r", self._pr(), set()))

    def test_only_unmerged_paths_make_a_conflict(self):
        s = arb.merged_checkout_script("pull/1/head", MAIN)
        self.assertIn("git ls-files -u", s)
        self.assertIn("RETRYABLE_INFRA_FAILURE git merge failed without a conflict", s)



DAY = 86400.0
BONSAI_ONLY = TEMPLATE.replace("- [ ] **Ternary-Bonsai-2-27B**", "- [x] **Ternary-Bonsai-2-27B**")


def qwen_stdout(top1="0.999", kl="0.001", drop=(), extra=()):
    """What a complete Qwen3.8 run prints (main's self-check included; a PR run ignores it)."""
    lines = ["PR_TIP " + "a" * 40, "REMOTE_SHA " + MAIN, "RESULT_DECODE128_TPS 80", "RESULT_PREFILL128_PP 4000",
             "RESULT_PREFILL16K_PP 8000"] + [f"RESULT_CB{c}_AGG {100 * c}" for c in qwen.CB_CONCS] + [
             f"METRIC top1={top1} kl={kl} ppl_pr=1.0 ppl_main=1.0",
             "SELFCHECK top1=1.000000 kl=0.000000 ppl_pr=1.0 ppl_main=1.0",
             "GUARD36 32768 50 900", "GUARDMO 32768 60 7000", "GUARDCBMO 16 800", "GUARDCBMO 32 1000",
             "GUARDMG_UNAVAILABLE", "GUARDBN 128 99 2000", "GUARDBN 32768 89 6500", "GUARD_END"]
    return "\n".join([l for l in lines if not any(l.startswith(d) for d in drop)] + list(extra)) + "\n"


class Iteration4Tests(unittest.TestCase):
    """Fixes from the post-merge review of main f9150ab, on all three bots."""

    def _bots(self):
        return Round3Tests._bots(self)

    def setUp(self):
        for mod, _tag, _r in self._bots():
            if _os.path.exists(mod.AUTHOR_WAIT_FILE):
                _os.remove(mod.AUTHOR_WAIT_FILE)

    def _pr(self, num=7, head="a" * 40, labels=(), **over):
        pr = {"number": num, "title": "t", "labels": [{"name": l} for l in labels], "isDraft": False,
              "headRefOid": head, "baseRefName": "main", "mergeable": "MERGEABLE", "body": TEMPLATE,
              "files": [{"path": "kernels/x.cu"}], "changedFiles": 1}
        pr.update(over)
        return pr

    def _stale(self, mod, tag, prs, now=10 * DAY, evaluated=(), scores=None, main=MAIN, dry_run=False):
        """The stale close on the real clock file (this suite's temp dir); the last commit is at 0."""
        with mock.patch.object(mod, "_pr_last_activity_ts", return_value=0.0), \
                mock.patch.object(mod.time, "time", return_value=now), \
                mock.patch.object(arb, "gh", return_value=run()), \
                mock.patch.object(arb, "current_main_sha", return_value=main), \
                mock.patch.object(arb, "greenlight_status", return_value=("ok", "claims a gain")), \
                mock.patch.object(mod, "_load_scores", return_value=scores or {}), \
                mock.patch.object(mod, "_verdict_heads", return_value=set(evaluated)), \
                mock.patch("builtins.print"):
            return getattr(mod, f"close_stale_{tag}_prs")("o/r", prs, dry_run=dry_run)

    def _clock(self, mod):
        with open(mod.AUTHOR_WAIT_FILE) as f:
            return json.load(f)

    def test_an_old_heads_needs_rebase_does_not_close_a_rebased_pr_the_bot_owes_a_verdict(self):
        for mod, tag, rebase in self._bots():
            with self.subTest(tag), mock.patch.object(arb.AuthorWaitClock, "since", _LONG_WAITED):
                pr = self._pr(labels=[rebase])       # left from an older head; the strip runs later
                self.assertEqual(self._stale(mod, tag, [pr]), set())
                # The box found that THIS head does not merge: the rebase is its author's.
                arb.record_strike(mod.STRIKES_FILE, 7, "a" * 40, "conflict")
                self.addCleanup(arb.clear_strikes, mod.STRIKES_FILE, 7)
                self.assertEqual(self._stale(mod, tag, [pr]), {7})
                arb.clear_strikes(mod.STRIKES_FILE, 7)

    def test_the_stale_clock_starts_when_the_pr_is_handed_to_its_author(self):
        for mod, tag, _r in self._bots():
            with self.subTest(tag):
                pr, done = self._pr(), {"a" * 40}   # its verdict is posted: it waits on its author
                # Its last commit is ten days old, but it has only just been handed back.
                self.assertEqual(self._stale(mod, tag, [pr], now=10 * DAY, evaluated=done), set())
                self.assertEqual(self._clock(mod)["7"], {"head": "a" * 40, "since": 10 * DAY})
                self.assertEqual(self._stale(mod, tag, [pr], now=10.9 * DAY, evaluated=done), set())
                self.assertEqual(self._stale(mod, tag, [pr], now=11.1 * DAY, evaluated=done), {7})
                # A close ends its clock: reopened, the PR has the whole period again.
                self.assertNotIn("7", self._clock(mod))
                self.assertEqual(self._stale(mod, tag, [pr], now=11.5 * DAY, evaluated=done), set())

    def test_a_pr_handed_back_within_the_day_closes_a_day_later_not_two(self):
        # The last commit is at 0. Its verdict lands in the round at 1 h; from the 2 h round on it waits
        # on its author. The clock used to start only once the commit was a day old: closed at 48 h.
        H = DAY / 24
        for mod, tag, _r in self._bots():
            with self.subTest(tag):
                self.assertEqual(self._stale(mod, tag, [self._pr()], now=1 * H, evaluated=set()), set())
                self.assertEqual(self._stale(mod, tag, [self._pr()], now=2 * H, evaluated={"a" * 40}), set())
                self.assertEqual(self._clock(mod)["7"]["since"], 2 * H)
                self.assertEqual(self._stale(mod, tag, [self._pr()], now=25 * H, evaluated={"a" * 40}), set())
                self.assertEqual(self._stale(mod, tag, [self._pr()], now=26 * H, evaluated={"a" * 40}), {7})

    def test_a_push_a_wait_on_the_bot_or_a_hold_restarts_the_clock(self):
        for mod, tag, _r in self._bots():
            with self.subTest(tag):
                self._stale(mod, tag, [self._pr()], now=10 * DAY, evaluated={"a" * 40})
                pushed = self._pr(head="b" * 40)
                self.assertEqual(self._stale(mod, tag, [pushed], now=11.5 * DAY, evaluated={"b" * 40}), set())
                self.assertEqual(self._clock(mod)["7"]["since"], 11.5 * DAY)       # a new head, a new clock
                # Greenlit with no verdict on its head: back with the bot, so the clock is dropped ...
                self.assertEqual(self._stale(mod, tag, [pushed], now=12 * DAY, evaluated=set()), set())
                self.assertNotIn("7", self._clock(mod))
                # ... and a `hold` drops it too.
                self._stale(mod, tag, [pushed], now=12.5 * DAY, evaluated={"b" * 40})
                self._stale(mod, tag, [self._pr(head="b" * 40, labels=["hold"])], now=13 * DAY, evaluated={"b" * 40})
                self.assertNotIn("7", self._clock(mod))
                self.assertEqual(self._stale(mod, tag, [pushed], now=13.5 * DAY, evaluated={"b" * 40}), set())

    def test_the_clock_forgets_closed_prs_and_a_dry_run_writes_nothing(self):
        for mod, tag, _r in self._bots():
            with self.subTest(tag):
                self._stale(mod, tag, [self._pr(), self._pr(num=8)], evaluated={"a" * 40}, dry_run=True)
                self.assertFalse(_os.path.exists(mod.AUTHOR_WAIT_FILE))
                self._stale(mod, tag, [self._pr(), self._pr(num=8)], evaluated={"a" * 40})
                self.assertEqual(set(self._clock(mod)), {"7", "8"})
                self._stale(mod, tag, [self._pr()], evaluated={"a" * 40})      # #8 is no longer open
                self.assertEqual(set(self._clock(mod)), {"7"})

    def test_a_ranking_loser_waits_for_the_winners_merge_not_on_its_author(self):
        for mod, tag, rebase in self._bots():
            pr = self._pr(labels=[rebase, f"eval-{tag}:L"])
            scores = {"7": {"commit": "a" * 40, "label": "L", "pass": True, "onto": MAIN}}
            with self.subTest(tag), mock.patch.object(arb.AuthorWaitClock, "since", _LONG_WAITED):
                # Main has not moved since its verdict: there is nothing to rebase onto yet.
                self.assertEqual(self._stale(mod, tag, [pr], evaluated={"a" * 40}, scores=scores), set())
                # The winner merged and main moved: the rebase is its author's now.
                self.assertEqual(self._stale(mod, tag, [pr], evaluated={"a" * 40}, scores=scores, main="d" * 40), {7})
                # A conflict is its author's at once.
                self.assertEqual(self._stale(mod, tag, [dict(pr, mergeable="CONFLICTING")], evaluated={"a" * 40},
                                             scores=scores), {7})

    def test_a_zero_threshold_or_autoclose_off_closes_nothing(self):
        for mod, tag, _r in self._bots():
            with self.subTest(tag), mock.patch.object(mod, "STALE_DAYS", 0.0), \
                    mock.patch.object(arb.AuthorWaitClock, "since", _LONG_WAITED):
                self.assertEqual(self._stale(mod, tag, [self._pr()], evaluated={"a" * 40}), set())
        with mock.patch.object(_bonsai, "AUTO_CLOSE", False), mock.patch.object(arb.AuthorWaitClock, "since", _LONG_WAITED):
            self.assertEqual(self._stale(_bonsai, "bonsai", [self._pr()], evaluated={"a" * 40}), set())

    def _verdict_close(self, mod, tag, view):
        calls = []

        def fake_gh(a):
            calls.append(a)
            return run(json.dumps(view)) if a[:2] == ["pr", "view"] else run()
        res = {"ok": True, "label": "REJECT", "delta_pct": -9.0, "pass": False, "accuracy_ok": True,
               "reason": "decode@128 regressed"}
        kw = {"body": BONSAI_ONLY} if tag == "bonsai" else {"pr_body": TEMPLATE}
        with mock.patch.object(arb, "gh", side_effect=fake_gh), \
                mock.patch.object(arb, "add_label"), mock.patch.object(arb, "remove_label"), \
                mock.patch.object(arb, "sync_generic_eval_label"), \
                mock.patch.object(arb, "labels_on_or_none", return_value=set()), \
                mock.patch.object(mod, f"strip_{tag}_eval_labels"), \
                mock.patch.object(mod, f"upload_{tag}_eval_log"), \
                mock.patch.object(mod, "format_comment", return_value="the verdict comment"), \
                mock.patch.object(mod, "_load_scores", return_value={}), \
                mock.patch.object(mod, "_save_scores"), \
                mock.patch("builtins.print"):
            if tag == "bonsai":
                with mock.patch.object(mod, "AUTO_CLOSE", True):
                    mod.apply_result("o/r", 1, "a" * 40, res, **kw)
            else:
                mod.apply_result("o/r", 1, "a" * 40, res, **kw)
        return any(c[:2] == ["pr", "close"] for c in calls)

    def test_a_hold_or_draft_made_while_the_round_ran_is_not_closed_over(self):
        base = {"headRefOid": "a" * 40, "state": "OPEN", "isDraft": False, "labels": []}
        for mod, tag, _r in self._bots():
            with self.subTest(tag):
                self.assertTrue(self._verdict_close(mod, tag, base))
                for view in (dict(base, labels=[{"name": "hold"}]), dict(base, isDraft=True),
                             dict(base, headRefOid="b" * 40), dict(base, state="CLOSED"), {}):
                    self.assertFalse(self._verdict_close(mod, tag, view), view)

    def test_an_unread_label_set_never_changes_a_generic_label(self):
        seen = []
        with mock.patch.object(arb, "labels_on_or_none", return_value=None), \
                mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: seen.append("-" + l)), \
                mock.patch.object(arb, "add_label", side_effect=lambda r, n, l: seen.append("+" + l)):
            self.assertIs(arb.sync_generic_eval_label("o/r", 1), False)
        self.assertEqual(seen, [])
        # The sync's read fails and the strip's own succeeds: eval:XL, still backed by Muse's XL, stays.
        for strip in (lambda labs: arb.strip_stale_verdict_labels("o/r", 1, set(labs), "eval-qwen38:", "b" * 40,
                                                                  {"a" * 40}),
                      lambda labs: arb.strip_foreign_stale_labels("o/r", 1, set(labs), "b" * 40, "eval-bonsai:")):
            labels = {"eval-museglimmer:XL", "eval:XL", "eval-qwen38:none"}
            reads = iter([None, set(labels)])
            with mock.patch.object(arb, "labels_on_or_none", side_effect=lambda r, n: next(reads)), \
                    mock.patch.object(arb, "bot_verdict_heads", return_value={"qwen38": {"a" * 40}}), \
                    mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: labels.discard(l)), \
                    mock.patch.object(arb, "add_label", side_effect=lambda r, n, l: labels.add(l)):
                strip(set(labels))
            self.assertEqual(labels, {"eval-museglimmer:XL", "eval:XL"})

    def test_a_bots_own_tier_is_removed_even_when_the_labels_cannot_be_read(self):
        for mod, tag, _r in self._bots():
            removed = []
            with self.subTest(tag), mock.patch.object(arb, "labels_on_or_none", return_value=None), \
                    mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: removed.append(l)):
                getattr(mod, f"strip_{tag}_eval_labels")("o/r", 1)
            self.assertEqual(set(removed), {f"eval-{tag}:{t}" for t in arb.GENERIC_TIER_RANK})
            removed = []
            with mock.patch.object(arb, "labels_on_or_none", return_value={f"eval-{tag}:XL", "eval:XL", "hold"}), \
                    mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: removed.append(l)):
                getattr(mod, f"strip_{tag}_eval_labels")("o/r", 1)
            self.assertEqual(removed, [f"eval-{tag}:XL"])

    def test_a_run_that_gave_up_on_a_pr_exits_3_with_the_gpu_down_or_the_lock_busy(self):
        prs = [Iteration3cTests._pr(self, number=5), Iteration3cTests._pr(self, number=6, headRefOid="b" * 40)]
        for mod, tag, _r in self._bots():
            self.addCleanup(mod.GAVE_UP.clear)
            self.addCleanup(arb.clear_strikes, mod.STRIKES_FILE, 5)
            for _ in range(arb.BOX_FAULT_STRIKES):
                arb.record_strike(mod.STRIKES_FILE, 5, "a" * 40, "error")
            with self.subTest(tag, case="gpu down"):
                code, out = Iteration3Tests._run_main(self, mod, tag, run(json.dumps(prs)), argv=("--only-prs", "5,6"))
                self.assertIn("GPU down", out)
                self.assertEqual(code, 3)
            mod.GAVE_UP.clear()
            with self.subTest(tag, case="lock busy"):
                code, out = Iteration3Tests._run_main(
                    self, mod, tag, run(json.dumps(prs)), argv=("--only-prs", "5,6"),
                    extra=((mod, "resolve_ssh", mock.Mock(return_value=("h", 1))),
                           (arb, "hold_bot_lock", mock.Mock(return_value=False))))
                self.assertIn("lock stayed busy", out)
                self.assertEqual(code, 3)
            mod.GAVE_UP.clear()

    def test_qwen38_checks_mains_score_dump_against_itself(self):
        m = qwen._remote_script("main", role="main")
        self.assertIn("--metric-label SELFCHECK", m)
        self.assertIn("IS_PR=0", m)                   # main takes the self-check branch, a PR the compare
        with mock.patch.object(qwen, "_ssh_run_resilient", return_value=run(qwen_stdout())):
            self.assertTrue(qwen.measure_main_baseline("h", 1)["ok"])
        for bad in (qwen_stdout(drop=("SELFCHECK",)),
                    qwen_stdout(drop=("SELFCHECK",), extra=("SELFCHECK top1=0 kl=99 ppl_pr=0 ppl_main=0",))):
            with mock.patch.object(qwen, "_ssh_run_resilient", return_value=run(bad)):
                m = qwen.measure_main_baseline("h", 1)
            self.assertFalse(m["ok"])
            self.assertIn("self-comparison", m["reason"])

    def test_a_guard_the_oom_killer_took_beside_a_failed_accuracy_gate_is_not_blamed(self):
        main = WiringTests._main(self)
        stdout = muse_stdout(top1="0.2", kl="2.0", drop=("GUARDMO ",)) + "GUARDMO_FAILED rc=137\n"
        with mock.patch.object(muse, "POLARIS_ENABLED", False), \
                mock.patch.object(muse, "_ssh_run_resilient", return_value=run(stdout)):
            res = muse.eval_museglimmer_on_box("h", 1, "pull/1/head", main)
        self.assertEqual((res["label"], res["modelopt_guard_ok"], res["guards_killed"]), ("REJECT", True, ["modelopt"]))
        self.assertNotIn("modelopt no-regression guard failed", res["reason"])
        self.assertIn("NOT MEASURED", muse.format_comment("a" * 40, res))
        with mock.patch.object(qwen, "_ssh_run_resilient", return_value=run(qwen_stdout())):
            qmain = qwen.measure_main_baseline("h", 1)
        stdout = qwen_stdout(top1="0.2", kl="2.0", drop=("GUARD36 ",), extra=("GUARD36_FAILED rc=137",))
        with mock.patch.object(qwen, "POLARIS_ENABLED", False), \
                mock.patch.object(qwen, "_ssh_run_resilient", return_value=run(stdout)):
            qres = qwen.eval_qwen38_on_box("h", 1, "pull/1/head", qmain)
        self.assertEqual((qres["label"], qres["q36_guard_ok"], qres["guards_killed"]), ("REJECT", True, ["qwen3.6"]))
        self.assertNotIn("qwen3.6 no-regression guard failed", qres["reason"])
        self.assertIn("NOT MEASURED", qwen.format_comment("a" * 40, qres))
        # The close comment names the accuracy gate, which is what failed.
        for mod, tag, r in ((muse, "museglimmer", res), (qwen, "qwen38", qres)):
            calls = []

            def fake_gh(a):
                calls.append(a)
                return run(json.dumps({"headRefOid": "a" * 40})) if a[:2] == ["pr", "view"] else run()
            with self.subTest(tag), mock.patch.object(arb, "gh", side_effect=fake_gh), \
                    mock.patch.object(arb, "add_label"), mock.patch.object(arb, "remove_label"), \
                    mock.patch.object(arb, "sync_generic_eval_label"), \
                    mock.patch.object(mod, f"upload_{tag}_eval_log"), mock.patch.object(mod, "_save_scores"), \
                    mock.patch("builtins.print"):
                mod.apply_result("o/r", 1, "a" * 40, dict(r, delta_pct=r.get("delta_pct") or 0.0), pr_body=TEMPLATE)
            close = next(" ".join(c) for c in calls if "auto-close -->" in " ".join(c))
            self.assertIn("failed the accuracy gate", close)
            self.assertNotIn("guard", close.split("Every")[0])

    def test_bonsai_quotes_the_rejects_own_reason_first(self):
        with open(_bonsai.__file__) as f:
            self.assertIn('reason = f"{reason} | {whys}"', f.read())

    def test_a_re_measure_onto_a_new_main_gets_its_own_eval_log_run(self):
        with mock.patch.object(arb, "LOG_DIR", _os.path.join(_STATE, "log")):
            base = "qwen38-0007-aaaaaaa"
            self.assertEqual(arb.eval_log_run_id(base, MAIN), base)
            run_dir = _os.path.join(arb.LOG_DIR, "runs", base)
            _os.makedirs(run_dir, exist_ok=True)
            with open(_os.path.join(run_dir, "result.json"), "w") as f:
                json.dump({"measured_onto": MAIN}, f)
            self.assertEqual(arb.eval_log_run_id(base, MAIN), base)                    # the same run again
            self.assertEqual(arb.eval_log_run_id(base, "d" * 40), base + "-onddddddd")  # a re-measure
            with open(_os.path.join(run_dir, "result.json"), "w") as f:
                json.dump({}, f)                                                         # from before the field
            self.assertEqual(arb.eval_log_run_id(base, "d" * 40), base)

    def test_a_conflict_is_remembered_for_the_commit_that_conflicted(self):
        err = "MERGE_CONFLICT_TIP " + "b" * 40 + "\nMERGE_CONFLICT bbbbbbb does not merge cleanly onto ccccccc\n"
        self.assertEqual(arb.merge_conflict_tip("", err), "b" * 40)
        self.assertEqual(arb.merge_conflict_tip("PR_TIP " + "b" * 40, err), "")          # not where it stopped
        self.assertIn("MERGE_CONFLICT_TIP", arb.merged_checkout_script("pull/1/head", MAIN))
        for mod, tag, _r in self._bots():
            ev = {"museglimmer": "eval_museglimmer_on_box", "qwen38": "eval_qwen38_on_box",
                  "bonsai": "eval_bonsai_on_box"}[tag]
            with self.subTest(tag), mock.patch.object(mod, "_ssh_run_resilient", return_value=run("", 1, err)):
                res = getattr(mod, ev)("h", 1, "pull/1/head", {"sha": MAIN})
            self.assertTrue(res.get("conflict"), res)
            self.assertEqual(arb.measured_commit("a" * 40, res), ("b" * 40, True))

    def test_the_boxs_own_failures_are_read_as_the_boxs(self):
        self.assertTrue(_bonsai._is_infra_failure("", "HARNESS_PIN_FAILED -- could not take the harness from x\n"))
        self.assertTrue(_bonsai._is_infra_failure("", "TOKENIZE_FAILED\n"))
        # A fault charged after BOX_FAULT_STRIKES rounds names its cause, not "no diagnostic".
        err = "RETRYABLE_INFRA_FAILURE build: Killed signal terminated program cc1plus\n"
        self.assertEqual(arb.infra_failure_line("", err), err.strip())
        with mock.patch.object(muse, "_ssh_run_resilient", return_value=run("PR_TIP " + "a" * 40 + "\n", 1, err)):
            res = muse.eval_museglimmer_on_box("h", 1, "pull/1/head", {"sha": MAIN})
        self.assertIn("Killed signal", res["reason"])
        # Qwen3.8: a width only the PR run could not complete is judged over rounds, not REJECTed once.
        cb = "RETRYABLE_INFRA_FAILURE concurrent decode failed at c=32 (see above)\n"
        with mock.patch.object(qwen, "_ssh_run_resilient", return_value=run("PR_TIP " + "a" * 40 + "\n", 75, cb)):
            q = qwen.eval_qwen38_on_box("h", 1, "pull/1/head", {"sha": MAIN})
        self.assertEqual((q["retry"], q["strike_key"]), (True, "cb"))

    def test_a_pr_based_on_a_newer_main_waits_a_round_with_no_strike(self):
        err = "BASE_AHEAD bbbbbbb is based on main ddddddd, newer than this round's baseline ccccccc\n"
        for mod, tag, _r in self._bots():
            ev = {"museglimmer": "eval_museglimmer_on_box", "qwen38": "eval_qwen38_on_box",
                  "bonsai": "eval_bonsai_on_box"}[tag]
            with self.subTest(tag), mock.patch.object(mod, "ssh_run", return_value=run("", 1, err)) as ssh, \
                    mock.patch("builtins.print"):
                res = getattr(mod, ev)("h", 1, "pull/1/head", {"sha": MAIN})
            self.assertEqual((res["ok"], res["retry"], res.get("strike_key")), (False, True, None), res)
            self.assertEqual(ssh.call_count, 1)             # stopped on purpose: not a hard kill to retry
            calls = []
            with mock.patch.object(arb, "gh", side_effect=lambda a: calls.append(a) or run()), \
                    mock.patch.object(arb, "add_label", side_effect=lambda *a: calls.append(a)), \
                    mock.patch("builtins.print"):
                mod.apply_result("o/r", 1, "a" * 40, res)
            self.assertEqual(calls, [])
            self.assertEqual(arb.strike_count(mod.STRIKES_FILE, 1, "a" * 40, "box"), 0)

    def test_a_verdict_comment_always_fits_on_github(self):
        body = "<!-- marker -->\n" + "x" * 100000
        fitted = arb.fit_comment(body)
        self.assertLessEqual(len(fitted), arb.COMMENT_LIMIT)
        self.assertTrue(fitted.startswith("<!-- marker -->"))
        self.assertEqual(arb.fit_comment("short"), "short")



class Iteration5Tests(unittest.TestCase):
    """Fixes from the post-merge review of main 5edba3c, on all three bots."""

    def _bots(self):
        out = []
        for mod, tag, rebase in Round3Tests._bots(self):
            up = {"museglimmer": "MUSEGLIMMER", "qwen38": "QWEN38", "bonsai": "BONSAI"}[tag]
            out.append((mod, tag, rebase, mod.EVAL_PREFIX, getattr(mod, f"{up}_MERGE_FIRST")))
        return out

    def setUp(self):
        for mod, *_ in self._bots():
            if _os.path.exists(mod.AUTHOR_WAIT_FILE):
                _os.remove(mod.AUTHOR_WAIT_FILE)

    def _reconcile(self, mod, tag, prs, scores, main=MAIN):
        """prs: {num: (labels, head)}. The merge gate is mocked; it refuses a PR's own needs-rebase
        unless reconcile says the label only records a lost ranking (ranking_loss_ok)."""
        calls, gate = [], []
        rebase = self._bots()[[b[1] for b in self._bots()].index(tag)][2]

        def fake_gh(a):
            if a[:2] == ["pr", "list"] and "open" in a:
                return run(json.dumps([{"number": n, "labels": [{"name": l} for l in labs], "headRefOid": head,
                                        "mergeable": "MERGEABLE", "isDraft": False, "body": TEMPLATE, "files": [],
                                        "baseRefName": "main"} for n, (labs, head) in prs.items()]))
            return run("[]")

        def fake_ok(r, n, require_merge_first=True, ranking_loss_ok=False):
            gate.append((n, ranking_loss_ok))
            if rebase in prs[n][0] and not ranking_loss_ok:
                return False, f"blocking label(s): {rebase}"
            return True, "ok"
        with mock.patch.object(arb, "gh", side_effect=fake_gh), \
                mock.patch.object(arb, "current_main_sha", return_value=main), \
                mock.patch.object(arb, "add_label", side_effect=lambda r, n, l: calls.append(("add", n, l))), \
                mock.patch.object(arb, "remove_label", side_effect=lambda r, n, l: calls.append(("rm", n, l))), \
                mock.patch.object(arb, "sync_generic_eval_label"), \
                mock.patch.object(mod, "_load_scores", return_value=scores), \
                mock.patch.object(mod, f"auto_merge_ok_{tag}", side_effect=fake_ok), \
                mock.patch.object(mod, "AUTO_MERGE", False), mock.patch("builtins.print"):
            getattr(mod, f"reconcile_{tag}_merge_labels")("o/r")
        return calls, gate

    def test_a_fresh_ranking_loser_is_still_in_the_running(self):
        for mod, tag, rebase, prefix, first in self._bots():
            scores = {"10": {"commit": "a" * 40, "label": "XS", "delta_pct": 2.5, "pass": True, "onto": MAIN},
                      "11": {"commit": "b" * 40, "label": "L", "delta_pct": 12.0, "pass": True, "onto": MAIN}}
            with self.subTest(tag, case="the winner re-measured lower"):
                calls, gate = self._reconcile(mod, tag, {10: ([prefix + "XS", first], "a" * 40),
                                                         11: ([prefix + "L", rebase], "b" * 40)}, scores)
                self.assertIn((11, True), gate)
                self.assertIn(("add", 11, first), calls)
                self.assertIn(("rm", 11, rebase), calls)
                self.assertIn(("add", 10, rebase), calls)
                self.assertIn(("rm", 10, first), calls)
            with self.subTest(tag, case="the winner was held"):
                calls, _ = self._reconcile(mod, tag, {10: ([prefix + "XL", first, "hold"], "a" * 40),
                                                      11: ([prefix + "L", rebase], "b" * 40)}, scores)
                self.assertIn(("add", 11, first), calls)
            with self.subTest(tag, case="main moved: the rebase is its author's"):
                calls, gate = self._reconcile(mod, tag, {10: ([prefix + "XS", first], "a" * 40),
                                                         11: ([prefix + "L", rebase], "b" * 40)}, scores,
                                              main="d" * 40)
                self.assertNotIn(("add", 11, first), calls)
                self.assertNotIn(11, [n for n, _ in gate])
            with self.subTest(tag, case="another block label"):
                calls, _ = self._reconcile(mod, tag, {11: ([prefix + "L", rebase, "penalty"], "b" * 40)}, scores)
                self.assertNotIn(("add", 11, first), calls)

    def test_reconcile_heals_a_generic_label_a_failed_sync_left_wrong(self):
        for mod, tag, rebase, prefix, first in self._bots():
            with self.subTest(tag):
                with mock.patch.object(arb, "gh", side_effect=lambda a: run(json.dumps([
                            {"number": 5, "labels": [{"name": prefix + "L"}, {"name": "eval:none"}]},
                            {"number": 6, "labels": [{"name": prefix + "S"}, {"name": "eval:S"}]},
                            {"number": 7, "labels": [{"name": "eval:M"}]},
                            # the retired AR bot's labels: its own rule (the failing side), not ours
                            {"number": 8, "labels": [{"name": "eval-qwen35:M"}, {"name": "eval-qwen36:XL"},
                                                     {"name": "eval:M"}]}]))
                            if a[:2] == ["pr", "list"] and "open" in a else run("[]")), \
                        mock.patch.object(arb, "sync_generic_eval_label") as sync, \
                        mock.patch.object(arb, "add_label"), mock.patch.object(arb, "remove_label"), \
                        mock.patch.object(mod, "_load_scores", return_value={}), \
                        mock.patch.object(mod, "AUTO_MERGE", False), mock.patch("builtins.print"):
                    getattr(mod, f"reconcile_{tag}_merge_labels")("o/r")
                self.assertEqual([c.args[1] for c in sync.call_args_list], [5])
        self.assertFalse(arb.generic_label_out_of_sync({"eval:M"}))                 # nothing to mirror
        self.assertTrue(arb.generic_label_out_of_sync({"eval-qwen38:S", "eval-bonsai:REJECT", "eval:S"}))

    def test_the_stale_close_leaves_another_bots_verified_speedup_alone(self):
        for mod, tag, rebase, prefix, first in self._bots():
            other = "eval-dspark:L"
            with self.subTest(tag), mock.patch.object(arb.AuthorWaitClock, "since", _LONG_WAITED):
                pr = Iteration4Tests._pr(self, labels=[other, "eval-dspark:none"])
                self.assertEqual(Iteration4Tests._stale(self, mod, tag, [pr], evaluated={"a" * 40}), set())
                own = Iteration4Tests._pr(self, labels=[prefix + "L"])       # its own tier protects nothing
                self.assertEqual(Iteration4Tests._stale(self, mod, tag, [own], evaluated={"a" * 40}), {7})
        self.assertIsNone(arb.stale_close_skip_reason({"labels": [{"name": "eval-qwen38:none"}]}, "muse",
                                                      "eval-museglimmer:"))

    def test_a_commit_dated_in_the_future_does_not_stop_the_clock(self):
        for mod, tag, *_ in self._bots():
            with self.subTest(tag), mock.patch.object(mod, "_pr_last_activity_ts", return_value=100 * DAY), \
                    mock.patch.object(mod.time, "time", return_value=10 * DAY), \
                    mock.patch.object(arb, "gh", return_value=run()), \
                    mock.patch.object(arb, "current_main_sha", return_value=MAIN), \
                    mock.patch.object(mod, "_verdict_heads", return_value={"a" * 40}), \
                    mock.patch.object(arb.AuthorWaitClock, "since", lambda self, n, h: 8 * DAY), \
                    mock.patch("builtins.print"):
                # Handed back at day 8: closed a day later, whatever date the commit claims.
                self.assertEqual(getattr(mod, f"close_stale_{tag}_prs")("o/r", [Iteration4Tests._pr(self)]), {7})
            with mock.patch.object(mod, "_pr_last_activity_ts", return_value=100 * DAY), \
                    mock.patch.object(mod.time, "time", return_value=10 * DAY), \
                    mock.patch.object(arb, "gh", return_value=run()), \
                    mock.patch.object(arb, "current_main_sha", return_value=MAIN), \
                    mock.patch.object(mod, "_verdict_heads", return_value={"a" * 40}), \
                    mock.patch.object(arb.AuthorWaitClock, "since", lambda self, n, h: 9.5 * DAY), \
                    mock.patch("builtins.print"):
                self.assertEqual(getattr(mod, f"close_stale_{tag}_prs")("o/r", [Iteration4Tests._pr(self)]), set())

    def test_the_clock_is_dropped_on_every_keep_path_and_never_starts_in_the_future(self):
        clock = arb.AuthorWaitClock(_os.path.join(_STATE, "clock.json"), [7], 100.0)
        clock.data["7"] = {"head": "a" * 40, "since": 500.0}                    # later than now: not trusted
        self.assertEqual(clock.since(7, "a" * 40), 100.0)
        for mod, tag, *_ in self._bots():
            for case, extra in (("gave up", ((arb, "gave_up", mock.Mock(return_value=True)),
                                              (mod, "_unmeasurable_reason", mock.Mock(return_value=None)))),
                                ("owed", ((mod, "_remeasure_state", mock.Mock(return_value=True)),
                                          (mod, "_unmeasurable_reason", mock.Mock(return_value=None))))):
                with self.subTest(tag, case=case):
                    Iteration4Tests._stale(self, mod, tag, [Iteration4Tests._pr(self)], evaluated={"a" * 40})
                    self.assertIn("7", Iteration4Tests._clock(self, mod))
                    ctx = [mock.patch.object(*x) for x in extra]
                    for c in ctx:
                        c.start()
                    try:
                        Iteration4Tests._stale(self, mod, tag, [Iteration4Tests._pr(self)], evaluated={"a" * 40})
                    finally:
                        for c in ctx:
                            c.stop()
                    self.assertNotIn("7", Iteration4Tests._clock(self, mod))

    def test_only_a_ranking_loss_waits_for_the_winner(self):
        for mod, tag, rebase, prefix, first in self._bots():
            entry = {"7": {"commit": "a" * 40, "label": "L", "pass": True, "onto": MAIN}}
            pr = Iteration4Tests._pr(self)
            with self.subTest(tag), mock.patch.object(mod, "_load_scores", return_value=entry):
                self.assertTrue(mod._waits_for_the_winner(pr, {rebase}, "a" * 40, MAIN))
                self.assertFalse(mod._waits_for_the_winner(pr, {rebase, "penalty"}, "a" * 40, MAIN))
                self.assertFalse(mod._waits_for_the_winner(pr, {rebase}, "a" * 40, "d" * 40))
                self.assertFalse(mod._waits_for_the_winner(pr, {rebase}, "b" * 40, MAIN))

    def test_a_pr_with_many_labels_is_read_whole(self):
        seen = []
        with mock.patch.object(arb, "gh", side_effect=lambda a: seen.append(a) or run("[]")):
            arb.labels_on_or_none("o/r", 1)
            arb.labels_on("o/r", 1)
        self.assertTrue(all("per_page=100" in a[1] for a in seen), seen)



class Iteration5bTests(unittest.TestCase):
    """The shell side of the post-merge review of 5edba3c: the reference server, coverage, causes."""

    ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(arb.__file__)))

    def _dump(self, path, positions):
        with open(path, "w") as f:
            for i in positions:
                f.write(f"S i={i} tgt=5 am=5 lp=-0.1 top=5:-0.1,6:-2.5\n")

    def _serve(self, handler_body):
        import http.server
        import threading

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                handler_body(self)

            def log_message(self, *a):
                pass
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)                               # runs first
        return f"http://127.0.0.1:{srv.server_address[1]}"

    def _compare(self, url, positions, n_ids=9):
        import subprocess
        d = _tempfile.mkdtemp(dir=_STATE)
        with open(_os.path.join(d, "tokenizers.py"), "w") as f:
            f.write("class Tokenizer:\n    pass\n")                  # the ids file is used, not a tokenizer
        self._dump(_os.path.join(d, "spark.txt"), positions)
        with open(_os.path.join(d, "ids.txt"), "w") as f:
            f.write(" ".join(str(i) for i in range(n_ids)))
        env = dict(_os.environ, PYTHONPATH=d)
        return subprocess.run(["python3", _os.path.join(self.ROOT, "bench/scripts/accuracy_compare.py"),
                               _os.path.join(d, "spark.txt"), "/dev/null", _os.path.join(d, "ids.txt"), url, "8"],
                              capture_output=True, text=True, timeout=60, env=env)

    def test_the_reference_server_failing_is_told_apart_from_the_prs_dump(self):
        def fail(h):
            h.send_response(500)
            h.end_headers()
        r = self._compare(self._serve(fail), range(8))
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("REFERENCE_FAILED", r.stderr)

        def empty(h):
            h.rfile.read(int(h.headers["Content-Length"]))
            body = json.dumps({"completion_probabilities": [{"top_logprobs": []}]}).encode()
            h.send_response(200)
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
        r = self._compare(self._serve(empty), range(8))              # an answer with nothing in it
        self.assertEqual(r.returncode, 3, r.stderr)

        def ok(h):
            h.rfile.read(int(h.headers["Content-Length"]))
            body = json.dumps({"completion_probabilities": [{"top_logprobs": [
                {"id": 5, "logprob": -0.1}, {"id": 6, "logprob": -2.5}]}]}).encode()
            h.send_response(200)
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
        url = self._serve(ok)
        full = self._compare(url, range(8))
        self.assertEqual(full.returncode, 0, full.stderr)
        self.assertIn(" n=8 n_expected=8", full.stdout)
        short = self._compare(url, range(3))                        # a dump that stopped early
        self.assertIn(" n=3 n_expected=8", short.stdout)
        # Muse reads exit 3 as its reference's failure (infra), and every wait on the server is bounded.
        s = muse._remote_script("pull/1/head", role="pr", onto=MAIN)
        self.assertIn('[ "$ACC_RC" = 124 ] || [ "$ACC_RC" = 3 ]', s)
        self.assertIn("ACCOUT=$(timeout 1800 python3 bench/scripts/accuracy_compare.py", s)
        self.assertNotIn('curl -s "http', s)
        self.assertEqual(s.count("curl -s --max-time 10"), 3)

    def test_a_score_dump_missing_positions_fails_the_accuracy_gate(self):
        import subprocess
        d = _tempfile.mkdtemp(dir=_STATE)
        self._dump(_os.path.join(d, "pr.txt"), range(3))
        self._dump(_os.path.join(d, "main.txt"), range(88))
        r = subprocess.run(["python3", _os.path.join(self.ROOT, "bench/scripts/accuracy_compare_pair.py"),
                            _os.path.join(d, "pr.txt"), _os.path.join(d, "main.txt")],
                           capture_output=True, text=True, timeout=60)
        metric = next(l for l in r.stdout.splitlines() if l.startswith("METRIC "))
        self.assertIn("top1=1.000000", metric)                      # the 3 it has agree ...
        self.assertIn("n=3 n_main=88", metric)                     # ... but it has only 3
        with mock.patch.object(qwen, "_ssh_run_resilient", return_value=run(qwen_stdout())):
            qmain = qwen.measure_main_baseline("h", 1)
        pr = qwen_stdout(drop=("METRIC",), extra=(metric,))
        with mock.patch.object(qwen, "POLARIS_ENABLED", False), \
                mock.patch.object(qwen, "_ssh_run_resilient", return_value=run(pr)), mock.patch("builtins.print"):
            res = qwen.eval_qwen38_on_box("h", 1, "pull/1/head", qmain)
        self.assertEqual((res["label"], res["accuracy_ok"]), ("REJECT", False))
        self.assertIn("covers 3 of 88 positions", res["reason"])
        main = WiringTests._main(self)
        with mock.patch.object(muse, "POLARIS_ENABLED", False), mock.patch("builtins.print"), \
                mock.patch.object(muse, "_ssh_run_resilient",
                                  return_value=run(muse_stdout() + "RESULT_ACC_POSITIONS 3 88\n")):
            mres = muse.eval_museglimmer_on_box("h", 1, "pull/1/head", main)
        self.assertEqual((mres["label"], mres["accuracy_ok"]), ("REJECT", False))
        self.assertIn("covers 3 of 88 positions", mres["reason"])
        # main's own dump short of the stream: the round is skipped, not every PR REJECTed.
        with mock.patch.object(muse, "_ssh_run_resilient", return_value=run(muse_stdout() + "RESULT_ACC_POSITIONS 7 8\n")):
            m = muse.measure_main_baseline("h", 1)
        self.assertFalse(m["ok"])
        self.assertIn("positions=7/8", m["reason"])

    def test_a_charged_box_fault_names_its_cause(self):
        err = ("RETRYABLE_INFRA_FAILURE GPU still holding 9000 MiB after 180s — refusing to start a load\n"
               "REMOTE_SCRIPT_FAILED line=88 exit=1 reason=\n")
        self.assertTrue(arb.failure_cause("REMOTE_SCRIPT_FAILED line=88 exit=1 reason=", "", err)
                        .startswith("RETRYABLE_INFRA_FAILURE GPU still holding"))
        self.assertEqual(arb.failure_cause("BUILD_FAILED: x.cu:1: error", "", err), "BUILD_FAILED: x.cu:1: error")
        self.assertEqual(arb.failure_cause(None, "", ""), None)

    def test_the_box_side_rules_no_other_test_pins(self):
        # Qwen3.8: main's score dump gone when the PR compared is the box's, bounded as such.
        with mock.patch.object(qwen, "_ssh_run_resilient",
                               return_value=run(qwen_stdout(drop=("METRIC",)), 0, "ACCURACY_NO_BASELINE\n")):
            q = qwen.eval_qwen38_on_box("h", 1, "pull/1/head", {"sha": MAIN})
        self.assertEqual((q["retry"], q["strike_key"]), (True, "box"))
        # Muse keeps a killed width's exit code; Bonsai calls a failed width the run's unless the GPU
        # did not drain; the round guard stops the orphan's whole ssh session.
        self.assertIn("|| { CB_RC=$?; false; }", muse._remote_script("pull/1/head", role="pr", onto=MAIN))
        self.assertIn("CB_WHY=run", _bonsai._remote_script("pull/1/head", role="pr", onto=MAIN))
        self.assertIn('pkill -TERM -s "$opid"', arb.round_guard_sh("qwen38"))



class Iteration6Tests(unittest.TestCase):
    """Fixes from the post-merge review of main 7c63687, on all three bots."""

    ROOT = Iteration5bTests.ROOT

    def _bots(self):
        return Iteration5Tests._bots(self)

    def _reconcile(self, mod, tag, prs, scores, evaluated=("a" * 40,)):
        with mock.patch.object(mod, "_verdict_heads", return_value=set(evaluated)):
            return Iteration5Tests._reconcile(self, mod, tag, prs, scores)

    def test_a_tier_whose_label_write_failed_is_put_back_from_the_recorded_verdict(self):
        for mod, tag, rebase, prefix, first in self._bots():
            xl = {"commit": "a" * 40, "label": "XL", "delta_pct": 30.0, "pass": True, "onto": MAIN}
            with self.subTest(tag, case="the add failed"):
                calls, _ = self._reconcile(mod, tag, {5: ([], "a" * 40)}, {"5": xl})
                self.assertIn(("add", 5, prefix + "XL"), calls)
                self.assertIn(("add", 5, first), calls)              # ... and it is ranked again
            with self.subTest(tag, case="a removal failed"):
                none = dict(xl, label="none", delta_pct=0.1, **{"pass": False})
                calls, _ = self._reconcile(mod, tag, {5: ([prefix + "XL", prefix + "none"], "a" * 40)}, {"5": none})
                self.assertIn(("rm", 5, prefix + "XL"), calls)
                self.assertNotIn(("add", 5, first), calls)
            for case, labels, entry, evaluated in (
                    ("parked by a noise ban", [prefix + "XL-p"], xl, ("a" * 40,)),
                    # its comment failed or was deleted, or its marker is an older schema: the selection
                    # drops the tier and measures the head again -- restoring it flipped it every round
                    ("no verdict marker on the head", [], xl, ()),
                    ("an entry for another commit", [], dict(xl, commit="b" * 40), ("a" * 40,))):
                with self.subTest(tag, case=case):
                    calls, _ = self._reconcile(mod, tag, {5: (labels, "a" * 40)}, {"5": entry}, evaluated)
                    self.assertFalse([c for c in calls if c[2].startswith(prefix)], calls)

    def test_every_posted_verdict_is_recorded_for_its_head(self):
        for mod, tag, *_ in self._bots():
            if True:
                saved = {}
                old = {"9": {"commit": "a" * 40, "label": "XL", "pass": True}}
                with self.subTest(tag), \
                        mock.patch.object(arb, "gh", return_value=run()), \
                        mock.patch.object(arb, "add_label"), mock.patch.object(arb, "remove_label"), \
                        mock.patch.object(arb, "sync_generic_eval_label"), \
                        mock.patch.object(mod, f"strip_{tag}_eval_labels"), \
                        mock.patch.object(mod, "format_comment", return_value="c"), \
                        mock.patch.object(mod, "_load_scores", return_value=dict(old)), \
                        mock.patch.object(mod, "_save_scores", side_effect=saved.update), \
                        mock.patch("builtins.print"):
                    # A re-measure of a verified head, charged as a failed run: the old XL must not stand.
                    mod.apply_result("o/r", 9, "a" * 40, {"ok": False, "retry": False, "reason": "killed", "log": ""})
                self.assertEqual((saved["9"]["commit"], saved["9"]["label"], saved["9"]["pass"]),
                                 ("a" * 40, "REJECT", False))

    def test_a_conflict_found_on_the_box_is_not_kept_in_the_running(self):
        for mod, tag, *_ in self._bots():
            pr = Iteration4Tests._pr(self)
            self.addCleanup(arb.clear_strikes, mod.STRIKES_FILE, 7)
            arb.record_strike(mod.STRIKES_FILE, 7, "a" * 40, "conflict")
            with self.subTest(tag):
                self.assertEqual(mod._unmeasurable_reason("o/r", pr, set()),
                                 "does not merge onto main on the box (needs a rebase)")
            arb.clear_strikes(mod.STRIKES_FILE, 7)

    def test_a_parked_reject_still_blocks_a_merge(self):
        for mod, tag, rebase, prefix, first in self._bots():
            info = {"state": "OPEN", "isDraft": False, "author": {"login": "dev"}, "mergeable": "MERGEABLE",
                    "files": [{"path": "kernels/x.cu"}], "changedFiles": 1, "headRefOid": "a" * 40,
                    "baseRefName": "main",
                    "labels": [{"name": prefix + "XL"}, {"name": first}, {"name": "eval-dspark:REJECT-p"}]}
            with self.subTest(tag), mock.patch.object(arb, "gh", return_value=run(json.dumps(info))), \
                    mock.patch.object(arb, "current_main_sha", return_value=MAIN), \
                    mock.patch.object(arb, "load_denylist", return_value=set()), \
                    mock.patch.object(arb, "author_penalty_until", return_value=None), \
                    mock.patch.object(mod, "_load_scores", return_value={"1": {
                        "commit": "a" * 40, "label": "XL", "pass": True, "onto": MAIN}}):
                ok, why = getattr(mod, f"auto_merge_ok_{tag}")("o/r", 1)
            self.assertFalse(ok)
            self.assertIn("REJECT", why)

    def test_accuracy_sh_fails_a_pass_that_did_not_cover_its_stream(self):
        import re as _re
        import subprocess
        with open(_os.path.join(self.ROOT, "bench/scripts/accuracy.sh")) as f:
            py = next(b for b in _re.findall(r"<<'PY'\n(.*?)\nPY\n", f.read(), _re.S) if "grab_short" in b)
        d = _tempfile.mkdtemp(dir=_STATE)

        def gate(short, longs=""):
            for name, text in (("s", short + "\n"), ("l", longs)):
                with open(_os.path.join(d, name), "w") as f:
                    f.write(text)
            r = subprocess.run(["python3", "-", _os.path.join(d, "s"), _os.path.join(d, "l")], input=py,
                               capture_output=True, text=True, timeout=60)
            return r.returncode, next(l for l in r.stdout.splitlines() if l.startswith("METRIC "))
        ok = "METRIC_SHORT top1=0.950000 kl=0.010000 ppl_spark=1 ppl_llama=1"
        self.assertEqual(gate(ok + " n=20 n_expected=20")[0], 0)
        self.assertEqual(gate(ok)[0], 0)                                     # a compare without counts
        self.assertIn("did not cover", gate(ok.replace("0.95", "1.00") + " n=16 n_expected=20")[1])
        self.assertIn("did not cover", gate(ok + " n=20 n_expected=20",
                                            "METRIC_LONG0 top1=0.95 kl=0.1 n=10 n_expected=16\n")[1])

    def test_a_secondary_rate_limit_is_retried(self):
        self.assertTrue(arb._GH_TRANSIENT_RE.search(
            "gh: You have exceeded a secondary rate limit. Please wait a few minutes (HTTP 403)"))
        self.assertFalse(arb._GH_TRANSIENT_RE.search("gh: Resource not accessible by integration (HTTP 403)"))

    def test_a_nan_row_is_unreadable_on_both_sides(self):
        import subprocess
        d = _tempfile.mkdtemp(dir=_STATE)
        nan = lambda i: f"S i={i} tgt=5 am=5 lp=-nan top=5:-nan,6:-nan\n"
        lp_nan = lambda i: f"S i={i} tgt=5 am=5 lp=nan top=5:-0.1,6:-2.5\n"      # a NaN in lp alone
        good = lambda i: f"S i={i} tgt=5 am=5 lp=-0.1 top=5:-0.1,6:-2.5\n"
        with open(_os.path.join(d, "main.txt"), "w") as f:
            f.write("".join(nan(i) if i == 5 else good(i) for i in range(10)) + "S i=10 tgt=5 am")   # + truncated
        with open(_os.path.join(d, "pr_ok.txt"), "w") as f:
            f.write("".join(good(i) for i in range(10)))
        with open(_os.path.join(d, "pr_nan.txt"), "w") as f:
            f.write("".join(nan(i) if i == 3 else lp_nan(i) if i == 7 else good(i) for i in range(10)))
        pair = _os.path.join(self.ROOT, "bench/scripts/accuracy_compare_pair.py")
        metric = lambda a, b, *x: next(l for l in subprocess.run(
            ["python3", pair, _os.path.join(d, a), _os.path.join(d, b), *x], capture_output=True, text=True,
            timeout=60).stdout.splitlines() if l.startswith(("METRIC ", "SELFCHECK ")))
        selfcheck = metric("main.txt", "main.txt", "--metric-label", "SELFCHECK")
        self.assertIn("top1=1.000000 kl=0.000000", selfcheck)
        self.assertNotIn("nan", selfcheck)                               # main's NaN row left out, not NaN
        self.assertIn("n=9 n_main=9", metric("pr_ok.txt", "main.txt"))   # a clean PR: judged on the rest
        self.assertIn("n=7 n_main=9", metric("pr_nan.txt", "main.txt"))  # a PR's NaNs: coverage fails

        def ok(h):
            h.rfile.read(int(h.headers["Content-Length"]))
            body = json.dumps({"completion_probabilities": [{"top_logprobs": [
                {"id": 5, "logprob": -0.1}, {"id": 6, "logprob": -2.5}]}]}).encode()
            h.send_response(200)
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
        def dump(path, positions):
            with open(path, "w") as f:
                f.write("".join(nan(i) if i == 3 else good(i) for i in positions))
        self._dump = dump                                                # the PR's dump, with a NaN row
        r = Iteration5bTests._compare(self, Iteration5bTests._serve(self, ok), range(8))
        self.assertIn(" n=7 n_expected=8", r.stdout)                     # Muse: a gap, not "kl=nan"



class Iteration7Tests(unittest.TestCase):
    """Fixes from the post-merge review of main e73fa96, on all three bots."""

    def _bots(self):
        return Iteration5Tests._bots(self)

    def test_a_head_reset_to_a_commit_measured_earlier_is_measured_again(self):
        # H0 (a) was measured, then H1 (b) -- REJECT, recorded; the author undoes H1 with a force-push.
        self.assertEqual(arb.recorded_verdict_heads({"a" * 40, "b" * 40}, {"commit": "b" * 40}), {"b" * 40})
        self.assertIsNone(arb.recorded_verdict_heads(None, {"commit": "b" * 40}))
        # No scores entry at all (a new controller, a missing file): the markers alone decide.
        self.assertEqual(arb.recorded_verdict_heads({"a" * 40}, None), {"a" * 40})
        for mod, tag, rebase, prefix, first in self._bots():
            pr = Iteration3cTests._pr(self, labels=[{"name": prefix + "REJECT"}, {"name": "eval:REJECT"}])
            removed = []
            with self.subTest(tag):
                code, out = Iteration3Tests._run_main(
                    self, mod, tag, run(json.dumps([pr])),
                    extra=((mod, f"{tag}_evaluated_commits", mock.Mock(return_value={"a" * 40, "b" * 40})),
                           (mod, "_load_scores", mock.Mock(return_value={"5": {"commit": "b" * 40, "label": "REJECT"}})),
                           (arb, "remove_label", mock.Mock(side_effect=lambda r, n, l: removed.append(l))),
                           (arb, "labels_on_or_none", mock.Mock(return_value=set()))))
                self.assertNotIn("already", out)                            # not skipped as measured
                self.assertIn(prefix + "REJECT", removed)                   # H1's tier is not H0's

    def test_the_heal_leaves_a_noise_ban_alone(self):
        self.assertFalse(arb.generic_label_out_of_sync({"eval-museglimmer:none", "eval-qwen38:XL-p", "eval:XL-p"}))
        self.assertTrue(arb.generic_label_out_of_sync({"eval-museglimmer:none", "eval-qwen38:XL", "eval:none"}))

    def test_no_verdict_close_when_the_verdict_comment_did_not_post(self):
        for mod, tag, *_ in self._bots():
            calls = []

            def fake_gh(a):
                calls.append(a)
                if a[:2] == ["pr", "comment"] and "auto-close" not in " ".join(a):
                    return run("", 1)                                         # the verdict's comment fails
                return run(json.dumps({"headRefOid": "a" * 40, "state": "OPEN", "isDraft": False, "labels": []}))
            res = {"ok": True, "label": "REJECT", "delta_pct": -9.0, "pass": False, "accuracy_ok": True,
                   "reason": "decode@128 regressed"}
            kw = {"body": BONSAI_ONLY} if tag == "bonsai" else {"pr_body": TEMPLATE}
            with self.subTest(tag), mock.patch.object(arb, "gh", side_effect=fake_gh), \
                    mock.patch.object(arb, "add_label"), mock.patch.object(arb, "remove_label"), \
                    mock.patch.object(arb, "sync_generic_eval_label"), \
                    mock.patch.object(arb, "labels_on_or_none", return_value=set()), \
                    mock.patch.object(mod, f"strip_{tag}_eval_labels"), \
                    mock.patch.object(mod, f"upload_{tag}_eval_log"), \
                    mock.patch.object(mod, "format_comment", return_value="the verdict comment"), \
                    mock.patch.object(mod, "_load_scores", return_value={}), \
                    mock.patch.object(mod, "_save_scores"), mock.patch("builtins.print"):
                if tag == "bonsai":
                    with mock.patch.object(mod, "AUTO_CLOSE", True):
                        mod.apply_result("o/r", 1, "a" * 40, res, **kw)
                else:
                    mod.apply_result("o/r", 1, "a" * 40, res, **kw)
            self.assertFalse(any(c[:2] == ["pr", "close"] for c in calls), tag)


if __name__ == "__main__":
    unittest.main()
