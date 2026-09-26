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

TEMPLATE = open(arb.os.path.join(arb.os.path.dirname(arb.os.path.dirname(arb.__file__)),
                                 ".github", "PULL_REQUEST_TEMPLATE.md")).read()


def run(stdout="", rc=0, stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


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

    def _ok(self, mod, prefix, first, info_over=None, scored=None, require_first=True):
        info = {"state": "OPEN", "isDraft": False, "labels": [{"name": prefix + "XL"}, {"name": first}],
                "author": {"login": "dev"}, "mergeable": "MERGEABLE", "files": [{"path": "kernels/x.cu"}],
                "headRefOid": "a" * 40}
        info.update(info_over or {})
        scored = {"1": {"commit": "a" * 40, "label": "XL", "pass": True}} if scored is None else scored
        with mock.patch.object(arb, "gh", return_value=run(json.dumps(info))), \
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
                                   scored={"1": {"commit": "a" * 40, "label": "none", "pass": True}})
                self.assertFalse(ok)
                self.assertIn("recorded verdict", why)

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

    def test_the_merge_is_pinned_to_the_checked_head(self):
        for mod, _p, _f, _n, ok_fn, try_fn, *_ in BOTS:
            calls = []

            def fake_gh(a):
                calls.append(a)
                return run('{"headRefOid": "' + "a" * 40 + '"}') if a[:2] == ["pr", "view"] else run("")
            name = ok_fn.__name__
            with self.subTest(mod.__name__), mock.patch.object(mod, name, return_value=(True, "ok")), \
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

    def test_an_exception_is_infra_and_the_baseline_is_guarded(self):
        for mod, *_ in BOTS:
            with open(mod.__file__) as f:
                src = f.read()
            with self.subTest(mod.__name__):
                self.assertIn('"ok": False, "retry": True, "reason": f"exception:', src)
                self.assertIn("main_result = measure_main_baseline(host, port)\n    except Exception", src)

    def test_a_guard_that_measured_nothing_retries_on_muse(self):
        # Only the Ternary-Bonsai guard used to take the infra path; the ModelOpt, unsloth and
        # Qwen3.6 guards REJECTed -- and so closed -- a PR over a measurement that never happened.
        def stdout(drop=()):
            lines = [f"MUSE {c} 100.0 5000.0" for c in muse.SCORED_CTXS]
            lines += [f"MUSECB {c} 500.0" for c in muse.CB_CONCS]
            lines += ["RESULT_DECODE_TPS 100.0", "RESULT_PREFILL128_PP 5000.0", "RESULT_TOP1 0.99",
                      "RESULT_KL 0.01", "ACCURACY_STAGE_DONE", "GUARD_START", "GUARD36 32768 50.0 900.0",
                      "GUARDMO 32768 60.0 7000.0", "GUARDUN 32768 55.0 6800.0",
                      "GUARDBN 128 99.0 2000.0", "GUARDBN 32768 89.0 6500.0", "GUARD_END"]
            return "\n".join(l for l in lines if not any(l.startswith(d) for d in drop)) + "\n"
        for tag in ("GUARDMO ", "GUARDUN ", "GUARD36 "):
            # The guard measured nothing on main (so nothing on either side): infra, retried.
            with mock.patch.object(muse, "_ssh_run_resilient", return_value=run(stdout(drop=(tag,)))):
                main = muse.measure_main_baseline("h", 1)
            self.assertTrue(main["ok"], main)
            with self.subTest(tag), mock.patch.object(muse, "POLARIS_ENABLED", False), \
                    mock.patch.object(muse, "_ssh_run_resilient", return_value=run(stdout(drop=(tag,)))):
                res = muse.eval_museglimmer_on_box("h", 1, "pull/1/head", main)
                self.assertFalse(res["ok"])
                self.assertTrue(res["retry"], res.get("reason"))
        # Unchanged: main measured it and only the PR's run lost it -> fail closed (a regression).
        with mock.patch.object(muse, "_ssh_run_resilient", return_value=run(stdout())):
            main = muse.measure_main_baseline("h", 1)
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

    def test_muse_reports_the_commit_it_built_and_retries_a_failed_fetch(self):
        s = muse._remote_script("pull/1/head")
        self.assertIn('echo "REMOTE_SHA $(git rev-parse HEAD)"', s)
        self.assertIn('RETRYABLE_INFRA_FAILURE git fetch pull/1/head failed', s)
        self.assertEqual(muse._parse_remote(f"REMOTE_SHA {'d' * 40}\n")["pr_tip"], "d" * 40)

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


if __name__ == "__main__":
    unittest.main()
