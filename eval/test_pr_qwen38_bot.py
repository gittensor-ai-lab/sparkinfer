import unittest

import pr_qwen38_bot as bot


class ConcurrencyAxesTests(unittest.TestCase):
    def test_concurrency_is_scored_and_c1_is_only_a_floor(self):
        # Issue #1080: the unsloth checkpoint's packed step takes FP8 and Q4_K paths no other
        # scored dimension enters, so c2..c32 must be able to earn a tier.
        for c in (2, 4, 8, 16, 32):
            self.assertIn(f"cb-decode@c{c}", bot.SCORING_DIMS)
        # c=1 is measured but never scored: it stops a PR buying scaling by slowing one stream.
        self.assertIn(1, bot.CB_CONCS)
        self.assertNotIn("cb-decode@c1", bot.SCORING_DIMS)
        self.assertIn("prefill@16k", bot.SCORING_DIMS)
        self.assertNotIn("decode@128", bot.SCORING_DIMS)

    def test_long_context_decode_is_scored_and_its_prefill_is_a_floor(self):
        # Issue #1113: decode@256k was defined in pr_dspark_bot.py but measured by no active bot,
        # so a change that only moves long-context decode scored "none" and was auto-closed.
        self.assertIn(bot.LONGCTX_DECODE_DIM, bot.SCORING_DIMS)
        self.assertNotIn(bot.LONGCTX_PREFILL_DIM, bot.SCORING_DIMS)
        self.assertEqual(bot.LONGCTX_CTX, 262144)
        script = bot._remote_script("main", role="main")
        # Its own sweep call: bench_sweep_run applies one rep count per call, so sharing the 5-rep
        # guard call would cost five 60 s rows.
        self.assertIn(f'bench_sweep_run "$MODELOPT_GUARD_MODEL_DIR" 128 {bot.LONGCTX_CTX} {bot.LONGCTX_REPS}', script)
        self.assertIn(f'echo "GUARDMO {bot.LONGCTX_CTX} ', script)
        # A missing or failed 256k sweep leaves the axis unscored; it never rejects a PR.
        self.assertIn("LONGCTX_FAILED", script)
        parsed = bot._parse_remote("LONGCTX_FAILED\n")
        self.assertTrue(parsed.get("longctx_unmeasured"))
        self.assertFalse(parsed.get("guardmo_failed"))

    def test_an_unmeasured_guard_retries_instead_of_rejecting(self):
        # 2026-09-18: one round produced no Muse concurrent rows, and "measurement unavailable"
        # was treated as a guard failure -- REJECT, then auto-closed -- on two PRs that had just
        # measured +10% on the 256k axis. An absent measurement is infra: retry, never reject.
        src = open(bot.__file__).read()
        self.assertIn('unavailable = [p for p in problems if p.endswith("measurement unavailable")]', src)
        self.assertIn("re-evaluated next round rather than rejected", src)
        # The coverage line makes a missing guard visible in the round log.
        self.assertIn("main guard coverage", src)
        self.assertIn("PR guard coverage", src)
        self.assertEqual(bot._guard_coverage({"guardmo": {32768: {}}, "guardcbmo": {16: {}, 32: {}}}),
                         "modelopt 1 ctx / 2 cc · muse 0 ctx / 0 cc · qwen3.6 0 ctx · bonsai 0 ctx")

    def test_schema_is_bumped_so_old_verdicts_re_evaluate(self):
        self.assertNotEqual(bot.EVAL_SCHEMA_VERSION, "v1-nvfp4-decode128")
        self.assertIn(bot.EVAL_SCHEMA_VERSION, bot.MARKER_RE.pattern.replace("\\", ""))

    def test_remote_script_measures_the_ladder_like_the_dspark_bot(self):
        script = bot._remote_script("main", role="main")
        self.assertIn("for CC in 1 2 4 8 16 32; do", script)
        # Same shape and env as pr_dspark_bot.py's ModelOpt rows, so the checkpoints compare.
        self.assertIn('cb_median "$MODEL_DIR" "$CC" SPARKINFER_QWEN38_PREFILL_NVFP4=1 '
                      'SPARKINFER_QWEN38_DECODE_NVFP4=1 SPARKINFER_KV_INT8=1', script)
        self.assertIn('build/runtime/qwen3_gguf_cb_bench "$ckpt" "$cc" 256 256 512', script)
        for env in ("SPARKINFER_QWEN38_PREFILL_NVFP4=1", "SPARKINFER_QWEN38_DECODE_NVFP4=1",
                    "SPARKINFER_KV_INT8=1"):
            self.assertIn(env, script)
        self.assertIn("qwen3_gguf_cb_bench -j", script)
        for kind in ("AGG", "ITL", "ERR"):
            self.assertIn(f"RESULT_CB${{CC}}_{kind}", script)
        # A harness that measures nothing is infra, never a regression to zero.
        self.assertIn("concurrent decode produced no positive metric", script)

    def test_remote_script_takes_the_harness_from_main(self):
        script = bot._remote_script("pull/1/head", role="pr")
        self.assertIn("git checkout -q origin/main -- runtime/examples/qwen3_gguf_bench.cpp", script)
        self.assertIn("runtime/examples/qwen3_gguf_cb_bench.cpp", script)
        self.assertIn("HARNESS_PINNED", script)
        for path in ("runtime/examples/qwen3_gguf_cb_bench.cpp", "eval/", "bench/scripts/"):
            self.assertIn(path, bot.HARNESS_PATHS)

    def test_parity_gate_is_not_run(self):
        # main fails prefill_parity_check on this checkpoint; an absolute gate main fails would
        # REJECT and auto-close every PR.
        self.assertNotIn('prefill_parity_check.py "', bot._remote_script("main", role="main"))

    def test_ladder_results_parse(self):
        out = bot._parse_remote("RESULT_CB2_AGG 155.9\nRESULT_CB2_ITL 12.59\n"
                                "RESULT_CB32_AGG 246.3\nRESULT_CB32_ERR 2\n")
        self.assertEqual(out["cb2_agg"], 155.9)
        self.assertEqual(out["cb2_itl"], 12.59)
        self.assertEqual(out["cb32_agg"], 246.3)
        self.assertEqual(out["cb32_err"], 2.0)

    def test_a_retryable_ladder_failure_is_retried_and_named(self):
        err = "RETRYABLE_INFRA_FAILURE concurrent-decode harness exited nonzero at c=32\n"
        self.assertTrue(bot._looks_like_hard_kill("", err))
        self.assertIn("c=32", bot._crash_reason("", err))


    def _cb_complete(self, c, tok, err):
        # Run the ladder's own bash check, extracted from the rendered script.
        import re, subprocess
        script = bot._remote_script("main", role="main")
        fn = re.search(r"^cb_complete\(\) \{\n.*?^\}\n", script, re.S | re.M).group(0)
        return subprocess.run(["bash", "-c", fn + f"cb_complete {c} {tok} {err}"]).returncode == 0

    def test_complete_runs_are_accepted(self):
        # Runs measured on main 507017b whose requests all finished or failed outright.
        self.assertTrue(self._cb_complete(1, 264, 0))
        self.assertTrue(self._cb_complete(16, 4104, 0))
        self.assertTrue(self._cb_complete(32, 7936, 2))   # one short request + the long one failed
        self.assertTrue(self._cb_complete(32, 7688, 2))   # two short requests failed
        self.assertTrue(self._cb_complete(32, 8200, 0))   # a PR that fixes the out-of-memory

    def test_a_run_whose_requests_stopped_part_way_is_rejected(self):
        # The c32 run that read 293.0 tok/s instead of ~249: 4,493 tokens with only 2 errors.
        self.assertFalse(self._cb_complete(32, 4493, 2))
        self.assertFalse(self._cb_complete(8, 2000, 0))

    def test_each_width_is_the_median_of_complete_runs(self):
        script = bot._remote_script("main", role="main")
        # c32 on identical code differs by up to 2.7% run to run -- outside the -2% reject band --
        # so one run per width could REJECT and auto-close an unchanged PR.
        self.assertEqual(bot.CB_REPS, 3)
        self.assertIn('while [ "$valid" -lt 3 ]; do', script)
        self.assertIn("statistics.median", script)
        # A partial run is re-run, never scored; too many of them fail the round as infra.
        self.assertIn("CB_PARTIAL c=$cc", script)
        self.assertIn('if [ "$attempt" -gt 5 ]; then', script)
        for kind in ("RUNS", "TOK"):
            self.assertIn(f"RESULT_CB${{CC}}_{kind}", script)

    def test_the_median_step_runs(self):
        import re, subprocess
        script = bot._remote_script("main", role="main")
        cmd = re.search(r'CB_AGG=\$\((python3 -c "import statistics.*?") \$CB_AGGS\)', script).group(1)
        out = subprocess.run(["bash", "-c", cmd + " 246.3 293.0 248.6"], capture_output=True, text=True)
        self.assertEqual(float(out.stdout), 248.6)

    def test_runs_parse(self):
        out = bot._parse_remote("RESULT_CB32_RUNS 246.3 250.2 247.9\nRESULT_CB32_AGG 247.9\n")
        self.assertEqual(out["cb32_runs"], [246.3, 250.2, 247.9])
        self.assertEqual(out["cb32_agg"], 247.9)


class CrossModelGuardTests(unittest.TestCase):
    """ModelOpt Qwen3.8 and Muse Glimmer no-regression guards, the pair pr_museglimmer_bot.py runs.
    The Muse bot skips Qwen3.8-only PRs, so this bot is the only check they get on either model."""

    MAIN = {"guard36": {32768: {"decode": 460.0, "prefill": 24800.0}},
            "guardmo": {32768: {"decode": 90.0, "prefill": 13000.0}},
            "guardmg": {32768: {"decode": 98.0, "prefill": 11000.0}}}

    def test_remote_script_runs_both_guards_before_the_end_marker(self):
        script = bot._remote_script("main", role="main")
        self.assertIn('bench_sweep_run "$MODELOPT_GUARD_MODEL_DIR" 128 32768 5', script)
        self.assertIn('bench_sweep_run "$MUSE_GUARD_GGUF" 128 32768 5', script)
        for marker in ("GUARDMO $ctx", "GUARDMO_FAILED", "GUARDMO_UNAVAILABLE",
                       "GUARDMG $ctx", "GUARDMG_FAILED", "GUARDMG_UNAVAILABLE"):
            self.assertIn(marker, script)
        # GUARD_END is the hard-kill retry's end-of-run marker, so it must come after both guards.
        self.assertLess(script.index("GUARDMG_UNAVAILABLE"), script.index('echo "GUARD_END"'))
        self.assertIn("cross-model-guards", bot.EVAL_SCHEMA_VERSION)

    def test_guard_lines_parse(self):
        out = bot._parse_remote("GUARDMO 32768 90.5 13010.0\nGUARDMG 32768 97.9 10990.0\n")
        self.assertEqual(out["guardmo"], {32768: {"decode": 90.5, "prefill": 13010.0}})
        self.assertEqual(out["guardmg"], {32768: {"decode": 97.9, "prefill": 10990.0}})
        failed = bot._parse_remote("GUARDMO_FAILED\nGUARDMG_UNAVAILABLE\n")
        self.assertTrue(failed["guardmo_failed"])
        self.assertTrue(failed["guardmg_unavailable"])

    def test_a_regression_or_missing_measurement_fails_the_guard(self):
        pr = {"guardmg": {32768: {"decode": 90.0, "prefill": 11000.0}}}
        ok, problems = bot.check_muse_guard(pr, self.MAIN)
        self.assertFalse(ok)
        self.assertIn("muse glimmer decode@32k", problems[0])
        ok, problems = bot.check_modelopt_guard({"guardmo": {}}, self.MAIN)
        self.assertFalse(ok)
        self.assertIn("modelopt guard measurement unavailable", problems)
        ok, _ = bot.check_modelopt_guard({"guardmo_failed": True, "guardmo": self.MAIN["guardmo"]}, self.MAIN)
        self.assertFalse(ok)

    def test_flat_numbers_pass_and_qwen36_behaviour_is_unchanged(self):
        pr = {k: {32768: {m: v * 0.99 for m, v in d[32768].items()}} for k, d in self.MAIN.items()}
        self.assertEqual(bot.check_modelopt_guard(pr, self.MAIN), (True, []))
        self.assertEqual(bot.check_muse_guard(pr, self.MAIN), (True, []))
        self.assertEqual(bot.check_q36_guard(pr, self.MAIN), (True, []))
        ok, problems = bot.check_q36_guard({"guard36": {}}, self.MAIN)
        self.assertIn("qwen3.6 guard measurement unavailable", problems)

    def test_comment_says_passed_failed_or_skipped(self):
        base = {"ok": True, "label": "none", "pr_decode_tps": 1.0, "main_decode_tps": 1.0,
                "pr_prefill_pp": 1.0, "main_prefill_pp": 1.0, "pr_prefill16k_pp": 1.0,
                "main_prefill16k_pp": 1.0, "accuracy_ok": True, "q36_guard_ok": True}
        body = bot.format_comment("c", {**base, "modelopt_guard_ok": True,
                                        "muse_guard_ok": False, "muse_guard_problems": ["muse glimmer decode@32k: x"],
                                        "muse_guard_skipped": False})
        self.assertIn("| modelopt guard | ✅ no regression", body)
        self.assertIn("| muse glimmer guard | ❌ **FAILED** — muse glimmer decode@32k: x", body)
        skipped = bot.format_comment("c", {**base, "modelopt_guard_ok": True, "modelopt_guard_skipped": True,
                                           "muse_guard_ok": True})
        self.assertIn("| modelopt guard | ⚠️ SKIPPED", skipped)
        self.assertNotIn("| modelopt guard | ✅", skipped)


class ConcurrencyGuardTests(unittest.TestCase):
    """The 32k guards run one request; this bot's PRs mostly change packed decode."""

    def test_both_models_get_concurrency_guards_before_the_end_marker(self):
        script = bot._remote_script("main", role="main")
        self.assertEqual(bot.CB_GUARD_CONCS, [16, 32])
        self.assertIn('cb_median "$MODELOPT_GUARD_MODEL_DIR" "$CC" SPARKINFER_QWEN38_PREFILL_NVFP4=1', script)
        self.assertIn('cb_median "$MUSE_GUARD_GGUF" "$CC"; then', script)
        for marker in ("GUARDCBMO $CC $CB_AGG", "GUARDCBMO_FAILED $CC", "GUARDCBMG $CC $CB_AGG", "GUARDCBMG_FAILED $CC"):
            self.assertIn(marker, script)
        self.assertLess(script.index("GUARDCBMG_FAILED"), script.index('echo "GUARD_END"'))
        # The schema names the axis set; it moved to v5 when decode@256k joined (#1113).
        self.assertIn("cross-model-guards-cb", bot.EVAL_SCHEMA_VERSION)

    def test_cb_median_returns_instead_of_exiting(self):
        script = bot._remote_script("main", role="main")
        fn = script[script.index("cb_median() {"):script.index("\n}\n", script.index("cb_median() {"))]
        self.assertNotIn("exit ", fn)
        self.assertIn("return 1", fn)

    def test_guard_lines_parse_and_a_regression_fails(self):
        main = bot._parse_remote("GUARDCBMO 16 1080.0\nGUARDCBMO 32 1600.0\nGUARDCBMG 16 400.0\n")
        self.assertEqual(main["guardcbmo"], {16: {"cb-decode": 1080.0}, 32: {"cb-decode": 1600.0}})
        pr = bot._parse_remote("GUARDCBMO 16 1079.0\nGUARDCBMO 32 1500.0\nGUARDCBMG_FAILED 16\n")
        ok, problems = bot.check_modelopt_cb_guard(pr, main)
        self.assertFalse(ok)
        self.assertEqual(len(problems), 1)
        self.assertIn("modelopt concurrent cb-decode@c32", problems[0])
        ok, problems = bot.check_muse_cb_guard(pr, main)
        self.assertFalse(ok)
        self.assertIn("muse glimmer concurrent guard measurement unavailable", problems)

if __name__ == "__main__":
    unittest.main()
