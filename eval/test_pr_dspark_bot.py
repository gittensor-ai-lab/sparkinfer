import subprocess
import unittest

from eval import pr_dspark_bot as bot


class Prefill256KEvalTests(unittest.TestCase):
    def test_256k_is_exact_scored_dimension(self):
        self.assertEqual(bot.DSPARK_PREFILL_CTX_256K, 262144)
        self.assertIn("target-prefill@256k", bot.SCORING_DIMS)
        self.assertIn("native-nvfp4-256k-prefill", bot.EVAL_SCHEMA_VERSION)

    def test_concurrency_is_scored_and_c1_is_only_a_floor(self):
        # Every other dimension measures ONE stream. Without these, a PR that fixed aggregate
        # throughput under concurrency scored exactly zero -- which is what #973 and #975 hit.
        for dim in ("cb-decode@c2", "cb-decode@c4", "cb-decode@c8"):
            self.assertIn(dim, bot.SCORING_DIMS)
        # c=1 must NOT be scored: it is the floor that stops a PR buying concurrency scaling by
        # slowing the single-stream path. Scoring it would let that trade earn a tier.
        self.assertNotIn("cb-decode@c1", bot.SCORING_DIMS)

    def test_remote_script_measures_the_concurrency_ladder(self):
        script = bot._remote_script("main", role="main")
        self.assertIn("qwen3_gguf_cb_bench", script)
        self.assertIn("for CC in 1 2 4 8; do", script)
        # 256 tokens per request, not 64. A ~1s run measures its own startup: on identical code
        # c=4 spread 3.40% and could land at -3.37%, hard-REJECTING a PR that changed nothing,
        # because REGRESS_TOL rejects below -2.00%. At 256 the worst case is -0.41%. Pinned here
        # so a later "save GPU time" edit cannot quietly reintroduce a spurious-reject generator.
        self.assertIn('cb_bench "$MODEL_DIR" "$CC" 256 256 512', script)
        # c=1 is measured even though it is not scored -- it is the floor.
        self.assertIn("RESULT_CB${CC}_AGG", script)
        self.assertIn("RESULT_CB${CC}_ITL", script)
        # A harness that runs but measures nothing must be infra, not a regression to zero:
        # scoring 0 would REJECT the PR for the harness's own failure.
        self.assertIn("concurrent decode produced no positive metric", script)
        # The measuring instrument comes from main, like every other harness file.
        self.assertIn("runtime/examples/qwen3_gguf_cb_bench.cpp", bot.HARNESS_PATHS)

    def test_concurrency_results_parse_into_the_keys_the_dims_table_reads(self):
        out = bot._parse_remote(
            "RESULT_CB1_AGG 82.0\n"
            "RESULT_CB2_AGG 69.2\n"
            "RESULT_CB4_AGG 71.1\n"
            "RESULT_CB8_AGG 71.4\n"
            "RESULT_CB2_ITL 30.18\n")
        self.assertEqual(out["cb1_agg"], 82.0)
        self.assertEqual(out["cb2_agg"], 69.2)
        self.assertEqual(out["cb4_agg"], 71.1)
        self.assertEqual(out["cb8_agg"], 71.4)
        self.assertEqual(out["cb2_itl"], 30.18)

    def test_remote_script_uses_one_pass_memory_safe_sweep(self):
        script = bot._remote_script("main", role="main")
        self.assertIn('PREFILL_CTX256=262144', script)
        self.assertIn('SPARKINFER_BENCH_SWEEP_CTXS="$PREFILL_CTX256"', script)
        self.assertIn('qwen3_gguf_bench "$MODEL_DIR" 128 sweep', script)
        self.assertIn('SPARKINFER_QWEN38_PREFILL_NVFP4=1', script)
        self.assertIn('SPARKINFER_QWEN38_DECODE_NVFP4=1', script)
        self.assertIn('SPARKINFER_KV_INT8=1', script)
        self.assertIn('RESULT_PREFILL256_PP', script)
        self.assertIn('runtime/examples/qwen3_gguf_bench.cpp', script)
        self.assertIn('runtime/examples/qwen_checkpoint.h', script)
        self.assertIn('runtime/examples/qwen3_gguf_config.h', script)
        self.assertIn('runtime/examples/qwen3_gguf_bench.cpp', bot.HARNESS_PATHS)
        # Do not pin the 128k batched ceiling: a PR that adds memory-safe chunked batched prefill
        # must be allowed to take that path and beat today's sequential baseline.
        self.assertNotIn('SPARKINFER_PREFILL_BATCHED_MAXCTX=', script)
        checked = subprocess.run(["bash", "-n"], input=script, text=True,
                                 capture_output=True)
        self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_parser_and_comment_keep_256k_metric(self):
        parsed = bot._parse_remote("RESULT_PREFILL256_PP 66.9306\n")
        self.assertEqual(parsed["prefill256_pp"], 66.9306)
        body = bot.format_comment("abcdef123", {
            "ok": True,
            "label": "XS",
            "pass": True,
            "pr_prefill256_pp": 70.0,
            "main_prefill256_pp": 66.93,
            "prefill256_delta_pct": 4.6,
        })
        self.assertIn("PR prefill @256k", body)
        self.assertIn("70.0 pp/s", body)
        self.assertIn("prefill @256k vs main | +4.6%", body)


if __name__ == "__main__":
    unittest.main()
