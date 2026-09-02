import subprocess
import unittest

from eval import pr_dspark_bot as bot


class Prefill256KEvalTests(unittest.TestCase):
    def test_256k_is_exact_scored_dimension(self):
        self.assertEqual(bot.DSPARK_PREFILL_CTX_256K, 262144)
        self.assertIn("target-prefill@256k", bot.SCORING_DIMS)
        self.assertIn("native-nvfp4-256k-prefill", bot.EVAL_SCHEMA_VERSION)

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
