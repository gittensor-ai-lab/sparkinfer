#!/usr/bin/env python3
"""Unit tests for the deterministic eval label policy.

Run from the repo root:
  python3 bench/scripts/test_label.py
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABEL = ROOT / "bench" / "scripts" / "label.py"


def score(tps, frontier=100.0, ceiling=200.0, top1=0.97, kl=0.02, commit="deadbee",
          prov=None, env=None):
    cmd = [sys.executable, str(LABEL), str(tps), str(frontier), str(ceiling),
           str(top1), str(kl), commit]
    if prov is not None:
        cmd.append(json.dumps(prov, separators=(",", ":")))
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    out = subprocess.check_output(cmd, text=True, env=run_env).strip()
    assert out.startswith("RESULT_JSON "), out
    return json.loads(out[len("RESULT_JSON "):])


class LabelPolicyTest(unittest.TestCase):
    def test_correctness_gate_rejects_bad_accuracy(self):
        low_top1 = score(130, top1=0.89, kl=0.02)
        self.assertEqual(low_top1["label"], "REJECT")
        self.assertFalse(low_top1["pass"])

        high_kl = score(130, top1=0.97, kl=0.21)
        self.assertEqual(high_kl["label"], "REJECT")
        self.assertFalse(high_kl["pass"])

    def test_significance_gate_is_strictly_above_two_percent(self):
        exact_floor = score(102.0)
        self.assertEqual(exact_floor["label"], "none")
        self.assertTrue(exact_floor["pass"])
        self.assertEqual(exact_floor["pct_over_frontier"], 2.0)

        just_above = score(102.1)
        self.assertEqual(just_above["label"], "XS")
        self.assertTrue(just_above["pass"])

    def test_llama_anchored_buckets_without_difficulty_boost(self):
        # The label tier is sized against the llama.cpp reference (DIFF_REF), not the frontier — so
        # equal tok/s of work earns the same tier at any maturity. With DIFF_REF=365.85 and frontier=100
        # (below the reference -> no boost), the S/M/L/XL thresholds (3.5/6/10/18% of llama) fall at
        # delta ~= 12.8/22/36.6/65.9 tok/s. A verified-but-smaller gain floors at XS.
        env = {"SPARKINFER_DIFFICULTY_BOOST": "0", "SPARKINFER_DIFFICULTY_REF": "365.85"}
        self.assertEqual(score(105.0, env=env)["label"], "XS")   # +5: verified over frontier, < S vs llama
        self.assertEqual(score(113.0, env=env)["label"], "S")    # +13 = 3.55% of llama
        self.assertEqual(score(122.0, env=env)["label"], "M")    # +22 = 6.01%
        self.assertEqual(score(137.0, env=env)["label"], "L")    # +37 = 10.11%
        self.assertEqual(score(166.0, env=env)["label"], "XL")   # +66 = 18.04%

    def test_unoptimized_model_does_not_mint_xl_from_low_hanging_fruit(self):
        # The unfairness this policy fixes: over an un-optimized frontier a small absolute gain used to
        # explode to XL. Sized against llama.cpp it earns a fair tier. (frontier 23, llama 190.)
        env = {"SPARKINFER_DIFFICULTY_BOOST": "1", "SPARKINFER_DIFFICULTY_REF": "190"}
        self.assertEqual(score(28.0, frontier=23.0, env=env)["label"], "XS")   # +5 (+22% over frontier) but 2.6% of llama
        self.assertEqual(score(171.0, frontier=23.0, env=env)["label"], "XL")  # a real 7x IS an XL (78% of llama)

    def test_difficulty_boost_high_cap_still_capped_at_twice_raw_speedup(self):
        # Even with a generous DIFF_MAX, g_eff cannot exceed 2× pct_over_frontier.
        res = score(484.79, frontier=469.13, ceiling=366.0, top1=0.9612, kl=0.0175,
                    commit="c30bf58",
                    env={"SPARKINFER_DIFFICULTY_BOOST": "1",
                         "SPARKINFER_DIFFICULTY_REF": "365.85",
                         "SPARKINFER_DIFFICULTY_K": "8",
                         "SPARKINFER_DIFFICULTY_MAX": "4"})
        self.assertEqual(res["label"], "M")
        self.assertEqual(res["pct_over_frontier"], 3.3)
        self.assertGreaterEqual(res["effective_pct"], 6.0)
        self.assertLess(res["effective_pct"], 10.0)

    def test_difficulty_boost_default_cap_is_one_point_five_x(self):
        # Default SPARKINFER_DIFFICULTY_MAX=1.5 with strict 2× raw cap.
        res = score(484.79, frontier=469.13, ceiling=366.0, top1=0.9612, kl=0.0175,
                    commit="c30bf58",
                    env={"SPARKINFER_DIFFICULTY_BOOST": "1",
                         "SPARKINFER_DIFFICULTY_REF": "365.85",
                         "SPARKINFER_DIFFICULTY_K": "8"})
        self.assertEqual(res["label"], "M")
        self.assertEqual(res["difficulty_mult"], 1.5)
        self.assertEqual(res["pct_over_frontier"], 3.3)
        self.assertGreaterEqual(res["effective_pct"], 6.0)
        self.assertLess(res["effective_pct"], 10.0)

    def test_strict_cap_limits_per_context_llama_ref_inflation(self):
        # Qwen3.6 @ 128: a low per-context llama ref must not push a ~3% raw gain to L.
        res = score(479.96, frontier=465.41, ceiling=366.0, top1=0.967, kl=0.0281,
                    commit="d12766f",
                    env={"SPARKINFER_DIFFICULTY_BOOST": "1",
                         "SPARKINFER_DIFFICULTY_REF": "275.81",
                         "SPARKINFER_DIFFICULTY_K": "8"})
        self.assertEqual(res["label"], "M")
        self.assertEqual(res["pct_over_frontier"], 3.1)
        self.assertGreaterEqual(res["effective_pct"], 6.0)
        self.assertLess(res["effective_pct"], 10.0)

    def test_long_context_metadata_is_preserved_in_verdict(self):
        prov = {
            "eval_mode": "longctx",
            "score_context": 16384,
            "best_context_label": "16k-context",
            "context_gains_pct": {"128-context": -2.04, "512-context": 1.25, "4k-context": 0.5, "16k-context": 20.0, "32k-context": 12.5},
            "ctx_128_tps": 480.0,
            "ctx_512_tps": 405.0,
            "ctx_4096_tps": 198.0,
            "ctx_16384_tps": 120.0,
            "ctx_32768_tps": 80.0,
            "guard_128_baseline": 490.0,
            "guard_128_ratio": 0.9796,
            "guard_128_pass": True,
            "guard_512_baseline": 400.0,
            "guard_512_ratio": 1.0125,
            "guard_512_pass": True,
            "guard_4k_baseline": 197.0,
            "guard_4k_ratio": 1.0051,
            "guard_4k_pass": True,
            "guard_16k_baseline": 100.0,
            "guard_16k_ratio": 1.2,
            "guard_16k_pass": True,
            "guard_32k_baseline": 71.11,
            "guard_32k_ratio": 1.125,
            "guard_32k_pass": True,
        }
        res = score(170.0, frontier=100.0, prov=prov)   # +70 = 19.1% of llama -> XL
        self.assertEqual(res["label"], "XL")
        self.assertEqual(res["eval_mode"], "longctx")
        self.assertEqual(res["score_context"], 16384)
        self.assertEqual(res["best_context_label"], "16k-context")
        self.assertEqual(res["context_gains_pct"]["16k-context"], 20.0)
        self.assertEqual(res["context_gains_pct"]["32k-context"], 12.5)
        self.assertEqual(res["ctx_128_tps"], 480.0)
        self.assertEqual(res["ctx_512_tps"], 405.0)
        self.assertEqual(res["ctx_4096_tps"], 198.0)
        self.assertEqual(res["ctx_32768_tps"], 80.0)
        self.assertTrue(res["guard_128_pass"])
        self.assertTrue(res["guard_512_pass"])
        self.assertTrue(res["guard_4k_pass"])
        self.assertTrue(res["guard_16k_pass"])
        self.assertTrue(res["guard_32k_pass"])

    def test_baseline_label_when_no_frontier_exists(self):
        res = score(120.0, frontier=0.0)
        self.assertEqual(res["label"], "BASELINE")
        self.assertTrue(res["pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
