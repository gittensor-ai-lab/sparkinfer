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

    def test_raw_speedup_buckets_without_difficulty_boost(self):
        env = {"SPARKINFER_DIFFICULTY_BOOST": "0"}
        self.assertEqual(score(104.0, env=env)["label"], "S")
        self.assertEqual(score(107.0, env=env)["label"], "M")
        self.assertEqual(score(111.0, env=env)["label"], "L")
        self.assertEqual(score(119.0, env=env)["label"], "XL")

    def test_difficulty_boost_changes_label_but_not_raw_percent(self):
        res = score(484.79, frontier=469.13, ceiling=366.0, top1=0.9612, kl=0.0175,
                    commit="c30bf58",
                    env={"SPARKINFER_DIFFICULTY_BOOST": "1",
                         "SPARKINFER_DIFFICULTY_REF": "365.85",
                         "SPARKINFER_DIFFICULTY_K": "8",
                         "SPARKINFER_DIFFICULTY_MAX": "4"})
        self.assertEqual(res["label"], "L")
        self.assertEqual(res["pct_over_frontier"], 3.3)
        self.assertGreaterEqual(res["effective_pct"], 10.0)
        self.assertLess(res["effective_pct"], 18.0)

    def test_long_context_metadata_is_preserved_in_verdict(self):
        prov = {
            "eval_mode": "longctx",
            "score_context": 16384,
            "ctx_2048_tps": 300.0,
            "ctx_16384_tps": 120.0,
            "ctx_32768_tps": 80.0,
            "guard_2k_baseline": 305.0,
            "guard_2k_ratio": 0.9836,
            "guard_2k_pass": True,
        }
        res = score(120.0, frontier=100.0, prov=prov)
        self.assertEqual(res["label"], "XL")
        self.assertEqual(res["eval_mode"], "longctx")
        self.assertEqual(res["score_context"], 16384)
        self.assertEqual(res["ctx_2048_tps"], 300.0)
        self.assertEqual(res["ctx_32768_tps"], 80.0)
        self.assertTrue(res["guard_2k_pass"])

    def test_baseline_label_when_no_frontier_exists(self):
        res = score(120.0, frontier=0.0)
        self.assertEqual(res["label"], "BASELINE")
        self.assertTrue(res["pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
