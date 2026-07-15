#!/usr/bin/env python3
"""Unit tests for bench sweep JSON parsing (_eval_speed.sh helper logic).

Run from the repo root:
  python3 bench/scripts/test_bench_sweep.py
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVAL_SPEED = ROOT / "bench" / "scripts" / "_eval_speed.sh"


def sweep_get(ctx: int, field: str, raw: str) -> float:
    cmd = [
        "bash", "-c",
        f'source "{EVAL_SPEED}"; _BENCH_SWEEP_JSON="$1"; _bench_sweep_get "$2" "$3"',
        "_",
        raw,
        str(ctx),
        field,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


class BenchSweepParseTest(unittest.TestCase):
    def test_decode_and_prefill(self):
        raw = json.dumps({
            "0": {"decode_tps": 350.5, "prefill_pp": 0.0},
            "4096": {"decode_tps": 290.1, "prefill_pp": 1200.3},
            "32768": {"decode_tps": 192.0, "prefill_pp": 450.7},
        }, separators=(",", ":"))
        self.assertAlmostEqual(sweep_get(0, "decode_tps", raw), 350.5)
        self.assertAlmostEqual(sweep_get(4096, "prefill_pp", raw), 1200.3)
        self.assertAlmostEqual(sweep_get(99999, "decode_tps", raw), 0.0)

    def test_malformed_json(self):
        self.assertEqual(sweep_get(0, "decode_tps", "not-json"), 0.0)


if __name__ == "__main__":
    unittest.main()
