#!/usr/bin/env python3
"""Unit tests for bidir per-GGUF accuracy cache key logic.

Run from the repo root:
  python3 bench/scripts/test_accuracy_cache.py
"""
import json
import unittest


def cache_key(models_dir: str, model_file: str) -> str:
    return f"{models_dir}/{model_file}"


def should_reuse(cached: dict, gguf: str, last_gguf: str) -> bool:
    return gguf in cached and gguf == last_gguf


class AccuracyCacheTest(unittest.TestCase):
    def test_bidir_pass_order(self):
        p35 = cache_key("/models35", "Qwythos-Q4_K_M.gguf")
        p36 = cache_key("/models36", "Qwen3.6-Q4_K_M.gguf")
        cached = {}
        order = [
            ("primary35", p35),
            ("guard36", p36),
            ("primary36", p36),
            ("guard35", p35),
        ]
        last = ""
        hits = []
        for _role, gguf in order:
            if gguf in cached:
                hits.append(gguf)
            else:
                cached[gguf] = {"top1": 0.97, "kl": 0.02}
            last = gguf
        self.assertEqual(hits, [p36, p35])
        self.assertEqual(len(cached), 2)

    def test_same_gguf_reap_bench(self):
        p36 = cache_key("/m36", "a.gguf")
        self.assertTrue(should_reuse({p36: 1}, p36, p36))
        self.assertFalse(should_reuse({p36: 1}, p36, "/other/a.gguf"))


if __name__ == "__main__":
    unittest.main()
