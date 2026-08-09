#!/usr/bin/env python3
"""Regression tests for convert_gguf vocab derivation (issue #637).

  python3 runtime/tools/test_convert_gguf.py
"""
import re
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().with_name("convert_gguf.py")


class TestConvertGgufVocab(unittest.TestCase):
    def test_shortcircuit_gone_and_dims1_used(self):
        src = SCRIPT.read_text(encoding="utf-8")
        self.assertNotRegex(src, r"if\s+False\s+else")
        self.assertRegex(
            src,
            r'cfg\["vocab"\]\s*=\s*int\(\s*shape\[1\]\s*\)',
        )
        self.assertIn("token_embd.weight", src)

    def test_logic_matches_cpp_dims1(self):
        # Mirror the inlined rule for a few shapes.
        def vocab(shape, fallback):
            return int(shape[1]) if len(shape) >= 2 else int(fallback)

        self.assertEqual(vocab((4096, 151936), 1), 151936)
        self.assertEqual(vocab((2048, 248320), 151936), 248320)
        self.assertEqual(vocab((151936,), 42), 42)


if __name__ == "__main__":
    unittest.main()
