#!/usr/bin/env python3
"""Regression tests for convert_gguf.py vocab derivation (issue #637).

`config.txt` must take vocab from the token_embd.weight tensor, preferring it
over `qwen3moe.vocab_size` metadata, so the converter agrees with the C++ loader
(`emb->dims[1]` in runtime/examples/qwen3_gguf_config.h). A prior revision wrote
the assignment as `... if False else cfg["vocab"]`, making it a no-op, so a GGUF
with missing or disagreeing metadata produced a wrong vocab and broke downstream
load_weights / LM-head sizing.

Axis note: gguf's ReaderTensor.shape is the RAW ggml dim order
([n_embd, n_vocab]); only `.data` is reshaped into reversed numpy order
([vocab, hidden]). Vocab is therefore dims[1], not dims[0].

convert_gguf.py imports `gguf` and `numpy` at module scope, which the repo does
not require for tests, so these tests exec the pure helper out of the source
rather than importing the module.

Run from the repo root:
  python3 runtime/tools/test_convert_gguf.py
"""
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "runtime" / "tools" / "convert_gguf.py"


def read_script():
    return SCRIPT.read_text(encoding="utf-8")


def load_vocab_helper():
    """exec only vocab_from_embedding() so no gguf/numpy import is needed."""
    src = read_script()
    m = re.search(r"^def vocab_from_embedding\(.*?(?=^\S|\Z)", src, re.M | re.S)
    assert m, "vocab_from_embedding() not found in %s" % SCRIPT.name
    namespace = {}
    exec(m.group(0), namespace)  # noqa: S102 - pure function, no imports
    return namespace["vocab_from_embedding"]


def vocab_assignment(src):
    """The line in main() that sets cfg["vocab"] from the embedding tensor."""
    m = re.search(r'^\s*cfg\["vocab"\]\s*=\s*(.+)$', src, re.M)
    assert m, 'cfg["vocab"] assignment not found'
    return m.group(1).strip()


class VocabFromEmbeddingTest(unittest.TestCase):
    """The helper must read ggml dims[1], matching qwen3_gguf_config.h."""

    def setUp(self):
        self.vocab_from_embedding = load_vocab_helper()

    def test_reads_dims_1_of_ggml_shape(self):
        # ggml order for token_embd.weight is [n_embd, n_vocab].
        self.assertEqual(self.vocab_from_embedding((2048, 151936), 0), 151936)

    def test_does_not_read_dims_0(self):
        """dims[0] is hidden size — returning it would mis-size the LM head."""
        self.assertNotEqual(self.vocab_from_embedding((2048, 151936), 0), 2048)

    def test_tensor_overrides_disagreeing_metadata(self):
        """The tensor wins; the metadata value is only a fallback."""
        self.assertEqual(self.vocab_from_embedding((2048, 151936), 151645), 151936)

    def test_non_2d_tensor_keeps_metadata_fallback(self):
        """Mirrors the C++ `emb->n_dims >= 2` guard."""
        self.assertEqual(self.vocab_from_embedding((151936,), 151645), 151645)
        self.assertEqual(self.vocab_from_embedding((), 151645), 151645)

    def test_returns_int(self):
        result = self.vocab_from_embedding([2048, 151936], 0)
        self.assertIsInstance(result, int)


class VocabAssignmentIsLiveTest(unittest.TestCase):
    """The assignment must not be short-circuited back to its own value."""

    def setUp(self):
        self.src = read_script()

    def test_assignment_is_not_dead_coded(self):
        rhs = vocab_assignment(self.src)
        self.assertNotIn("if False", rhs,
                         "vocab assignment is dead-coded: %s" % rhs)
        self.assertNotRegex(rhs, r'else\s+cfg\["vocab"\]',
                            "vocab assignment resolves back to itself: %s" % rhs)

    def test_assignment_uses_the_helper(self):
        self.assertIn("vocab_from_embedding", vocab_assignment(self.src))


if __name__ == "__main__":
    unittest.main(verbosity=2)
