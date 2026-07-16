#!/usr/bin/env python3
"""Unit tests for the Polaris receipt library.

Covers: canonicalization, building, signing, verification, tamper detection,
schema validation, and edge cases.

Run:  python3 -m pytest eval/polaris/test_receipt.py -v
  or:  python3 eval/polaris/test_receipt.py
"""

import base64
import json
import os
import sys
import tempfile
import unittest

# Ensure we can import from eval/polaris/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from eval.polaris.receipt import (
    AttestationBuilder,
    ReceiptValidator,
    build_receipt,
    canonicalize,
    compute_build_hash,
    generate_keypair,
    model_sha256,
    receipt_id_of,
    sign_attestation,
    verify_attestation,
)


# ---- Sample RESULT_JSON (based on real PR #279 eval) ----
SAMPLE_RESULT = {
    "commit": "d3bb8e7",
    "tps": 391.17,
    "top1": 0.9505,
    "kl": 0.0207,
    "frontier_tps": 381.29,
    "label": "S",
    "pass": True,
    "pct_over_frontier": 2.6,
    "delta_tps": 9.88,
    "model": "Qwen3.6-35B-A3B",
    "guard_model": "Qwen3-30B-A3B",
    "eval_mode": "longctx",
    "score_context": 128,
    "best_context_label": "4k-context",
    "context_gains_pct": {"128-context": 2.59, "512-context": 2.61, "4k-context": 2.84},
    "regression_labels": [],
    "guard_regression_labels": [],
    "ctx_128_tps": 391.17,
    "ctx_512_tps": 384.70,
    "ctx_4096_tps": 369.78,
    "ctx_16384_tps": 0.0,
    "ctx_32768_tps": 0.0,
    "guard_128_baseline": 381.29,
    "guard_128_ratio": 1.0259,
    "guard_128_pass": True,
    "guard_512_baseline": 374.93,
    "guard_512_ratio": 1.0261,
    "guard_512_pass": True,
    "guard_4k_baseline": 359.58,
    "guard_4k_ratio": 1.0284,
    "guard_4k_pass": True,
    "guard_16k_baseline": 0.0,
    "guard_16k_ratio": 0.0,
    "guard_16k_pass": True,
    "guard_32k_baseline": 0.0,
    "guard_32k_ratio": 0.0,
    "guard_32k_pass": True,
    "clocks_pinned": True,
    "clock_mhz": "2505",
    "clock_spread_mhz": "0",
    "pin_target_mhz": "2505",
    "eval_seed": "a1b2c3d4e5f6a7b8",
    "guard": {
        "pass": True,
        "top1": 0.9700,
        "kl": 0.0200,
        "label": "none",
        "ctx_128_tps": 493.56,
        "ctx_512_tps": 469.58,
        "ctx_4096_tps": 392.65,
        "ctx_16384_tps": 330.20,
        "ctx_32768_tps": 0.0,
        "guard_128_pass": True,
        "guard_512_pass": True,
        "guard_4k_pass": True,
        "guard_16k_pass": True,
        "guard_32k_pass": True,
        "speed_ok": True,
        "accuracy_ok": True,
    },
}


def build_sample_attestation():
    """Build a complete attestation from SAMPLE_RESULT."""
    b = AttestationBuilder()
    b.set_code(
        repo="https://github.com/gittensor-ai-lab/sparkinfer",
        commit="d3bb8e7f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d",
        build_hash="abc123def456",
        scoring_scripts_commit="x" * 40,
    )
    b.set_references(
        model_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        model_file="Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        guard_model_sha256="a" * 64,
        guard_model_file="Qwen3-30B-A3B-Q4_K_M.gguf",
        llamacpp_commit="b" * 40,
        eval_seed="a1b2c3d4e5f6a7b8",
    )
    b.set_environment(
        eval_mode="longctx",
        decode_tokens=128,
        gpu_name="NVIDIA RTX 5090",
        gpu_arch="sm_120",
        clocks_pinned=True,
        clock_mhz=2505,
        clock_spread_mhz=0,
        pin_target_mhz=2505,
        cuda_version="13.2",
        driver_version="570",
    )
    b.set_measurements(SAMPLE_RESULT)
    b.set_verdict(SAMPLE_RESULT)
    b.set_timestamp("2026-07-08T12:00:00Z")
    return b.build()


class TestCanonicalization(unittest.TestCase):
    """Canonical JSON must be deterministic and idempotent."""

    def test_canonicalize_idempotent(self):
        att = build_sample_attestation()
        c1 = canonicalize(att)
        c2 = canonicalize(att)
        self.assertEqual(c1, c2)

    def test_canonicalize_sorted_keys(self):
        # Build a dict with unsorted keys and verify canonical form is sorted
        att = build_sample_attestation()
        cb = canonicalize(att)
        # Re-parse and check key order
        d = json.loads(cb)
        keys = list(d.keys())
        self.assertEqual(keys, sorted(keys))

    def test_receipt_id_deterministic(self):
        att = build_sample_attestation()
        rid1 = receipt_id_of(att)
        rid2 = receipt_id_of(att)
        self.assertEqual(rid1, rid2)
        self.assertEqual(len(rid1), 64)  # SHA256 hex


class TestSigning(unittest.TestCase):
    """Ed25519 sign + verify round-trip."""

    def setUp(self):
        self.priv, self.pub = generate_keypair()
        self.priv_b64 = base64.b64encode(self.priv).decode()
        self.pub_b64 = base64.b64encode(self.pub).decode()

    def test_generate_keypair(self):
        self.assertEqual(len(self.priv), 32)
        self.assertEqual(len(self.pub), 32)

    def test_sign_and_verify(self):
        att = build_sample_attestation()
        sig = sign_attestation(att, self.priv)
        self.assertTrue(verify_attestation(att, sig, self.pub_b64))

    def test_wrong_key_fails(self):
        att = build_sample_attestation()
        sig = sign_attestation(att, self.priv)
        _, other_pub = generate_keypair()
        other_pub_b64 = base64.b64encode(other_pub).decode()
        self.assertFalse(verify_attestation(att, sig, other_pub_b64))

    def test_tampered_attestation_fails(self):
        att = build_sample_attestation()
        sig = sign_attestation(att, self.priv)
        # Tamper with TPS
        att["verdict"]["tps"] = 999.99
        self.assertFalse(verify_attestation(att, sig, self.pub_b64))

    def test_tampered_label_fails(self):
        att = build_sample_attestation()
        sig = sign_attestation(att, self.priv)
        att["verdict"]["label"] = "XL"
        self.assertFalse(verify_attestation(att, sig, self.pub_b64))


class TestReceiptBuilding(unittest.TestCase):
    """End-to-end receipt assembly."""

    def setUp(self):
        self.priv, self.pub = generate_keypair()

    def test_build_complete_receipt(self):
        att = build_sample_attestation()
        receipt = build_receipt(att, self.priv, prev_receipt_hash=None, chain_index=0)

        self.assertEqual(receipt["polaris_version"], 1)
        self.assertEqual(len(receipt["receipt_id"]), 64)
        self.assertIsNone(receipt["chain"]["prev_receipt_hash"])
        self.assertEqual(receipt["chain"]["chain_index"], 0)
        self.assertIn("signature", receipt)
        self.assertIn("public_key", receipt)
        self.assertEqual(receipt["attestation"], att)

    def test_receipt_chain(self):
        att1 = build_sample_attestation()
        r1 = build_receipt(att1, self.priv, prev_receipt_hash=None, chain_index=0)

        # Second receipt chains to first
        att2 = build_sample_attestation()
        att2["timestamp_utc"] = "2026-07-08T13:00:00Z"
        r2 = build_receipt(att2, self.priv,
                           prev_receipt_hash=r1["receipt_id"], chain_index=1)

        self.assertEqual(r2["chain"]["prev_receipt_hash"], r1["receipt_id"])
        self.assertEqual(r2["chain"]["chain_index"], 1)

    def test_receipt_id_matches(self):
        att = build_sample_attestation()
        receipt = build_receipt(att, self.priv)
        expected = receipt_id_of(att)
        self.assertEqual(receipt["receipt_id"], expected)


class TestReceiptValidator(unittest.TestCase):
    """Receipt validation: schema, consistency, signature checks."""

    def setUp(self):
        self.priv, self.pub = generate_keypair()
        self.pub_b64 = base64.b64encode(self.pub).decode()

    def _make_receipt(self, attestation=None):
        att = attestation or build_sample_attestation()
        return build_receipt(att, self.priv)

    def test_valid_receipt_passes_schema(self):
        receipt = self._make_receipt()
        v = ReceiptValidator(receipt)
        issues = v.validate_schema()
        self.assertEqual(issues, [], f"Schema issues: {issues}")

    def test_valid_receipt_passes_all(self):
        receipt = self._make_receipt()
        v = ReceiptValidator(receipt)
        passed, results = v.verify(public_key_b64=self.pub_b64)
        self.assertTrue(passed, f"Verification failed:\n" + "\n".join(results))

    def test_missing_top_level_field(self):
        receipt = self._make_receipt()
        del receipt["signature"]
        v = ReceiptValidator(receipt)
        issues = v.validate_schema()
        self.assertTrue(any("signature" in i for i in issues))

    def test_missing_attestation_field(self):
        receipt = self._make_receipt()
        del receipt["attestation"]["code"]
        v = ReceiptValidator(receipt)
        issues = v.validate_schema()
        self.assertTrue(any("code" in i for i in issues))

    def test_tampered_receipt_rejected(self):
        receipt = self._make_receipt()
        # Tamper with TPS value
        receipt["attestation"]["verdict"]["tps"] = 999.99
        v = ReceiptValidator(receipt)
        passed, results = v.verify(public_key_b64=self.pub_b64)
        self.assertFalse(passed)

    def test_wrong_public_key_rejected(self):
        receipt = self._make_receipt()
        _, other_pub = generate_keypair()
        other_b64 = base64.b64encode(other_pub).decode()
        v = ReceiptValidator(receipt)
        passed, results = v.verify(public_key_b64=other_b64)
        self.assertFalse(passed)

    def test_consistency_label_reject_pass_false(self):
        """If label is REJECT, pass must be false."""
        att = build_sample_attestation()
        att["verdict"]["label"] = "REJECT"
        att["verdict"]["pass"] = True  # inconsistent
        receipt = self._make_receipt(att)
        v = ReceiptValidator(receipt)
        issues = v.validate_consistency()
        self.assertTrue(any("REJECT" in i for i in issues))

    def test_consistency_negative_tps(self):
        att = build_sample_attestation()
        att["measurements"]["primary"]["ctx_128_tps"] = -1.0
        receipt = self._make_receipt(att)
        v = ReceiptValidator(receipt)
        issues = v.validate_consistency()
        self.assertTrue(any("negative" in i for i in issues))


class TestEdgeCases(unittest.TestCase):
    """Corner cases and error handling."""

    def setUp(self):
        self.priv, self.pub = generate_keypair()

    def _make_receipt(self, attestation=None):
        att = attestation or build_sample_attestation()
        return build_receipt(att, self.priv)

    def test_no_guard_model(self):
        """Receipts work without a guard model."""
        att = build_sample_attestation()
        del att["measurements"]["guard"]
        receipt = self._make_receipt(att)
        v = ReceiptValidator(receipt)
        passed, results = v.verify()
        self.assertTrue(passed)

    def test_reject_result(self):
        """REJECT verdicts should verify OK (they're still valid receipts)."""
        reject_result = dict(SAMPLE_RESULT)
        reject_result.update({"label": "REJECT", "pass": False, "tps": 354.31,
                              "reason": "guard regression", "auto_close": True})
        b = AttestationBuilder()
        b.set_code(repo="...", commit="x" * 40, build_hash="abc")
        b.set_references(model_sha256="x" * 64, model_file="test.gguf")
        b.set_environment(eval_mode="longctx", decode_tokens=128,
                          gpu_name="RTX 5090", gpu_arch="sm_120",
                          clocks_pinned=True, clock_mhz=2505,
                          clock_spread_mhz=0, pin_target_mhz=2505)
        b.set_measurements(reject_result)
        b.set_verdict(reject_result)
        b.set_timestamp("2026-07-08T12:00:00Z")
        att = b.build()

        receipt = self._make_receipt(att)
        v = ReceiptValidator(receipt)
        passed, results = v.verify()
        self.assertTrue(passed, f"REJECT receipt should verify:\n" + "\n".join(results))

    def test_build_hash_compute(self):
        """compute_build_hash returns empty string for nonexistent dir."""
        h = compute_build_hash("/nonexistent/path")
        self.assertEqual(h, "")

    def test_model_sha256_compute(self):
        """model_sha256 returns empty string for nonexistent file."""
        h = model_sha256("/nonexistent/model.gguf")
        self.assertEqual(h, "")

    def test_canonicalize_preserves_types(self):
        att = build_sample_attestation()
        cb = canonicalize(att)
        d = json.loads(cb)
        self.assertIsInstance(d["verdict"]["label"], str)
        self.assertIsInstance(d["verdict"]["tps"], float)
        self.assertIsInstance(d["verdict"]["pass"], bool)
        self.assertIsInstance(d["environment"]["clocks_pinned"], bool)

    def test_empty_receipt_rejected(self):
        v = ReceiptValidator({})
        issues = v.validate_schema()
        self.assertTrue(len(issues) > 0)


class TestRealWorldScenarios(unittest.TestCase):
    """Tests that mirror real eval scenarios."""

    def setUp(self):
        self.priv, self.pub = generate_keypair()

    def _make(self, att):
        return build_receipt(att, self.priv)

    def test_baseline_eval(self):
        """A BASELINE (no frontier) should verify."""
        baseline = dict(SAMPLE_RESULT)
        baseline.update({"label": "BASELINE", "pass": True, "frontier_tps": 0, "pct_over_frontier": 0})
        b = AttestationBuilder()
        b.set_code(repo="...", commit="x" * 40, build_hash="abc")
        b.set_references(model_sha256="x" * 64, model_file="test.gguf")
        b.set_environment(eval_mode="longctx", decode_tokens=128,
                          gpu_name="RTX 5090", gpu_arch="sm_120",
                          clocks_pinned=True, clock_mhz=2505,
                          clock_spread_mhz=0, pin_target_mhz=2505)
        b.set_measurements(baseline)
        b.set_verdict(baseline)
        b.set_timestamp("2026-07-08T12:00:00Z")
        att = b.build()

        receipt = self._make(att)
        v = ReceiptValidator(receipt)
        passed, results = v.verify()
        self.assertTrue(passed)

    def test_none_label(self):
        """A 'none' label (within significance gate) should verify."""
        none_result = dict(SAMPLE_RESULT)
        none_result.update({"label": "none", "pass": True, "pct_over_frontier": 0.5, "delta_tps": 1.5})
        b = AttestationBuilder()
        b.set_code(repo="...", commit="x" * 40, build_hash="abc")
        b.set_references(model_sha256="x" * 64, model_file="test.gguf")
        b.set_environment(eval_mode="longctx", decode_tokens=128,
                          gpu_name="RTX 5090", gpu_arch="sm_120",
                          clocks_pinned=True, clock_mhz=2505,
                          clock_spread_mhz=0, pin_target_mhz=2505)
        b.set_measurements(none_result)
        b.set_verdict(none_result)
        b.set_timestamp("2026-07-08T12:00:00Z")
        att = b.build()

        receipt = self._make(att)
        v = ReceiptValidator(receipt)
        passed, results = v.verify()
        self.assertTrue(passed)


class TestCanonicalizationEdgeCases(unittest.TestCase):
    """Floating point and Unicode in canonicalization."""

    def test_float_precision(self):
        """Fl oats are rounded to 6dp for canonical stability."""
        att = build_sample_attestation()
        att["verdict"]["tps"] = 391.1700000001  # Excess precision
        c1 = canonicalize(att)
        att["verdict"]["tps"] = 391.17
        c2 = canonicalize(att)
        self.assertEqual(c1, c2)

    def test_unicode_in_model_name(self):
        att = build_sample_attestation()
        att["measurements"]["primary"]["model"] = "Qwen3.6-35B-A3B \U0001f680"
        c = canonicalize(att)
        # Should not raise
        self.assertIsInstance(c, bytes)


if __name__ == "__main__":
    unittest.main()
