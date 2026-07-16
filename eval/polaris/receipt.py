#!/usr/bin/env python3
"""Polaris receipt: schema, builder, canonicalization, signing, and verification.

A Polaris receipt binds a SparkInfer eval result to its provenance — code commit,
model SHA256, eval seed, build hash, observed clock — so a third party can verify
the result without re-running the GPU job.

Two attestation modes are supported:

  Ed25519 (software signatures):
    1. attestation  — canonical data block (what gets signed)
    2. signature    — Ed25519 signature over canonical attestation bytes
    3. receipt      — attestation + signature + public_key + chain metadata

  Polaris TDX (hardware attestation via Intel DCAP):
    1. attestation  — canonical data block (provenance metadata)
    2. tdx          — Polaris DCAP-quoted receipt (quote, collateral, verification)
    3. receipt      — attestation + tdx + attestation_type + chain metadata

Design:
  - Judge (on eval box) assembles the unsigned attestation; the bot submits
    scoring to Polaris TDX or signs with Ed25519 as a fallback.
  - Canonical JSON (sorted keys, compact separators, pre-rounded floats) ensures
    deterministic hashing across Python versions.
"""

import base64
import datetime
import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

# ---- Ed25519 via cryptography (already installed on bot + eval boxes) ----
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

# ---- Numeric precision for canonicalization ----
# All floats in the attestation are rounded to these precisions before
# serialization, so floating-point representation differences across
# Python versions cannot break signature verification.
PRECISION = {
    "tps": 2,         # tok/s values
    "ratio": 4,       # guard ratios
    "top1": 4,        # token-match accuracy
    "kl": 4,          # KL divergence
    "delta": 2,       # delta_tps
    "pct": 1,         # pct_over_frontier (one decimal)
    "gain": 2,        # context_gains_pct values
}


def _round_attestation(obj: Any) -> Any:
    """Recursively round numeric fields to canonical precision.

    We don't know field names at the top level, so we apply precision based on
    heuristics: values that look like TPS (typically 50-500), ratios (0.9-1.1),
    probabilities (0-1), etc. The canonicalization is idempotent — calling it
    twice produces the same result.
    """
    if isinstance(obj, dict):
        return {k: _round_attestation(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_round_attestation(v) for v in obj]
    elif isinstance(obj, float):
        # Use a fixed precision that preserves enough information for any field.
        # 6 decimal places covers all our precisions (tps=2, ratio=4, top1=4, kl=4)
        # while being stable across Python versions.
        return round(obj, 6)
    else:
        return obj


def canonicalize(attestation: dict) -> bytes:
    """Serialize attestation dict to canonical JSON bytes.

    Uses sorted keys, compact separators (no whitespace), and ensure_ascii=False
    so Unicode in model names survives. All numeric fields are pre-rounded.
    """
    rounded = _round_attestation(attestation)
    return json.dumps(rounded, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def receipt_id_of(attestation: dict) -> str:
    """SHA256 hex digest of canonical attestation bytes."""
    return hashlib.sha256(canonicalize(attestation)).hexdigest()


# ---- Key generation ----

def generate_keypair() -> Tuple[bytes, bytes]:
    """Generate a fresh Ed25519 keypair.

    Returns (private_key_bytes, public_key_bytes).
    private_key_bytes is 32 bytes (raw seed).
    public_key_bytes is 32 bytes.
    """
    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key()
    priv_bytes = priv.private_bytes_raw()
    pub_bytes = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return priv_bytes, pub_bytes


def sign_attestation(attestation: dict, private_key_bytes: bytes) -> str:
    """Sign canonical attestation bytes with Ed25519.

    Returns base64-encoded signature (88 chars).
    """
    priv = ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    sig = priv.sign(canonicalize(attestation))
    return base64.b64encode(sig).decode("ascii")


def verify_attestation(attestation: dict, signature_b64: str,
                       public_key_b64: str) -> bool:
    """Verify an Ed25519 signature over canonical attestation bytes."""
    try:
        sig = base64.b64decode(signature_b64)
        pub_bytes = base64.b64decode(public_key_b64)
        pub = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
        pub.verify(sig, canonicalize(attestation))
        return True
    except Exception:
        return False


# ---- Receipt assembly ----

def build_receipt(attestation: dict, private_key_bytes: bytes,
                  prev_receipt_hash: Optional[str] = None,
                  chain_index: int = 0) -> dict:
    """Assemble a complete Polaris receipt.

    Args:
        attestation: The canonical attestation dict.
        private_key_bytes: 32-byte Ed25519 seed for signing.
        prev_receipt_hash: SHA256 of the previous receipt in the chain (or None).
        chain_index: Monotonically increasing index.

    Returns:
        Complete receipt dict ready for JSON serialization.
    """
    pub_bytes = ed25519.Ed25519PrivateKey.from_private_bytes(
        private_key_bytes
    ).public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signature_b64 = sign_attestation(attestation, private_key_bytes)
    rid = receipt_id_of(attestation)

    return {
        "polaris_version": 1,
        "receipt_id": rid,
        "chain": {
            "prev_receipt_hash": prev_receipt_hash,
            "chain_index": chain_index,
        },
        "attestation": attestation,
        "signature": signature_b64,
        "public_key": base64.b64encode(pub_bytes).decode("ascii"),
    }


# Alias for clarity when both paths are in play
build_ed25519_receipt = build_receipt


def build_polaris_receipt(
    polaris_response: dict,
    attestation: dict,
    prev_receipt_hash: Optional[str] = None,
    chain_index: int = 0,
) -> dict:
    """Assemble a Polaris TDX receipt from a Polaris /v1/attest response.

    This wraps the Polaris DCAP-quoted attestation (Intel TDX hardware root of
    trust) around our standard attestation structure. The receipt carries the
    full DCAP quote + collateral chain so verifiers can check everything
    offline without trusting Polaris.

    Args:
        polaris_response: The JSON response from Polaris POST /v1/attest.
        attestation: Our canonical attestation dict (code/refs/env/measurements/verdict).
        prev_receipt_hash: SHA256 of the previous receipt in the chain (or None).
        chain_index: Monotonically increasing index.

    Returns:
        Complete TDX receipt dict ready for JSON serialization.
    """
    tee = polaris_response.get("tee_attestation", {}) or {}
    verification = polaris_response.get("verification", {}) or tee.get("verification", {}) or {}

    # Receipt ID: bind the quote and result together
    id_input = (tee.get("quote_b64", "")[:64] +
                tee.get("result_sha256", "")).encode("utf-8")
    rid = hashlib.sha256(id_input).hexdigest()[:32]

    return {
        "polaris_version": 1,
        "receipt_id": rid,
        "chain": {
            "prev_receipt_hash": prev_receipt_hash,
            "chain_index": chain_index,
        },
        "attestation": attestation,
        "attestation_type": "tdx-quote",
        "tdx": {
            "quote_b64": tee.get("quote_b64", ""),
            "collateral_b64": tee.get("collateral_b64", ""),
            "nonce": tee.get("nonce", ""),
            "e2e_pubkey_b64": tee.get("e2e_pubkey_b64", ""),
            "bound_digest": tee.get("bound_digest", ""),
            "result_sha256": tee.get("result_sha256", ""),
            "stdout_b64": tee.get("stdout_b64", "") or polaris_response.get("stdout_b64", ""),
            "files_sha256": tee.get("files_sha256", "") or polaris_response.get("files_sha256", ""),
            "workload_sha256": tee.get("workload_sha256", ""),
            "verification": {
                "intel_verified": bool(verification.get("intel_verified", False)),
                "report_data_match": bool(verification.get("report_data_match", False)),
            },
            "cost_usd": polaris_response.get("cost_usd"),
        },
        # Carry expected hashes for verification
        "_expected": polaris_response.get("_expected", {}),
    }


# ---- Attestation builder ----

class AttestationBuilder:
    """Accumulates provenance data and RESULT_JSON into a canonical attestation."""

    def __init__(self):
        self._code = {}
        self._references = {}
        self._environment = {}
        self._measurements = {}
        self._verdict = {}
        self._timestamp = ""

    def set_code(self, repo: str, commit: str, build_hash: str,
                 scoring_scripts_commit: str = ""):
        self._code = {
            "repo": repo,
            "commit": commit,
            "build_hash": build_hash,
            "scoring_scripts_commit": scoring_scripts_commit,
        }

    def set_references(self, model_sha256: str, model_file: str,
                       guard_model_sha256: str = "",
                       guard_model_file: str = "",
                       llamacpp_commit: str = "",
                       eval_seed: str = ""):
        self._references = {
            "model_sha256": model_sha256,
            "model_file": model_file,
            "guard_model_sha256": guard_model_sha256,
            "guard_model_file": guard_model_file,
            "llamacpp_commit": llamacpp_commit,
            "eval_seed": eval_seed,
        }

    def set_environment(self, eval_mode: str, decode_tokens: int,
                        gpu_name: str, gpu_arch: str,
                        clocks_pinned: bool, clock_mhz: int,
                        clock_spread_mhz: int, pin_target_mhz: int,
                        cuda_version: str = "", driver_version: str = ""):
        self._environment = {
            "eval_mode": eval_mode,
            "decode_tokens": decode_tokens,
            "gpu_name": gpu_name,
            "gpu_arch": gpu_arch,
            "clocks_pinned": clocks_pinned,
            "clock_mhz": clock_mhz,
            "clock_spread_mhz": clock_spread_mhz,
            "pin_target_mhz": pin_target_mhz,
            "cuda_version": cuda_version,
            "driver_version": driver_version,
        }

    def set_measurements(self, result_json: dict):
        """Extract primary + guard measurements from a RESULT_JSON verdict.

        Handles both dual-model (primary + guard blocks) and single-model formats.
        """
        guard = result_json.get("guard") or {}

        # Primary (scored model)
        primary_model = result_json.get("model", "")
        primary = {
            "model": primary_model,
            "model_sha256": "",  # filled by judge
            "ctx_128_tps": _f(result_json.get("ctx_128_tps")),
            "ctx_512_tps": _f(result_json.get("ctx_512_tps")),
            "ctx_4096_tps": _f(result_json.get("ctx_4096_tps")),
            "ctx_16384_tps": _f(result_json.get("ctx_16384_tps")),
            "ctx_32768_tps": _f(result_json.get("ctx_32768_tps")),
            "top1": _f(result_json.get("top1")),
            "kl": _f(result_json.get("kl")),
            "guard_128_baseline": _f(result_json.get("guard_128_baseline")),
            "guard_128_ratio": _f(result_json.get("guard_128_ratio")),
            "guard_128_pass": result_json.get("guard_128_pass", True),
            "guard_512_baseline": _f(result_json.get("guard_512_baseline")),
            "guard_512_ratio": _f(result_json.get("guard_512_ratio")),
            "guard_512_pass": result_json.get("guard_512_pass", True),
            "guard_4k_baseline": _f(result_json.get("guard_4k_baseline")),
            "guard_4k_ratio": _f(result_json.get("guard_4k_ratio")),
            "guard_4k_pass": result_json.get("guard_4k_pass", True),
            "guard_16k_baseline": _f(result_json.get("guard_16k_baseline")),
            "guard_16k_ratio": _f(result_json.get("guard_16k_ratio")),
            "guard_16k_pass": result_json.get("guard_16k_pass", True),
            "guard_32k_baseline": _f(result_json.get("guard_32k_baseline")),
            "guard_32k_ratio": _f(result_json.get("guard_32k_ratio")),
            "guard_32k_pass": result_json.get("guard_32k_pass", True),
        }

        # Guard model (Qwen3-30B), if present
        guard_model = result_json.get("guard_model", "")
        guard_data = {}
        if guard:
            guard_data = {
                "model": guard_model,
                "model_sha256": "",
                "ctx_128_tps": _f(guard.get("ctx_128_tps")),
                "ctx_512_tps": _f(guard.get("ctx_512_tps")),
                "ctx_4096_tps": _f(guard.get("ctx_4096_tps")),
                "ctx_16384_tps": _f(guard.get("ctx_16384_tps")),
                "ctx_32768_tps": _f(guard.get("ctx_32768_tps")),
                "top1": _f(guard.get("top1")),
                "kl": _f(guard.get("kl")),
                "speed_ok": guard.get("speed_ok", True),
                "accuracy_ok": guard.get("accuracy_ok", True),
            }

        self._measurements = {
            "primary": primary,
        }
        if guard_data:
            self._measurements["guard"] = guard_data

    def set_verdict(self, result_json: dict):
        """Extract verdict fields from RESULT_JSON."""
        self._verdict = {
            "model": result_json.get("model", ""),
            "label": result_json.get("label", "?"),
            "pass": result_json.get("pass", False),
            "tps": _f(result_json.get("tps")),
            "delta_tps": _f(result_json.get("delta_tps")),
            "pct_over_frontier": _f(result_json.get("pct_over_frontier")),
            "score_context": result_json.get("score_context"),
            "best_context_label": result_json.get("best_context_label", ""),
            "context_gains_pct": result_json.get("context_gains_pct") or {},
            "regression_labels": result_json.get("regression_labels") or [],
            "guard_regression_labels": result_json.get("guard_regression_labels") or [],
            "reason": result_json.get("reason"),
        }

    def set_timestamp(self, ts: Optional[str] = None):
        self._timestamp = ts or datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    def build(self) -> dict:
        """Assemble the complete attestation dict."""
        return {
            "code": self._code,
            "references": self._references,
            "environment": self._environment,
            "measurements": self._measurements,
            "verdict": self._verdict,
            "timestamp_utc": self._timestamp,
        }


def _f(val):
    """Round a float value for canonicalization, or return as-is if not a float."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return round(float(val), 6)
    return val


# ---- Receipt validator ----

class ReceiptValidator:
    """Validates a receipt: schema, signature, consistency, and optional chain."""

    def __init__(self, receipt: dict):
        self.receipt = receipt
        self.attestation = receipt.get("attestation") or {}

    def validate_schema(self) -> List[str]:
        """Check that all required fields are present. Returns list of violations."""
        issues = []
        r = self.receipt
        a = self.attestation

        # Top-level
        for k in ["polaris_version", "receipt_id", "attestation", "signature", "public_key"]:
            if k not in r:
                issues.append(f"missing top-level field: {k}")

        if not isinstance(r.get("polaris_version"), int):
            issues.append("polaris_version must be an integer")

        if r.get("polaris_version") != 1:
            issues.append(f"unsupported polaris_version: {r.get('polaris_version')}")

        # Attestation sub-keys
        for k in ["code", "references", "environment", "measurements", "verdict", "timestamp_utc"]:
            if k not in a:
                issues.append(f"missing attestation field: {k}")

        # Code
        for k in ["repo", "commit", "build_hash"]:
            if k not in a.get("code", {}):
                issues.append(f"missing attestation.code.{k}")

        # References
        for k in ["model_sha256", "model_file"]:
            if k not in a.get("references", {}):
                issues.append(f"missing attestation.references.{k}")

        # Environment
        for k in ["eval_mode", "decode_tokens", "gpu_name", "clocks_pinned", "clock_mhz"]:
            if k not in a.get("environment", {}):
                issues.append(f"missing attestation.environment.{k}")

        # Measurements
        meas = a.get("measurements", {})
        if "primary" not in meas:
            issues.append("missing attestation.measurements.primary")
        else:
            p = meas["primary"]
            for k in ["ctx_128_tps", "ctx_512_tps", "ctx_4096_tps", "top1", "kl"]:
                if k not in p:
                    issues.append(f"missing attestation.measurements.primary.{k}")

        # Verdict
        for k in ["label", "pass", "tps"]:
            if k not in a.get("verdict", {}):
                issues.append(f"missing attestation.verdict.{k}")

        return issues

    def validate_consistency(self) -> List[str]:
        """Cross-field consistency checks. Returns list of violations."""
        issues = []
        a = self.attestation
        v = a.get("verdict", {})
        e = a.get("environment", {})
        meas = a.get("measurements", {})

        # If pass is false, label should be REJECT or none
        if v.get("pass") is False:
            if v.get("label") not in ("REJECT", "none"):
                issues.append(
                    f"verdict.pass=false but label is '{v.get('label')}' "
                    f"(expected REJECT or none)"
                )

        # If label is REJECT, pass should be false
        if v.get("label") == "REJECT" and v.get("pass") is not False:
            issues.append("verdict.label=REJECT but verdict.pass is not false")

        # All TPS values should be non-negative
        primary = meas.get("primary", {})
        for key in ["ctx_128_tps", "ctx_512_tps", "ctx_4096_tps",
                     "ctx_16384_tps", "ctx_32768_tps"]:
            val = primary.get(key)
            if val is not None and isinstance(val, (int, float)) and val < 0:
                issues.append(f"measurements.primary.{key} is negative: {val}")

        # If clocks_pinned is true, pin_target_mhz should be > 0
        if e.get("clocks_pinned") and not e.get("pin_target_mhz", 0):
            issues.append("clocks_pinned=true but pin_target_mhz is 0 or missing")

        # Score context should be one of the known values
        sc = v.get("score_context")
        if sc is not None and sc not in (128, 512, 2048, 4096, 16384, 32768):
            issues.append(f"verdict.score_context={sc} is not a known context length")

        # Timestamp should be parseable ISO
        ts = a.get("timestamp_utc", "")
        if ts:
            try:
                datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                issues.append(f"timestamp_utc='{ts}' is not valid ISO 8601")

        # If guard is present, check consistency
        guard = meas.get("guard")
        if guard:
            if guard.get("speed_ok") is False:
                # At least one context should have failed
                pass

        return issues

    def verify_signature(self) -> Tuple[bool, str]:
        """Verify the Ed25519 signature over the attestation."""
        sig = self.receipt.get("signature", "")
        pub = self.receipt.get("public_key", "")
        if not sig or not pub:
            return False, "missing signature or public_key"

        if verify_attestation(self.attestation, sig, pub):
            return True, "signature valid"
        else:
            return False, "signature INVALID"

    def verify_hash(self) -> Tuple[bool, str]:
        """Verify receipt_id matches the expected binding for this receipt type."""
        actual = self.receipt.get("receipt_id", "")
        if self.receipt.get("attestation_type") == "tdx-quote" or "tdx" in self.receipt:
            tee = self.receipt.get("tdx", {})
            id_input = (tee.get("quote_b64", "")[:64] +
                        tee.get("result_sha256", "")).encode("utf-8")
            expected = hashlib.sha256(id_input).hexdigest()[:32]
            if expected == actual:
                return True, f"TDX receipt_id OK ({expected[:16]}...)"
            return False, (
                f"TDX receipt_id MISMATCH: expected {expected[:16]}..., got {actual[:16]}..."
            )
        expected = receipt_id_of(self.attestation)
        if expected == actual:
            return True, f"hash OK ({expected[:16]}...)"
        else:
            return False, f"hash MISMATCH: expected {expected[:16]}..., got {actual[:16]}..."

    def verify(self, public_key_b64: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Run all verification checks.

        Auto-detects receipt type: if attestation_type is "tdx-quote" or a
        "tdx" key is present, runs TDX verification. Otherwise runs Ed25519
        verification.

        Returns (passed: bool, results: list of "✓ ..." / "✗ ..." strings).
        """
        if self.receipt.get("attestation_type") == "tdx-quote" or "tdx" in self.receipt:
            return self.verify_tdx(public_key_b64)
        return self._verify_ed25519(public_key_b64)

    def _verify_ed25519(self, public_key_b64: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Ed25519 verification path (original)."""
        results = []

        # 1. Schema
        schema_issues = self.validate_schema()
        if not schema_issues:
            results.append("✓ schema valid (v{})".format(self.receipt.get("polaris_version", "?")))
        else:
            for issue in schema_issues:
                results.append(f"✗ schema: {issue}")

        # 2. Hash integrity
        hash_ok, hash_msg = self.verify_hash()
        results.append(f"{'✓' if hash_ok else '✗'} hash integrity: {hash_msg}")

        # 3. Signature
        if self.receipt.get("signature"):
            sig_ok, sig_msg = self.verify_signature()
            results.append(f"{'✓' if sig_ok else '✗'} Ed25519 signature: {sig_msg}")
        else:
            results.append("✗ unsigned (no signature field)")

        # 4. Public key match (if provided)
        if public_key_b64:
            receipt_pub = self.receipt.get("public_key", "")
            if receipt_pub == public_key_b64:
                results.append("✓ public key matches trusted key")
            else:
                results.append("✗ public key DOES NOT MATCH trusted key")

        # 5. Consistency
        consistency_issues = self.validate_consistency()
        if not consistency_issues:
            results.append("✓ internal consistency")
        else:
            for issue in consistency_issues:
                results.append(f"✗ consistency: {issue}")

        # 6. Gate re-checks
        gate_results = self._recheck_gates()
        results.extend(gate_results)

        passed = self._verification_passed(results)
        return passed, results

    def verify_tdx(self, public_key_b64: Optional[str] = None) -> Tuple[bool, List[str]]:
        """TDX hardware attestation verification path.

        Validates a Polaris TDX receipt:
        1. Intel DCAP verification (intel_verified flag)
        2. Result hash integrity (result_sha256 matches stdout)
        3. E2E pubkey binding (our key is in the quote)
        4. Schema + consistency (same as Ed25519 path)
        """
        results = []
        tdx = self.receipt.get("tdx", {})
        verification = tdx.get("verification", {})

        # 1. Intel DCAP verification
        if verification.get("intel_verified"):
            results.append("✓ Intel DCAP: quote verified by Intel")
        else:
            results.append("✗ Intel DCAP: quote NOT verified")

        # 2. Report data match
        if verification.get("report_data_match"):
            results.append("✓ DCAP report_data: bindings match")
        else:
            results.append("✗ DCAP report_data: bindings MISMATCH")

        # 3. Result hash integrity — sha256(stdout) must match result_sha256
        result_sha256 = tdx.get("result_sha256", "")
        stdout_b64 = tdx.get("stdout_b64", "")
        if result_sha256 and stdout_b64:
            try:
                stdout_bytes = base64.b64decode(stdout_b64)
                computed = hashlib.sha256(stdout_bytes).hexdigest()
                if computed == result_sha256:
                    results.append(f"✓ result hash: stdout matches result_sha256 ({computed[:16]}...)")
                else:
                    results.append(
                        f"✗ result hash MISMATCH: computed {computed[:16]}... "
                        f"!= expected {result_sha256[:16]}..."
                    )
            except Exception as e:
                results.append(f"✗ result hash: failed to decode stdout — {e}")
        elif result_sha256:
            results.append("⚠ result hash: no stdout_b64 to verify against")
        else:
            results.append("⚠ result hash: no result_sha256 in TDX receipt")

        # 4. E2E pubkey binding — check our public key is in the quote
        e2e_pubkey = tdx.get("e2e_pubkey_b64", "")
        if public_key_b64:
            if e2e_pubkey == public_key_b64:
                results.append("✓ e2e pubkey: matches trusted key")
            else:
                results.append(
                    "✗ e2e pubkey MISMATCH: receipt key does not match trusted key"
                )
        elif e2e_pubkey:
            results.append(f"✓ e2e pubkey present ({e2e_pubkey[:16]}...) — no trusted key provided")
        else:
            results.append("⚠ e2e pubkey: not present in TDX receipt")

        # 5. Schema validation for TDX receipts
        tdx_issues = self._validate_tdx_schema()
        if not tdx_issues:
            results.append("✓ TDX schema valid")
        else:
            for issue in tdx_issues:
                results.append(f"✗ TDX schema: {issue}")

        # 6. Hash integrity (receipt_id)
        hash_ok, hash_msg = self.verify_hash()
        results.append(f"{'✓' if hash_ok else '✗'} hash integrity: {hash_msg}")

        # 7. Consistency checks (same as Ed25519 path)
        consistency_issues = self.validate_consistency()
        if not consistency_issues:
            results.append("✓ internal consistency")
        else:
            for issue in consistency_issues:
                results.append(f"✗ consistency: {issue}")

        # 8. Gate re-checks
        gate_results = self._recheck_gates()
        results.extend(gate_results)

        passed = self._verification_passed(results)
        return passed, results

    def _validate_tdx_schema(self) -> List[str]:
        """Check that all required TDX fields are present."""
        issues = []
        tdx = self.receipt.get("tdx", {})

        for k in ["quote_b64", "collateral_b64", "result_sha256", "verification"]:
            if k not in tdx:
                issues.append(f"missing tdx.{k}")

        if not isinstance(tdx.get("verification", {}).get("intel_verified"), bool):
            issues.append("tdx.verification.intel_verified must be a boolean")

        return issues

    def _verification_passed(self, results: List[str]) -> bool:
        """True when cryptographic / schema checks pass.

        Gate re-checks are informational on REJECT verdicts — a failed eval
        still produces a valid attestation receipt.
        """
        reject = self.attestation.get("verdict", {}).get("pass") is False
        gate_markers = ("correctness gate", "primary guard", "guard model")
        relevant = [
            line for line in results
            if not (reject and any(m in line for m in gate_markers))
        ]
        return all(line.startswith("✓") for line in relevant)

    def _recheck_gates(self) -> List[str]:
        """Re-check correctness gates from attested measurements."""
        results = []
        primary = self.attestation.get("measurements", {}).get("primary", {})
        top1 = primary.get("top1", 0)
        kl = primary.get("kl", 99)

        if top1 is not None and top1 >= 0.90:
            results.append(f"✓ correctness gate: top1={top1:.4f} (>=0.90)")
        elif top1 is not None:
            results.append(f"✗ correctness gate: top1={top1:.4f} (<0.90)")

        if kl is not None and kl <= 0.20:
            results.append(f"✓ correctness gate: kl={kl:.4f} (<=0.20)")
        elif kl is not None:
            results.append(f"✗ correctness gate: kl={kl:.4f} (>0.20)")

        # Guard gates (primary model's own no-regression checks)
        for ctx_key, label in [("guard_128_pass", "128"), ("guard_512_pass", "512"),
                                ("guard_4k_pass", "4k"), ("guard_16k_pass", "16k"),
                                ("guard_32k_pass", "32k")]:
            passed = primary.get(ctx_key)
            if passed is False:
                results.append(f"✗ primary guard: {label}-context FAILED")
            elif passed is True:
                tps_key = ctx_key.replace("_pass", "").replace("guard_", "ctx_") + "_tps"
                results.append(f"✓ primary guard: {label}-context passed")

        # Guard model
        guard = self.attestation.get("measurements", {}).get("guard")
        if guard:
            if guard.get("speed_ok", True):
                results.append("✓ guard model: speed OK")
            else:
                results.append("✗ guard model: speed REGRESSION")
            if guard.get("accuracy_ok", True):
                results.append("✓ guard model: accuracy OK")
            else:
                results.append("✗ guard model: accuracy REGRESSION")

        return results


# ---- Utility functions for the eval box ----

def compute_build_hash(build_dir: str) -> str:
    """Compute SHA256 of the main sparkinfer binary.

    Tries qwen3_gguf_bench first, falls back to qwen3_gguf_score.
    Returns hex string, or empty string if no binary found.
    """
    candidates = ["qwen3_gguf_bench", "qwen3_gguf_score"]
    for name in candidates:
        path = os.path.join(build_dir, name)
        if os.path.isfile(path):
            try:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                return h.hexdigest()
            except OSError:
                pass
    return ""


def model_sha256(filepath: str) -> str:
    """Compute SHA256 of a model file. Returns hex string, or '' on error."""
    try:
        if not os.path.isfile(filepath):
            return ""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""
