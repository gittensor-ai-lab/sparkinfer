#!/usr/bin/env python3
"""Polaris receipt verifier — standalone CLI. No GPU required.

Verifies a Polaris receipt JSON file: checks the schema, hash integrity,
Ed25519 signature (software receipts) or Intel DCAP attestation (TDX receipts),
correctness gates, guard gates, and internal consistency.

Usage:
  python3 eval/polaris/verify.py receipt.json
  python3 eval/polaris/verify.py receipt.json --public-key <base64-key>
  python3 eval/polaris/verify.py receipt.json --strict

Exit code: 0 = verified, 1 = rejected.
"""

import argparse
import base64
import json
import os
import sys

# Allow running from repo root or eval/polaris/
try:
    from eval.polaris.receipt import ReceiptValidator
except ImportError:
    from receipt import ReceiptValidator


def load_trusted_key(filepath: str) -> str:
    """Load a trusted public key from a file.

    File format: one line with base64-encoded 32-byte Ed25519 public key.
    Lines starting with # are comments.
    """
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Validate it's valid base64
                base64.b64decode(line)
                return line
    except Exception as e:
        print(f"error reading trusted key file: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"error: no key found in {filepath}", file=sys.stderr)
    sys.exit(2)


def load_trusted_keys(filepath: str) -> list:
    """Load a JSON array of trusted public keys."""
    try:
        with open(filepath) as f:
            keys = json.load(f)
        if not isinstance(keys, list):
            print("error: trusted keys file must contain a JSON array", file=sys.stderr)
            sys.exit(2)
        # Validate each is valid base64
        for k in keys:
            base64.b64decode(k)
        return keys
    except json.JSONDecodeError as e:
        print(f"error: invalid JSON in trusted keys file: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"error reading trusted keys file: {e}", file=sys.stderr)
        sys.exit(2)


def verify_strict(receipt: dict, validator: ReceiptValidator) -> list:
    """Additional strict-mode checks beyond the basic verification."""
    issues = []
    a = receipt.get("attestation", {})
    e = a.get("environment", {})
    r = a.get("references", {})

    # Clock must be pinned
    if not e.get("clocks_pinned"):
        issues.append("✗ strict: clock was NOT pinned (clocks_pinned=false)")

    # Model SHA256 must be present
    if not r.get("model_sha256"):
        issues.append("✗ strict: model_sha256 is missing")

    # llama.cpp commit must be pinned
    if not r.get("llamacpp_commit"):
        issues.append("✗ strict: llamacpp_commit is not pinned")

    # Build hash must be present
    if not a.get("code", {}).get("build_hash"):
        issues.append("✗ strict: build_hash is missing")

    # Eval seed must be present (H1)
    if not r.get("eval_seed"):
        issues.append("✗ strict: eval_seed is missing (H1 held-out prompt)")

    return issues


def print_receipt_summary(receipt: dict):
    """Print a human-readable summary of the attested data."""
    a = receipt.get("attestation", {})
    c = a.get("code", {})
    r = a.get("references", {})
    e = a.get("environment", {})
    m = a.get("measurements", {})
    v = a.get("verdict", {})
    primary = m.get("primary", {})
    guard = m.get("guard", {})

    print()
    print("=== Attested Measurements ===")
    print(f"Commit:      {c.get('commit', '?')[:12]}...")
    print(f"Model:       {r.get('model_file', '?')} (sha: {r.get('model_sha256', '?')[:12]}...)")
    print(f"Eval seed:   {r.get('eval_seed', '?')[:16]}...")
    build = c.get("build_hash", "")
    print(f"Build hash:  {build[:12]}..." if build else "Build hash:  (missing)")
    clock_info = f"{e.get('clock_mhz', '?')} MHz"
    if e.get("clocks_pinned"):
        clock_info += f" (pinned, target {e.get('pin_target_mhz', '?')} MHz)"
    else:
        clock_info += " (NOT pinned)"
    print(f"GPU:         {e.get('gpu_name', '?')} ({e.get('gpu_arch', '?')})")
    print(f"Clock:       {clock_info}")
    print(f"Mode:        {e.get('eval_mode', '?')} · {e.get('decode_tokens', '?')} tokens")
    print()

    # Per-context TPS
    for label, key in [("128-token", "ctx_128_tps"), ("512", "ctx_512_tps"),
                        ("4k", "ctx_4096_tps"), ("16k", "ctx_16384_tps"),
                        ("32k", "ctx_32768_tps")]:
        tps = primary.get(key)
        if tps:
            print(f"TPS ({label:>8}):  {tps:.2f}" if isinstance(tps, (int, float)) else f"TPS ({label:>8}):  {tps}")

    print(f"Top-1:       {primary.get('top1', '?')}")
    print(f"KL:          {primary.get('kl', '?')}")
    print()

    # Verdict
    label = v.get("label", "?")
    pct = v.get("pct_over_frontier", 0) or 0
    print(f"Verdict:     eval:{label}  ({pct:+.1f}% over same-box main)")
    tps_val = v.get("tps", 0)
    print(f"Scored TPS:  {tps_val:.2f}" if isinstance(tps_val, (int, float)) else f"Scored TPS:  {tps_val}")
    sc = v.get("score_context", "")
    best = v.get("best_context_label", "")
    if sc and best:
        print(f"Context:     {sc}-ctx · best: {best}")
    if v.get("reason"):
        print(f"Reason:      {v.get('reason')}")

    # Guard model
    if guard:
        print()
        gspeed = "✓" if guard.get("speed_ok", True) else "✗"
        gacc = "✓" if guard.get("accuracy_ok", True) else "✗"
        print(f"Guard model: {guard.get('model', '?')}")
        print(f"  Speed:     {gspeed}  Accuracy: {gacc}")
        for label, key in [("128", "ctx_128_tps"), ("512", "ctx_512_tps"),
                            ("4k", "ctx_4096_tps"), ("16k", "ctx_16384_tps"),
                            ("32k", "ctx_32768_tps")]:
            tps = guard.get(key)
            if tps:
                print(f"  TPS ({label:>3}):    {tps:.2f}" if isinstance(tps, (int, float)) else f"  TPS ({label:>3}):    {tps}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Polaris receipt verifier — check an eval receipt without a GPU"
    )
    ap.add_argument("receipt", help="Path to receipt JSON file")
    ap.add_argument("--public-key", "-k", default="",
                    help="Base64-encoded Ed25519 public key to trust")
    ap.add_argument("--trusted-keys", default="",
                    help="JSON file with array of trusted public keys")
    ap.add_argument("--pubkey-file", default="",
                    help="File containing a trusted public key (one line, base64)")
    ap.add_argument("--strict", action="store_true",
                    help="Require pinned clocks, model SHA, and other strict checks")
    ap.add_argument("--summary", action="store_true",
                    help="Print a human-readable summary of the attested data")
    args = ap.parse_args()

    # ---- Load receipt ----
    try:
        with open(args.receipt) as f:
            receipt = json.load(f)
    except FileNotFoundError:
        print(f"error: file not found: {args.receipt}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"error: invalid JSON: {e}", file=sys.stderr)
        sys.exit(2)

    # ---- Determine trusted key(s) ----
    trusted_key = ""
    trusted_key_set = set()

    if args.public_key:
        trusted_key = args.public_key
        trusted_key_set.add(args.public_key)
    if args.pubkey_file:
        k = load_trusted_key(args.pubkey_file)
        if not trusted_key:
            trusted_key = k
        trusted_key_set.add(k)
    if args.trusted_keys:
        for k in load_trusted_keys(args.trusted_keys):
            trusted_key_set.add(k)
            if not trusted_key:
                trusted_key = k

    # ---- Verify ----
    validator = ReceiptValidator(receipt)
    passed, results = validator.verify(public_key_b64=trusted_key)

    is_tdx = receipt.get("attestation_type") == "tdx-quote" or "tdx" in receipt

    # Strict checks
    if args.strict:
        strict_issues = verify_strict(receipt, validator)
        results.extend(strict_issues)
        if any(line.startswith("✗") for line in strict_issues):
            passed = False

    # Trusted key set check (Ed25519: public_key; TDX: e2e_pubkey_b64)
    if trusted_key_set:
        if is_tdx:
            receipt_pub = receipt.get("tdx", {}).get("e2e_pubkey_b64", "")
        else:
            receipt_pub = receipt.get("public_key", "")
        if receipt_pub and receipt_pub not in trusted_key_set:
            results.append("✗ public key NOT in trusted key set")
            passed = False

    # ---- Output ----
    print()
    rid = receipt.get("receipt_id", "?")[:16]
    ts = receipt.get("attestation", {}).get("timestamp_utc", "?")

    if is_tdx:
        tdx = receipt.get("tdx", {})
        intel_ok = tdx.get("verification", {}).get("intel_verified", False)
        e2e_pub = (tdx.get("e2e_pubkey_b64", "") or "")[:16]
        print(f"=== Polaris TDX Receipt Verification ===")
        print(f"Receipt ID:  {rid}...")
        print(f"Intel DCAP:  {'✓ verified' if intel_ok else '✗ NOT verified'}")
        print(f"E2E pubkey:  {e2e_pub}..." if e2e_pub else f"E2E pubkey:  (not set)")
        print(f"Signed at:   {ts}")
    else:
        pub_short = receipt.get("public_key", "?")[:16]
        print(f"=== Polaris Receipt Verification ===")
        print(f"Receipt ID:  {rid}...")
        print(f"Public key:  {pub_short}...")
        print(f"Signed at:   {ts}")
    print()

    for line in results:
        print(line)

    if args.summary:
        print_receipt_summary(receipt)

    print()
    if passed:
        attest_type = "TDX Intel DCAP" if is_tdx else "Ed25519"
        print(f"=== Status: {attest_type} VERIFIED — {len([r for r in results if r.startswith('✓')])}/{len(results)} checks passed ===")
        sys.exit(0)
    else:
        failed = len([r for r in results if r.startswith("✗")])
        print(f"=== Status: REJECTED — {failed} check(s) failed ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
