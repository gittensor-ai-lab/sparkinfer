#!/usr/bin/env python3
"""Self-contained scoring script that runs inside a Polaris TDX enclave.

This script replicates the logic from evaluate_dual.sh lines 148-212 in pure
Python (stdlib only — no dependencies). It reads the eval result JSON from
/submission/result.json (mounted by the Polaris attest workflow), applies the
correctness gate and no-regression guard, and outputs the final verdict as JSON
to stdout.

The DCAP quote proves this exact script, with these exact inputs, produced
this exact output — hardware-attested scoring.
"""

import json
import sys

TOP1_BAR = 0.90
KL_BAR = 0.20

GUARD_CTX_KEYS = [
    "guard_128_pass",
    "guard_512_pass",
    "guard_4k_pass",
    "guard_16k_pass",
    "guard_32k_pass",
]

CTX_LABEL_MAP = {
    "guard_128_pass": "128",
    "guard_512_pass": "512",
    "guard_4k_pass": "4k",
    "guard_16k_pass": "16k",
    "guard_32k_pass": "32k",
}


def load_result(path: str = "/submission/result.json") -> dict:
    """Load the eval result JSON from the submission mount."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(json.dumps({
            "label": "REJECT",
            "pass": False,
            "reason": f"scoring: failed to load result.json: {e}",
        }))
        sys.exit(1)


def score(result: dict) -> dict:
    """Apply correctness gate + no-regression guard and return the final verdict.

    Args:
        result: The merged RESULT_JSON dict (primary + guard fields).

    Returns:
        A verdict dict with at least: label, pass, reason.
    """
    primary = result.get("primary", result)
    guard = result.get("guard") or {}

    # An empty primary is an infra failure — fail closed.
    if not primary:
        return {
            "label": "REJECT",
            "pass": False,
            "reason": "primary (Qwen3.6) eval produced no verdict (infra error)",
        }

    # ---- Correctness gate ----
    top1 = float(primary.get("top1", 0))
    kl = float(primary.get("kl", 99))
    acc_ok = top1 >= TOP1_BAR and kl <= KL_BAR

    if not acc_ok:
        reasons = []
        if top1 < TOP1_BAR:
            reasons.append(f"top1={top1:.4f} (need >= {TOP1_BAR})")
        if kl > KL_BAR:
            reasons.append(f"kl={kl:.4f} (need <= {KL_BAR})")
        return {
            "label": "REJECT",
            "pass": False,
            "reason": "correctness gate failed: " + "; ".join(reasons),
        }

    # ---- No-regression guard ----
    # Every measured context must hold >= tolerance vs main branch baseline.
    present = [k for k in GUARD_CTX_KEYS if k in guard]
    speed_ok = all(guard.get(k, True) for k in present)
    g_top1 = float(guard.get("top1", 0))
    g_kl = float(guard.get("kl", 99))
    g_acc_ok = g_top1 >= TOP1_BAR and g_kl <= KL_BAR
    guard_ok = bool(guard) and speed_ok and g_acc_ok

    regressed = [CTX_LABEL_MAP[k] for k in present if not guard.get(k, True)]

    # Build the final result — carry primary metrics verbatim, overlay guard.
    final = dict(primary)
    final["guard"] = {
        k: guard[k]
        for k in (
            "pass", "top1", "kl", "label",
            "ctx_128_tps", "ctx_512_tps", "ctx_4096_tps",
            "ctx_16384_tps", "ctx_32768_tps",
            "guard_128_pass", "guard_512_pass", "guard_4k_pass",
            "guard_16k_pass", "guard_32k_pass",
        )
        if k in guard
    }
    final["guard"]["speed_ok"] = speed_ok
    final["guard"]["accuracy_ok"] = g_acc_ok

    if not guard_ok:
        reasons = []
        if regressed:
            reasons.append(
                "Qwen3-30B decode regressed at context(s): " + ", ".join(regressed)
            )
        if not g_acc_ok:
            reasons.append(
                f"Qwen3-30B accuracy broke (top1={g_top1} need>={TOP1_BAR}, "
                f"kl={g_kl} need<={KL_BAR})"
            )
        if not bool(guard):
            reasons.append("Qwen3-30B guard produced no verdict (infra error)")
        if not reasons:
            reasons.append("Qwen3-30B no-regression guard failed")
        final["label"] = "REJECT"
        final["pass"] = False
        final["reason"] = "no-regression guard: " + "; ".join(reasons)
        final["guard_regression_labels"] = [
            "regression-qwen3-" + c for c in regressed
        ]

    return final


def main():
    result = load_result()
    verdict = score(result)
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
