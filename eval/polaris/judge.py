#!/usr/bin/env python3
"""Polaris judge — assembles an unsigned attestation on the eval box.

Runs on the eval box AFTER the evaluation script completes, or wraps it as a
subprocess. Collects provenance data that only the eval box can observe:
  - Model SHA256 (from the actual GGUF files on disk)
  - Build hash (SHA256 of the compiled binary)
  - GPU clock info (from nvidia-smi)
  - Git commit info

The judge does NOT sign — that happens on the bot host where the private key lives.
It prints POLARIS_ATTESTATION <json> to stdout so vast_eval.py can parse it.

Usage (wrapper mode):
  python3 eval/polaris/judge.py \
    --ref pull/279/head --ceiling 500 \
    --script bench/scripts/evaluate_dual.sh \
    --model-file /workspace/models36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
    --guard-model-file /workspace/models/Qwen3-30B-A3B-Q4_K_M.gguf \
    --build-dir /root/sparkinfer/build/runtime

Usage (post-hoc mode, reads existing RESULT_JSON from stdin):
  eval/polaris/judge.py --from-stdin \
    --model-file ... --build-dir ...

Environment variables used:
  SPARKINFER_EVAL_SEED, MODEL_SHA256, LLAMACPP_COMMIT, GPU_CLOCKS_PINNED,
  PINNED_GCLK, SPARKINFER_EVAL_MODE, SPARKINFER_DECODE_TOKENS
"""

import argparse
import json
import os
import subprocess
import sys
import time

# Allow running from repo root or eval/polaris/
try:
    from eval.polaris.receipt import (
        AttestationBuilder, compute_build_hash, model_sha256,
    )
except ImportError:
    # Running from within eval/polaris/ directory
    from receipt import (
        AttestationBuilder, compute_build_hash, model_sha256,
    )


def _git(cmd: str, cwd: str = "/root/sparkinfer") -> str:
    """Run a git command and return stripped stdout, or empty string on failure."""
    try:
        r = subprocess.run(
            ["git", "-C", cwd] + cmd.split(),
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def _nvidia_smi(field: str) -> str:
    """Query nvidia-smi for a field. Returns stripped output or ''."""
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip().split("\n")[0].strip()
    except Exception:
        return ""


def collect_metadata(args, result_json: dict) -> dict:
    """Gather all provenance data from the eval box environment.

    Returns kwargs dict for AttestationBuilder methods.
    """
    meta = {}

    # --- Code provenance ---
    repo = os.environ.get("EVAL_REPO", "https://github.com/gittensor-ai-lab/sparkinfer")
    commit = _git("rev-parse HEAD")
    scoring_commit = _git("rev-parse origin/main")
    build_hash = compute_build_hash(args.build_dir) if args.build_dir else ""

    meta["code_repo"] = repo
    meta["code_commit"] = commit
    meta["code_build_hash"] = build_hash
    meta["code_scoring_scripts_commit"] = scoring_commit

    # --- Reference pins ---
    p_model_file = args.model_file or ""
    p_model_sha = model_sha256(p_model_file) if p_model_file else ""
    g_model_file = args.guard_model_file or ""
    g_model_sha = model_sha256(g_model_file) if g_model_file else ""

    meta["model_sha256"] = os.environ.get("QWEN36_MODEL_SHA256",
                                           os.environ.get("MODEL_SHA256", p_model_sha))
    meta["model_file"] = os.path.basename(p_model_file) or "unknown"
    meta["guard_model_sha256"] = os.environ.get("MODEL_SHA256", g_model_sha)
    meta["guard_model_file"] = os.path.basename(g_model_file) or ""
    meta["llamacpp_commit"] = os.environ.get("LLAMACPP_COMMIT", "")
    meta["eval_seed"] = os.environ.get("SPARKINFER_EVAL_SEED", "")

    # --- Environment ---
    meta["eval_mode"] = os.environ.get("SPARKINFER_EVAL_MODE", "longctx")
    meta["decode_tokens"] = int(os.environ.get("SPARKINFER_DECODE_TOKENS", "128"))
    meta["gpu_name"] = _nvidia_smi("name")
    meta["gpu_arch"] = f"sm_{_nvidia_smi('compute_cap').replace('.', '')}"
    meta["cuda_version"] = os.environ.get("CUDA_VERSION", "")
    meta["driver_version"] = _nvidia_smi("driver_version")

    # Clock info (M1 reproducibility)
    clocks_pinned = os.environ.get("GPU_CLOCKS_PINNED", "0") == "1"
    pin_target = int(os.environ.get("PINNED_GCLK", "0") or "0")
    # Also read from result_json provenance
    prov_clocks = result_json.get("clocks_pinned")
    if prov_clocks is not None:
        clocks_pinned = bool(prov_clocks)
    prov_clock = result_json.get("clock_mhz")
    prov_spread = result_json.get("clock_spread_mhz", 0)
    prov_target = result_json.get("pin_target_mhz", 0)

    meta["clocks_pinned"] = clocks_pinned
    meta["clock_mhz"] = int(prov_clock) if prov_clock else 0
    meta["clock_spread_mhz"] = int(prov_spread) if prov_spread else 0
    meta["pin_target_mhz"] = int(prov_target) if prov_target else pin_target

    return meta


def run_eval(args) -> dict:
    """Run the eval script as a subprocess and parse RESULT_JSON."""
    env = os.environ.copy()

    # Ensure SI_NO_CHECKOUT is set (bot pre-checks-out)
    if "SI_NO_CHECKOUT" not in env:
        env["SI_NO_CHECKOUT"] = "1"

    cmd = ["bash", args.script, "--ref", args.ref]
    if args.ceiling:
        cmd.extend(["--ceiling", str(args.ceiling)])

    print(f">> [polaris-judge] running: {' '.join(cmd)}", file=sys.stderr)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=14400, env=env)
    elapsed = time.time() - t0
    print(f">> [polaris-judge] eval completed in {elapsed:.0f}s (rc={r.returncode})", file=sys.stderr)

    # Pass through stdout/stderr
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)

    # Parse RESULT_JSON
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            try:
                return json.loads(line[len("RESULT_JSON "):])
            except json.JSONDecodeError as e:
                print(f">> [polaris-judge] ERROR parsing RESULT_JSON: {e}", file=sys.stderr)
                return {}

    print(">> [polaris-judge] ERROR: no RESULT_JSON found in eval output", file=sys.stderr)
    return {}


def main():
    ap = argparse.ArgumentParser(description="Polaris judge — assemble an eval attestation")
    ap.add_argument("--ref", default="main", help="Git ref to evaluate")
    ap.add_argument("--ceiling", type=float, default=0, help="Roofline TPS cap")
    ap.add_argument("--script", default="bench/scripts/evaluate_dual.sh",
                    help="Eval script to run (wrapper mode)")
    ap.add_argument("--from-stdin", action="store_true",
                    help="Read RESULT_JSON from stdin instead of running eval")
    ap.add_argument("--model-file", default="",
                    help="Path to primary model GGUF (for SHA256)")
    ap.add_argument("--guard-model-file", default="",
                    help="Path to guard model GGUF (for SHA256)")
    ap.add_argument("--build-dir", default="/root/sparkinfer/build/runtime",
                    help="Directory with compiled sparkinfer binaries")
    ap.add_argument("--sparkinfer-root", default="/root/sparkinfer",
                    help="Path to sparkinfer repo checkout")
    args = ap.parse_args()

    # Allow --build-dir to override the git working directory
    git_root = args.sparkinfer_root

    # ---- Get RESULT_JSON ----
    if args.from_stdin:
        raw = sys.stdin.read()
        result_json = {}
        for line in raw.splitlines():
            if line.startswith("RESULT_JSON "):
                try:
                    result_json = json.loads(line[len("RESULT_JSON "):])
                except json.JSONDecodeError:
                    pass
        if not result_json:
            print(">> [polaris-judge] ERROR: no RESULT_JSON in stdin", file=sys.stderr)
            sys.exit(1)
    else:
        result_json = run_eval(args)
        if not result_json:
            sys.exit(1)

    # ---- Collect metadata ----
    meta = collect_metadata(args, result_json)

    # ---- Build attestation ----
    builder = AttestationBuilder()

    builder.set_code(
        repo=meta["code_repo"],
        commit=meta["code_commit"],
        build_hash=meta["code_build_hash"],
        scoring_scripts_commit=meta["code_scoring_scripts_commit"],
    )
    builder.set_references(
        model_sha256=meta["model_sha256"],
        model_file=meta["model_file"],
        guard_model_sha256=meta["guard_model_sha256"],
        guard_model_file=meta["guard_model_file"],
        llamacpp_commit=meta["llamacpp_commit"],
        eval_seed=meta["eval_seed"],
    )
    builder.set_environment(
        eval_mode=meta["eval_mode"],
        decode_tokens=meta["decode_tokens"],
        gpu_name=meta["gpu_name"],
        gpu_arch=meta["gpu_arch"],
        clocks_pinned=meta["clocks_pinned"],
        clock_mhz=meta["clock_mhz"],
        clock_spread_mhz=meta["clock_spread_mhz"],
        pin_target_mhz=meta["pin_target_mhz"],
        cuda_version=meta["cuda_version"],
        driver_version=meta["driver_version"],
    )
    builder.set_measurements(result_json)
    builder.set_verdict(result_json)
    builder.set_timestamp()

    attestation = builder.build()

    # ---- Output ----
    # Print attestation as a single line for easy parsing
    print(f"POLARIS_ATTESTATION {json.dumps(attestation, separators=(',', ':'))}")

    # Also save to a temp file for debugging
    debug_path = "/tmp/polaris_attestation.json"
    try:
        with open(debug_path, "w") as f:
            json.dump(attestation, f, indent=2)
        print(f">> [polaris-judge] attestation saved to {debug_path}", file=sys.stderr)
    except Exception:
        pass


if __name__ == "__main__":
    main()
