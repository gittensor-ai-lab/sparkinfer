#!/usr/bin/env python3
"""Smoke checks for the experimental CuTe DSL package.

This intentionally runs without CUDA by validating the CPU references and import
boundaries. Use `--require-dsl` on a GPU node to assert the NVIDIA DSL package is
installed before trying real launches.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sparkinfer_cutedsl.moe_swiglu import build_routed_swiglu_launcher
from sparkinfer_cutedsl.reference import router_gemm_reference, routed_swiglu_reference
from sparkinfer_cutedsl.router_gemm import build_router_gemm_launcher


def _assert_close(got: list[float], want: list[float], tol: float = 1e-6) -> None:
    if len(got) != len(want):
        raise AssertionError(f"length mismatch: {len(got)} != {len(want)}")
    for i, (g, w) in enumerate(zip(got, want)):
        if not math.isclose(g, w, rel_tol=tol, abs_tol=tol):
            raise AssertionError(f"mismatch at {i}: got {g}, want {w}")


def run_reference_checks() -> None:
    logits = router_gemm_reference(
        input_rows=[1.0, 2.0, 3.0, 4.0],
        router_w=[
            0.5,
            -1.0,
            1.5,
            2.0,
            -0.5,
            0.25,
        ],
        num_tokens=2,
        hidden_dim=2,
        num_experts=3,
    )
    _assert_close(logits, [4.5, -2.0, 2.0, 9.5, -5.0, 5.5])

    h = routed_swiglu_reference(
        input_rows=[1.0, 2.0],
        gate_w=[
            0.5,
            1.0,
            1.5,
            -0.5,
        ],
        up_w=[
            1.0,
            -1.0,
            0.25,
            2.0,
        ],
        expert_ids=[0],
        num_tokens=1,
        top_k=1,
        num_experts=1,
        hidden_dim=2,
        ffn_dim=2,
    )
    gate0 = 1.0 * 0.5 + 2.0 * 1.5
    up0 = 1.0 * 1.0 + 2.0 * 0.25
    gate1 = 1.0 * 1.0 + 2.0 * -0.5
    up1 = 1.0 * -1.0 + 2.0 * 2.0
    want = [gate0 / (1.0 + math.exp(-gate0)) * up0, 0.0 / (1.0 + math.exp(0.0)) * up1]
    _assert_close(h, want)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-dsl", action="store_true", help="fail if nvidia-cutlass-dsl is unavailable")
    args = parser.parse_args()

    run_reference_checks()
    print("[cutedsl] CPU reference checks passed")

    if args.require_dsl:
        build_router_gemm_launcher()
        build_routed_swiglu_launcher()
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "bench_cutedsl.py"),
                "--hidden",
                "32",
                "--experts",
                "4",
                "--ffn",
                "8",
                "--iters",
                "1",
                "--warmup",
                "1",
            ],
            check=True,
        )
        print("[cutedsl] CuTe DSL package import, JIT, GPU launch, and correctness passed")
    else:
        print("[cutedsl] skipped CuTe DSL import; pass --require-dsl on a CUDA node")


if __name__ == "__main__":
    main()
