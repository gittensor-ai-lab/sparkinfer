#!/usr/bin/env python3
"""Correctness and benchmark runner for experimental CuTe DSL kernels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cupy as cp
import numpy as np
from cutlass.cute.runtime import from_dlpack

from sparkinfer_cutedsl.moe_swiglu import build_routed_swiglu_launcher
from sparkinfer_cutedsl.router_gemm import build_router_gemm_launcher
import cutlass.cute as cute


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    cp.cuda.runtime.deviceSynchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / iters


def _max_abs(a: cp.ndarray, b: cp.ndarray) -> float:
    return float(cp.max(cp.abs(a - b)).get())


def bench_router(args) -> dict:
    rng = cp.random.default_rng(args.seed)
    x = rng.standard_normal((args.tokens, args.hidden), dtype=cp.float32) * args.input_scale
    w = rng.standard_normal((args.hidden, args.experts), dtype=cp.float32) * args.weight_scale
    out = cp.zeros((args.tokens, args.experts), dtype=cp.float32)
    ref = x @ w

    x_t = from_dlpack(x)
    w_t = from_dlpack(w)
    out_t = from_dlpack(out)
    launch_jit = build_router_gemm_launcher()
    launcher = cute.compile(launch_jit, x_t, w_t, out_t, args.tokens, args.hidden, args.experts)

    def run():
        launcher(x_t, w_t, out_t, args.tokens, args.hidden, args.experts)

    run()
    cp.cuda.runtime.deviceSynchronize()
    err = _max_abs(out, ref)
    ms = _time_ms(run, args.warmup, args.iters)
    dots = args.tokens * args.experts
    flops = 2.0 * dots * args.hidden
    return {
        "kernel": "router_gemm_cutedsl",
        "tokens": args.tokens,
        "hidden": args.hidden,
        "experts": args.experts,
        "time_ms": ms,
        "gflop_s": flops / (ms * 1.0e6),
        "max_abs_err": err,
        "correct": bool(math.isfinite(err) and err <= args.tolerance),
    }


def bench_swiglu(args) -> dict:
    rng = cp.random.default_rng(args.seed + 1)
    x = rng.standard_normal((args.tokens, args.hidden), dtype=cp.float32) * args.input_scale
    gate = rng.standard_normal((args.experts, args.hidden, args.ffn), dtype=cp.float32) * args.weight_scale
    up = rng.standard_normal((args.experts, args.hidden, args.ffn), dtype=cp.float32) * args.weight_scale
    expert_ids = cp.arange(args.tokens * args.top_k, dtype=cp.int32).reshape(args.tokens, args.top_k) % args.experts
    out = cp.zeros((args.tokens * args.top_k, args.ffn), dtype=cp.float32)

    # Small reference on CPU to avoid invoking non-DSL GPU kernels for correctness.
    xh = cp.asnumpy(x)
    gh = cp.asnumpy(gate)
    uh = cp.asnumpy(up)
    eh = cp.asnumpy(expert_ids)
    ref = np.empty((args.tokens * args.top_k, args.ffn), dtype=np.float32)
    for t in range(args.tokens):
        for k in range(args.top_k):
            e = int(eh[t, k])
            g = xh[t] @ gh[e]
            u = xh[t] @ uh[e]
            ref[t * args.top_k + k] = (g / (1.0 + np.exp(-g))) * u
    ref_d = cp.asarray(ref)

    x_t = from_dlpack(x)
    gate_t = from_dlpack(gate)
    up_t = from_dlpack(up)
    expert_t = from_dlpack(expert_ids)
    out_t = from_dlpack(out)
    launch_jit = build_routed_swiglu_launcher()
    launcher = cute.compile(
        launch_jit,
        x_t,
        gate_t,
        up_t,
        expert_t,
        out_t,
        args.tokens,
        args.top_k,
        args.hidden,
        args.ffn,
    )

    def run():
        launcher(x_t, gate_t, up_t, expert_t, out_t, args.tokens, args.top_k, args.hidden, args.ffn)

    run()
    cp.cuda.runtime.deviceSynchronize()
    err = _max_abs(out, ref_d)
    ms = _time_ms(run, args.warmup, args.iters)
    outputs = args.tokens * args.top_k * args.ffn
    flops = 4.0 * outputs * args.hidden
    return {
        "kernel": "routed_swiglu_cutedsl",
        "tokens": args.tokens,
        "top_k": args.top_k,
        "hidden": args.hidden,
        "ffn": args.ffn,
        "experts": args.experts,
        "time_ms": ms,
        "gflop_s": flops / (ms * 1.0e6),
        "max_abs_err": err,
        "correct": bool(math.isfinite(err) and err <= args.tolerance),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ffn", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-scale", type=float, default=0.02)
    parser.add_argument("--weight-scale", type=float, default=0.02)
    parser.add_argument("--tolerance", type=float, default=2e-3)
    parser.add_argument("--output", type=Path, help="optional path to write benchmark JSON")
    args = parser.parse_args()

    if args.tokens * args.top_k * args.ffn * args.hidden > 50_000_000:
        print("[cutedsl] refusing very large scalar baseline benchmark; lower --ffn/--hidden", file=sys.stderr)
        raise SystemExit(2)

    results = [bench_router(args), bench_swiglu(args)]
    payload = {
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8"),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "shape": {
            "tokens": args.tokens,
            "hidden": args.hidden,
            "experts": args.experts,
            "top_k": args.top_k,
            "ffn": args.ffn,
        },
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    for r in results:
        if not r["correct"]:
            raise SystemExit(f"{r['kernel']} correctness failed: max_abs_err={r['max_abs_err']}")


if __name__ == "__main__":
    main()
