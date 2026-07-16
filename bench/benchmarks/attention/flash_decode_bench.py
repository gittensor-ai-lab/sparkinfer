"""
Flash decode attention benchmark.

Measures latency and memory bandwidth utilization for single-token decode
across varying batch sizes and context lengths.

Target hardware: RTX 5090, RTX Spark, DGX Spark, Jetson Thor
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List

import torch


@dataclass
class BenchConfig:
    batch_sizes: List[int]
    context_lens: List[int]
    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: str = "bfloat16"
    warmup_iters: int = 50
    bench_iters: int = 200
    device: str = "cuda"


@dataclass
class BenchResult:
    batch_size: int
    context_len: int
    latency_ms: float
    bandwidth_gbps: float
    tokens_per_sec: float
    device_name: str
    cuda_arch: str


def get_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def run_flash_decode_bench(cfg: BenchConfig) -> List[BenchResult]:
    results = []
    dtype = get_dtype(cfg.dtype)
    device = torch.device(cfg.device)

    device_name = torch.cuda.get_device_name(0)
    cuda_arch = f"sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}"

    for bs in cfg.batch_sizes:
        for ctx in cfg.context_lens:
            q = torch.randn(bs, cfg.num_heads, cfg.head_dim, dtype=dtype, device=device)
            k = torch.randn(bs, ctx, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=device)
            v = torch.randn(bs, ctx, cfg.num_kv_heads, cfg.head_dim, dtype=dtype, device=device)

            scale = cfg.head_dim ** -0.5

            def _run():
                # GQA-style grouped repeat for baselines without native GQA
                num_repeat = cfg.num_heads // cfg.num_kv_heads
                k_exp = k.repeat_interleave(num_repeat, dim=2).transpose(1, 2)
                v_exp = v.repeat_interleave(num_repeat, dim=2).transpose(1, 2)
                scores = torch.einsum("bhd,bhsd->bhs", q, k_exp) * scale
                weights = scores.softmax(dim=-1)
                return torch.einsum("bhs,bhsd->bhd", weights, v_exp)

            # warmup
            for _ in range(cfg.warmup_iters):
                _run()
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(cfg.bench_iters):
                _run()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1e3 / cfg.bench_iters

            # KV bytes read per step (dominant cost in decode)
            kv_bytes = 2 * bs * ctx * cfg.num_kv_heads * cfg.head_dim * (2 if dtype == torch.float16 else 2)
            bandwidth_gbps = kv_bytes / (elapsed_ms * 1e-3) / 1e9

            results.append(BenchResult(
                batch_size=bs,
                context_len=ctx,
                latency_ms=elapsed_ms,
                bandwidth_gbps=bandwidth_gbps,
                tokens_per_sec=bs / (elapsed_ms * 1e-3),
                device_name=device_name,
                cuda_arch=cuda_arch,
            ))
            print(f"bs={bs:4d} ctx={ctx:6d}  {elapsed_ms:7.3f}ms  "
                  f"{bandwidth_gbps:6.1f} GB/s  {bs/(elapsed_ms*1e-3):8.0f} tok/s")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16, 64, 256])
    parser.add_argument("--context-lens", nargs="+", type=int, default=[512, 2048, 8192, 32768])
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", default="results/flash_decode.json")
    args = parser.parse_args()

    cfg = BenchConfig(
        batch_sizes=args.batch_sizes,
        context_lens=args.context_lens,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=args.dtype,
    )

    results = run_flash_decode_bench(cfg)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
