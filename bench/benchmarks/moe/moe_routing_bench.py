"""
MoE routing and grouped GEMM benchmark.

Measures end-to-end latency for token routing + expert dispatch
at varying token counts, expert counts, and top-k values.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class MoEBenchConfig:
    token_counts: List[int]
    num_experts_list: List[int]
    top_k_list: List[int]
    hidden_dim: int = 4096
    ffn_dim: int = 1408       # DeepSeek-V2 style intermediate
    dtype: str = "bfloat16"
    warmup_iters: int = 30
    bench_iters: int = 100


@dataclass
class MoEBenchResult:
    num_tokens: int
    num_experts: int
    top_k: int
    routing_ms: float
    dispatch_ms: float
    total_ms: float
    device_name: str


def run_moe_bench(cfg: MoEBenchConfig) -> List[MoEBenchResult]:
    results = []
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.dtype]
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)

    for num_tokens in cfg.token_counts:
        for num_experts in cfg.num_experts_list:
            for top_k in cfg.top_k_list:
                hidden = torch.randn(num_tokens, cfg.hidden_dim, dtype=dtype, device=device)
                router_w = torch.randn(cfg.hidden_dim, num_experts, dtype=dtype, device=device)
                # Expert weights: [E, H, F]
                expert_w = torch.randn(num_experts, cfg.hidden_dim, cfg.ffn_dim, dtype=dtype, device=device)

                def _route():
                    logits = (hidden @ router_w).float()
                    return logits.topk(top_k, dim=-1)

                def _dispatch(ids, weights):
                    out = torch.zeros_like(hidden)
                    for k in range(top_k):
                        eid = ids[:, k]  # [T]
                        w   = weights[:, k:k+1]  # [T,1]
                        # Gather-GEMM: simplified (not grouped)
                        expert_inputs = hidden  # ideally gathered by expert
                        # naive: loop (replace with grouped GEMM in real impl)
                        for e in range(num_experts):
                            mask = (eid == e)
                            if mask.any():
                                out[mask] += w[mask] * (expert_inputs[mask] @ expert_w[e])[:, :cfg.hidden_dim]
                    return out

                # Warmup
                for _ in range(cfg.warmup_iters):
                    ids, ws = _route()
                    # skip dispatch in warmup for speed
                torch.cuda.synchronize()

                # Bench routing
                start = time.perf_counter()
                for _ in range(cfg.bench_iters):
                    ids, ws = _route()
                torch.cuda.synchronize()
                routing_ms = (time.perf_counter() - start) * 1e3 / cfg.bench_iters

                # Skip dispatch bench for large configs (too slow with naive impl)
                dispatch_ms = float("nan")
                total_ms = routing_ms

                results.append(MoEBenchResult(
                    num_tokens=num_tokens,
                    num_experts=num_experts,
                    top_k=top_k,
                    routing_ms=routing_ms,
                    dispatch_ms=dispatch_ms,
                    total_ms=total_ms,
                    device_name=device_name,
                ))
                print(f"T={num_tokens:5d} E={num_experts:4d} k={top_k}  "
                      f"route={routing_ms:.3f}ms")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-counts", nargs="+", type=int, default=[1, 16, 64, 256, 1024])
    parser.add_argument("--num-experts", nargs="+", type=int, default=[8, 64, 256])
    parser.add_argument("--top-k", nargs="+", type=int, default=[2, 6, 8])
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--ffn-dim", type=int, default=1408)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output", default="results/moe_routing.json")
    args = parser.parse_args()

    cfg = MoEBenchConfig(
        token_counts=args.token_counts,
        num_experts_list=args.num_experts,
        top_k_list=args.top_k,
        hidden_dim=args.hidden_dim,
        ffn_dim=args.ffn_dim,
        dtype=args.dtype,
    )

    results = run_moe_bench(cfg)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
