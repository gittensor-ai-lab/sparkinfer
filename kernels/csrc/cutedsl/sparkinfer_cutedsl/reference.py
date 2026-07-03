"""Small CPU references for the experimental CuTe DSL kernels.

The references use plain Python lists so they can run on a development machine
without NumPy, CUDA, or CUTLASS installed.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _as_list(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def router_gemm_reference(
    input_rows: Sequence[float],
    router_w: Sequence[float],
    num_tokens: int,
    hidden_dim: int,
    num_experts: int,
) -> list[float]:
    """Return row-major logits [num_tokens, num_experts].

    Layouts match `launch_moe_router_gemm`:
    - input_rows: [num_tokens, hidden_dim]
    - router_w:   [hidden_dim, num_experts]
    """

    x = _as_list(input_rows)
    w = _as_list(router_w)
    out = [0.0] * (num_tokens * num_experts)
    for t in range(num_tokens):
        for e in range(num_experts):
            acc = 0.0
            for h in range(hidden_dim):
                acc += x[t * hidden_dim + h] * w[h * num_experts + e]
            out[t * num_experts + e] = acc
    return out


def routed_swiglu_reference(
    input_rows: Sequence[float],
    gate_w: Sequence[float],
    up_w: Sequence[float],
    expert_ids: Sequence[int],
    num_tokens: int,
    top_k: int,
    num_experts: int,
    hidden_dim: int,
    ffn_dim: int,
) -> list[float]:
    """Return row-major h_scratch [num_tokens * top_k, ffn_dim].

    Dense reference for the first half of MoE FFN:
    `SiLU(input @ gate_w[e]) * (input @ up_w[e])`.

    Layouts:
    - input_rows: [num_tokens, hidden_dim]
    - gate_w:     [num_experts, hidden_dim, ffn_dim]
    - up_w:       [num_experts, hidden_dim, ffn_dim]
    - expert_ids: [num_tokens, top_k]
    """

    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    x = _as_list(input_rows)
    gate = _as_list(gate_w)
    up = _as_list(up_w)
    out = [0.0] * (num_tokens * top_k * ffn_dim)
    for t in range(num_tokens):
        for k in range(top_k):
            e = int(expert_ids[t * top_k + k])
            if e < 0 or e >= num_experts:
                raise ValueError(f"expert id {e} out of range [0, {num_experts})")
            ts = t * top_k + k
            for f in range(ffn_dim):
                g_acc = 0.0
                u_acc = 0.0
                for h in range(hidden_dim):
                    xv = x[t * hidden_dim + h]
                    base = (e * hidden_dim + h) * ffn_dim + f
                    g_acc += xv * gate[base]
                    u_acc += xv * up[base]
                out[ts * ffn_dim + f] = silu(g_acc) * u_acc
    return out

