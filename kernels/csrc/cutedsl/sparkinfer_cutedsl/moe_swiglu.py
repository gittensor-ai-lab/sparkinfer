"""CuTe DSL prototype for dense routed gate/up SwiGLU.

Production CUDA equivalents:
- dense path: `kernels/csrc/cuda/moe/expert_ffn.cu`
- quantized production path: `kernels/csrc/cuda/moe/expert_ffn_q4k.cu`

This prototype covers the dense bf16/fp16 gate/up half:
`h = SiLU(input @ gate_w[e]) * (input @ up_w[e])`.
It deliberately does not replace the Q4_K/Q6_K decode kernels.
"""

from __future__ import annotations

from typing import Any


THREADS = 128


def _load_cutlass_dsl() -> tuple[Any, Any]:
    try:
        import cutlass  # type: ignore
        import cutlass.cute as cute  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CuTe DSL is not installed. Install CUDA 13.1 support with: "
            'python3 -m pip install "nvidia-cutlass-dsl[cu13]" cuda-python'
        ) from exc
    return cutlass, cute


def build_routed_swiglu_kernel():
    """Create the JIT kernel for dense routed gate/up SwiGLU.

    Layouts:
    - input:      [num_tokens, hidden_dim]
    - gate_w/up_w:[num_experts, hidden_dim, ffn_dim]
    - expert_ids: [num_tokens, top_k]
    - h_scratch:  [num_tokens * top_k, ffn_dim], fp32

    Grid maps one CTA to one `(token, route, ffn_column)` dot pair. This is the
    smallest useful CuTe DSL port of sparkinfer's MoE work and a stepping stone
    to grouped tensor-core GEMM with a fused SwiGLU epilogue.
    """

    cutlass, cute = _load_cutlass_dsl()

    @cute.kernel
    def _routed_swiglu_kernel(
        input_ptr,
        gate_w_ptr,
        up_w_ptr,
        expert_ids_ptr,
        h_scratch_ptr,
        num_tokens: cutlass.Int32,
        top_k: cutlass.Int32,
        hidden_dim: cutlass.Int32,
        ffn_dim: cutlass.Int32,
    ):
        bx, by, _ = cute.arch.block_idx()
        tx, _, _ = cute.arch.thread_idx()
        ffn_col = bx
        ts = by
        token = ts // top_k
        route = ts - token * top_k
        expert = expert_ids_ptr[token * top_k + route]

        g_acc = cutlass.Float32(0.0)
        u_acc = cutlass.Float32(0.0)
        h = tx
        while h < hidden_dim:
            xv = cutlass.Float32(input_ptr[token * hidden_dim + h])
            w_base = (expert * hidden_dim + h) * ffn_dim + ffn_col
            g_acc += xv * cutlass.Float32(gate_w_ptr[w_base])
            u_acc += xv * cutlass.Float32(up_w_ptr[w_base])
            h += THREADS

        smem = cute.utils.SmemAllocator()
        g_part = smem.allocate(cutlass.Float32, THREADS)
        u_part = smem.allocate(cutlass.Float32, THREADS)
        g_part[tx] = g_acc
        u_part[tx] = u_acc
        cute.arch.sync_threads()

        stride = THREADS // 2
        while stride > 0:
            if tx < stride:
                g_part[tx] += g_part[tx + stride]
                u_part[tx] += u_part[tx + stride]
            cute.arch.sync_threads()
            stride //= 2

        if tx == 0 and token < num_tokens and ffn_col < ffn_dim:
            gate = g_part[0]
            up = u_part[0]
            h_scratch_ptr[ts * ffn_dim + ffn_col] = (gate / (cutlass.Float32(1.0) + cute.exp(-gate))) * up

    return _routed_swiglu_kernel


def launch_routed_swiglu_cutedsl(
    input_ptr,
    gate_w_ptr,
    up_w_ptr,
    expert_ids_ptr,
    h_scratch_ptr,
    num_tokens: int,
    top_k: int,
    hidden_dim: int,
    ffn_dim: int,
    stream=None,
):
    """Launch the experimental dense routed SwiGLU kernel."""

    kernel = build_routed_swiglu_kernel()
    kernel(
        input_ptr,
        gate_w_ptr,
        up_w_ptr,
        expert_ids_ptr,
        h_scratch_ptr,
        num_tokens,
        top_k,
        hidden_dim,
        ffn_dim,
    ).launch(
        grid=[ffn_dim, num_tokens * top_k, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )

