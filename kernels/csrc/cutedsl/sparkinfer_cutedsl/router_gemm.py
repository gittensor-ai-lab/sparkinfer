"""CuTe DSL prototype for the MoE router projection.

Production CUDA equivalent:
`kernels/csrc/cuda/moe/router_gemm.cu`

This file is intentionally experimental. It is import-safe without CUTLASS
installed; the DSL symbols are loaded only when `launch_router_gemm_cutedsl` is
called.
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


def build_router_gemm_kernel():
    """Create the JIT launcher for `logits = input @ router_w`.

    Expected device layouts:
    - input:    [num_tokens, hidden_dim], bf16/fp16
    - router_w: [hidden_dim, num_experts], bf16/fp16
    - logits:   [num_tokens, num_experts], fp32

    Grid maps one CTA to one `(token, expert)` dot product. This mirrors the
    production bs=1/decode router shape and gives a simple first CuTe DSL target
    before moving to tensor-core grouped GEMM.
    """

    cutlass, cute = _load_cutlass_dsl()

    @cute.kernel
    def _router_gemm_kernel(
        input_ptr,
        router_w_ptr,
        logits_ptr,
        num_tokens: cutlass.Int32,
        hidden_dim: cutlass.Int32,
        num_experts: cutlass.Int32,
    ):
        bx, by, _ = cute.arch.block_idx()
        tx, _, _ = cute.arch.thread_idx()
        expert = bx
        token = by

        acc = cutlass.Float32(0.0)
        h = tx
        while h < hidden_dim:
            xv = cutlass.Float32(input_ptr[token * hidden_dim + h])
            wv = cutlass.Float32(router_w_ptr[h * num_experts + expert])
            acc += xv * wv
            h += THREADS

        # Tree reduce through shared memory. This is intentionally SIMT/simple;
        # tensor-core CuTe comes next for grouped dense experts.
        smem = cute.utils.SmemAllocator()
        partial = smem.allocate(cutlass.Float32, THREADS)
        partial[tx] = acc
        cute.arch.sync_threads()

        stride = THREADS // 2
        while stride > 0:
            if tx < stride:
                partial[tx] += partial[tx + stride]
            cute.arch.sync_threads()
            stride //= 2

        if tx == 0 and token < num_tokens and expert < num_experts:
            logits_ptr[token * num_experts + expert] = partial[0]

    return _router_gemm_kernel


def launch_router_gemm_cutedsl(
    input_ptr,
    router_w_ptr,
    logits_ptr,
    num_tokens: int,
    hidden_dim: int,
    num_experts: int,
    stream=None,
):
    """Launch the experimental CuTe DSL router GEMM.

    This wrapper accepts CuTe/CUDA-compatible device pointers or tensors as
    expected by the installed CUTLASS DSL runtime. It is not called by
    sparkinfer's production runtime.
    """

    kernel = build_router_gemm_kernel()
    kernel(input_ptr, router_w_ptr, logits_ptr, num_tokens, hidden_dim, num_experts).launch(
        grid=[num_experts, num_tokens, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )

