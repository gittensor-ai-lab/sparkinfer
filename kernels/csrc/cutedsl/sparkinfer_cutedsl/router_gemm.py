"""Executable CuTe DSL prototype for the MoE router projection.

Production CUDA equivalent:
`kernels/csrc/cuda/moe/router_gemm.cu`

This kernel is deliberately isolated from the runtime. It is a correctness and
benchmark target for the CuTe DSL backend, not a production replacement yet.
"""

from __future__ import annotations

from typing import Any


def _load_cutlass_dsl() -> tuple[Any, Any, Any]:
    try:
        import cutlass  # type: ignore
        import cutlass.cute as cute  # type: ignore
        from cutlass.cute.runtime import from_dlpack  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CuTe DSL is not installed. Install CUDA 13.1 support with: "
            'python3 -m pip install "nvidia-cutlass-dsl[cu13]" cuda-python'
        ) from exc
    return cutlass, cute, from_dlpack


def build_router_gemm_launcher():
    """Return a JIT launcher for `logits = input @ router_w`.

    Device layouts:
    - input:    [num_tokens, hidden_dim], fp16/bf16/fp32
    - router_w: [hidden_dim, num_experts], fp16/bf16/fp32
    - logits:   [num_tokens, num_experts], fp32

    Grid maps one CTA to one `(token, expert)` output. The implementation is a
    pure CuTe DSL scalar-dot baseline. It is intentionally simple so the first
    branch can validate DSL installation, launch, correctness, and timing before
    moving to tiled/tensor-core variants.
    """

    cutlass, cute, _ = _load_cutlass_dsl()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute

    @cute.kernel
    def _router_gemm_kernel(
        input_tensor: cute.Tensor,
        router_w_tensor: cute.Tensor,
        logits_tensor: cute.Tensor,
        hidden_dim: cutlass.Int32,
        num_experts: cutlass.Int32,
    ):
        bx, _, _ = cute.arch.block_idx()
        token = bx // num_experts
        expert = bx - token * num_experts

        acc = cutlass.Float32(0.0)
        h = cutlass.Int32(0)
        while h < hidden_dim:
            acc += input_tensor[token, h].to(cutlass.Float32) * router_w_tensor[h, expert].to(cutlass.Float32)
            h += 1

        logits_tensor[token, expert] = acc

    @cute.jit
    def _router_gemm_launch(
        input_tensor: cute.Tensor,
        router_w_tensor: cute.Tensor,
        logits_tensor: cute.Tensor,
        num_tokens: cutlass.Int32,
        hidden_dim: cutlass.Int32,
        num_experts: cutlass.Int32,
    ):
        import cutlass as _cutlass

        _cutlass.cuda.initialize_cuda_context()
        _router_gemm_kernel(input_tensor, router_w_tensor, logits_tensor, hidden_dim, num_experts).launch(
            grid=(num_tokens * num_experts, 1, 1),
            block=(1, 1, 1),
        )

    return _router_gemm_launch


def launch_router_gemm_cutedsl(
    input_array,
    router_w_array,
    logits_array,
    num_tokens: int,
    hidden_dim: int,
    num_experts: int,
):
    """Launch the compiled CuTe DSL router GEMM on DLPack-capable arrays."""

    _, _, from_dlpack = _load_cutlass_dsl()
    _, cute, _ = _load_cutlass_dsl()
    launch_jit = build_router_gemm_launcher()
    args = (
        from_dlpack(input_array),
        from_dlpack(router_w_array),
        from_dlpack(logits_array),
        num_tokens,
        hidden_dim,
        num_experts,
    )
    launcher = cute.compile(launch_jit, *args)
    launcher(*args)
