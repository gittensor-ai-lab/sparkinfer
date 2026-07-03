"""Executable CuTe DSL prototype for dense routed gate/up SwiGLU.

Production CUDA equivalents:
- dense path: `kernels/csrc/cuda/moe/expert_ffn.cu`
- quantized production path: `kernels/csrc/cuda/moe/expert_ffn_q4k.cu`

This prototype computes only the dense gate/up half:
`h = SiLU(input @ gate_w[e]) * (input @ up_w[e])`.
It does not replace the Q4_K/Q6_K decode kernels.
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


def build_routed_swiglu_launcher():
    """Return a JIT launcher for dense routed SwiGLU.

    Device layouts:
    - input:      [num_tokens, hidden_dim]
    - gate_w/up_w:[num_experts, hidden_dim, ffn_dim]
    - expert_ids: [num_tokens, top_k], int32
    - h_scratch:  [num_tokens * top_k, ffn_dim], fp32
    """

    cutlass, cute, _ = _load_cutlass_dsl()
    globals()["cutlass"] = cutlass
    globals()["cute"] = cute

    @cute.kernel
    def _routed_swiglu_kernel(
        input_tensor: cute.Tensor,
        gate_w_tensor: cute.Tensor,
        up_w_tensor: cute.Tensor,
        expert_ids_tensor: cute.Tensor,
        h_scratch_tensor: cute.Tensor,
        top_k: cutlass.Int32,
        hidden_dim: cutlass.Int32,
        ffn_dim: cutlass.Int32,
    ):
        bx, _, _ = cute.arch.block_idx()
        ts = bx // ffn_dim
        f = bx - ts * ffn_dim
        token = ts // top_k
        route = ts - token * top_k
        expert = expert_ids_tensor[token, route]

        g_acc = cutlass.Float32(0.0)
        u_acc = cutlass.Float32(0.0)
        h = cutlass.Int32(0)
        while h < hidden_dim:
            xv = input_tensor[token, h].to(cutlass.Float32)
            g_acc += xv * gate_w_tensor[expert, h, f].to(cutlass.Float32)
            u_acc += xv * up_w_tensor[expert, h, f].to(cutlass.Float32)
            h += 1

        h_scratch_tensor[ts, f] = (g_acc / (cutlass.Float32(1.0) + cute.exp(-g_acc))) * u_acc

    @cute.jit
    def _routed_swiglu_launch(
        input_tensor: cute.Tensor,
        gate_w_tensor: cute.Tensor,
        up_w_tensor: cute.Tensor,
        expert_ids_tensor: cute.Tensor,
        h_scratch_tensor: cute.Tensor,
        num_tokens: cutlass.Int32,
        top_k: cutlass.Int32,
        hidden_dim: cutlass.Int32,
        ffn_dim: cutlass.Int32,
    ):
        import cutlass as _cutlass

        _cutlass.cuda.initialize_cuda_context()
        _routed_swiglu_kernel(
            input_tensor,
            gate_w_tensor,
            up_w_tensor,
            expert_ids_tensor,
            h_scratch_tensor,
            top_k,
            hidden_dim,
            ffn_dim,
        ).launch(
            grid=(num_tokens * top_k * ffn_dim, 1, 1),
            block=(1, 1, 1),
        )

    return _routed_swiglu_launch


def launch_routed_swiglu_cutedsl(
    input_array,
    gate_w_array,
    up_w_array,
    expert_ids_array,
    h_scratch_array,
    num_tokens: int,
    top_k: int,
    hidden_dim: int,
    ffn_dim: int,
):
    """Launch the compiled CuTe DSL dense routed SwiGLU on DLPack arrays."""

    _, _, from_dlpack = _load_cutlass_dsl()
    _, cute, _ = _load_cutlass_dsl()
    launch_jit = build_routed_swiglu_launcher()
    args = (
        from_dlpack(input_array),
        from_dlpack(gate_w_array),
        from_dlpack(up_w_array),
        from_dlpack(expert_ids_array),
        from_dlpack(h_scratch_array),
        num_tokens,
        top_k,
        hidden_dim,
        ffn_dim,
    )
    launcher = cute.compile(launch_jit, *args)
    launcher(*args)
