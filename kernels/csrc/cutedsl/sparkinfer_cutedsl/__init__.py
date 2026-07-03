"""Experimental CuTe DSL kernels for sparkinfer.

This package is intentionally not imported by the production runtime. It exists
to develop and benchmark CuTe DSL variants beside the optimized CUDA kernels.
"""

from .reference import router_gemm_reference, routed_swiglu_reference, silu

__all__ = [
    "router_gemm_reference",
    "routed_swiglu_reference",
    "silu",
]

