# Experimental CuTe DSL kernels

This directory is an isolated research backend for NVIDIA CUTLASS CuTe DSL. It is
not linked into `sparkinfer_kernels`, not used by the runtime, and not part of
the default CMake build.

The goal is to port selected optimized CUDA math surfaces into CuTe DSL without
risking the production path:

- `router_gemm.py` — bf16/fp16 router projection, `logits = input @ router_w`.
- `moe_swiglu.py` — dense routed gate/up projection with SwiGLU epilogue.
- `reference.py` — small CPU references used by smoke tests and future GPU
  correctness checks.

The current Q4_K/Q6_K production MoE kernels remain handwritten CUDA because
their hot path is byte/nibble decode, `dp4a`, warp reductions, and exact GGUF
math. CuTe DSL becomes useful first for dense/repacked tensor-core paths and
future expert layouts, then it can be benchmarked against the quantized path.

## Install

CUDA 13.1 machines:

```bash
python3 -m pip install -r kernels/csrc/cutedsl/requirements-cu13.txt
```

CUDA 12.9 machines:

```bash
python3 -m pip install nvidia-cutlass-dsl
```

The official CUTLASS DSL quick start states that `nvidia-cutlass-dsl[cu13]` is
the CUDA 13.1 wheel and that Python 3.10-3.14 on Linux is supported.

## Smoke test

The local smoke test exercises the CPU references even without a CUDA toolkit:

```bash
python3 kernels/csrc/cutedsl/run_smoke.py
```

On an RTX 5090 node with the DSL installed, pass `--require-dsl` to fail if the
CuTe DSL package is missing:

```bash
python3 kernels/csrc/cutedsl/run_smoke.py --require-dsl
```

## Integration rule

Do not call these kernels from the runtime until all of the following are true:

1. DSL kernel output matches the CUDA production path within the same top-1/KL
   gates used by the evaluation bot.
2. Same-box RTX 5090 benchmarks beat or match the handwritten CUDA kernel for
   the target surface.
3. The production path keeps a build flag fallback to handwritten CUDA.

