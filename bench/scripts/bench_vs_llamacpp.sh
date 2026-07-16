#!/usr/bin/env bash
# Decode-throughput comparison: sparkinfer vs llama.cpp, same GGUF, same GPU.
#
# Measures single-stream (batch=1) generation tokens/sec on Qwen3-30B-A3B Q4_K_M.
# This is an honest baseline — sparkinfer's kernels are unoptimized (no fused
# quantized matmul, naive per-token expert dequant); the gap vs llama.cpp is the
# optimization target.
#
# Prereqs:
#   - sparkinfer superbuild built (bin/qwen3_bench) — see runtime/scripts/build.sh
#   - llama.cpp built with CUDA: cmake -B build -DGGML_CUDA=ON \
#       -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build --target llama-bench
#   - Qwen3-30B-A3B-Q4_K_M.gguf (and bf16 weights dir from convert_gguf.py, optional)
#
# Usage: bench_vs_llamacpp.sh <model.gguf> <llama-bench> <qwen3_bench> [bf16_dir]
set -euo pipefail
GGUF="${1:?path to .gguf}"; LLAMA_BENCH="${2:?path to llama-bench}"; SI_BENCH="${3:?path to qwen3_bench}"
BF16_DIR="${4:-}"
N=64

echo "================= sparkinfer (native Q4_K_M) ================="
"$SI_BENCH" "$GGUF" "$N"

if [ -n "$BF16_DIR" ]; then
  echo "================= sparkinfer (bf16) ================="
  "$SI_BENCH" "$BF16_DIR" "$N"
fi

echo "================= llama.cpp (same Q4_K_M GGUF) ================="
"$LLAMA_BENCH" -m "$GGUF" -ngl 99 -p 512 -n "$N"
