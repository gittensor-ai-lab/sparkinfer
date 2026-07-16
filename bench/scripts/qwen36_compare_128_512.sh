#!/usr/bin/env bash
# Same-GGUF Qwen3.6 decode comparison: sparkinfer vs llama.cpp at 128 and 512
# generated tokens, batch size 1.
#
# Usage:
#   bench/scripts/qwen36_compare_128_512.sh [model.gguf]
#
# Defaults to the Unsloth Qwen3.6 35B-A3B Q4_K_M GGUF:
#   unsloth/Qwen3.6-35B-A3B-GGUF / Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
set -euo pipefail

export MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
export MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
export TOK_REPO="${TOK_REPO:-Qwen/Qwen3.6-35B-A3B}"

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

GGUF="${1:-$MODELS_DIR/$MODEL_FILE}"
ARCH="$(detect_arch)"

echo ">> GPU arch: sm_$ARCH"
resolve_runner "$ARCH"
if [ "$GGUF" = "$MODELS_DIR/$MODEL_FILE" ]; then
  ensure_model
else
  [ -f "$GGUF" ] || { echo "!! GGUF not found: $GGUF"; exit 1; }
fi
ensure_llamacpp "$ARCH"

echo ">> model: $GGUF"
echo ">> sparkinfer commit: $(git -C "$ROOT" rev-parse --short HEAD)"
echo ">> llama.cpp commit: $(git -C "$LLAMACPP_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"

for TOKENS in 128 512; do
  echo
  echo "================= sparkinfer Qwen3.6 (n=$TOKENS, bs=1) ================="
  si_run qwen3_gguf_bench "$GGUF" "$TOKENS"

  echo
  echo "================= llama.cpp Qwen3.6 (n=$TOKENS, bs=1) ================="
  "$LLAMACPP_DIR/build/bin/llama-bench" -m "$GGUF" -p 0 -n "$TOKENS" -ngl 99
done
