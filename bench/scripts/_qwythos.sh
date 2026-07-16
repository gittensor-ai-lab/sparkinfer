#!/usr/bin/env bash
# Qwythos-9B (Qwen3.5-9B) model paths — sourced by evaluate_triple.sh and probe_max_ctx.sh.
# HF: https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF

QWYTHOS_REPO="${QWYTHOS_REPO:-empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF}"
QWYTHOS_TOK_REPO="${QWYTHOS_TOK_REPO:-Qwen/Qwen3.5-9B}"
QWYTHOS_MODELS_DIR="${QWYTHOS_MODELS_DIR:-${MODELS_DIR:-/workspace/models}35}"

# PRIMARY_QUANT: Q4_K_M (default) | Q8_0 | BF16
qwythos_quant_file() {
  local q="${PRIMARY_QUANT:-Q4_K_M}"
  case "${q^^}" in
    BF16)   echo "Qwythos-9B-Claude-Mythos-5-1M-BF16.gguf" ;;
    Q8_0)   echo "Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf" ;;
    Q4_K_M|Q4K|*) echo "Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf" ;;
  esac
}

qwythos_sha_var() {
  local q="${PRIMARY_QUANT:-Q4_K_M}"
  case "${q^^}" in
    BF16)   echo "${QWEN35_9B_BF16_SHA256:-}" ;;
    Q8_0)   echo "${QWEN35_9B_Q8_SHA256:-}" ;;
    *)      echo "${QWEN35_9B_Q4K_SHA256:-}" ;;
  esac
}
