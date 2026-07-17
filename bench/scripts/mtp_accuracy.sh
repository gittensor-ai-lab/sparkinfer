#!/usr/bin/env bash
# MTP accuracy gate: speculative output vs AR + draft-head top-1/KL vs main model.
# Also runs trunk vs llama.cpp if llama-server is available.
#
#   bench/scripts/mtp_accuracy.sh [--download | <mtp_model.gguf>]
#
# Env: SPARKINFER_MTP=1 SPARKINFER_MTP_DRAFT_MAX=3 SPARKINFER_MTP_FAST=0
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"

GGUF=""
while [ $# -gt 0 ]; do case "$1" in
  --download) GGUF="$MODELS_DIR/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf" ;;
  -h|--help)  sed -n '2,10p' "$0"; exit 0 ;;
  *)          GGUF="$1" ;;
esac; shift; done
[ -z "$GGUF" ] && GGUF="${SPARKINFER_MTP_GGUF:-$MODELS_DIR/Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf}"

ARCH="$(detect_arch)"
resolve_runner "$ARCH"
[ -f "$GGUF" ] || { echo "!! GGUF not found: $GGUF"; exit 1; }

export SPARKINFER_MTP="${SPARKINFER_MTP:-1}"
export SPARKINFER_MTP_DRAFT_MAX="${SPARKINFER_MTP_DRAFT_MAX:-3}"
export SPARKINFER_MTP_FAST="${SPARKINFER_MTP_FAST:-0}"
export SPARKINFER_MTP_ADAPTIVE="${SPARKINFER_MTP_ADAPTIVE:-0}"

SEED="${SPARKINFER_EVAL_SEED:-fixed}"
IDS_FILE="/tmp/mtp_eval_ids.txt"
ensure_tokenizer
python3 "$HERE/gen_eval_prompt.py" "$SEED" "$MODELS_DIR/tokenizer.json" "$HERE/eval_corpus.txt" "$HERE/eval_text.txt" > "$IDS_FILE"
IDS="$(cat "$IDS_FILE")"
NIDS=$(echo "$IDS" | wc -w)
MAX_NEW="${SPARKINFER_MTP_CHECK_NEW:-64}"
echo ">> MTP check: seed=$SEED prompt_tokens=$NIDS max_new=$MAX_NEW draft_max=$SPARKINFER_MTP_DRAFT_MAX"

si_run qwen3_gguf_mtp_check "$GGUF" "$MAX_NEW" $IDS > /tmp/mtp_check.txt 2>&1 || true
if ! grep -q "^METRIC" /tmp/mtp_check.txt; then
  fallback_build "$ARCH"
  si_run qwen3_gguf_mtp_check "$GGUF" "$MAX_NEW" $IDS > /tmp/mtp_check.txt 2>&1
fi
cat /tmp/mtp_check.txt

echo
echo "=== trunk vs llama.cpp (teacher-forced, MTP GGUF) ==="
if [ -x "${LLAMACPP_DIR:-/dev/null}/build/bin/llama-server" ] || ensure_llamacpp "$ARCH" 2>/dev/null; then
  bash "$HERE/accuracy.sh" "$GGUF" 2>/dev/null | tail -8 || echo ">> llama.cpp compare skipped (server unavailable)"
else
  echo ">> llama.cpp not available — skipping trunk top1/KL vs reference"
fi

grep -q "^VERDICT PASS" /tmp/mtp_check.txt
