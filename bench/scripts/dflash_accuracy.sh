#!/usr/bin/env bash
# DFlash accuracy gate: greedy DFlash output must match AR (SPEC_AGREE = 100%).
#
#   bench/scripts/dflash_accuracy.sh <target.gguf> <draft_dir>
#
# Env: SPARKINFER_DFLASH_CHECK_NEW (default 32)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"

GGUF="${1:-}"
DRAFT="${2:-}"
if [ -z "$GGUF" ] || [ -z "$DRAFT" ]; then
  echo "usage: $0 <target.gguf> <draft_dir>"
  exit 2
fi
[ -f "$GGUF" ] || { echo "!! GGUF not found: $GGUF"; exit 1; }
[ -d "$DRAFT" ] || { echo "!! draft dir not found: $DRAFT"; exit 1; }

ARCH="$(detect_arch)"
resolve_runner "$ARCH"

SEED="${SPARKINFER_EVAL_SEED:-fixed}"
IDS_FILE="/tmp/dflash_eval_ids.txt"
ensure_tokenizer
python3 "$HERE/gen_eval_prompt.py" "$SEED" "$MODELS_DIR/tokenizer.json" \
  "$HERE/eval_corpus.txt" "$HERE/eval_text.txt" > "$IDS_FILE"
IDS="$(cat "$IDS_FILE")"
NIDS=$(echo "$IDS" | wc -w)
MAX_NEW="${SPARKINFER_DFLASH_CHECK_NEW:-32}"
echo ">> DFlash check: seed=$SEED prompt_tokens=$NIDS max_new=$MAX_NEW"

si_run qwen3_gguf_dflash_check "$GGUF" "$DRAFT" "$MAX_NEW" $IDS > /tmp/dflash_check.txt 2>&1 || true
if ! grep -q "^METRIC" /tmp/dflash_check.txt; then
  fallback_build "$ARCH"
  si_run qwen3_gguf_dflash_check "$GGUF" "$DRAFT" "$MAX_NEW" $IDS > /tmp/dflash_check.txt 2>&1
fi
cat /tmp/dflash_check.txt
grep -q "^VERDICT PASS" /tmp/dflash_check.txt
