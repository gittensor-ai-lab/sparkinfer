#!/usr/bin/env bash
# Turnkey accuracy gate: token-match / KL / perplexity of sparkinfer vs llama.cpp on
# the SAME GGUF (teacher-forced over a fixed text). Builds whatever is missing.
#
#   bench/scripts/accuracy.sh [--download | <model.gguf>] [--text FILE]
#
# Env overrides: MODELS_DIR, MODEL_FILE, ARCH, LLAMACPP_DIR.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"

GGUF=""; TEXT="$HERE/eval_text.txt"
while [ $# -gt 0 ]; do case "$1" in
  --download) GGUF="$MODELS_DIR/$MODEL_FILE" ;;
  --text)     shift; TEXT="$1" ;;
  -h|--help)  sed -n '2,8p' "$0"; exit 0 ;;
  *)          GGUF="$1" ;;
esac; shift; done
[ -z "$GGUF" ] && GGUF="$MODELS_DIR/$MODEL_FILE"

ARCH="$(detect_arch)"
resolve_runner "$ARCH"     # prebuilt binaries if available, else build from source
[ "$GGUF" = "$MODELS_DIR/$MODEL_FILE" ] && ensure_model
ensure_tokenizer
ensure_llamacpp "$ARCH"
[ -f "$GGUF" ] || { echo "!! GGUF not found: $GGUF"; exit 1; }

IDS="$(python3 -c "from tokenizers import Tokenizer; print(' '.join(map(str, Tokenizer.from_file('$MODELS_DIR/tokenizer.json').encode(open('$TEXT').read().strip()).ids)))")"
echo ">> eval tokens: $(echo "$IDS" | wc -w)"

echo ">> sparkinfer teacher-forced score ..."
si_run qwen3_gguf_score "$GGUF" 20 $IDS > /tmp/spark_score.txt 2>/dev/null || true
if ! grep -q "^PPL" /tmp/spark_score.txt; then   # prebuilt incompatible -> rebuild
  fallback_build "$ARCH"
  si_run qwen3_gguf_score "$GGUF" 20 $IDS > /tmp/spark_score.txt 2>/dev/null
fi

echo ">> starting llama.cpp server (reference) ..."
"$LLAMACPP_DIR/build/bin/llama-server" -m "$GGUF" -ngl 99 -c 2048 --port 8081 >/tmp/llama_srv.log 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null || true' EXIT   # reap server (frees VRAM) before exit
for _ in $(seq 1 120); do curl -s http://localhost:8081/health 2>/dev/null | grep -q '"ok"' && break; sleep 2; done

echo; echo "=== accuracy: sparkinfer vs llama.cpp ==="
python3 "$HERE/accuracy_compare.py" /tmp/spark_score.txt "$MODELS_DIR/tokenizer.json" "$TEXT"
