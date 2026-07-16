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

# H1: the prompt is held-out / fuzzed by EVAL_SEED (set by the eval bot to a fresh, unpredictable
# value each run) so a submission can't overfit the in-repo text. seed="fixed" (the default for a
# manual run) reproduces the legacy fixed prompt. The exact ids are written once and fed to BOTH
# sparkinfer and llama, so they score the identical sequence; the seed is logged for reproduction.
SEED="${SPARKINFER_EVAL_SEED:-fixed}"
IDS_FILE="/tmp/eval_ids.txt"
python3 "$HERE/gen_eval_prompt.py" "$SEED" "$MODELS_DIR/tokenizer.json" "$HERE/eval_corpus.txt" "$TEXT" > "$IDS_FILE"
IDS="$(cat "$IDS_FILE")"
echo ">> eval prompt: seed=$SEED tokens=$(echo "$IDS" | wc -w)"

echo ">> sparkinfer teacher-forced score ..."
# Dump top-128 (>= the llama top-k queried in accuracy_compare). A shallow dump made the KL a
# truncation artifact: any llama-tail token outside sparkinfer's dump was floored (exp(-20)) and
# massively over-penalized, inflating KL to 0.14-0.33 on flat distributions. With the dump covering
# llama's query, KL reflects the true ~0.01-0.03 divergence. Scoring-only — no decode-speed impact.
si_run qwen3_gguf_score "$GGUF" 128 $IDS > /tmp/spark_score.txt 2>/dev/null || true
if ! grep -q "^PPL" /tmp/spark_score.txt; then   # prebuilt incompatible -> rebuild
  fallback_build "$ARCH"
  si_run qwen3_gguf_score "$GGUF" 128 $IDS > /tmp/spark_score.txt 2>/dev/null
fi

echo ">> starting llama.cpp server (reference) ..."
"$LLAMACPP_DIR/build/bin/llama-server" -m "$GGUF" -ngl 99 -c 2048 --port 8081 --no-jinja >/tmp/llama_srv.log 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null || true' EXIT   # reap server (frees VRAM) before exit
for _ in $(seq 1 120); do curl -s http://localhost:8081/health 2>/dev/null | grep -q '"ok"' && break; sleep 2; done

echo; echo "=== accuracy: sparkinfer vs llama.cpp ==="
python3 "$HERE/accuracy_compare.py" /tmp/spark_score.txt "$MODELS_DIR/tokenizer.json" "$IDS_FILE"
