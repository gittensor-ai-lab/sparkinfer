#!/usr/bin/env bash
# Orchestrate Qwen3.6 quality parity: sparkinfer first (exclusive GPU), then llama-server.
# Matches bench/quality/README.md — do not keep both engines resident on 32 GB cards.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
QUALITY="$ROOT/bench/quality"
OUT_DIR="${OUT_DIR:-/tmp/qwen36_quality}"
MODEL="${MODEL:-$ROOT/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
TOKENIZER="${TOKENIZER:-$ROOT/models/tokenizer.json}"
SPARK_BIN="${SPARK_BIN:-$ROOT/build/runtime/qwen3_gguf_generate}"
LLAMA_SERVER="${LLAMA_SERVER:-$ROOT/.llamacpp/build/bin/llama-server}"
LLAMA_URL="${LLAMA_URL:-http://127.0.0.1:8082}"
LLAMA_PORT="${LLAMA_PORT:-8082}"
TIER="${TIER:-benchmark}"
NGL="${NGL:-99}"

mkdir -p "$OUT_DIR"

die() { echo "error: $*" >&2; exit 1; }
need() { [[ -e "$1" ]] || die "missing $1"; }

need "$MODEL"
need "$QUALITY/run_quality.py"
need "$SPARK_BIN"
need "$LLAMA_SERVER"
need "$TOKENIZER"

echo "==> [1/3] sparkinfer quality ($TIER) — exclusive GPU"
python3 "$QUALITY/run_quality.py" \
  --backend sparkinfer \
  --model "$MODEL" \
  --bin "$SPARK_BIN" \
  --tokenizer "$TOKENIZER" \
  --tier "$TIER" \
  --out "$OUT_DIR/sparkinfer_${TIER}.jsonl"

echo "==> [2/3] start llama-server on :$LLAMA_PORT"
"$LLAMA_SERVER" -m "$MODEL" --port "$LLAMA_PORT" -ngl "$NGL" --host 127.0.0.1 \
  >"$OUT_DIR/llama-server.log" 2>&1 &
LLAMA_PID=$!
cleanup() {
  if kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo "==> stopping llama-server pid=$LLAMA_PID"
    kill "$LLAMA_PID" 2>/dev/null || true
    wait "$LLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Wait until /completion answers (or timeout).
for i in $(seq 1 120); do
  if curl -sf -o /dev/null -X POST "$LLAMA_URL/completion" \
      -H 'Content-Type: application/json' \
      -d '{"prompt":"ping","n_predict":1,"temperature":0}'; then
    break
  fi
  sleep 2
  if [[ "$i" -eq 120 ]]; then
    die "llama-server did not become ready; see $OUT_DIR/llama-server.log"
  fi
done

echo "==> [3/3] llama-server quality ($TIER)"
python3 "$QUALITY/run_quality.py" \
  --backend llama-server \
  --llama-url "$LLAMA_URL" \
  --tier "$TIER" \
  --out "$OUT_DIR/llama_${TIER}.jsonl"

echo "==> done. results in $OUT_DIR"
