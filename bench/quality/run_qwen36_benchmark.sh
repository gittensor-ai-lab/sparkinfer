#!/usr/bin/env bash
# Quality-parity A/B for Qwen3.6: sparkinfer first (exclusive GPU), then llama-server.
#
#   bench/quality/run_qwen36_benchmark.sh /path/to/model.gguf
#
# Env (optional): TIER OUT_DIR PORT SPARK_BIN TOKENIZER LLAMA_SERVER NGL BENCHMARKS
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-${MODEL:-}}"
if [ -z "$MODEL" ]; then
  echo "usage: $0 <model.gguf>   (or MODEL=... $0)" >&2
  exit 1
fi
[ -f "$MODEL" ] || { echo "no such model: $MODEL" >&2; exit 1; }

TIER="${TIER:-benchmark}"
OUT_DIR="${OUT_DIR:-./qwen36_quality}"
PORT="${PORT:-8082}"
SPARK_BIN="${SPARK_BIN:-$HERE/../../build/qwen3_gguf_generate}"
TOKENIZER="${TOKENIZER:-$(dirname "$MODEL")/tokenizer.json}"
LLAMA_SERVER="${LLAMA_SERVER:-$HERE/../../.llamacpp/build/bin/llama-server}"
NGL="${NGL:-99}"
BENCH_ARG=()
[ -n "${BENCHMARKS:-}" ] && BENCH_ARG=(--benchmarks "$BENCHMARKS")

mkdir -p "$OUT_DIR"
echo "model : $MODEL"
echo "tier  : $TIER"
echo "out   : $OUT_DIR"

echo
echo "=== [1/2] sparkinfer ==="
for f in "$SPARK_BIN" "$TOKENIZER"; do
  [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
python3 "$HERE/run_quality.py" --backend sparkinfer --tier "$TIER" "${BENCH_ARG[@]}" \
  --model "$MODEL" --bin "$SPARK_BIN" --tokenizer "$TOKENIZER" \
  --out "$OUT_DIR/sparkinfer.jsonl"

echo
echo "=== [2/2] llama-server ==="
[ -x "$LLAMA_SERVER" ] || { echo "missing llama-server: $LLAMA_SERVER" >&2; exit 1; }

SERVER_PID=""
cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$LLAMA_SERVER" -m "$MODEL" --port "$PORT" -ngl "$NGL" --host 127.0.0.1 \
  >"$OUT_DIR/llama-server.log" 2>&1 &
SERVER_PID=$!

printf 'waiting for llama-server on :%s' "$PORT"
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
     || curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo " up"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo
    echo "llama-server died — see $OUT_DIR/llama-server.log" >&2
    exit 1
  fi
  printf '.'
  sleep 1
done

python3 "$HERE/run_quality.py" --backend llama-server --tier "$TIER" "${BENCH_ARG[@]}" \
  --llama-url "http://127.0.0.1:$PORT" --model "$MODEL" \
  --out "$OUT_DIR/llama.jsonl"

cleanup
SERVER_PID=""

echo
echo "=== done ==="
echo "sparkinfer : $OUT_DIR/sparkinfer.jsonl"
echo "llama.cpp  : $OUT_DIR/llama.jsonl"
