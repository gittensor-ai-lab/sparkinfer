#!/usr/bin/env bash
# Compare API decode throughput (C++ tokenizer) vs native qwen3_gguf_bench.
# Usage: bench_api_vs_native.sh [model.gguf] [api_url]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GGUF="${1:-$ROOT/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
API="${2:-http://127.0.0.1:8080}"
BENCH="${ROOT}/build/runtime/qwen3_gguf_bench"
PY="${ROOT}/server/scripts/bench_api_vs_native.py"

if [ ! -x "$BENCH" ]; then
  echo "!! missing $BENCH — build with cmake -DBUILD_SERVER=ON (and the runtime examples)" >&2
  exit 1
fi
if [ ! -f "$PY" ]; then
  echo "!! missing $PY" >&2
  exit 1
fi

exec python3 "$PY" "$ROOT" "$API" "$GGUF"
