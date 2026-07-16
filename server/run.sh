#!/usr/bin/env bash
# Start sparkinfer-server with Qwen3.6-35B-A3B (SOTA) defaults.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SPARKINFER_ROOT="$ROOT"
# Server default model — override via env if needed.
export MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
export MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
export MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
export TOK_REPO="${TOK_REPO:-Qwen/Qwen3.6-35B-A3B}"
export MODEL_NAME="${MODEL_NAME:-qwen3.6-35b-a3b}"
# shellcheck source=../bench/scripts/_common.sh
source "$ROOT/bench/scripts/_common.sh"

ARCH="$(detect_arch)"
GGUF="${1:-$MODELS_DIR/$MODEL_FILE}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-36864}"

while [ $# -gt 0 ]; do
  case "$1" in
    --download) GGUF="$MODELS_DIR/$MODEL_FILE" ;;
    --port) PORT="$2"; shift ;;
    --host) HOST="$2"; shift ;;
    --ctx) CTX="$2"; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) GGUF="$1" ;;
  esac
  shift
done

[ "$GGUF" = "$MODELS_DIR/$MODEL_FILE" ] && ensure_model
ensure_tokenizer
ensure_sparkinfer "$ARCH"

BIN="$ROOT/build/server/sparkinfer_server"
if [ ! -x "$BIN" ]; then
  echo ">> building sparkinfer_server ..."
  cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SERVER=ON $CUDA_HOST_FLAG >/dev/null
  cmake --build "$ROOT/build" -j"$(nproc)" --target sparkinfer_server
fi

ARGS=(-m "$GGUF" --host "$HOST" --port "$PORT" --tokenizer "$MODELS_DIR/tokenizer.json" --model-name "$MODEL_NAME")
[ "$CTX" != "0" ] && ARGS+=(--ctx "$CTX")
[ -n "${API_KEY:-}" ] && ARGS+=(--api-key "$API_KEY")

echo ">> sparkinfer-server http://${HOST}:${PORT} model=$GGUF"
exec "$BIN" "${ARGS[@]}"
