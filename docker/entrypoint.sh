#!/usr/bin/env bash
# Default: serve (AR + image + video) over an OpenAI-compatible API.
# `bench [workload]`: run the DSpark speculative benchmark instead.
set -euo pipefail

fetch() {  # repo dir label
  if [ ! -f "$2/config.json" ]; then
    echo "[sparkinfer] downloading $3 ($1) — first run only, cached in /models"
    hf download "$1" --local-dir "$2"
  fi
}

if [ "${1:-}" = "bench" ]; then
  shift
  WORKLOAD="${1:-code}"
  fetch "$MODEL_REPO" "$MODEL_DIR" "target"
  fetch "$DRAFT_REPO" "$DRAFT_DIR" "DSpark drafter"
  echo "[sparkinfer] DSpark bench · workload=$WORKLOAD · ${BENCH_TOKENS:-256} tokens"
  MODEL_DIR="$MODEL_DIR" python3 /opt/sparkinfer/mkids.py "$WORKLOAD" > /tmp/ids.txt
  exec /opt/sparkinfer/bin/qwen38_hf_dflash_bench \
       "$MODEL_DIR" "$DRAFT_DIR" "${BENCH_TOKENS:-256}" $(cat /tmp/ids.txt)
fi

fetch "$MODEL_REPO" "$MODEL_DIR" "target"
echo "[sparkinfer] serving $MODEL_DIR as '$MODEL_NAME' on $HOST:$PORT (ctx $CTX, max output $SPARKINFER_MAX_OUTPUT_TOKENS)"
exec /opt/sparkinfer/bin/sparkinfer_server \
  -m "$MODEL_DIR" --tokenizer "$MODEL_DIR/tokenizer.json" \
  --model-name "$MODEL_NAME" --ctx "$CTX" --host "$HOST" --port "$PORT" "$@"
