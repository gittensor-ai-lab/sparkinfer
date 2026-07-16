#!/usr/bin/env bash
# Probe maximum decode context for Qwythos-9B on the current GPU (RTX 5090 target).
# Steps upward from 32k until qwen3_gguf_bench fails (OOM or error).
#
#   PRIMARY_QUANT=Q4_K_M|Q8_0|BF16 ./bench/scripts/probe_max_ctx.sh [model.gguf]
#
# Prints JSON: {"quant":"...","max_ctx":N,"attempts":[...]}
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"
source "$HERE/_qwythos.sh"

reap() { pkill -f qwen3_gguf 2>/dev/null || true; sleep 1; true; }

ARCH="$(detect_arch)"
FILE="${1:-${QWYTHOS_MODELS_DIR}/$(qwythos_quant_file)}"
QUANT="${PRIMARY_QUANT:-Q4_K_M}"
NTOK=32
OUT_JSON="${PROBE_OUT_JSON:-/tmp/qwythos_max_ctx_${QUANT}.json}"

[ -f "$FILE" ] || { echo "!! missing model: $FILE" >&2; exit 1; }
ensure_sparkinfer "$ARCH"
resolve_runner "$ARCH"

# Candidate contexts (KV prefill tokens). YaRN supports 1M in llama.cpp; find sparkinfer limit.
CANDIDATES=(32768 65536 98304 131072 196608 262144 393216 524288 786432 1048576)
MAX_OK=0
ATTEMPTS_JSON="[]"

try_ctx() {
  local ctx="$1"
  local out rc tps err
  out="$(si_run qwen3_gguf_bench "$FILE" "$NTOK" "$ctx" 2>&1)" && rc=0 || rc=$?
  tps="$(echo "$out" | sed -n 's/.*decode tg *: *\([0-9.][0-9.]*\).*/\1/p' | tail -1)"
  if [ "$rc" -eq 0 ] && [ -n "$tps" ]; then
    ATTEMPTS_JSON="$(python3 -c "import json; a=json.loads('''$ATTEMPTS_JSON'''); a.append({'ctx':$ctx,'ok':True,'tps':float('$tps')}); print(json.dumps(a))")"
    return 0
  fi
  err="$(echo "$out" | tail -3 | tr '\n' ' ' | sed 's/"/\\"/g')"
  ATTEMPTS_JSON="$(python3 -c "import json; a=json.loads('''$ATTEMPTS_JSON'''); a.append({'ctx':$ctx,'ok':False,'error':'''${err}'''}); print(json.dumps(a))")"
  return 1
}

echo ">> probing max context: $FILE (quant=$QUANT, arch=sm_$ARCH)" >&2
for ctx in "${CANDIDATES[@]}"; do
  echo ">> try ctx=$ctx ..." >&2
  if try_ctx "$ctx"; then
    MAX_OK="$ctx"
  else
    echo ">> failed at ctx=$ctx — stopping" >&2
    break
  fi
  reap
  sleep 2
done
reap

python3 - <<PY | tee "$OUT_JSON"
import json, os
print(json.dumps({
    "model": os.path.basename("$FILE"),
    "quant": "$QUANT",
    "gpu_arch": "sm_$ARCH",
    "max_ctx": int("$MAX_OK"),
    "decode_tokens": $NTOK,
    "attempts": json.loads('''$ATTEMPTS_JSON'''),
}, indent=2))
PY
