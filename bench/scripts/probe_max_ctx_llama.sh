#!/usr/bin/env bash
# Probe maximum context for Qwythos via llama.cpp (reference ceiling on RTX 5090).
# Use until sparkinfer loads Qwen3.5-9B dense hybrid GGUFs (probe_max_ctx.sh).
#
#   PRIMARY_QUANT=Q4_K_M|Q8_0|BF16 ./bench/scripts/probe_max_ctx_llama.sh [model.gguf]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_qwythos.sh"

LLAMA_BENCH="${LLAMA_BENCH:-/workspace/.llamacpp/build/bin/llama-bench}"
FILE="${1:-${QWYTHOS_MODELS_DIR}/$(qwythos_quant_file)}"
QUANT="${PRIMARY_QUANT:-Q4_K_M}"
NTOK=32
OUT_JSON="${PROBE_LLAMA_OUT_JSON:-/tmp/qwythos_max_ctx_llama_${QUANT}.json}"

[ -x "$LLAMA_BENCH" ] || { echo "!! missing llama-bench: $LLAMA_BENCH" >&2; exit 1; }
[ -f "$FILE" ] || { echo "!! missing model: $FILE" >&2; exit 1; }

CANDIDATES=(32768 65536 131072 262144 524288 786432 819200 851968 917504 983040 1048576)
MAX_OK=0
ATTEMPTS_JSON="[]"

try_ctx() {
  local ctx="$1" out rc tps err
  out="$("$LLAMA_BENCH" -m "$FILE" -p "$ctx" -n "$NTOK" -r 1 --no-warmup -o json 2>&1)" && rc=0 || rc=$?
  tps="$(python3 -c "import re,sys; m=re.findall(r'\"samples_ts\":\s*\[\s*([0-9.]+)', sys.stdin.read()); print(m[-1] if m else '')" <<<"$out")"
  if [ "$rc" -eq 0 ] && [ -n "$tps" ]; then
    ATTEMPTS_JSON="$(python3 -c "import json; a=json.loads('''$ATTEMPTS_JSON'''); a.append({'ctx':$ctx,'ok':True,'tps':float('$tps')}); print(json.dumps(a))")"
    return 0
  fi
  err="$(echo "$out" | grep -E 'error|failed|OOM' | tail -1 | tr '\n' ' ' | sed 's/"/\\"/g')"
  ATTEMPTS_JSON="$(python3 -c "import json; a=json.loads('''$ATTEMPTS_JSON'''); a.append({'ctx':$ctx,'ok':False,'error':'''${err:-load failed}'''}); print(json.dumps(a))")"
  return 1
}

echo ">> llama.cpp max-context probe: $FILE (quant=$QUANT)" >&2
for ctx in "${CANDIDATES[@]}"; do
  echo ">> try ctx=$ctx ..." >&2
  if try_ctx "$ctx"; then MAX_OK="$ctx"; else echo ">> failed at ctx=$ctx" >&2; break; fi
done

python3 - <<PY | tee "$OUT_JSON"
import json
print(json.dumps({
    "engine": "llama.cpp",
    "model": "$(basename "$FILE")",
    "quant": "$QUANT",
    "gpu": "RTX 5090",
    "max_ctx": int("$MAX_OK"),
    "decode_tokens": $NTOK,
    "attempts": json.loads('''$ATTEMPTS_JSON'''),
}, indent=2))
PY
