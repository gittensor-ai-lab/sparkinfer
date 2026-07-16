# Shared decode + prefill speed helpers for evaluate.sh (sourced, not executed).
# median_bench_metric CTX REPS PATTERN — median tok/s from qwen3_gguf_bench output lines.

_BENCH_SWEEP_JSON=""

bench_sweep_enabled() {
  [ "${SPARKINFER_BENCH_SWEEP:-1}" != "0" ]
}

# Read decode_tps or prefill_pp for one context from cached SWEEP_JSON.
_bench_sweep_get() {
  local ctx="$1" field="$2"
  python3 - "$ctx" "$field" "$_BENCH_SWEEP_JSON" <<'PY'
import json, sys
ctx, field, raw = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.loads(raw)
    row = d.get(str(ctx), {})
    v = row.get(field, 0)
    print(v if v else 0)
except Exception:
    print(0)
PY
}

# One model load, many contexts. Args: GGUF N_TOKENS (CTX REPS)+
bench_sweep_run() {
  local gguf="$1" n_tokens="$2"
  shift 2
  local ctxs=() ctx reps max_reps=1
  while [ $# -ge 2 ]; do
    ctx="$1"; reps="$2"; shift 2
    [ "${reps:-0}" -le 0 ] && continue
    ctxs+=("$ctx")
    [ "$reps" -gt "$max_reps" ] && max_reps="$reps"
  done
  [ ${#ctxs[@]} -eq 0 ] && { _BENCH_SWEEP_JSON=""; return 1; }
  local csv
  csv="$(printf '%s\n' "${ctxs[@]}" | sort -nu | paste -sd,)"
  local out rc=0
  export SPARKINFER_BENCH_SWEEP_CTXS="$csv"
  export SPARKINFER_BENCH_SWEEP_REPS="$max_reps"
  out="$(si_run qwen3_gguf_bench "$gguf" "$n_tokens" sweep 2>&1)" || rc=$?
  _BENCH_SWEEP_JSON="$(printf '%s\n' "$out" | sed -n 's/^SWEEP_JSON //p' | tail -1)"
  if [ "$rc" != 0 ] || [ -z "$_BENCH_SWEEP_JSON" ]; then
    echo ">> WARN: bench sweep failed (rc=$rc): ${out##*$'\n'}" >&2
    _BENCH_SWEEP_JSON=""
    return 1
  fi
  gclks+=("$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')")
  echo ">> bench sweep ok (${csv})" >&2
  return 0
}

median_bench_metric() {
  local ctx="$1" reps="$2" pat="$3"
  local vals=() t rc out
  [ "${reps:-0}" -le 0 ] && { echo 0; return; }
  for _ in $(seq 1 "$reps"); do
    out="$(si_run qwen3_gguf_bench "$GGUF" "$DECODE_TOKENS" "$ctx" 2>&1)" || rc=$?
    t=$(printf '%s\n' "$out" | sed -n "s/.*${pat} *: *\\([0-9.][0-9.]*\\).*/\\1/p" | tail -1 || true)
    if [ "${rc:-0}" != 0 ] || [ -z "$t" ]; then
      echo ">> WARN: bench metric '${pat}' failed (ctx=$ctx rc=${rc:-0}): ${out##*$'\n'}" >&2
      t=0
    fi
    vals+=("${t:-0}")
    gclks+=("$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')")
  done
  printf '%s\n' "${vals[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'
}

median_ctx() { median_bench_metric "$1" "$2" "decode tg"; }
median_ctx_pp() { median_bench_metric "$1" "$2" "prefill pp"; }

# Build SCORE_SELECT JSON for decode or prefill (metric key in context rows: tps).
# Args: GUARD_TPS GUARD_512 ... TPS_16K GUARD_32K GUARD_64K GUARD_128K
#       baselines and llama refs for each context.
build_score_select() {
  local metric="$1"
  python3 - "$metric" <<'PY'
import json, os, sys
metric = sys.argv[1]
def f(k, d=0.0):
    v = os.environ.get(k, "")
    try:
        return float(v) if v else d
    except ValueError:
        return d
contexts = [
  {"ctx":128, "label":"128-context", "tps":f("GUARD_TPS"), "base":f("GUARD_BASELINE"), "llama":f("LLAMA_128")},
  {"ctx":512, "label":"512-context", "tps":f("GUARD_512_TPS"), "base":f("GUARD_512_BASELINE"), "llama":f("LLAMA_512")},
  {"ctx":4096, "label":"4k-context", "tps":f("GUARD_4K_TPS"), "base":f("GUARD_4K_BASELINE"), "llama":f("LLAMA_4K")},
  {"ctx":16384, "label":"16k-context", "tps":f("TPS_16K"), "base":f("GUARD_16K_BASELINE") or f("FRONTIER"), "llama":f("LLAMA_16K")},
  {"ctx":32768, "label":"32k-context", "tps":f("GUARD_32K_TPS"), "base":f("GUARD_32K_BASELINE"), "llama":f("LLAMA_32K")},
  {"ctx":65536, "label":"64k-context", "tps":f("GUARD_64K_TPS"), "base":f("GUARD_64K_BASELINE"), "llama":f("LLAMA_64K")},
  {"ctx":131072, "label":"128k-context", "tps":f("GUARD_128K_TPS"), "base":f("GUARD_128K_BASELINE"), "llama":f("LLAMA_128K")},
]
for c in contexts:
    c["gain"] = 0.0 if c["base"] <= 0 else (c["tps"] - c["base"]) / c["base"]
scorable = [c for c in contexts if c["base"] > 0 and c["tps"] > 0]
score_ctx = int(os.environ.get("SCORE_CTX", "16384"))
chosen = max(scorable, key=lambda c: c["gain"]) if scorable else next(c for c in contexts if c["ctx"] == score_ctx)
print(json.dumps({"chosen": chosen, "contexts": contexts, "metric": metric}, separators=(",", ":")))
PY
}

guard_ratio() {
  python3 - <<PY
base=float("$1"); cur=float("$2")
print(0 if base <= 0 else cur / base)
PY
}

guard_pass() {
  python3 - <<PY
base=float("$1"); cur=float("$2"); tol=float("$3")
print("true" if base <= 0 or cur >= base * tol else "false")
PY
}

# Build SCORE_SELECT JSON for Qwen3.5 prefill pp (4k/32k/64k/128k only).
# Uses bash expansion (not os.environ) so callers need not export GUARD_*_PP_*.
build_pp_score_select() {
  python3 - <<PY
import json
contexts = [
  {"ctx":4096, "label":"4k-context", "tps":float("$GUARD_4K_PP_TPS"), "base":float("$GUARD_4K_PP_BASELINE"), "llama":float("$LLAMA_4K_PP")},
  {"ctx":32768, "label":"32k-context", "tps":float("$GUARD_32K_PP_TPS"), "base":float("$GUARD_32K_PP_BASELINE"), "llama":float("$LLAMA_32K_PP")},
  {"ctx":65536, "label":"64k-context", "tps":float("$GUARD_64K_PP_TPS"), "base":float("$GUARD_64K_PP_BASELINE"), "llama":float("$LLAMA_64K_PP")},
  {"ctx":131072, "label":"128k-context", "tps":float("$GUARD_128K_PP_TPS"), "base":float("$GUARD_128K_PP_BASELINE"), "llama":float("$LLAMA_128K_PP")},
]
for c in contexts:
    c["gain"] = 0.0 if c["base"] <= 0 else (c["tps"] - c["base"]) / c["base"]
scorable = [c for c in contexts if c["base"] > 0 and c["tps"] > 0]
fallback = {"ctx":4096, "label":"4k-context", "tps":0, "base":0, "llama":0, "gain":0}
chosen = max(scorable, key=lambda c: c["gain"]) if scorable else fallback
print(json.dumps({"chosen": chosen, "contexts": contexts, "metric": "prefill"}, separators=(",", ":")))
PY
}
