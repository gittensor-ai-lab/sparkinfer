#!/usr/bin/env bash
# Bidirectional evaluation: Qwen3.5-9B (Qwythos) and Qwen3.6-35B-A3B — one build, one box.
#
# Runs BOTH scoring directions in one submission:
#   • score_qwen35 — optimize Qwen3.5 (128/4k/32k/64k decode + 4k/32k/64k/128k prefill), guard Qwen3.6 (5 contexts)
#   • score_qwen36 — optimize Qwen3.6 (5 decode contexts + 128/512/4k/16k/32k prefill), guard Qwen3.5 (128/4k/32k/64k)
#
#   bench/scripts/evaluate_bidir.sh [--ref GIT_REF] [--ceiling TPS]
#
# PRIMARY_QUANT: Q4_K_M (default) | Q8_0 | BF16
#
# Env (optional):
#   SPARKINFER_P35_GUARD_*     Qwen3.5 same-box main decode tok/s (128/4k/32k/64k)
#   SPARKINFER_P36_GUARD_*_PP  Qwen3.6 same-box main prefill pp tok/s (128/512/4k/16k/32k)
#   SPARKINFER_P36_GUARD_*     Qwen3.6 same-box main tok/s (128/512/4k/16k/32k)
#   SPARKINFER_G36_GUARD_*   Qwen3.6 guard baselines (for score_qwen35)
#   SPARKINFER_G35_GUARD_*   Qwen3.5 guard baselines (for score_qwen36)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"
source "$HERE/_eval_speed.sh"
source "$HERE/_qwythos.sh"

REF=""; CEILING=0; BASELINE_ONLY=0
while [ $# -gt 0 ]; do case "$1" in
  --ref) shift; REF="$1" ;; --ceiling) shift; CEILING="$1" ;;
  --baseline-only) BASELINE_ONLY=1 ;;
  *) ;;
esac; shift; done

export LLAMACPP_DIR="${LLAMACPP_DIR:-/workspace/.llamacpp}"
ARCH="$(detect_arch)"

if [ -n "$REF" ] && [ -z "${SI_NO_CHECKOUT:-}" ]; then
  git -C "$ROOT" fetch -q origin "$REF" 2>/dev/null || true; git -C "$ROOT" checkout -q "$REF"
fi
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"

P35_FILE="$(qwythos_quant_file)"
P35_REPO="${QWYTHOS_REPO}"
P35_TOK="${QWYTHOS_TOK_REPO}"
P35_DIR="${QWYTHOS_MODELS_DIR}"
P35_SHA="$(qwythos_sha_var)"

P36_FILE="${PRIMARY36_MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
P36_REPO="${PRIMARY36_MODEL_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
P36_TOK="${PRIMARY36_TOK_REPO:-Qwen/Qwen3.6-35B-A3B}"
P36_DIR="${PRIMARY36_MODELS_DIR:-${MODELS_DIR:-$ROOT/models}36}"

QUANT="${PRIMARY_QUANT:-Q4_K_M}"
echo ">> bidir: Qwen3.5=$P35_FILE (quant=$QUANT, ctx=128/4k/32k/64k) + Qwen3.6=$P36_FILE (ctx=128/512/4k/16k/32k)" >&2

reap() {
  pkill -f llama-server 2>/dev/null || true
  pkill -f qwen3_gguf_bench 2>/dev/null || true
  pkill -f qwen3_gguf 2>/dev/null || true
  sleep 3
  true
}
reap_bench() {
  pkill -f qwen3_gguf_bench 2>/dev/null || true
  pkill -f qwen3_gguf 2>/dev/null || true
  sleep 3
  true
}

declare -A _ACC_TOP1 _ACC_KL
_LAST_GGUF=""

_models_dir_from_args() {
  local file="$1"; shift
  local mdir="${MODELS_DIR:-$ROOT/models}"
  local a
  for a in "$@"; do
    case "$a" in MODELS_DIR=*) mdir="${a#MODELS_DIR=}" ;; esac
  done
  echo "${mdir}/${file}"
}

run_model() {
  local role="$1" file="$2" repo="$3" tok="$4" frontier="$5"; shift 5
  local gguf skip_env=() json
  gguf="$(_models_dir_from_args "$file" "$@")"
  if [ -n "${_ACC_TOP1[$gguf]:-}" ]; then
    skip_env=(SPARKINFER_SKIP_ACCURACY=1
              SPARKINFER_ACCURACY_TOP1="${_ACC_TOP1[$gguf]}"
              SPARKINFER_ACCURACY_KL="${_ACC_KL[$gguf]}")
    echo ">> [$role] reusing cached accuracy for $file (top1=${_ACC_TOP1[$gguf]}, kl=${_ACC_KL[$gguf]})" >&2
    if [ "$gguf" = "$_LAST_GGUF" ]; then reap_bench; else reap; fi
  else
    reap
  fi
  echo ">> [$role] model $file ..." >&2
  json="$(env SI_SKIP_BUILD=1 SI_NO_CHECKOUT=1 \
      MODEL_FILE="$file" MODEL_REPO="$repo" TOK_REPO="$tok" \
      "${skip_env[@]}" \
      "$@" \
      "$HERE/evaluate.sh" --ref "$REF" --frontier "$frontier" --ceiling "$CEILING" \
    | sed -n 's/^RESULT_JSON //p' | tail -1)"
  if [ -z "${_ACC_TOP1[$gguf]:-}" ] && [ -n "$json" ]; then
    read -r _t _k < <(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("top1",0), d.get("kl",99))' "$json")
    _ACC_TOP1[$gguf]="$_t"
    _ACC_KL[$gguf]="$_k"
  fi
  _LAST_GGUF="$gguf"
  echo "$json"
}

if [ "$BASELINE_ONLY" = 1 ]; then
  echo ">> bidir baseline-only: ctx speed sweep on origin/main (skip PR build + dual eval)" >&2
  reap
  git -C "$ROOT" fetch -q origin main 2>/dev/null || true
  if [ -n "${SI_NO_CHECKOUT:-}" ]; then
    _bl_head="$(git -C "$ROOT" rev-parse origin/main)"
    COMMIT="$(git -C "$ROOT" rev-parse --short origin/main)"
    _bs_save="$(mktemp -d)"
    cp -a "$ROOT/bench/scripts/." "$_bs_save/"
    git -C "$ROOT" checkout -qf origin/main
    cp -a "$_bs_save/." "$ROOT/bench/scripts/"
    rm -rf "$_bs_save"
  else
    git -C "$ROOT" checkout -qf origin/main
    COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
    _bl_head="$(git -C "$ROOT" rev-parse HEAD)"
  fi
  _bl_mark="$ROOT/build/.baseline_commit"
  if [ ! -x "$ROOT/build/runtime/qwen3_gguf_bench" ] || \
     [ "$(cat "$_bl_mark" 2>/dev/null)" != "$_bl_head" ]; then
    echo ">> baseline-only: building origin/main ($_bl_head) ..." >&2
    rm -rf "$ROOT/build"
    if ! NO_PREBUILT=1 ensure_sparkinfer "$ARCH"; then
      printf 'RESULT_JSON {"commit": "%s", "tps": 0, "top1": 0, "kl": 99, "label": "REJECT", "reason": "baseline build failed", "pass": false, "mode": "bidir"}\n' "$COMMIT"
      exit 0
    fi
    echo "$_bl_head" > "$_bl_mark"
  fi
  SPARKINFER_P36_GUARD_128_BASELINE=0
  SPARKINFER_P36_GUARD_512_BASELINE=0
  SPARKINFER_P36_GUARD_4K_BASELINE=0
  SPARKINFER_P36_GUARD_16K_BASELINE=0
  SPARKINFER_P36_GUARD_32K_BASELINE=0
  SPARKINFER_P36_GUARD_128_PP_BASELINE=0
  SPARKINFER_P36_GUARD_512_PP_BASELINE=0
  SPARKINFER_P36_GUARD_4K_PP_BASELINE=0
  SPARKINFER_P36_GUARD_16K_PP_BASELINE=0
  SPARKINFER_P36_GUARD_32K_PP_BASELINE=0
  SPARKINFER_P35_GUARD_128_BASELINE=0
  SPARKINFER_P35_GUARD_4K_BASELINE=0
  SPARKINFER_P35_GUARD_32K_BASELINE=0
  SPARKINFER_P35_GUARD_64K_BASELINE=0
  SPARKINFER_P35_GUARD_128K_BASELINE=0
  SPARKINFER_P35_GUARD_4K_PP_BASELINE=0
  SPARKINFER_P35_GUARD_32K_PP_BASELINE=0
  SPARKINFER_P35_GUARD_64K_PP_BASELINE=0
  SPARKINFER_P35_GUARD_128K_PP_BASELINE=0
  SPARKINFER_P35_GUARD_CB_TTFT_BASELINE=0
  SPARKINFER_P36_GUARD_CB_TTFT_BASELINE=0
else
  echo ">> [build] submission ($COMMIT) from source (sm_$ARCH) — shared by both models ..." >&2
  rm -rf "$ROOT/build"
  if ! NO_PREBUILT=1 ensure_sparkinfer "$ARCH"; then
    echo ">> build FAILED — submission does not compile (sm_$ARCH)" >&2
    printf 'RESULT_JSON {"commit": "%s", "tps": 0, "top1": 0, "kl": 99, "frontier_tps": 0, "label": "REJECT", "reason": "build failed (does not compile)", "pass": false, "mode": "bidir"}\n' "$COMMIT"
    exit 0
  fi
  export SPARKINFER_DOWN_REQUANT_Q4K=1
fi

# Qwen3.5: score/guard at 128/4k/32k/64k decode + 4k/32k/64k/128k prefill (skip 512, 16k decode).
Q35_CTX_ENVS=(
  SPARKINFER_SCORE_REPS=0
  SPARKINFER_GUARD_512_REPS=0
  SPARKINFER_GUARD_16K_REPS=0
  SPARKINFER_GUARD_REPS=1
  SPARKINFER_GUARD_4K_REPS=1
  SPARKINFER_GUARD_32K_REPS=1
  SPARKINFER_GUARD_64K_REPS=1
  SPARKINFER_GUARD_128K_REPS=1
  SPARKINFER_GUARD_16K_BASELINE=0
  SPARKINFER_GUARD_512_BASELINE=0
  SPARKINFER_GUARD_128K_BASELINE=0
)

# Qwen3.6: full 5-context sweep.
Q36_FULL_ENVS=(
  SPARKINFER_SCORE_REPS=1
  SPARKINFER_GUARD_32K_REPS=1
  SPARKINFER_GUARD_REPS=1
  SPARKINFER_GUARD_512_REPS=1
  SPARKINFER_GUARD_4K_REPS=1
)

resolve_runner "$ARCH"
pin_clocks
trap 'unpin_clocks' EXIT

_bench_decode_tps() {
  local out rc=0
  out="$(si_run qwen3_gguf_bench "$1" 128 "$2" 2>&1)" || rc=$?
  if [ "$rc" != 0 ]; then
    echo ">> WARN: bench failed (ctx=$2 rc=$rc): ${out##*$'\n'}" >&2
    echo 0
    return 0
  fi
  echo "$out" | sed -n 's/.*decode tg *: *\([0-9.][0-9.]*\).*/\1/p' | tail -1
}

_bench_prefill_pp() {
  local out rc=0
  [ "${2:-0}" -le 0 ] && { echo 0; return 0; }
  out="$(si_run qwen3_gguf_bench "$1" 128 "$2" 2>&1)" || rc=$?
  if [ "$rc" != 0 ]; then
    echo ">> WARN: prefill bench failed (ctx=$2 rc=$rc): ${out##*$'\n'}" >&2
    echo 0
    return 0
  fi
  echo "$out" | sed -n 's/.*prefill pp *: *\([0-9.][0-9.]*\).*/\1/p' | tail -1
}

if [ "${SPARKINFER_P36_GUARD_128_BASELINE:-0}" = "0" ]; then
  echo ">> measuring Qwen3.6 same-box main (5 contexts, one load) ..." >&2
  P36_GGUF="${P36_DIR}/${P36_FILE}"
  if bench_sweep_enabled && bench_sweep_run "$P36_GGUF" 128 0 1 512 1 4096 1 16384 1 32768 1; then
    B36_128="$(_bench_sweep_get 0 decode_tps)"
    B36_512="$(_bench_sweep_get 512 decode_tps)"
    B36_4K="$(_bench_sweep_get 4096 decode_tps)"
    B36_16K="$(_bench_sweep_get 16384 decode_tps)"
    B36_32K="$(_bench_sweep_get 32768 decode_tps)"
    if [ "${SPARKINFER_P36_GUARD_4K_PP_BASELINE:-0}" = "0" ]; then
      B36_128_PP="$(_bench_sweep_get 0 prefill_pp)"
      B36_512_PP="$(_bench_sweep_get 512 prefill_pp)"
      B36_4K_PP="$(_bench_sweep_get 4096 prefill_pp)"
      B36_16K_PP="$(_bench_sweep_get 16384 prefill_pp)"
      B36_32K_PP="$(_bench_sweep_get 32768 prefill_pp)"
    fi
  else
    for ctx in 0 512 4096 16384 32768; do
      t="$(_bench_decode_tps "$P36_GGUF" "$ctx")"
      t="${t:-0}"
      case "$ctx" in
        0)     B36_128="${t:-0}" ;;
        512)   B36_512="${t:-0}" ;;
        4096)  B36_4K="${t:-0}" ;;
        16384) B36_16K="${t:-0}" ;;
        32768) B36_32K="${t:-0}" ;;
      esac
    done
    if [ "${SPARKINFER_P36_GUARD_4K_PP_BASELINE:-0}" = "0" ]; then
      for ctx in 0 512 4096 16384 32768; do
        t="$(_bench_prefill_pp "$P36_GGUF" "$ctx")"
        t="${t:-0}"
        case "$ctx" in
          0)      B36_128_PP="${t:-0}" ;;
          512)    B36_512_PP="${t:-0}" ;;
          4096)   B36_4K_PP="${t:-0}" ;;
          16384)  B36_16K_PP="${t:-0}" ;;
          32768)  B36_32K_PP="${t:-0}" ;;
        esac
      done
    fi
  fi
  echo ">> Qwen3.6 main: 128=${B36_128:-0} 512=${B36_512:-0} 4k=${B36_4K:-0} 16k=${B36_16K:-0} 32k=${B36_32K:-0} tok/s" >&2
  echo ">> Qwen3.6 prefill: 128=${B36_128_PP:-0} 512=${B36_512_PP:-0} 4k=${B36_4K_PP:-0} 16k=${B36_16K_PP:-0} 32k=${B36_32K_PP:-0} pp tok/s" >&2
fi

if [ "${SPARKINFER_P35_GUARD_128_BASELINE:-0}" = "0" ] || [ "${SPARKINFER_P35_GUARD_4K_PP_BASELINE:-0}" = "0" ]; then
  echo ">> measuring Qwen3.5 same-box main (decode + prefill, one load) ..." >&2
  P35_GGUF="${P35_DIR}/${P35_FILE}"
  if bench_sweep_enabled && bench_sweep_run "$P35_GGUF" 128 0 1 4096 1 32768 1 65536 1 131072 1; then
    if [ "${SPARKINFER_P35_GUARD_128_BASELINE:-0}" = "0" ]; then
      B35_128="$(_bench_sweep_get 0 decode_tps)"
      B35_4K="$(_bench_sweep_get 4096 decode_tps)"
      B35_32K="$(_bench_sweep_get 32768 decode_tps)"
      B35_64K="$(_bench_sweep_get 65536 decode_tps)"
    fi
    if [ "${SPARKINFER_P35_GUARD_4K_PP_BASELINE:-0}" = "0" ]; then
      B35_4K_PP="$(_bench_sweep_get 4096 prefill_pp)"
      B35_32K_PP="$(_bench_sweep_get 32768 prefill_pp)"
      B35_64K_PP="$(_bench_sweep_get 65536 prefill_pp)"
      B35_128K_PP="$(_bench_sweep_get 131072 prefill_pp)"
    fi
  else
    if [ "${SPARKINFER_P35_GUARD_128_BASELINE:-0}" = "0" ]; then
      for ctx in 0 4096 32768 65536; do
        t="$(_bench_decode_tps "$P35_GGUF" "$ctx")"
        t="${t:-0}"
        case "$ctx" in
          0)      B35_128="${t:-0}" ;;
          4096)   B35_4K="${t:-0}" ;;
          32768)  B35_32K="${t:-0}" ;;
          65536)  B35_64K="${t:-0}" ;;
        esac
      done
    fi
    if [ "${SPARKINFER_P35_GUARD_4K_PP_BASELINE:-0}" = "0" ]; then
      for ctx in 4096 32768 65536 131072; do
        t="$(_bench_prefill_pp "$P35_GGUF" "$ctx")"
        t="${t:-0}"
        case "$ctx" in
          4096)    B35_4K_PP="${t:-0}" ;;
          32768)   B35_32K_PP="${t:-0}" ;;
          65536)   B35_64K_PP="${t:-0}" ;;
          131072)  B35_128K_PP="${t:-0}" ;;
        esac
      done
    fi
  fi
  echo ">> Qwen3.5 main: 128=${B35_128:-0} 4k=${B35_4K:-0} 32k=${B35_32K:-0} 64k=${B35_64K:-0} tok/s" >&2
  echo ">> Qwen3.5 prefill: 4k=${B35_4K_PP:-0} 32k=${B35_32K_PP:-0} 64k=${B35_64K_PP:-0} 128k=${B35_128K_PP:-0} pp tok/s" >&2
fi

# CB mixed-load TTFT baselines (Qwen3.5 + Qwen3.6) — same fixed recipe as evaluate.sh.
B35_CB_TTFT="${SPARKINFER_P35_GUARD_CB_TTFT_BASELINE:-0}"
B36_CB_TTFT="${SPARKINFER_P36_GUARD_CB_TTFT_BASELINE:-0}"
if [ "${SPARKINFER_EVAL_PREFILL_CB:-1}" = "1" ]; then
  if ensure_cb_bench "$ARCH"; then
    if [ "${B35_CB_TTFT}" = "0" ]; then
      echo ">> measuring Qwen3.5 CB mixed-load TTFT (4×decode + 8k interrupt) ..." >&2
      P35_GGUF="${P35_DIR}/${P35_FILE}"
      CB_PAIR="$(run_cb_ttft "$P35_GGUF" || true)"
      B35_CB_TTFT="$(printf '%s\n' "$CB_PAIR" | awk '{print $1+0}')"
      echo ">> Qwen3.5 CB TTFT main: ${B35_CB_TTFT}s" >&2
    fi
    if [ "${B36_CB_TTFT}" = "0" ]; then
      echo ">> measuring Qwen3.6 CB mixed-load TTFT (4×decode + 8k interrupt) ..." >&2
      P36_GGUF="${P36_DIR}/${P36_FILE}"
      CB_PAIR="$(run_cb_ttft "$P36_GGUF" || true)"
      B36_CB_TTFT="$(printf '%s\n' "$CB_PAIR" | awk '{print $1+0}')"
      echo ">> Qwen3.6 CB TTFT main: ${B36_CB_TTFT}s" >&2
    fi
  else
    echo ">> WARN: qwen3_gguf_cb_bench unavailable — CB TTFT baselines skipped" >&2
  fi
fi
B35_CB_TTFT="${B35_CB_TTFT:-0}"
B36_CB_TTFT="${B36_CB_TTFT:-0}"

B36_128="${B36_128:-${SPARKINFER_P36_GUARD_128_BASELINE:-0}}"
B36_512="${B36_512:-${SPARKINFER_P36_GUARD_512_BASELINE:-0}}"
B36_4K="${B36_4K:-${SPARKINFER_P36_GUARD_4K_BASELINE:-0}}"
B36_16K="${B36_16K:-${SPARKINFER_P36_GUARD_16K_BASELINE:-0}}"
B36_32K="${B36_32K:-${SPARKINFER_P36_GUARD_32K_BASELINE:-0}}"
B36_128_PP="${B36_128_PP:-${SPARKINFER_P36_GUARD_128_PP_BASELINE:-0}}"
B36_512_PP="${B36_512_PP:-${SPARKINFER_P36_GUARD_512_PP_BASELINE:-0}}"
B36_4K_PP="${B36_4K_PP:-${SPARKINFER_P36_GUARD_4K_PP_BASELINE:-0}}"
B36_16K_PP="${B36_16K_PP:-${SPARKINFER_P36_GUARD_16K_PP_BASELINE:-0}}"
B36_32K_PP="${B36_32K_PP:-${SPARKINFER_P36_GUARD_32K_PP_BASELINE:-0}}"
if [ "$BASELINE_ONLY" != 1 ]; then
  [ "${B36_128}" = "0" ] && B36_128="300.16"
  [ "${B36_512}" = "0" ] && B36_512="296.76"
  [ "${B36_4K}"  = "0" ] && B36_4K="287.91"
  [ "${B36_16K}" = "0" ] && B36_16K="338.55"
  [ "${B36_32K}" = "0" ] && B36_32K="301.19"
fi

B35_128="${B35_128:-${SPARKINFER_P35_GUARD_128_BASELINE:-0}}"
B35_4K="${B35_4K:-${SPARKINFER_P35_GUARD_4K_BASELINE:-0}}"
B35_32K="${B35_32K:-${SPARKINFER_P35_GUARD_32K_BASELINE:-0}}"
B35_64K="${B35_64K:-${SPARKINFER_P35_GUARD_64K_BASELINE:-0}}"
B35_128K="${B35_128K:-${SPARKINFER_P35_GUARD_128K_BASELINE:-0}}"
B35_4K_PP="${B35_4K_PP:-${SPARKINFER_P35_GUARD_4K_PP_BASELINE:-0}}"
B35_32K_PP="${B35_32K_PP:-${SPARKINFER_P35_GUARD_32K_PP_BASELINE:-0}}"
B35_64K_PP="${B35_64K_PP:-${SPARKINFER_P35_GUARD_64K_PP_BASELINE:-0}}"
B35_128K_PP="${B35_128K_PP:-${SPARKINFER_P35_GUARD_128K_PP_BASELINE:-0}}"

G36_128="${SPARKINFER_G36_GUARD_128_BASELINE:-$B36_128}"
G36_512="${SPARKINFER_G36_GUARD_512_BASELINE:-$B36_512}"
G36_4K="${SPARKINFER_G36_GUARD_4K_BASELINE:-$B36_4K}"
G36_16K="${SPARKINFER_G36_GUARD_16K_BASELINE:-$B36_16K}"
G36_32K="${SPARKINFER_G36_GUARD_32K_BASELINE:-$B36_32K}"

G35_128="${SPARKINFER_G35_GUARD_128_BASELINE:-$B35_128}"
G35_4K="${SPARKINFER_G35_GUARD_4K_BASELINE:-$B35_4K}"
G35_32K="${SPARKINFER_G35_GUARD_32K_BASELINE:-$B35_32K}"
G35_64K="${SPARKINFER_G35_GUARD_64K_BASELINE:-$B35_64K}"
G35_128K="${SPARKINFER_G35_GUARD_128K_BASELINE:-$B35_128K}"

P_DIFF_REF="${SPARKINFER_DIFFICULTY_REF_OVERRIDE:-365.85}"

if [ "$BASELINE_ONLY" = 1 ]; then
  echo ">> bidir baseline-only: ctx sweep complete — skipping full dual-model eval" >&2
  if [ "${B36_128:-0}" = "0" ] || [ "${B35_128:-0}" = "0" ]; then
    echo ">> baseline-only FAILED: Qwen3.6=${B36_128:-0} Qwythos=${B35_128:-0} tok/s at 128 ctx" >&2
    printf 'RESULT_JSON {"commit": "%s", "tps": 0, "top1": 0, "kl": 99, "label": "REJECT", "reason": "baseline ctx sweep failed (0 tok/s)", "pass": false, "mode": "bidir"}\n' "$COMMIT"
    exit 0
  fi
  QUANT="${PRIMARY_QUANT:-Q4_K_M}"
  python3 - <<PY
import json
commit = "$COMMIT"
quant = "$QUANT"
def stub(tps, ctx128, ctx4k, ctx32k=0, ctx64k=0, ctx128k=0, ctx512=0, ctx16k=0,
         pp4k=0, pp32k=0, pp64k=0, pp128k=0, pp128=0, pp512=0, pp16k=0):
    return {"pass": True, "label": "BASELINE", "tps": float(tps or ctx128 or 0),
            "top1": 1.0, "kl": 0.0, "ctx_128_tps": float(ctx128 or 0),
            "ctx_512_tps": float(ctx512 or 0), "ctx_4096_tps": float(ctx4k or 0),
            "ctx_16384_tps": float(ctx16k or 0), "ctx_32768_tps": float(ctx32k or 0),
            "ctx_65536_tps": float(ctx64k or 0), "ctx_131072_tps": float(ctx128k or 0),
            "ctx_128_pp_tps": float(pp128 or 0), "ctx_512_pp_tps": float(pp512 or 0),
            "ctx_4096_pp_tps": float(pp4k or 0), "ctx_16384_pp_tps": float(pp16k or 0),
            "ctx_32768_pp_tps": float(pp32k or 0), "ctx_65536_pp_tps": float(pp64k or 0),
            "ctx_131072_pp_tps": float(pp128k or 0),
            "guard_128_pass": True, "guard_512_pass": True, "guard_4k_pass": True,
            "guard_16k_pass": True, "guard_32k_pass": True,
            "guard_64k_pass": True, "guard_128k_pass": True,
            "guard_128_pp_pass": True, "guard_512_pp_pass": True,
            "guard_4k_pp_pass": True, "guard_16k_pp_pass": True, "guard_32k_pp_pass": True,
            "guard_64k_pp_pass": True, "guard_128k_pp_pass": True,
            "eval_prefill": bool(float(pp4k or pp32k or pp64k or pp128k or pp128 or pp512 or pp16k or 0) > 0)}
s35 = stub("$B35_128", "$B35_128", "$B35_4K", "$B35_32K", "$B35_64K",
           pp4k="$B35_4K_PP", pp32k="$B35_32K_PP", pp64k="$B35_64K_PP", pp128k="$B35_128K_PP")
s35["cb_ttft_s"] = float("${B35_CB_TTFT:-0}" or 0)
s36 = stub("$B36_128", "$B36_128", "$B36_4K", "$B36_32K", ctx512="$B36_512", ctx16k="$B36_16K",
           pp128="$B36_128_PP", pp512="$B36_512_PP", pp4k="$B36_4K_PP", pp16k="$B36_16K_PP", pp32k="$B36_32K_PP")
s36["cb_ttft_s"] = float("${B36_CB_TTFT:-0}" or 0)
out = {"pass": True, "label": "BASELINE", "commit": commit, "mode": "bidir", "model": "bidir",
       "tps": s36["tps"], "top1": 1.0, "kl": 0.0, "primary_quant": quant,
       "score_qwen35": s35, "score_qwen36": s36, "pass_qwen35": True, "pass_qwen36": True,
       "label_qwen35": "BASELINE", "label_qwen36": "BASELINE",
       "cb_ttft_s": float("${B35_CB_TTFT:-0}" or 0),
       "cb_ttft_s_qwen36": float("${B36_CB_TTFT:-0}" or 0)}
print("RESULT_JSON " + json.dumps(out))
PY
  exit 0
fi

PRIMARY35_JSON="$(run_model primary-qwen35 "$P35_FILE" "$P35_REPO" "$P35_TOK" 0 \
  MODELS_DIR="$P35_DIR" MODEL_SHA256="${P35_SHA}" \
  "${Q35_CTX_ENVS[@]}" \
  SPARKINFER_EVAL_PREFILL=1 SPARKINFER_PREFILL_PROFILE=qwen35 \
  SPARKINFER_DIFFICULTY_BOOST=1 SPARKINFER_DIFFICULTY_REF="${P_DIFF_REF}" \
  SPARKINFER_GUARD_128_BASELINE="${B35_128}" \
  SPARKINFER_GUARD_4K_BASELINE="${B35_4K}" \
  SPARKINFER_GUARD_32K_BASELINE="${B35_32K}" \
  SPARKINFER_GUARD_64K_BASELINE="${B35_64K}" \
  SPARKINFER_GUARD_4K_PP_BASELINE="${B35_4K_PP}" \
  SPARKINFER_GUARD_32K_PP_BASELINE="${B35_32K_PP}" \
  SPARKINFER_GUARD_64K_PP_BASELINE="${B35_64K_PP}" \
  SPARKINFER_GUARD_128K_PP_BASELINE="${B35_128K_PP}" \
  SPARKINFER_GUARD_CB_TTFT_BASELINE="${B35_CB_TTFT}" \
  SPARKINFER_LLAMA_128_BASELINE="${SPARKINFER_P35_LLAMA_128_BASELINE:-${QWEN35_9B_LLAMA_128:-0}}" \
  SPARKINFER_LLAMA_4K_BASELINE="${SPARKINFER_P35_LLAMA_4K_BASELINE:-${QWEN35_9B_LLAMA_4K:-0}}" \
  SPARKINFER_LLAMA_32K_BASELINE="${SPARKINFER_P35_LLAMA_32K_BASELINE:-${QWEN35_9B_LLAMA_32K:-0}}" \
  SPARKINFER_LLAMA_64K_BASELINE="${SPARKINFER_P35_LLAMA_64K_BASELINE:-${QWEN35_9B_LLAMA_64K:-0}}" \
  SPARKINFER_LLAMA_4K_PP_BASELINE="${SPARKINFER_P35_LLAMA_4K_PP_BASELINE:-${QWEN35_9B_LLAMA_4K_PP:-0}}" \
  SPARKINFER_LLAMA_32K_PP_BASELINE="${SPARKINFER_P35_LLAMA_32K_PP_BASELINE:-${QWEN35_9B_LLAMA_32K_PP:-0}}" \
  SPARKINFER_LLAMA_64K_PP_BASELINE="${SPARKINFER_P35_LLAMA_64K_PP_BASELINE:-${QWEN35_9B_LLAMA_64K_PP:-0}}" \
  SPARKINFER_LLAMA_128K_PP_BASELINE="${SPARKINFER_P35_LLAMA_128K_PP_BASELINE:-${QWEN35_9B_LLAMA_128K_PP:-0}}")"

GUARD36_JSON="$(run_model guard36 "$P36_FILE" "$P36_REPO" "$P36_TOK" 0 \
  MODELS_DIR="$P36_DIR" MODEL_SHA256="${QWEN36_MODEL_SHA256:-}" \
  SPARKINFER_EVAL_PREFILL=0 \
  "${Q36_FULL_ENVS[@]}" \
  SPARKINFER_GUARD_128_BASELINE="${G36_128}" \
  SPARKINFER_GUARD_512_BASELINE="${G36_512}" \
  SPARKINFER_GUARD_4K_BASELINE="${G36_4K}" \
  SPARKINFER_GUARD_16K_BASELINE="${G36_16K}" \
  SPARKINFER_GUARD_32K_BASELINE="${G36_32K}")"

PRIMARY36_JSON="$(run_model primary-qwen36 "$P36_FILE" "$P36_REPO" "$P36_TOK" 0 \
  MODELS_DIR="$P36_DIR" MODEL_SHA256="${QWEN36_MODEL_SHA256:-}" \
  SPARKINFER_EVAL_PREFILL=1 SPARKINFER_PREFILL_PROFILE=qwen36 \
  "${Q36_FULL_ENVS[@]}" \
  SPARKINFER_DIFFICULTY_BOOST=1 SPARKINFER_DIFFICULTY_REF="${P_DIFF_REF}" \
  SPARKINFER_GUARD_128_BASELINE="${B36_128}" \
  SPARKINFER_GUARD_512_BASELINE="${B36_512}" \
  SPARKINFER_GUARD_4K_BASELINE="${B36_4K}" \
  SPARKINFER_GUARD_16K_BASELINE="${B36_16K}" \
  SPARKINFER_GUARD_32K_BASELINE="${B36_32K}" \
  SPARKINFER_GUARD_128_PP_BASELINE="${B36_128_PP}" \
  SPARKINFER_GUARD_512_PP_BASELINE="${B36_512_PP}" \
  SPARKINFER_GUARD_4K_PP_BASELINE="${B36_4K_PP}" \
  SPARKINFER_GUARD_16K_PP_BASELINE="${B36_16K_PP}" \
  SPARKINFER_GUARD_32K_PP_BASELINE="${B36_32K_PP}" \
  SPARKINFER_GUARD_CB_TTFT_BASELINE="${B36_CB_TTFT}" \
  SPARKINFER_LLAMA_128_BASELINE="${SPARKINFER_P36_LLAMA_128_BASELINE:-275.81}" \
  SPARKINFER_LLAMA_512_BASELINE="${SPARKINFER_P36_LLAMA_512_BASELINE:-275.61}" \
  SPARKINFER_LLAMA_4K_BASELINE="${SPARKINFER_P36_LLAMA_4K_BASELINE:-276.30}" \
  SPARKINFER_LLAMA_16K_BASELINE="${SPARKINFER_P36_LLAMA_16K_BASELINE:-280.66}" \
  SPARKINFER_LLAMA_32K_BASELINE="${SPARKINFER_P36_LLAMA_32K_BASELINE:-279.83}" \
  SPARKINFER_LLAMA_128_PP_BASELINE="${SPARKINFER_P36_LLAMA_128_PP_BASELINE:-${QWEN36_LLAMA_128_PP:-0}}" \
  SPARKINFER_LLAMA_512_PP_BASELINE="${SPARKINFER_P36_LLAMA_512_PP_BASELINE:-${QWEN36_LLAMA_512_PP:-0}}" \
  SPARKINFER_LLAMA_4K_PP_BASELINE="${SPARKINFER_P36_LLAMA_4K_PP_BASELINE:-${QWEN36_LLAMA_4K_PP:-0}}" \
  SPARKINFER_LLAMA_16K_PP_BASELINE="${SPARKINFER_P36_LLAMA_16K_PP_BASELINE:-${QWEN36_LLAMA_16K_PP:-0}}" \
  SPARKINFER_LLAMA_32K_PP_BASELINE="${SPARKINFER_P36_LLAMA_32K_PP_BASELINE:-${QWEN36_LLAMA_32K_PP:-0}}")"

GUARD35_JSON="$(run_model guard35 "$P35_FILE" "$P35_REPO" "$P35_TOK" 0 \
  MODELS_DIR="$P35_DIR" MODEL_SHA256="${P35_SHA}" \
  SPARKINFER_EVAL_PREFILL=0 \
  "${Q35_CTX_ENVS[@]}" \
  SPARKINFER_GUARD_128_BASELINE="${G35_128}" \
  SPARKINFER_GUARD_4K_BASELINE="${G35_4K}" \
  SPARKINFER_GUARD_32K_BASELINE="${G35_32K}" \
  SPARKINFER_GUARD_64K_BASELINE="${G35_64K}")"
reap

PRIMARY35_JSON="$PRIMARY35_JSON" GUARD36_JSON="$GUARD36_JSON" \
PRIMARY36_JSON="$PRIMARY36_JSON" GUARD35_JSON="$GUARD35_JSON" \
COMMIT="$COMMIT" QUANT="$QUANT" python3 - <<'PY'
import json, os

TIER_RANK = {"XL": 6, "L": 5, "M": 4, "S": 3, "XS": 2, "none": 1, "BASELINE": 0, "REJECT": -1}

def load(name):
    raw = os.environ.get(name, "").strip()
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}

def guard_ok(guard):
    if guard.get("infra_error"):
        return True, [], True, True
    gctx = ["guard_128_pass", "guard_512_pass", "guard_4k_pass", "guard_16k_pass", "guard_32k_pass",
            "guard_64k_pass", "guard_128k_pass"]
    present = [k for k in gctx if k in guard]
    top1_bar = float(os.environ.get("SPARKINFER_TOP1_BAR", "0.90"))
    kl_bar = float(os.environ.get("SPARKINFER_KL_BAR", "0.20"))
    speed_ok = all(guard.get(k, True) for k in present)
    g_top1 = float(guard.get("top1", 0)); g_kl = float(guard.get("kl", 99))
    acc_ok = g_top1 >= top1_bar and g_kl <= kl_bar
    label_map = {"guard_128_pass": "128", "guard_512_pass": "512", "guard_4k_pass": "4k",
                 "guard_16k_pass": "16k", "guard_32k_pass": "32k",
                 "guard_64k_pass": "64k", "guard_128k_pass": "128k"}
    regressed = [label_map[k] for k in present if not guard.get(k, True)]
    return bool(guard) and speed_ok and acc_ok, regressed, speed_ok, acc_ok

def merge_primary(primary, guard, scored_model, guard_model, guard_prefix):
    if not primary:
        return {"label": "REJECT", "pass": False,
                "reason": f"primary ({scored_model}) produced no verdict (infra error)"}
    if primary.get("infra_error"):
        return dict(primary)
    _, regressed, speed_ok, acc_ok = guard_ok(guard)
    out = dict(primary)
    out["model"] = scored_model
    out["guard_model"] = guard_model
    out["guard"] = {k: guard.get(k) for k in (
        "pass", "top1", "kl", "label", "ctx_128_tps", "ctx_512_tps", "ctx_4096_tps",
        "ctx_16384_tps", "ctx_32768_tps", "ctx_65536_tps", "ctx_131072_tps",
        "guard_128_pass", "guard_512_pass", "guard_4k_pass", "guard_16k_pass", "guard_32k_pass",
        "guard_64k_pass", "guard_128k_pass") if k in guard}
    out["guard"]["speed_ok"] = speed_ok
    out["guard"]["accuracy_ok"] = acc_ok
    if not speed_ok:
        reasons = []
        if regressed:
            reasons.append(f"{guard_model} decode regressed at: " + ", ".join(regressed))
        if not guard:
            reasons.append(f"{guard_model} guard produced no verdict")
        out["label"] = "REJECT"
        out["pass"] = False
        out["reason"] = "no-regression guard: " + "; ".join(reasons or ["guard failed"])
        out["guard_regression_labels"] = [f"regression-{guard_prefix}-" + c for c in regressed]
    elif not acc_ok:
        reasons = []
        if not acc_ok and guard:
            reasons.append(f"{guard_model} accuracy broke (top1={guard.get('top1')}, kl={guard.get('kl')})")
        out["pass"] = False
        out["reason"] = "no-regression guard: " + "; ".join(reasons or ["guard failed"])
        # Keep primary speed tier (e.g. eval-qwen35:M) — guard accuracy fail is pass=false, not tier REJECT.
    return out

commit = os.environ["COMMIT"]
quant = os.environ.get("QUANT", "Q4_K_M")
q35_name = f"Qwythos-9B ({quant})"
q36_name = "Qwen3.6-35B-A3B"

score35 = merge_primary(load("PRIMARY35_JSON"), load("GUARD36_JSON"), q35_name, q36_name, "qwen36")
score36 = merge_primary(load("PRIMARY36_JSON"), load("GUARD35_JSON"), q36_name, q35_name, "qwen35")

def pick_best(a, b):
    ra, rb = TIER_RANK.get(a.get("label"), -1), TIER_RANK.get(b.get("label"), -1)
    if ra != rb:
        return a if ra > rb else b
    return a if float(a.get("tps") or 0) >= float(b.get("tps") or 0) else b

def pick_headline(a, b):
    """Headline verdict: either-side fail blocks merge; REJECT beats a passing XL/L/… from the other model."""
    if not a.get("pass") or not b.get("pass"):
        for s in (a, b):
            if s.get("infra_error"):
                continue
            if not s.get("pass") and s.get("label") == "REJECT":
                return dict(s)
        for s in (a, b):
            if s.get("infra_error"):
                continue
            if not s.get("pass"):
                return dict(s)
    return pick_best(a, b)

best = pick_headline(score35, score36)

final = dict(best)
final["commit"] = commit
final["mode"] = "bidir"
final["score_qwen35"] = score35
final["score_qwen36"] = score36
final["label_qwen35"] = score35.get("label")
final["label_qwen36"] = score36.get("label")
final["pass_qwen35"] = bool(score35.get("pass"))
final["pass_qwen36"] = bool(score36.get("pass"))
final["model"] = "bidir"
final["primary_quant"] = quant
final["regression_labels"] = list(dict.fromkeys(
    (score35.get("guard_regression_labels") or []) + (score36.get("guard_regression_labels") or [])))

print("RESULT_JSON " + json.dumps(final))
PY
