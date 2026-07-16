#!/usr/bin/env bash
# Triple-model evaluation: score Qwythos-9B (Qwen3.5-9B primary) and guard BOTH
# Qwen3.6-35B-A3B + Qwen3-30B-A3B against no-regression — one build, one box.
#
#   bench/scripts/evaluate_triple.sh [--ref GIT_REF] [--ceiling TPS]
#
# PRIMARY_QUANT: Q4_K_M (default) | Q8_0 | BF16
#
# Env (all optional):
#   SPARKINFER_P_GUARD_*     Qwythos same-box main tok/s (scored target guards)
#   SPARKINFER_P_LLAMA_*     Qwythos llama.cpp refs
#   SPARKINFER_G36_GUARD_*   Qwen3.6 guard baselines
#   SPARKINFER_G3_GUARD_*    Qwen3-30B guard baselines
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"
source "$HERE/_qwythos.sh"

REF=""; CEILING=0
while [ $# -gt 0 ]; do case "$1" in
  --ref) shift; REF="$1" ;; --ceiling) shift; CEILING="$1" ;; *) ;;
esac; shift; done

export LLAMACPP_DIR="${LLAMACPP_DIR:-/workspace/.llamacpp}"
ARCH="$(detect_arch)"

if [ -n "$REF" ] && [ -z "${SI_NO_CHECKOUT:-}" ]; then
  git -C "$ROOT" fetch -q origin "$REF" 2>/dev/null || true; git -C "$ROOT" checkout -q "$REF"
fi
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"

echo ">> [build] submission ($COMMIT) from source (sm_$ARCH) — shared by all models ..." >&2
rm -rf "$ROOT/build"
if ! NO_PREBUILT=1 ensure_sparkinfer "$ARCH"; then
  echo ">> build FAILED — submission does not compile (sm_$ARCH)" >&2
  printf 'RESULT_JSON {"commit": "%s", "tps": 0, "top1": 0, "kl": 99, "frontier_tps": 0, "label": "REJECT", "reason": "build failed (does not compile)", "pass": false}\n' "$COMMIT"
  exit 0
fi

P_FILE="$(qwythos_quant_file)"
P_REPO="${QWYTHOS_REPO}"
P_TOK="${QWYTHOS_TOK_REPO}"
P_DIR="${QWYTHOS_MODELS_DIR}"
P_SHA="$(qwythos_sha_var)"

G36_FILE="${GUARD36_MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
G36_REPO="${GUARD36_MODEL_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
G36_TOK="${GUARD36_TOK_REPO:-Qwen/Qwen3.6-35B-A3B}"
G36_DIR="${GUARD36_MODELS_DIR:-${MODELS_DIR:-$ROOT/models}36}"

G3_FILE="${GUARD3_MODEL_FILE:-Qwen3-30B-A3B-Q4_K_M.gguf}"
G3_REPO="${GUARD3_MODEL_REPO:-Qwen/Qwen3-30B-A3B-GGUF}"
G3_TOK="${GUARD3_TOK_REPO:-Qwen/Qwen3-30B-A3B}"
G3_DIR="${GUARD3_MODELS_DIR:-${MODELS_DIR:-$ROOT/models}}"

echo ">> primary: $P_FILE (quant=${PRIMARY_QUANT:-Q4_K_M})" >&2
echo ">> guards:  Qwen3.6=$G36_FILE  Qwen3-30B=$G3_FILE" >&2

reap() { pkill -f llama-server 2>/dev/null || true; pkill -f qwen3_gguf 2>/dev/null || true; sleep 1; true; }

run_model() {
  local role="$1" file="$2" repo="$3" tok="$4" frontier="$5"; shift 5
  reap
  echo ">> [$role] scoring model $file ..." >&2
  env SI_SKIP_BUILD=1 SI_NO_CHECKOUT=1 \
      MODEL_FILE="$file" MODEL_REPO="$repo" TOK_REPO="$tok" \
      "$@" \
      "$HERE/evaluate.sh" --ref "$REF" --frontier "$frontier" --ceiling "$CEILING" \
    | sed -n 's/^RESULT_JSON //p' | tail -1
}

# Auto-measure Qwythos same-box main if baselines unset.
if [ "${SPARKINFER_P_GUARD_128_BASELINE:-0}" = "0" ]; then
  echo ">> measuring Qwythos same-box main baseline (5-context sweep) ..." >&2
  P_GGUF="${P_DIR}/${P_FILE}"
  resolve_runner "$ARCH"
  for ctx in 0 512 4096 16384 32768; do
    t="$(si_run qwen3_gguf_bench "$P_GGUF" 128 "$ctx" 2>/dev/null | \
         sed -n 's/.*decode tg *: *\([0-9.][0-9.]*\).*/\1/p' | tail -1)"
    case "$ctx" in
      0)     P_128="${t:-0}" ;;
      512)   P_512="${t:-0}" ;;
      4096)  P_4K="${t:-0}" ;;
      16384) P_16K="${t:-0}" ;;
      32768) P_32K="${t:-0}" ;;
    esac
  done
  echo ">> Qwythos main: 128=${P_128:-0} 512=${P_512:-0} 4k=${P_4K:-0} 16k=${P_16K:-0} 32k=${P_32K:-0} tok/s" >&2
fi

PRIMARY_JSON="$(run_model primary "$P_FILE" "$P_REPO" "$P_TOK" 0 \
  MODELS_DIR="$P_DIR" MODEL_SHA256="${P_SHA}" \
  SPARKINFER_SCORE_REPS=1 SPARKINFER_GUARD_32K_REPS=1 \
  SPARKINFER_GUARD_REPS=1 SPARKINFER_GUARD_512_REPS=1 SPARKINFER_GUARD_4K_REPS=1 \
  SPARKINFER_DIFFICULTY_BOOST=1 SPARKINFER_DIFFICULTY_REF="${SPARKINFER_DIFFICULTY_REF_OVERRIDE:-365.85}" \
  SPARKINFER_GUARD_128_BASELINE="${P_128:-${SPARKINFER_P_GUARD_128_BASELINE:-0}}" \
  SPARKINFER_GUARD_512_BASELINE="${P_512:-${SPARKINFER_P_GUARD_512_BASELINE:-0}}" \
  SPARKINFER_GUARD_4K_BASELINE="${P_4K:-${SPARKINFER_P_GUARD_4K_BASELINE:-0}}" \
  SPARKINFER_GUARD_16K_BASELINE="${P_16K:-${SPARKINFER_P_GUARD_16K_BASELINE:-0}}" \
  SPARKINFER_GUARD_32K_BASELINE="${P_32K:-${SPARKINFER_P_GUARD_32K_BASELINE:-0}}" \
  SPARKINFER_LLAMA_128_BASELINE="${SPARKINFER_P_LLAMA_128_BASELINE:-${QWEN35_9B_LLAMA_128:-0}}" \
  SPARKINFER_LLAMA_512_BASELINE="${SPARKINFER_P_LLAMA_512_BASELINE:-${QWEN35_9B_LLAMA_512:-0}}" \
  SPARKINFER_LLAMA_4K_BASELINE="${SPARKINFER_P_LLAMA_4K_BASELINE:-${QWEN35_9B_LLAMA_4K:-0}}" \
  SPARKINFER_LLAMA_16K_BASELINE="${SPARKINFER_P_LLAMA_16K_BASELINE:-0}" \
  SPARKINFER_LLAMA_32K_BASELINE="${SPARKINFER_P_LLAMA_32K_BASELINE:-0}")"

GUARD36_JSON="$(run_model guard36 "$G36_FILE" "$G36_REPO" "$G36_TOK" 0 \
  MODELS_DIR="$G36_DIR" MODEL_SHA256="${QWEN36_MODEL_SHA256:-}" \
  SPARKINFER_GUARD_128_BASELINE="${SPARKINFER_G36_GUARD_128_BASELINE:-0}" \
  SPARKINFER_GUARD_512_BASELINE="${SPARKINFER_G36_GUARD_512_BASELINE:-0}" \
  SPARKINFER_GUARD_4K_BASELINE="${SPARKINFER_G36_GUARD_4K_BASELINE:-0}" \
  SPARKINFER_GUARD_16K_BASELINE="${SPARKINFER_G36_GUARD_16K_BASELINE:-0}" \
  SPARKINFER_GUARD_32K_BASELINE="${SPARKINFER_G36_GUARD_32K_BASELINE:-0}")"

GUARD3_JSON="$(run_model guard3 "$G3_FILE" "$G3_REPO" "$G3_TOK" 0 \
  MODELS_DIR="$G3_DIR" \
  SPARKINFER_GUARD_128_BASELINE="${SPARKINFER_G3_GUARD_128_BASELINE:-0}" \
  SPARKINFER_GUARD_512_BASELINE="${SPARKINFER_G3_GUARD_512_BASELINE:-0}" \
  SPARKINFER_GUARD_4K_BASELINE="${SPARKINFER_G3_GUARD_4K_BASELINE:-0}" \
  SPARKINFER_GUARD_16K_BASELINE="${SPARKINFER_G3_GUARD_16K_BASELINE:-0}" \
  SPARKINFER_GUARD_32K_BASELINE="${SPARKINFER_G3_GUARD_32K_BASELINE:-0}")"
reap

PRIMARY_JSON="$PRIMARY_JSON" GUARD36_JSON="$GUARD36_JSON" GUARD3_JSON="$GUARD3_JSON" \
COMMIT="$COMMIT" P_FILE="$P_FILE" G36_FILE="$G36_FILE" G3_FILE="$G3_FILE" \
PRIMARY_QUANT="${PRIMARY_QUANT:-Q4_K_M}" python3 - <<'PY'
import json, os

def load(name):
    raw = os.environ.get(name, "").strip()
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}

def guard_ok(guard, prefix):
    gctx = ["guard_128_pass", "guard_512_pass", "guard_4k_pass", "guard_16k_pass", "guard_32k_pass"]
    present = [k for k in gctx if k in guard]
    top1_bar = float(os.environ.get("SPARKINFER_TOP1_BAR", "0.90"))
    kl_bar = float(os.environ.get("SPARKINFER_KL_BAR", "0.20"))
    speed_ok = all(guard.get(k, True) for k in present)
    g_top1 = float(guard.get("top1", 0)); g_kl = float(guard.get("kl", 99))
    acc_ok = g_top1 >= top1_bar and g_kl <= kl_bar
    label_map = {"guard_128_pass": "128", "guard_512_pass": "512", "guard_4k_pass": "4k",
                 "guard_16k_pass": "16k", "guard_32k_pass": "32k"}
    regressed = [label_map[k] for k in present if not guard.get(k, True)]
    return bool(guard) and speed_ok and acc_ok, regressed, speed_ok, acc_ok

primary = load("PRIMARY_JSON")
g36 = load("GUARD36_JSON")
g3 = load("GUARD3_JSON")
commit = os.environ["COMMIT"]
quant = os.environ.get("PRIMARY_QUANT", "Q4_K_M")

if not primary:
    print("RESULT_JSON " + json.dumps({
        "commit": commit, "label": "REJECT", "pass": False,
        "reason": "primary (Qwythos-9B) eval produced no verdict (infra error)"}))
    raise SystemExit(0)

g36_ok, g36_reg, g36_speed, g36_acc = guard_ok(g36, "qwen36")
g3_ok, g3_reg, g3_speed, g3_acc = guard_ok(g3, "qwen3")

final = dict(primary)
final["commit"] = commit
final["model"] = f"Qwythos-9B ({quant})"
final["guard36_model"] = "Qwen3.6-35B-A3B"
final["guard3_model"] = "Qwen3-30B-A3B"
final["primary_quant"] = quant
final["guard36"] = {k: g36.get(k) for k in (
    "pass", "top1", "kl", "label", "ctx_128_tps", "ctx_512_tps", "ctx_4096_tps",
    "ctx_16384_tps", "ctx_32768_tps", "guard_128_pass", "guard_512_pass", "guard_4k_pass",
    "guard_16k_pass", "guard_32k_pass") if k in g36}
final["guard36"]["speed_ok"] = g36_speed
final["guard36"]["accuracy_ok"] = g36_acc
final["guard3"] = {k: g3.get(k) for k in (
    "pass", "top1", "kl", "label", "ctx_128_tps", "ctx_512_tps", "ctx_4096_tps",
    "ctx_16384_tps", "ctx_32768_tps", "guard_128_pass", "guard_512_pass", "guard_4k_pass",
    "guard_16k_pass", "guard_32k_pass") if k in g3}
final["guard3"]["speed_ok"] = g3_speed
final["guard3"]["accuracy_ok"] = g3_acc

if not (g36_ok and g3_ok):
    reasons = []
    if g36_reg:
        reasons.append("Qwen3.6 decode regressed at: " + ", ".join(g36_reg))
    if not g36_acc and g36:
        reasons.append(f"Qwen3.6 accuracy broke (top1={g36.get('top1')}, kl={g36.get('kl')})")
    if not bool(g36):
        reasons.append("Qwen3.6 guard produced no verdict")
    if g3_reg:
        reasons.append("Qwen3-30B decode regressed at: " + ", ".join(g3_reg))
    if not g3_acc and g3:
        reasons.append(f"Qwen3-30B accuracy broke (top1={g3.get('top1')}, kl={g3.get('kl')})")
    if not bool(g3):
        reasons.append("Qwen3-30B guard produced no verdict")
    final["label"] = "REJECT"
    final["pass"] = False
    final["reason"] = "no-regression guard: " + "; ".join(reasons or ["guard failed"])
    labels = []
    labels += ["regression-qwen36-" + c for c in g36_reg]
    labels += ["regression-qwen3-" + c for c in g3_reg]
    final["regression_labels"] = labels

print("RESULT_JSON " + json.dumps(final))
PY
