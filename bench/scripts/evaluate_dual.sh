#!/usr/bin/env bash
# Dual-model evaluation: score Qwen3.6-35B-A3B (the primary target) and guard
# Qwen3-30B-A3B against no-regression — in one build, on one box.
#
#   bench/scripts/evaluate_dual.sh [--ref GIT_REF] [--ceiling TPS] [--primary-frontier TPS]
#
# Builds the submission ONCE (the build is model-agnostic), then:
#   • PRIMARY  (Qwen3.6): full speed sweep + token-match/KL vs llama.cpp -> the scored eval:<label>.
#   • GUARD    (Qwen3-30B): the same speed sweep + accuracy gate, but never scored — it only has to
#                           NOT regress. If Qwen3 loses speed at any context OR breaks parity with
#                           llama.cpp, the whole submission is REJECTed (an optimization that helps
#                           Qwen3.6 must not quietly wreck the shipped Qwen3 path).
#
# Final verdict = the Qwen3.6 label, forced to REJECT if the Qwen3 guard fails. Both models'
# measurements are merged into one RESULT_JSON so the record is self-describing.
#
# Model / baseline selection (env, all optional — sane Qwen3.6/Qwen3 defaults):
#   PRIMARY_MODEL_FILE/REPO/TOK, GUARD_MODEL_FILE/REPO/TOK
#   SPARKINFER_P_GUARD_{128,512,4K,16K,32K}_BASELINE   Qwen3.6 same-box main tok/s (its guards)
#   SPARKINFER_P_LLAMA_{128,512,4K,16K,32K}_BASELINE   Qwen3.6 llama.cpp tok/s (display + difficulty ref)
#   SPARKINFER_G_GUARD_{128,512,4K,16K,32K}_BASELINE   Qwen3-30B same-box main tok/s (its guards)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/_common.sh"

REF=""; CEILING=0
while [ $# -gt 0 ]; do case "$1" in
  --ref) shift; REF="$1" ;; --ceiling) shift; CEILING="$1" ;; *) ;;
esac; shift; done

export LLAMACPP_DIR="${LLAMACPP_DIR:-/workspace/.llamacpp}"
ARCH="$(detect_arch)"

# Same trust model as evaluate.sh: the bot pre-checks-out the submission + pins bench/scripts to the
# protected branch and sets SI_NO_CHECKOUT=1. A manual self-test with --ref checks it out here.
if [ -n "$REF" ] && [ -z "${SI_NO_CHECKOUT:-}" ]; then
  git -C "$ROOT" fetch -q origin "$REF" 2>/dev/null || true; git -C "$ROOT" checkout -q "$REF"
fi
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"

# --- build ONCE (model-agnostic) ; a non-compiling submission is a clean REJECT ------------------
echo ">> [build] submission ($COMMIT) from source (sm_$ARCH) — shared by both models ..." >&2
rm -rf "$ROOT/build"
if ! NO_PREBUILT=1 ensure_sparkinfer "$ARCH"; then
  echo ">> build FAILED — submission does not compile (sm_$ARCH)" >&2
  printf 'RESULT_JSON {"commit": "%s", "tps": 0, "top1": 0, "kl": 99, "frontier_tps": 0, "label": "REJECT", "reason": "build failed (does not compile)", "pass": false}\n' "$COMMIT"
  exit 0
fi

# Primary = Qwen3.6 (Unsloth Dynamic UD-Q4_K_M — mixes Q5_K, the default UD path). Guard = Qwen3-30B.
# The two models have DIFFERENT tokenizers (Qwen3.6 vocab 248k vs Qwen3 152k), and each model's dir
# holds its own GGUF + tokenizer.json — so they must live in SEPARATE MODELS_DIRs or accuracy.sh would
# score one model's ids against the other's tokenizer. Guard keeps the inherited MODELS_DIR.
P_FILE="${PRIMARY_MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
P_REPO="${PRIMARY_MODEL_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
P_TOK="${PRIMARY_TOK_REPO:-Qwen/Qwen3.6-35B-A3B}"
P_DIR="${PRIMARY_MODELS_DIR:-${MODELS_DIR:-$ROOT/models}36}"    # e.g. /workspace/models -> /workspace/models36
G_FILE="${GUARD_MODEL_FILE:-Qwen3-30B-A3B-Q4_K_M.gguf}"
G_REPO="${GUARD_MODEL_REPO:-Qwen/Qwen3-30B-A3B-GGUF}"
G_TOK="${GUARD_TOK_REPO:-Qwen/Qwen3-30B-A3B}"
G_DIR="${GUARD_MODELS_DIR:-${MODELS_DIR:-$ROOT/models}}"

reap() { pkill -f llama-server 2>/dev/null || true; pkill -f qwen3_gguf 2>/dev/null || true; sleep 1; true; }

run_model() {  # $1=role  $2=file $3=repo $4=tok  $5=frontier  $6..=SPARKINFER_* baseline assignments
  local role="$1" file="$2" repo="$3" tok="$4" frontier="$5"; shift 5
  reap
  echo ">> [$role] scoring model $file ..." >&2
  # SI_SKIP_BUILD reuses the single build; SI_NO_CHECKOUT keeps the trusted-harness pin intact.
  env SI_SKIP_BUILD=1 SI_NO_CHECKOUT=1 \
      MODEL_FILE="$file" MODEL_REPO="$repo" TOK_REPO="$tok" \
      "$@" \
      "$HERE/evaluate.sh" --ref "$REF" --frontier "$frontier" --ceiling "$CEILING" \
    | sed -n 's/^RESULT_JSON //p' | tail -1
}

# Difficulty anchor = Qwen3-30B llama.cpp 128-tok reference (365.85 default), NOT Qwen3.6's.
# Qwen3.6 is artificially slow in llama.cpp (~275 tok/s, not yet optimized there), so using
# its llama ref would make the frontier look "past-llama" prematurely and inflate difficulty_mult.
# Qwen3-30B is the well-optimized reference model — use its llama.cpp speed as the maturity anchor.
# To override: set SPARKINFER_DIFFICULTY_REF_OVERRIDE in the environment.
P_DIFF_REF="${SPARKINFER_DIFFICULTY_REF_OVERRIDE:-}"

# When the Qwen3.6 guard baselines aren't pre-set (bot passes 0), measure Qwen3.6 main
# speed directly on the box — a quick 5-context decode sweep against the already-built
# origin/main (128/512/4k/16k/32k). This auto-calibrates every bot run so the scoring base never goes stale.
if [ "${SPARKINFER_P_GUARD_128_BASELINE:-0}" = "0" ] || \
   [ "${SPARKINFER_P_GUARD_512_BASELINE:-0}" = "0" ] || \
   [ "${SPARKINFER_P_GUARD_4K_BASELINE:-0}"  = "0" ] || \
   [ "${SPARKINFER_P_GUARD_16K_BASELINE:-0}" = "0" ] || \
   [ "${SPARKINFER_P_GUARD_32K_BASELINE:-0}" = "0" ]; then
  echo ">> measuring Qwen3.6 same-box main baseline (5-context sweep) ..." >&2
  P36_GGUF="${P_DIR}/${P_FILE}"
  SI_BIN="$ROOT/build/runtime"   # ensure si_run finds the already-built binaries
  [ -f "$P36_GGUF" ] || P36_GGUF="${MODELS_DIR:-$ROOT/models}/${P_FILE}"
  SI_BIN="$ROOT/build/runtime"   # ensure si_run finds the already-built binaries
  for ctx in 0 512 4096 16384 32768; do
    t="$(si_run qwen3_gguf_bench "$P36_GGUF" 128 "$ctx" 2>/dev/null | \
         sed -n 's/.*decode tg *: *\([0-9.][0-9.]*\).*/\1/p' | tail -1)"
    case "$ctx" in
      0)     P36_128="${t:-0}" ;;
      512)   P36_512="${t:-0}" ;;
      4096)  P36_4K="${t:-0}" ;;
      16384) P36_16K="${t:-0}" ;;
      32768) P36_32K="${t:-0}" ;;
    esac
  done
  echo ">> Qwen3.6 same-box main: 128=${P36_128} 512=${P36_512} 4k=${P36_4K} 16k=${P36_16K} 32k=${P36_32K} tok/s" >&2
fi
P36_128="${P36_128:-0}"
P36_512="${P36_512:-0}"
P36_4K="${P36_4K:-0}"
P36_16K="${P36_16K:-0}"
P36_32K="${P36_32K:-0}"
# Fall back to the bot's config defaults if measurement produced 0.
# Two-step: try the env var first; if it is also unset/empty/"0", use the hardcoded default.
# A single ${VAR:-default} doesn't work because the bot passes VAR=0 explicitly, and
# "0" is a non-empty string so the :- expansion never fires (same pitfall as P36_* above).
[ "${P36_128}" = "0" ] && P36_128="${SPARKINFER_P_GUARD_128_BASELINE}"
[ "${P36_128}" = "0" ] && P36_128="300.16"
[ "${P36_512}" = "0" ] && P36_512="${SPARKINFER_P_GUARD_512_BASELINE}"
[ "${P36_512}" = "0" ] && P36_512="296.76"
[ "${P36_4K}"  = "0" ] && P36_4K="${SPARKINFER_P_GUARD_4K_BASELINE}"
[ "${P36_4K}"  = "0" ] && P36_4K="287.91"
[ "${P36_16K}" = "0" ] && P36_16K="${SPARKINFER_P_GUARD_16K_BASELINE}"
[ "${P36_16K}" = "0" ] && P36_16K="338.55"
[ "${P36_32K}" = "0" ] && P36_32K="${SPARKINFER_P_GUARD_32K_BASELINE}"
[ "${P36_32K}" = "0" ] && P36_32K="301.19"

PRIMARY_JSON="$(run_model primary "$P_FILE" "$P_REPO" "$P_TOK" 0 \
  MODELS_DIR="$P_DIR" MODEL_SHA256="${QWEN36_MODEL_SHA256:-}" \
  SPARKINFER_SCORE_REPS=3 SPARKINFER_GUARD_32K_REPS=1 \
  SPARKINFER_GUARD_REPS=3 SPARKINFER_GUARD_512_REPS=3 SPARKINFER_GUARD_4K_REPS=3 \
  SPARKINFER_DIFFICULTY_BOOST=1 SPARKINFER_DIFFICULTY_REF="${P_DIFF_REF:-365.85}" \
  SPARKINFER_GUARD_128_BASELINE="${P36_128}" \
  SPARKINFER_GUARD_512_BASELINE="${P36_512}" \
  SPARKINFER_GUARD_4K_BASELINE="${P36_4K}" \
  SPARKINFER_GUARD_16K_BASELINE="${P36_16K}" \
  SPARKINFER_GUARD_32K_BASELINE="${P36_32K}" \
  SPARKINFER_LLAMA_128_BASELINE="${SPARKINFER_P_LLAMA_128_BASELINE:-0}" \
  SPARKINFER_LLAMA_512_BASELINE="${SPARKINFER_P_LLAMA_512_BASELINE:-0}" \
  SPARKINFER_LLAMA_4K_BASELINE="${SPARKINFER_P_LLAMA_4K_BASELINE:-0}" \
  SPARKINFER_LLAMA_16K_BASELINE="${SPARKINFER_P_LLAMA_16K_BASELINE:-280.66}" \
  SPARKINFER_LLAMA_32K_BASELINE="${SPARKINFER_P_LLAMA_32K_BASELINE:-279.83}")"

# Guard runs at frontier 0 (never scored). Its main-branch baselines make every context a
# no-regression gate; the merge below fails the submission if any gate or the accuracy gate breaks.
GUARD_JSON="$(run_model guard "$G_FILE" "$G_REPO" "$G_TOK" 0 \
  MODELS_DIR="$G_DIR" \
  SPARKINFER_GUARD_128_BASELINE="${SPARKINFER_G_GUARD_128_BASELINE:-0}" \
  SPARKINFER_GUARD_512_BASELINE="${SPARKINFER_G_GUARD_512_BASELINE:-0}" \
  SPARKINFER_GUARD_4K_BASELINE="${SPARKINFER_G_GUARD_4K_BASELINE:-0}" \
  SPARKINFER_GUARD_16K_BASELINE="${SPARKINFER_G_GUARD_16K_BASELINE:-0}" \
  SPARKINFER_GUARD_32K_BASELINE="${SPARKINFER_G_GUARD_32K_BASELINE:-0}")"
reap

# --- merge: final = Qwen3.6 verdict, REJECTed if the Qwen3 guard regressed (speed OR accuracy) ----
PRIMARY_JSON="$PRIMARY_JSON" GUARD_JSON="$GUARD_JSON" COMMIT="$COMMIT" \
P_FILE="$P_FILE" G_FILE="$G_FILE" python3 - <<'PY'
import json, os

def load(name):
    raw = os.environ.get(name, "").strip()
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}

primary = load("PRIMARY_JSON")
guard   = load("GUARD_JSON")
commit  = os.environ["COMMIT"]
TOP1_BAR = float(os.environ.get("SPARKINFER_TOP1_BAR", "0.90"))
KL_BAR   = float(os.environ.get("SPARKINFER_KL_BAR",   "0.20"))

# An empty/garbled sub-verdict is an infra failure, not a pass — fail closed.
if not primary:
    print("RESULT_JSON " + json.dumps({
        "commit": commit, "label": "REJECT", "pass": False,
        "reason": "primary (Qwen3.6) eval produced no verdict (infra error)"}))
    raise SystemExit(0)

# Guard verdict: strict — EVERY measured context must hold >= tol of Qwen3 main, and Qwen3 must keep
# llama.cpp parity. Read the measured per-context pass flags directly (not the guard's own lenient
# "one win excuses a regression" scoring path, which we deliberately bypass for the guard role).
gctx = ["guard_128_pass", "guard_512_pass", "guard_4k_pass", "guard_16k_pass", "guard_32k_pass"]
present = [k for k in gctx if k in guard]
speed_ok = all(guard.get(k, True) for k in present)
g_top1 = float(guard.get("top1", 0)); g_kl = float(guard.get("kl", 99))
acc_ok = g_top1 >= TOP1_BAR and g_kl <= KL_BAR
guard_ok = bool(guard) and speed_ok and acc_ok

# Which Qwen3 contexts regressed (for the reason + labels).
label_map = {"guard_128_pass": "128", "guard_512_pass": "512", "guard_4k_pass": "4k",
             "guard_16k_pass": "16k", "guard_32k_pass": "32k"}
regressed = [label_map[k] for k in present if not guard.get(k, True)]

final = dict(primary)                       # carry the Qwen3.6 label, metrics, provenance verbatim
final["commit"] = commit
final["model"] = "Qwen3.6-35B-A3B"
final["guard_model"] = "Qwen3-30B-A3B"
final["guard"] = {k: guard.get(k) for k in (
    "pass", "top1", "kl", "label", "ctx_128_tps", "ctx_512_tps", "ctx_4096_tps",
    "ctx_16384_tps", "ctx_32768_tps", "guard_128_pass", "guard_512_pass", "guard_4k_pass",
    "guard_16k_pass", "guard_32k_pass") if k in guard}
final["guard"]["speed_ok"] = speed_ok
final["guard"]["accuracy_ok"] = acc_ok

if not guard_ok:
    reasons = []
    if regressed:
        reasons.append("Qwen3-30B decode regressed at context(s): " + ", ".join(regressed))
    if not acc_ok:
        reasons.append(f"Qwen3-30B accuracy broke (top1={g_top1} need>={TOP1_BAR}, kl={g_kl} need<={KL_BAR})")
    if not bool(guard):
        reasons.append("Qwen3-30B guard produced no verdict (infra error)")
    if not reasons:
        reasons.append("Qwen3-30B no-regression guard failed")
    final["label"] = "REJECT"
    final["pass"] = False
    final["reason"] = "no-regression guard: " + "; ".join(reasons)
    final["guard_regression_labels"] = ["regression-qwen3-" + c for c in regressed]

print("RESULT_JSON " + json.dumps(final))
PY
