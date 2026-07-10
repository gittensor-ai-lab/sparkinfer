#!/usr/bin/env bash
# Automatic evaluation of a sparkinfer build: build → correctness → speed → label.
# Runs ON a GPU box (the vast orchestrator clones the repo + invokes this). Emits a JSON
# verdict as the last stdout line:  RESULT_JSON {...}
#
#   bench/scripts/evaluate.sh [--ref GIT_REF] [--frontier TPS] [--ceiling TPS] [--gguf PATH]
#
# correctness = token-match / KL vs llama.cpp (accuracy.sh) · speed = median of 3 bench runs
# · label = significance gate + headroom bucket (label.py). Source-built (NO_PREBUILT) so the
# measured artifact is the submitted code.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/_common.sh"

REF=""; FRONTIER=0; CEILING=0; GGUF=""
while [ $# -gt 0 ]; do case "$1" in
  --ref) shift; REF="$1" ;; --frontier) shift; FRONTIER="$1" ;;
  --ceiling) shift; CEILING="$1" ;; --gguf) shift; GGUF="$1" ;; *) ;;
esac; shift; done
[ -z "$GGUF" ] && GGUF="$MODELS_DIR/$MODEL_FILE"
export LLAMACPP_DIR="${LLAMACPP_DIR:-/workspace/.llamacpp}"   # persist llama.cpp across evals
# Single-model Qwen3.6 evals must pin the Qwen3.6 GGUF sha (evaluate_dual.sh does this per-model).
case "$MODEL_FILE" in
  *Qwen3.6*) MODEL_SHA256="${MODEL_SHA256:-${QWEN36_MODEL_SHA256:-}}" ;;
esac
ARCH="$(detect_arch)"

# Self-test convenience: check out the submitted ref. The bot pre-checks-out the submission and
# pins bench/scripts to the protected branch, then sets SI_NO_CHECKOUT=1 so this can't restore the
# submission's (untrusted) copy of the scoring harness over the trusted one.
if [ -n "$REF" ] && [ -z "${SI_NO_CHECKOUT:-}" ]; then
  git -C "$ROOT" fetch -q origin "$REF" 2>/dev/null || true; git -C "$ROOT" checkout -q "$REF"
fi
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"

# SI_SKIP_BUILD=1: the caller (evaluate_dual.sh) already built this exact tree from source and is
# now scoring a second model against the same binaries — skip the rebuild so a two-model eval pays
# the (model-agnostic) compile cost once, not twice.
if [ -n "${SI_SKIP_BUILD:-}" ] && [ -x "$ROOT/build/runtime/qwen3_gguf_bench" ]; then
  echo ">> [1/3] reusing pre-built submission ($COMMIT) (SI_SKIP_BUILD) ..." >&2
else
  echo ">> [1/3] build submission ($COMMIT) from source (sm_$ARCH) ..." >&2
  rm -rf "$ROOT/build"
  # A submission that does not compile is invalid -> clean REJECT (not an infra error). The `if !`
  # guard suppresses `set -e` for the build so we can emit a verdict instead of aborting silently.
  if ! NO_PREBUILT=1 ensure_sparkinfer "$ARCH"; then
    echo ">> build FAILED — submission does not compile (sm_$ARCH)" >&2
    printf 'RESULT_JSON {"commit": "%s", "tps": 0, "top1": 0, "kl": 99, "frontier_tps": %s, "label": "REJECT", "reason": "build failed (does not compile)", "pass": false}\n' "$COMMIT" "$FRONTIER"
    exit 0
  fi
fi
SI_BIN="$ROOT/build/runtime"; SI_LD=""

# One-time setup: download model (~17 GB) and build llama.cpp if not already cached.
# /workspace persists across vast stop/start; skipped on reuse.
ensure_model
ensure_llamacpp "$ARCH"

EVAL_MODE="${SPARKINFER_EVAL_MODE:-longctx}"
SCORE_CTX="${SPARKINFER_SCORE_CTX:-16384}"
GUARD_CTX="${SPARKINFER_GUARD_CTX:-0}"
GUARD_512_CTX="${SPARKINFER_GUARD_512_CTX:-512}"
GUARD_4K_CTX="${SPARKINFER_GUARD_4K_CTX:-4096}"
GUARD_32K_CTX="${SPARKINFER_GUARD_32K_CTX:-32768}"
DECODE_TOKENS="${SPARKINFER_DECODE_TOKENS:-128}"
SCORE_REPS="${SPARKINFER_SCORE_REPS:-3}"
GUARD_REPS="${SPARKINFER_GUARD_REPS:-1}"
GUARD_512_REPS="${SPARKINFER_GUARD_512_REPS:-1}"
GUARD_4K_REPS="${SPARKINFER_GUARD_4K_REPS:-1}"
GUARD_32K_REPS="${SPARKINFER_GUARD_32K_REPS:-1}"
GUARD_BASELINE="${SPARKINFER_GUARD_128_BASELINE:-${SPARKINFER_GUARD_2K_BASELINE:-0}}"
GUARD_512_BASELINE="${SPARKINFER_GUARD_512_BASELINE:-0}"
GUARD_4K_BASELINE="${SPARKINFER_GUARD_4K_BASELINE:-0}"
GUARD_16K_BASELINE="${SPARKINFER_GUARD_16K_BASELINE:-0}"
GUARD_32K_BASELINE="${SPARKINFER_GUARD_32K_BASELINE:-0}"
GUARD_TOL="${SPARKINFER_GUARD_128_TOL:-${SPARKINFER_GUARD_2K_TOL:-0.98}}"
GUARD_512_TOL="${SPARKINFER_GUARD_512_TOL:-$GUARD_TOL}"
GUARD_4K_TOL="${SPARKINFER_GUARD_4K_TOL:-$GUARD_TOL}"
GUARD_16K_TOL="${SPARKINFER_GUARD_16K_TOL:-$GUARD_TOL}"
GUARD_32K_TOL="${SPARKINFER_GUARD_32K_TOL:-$GUARD_TOL}"
LLAMA_128_BASELINE="${SPARKINFER_LLAMA_128_BASELINE:-365.85}"
LLAMA_512_BASELINE="${SPARKINFER_LLAMA_512_BASELINE:-342.59}"
LLAMA_4K_BASELINE="${SPARKINFER_LLAMA_4K_BASELINE:-292.99}"
LLAMA_16K_BASELINE="${SPARKINFER_LLAMA_16K_BASELINE:-245.53}"
LLAMA_32K_BASELINE="${SPARKINFER_LLAMA_32K_BASELINE:-192.62}"

echo ">> [2/3] speed — ${EVAL_MODE} decode benchmark ..." >&2
# M1: pin the GPU clock so the absolute tok/s is reproducible (not just same-box-cancelled). Best-
# effort; reset on exit no matter how we leave. Warmup still runs as the fallback when pinning is
# refused, and to spin clocks up before the first timed build (the cold-clock artifact that once
# mislabeled minor PRs as XL above the ceiling).
pin_clocks
trap 'unpin_clocks' EXIT

gclks=()
median_ctx() {  # $1=context tokens, $2=repetitions ; reps<=0 SKIPS the context (returns 0, no run)
  local ctx="$1" reps="$2" vals=() t
  [ "${reps:-0}" -le 0 ] && { echo 0; return; }
  for _ in $(seq 1 "$reps"); do
    t=$(si_run qwen3_gguf_bench "$GGUF" "$DECODE_TOKENS" "$ctx" 2>/dev/null |
        sed -n 's/.*decode tg *: *\([0-9.][0-9.]*\).*/\1/p' || true)
    vals+=("${t:-0}")
    gclks+=("$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')")
  done
  printf '%s\n' "${vals[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'
}

if [ "$EVAL_MODE" = "short" ]; then
  si_run qwen3_gguf_bench "$GGUF" 192 0 >/dev/null 2>&1 || true
  TPS="$(median_ctx 0 3)"
  GUARD_TPS=0; GUARD_512_TPS=0; GUARD_4K_TPS=0; GUARD_32K_TPS=0
  GUARD_PASS=true; GUARD_512_PASS=true; GUARD_4K_PASS=true; GUARD_16K_PASS=true; ALL_GUARDS_PASS=true
  GUARD_RATIO=0; GUARD_512_RATIO=0; GUARD_4K_RATIO=0; GUARD_16K_RATIO=0
  SELECTED_TPS="$TPS"; SELECTED_FRONTIER="$FRONTIER"; SELECTED_CTX=128
  SELECTED_CONTEXT_LABEL="128-context"; BEST_CONTEXT_LABEL="128-context"; SELECTED_LLAMA_REF="$LLAMA_128_BASELINE"
  CONTEXT_GAINS_JSON='{}'
else
  echo ">> context policy: ${DECODE_TOKENS}-token decode at 128/512/4k/16k/32k; all contexts guarded; best context scores" >&2
  si_run qwen3_gguf_bench "$GGUF" 64 "$GUARD_CTX" >/dev/null 2>&1 || true
  GUARD_TPS="$(median_ctx "$GUARD_CTX" "$GUARD_REPS")"
  GUARD_512_TPS="$(median_ctx "$GUARD_512_CTX" "$GUARD_512_REPS")"
  GUARD_4K_TPS="$(median_ctx "$GUARD_4K_CTX" "$GUARD_4K_REPS")"
  TPS="$(median_ctx "$SCORE_CTX" "$SCORE_REPS")"
  GUARD_32K_TPS="$(median_ctx "$GUARD_32K_CTX" "$GUARD_32K_REPS")"
  GUARD_RATIO="$(python3 - <<PY
base=float("$GUARD_BASELINE")
cur=float("$GUARD_TPS")
print(0 if base <= 0 else cur / base)
PY
)"
  GUARD_512_RATIO="$(python3 - <<PY
base=float("$GUARD_512_BASELINE")
cur=float("$GUARD_512_TPS")
print(0 if base <= 0 else cur / base)
PY
)"
  GUARD_4K_RATIO="$(python3 - <<PY
base=float("$GUARD_4K_BASELINE")
cur=float("$GUARD_4K_TPS")
print(0 if base <= 0 else cur / base)
PY
)"
  GUARD_16K_RATIO="$(python3 - <<PY
base=float("$GUARD_16K_BASELINE")
cur=float("$TPS")
print(0 if base <= 0 else cur / base)
PY
)"
  GUARD_32K_RATIO="$(python3 - <<PY
base=float("$GUARD_32K_BASELINE")
cur=float("$GUARD_32K_TPS")
print(0 if base <= 0 else cur / base)
PY
)"
  GUARD_PASS="$(python3 - <<PY
base=float("$GUARD_BASELINE")
cur=float("$GUARD_TPS")
tol=float("$GUARD_TOL")
print("true" if base <= 0 or cur >= base * tol else "false")
PY
)"
  GUARD_512_PASS="$(python3 - <<PY
base=float("$GUARD_512_BASELINE")
cur=float("$GUARD_512_TPS")
tol=float("$GUARD_512_TOL")
print("true" if base <= 0 or cur >= base * tol else "false")
PY
)"
  GUARD_4K_PASS="$(python3 - <<PY
base=float("$GUARD_4K_BASELINE")
cur=float("$GUARD_4K_TPS")
tol=float("$GUARD_4K_TOL")
print("true" if base <= 0 or cur >= base * tol else "false")
PY
)"
  GUARD_16K_PASS="$(python3 - <<PY
base=float("$GUARD_16K_BASELINE")
cur=float("$TPS")
tol=float("$GUARD_16K_TOL")
print("true" if base <= 0 or cur >= base * tol else "false")
PY
)"
  GUARD_32K_PASS="$(python3 - <<PY
base=float("$GUARD_32K_BASELINE")
cur=float("$GUARD_32K_TPS")
tol=float("$GUARD_32K_TOL")
print("true" if base <= 0 or cur >= base * tol else "false")
PY
)"
  SCORE_SELECT="$(python3 - <<PY
import json
contexts = [
  {"ctx":128, "label":"128-context", "tps":float("$GUARD_TPS"), "base":float("$GUARD_BASELINE"), "llama":float("$LLAMA_128_BASELINE")},
  {"ctx":512, "label":"512-context", "tps":float("$GUARD_512_TPS"), "base":float("$GUARD_512_BASELINE"), "llama":float("$LLAMA_512_BASELINE")},
  {"ctx":4096, "label":"4k-context", "tps":float("$GUARD_4K_TPS"), "base":float("$GUARD_4K_BASELINE"), "llama":float("$LLAMA_4K_BASELINE")},
  {"ctx":16384, "label":"16k-context", "tps":float("$TPS"), "base":float("$GUARD_16K_BASELINE") or float("$FRONTIER"), "llama":float("$LLAMA_16K_BASELINE")},
  {"ctx":32768, "label":"32k-context", "tps":float("$GUARD_32K_TPS"), "base":float("$GUARD_32K_BASELINE"), "llama":float("$LLAMA_32K_BASELINE")},
]
for c in contexts:
    # The scoring base is the SAME-BOX origin/main measurement — not a passed-in frontier number.
    # The bot already measures main on this box at the start of every run and passes it as the
    # GUARD_BASELINE. Each PR scores directly against that same-box baseline: the gain is "how
    # much faster is this PR than main on the same box?" — hardware-independent and always current.
    # The only exception is 16k, which falls back to the old FRONTIER if no baseline is set;
    # for models that skip 16k (reps=0), it never enters scoring anyway.
    c["gain"] = 0.0 if c["base"] <= 0 else (c["tps"] - c["base"]) / c["base"]
# A context measured with 0 reps (tps<=0) was intentionally skipped (e.g. Qwen3.6 runs only
# 128/512/4k for now) — exclude it from scoring so a skipped context is never chosen or penalized.
scorable = [c for c in contexts if c["base"] > 0 and c["tps"] > 0]
chosen = max(scorable, key=lambda c: c["gain"]) if scorable else next(c for c in contexts if c["ctx"] == int("$SCORE_CTX"))
print(json.dumps({"chosen": chosen, "contexts": contexts}, separators=(",", ":")))
PY
)"
  SELECTED_TPS="$(SCORE_SELECT="$SCORE_SELECT" python3 - <<'PY'
import json, os
print(json.loads(os.environ["SCORE_SELECT"])["chosen"]["tps"])
PY
)"
  SELECTED_FRONTIER="$(SCORE_SELECT="$SCORE_SELECT" python3 - <<'PY'
import json, os
print(json.loads(os.environ["SCORE_SELECT"])["chosen"]["base"])
PY
)"
  SELECTED_CTX="$(SCORE_SELECT="$SCORE_SELECT" python3 - <<'PY'
import json, os
print(json.loads(os.environ["SCORE_SELECT"])["chosen"]["ctx"])
PY
)"
  SELECTED_CONTEXT_LABEL="$(SCORE_SELECT="$SCORE_SELECT" python3 - <<'PY'
import json, os
print(json.loads(os.environ["SCORE_SELECT"])["chosen"]["label"])
PY
)"
  BEST_CONTEXT_LABEL="$SELECTED_CONTEXT_LABEL"
  SELECTED_LLAMA_REF="$(SCORE_SELECT="$SCORE_SELECT" python3 - <<'PY'
import json, os
print(json.loads(os.environ["SCORE_SELECT"])["chosen"]["llama"])
PY
)"
  SELECTED_GAIN="$(SCORE_SELECT="$SCORE_SELECT" python3 - <<'PY'
import json, os
print(json.loads(os.environ["SCORE_SELECT"])["chosen"]["gain"])
PY
)"
  CONTEXT_GAINS_JSON="$(SCORE_SELECT="$SCORE_SELECT" python3 - <<'PY'
import json, os
print(json.dumps({c["label"]: round(100*c["gain"], 2) for c in json.loads(os.environ["SCORE_SELECT"])["contexts"] if c["tps"] > 0}, separators=(",", ":")))
PY
)"
  REGRESSION_LABELS_JSON="$(python3 - <<PY
import json
labels = []
if "$GUARD_PASS" != "true": labels.append("regression-128")
if "$GUARD_512_PASS" != "true": labels.append("regression-512")
if "$GUARD_4K_PASS" != "true": labels.append("regression-4k")
if "$GUARD_16K_PASS" != "true": labels.append("regression-16k")
if "$GUARD_32K_PASS" != "true": labels.append("regression-32k")
print(json.dumps(labels, separators=(",", ":")))
PY
)"
  ALL_GUARDS_PASS="$(python3 - <<PY
print("true" if all(x == "true" for x in ["$GUARD_PASS", "$GUARD_512_PASS", "$GUARD_4K_PASS", "$GUARD_16K_PASS", "$GUARD_32K_PASS"]) else "false")
PY
)"
  HAS_VERIFIED_CONTEXT_GAIN="$(python3 - <<PY
print("true" if float("$SELECTED_GAIN") > 0.02 else "false")
PY
)"
fi
# M1: record the graphics clock the number was produced at — the reproducibility anchor. Equals the
# pin target where -lgc is permitted (bare-metal/datacenter); on a restricted container (vast lacks
# cap_sys_admin) it's the OBSERVED median, so the absolute tok/s stays interpretable and a verifier
# can confirm they reproduced at the same clock. clock_spread exposes how stable it was.
GCLK=$(printf '%s\n' "${gclks[@]}" | sort -n | awk 'NF{a[++n]=$1} END{print (n?a[int((n+1)/2)]:0)}')
GSPREAD=$(printf '%s\n' "${gclks[@]}" | sort -n | awk 'NF{a[++n]=$1} END{print (n?a[n]-a[1]:0)}')

echo ">> [3/3] correctness — token-match / KL vs llama.cpp (held-out prompt) ..." >&2
# H1: the accuracy gate scores a held-out / fuzzed prompt chosen by EVAL_SEED (set by the bot to a
# fresh, unpredictable value each eval), so a submission can't overfit the in-repo prompt. The seed
# is recorded below so any verifier reproduces the exact token stream.
EVAL_SEED="${SPARKINFER_EVAL_SEED:-fixed}"
acc=$(SPARKINFER_EVAL_SEED="$EVAL_SEED" "$HERE/accuracy.sh" "$GGUF" 2>/dev/null || true)
# parse the unambiguous METRIC line (not the human-readable text, which contains "bar >= 0.90")
TOP1=$(printf '%s\n' "$acc" | sed -n 's/.*METRIC .*top1=\([0-9.][0-9.]*\).*/\1/p' | head -1)
KL=$(printf   '%s\n' "$acc" | sed -n 's/.*METRIC .*kl=\([0-9.][0-9.]*\).*/\1/p' | head -1)
TOP1="${TOP1:-0}"; KL="${KL:-99}"

# Provenance merged into the verdict (M1 clock, H1 seed, C2 reference pins) — non-scoring, for the log.
[ "$GPU_CLOCKS_PINNED" = 1 ] && CP=true || CP=false
[ -n "${MODEL_SHA256:-}" ] && MP=true || MP=false
PROV="$(python3 - <<PY
import json, os
score_ctx = 128 if "$EVAL_MODE" == "short" else int("$SELECTED_CTX")
guard_ctx = 0 if "$EVAL_MODE" == "short" else int("$GUARD_CTX")
data = {
  "clocks_pinned": "$CP" == "true",
  "clock_mhz": "$GCLK",
  "clock_spread_mhz": "$GSPREAD",
  "pin_target_mhz": "$PINNED_GCLK",
  "eval_seed": "$EVAL_SEED",
  "model_sha_pinned": "$MP" == "true",
  "llama_commit": "${LLAMACPP_COMMIT:-unpinned}",
  "eval_mode": "$EVAL_MODE",
  "decode_tokens": int("$DECODE_TOKENS"),
  "score_context": score_ctx,
  "best_context_label": "$BEST_CONTEXT_LABEL",
  "context_gains_pct": json.loads('''$CONTEXT_GAINS_JSON'''),
  "regression_labels": json.loads('''${REGRESSION_LABELS_JSON:-[]}'''),
}
if "$EVAL_MODE" != "short":
  data.update({
    "guard_context": guard_ctx,
    "ctx_128_tps": round(float("$GUARD_TPS"), 2),
    "ctx_512_tps": round(float("$GUARD_512_TPS"), 2),
    "ctx_4096_tps": round(float("$GUARD_4K_TPS"), 2),
    "ctx_16384_tps": round(float("$TPS"), 2),
    "ctx_32768_tps": round(float("$GUARD_32K_TPS"), 2),
    "guard_128_baseline": round(float("$GUARD_BASELINE"), 2),
    "guard_128_ratio": round(float("$GUARD_RATIO"), 4),
    "guard_128_tol": float("$GUARD_TOL"),
    "guard_128_pass": "$GUARD_PASS" == "true",
    "guard_512_baseline": round(float("$GUARD_512_BASELINE"), 2),
    "guard_512_ratio": round(float("$GUARD_512_RATIO"), 4),
    "guard_512_tol": float("$GUARD_512_TOL"),
    "guard_512_pass": "$GUARD_512_PASS" == "true",
    "guard_4k_baseline": round(float("$GUARD_4K_BASELINE"), 2),
    "guard_4k_ratio": round(float("$GUARD_4K_RATIO"), 4),
    "guard_4k_tol": float("$GUARD_4K_TOL"),
    "guard_4k_pass": "$GUARD_4K_PASS" == "true",
    "guard_16k_baseline": round(float("$GUARD_16K_BASELINE"), 2),
    "guard_16k_ratio": round(float("$GUARD_16K_RATIO"), 4),
    "guard_16k_tol": float("$GUARD_16K_TOL"),
    "guard_16k_pass": "$GUARD_16K_PASS" == "true",
    "guard_32k_baseline": round(float("$GUARD_32K_BASELINE"), 2),
    "guard_32k_ratio": round(float("$GUARD_32K_RATIO"), 4),
    "guard_32k_tol": float("$GUARD_32K_TOL"),
    "guard_32k_pass": "$GUARD_32K_PASS" == "true",
  })
print(json.dumps(data, separators=(",", ":")))
PY
)"
if [ "$EVAL_MODE" != "short" ] && [ "$ALL_GUARDS_PASS" != "true" ] && [ "$HAS_VERIFIED_CONTEXT_GAIN" != "true" ]; then
  PROV="$PROV" python3 - <<PY
import json, os
tps=float("$SELECTED_TPS"); frontier=float("$SELECTED_FRONTIER"); guard=float("$GUARD_TPS")
base=float("$GUARD_BASELINE"); tol=float("$GUARD_TOL")
guard512=float("$GUARD_512_TPS"); base512=float("$GUARD_512_BASELINE"); tol512=float("$GUARD_512_TOL")
guard4k=float("$GUARD_4K_TPS"); base4k=float("$GUARD_4K_BASELINE"); tol4k=float("$GUARD_4K_TOL")
guard16k=float("$TPS"); base16k=float("$GUARD_16K_BASELINE"); tol16k=float("$GUARD_16K_TOL")
guard32k=float("$GUARD_32K_TPS"); base32k=float("$GUARD_32K_BASELINE"); tol32k=float("$GUARD_32K_TOL")
reasons = []
if base > 0 and guard < base * tol:
    reasons.append(f"128-token decode no-regression gate: {guard:.2f} tok/s < {tol:.0%} of main {base:.2f} tok/s")
if base512 > 0 and guard512 < base512 * tol512:
    reasons.append(f"512-context decode no-regression gate: {guard512:.2f} tok/s < {tol512:.0%} of main {base512:.2f} tok/s")
if base4k > 0 and guard4k < base4k * tol4k:
    reasons.append(f"4k-context decode no-regression gate: {guard4k:.2f} tok/s < {tol4k:.0%} of main {base4k:.2f} tok/s")
if base16k > 0 and guard16k < base16k * tol16k:
    reasons.append(f"16k-context decode no-regression gate: {guard16k:.2f} tok/s < {tol16k:.0%} of main {base16k:.2f} tok/s")
if base32k > 0 and guard32k < base32k * tol32k:
    reasons.append(f"32k-context decode no-regression gate: {guard32k:.2f} tok/s < {tol32k:.0%} of main {base32k:.2f} tok/s")
res = {
  "commit": "$COMMIT",
  "tps": round(tps, 2),
  "top1": round(float("$TOP1"), 4),
  "kl": round(float("$KL"), 4),
  "frontier_tps": round(frontier, 2),
  "label": "REJECT",
  "pass": False,
  "reason": "; ".join(reasons) or "decode no-regression guard failed",
  "auto_close": True,
}
if frontier > 0:
  res["delta_tps"] = round(tps - frontier, 2)
  res["pct_over_frontier"] = round(100 * (tps - frontier) / frontier, 1)
res.update(json.loads(os.environ["PROV"]))
print("RESULT_JSON " + json.dumps(res))
PY
  exit 0
fi
SPARKINFER_DIFFICULTY_REF="$SELECTED_LLAMA_REF" python3 "$HERE/label.py" "$SELECTED_TPS" "$SELECTED_FRONTIER" "$CEILING" "$TOP1" "$KL" "$COMMIT" "$PROV"
