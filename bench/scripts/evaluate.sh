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
ARCH="$(detect_arch)"

# Self-test convenience: check out the submitted ref. The bot pre-checks-out the submission and
# pins bench/scripts to the protected branch, then sets SI_NO_CHECKOUT=1 so this can't restore the
# submission's (untrusted) copy of the scoring harness over the trusted one.
if [ -n "$REF" ] && [ -z "${SI_NO_CHECKOUT:-}" ]; then
  git -C "$ROOT" fetch -q origin "$REF" 2>/dev/null || true; git -C "$ROOT" checkout -q "$REF"
fi
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"

echo ">> [1/3] build submission ($COMMIT) from source (sm_$ARCH) ..." >&2
rm -rf "$ROOT/build"
# A submission that does not compile is invalid -> clean REJECT (not an infra error). The `if !`
# guard suppresses `set -e` for the build so we can emit a verdict instead of aborting silently.
if ! NO_PREBUILT=1 ensure_sparkinfer "$ARCH"; then
  echo ">> build FAILED — submission does not compile (sm_$ARCH)" >&2
  printf 'RESULT_JSON {"commit": "%s", "tps": 0, "top1": 0, "kl": 99, "frontier_tps": %s, "label": "REJECT", "reason": "build failed (does not compile)", "pass": false}\n' "$COMMIT" "$FRONTIER"
  exit 0
fi
SI_BIN="$ROOT/build/runtime"; SI_LD=""

# One-time setup: download model (~17 GB) and build llama.cpp if not already cached.
# /workspace persists across vast stop/start; skipped on reuse.
ensure_model
ensure_llamacpp "$ARCH"

EVAL_MODE="${SPARKINFER_EVAL_MODE:-longctx}"
SCORE_CTX="${SPARKINFER_SCORE_CTX:-16384}"
GUARD_CTX="${SPARKINFER_GUARD_CTX:-2048}"
REPORT_CTX="${SPARKINFER_REPORT_CTX:-32768}"
DECODE_TOKENS="${SPARKINFER_DECODE_TOKENS:-128}"
SCORE_REPS="${SPARKINFER_SCORE_REPS:-3}"
GUARD_REPS="${SPARKINFER_GUARD_REPS:-1}"
REPORT_REPS="${SPARKINFER_REPORT_REPS:-1}"
GUARD_BASELINE="${SPARKINFER_GUARD_2K_BASELINE:-0}"
GUARD_TOL="${SPARKINFER_GUARD_2K_TOL:-0.98}"

echo ">> [2/3] speed — ${EVAL_MODE} decode benchmark ..." >&2
# M1: pin the GPU clock so the absolute tok/s is reproducible (not just same-box-cancelled). Best-
# effort; reset on exit no matter how we leave. Warmup still runs as the fallback when pinning is
# refused, and to spin clocks up before the first timed build (the cold-clock artifact that once
# mislabeled minor PRs as XL above the ceiling).
pin_clocks
trap 'unpin_clocks' EXIT

gclks=()
median_ctx() {  # $1=context tokens, $2=repetitions
  local ctx="$1" reps="$2" vals=() t
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
  GUARD_TPS=0; REPORT_TPS=0; GUARD_PASS=true; GUARD_RATIO=0
else
  echo ">> long-context policy: ${GUARD_CTX} ctx no-regression gate; ${SCORE_CTX} ctx scored; ${REPORT_CTX} ctx telemetry" >&2
  si_run qwen3_gguf_bench "$GGUF" 64 "$GUARD_CTX" >/dev/null 2>&1 || true
  GUARD_TPS="$(median_ctx "$GUARD_CTX" "$GUARD_REPS")"
  TPS="$(median_ctx "$SCORE_CTX" "$SCORE_REPS")"
  REPORT_TPS="$(median_ctx "$REPORT_CTX" "$REPORT_REPS")"
  GUARD_RATIO="$(python3 - <<PY
base=float("$GUARD_BASELINE")
cur=float("$GUARD_TPS")
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
import json
score_ctx = 128 if "$EVAL_MODE" == "short" else int("$SCORE_CTX")
guard_ctx = 0 if "$EVAL_MODE" == "short" else int("$GUARD_CTX")
report_ctx = 0 if "$EVAL_MODE" == "short" else int("$REPORT_CTX")
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
}
if "$EVAL_MODE" != "short":
  data.update({
    "guard_context": guard_ctx,
    "report_context": report_ctx,
    "ctx_2048_tps": round(float("$GUARD_TPS"), 2),
    "ctx_16384_tps": round(float("$TPS"), 2),
    "ctx_32768_tps": round(float("$REPORT_TPS"), 2),
    "guard_2k_baseline": round(float("$GUARD_BASELINE"), 2),
    "guard_2k_ratio": round(float("$GUARD_RATIO"), 4),
    "guard_2k_tol": float("$GUARD_TOL"),
    "guard_2k_pass": "$GUARD_PASS" == "true",
  })
print(json.dumps(data, separators=(",", ":")))
PY
)"
if [ "$EVAL_MODE" != "short" ] && [ "$GUARD_PASS" != "true" ]; then
  PROV="$PROV" python3 - <<PY
import json, os
tps=float("$TPS"); frontier=float("$FRONTIER"); guard=float("$GUARD_TPS")
base=float("$GUARD_BASELINE"); tol=float("$GUARD_TOL")
res = {
  "commit": "$COMMIT",
  "tps": round(tps, 2),
  "top1": round(float("$TOP1"), 4),
  "kl": round(float("$KL"), 4),
  "frontier_tps": round(frontier, 2),
  "label": "REJECT",
  "pass": False,
  "reason": f"2k no-regression gate: {guard:.2f} tok/s < {tol:.0%} of main {base:.2f} tok/s",
}
if frontier > 0:
  res["delta_tps"] = round(tps - frontier, 2)
  res["pct_over_frontier"] = round(100 * (tps - frontier) / frontier, 1)
res.update(json.loads(os.environ["PROV"]))
print("RESULT_JSON " + json.dumps(res))
PY
  exit 0
fi
python3 "$HERE/label.py" "$TPS" "$FRONTIER" "$CEILING" "$TOP1" "$KL" "$COMMIT" "$PROV"
