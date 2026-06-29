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

echo ">> [2/3] speed — median of 3 bench runs ..." >&2
# M1: pin the GPU clock so the absolute tok/s is reproducible (not just same-box-cancelled). Best-
# effort; reset on exit no matter how we leave. Warmup still runs as the fallback when pinning is
# refused, and to spin clocks up before the first timed build (the cold-clock artifact that once
# mislabeled minor PRs as XL above the ceiling).
pin_clocks
trap 'unpin_clocks' EXIT
si_run qwen3_gguf_bench "$GGUF" 192 >/dev/null 2>&1 || true
ts=(); gclks=()
for _ in 1 2 3; do
  t=$(si_run qwen3_gguf_bench "$GGUF" 128 2>/dev/null | sed -n 's/.*decode tg *: *\([0-9.][0-9.]*\).*/\1/p' || true)
  ts+=("${t:-0}")
  gclks+=("$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')")
done
TPS=$(printf '%s\n' "${ts[@]}" | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
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
PROV="{\"clocks_pinned\":$CP,\"clock_mhz\":\"${GCLK}\",\"clock_spread_mhz\":\"${GSPREAD}\",\"pin_target_mhz\":\"${PINNED_GCLK}\",\"eval_seed\":\"${EVAL_SEED}\",\"model_sha_pinned\":$MP,\"llama_commit\":\"${LLAMACPP_COMMIT:-unpinned}\"}"
python3 "$HERE/label.py" "$TPS" "$FRONTIER" "$CEILING" "$TOP1" "$KL" "$COMMIT" "$PROV"
