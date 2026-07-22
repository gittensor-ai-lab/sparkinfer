#!/usr/bin/env bash
# Turnkey accuracy gate: token-match / KL / perplexity of sparkinfer vs llama.cpp on
# the SAME GGUF (teacher-forced over a fixed text). Builds whatever is missing.
#
#   bench/scripts/accuracy.sh [--download | <model.gguf>] [--text FILE]
#
# Env overrides: MODELS_DIR, MODEL_FILE, ARCH, LLAMACPP_DIR.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_common.sh"

GGUF=""; TEXT="$HERE/eval_text.txt"
while [ $# -gt 0 ]; do case "$1" in
  --download) GGUF="$MODELS_DIR/$MODEL_FILE" ;;
  --text)     shift; TEXT="$1" ;;
  -h|--help)  sed -n '2,8p' "$0"; exit 0 ;;
  *)          GGUF="$1" ;;
esac; shift; done
[ -z "$GGUF" ] && GGUF="$MODELS_DIR/$MODEL_FILE"

ARCH="$(detect_arch)"
resolve_runner "$ARCH"     # prebuilt binaries if available, else build from source
[ "$GGUF" = "$MODELS_DIR/$MODEL_FILE" ] && ensure_model
ensure_tokenizer
ensure_llamacpp "$ARCH"
[ -f "$GGUF" ] || { echo "!! GGUF not found: $GGUF"; exit 1; }

# H1: the prompt is held-out / fuzzed by EVAL_SEED (set by the eval bot to a fresh, unpredictable
# value each run) so a submission can't overfit the in-repo text. seed="fixed" (the default for a
# manual run) reproduces the legacy fixed prompt. The exact ids are written once and fed to BOTH
# sparkinfer and llama, so they score the identical sequence; the seed is logged for reproduction.
SEED="${SPARKINFER_EVAL_SEED:-fixed}"
IDS_FILE="/tmp/eval_ids.txt"
python3 "$HERE/gen_eval_prompt.py" "$SEED" "$MODELS_DIR/tokenizer.json" "$HERE/eval_corpus.txt" "$TEXT" > "$IDS_FILE"
IDS="$(cat "$IDS_FILE")"
echo ">> eval prompt: seed=$SEED tokens=$(echo "$IDS" | wc -w)"

# H2 (long-context probe): the short prompt above is ~200-360 tokens, so qwen3_gguf_score leaves
# int8 KV OFF and the gate scores the bf16 tile path — while bench.cpp/model_engine.cpp turn int8 KV
# ON at every ctx >= 4096. The gate therefore never executed the kernels the tok/s numbers and the
# server come from, and four correctness defects reached main through that gap (#300, #388, #393,
# #517), all int8-path-specific.
#
# This second pass scores the SHIPPED config — int8 KV on, >= LONG_N held-out tokens — against the
# SAME llama.cpp reference, over the last LONG_TAIL positions (all past the engagement thresholds):
#   int8-MMA flash-decode : chunk >= 32 at n_splits=160 -> seqlen >= 4961
#   sparse KV (Qwythos)   : seqlen >= sparse_min_ctx (8192)
# llama is queried once per scored position, so we score a TAIL window (long prefix kept warm by
# cache_prompt) rather than all LONG_N positions.
#
# This second pass scores the SHIPPED config — int8 KV on, >= LONG_N held-out tokens — against the
# SAME llama.cpp reference, over the last LONG_TAIL positions (all past the engagement thresholds):
#   int8-MMA flash-decode : chunk >= 32 at n_splits=160 -> seqlen >= 4961
#   sparse KV (Qwythos)   : seqlen >= sparse_min_ctx (8192)
# llama is queried once per scored position, so we score a TAIL window (long prefix kept warm by
# cache_prompt) rather than all LONG_N positions.
#
# It runs LONG_SEEDS independent held-out streams (derived from the eval SEED) and AVERAGES their
# top-1 and KL, then VETOES on the mean: mean_top1 < LONG_TOP1_BAR OR mean_kl > LONG_KL_BAR. The bars
# are deliberately NOT the short pass's — int8 KV is lossy by construction, so a CORRECT int8 build
# diverges from llama more than bf16 does. They come from a 27-seed sweep on RTX 5090 (Qwen3.6, 8k,
# tail-128), across all 3-seed combinations of 16 correct (#517-fixed) and 11 broken (#517-live)
# builds:
#     metric   correct 3-seed range   broken 3-seed range   bar
#     top-1     [0.870, 0.974]         [0.568, 0.786]        >=0.875
#     KL        [0.215, 1.367]         [1.444, 2.461]        <=1.0
# Single-seed the two KL distributions OVERLAP (correct up to 1.66, broken down to 1.25) and no KL
# bar works — averaging LONG_SEEDS is what opens the gap. The KL bar (1.0) is far looser than the
# short pass's (0.20) precisely to allow int8's honest quantization loss: reading that loss as
# "corruption" and tightening the bar is what turned this probe off in #227 and created the blind
# spot. In this data top-1 alone already separates cleanly (broken tops out at 0.786), so the KL
# veto is defense-in-depth against a future bug that spares top-1 but wrecks KL. Reject on EITHER.
# All bars, the seed count, window, and token length are env-overridable.
LONGCTX="${SPARKINFER_EVAL_LONGCTX:-1}"
LONG_N="${SPARKINFER_EVAL_LONGCTX_TOKENS:-8448}"
LONG_TAIL="${SPARKINFER_EVAL_LONGCTX_TAIL:-128}"
LONG_SEEDS="${SPARKINFER_EVAL_LONGCTX_SEEDS:-3}"
LONG_TOP1_BAR="${SPARKINFER_EVAL_LONGCTX_TOP1_BAR:-0.875}"
LONG_KL_BAR="${SPARKINFER_EVAL_LONGCTX_KL_BAR:-1.0}"
if [ "$LONGCTX" = "1" ]; then
  for j in $(seq 0 $((LONG_SEEDS - 1))); do
    python3 "$HERE/gen_eval_prompt.py" "${SEED}:L${j}" "$MODELS_DIR/tokenizer.json" "$HERE/eval_corpus.txt" \
            --len "$LONG_N" > "/tmp/eval_ids_long_${j}.txt"
  done
  echo ">> long-context probe: ${LONG_SEEDS} seeds from '$SEED', ${LONG_N} tokens each, scored_tail=$LONG_TAIL, int8_kv=1 vs llama"
fi

echo ">> sparkinfer teacher-forced score ..."
# Dump top-128 (>= the llama top-k queried in accuracy_compare). A shallow dump made the KL a
# truncation artifact: any llama-tail token outside sparkinfer's dump was floored (exp(-20)) and
# massively over-penalized, inflating KL to 0.14-0.33 on flat distributions. With the dump covering
# llama's query, KL reflects the true ~0.01-0.03 divergence. Scoring-only — no decode-speed impact.
# All sparkinfer scores run BEFORE the llama server starts, so the ~21 GB model isn't resident twice
# on the 32 GB card (llama + sparkinfer together OOM the load).
si_run qwen3_gguf_score "$GGUF" 128 $IDS > /tmp/spark_score.txt 2>/dev/null || true
if ! grep -q "^PPL" /tmp/spark_score.txt; then   # prebuilt incompatible -> rebuild
  fallback_build "$ARCH"
  si_run qwen3_gguf_score "$GGUF" 128 $IDS > /tmp/spark_score.txt 2>/dev/null
fi

if [ "$LONGCTX" = "1" ]; then
  for j in $(seq 0 $((LONG_SEEDS - 1))); do
    echo ">> sparkinfer long-context score — int8 KV, seed L${j} ..."
    SPARKINFER_KV_INT8=1 SPARKINFER_SCORE_MAX_SEQ=$((LONG_N + 256)) \
      si_run qwen3_gguf_score "$GGUF" 128 $(cat "/tmp/eval_ids_long_${j}.txt") > "/tmp/spark_long_int8_${j}.txt" 2>/dev/null || true
    grep -q "^PPL" "/tmp/spark_long_int8_${j}.txt" || { echo "!! long score L${j} produced no PPL — skipping H2 veto"; LONGCTX=0; break; }
  done
fi

echo ">> starting llama.cpp server (reference) ..."
# -c covers the long stream (+ margin); the short pass shares this one server.
LLAMA_CTX=2048
[ "$LONGCTX" = "1" ] && LLAMA_CTX=$((LONG_N + 256))
"$LLAMACPP_DIR/build/bin/llama-server" -m "$GGUF" -ngl 99 -c "$LLAMA_CTX" --port 8081 --no-jinja >/tmp/llama_srv.log 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null || true' EXIT   # reap server (frees VRAM) before exit
for _ in $(seq 1 120); do curl -s http://localhost:8081/health 2>/dev/null | grep -q '"ok"' && break; sleep 2; done

echo; echo "=== accuracy: sparkinfer vs llama.cpp (short, bf16 KV) ==="
python3 "$HERE/accuracy_compare.py" /tmp/spark_score.txt "$MODELS_DIR/tokenizer.json" "$IDS_FILE" \
        --metric-label METRIC_SHORT | tee /tmp/acc_short.txt

rm -f /tmp/acc_long.txt
if [ "$LONGCTX" = "1" ]; then
  for j in $(seq 0 $((LONG_SEEDS - 1))); do
    echo; echo "=== accuracy: sparkinfer vs llama.cpp (long-context int8, seed L${j}) ==="
    python3 "$HERE/accuracy_compare.py" "/tmp/spark_long_int8_${j}.txt" "$MODELS_DIR/tokenizer.json" \
            "/tmp/eval_ids_long_${j}.txt" http://localhost:8081 64 --tail "$LONG_TAIL" \
            --metric-label "METRIC_LONG${j}" | tee -a /tmp/acc_long.txt
  done
fi

# evaluate.sh parses the single `METRIC ` line (grep 'METRIC ' + head -1; the METRIC_SHORT /
# METRIC_LONG* diagnostics carry no trailing space so they don't match). The emitted gate is the
# SHORT vs-llama result — so label.py's existing top1/KL bars are unchanged — but the long pass
# AVERAGES the LONG_SEEDS runs and VETOES if mean_top1 < LONG_TOP1_BAR OR mean_kl > LONG_KL_BAR.
# On veto, top1 is forced to 0 (REJECT). The long pass can only reject, never rescue; its bars are
# long-context/int8-specific and never touch the short pass's bars.
echo
LONG_TOP1_BAR="$LONG_TOP1_BAR" LONG_KL_BAR="$LONG_KL_BAR" python3 - /tmp/acc_short.txt /tmp/acc_long.txt <<'PY'
import re, sys, os
t1_bar = float(os.environ.get("LONG_TOP1_BAR", "0.875"))
kl_bar = float(os.environ.get("LONG_KL_BAR", "1.0"))
def grab_short(path):
    for line in open(path):
        m = re.match(r'^METRIC_SHORT top1=([\d.]+) kl=([\d.]+)', line)
        if m: return float(m.group(1)), float(m.group(2))
    return None
def grab_longs(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            m = re.match(r'^METRIC_LONG\d+ top1=([\d.]+) kl=([\d.]+)', line)
            if m: out.append((float(m.group(1)), float(m.group(2))))
    return out
short = grab_short(sys.argv[1])
longs = grab_longs(sys.argv[2]) if len(sys.argv) > 2 else []
if not short:
    print("METRIC top1=0 kl=99 ppl_spark=0 ppl_llama=0   (short pass produced nothing)"); sys.exit(1)
t, k = short
print(f"  short vs-llama (bf16)  top1={t:.4f} kl={k:.4f}   (gates top1>=0.90, kl<=0.20)")
if longs:
    for j, (lt, lk) in enumerate(longs):
        print(f"  long  L{j} int8-vs-llama  top1={lt:.4f} kl={lk:.4f}")
    mt = sum(x for x, _ in longs) / len(longs)
    mk = sum(y for _, y in longs) / len(longs)
    reasons = []
    if mt < t1_bar: reasons.append(f"mean top1 {mt:.4f} < {t1_bar:.2f}")
    if mk > kl_bar: reasons.append(f"mean KL {mk:.4f} > {kl_bar:.2f}")
    print(f"  long  MEAN of {len(longs)}     top1={mt:.4f} kl={mk:.4f}   (veto if top1<{t1_bar:.2f} or KL>{kl_bar:.2f}){'  -> VETO' if reasons else ''}")
    if reasons:
        print(f"=== H2 long-context veto: int8 config diverges from llama ({'; '.join(reasons)}) ===")
        print(f"METRIC top1=0.000000 kl={mk:.6f} ppl_spark=0 ppl_llama=0"); sys.exit(0)
else:
    print("  long  int8-vs-llama  SKIPPED")
print("=== gate = short vs-llama (long pass did not veto) ===")
print(f"METRIC top1={t:.6f} kl={k:.6f} ppl_spark=0 ppl_llama=0")
PY
