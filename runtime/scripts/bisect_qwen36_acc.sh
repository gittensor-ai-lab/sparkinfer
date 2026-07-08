#!/usr/bin/env bash
# Quick Qwen3.6 accuracy bisect: compare teacher-forced argmax top-1 vs a reference dump.
# Usage: bisect_qwen36_acc.sh <ref_score.txt> <label> [env exports...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH="$ROOT/bench/scripts"
source "$BENCH/_common.sh"

REF="${1:?ref score dump}"; LABEL="${2:?label}"; shift 2
GGUF="${GGUF:-/workspace/models36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
ARCH="$(detect_arch)"
resolve_runner "$ARCH"
ensure_tokenizer

SEED="${SPARKINFER_EVAL_SEED:-fixed}"
IDS_FILE="/tmp/bisect_ids.txt"
python3 "$BENCH/gen_eval_prompt.py" "$SEED" "$MODELS_DIR/tokenizer.json" \
  "$BENCH/eval_corpus.txt" "$BENCH/eval_text.txt" > "$IDS_FILE"
IDS="$(cat "$IDS_FILE")"

OUT="/tmp/bisect_${LABEL}.txt"
env "$@" si_run qwen3_gguf_score "$GGUF" 128 $IDS > "$OUT"

python3 - "$REF" "$OUT" "$LABEL" <<'PY'
import re, sys
ref_path, cand_path, label = sys.argv[1:4]

def argmatch(path):
    for line in open(path):
        m = re.search(r"ARGMATCH (\d+)/(\d+)", line)
        if m:
            return int(m.group(1)), int(m.group(2))
    tops = []
    for line in open(path):
        m = re.match(r"S i=\d+ tgt=(\d+) am=(\d+)", line)
        if m:
            tops.append(int(m.group(1)) == int(m.group(2)))
    if tops:
        return sum(tops), len(tops)
    return 0, 0

rm, rn = argmatch(ref_path)
cm, cn = argmatch(cand_path)
n = min(rn, cn)
if n == 0:
    print(f"{label}: ERROR no positions parsed"); sys.exit(1)
ref_am = []
for line in open(ref_path):
    m = re.match(r"S i=\d+ tgt=\d+ am=(\d+)", line)
    if m: ref_am.append(int(m.group(1)))
cand_am = []
for line in open(cand_path):
    m = re.match(r"S i=\d+ tgt=\d+ am=(\d+)", line)
    if m: cand_am.append(int(m.group(1)))
match = sum(1 for a, b in zip(ref_am, cand_am) if a == b)
print(f"{label}: argmax match {match}/{len(ref_am)} ({100*match/max(1,len(ref_am)):.1f}%) vs reference")
PY
