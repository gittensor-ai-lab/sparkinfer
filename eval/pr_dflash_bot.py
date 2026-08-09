#!/usr/bin/env python3
"""sparkinfer DFlash PR auto-evaluator.

Sibling of pr_eval_bot.py — evaluates any PR whose description claims a real DFlash speedup
(the same "Tested on RTX 5090" + before/after checklist greenlight_status() already parses),
regardless of which files it touches.
Scores same-box PR DFlash tok/s vs origin/main DFlash tok/s, applies eval-dflash:*
tiers, picks dflash-merge-first, and optionally auto-merges (SPARKINFER_AUTOMERGE=1).

  python eval/pr_dflash_bot.py --instance 46074104
  python eval/pr_dflash_bot.py --only-prs 636 --reeval

Never rents a GPU. Shares the pinned box with the AR bot via flock in the cron wrapper.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ssh_box import ssh_box_enabled, ssh_box_endpoint, ssh_box_user  # noqa: E402

# Reuse shared helpers from the AR bot (labels, greenlight, denylist, …).
import pr_eval_bot as arb  # noqa: E402

SPEEDUP_LABELS = {"XL", "L", "M", "S", "XS"}
SIG = 0.02
REGRESS_TOL = 0.98
BUCKETS = [(0.18, "XL"), (0.10, "L"), (0.06, "M"), (0.035, "S"), (SIG, "XS")]

EVAL_PREFIX = "eval-dflash:"
DFLASH_MERGE_FIRST = "dflash-merge-first"
DFLASH_NEEDS_REBASE = "dflash-needs-rebase"
# Bumped when the scoring/guard logic changes materially (e.g. adding the Qwen3.5/3.6
# no-regression guard) — old markers from before the bump deliberately DON'T match, so a PR
# whose head commit hasn't moved since a pre-guard eval is treated as never-evaluated and gets
# a fresh, guarded run instead of keeping its stale (unguarded) label/score forever.
EVAL_SCHEMA_VERSION = "v2-qwenguard"
MARKER_RE = re.compile(
    r"<!-- sparkinfer-dflash-eval:" + re.escape(EVAL_SCHEMA_VERSION) + r":([0-9a-f]+)(?:\s+(\{.*?\}))? -->",
    re.DOTALL,
)

DEFAULT_GGUF = os.environ.get(
    "DFLASH_GGUF", "/workspace/models36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
)
DEFAULT_DRAFT = os.environ.get(
    "DFLASH_DRAFT", "/workspace/models_dflash/Qwen3.6-35B-A3B-DFlash"
)
DEFAULT_MODELS_DIR = os.environ.get("DFLASH_MODELS_DIR", "/workspace/models36")
REMOTE_REPO = os.environ.get("DFLASH_REMOTE_REPO", "/root/sparkinfer")
BENCH_TOKENS = int(os.environ.get("DFLASH_BENCH_TOKENS", "128"))

# Qwen3.5 (Qwythos) + Qwen3.6 no-regression guard — since #667-era policy, DFlash is the only
# thing that gets a score; it's only valid if the *same PR build* doesn't regress Qwen3.5/3.6
# decode or prefill vs same-box origin/main. Qwen3.6 reuses the DFlash GGUF (one copy, no extra
# download); Qwen3.5 uses the standard Qwythos path already used by the AR/bidir bot.
Q36_GUARD_MODEL_FILE = os.path.basename(DEFAULT_GGUF)
Q36_GUARD_MODELS_DIR = os.path.dirname(DEFAULT_GGUF) or DEFAULT_MODELS_DIR
Q36_GUARD_MODEL_REPO = os.environ.get("PRIMARY36_MODEL_REPO", "unsloth/Qwen3.6-35B-A3B-GGUF")
Q36_GUARD_TOK_REPO = os.environ.get("PRIMARY36_TOK_REPO", "Qwen/Qwen3.6-35B-A3B")
Q35_GUARD_MODELS_DIR = os.environ.get("QWYTHOS_MODELS_DIR", "/workspace/models35")
_Q35_QUANT_FILES = {
    "Q4_K_M": "Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf",
    "Q8_0": "Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf",
    "BF16": "Qwythos-9B-Claude-Mythos-5-1M-BF16.gguf",
}
Q35_GUARD_MODEL_FILE = _Q35_QUANT_FILES.get(
    os.environ.get("PRIMARY_QUANT", "Q4_K_M").upper(), _Q35_QUANT_FILES["Q4_K_M"]
)
GUARD_CTX_LABEL = {0: "128", 512: "512", 4096: "4k", 16384: "16k", 32768: "32k",
                   65536: "64k", 131072: "128k"}

AUTO_MERGE = os.environ.get("SPARKINFER_AUTOMERGE", "0") == "1"
AUTOMERGE_BLOCK = {
    "copycat", "copycat-warn", "flagged:gaming", "penalty", "needs-benchmark",
    DFLASH_NEEDS_REBASE, arb.REEVALUATE_LABEL, arb.HOLD_LABEL, *arb.REGRESSION_LABELS,
}

SCORES_FILE = os.path.expanduser(
    os.environ.get("DFLASH_SCORES_FILE", "~/.sparkinfer_dflash_scores.json")
)

# Polaris verifiable-compute receipts — same policy as the AR bot (on by default; TDX via
# POLARIS_API_KEY when configured, else Ed25519 fallback via SPARKINFER_POLARIS_PRIVATE_KEY).
POLARIS_ENABLED = os.environ.get("POLARIS", "1") != "0"
POLARIS_API_KEY = os.environ.get("POLARIS_API_KEY", "")
_POLARIS_PUBKEY_FILE = os.path.join(HERE, "polaris", "sparkinfer_eval.pub")


def _load_polaris_pubkey():
    try:
        with open(_POLARIS_PUBKEY_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    except Exception:
        pass
    return ""


def _load_scores():
    try:
        return json.load(open(SCORES_FILE))
    except Exception:
        return {}


def _save_scores(data):
    try:
        with open(SCORES_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f">> dflash scores save skipped: {e}")


def tier_from_gain(pr_tps: float, main_tps: float):
    """Return (label, delta_pct, pass_ok, reason)."""
    if main_tps <= 0:
        return "REJECT", 0.0, False, "main DFlash baseline is 0"
    if pr_tps < REGRESS_TOL * main_tps:
        pct = 100.0 * (pr_tps - main_tps) / main_tps
        return "REJECT", round(pct, 1), False, (
            f"DFlash regression: {pr_tps:.2f} < {100 * REGRESS_TOL:.0f}% of main {main_tps:.2f}"
        )
    g = (pr_tps - main_tps) / main_tps
    pct = round(100.0 * g, 1)
    if g < SIG:
        return "none", pct, True, "within significance gate — not a verified DFlash improvement"
    for thr, name in BUCKETS:
        if g >= thr:
            return name, pct, True, "ok"
    return "none", pct, True, "ok"


def score_dflash_multi_ctx(dflash_ctx: dict):
    """Score DFlash across every measured context size (128/512/4k) instead of just the short
    prompt: reject if ANY context regresses beyond REGRESS_TOL — or is missing/zero where main has
    a real baseline, same fail-closed discipline as check_qwen_guard — otherwise tier from
    whichever context shows the BEST gain. A PR that wins big at one context and regresses at
    another must not slip through on its best number alone.

    Returns (label, delta_pct, passed, reason, ctx). ctx is the winning context on success, the
    worst-regressing context on a regression REJECT, or None only in the degenerate case where no
    context has any comparable measurement at all — delta_pct is always a real number otherwise,
    never None, so callers never need to special-case it."""
    problems = []
    worst_ctx, worst_pct = None, None
    candidates = {}  # ctx -> (pr_tps, main_tps)
    for ctx, e in sorted(dflash_ctx.items()):
        clabel = GUARD_CTX_LABEL.get(ctx, str(ctx))
        main_tps = e.get("main_dflash_tps") or 0
        pr_tps = e.get("pr_dflash_tps") or 0
        if main_tps <= 0:
            continue  # main itself has no baseline here — not comparable
        if pr_tps < REGRESS_TOL * main_tps:
            pct = 100.0 * (pr_tps - main_tps) / main_tps
            problems.append(
                f"DFlash@{clabel}: {pr_tps:.2f} < {100 * REGRESS_TOL:.0f}% of main "
                f"{main_tps:.2f} ({pct:+.1f}%)"
            )
            if worst_pct is None or pct < worst_pct:
                worst_pct, worst_ctx = pct, ctx
            continue
        candidates[ctx] = (pr_tps, main_tps)
    if problems:
        return "REJECT", round(worst_pct, 1), False, "DFlash regression at: " + "; ".join(problems), worst_ctx
    if not candidates:
        return "REJECT", 0.0, False, "no comparable DFlash context measurements", None
    best_ctx = max(candidates, key=lambda c: candidates[c][0] / candidates[c][1])
    pr_tps, main_tps = candidates[best_ctx]
    label, delta_pct, passed, reason = tier_from_gain(pr_tps, main_tps)
    clabel = GUARD_CTX_LABEL.get(best_ctx, str(best_ctx))
    return label, delta_pct, passed, f"{reason} (best at {clabel} context)", best_ctx


def dflash_evaluated_commits(repo, num):
    """Head commits that already have a REAL scoring verdict posted — infra/transport failures
    (label:null in the marker's meta JSON) don't count, so a commit that hit a flaky SSH/CUDA/box
    crash gets picked up again on the next tick instead of being silently skipped forever."""
    r = arb.gh(["pr", "view", str(num), "-R", repo, "--json", "comments"])
    done = set()
    for c in json.loads(r.stdout or "{}").get("comments", []):
        body = c.get("body") or ""
        m = MARKER_RE.search(body)
        if not m or "sparkinfer dflash auto-eval" not in body:
            continue
        meta_raw = m.group(2)
        try:
            meta = json.loads(meta_raw) if meta_raw else {}
        except json.JSONDecodeError:
            meta = {}
        if meta.get("label") is None:
            continue
        done.add(m.group(1))
    return done


def strip_dflash_eval_labels(repo, num):
    for lab in list(arb.labels_on(repo, num)):
        if lab.startswith(EVAL_PREFIX):
            arb.remove_label(repo, num, lab)


def resolve_ssh(instance_id: int):
    """Return (host, port) for the pinned box."""
    if ssh_box_enabled():
        ep = ssh_box_endpoint()
        if not ep:
            raise RuntimeError("EVAL_TRANSPORT=ssh but EVAL_SSH_HOST unset")
        return ep
    key = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/speedy"))
    os.environ.setdefault("SSH_KEY", key)
    iid = arb.current_instance(instance_id) or instance_id
    raw = subprocess.run(
        ["vastai", "show", "instance", str(iid), "--raw"],
        capture_output=True, text=True, timeout=60,
    )
    if raw.returncode != 0 or not (raw.stdout or "").strip():
        raise RuntimeError(f"vastai show instance {iid} failed: {(raw.stderr or '')[:200]}")
    info = json.loads(raw.stdout)
    ip = (info.get("public_ipaddr") or "").strip()
    ports = info.get("ports") or {}
    m = ports.get("22/tcp") or [{}]
    port = int((m[0] or {}).get("HostPort") or 0)
    if info.get("actual_status") != "running" or not ip or not port:
        raise RuntimeError(
            f"pinned instance {iid} not SSH-ready (status={info.get('actual_status')})"
        )
    return ip, port


def ssh_run(host, port, cmd, timeout=7200, stdin_data=None):
    key = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/speedy"))
    # vast.ai images run as root; a bare-metal SSH box (EVAL_TRANSPORT=ssh) may have a
    # non-root default account instead — EVAL_SSH_USER overrides (defaults to root).
    user = ssh_box_user() if ssh_box_enabled() else "root"
    return subprocess.run(
        [
            "ssh", "-i", key,
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=40",
            "-p", str(port), f"{user}@{host}", cmd,
        ],
        capture_output=True, text=True, timeout=timeout, input=stdin_data,
    )


def _remote_script(ref: str, do_accuracy: bool, prompt_ids: str | None, n_tokens: int,
                    prompt_ids_ctx: dict | None = None) -> str:
    """Bash run on the eval box: checkout ref, build, optional accuracy, bench."""
    gguf = shlex.quote(DEFAULT_GGUF)
    draft = shlex.quote(DEFAULT_DRAFT)
    models = shlex.quote(DEFAULT_MODELS_DIR)
    repo = shlex.quote(REMOTE_REPO)
    ref_q = shlex.quote(ref)
    hf = shlex.quote(os.environ.get("HF_TOKEN", ""))
    ids_export = ""
    if prompt_ids:
        ids_export = f"PROMPT_IDS={shlex.quote(prompt_ids)}\n"
    for ctx, ids in (prompt_ids_ctx or {}).items():
        if ids:
            ids_export += f"PROMPT_IDS_{ctx}={shlex.quote(ids)}\n"
    acc = "1" if do_accuracy else "0"
    q36_file = shlex.quote(Q36_GUARD_MODEL_FILE)
    q36_dir = shlex.quote(Q36_GUARD_MODELS_DIR)
    q36_repo = shlex.quote(Q36_GUARD_MODEL_REPO)
    q36_tok = shlex.quote(Q36_GUARD_TOK_REPO)
    q35_dir = shlex.quote(Q35_GUARD_MODELS_DIR)
    return f"""
set -euo pipefail
# Surface *why* a crash happened instead of just dying silently under set -e — decode common
# kill/crash exit codes into a human reason and snapshot GPU memory at the moment of failure, so
# a REJECT from an infra crash (e.g. #684's OOM-during-model-load) shows a real cause instead of
# a bare truncated stdout tail.
trap 'rc=$?; ln=$LINENO; reason=""; \\
  case $rc in \\
    137) reason="likely OOM-killed (SIGKILL)" ;; \\
    139) reason="likely segfault (SIGSEGV)" ;; \\
    134) reason="likely abort (SIGABRT)" ;; \\
    124) reason="likely timeout" ;; \\
  esac; \\
  echo "REMOTE_SCRIPT_FAILED line=$ln exit=$rc reason=$reason" >&2; \\
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader >&2 2>/dev/null || true' ERR
# One run does 4 back-to-back multi-GB model load/unload cycles (PR/main DFlash bench + Qwen3.6
# guard + Qwen3.5 guard) with no gap between them. Observed live (#684, #690): starting the next
# heavy load before the previous process's VRAM is actually reclaimed by the driver appears to be
# what silently kills the *whole* remote script (not just the loading binary) around a reload
# boundary — invisible to the ERR trap above, since a hard kill of the interpreter itself never
# reaches trap handling. Poll down to a near-empty GPU before each heavy load instead of assuming
# the previous process's exit already means its memory is free.
wait_gpu_clear() {{
  local tries=0 used
  while [ "$tries" -lt 30 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -lt 1024 ] 2>/dev/null && return 0
    sleep 1
    tries=$((tries + 1))
  done
  echo "WARN: GPU memory still ${{used:-unknown}} MiB after ${{tries}}s wait — proceeding anyway" >&2
}}
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
export CUDA_HOME=${{CUDA_HOME:-/usr/local/cuda-13.0}}
export HF_TOKEN={hf}
export HF_HUB_DISABLE_XET=1
REPO={repo}
GGUF={gguf}
DRAFT={draft}
MODELS_DIR={models}
NTOK={n_tokens}
DO_ACC={acc}
Q36_GUARD_MODEL_FILE={q36_file}
Q36_GUARD_MODELS_DIR={q36_dir}
Q36_GUARD_MODEL_REPO={q36_repo}
Q36_GUARD_TOK_REPO={q36_tok}
Q35_GUARD_MODELS_DIR={q35_dir}
{ids_export}
cd "$REPO"
git remote set-url origin https://github.com/gittensor-ai-lab/sparkinfer.git 2>/dev/null || true
git fetch -q origin {ref_q}
git reset -q --hard
git clean -qfd
git checkout -qf FETCH_HEAD
HEAD=$(git rev-parse --short HEAD)
echo "REMOTE_HEAD $HEAD"

# Ensure draft weights exist
if [ ! -f "$DRAFT/model.safetensors" ] && [ ! -f "$DRAFT/model.safetensors.index.json" ]; then
  mkdir -p "$(dirname "$DRAFT")"
  hf download z-lab/Qwen3.6-35B-A3B-DFlash --local-dir "$DRAFT"
fi
test -f "$GGUF" || {{ echo "FAIL missing GGUF $GGUF"; exit 1; }}

# Build dflash tools + qwen3_gguf_bench (incremental build — the latter drives the Qwen3.5/3.6
# guard below — but ALWAYS reconfigure). build/ is gitignored so it survives every checkout on
# this box; skipping `cmake -S . -B build` whenever CMakeCache.txt already exists left stale
# generated Makefiles pointing at source files from a *different* PR's branch that added them —
# switching checkout to main (or another PR without those files) then failed with
# "No such file or directory" for files main never even references (#693, #694). cmake's own
# configure step is cheap and idempotent on an existing cache, so there's no reason to skip it.
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/dflash_cmake.log 2>&1
cmake --build build --target qwen3_gguf_dflash_check qwen3_gguf_dflash_bench qwen3_gguf_bench -j"$(nproc)" >/tmp/dflash_build.log 2>&1 || {{
  echo "BUILD_FAILED — tail of /tmp/dflash_build.log:" >&2
  tail -80 /tmp/dflash_build.log >&2
  exit 1
}}
test -x build/runtime/qwen3_gguf_dflash_bench
test -x build/runtime/qwen3_gguf_bench

if [ "$DO_ACC" = "1" ]; then
  export MODELS_DIR
  # dflash_accuracy.sh's ensure_tokenizer needs TOK_REPO for the *target* model (Qwen3.6) —
  # without it, _common.sh's TOK_REPO default ("Qwen/Qwen3-30B-A3B") silently wins, so
  # gen_eval_prompt.py tokenizes the scored prompt with the wrong (smaller) vocabulary. The
  # mismatch is invisible downstream: Qwen3.6's vocab is larger, so every resulting id is
  # still "valid", just semantically meaningless. That corrupted PROMPT_IDS stream is what
  # both the accuracy check AND the speed bench below score against.
  export TOK_REPO="$Q36_GUARD_TOK_REPO"
  bash bench/scripts/dflash_accuracy.sh "$GGUF" "$DRAFT" | tee /tmp/dflash_check_out.txt
  grep -q "^VERDICT PASS" /tmp/dflash_check_out.txt
  if [ -z "${{PROMPT_IDS:-}}" ] && [ -f /tmp/dflash_eval_ids.txt ]; then
    PROMPT_IDS=$(cat /tmp/dflash_eval_ids.txt)
  fi
  echo "PROMPT_IDS $PROMPT_IDS"
  SPEC=$(grep '^METRIC SPEC_AGREE' /tmp/dflash_check_out.txt | tail -1 || true)
  echo "$SPEC"
else
  PROMPT_IDS="${{PROMPT_IDS:-}}"
  if [ -z "$PROMPT_IDS" ] && [ -f /tmp/dflash_eval_ids.txt ]; then
    PROMPT_IDS=$(cat /tmp/dflash_eval_ids.txt)
  fi
  echo "PROMPT_IDS $PROMPT_IDS"
fi

wait_gpu_clear
OUT=$(build/runtime/qwen3_gguf_dflash_bench "$GGUF" "$DRAFT" "$NTOK" $PROMPT_IDS | tee /tmp/dflash_bench_out.txt)
echo "$OUT"
AR=$(echo "$OUT" | grep '^METRIC AR_TPS' | awk '{{print $3}}' | tail -1)
DF=$(echo "$OUT" | grep '^METRIC DFLASH_TPS' | awk '{{print $3}}' | tail -1)
TAU=$(echo "$OUT" | grep '^METRIC MEAN_ACCEPT' | awk '{{print $3}}' | tail -1)
echo "RESULT_AR_TPS $AR"
echo "RESULT_DFLASH_TPS $DF"
echo "RESULT_MEAN_ACCEPT $TAU"

# --- DFlash speed at additional context sizes (512, 4k) — baseline/informational data only for
# now, not part of the pass/fail gate. Same held-out-prompt discipline as the primary bench above:
# the PR run generates+caches a fresh held-out prompt per size, the main run reuses the identical
# ids (passed in via env, below) so PR vs main compares the same prompt, not two independent draws.
source bench/scripts/_common.sh
export MODELS_DIR
export TOK_REPO="$Q36_GUARD_TOK_REPO"
ensure_tokenizer || echo "WARN: extra-context tokenizer setup failed" >&2
for ctx in 512 4096; do
  idsfile="/tmp/dflash_eval_ids_${{ctx}}.txt"
  case "$ctx" in
    512) ids="${{PROMPT_IDS_512:-}}" ;;
    4096) ids="${{PROMPT_IDS_4096:-}}" ;;
  esac
  if [ -z "$ids" ] && [ -f "$idsfile" ]; then
    ids="$(cat "$idsfile")"
  fi
  if [ -z "$ids" ]; then
    ids="$(python3 bench/scripts/gen_eval_prompt.py "${{SPARKINFER_EVAL_SEED:-fixed}}" "$MODELS_DIR/tokenizer.json" bench/scripts/eval_corpus.txt --len "$ctx")"
    echo "$ids" > "$idsfile"
  fi
  wait_gpu_clear
  CTXOUT=$(build/runtime/qwen3_gguf_dflash_bench "$GGUF" "$DRAFT" "$NTOK" $ids)
  CAR=$(echo "$CTXOUT" | grep '^METRIC AR_TPS' | awk '{{print $3}}' | tail -1)
  CDF=$(echo "$CTXOUT" | grep '^METRIC DFLASH_TPS' | awk '{{print $3}}' | tail -1)
  CTAU=$(echo "$CTXOUT" | grep '^METRIC MEAN_ACCEPT' | awk '{{print $3}}' | tail -1)
  echo "DFLASH_CTX $ctx ${{CDF:-0}} ${{CAR:-0}} ${{CTAU:-0}}"
  echo "PROMPT_IDS_$ctx $ids"
  # Speed alone at a context isn't proof of correctness — DFlash is a lossless speculative
  # accelerator, so it should exactly reproduce greedy AR at every context, same bar as the short
  # prompt. Checked only on the PR side (DO_ACC=1): main's own correctness at these contexts isn't
  # in question here, only whether the PR regresses it. #707 proved this matters — it fixed a
  # crash at 512 that a speed-only check could never have caught, and separately looked fast at 4k
  # while actually diverging from AR (0.0625 SPEC_AGREE) — a corruption-driven speedup that a
  # speed-only regression check would have happily accepted.
  if [ "$DO_ACC" = "1" ]; then
    wait_gpu_clear
    CHKOUT=$(build/runtime/qwen3_gguf_dflash_check "$GGUF" "$DRAFT" "${{SPARKINFER_DFLASH_CHECK_NEW:-32}}" $ids)
    CRATIO=$(echo "$CHKOUT" | grep '^METRIC SPEC_AGREE' | tail -1 | awk '{{print $NF}}')
    CVERDICT=$(echo "$CHKOUT" | grep '^VERDICT' | tail -1 | awk '{{print $2}}')
    echo "DFLASH_CTX_ACC $ctx ${{CRATIO:-0}} ${{CVERDICT:-FAIL}}"
  fi
done

# --- Qwen3.5 / Qwen3.6 no-regression guard (decode + prefill, same build as above) ---
source bench/scripts/_eval_speed.sh
# Pin QWYTHOS_MODELS_DIR before sourcing _qwythos.sh: its own default derives from the ambient
# $MODELS_DIR, which the DFlash steps above already repointed at /workspace/models36 — falling
# through to that default here would silently resolve to .../models3635.
export QWYTHOS_MODELS_DIR="$Q35_GUARD_MODELS_DIR"
source bench/scripts/_qwythos.sh
SI_BIN="$PWD/build/runtime"; SI_LD=""
gclks=()

Q35_FILE="$(qwythos_quant_file)"
export MODELS_DIR="$QWYTHOS_MODELS_DIR" MODEL_REPO="$QWYTHOS_REPO" MODEL_FILE="$Q35_FILE" TOK_REPO="$QWYTHOS_TOK_REPO"
export MODEL_SHA256="$(qwythos_sha_var)"
( ensure_model && ensure_tokenizer ) || echo "WARN: qwen3.5 guard model setup failed" >&2
Q35_GGUF="$QWYTHOS_MODELS_DIR/$Q35_FILE"

export MODELS_DIR="$Q36_GUARD_MODELS_DIR" MODEL_REPO="$Q36_GUARD_MODEL_REPO" MODEL_FILE="$Q36_GUARD_MODEL_FILE" TOK_REPO="$Q36_GUARD_TOK_REPO"
export MODEL_SHA256="${{QWEN36_MODEL_SHA256:-}}"
( ensure_model && ensure_tokenizer ) || echo "WARN: qwen3.6 guard model setup failed" >&2
Q36_GGUF="$Q36_GUARD_MODELS_DIR/$Q36_GUARD_MODEL_FILE"

echo "GUARD_START"
wait_gpu_clear
if bench_sweep_run "$Q36_GGUF" 128 0 1 512 1 4096 1 16384 1 32768 1; then
  for ctx in 0 512 4096 16384 32768; do
    echo "GUARD36 $ctx $(_bench_sweep_get $ctx decode_tps) $(_bench_sweep_get $ctx prefill_pp)"
  done
else
  echo "GUARD36_FAILED"
fi
wait_gpu_clear
if bench_sweep_run "$Q35_GGUF" 128 0 1 4096 1 32768 1 65536 1 131072 1; then
  for ctx in 0 4096 32768 65536 131072; do
    echo "GUARD35 $ctx $(_bench_sweep_get $ctx decode_tps) $(_bench_sweep_get $ctx prefill_pp)"
  done
else
  echo "GUARD35_FAILED"
fi
echo "GUARD_END"
"""


def _parse_remote(stdout: str) -> dict:
    out = {}
    guard36, guard35 = {}, {}
    dflash_ctx, dflash_ctx_acc, prompt_ids_ctx = {}, {}, {}
    for line in (stdout or "").splitlines():
        if line.startswith("RESULT_AR_TPS "):
            out["ar_tps"] = float(line.split()[1])
        elif line.startswith("RESULT_DFLASH_TPS "):
            out["dflash_tps"] = float(line.split()[1])
        elif line.startswith("RESULT_MEAN_ACCEPT "):
            out["mean_accept"] = float(line.split()[1])
        elif line.startswith("PROMPT_IDS "):
            out["prompt_ids"] = line[len("PROMPT_IDS "):].strip()
        elif line.startswith("DFLASH_CTX "):
            parts = line.split()
            if len(parts) >= 5:
                try:
                    dflash_ctx[int(parts[1])] = {
                        "dflash_tps": float(parts[2]), "ar_tps": float(parts[3]),
                        "mean_accept": float(parts[4]),
                    }
                except ValueError:
                    pass
        elif line.startswith("DFLASH_CTX_ACC "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    dflash_ctx_acc[int(parts[1])] = {"spec_agree": float(parts[2]), "verdict": parts[3]}
                except ValueError:
                    pass
        elif line.startswith("PROMPT_IDS_"):
            rest = line[len("PROMPT_IDS_"):]
            ctx_str, _, ids_str = rest.partition(" ")
            if ctx_str.isdigit():
                prompt_ids_ctx[int(ctx_str)] = ids_str.strip()
        elif line.startswith("REMOTE_HEAD "):
            out["head"] = line.split()[1]
        elif line.startswith("METRIC SPEC_AGREE"):
            out["spec_agree"] = line.strip()
        elif line.startswith("GUARD36 "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    guard36[int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.startswith("GUARD35 "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    guard35[int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.strip() == "GUARD36_FAILED":
            out["guard36_failed"] = True
        elif line.strip() == "GUARD35_FAILED":
            out["guard35_failed"] = True
    out["guard36"] = guard36
    out["guard35"] = guard35
    out["dflash_ctx"] = dflash_ctx
    out["dflash_ctx_acc"] = dflash_ctx_acc
    out["prompt_ids_ctx"] = prompt_ids_ctx
    return out


def check_qwen_guard(pr: dict, main: dict, tol: float = REGRESS_TOL):
    """No-regression check: PR vs same-box main, Qwen3.5 + Qwen3.6, decode + prefill, every
    measured context. Returns (ok, [human-readable regression/failure strings])."""
    problems = []
    if pr.get("guard36_failed") or main.get("guard36_failed") or not pr.get("guard36") or not main.get("guard36"):
        problems.append("qwen3.6 guard measurement unavailable")
    if pr.get("guard35_failed") or main.get("guard35_failed") or not pr.get("guard35") or not main.get("guard35"):
        problems.append("qwen3.5 guard measurement unavailable")
    # Iterate over MAIN's contexts (the reference/expected set), not the PR's — a PR build that
    # crashes partway through its sweep and never reports a context must not make that context
    # silently uncheckable. Fail closed: a real main baseline (base > 0) with a missing or zero
    # PR measurement (cur <= 0) is flagged as a regression, never skipped.
    for model_name, pr_ctxs, main_ctxs in (
        ("qwen3.6", pr.get("guard36") or {}, main.get("guard36") or {}),
        ("qwen3.5", pr.get("guard35") or {}, main.get("guard35") or {}),
    ):
        for ctx, main_vals in main_ctxs.items():
            label = GUARD_CTX_LABEL.get(ctx, str(ctx))
            pr_vals = pr_ctxs.get(ctx) or {}
            for metric in ("decode", "prefill"):
                base = main_vals.get(metric, 0)
                if base <= 0:
                    continue  # main itself has no baseline for this metric/ctx — not comparable
                cur = pr_vals.get(metric, 0)
                if cur <= 0:
                    problems.append(
                        f"{model_name} {metric}@{label}: PR measurement missing/zero "
                        f"(main {base:.1f}) — treated as regression"
                    )
                    continue
                if cur < base * tol:
                    pct = 100.0 * (cur - base) / base
                    problems.append(
                        f"{model_name} {metric}@{label}: {cur:.1f} < {100 * tol:.0f}% of main "
                        f"{base:.1f} ({pct:+.1f}%)"
                    )
    return (len(problems) == 0, problems)


def push_eval_polaris(host, port):
    """Sync eval/polaris/ (judge.py + receipt.py) to the box from a TRUSTED source before
    running attestation — mirrors vast_eval.py's push_bench_scripts() protection of the
    scoring harness. The box's git checkout is whatever ref is being evaluated (the PR's own
    branch, or main); letting a PR's own commits supply the code that produces its own
    attestation would let it fake a clean receipt, defeating the entire point of an
    independently-verifiable one. Prefers origin/main (fetched fresh — a stale local dev tree
    would be just as untrustworthy a source of truth as the PR itself); set
    SPARKINFER_USE_LOCAL_POLARIS=1 to force the local working tree instead, for testing
    eval/polaris changes before they're merged. Returns True on success."""
    use_local = os.environ.get("SPARKINFER_USE_LOCAL_POLARIS", "").strip().lower() in ("1", "true", "yes")
    tar_data = None
    source = "local checkout"
    extract_root = os.path.join(REMOTE_REPO, "eval")
    if not use_local:
        subprocess.run(["git", "fetch", "-q", "origin", "main"], cwd=ROOT, capture_output=True, timeout=120)
        arch = subprocess.run(
            ["git", "archive", "--format=tar.gz", "origin/main", "eval/polaris"],
            cwd=ROOT, capture_output=True, timeout=120,
        )
        if arch.returncode == 0 and arch.stdout:
            tar_data = arch.stdout
            source = "origin/main"
            extract_root = REMOTE_REPO
        else:
            # Do NOT fall back to the local working tree here — that tree can hold unreviewed,
            # uncommitted edits, and silently attesting with it on a mere network/git hiccup would
            # defeat the whole point of push_eval_polaris (never trust unreviewed code for
            # attestation). Fail closed instead; set SPARKINFER_USE_LOCAL_POLARIS=1 to explicitly
            # opt into the local tree for pre-merge testing.
            print(">> WARN: origin/main eval/polaris fetch failed — refusing to fall back to the "
                  "local working tree for a real run — attestation unavailable this run")
            return False
    else:
        polaris_dir = os.path.join(HERE, "polaris")
        if os.path.isdir(polaris_dir):
            tar = subprocess.run(["tar", "-C", HERE, "-czf", "-", "polaris"],
                                  capture_output=True, timeout=120)
            if tar.returncode == 0 and tar.stdout:
                tar_data = tar.stdout
                source = "local checkout"
                extract_root = os.path.join(REMOTE_REPO, "eval")
    if tar_data is None:
        print(">> WARN: no eval/polaris archive — attestation unavailable this run")
        return False
    key = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/speedy"))
    user = ssh_box_user() if ssh_box_enabled() else "root"
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
            tmp.write(tar_data)
            tmp_path = tmp.name
        scp = subprocess.run(
            ["scp", "-P", str(port), "-i", key, "-o", "StrictHostKeyChecking=accept-new",
             "-o", "BatchMode=yes", tmp_path, f"{user}@{host}:/tmp/si_polaris.tgz"],
            capture_output=True, text=True, timeout=120,
        )
        if scp.returncode != 0:
            print(f">> WARN: eval/polaris scp failed (rc={scp.returncode}): {scp.stderr[-500:]}")
            return False
        extract = (
            f"mkdir -p {shlex.quote(extract_root)} && "
            f"tar -xzf /tmp/si_polaris.tgz -C {shlex.quote(extract_root)} && "
            "rm -f /tmp/si_polaris.tgz"
        )
        r = ssh_run(host, port, extract, timeout=60)
        if r.returncode != 0:
            print(f">> WARN: eval/polaris extract failed (rc={r.returncode}): {(r.stdout + r.stderr)[-500:]}")
            return False
        print(f">> eval/polaris synced from {source} (trusted attestation code)")
        return True
    except subprocess.TimeoutExpired:
        print(">> WARN: eval/polaris sync timed out — attestation unavailable this run")
        return False
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def collect_polaris_attestation(host, port, res: dict, pr_ref: str):
    """Run judge.py --dflash on the box to assemble an unsigned attestation from the eval
    result, then sign it (TDX via POLARIS_API_KEY, else Ed25519 fallback) — same policy as the
    AR bot. Returns {"attestation":..., "receipt":...} (receipt omitted if unsigned), or None
    if Polaris is disabled. Never raises — a Polaris failure must not block the DFlash verdict
    itself, only omit its receipt.

    Two correctness fixes baked in here:
    - eval_dflash_on_box() evaluates the PR ref first, then main second, and leaves the box
      checked out on whichever ran last (main) — so judge.py's `git rev-parse HEAD` would
      silently attest to the BASELINE commit, not the PR commit the verdict is actually about.
      Re-checkout pr_ref explicitly before invoking judge.py.
    - build_polaris_receipt_from_attestation()'s nonce is sha256(commit + model_sha256 +
      eval_seed); with eval_seed empty (the default), re-evaluating the same commit+model
      combo — which DFlash re-evals routinely do — produces an IDENTICAL nonce every time.
      Observed in testing: two attestations with genuinely different measurements came back
      with the same tdx.result_sha256, consistent with Polaris treating a repeated nonce as
      a dedup/idempotency key and replaying a cached quote rather than a fresh one. Force a
      unique eval_seed per call so the nonce — and therefore the quote — can never repeat.
    """
    if not POLARIS_ENABLED:
        return None
    # Order matters: checkout pr_ref FIRST (so git HEAD is the commit being attested), THEN
    # sync the trusted judge.py on top of it (so the PR's own — possibly stale or, if malicious,
    # deliberately broken — copy of eval/polaris/ never runs). Reversed, the checkout would
    # clobber the just-synced trusted files right back to whatever the PR ref itself carries.
    checkout_cmd = (
        f"cd {shlex.quote(REMOTE_REPO)} && "
        f"git fetch -q origin {shlex.quote(pr_ref)} && git checkout -qf FETCH_HEAD"
    )
    r0 = ssh_run(host, port, checkout_cmd, timeout=60)
    if r0.returncode != 0:
        print(f">> Polaris: could not checkout {pr_ref} for attestation (rc={r0.returncode}): "
              f"{(r0.stderr or '')[-300:]}")
        return None
    if not push_eval_polaris(host, port):
        return None
    q35_gguf = f"{Q35_GUARD_MODELS_DIR}/{Q35_GUARD_MODEL_FILE}"
    q36_gguf = f"{Q36_GUARD_MODELS_DIR}/{Q36_GUARD_MODEL_FILE}"
    eval_seed = f"dflash-{int(time.time() * 1000)}"
    cmd = (
        f"cd {shlex.quote(REMOTE_REPO)} && "
        f"SPARKINFER_EVAL_SEED={shlex.quote(eval_seed)} python3 eval/polaris/judge.py --dflash "
        f"--sparkinfer-root {shlex.quote(REMOTE_REPO)} "
        f"--build-dir {shlex.quote(REMOTE_REPO)}/build/runtime "
        f"--model-file {shlex.quote(q36_gguf)} "
        f"--guard-model-file {shlex.quote(q35_gguf)}"
    )
    try:
        r = ssh_run(host, port, cmd, timeout=120, stdin_data=json.dumps(res))
    except Exception as e:
        print(f">> Polaris judge SSH failed: {e}")
        return None
    if r.returncode != 0:
        print(f">> Polaris judge failed (rc={r.returncode}): {(r.stderr or '')[-500:]}")
        return None
    polaris_line = next((l for l in (r.stdout or "").splitlines()
                         if l.startswith("POLARIS_ATTESTATION ")), None)
    if not polaris_line:
        print(">> Polaris judge produced no attestation")
        return None
    try:
        attestation = json.loads(polaris_line[len("POLARIS_ATTESTATION "):])
    except json.JSONDecodeError as e:
        print(f">> Polaris attestation JSON parse failed: {e}")
        return None
    privkey = arb._load_polaris_privkey()
    if not POLARIS_API_KEY and not privkey:
        print(">> Polaris: attestation collected but NOT signed (no key configured)")
        return {"attestation": attestation}
    try:
        receipt = arb.build_polaris_receipt_from_attestation(
            attestation, api_key=POLARIS_API_KEY, privkey=privkey, pubkey=_load_polaris_pubkey(),
        )
        return {"attestation": attestation, "receipt": receipt}
    except Exception as e:
        print(f">> Polaris signing failed: {e}")
        return {"attestation": attestation}


def _crash_reason(*outputs: str) -> str | None:
    """Pull the ERR-trap diagnostic line (line/exit-code/signal + GPU mem snapshot) out of remote
    script output, so a REJECT from an infra crash carries a real cause instead of just a bare
    truncated log tail the reader has to decode themselves."""
    combined = "\n".join(o or "" for o in outputs)
    lines = combined.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("REMOTE_SCRIPT_FAILED "):
            extra = lines[i + 1].strip() if i + 1 < len(lines) else ""
            return line.strip() + (f" | gpu: {extra}" if extra else "")
    return None


def _looks_like_hard_kill(stdout: str, stderr: str) -> bool:
    """True when a failed run has neither the ERR-trap diagnostic NOR a graceful
    GUARD36_FAILED/GUARD35_FAILED marker — i.e. the whole remote bash process was killed outright
    (OOM/session drop around a heavy model-reload boundary, per #684/#690) rather than a single
    step failing in a way the script could catch and report on its own. A command that's the
    condition of an `if` (like the guard sweeps) never trips the ERR trap even on an ordinary
    failure, so seeing neither signal means something killed the interpreter itself."""
    combined = (stdout or "") + "\n" + (stderr or "")
    if _crash_reason(stdout, stderr):
        return False
    if "GUARD36_FAILED" in combined or "GUARD35_FAILED" in combined:
        return False
    return True


def _ssh_run_resilient(host, port, script: str, label: str):
    """ssh_run() with one automatic retry when the failure looks like a hard kill rather than a
    graceful, self-reported failure — cheap insurance against the exact silent-kill pattern that
    cost #684 and #690 a full eval slot each, since a hard kill gives no actionable diagnostic to
    act on anyway and a retry is the only way to tell transient from reproducible."""
    r = ssh_run(host, port, script)
    if r.returncode != 0 and _looks_like_hard_kill(r.stdout, r.stderr):
        print(f">> {label}: looks like a hard kill (no ERR-trap diagnostic, no graceful "
              f"GUARD*_FAILED marker) — retrying once")
        r = ssh_run(host, port, script)
    return r


def eval_dflash_on_box(host, port, pr_ref: str):
    """Run PR accuracy+bench then main bench with same prompt ids. Returns result dict."""
    print(f">> DFlash eval on box: PR ref={pr_ref}")
    r = _ssh_run_resilient(host, port, _remote_script(pr_ref, do_accuracy=True, prompt_ids=None,
                                                       n_tokens=BENCH_TOKENS), "PR run")
    if r.returncode != 0:
        tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-2000:]
        crash = _crash_reason(r.stdout, r.stderr)
        reason = "PR accuracy/bench failed" + (f" — {crash}" if crash else " (no crash diagnostic captured, possible hard kill — retried once)")
        return {"ok": False, "reason": reason, "log": tail}
    pr = _parse_remote(r.stdout or "")
    if "dflash_tps" not in pr:
        return {"ok": False, "reason": "PR bench missing DFLASH_TPS", "log": (r.stdout or "")[-1500:]}
    ids = pr.get("prompt_ids") or ""
    ctx_ids = pr.get("prompt_ids_ctx") or {}
    print(f">> PR DFlash={pr['dflash_tps']:.2f} AR={pr.get('ar_tps', 0):.2f} — measuring main …")
    r2 = _ssh_run_resilient(host, port, _remote_script("main", do_accuracy=False, prompt_ids=ids,
                                                        n_tokens=BENCH_TOKENS,
                                                        prompt_ids_ctx=ctx_ids), "main run")
    if r2.returncode != 0:
        tail = ((r2.stdout or "") + "\n" + (r2.stderr or ""))[-2000:]
        crash = _crash_reason(r2.stdout, r2.stderr)
        reason = "main bench failed" + (f" — {crash}" if crash else " (no crash diagnostic captured, possible hard kill — retried once)")
        return {"ok": False, "reason": reason, "log": tail, "pr": pr}
    main = _parse_remote(r2.stdout or "")
    if "dflash_tps" not in main:
        return {"ok": False, "reason": "main bench missing DFLASH_TPS", "log": (r2.stdout or "")[-1500:],
                "pr": pr}
    # Fold the primary/short bench into the SAME per-context structure under ctx=0 (matches
    # GUARD_CTX_LABEL's own "0 -> 128" convention), then iterate MAIN's contexts as the fail-closed
    # base — same discipline as check_qwen_guard: a PR that crashes partway through its own 512/4k
    # sweep must not silently drop that context from scoring instead of failing it.
    main_ctx_raw = {0: {"dflash_tps": main["dflash_tps"], "ar_tps": main.get("ar_tps")},
                     **(main.get("dflash_ctx") or {})}
    pr_ctx_raw = {0: {"dflash_tps": pr["dflash_tps"], "ar_tps": pr.get("ar_tps")},
                  **(pr.get("dflash_ctx") or {})}
    dflash_ctx = {}
    for ctx, m in main_ctx_raw.items():
        p = pr_ctx_raw.get(ctx) or {}
        dp, dm = p.get("dflash_tps") or 0, m.get("dflash_tps") or 0
        dflash_ctx[ctx] = {
            "pr_dflash_tps": dp, "main_dflash_tps": dm,
            "pr_ar_tps": p.get("ar_tps"), "main_ar_tps": m.get("ar_tps"),
            "delta_pct": round(100.0 * (dp - dm) / dm, 1) if dm else None,
        }

    # Score across ALL measured contexts (128/512/4k), not just the short prompt — regression at
    # ANY of them rejects the whole PR; otherwise the tier comes from whichever context shows the
    # best gain. A PR that wins big at one context and quietly regresses at another must not slip
    # through on its best number alone.
    label, delta_pct, passed, reason, best_ctx = score_dflash_multi_ctx(dflash_ctx)

    # Speed alone at a context is not proof of correctness — DFlash is a lossless speculative
    # accelerator, so it must exactly reproduce greedy AR (SPEC_AGREE VERDICT PASS) at every
    # measured context, same bar the short prompt already enforces. #707 proved why this matters:
    # it fixed a crash at 512-ctx that a speed-only check could never catch, and separately looked
    # faster at 4k while actually diverging from AR — a corruption-driven "speedup" that
    # score_dflash_multi_ctx alone would have happily accepted. Checked on the PR side only
    # (main's own correctness there isn't in question, only whether the PR breaks it).
    acc_problems = []
    for ctx, e in sorted((pr.get("dflash_ctx_acc") or {}).items()):
        clabel = GUARD_CTX_LABEL.get(ctx, str(ctx))
        if e.get("verdict") != "PASS":
            acc_problems.append(
                f"DFlash@{clabel}: SPEC_AGREE={e.get('spec_agree', 0):.4f} VERDICT={e.get('verdict', '?')}"
            )
    if acc_problems:
        label = "REJECT"
        passed = False
        reason = "DFlash accuracy failed at: " + "; ".join(acc_problems)

    guard_ok, guard_problems = check_qwen_guard(pr, main)
    if not guard_ok:
        # DFlash-only scoring is only valid alongside a clean Qwen3.5/3.6 guard — a regression
        # there overrides any DFlash tier, however good, into REJECT.
        label = "REJECT"
        passed = False
        reason = "qwen3.5/qwen3.6 no-regression guard failed: " + "; ".join(guard_problems[:6])
        if acc_problems:
            reason = ("DFlash accuracy failed at: " + "; ".join(acc_problems) +
                       " | also: " + reason)

    # Headline PR/main tok/s reflect whichever context the label was actually scored at; REJECT
    # has no winning context, so fall back to the primary/short one (0) for display purposes.
    h = dflash_ctx[best_ctx] if best_ctx is not None else dflash_ctx[0]
    res = {
        "ok": True,
        "label": label,
        "pass": passed and label != "REJECT",
        "reason": reason,
        "delta_pct": delta_pct,
        "pr_dflash_tps": h["pr_dflash_tps"],
        "pr_ar_tps": h.get("pr_ar_tps"),
        "main_dflash_tps": h["main_dflash_tps"],
        "main_ar_tps": h.get("main_ar_tps"),
        "scored_ctx": GUARD_CTX_LABEL.get(best_ctx, str(best_ctx)) if best_ctx is not None else None,
        "mean_accept": pr.get("mean_accept"),
        "spec_agree": pr.get("spec_agree"),
        "prompt_ids": ids,
        "speedup_vs_main": round(h["pr_dflash_tps"] / h["main_dflash_tps"], 3) if h.get("main_dflash_tps") else 0,
        "speedup_vs_ar": round(h["pr_dflash_tps"] / h["pr_ar_tps"], 3) if h.get("pr_ar_tps") else 0,
        "guard_ok": guard_ok,
        "guard_problems": guard_problems,
        "guard36": pr.get("guard36"),
        "guard35": pr.get("guard35"),
        "dflash_ctx": dflash_ctx,
        "dflash_ctx_acc": pr.get("dflash_ctx_acc"),
    }
    polaris = collect_polaris_attestation(host, port, res, pr_ref)
    if polaris:
        res["polaris"] = polaris
    return res


def format_comment(commit: str, res: dict) -> str:
    meta = {
        "label": res.get("label"),
        "delta_pct": res.get("delta_pct"),
        "pr_dflash_tps": res.get("pr_dflash_tps"),
        "main_dflash_tps": res.get("main_dflash_tps"),
        "pass": res.get("pass"),
    }
    marker = (
        f"<!-- sparkinfer-dflash-eval:{EVAL_SCHEMA_VERSION}:{commit} "
        f"{json.dumps(meta, separators=(',', ':'))} -->"
    )
    if not res.get("ok"):
        return (
            f"{marker}\n## sparkinfer dflash auto-eval — error\n\n"
            f"**reason:** `{res.get('reason')}`\n\n"
            f"<details><summary>log tail</summary>\n\n```\n{(res.get('log') or '')[:1800]}\n```\n</details>\n"
        )
    lab = res["label"]
    guard_ok = res.get("guard_ok")
    if guard_ok is None:
        guard_row = "| Qwen3.5/3.6 guard | — |\n"
    elif guard_ok:
        guard_row = "| Qwen3.5/3.6 guard | ✅ no regression (decode + prefill) |\n"
    else:
        problems = res.get("guard_problems") or []
        guard_row = "| Qwen3.5/3.6 guard | ❌ **REGRESSED** — DFlash score voided |\n"
        guard_row += "".join(f"| &nbsp;&nbsp;↳ | `{p}` |\n" for p in problems[:8])
    polaris = res.get("polaris") or {}
    receipt = polaris.get("receipt")
    if receipt:
        rtype = "TDX (Intel hardware attestation)" if receipt.get("attestation_type") == "tdx-quote" \
            else "Ed25519 (SparkInfer key)"
        polaris_row = (
            f"| Polaris receipt | `{receipt.get('receipt_id', '?')[:16]}…` — {rtype} |\n"
        )
    elif polaris.get("attestation"):
        polaris_row = "| Polaris receipt | collected, not signed (no key configured) |\n"
    else:
        polaris_row = ""
    scored_ctx = res.get("scored_ctx")
    dctx_acc = res.get("dflash_ctx_acc") or {}
    ctx_rows = ""
    dctx = res.get("dflash_ctx") or {}
    for ctx in sorted(dctx):
        e = dctx[ctx]
        clabel = GUARD_CTX_LABEL.get(ctx, str(ctx))
        dp, dm, delta = e.get("pr_dflash_tps"), e.get("main_dflash_tps"), e.get("delta_pct")
        dtxt = f"{delta:+.1f}%" if delta is not None else "n/a"
        star = " ⭐ *(scored)*" if scored_ctx is not None and clabel == scored_ctx else ""
        ctx_rows += f"| DFlash @{clabel} tok/s{star} | PR {dp:.2f} vs main {dm:.2f} ({dtxt}) |\n"
        acc = dctx_acc.get(ctx)
        if acc:
            atxt = "✅ PASS" if acc.get("verdict") == "PASS" else f"❌ **{acc.get('verdict', 'FAIL')}**"
            ctx_rows += f"| &nbsp;&nbsp;↳ accuracy | {atxt} (SPEC_AGREE={acc.get('spec_agree', 0):.4f}) |\n"
    return (
        f"{marker}\n## sparkinfer dflash auto-eval — `eval-dflash:{lab}`\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| **label** | `eval-dflash:{lab}` |\n"
        f"| scored at context | {scored_ctx or 'n/a'} (best of 128/512/4k; regression at ANY rejects) |\n"
        f"| PR DFlash tok/s | {res['pr_dflash_tps']:.2f} |\n"
        f"| main DFlash tok/s | {res['main_dflash_tps']:.2f} |\n"
        f"| speedup vs main | **{res['speedup_vs_main']:.2f}×** ({res['delta_pct']:+.1f}%) |\n"
        f"| PR AR tok/s | {res.get('pr_ar_tps') or 0:.2f} |\n"
        f"| DFlash vs AR | {res.get('speedup_vs_ar') or 0:.2f}× |\n"
        f"| mean accept τ | {res.get('mean_accept') or 0:.3f} |\n"
        f"| accuracy | {res.get('spec_agree') or 'VERDICT PASS'} |\n"
        f"{guard_row}"
        f"{ctx_rows}"
        f"{polaris_row}"
        f"| commit | `{commit[:9]}` |\n\n"
        f"{res.get('reason') or ''}\n\n"
        "<sub>Scored on pinned RTX 5090 vs same-box `origin/main` DFlash, gated by a "
        "same-build Qwen3.5/3.6 decode+prefill no-regression guard. "
        "AR `eval:*` labels are frozen (casual bidir eval retired).</sub>\n"
    )


def auto_merge_ok_dflash(repo, num):
    info = json.loads(arb.gh([
        "pr", "view", str(num), "-R", repo, "--json",
        "state,isDraft,labels,author,mergeable,files",
    ]).stdout or "{}")
    if info.get("state") != "OPEN" or info.get("isDraft"):
        return False, "not an open, non-draft PR"
    labs = {l["name"] for l in info.get("labels", [])}
    tiers = {l.split(":", 1)[1] for l in labs if l.startswith(EVAL_PREFIX)}
    if not (tiers & SPEEDUP_LABELS):
        return False, "no verified eval-dflash:speedup label"
    if DFLASH_MERGE_FIRST not in labs:
        return False, "not dflash-merge-first"
    blocked = labs & AUTOMERGE_BLOCK
    if blocked:
        return False, f"blocking label(s): {', '.join(sorted(blocked))}"
    author = (info.get("author") or {}).get("login", "")
    if author.lower() in arb.load_denylist():
        return False, f"author {author} is blocked"
    if arb.author_penalty_until(author):
        return False, f"author {author} is under penalty"
    sens = [f["path"] for f in info.get("files", [])
            if any(f["path"].startswith(p) for p in arb.AUTOMERGE_SENSITIVE)]
    if sens:
        return False, f"touches protected paths: {', '.join(sens[:3])}"
    if arb.pr_merge_conflict(info.get("mergeable")):
        return False, "merge conflict with base"
    if info.get("mergeable") != "MERGEABLE":
        return False, f"not cleanly mergeable ({info.get('mergeable')})"
    return True, "ok"


def try_auto_merge_dflash(repo, num):
    ok, reason = auto_merge_ok_dflash(repo, num)
    if not ok:
        print(f">> dflash auto-merge SKIP #{num}: {reason}")
        return False
    r = arb.gh(["pr", "merge", str(num), "-R", repo, "--squash"])
    if r.returncode != 0 and os.environ.get("SPARKINFER_AUTOMERGE_ADMIN", "1") == "1":
        err = ((r.stderr or "") + (r.stdout or "")).lower()
        if "not mergeable" in err or "branch policy" in err or "required" in err or "prohibited" in err:
            print(">> dflash auto-merge: branch policy blocked — retrying with --admin")
            r = arb.gh(["pr", "merge", str(num), "-R", repo, "--squash", "--admin"])
    if r.returncode == 0:
        print(f">> DFLASH AUTO-MERGED #{num} (dflash-merge-first)")
        arb.gh(["pr", "comment", str(num), "-R", repo, "--body",
                "<!-- sparkinfer-dflash-automerge -->\n"
                "Auto-merged as the round's `dflash-merge-first` winner — "
                "verified same-box DFlash speedup over `main` with SPEC_AGREE pass."])
        return True
    print(f">> dflash auto-merge BLOCKED #{num}: {(r.stderr or r.stdout or '')[:200]}")
    return False


def reconcile_dflash_merge_labels(repo, dry_run=False):
    scores = _load_scores()
    open_prs = json.loads(arb.gh([
        "pr", "list", "-R", repo, "--state", "open",
        "--json", "number,labels", "--limit", "80",
    ]).stdout or "[]")
    open_labels = {p["number"]: {l["name"] for l in p["labels"]} for p in open_prs}

    # Clear dflash-merge-first from recently merged PRs
    merged = json.loads(arb.gh([
        "pr", "list", "-R", repo, "--state", "merged", "--label", DFLASH_MERGE_FIRST,
        "--json", "number", "--limit", "10",
    ]).stdout or "[]")
    for m in merged:
        if not dry_run:
            arb.remove_label(repo, m["number"], DFLASH_MERGE_FIRST)

    scored = []
    for num, labs in open_labels.items():
        if DFLASH_NEEDS_REBASE in labs:
            continue
        tiers = {l.split(":", 1)[1] for l in labs if l.startswith(EVAL_PREFIX)}
        tier = next((t for t in tiers if t in SPEEDUP_LABELS), None)
        if not tier:
            continue
        entry = scores.get(str(num)) or {}
        if entry.get("label") not in SPEEDUP_LABELS:
            # Prefer live label; use stored delta if present
            if tier not in SPEEDUP_LABELS:
                continue
            entry = {"label": tier, "delta_pct": entry.get("delta_pct") or 0}
        scored.append((num, float(entry.get("delta_pct") or 0), entry.get("label") or tier))

    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        print(">> dflash round: no verified speedup PRs")
        return
    winner = scored[0][0]
    print(f">> dflash round: merge-first #{winner}; rebase {[n for n,_,_ in scored[1:]] or 'none'}")
    if dry_run:
        return
    arb.add_label(repo, winner, DFLASH_MERGE_FIRST)
    arb.remove_label(repo, winner, DFLASH_NEEDS_REBASE)
    for num, _, _ in scored[1:]:
        arb.add_label(repo, num, DFLASH_NEEDS_REBASE)
        arb.remove_label(repo, num, DFLASH_MERGE_FIRST)
    if AUTO_MERGE:
        try_auto_merge_dflash(repo, winner)


def upload_dflash_eval_log(repo, num, title, oid, res):
    """Commit the DFlash eval result (+ Polaris receipt/attestation) to sparkinfer-log — the
    same public ledger the AR bot writes to (gittensor-ai-lab/sparkinfer-log), just under a
    dflash-prefixed run id so it can't collide with an AR-bot entry for the same PR number.
    Best-effort: never blocks or raises — a log-upload failure must not affect the verdict
    already posted to the PR."""
    try:
        rid = f"dflash-{int(num):04d}-{oid[:7]}"
        arb._ensure_log_repo()
        rundir = os.path.join(arb.LOG_DIR, "runs", rid)
        os.makedirs(rundir, exist_ok=True)
        polaris = res.get("polaris") or {}
        receipt = polaris.get("receipt")
        result = {
            "id": rid, "pr": int(num), "title": title,
            "url": f"https://github.com/{repo}/pull/{num}", "commit": oid[:7],
            "eval_mode": "dflash",
            "label": res.get("label"), "pass": res.get("pass"), "reason": res.get("reason"),
            "delta_pct": res.get("delta_pct"),
            "pr_dflash_tps": res.get("pr_dflash_tps"), "main_dflash_tps": res.get("main_dflash_tps"),
            "pr_ar_tps": res.get("pr_ar_tps"), "main_ar_tps": res.get("main_ar_tps"),
            "speedup_vs_main": res.get("speedup_vs_main"), "speedup_vs_ar": res.get("speedup_vs_ar"),
            "mean_accept": res.get("mean_accept"), "spec_agree": res.get("spec_agree"),
            "guard_ok": res.get("guard_ok"), "guard_problems": res.get("guard_problems"),
            "dflash_ctx": res.get("dflash_ctx"), "scored_ctx": res.get("scored_ctx"),
            "gpu": "RTX 5090 (sm_120)", "date": arb.datetime.date.today().isoformat(),
        }
        if receipt:
            result["polaris"] = True
            result["polaris_receipt_id"] = receipt.get("receipt_id")
        json.dump(result, open(os.path.join(rundir, "result.json"), "w"), indent=2)
        if polaris.get("attestation"):
            json.dump(polaris["attestation"], open(os.path.join(rundir, "attestation.json"), "w"), indent=2)
        if receipt:
            json.dump(receipt, open(os.path.join(rundir, "receipt.json"), "w"), indent=2)
        ipath = os.path.join(arb.LOG_DIR, "index.json")
        idx = json.load(open(ipath)) if os.path.exists(ipath) else []
        idx = [e for e in idx if e.get("id") != rid]
        idx_entry = {"id": rid, "pr": int(num), "title": title, "label": res.get("label"),
                     "delta_pct": res.get("delta_pct"), "eval_mode": "dflash", "date": result["date"]}
        if receipt:
            idx_entry["polaris"] = True
            idx_entry["polaris_receipt_id"] = receipt.get("receipt_id", "")[:16]
        idx.append(idx_entry)
        idx.sort(key=lambda x: x["id"])
        json.dump(idx, open(ipath, "w"), indent=2)
        subprocess.run(["git", "-C", arb.LOG_DIR, "add", "-A"], check=True)
        msg = f"dflash-eval: #{num} {oid[:7]} -> eval-dflash:{res.get('label')}"
        if receipt:
            msg += f" + polaris {receipt.get('receipt_id', '?')[:16]}"
        commit = subprocess.run(["git", "-C", arb.LOG_DIR, "commit", "-q", "-m", msg], check=False)
        if commit.returncode != 0:
            print(">> dflash eval-log upload skipped: nothing to commit")
            return None
        push = subprocess.run(["git", "-C", arb.LOG_DIR, "push", "-q"], check=False)
        if push.returncode != 0:
            print(f">> dflash eval-log push failed (rc={push.returncode})")
            return None
        url = arb.LOG_PAGE + rid
        print(f">> dflash eval log: {url}")
        return url
    except Exception as e:
        print(f">> dflash eval-log upload failed: {e}")
        return None


def apply_result(repo, num, commit, res, title="", dry_run=False):
    body = format_comment(commit, res)
    label = res.get("label") if res.get("ok") else "REJECT"
    if not res.get("ok"):
        label = "REJECT"
    print(f"PR #{num}: eval-dflash:{label}  "
          f"PR={res.get('pr_dflash_tps')} main={res.get('main_dflash_tps')} "
          f"delta={res.get('delta_pct')}%")
    if dry_run:
        print(body[:500])
        return
    strip_dflash_eval_labels(repo, num)
    arb.add_label(repo, num, f"{EVAL_PREFIX}{label}")
    # Also apply the AR bot's eval:* convention (same tier value) — SN74 scoring reads eval:*
    # tiers, not eval-dflash:*, and the AR bot no longer runs on cron to post them itself. Without
    # this a DFlash-verified speedup would be invisible to anything that only looks at eval:*.
    for lab in {l for l in arb.labels_on(repo, num) if l.startswith("eval:")}:
        arb.remove_label(repo, num, lab)
    arb.add_label(repo, num, f"eval:{label}")
    arb.gh(["pr", "comment", str(num), "-R", repo, "--body", body])
    if res.get("ok"):
        upload_dflash_eval_log(repo, num, title, commit, res)
    if res.get("ok") and res.get("delta_pct") is not None:
        scores = _load_scores()
        scores[str(num)] = {
            "commit": commit,
            "label": label,
            "delta_pct": res.get("delta_pct"),
            "pr_dflash_tps": res.get("pr_dflash_tps"),
            "main_dflash_tps": res.get("main_dflash_tps"),
            "pass": res.get("pass"),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _save_scores(scores)
        # Auto-close on a genuine none/REJECT verdict (a real benchmark that ran to completion
        # and just wasn't a win) -- but ONLY when res["ok"] and delta_pct actually exist, i.e. a
        # real pair of numbers came back. An infra failure (crash, OOM, timeout) also gets
        # labeled REJECT by the fallback above with delta_pct=None, and must NOT close the PR --
        # #720 crashed on one eval (REJECT, PR=None main=None) then came back +65% (XL) on a
        # clean re-run; closing on that first crash would have killed a real win.
        if label in ("none", "REJECT"):
            close_body = (
                "<!-- sparkinfer-dflash-auto-close -->\n"
                f"## Closed: sparkinfer dflash auto-eval — `eval-dflash:{label}`\n\n"
                f"This PR's DFlash speed measured **{res.get('delta_pct')}%** vs main "
                f"({'no verified improvement' if label == 'none' else 'regression'}) — "
                "closing automatically. Reopen (or open a fresh PR) if you have a fix or a "
                "different approach."
            )
            arb.gh(["pr", "comment", str(num), "-R", repo, "--body", close_body])
            arb.gh(["pr", "close", str(num), "-R", repo])
            print(f">> auto-closed PR #{num} (eval-dflash:{label})")


def main():
    ap = argparse.ArgumentParser(description="DFlash PR eval bot")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--repo", default="gittensor-ai-lab/sparkinfer")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reeval", action="store_true")
    ap.add_argument("--labels-only", action="store_true",
                    help="reconcile dflash-merge-first only — no GPU")
    ap.add_argument("--only-prs", default="",
                    help="comma-separated PR numbers (bypass greenlight)")
    args = ap.parse_args()

    only = {int(x) for x in args.only_prs.split(",") if x.strip().isdigit()}

    print(f">> dflash eval transport: "
          f"{'ssh' if ssh_box_enabled() else f'vast.ai (instance {arb.current_instance(args.instance) or args.instance})'}")
    print(f">> AUTOMERGE={int(AUTO_MERGE)}")

    if args.labels_only:
        reconcile_dflash_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — dflash labels only (no GPU).")
        return

    prs = json.loads(arb.gh([
        "pr", "list", "-R", args.repo, "--state", "open",
        "--json", "number,title,labels,isDraft,headRefOid,headRefName,mergeable,author,body",
        "--limit", "80",
    ]).stdout or "[]")
    prs.sort(key=lambda p: p["number"])

    pending = []
    for pr in prs:
        num = pr["number"]
        if only and num not in only:
            continue
        if pr.get("isDraft"):
            continue
        labs = {l["name"] for l in pr.get("labels", [])}
        if arb.HOLD_LABEL in labs:
            print(f"PR #{num}: hold — skip")
            continue
        head = (pr.get("headRefOid") or "")[:40]
        short = head[:9]
        if not args.reeval and head and head in dflash_evaluated_commits(args.repo, num):
            print(f"PR #{num} @ {short}: already dflash-evaluated — skip")
            continue
        if arb.pr_merge_conflict(pr.get("mergeable")):
            print(f"PR #{num}: merge conflict — dflash-needs-rebase")
            if not args.dry_run:
                arb.add_label(args.repo, num, DFLASH_NEEDS_REBASE)
            continue

        if not only:
            status, why = arb.greenlight_status(args.repo, num, labs)
            if status != "ok":
                print(f"PR #{num}: not greenlit ({why}) — skip dflash eval")
                continue
            print(f"PR #{num}: greenlit ({why})")
        else:
            print(f"PR #{num}: --only-prs targeted")

        ref = f"pull/{num}/head"
        # Same-repo branches can use headRefName; pull/N/head always works.
        pending.append((num, head, short, ref, pr.get("title", "")))

    if not pending:
        reconcile_dflash_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — no dflash PRs to evaluate.")
        return

    if args.dry_run:
        print("--- dry-run would evaluate: " + ", ".join(f"#{n}" for n, *_ in pending))
        return

    # Pin instance file
    pin = arb.PINNED_INSTANCE
    if pin and not ssh_box_enabled():
        with open(arb.INSTANCE_FILE, "w") as f:
            f.write(str(pin))

    try:
        host, port = resolve_ssh(args.instance)
    except Exception as e:
        print(f">> GPU unavailable: {e}")
        reconcile_dflash_merge_labels(args.repo, dry_run=False)
        print("done — dflash labels only (GPU down).")
        return

    _ssh_user = ssh_box_user() if ssh_box_enabled() else "root"
    print(f">> SSH {_ssh_user}@{host}:{port}")

    for num, head, short, ref, title in pending:
        print(f"PR #{num} @ {short}: evaluating DFlash '{ref}' …")
        try:
            res = eval_dflash_on_box(host, port, ref)
        except Exception as e:
            res = {"ok": False, "reason": f"exception: {e}"}
        apply_result(args.repo, num, head or short, res, title=title, dry_run=False)

    reconcile_dflash_merge_labels(args.repo, dry_run=False)
    print("done — dflash eval pass complete.")


if __name__ == "__main__":
    main()
