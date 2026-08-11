#!/usr/bin/env python3
"""sparkinfer Muse Glimmer PR auto-evaluator.

Sibling of pr_dflash_bot.py — narrowly scoped to Muse Glimmer's plain AR (autoregressive)
decode at 128-token context ONLY. Not DFlash, not long-context, not prefill. This scope is
deliberate: Muse Glimmer is a very young architecture (4 real correctness bugs found and fixed
in a single bring-up session, commit 6d911d4 and preceding) with essentially zero production
track record, so this bot stays small and strict rather than growing the same multi-context /
cross-model-guard surface pr_dflash_bot.py has.

Scoring, same-box PR-vs-main on a single pinned GPU:
  1. Speed  — qwen3_gguf_bench <gguf> 128 0 (128-token decode, no prefill context), PR vs a
              freshly-measured origin/main, same box, same run. Same tier buckets as the AR and
              DFlash bots (BUCKETS/SIG/REGRESS_TOL below — copied, not reinvented).
  2. Accuracy gate — teacher-forced qwen3_gguf_score vs a live llama-server reference on the
              SAME GGUF, exactly the methodology validated by hand this session (commit
              6d911d4's message): qwen3_gguf_score dumps sparkinfer's per-position distribution,
              llama-server answers /completion (n_probs + cache_prompt + temperature=0) for the
              same positions, accuracy_compare.py reports top1/KL. Bar: top1 >= 0.90, KL <= 0.10
              (this session achieved 0.980 / 0.0029 on the exact eval_text.txt corpus reused
              here). A PR that fails this bar is REJECTed regardless of speed — speed is
              meaningless if a PR silently breaks a still-fragile architecture's correctness.

Applies `eval-museglimmer:<TIER>` AND mirrors it to the generic `eval:<TIER>` label (SN74
scoring reads eval:* tiers) — explicit user decision, 2026-08-11; originally NOT mirrored given
Muse Glimmer's youth at the time, see git history on apply_result() for that reasoning. Auto-
close on none/REJECT and auto-merge on a verified speedup are both live (SPARKINFER_MUSEGLIMMER_
AUTOMERGE=1 in .env.eval) — same policy as pr_dflash_bot.py, also an explicit user decision
after this bot's first live run wrongly auto-closed an unrelated PR (#768, reopened +
apologized); the user was told the risk directly and chose to accept it rather than narrow the
evaluation scope. No Qwen3.5/Qwen3.6/Qwythos cross-model guard — that safety net is a pushed
backup branch (backup/qwen-model-optimization-20260811 @ 6d911d4) instead of a per-PR check,
per explicit scope decision.

  python eval/pr_museglimmer_bot.py --instance 46074104
  python eval/pr_museglimmer_bot.py --only-prs 636 --reeval

Never rents a GPU. Shares the pinned box with the AR and DFlash bots via flock in the cron
wrapper (run_museglimmer_cron.sh) — all three MUST share /tmp/sparkinfer_bot.lock.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ssh_box import ssh_box_enabled, ssh_box_endpoint, ssh_box_user  # noqa: E402

# Reuse shared helpers from the AR bot (labels, greenlight, denylist, gh() wrapper, Polaris
# signing, stale-close primitives, …) — same import pattern as pr_dflash_bot.py.
import pr_eval_bot as arb  # noqa: E402

SPEEDUP_LABELS = {"XL", "L", "M", "S", "XS"}
# Same tier-bucketing constants as pr_dflash_bot.py / pr_eval_bot.py — copied verbatim, not
# reinvented, per explicit instruction.
SIG = 0.02
REGRESS_TOL = 0.98
BUCKETS = [(0.18, "XL"), (0.10, "L"), (0.06, "M"), (0.035, "S"), (SIG, "XS")]

# Accuracy gate bars — validated by hand this session (6d911d4: top1=0.980, KL=0.0029 achieved
# on the 99-token eval_text.txt corpus; bars themselves are the short-pass bars this codebase
# already uses elsewhere, see accuracy_compare.py's own docstring bar for top-1).
ACC_TOP1_BAR = float(os.environ.get("MUSEGLIMMER_ACC_TOP1_BAR", "0.90"))
ACC_KL_BAR = float(os.environ.get("MUSEGLIMMER_ACC_KL_BAR", "0.10"))

EVAL_PREFIX = "eval-museglimmer:"
MUSEGLIMMER_MERGE_FIRST = "museglimmer-merge-first"
MUSEGLIMMER_NEEDS_REBASE = "museglimmer-needs-rebase"
EVAL_SCHEMA_VERSION = "v1"
MARKER_RE = re.compile(
    r"<!-- sparkinfer-museglimmer-eval:" + re.escape(EVAL_SCHEMA_VERSION) + r":([0-9a-f]+)(?:\s+(\{.*?\}))? -->",
    re.DOTALL,
)

# --- box paths (see .env.eval's MUSEGLIMMER_* block) ---
# Deliberately a SEPARATE clone from DFLASH_REMOTE_REPO/pr_eval_bot's /root/sparkinfer: this
# session validated Muse Glimmer support against /root/sparkinfer_mg specifically, while
# /root/sparkinfer carries unrelated uncommitted work from a different task that must not be
# touched or built against.
REMOTE_REPO = os.environ.get("MUSEGLIMMER_REMOTE_REPO", "/root/sparkinfer_mg")
DEFAULT_GGUF = os.environ.get(
    "MUSEGLIMMER_GGUF", "/root/workspace/models_muse_glimmer/muse-glimmer-30B-kquant-17gb.gguf"
)
DEFAULT_MODELS_DIR = os.environ.get("MUSEGLIMMER_MODELS_DIR", "/root/workspace/models_muse_glimmer")
# Shared llama.cpp checkout used by every eval bot on this box (persists outside any repo
# checkout so `git clean -qfd` in the remote script's checkout step can never delete it).
LLAMACPP_DIR = os.environ.get("LLAMACPP_DIR", "/root/workspace/.llamacpp")
BENCH_TOKENS = int(os.environ.get("MUSEGLIMMER_BENCH_TOKENS", "128"))
ACC_TOPK = int(os.environ.get("MUSEGLIMMER_ACC_TOPK", "128"))
# Fixed local port for the reference llama-server this bot starts/stops per run. Distinct from
# accuracy.sh's interactive default (8081) purely so a manual accuracy.sh run on the same box
# can't collide with a bot tick (the flock in the cron wrapper already prevents two bot ticks
# from overlapping with each other).
LLAMA_SERVER_PORT = int(os.environ.get("MUSEGLIMMER_LLAMA_PORT", "8097"))
EVAL_TEXT = "bench/scripts/eval_text.txt"  # the exact known-good corpus from this session's fixes

# Auto-merge is wired (mirrors pr_dflash_bot.py's auto_merge_ok_dflash/try_auto_merge_dflash
# shape) but OFF unless this exact env var is set — NOT set in .env.eval, so it stays fully
# inert until a human deliberately flips it on. Single-line change to enable later.
AUTO_MERGE = os.environ.get("SPARKINFER_MUSEGLIMMER_AUTOMERGE") == "1"
AUTOMERGE_BLOCK = {
    "copycat", "copycat-warn", "flagged:gaming", "penalty", "needs-benchmark",
    MUSEGLIMMER_NEEDS_REBASE, arb.REEVALUATE_LABEL, arb.HOLD_LABEL, *arb.REGRESSION_LABELS,
}

SCORES_FILE = os.path.expanduser(
    os.environ.get("MUSEGLIMMER_SCORES_FILE", "~/.sparkinfer_museglimmer_scores.json")
)

# Polaris verifiable-compute receipts — same policy/keys as the AR and DFlash bots (on by
# default; TDX via POLARIS_API_KEY when configured, else Ed25519 fallback). Wired through
# judge.py's --from-stdin generic RESULT_JSON path (NOT --dflash, which hardcodes a
# DFlash-shaped measurement block and eval_mode="dflash" — reusing it here would produce a
# mislabeled, semantically wrong attestation). SPARKINFER_EVAL_MODE is set explicitly below so
# the attestation correctly records "museglimmer-128", not the AR bot's "longctx" default.
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
        print(f">> museglimmer scores save skipped: {e}")


def tier_from_gain(pr_tps: float, main_tps: float):
    """Return (label, delta_pct, pass_ok, reason). Identical logic to pr_dflash_bot.py's
    tier_from_gain — same bucket thresholds, same no-regression floor."""
    if main_tps <= 0:
        return "REJECT", 0.0, False, "main decode baseline is 0"
    if pr_tps < REGRESS_TOL * main_tps:
        pct = 100.0 * (pr_tps - main_tps) / main_tps
        return "REJECT", round(pct, 1), False, (
            f"decode regression: {pr_tps:.2f} < {100 * REGRESS_TOL:.0f}% of main {main_tps:.2f}"
        )
    g = (pr_tps - main_tps) / main_tps
    pct = round(100.0 * g, 1)
    if g < SIG:
        return "none", pct, True, "within significance gate — not a verified decode improvement"
    for thr, name in BUCKETS:
        if g >= thr:
            return name, pct, True, "ok"
    return "none", pct, True, "ok"


def museglimmer_evaluated_commits(repo, num):
    """Head commits that already have a REAL scoring verdict posted — mirrors
    dflash_evaluated_commits: infra/transport failures (label:null in the marker) don't count."""
    r = arb.gh(["pr", "view", str(num), "-R", repo, "--json", "comments"])
    done = set()
    for c in json.loads(r.stdout or "{}").get("comments", []):
        body = c.get("body") or ""
        m = MARKER_RE.search(body)
        if not m or "sparkinfer museglimmer auto-eval" not in body:
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


def strip_museglimmer_eval_labels(repo, num):
    for lab in list(arb.labels_on(repo, num)):
        if lab.startswith(EVAL_PREFIX):
            arb.remove_label(repo, num, lab)


STALE_DAYS = float(os.environ.get("MUSEGLIMMER_STALE_DAYS", "1"))


def _pr_last_activity_ts(repo, num):
    """Last real author activity (most recent commit's committedDate), not PR updatedAt — same
    rationale as pr_dflash_bot.py's copy of this helper (bot comments/labels bump updatedAt)."""
    r = arb.gh(["pr", "view", str(num), "-R", repo, "--json", "commits,createdAt"])
    try:
        info = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return None
    dates = [c.get("committedDate") for c in (info.get("commits") or []) if c.get("committedDate")]
    ts_str = max(dates) if dates else info.get("createdAt")
    if not ts_str:
        return None
    try:
        return time.mktime(time.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return None


def close_stale_museglimmer_prs(repo, prs, dry_run=False):
    """Close open PRs with no author commit activity in STALE_DAYS+ days. HOLD_LABEL and the
    current museglimmer-merge-first winner are exempt."""
    closed = set()
    now = time.time()
    for pr in prs:
        num = pr["number"]
        if pr.get("isDraft"):
            continue
        labs = {l["name"] for l in pr.get("labels", [])}
        if arb.HOLD_LABEL in labs or MUSEGLIMMER_MERGE_FIRST in labs:
            continue
        ts = _pr_last_activity_ts(repo, num)
        if ts is None:
            continue
        age_days = (now - ts) / 86400
        if age_days < STALE_DAYS:
            continue
        print(f"PR #{num}: stale ({age_days:.1f}d since last commit, threshold {STALE_DAYS}d) — closing")
        closed.add(num)
        if dry_run:
            continue
        body = (
            "<!-- sparkinfer-museglimmer-auto-close-stale -->\n"
            f"## Closed: stale — no commits in {age_days:.1f} days\n\n"
            f"This PR has had no new commits in over {STALE_DAYS:g} days — closing automatically "
            "to keep the Muse Glimmer eval queue clean. Reopen (or push a new commit / open a "
            "fresh PR) whenever you're ready to continue; it'll be picked back up on the next "
            "eval cycle."
        )
        arb.gh(["pr", "comment", str(num), "-R", repo, "--body", body])
        arb.gh(["pr", "close", str(num), "-R", repo])
    return closed


def resolve_ssh(instance_id: int):
    """Return (host, port) for the pinned box. Same logic as pr_dflash_bot.py's resolve_ssh."""
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


def ssh_run(host, port, cmd, timeout=7200, stdin_data=None, via_stdin=False):
    """Same shape as pr_dflash_bot.py's ssh_run — via_stdin=True avoids MAX_ARG_STRLEN limits
    for the (much smaller here, but still nontrivial) remote script text."""
    key = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/speedy"))
    user = ssh_box_user() if ssh_box_enabled() else "root"
    remote = ["bash", "-s"] if via_stdin else [cmd]
    return subprocess.run(
        [
            "ssh", "-i", key,
            "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=40",
            "-p", str(port), f"{user}@{host}", *remote,
        ],
        capture_output=True, text=True, timeout=timeout,
        input=cmd if via_stdin else stdin_data,
    )


def _crash_reason(*outputs: str) -> str | None:
    """Same ERR-trap diagnostic extraction as pr_dflash_bot.py."""
    combined = "\n".join(o or "" for o in outputs)
    lines = combined.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("REMOTE_SCRIPT_FAILED "):
            extra = lines[i + 1].strip() if i + 1 < len(lines) else ""
            return line.strip() + (f" | gpu: {extra}" if extra else "")
    return None


def _looks_like_hard_kill(stdout: str, stderr: str) -> bool:
    """Same hard-kill heuristic as pr_dflash_bot.py (no ERR-trap diagnostic captured at all)."""
    combined = (stdout or "") + "\n" + (stderr or "")
    if _crash_reason(stdout, stderr):
        return False
    return "ACCURACY_STAGE_DONE" not in combined  # never reached even the accuracy checkpoint


def _ssh_run_resilient(host, port, script: str, label: str):
    """One automatic retry on an apparent hard kill — same insurance pr_dflash_bot.py added
    after #684/#690 (heavy model-reload boundaries silently killing the whole remote shell)."""
    r = ssh_run(host, port, script, via_stdin=True)
    if r.returncode != 0 and _looks_like_hard_kill(r.stdout, r.stderr):
        print(f">> {label}: looks like a hard kill (no ERR-trap diagnostic, no accuracy-stage "
              f"checkpoint reached) — retrying once")
        r = ssh_run(host, port, script, via_stdin=True)
    return r


def _remote_script(ref: str) -> str:
    """Bash run on the eval box: checkout ref, build, 128-decode speed bench, accuracy gate vs
    a live llama-server reference. Run once per ref (PR, then "main") — identical script both
    times so the two measurements are directly comparable."""
    repo = shlex.quote(REMOTE_REPO)
    gguf = shlex.quote(DEFAULT_GGUF)
    llamacpp_dir = shlex.quote(LLAMACPP_DIR)
    ref_q = shlex.quote(ref)
    ntok = BENCH_TOKENS
    topk = ACC_TOPK
    port = LLAMA_SERVER_PORT
    eval_text = shlex.quote(EVAL_TEXT)
    return f"""
set -euo pipefail
# Surface *why* a crash happened instead of dying silently — same diagnostic trap as
# pr_dflash_bot.py's _remote_script (a REJECT from an infra crash should carry a real cause).
trap 'rc=$?; ln=$LINENO; reason=""; \\
  case $rc in \\
    137) reason="likely OOM-killed (SIGKILL)" ;; \\
    139) reason="likely segfault (SIGSEGV)" ;; \\
    134) reason="likely abort (SIGABRT)" ;; \\
    124) reason="likely timeout" ;; \\
  esac; \\
  echo "REMOTE_SCRIPT_FAILED line=$ln exit=$rc reason=$reason" >&2; \\
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader >&2 2>/dev/null || true' ERR

# One run does several back-to-back multi-GB model load/unload cycles (sparkinfer speed bench,
# sparkinfer score dump, llama-server) — poll GPU memory down to near-empty before each heavy
# load instead of assuming the previous process's exit already freed it (same lesson as
# pr_dflash_bot.py's wait_gpu_clear, #684/#690).
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
REPO={repo}
GGUF={gguf}
NTOK={ntok}
TOPK={topk}
PORT={port}
EVAL_TEXT={eval_text}

cd "$REPO"
git remote set-url origin https://github.com/gittensor-ai-lab/sparkinfer.git 2>/dev/null || true
git fetch -q origin {ref_q}
git reset -q --hard
git clean -qfd
git checkout -qf FETCH_HEAD
HEAD=$(git rev-parse --short HEAD)
echo "REMOTE_HEAD $HEAD"

test -f "$GGUF" || {{ echo "FAIL missing GGUF $GGUF"; exit 1; }}

# Build sparkinfer's speed-bench + teacher-forced-score binaries. Always reconfigure (cheap,
# idempotent) — skipping it on an existing CMakeCache left stale generated Makefiles pointing at
# a DIFFERENT PR branch's files once the checkout switched underneath it (pr_dflash_bot.py
# #693/#694 hit exactly this).
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/mg_cmake.log 2>&1
cmake --build build --target qwen3_gguf_bench qwen3_gguf_score -j"$(nproc)" >/tmp/mg_build.log 2>&1 || {{
  echo "BUILD_FAILED — tail of /tmp/mg_build.log:" >&2
  tail -80 /tmp/mg_build.log >&2
  exit 1
}}
test -x build/runtime/qwen3_gguf_bench
test -x build/runtime/qwen3_gguf_score

# --- 128-token decode speed (ctx=0, n=128 — the SAME "128-context, no prefill" convention the
# rest of this codebase's decode buckets use, e.g. pr_eval_bot.py's CTX_SERIES[128]) ---
wait_gpu_clear
OUT=$(build/runtime/qwen3_gguf_bench "$GGUF" "$NTOK" 0)
echo "$OUT"
DECODE_TPS=$(echo "$OUT" | grep 'decode tg' | tail -1 | sed -E 's/.*decode tg[[:space:]]*:[[:space:]]*([0-9.]+).*/\\1/')
echo "RESULT_DECODE_TPS ${{DECODE_TPS:-0}}"

# --- accuracy gate: sparkinfer teacher-forced score vs a live llama-server reference, same
# GGUF, same eval_text.txt corpus this session already validated by hand (6d911d4) ---
#
# Deliberately NOT bench/scripts/_common.sh's ensure_llamacpp()/reference.lock here: reference.lock
# pins LLAMACPP_COMMIT to a July-2026 commit that predates llama.cpp's native muse-glimmer
# architecture support entirely (merged ~2026-08-10). ensure_llamacpp() tamper-checks the
# checkout's HEAD against that pin and — on mismatch — fetches and resets to the OLD pinned
# commit, which cannot even load a general.architecture=muse-glimmer GGUF. Calling it here would
# silently wreck the box's already-working llama.cpp checkout on the very first tick and break
# every subsequent accuracy gate. reference.lock is shared with the Qwen3.5/Qwen3.6/Qwythos evals
# (pr_eval_bot.py/pr_dflash_bot.py) — bumping it to a bleeding-edge commit to fix this would risk
# changing THEIR baseline numbers, which is out of scope here. Build straight off whatever's
# already checked out at LLAMACPP_DIR instead, unpinned, isolated to this bot only.
LLAMACPP_DIR={llamacpp_dir}
if [ ! -d "$LLAMACPP_DIR/.git" ]; then
  echo "FAIL: $LLAMACPP_DIR missing or not a git checkout. Muse Glimmer's accuracy gate needs a" >&2
  echo "llama.cpp build with native muse-glimmer support (src/models/muse-glimmer.cpp) -- clone" >&2
  echo "https://github.com/ggml-org/llama.cpp fresh into this path before the bot's first run." >&2
  exit 1
fi
mkdir -p "$LLAMACPP_DIR/build"
cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \\
  -DCMAKE_BUILD_TYPE=Release >/tmp/mg_llamacpp_cmake.log 2>&1 || {{
  echo "LLAMACPP_CONFIGURE_FAILED — tail:" >&2
  tail -60 /tmp/mg_llamacpp_cmake.log >&2
  exit 1
}}
cmake --build "$LLAMACPP_DIR/build" -j"$(nproc)" --target llama-server llama-tokenize \\
  >/tmp/mg_llamacpp_build.log 2>&1 || {{
  echo "LLAMACPP_BUILD_FAILED — tail:" >&2
  tail -80 /tmp/mg_llamacpp_build.log >&2
  exit 1
}}
test -x "$LLAMACPP_DIR/build/bin/llama-tokenize"
test -x "$LLAMACPP_DIR/build/bin/llama-server"

# Tokenize the known-good corpus with llama-tokenize (not gen_eval_prompt.py's HF-tokenizer
# path) — guarantees byte-identical tokenization to the llama-server reference queried below,
# same approach already proven this session.
TOKOUT=$("$LLAMACPP_DIR/build/bin/llama-tokenize" -m "$GGUF" -f "$EVAL_TEXT" --ids)
IDS=$(echo "$TOKOUT" | grep -oE '[0-9]+' | tr '\\n' ' ')
echo "$IDS" > /tmp/mg_eval_ids.txt
test -n "$IDS" || {{ echo "FAIL empty tokenized IDS from llama-tokenize"; exit 1; }}
echo "TOKEN_COUNT $(echo "$IDS" | wc -w)"

wait_gpu_clear
build/runtime/qwen3_gguf_score "$GGUF" "$TOPK" $IDS > /tmp/mg_score.txt
grep -q '^PPL' /tmp/mg_score.txt || echo "WARN: qwen3_gguf_score produced no PPL line" >&2

wait_gpu_clear
"$LLAMACPP_DIR/build/bin/llama-server" -m "$GGUF" -ngl 99 -c 2048 --port "$PORT" --no-jinja \\
  >/tmp/mg_llama_srv.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 2
done
echo "ACCURACY_STAGE_DONE"

# /dev/null as the tokenizer-path arg is safe: accuracy_compare.py's 3rd positional arg is a
# file of already-tokenized space-separated ids (produced above), so its all-digit check skips
# the HF-tokenizer-load code path entirely — the tokenizer path is never opened.
ACCOUT=$(python3 bench/scripts/accuracy_compare.py /tmp/mg_score.txt /dev/null /tmp/mg_eval_ids.txt \\
         "http://localhost:$PORT" "$TOPK")
echo "$ACCOUT"
kill $SRV 2>/dev/null || true
wait $SRV 2>/dev/null || true
trap - EXIT

METRIC_LINE=$(echo "$ACCOUT" | grep '^METRIC ' | tail -1)
TOP1=$(echo "$METRIC_LINE" | sed -E 's/.*top1=([0-9.]+).*/\\1/')
KL=$(echo "$METRIC_LINE" | sed -E 's/.*kl=([0-9.]+).*/\\1/')
PPLS=$(echo "$METRIC_LINE" | sed -E 's/.*ppl_spark=([0-9.]+).*/\\1/')
PPLL=$(echo "$METRIC_LINE" | sed -E 's/.*ppl_llama=([0-9.]+).*/\\1/')
echo "RESULT_TOP1 ${{TOP1:-0}}"
echo "RESULT_KL ${{KL:-99}}"
echo "RESULT_PPL_SPARK ${{PPLS:-0}}"
echo "RESULT_PPL_LLAMA ${{PPLL:-0}}"
"""


def _parse_remote(stdout: str) -> dict:
    out = {}
    for line in (stdout or "").splitlines():
        if line.startswith("REMOTE_HEAD "):
            out["head"] = line.split()[1]
        elif line.startswith("RESULT_DECODE_TPS "):
            try:
                out["decode_tps"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_TOP1 "):
            try:
                out["top1"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_KL "):
            try:
                out["kl"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_PPL_SPARK "):
            try:
                out["ppl_spark"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_PPL_LLAMA "):
            try:
                out["ppl_llama"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("TOKEN_COUNT "):
            try:
                out["token_count"] = int(line.split()[1])
            except ValueError:
                pass
    return out


def push_eval_polaris(host, port):
    """Sync eval/polaris/ (judge.py + receipt.py) from a TRUSTED source (origin/main) before
    running attestation — same protection as pr_dflash_bot.py's push_eval_polaris (a PR's own
    checkout must never supply the code that produces its own attestation)."""
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
    import tempfile
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
            tmp.write(tar_data)
            tmp_path = tmp.name
        scp = subprocess.run(
            ["scp", "-P", str(port), "-i", key, "-o", "IdentitiesOnly=yes",
             "-o", "StrictHostKeyChecking=accept-new",
             "-o", "BatchMode=yes", tmp_path, f"{user}@{host}:/tmp/si_polaris_mg.tgz"],
            capture_output=True, text=True, timeout=120,
        )
        if scp.returncode != 0:
            print(f">> WARN: eval/polaris scp failed (rc={scp.returncode}): {scp.stderr[-500:]}")
            return False
        extract = (
            f"mkdir -p {shlex.quote(extract_root)} && "
            f"tar -xzf /tmp/si_polaris_mg.tgz -C {shlex.quote(extract_root)} && "
            "rm -f /tmp/si_polaris_mg.tgz"
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
    """Run judge.py --from-stdin (a RESULT_JSON-prefixed line, the AR bot's own generic single-
    model shape) to assemble+sign an attestation. NOT --dflash — that mode hardcodes a
    DFlash-shaped measurement block and eval_mode="dflash", which would be a factually wrong
    label for a plain-AR Muse Glimmer eval. Never raises — a Polaris failure must not block the
    verdict itself, only omit its receipt."""
    if not POLARIS_ENABLED:
        return None
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
    result_json = {
        "model": "museglimmer-128",
        "label": res.get("label"),
        "pass": res.get("pass"),
        "tps": res.get("pr_decode_tps"),
        "delta_tps": (res.get("pr_decode_tps") or 0) - (res.get("main_decode_tps") or 0),
        "pct_over_frontier": res.get("delta_pct"),
        "score_context": 128,
        "best_context_label": "128",
        "ctx_128_tps": res.get("pr_decode_tps"),
        "top1": res.get("pr_top1"),
        "kl": res.get("pr_kl"),
    }
    eval_seed = f"museglimmer-{int(time.time() * 1000)}"  # unique nonce per attestation
    stdin_payload = "RESULT_JSON " + json.dumps(result_json)
    cmd = (
        f"cd {shlex.quote(REMOTE_REPO)} && "
        f"SPARKINFER_EVAL_MODE=museglimmer-128 SPARKINFER_DECODE_TOKENS={BENCH_TOKENS} "
        f"SPARKINFER_EVAL_SEED={shlex.quote(eval_seed)} python3 eval/polaris/judge.py --from-stdin "
        f"--model-file {shlex.quote(DEFAULT_GGUF)} "
        f"--build-dir {shlex.quote(REMOTE_REPO)}/build/runtime "
        f"--sparkinfer-root {shlex.quote(REMOTE_REPO)}"
    )
    try:
        r = ssh_run(host, port, cmd, timeout=60, stdin_data=stdin_payload)
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


def eval_museglimmer_on_box(host, port, pr_ref: str):
    """Run the PR ref's speed+accuracy script, then main's, on the same box. Returns result."""
    print(f">> Muse Glimmer eval on box: PR ref={pr_ref}")
    r = _ssh_run_resilient(host, port, _remote_script(pr_ref), "PR run")
    if r.returncode != 0:
        tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-2000:]
        crash = _crash_reason(r.stdout, r.stderr)
        reason = "PR speed/accuracy run failed" + (f" — {crash}" if crash else " (no crash diagnostic captured, possible hard kill — retried once)")
        return {"ok": False, "reason": reason, "log": tail}
    pr = _parse_remote(r.stdout or "")
    if "decode_tps" not in pr:
        return {"ok": False, "reason": "PR bench missing decode tok/s", "log": (r.stdout or "")[-1500:]}
    if "top1" not in pr or "kl" not in pr:
        return {"ok": False, "reason": "PR run missing accuracy METRIC line", "log": (r.stdout or "")[-1500:]}
    print(f">> PR decode={pr['decode_tps']:.2f} top1={pr.get('top1', 0):.3f} kl={pr.get('kl', 99):.4f} "
          f"— measuring main …")

    r2 = _ssh_run_resilient(host, port, _remote_script("main"), "main run")
    if r2.returncode != 0:
        tail = ((r2.stdout or "") + "\n" + (r2.stderr or ""))[-2000:]
        crash = _crash_reason(r2.stdout, r2.stderr)
        reason = "main run failed" + (f" — {crash}" if crash else " (no crash diagnostic captured, possible hard kill — retried once)")
        return {"ok": False, "reason": reason, "log": tail, "pr": pr}
    main = _parse_remote(r2.stdout or "")
    if "decode_tps" not in main:
        return {"ok": False, "reason": "main bench missing decode tok/s", "log": (r2.stdout or "")[-1500:], "pr": pr}

    label, delta_pct, passed, speed_reason = tier_from_gain(pr["decode_tps"], main["decode_tps"])

    pr_top1 = pr.get("top1", 0.0)
    pr_kl = pr.get("kl", 99.0)
    accuracy_ok = pr_top1 >= ACC_TOP1_BAR and pr_kl <= ACC_KL_BAR
    reason = speed_reason
    if not accuracy_ok:
        # Accuracy gate is a hard REJECT regardless of speed — same discipline as
        # pr_dflash_bot.py's SPEC_AGREE veto: a fast-but-wrong PR is worthless on a still-fragile
        # architecture, and speed alone can't prove correctness.
        acc_reason = (f"accuracy gate failed: top1={pr_top1:.3f} (bar >={ACC_TOP1_BAR}) "
                      f"kl={pr_kl:.4f} (bar <={ACC_KL_BAR})")
        reason = f"{acc_reason} | speed: {speed_reason}"
        label = "REJECT"
        passed = False

    res = {
        "ok": True,
        "label": label,
        "pass": passed and label != "REJECT",
        "reason": reason,
        "delta_pct": delta_pct,
        "pr_decode_tps": pr["decode_tps"],
        "main_decode_tps": main["decode_tps"],
        "speedup_vs_main": round(pr["decode_tps"] / main["decode_tps"], 3) if main.get("decode_tps") else 0,
        "pr_top1": pr_top1,
        "pr_kl": pr_kl,
        "pr_ppl_spark": pr.get("ppl_spark"),
        "pr_ppl_llama": pr.get("ppl_llama"),
        "main_top1": main.get("top1"),
        "main_kl": main.get("kl"),
        "accuracy_ok": accuracy_ok,
        "pr_head": pr.get("head"),
        "main_head": main.get("head"),
    }
    polaris = collect_polaris_attestation(host, port, res, pr_ref)
    if polaris:
        res["polaris"] = polaris
    return res


def format_comment(commit: str, res: dict) -> str:
    meta = {
        "label": res.get("label"),
        "delta_pct": res.get("delta_pct"),
        "pr_decode_tps": res.get("pr_decode_tps"),
        "main_decode_tps": res.get("main_decode_tps"),
        "pass": res.get("pass"),
        "accuracy_ok": res.get("accuracy_ok"),
    }
    marker = (
        f"<!-- sparkinfer-museglimmer-eval:{EVAL_SCHEMA_VERSION}:{commit} "
        f"{json.dumps(meta, separators=(',', ':'))} -->"
    )
    if not res.get("ok"):
        return (
            f"{marker}\n## sparkinfer museglimmer auto-eval — error\n\n"
            f"**reason:** `{res.get('reason')}`\n\n"
            f"<details><summary>log tail</summary>\n\n```\n{(res.get('log') or '')[:1800]}\n```\n</details>\n"
        )
    lab = res["label"]
    if res.get("accuracy_ok"):
        acc_row = (f"| accuracy gate | ✅ top1={res.get('pr_top1', 0):.3f} "
                    f"(bar >={ACC_TOP1_BAR}) · KL={res.get('pr_kl', 0):.4f} (bar <={ACC_KL_BAR}) |\n")
    else:
        acc_row = (f"| accuracy gate | ❌ **FAILED** — top1={res.get('pr_top1', 0):.3f} "
                    f"(bar >={ACC_TOP1_BAR}) · KL={res.get('pr_kl', 0):.4f} (bar <={ACC_KL_BAR}) — "
                    "**verdict forced to REJECT regardless of speed** |\n")
    main_acc_note = ""
    if res.get("main_top1") is not None and (res.get("main_top1", 1) < ACC_TOP1_BAR or (res.get("main_kl") or 0) > ACC_KL_BAR):
        main_acc_note = (f"| ⚠️ same-box main accuracy | top1={res.get('main_top1'):.3f} "
                          f"kl={res.get('main_kl'):.4f} — main ALSO misses the bar (informational; "
                          "not gated on main, but check the box/corpus if this persists) |\n")
    polaris = res.get("polaris") or {}
    receipt = polaris.get("receipt")
    if receipt:
        rtype = "TDX (Intel hardware attestation)" if receipt.get("attestation_type") == "tdx-quote" \
            else "Ed25519 (SparkInfer key)"
        polaris_row = f"| Polaris receipt | `{receipt.get('receipt_id', '?')[:16]}…` — {rtype} |\n"
    elif polaris.get("attestation"):
        polaris_row = "| Polaris receipt | collected, not signed (no key configured) |\n"
    else:
        polaris_row = ""
    return (
        f"{marker}\n## sparkinfer museglimmer auto-eval — `eval-museglimmer:{lab}`\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| **label** | `eval-museglimmer:{lab}` |\n"
        f"| scored at | 128-token decode (ctx=0), no prefill context |\n"
        f"| PR decode tok/s | {res['pr_decode_tps']:.2f} |\n"
        f"| main decode tok/s | {res['main_decode_tps']:.2f} |\n"
        f"| speedup vs main | **{res.get('speedup_vs_main', 0):.2f}×** ({res.get('delta_pct', 0):+.1f}%) |\n"
        f"{acc_row}"
        f"{main_acc_note}"
        f"| PPL sparkinfer / llama.cpp | {res.get('pr_ppl_spark') or '?'} / {res.get('pr_ppl_llama') or '?'} |\n"
        f"{polaris_row}"
        f"| commit | `{commit[:9]}` |\n\n"
        f"{res.get('reason') or ''}\n\n"
        "<sub>Scored on the pinned eval box vs same-box `origin/main`, 128-token AR decode only "
        "(no DFlash, no long-context, no prefill) — Muse Glimmer's narrow, deliberately strict "
        "eval scope. This is informational, not a judgment on your PR: a `none` label just means "
        "no measurable Muse Glimmer 128-decode speedup was found, which is expected and fine if "
        "that isn't what your change is about — this bot never closes PRs. "
        "Correctness gated against a live llama.cpp reference on the same GGUF. "
        "Automated — **not merged**; merge manually after review.</sub>\n"
    )


def auto_merge_ok_museglimmer(repo, num):
    info = json.loads(arb.gh([
        "pr", "view", str(num), "-R", repo, "--json",
        "state,isDraft,labels,author,mergeable,files",
    ]).stdout or "{}")
    if info.get("state") != "OPEN" or info.get("isDraft"):
        return False, "not an open, non-draft PR"
    labs = {l["name"] for l in info.get("labels", [])}
    tiers = {l.split(":", 1)[1] for l in labs if l.startswith(EVAL_PREFIX)}
    if not (tiers & SPEEDUP_LABELS):
        return False, "no verified eval-museglimmer:speedup label"
    if MUSEGLIMMER_MERGE_FIRST not in labs:
        return False, "not museglimmer-merge-first"
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


def try_auto_merge_museglimmer(repo, num):
    ok, reason = auto_merge_ok_museglimmer(repo, num)
    if not ok:
        print(f">> museglimmer auto-merge SKIP #{num}: {reason}")
        return False
    r = arb.gh(["pr", "merge", str(num), "-R", repo, "--squash"])
    if r.returncode != 0 and os.environ.get("SPARKINFER_AUTOMERGE_ADMIN", "1") == "1":
        err = ((r.stderr or "") + (r.stdout or "")).lower()
        if "not mergeable" in err or "branch policy" in err or "required" in err or "prohibited" in err:
            print(">> museglimmer auto-merge: branch policy blocked — retrying with --admin")
            r = arb.gh(["pr", "merge", str(num), "-R", repo, "--squash", "--admin"])
    if r.returncode == 0:
        print(f">> MUSEGLIMMER AUTO-MERGED #{num} (museglimmer-merge-first)")
        arb.gh(["pr", "comment", str(num), "-R", repo, "--body",
                "<!-- sparkinfer-museglimmer-automerge -->\n"
                "Auto-merged as the round's `museglimmer-merge-first` winner — verified same-box "
                "128-token decode speedup over `main`, accuracy-gated vs llama.cpp."])
        return True
    print(f">> museglimmer auto-merge BLOCKED #{num}: {(r.stderr or r.stdout or '')[:200]}")
    return False


def reconcile_museglimmer_merge_labels(repo, dry_run=False):
    scores = _load_scores()
    open_prs = json.loads(arb.gh([
        "pr", "list", "-R", repo, "--state", "open",
        "--json", "number,labels", "--limit", "80",
    ]).stdout or "[]")
    open_labels = {p["number"]: {l["name"] for l in p["labels"]} for p in open_prs}

    merged = json.loads(arb.gh([
        "pr", "list", "-R", repo, "--state", "merged", "--label", MUSEGLIMMER_MERGE_FIRST,
        "--json", "number", "--limit", "10",
    ]).stdout or "[]")
    for m in merged:
        if not dry_run:
            arb.remove_label(repo, m["number"], MUSEGLIMMER_MERGE_FIRST)

    scored = []
    for num, labs in open_labels.items():
        if MUSEGLIMMER_NEEDS_REBASE in labs:
            continue
        tiers = {l.split(":", 1)[1] for l in labs if l.startswith(EVAL_PREFIX)}
        tier = next((t for t in tiers if t in SPEEDUP_LABELS), None)
        if not tier:
            continue
        entry = scores.get(str(num)) or {}
        if entry.get("label") not in SPEEDUP_LABELS:
            if tier not in SPEEDUP_LABELS:
                continue
            entry = {"label": tier, "delta_pct": entry.get("delta_pct") or 0}
        scored.append((num, float(entry.get("delta_pct") or 0), entry.get("label") or tier))

    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        print(">> museglimmer round: no verified speedup PRs")
        return
    winner = scored[0][0]
    print(f">> museglimmer round: merge-first #{winner}; rebase {[n for n,_,_ in scored[1:]] or 'none'}")
    if dry_run:
        return
    arb.add_label(repo, winner, MUSEGLIMMER_MERGE_FIRST)
    arb.remove_label(repo, winner, MUSEGLIMMER_NEEDS_REBASE)
    for num, _, _ in scored[1:]:
        arb.add_label(repo, num, MUSEGLIMMER_NEEDS_REBASE)
        arb.remove_label(repo, num, MUSEGLIMMER_MERGE_FIRST)
    if AUTO_MERGE:
        try_auto_merge_museglimmer(repo, winner)


def upload_museglimmer_eval_log(repo, num, title, oid, res):
    """Commit the eval result (+ Polaris receipt/attestation) to sparkinfer-log, mirroring
    pr_dflash_bot.py's upload_dflash_eval_log with a museglimmer-prefixed run id."""
    try:
        rid = f"museglimmer-{int(num):04d}-{oid[:7]}"
        arb._ensure_log_repo()
        rundir = os.path.join(arb.LOG_DIR, "runs", rid)
        os.makedirs(rundir, exist_ok=True)
        polaris = res.get("polaris") or {}
        receipt = polaris.get("receipt")
        result = {
            "id": rid, "pr": int(num), "title": title,
            "url": f"https://github.com/{repo}/pull/{num}", "commit": oid[:7],
            "eval_mode": "museglimmer-128",
            "label": res.get("label"), "pass": res.get("pass"), "reason": res.get("reason"),
            "delta_pct": res.get("delta_pct"),
            "pr_decode_tps": res.get("pr_decode_tps"), "main_decode_tps": res.get("main_decode_tps"),
            "speedup_vs_main": res.get("speedup_vs_main"),
            "pr_top1": res.get("pr_top1"), "pr_kl": res.get("pr_kl"),
            "accuracy_ok": res.get("accuracy_ok"),
            "gpu": "pinned eval box", "date": arb.datetime.date.today().isoformat(),
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
                     "delta_pct": res.get("delta_pct"), "eval_mode": "museglimmer-128", "date": result["date"]}
        if receipt:
            idx_entry["polaris"] = True
            idx_entry["polaris_receipt_id"] = receipt.get("receipt_id", "")[:16]
        idx.append(idx_entry)
        idx.sort(key=lambda x: x["id"])
        json.dump(idx, open(ipath, "w"), indent=2)
        subprocess.run(["git", "-C", arb.LOG_DIR, "add", "-A"], check=True)
        msg = f"museglimmer-eval: #{num} {oid[:7]} -> eval-museglimmer:{res.get('label')}"
        if receipt:
            msg += f" + polaris {receipt.get('receipt_id', '?')[:16]}"
        commit = subprocess.run(["git", "-C", arb.LOG_DIR, "commit", "-q", "-m", msg], check=False)
        if commit.returncode != 0:
            print(">> museglimmer eval-log upload skipped: nothing to commit")
            return None
        push = subprocess.run(["git", "-C", arb.LOG_DIR, "push", "-q"], check=False)
        if push.returncode != 0:
            print(f">> museglimmer eval-log push failed (rc={push.returncode})")
            return None
        url = arb.LOG_PAGE + rid
        print(f">> museglimmer eval log: {url}")
        return url
    except Exception as e:
        print(f">> museglimmer eval-log upload failed: {e}")
        return None


def apply_result(repo, num, commit, res, title="", dry_run=False):
    body = format_comment(commit, res)
    label = res.get("label") if res.get("ok") else "REJECT"
    if not res.get("ok"):
        label = "REJECT"
    print(f"PR #{num}: eval-museglimmer:{label}  "
          f"PR={res.get('pr_decode_tps')} main={res.get('main_decode_tps')} "
          f"delta={res.get('delta_pct')}%  accuracy_ok={res.get('accuracy_ok')}")
    if dry_run:
        print(body[:500])
        return
    strip_museglimmer_eval_labels(repo, num)
    arb.add_label(repo, num, f"{EVAL_PREFIX}{label}")
    # Mirrored to the generic `eval:*` label, same as pr_dflash_bot.py -- SN74 scoring reads
    # eval:* tiers, so this makes Muse Glimmer submissions count toward that live incentive
    # mechanism. Explicit user decision, 2026-08-11 (originally deliberately NOT mirrored, given
    # Muse Glimmer's youth at the time -- see git history on this line for that reasoning).
    for lab in {l for l in arb.labels_on(repo, num) if l.startswith("eval:")}:
        arb.remove_label(repo, num, lab)
    arb.add_label(repo, num, f"eval:{label}")
    arb.gh(["pr", "comment", str(num), "-R", repo, "--body", body])
    if res.get("ok"):
        upload_museglimmer_eval_log(repo, num, title, commit, res)
    if res.get("ok") and res.get("delta_pct") is not None:
        scores = _load_scores()
        scores[str(num)] = {
            "commit": commit,
            "label": label,
            "delta_pct": res.get("delta_pct"),
            "pr_decode_tps": res.get("pr_decode_tps"),
            "main_decode_tps": res.get("main_decode_tps"),
            "pass": res.get("pass"),
            "accuracy_ok": res.get("accuracy_ok"),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _save_scores(scores)
        # Auto-close on "none"/"REJECT" -- same policy as pr_dflash_bot.py. Re-enabled 2026-08-11
        # after an explicit, informed decision: the very first supervised run of this bot closed a
        # real external contributor's unrelated PR (#768) this exact same way, since
        # arb.greenlight_status() is generic (any PR with a checked "tested" box + a decode/
        # prefill before/after table) and matches essentially any performance PR in the repo, not
        # just Muse-Glimmer-relevant ones -- "none" is the expected, non-judgmental outcome for
        # most PRs this bot evaluates, not a rejection of the PR's actual purpose. That PR was
        # reopened + apologized for. The user was told this risk explicitly and chose to accept it
        # (broad scope, matching pr_dflash_bot.py) rather than narrow evaluation to only
        # Muse-Glimmer-relevant PRs. If this causes another wrongful close, reopen + apologize the
        # same way, and reconsider the scope-narrowing alternative that was declined here.
        if label in ("none", "REJECT"):
            close_body = (
                "<!-- sparkinfer-museglimmer-auto-close -->\n"
                f"## Closed: sparkinfer museglimmer auto-eval — `eval-museglimmer:{label}`\n\n"
                f"This PR's Muse Glimmer 128-decode speed measured **{res.get('delta_pct')}%** vs "
                f"main, {'and failed the accuracy gate' if not res.get('accuracy_ok') else 'with no verified improvement' if label == 'none' else '(regression)'} "
                "— closing automatically. This bot evaluates every eligible PR in the repo "
                "against Muse Glimmer's decode speed specifically, regardless of what the PR is "
                "actually about — a close here isn't a judgment on the PR's purpose, just that it "
                "didn't move this particular metric. Reopen (or open a fresh PR) if you have a fix "
                "or a different approach."
            )
            arb.gh(["pr", "comment", str(num), "-R", repo, "--body", close_body])
            arb.gh(["pr", "close", str(num), "-R", repo])
            print(f">> auto-closed PR #{num} (eval-museglimmer:{label})")


def main():
    ap = argparse.ArgumentParser(description="Muse Glimmer 128-decode PR eval bot")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--repo", default="gittensor-ai-lab/sparkinfer")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reeval", action="store_true")
    ap.add_argument("--labels-only", action="store_true",
                    help="reconcile museglimmer-merge-first only — no GPU")
    ap.add_argument("--only-prs", default="",
                    help="comma-separated PR numbers (bypass greenlight)")
    args = ap.parse_args()

    only = {int(x) for x in args.only_prs.split(",") if x.strip().isdigit()}

    print(f">> museglimmer eval transport: "
          f"{'ssh' if ssh_box_enabled() else f'vast.ai (instance {arb.current_instance(args.instance) or args.instance})'}")
    print(f">> AUTOMERGE={int(AUTO_MERGE)}")

    if args.labels_only:
        reconcile_museglimmer_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — museglimmer labels only (no GPU).")
        return

    prs = json.loads(arb.gh([
        "pr", "list", "-R", args.repo, "--state", "open",
        "--json", "number,title,labels,isDraft,headRefOid,headRefName,mergeable,author,body",
        "--limit", "80",
    ]).stdout or "[]")
    prs.sort(key=lambda p: p["number"])

    stale_closed = close_stale_museglimmer_prs(args.repo, prs, dry_run=args.dry_run) if not only else set()

    pending = []
    for pr in prs:
        num = pr["number"]
        if num in stale_closed:
            continue
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
        if not args.reeval and head and head in museglimmer_evaluated_commits(args.repo, num):
            print(f"PR #{num} @ {short}: already museglimmer-evaluated — skip")
            continue
        if arb.pr_merge_conflict(pr.get("mergeable")):
            print(f"PR #{num}: merge conflict — museglimmer-needs-rebase")
            if not args.dry_run:
                arb.add_label(args.repo, num, MUSEGLIMMER_NEEDS_REBASE)
            continue

        if not only:
            status, why = arb.greenlight_status(args.repo, num, labs)
            if status != "ok":
                print(f"PR #{num}: not greenlit ({why}) — skip museglimmer eval")
                continue
            print(f"PR #{num}: greenlit ({why})")
        else:
            print(f"PR #{num}: --only-prs targeted")

        ref = f"pull/{num}/head"
        pending.append((num, head, short, ref, pr.get("title", "")))

    if not pending:
        reconcile_museglimmer_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — no museglimmer PRs to evaluate.")
        return

    if args.dry_run:
        print("--- dry-run would evaluate: " + ", ".join(f"#{n}" for n, *_ in pending))
        return

    pin = arb.PINNED_INSTANCE
    if pin and not ssh_box_enabled():
        with open(arb.INSTANCE_FILE, "w") as f:
            f.write(str(pin))

    try:
        host, port = resolve_ssh(args.instance)
    except Exception as e:
        print(f">> GPU unavailable: {e}")
        reconcile_museglimmer_merge_labels(args.repo, dry_run=False)
        print("done — museglimmer labels only (GPU down).")
        return

    _ssh_user = ssh_box_user() if ssh_box_enabled() else "root"
    print(f">> SSH {_ssh_user}@{host}:{port}")

    for num, head, short, ref, title in pending:
        print(f"PR #{num} @ {short}: evaluating Muse Glimmer '{ref}' …")
        try:
            res = eval_museglimmer_on_box(host, port, ref)
        except Exception as e:
            res = {"ok": False, "reason": f"exception: {e}"}
        apply_result(args.repo, num, head or short, res, title=title, dry_run=False)

    reconcile_museglimmer_merge_labels(args.repo, dry_run=False)
    print("done — museglimmer eval pass complete.")


if __name__ == "__main__":
    main()
