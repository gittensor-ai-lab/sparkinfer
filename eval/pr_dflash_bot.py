#!/usr/bin/env python3
"""sparkinfer DFlash PR auto-evaluator.

Sibling of pr_eval_bot.py for PRs that touch the DFlash speculative-decode path.
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
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ssh_box import ssh_box_enabled, ssh_box_endpoint  # noqa: E402

# Reuse shared helpers from the AR bot (labels, greenlight, denylist, …).
import pr_eval_bot as arb  # noqa: E402

SPEEDUP_LABELS = {"XL", "L", "M", "S", "XS"}
SIG = 0.02
REGRESS_TOL = 0.98
BUCKETS = [(0.18, "XL"), (0.10, "L"), (0.06, "M"), (0.035, "S"), (SIG, "XS")]

EVAL_PREFIX = "eval-dflash:"
DFLASH_MERGE_FIRST = "dflash-merge-first"
DFLASH_NEEDS_REBASE = "dflash-needs-rebase"
MARKER_RE = re.compile(
    r"<!-- sparkinfer-dflash-eval:([0-9a-f]+)(?:\s+(\{.*?\}))? -->",
    re.DOTALL,
)
DFLASH_PATH_RE = re.compile(r"(?:^|/)(?:dflash|qwen3_gguf_dflash_|dflash_accuracy\.sh)", re.I)

DEFAULT_GGUF = os.environ.get(
    "DFLASH_GGUF", "/workspace/models36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
)
DEFAULT_DRAFT = os.environ.get(
    "DFLASH_DRAFT", "/workspace/models_dflash/Qwen3.6-35B-A3B-DFlash"
)
DEFAULT_MODELS_DIR = os.environ.get("DFLASH_MODELS_DIR", "/workspace/models36")
REMOTE_REPO = os.environ.get("DFLASH_REMOTE_REPO", "/root/sparkinfer")
# 512 gen crosses the adaptive KV-split knee (prompt + ntok > 2*split_chunk) so DFlash
# verify graphs are captured at the terminal n_splits tier, not stuck at 32.
BENCH_TOKENS = int(os.environ.get("DFLASH_BENCH_TOKENS", "512"))

AUTO_MERGE = os.environ.get("SPARKINFER_AUTOMERGE", "0") == "1"
AUTOMERGE_BLOCK = {
    "copycat", "copycat-warn", "flagged:gaming", "penalty", "needs-benchmark",
    DFLASH_NEEDS_REBASE, arb.REEVALUATE_LABEL, arb.HOLD_LABEL, *arb.REGRESSION_LABELS,
}

SCORES_FILE = os.path.expanduser(
    os.environ.get("DFLASH_SCORES_FILE", "~/.sparkinfer_dflash_scores.json")
)


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


def is_dflash_pr(repo, num) -> bool:
    files = json.loads(
        arb.gh(["pr", "view", str(num), "-R", repo, "--json", "files"]).stdout or "{}"
    ).get("files", [])
    for f in files:
        path = f.get("path") or ""
        if DFLASH_PATH_RE.search(path):
            return True
    return False


def dflash_evaluated_commits(repo, num):
    r = arb.gh(["pr", "view", str(num), "-R", repo, "--json", "comments"])
    done = set()
    for c in json.loads(r.stdout or "{}").get("comments", []):
        m = MARKER_RE.search(c.get("body") or "")
        if m and "sparkinfer dflash auto-eval" in (c.get("body") or ""):
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


def ssh_run(host, port, cmd, timeout=7200):
    key = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/speedy"))
    return subprocess.run(
        [
            "ssh", "-i", key,
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=40",
            "-p", str(port), f"root@{host}", cmd,
        ],
        capture_output=True, text=True, timeout=timeout,
    )


def _remote_script(ref: str, do_accuracy: bool, prompt_ids: str | None, n_tokens: int) -> str:
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
    acc = "1" if do_accuracy else "0"
    return f"""
set -euo pipefail
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

# Build dflash tools (incremental)
mkdir -p build
if [ ! -f build/CMakeCache.txt ]; then
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/dflash_cmake.log 2>&1
fi
cmake --build build --target qwen3_gguf_dflash_check qwen3_gguf_dflash_bench -j"$(nproc)" >/tmp/dflash_build.log 2>&1
test -x build/runtime/qwen3_gguf_dflash_bench

if [ "$DO_ACC" = "1" ]; then
  export MODELS_DIR
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

OUT=$(build/runtime/qwen3_gguf_dflash_bench "$GGUF" "$DRAFT" "$NTOK" $PROMPT_IDS | tee /tmp/dflash_bench_out.txt)
echo "$OUT"
AR=$(echo "$OUT" | grep '^METRIC AR_TPS' | awk '{{print $3}}' | tail -1)
DF=$(echo "$OUT" | grep '^METRIC DFLASH_TPS' | awk '{{print $3}}' | tail -1)
TAU=$(echo "$OUT" | grep '^METRIC MEAN_ACCEPT' | awk '{{print $3}}' | tail -1)
echo "RESULT_AR_TPS $AR"
echo "RESULT_DFLASH_TPS $DF"
echo "RESULT_MEAN_ACCEPT $TAU"
"""


def _parse_remote(stdout: str) -> dict:
    out = {}
    for line in (stdout or "").splitlines():
        if line.startswith("RESULT_AR_TPS "):
            out["ar_tps"] = float(line.split()[1])
        elif line.startswith("RESULT_DFLASH_TPS "):
            out["dflash_tps"] = float(line.split()[1])
        elif line.startswith("RESULT_MEAN_ACCEPT "):
            out["mean_accept"] = float(line.split()[1])
        elif line.startswith("PROMPT_IDS "):
            out["prompt_ids"] = line[len("PROMPT_IDS "):].strip()
        elif line.startswith("REMOTE_HEAD "):
            out["head"] = line.split()[1]
        elif line.startswith("METRIC SPEC_AGREE"):
            out["spec_agree"] = line.strip()
    return out


def eval_dflash_on_box(host, port, pr_ref: str):
    """Run PR accuracy+bench then main bench with same prompt ids. Returns result dict."""
    print(f">> DFlash eval on box: PR ref={pr_ref}")
    r = ssh_run(host, port, _remote_script(pr_ref, do_accuracy=True, prompt_ids=None,
                                           n_tokens=BENCH_TOKENS))
    if r.returncode != 0:
        tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-2000:]
        return {"ok": False, "reason": "PR accuracy/bench failed", "log": tail}
    pr = _parse_remote(r.stdout or "")
    if "dflash_tps" not in pr:
        return {"ok": False, "reason": "PR bench missing DFLASH_TPS", "log": (r.stdout or "")[-1500:]}
    ids = pr.get("prompt_ids") or ""
    print(f">> PR DFlash={pr['dflash_tps']:.2f} AR={pr.get('ar_tps', 0):.2f} — measuring main …")
    r2 = ssh_run(host, port, _remote_script("main", do_accuracy=False, prompt_ids=ids,
                                            n_tokens=BENCH_TOKENS))
    if r2.returncode != 0:
        tail = ((r2.stdout or "") + "\n" + (r2.stderr or ""))[-2000:]
        return {"ok": False, "reason": "main bench failed", "log": tail, "pr": pr}
    main = _parse_remote(r2.stdout or "")
    if "dflash_tps" not in main:
        return {"ok": False, "reason": "main bench missing DFLASH_TPS", "log": (r2.stdout or "")[-1500:],
                "pr": pr}
    label, delta_pct, passed, reason = tier_from_gain(pr["dflash_tps"], main["dflash_tps"])
    return {
        "ok": True,
        "label": label,
        "pass": passed and label != "REJECT",
        "reason": reason,
        "delta_pct": delta_pct,
        "pr_dflash_tps": pr["dflash_tps"],
        "pr_ar_tps": pr.get("ar_tps"),
        "main_dflash_tps": main["dflash_tps"],
        "main_ar_tps": main.get("ar_tps"),
        "mean_accept": pr.get("mean_accept"),
        "spec_agree": pr.get("spec_agree"),
        "prompt_ids": ids,
        "speedup_vs_main": round(pr["dflash_tps"] / main["dflash_tps"], 3) if main["dflash_tps"] else 0,
        "speedup_vs_ar": round(pr["dflash_tps"] / pr["ar_tps"], 3) if pr.get("ar_tps") else 0,
        "bench_tokens": BENCH_TOKENS,
    }


def format_comment(commit: str, res: dict) -> str:
    meta = {
        "label": res.get("label"),
        "delta_pct": res.get("delta_pct"),
        "pr_dflash_tps": res.get("pr_dflash_tps"),
        "main_dflash_tps": res.get("main_dflash_tps"),
        "pass": res.get("pass"),
        "bench_tokens": res.get("bench_tokens", BENCH_TOKENS),
    }
    marker = f"<!-- sparkinfer-dflash-eval:{commit} {json.dumps(meta, separators=(',', ':'))} -->"
    if not res.get("ok"):
        return (
            f"{marker}\n## sparkinfer dflash auto-eval — error\n\n"
            f"**reason:** `{res.get('reason')}`\n\n"
            f"<details><summary>log tail</summary>\n\n```\n{(res.get('log') or '')[:1800]}\n```\n</details>\n"
        )
    lab = res["label"]
    return (
        f"{marker}\n## sparkinfer dflash auto-eval — `eval-dflash:{lab}`\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| **label** | `eval-dflash:{lab}` |\n"
        f"| bench gen tokens | {res.get('bench_tokens', BENCH_TOKENS)} |\n"
        f"| PR DFlash tok/s | {res['pr_dflash_tps']:.2f} |\n"
        f"| main DFlash tok/s | {res['main_dflash_tps']:.2f} |\n"
        f"| speedup vs main | **{res['speedup_vs_main']:.2f}×** ({res['delta_pct']:+.1f}%) |\n"
        f"| PR AR tok/s | {res.get('pr_ar_tps') or 0:.2f} |\n"
        f"| DFlash vs AR | {res.get('speedup_vs_ar') or 0:.2f}× |\n"
        f"| mean accept τ | {res.get('mean_accept') or 0:.3f} |\n"
        f"| accuracy | {res.get('spec_agree') or 'VERDICT PASS'} |\n"
        f"| commit | `{commit[:9]}` |\n\n"
        f"{res.get('reason') or ''}\n\n"
        "<sub>Scored on pinned RTX 5090 vs same-box `origin/main` DFlash. "
        "AR `eval:*` labels are unchanged.</sub>\n"
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


def apply_result(repo, num, commit, res, dry_run=False):
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
    arb.gh(["pr", "comment", str(num), "-R", repo, "--body", body])
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


def main():
    ap = argparse.ArgumentParser(description="DFlash PR eval bot")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--repo", default="gittensor-ai-lab/sparkinfer")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reeval", action="store_true")
    ap.add_argument("--labels-only", action="store_true",
                    help="reconcile dflash-merge-first only — no GPU")
    ap.add_argument("--only-prs", default="",
                    help="comma-separated PR numbers (bypass greenlight; still require dflash paths unless forced)")
    ap.add_argument("--force-prs", action="store_true",
                    help="with --only-prs, evaluate even if paths do not match dflash filter")
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
        if not only or not args.force_prs:
            if not is_dflash_pr(args.repo, num):
                if only:
                    print(f"PR #{num}: not a dflash-path PR — skip (use --force-prs)")
                continue
        else:
            if not is_dflash_pr(args.repo, num):
                print(f"PR #{num}: force — evaluating despite non-dflash paths")

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
        pending.append((num, head, short, ref))

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

    print(f">> SSH root@{host}:{port}")

    for num, head, short, ref in pending:
        print(f"PR #{num} @ {short}: evaluating DFlash '{ref}' …")
        try:
            res = eval_dflash_on_box(host, port, ref)
        except Exception as e:
            res = {"ok": False, "reason": f"exception: {e}"}
        apply_result(args.repo, num, head or short, res, dry_run=False)

    reconcile_dflash_merge_labels(args.repo, dry_run=False)
    print("done — dflash eval pass complete.")


if __name__ == "__main__":
    main()
