#!/usr/bin/env python3
"""sparkinfer PR auto-evaluator (bot).

Polls open PRs; for any PR whose head commit hasn't been evaluated yet, runs the vast.ai
evaluation (build → correctness → speed → label), applies an `eval:<LABEL>` label, and posts the
result as a PR comment. **Never merges** — merging is manual after review.

Designed to run on a 30-min schedule (system cron or a Claude agent). Idempotent: a commit is
evaluated once (tracked by a hidden marker in the bot's comment), so re-runs only pick up new
commits and only spin the GPU when there's new work.

  python eval/pr_eval_bot.py --instance 42134865 --frontier 164 --ceiling 366

Needs: `gh` authenticated, VAST_API_KEY saved (vastai), and the eval:* labels (eval/setup_labels.sh).
"""
import argparse, datetime, hashlib, json, os, re, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ssh_box import ssh_box_enabled, ssh_box_endpoint, ssh_box_arg, vast_enabled

# Reuse vast_eval's SSH plumbing for the Qwen3.6 baseline bench (same box, same keys).
# The bot shells out to vast_eval for the full accuracy-gated Qwen3-30B baseline, but
# the Qwen3.6 primary only needs a speed sweep — a direct SSH bench is faster.
try:
    import importlib.util
    _ve_spec = importlib.util.spec_from_file_location("vast_eval", os.path.join(HERE, "vast_eval.py"))
    _ve = importlib.util.module_from_spec(_ve_spec); _ve_spec.loader.exec_module(_ve)
    _vast_sh  = _ve.sh
    _vast_endpoint = _ve.endpoint
    _vast_info_of  = _ve.info_of
except Exception:
    _vast_sh = _vast_endpoint = _vast_info_of = None

# vast_eval.py self-heals dead boxes (recreates them) and writes the working instance id here;
# prefer it over --instance so we reuse the recreated box instead of retrying the dead one.
INSTANCE_FILE = os.path.expanduser(os.environ.get("VAST_INSTANCE_FILE", "~/.sparkinfer_vast_instance"))
def current_instance(default):
    try: return int(open(INSTANCE_FILE).read().strip())
    except Exception: return default

# Pinned default box: a stable, known-good instance (cached model, good download speed) that we
# reuse on every run and NEVER destroy. On bring-up failure vast_eval exits PINNED_RETRY_RC for
# the next scheduled run. Set VAST_DEFAULT_INSTANCE="" to disable the pin.
PIN_FILE = os.path.expanduser(os.environ.get("VAST_PIN_FILE", "~/.sparkinfer_pinned_instance"))
def _read_pin():
    try:
        v = open(PIN_FILE).read().strip()
        if v: return v
    except Exception: pass
    return os.environ.get("VAST_DEFAULT_INSTANCE", "44206573").strip()
def _write_pin(iid):
    try:
        with open(PIN_FILE, "w") as f: f.write(str(iid))
    except Exception: pass
PINNED_INSTANCE = _read_pin() if not ssh_box_enabled() else ""
PINNED_RETRY_RC = 75   # must match vast_eval.PINNED_RETRY_RC


def _vast_eval_transport_args(instance_id):
    """Return CLI args for vast_eval.py: either --ssh or --reuse."""
    if ssh_box_enabled():
        return ["--ssh", ssh_box_arg()]
    return ["--reuse", str(current_instance(instance_id))]


def _bidir_baseline_args(q36, q35):
    """Guard baseline CLI args for vast_eval.py bidir runs."""
    return [
        "--p35-guard-128-baseline", str(q35["128"]),
        "--p35-guard-4k-baseline",  str(q35["4k"]),
        "--p35-guard-32k-baseline", str(q35["32k"]),
        "--p35-guard-64k-baseline", str(q35["64k"]),
        "--p35-guard-128k-baseline", str(q35["128k"]),
        "--p35-guard-4k-pp-baseline", str(q35.get("4k_pp", 0)),
        "--p35-guard-32k-pp-baseline", str(q35.get("32k_pp", 0)),
        "--p35-guard-64k-pp-baseline", str(q35.get("64k_pp", 0)),
        "--p35-guard-128k-pp-baseline", str(q35.get("128k_pp", 0)),
        "--p-guard-128-baseline", str(q36["128"]),
        "--p-guard-512-baseline", str(q36["512"]),
        "--p-guard-4k-baseline",  str(q36["4k"]),
        "--p-guard-16k-baseline", str(q36["16k"]),
        "--p-guard-32k-baseline", str(q36["32k"]),
        "--p36-guard-128-pp-baseline", str(q36.get("128_pp", 0)),
        "--p36-guard-512-pp-baseline", str(q36.get("512_pp", 0)),
        "--p36-guard-4k-pp-baseline", str(q36.get("4k_pp", 0)),
        "--p36-guard-16k-pp-baseline", str(q36.get("16k_pp", 0)),
        "--p36-guard-32k-pp-baseline", str(q36.get("32k_pp", 0)),
        "--g36-guard-128-baseline", str(q36["128"]),
        "--g36-guard-512-baseline", str(q36["512"]),
        "--g36-guard-4k-baseline",  str(q36["4k"]),
        "--g36-guard-16k-baseline", str(q36["16k"]),
        "--g36-guard-32k-baseline", str(q36["32k"]),
        "--g35-guard-128-baseline", str(q35["128"]),
        "--g35-guard-4k-baseline",  str(q35["4k"]),
        "--g35-guard-32k-baseline", str(q35["32k"]),
        "--g35-guard-64k-baseline", str(q35["64k"]),
        "--g35-guard-128k-baseline", str(q35["128k"]),
        "--p-llama-128-baseline", str(q36["llama128"]),
        "--p-llama-512-baseline", str(q36["llama512"]),
        "--p-llama-4k-baseline",  str(q36["llama4k"]),
    ]


def _apply_bidir_ctx_from_bres(bres, q36, q35):
    """Fill QWEN36_BASE / QWYTHOS_BASE from bidir RESULT_JSON; return True if both models measured."""
    ctx_map = {
        "128": "ctx_128_tps", "512": "ctx_512_tps", "4k": "ctx_4096_tps",
        "16k": "ctx_16384_tps", "32k": "ctx_32768_tps",
        "64k": "ctx_65536_tps", "128k": "ctx_131072_tps",
    }
    pp_map_q35 = {
        "4k_pp": "ctx_4096_pp_tps", "32k_pp": "ctx_32768_pp_tps",
        "64k_pp": "ctx_65536_pp_tps", "128k_pp": "ctx_131072_pp_tps",
    }
    pp_map_q36 = {
        "128_pp": "ctx_128_pp_tps", "512_pp": "ctx_512_pp_tps",
        "4k_pp": "ctx_4096_pp_tps", "16k_pp": "ctx_16384_pp_tps",
        "32k_pp": "ctx_32768_pp_tps",
    }

    def _fill(score, store, keys):
        if not score:
            return False
        for k in keys:
            v = float(score.get(ctx_map[k]) or 0)
            if v <= 0:
                return False
            store[k] = v
        return True

    def _fill_pp(score, store, keys, mapping):
        """Fill whichever prefill contexts measured >0. Qwen3.6 often has 128_pp=0; requiring
        every key used to abort the whole fill and leave all baselines at 0 (PR #403)."""
        if not score:
            return False
        got = False
        for k in keys:
            v = float(score.get(mapping[k]) or 0)
            if v > 0:
                store[k] = v
                got = True
        return got

    got36 = _fill(bres.get("score_qwen36"), q36, ("128", "512", "4k", "16k", "32k"))
    got35 = _fill(bres.get("score_qwen35"), q35, ("128", "4k", "32k", "64k"))
    _fill_pp(bres.get("score_qwen36"), q36, ("128_pp", "512_pp", "4k_pp", "16k_pp", "32k_pp"), pp_map_q36)
    _fill_pp(bres.get("score_qwen35"), q35, ("4k_pp", "32k_pp", "64k_pp", "128k_pp"), pp_map_q35)
    return got36 and got35


def _bidir_baseline_sane(q36, q35):
    """Reject dashboard stubs / failed sweeps (Qwen3.6 ~400+, Qwythos ~120+ on 5090)."""
    return q36.get("128", 0) >= 200 and q35.get("128", 0) >= 80

BASELINE_CACHE_FILE = os.path.join(HERE, ".baseline_cache.json")


def _baseline_box_id(instance=0):
    if ssh_box_enabled():
        h, p = ssh_box_endpoint()
        return f"ssh:{h}:{p}"
    return f"vast:{instance or current_instance(0)}"


def _save_baseline_cache(box_id, q36, q35, bres):
    try:
        with open(BASELINE_CACHE_FILE, "w") as f:
            json.dump({"box": box_id, "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "q36": q36, "q35": q35, "bres": bres}, f)
    except OSError:
        pass


def _origin_main_short():
    """Short SHA of origin/main for baseline cache invalidation after merges."""
    r = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "origin/main"],
                       capture_output=True, text=True)
    return (r.stdout or "").strip() or None


def _bench_harness_changed_between(old_commit, new_commit):
    """True when bench/scripts differs between two main commits (needs fresh baseline)."""
    if not old_commit or not new_commit or old_commit == new_commit:
        return False
    r = subprocess.run(
        ["git", "-C", ROOT, "diff", "--quiet", f"{old_commit}..{new_commit}", "--", "bench/scripts"],
        capture_output=True, text=True)
    return r.returncode != 0


def _baseline_cache_valid(cache, bidir, q36, q35, main_commit=None):
    """Return True when a loaded baseline cache entry is safe to reuse."""
    if not cache:
        return False
    bres = cache.get("bres") or {}
    cached_commit = bres.get("commit")
    if main_commit and cached_commit and cached_commit != main_commit:
        if _bench_harness_changed_between(cached_commit, main_commit):
            return False
    if bidir:
        probe36, probe35 = dict(q36), dict(q35)
        return (_apply_bidir_ctx_from_bres(bres, probe36, probe35)
                and _bidir_baseline_sane(probe36, probe35))
    return bool(bres.get("pass") and bres.get("tps"))


def _load_baseline_cache(box_id, max_age_hours=12):
    try:
        with open(BASELINE_CACHE_FILE) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("box") != box_id:
        return None
    try:
        ts = datetime.datetime.fromisoformat(data["ts"].replace("Z", "+00:00"))
        age_h = (datetime.datetime.now(datetime.timezone.utc) - ts).total_seconds() / 3600
        if age_h > max_age_hours:
            return None
    except (TypeError, ValueError):
        return None
    return data


def _parse_only_prs(only_pr, only_prs):
    out = set()
    if only_pr:
        for part in str(only_pr).split(","):
            part = part.strip()
            if part:
                out.add(int(part))
    for part in (only_prs or "").split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out

# Subsystem buckets for the deterministic area:<name> label (from a PR's top-level changed
# dirs — no AI). Categorization/display only: SN74 scoring is speedup-only (the eval:* tier),
# NOT a per-subsystem budget.
AREAS = {"kernels", "runtime", "moe", "bench"}

# RTX 5090 evaluation is OPT-IN *and* proof-gated. A PR is only evaluated if it ticks the
# "Tested on RTX 5090" box AND fills the decode and/or Qwen3.5 prefill before/after tables with
# real numbers showing a clear improvement (after > before) on at least one metric — checking the
# box alone is not enough (it wasted GPU on PRs whose tables were still placeholder). States:
# greenlit -> test-on-5090 (eval); box ticked but no valid before<after gain ->
# needs-benchmark (skip + ask for numbers); box unticked -> auto-close (rtx5090-required).
# PRs touching runtime/ must tick the box even when the proof section was removed — non-speed
# runtime fixes stay out of the open queue until greenlit or exempted via hold/maintainer.
EVAL_GATE_LABEL  = "test-on-5090"     # bot-set marker: greenlit, will be evaluated
NOT_TESTED_LABEL = "not-tested"       # legacy label — removed on close; no longer applied
NEEDS_BENCH_LABEL = "needs-benchmark" # box ticked but decode/prefill tables missing/invalid/no-gain
# Per-round merge workflow (all queued PRs graded vs the same-box main in one round):
MERGE_FIRST_LABEL  = "merge-first"    # the round's biggest verified speedup — merge this one first
NEEDS_REBASE_LABEL = "needs-rebase"   # also a verified speedup, but not the round winner
REEVALUATE_LABEL   = "re-evaluate"    # winner merged → rebase onto new main; bot re-evals on push
HOLD_LABEL         = "hold"           # maintainer override: never auto-merge this PR
RTX5090_CLOSE_SKIP_LABELS = {HOLD_LABEL}
RTX5090_EXEMPT_LOGINS = {"ai-hpc"}
RTX5090_EXEMPT_ASSOC = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
STALE_PR_DAYS      = int(os.environ.get("SPARKINFER_STALE_PR_DAYS", "2"))
DRAFT_STALE_DAYS   = int(os.environ.get("SPARKINFER_DRAFT_STALE_DAYS", "4"))
STALE_CLOSE_SKIP_LABELS = {HOLD_LABEL, MERGE_FIRST_LABEL}  # protected from auto-close
EXHAUSTED_EVAL_MAX = int(os.environ.get("SPARKINFER_EXHAUSTED_EVAL_MAX", "2"))
FAIL_VERDICT_LABELS = frozenset({"none", "REJECT"})
INFRA_LABEL = "infra-error"
CONTEXT_LABELS     = {"128-context", "512-context", "4k-context", "16k-context", "32k-context",
                      "64k-context", "128k-context"}
REGRESSION_LABELS  = {"regression-128", "regression-512", "regression-4k", "regression-16k",
                      "regression-32k", "regression-64k", "regression-128k",
                      "regression-128-pp", "regression-512-pp", "regression-4k-pp",
                      "regression-16k-pp", "regression-32k-pp", "regression-64k-pp", "regression-128k-pp"}

# Per-context guard baseline fallbacks for display when the RESULT_JSON baseline is 0.
# Mirrors evaluate_dual.sh hardcoded defaults (used when both eval-box measurement and
# bot env var are unavailable).
_GUARD_BASE_FALLBACK = {
    "guard_128_baseline": 300.16,
    "guard_512_baseline": 296.76,
    "guard_4k_baseline":  287.91,
    "guard_16k_baseline": 338.55,
    "guard_32k_baseline": 301.19,
}

# Auto-merge the round's merge-first winner — OFF unless SPARKINFER_AUTOMERGE=1. Heavily guarded:
# the eval only verifies speed + token-match, so auto-merge is gated on labels, author standing,
# changed paths, and branch protection (gh refuses if checks/reviews aren't satisfied).
AUTO_MERGE_FIRST = os.environ.get("SPARKINFER_AUTOMERGE", "0") == "1"
# Auto-merge is BLOCKED if the PR carries any of these labels:
AUTOMERGE_BLOCK_LABELS = {"copycat", "copycat-warn", "flagged:gaming", "penalty", "needs-benchmark",
                          NEEDS_REBASE_LABEL, REEVALUATE_LABEL, HOLD_LABEL, *REGRESSION_LABELS}
# ...or touches any maintainer-owned / governance path (contributor speedups live in kernels|runtime|moe):
AUTOMERGE_SENSITIVE = ("eval/", "bench/scripts/", ".gittensor/", ".github/", "dashboard/", "CODEOWNERS")

def gh(args):
    return subprocess.run(["gh"] + args, capture_output=True, text=True)

# ---- contributor denylist (eval-gaming / sybil block) ----
# .github/blocked-contributors.txt lists GitHub logins (one per line, # = comment). A PR is blocked
# if its opener OR any commit author/committer is listed. Blocked PRs are labeled, commented, closed,
# and NOT evaluated. Evidence per account: .github/FLAGGED.md
DENYLIST_FILE = os.path.join(ROOT, ".github", "blocked-contributors.txt")
FLAG_LABEL = "flagged:gaming"

def load_denylist():
    try:
        out = set()
        for line in open(DENYLIST_FILE):
            s = line.split("#", 1)[0].strip().lower()
            if s: out.add(s)
        return out
    except Exception:
        return set()

def pr_involved_logins(repo, num):
    """Every GitHub login tied to a PR: opener + each commit's author and committer."""
    owner, r = _owner_repo(repo)
    logins = set()
    info = json.loads(gh(["pr", "view", str(num), "-R", repo, "--json", "author"]).stdout or "{}")
    if info.get("author", {}).get("login"): logins.add(info["author"]["login"].lower())
    out = subprocess.run(["gh", "api", f"repos/{owner}/{r}/pulls/{num}/commits",
                          "--jq", ".[] | (.author.login // \"\") + \"\\n\" + (.committer.login // \"\")"],
                         capture_output=True, text=True)
    for l in (out.stdout or "").splitlines():
        if l.strip(): logins.add(l.strip().lower())
    return logins

def _parse_github_time(ts):
    """Parse GitHub ISO-8601 timestamp to UTC datetime."""
    if not ts:
        return None
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.datetime.fromisoformat(ts)


def pr_inactive_days(pr, now=None):
    """Days since the PR's last activity (updatedAt), or None if unknown."""
    updated = _parse_github_time(pr.get("updatedAt"))
    if not updated:
        return None
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=datetime.timezone.utc)
    return (now - updated).total_seconds() / 86400.0


def pr_draft_since(repo, pr):
    """When the PR's current draft period began.

    Uses the latest ``converted_to_draft`` timeline event when present; otherwise
    ``createdAt`` for PRs opened as draft.
    """
    owner, r = _owner_repo(repo)
    num = pr["number"]
    latest_convert = None
    out = gh(["api", f"repos/{owner}/{r}/issues/{num}/timeline"])
    if out.returncode == 0:
        for e in json.loads(out.stdout or "[]"):
            if e.get("event") != "converted_to_draft":
                continue
            ts = _parse_github_time(e.get("created_at"))
            if ts and (latest_convert is None or ts > latest_convert):
                latest_convert = ts
    if latest_convert:
        return latest_convert
    return _parse_github_time(pr.get("createdAt"))


def pr_draft_days(repo, pr, now=None, *, draft_since=None):
    """Days the PR has been in draft status (activity ignored), or None if unknown."""
    since = draft_since if draft_since is not None else pr_draft_since(repo, pr)
    if not since:
        return None
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if since.tzinfo is None:
        since = since.replace(tzinfo=datetime.timezone.utc)
    return (now - since).total_seconds() / 86400.0


def close_stale_prs(repo, days=STALE_PR_DAYS, dry_run=False, *, drafts_only=None):
    """Close open PRs with no GitHub activity for more than `days` days.

    Uses PR updatedAt (commits, comments, reviews, label changes). Skips PRs labeled
    `hold` or `merge-first`. When `drafts_only` is False, draft PRs are skipped (they are
    handled by close_stale_draft_prs). Returns the set of PR numbers closed (or would-close
    in dry-run).
    """
    if days <= 0:
        return set()
    now = datetime.datetime.now(datetime.timezone.utc)
    prs = json.loads(gh(["pr", "list", "-R", repo, "--state", "open",
                         "--json", "number,title,updatedAt,labels,isDraft"]).stdout or "[]")
    closed = set()
    for pr in prs:
        num = pr["number"]
        is_draft = bool(pr.get("isDraft"))
        if drafts_only is True and not is_draft:
            continue
        if drafts_only is False and is_draft:
            continue
        labels = {l["name"] for l in pr.get("labels", [])}
        if labels & STALE_CLOSE_SKIP_LABELS:
            continue
        inactive = pr_inactive_days(pr, now)
        if inactive is None or inactive <= days:
            continue
        days_idle = int(inactive)
        kind = "draft stale" if is_draft else "stale"
        print(f"PR #{num}: {kind} — no activity for {days_idle}d (limit {days}d)"
              f"{' — would close' if dry_run else ' — closing'}")
        if dry_run:
            closed.add(num)
            continue
        marker = "<!-- sparkinfer-stale-draft -->" if is_draft else "<!-- sparkinfer-stale -->"
        if is_draft:
            body = (f"{marker}\n"
                    f"## Closed: draft inactive for {days}+ days\n\n"
                    f"This **draft** PR has had no updates (commits, comments, reviews, or label "
                    f"changes) for **{days_idle} days** (draft threshold: {days} days).\n\n"
                    "Mark ready for review or reopen when you're ready to continue.")
        else:
            body = (f"{marker}\n"
                    f"## Closed: no activity for {days}+ days\n\n"
                    f"This PR has had no updates (commits, comments, reviews, or label changes) "
                    f"for **{days_idle} days** (threshold: {days} days).\n\n"
                    "Reopen this PR or open a fresh one when you're ready to continue.")
        gh(["pr", "comment", str(num), "-R", repo, "--body", body])
        if gh(["pr", "close", str(num), "-R", repo]).returncode == 0:
            closed.add(num)
    if closed:
        tag = "close_stale_draft_prs" if drafts_only else "close_stale_prs"
        print(f">> {tag}: {'would close' if dry_run else 'closed'} {len(closed)} PR(s): "
              f"{sorted(closed)}")
    return closed


def close_stale_draft_prs(repo, days=DRAFT_STALE_DAYS, dry_run=False):
    """Close draft PRs that have remained in draft status for more than `days` days.

    Uses draft duration (createdAt or latest converted_to_draft), not updatedAt —
    activity does not reset the clock. Skips `hold` and `merge-first`.
    """
    if days <= 0:
        return set()
    now = datetime.datetime.now(datetime.timezone.utc)
    prs = json.loads(gh(["pr", "list", "-R", repo, "--state", "open", "--draft",
                         "--json", "number,title,createdAt,labels,isDraft"]).stdout or "[]")
    closed = set()
    for pr in prs:
        if not pr.get("isDraft"):
            continue
        num = pr["number"]
        labels = {l["name"] for l in pr.get("labels", [])}
        if labels & STALE_CLOSE_SKIP_LABELS:
            continue
        drafted = pr_draft_days(repo, pr, now)
        if drafted is None or drafted <= days:
            continue
        days_drafted = int(drafted)
        print(f"PR #{num}: draft age — in draft for {days_drafted}d (limit {days}d)"
              f"{' — would close' if dry_run else ' — closing'}")
        if dry_run:
            closed.add(num)
            continue
        body = ("<!-- sparkinfer-stale-draft -->\n"
                f"## Closed: draft open for {days}+ days\n\n"
                f"This PR has remained a **draft** for **{days_drafted} days** "
                f"(threshold: {days} days). Recent commits or comments do not reset this timer.\n\n"
                "Mark ready for review or reopen when you're ready to continue.")
        gh(["pr", "comment", str(num), "-R", repo, "--body", body])
        if gh(["pr", "close", str(num), "-R", repo]).returncode == 0:
            closed.add(num)
    if closed:
        print(f">> close_stale_draft_prs: {'would close' if dry_run else 'closed'} "
              f"{len(closed)} PR(s): {sorted(closed)}")
    return closed


def rtx5090_close_exempt(pr):
    """Skip auto-close for drafts, hold, maintainers, and trusted bots."""
    if pr.get("isDraft"):
        return True
    labels = {l["name"] for l in pr.get("labels", [])}
    if labels & RTX5090_CLOSE_SKIP_LABELS:
        return True
    login = (pr.get("author") or {}).get("login", "")
    if login in RTX5090_EXEMPT_LOGINS:
        return True
    if pr.get("authorAssociation") in RTX5090_EXEMPT_ASSOC:
        return True
    return False


def post_rtx5090_unchecked_comment(repo, num, runtime=False):
    owner, r = _owner_repo(repo)
    if runtime:
        reason = (
            "This PR touches **`runtime/`** and was auto-closed because **Tested on RTX 5090** "
            "is not ticked (`- [x]`). Runtime changes must show a real 5090 before/after benchmark "
            "to enter the eval queue."
        )
        reopen = (
            "1. **Edit this PR description** (you can edit while closed): add or restore the "
            "proof-of-speedup section and tick `- [x] Tested on RTX 5090`\n"
            "2. Fill the decode and/or prefill **before → after** tables with real "
            "`bench/scripts/bench.sh` numbers\n"
            "3. **Reopen** this PR\n\n"
            "Non-speed runtime fixes that should not be evaluated need a maintainer `hold` label."
        )
    else:
        reason = (
            "This PR was auto-closed because the template includes **Tested on RTX 5090** "
            "as `- [ ]` (unchecked). Evaluation is opt-in — tick the box only after a real "
            "5090 run."
        )
        reopen = (
            "1. **Edit this PR description** (you can edit while closed): change to "
            "`- [x] Tested on RTX 5090`\n"
            "2. Fill the decode and/or prefill **before → after** tables with real "
            "`bench/scripts/bench.sh` numbers showing improvement\n"
            "3. **Reopen** this PR\n\n"
            "If this PR does not need GPU eval (e.g. docs-only), remove the proof-of-speedup "
            "section from the description instead of leaving an unchecked box."
        )
    body = (
        "<!-- sparkinfer-rtx5090-required -->\n"
        f"## Closed — RTX 5090 checkbox not ticked\n\n"
        f"{reason}\n\n"
        "To submit for review:\n\n"
        f"{reopen}\n\n"
        f"[CONTRIBUTING.md](https://github.com/{owner}/{r}/blob/main/CONTRIBUTING.md)\n\n"
        "<sub>Automated by eval bot / rtx5090-required CI.</sub>"
    )
    gh(["pr", "comment", str(num), "-R", repo, "--body", body])


def close_rtx5090_unchecked_pr(repo, num, dry_run=False, runtime=False):
    """Close one PR with unticked RTX 5090 checkbox (or legacy not-tested label)."""
    if dry_run:
        return True
    for lab in (EVAL_GATE_LABEL, NOT_TESTED_LABEL, NEEDS_BENCH_LABEL):
        if lab in labels_on(repo, num):
            remove_label(repo, num, lab)
    post_rtx5090_unchecked_comment(repo, num, runtime=runtime)
    gh(["pr", "close", str(num), "-R", repo, "-c", "closed"])
    return True


def close_unchecked_rtx5090_prs(repo, dry_run=False):
    """Close open PRs with unticked RTX 5090 checkbox, legacy not-tested label, or runtime/
    changes without a ticked box.

    Docs-only PRs outside runtime/ without the template checkbox are left open.
    Returns PR numbers closed.
    """
    prs = json.loads(gh(["pr", "list", "-R", repo, "--state", "open",
                         "--json", "number,title,labels,isDraft,author,authorAssociation"]).stdout or "[]")
    closed = set()
    for pr in prs:
        num = pr["number"]
        if rtx5090_close_exempt(pr):
            continue
        labels = {l["name"] for l in pr.get("labels", [])}
        legacy = NOT_TESTED_LABEL in labels
        body = (json.loads(gh(["pr", "view", str(num), "-R", repo, "--json", "body"]).stdout or "{}")
                .get("body") or "")
        areas = areas_for_pr(repo, num)
        runtime_gate = "runtime" in areas
        if not legacy and not rtx5090_should_close(body, areas):
            continue
        why = "legacy not-tested" if legacy else ("runtime without ticked box" if runtime_gate
                                                    else "unchecked checkbox")
        print(f"PR #{num}: RTX 5090 gate ({why})"
              f"{' — would close' if dry_run else ' — closing'}")
        if dry_run:
            closed.add(num)
            continue
        close_rtx5090_unchecked_pr(repo, num, runtime=runtime_gate and not legacy)
        closed.add(num)
    if closed:
        print(f">> close_unchecked_rtx5090: {'would close' if dry_run else 'closed'} "
              f"{len(closed)} PR(s): {sorted(closed)}")
    return closed


def _eval_verdict_from_comment(body):
    """Return eval tier label from a completed auto-eval comment, or None."""
    if _evaluated_commit_from_comment(body) is None:
        return None
    m = re.search(r"\|\s*\*\*label\*\*\s*\|\s*`eval:([^`]+)`", body)
    return m.group(1) if m else None


def _public_eval_label(res):
    """GitHub label tier for a verdict — infra failures are not scored REJECTs.

    Bidir: either-side fail headlines from the failing side (REJECT preferred), so a passing
    Qwen3.6 XL cannot paper over a Qwen3.5 regression (PR #555).
    """
    if not res:
        return None
    if res.get("infra_error"):
        return INFRA_LABEL
    if res.get("mode") == "bidir" or res.get("score_qwen35") or res.get("score_qwen36"):
        sides = (
            res.get("score_qwen35") or {"label": res.get("label_qwen35"), "pass": res.get("pass_qwen35")},
            res.get("score_qwen36") or {"label": res.get("label_qwen36"), "pass": res.get("pass_qwen36")},
        )
        if any(s.get("pass") is False for s in sides):
            for s in sides:
                if s.get("pass") is False and s.get("label") == "REJECT":
                    return "REJECT"
            for s in sides:
                if s.get("pass") is False and s.get("label"):
                    return s["label"]
    return res.get("label")


def none_reject_eval_count(repo, num):
    """Count completed auto-eval comments whose verdict is none or REJECT."""
    r = gh(["pr", "view", str(num), "-R", repo, "--json", "comments"])
    n = 0
    for c in json.loads(r.stdout or "{}").get("comments", []):
        if _eval_verdict_from_comment(c.get("body", "")) in FAIL_VERDICT_LABELS:
            n += 1
    return n


def close_exhausted_eval_prs(repo, max_none_reject=EXHAUSTED_EVAL_MAX, dry_run=False):
    """Close open PRs with more than `max_none_reject` completed none/REJECT evals.

    Default max is 2 — a third none or REJECT triggers auto-close. Skips `hold`, `merge-first`,
    and draft PRs. Returns PR numbers closed (or would-close in dry-run).
    """
    if max_none_reject < 0:
        return set()
    prs = json.loads(gh(["pr", "list", "-R", repo, "--state", "open",
                         "--json", "number,title,labels,isDraft"]).stdout or "[]")
    closed = set()
    for pr in prs:
        num = pr["number"]
        if pr.get("isDraft"):
            continue
        labels = {l["name"] for l in pr.get("labels", [])}
        if labels & STALE_CLOSE_SKIP_LABELS:
            continue
        count = none_reject_eval_count(repo, num)
        if count <= max_none_reject:
            continue
        print(f"PR #{num}: {count} none/REJECT eval(s) (limit >{max_none_reject})"
              f"{' — would close' if dry_run else ' — closing'}")
        if dry_run:
            closed.add(num)
            continue
        body = ("<!-- sparkinfer-exhausted -->\n"
                f"## Closed: {count} evaluations with no verified speedup or rejection\n\n"
                f"This PR received **{count}** completed sparkinfer auto-evaluations labeled "
                f"`eval:none` or `eval:REJECT` (limit: more than {max_none_reject}).\n\n"
                "Open a fresh PR if you have a new optimization to try.")
        gh(["pr", "comment", str(num), "-R", repo, "--body", body])
        if gh(["pr", "close", str(num), "-R", repo]).returncode == 0:
            closed.add(num)
    if closed:
        print(f">> close_exhausted_eval_prs: {'would close' if dry_run else 'closed'} "
              f"{len(closed)} PR(s): {sorted(closed)}")
    return closed


def run_poll_auto_closes(repo, dry_run=False):
    """Stale / draft-stale / unchecked-5090 / exhausted-eval sweeps — run at each bot poll tick."""
    closed = set()
    for got in (
        close_stale_prs(repo, days=STALE_PR_DAYS, dry_run=dry_run, drafts_only=False),
        close_stale_draft_prs(repo, dry_run=dry_run),
        close_unchecked_rtx5090_prs(repo, dry_run=dry_run),
        close_exhausted_eval_prs(repo, dry_run=dry_run),
    ):
        closed |= got
    return closed


def maybe_close_exhausted_pr(repo, num, dry_run=False):
    """Close one PR if it just crossed the none/REJECT exhaustion threshold."""
    r = gh(["pr", "view", str(num), "-R", repo, "--json", "labels,isDraft"])
    pr = json.loads(r.stdout or "{}")
    if pr.get("isDraft"):
        return False
    labels = {l["name"] for l in pr.get("labels", [])}
    if labels & STALE_CLOSE_SKIP_LABELS:
        return False
    count = none_reject_eval_count(repo, num)
    if count <= EXHAUSTED_EVAL_MAX:
        return False
    print(f"PR #{num}: {count} none/REJECT eval(s) (limit >{EXHAUSTED_EVAL_MAX})"
          f"{' — would close' if dry_run else ' — closing'}")
    if dry_run:
        return True
    body = ("<!-- sparkinfer-exhausted -->\n"
            f"## Closed: {count} evaluations with no verified speedup or rejection\n\n"
            f"This PR received **{count}** completed sparkinfer auto-evaluations labeled "
            f"`eval:none` or `eval:REJECT` (limit: more than {EXHAUSTED_EVAL_MAX}).\n\n"
            "Open a fresh PR if you have a new optimization to try.")
    gh(["pr", "comment", str(num), "-R", repo, "--body", body])
    return gh(["pr", "close", str(num), "-R", repo]).returncode == 0


def close_blocked_pr(repo, num, hits):
    """Label flagged:gaming, comment with the reason, and close the PR. Returns True on close."""
    add_label(repo, num, FLAG_LABEL)
    who = ", ".join(f"`{h}`" for h in sorted(hits))
    body = ("<!-- sparkinfer-flagged -->\n"
            "## 🚩 Flagged: eval-gaming\n\n"
            f"This PR involves an account blocked for gaming the SN74 emission mechanism "
            f"(sybil / coordinated duplicate farming): {who}.\n\n"
            "Per the project's no-gaming policy these accounts are blocked: the PR is **not "
            "evaluated, scored, or merged**. See [`.github/FLAGGED.md`]"
            "(../blob/main/.github/FLAGGED.md) for the evidence and record.")
    gh(["pr", "comment", str(num), "-R", repo, "--body", body])
    return gh(["pr", "close", str(num), "-R", repo]).returncode == 0

def block_account(login, reason):
    """Append a login to the denylist file + a reason to FLAGGED.md (deduped)."""
    cur = load_denylist()
    if login.lower() not in cur:
        with open(DENYLIST_FILE, "a") as f: f.write(f"\n{login}\n")
    with open(FLAG_FILE, "a") as f:
        f.write(f"\n## {datetime.date.today().isoformat()} — `{login}` (auto-blocked)\n\n{reason}\n")

# ---- copycat detection (a later PR that re-submits an earlier PR's diff) ----
# Tiered policy (shared with eval/copycat_policy.py + copycat_guard.py):
#   ≥85% containment → block + close; 75–84% → copycat-warn; 3 warns → block.
from copycat_policy import COPYCAT_BLOCK, COPYCAT_WARN, MAX_WARNINGS, skip_copycat_scoring
from copycat_guard import warn_copycat, list_reference_prs

FLAG_FILE = os.path.join(ROOT, ".github", "FLAGGED.md")
COPYCAT_LABEL = "copycat"
COPYCAT_WARN_LABEL = "copycat-warn"
COPYCAT_CLEARED_LABEL = "copycat-cleared"
COPYCAT_LOG = os.path.join(ROOT, ".github", "copycats.json")
COPYCAT_CONTAINMENT = COPYCAT_BLOCK   # back-compat alias
PENALTY_DAYS = 5             # legacy penalty window for old log entries
PENALTY_LABEL = "penalty"

def author_penalty_until(author):
    """If `author` has an active copycat strike, return the date the penalty lifts, else None.
    Each strike lifts at its date + its own `penalty_days` (default PENALTY_DAYS); a strike entry
    may override `penalty_days` for leniency (e.g. a first-time contributor's first mistake). The
    author is penalized until the latest lift date. Applies from the FIRST strike onward."""
    if not author: return None
    lifts = []
    for e in load_copycat_log():
        if str(e.get("author", "")).lower() == author.lower():
            if e.get("blocked") is False and int(e.get("penalty_days", PENALTY_DAYS)) == 0:
                continue
            try:
                d = datetime.date.fromisoformat(e["date"])
                days = int(e.get("penalty_days", PENALTY_DAYS))
                lifts.append(d + datetime.timedelta(days=days))
            except Exception:
                pass
    if not lifts: return None
    until = max(lifts)
    return until if datetime.date.today() <= until else None

def pr_fingerprint(repo, num):
    """(changed files, normalized non-empty added lines) from the PR's unified diff."""
    diff = gh(["pr", "diff", str(num), "-R", repo]).stdout or ""
    files, added = set(), set()
    for line in diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            p = line[4:].strip()
            if p.startswith(("a/", "b/")): p = p[2:]
            if p and p != "/dev/null": files.add(p)
        elif line.startswith("+") and not line.startswith("+++"):
            s = line[1:].strip()
            if s and not s.startswith(("//", "#", "/*", "*")): added.add(s)  # skip comment-only lines
    return files, added

def containment(copy_added, orig_added):
    if not copy_added: return 0.0
    return len(copy_added & orig_added) / len(copy_added)

def load_copycat_log():
    try: return json.load(open(COPYCAT_LOG))
    except Exception: return []

def save_copycat_log(log):
    os.makedirs(os.path.dirname(COPYCAT_LOG), exist_ok=True)
    with open(COPYCAT_LOG, "w") as f: json.dump(log, f, indent=2)

def push_github_state(msg):
    subprocess.run(["git", "-C", ROOT, "add", ".github/copycats.json",
                    ".github/blocked-contributors.txt", ".github/FLAGGED.md"], capture_output=True)
    if subprocess.run(["git", "-C", ROOT, "diff", "--cached", "--quiet"]).returncode == 0:
        return
    subprocess.run(["git", "-C", ROOT, "commit", "-q", "-m", msg], capture_output=True)
    subprocess.run(["git", "-C", ROOT, "pull", "-q", "--rebase", "origin", "main"], capture_output=True)
    subprocess.run(["git", "-C", ROOT, "push", "-q", "origin", "main"], capture_output=True)

def flag_copycat(repo, num, original, author):
    add_label(repo, num, COPYCAT_LABEL)
    body = (f"<!-- sparkinfer-copycat -->\n## 🐈 Flagged: copycat\n\n"
            f"This PR re-submits substantially the same diff (≥85% line overlap) as the earlier "
            f"#{original}. Duplicating another contributor's work is treated as gaming the SN74 "
            f"emission mechanism. The account has been **blocked** and this PR **closed**.\n\n"
            f"See [`.github/COPYCATS.md`](../blob/main/.github/COPYCATS.md).")
    gh(["pr", "comment", str(num), "-R", repo, "--body", body])

def _evaluated_commit_from_comment(body):
    """Return commit oid if a PR comment records a completed eval verdict, else None.

    Error posts reuse the eval marker but lack the verdict header — they must not block re-runs."""
    if not body or "auto-eval errored" in body:
        return None
    m = re.search(r"<!-- sparkinfer-eval:([0-9a-f]+) -->", body)
    if not m or "sparkinfer auto-eval —" not in body:
        return None
    return m.group(1)

def evaluated_commits(repo, num):
    r = gh(["pr", "view", str(num), "-R", repo, "--json", "comments"])
    done = set()
    for c in json.loads(r.stdout or "{}").get("comments", []):
        oid = _evaluated_commit_from_comment(c.get("body", ""))
        if oid:
            done.add(oid)
    return done

def areas_for_pr(repo, num):
    """Subsystems a PR touches, from its changed file paths (deterministic, no AI)."""
    files = json.loads(gh(["pr", "view", str(num), "-R", repo, "--json", "files"]).stdout or "{}").get("files", [])
    return {f["path"].split("/", 1)[0] for f in files} & set(AREAS)

def _table_val(body, key, metric="decode"):
    """Pull a numeric tok/s from a PR template table row '| before… | <n> |'.

    metric='decode' skips prefill rows; metric='prefill' requires 'prefill' in the row label."""
    for ln in body.splitlines():
        low = ln.lower()
        if metric == "decode" and "prefill" in low:
            continue
        if metric == "prefill" and "prefill" not in low:
            continue
        m = re.match(rf"\s*\|\s*{key}\b[^|]*\|\s*([^|]*?)\s*\|", ln, re.I)
        if not m:
            continue
        num = re.search(r"[-+]?\d+\.?\d*", m.group(1))
        try:
            return float(num.group(0)) if num else None
        except ValueError:
            return None
    return None

def _decode_val(body, key):
    """Decode tok/s from the template decode table (not the prefill table)."""
    return _table_val(body, key, metric="decode")

def _prefill_val(body, key):
    """Qwythos prefill pp tok/s from the template prefill table."""
    return _table_val(body, key, metric="prefill")

def _claimed_gain(before, after):
    """True when both numbers are present and after > before."""
    return before is not None and after is not None and after > before

def _rtx5090_line(ln):
    return "5090" in ln

def rtx5090_has_checkbox(body):
    """True when the PR body still has the template RTX 5090 markdown checkbox line."""
    for ln in (body or "").splitlines():
        if not _rtx5090_line(ln):
            continue
        if re.search(r"-\s*\[\s*[xX]\s*\]", ln) or re.search(r"-\s*\[\s*\]", ln):
            return True
    return False

def rtx5090_box_checked(body):
    """True when the PR template's 'Tested on RTX 5090' checkbox is ticked."""
    return any(
        _rtx5090_line(ln) and re.search(r"-\s*\[\s*[xX]\s*\]", ln)
        for ln in (body or "").splitlines()
    )

def rtx5090_should_close(body, areas=None):
    """True when the 5090 box is not ticked and either the template checkbox is present
    unchecked, or the PR touches runtime/ (runtime changes require greenlight)."""
    if rtx5090_box_checked(body):
        return False
    areas = areas or set()
    if "runtime" in areas:
        return True
    return rtx5090_has_checkbox(body)

def greenlight_status(repo, num, pr_labels):
    """Decide whether a PR may be evaluated. Returns (status, reason):
      'ok'        — greenlit: box ticked + real decode and/or prefill before<after gain
      'no-bench'  — box ticked but neither table shows a claimed improvement
      'unchecked' — the 'Tested on RTX 5090' box is not ticked
    Checking the box is necessary but NOT sufficient — decode or prefill gain must be claimed."""
    body = (json.loads(gh(["pr", "view", str(num), "-R", repo, "--json", "body"]).stdout or "{}")
            .get("body") or "")
    if not rtx5090_box_checked(body):
        return "unchecked", "RTX-5090 box unchecked"
    d_before, d_after = _decode_val(body, "before"), _decode_val(body, "after")
    p_before, p_after = _prefill_val(body, "before"), _prefill_val(body, "after")
    decode_ok = _claimed_gain(d_before, d_after)
    prefill_ok = _claimed_gain(p_before, p_after)
    if decode_ok and prefill_ok:
        return ("ok", f"ticked + decode {d_before}→{d_after} tok/s (+{d_after - d_before:.1f}); "
                f"prefill {p_before}→{p_after} pp tok/s (+{p_after - p_before:.1f})")
    if decode_ok:
        return "ok", f"ticked + decode {d_before}→{d_after} tok/s (+{d_after - d_before:.1f})"
    if prefill_ok:
        return "ok", (f"ticked + prefill {p_before}→{p_after} pp tok/s "
                      f"(+{p_after - p_before:.1f})")
    if d_before is not None and d_after is not None and d_after <= d_before:
        if p_before is not None and p_after is not None and p_after <= p_before:
            return ("no-bench",
                    f"claimed decode {d_before}≥{d_after} and prefill {p_before}≥{p_after} (no improvement)")
        return "no-bench", f"claimed decode before={d_before} ≥ after={d_after} (no improvement)"
    if p_before is not None and p_after is not None and p_after <= p_before:
        return ("no-bench",
                f"claimed prefill before={p_before} ≥ after={p_after} (no improvement)")
    if (d_before is None or d_after is None) and (p_before is None or p_after is None):
        return ("no-bench",
                "box ticked but decode and/or prefill before/after not filled with real numbers")
    return "no-bench", "box ticked but no claimed decode or prefill improvement"

def pr_merge_conflict(mergeable):
    """True when GitHub reports the PR cannot merge cleanly into its base branch."""
    return mergeable == "CONFLICTING"

def post_merge_conflict_comment(repo, num):
    body = ("<!-- sparkinfer-merge-conflict -->\n## ⏸ Merge conflict — rebase before eval\n\n"
            "This branch **conflicts with `main`**, so the RTX 5090 eval is skipped until you rebase.\n\n"
            "Please **rebase onto `main`** and push — the bot re-evaluates on the next poll once the "
            "PR is cleanly mergeable.")
    gh(["pr", "comment", str(num), "-R", repo, "--body", body])

def post_needs_bench_comment(repo, num):
    body = ("<!-- sparkinfer-needs-bench -->\n## ⏳ Needs a benchmark to be evaluated\n\n"
            "You ticked **Tested on RTX 5090** but neither the decode **nor** the Qwen3.5 prefill "
            "**before → after** table shows a real claimed improvement. The on-device eval won't run "
            "until at least one does.\n\n**Decode** (end-to-end, not an isolated-kernel microbench):\n"
            "```bash\nbench/scripts/bench.sh --download            # baseline (before)\n"
            "bench/scripts/bench.sh --download            # your branch (after)\n```\n\n"
            "**Prefill pp** (Qwythos — report your best of 4k/32k/64k from the `prefill pp` "
            "line in bench output):\n```bash\nbench/scripts/bench.sh --download --ctx 4096   # try "
            "4096 / 32768 / 65536 / 131072\nbench/scripts/bench.sh --download --ctx 4096   # your "
            "branch\n```\n\nThen the bot greenlights it on the next poll and evaluates it on an "
            "RTX 5090.")
    gh(["pr", "comment", str(num), "-R", repo, "--body", body])

def _owner_repo(repo):
    parts = repo.split("/"); return parts[0], parts[1]

def labels_on(repo, num):
    owner, r = _owner_repo(repo)
    out = subprocess.run(["gh", "api", f"repos/{owner}/{r}/issues/{num}/labels",
                          "--jq", "[.[].name]"], capture_output=True, text=True)
    try: return set(json.loads(out.stdout))
    except Exception: return set()

def add_label(repo, num, label):
    owner, r = _owner_repo(repo)
    subprocess.run(["gh", "api", f"repos/{owner}/{r}/issues/{num}/labels",
                    "--method", "POST", "-f", f"labels[]={label}"],
                   capture_output=True, text=True)

def remove_label(repo, num, label):
    owner, r = _owner_repo(repo)
    subprocess.run(["gh", "api", f"repos/{owner}/{r}/issues/{num}/labels/{label}",
                    "--method", "DELETE"], capture_output=True, text=True)

def apply_area_labels(repo, num, areas):
    want = {f"area:{a}" for a in areas}
    have = {l for l in labels_on(repo, num) if l.startswith("area:")}
    for lab in want - have: add_label(repo, num, lab)
    for lab in have - want: remove_label(repo, num, lab)

def apply_context_label(repo, num, cur, label):
    if label not in CONTEXT_LABELS:
        return
    for lab in cur & CONTEXT_LABELS:
        if lab != label:
            remove_label(repo, num, lab)
    if label not in cur:
        add_label(repo, num, label)

def apply_regression_labels(repo, num, cur, labels):
    want = {l for l in (labels or []) if l in REGRESSION_LABELS}
    for lab in (cur & REGRESSION_LABELS) - want:
        remove_label(repo, num, lab)
    for lab in want - cur:
        add_label(repo, num, lab)

_PREFILL_CTX_METRICS_Q35 = (
    ("ctx_4096_pp_tps", 4096, "4k-context"),
    ("ctx_32768_pp_tps", 32768, "32k-context"),
    ("ctx_65536_pp_tps", 65536, "64k-context"),
    ("ctx_131072_pp_tps", 131072, "128k-context"),
)
_PREFILL_CTX_METRICS_Q36 = (
    ("ctx_128_pp_tps", 128, "128-context"),
    ("ctx_512_pp_tps", 512, "512-context"),
    ("ctx_4096_pp_tps", 4096, "4k-context"),
    ("ctx_16384_pp_tps", 16384, "16k-context"),
    ("ctx_32768_pp_tps", 32768, "32k-context"),
)
_PREFILL_PP_GATES_Q35 = (
    ("ctx_4096_pp_tps", "guard_4k_pp_pass", "guard_4k_pp_baseline", "4k prefill"),
    ("ctx_32768_pp_tps", "guard_32k_pp_pass", "guard_32k_pp_baseline", "32k prefill"),
    ("ctx_65536_pp_tps", "guard_64k_pp_pass", "guard_64k_pp_baseline", "64k prefill"),
    ("ctx_131072_pp_tps", "guard_128k_pp_pass", "guard_128k_pp_baseline", "128k prefill"),
)
_PREFILL_PP_GATES_Q36 = (
    ("ctx_128_pp_tps", "guard_128_pp_pass", "guard_128_pp_baseline", "128 prefill"),
    ("ctx_512_pp_tps", "guard_512_pp_pass", "guard_512_pp_baseline", "512 prefill"),
    ("ctx_4096_pp_tps", "guard_4k_pp_pass", "guard_4k_pp_baseline", "4k prefill"),
    ("ctx_16384_pp_tps", "guard_16k_pp_pass", "guard_16k_pp_baseline", "16k prefill"),
    ("ctx_32768_pp_tps", "guard_32k_pp_pass", "guard_32k_pp_baseline", "32k prefill"),
)

def _prefill_metrics_for_block(block):
    profile = block.get("prefill_profile")
    if profile == "qwen36" or block.get("ctx_128_pp_tps"):
        return _PREFILL_CTX_METRICS_Q36
    return _PREFILL_CTX_METRICS_Q35

def _prefill_gates_for_block(block):
    profile = block.get("prefill_profile")
    if profile == "qwen36" or block.get("ctx_128_pp_tps"):
        return _PREFILL_PP_GATES_Q36
    return _PREFILL_PP_GATES_Q35

def _best_prefill_measurement(block):
    """Return (tps, ctx, label) for the highest measured prefill context, if any."""
    best = None
    for key, ctx, lbl in _prefill_metrics_for_block(block):
        tps = float(block.get(key) or 0)
        if tps > 0 and (best is None or tps > best[0]):
            best = (tps, ctx, lbl)
    return best

def render(res, oid):
    label = res.get("label", "?")
    display_label = _public_eval_label(res) or label
    if res.get("infra_error"):
        icon = "⚠️"
    else:
        icon = {"REJECT": "❌", "none": "⚪", "BASELINE": "📊"}.get(label, "✅")
    # A passing speedup (XL/L/M/S/XS) clears the significance gate, so its tps becomes the NEW frontier.
    advanced = label in {"XL", "L", "M", "S", "XS"} and res.get("pass")
    ctx_label = res.get("best_context_label")
    # Dual-model runs carry a "guard" block (Qwen3-30B) + "model" (the scored target, Qwen3.6). The
    # verdict can REJECT on the GUARD even when every scored-model gate passes, so the guard's own
    # per-context results must be shown — otherwise the table reads "all pass" next to a REJECT.
    guard = res.get("guard") or {}
    guard36 = res.get("guard36") or {}
    guard3 = res.get("guard3") or {}
    scored_model = res.get("model")
    bidir = res.get("mode") == "bidir" or bool(res.get("score_qwen35"))
    dual = bool(guard) and bool(scored_model) and not (guard36 or guard3) and not bidir
    triple = bool(guard36 or guard3) and bool(scored_model) and not bidir
    short = scored_model.split("(")[0].strip() if scored_model else ""
    if not short and scored_model:
        short = scored_model.split("-")[0]
    gname = res.get("guard_model", "Qwen3-30B")
    rows = [f"| **label** | `eval:{display_label}` |"]
    if bidir:
        if res.get("infra_error"):
            rows.append("| Qwen3.5 score | infra error (not graded) |")
            rows.append("| Qwen3.6 score | infra error (not graded) |")
        else:
            rows.append(f"| Qwen3.5 score | `eval-qwen35:{res.get('label_qwen35', '?')}` "
                        f"({'pass' if res.get('pass_qwen35') else 'fail'}) |")
            rows.append(f"| Qwen3.6 score | `eval-qwen36:{res.get('label_qwen36', '?')}` "
                        f"({'pass' if res.get('pass_qwen36') else 'fail'}) |")
        for title, block in [("Qwen3.5", res.get("score_qwen35") or {}),
                             ("Qwen3.6", res.get("score_qwen36") or {})]:
            if res.get("infra_error"):
                continue
            if not block:
                continue
            bctx = block.get("best_context_label")
            sc = block.get("score_context", 128)
            if block.get("frontier_tps", 0) > 0:
                rows.append(f"| {title} vs same-box main | {block['frontier_tps']} tok/s → "
                            f"{block.get('pct_over_frontier', 0):+.1f}% "
                            f"({block.get('delta_tps', 0):+.1f}) |")
            if block.get("score_metric") == "prefill":
                dsc = block.get("decode_score_context", sc)
                dbctx = block.get("decode_best_context_label", bctx)
                dtps = block.get("decode_tps", block.get("tps", "?"))
                rows.append(f"| {title} scored decode ({dsc} ctx"
                            f"{f' · {dbctx}' if dbctx else ''}) | {dtps} tok/s |")
            else:
                rows.append(f"| {title} scored decode ({sc} ctx"
                            f"{f' · {bctx}' if bctx else ''}) | {block.get('tps', '?')} tok/s |")
            if block.get("prefill_label"):
                pctx = block.get("score_prefill_context", 0)
                plbl = block.get("best_prefill_context_label", "")
                rows.append(f"| {title} scored prefill ({pctx} ctx"
                            f"{f' · {plbl}' if plbl else ''}) | {block.get('prefill_tps', '?')} pp tok/s · "
                            f"`eval-prefill:{block.get('prefill_label')}` |")
            elif block.get("eval_prefill"):
                best_pp = _best_prefill_measurement(block)
                if best_pp:
                    pp_tps, pp_ctx, pp_lbl = best_pp
                    pctx = block.get("score_prefill_context") or pp_ctx
                    plbl = block.get("best_prefill_context_label") or pp_lbl
                    rows.append(f"| {title} scored prefill ({pctx} ctx"
                                f"{f' · {plbl}' if plbl else ''}) | {pp_tps} pp tok/s |")
                else:
                    rows.append(f"| {title} scored prefill | not measured (0 pp tok/s on all contexts) |")
            rows.append(f"| {title} correctness | top-1 {block.get('top1', 0) * 100:.1f}% · "
                        f"KL {block.get('kl', '?')} |")
            for key, gkey, bkey, lbl in [
                    ("ctx_128_tps", "guard_128_pass", "guard_128_baseline", "128-token"),
                    ("ctx_512_tps", "guard_512_pass", "guard_512_baseline", "512-context"),
                    ("ctx_4096_tps", "guard_4k_pass", "guard_4k_baseline", "4k-context"),
                    ("ctx_16384_tps", "guard_16k_pass", "guard_16k_baseline", "16k-context"),
                    ("ctx_32768_tps", "guard_32k_pass", "guard_32k_baseline", "32k-context"),
                    ("ctx_65536_tps", "guard_64k_pass", "guard_64k_baseline", "64k-context"),
                    ("ctx_131072_tps", "guard_128k_pass", "guard_128k_baseline", "128k-context")]:
                tps = block.get(key)
                if tps is None:
                    continue
                if not tps:
                    continue
                gate = "pass" if block.get(gkey, True) else "fail"
                base = block.get(bkey) or _GUARD_BASE_FALLBACK.get(bkey, 0)
                rows.append(f"| {title} {lbl} no-regression gate | {tps} tok/s"
                            f"{f' vs main {base} tok/s' if base else ''} · {gate} |")
            if block.get("eval_prefill"):
                for key, gkey, bkey, lbl in _prefill_gates_for_block(block):
                    if key not in block and bkey not in block:
                        continue
                    tps = block.get(key, 0)
                    gate = "pass" if block.get(gkey, True) else "fail"
                    base = block.get(bkey) or 0
                    rows.append(f"| {title} {lbl} no-regression gate | {tps} pp tok/s"
                                f"{f' vs main {base} pp tok/s' if base else ''} · {gate} |")
    if not bidir:
        rows += [
            f"| scored decode ({res.get('score_context', 128)} ctx{f' · {ctx_label}' if ctx_label else ''}{f' · {short}' if dual or triple else ''}) | {res.get('tps','?')} tok/s |",
            f"| correctness{f' ({short} vs llama.cpp)' if dual or triple else ''} | top-1 {res.get('top1',0)*100:.1f}% · KL {res.get('kl','?')} |"]
        # scored-model per-context no-regression gates — SKIP contexts not measured (tps 0/None) so a
        # deliberately-skipped 16k/32k never renders a misleading "0.0 tok/s · pass".
        for key, gkey, bkey, lbl in [("ctx_128_tps", "guard_128_pass", "guard_128_baseline", "128-token"),
                                     ("ctx_512_tps", "guard_512_pass", "guard_512_baseline", "512-context"),
                                     ("ctx_4096_tps", "guard_4k_pass", "guard_4k_baseline", "4k-context"),
                                     ("ctx_16384_tps", "guard_16k_pass", "guard_16k_baseline", "16k-context"),
                                     ("ctx_32768_tps", "guard_32k_pass", "guard_32k_baseline", "32k-context"),
                                     ("ctx_65536_tps", "guard_64k_pass", "guard_64k_baseline", "64k-context"),
                                     ("ctx_131072_tps", "guard_128k_pass", "guard_128k_baseline", "128k-context")]:
            tps = res.get(key)
            if not tps:
                continue
            gate = "pass" if res.get(gkey, True) else "fail"
            base = res.get(bkey) or _GUARD_BASE_FALLBACK.get(bkey, 0)
            rows.append(f"| {f'{short} ' if dual or triple else ''}{lbl} no-regression gate | {tps} tok/s"
                        f"{f' vs main {base} tok/s' if base else ''} · {gate} |")
        if res.get("ctx_2048_tps") is not None and res.get("ctx_512_tps") is None:
            gate = "pass" if res.get("guard_2k_pass", True) else "fail"
            base = res.get("guard_2k_baseline") or 0
            rows.append(f"| legacy 2k no-regression gate | {res.get('ctx_2048_tps')} tok/s"
                        f"{f' vs main {base} tok/s' if base else ''} · {gate} |")
    # The Qwen3-30B no-regression guard — the check that actually gates a dual verdict.
    if bidir and not res.get("infra_error"):
        for title, block in [("Qwen3.5 optimize", res.get("score_qwen35") or {}),
                             ("Qwen3.6 optimize", res.get("score_qwen36") or {})]:
            if not block:
                continue
            rows.append(f"| **{title}** | `eval:{block.get('label','?')}` · "
                        f"{block.get('tps','?')} tok/s · {'pass' if block.get('pass') else 'fail'} |")
            g = block.get("guard") or {}
            gname = block.get("guard_model", "guard")
            mname = block.get("model", title.replace(" optimize", ""))
            if g:
                rows.append(f"| {title} — {gname} guard accuracy | "
                            f"top-1 {g.get('top1',0)*100:.1f}% · KL {g.get('kl','?')} · "
                            f"{'pass' if g.get('accuracy_ok', True) else '**FAIL**'} |")
                for key, gkey, lbl in [("ctx_128_tps", "guard_128_pass", "128"),
                                       ("ctx_512_tps", "guard_512_pass", "512"),
                                       ("ctx_4096_tps", "guard_4k_pass", "4k"),
                                       ("ctx_16384_tps", "guard_16k_pass", "16k"),
                                       ("ctx_32768_tps", "guard_32k_pass", "32k"),
                                       ("ctx_65536_tps", "guard_64k_pass", "64k"),
                                       ("ctx_131072_tps", "guard_128k_pass", "128k")]:
                    tps = block.get(key)
                    if not tps:
                        continue
                    rows.append(f"| {title} — {mname} {lbl} | {tps} tok/s · "
                                f"{'pass' if block.get(gkey, True) else '**fail**'} |")
    if dual:
        acc_ok = "pass" if guard.get("accuracy_ok", True) else "**FAIL**"
        rows.append(f"| **{gname} guard — accuracy** | top-1 {guard.get('top1',0)*100:.1f}% · KL {guard.get('kl','?')} · {acc_ok} |")
        for key, gkey, lbl in [("ctx_128_tps", "guard_128_pass", "128-token"),
                               ("ctx_512_tps", "guard_512_pass", "512-context"),
                               ("ctx_4096_tps", "guard_4k_pass", "4k-context"),
                               ("ctx_16384_tps", "guard_16k_pass", "16k-context"),
                               ("ctx_32768_tps", "guard_32k_pass", "32k-context"),
                               ("ctx_65536_tps", "guard_64k_pass", "64k-context"),
                               ("ctx_131072_tps", "guard_128k_pass", "128k-context")]:
            tps = guard.get(key)
            if not tps:
                continue
            rows.append(f"| {gname} guard — {lbl} | {tps} tok/s · {'pass' if guard.get(gkey, True) else '**fail**'} |")
    if triple:
        for gblock, gtitle in [(guard36, res.get("guard36_model", "Qwen3.6")),
                               (guard3, res.get("guard3_model", "Qwen3-30B"))]:
            if not gblock:
                continue
            acc_ok = "pass" if gblock.get("accuracy_ok", True) else "**FAIL**"
            rows.append(f"| **{gtitle} guard — accuracy** | top-1 {gblock.get('top1',0)*100:.1f}% · KL {gblock.get('kl','?')} · {acc_ok} |")
            for key, gkey, lbl in [("ctx_128_tps", "guard_128_pass", "128-token"),
                                   ("ctx_512_tps", "guard_512_pass", "512-context"),
                                   ("ctx_4096_tps", "guard_4k_pass", "4k-context"),
                                   ("ctx_16384_tps", "guard_16k_pass", "16k-context"),
                                   ("ctx_32768_tps", "guard_32k_pass", "32k-context"),
                                   ("ctx_65536_tps", "guard_64k_pass", "64k-context"),
                                   ("ctx_131072_tps", "guard_128k_pass", "128k-context")]:
                tps = gblock.get(key)
                if not tps:
                    continue
                rows.append(f"| {gtitle} guard — {lbl} | {tps} tok/s · {'pass' if gblock.get(gkey, True) else '**fail**'} |")
    if res.get("regression_labels") or res.get("guard_regression_labels"):
        allregs = (res.get("regression_labels") or []) + (res.get("guard_regression_labels") or [])
        rows.append(f"| regressions | {', '.join(allregs)} |")
    if not bidir and "frontier_tps" in res and res.get("frontier_tps", 0) > 0:
        # "frontier_tps" is now the SAME-BOX origin/main baseline — the gain is measured directly
        # against main on the same GPU in the same run, not a passed-in frontier number.
        rows.insert(2, f"| vs same-box main | {res['frontier_tps']} tok/s → "
                       f"{res.get('pct_over_frontier', 0):+.1f}% ({res.get('delta_tps',0):+.1f}) |")
    note = {"REJECT": f"**Rejected** — {res.get('reason','')}.",
            "none": "Within the significance gate — no *verified* speedup over same-box main.",
            "BASELINE": "No same-box main baseline was set; this run establishes one."
            }.get(label, f"Verified speedup over same-box origin/main — "
                         f"{res.get('tps')} tok/s (main was {res.get('frontier_tps','?')} tok/s).")
    if label == "REJECT" and res.get("auto_close") and not res.get("infra_error"):
        note = "No context cleared the 2% significance gate while at least one context regressed. Auto-closing this PR."
    if res.get("infra_error") or str(res.get("reason") or "").startswith("infra error"):
        reason = str(res.get("reason") or "infra error").removeprefix("infra error: ").strip()
        note = ("Infra failure during eval (not a verified PR regression) — re-run recommended."
                + (f"\n\n`{reason}`" if reason else ""))
    target_note = ("128/512/4k/16k/32k guarded · Qwen3.5 prefill at 4k/32k/64k/128k · "
                   "Qwen3.6 prefill at 128/512/4k/16k/32k · scored vs same-box main"
                   if res.get("eval_mode") == "longctx" and res.get("mode") == "bidir"
                   else "128/512/4k/16k/32k guarded · Qwen3.5 prefill at 4k/32k/64k/128k · scored vs same-box main"
                   if res.get("eval_mode") == "longctx" and (res.get("score_qwen35") or {}).get("eval_prefill")
                   else "128/512/4k/16k/32k guarded · scored vs same-box main · strongest context scores"
                   if res.get("eval_mode") == "longctx" else "128-token decode scored vs same-box main")
    return (f"<!-- sparkinfer-eval:{oid} -->\n"
            f"## {icon} sparkinfer auto-eval — `{oid}`\n\n"
            f"| metric | value |\n|---|---|\n" + "\n".join(rows) + "\n\n"
            f"{note}\n\n"
            f"_RTX 5090 (sm_120) · {target_note} · built from source · correctness vs llama.cpp. "
            f"Automated — **not merged**; merge manually after review._")

# ---- live dashboard: owned by sparkinfer-web (public/dashboard/data.json) ----
# The old in-repo dashboard/ under sparkinfer is no longer updated by the bot.
WEB_REPO = os.environ.get("SPARKINFER_WEB_REPO", "https://github.com/gittensor-ai-lab/sparkinfer-web.git")
WEB_DIR = os.path.expanduser(os.environ.get("SPARKINFER_WEB_DIR", "~/.sparkinfer_web_checkout"))
# Default branch currently hosts the Next.js dashboard (main may not have public/dashboard yet).
WEB_BRANCH = os.environ.get("SPARKINFER_WEB_BRANCH", "feat/landing-page")
DASH = os.path.join(WEB_DIR, "public", "dashboard")
DATA_JSON = os.path.join(DASH, "data.json")
FRONTIER_LABELS = {"XL", "L", "M", "S", "XS", "BASELINE"}
SPEEDUP_LABELS = {"XL", "L", "M", "S", "XS"}   # verified speedup over main (BASELINE excluded)
CTX_SERIES = {
    128:   {"metric": "ctx_128_tps",   "guard": "guard_128_baseline",  "status": "frontier_tps",     "label": "128", "color": "#D14D72", "llama": 365.85, "note": "128-token decode, no prefill context"},
    512:   {"metric": "ctx_512_tps",   "guard": "guard_512_baseline",  "status": "longctx_512_tps",  "label": "512", "color": "#7B5DFF", "llama": 342.59, "note": "llama-batched-bench npp=512 ntg=128 npl=1"},
    4096:  {"metric": "ctx_4096_tps",  "guard": "guard_4k_baseline",   "status": "longctx_4k_tps",   "label": "4k", "color": "#0E8A16", "llama": 292.99, "note": "llama-batched-bench npp=4096 ntg=128 npl=1"},
    16384: {"metric": "ctx_16384_tps", "guard": "guard_16k_baseline",  "status": "longctx_16k_tps",  "label": "16k", "color": "#B8860B", "llama": 245.53, "note": "llama-batched-bench npp=16384 ntg=128 npl=1"},
    32768: {"metric": "ctx_32768_tps", "guard": "guard_32k_baseline",  "status": "longctx_32k_tps",  "label": "32k", "color": "#6F42C1", "llama": 192.62, "note": "release-log llama.cpp estimate at 32k, ntg=128"},
    65536: {"metric": "ctx_65536_tps", "guard": "guard_64k_baseline",  "status": "longctx_64k_tps",  "label": "64k", "color": "#E67E22", "llama": 0, "note": "Qwythos long-context decode at 64k, ntg=128"},
    131072: {"metric": "ctx_131072_tps", "guard": "guard_128k_baseline", "status": "longctx_128k_tps", "label": "128k", "color": "#17A2B8", "llama": 0, "note": "Qwythos long-context decode at 128k, ntg=128"},
}
# Qwen3.5 (Qwythos) per-context llama.cpp anchors — colors match CTX_SERIES.
Q35_CTX_ORDER = ("128", "4k", "32k", "64k")
Q35_CTX_SERIES = {
    128:    {"label": "128",  "ref_tps": 220.84},
    4096:   {"label": "4k",   "ref_tps": 221.80},
    32768:  {"label": "32k",  "ref_tps": 221.11},
    65536:  {"label": "64k",  "ref_tps": 220.54},
}
# Qwen3.5 prefill pp anchors — pinned in bench/scripts/reference.lock (RTX 5090).
Q35_PP_ORDER = ("4k", "32k", "64k", "128k")
Q35_PP_SERIES = {
    4096:   {"label": "4k",   "metric": "ctx_4096_pp_tps",   "ref_pp": 11104.62, "color": "#0E8A16"},
    32768:  {"label": "32k",  "metric": "ctx_32768_pp_tps",  "ref_pp": 9772.31,  "color": "#6F42C1"},
    65536:  {"label": "64k",  "metric": "ctx_65536_pp_tps",  "ref_pp": 8153.53,  "color": "#E67E22"},
    131072: {"label": "128k", "metric": "ctx_131072_pp_tps", "ref_pp": 5999.59,  "color": "#17A2B8"},
}
Q36_PP_ORDER = ("128", "512", "4k", "16k", "32k")
Q36_PP_SERIES = {
    128:    {"label": "128",  "metric": "ctx_128_pp_tps",    "ref_pp": 0, "color": "#D14D72"},
    512:    {"label": "512",  "metric": "ctx_512_pp_tps",    "ref_pp": 0, "color": "#7B5DFF"},
    4096:   {"label": "4k",   "metric": "ctx_4096_pp_tps",   "ref_pp": 0, "color": "#0E8A16"},
    16384:  {"label": "16k",  "metric": "ctx_16384_pp_tps",  "ref_pp": 0, "color": "#B8860B"},
    32768:  {"label": "32k",  "metric": "ctx_32768_pp_tps",  "ref_pp": 0, "color": "#6F42C1"},
}
Q36_CTX_ORDER = ("128", "512", "4k", "16k", "32k")
# Qwen3.6-35B-A3B llama.cpp decode refs (RTX 5090) — pinned in bench/scripts/reference.lock
Q36_LLAMA_REF = {128: 275.81, 512: 275.61, 4096: 276.3, 16384: 280.66, 32768: 279.83}

def _dash_uses_web_repo():
    """True when DASH points at the sparkinfer-web checkout (False under unit-test patches)."""
    try:
        return os.path.abspath(DASH).startswith(os.path.abspath(WEB_DIR) + os.sep)
    except Exception:
        return False

def _ensure_web_repo():
    """Clone or update the sparkinfer-web checkout on WEB_BRANCH."""
    if not os.path.isdir(os.path.join(WEB_DIR, ".git")):
        subprocess.run(["git", "clone", "-q", "--branch", WEB_BRANCH, WEB_REPO, WEB_DIR], check=True)
        return
    origin = subprocess.run(
        ["git", "-C", WEB_DIR, "remote", "get-url", "origin"],
        capture_output=True, text=True,
    ).stdout.strip()
    if origin.rstrip("/") != WEB_REPO.rstrip("/"):
        shutil.rmtree(WEB_DIR)
        subprocess.run(["git", "clone", "-q", "--branch", WEB_BRANCH, WEB_REPO, WEB_DIR], check=True)
        return
    subprocess.run(["git", "-C", WEB_DIR, "fetch", "-q", "origin", WEB_BRANCH], check=False)
    subprocess.run(["git", "-C", WEB_DIR, "checkout", "-q", WEB_BRANCH], check=False)
    subprocess.run(["git", "-C", WEB_DIR, "pull", "-q", "--rebase", "origin", WEB_BRANCH], check=False)

def load_dash():
    if _dash_uses_web_repo():
        try:
            _ensure_web_repo()
        except Exception as ex:
            print(f">> dashboard checkout unavailable: {ex}")
            return None
    try:
        with open(DATA_JSON) as f: return json.load(f)
    except Exception:
        return None

def write_dash(data):
    os.makedirs(DASH, exist_ok=True)
    with open(DATA_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    # Legacy static page only — sparkinfer-web serves data.json; skip unless data.js already exists.
    js_path = os.path.join(DASH, "data.js")
    if os.path.exists(js_path):
        with open(js_path, "w") as f:
            f.write("// Generated by eval/pr_eval_bot.py from data.json — do not edit by hand.\n")
            f.write("window.SPARKINFER = " + json.dumps(data, indent=2, ensure_ascii=False) + ";\n")

def push_dash(msg):
    """Commit + push public/dashboard/data.json to sparkinfer-web (not sparkinfer)."""
    if _dash_uses_web_repo():
        try:
            _ensure_web_repo()
        except Exception as ex:
            print(f">> dashboard push skipped (web repo): {ex}")
            return
        git_root = WEB_DIR
        branch = WEB_BRANCH
        paths = ["public/dashboard/data.json"]
        if os.path.exists(os.path.join(DASH, "data.js")):
            paths.append("public/dashboard/data.js")
    else:
        # Unit tests / local backfill with patched DASH — no remote push.
        git_root = ROOT
        branch = "main"
        paths = [os.path.relpath(DATA_JSON, ROOT)]
        js = os.path.join(DASH, "data.js")
        if os.path.exists(js):
            paths.append(os.path.relpath(js, ROOT))
    subprocess.run(["git", "-C", git_root, "add", *paths], capture_output=True)
    if subprocess.run(["git", "-C", git_root, "diff", "--cached", "--quiet"]).returncode == 0:
        return  # nothing changed
    subprocess.run(["git", "-C", git_root, "commit", "-q", "-m", msg], capture_output=True)
    if not _dash_uses_web_repo():
        return
    subprocess.run(["git", "-C", git_root, "pull", "-q", "--rebase", "origin", branch], capture_output=True)
    pr = subprocess.run(["git", "-C", git_root, "push", "-q", "origin", branch], capture_output=True, text=True)
    if pr.returncode != 0:
        print(f">> dashboard push to {WEB_REPO} ({branch}) failed: {(pr.stderr or pr.stdout or '').strip()[-500:]}")
    else:
        print(f">> dashboard pushed to sparkinfer-web ({branch})")

LOG_REPO  = os.environ.get("SPARKINFER_LOG_REPO", "https://github.com/gittensor-ai-lab/sparkinfer-log.git")
LOG_DIR   = os.path.expanduser(os.environ.get("SPARKINFER_LOG_DIR", "~/.sparkinfer_log_checkout"))
LOG_PAGE  = "https://gittensor-ai-lab.github.io/sparkinfer-log/?run="

def _ensure_log_repo():
    """Clone or update the public sparkinfer-log checkout."""
    if not os.path.isdir(os.path.join(LOG_DIR, ".git")):
        subprocess.run(["git", "clone", "-q", LOG_REPO, LOG_DIR], check=True)
        return
    origin = subprocess.run(
        ["git", "-C", LOG_DIR, "remote", "get-url", "origin"],
        capture_output=True, text=True,
    ).stdout.strip()
    if origin.rstrip("/") != LOG_REPO.rstrip("/"):
        shutil.rmtree(LOG_DIR)
        subprocess.run(["git", "clone", "-q", LOG_REPO, LOG_DIR], check=True)
        return
    subprocess.run(["git", "-C", LOG_DIR, "pull", "-q", "--rebase"], check=False)

def upload_eval_log(repo, num, title, oid, res, log_text, baseline, polaris=None):
    """Commit eval log (+ optional Polaris receipt/attestation) to sparkinfer-log.

    polaris: optional dict with keys ``receipt`` and/or ``attestation`` (JSON-serializable).
    Best-effort: never blocks the eval. Returns the page URL only after a successful push.
    """
    try:
        rid = f"{int(num):04d}-{oid[:7]}"
        _ensure_log_repo()
        rundir = os.path.join(LOG_DIR, "runs", rid); os.makedirs(rundir, exist_ok=True)
        result = {"id": rid, "pr": int(num), "title": title,
                  "url": f"https://github.com/{repo}/pull/{num}", "commit": oid[:7],
                  "label": res.get("label"), "tps": res.get("tps"),
                  "baseline_tps": round(baseline, 2) if baseline else None,
                  "delta_pct": res.get("pct_over_frontier"), "delta_tps": res.get("delta_tps"),
                  "top1": res.get("top1"), "kl": res.get("kl"),
                  "gpu": "RTX 5090 (sm_120) · vast.ai", "date": datetime.date.today().isoformat(),
                  "frontier": res.get("frontier_tps"),
                  "eval_mode": res.get("eval_mode"), "score_context": res.get("score_context"),
                  "best_context_label": res.get("best_context_label"),
                  "context_gains_pct": res.get("context_gains_pct"),
                  "regression_labels": res.get("regression_labels"),
                  "auto_close": res.get("auto_close"),
                  "ctx_128_tps": res.get("ctx_128_tps"), "ctx_512_tps": res.get("ctx_512_tps"),
                  "ctx_2048_tps": res.get("ctx_2048_tps"),
                  "ctx_4096_tps": res.get("ctx_4096_tps"),
                  "ctx_16384_tps": res.get("ctx_16384_tps"),
                  "ctx_32768_tps": res.get("ctx_32768_tps"),
                  "guard_128_baseline": res.get("guard_128_baseline"),
                  "guard_128_ratio": res.get("guard_128_ratio"),
                  "guard_128_pass": res.get("guard_128_pass"),
                  "guard_512_baseline": res.get("guard_512_baseline"),
                  "guard_512_ratio": res.get("guard_512_ratio"),
                  "guard_512_pass": res.get("guard_512_pass"),
                  "guard_4k_baseline": res.get("guard_4k_baseline"),
                  "guard_4k_ratio": res.get("guard_4k_ratio"),
                  "guard_4k_pass": res.get("guard_4k_pass"),
                  "guard_16k_baseline": res.get("guard_16k_baseline"),
                  "guard_16k_ratio": res.get("guard_16k_ratio"),
                  "guard_16k_pass": res.get("guard_16k_pass"),
                  "guard_32k_baseline": res.get("guard_32k_baseline"),
                  "guard_32k_ratio": res.get("guard_32k_ratio"),
                  "guard_32k_pass": res.get("guard_32k_pass"),
                  "guard_2k_baseline": res.get("guard_2k_baseline"),
                  "guard_2k_ratio": res.get("guard_2k_ratio"),
                  "guard_2k_pass": res.get("guard_2k_pass"),
                  # M1/H1/C2 provenance — makes the immutable log self-describing + reproducible
                  "clocks_pinned": res.get("clocks_pinned"), "clock_mhz": res.get("clock_mhz"),
                  "clock_spread_mhz": res.get("clock_spread_mhz"), "eval_seed": res.get("eval_seed"),
                  "model_sha_pinned": res.get("model_sha_pinned"), "llama_commit": res.get("llama_commit")}
        if polaris and polaris.get("receipt"):
            result["polaris"] = True
            result["polaris_receipt_id"] = polaris["receipt"].get("receipt_id")
        json.dump(result, open(os.path.join(rundir, "result.json"), "w"), indent=2)
        clean = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<ip>", log_text or "")   # scrub host IPs
        open(os.path.join(rundir, "log.txt"), "w").write(clean)
        if polaris:
            if polaris.get("attestation"):
                json.dump(polaris["attestation"], open(os.path.join(rundir, "attestation.json"), "w"), indent=2)
            if polaris.get("receipt"):
                json.dump(polaris["receipt"], open(os.path.join(rundir, "receipt.json"), "w"), indent=2)
        ipath = os.path.join(LOG_DIR, "index.json")
        idx = json.load(open(ipath)) if os.path.exists(ipath) else []
        idx = [e for e in idx if e.get("id") != rid]
        idx_entry = {"id": rid, "pr": int(num), "title": title, "label": res.get("label"),
                     "delta_pct": res.get("pct_over_frontier"), "tps": res.get("tps"),
                     "score_context": res.get("score_context"), "date": result["date"]}
        if polaris and polaris.get("receipt"):
            idx_entry["polaris"] = True
            idx_entry["polaris_receipt_id"] = polaris["receipt"].get("receipt_id", "")[:16]
        idx.append(idx_entry)
        idx.sort(key=lambda x: x["id"])
        json.dump(idx, open(ipath, "w"), indent=2)
        subprocess.run(["git", "-C", LOG_DIR, "add", "-A"], check=True)
        msg = f"eval: #{num} {oid[:7]} -> eval:{res.get('label')}"
        if polaris and polaris.get("receipt"):
            msg += f" + polaris {polaris['receipt'].get('receipt_id', '?')[:16]}"
        commit = subprocess.run(["git", "-C", LOG_DIR, "commit", "-q", "-m", msg], check=False)
        if commit.returncode != 0:
            print(">> eval-log upload skipped: nothing to commit")
            return None
        push = subprocess.run(["git", "-C", LOG_DIR, "push", "-q"], check=False)
        if push.returncode != 0:
            print(f">> eval-log push failed (rc={push.returncode})")
            return None
        url = LOG_PAGE + rid
        print(f">> eval log: {url}")
        if polaris and polaris.get("receipt"):
            print(f">> Polaris receipt: {url}")
        return url
    except Exception as e:
        print(f">> eval-log upload skipped: {e}")
        return None

def update_dashboard(repo, pr, areas, res, proof_url=None):
    """Upsert the PR's eval verdict into the dashboard TABLE (`prs`) only. The frontier and the
    journey (`landed`) advance only when a PR is actually MERGED — see record_merge() — so the
    chart shows shipped code, never unmerged evals or a losing rival in the same round."""
    data = load_dash()
    if data is None: return
    num = pr["number"]
    entry = {"num": num, "title": pr.get("title", ""), "areas": sorted(areas),
             "label": res.get("label"), "tps": res.get("tps"),
             "delta_pct": res.get("pct_over_frontier"),
             "top1": res.get("top1"), "kl": res.get("kl"),
             "url": f"https://github.com/{repo}/pull/{num}",
             "model": res.get("model", "")}
    # Polaris receipt links (optional — only present when --polaris is used)
    if res.get("polaris_receipt_url"):
        entry["polaris_receipt_url"] = res["polaris_receipt_url"]
    if res.get("polaris_receipt_hash"):
        entry["polaris_receipt_hash"] = res["polaris_receipt_hash"]
    for k in ("eval_mode", "score_context", "best_context_label", "context_gains_pct",
              "regression_labels", "auto_close",
              "mode", "label_qwen35", "label_qwen36", "pass_qwen35", "pass_qwen36",
              "score_qwen35", "score_qwen36",
              # Headline metric so the dashboard can split Evaluated PRs into decode vs prefill.
              "score_metric", "prefill_label", "prefill_tps", "decode_label", "decode_tps",
              "ctx_128_tps", "ctx_512_tps", "ctx_2048_tps", "ctx_4096_tps",
              "ctx_16384_tps", "ctx_32768_tps",
              "guard_128_baseline", "guard_128_ratio", "guard_128_pass",
              "guard_512_baseline", "guard_512_ratio", "guard_512_pass",
              "guard_4k_baseline", "guard_4k_ratio", "guard_4k_pass",
              "guard_16k_baseline", "guard_16k_ratio", "guard_16k_pass",
              "guard_32k_baseline", "guard_32k_ratio", "guard_32k_pass",
              "guard_2k_baseline", "guard_2k_ratio", "guard_2k_pass"):
        if res.get(k) is not None:
            entry[k] = res.get(k)
    if proof_url:
        entry["proof_url"] = proof_url
        m = re.search(r"[?&]run=([^&]+)", proof_url)
        if m:
            entry["proof_run"] = m.group(1)
    data["prs"] = [p for p in data.get("prs", []) if p.get("num") != num]
    data["prs"].insert(0, entry)
    data["prs"] = data["prs"][:50]
    data["updated"] = datetime.date.today().isoformat()
    write_dash(data)
    push_dash(f"dashboard: PR #{num} -> eval:{res.get('label')} ({res.get('tps')} tok/s)")

def _merge_recorded(data, num, e):
    """True if PR #num's merge is already reflected in dashboard journey/frontier."""
    if not e:
        return False
    bidir = e.get("mode") == "bidir" or e.get("model") == "bidir"
    if bidir:
        q36 = bool(e.get("pass_qwen36") and e.get("label_qwen36") in SPEEDUP_LABELS)
        q35 = bool(e.get("pass_qwen35") and e.get("label_qwen35") in SPEEDUP_LABELS)
        in36 = any(m.get("pr") == num for m in data.get("landed_qwen36", []))
        in35 = any(m.get("pr") == num for m in data.get("landed_qwen35", []))
        if q36 and q35:
            return in36 and in35
        if q36:
            return in36
        if q35:
            return in35
        return False
    if e.get("label") not in SPEEDUP_LABELS:
        return True
    if str(e.get("model") or "").startswith("Qwen3.6"):
        return any(m.get("pr") == num for m in data.get("landed_qwen36", []))
    score_ctx = int(e.get("score_context") or 0)
    if e.get("eval_mode") == "longctx" and score_ctx in CTX_SERIES:
        return any(m.get("pr") == num for m in data.get("landed_longctx", []))
    return any(m.get("pr") == num for m in data.get("landed", []))

def sync_merged_dashboard(repo, limit=40):
    """Advance dashboard frontier/journey for recently MERGED PRs with eval rows.

    Eval grades upsert `prs` immediately; `record_merge()` used to run only when a merged PR
    still carried `merge-first`, so manual merges (or label cleanup before sync) left the public
    dashboard stale. This syncs any merged PR that has dashboard eval data but is not yet on
    the journey charts — idempotent and safe to run every cron tick."""
    data = load_dash()
    if data is None:
        return
    by_num = {p["num"]: p for p in data.get("prs", [])}
    if not by_num:
        return
    try:
        merged = json.loads(gh(["pr", "list", "-R", repo, "--state", "merged",
                                "--json", "number", "--limit", str(limit)]).stdout or "[]")
    except Exception as ex:
        print(f">> sync_merged_dashboard: gh list failed: {ex}")
        return
    synced = 0
    for pr in merged:
        num = pr.get("number")
        e = by_num.get(num)
        if e is None or _merge_recorded(data, num, e):
            continue
        print(f">> sync merged PR #{num} -> dashboard frontier/journey")
        record_merge(repo, num)
        synced += 1
        data = load_dash() or data
        by_num = {p["num"]: p for p in data.get("prs", [])}
    if synced:
        print(f">> sync_merged_dashboard: applied {synced} merge(s)")
    # Repair Qwen3.5 journey from prs rows (fixes stale cumulative-bar plateau / wrong sort).
    data = load_dash()
    if data is not None:
        before = json.dumps(data.get("landed_qwen35", []), sort_keys=True)
        before_pp = json.dumps(data.get("landed_qwen35_pp", []), sort_keys=True)
        _rebuild_qwen35_journey(data)
        _rebuild_qwen35_pp_journey(data)
        if (json.dumps(data.get("landed_qwen35", []), sort_keys=True) != before
                or json.dumps(data.get("landed_qwen35_pp", []), sort_keys=True) != before_pp):
            data["updated"] = datetime.date.today().isoformat()
            write_dash(data)
            push_dash("dashboard: repair Qwen3.5 optimization journey")

def _scaled_context_tps(old_tps, measured_tps, guard_tps):
    if measured_tps is None:
        return None
    measured_tps = float(measured_tps)
    old_tps = float(old_tps or 0)
    guard_tps = float(guard_tps or 0)
    if old_tps > 0 and guard_tps > 0:
        return round(old_tps * (measured_tps / guard_tps), 2)
    return round(measured_tps, 2)

def _load_polaris_privkey():
    """Load SparkInfer Ed25519 signing key from SPARKINFER_POLARIS_PRIVATE_KEY (base64, 32 bytes)."""
    import base64
    key_b64 = os.environ.get("SPARKINFER_POLARIS_PRIVATE_KEY", "")
    if not key_b64:
        return None
    try:
        return base64.b64decode(key_b64)
    except Exception as e:
        print(f">> Polaris Ed25519 key load failed: {e}")
        return None

def build_polaris_receipt_from_attestation(attestation, api_key="", privkey=None, pubkey=""):
    """Build a Polaris receipt: TDX via API when available, else Ed25519 fallback."""
    if api_key:
        try:
            from eval.polaris.receipt import build_polaris_receipt
            from eval.polaris.client import PolarisClient

            nonce_input = (
                attestation.get("code", {}).get("commit", "") +
                attestation.get("references", {}).get("model_sha256", "") +
                attestation.get("references", {}).get("eval_seed", "")
            ).encode("utf-8")
            nonce = hashlib.sha256(nonce_input).hexdigest()[:64]
            polaris_resp = PolarisClient(api_key).attest_scoring(
                attestation.get("measurements", {}),
                nonce,
                pubkey,
            )
            receipt = build_polaris_receipt(polaris_resp, attestation)
            tee = polaris_resp.get("tee_attestation", {}) or {}
            intel = (tee.get("verification", {}) or polaris_resp.get("verification", {}) or {}
                     ).get("intel_verified")
            print(f">> Polaris TDX: Intel verified={intel}")
            return receipt
        except Exception as e:
            print(f">> Polaris TDX unavailable: {e}")
            if not privkey:
                raise
            print(">> Polaris: falling back to Ed25519 signing")
    if privkey:
        from eval.polaris.receipt import build_receipt
        receipt = build_receipt(attestation, privkey)
        print(">> Polaris Ed25519: signed with SparkInfer key")
        return receipt
    return None

def _upsert_context_baselines(data, e):
    """Update the displayed per-context live baselines from a merged longctx eval.

    The eval runs against a same-box main baseline, while the dashboard is calibrated from prior
    runs. Apply each context's same-box ratio to the displayed value so hardware variance does not
    make the public chart jump around. Rows never move down from a passing merge; regressions are
    surfaced by eval labels before merge, not by degrading the historical dashboard frontier.
    """
    rows = data.setdefault("context_baselines", [])
    by_ctx = {int(r.get("ctx")): r for r in rows if r.get("ctx") is not None}
    changed = {}
    for ctx, meta in CTX_SERIES.items():
        measured = e.get(meta["metric"])
        if measured is None:
            continue
        row = by_ctx.get(ctx)
        old = row.get("sparkinfer_tps") if row else data.get("status", {}).get(meta["status"], 0)
        new = _scaled_context_tps(old, measured, e.get(meta["guard"]))
        if new is None:
            continue
        if old and new < round(float(old), 2):
            new = round(float(old), 2)
        if row:
            if round(float(row.get("sparkinfer_tps") or 0), 2) != new:
                row["sparkinfer_tps"] = new
                changed[ctx] = new
        else:
            rows.append({"ctx": ctx, "label": meta["label"], "color": meta["color"], "tokens": 128,
                         "sparkinfer_tps": new, "llamacpp_decode_tps": meta["llama"],
                         "llamacpp_note": meta["note"]})
            changed[ctx] = new
        if data.get("status") is not None:
            cur = data["status"].get(meta["status"])
            if cur is None or new > round(float(cur), 2):
                data["status"][meta["status"]] = new
    rows.sort(key=lambda r: int(r.get("ctx") or 0))
    return changed

def _upsert_qwen35_ctx(data, sub):
    """Refresh Qwen3.5 per-context sparkinfer bars from a merged bidir score_qwen35 block.

    Uses same-box measured tok/s directly (vs llama.cpp ref anchors). Unlike Qwen3.6
    context_baselines, do not apply _scaled_context_tps — that compounded merge ratios and
    inflated ctx bars above frontier_tps.
    """
    q35 = data.setdefault("qwen35", {})
    ctx_rows = {r.get("label"): r for r in q35.get("ctx") or []}
    changed = False
    for ctx, meta in Q35_CTX_SERIES.items():
        measured = sub.get(CTX_SERIES[ctx]["metric"])
        if measured is None:
            continue
        new = round(float(measured), 2)
        old = round(float((ctx_rows.get(meta["label"]) or {}).get("tps") or 0), 2)
        if old and new < old:
            continue
        row = ctx_rows.get(meta["label"])
        if row:
            if old != new:
                row["tps"] = new
                row["color"] = CTX_SERIES[ctx]["color"]
                changed = True
        else:
            ctx_rows[meta["label"]] = {
                "label": meta["label"], "color": CTX_SERIES[ctx]["color"],
                "tps": new, "ref_tps": meta["ref_tps"],
            }
            changed = True
    if changed:
        q35["ctx"] = [ctx_rows[k] for k in Q35_CTX_ORDER if k in ctx_rows]
    return changed

def _upsert_qwen35_pp(data, sub):
    """Refresh Qwen3.5 per-context prefill pp bars from a merged bidir score_qwen35 block."""
    q35 = data.setdefault("qwen35", {})
    pp_rows = {r.get("label"): r for r in q35.get("pp") or []}
    changed = False
    for ctx, meta in Q35_PP_SERIES.items():
        measured = sub.get(meta["metric"])
        if measured is None:
            continue
        new = round(float(measured), 2)
        old = round(float((pp_rows.get(meta["label"]) or {}).get("pp") or 0), 2)
        if old and new < old:
            continue
        row = pp_rows.get(meta["label"])
        if row:
            if old != new:
                row["pp"] = new
                row["color"] = meta["color"]
                changed = True
        else:
            pp_rows[meta["label"]] = {
                "label": meta["label"], "color": meta["color"],
                "pp": new, "ref_pp": meta["ref_pp"],
            }
            changed = True
    if changed:
        q35["pp"] = [pp_rows[k] for k in Q35_PP_ORDER if k in pp_rows]
    scored = sub.get("prefill_tps")
    if scored is not None:
        old_f = round(float(q35.get("prefill_frontier_pp") or 0), 2)
        new_f = round(max(old_f, float(scored)), 2)
        if new_f != old_f:
            q35["prefill_frontier_pp"] = new_f
            changed = True
    elif changed and q35.get("pp"):
        peak = max(float(r.get("pp") or 0) for r in q35["pp"])
        old_f = round(float(q35.get("prefill_frontier_pp") or 0), 2)
        if peak > old_f:
            q35["prefill_frontier_pp"] = round(peak, 2)
    if sub.get("prefill_label"):
        q35["prefill_label"] = sub["prefill_label"]
    return changed

def _upsert_qwen36_pp(data, sub):
    """Refresh Qwen3.6 per-context prefill pp bars from a merged bidir score_qwen36 block."""
    q36 = data.setdefault("qwen36", {})
    pp_rows = {r.get("label"): r for r in q36.get("pp") or []}
    changed = False
    for ctx, meta in Q36_PP_SERIES.items():
        measured = sub.get(meta["metric"])
        if measured is None:
            continue
        new = round(float(measured), 2)
        old = round(float((pp_rows.get(meta["label"]) or {}).get("pp") or 0), 2)
        if old and new < old:
            continue
        row = pp_rows.get(meta["label"])
        if row:
            if old != new:
                row["pp"] = new
                row["color"] = meta["color"]
                changed = True
        else:
            pp_rows[meta["label"]] = {
                "label": meta["label"], "color": meta["color"],
                "pp": new, "ref_pp": meta["ref_pp"],
            }
            changed = True
    if changed:
        q36["pp"] = [pp_rows[k] for k in Q36_PP_ORDER if k in pp_rows]
    scored = sub.get("prefill_tps")
    if scored is not None:
        old_f = round(float(q36.get("prefill_frontier_pp") or 0), 2)
        new_f = round(max(old_f, float(scored)), 2)
        if new_f != old_f:
            q36["prefill_frontier_pp"] = new_f
            changed = True
    elif changed and q36.get("pp"):
        peak = max(float(r.get("pp") or 0) for r in q36["pp"])
        old_f = round(float(q36.get("prefill_frontier_pp") or 0), 2)
        if peak > old_f:
            q36["prefill_frontier_pp"] = round(peak, 2)
    if sub.get("prefill_label"):
        q36["prefill_label"] = sub["prefill_label"]
    return changed

def _upsert_qwen36_ctx(data, sub):
    """Refresh Qwen3.6 per-context sparkinfer bars from a merged bidir score_qwen36 block.

    Uses same-box measured tok/s directly (vs llama.cpp ref anchors). Ratchets up only —
    same policy as _upsert_qwen35_ctx.
    """
    q36 = data.setdefault("qwen36", {})
    ctx_rows = {r.get("label"): r for r in q36.get("ctx") or []}
    changed = False
    for ctx, meta in CTX_SERIES.items():
        measured = sub.get(meta["metric"])
        if measured is None:
            continue
        new = round(float(measured), 2)
        label = meta["label"]
        old = round(float((ctx_rows.get(label) or {}).get("tps") or 0), 2)
        if old and new < old:
            continue
        row = ctx_rows.get(label)
        if row:
            if old != new:
                row["tps"] = new
                row["color"] = meta["color"]
                changed = True
        else:
            ctx_rows[label] = {
                "label": label, "color": meta["color"],
                "tps": new, "ref_tps": Q36_LLAMA_REF.get(ctx, meta["llama"]),
            }
            changed = True
    if changed:
        q36["ctx"] = [ctx_rows[k] for k in Q36_CTX_ORDER if k in ctx_rows]
    return changed

def _qwen36_journey_tps(sub):
    """128-token headline for Qwen3.6 journey steps (matches chart hint + landed history)."""
    return float(sub.get("ctx_128_tps") or sub.get("tps") or 0)

def _qwen35_journey_tps(sub):
    """128-token decode headline for Qwen3.5 journey (not longctx best-context tps)."""
    return float(sub.get("ctx_128_tps") or sub.get("tps") or 0)

def _qwen35_journey_pp(sub):
    """Scored prefill headline for Qwen3.5 prefill optimization journey."""
    return float(sub.get("prefill_tps") or 0)

def _qwen35_baseline_from_prs(data):
    """Same-box origin/main 128-tok speed before the first landed Qwen3.5 optimization."""
    earliest_num, guard = None, None
    for e in data.get("prs", []):
        if not (e.get("pass_qwen35") and e.get("label_qwen35") in SPEEDUP_LABELS):
            continue
        num = e.get("num")
        if num is None:
            continue
        sub = e.get("score_qwen35") or e
        g = sub.get("guard_128_baseline")
        if g is None:
            continue
        if earliest_num is None or num < earliest_num:
            earliest_num, guard = num, float(g)
    return round(guard, 2) if guard is not None else None

def _qwen35_pp_baseline_from_prs(data):
    """Same-box origin/main prefill speed before the first landed Qwen3.5 prefill win."""
    earliest_num, baseline = None, None
    for e in data.get("prs", []):
        if not (e.get("pass_qwen35") and e.get("label_qwen35") in SPEEDUP_LABELS):
            continue
        num = e.get("num")
        if num is None:
            continue
        sub = e.get("score_qwen35") or e
        if _qwen35_journey_pp(sub) <= 0:
            continue
        bl = sub.get("frontier_tps")
        if bl is None or float(bl) <= 0:
            continue
        if earliest_num is None or num < earliest_num:
            earliest_num, baseline = num, float(bl)
    return round(baseline, 2) if baseline is not None else None

def _rebuild_qwen35_journey(data):
    """Recompute landed_qwen35 + baseline/frontier from stored prs rows (merge-chronological).

    Each bar is the running 128-tok frontier after that merge (monotonic ratchet), not the raw
    same-box measurement for that PR. Raw tok/s zig-zags with box state and with PRs that target
    long-context only; the ratchet matches Qwen3-MoE journey semantics and the headline frontier.
    """
    existing_dates = {m["pr"]: m.get("date") for m in data.get("landed_qwen35", []) if m.get("pr")}
    pending = []
    for e in data.get("prs", []):
        if not (e.get("pass_qwen35") and e.get("label_qwen35") in SPEEDUP_LABELS):
            continue
        num = e["num"]
        sub = e.get("score_qwen35") or e
        raw = round(_qwen35_journey_tps(sub), 2)
        short = re.sub(r"^\w+(\([^)]*\))?:\s*", "", e.get("title", ""))[:28]
        pending.append({
            "name": short or f"PR #{num}",
            "raw_tps": raw,
            "pr": num,
            "date": existing_dates.get(num) or datetime.date.today().isoformat(),
            "label": e.get("label_qwen35"),
        })
    pending.sort(key=lambda m: (m.get("date", ""), m.get("pr", 0)))
    q35 = data.setdefault("qwen35", {})
    bl = _qwen35_baseline_from_prs(data)
    if bl is not None:
        q35["baseline_tps"] = bl
    running = round(float(bl or 0), 2)
    landed = []
    for ent in pending:
        raw = ent.pop("raw_tps")
        running = round(max(running, raw), 2)
        ent["tps"] = running
        if raw != running:
            ent["raw_tps"] = raw
        landed.append(ent)
    data["landed_qwen35"] = landed
    if landed:
        q35["frontier_tps"] = running

def _rebuild_qwen35_pp_journey(data):
    """Recompute landed_qwen35_pp + prefill baseline/frontier from stored prs rows."""
    decode_dates = {m["pr"]: m.get("date") for m in data.get("landed_qwen35", []) if m.get("pr")}
    existing_dates = {m["pr"]: m.get("date") for m in data.get("landed_qwen35_pp", []) if m.get("pr")}
    existing_dates = {**decode_dates, **existing_dates}
    pending = []
    for e in data.get("prs", []):
        if not (e.get("pass_qwen35") and e.get("label_qwen35") in SPEEDUP_LABELS):
            continue
        num = e["num"]
        sub = e.get("score_qwen35") or e
        raw = round(_qwen35_journey_pp(sub), 2)
        if raw <= 0:
            continue
        short = re.sub(r"^\w+(\([^)]*\))?:\s*", "", e.get("title", ""))[:28]
        pending.append({
            "name": short or f"PR #{num}",
            "raw_tps": raw,
            "pr": num,
            "date": existing_dates.get(num) or datetime.date.today().isoformat(),
            "label": e.get("label_qwen35") or sub.get("prefill_label"),
        })
    pending.sort(key=lambda m: (m.get("date", ""), m.get("pr", 0)))
    q35 = data.setdefault("qwen35", {})
    bl = _qwen35_pp_baseline_from_prs(data)
    if bl is not None:
        q35["baseline_pp"] = bl
    running = round(float(bl or 0), 2)
    landed = []
    for ent in pending:
        raw = ent.pop("raw_tps")
        running = round(max(running, raw), 2)
        ent["tps"] = running
        if raw != running:
            ent["raw_tps"] = raw
        landed.append(ent)
    data["landed_qwen35_pp"] = landed
    if landed:
        q35["prefill_frontier_pp"] = running

def record_merge(repo, num):
    """A frontier-advancing PR was MERGED → advance the displayed frontier by its verified same-box
    relative gain and add it to the journey (`landed`). Hardware-independent and merged-only;
    idempotent (dedupe by PR). Reads the PR's stored eval from `prs`."""
    data = load_dash()
    if data is None: return
    e = next((p for p in data.get("prs", []) if p.get("num") == num), None)
    if not e:
        return
    bidir = e.get("mode") == "bidir" or e.get("model") == "bidir"
    if bidir:
        if not ((e.get("pass_qwen35") and e.get("label_qwen35") in SPEEDUP_LABELS) or
                (e.get("pass_qwen36") and e.get("label_qwen36") in SPEEDUP_LABELS) or
                e.get("label") in SPEEDUP_LABELS):
            return
        changed = False
        if e.get("pass_qwen36") and e.get("label_qwen36") in SPEEDUP_LABELS:
            sub = e.get("score_qwen36") or e
            q36 = data.setdefault("qwen36", {})
            old_f = round(q36.get("frontier_tps") or q36.get("baseline_tps") or 0, 2)
            step = round(_qwen36_journey_tps(sub), 2)
            new_f = round(max(old_f, step), 2)
            q36["frontier_tps"] = new_f
            if sub.get("top1") is not None: q36["token_match"] = round(sub["top1"], 4)
            if sub.get("kl") is not None:   q36["kl"] = round(sub["kl"], 4)
            _upsert_qwen36_ctx(data, sub)
            if sub.get("eval_prefill"):
                _upsert_qwen36_pp(data, sub)
            short = re.sub(r"^\w+(\([^)]*\))?:\s*", "", e.get("title", ""))[:28]
            landed = [m for m in data.get("landed_qwen36", []) if m.get("pr") != num and not m.get("baseline")]
            landed.append({"name": short or f"PR #{num}", "tps": step, "pr": num,
                           "date": datetime.date.today().isoformat(), "label": e.get("label_qwen36")})
            data["landed_qwen36"] = sorted(landed, key=lambda m: m["tps"])
            changed = True
        if e.get("pass_qwen35") and e.get("label_qwen35") in SPEEDUP_LABELS:
            sub = e.get("score_qwen35") or e
            q35 = data.setdefault("qwen35", {})
            if sub.get("top1") is not None: q35["token_match"] = round(sub["top1"], 4)
            if sub.get("kl") is not None:   q35["kl"] = round(sub["kl"], 4)
            _upsert_qwen35_ctx(data, sub)
            _upsert_qwen35_pp(data, sub)
            _rebuild_qwen35_journey(data)
            _rebuild_qwen35_pp_journey(data)
            changed = True
        if changed:
            data["updated"] = datetime.date.today().isoformat()
            write_dash(data)
            push_dash(f"dashboard: PR #{num} merged -> bidir frontier update")
        return
    if e.get("label") not in SPEEDUP_LABELS:
        return
    # A Qwen3.6 dual-eval result must advance the Qwen3.6 frontier/journey, not Qwen3-MoE's — its
    # delta_pct was measured against a different baseline (23 tok/s, +635%), and applying that gain
    # to Qwen3-MoE's 493 frontier inflates the chart 7.4× per PR. Route to landed_qwen36.
    scored_qwen36 = str(e.get("model") or "").startswith("Qwen3.6")
    if scored_qwen36:
        q36 = data.setdefault("qwen36", {})
        old_f = round(q36.get("frontier_tps") or q36.get("baseline_tps") or 0, 2)
        step = round(_qwen36_journey_tps(e), 2)
        new_f = round(max(old_f, step), 2)          # Qwen3.6 frontier: take the max 128-tok tps seen
        q36["frontier_tps"] = new_f
        if e.get("top1") is not None: q36["token_match"] = round(e["top1"], 4)
        if e.get("kl") is not None:   q36["kl"] = round(e["kl"], 4)
        short = re.sub(r"^\w+(\([^)]*\))?:\s*", "", e.get("title", ""))[:28]
        landed = [m for m in data.get("landed_qwen36", []) if m.get("pr") != num and not m.get("baseline")]
        landed.append({"name": short or f"PR #{num}", "tps": step, "pr": num,
                       "date": datetime.date.today().isoformat(), "label": e.get("label")})
        data["landed_qwen36"] = sorted(landed, key=lambda m: m["tps"])
        data["updated"] = datetime.date.today().isoformat()
        write_dash(data)
        push_dash(f"dashboard: PR #{num} merged -> Qwen3.6 frontier {new_f} tok/s")
        return
    if e.get("eval_mode") == "longctx" and int(e.get("score_context") or 0) in (512, 4096, 16384, 32768, 65536, 131072):
        if any(m.get("pr") == num for m in data.get("landed_longctx", [])): return
        score_ctx = int(e.get("score_context") or 0)
        old_f = round((next((r.get("sparkinfer_tps") for r in data.get("context_baselines", [])
                             if int(r.get("ctx") or 0) == score_ctx), None)
                       or data["status"].get(CTX_SERIES[score_ctx]["status"])
                       or e.get("frontier_tps") or 0), 2)
        changed = _upsert_context_baselines(data, e)
        new_f = round((next((r.get("sparkinfer_tps") for r in data.get("context_baselines", [])
                             if int(r.get("ctx") or 0) == score_ctx), None)
                       or changed.get(score_ctx) or e.get("tps") or 0), 2)
        if e.get("top1") is not None: data["status"]["longctx_token_match"] = round(e["top1"], 4)
        if e.get("kl") is not None:   data["status"]["longctx_kl"] = round(e["kl"], 4)
        short = re.sub(r"^\w+(\([^)]*\))?:\s*", "", e.get("title", ""))[:28]
        landed = [m for m in data.get("landed_longctx", []) if m.get("pr") != num]
        landed.append({"name": short or f"PR #{num}", "tps": new_f, "pr": num,
                       "ctx": score_ctx, "date": datetime.date.today().isoformat()})
        data["landed_longctx"] = sorted(landed, key=lambda m: m["tps"])
        data["updated"] = datetime.date.today().isoformat()
        write_dash(data)
        push_dash(f"dashboard: PR #{num} merged -> {CTX_SERIES[score_ctx]['label']} frontier {new_f} tok/s")
        append_frontier_ledger(repo, num, e, old_f, new_f)
        return

    if any(m.get("pr") == num for m in data.get("landed", [])): return       # already recorded
    # Bidir PRs must not advance the Qwen3-MoE frontier — route via the bidir branch above.
    if bidir:
        return
    # Advance by the VERIFIED SAME-BOX RELATIVE GAIN (delta_pct), NOT the raw measured tps. Raw tok/s
    # zig-zags ±2% with whichever box ran (hot vs cool) and breaks the journey's monotonicity; applying
    # the same-box gain to the displayed frontier keeps the headline hardware-independent and the journey
    # a clean calibrated ladder. Falls back to raw max() only if the gain wasn't recorded.
    old_f = round(data["status"].get("frontier_tps") or 0, 2)
    gain = (e.get("delta_pct") or 0) / 100.0
    # Safety: if the model field is missing (stale dual-eval record before the Qwen3.6 routing was
    # added) the delta_pct may be from a different model's baseline, producing impossible gains
    # (e.g. Qwen3.6's +50% applied to Qwen3-30B's frontier). A >30% single-step gain on Qwen3-30B
    # is physically implausible at this stage — treat it as a routing error and fall back to raw tps.
    model_name = str(e.get("model") or "")
    if gain > 0.30 and not model_name.startswith("Qwen3.6"):
        print(f">> record_merge: PR #{num} delta_pct={e.get('delta_pct')}% applied to Qwen3-30B "
              f"frontier {old_f} is implausible (model={model_name!r}) — falling back to raw tps")
        new_f = max(old_f, round(e.get("tps") or 0, 2))
    else:
        new_f = round(old_f * (1 + gain), 2) if gain > 0 else max(old_f, round(e.get("tps") or 0, 2))
    data["status"]["frontier_tps"] = new_f
    if e.get("eval_mode") == "longctx":
        _upsert_context_baselines(data, e)
        row128 = next((r for r in data.get("context_baselines", []) if int(r.get("ctx") or 0) == 128), None)
        if row128:
            row128["sparkinfer_tps"] = max(round(float(row128.get("sparkinfer_tps") or 0), 2), new_f)
    if e.get("top1") is not None: data["status"]["token_match"] = round(e["top1"], 4)
    if e.get("kl") is not None:   data["status"]["kl"] = round(e["kl"], 4)
    short = re.sub(r"^\w+(\([^)]*\))?:\s*", "", e.get("title", ""))[:28]      # strip "area(x): " prefix
    landed = [m for m in data.get("landed", []) if m.get("pr") != num]
    landed.append({"name": short or f"PR #{num}", "tps": new_f, "pr": num,
                   "date": datetime.date.today().isoformat()})
    data["landed"] = sorted(landed, key=lambda m: m["tps"])
    data["updated"] = datetime.date.today().isoformat()
    write_dash(data)
    push_dash(f"dashboard: PR #{num} merged -> frontier {new_f} tok/s")
    append_frontier_ledger(repo, num, e, old_f, new_f)                        # H2: immutable ledger

def append_frontier_ledger(repo, num, e, prev_f, new_f):
    """H2 (verifiable frontier ledger): append an immutable, GitHub-timestamped line to the public
    ledger for every frontier advance — (date, pr, author, merge commit, same-box delta%, prev->new
    frontier, eval-log proof URL). The append-only commit history IS the signature: the frontier
    history is independently auditable line-by-line against the per-run eval logs, so no advance can
    be silently inserted or rewritten. Best-effort; never blocks the merge."""
    try:
        info = json.loads(gh(["pr", "view", str(num), "-R", repo, "--json",
                              "author,mergeCommit"]).stdout or "{}")
        author = (info.get("author") or {}).get("login", "?")
        commit = ((info.get("mergeCommit") or {}).get("oid") or "")[:9]
        _ensure_log_repo()
        entry = {"date": datetime.date.today().isoformat(), "pr": int(num), "author": author,
                 "commit": commit, "delta_pct": e.get("delta_pct"),
                 "prev_frontier": prev_f, "new_frontier": new_f, "proof": e.get("proof_url")}
        with open(os.path.join(LOG_DIR, "ledger.jsonl"), "a") as f:
            f.write(json.dumps(entry) + "\n")
        subprocess.run(["git", "-C", LOG_DIR, "add", "ledger.jsonl"], check=True)
        subprocess.run(["git", "-C", LOG_DIR, "commit", "-q", "-m",
                        f"ledger: #{num} {commit} frontier {prev_f} -> {new_f} (+{entry['delta_pct']}%)"], check=False)
        subprocess.run(["git", "-C", LOG_DIR, "push", "-q"], check=False)
        print(f">> frontier ledger += #{num} ({prev_f} -> {new_f} tok/s)")
    except Exception as ex:
        print(f">> ledger append skipped: {ex}")

def auto_merge_ok(repo, num):
    """Guardrails for auto-merging the merge-first winner. Returns (ok, reason)."""
    info = json.loads(gh(["pr", "view", str(num), "-R", repo, "--json",
                          "state,isDraft,labels,author,mergeable,files"]).stdout or "{}")
    if info.get("state") != "OPEN" or info.get("isDraft"):
        return False, "not an open, non-draft PR"
    labs = {l["name"] for l in info.get("labels", [])}
    eval_tiers = {l.split(":", 1)[1] for l in labs if l.startswith("eval:")}
    if not (eval_tiers & SPEEDUP_LABELS):
        return False, "no verified eval:speedup label"
    blocked = labs & AUTOMERGE_BLOCK_LABELS
    if blocked:
        return False, f"blocking label(s): {', '.join(sorted(blocked))}"
    author = (info.get("author") or {}).get("login", "")
    if author.lower() in load_denylist():
        return False, f"author {author} is blocked"
    if author_penalty_until(author):
        return False, f"author {author} is under penalty"
    sens = [f["path"] for f in info.get("files", []) if any(f["path"].startswith(p) for p in AUTOMERGE_SENSITIVE)]
    if sens:
        return False, f"touches protected paths: {', '.join(sens[:3])}"
    if pr_merge_conflict(info.get("mergeable")):
        return False, "merge conflict with base"
    if info.get("mergeable") != "MERGEABLE":
        return False, f"not cleanly mergeable ({info.get('mergeable')})"
    return True, "ok"

def try_auto_merge(repo, num):
    """Auto-merge the merge-first winner iff all guardrails pass. gh still enforces branch protection
    (required checks/reviews) and refuses otherwise — a safe backstop. Returns True if merged."""
    ok, reason = auto_merge_ok(repo, num)
    if not ok:
        print(f">> auto-merge SKIP #{num}: {reason}"); return False
    r = gh(["pr", "merge", str(num), "-R", repo, "--squash"])
    if r.returncode == 0:
        print(f">> AUTO-MERGED #{num} (merge-first winner)")
        gh(["pr", "comment", str(num), "-R", repo, "--body",
            "<!-- sparkinfer-automerge -->\n✅ Auto-merged as the round's `merge-first` winner — "
            "verified same-box speedup over `main`, all checks green. Thanks for the contribution!"])
        return True
    print(f">> auto-merge BLOCKED #{num} (branch protection/checks): {(r.stderr or r.stdout).strip()[:200]}")
    return False

def reconcile_merge_labels(repo):
    """Per-round merge workflow. After all queued PRs are graded against the same-box main:
      0. Sync recently merged PRs onto the dashboard (manual merges, not only merge-first).
      1. If a `merge-first` PR has since MERGED, its rivals are stale → tag them `re-evaluate` and
         ask them to rebase onto the new main (the bot re-evals automatically on the rebased commit).
      2. Among the still-open PRs with a verified speedup, label the biggest `merge-first` and the
         rest `needs-rebase`. Ranking uses the same-box % gain over main (data.json `delta_pct`)."""
    sync_merged_dashboard(repo)
    data = load_dash() or {}
    by_num = {p["num"]: p for p in data.get("prs", [])}
    open_prs = json.loads(gh(["pr", "list", "-R", repo, "--state", "open",
                              "--json", "number,labels", "--limit", "80"]).stdout or "[]")
    open_labels = {p["number"]: {l["name"] for l in p["labels"]} for p in open_prs}

    # 1) A merge-first PR that merged → its rivals must rebase + re-eval against the new main.
    merged_first = json.loads(gh(["pr", "list", "-R", repo, "--state", "merged", "--label",
                                  MERGE_FIRST_LABEL, "--json", "number", "--limit", "10"]).stdout or "[]")
    if merged_first:
        for m in merged_first:
            record_merge(repo, m["number"])      # idempotent; sync_merged_dashboard usually did this
            remove_label(repo, m["number"], MERGE_FIRST_LABEL)
        # Rivals stay `needs-rebase` (they have NOT rebased yet — that's exactly why `re-evaluate`
        # would be wrong here). Just nudge them to rebase; the eval re-runs on the rebased commit.
        for num, labs in open_labels.items():
            if NEEDS_REBASE_LABEL in labs:
                gh(["pr", "comment", str(num), "-R", repo, "--body",
                    "<!-- sparkinfer-rebase -->\nThe round's `merge-first` PR was just merged. Please "
                    "**rebase this branch onto `main`** — once you push the rebase the bot re-evaluates "
                    "it against the new frontier (crediting your *marginal* gain on top of what merged)."])

    # 2) Rank the open verified-speedup PRs that are FRESH (graded vs the CURRENT main) — i.e. NOT
    #    `needs-rebase`. A needs-rebase PR's score is stale (an older main), so it can't be this
    #    round's winner until it rebases and the bot re-grades it (which clears needs-rebase).
    scored = sorted(((num, by_num[num].get("delta_pct") or 0) for num in open_labels
                     if num in by_num and by_num[num].get("label") in SPEEDUP_LABELS
                     and NEEDS_REBASE_LABEL not in open_labels.get(num, set())),
                    key=lambda x: x[1], reverse=True)
    if not scored: return
    winner = scored[0][0]
    add_label(repo, winner, MERGE_FIRST_LABEL)
    for L in (NEEDS_REBASE_LABEL, REEVALUATE_LABEL): remove_label(repo, winner, L)
    for num, _ in scored[1:]:
        add_label(repo, num, NEEDS_REBASE_LABEL)
        for L in (MERGE_FIRST_LABEL, REEVALUATE_LABEL): remove_label(repo, num, L)
    print(f">> round labels: merge-first #{winner}; needs-rebase {[n for n,_ in scored[1:]] or 'none'}")

    # Optionally auto-merge the winner (guarded). Rivals keep needs-rebase + a rebase nudge.
    if AUTO_MERGE_FIRST and try_auto_merge(repo, winner):
        record_merge(repo, winner)               # merged now → advance the journey/frontier
        for num, _ in scored[1:]:
            gh(["pr", "comment", str(num), "-R", repo, "--body",
                "<!-- sparkinfer-rebase -->\nThe round's `merge-first` PR was just merged. Please "
                "**rebase this branch onto `main`** — the bot re-evaluates it on push (crediting your "
                "*marginal* gain on top of what merged)."])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=int, default=0,
                    help="vast.ai instance id (EVAL_TRANSPORT=vast only; ignored for ssh)")
    ap.add_argument("--frontier", type=float, default=0, help="DEPRECATED: scoring now uses same-box origin/main baseline")
    ap.add_argument("--ceiling", type=float, default=0)
    ap.add_argument("--repo", default="gittensor-ai-lab/sparkinfer")
    ap.add_argument("--dry-run", action="store_true", help="evaluate + print, but don't label/comment")
    # Dual-model: score Qwen3.6-35B-A3B (primary) and guard Qwen3-30B against no-regression.
    # Each PR is scored directly against the SAME-BOX origin/main baseline (measured once per run),
    # not a passed-in frontier number — so the gain is hardware-independent and always current.
    ap.add_argument("--bidir", action="store_true",
                    help="bidirectional Qwen3.5 + Qwen3.6 eval via evaluate_bidir.sh")
    ap.add_argument("--dual", action="store_true",
                    help="legacy alias for --bidir")
    ap.add_argument("--triple", action="store_true",
                    help="legacy alias for --bidir")
    ap.add_argument("--primary-quant", default=os.environ.get("PRIMARY_QUANT", "Q4_K_M"),
                    choices=["Q4_K_M", "Q8_0", "BF16"],
                    help="[--triple] Qwythos GGUF quant (default Q4_K_M)")
    ap.add_argument("--polaris", action="store_true",
                    help="generate a Polaris verifiable receipt for each eval (default: on)")
    ap.add_argument("--no-polaris", action="store_true",
                    help="disable Polaris TDX receipts (overrides POLARIS=1)")
    ap.add_argument("--only-pr", default="",
                    help="evaluate only this PR number (comma-separated list ok, e.g. 531,511,530)")
    ap.add_argument("--only-prs", default="",
                    help="comma-separated PR numbers — one baseline, then eval each (e.g. 387,389)")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="require cached same-box baseline (≤12h, same box, same origin/main); abort if missing")
    ap.add_argument("--reeval", action="store_true",
                    help="re-run eval even if this commit was already graded (use with --only-pr)")
    args = ap.parse_args()
    if args.triple or args.dual:
        args.bidir = True
    if os.environ.get("BIDIR", "1") != "0" and not any(
            a in sys.argv for a in ("--bidir", "--triple", "--dual")):
        args.bidir = True
    if args.no_polaris:
        args.polaris = False
    elif not args.polaris and os.environ.get("POLARIS", "1") != "0":
        args.polaris = True
    if not ssh_box_enabled() and not args.instance:
        ap.error("--instance is required for vast.ai transport (or set EVAL_TRANSPORT=ssh + EVAL_SSH_HOST)")
    if ssh_box_enabled():
        h, p = ssh_box_endpoint()
        print(f">> eval transport: fixed SSH root@{h}:{p} (EVAL_TRANSPORT=ssh, vast.ai disabled)")
    elif vast_enabled():
        print(f">> eval transport: vast.ai (instance {args.instance or current_instance(0)})")
    # Qwen3.6 same-box origin/main baselines (128/512/4k/16k/32k). Env-overridable; measured on RTX 5090.
    QWEN36_BASE = {
        "128": float(os.environ.get("SPARKINFER_QWEN36_128", "300.16")),
        "512": float(os.environ.get("SPARKINFER_QWEN36_512", "296.76")),
        "4k":  float(os.environ.get("SPARKINFER_QWEN36_4K",  "287.91")),
        "16k": float(os.environ.get("SPARKINFER_QWEN36_16K", "338.55")),
        "32k": float(os.environ.get("SPARKINFER_QWEN36_32K", "301.19")),
        "128_pp": float(os.environ.get("SPARKINFER_QWEN36_128_PP", "0")),
        "512_pp": float(os.environ.get("SPARKINFER_QWEN36_512_PP", "0")),
        "4k_pp": float(os.environ.get("SPARKINFER_QWEN36_4K_PP", "0")),
        "16k_pp": float(os.environ.get("SPARKINFER_QWEN36_16K_PP", "0")),
        "32k_pp": float(os.environ.get("SPARKINFER_QWEN36_32K_PP", "0")),
        "llama128": float(os.environ.get("SPARKINFER_QWEN36_LLAMA_128", "275.81")),
        "llama512": float(os.environ.get("SPARKINFER_QWEN36_LLAMA_512", "275.61")),
        "llama4k":  float(os.environ.get("SPARKINFER_QWEN36_LLAMA_4K",  "276.30")),
    }
    QWYTHOS_BASE = {
        "128": float(os.environ.get("SPARKINFER_QWYTHOS_128", "0")),
        "4k":  float(os.environ.get("SPARKINFER_QWYTHOS_4K",  "0")),
        "32k": float(os.environ.get("SPARKINFER_QWYTHOS_32K", "0")),
        "64k": float(os.environ.get("SPARKINFER_QWYTHOS_64K", "0")),
        "128k": float(os.environ.get("SPARKINFER_QWYTHOS_128K", "0")),
        "llama128": float(os.environ.get("QWEN35_9B_LLAMA_128", "0")),
        "llama4k":  float(os.environ.get("QWEN35_9B_LLAMA_4K",  "0")),
        "llama32k": float(os.environ.get("QWEN35_9B_LLAMA_32K", "0")),
        "llama64k": float(os.environ.get("QWEN35_9B_LLAMA_64K", "0")),
        "llama128k": float(os.environ.get("QWEN35_9B_LLAMA_128K", "0")),
        "4k_pp": float(os.environ.get("SPARKINFER_QWYTHOS_4K_PP", "0")),
        "32k_pp": float(os.environ.get("SPARKINFER_QWYTHOS_32K_PP", "0")),
        "64k_pp": float(os.environ.get("SPARKINFER_QWYTHOS_64K_PP", "0")),
        "128k_pp": float(os.environ.get("SPARKINFER_QWYTHOS_128K_PP", "0")),
        "llama4k_pp": float(os.environ.get("QWEN35_9B_LLAMA_4K_PP", "0")),
        "llama32k_pp": float(os.environ.get("QWEN35_9B_LLAMA_32K_PP", "0")),
        "llama64k_pp": float(os.environ.get("QWEN35_9B_LLAMA_64K_PP", "0")),
        "llama128k_pp": float(os.environ.get("QWEN35_9B_LLAMA_128K_PP", "0")),
    }

    # --- Polaris verifiable compute ---
    # TDX (preferred): POLARIS_API_KEY → scoring inside Intel TDX enclave.
    # Ed25519 fallback: SPARKINFER_POLARIS_PRIVATE_KEY when TDX is down or unset.
    # The private key NEVER touches the eval box — the bot signs/submits here on the bot host.
    POLARIS_PRIVKEY = _load_polaris_privkey() if args.polaris else None
    POLARIS_API_KEY = os.environ.get("POLARIS_API_KEY", "")
    POLARIS_PUBKEY = ""  # SparkInfer's Ed25519 public key (used as e2e_pubkey for TDX)

    if args.polaris:
        import base64 as _b64
        # Load the public key from the committed trust anchor
        _pubkey_file = os.path.join(HERE, "polaris", "sparkinfer_eval.pub")
        try:
            with open(_pubkey_file) as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line and not _line.startswith("#"):
                        _b64.b64decode(_line)  # validate
                        POLARIS_PUBKEY = _line
                        break
        except Exception:
            pass

        if POLARIS_API_KEY:
            if POLARIS_PRIVKEY:
                print(">> Polaris TDX enabled (Ed25519 fallback if TDX unavailable)")
            else:
                print(">> Polaris TDX enabled — scoring will run inside Intel TDX enclave")
        elif POLARIS_PRIVKEY:
            print(">> Polaris Ed25519 enabled — receipts will be signed")
        else:
            print(">> Polaris enabled but no POLARIS_API_KEY or SPARKINFER_POLARIS_PRIVATE_KEY set — "
                  "attestations will be collected but NOT signed")

    dash = load_dash()
    frontier = dash["status"]["frontier_tps"] if dash else args.frontier   # live ledger frontier
    # OLDEST-FIRST: evaluate ascending by PR number so the original of any duplicate is seen before
    # its copy, and the earliest submitter is graded first (fairness + copycat attribution).
    prs = json.loads(gh(["pr", "list", "-R", args.repo, "--state", "open",
                         "--json", "number,headRefName,headRefOid,title,isCrossRepository,labels,isDraft,mergeable,updatedAt"]).stdout or "[]")
    prs.sort(key=lambda p: p["number"])
    only_set = _parse_only_prs(args.only_pr, args.only_prs)
    targeted = bool(only_set)
    if not targeted:
        auto_closed = run_poll_auto_closes(args.repo, dry_run=args.dry_run)
        if auto_closed:
            prs = [p for p in prs if p["number"] not in auto_closed]
    if only_set:
        prs = [p for p in prs if p["number"] in only_set]
        if not prs:
            print(f"PR(s) {sorted(only_set)} not open"); return
    if not prs:
        print("no open PRs"); return

    # Fingerprint open PRs only — copycat comparison is against still-open earlier PRs
    # (closed/merged are excluded; see eval/copycat_policy.py COPYCAT_REFERENCE_STATE).
    ref_prs = list_reference_prs(args.repo)
    all_nums = sorted(p["number"] for p in ref_prs)
    pr_author = {p["number"]: (p.get("author") or {}).get("login", "?") for p in ref_prs}
    fps = {n: pr_fingerprint(args.repo, n) for n in all_nums}
    copy_log = load_copycat_log()
    logged_blocked = {e["pr"] for e in copy_log if e.get("blocked", True)}
    logged_any = {e["pr"] for e in copy_log}
    cleared_copycats = {e["pr"] for e in copy_log if e.get("blocked") is False}
    state_changed = False

    def find_copycat_match(num):
        """Best earlier different-author match with containment >= COPYCAT_WARN, else None."""
        files, added = fps.get(num, (set(), set()))
        if not added:
            return None, 0.0
        me = pr_author.get(num, "?")
        best_orig = None
        best_c = 0.0
        for earlier in all_nums:
            if earlier >= num:
                break
            ea_login = pr_author.get(earlier, "?")
            if ea_login == me:
                continue
            if ea_login.lower() in denylist:
                continue
            if earlier in logged_blocked:
                continue
            ef, ea = fps.get(earlier, (set(), set()))
            if not (files & ef):
                continue
            c = containment(added, ea)
            if c >= COPYCAT_WARN and c > best_c:
                best_c = c
                best_orig = earlier
        return best_orig, best_c

    # Collect PRs that actually need evaluation before starting the GPU instance.
    denylist = load_denylist()
    pending = []
    for pr in prs:
        num, branch, oid = pr["number"], pr["headRefName"], pr["headRefOid"][:7]
        ref = f"pull/{num}/head" if pr.get("isCrossRepository") else branch
        # Gate 0 — draft PRs are work-in-progress: never evaluate them. Skip entirely (no greenlight,
        # no labels). The bot picks them up once they're marked "Ready for review".
        if pr.get("isDraft"):
            print(f"PR #{num}: draft — skip (not evaluated until marked ready for review)")
            continue
        # Gate 1 — blocked contributor: never spend GPU on a flagged/sybil PR.
        hits = pr_involved_logins(args.repo, num) & denylist
        if hits:
            print(f"PR #{num}: BLOCKED (denylisted: {', '.join(sorted(hits))}) — flag + close, no eval")
            if not args.dry_run: close_blocked_pr(args.repo, num, hits)
            continue
        # Gate 2 — copycat: tiered containment vs earlier open PRs (not closed/merged).
        pr_labels = {l["name"] for l in pr.get("labels", [])}
        if COPYCAT_CLEARED_LABEL in pr_labels or num in cleared_copycats:
            print(f"PR #{num}: copycat-cleared — skip copycat gate")
        else:
            original, copy_c = find_copycat_match(num)
            if original is not None:
                author = pr_author.get(num, "?")
                _, added = fps.get(num, (set(), set()))
                if skip_copycat_scoring(added, copy_c):
                    print(f"PR #{num}: copycat-like #{original} at {copy_c:.0%} but too few added lines — allow eval")
                elif copy_c >= COPYCAT_BLOCK:
                    print(f"PR #{num}: COPYCAT ≥85% of #{original} by {pr_author.get(original,'?')} "
                          f"(author {author}) — block, no eval")
                    if not args.dry_run and num not in logged_any:
                        flag_copycat(args.repo, num, original, author)
                        copy_log.append({"pr": num, "author": author, "original": original,
                                         "date": datetime.date.today().isoformat(), "blocked": True})
                        logged_blocked.add(num); logged_any.add(num); state_changed = True
                        if author.lower() not in load_denylist():
                            block_account(author, f"#{num} ≥85% copycat of #{original} ({copy_c:.0%})")
                            close_blocked_pr(args.repo, num, {author})
                    continue
                else:
                    print(f"PR #{num}: COPYCAT WARN {copy_c:.0%} of #{original} by {pr_author.get(original,'?')} "
                          f"(author {author}) — warn, skip eval")
                    if not args.dry_run and num not in logged_any:
                        warn_strikes = sum(1 for e in copy_log
                                           if e.get("author") == author and not e.get("blocked", True)
                                           and int(e.get("penalty_days", PENALTY_DAYS)) != 0)
                        strike = warn_strikes + 1
                        will_block = warn_copycat(args.repo, num, original, author, strike, copy_c)
                        copy_log.append({"pr": num, "author": author, "original": original,
                                         "date": datetime.date.today().isoformat(), "blocked": False,
                                         "strike": strike, "containment": round(copy_c, 3)})
                        logged_any.add(num); state_changed = True
                        if will_block and author.lower() not in load_denylist():
                            block_account(author, f"{MAX_WARNINGS} copycat strikes: #{num} (vs #{original})")
                            close_blocked_pr(args.repo, num, {author})
                    continue
        areas = areas_for_pr(args.repo, num)
        print(f"PR #{num} @ {oid}: areas={sorted(areas) or ['(none)']} ref={ref}")
        if not args.dry_run: apply_area_labels(args.repo, num, areas)
        if not args.reeval and oid in evaluated_commits(args.repo, num):
            print(f"PR #{num} @ {oid}: already evaluated — skip eval"); continue
        # Gate 2.5 — copycat penalty: a copycat strike freezes the author's evaluations for
        # PENALTY_DAYS (from the first strike). During the window the bot does NOT greenlight any of
        # their PRs — it applies `penalty` and skips, instead of `test-on-5090`.
        pen_until = author_penalty_until(pr_author.get(num, "?"))
        if pen_until:
            print(f"PR #{num}: author {pr_author.get(num,'?')} under copycat penalty until {pen_until} "
                  f"— {PENALTY_LABEL}, skip eval")
            if not args.dry_run:
                cur = {l["name"] for l in pr.get("labels", [])}
                if PENALTY_LABEL not in cur: add_label(args.repo, num, PENALTY_LABEL)
                for L in (EVAL_GATE_LABEL, NOT_TESTED_LABEL, NEEDS_BENCH_LABEL):
                    if L in cur: remove_label(args.repo, num, L)
            continue
        # Gate 2.6 — merge conflict: don't spend GPU until the branch rebases cleanly onto main.
        if pr_merge_conflict(pr.get("mergeable")):
            print(f"PR #{num}: merge conflict with base — {NEEDS_REBASE_LABEL}, skip eval")
            if not args.dry_run:
                cur = {l["name"] for l in pr.get("labels", [])}
                if NEEDS_REBASE_LABEL not in cur:
                    add_label(args.repo, num, NEEDS_REBASE_LABEL)
                    post_merge_conflict_comment(args.repo, num)
                if EVAL_GATE_LABEL in cur:
                    remove_label(args.repo, num, EVAL_GATE_LABEL)
            continue
        # Gate 3 — greenlight (proof-gated): evaluate only if the PR ticks the RTX-5090 box AND
        # claims a real decode and/or Qwen3.5 prefill improvement in the template tables.
        # Reconcile labels each poll so a stale test-on-5090 can't keep a no-benchmark PR in the queue.
        pr_labels = {l["name"] for l in pr.get("labels", [])}
        def _reconcile(keep, drop):
            if args.dry_run: return
            if keep not in pr_labels: add_label(args.repo, num, keep)
            for L in drop:
                if L in pr_labels: remove_label(args.repo, num, L)
        status, reason = greenlight_status(args.repo, num, pr_labels)
        if targeted:
            print(f"PR #{num}: maintainer-targeted eval (greenlight: {status} — {reason})")
            pending.append((pr, num, branch, oid, ref, areas))
        elif status == "ok":
            print(f"PR #{num}: greenlit ({reason})")
            _reconcile(EVAL_GATE_LABEL, [NOT_TESTED_LABEL, NEEDS_BENCH_LABEL])
            pending.append((pr, num, branch, oid, ref, areas))
        elif status == "no-bench":
            print(f"PR #{num}: NOT greenlit ({reason}) — needs-benchmark, skip eval")
            first_time = NEEDS_BENCH_LABEL not in pr_labels
            _reconcile(NEEDS_BENCH_LABEL, [EVAL_GATE_LABEL, NOT_TESTED_LABEL])
            if first_time and not args.dry_run: post_needs_bench_comment(args.repo, num)
        else:  # unchecked
            print(f"PR #{num}: not greenlit ({reason}) — close (RTX 5090 unchecked)")
            if not args.dry_run:
                close_rtx5090_unchecked_pr(args.repo, num)
            continue

    if not args.dry_run and state_changed:
        save_copycat_log(copy_log)
        push_github_state("eval: record copycat detections + any auto-blocks")

    if not pending:
        # No new commits to grade, but still run the merge workflow: auto-merge a standing
        # `merge-first` winner from a previous round and flag rivals of a just-merged winner.
        if not args.dry_run:
            reconcile_merge_labels(args.repo)
        print("done — no merges (manual)."); return

    if args.dry_run:
        print("--- dry-run: would evaluate (oldest-first): " +
              ", ".join(f"#{n}" for _, n, *_ in pending)); return

    # Reuse the pinned stable box first (cached model, good download speed). Skip when on bare metal.
    if PINNED_INSTANCE and not ssh_box_enabled():
        with open(INSTANCE_FILE, "w") as f: f.write(PINNED_INSTANCE)

    # --- Same-box baseline (once per bot run; reused for every PR in pending) ----------------------
    base_iid = current_instance(args.instance) if args.instance else 0
    box_id = _baseline_box_id(base_iid)
    main_commit = _origin_main_short()
    bres = {}
    cache = _load_baseline_cache(box_id)
    cache_ok = cache and _baseline_cache_valid(cache, args.bidir, QWEN36_BASE, QWYTHOS_BASE, main_commit)
    if cache_ok:
        if args.bidir:
            QWEN36_BASE.update(cache["q36"])
            QWYTHOS_BASE.update(cache["q35"])
        bres = cache["bres"]
        # Re-derive pp from bres so older caches that zeroed all q36 pp (128_pp=0 abort) self-heal.
        if args.bidir and bres:
            _apply_bidir_ctx_from_bres(bres, QWEN36_BASE, QWYTHOS_BASE)
        print(f">> reusing cached same-box baseline from {cache['ts']} ({box_id}, main={main_commit or '?'})")
    elif args.skip_baseline:
        print(f">> --skip-baseline: no valid cache for {box_id} — aborting")
        return
    else:
        if cache and main_commit:
            cached_commit = (cache.get("bres") or {}).get("commit")
            if (cached_commit and cached_commit != main_commit
                    and _bench_harness_changed_between(cached_commit, main_commit)):
                print(f">> baseline cache stale (bench/scripts changed {cached_commit}..{main_commit}) "
                      f"— remeasuring")
        bcmd = [sys.executable, os.path.join(HERE, "vast_eval.py"),
                *_vast_eval_transport_args(args.instance),
                "--ref", "origin/main", "--frontier", "0", "--ceiling", str(args.ceiling),
                "--eval-mode", "longctx"]
        if args.bidir:
            bcmd += ["--bidir", "--primary-quant", args.primary_quant, "--baseline-only"]
        if PINNED_INSTANCE and not ssh_box_enabled() and str(base_iid) == PINNED_INSTANCE:
            bcmd.append("--pinned")
        box_label = ssh_box_arg() if ssh_box_enabled() else f"instance {base_iid}"
        print(f">> measuring same-box baseline (origin/main) on {box_label} ...")
        br = subprocess.run(bcmd, cwd=ROOT, capture_output=True, text=True, timeout=14400)
        if br.returncode == PINNED_RETRY_RC:
            tail = next((l for l in reversed((br.stdout + br.stderr).splitlines()) if l.strip()), "")
            print(f">> {tail}\n>> aborting this run — next scheduled run retries the pinned box."); return
        for l in br.stdout.splitlines():
            if l.startswith("NEW_INSTANCE_ID "):
                try:
                    nid = int(l.split()[1])
                    with open(INSTANCE_FILE, "w") as f: f.write(str(nid))
                    print(f"  (instance updated to {nid})")
                except Exception: pass
        bline = next((l for l in br.stdout.splitlines() if l.startswith("RESULT_JSON")), None)
        bres = json.loads(bline[len("RESULT_JSON "):]) if bline else {}
        if not bres.get("label"):
            log = (br.stdout + br.stderr)[-1200:]
            print(f">> same-box baseline (origin/main) failed ({bres.get('label','no result')}) — "
                  f"aborting; no PRs graded.\n{log}"); return
        if args.bidir and not (bres.get("pass") or bres.get("score_qwen35") or bres.get("score_qwen36")):
            log = (br.stdout + br.stderr)[-1200:]
            print(f">> bidir baseline (origin/main) failed — aborting; no PRs graded.\n{log}"); return
        if args.bidir:
            if not _apply_bidir_ctx_from_bres(bres, QWEN36_BASE, QWYTHOS_BASE) or not _bidir_baseline_sane(QWEN36_BASE, QWYTHOS_BASE):
                log = (br.stdout + br.stderr)[-1200:]
                print(f">> bidir baseline measurement invalid — aborting; no PRs graded.\n{log}"); return
        if not args.bidir and (not bres.get("pass") or not bres.get("tps")):
            log = (br.stdout + br.stderr)[-1200:]
            print(f">> same-box baseline (origin/main) failed — aborting; no PRs graded.\n{log}"); return
        _save_baseline_cache(box_id, dict(QWEN36_BASE), dict(QWYTHOS_BASE), bres)
    if args.bidir:
        run_baseline = float(QWEN36_BASE["128"])
        run_guard_128 = float(QWEN36_BASE["128"])
        run_guard_512 = float(QWEN36_BASE["512"])
        run_guard_4k = float(QWEN36_BASE["4k"])
        run_guard_16k = float(QWEN36_BASE["16k"])
        run_guard_32k = float(QWEN36_BASE["32k"])
        score_ctx = 128
    else:
        run_baseline = bres["tps"]
        run_guard_128 = float(bres.get("ctx_128_tps") or bres.get("tps") or 0)
        run_guard_512 = float(bres.get("ctx_512_tps") or 0)
        run_guard_4k = float(bres.get("ctx_4096_tps") or 0)
        run_guard_16k = float(bres.get("ctx_16384_tps") or bres.get("tps") or 0)
        run_guard_32k = float(bres.get("ctx_32768_tps") or 0)
        score_ctx = int(bres.get("score_context") or 128)
    if score_ctx == 128:
        print(f">> same-box baseline: origin/main = {run_baseline} tok/s on this box")
    else:
        print(f">> same-box baseline: origin/main contexts: "
              f"128={run_guard_128} tok/s; 512={run_guard_512} tok/s; "
              f"4k={run_guard_4k} tok/s; 16k={run_guard_16k} tok/s; "
              f"32k={run_guard_32k} tok/s")
    # Sanity guard: origin/main IS the merged frontier code, so on a healthy box it should measure
    # within ~10% of the known frontier. A baseline well below that means the box is cold/throttling
    # or degraded — grading PRs against it inflates every delta (the cold-clock artifact that once
    # mislabeled minor PRs as XL above the ceiling). Abort rather than post bogus labels.
    SANITY_FRAC = float(os.environ.get("SPARKINFER_BASELINE_SANITY", "0.90"))
    # Dashboard frontier is currently the 128-token headline. Long-context eval establishes a fresh
    # same-box 16k frontier every run, so do not compare the 16k baseline to the 128-token dashboard.
    known_frontier = float(args.frontier or 0) if score_ctx == 128 else 0
    if known_frontier > 0 and run_baseline < SANITY_FRAC * known_frontier:
        print(f">> baseline {run_baseline} < {SANITY_FRAC:.0%} of known frontier {known_frontier} "
              f"(= {SANITY_FRAC*known_frontier:.1f}) — box underperforming (cold/throttling/degraded). "
              f"Aborting; NO PRs graded. Re-run on a warm, stable box.")
        return

    # Bidir: ctx speeds already in QWEN36_BASE / QWYTHOS_BASE from baseline RESULT_JSON.
    if args.bidir:
        print(f"  Qwen3.6 same-box main: 128={QWEN36_BASE['128']} 512={QWEN36_BASE['512']} "
              f"4k={QWEN36_BASE['4k']} 16k={QWEN36_BASE['16k']} 32k={QWEN36_BASE['32k']} tok/s")
        if any(QWEN36_BASE.get(k, 0) for k in ("128_pp", "512_pp", "4k_pp", "16k_pp", "32k_pp")):
            print(f"  Qwen3.6 prefill pp: 128={QWEN36_BASE.get('128_pp', 0)} "
                  f"512={QWEN36_BASE.get('512_pp', 0)} 4k={QWEN36_BASE.get('4k_pp', 0)} "
                  f"16k={QWEN36_BASE.get('16k_pp', 0)} 32k={QWEN36_BASE.get('32k_pp', 0)} pp tok/s")
        print(f"  Qwythos ({args.primary_quant}) same-box main: "
              f"128={QWYTHOS_BASE['128']} 4k={QWYTHOS_BASE['4k']} "
              f"32k={QWYTHOS_BASE['32k']} 64k={QWYTHOS_BASE['64k']} tok/s")
        if any(QWYTHOS_BASE.get(k, 0) for k in ("4k_pp", "32k_pp", "64k_pp", "128k_pp")):
            print(f"  Qwythos prefill pp: 4k={QWYTHOS_BASE.get('4k_pp', 0)} "
                  f"32k={QWYTHOS_BASE.get('32k_pp', 0)} 64k={QWYTHOS_BASE.get('64k_pp', 0)} "
                  f"128k={QWYTHOS_BASE.get('128k_pp', 0)} pp tok/s")

    # Run all pending evals on the same vast instance (left running between PRs and after the run).
    for i, (pr, num, branch, oid, ref, areas) in enumerate(pending):
        # Grade against the same-box baseline = MERGED origin/main (measured above). Every PR in the
        # run is graded against main, NOT against other PRs in the run — #67 and #70 are independent
        # branches off main, so each must get its own gain over main (the old within-run ratchet made
        # whichever ran second look like "none"). The frontier advances when you MERGE; to see if two
        # optimizations STACK, re-evaluate the second after merging the first. Literal duplicates are
        # caught by copycat detection; emission only pays MERGED PRs, so the maintainer's merge choice
        # (not eval order) decides what counts.
        cur_iid = current_instance(args.instance) if args.instance else 0
        cmd = [sys.executable, os.path.join(HERE, "vast_eval.py"),
               *_vast_eval_transport_args(args.instance),
               "--ref", ref,
               "--frontier", "0", "--ceiling", str(args.ceiling),
               "--eval-mode", "longctx", "--guard-128-baseline", str(run_guard_128),
               "--guard-512-baseline", str(run_guard_512),
               "--guard-4k-baseline", str(run_guard_4k),
               "--guard-16k-baseline", str(run_guard_16k),
               "--guard-32k-baseline", str(run_guard_32k)]
        if args.bidir:
            cmd += ["--bidir", "--primary-quant", args.primary_quant] + _bidir_baseline_args(QWEN36_BASE, QWYTHOS_BASE)
        if PINNED_INSTANCE and not ssh_box_enabled() and str(cur_iid) == PINNED_INSTANCE:
            cmd.append("--pinned")
        if args.polaris:
            cmd.append("--polaris")
        pinned = "--pinned" in cmd
        box_label = ssh_box_arg() if ssh_box_enabled() else f"instance {cur_iid}"
        print(f"PR #{num} @ {oid}: evaluating '{ref}' (vs same-box main) on {box_label}"
              f"{' [pinned]' if pinned else ''} ...")
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=14400)
        if r.returncode == PINNED_RETRY_RC:
            tail = next((l for l in reversed((r.stdout + r.stderr).splitlines()) if l.strip()), "")
            print(f">> {tail}\n>> aborting this run — the next scheduled run retries the "
                  f"pinned box. No PRs evaluated this tick."); return
        # Legacy: --allow-provision may emit NEW_INSTANCE_ID (auto-rent is off by default).
        for l in r.stdout.splitlines():
            if l.startswith("NEW_INSTANCE_ID "):
                try:
                    new_id = int(l.split()[1])
                    with open(INSTANCE_FILE, "w") as f: f.write(str(new_id))
                    print(f"  (instance updated to {new_id})")
                except Exception: pass
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT_JSON")), None)
        if not line:
            log = (r.stdout + r.stderr)[-1500:]
            print(f"PR #{num}: eval produced no result\n{log}")
            body = (f"<!-- sparkinfer-eval-error:{oid} -->\n⚠️ **sparkinfer auto-eval errored** for `{oid}` "
                    f"— re-run manually.\n\n<details><summary>log tail</summary>\n\n```\n{log}\n```\n</details>")
            res, label = None, None
        else:
            res = json.loads(line[len("RESULT_JSON "):])
            label = _public_eval_label(res)
            # Keep RESULT_JSON fields consistent with the public headline (bidir either-side REJECT).
            if label and res.get("label") != label:
                res["label"] = label
                if label == "REJECT":
                    res["pass"] = False
            body = render(res, oid)
            print(f"PR #{num}: {json.dumps(res)}")

            # --- Polaris: parse unsigned attestation from eval box, attest it, upload with eval log ---
            polaris_line = next((l for l in r.stdout.splitlines()
                                 if l.startswith("POLARIS_ATTESTATION ")), None)
            polaris_bundle = None
            if polaris_line and res:
                try:
                    attestation = json.loads(polaris_line[len("POLARIS_ATTESTATION "):])
                    receipt = build_polaris_receipt_from_attestation(
                        attestation,
                        api_key=POLARIS_API_KEY,
                        privkey=POLARIS_PRIVKEY,
                        pubkey=POLARIS_PUBKEY,
                    )
                    if receipt:
                        polaris_bundle = {"receipt": receipt, "attestation": attestation}
                        res["polaris_receipt_hash"] = receipt["receipt_id"][:16]
                    else:
                        print(">> Polaris attestation collected but NOT attested (no key configured)")
                except Exception as e:
                    import traceback
                    print(f">> Polaris receipt failed: {e}")
                    traceback.print_exc()
        if args.dry_run:
            print("--- dry-run, not posting ---\n" + body); continue
        if label:
            cur = labels_on(args.repo, num)
            for lab in {l for l in cur if l.startswith("eval:") or l.startswith("eval-qwen")
                        or l.startswith("eval-prefill:")}:
                remove_label(args.repo, num, lab)
            add_label(args.repo, num, f"eval:{label}")
            if res and res.get("mode") == "bidir" and not res.get("infra_error"):
                if res.get("label_qwen35"):
                    add_label(args.repo, num, f"eval-qwen35:{res['label_qwen35']}")
                if res.get("label_qwen36"):
                    add_label(args.repo, num, f"eval-qwen36:{res['label_qwen36']}")
                # Annotate verified prefill progress even when headline is REJECT (correctness).
                prefill_tiers = []
                for block in (res.get("score_qwen35") or {}, res.get("score_qwen36") or {}, res):
                    pl = block.get("prefill_label")
                    if pl in SPEEDUP_LABELS:
                        prefill_tiers.append(pl)
                if prefill_tiers:
                    best_pf = max(prefill_tiers, key=lambda t: (
                        {"XL": 6, "L": 5, "M": 4, "S": 3, "XS": 2}.get(t, 0)))
                    add_label(args.repo, num, f"eval-prefill:{best_pf}")
            elif res and res.get("infra_error"):
                add_label(args.repo, num, REEVALUATE_LABEL)
            apply_context_label(args.repo, num, cur, res.get("best_context_label"))
            apply_regression_labels(args.repo, num, cur, res.get("regression_labels"))
            # This was just graded against the CURRENT main, so it's no longer stale: clear
            # needs-rebase. If it carried needs-rebase, it was a post-rebase re-eval → tag re-evaluate.
            if NEEDS_REBASE_LABEL in cur:
                remove_label(args.repo, num, NEEDS_REBASE_LABEL)
                add_label(args.repo, num, REEVALUATE_LABEL)
        gh(["pr", "comment", str(num), "-R", args.repo, "--body", body])
        print(f"PR #{num}: posted {'eval:'+label if label else 'error'} — NOT merged.")
        if label in FAIL_VERDICT_LABELS and not targeted:
            maybe_close_exhausted_pr(args.repo, num, dry_run=args.dry_run)
        if res:
            proof = upload_eval_log(args.repo, num, pr.get("title", ""), oid, res,
                                    r.stdout + r.stderr, run_baseline, polaris=polaris_bundle)
            if proof and polaris_bundle:
                res["polaris_receipt_url"] = proof
            update_dashboard(args.repo, pr, areas, res, proof_url=proof)
            # auto-close on REJECT is DISABLED — merge-first is the only automated action.
            # Rejected PRs stay open so authors can rebase and re-submit.
        # NB: run_baseline is NOT ratcheted here — every PR is graded against merged origin/main, so
        # independent optimizations each get their true gain (the frontier advances on MERGE, not eval).

    # Re-scan exhausted PRs after grading — a none/REJECT posted this round may cross the limit.
    if not targeted and not args.dry_run:
        run_poll_auto_closes(args.repo)

    # Per-round merge workflow: among the PRs graded this round, label the biggest verified speedup
    # `merge-first` and the rest `needs-rebase`; if a prior winner merged, flag its rivals `re-evaluate`.
    if not args.dry_run:
        reconcile_merge_labels(args.repo)

    print("done — no merges (manual).")

if __name__ == "__main__":
    main()
