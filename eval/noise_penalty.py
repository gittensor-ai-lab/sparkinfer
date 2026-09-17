#!/usr/bin/env python3
"""Noise ban: park a noisy account's eval tiers for 3 days, then put them back.

Policy (list: .github/noise-ban-list.txt, CI: .github/workflows/noise-penalty.yml)
--------------------------------------------------------------------------------
An account listed for noise has every eval verdict on its PRs suspended for BAN_DAYS days:

    eval:XL  ->  eval:XL-p        eval-qwen38:M  ->  eval-qwen38:M-p
    eval:none                     (exempt, every family)

SN74 scoring reads `eval:*`, so a parked tier earns nothing while the ban is active. When the
window closes the `-p` comes off and the original tier is back.

Why rename instead of delete: the tier is still spelled out in the parked label, so the restore
is exact and needs no state file to trust. A ban costs the account 3 days of emissions; it never
costs them a verdict the eval actually measured, and it never re-runs a GPU to get it back.

Why `:none` is exempt: it is worth nothing already, and it is how the bots record "evaluated, no
speedup". Parking it would delete that record and buy nothing.

This is deliberately NOT the eval-gaming denylist (.github/blocked-contributors.txt), which is
permanent, closes the PR, and refuses to evaluate it. Noise is a timeout, not an expulsion —
nothing here closes, comments on, or re-scores a PR.

Runs
----
    python3 eval/noise_penalty.py --check          # validate the list, touch nothing (CI)
    python3 eval/noise_penalty.py --dry-run        # print the label edits, make none
    python3 eval/noise_penalty.py                  # apply

Every run is idempotent and self-correcting: it re-derives the whole desired state from the list
plus today's date, so a missed run (or ten) costs nothing but latency, and a line deleted from
the list early still gets its labels restored by the orphan sweep.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

BAN_LIST_FILE = os.path.join(ROOT, ".github", "noise-ban-list.txt")
DEFAULT_REPO = "gittensor-ai-lab/sparkinfer"

BAN_DAYS = 3                 # a ban starting on D is active on D, D+1, D+2 and lifts on D+3
PENALTY_SUFFIX = "-p"
EXEMPT_TIERS = {"none"}      # `eval:none` and its per-family twins are never parked
PENALTY_COLOR = "6A737D"

# Every eval verdict family the bots apply: eval:, eval-prefill:, eval-dflash:,
# eval-museglimmer:, eval-qwen38: (see eval/setup_labels.sh). Matching the shape rather than a
# hard-coded list means a new bot's family is covered the day it ships.
EVAL_LABEL_RE = re.compile(r"^(eval(?:-[a-z0-9]+)?):(.+)$")

_GH_TRANSIENT_RE = re.compile(
    r"\b(502|503|504|timeout|timed out|connection reset|EOF|temporarily unavailable|"
    r"rate limit|secondary rate)\b", re.I)


def gh(args, retries=4, quiet=False):
    """Run `gh`, retrying transient GitHub failures, warning (never silent) on a real one."""
    delay, last = 3, None
    for attempt in range(1, max(1, retries) + 1):
        last = subprocess.run(["gh"] + args, capture_output=True, text=True)
        if last.returncode == 0:
            return last
        blob = (last.stderr or "") + (last.stdout or "")
        if attempt == retries or not _GH_TRANSIENT_RE.search(blob):
            break
        time.sleep(delay)
        delay = min(delay * 2, 30)
    if last is not None and last.returncode != 0 and not quiet:
        first = ((last.stderr or last.stdout or "").strip().splitlines() or [""])[0]
        print(f">> WARN: gh {' '.join(args[:3])} failed (rc={last.returncode}): {first[:200]}",
              file=sys.stderr)
    return last


# ---- label arithmetic (pure) ----

def penalty_name(label):
    """`eval:XL` -> `eval:XL-p`. None if the label is not a parkable eval verdict.

    Not parkable: anything outside the eval families, an exempt tier (`:none`), and a label that
    is already parked — so applying a ban twice is a no-op rather than `eval:XL-p-p`.
    """
    m = EVAL_LABEL_RE.match(label or "")
    if not m:
        return None
    tier = m.group(2)
    if tier in EXEMPT_TIERS or tier.endswith(PENALTY_SUFFIX):
        return None
    return f"{m.group(1)}:{tier}{PENALTY_SUFFIX}"


def original_name(label):
    """`eval:XL-p` -> `eval:XL`. None if the label is not a parked eval verdict."""
    m = EVAL_LABEL_RE.match(label or "")
    if not m:
        return None
    tier = m.group(2)
    if not tier.endswith(PENALTY_SUFFIX):
        return None
    tier = tier[: -len(PENALTY_SUFFIX)]
    if not tier or tier in EXEMPT_TIERS:
        return None
    return f"{m.group(1)}:{tier}"


def plan_labels(labels, penalize):
    """The (add, remove) label edit that parks (or restores) one PR's verdicts.

    Returns empty lists when there is nothing to do, which is how callers skip the API call.
    """
    add, remove = [], []
    for label in labels:
        want = penalty_name(label) if penalize else original_name(label)
        if not want or want in add:
            continue
        add.append(want)
        remove.append(label)
    return add, remove


# ---- ban list (pure) ----

def parse_ban_list(text):
    """Parse the ban list into (entries, problems).

    An entry is {login, start: date, reason, line}. Anything malformed lands in `problems` and is
    skipped rather than guessed at: a line with no date could only mean "banned from now", which
    would re-arm every run and never lift.
    """
    entries, problems, seen = [], [], {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        body = raw.split("#", 1)[0].strip()
        if not body:
            continue
        parts = body.split()
        login = parts[0].lower()
        reason = raw.split("#", 1)[1].strip() if "#" in raw else ""
        if len(parts) < 2:
            problems.append(f"line {lineno}: `{login}` has no ban-start date "
                            f"(expected `<login> YYYY-MM-DD`)")
            continue
        try:
            start = datetime.date.fromisoformat(parts[1])
        except ValueError:
            problems.append(f"line {lineno}: `{login}` has an unparseable ban-start date "
                            f"{parts[1]!r} (expected YYYY-MM-DD)")
            continue
        if len(parts) > 2:
            problems.append(f"line {lineno}: `{login}` has trailing text "
                            f"{' '.join(parts[2:])!r} — put notes after a `#`")
            continue
        if login in seen:
            problems.append(f"line {lineno}: `{login}` is listed twice (first at line "
                            f"{seen[login]}) — move the date instead of adding a second ban")
            continue
        seen[login] = lineno
        entries.append({"login": login, "start": start, "reason": reason, "line": lineno})
    return entries, problems


def load_ban_list(path=BAN_LIST_FILE):
    try:
        with open(path) as f:
            return parse_ban_list(f.read())
    except FileNotFoundError:
        return [], [f"ban list not found: {path}"]


def lift_date(start):
    return start + datetime.timedelta(days=BAN_DAYS)


def ban_status(entry, today):
    """('active'|'pending'|'expired', lift_date) for this entry as of `today` (UTC)."""
    lift = lift_date(entry["start"])
    if today < entry["start"]:
        return "pending", lift
    if today < lift:
        return "active", lift
    return "expired", lift


# ---- GitHub side ----

def author_prs(repo, login, limit):
    out = gh(["pr", "list", "-R", repo, "--author", login, "--state", "all",
              "--limit", str(limit), "--json", "number,labels,author"])
    if out is None or out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout or "[]")
    except json.JSONDecodeError:
        return None


def prs_with_label(repo, label, limit):
    out = gh(["pr", "list", "-R", repo, "--label", label, "--state", "all",
              "--limit", str(limit), "--json", "number,labels,author"])
    if out is None or out.returncode != 0:
        return []
    try:
        return json.loads(out.stdout or "[]")
    except json.JSONDecodeError:
        return []


def repo_penalty_labels(repo):
    """Parked labels that actually exist in the repo.

    They are created on demand, so this stays small (only tiers a ban has really touched) and
    shrinks to nothing in a repo that has never used one.
    """
    owner_repo = repo
    out = gh(["api", f"repos/{owner_repo}/labels", "--paginate", "--jq", ".[].name"], quiet=True)
    if out is None or out.returncode != 0:
        return []
    return [n.strip() for n in (out.stdout or "").splitlines()
            if n.strip() and original_name(n.strip())]


def label_names(pr):
    return [l.get("name", "") for l in (pr.get("labels") or [])]


def pr_author(pr):
    return ((pr.get("author") or {}).get("login") or "").lower()


def ensure_label(repo, name, dry_run):
    """Upsert a parked label. `--force` makes this idempotent, so it is safe to call every time."""
    if dry_run:
        return
    restored = original_name(name) or name
    gh(["label", "create", name, "-R", repo, "--color", PENALTY_COLOR, "--force",
        "--description", f"eval verdict parked by a 3-day noise ban — restores to {restored}"],
       quiet=True)


def edit_pr(repo, num, add, remove, dry_run):
    for name in add:
        ensure_label(repo, name, dry_run)
    cmd = ["pr", "edit", str(num), "-R", repo]
    for name in add:
        cmd += ["--add-label", name]
    for name in remove:
        cmd += ["--remove-label", name]
    if dry_run:
        return True
    out = gh(cmd)
    return out is not None and out.returncode == 0


def process_pr(repo, pr, penalize, dry_run, verb):
    add, remove = plan_labels(label_names(pr), penalize)
    if not add:
        return 0
    num = pr.get("number")
    print(f"   PR #{num}: {verb} {', '.join(remove)} -> {', '.join(add)}")
    return 1 if edit_pr(repo, num, add, remove, dry_run) else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Apply and lift 3-day noise-ban eval penalties.")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO)
    ap.add_argument("--list", dest="list_file", default=BAN_LIST_FILE)
    ap.add_argument("--today", default=None,
                    help="override today's UTC date (YYYY-MM-DD); for testing")
    # Deliberately generous: emissions are scored from `eval:*` on MERGED PRs (.gittensor/
    # weights.json), so a prolific account's history is all in scope. Truncating here would
    # under-park on the way in and, worse, strand labels at `-p` on the way out.
    ap.add_argument("--limit", type=int, default=1000, help="max PRs inspected per query")
    ap.add_argument("--dry-run", action="store_true", help="print the edits, make none")
    ap.add_argument("--check", action="store_true",
                    help="validate the ban list and exit; no GitHub calls")
    args = ap.parse_args(argv)

    entries, problems = load_ban_list(args.list_file)
    for p in problems:
        print(f">> ERROR: ban list: {p}", file=sys.stderr)

    today = (datetime.date.fromisoformat(args.today) if args.today
             else datetime.datetime.now(datetime.timezone.utc).date())

    if args.check:
        print(f"ban list: {args.list_file} — {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}")
        for e in entries:
            state, lift = ban_status(e, today)
            print(f"  {e['login']:<24} start {e['start']}  lift {lift}  [{state}]"
                  + (f"  # {e['reason']}" if e["reason"] else ""))
        if problems:
            print(f">> {len(problems)} problem(s) — fix the list", file=sys.stderr)
            return 1
        return 0

    # A malformed list is never enforced half-way: a typo in one line must not silently leave
    # another account's tiers parked past its window.
    if problems:
        print(">> refusing to enforce a malformed ban list", file=sys.stderr)
        return 1

    active, inactive = {}, []
    for e in entries:
        state, lift = ban_status(e, today)
        if state == "active":
            active[e["login"]] = lift
        else:
            inactive.append((e, state))

    print(f"noise penalty on {args.repo} — {today} (UTC), {len(active)} active ban(s)"
          + (" [dry run]" if args.dry_run else ""))

    edits = 0
    for login, lift in sorted(active.items()):
        prs = author_prs(args.repo, login, args.limit)
        if prs is None:
            print(f">> WARN: could not list PRs for {login}; leaving it alone this run",
                  file=sys.stderr)
            continue
        print(f" * {login}: ban active until {lift} — {len(prs)} PR(s)")
        for pr in prs:
            edits += process_pr(args.repo, pr, True, args.dry_run, "park")

    for e, state in inactive:
        prs = author_prs(args.repo, e["login"], args.limit)
        if prs is None:
            continue
        restorable = sum(1 for pr in prs if plan_labels(label_names(pr), False)[0])
        if restorable:
            print(f" * {e['login']}: ban {state} (lifted {lift_date(e['start'])}) — restoring")
        for pr in prs:
            edits += process_pr(args.repo, pr, False, args.dry_run, "restore")

    # Sweep: catch parked labels whose owner has left the list entirely (a maintainer deleted the
    # line early, or the PR was opened under a login the list no longer names). Without this, a
    # deleted line would strand a tier at `-p` forever.
    for label in repo_penalty_labels(args.repo):
        for pr in prs_with_label(args.repo, label, args.limit):
            if pr_author(pr) in active:
                continue
            edits += process_pr(args.repo, pr, False, args.dry_run, "restore (sweep)")

    print(f"done — {edits} PR label edit(s)" + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
