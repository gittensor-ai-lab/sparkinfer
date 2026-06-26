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
import argparse, datetime, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# vast_eval.py self-heals dead boxes (recreates them) and writes the working instance id here;
# prefer it over --instance so we reuse the recreated box instead of retrying the dead one.
INSTANCE_FILE = os.path.expanduser(os.environ.get("VAST_INSTANCE_FILE", "~/.sparkinfer_vast_instance"))
def current_instance(default):
    try: return int(open(INSTANCE_FILE).read().strip())
    except Exception: return default

# Pinned default box: a stable, known-good instance (cached model, good download speed) that we
# reuse first on every run and NEVER destroy. vast_eval.py is invoked with --pinned for it, so on
# bring-up failure it retries on later scheduled runs instead of provisioning immediately;
# only after VAST_REUSE_MAX_RETRIES misses does it spin up a new box (the pinned one is kept).
# Set VAST_DEFAULT_INSTANCE="" to disable the pin and always provision fresh.
# The pin id lives in a file so it can self-heal: when the pinned box is reclaimed and the eval
# provisions a fresh one, we re-pin to that fresh box (write its id here). Seed/override via
# VAST_DEFAULT_INSTANCE; set it to "" to disable pinning entirely (always provision fresh).
PIN_FILE = os.path.expanduser(os.environ.get("VAST_PIN_FILE", "~/.sparkinfer_pinned_instance"))
def _read_pin():
    try:
        v = open(PIN_FILE).read().strip()
        if v: return v
    except Exception: pass
    return os.environ.get("VAST_DEFAULT_INSTANCE", "42682383").strip()
def _write_pin(iid):
    try:
        with open(PIN_FILE, "w") as f: f.write(str(iid))
    except Exception: pass
PINNED_INSTANCE = _read_pin()
PINNED_RETRY_RC = 75   # must match vast_eval.PINNED_RETRY_RC

# Subsystem buckets for the deterministic area:<name> label (from a PR's top-level changed
# dirs — no AI). Categorization/display only: SN74 scoring is speedup-only (the eval:* tier),
# NOT a per-subsystem budget.
AREAS = {"kernels", "runtime", "moe", "bench"}

# RTX 5090 evaluation is OPT-IN *and* proof-gated. A PR is only evaluated if it ticks the
# "Tested on RTX 5090" box AND fills the decode before/after table with real numbers showing a
# clear improvement (after > before) — checking the box alone is not enough (it wasted GPU on PRs
# whose decode table was still the placeholder). States: greenlit -> test-on-5090 (eval); box
# ticked but no valid before<after -> needs-benchmark (skip + ask for numbers); box unticked ->
# not-tested (skip). There is NO manual override — every eval must pass the gate on real RTX 5090
# before/after numbers; nothing bypasses the benchmark check.
EVAL_GATE_LABEL  = "test-on-5090"     # bot-set marker: greenlit, will be evaluated
NOT_TESTED_LABEL = "not-tested"       # box not ticked
NEEDS_BENCH_LABEL = "needs-benchmark" # box ticked but decode before/after missing/invalid/no-gain
# Per-round merge workflow (all queued PRs graded vs the same-box main in one round):
MERGE_FIRST_LABEL  = "merge-first"    # the round's biggest verified speedup — merge this one first
NEEDS_REBASE_LABEL = "needs-rebase"   # also a verified speedup, but not the round winner
REEVALUATE_LABEL   = "re-evaluate"    # winner merged → rebase onto new main; bot re-evals on push
HOLD_LABEL         = "hold"           # maintainer override: never auto-merge this PR

# Auto-merge the round's merge-first winner — OFF unless SPARKINFER_AUTOMERGE=1. Heavily guarded:
# the eval only verifies speed + token-match, so auto-merge is gated on labels, author standing,
# changed paths, and branch protection (gh refuses if checks/reviews aren't satisfied).
AUTO_MERGE_FIRST = os.environ.get("SPARKINFER_AUTOMERGE", "0") == "1"
# Auto-merge is BLOCKED if the PR carries any of these labels:
AUTOMERGE_BLOCK_LABELS = {"copycat", "flagged:gaming", "penalty", "needs-benchmark", "not-tested",
                          NEEDS_REBASE_LABEL, REEVALUATE_LABEL, HOLD_LABEL}
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
# A PR is a copycat if its added lines are largely contained in an EARLIER PR touching the same
# file(s). Copycats are labeled `copycat`, commented (citing the original), and NOT evaluated.
# Logged to .github/copycats.json; a 2nd copycat by the same author auto-adds them to the denylist.
FLAG_FILE = os.path.join(ROOT, ".github", "FLAGGED.md")
COPYCAT_LABEL = "copycat"
COPYCAT_LOG = os.path.join(ROOT, ".github", "copycats.json")
COPYCAT_CONTAINMENT = 0.80   # ≥80% of the copy's added lines also appear in the original
COPYCAT_STRIKES = 2          # this many copycats by one author -> denylist (permanent block)
PENALTY_DAYS = 5             # every copycat strike freezes the author's evals for this long
PENALTY_LABEL = "penalty"    # applied to a penalized author's PRs instead of greenlighting them

def author_penalty_until(author):
    """If `author` has an active copycat strike, return the date the penalty lifts, else None.
    Each strike lifts at its date + its own `penalty_days` (default PENALTY_DAYS); a strike entry
    may override `penalty_days` for leniency (e.g. a first-time contributor's first mistake). The
    author is penalized until the latest lift date. Applies from the FIRST strike onward."""
    if not author: return None
    lifts = []
    for e in load_copycat_log():
        if str(e.get("author", "")).lower() == author.lower():
            try:
                d = datetime.date.fromisoformat(e["date"])
                days = int(e.get("penalty_days", PENALTY_DAYS))
                lifts.append(d + datetime.timedelta(days=days))
            except Exception: pass
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
            f"This PR re-submits substantially the same diff as the earlier #{original}. "
            f"Duplicating an existing PR's work does not earn a score — copycats are **not "
            f"evaluated or merged**.\n\nRepeated copycatting (≥{COPYCAT_STRIKES}) results in an "
            f"automatic block. See [`.github/COPYCATS.md`](../blob/main/.github/COPYCATS.md).")
    gh(["pr", "comment", str(num), "-R", repo, "--body", body])

def evaluated_commits(repo, num):
    r = gh(["pr", "view", str(num), "-R", repo, "--json", "comments"])
    done = set()
    for c in json.loads(r.stdout or "{}").get("comments", []):
        for m in re.finditer(r"<!-- sparkinfer-eval:([0-9a-f]+) -->", c.get("body", "")):
            done.add(m.group(1))
    return done

def areas_for_pr(repo, num):
    """Subsystems a PR touches, from its changed file paths (deterministic, no AI)."""
    files = json.loads(gh(["pr", "view", str(num), "-R", repo, "--json", "files"]).stdout or "{}").get("files", [])
    return {f["path"].split("/", 1)[0] for f in files} & set(AREAS)

def _decode_val(body, key):
    """Pull a numeric decode tok/s out of the template table row '| before|after | <n> |'.
    Returns the float, or None if the cell is the placeholder / non-numeric / absent."""
    for ln in body.splitlines():
        m = re.match(rf"\s*\|\s*{key}\b[^|]*\|\s*([^|]*?)\s*\|", ln, re.I)
        if not m: continue
        num = re.search(r"[-+]?\d+\.?\d*", m.group(1))
        try: return float(num.group(0)) if num else None
        except ValueError: return None
    return None

def greenlight_status(repo, num, pr_labels):
    """Decide whether a PR may be evaluated. Returns (status, reason):
      'ok'        — greenlit: box ticked + real before<after decode numbers (no override exists)
      'no-bench'  — box ticked but the decode before/after table is missing/placeholder/no-gain
      'unchecked' — the 'Tested on RTX 5090' box is not ticked
    Checking the box is necessary but NOT sufficient — a clear decode improvement must be claimed."""
    body = (json.loads(gh(["pr", "view", str(num), "-R", repo, "--json", "body"]).stdout or "{}")
            .get("body") or "")
    if not any(re.search(r"\[\s*[xX]\s*\]", ln) and "5090" in ln for ln in body.splitlines()):
        return "unchecked", "RTX-5090 box unchecked"
    before, after = _decode_val(body, "before"), _decode_val(body, "after")
    if before is None or after is None:
        return "no-bench", "box ticked but decode before/after not filled with real numbers"
    if after <= before:
        return "no-bench", f"claimed decode before={before} ≥ after={after} (no improvement)"
    return "ok", f"ticked + decode {before}→{after} tok/s (+{after - before:.1f})"

def post_needs_bench_comment(repo, num):
    body = ("<!-- sparkinfer-needs-bench -->\n## ⏳ Needs a benchmark to be evaluated\n\n"
            "You ticked **Tested on RTX 5090** but the decode **before → after tok/s** table is still "
            "empty / placeholder (or shows no gain). The on-device eval won't run until it shows a real "
            "improvement.\n\nFill it from the **end-to-end** decode bench (not an isolated-kernel "
            "microbench):\n```bash\nbench/scripts/bench.sh --download            # baseline (before)\n"
            "bench/scripts/bench.sh --download            # your branch (after)\n```\n"
            "Then the bot greenlights it on the next poll and evaluates it on an RTX 5090.")
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

def render(res, oid):
    label = res.get("label", "?")
    icon = {"REJECT": "❌", "none": "⚪", "BASELINE": "📊"}.get(label, "✅")
    # A passing speedup (XL/L/M/S/XS) clears the significance gate, so its tps becomes the NEW frontier.
    advanced = label in {"XL", "L", "M", "S", "XS"} and res.get("pass")
    rows = [f"| **label** | `eval:{label}` |",
            f"| decode | {res.get('tps','?')} tok/s |",
            f"| correctness | top-1 {res.get('top1',0)*100:.1f}% · KL {res.get('kl','?')} |"]
    if "frontier_tps" in res and res["frontier_tps"]:
        # Label it "prior frontier" when this PR superseded it, so the old value isn't mistaken
        # for the current live frontier (which is now this PR's tps).
        rows.insert(2, f"| {'vs prior frontier' if advanced else 'vs frontier'} | {res['frontier_tps']} tok/s → "
                       f"{res.get('pct_over_frontier', 0):+.1f}% ({res.get('delta_tps',0):+.1f}) |")
    if advanced:
        rows.insert(3, f"| **→ new frontier** | **{res.get('tps')} tok/s** |")
    note = {"REJECT": f"Failed the correctness gate: {res.get('reason','')}. Not a valid submission.",
            "none": "Within the significance gate — no *verified* speedup over the current frontier.",
            "BASELINE": "No frontier was set; this run establishes it."
            }.get(label, f"Verified speedup — **sets the new frontier to {res.get('tps')} tok/s** "
                         f"(was {res.get('frontier_tps','?')}).")
    return (f"<!-- sparkinfer-eval:{oid} -->\n"
            f"## {icon} sparkinfer auto-eval — `{oid}`\n\n"
            f"| metric | value |\n|---|---|\n" + "\n".join(rows) + "\n\n"
            f"{note}\n\n"
            f"_RTX 5090 (sm_120) · built from source · correctness vs llama.cpp. "
            f"Automated — **not merged**; merge manually after review._")

# ---- live dashboard: data.json is canonical; data.js is generated for the page ----
DASH = os.path.join(ROOT, "dashboard")
DATA_JSON = os.path.join(DASH, "data.json")
FRONTIER_LABELS = {"XL", "L", "M", "S", "XS", "BASELINE"}
SPEEDUP_LABELS = {"XL", "L", "M", "S", "XS"}   # verified speedup over main (BASELINE excluded)

def load_dash():
    try:
        with open(DATA_JSON) as f: return json.load(f)
    except Exception:
        return None

def write_dash(data):
    with open(DATA_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(DASH, "data.js"), "w") as f:
        f.write("// Generated by eval/pr_eval_bot.py from data.json — do not edit by hand.\n")
        f.write("window.SPARKINFER = " + json.dumps(data, indent=2, ensure_ascii=False) + ";\n")

def push_dash(msg):
    subprocess.run(["git", "-C", ROOT, "add", "dashboard/data.json", "dashboard/data.js"], capture_output=True)
    if subprocess.run(["git", "-C", ROOT, "diff", "--cached", "--quiet"]).returncode == 0:
        return  # nothing changed
    subprocess.run(["git", "-C", ROOT, "commit", "-q", "-m", msg], capture_output=True)
    subprocess.run(["git", "-C", ROOT, "pull", "-q", "--rebase", "origin", "main"], capture_output=True)
    subprocess.run(["git", "-C", ROOT, "push", "-q", "origin", "main"], capture_output=True)

def update_dashboard(repo, pr, areas, res):
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
             "url": f"https://github.com/{repo}/pull/{num}"}
    data["prs"] = [p for p in data.get("prs", []) if p.get("num") != num]
    data["prs"].insert(0, entry)
    data["prs"] = data["prs"][:50]
    data["updated"] = datetime.date.today().isoformat()
    write_dash(data)
    push_dash(f"dashboard: PR #{num} -> eval:{res.get('label')} ({res.get('tps')} tok/s)")

def record_merge(repo, num):
    """A frontier-advancing PR was MERGED → advance the displayed frontier by its verified same-box
    relative gain and add it to the journey (`landed`). Hardware-independent and merged-only;
    idempotent (dedupe by PR). Reads the PR's stored eval from `prs`."""
    data = load_dash()
    if data is None: return
    if any(m.get("pr") == num for m in data.get("landed", [])): return       # already recorded
    e = next((p for p in data.get("prs", []) if p.get("num") == num), None)
    if not e or e.get("label") not in SPEEDUP_LABELS: return                 # only verified speedups
    old_f = data["status"].get("frontier_tps") or 0
    gain = (e.get("delta_pct") or 0) / 100.0
    new_f = round(old_f * (1 + gain), 2) if old_f else round(e.get("tps") or 0, 2)
    data["status"]["frontier_tps"] = new_f
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
      1. If a `merge-first` PR has since MERGED, its rivals are stale → tag them `re-evaluate` and
         ask them to rebase onto the new main (the bot re-evals automatically on the rebased commit).
      2. Among the still-open PRs with a verified speedup, label the biggest `merge-first` and the
         rest `needs-rebase`. Ranking uses the same-box % gain over main (data.json `delta_pct`)."""
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
            record_merge(repo, m["number"])      # advance the journey/frontier for the merged winner
            remove_label(repo, m["number"], MERGE_FIRST_LABEL)
        for num, labs in open_labels.items():
            if NEEDS_REBASE_LABEL in labs and REEVALUATE_LABEL not in labs:
                add_label(repo, num, REEVALUATE_LABEL)
                gh(["pr", "comment", str(num), "-R", repo, "--body",
                    "<!-- sparkinfer-reeval -->\nThe round's `merge-first` PR was just merged. Please "
                    "**rebase this branch onto `main`** — the bot will re-evaluate it against the new "
                    "frontier (so you're credited for the *marginal* gain on top of what merged)."])

    # 2) Rank open verified-speedup PRs; biggest → merge-first, rest → needs-rebase.
    scored = sorted(((num, by_num[num].get("delta_pct") or 0) for num in open_labels
                     if num in by_num and by_num[num].get("label") in SPEEDUP_LABELS),
                    key=lambda x: x[1], reverse=True)
    if not scored: return
    winner = scored[0][0]
    add_label(repo, winner, MERGE_FIRST_LABEL)
    for L in (NEEDS_REBASE_LABEL, REEVALUATE_LABEL): remove_label(repo, winner, L)
    for num, _ in scored[1:]:
        add_label(repo, num, NEEDS_REBASE_LABEL)
        remove_label(repo, num, MERGE_FIRST_LABEL)
    print(f">> round labels: merge-first #{winner}; needs-rebase {[n for n,_ in scored[1:]] or 'none'}")

    # Optionally auto-merge the winner (guarded), then flag the rivals to rebase + re-eval vs new main.
    if AUTO_MERGE_FIRST and try_auto_merge(repo, winner):
        record_merge(repo, winner)               # merged now → advance the journey/frontier
        for num, _ in scored[1:]:
            add_label(repo, num, REEVALUATE_LABEL)
            gh(["pr", "comment", str(num), "-R", repo, "--body",
                "<!-- sparkinfer-reeval -->\nThe round's `merge-first` PR was just merged. Please "
                "**rebase this branch onto `main`** — the bot re-evaluates it against the new frontier "
                "(crediting your *marginal* gain on top of what merged)."])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=int, required=True, help="vast.ai instance id to reuse")
    ap.add_argument("--frontier", type=float, default=0)
    ap.add_argument("--ceiling", type=float, default=0)
    ap.add_argument("--repo", default="gittensor-ai-lab/sparkinfer")
    ap.add_argument("--dry-run", action="store_true", help="evaluate + print, but don't label/comment")
    args = ap.parse_args()

    dash = load_dash()
    frontier = dash["status"]["frontier_tps"] if dash else args.frontier   # live ledger frontier
    # OLDEST-FIRST: evaluate ascending by PR number so the original of any duplicate is seen before
    # its copy, and the earliest submitter is graded first (fairness + copycat attribution).
    prs = json.loads(gh(["pr", "list", "-R", args.repo, "--state", "open",
                         "--json", "number,headRefName,headRefOid,title,isCrossRepository,labels,isDraft"]).stdout or "[]")
    prs.sort(key=lambda p: p["number"])
    if not prs:
        print("no open PRs"); return

    # Fingerprint EVERY PR (open + closed + merged) so an open PR can be compared against any earlier
    # one — copies often target already-merged originals. Built once, ascending by number.
    all_prs = json.loads(gh(["pr", "list", "-R", args.repo, "--state", "all",
                             "--json", "number,author", "-L", "300"]).stdout or "[]")
    all_nums = sorted(p["number"] for p in all_prs)
    pr_author = {p["number"]: (p.get("author") or {}).get("login", "?") for p in all_prs}
    fps = {n: pr_fingerprint(args.repo, n) for n in all_nums}
    copy_log = load_copycat_log()
    logged_copycats = {e["pr"] for e in copy_log}
    state_changed = False

    def find_original(num):
        """Earliest PR by a DIFFERENT author (shared file) whose added lines contain this PR's diff.
        Self-resubmissions (same author iterating on their own earlier PR) are NOT copycats."""
        files, added = fps.get(num, (set(), set()))
        if not added: return None
        me = pr_author.get(num, "?")
        for earlier in all_nums:
            if earlier >= num: break
            if pr_author.get(earlier, "?") == me: continue   # ignore one's own earlier PRs
            ef, ea = fps.get(earlier, (set(), set()))
            if (files & ef) and containment(added, ea) >= COPYCAT_CONTAINMENT:
                return earlier
        return None

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
        # Gate 2 — copycat: re-submits a DIFFERENT author's earlier diff. Label, log, skip eval;
        # 2nd strike -> block. (Self-resubmissions are excluded by find_original.)
        original = find_original(num)
        if original is not None:
            author = pr_author.get(num, "?")
            print(f"PR #{num}: COPYCAT of #{original} by {pr_author.get(original,'?')} "
                  f"(author {author}) — flag, no eval")
            if not args.dry_run and num not in logged_copycats:
                flag_copycat(args.repo, num, original, author)
                copy_log.append({"pr": num, "author": author, "original": original,
                                 "date": datetime.date.today().isoformat()})
                logged_copycats.add(num); state_changed = True
                strikes = sum(1 for e in copy_log if e["author"] == author)
                if strikes >= COPYCAT_STRIKES and author.lower() not in load_denylist():
                    print(f"  -> {author} hit {strikes} copycats — auto-blocking")
                    block_account(author, f"Auto-blocked after {strikes} copycat PRs "
                                  f"(#{', #'.join(str(e['pr']) for e in copy_log if e['author']==author)}).")
                    close_blocked_pr(args.repo, num, {author})
            continue
        areas = areas_for_pr(args.repo, num)
        print(f"PR #{num} @ {oid}: areas={sorted(areas) or ['(none)']} ref={ref}")
        if not args.dry_run: apply_area_labels(args.repo, num, areas)
        if oid in evaluated_commits(args.repo, num):
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
        # Gate 3 — greenlight (proof-gated): evaluate only if the PR ticks the RTX-5090 box AND
        # fills the decode before/after table with a real improvement. No override exists.
        # Reconcile labels each poll so a stale test-on-5090 can't keep a no-benchmark PR in the queue.
        pr_labels = {l["name"] for l in pr.get("labels", [])}
        def _reconcile(keep, drop):
            if args.dry_run: return
            if keep not in pr_labels: add_label(args.repo, num, keep)
            for L in drop:
                if L in pr_labels: remove_label(args.repo, num, L)
        status, reason = greenlight_status(args.repo, num, pr_labels)
        if status == "ok":
            print(f"PR #{num}: greenlit ({reason})")
            _reconcile(EVAL_GATE_LABEL, [NOT_TESTED_LABEL, NEEDS_BENCH_LABEL])
            pending.append((pr, num, branch, oid, ref, areas))
        elif status == "no-bench":
            print(f"PR #{num}: NOT greenlit ({reason}) — needs-benchmark, skip eval")
            first_time = NEEDS_BENCH_LABEL not in pr_labels
            _reconcile(NEEDS_BENCH_LABEL, [EVAL_GATE_LABEL, NOT_TESTED_LABEL])
            if first_time and not args.dry_run: post_needs_bench_comment(args.repo, num)
        else:  # unchecked
            print(f"PR #{num}: not greenlit ({reason}) — mark not-tested, skip eval")
            _reconcile(NOT_TESTED_LABEL, [EVAL_GATE_LABEL, NEEDS_BENCH_LABEL])

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

    # Reuse the pinned stable box first (cached model, good download speed). Reset the pointer to it
    # at the start of each run so the pin is always tried before any fallback box left from a prior run.
    if PINNED_INSTANCE:
        with open(INSTANCE_FILE, "w") as f: f.write(PINNED_INSTANCE)

    # --- Same-box baseline -------------------------------------------------------------------------
    # vast boxes vary in speed, so comparing a PR's tok/s against a frontier measured on a DIFFERENT
    # box leaks hardware variance into the delta. Build+bench origin/main on THIS box first and grade
    # every PR against that same-box number (+ any PR that lands earlier in this run). Measured ONCE
    # per run, not per PR — otherwise two PRs targeting the same optimization could both "beat" main.
    base_iid = current_instance(args.instance)
    bcmd = [sys.executable, os.path.join(HERE, "vast_eval.py"), "--reuse", str(base_iid),
            "--ref", "origin/main", "--frontier", "0", "--ceiling", str(args.ceiling), "--keep"]
    if PINNED_INSTANCE and str(base_iid) == PINNED_INSTANCE: bcmd.append("--pinned")
    print(f">> measuring same-box baseline (origin/main) on instance {base_iid} ...")
    br = subprocess.run(bcmd, cwd=ROOT, capture_output=True, text=True, timeout=14400)
    if br.returncode == PINNED_RETRY_RC:
        tail = next((l for l in reversed((br.stdout + br.stderr).splitlines()) if l.strip()), "")
        print(f">> {tail}\n>> aborting this run — next scheduled run retries the pinned box."); return
    for l in br.stdout.splitlines():
        if l.startswith("NEW_INSTANCE_ID "):
            try:
                nid = int(l.split()[1])
                with open(INSTANCE_FILE, "w") as f: f.write(str(nid))
                if PINNED_INSTANCE: _write_pin(nid)
                print(f"  (instance updated to {nid}{'; re-pinned' if PINNED_INSTANCE else ''})")
            except Exception: pass
    bline = next((l for l in br.stdout.splitlines() if l.startswith("RESULT_JSON")), None)
    bres = json.loads(bline[len("RESULT_JSON "):]) if bline else {}
    if not bres.get("pass") or not bres.get("tps"):
        log = (br.stdout + br.stderr)[-1200:]
        print(f">> same-box baseline (origin/main) failed ({bres.get('label','no result')}) — "
              f"aborting; no PRs graded.\n{log}"); return
    run_baseline = bres["tps"]
    print(f">> same-box baseline: origin/main = {run_baseline} tok/s on this box")

    # Run all pending evals on the SAME instance: pass --keep so vast_eval.py never stops/destroys
    # the box mid-queue. The bot stops the instance once after ALL PRs finish (or if the instance
    # dies, subsequent PRs self-heal by provisioning a new one).
    for i, (pr, num, branch, oid, ref, areas) in enumerate(pending):
        # Grade against the same-box baseline = MERGED origin/main (measured above). Every PR in the
        # run is graded against main, NOT against other PRs in the run — #67 and #70 are independent
        # branches off main, so each must get its own gain over main (the old within-run ratchet made
        # whichever ran second look like "none"). The frontier advances when you MERGE; to see if two
        # optimizations STACK, re-evaluate the second after merging the first. Literal duplicates are
        # caught by copycat detection; emission only pays MERGED PRs, so the maintainer's merge choice
        # (not eval order) decides what counts.
        cur_frontier = run_baseline
        cur_iid = current_instance(args.instance)
        cmd = [sys.executable, os.path.join(HERE, "vast_eval.py"),
               "--reuse", str(cur_iid), "--ref", ref,
               "--frontier", str(cur_frontier), "--ceiling", str(args.ceiling),
               "--keep"]            # keep instance alive — bot stops it after all PRs
        if PINNED_INSTANCE and str(cur_iid) == PINNED_INSTANCE:
            cmd.append("--pinned")  # never destroy the pin; retry-then-fallback on bring-up failure
        pinned = "--pinned" in cmd
        print(f"PR #{num} @ {oid}: evaluating '{ref}' (frontier={cur_frontier}) on instance "
              f"{cur_iid}{' [pinned]' if pinned else ''} ...")
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=14400)
        if r.returncode == PINNED_RETRY_RC:
            tail = next((l for l in reversed((r.stdout + r.stderr).splitlines()) if l.strip()), "")
            print(f">> {tail}\n>> aborting this run — the next scheduled run retries the "
                  f"pinned box. No PRs evaluated this tick."); return
        # If vast_eval self-healed/fell back to a new instance, track the new id for the next PR.
        for l in r.stdout.splitlines():
            if l.startswith("NEW_INSTANCE_ID "):
                try:
                    new_id = int(l.split()[1])
                    with open(INSTANCE_FILE, "w") as f: f.write(str(new_id))
                    if PINNED_INSTANCE: _write_pin(new_id)   # self-heal: re-pin to the fresh box
                    print(f"  (instance updated to {new_id}{'; re-pinned' if PINNED_INSTANCE else ''})")
                except Exception: pass
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT_JSON")), None)
        if not line:
            log = (r.stdout + r.stderr)[-1500:]
            print(f"PR #{num}: eval produced no result\n{log}")
            body = (f"<!-- sparkinfer-eval:{oid} -->\n⚠️ **sparkinfer auto-eval errored** for `{oid}` "
                    f"— re-run manually.\n\n<details><summary>log tail</summary>\n\n```\n{log}\n```\n</details>")
            res, label = None, None
        else:
            res = json.loads(line[len("RESULT_JSON "):]); label = res["label"]; body = render(res, oid)
            print(f"PR #{num}: {json.dumps(res)}")
        if args.dry_run:
            print("--- dry-run, not posting ---\n" + body); continue
        if label:
            for lab in {l for l in labels_on(args.repo, num) if l.startswith("eval:")}:
                remove_label(args.repo, num, lab)
            add_label(args.repo, num, f"eval:{label}")
        gh(["pr", "comment", str(num), "-R", args.repo, "--body", body])
        print(f"PR #{num}: posted {'eval:'+label if label else 'error'} — NOT merged.")
        if res: update_dashboard(args.repo, pr, areas, res)
        # NB: run_baseline is NOT ratcheted here — every PR is graded against merged origin/main, so
        # independent optimizations each get their true gain (the frontier advances on MERGE, not eval).

    # Per-round merge workflow: among the PRs graded this round, label the biggest verified speedup
    # `merge-first` and the rest `needs-rebase`; if a prior winner merged, flag its rivals `re-evaluate`.
    if not args.dry_run:
        reconcile_merge_labels(args.repo)

    # Stop (not destroy) the instance after all PRs — disk/model cache persists for next run.
    final_iid = current_instance(args.instance)
    if final_iid:
        print(f">> stopping instance {final_iid} — model cache persists for next run")
        subprocess.run(["vastai", "stop", "instance", str(final_iid)], capture_output=True)
    print("done — no merges (manual).")

if __name__ == "__main__":
    main()
