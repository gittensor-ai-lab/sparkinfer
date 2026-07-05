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
CONTEXT_LABELS     = {"128-context", "512-context", "4k-context", "16k-context", "32k-context"}
REGRESSION_LABELS  = {"regression-128", "regression-512", "regression-4k", "regression-16k", "regression-32k"}

# Auto-merge the round's merge-first winner — OFF unless SPARKINFER_AUTOMERGE=1. Heavily guarded:
# the eval only verifies speed + token-match, so auto-merge is gated on labels, author standing,
# changed paths, and branch protection (gh refuses if checks/reviews aren't satisfied).
AUTO_MERGE_FIRST = os.environ.get("SPARKINFER_AUTOMERGE", "0") == "1"
# Auto-merge is BLOCKED if the PR carries any of these labels:
AUTOMERGE_BLOCK_LABELS = {"copycat", "flagged:gaming", "penalty", "needs-benchmark", "not-tested",
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
# Logged to .github/copycats.json; ANY copycat immediately blocks the author and closes the PR
# (zero tolerance — no penalty period, no strike threshold).
FLAG_FILE = os.path.join(ROOT, ".github", "FLAGGED.md")
COPYCAT_LABEL = "copycat"
COPYCAT_LOG = os.path.join(ROOT, ".github", "copycats.json")
COPYCAT_CONTAINMENT = 0.80   # ≥80% of the copy's added lines also appear in the original
COPYCAT_STRIKES = 1          # zero tolerance: the FIRST copycat denylists the author + closes the PR
PENALTY_DAYS = 5             # (legacy; copycats now block immediately, so no penalty period applies)
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
            f"Duplicating another contributor's work is treated as gaming the SN74 emission "
            f"mechanism. The account has been **blocked** and this PR **closed** — zero tolerance, "
            f"no warning. See [`.github/COPYCATS.md`](../blob/main/.github/COPYCATS.md).")
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

def render(res, oid):
    label = res.get("label", "?")
    icon = {"REJECT": "❌", "none": "⚪", "BASELINE": "📊"}.get(label, "✅")
    # A passing speedup (XL/L/M/S/XS) clears the significance gate, so its tps becomes the NEW frontier.
    advanced = label in {"XL", "L", "M", "S", "XS"} and res.get("pass")
    ctx_label = res.get("best_context_label")
    # Dual-model runs carry a "guard" block (Qwen3-30B) + "model" (the scored target, Qwen3.6). The
    # verdict can REJECT on the GUARD even when every scored-model gate passes, so the guard's own
    # per-context results must be shown — otherwise the table reads "all pass" next to a REJECT.
    guard = res.get("guard") or {}
    scored_model = res.get("model")
    dual = bool(guard) and bool(scored_model)
    short = scored_model.split("-")[0] if scored_model else ""       # "Qwen3.6"
    gname = res.get("guard_model", "Qwen3-30B")
    rows = [f"| **label** | `eval:{label}` |",
            f"| scored decode ({res.get('score_context', 128)} ctx{f' · {ctx_label}' if ctx_label else ''}{f' · {short}' if dual else ''}) | {res.get('tps','?')} tok/s |",
            f"| correctness{f' ({short} vs llama.cpp)' if dual else ''} | top-1 {res.get('top1',0)*100:.1f}% · KL {res.get('kl','?')} |"]
    # scored-model per-context no-regression gates — SKIP contexts not measured (tps 0/None) so a
    # deliberately-skipped 16k/32k never renders a misleading "0.0 tok/s · pass".
    for key, gkey, bkey, lbl in [("ctx_128_tps", "guard_128_pass", "guard_128_baseline", "128-token"),
                                 ("ctx_512_tps", "guard_512_pass", "guard_512_baseline", "512-context"),
                                 ("ctx_4096_tps", "guard_4k_pass", "guard_4k_baseline", "4k-context"),
                                 ("ctx_16384_tps", "guard_16k_pass", "guard_16k_baseline", "16k-context"),
                                 ("ctx_32768_tps", "guard_32k_pass", "guard_32k_baseline", "32k-context")]:
        tps = res.get(key)
        if not tps:
            continue
        gate = "pass" if res.get(gkey, True) else "fail"
        base = res.get(bkey) or (res.get("frontier_tps") if key == "ctx_16384_tps" else 0) or 0
        rows.append(f"| {f'{short} ' if dual else ''}{lbl} no-regression gate | {tps} tok/s"
                    f"{f' vs main {base} tok/s' if base else ''} · {gate} |")
    if res.get("ctx_2048_tps") is not None and res.get("ctx_512_tps") is None:
        gate = "pass" if res.get("guard_2k_pass", True) else "fail"
        base = res.get("guard_2k_baseline") or 0
        rows.append(f"| legacy 2k no-regression gate | {res.get('ctx_2048_tps')} tok/s"
                    f"{f' vs main {base} tok/s' if base else ''} · {gate} |")
    # The Qwen3-30B no-regression guard — the check that actually gates a dual verdict.
    if dual:
        acc_ok = "pass" if guard.get("accuracy_ok", True) else "**FAIL**"
        rows.append(f"| **{gname} guard — accuracy** | top-1 {guard.get('top1',0)*100:.1f}% · KL {guard.get('kl','?')} · {acc_ok} |")
        for key, gkey, lbl in [("ctx_128_tps", "guard_128_pass", "128-token"),
                               ("ctx_512_tps", "guard_512_pass", "512-context"),
                               ("ctx_4096_tps", "guard_4k_pass", "4k-context"),
                               ("ctx_16384_tps", "guard_16k_pass", "16k-context"),
                               ("ctx_32768_tps", "guard_32k_pass", "32k-context")]:
            tps = guard.get(key)
            if not tps:
                continue
            rows.append(f"| {gname} guard — {lbl} | {tps} tok/s · {'pass' if guard.get(gkey, True) else '**fail**'} |")
    if res.get("regression_labels") or res.get("guard_regression_labels"):
        allregs = (res.get("regression_labels") or []) + (res.get("guard_regression_labels") or [])
        rows.append(f"| regressions | {', '.join(allregs)} |")
    if "frontier_tps" in res and res["frontier_tps"]:
        # Label it "prior frontier" when this PR superseded it, so the old value isn't mistaken
        # for the current live frontier (which is now this PR's tps).
        rows.insert(2, f"| {'vs prior frontier' if advanced else 'vs frontier'} | {res['frontier_tps']} tok/s → "
                       f"{res.get('pct_over_frontier', 0):+.1f}% ({res.get('delta_tps',0):+.1f}) |")
    if advanced:
        rows.insert(3, f"| **→ new frontier** | **{res.get('tps')} tok/s** |")
    note = {"REJECT": f"**Rejected** — {res.get('reason','')}.",
            "none": "Within the significance gate — no *verified* speedup over the current frontier.",
            "BASELINE": "No frontier was set; this run establishes it."
            }.get(label, f"Verified speedup — **sets the new frontier to {res.get('tps')} tok/s** "
                         f"(was {res.get('frontier_tps','?')}).")
    if label == "REJECT" and res.get("auto_close"):
        note = "No context cleared the 2% significance gate while at least one context regressed. Auto-closing this PR."
    target_note = ("128/512/4k/16k/32k guarded · strongest context scores"
                   if res.get("eval_mode") == "longctx" else "128-token decode frontier")
    return (f"<!-- sparkinfer-eval:{oid} -->\n"
            f"## {icon} sparkinfer auto-eval — `{oid}`\n\n"
            f"| metric | value |\n|---|---|\n" + "\n".join(rows) + "\n\n"
            f"{note}\n\n"
            f"_RTX 5090 (sm_120) · {target_note} · built from source · correctness vs llama.cpp. "
            f"Automated — **not merged**; merge manually after review._")

# ---- live dashboard: data.json is canonical; data.js is generated for the page ----
DASH = os.path.join(ROOT, "dashboard")
DATA_JSON = os.path.join(DASH, "data.json")
FRONTIER_LABELS = {"XL", "L", "M", "S", "XS", "BASELINE"}
SPEEDUP_LABELS = {"XL", "L", "M", "S", "XS"}   # verified speedup over main (BASELINE excluded)
CTX_SERIES = {
    128:   {"metric": "ctx_128_tps",   "guard": "guard_128_baseline",  "status": "frontier_tps",     "label": "128", "color": "#D14D72", "llama": 365.85, "note": "128-token decode, no prefill context"},
    512:   {"metric": "ctx_512_tps",   "guard": "guard_512_baseline",  "status": "longctx_512_tps",  "label": "512", "color": "#7B5DFF", "llama": 342.59, "note": "llama-batched-bench npp=512 ntg=128 npl=1"},
    4096:  {"metric": "ctx_4096_tps",  "guard": "guard_4k_baseline",   "status": "longctx_4k_tps",   "label": "4k", "color": "#0E8A16", "llama": 292.99, "note": "llama-batched-bench npp=4096 ntg=128 npl=1"},
    16384: {"metric": "ctx_16384_tps", "guard": "guard_16k_baseline",  "status": "longctx_16k_tps",  "label": "16k", "color": "#B8860B", "llama": 245.53, "note": "llama-batched-bench npp=16384 ntg=128 npl=1"},
    32768: {"metric": "ctx_32768_tps", "guard": "guard_32k_baseline",  "status": "longctx_32k_tps",  "label": "32k", "color": "#6F42C1", "llama": 192.62, "note": "release-log llama.cpp estimate at 32k, ntg=128"},
}

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

LOG_REPO  = os.environ.get("SPARKINFER_LOG_REPO", "https://github.com/gittensor-ai-lab/sparkinfer-log.git")
LOG_DIR   = os.path.expanduser(os.environ.get("SPARKINFER_LOG_DIR", "~/.sparkinfer_log_checkout"))
LOG_PAGE  = "https://gittensor-ai-lab.github.io/sparkinfer-log/?run="

def upload_eval_log(repo, num, title, oid, res, log_text, baseline):
    """Commit this eval's raw log + result to the public sparkinfer-log repo (immutable record),
    rendered at a unique per-run URL. Best-effort: never blocks the eval. Returns the page URL."""
    try:
        rid = f"{int(num):04d}-{oid[:7]}"
        if not os.path.isdir(os.path.join(LOG_DIR, ".git")):
            subprocess.run(["git", "clone", "-q", LOG_REPO, LOG_DIR], check=True)
        else:
            subprocess.run(["git", "-C", LOG_DIR, "pull", "-q", "--rebase"], check=False)
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
        json.dump(result, open(os.path.join(rundir, "result.json"), "w"), indent=2)
        clean = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<ip>", log_text or "")   # scrub host IPs
        open(os.path.join(rundir, "log.txt"), "w").write(clean)
        ipath = os.path.join(LOG_DIR, "index.json")
        idx = json.load(open(ipath)) if os.path.exists(ipath) else []
        idx = [e for e in idx if e.get("id") != rid]
        idx.append({"id": rid, "pr": int(num), "title": title, "label": res.get("label"),
                    "delta_pct": res.get("pct_over_frontier"), "tps": res.get("tps"),
                    "score_context": res.get("score_context"), "date": result["date"]})
        idx.sort(key=lambda x: x["id"])
        json.dump(idx, open(ipath, "w"), indent=2)
        subprocess.run(["git", "-C", LOG_DIR, "add", "-A"], check=True)
        subprocess.run(["git", "-C", LOG_DIR, "commit", "-q", "-m",
                        f"eval: #{num} {oid[:7]} -> eval:{res.get('label')}"], check=False)
        subprocess.run(["git", "-C", LOG_DIR, "push", "-q"], check=False)
        url = LOG_PAGE + rid
        print(f">> eval log: {url}")
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
             "url": f"https://github.com/{repo}/pull/{num}"}
    for k in ("eval_mode", "score_context", "best_context_label", "context_gains_pct",
              "regression_labels", "auto_close",
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
    if proof_url: entry["proof_url"] = proof_url
    data["prs"] = [p for p in data.get("prs", []) if p.get("num") != num]
    data["prs"].insert(0, entry)
    data["prs"] = data["prs"][:50]
    data["updated"] = datetime.date.today().isoformat()
    write_dash(data)
    push_dash(f"dashboard: PR #{num} -> eval:{res.get('label')} ({res.get('tps')} tok/s)")

def _scaled_context_tps(old_tps, measured_tps, guard_tps):
    if measured_tps is None:
        return None
    measured_tps = float(measured_tps)
    old_tps = float(old_tps or 0)
    guard_tps = float(guard_tps or 0)
    if old_tps > 0 and guard_tps > 0:
        return round(old_tps * (measured_tps / guard_tps), 2)
    return round(measured_tps, 2)

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

def record_merge(repo, num):
    """A frontier-advancing PR was MERGED → advance the displayed frontier by its verified same-box
    relative gain and add it to the journey (`landed`). Hardware-independent and merged-only;
    idempotent (dedupe by PR). Reads the PR's stored eval from `prs`."""
    data = load_dash()
    if data is None: return
    e = next((p for p in data.get("prs", []) if p.get("num") == num), None)
    if not e or e.get("label") not in SPEEDUP_LABELS: return                 # only verified speedups
    if e.get("eval_mode") == "longctx" and int(e.get("score_context") or 0) in (512, 4096, 16384, 32768):
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
    # Advance by the VERIFIED SAME-BOX RELATIVE GAIN (delta_pct), NOT the raw measured tps. Raw tok/s
    # zig-zags ±2% with whichever box ran (hot vs cool) and breaks the journey's monotonicity; applying
    # the same-box gain to the displayed frontier keeps the headline hardware-independent and the journey
    # a clean calibrated ladder. Falls back to raw max() only if the gain wasn't recorded.
    old_f = round(data["status"].get("frontier_tps") or 0, 2)
    gain = (e.get("delta_pct") or 0) / 100.0
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
        if not os.path.isdir(os.path.join(LOG_DIR, ".git")):
            subprocess.run(["git", "clone", "-q", LOG_REPO, LOG_DIR], check=True)
        else:
            subprocess.run(["git", "-C", LOG_DIR, "pull", "-q", "--rebase"], check=False)
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
    ap.add_argument("--instance", type=int, required=True, help="vast.ai instance id to reuse")
    ap.add_argument("--frontier", type=float, default=0)
    ap.add_argument("--ceiling", type=float, default=0)
    ap.add_argument("--repo", default="gittensor-ai-lab/sparkinfer")
    ap.add_argument("--dry-run", action="store_true", help="evaluate + print, but don't label/comment")
    # Dual-model: score Qwen3.6-35B-A3B (primary) and guard Qwen3-30B against no-regression. The
    # Qwen3-30B guard baselines are still measured same-box each run (run_guard_*); the Qwen3.6
    # primary baselines are stable config constants (no Qwen3.6-optimizing PR has moved them yet) —
    # overridable via env until a same-box Qwen3.6 baseline pass is added.
    ap.add_argument("--dual", action="store_true",
                    help="score Qwen3.6 (primary) + guard Qwen3-30B (no-regression) via evaluate_dual.sh")
    ap.add_argument("--primary-frontier", type=float,
                    default=float(os.environ.get("SPARKINFER_QWEN36_FRONTIER", "23.0")),
                    help="[--dual] Qwen3.6 current best verified 128/512/4k tok/s (the scored frontier)")
    args = ap.parse_args()
    # Qwen3.6 same-box origin/main baselines (128/512/4k). Env-overridable; measured 2026-07 on RTX 5090.
    QWEN36_BASE = {
        "128": float(os.environ.get("SPARKINFER_QWEN36_128", "23.22")),
        "512": float(os.environ.get("SPARKINFER_QWEN36_512", "23.16")),
        "4k":  float(os.environ.get("SPARKINFER_QWEN36_4K",  "23.03")),
        "llama128": float(os.environ.get("SPARKINFER_QWEN36_LLAMA_128", "190")),
    }

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
            ea_login = pr_author.get(earlier, "?")
            if ea_login == me: continue                      # ignore one's own earlier PRs
            # A blocked copier's PR (or one already adjudicated as a copy) must NOT be usable as the
            # "original": otherwise a copier can front-run the real author by opening an earlier-numbered
            # PR (even an empty placeholder later force-pushed), get flagged, yet still frame the author.
            if ea_login.lower() in denylist: continue
            if earlier in logged_copycats: continue
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
        # Gate 2 — copycat: re-submits a DIFFERENT author's earlier diff. Zero tolerance — flag,
        # block the author, and close the PR immediately (no eval, no penalty, no strike threshold).
        # (Self-resubmissions are excluded by find_original.)
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
            "--ref", "origin/main", "--frontier", "0", "--ceiling", str(args.ceiling),
            "--eval-mode", "longctx", "--keep"]
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
               "--eval-mode", "longctx", "--guard-128-baseline", str(run_guard_128),
               "--guard-512-baseline", str(run_guard_512),
               "--guard-4k-baseline", str(run_guard_4k),
               "--guard-16k-baseline", str(run_guard_16k),
               "--guard-32k-baseline", str(run_guard_32k),
               "--keep"]            # keep instance alive — bot stops it after all PRs
        if args.dual:
            # Qwen3.6 scored (128/512/4k); the --guard-*-baseline above become the Qwen3-30B guard.
            cmd[cmd.index("--keep"):cmd.index("--keep")] = [
                "--dual", "--primary-frontier", str(args.primary_frontier),
                "--p-guard-128-baseline", str(QWEN36_BASE["128"]),
                "--p-guard-512-baseline", str(QWEN36_BASE["512"]),
                "--p-guard-4k-baseline",  str(QWEN36_BASE["4k"]),
                "--p-llama-128-baseline", str(QWEN36_BASE["llama128"])]
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
            cur = labels_on(args.repo, num)
            for lab in {l for l in cur if l.startswith("eval:")}:
                remove_label(args.repo, num, lab)
            add_label(args.repo, num, f"eval:{label}")
            apply_context_label(args.repo, num, cur, res.get("best_context_label"))
            apply_regression_labels(args.repo, num, cur, res.get("regression_labels"))
            # This was just graded against the CURRENT main, so it's no longer stale: clear
            # needs-rebase. If it carried needs-rebase, it was a post-rebase re-eval → tag re-evaluate.
            if NEEDS_REBASE_LABEL in cur:
                remove_label(args.repo, num, NEEDS_REBASE_LABEL)
                add_label(args.repo, num, REEVALUATE_LABEL)
        gh(["pr", "comment", str(num), "-R", args.repo, "--body", body])
        print(f"PR #{num}: posted {'eval:'+label if label else 'error'} — NOT merged.")
        if res:
            proof = upload_eval_log(args.repo, num, pr.get("title", ""), oid, res, r.stdout + r.stderr, run_baseline)
            update_dashboard(args.repo, pr, areas, res, proof_url=proof)
            if res.get("auto_close"):
                gh(["pr", "close", str(num), "-R", args.repo,
                    "--comment", "Auto-closed by sparkinfer eval: no single context cleared the 2% improvement gate and at least one context regressed."])
                print(f"PR #{num}: auto-closed after regression-only eval reject.")
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
