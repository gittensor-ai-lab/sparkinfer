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
AUTOMERGE_BLOCK_LABELS = {"copycat", "copycat-warn", "flagged:gaming", "penalty", "needs-benchmark", "not-tested",
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
# Tiered policy (shared with eval/copycat_policy.py + copycat_guard.py):
#   ≥85% containment → block + close; 75–84% → copycat-warn; 3 warns → block.
from copycat_policy import COPYCAT_BLOCK, COPYCAT_WARN, MAX_WARNINGS, skip_copycat_scoring
from copycat_guard import warn_copycat

FLAG_FILE = os.path.join(ROOT, ".github", "FLAGGED.md")
COPYCAT_LABEL = "copycat"
COPYCAT_WARN_LABEL = "copycat-warn"
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
            f"This PR re-submits substantially the same diff (≥85% line overlap) as the earlier "
            f"#{original}. Duplicating another contributor's work is treated as gaming the SN74 "
            f"emission mechanism. The account has been **blocked** and this PR **closed**.\n\n"
            f"See [`.github/COPYCATS.md`](../blob/main/.github/COPYCATS.md).")
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
    rows = [f"| **label** | `eval:{label}` |"]
    if bidir:
        rows.append(f"| Qwen3.5 score | `eval-qwen35:{res.get('label_qwen35', '?')}` "
                    f"({'pass' if res.get('pass_qwen35') else 'fail'}) |")
        rows.append(f"| Qwen3.6 score | `eval-qwen36:{res.get('label_qwen36', '?')}` "
                    f"({'pass' if res.get('pass_qwen36') else 'fail'}) |")
    rows += [
            f"| scored decode ({res.get('score_context', 128)} ctx{f' · {ctx_label}' if ctx_label else ''}{f' · {short}' if dual or triple or bidir else ''}) | {res.get('tps','?')} tok/s |",
            f"| correctness{f' ({short} vs llama.cpp)' if dual or triple or bidir else ''} | top-1 {res.get('top1',0)*100:.1f}% · KL {res.get('kl','?')} |"]
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
        base = res.get(bkey) or _GUARD_BASE_FALLBACK.get(bkey, 0)
        rows.append(f"| {f'{short} ' if dual or triple else ''}{lbl} no-regression gate | {tps} tok/s"
                    f"{f' vs main {base} tok/s' if base else ''} · {gate} |")
    if res.get("ctx_2048_tps") is not None and res.get("ctx_512_tps") is None:
        gate = "pass" if res.get("guard_2k_pass", True) else "fail"
        base = res.get("guard_2k_baseline") or 0
        rows.append(f"| legacy 2k no-regression gate | {res.get('ctx_2048_tps')} tok/s"
                    f"{f' vs main {base} tok/s' if base else ''} · {gate} |")
    # The Qwen3-30B no-regression guard — the check that actually gates a dual verdict.
    if bidir:
        for title, block in [("Qwen3.5 optimize", res.get("score_qwen35") or {}),
                             ("Qwen3.6 optimize", res.get("score_qwen36") or {})]:
            if not block:
                continue
            rows.append(f"| **{title}** | `eval:{block.get('label','?')}` · "
                        f"{block.get('tps','?')} tok/s · {'pass' if block.get('pass') else 'fail'} |")
            g = block.get("guard") or {}
            gname = block.get("guard_model", "guard")
            if g:
                rows.append(f"| {title} — {gname} guard accuracy | "
                            f"top-1 {g.get('top1',0)*100:.1f}% · KL {g.get('kl','?')} · "
                            f"{'pass' if g.get('accuracy_ok', True) else '**FAIL**'} |")
                for key, gkey, lbl in [("ctx_128_tps", "guard_128_pass", "128"),
                                       ("ctx_512_tps", "guard_512_pass", "512"),
                                       ("ctx_4096_tps", "guard_4k_pass", "4k"),
                                       ("ctx_16384_tps", "guard_16k_pass", "16k"),
                                       ("ctx_32768_tps", "guard_32k_pass", "32k")]:
                    tps = g.get(key)
                    if not tps:
                        continue
                    rows.append(f"| {title} — {gname} {lbl} | {tps} tok/s · "
                                f"{'pass' if g.get(gkey, True) else '**fail**'} |")
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
                                   ("ctx_32768_tps", "guard_32k_pass", "32k-context")]:
                tps = gblock.get(key)
                if not tps:
                    continue
                rows.append(f"| {gtitle} guard — {lbl} | {tps} tok/s · {'pass' if gblock.get(gkey, True) else '**fail**'} |")
    if res.get("regression_labels") or res.get("guard_regression_labels"):
        allregs = (res.get("regression_labels") or []) + (res.get("guard_regression_labels") or [])
        rows.append(f"| regressions | {', '.join(allregs)} |")
    if "frontier_tps" in res and res.get("frontier_tps", 0) > 0:
        # "frontier_tps" is now the SAME-BOX origin/main baseline — the gain is measured directly
        # against main on the same GPU in the same run, not a passed-in frontier number.
        rows.insert(2, f"| vs same-box main | {res['frontier_tps']} tok/s → "
                       f"{res.get('pct_over_frontier', 0):+.1f}% ({res.get('delta_tps',0):+.1f}) |")
    note = {"REJECT": f"**Rejected** — {res.get('reason','')}.",
            "none": "Within the significance gate — no *verified* speedup over same-box main.",
            "BASELINE": "No same-box main baseline was set; this run establishes one."
            }.get(label, f"Verified speedup over same-box origin/main — "
                         f"{res.get('tps')} tok/s (main was {res.get('frontier_tps','?')} tok/s).")
    if label == "REJECT" and res.get("auto_close"):
        note = "No context cleared the 2% significance gate while at least one context regressed. Auto-closing this PR."
    target_note = ("128/512/4k/16k/32k guarded · scored vs same-box main · strongest context scores"
                   if res.get("eval_mode") == "longctx" else "128-token decode scored vs same-box main")
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
    if not e:
        return
    if e.get("mode") == "bidir":
        if not ((e.get("pass_qwen35") and e.get("label_qwen35") in SPEEDUP_LABELS) or
                (e.get("pass_qwen36") and e.get("label_qwen36") in SPEEDUP_LABELS) or
                e.get("label") in SPEEDUP_LABELS):
            return
        changed = False
        if e.get("pass_qwen36") and e.get("label_qwen36") in SPEEDUP_LABELS:
            sub = e.get("score_qwen36") or e
            q36 = data.setdefault("qwen36", {})
            old_f = round(q36.get("frontier_tps") or q36.get("baseline_tps") or 0, 2)
            new_f = round(max(old_f, sub.get("tps") or 0), 2)
            q36["frontier_tps"] = new_f
            if sub.get("top1") is not None: q36["token_match"] = round(sub["top1"], 4)
            if sub.get("kl") is not None:   q36["kl"] = round(sub["kl"], 4)
            short = re.sub(r"^\w+(\([^)]*\))?:\s*", "", e.get("title", ""))[:28]
            landed = [m for m in data.get("landed_qwen36", []) if m.get("pr") != num and not m.get("baseline")]
            landed.append({"name": short or f"PR #{num}", "tps": new_f, "pr": num,
                           "date": datetime.date.today().isoformat(), "label": e.get("label_qwen36")})
            data["landed_qwen36"] = sorted(landed, key=lambda m: m["tps"])
            changed = True
        if e.get("pass_qwen35") and e.get("label_qwen35") in SPEEDUP_LABELS:
            sub = e.get("score_qwen35") or e
            q35 = data.setdefault("qwen35", {})
            old_f = round(q35.get("frontier_tps") or q35.get("baseline_tps") or 0, 2)
            new_f = round(max(old_f, sub.get("tps") or 0), 2)
            q35["frontier_tps"] = new_f
            if sub.get("top1") is not None: q35["token_match"] = round(sub["top1"], 4)
            if sub.get("kl") is not None:   q35["kl"] = round(sub["kl"], 4)
            short = re.sub(r"^\w+(\([^)]*\))?:\s*", "", e.get("title", ""))[:28]
            landed = [m for m in data.get("landed_qwen35", []) if m.get("pr") != num]
            landed.append({"name": short or f"PR #{num}", "tps": new_f, "pr": num,
                           "date": datetime.date.today().isoformat(), "label": e.get("label_qwen35")})
            data["landed_qwen35"] = sorted(landed, key=lambda m: m["tps"])
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
        new_f = round(max(old_f, e.get("tps") or 0), 2)          # Qwen3.6 frontier: take the max tps seen
        q36["frontier_tps"] = new_f
        if e.get("top1") is not None: q36["token_match"] = round(e["top1"], 4)
        if e.get("kl") is not None:   q36["kl"] = round(e["kl"], 4)
        short = re.sub(r"^\w+(\([^)]*\))?:\s*", "", e.get("title", ""))[:28]
        landed = [m for m in data.get("landed_qwen36", []) if m.get("pr") != num and not m.get("baseline")]
        landed.append({"name": short or f"PR #{num}", "tps": new_f, "pr": num,
                       "date": datetime.date.today().isoformat(), "label": e.get("label")})
        data["landed_qwen36"] = sorted(landed, key=lambda m: m["tps"])
        data["updated"] = datetime.date.today().isoformat()
        write_dash(data)
        push_dash(f"dashboard: PR #{num} merged -> Qwen3.6 frontier {new_f} tok/s")
        return
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
    ap.add_argument("--only-pr", type=int, default=0,
                    help="evaluate only this PR number (must be open)")
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
        "llama128": float(os.environ.get("SPARKINFER_QWEN36_LLAMA_128", "275.81")),
        "llama512": float(os.environ.get("SPARKINFER_QWEN36_LLAMA_512", "275.61")),
        "llama4k":  float(os.environ.get("SPARKINFER_QWEN36_LLAMA_4K",  "276.30")),
    }
    QWYTHOS_BASE = {
        "128": float(os.environ.get("SPARKINFER_QWYTHOS_128", "0")),
        "512": float(os.environ.get("SPARKINFER_QWYTHOS_512", "0")),
        "4k":  float(os.environ.get("SPARKINFER_QWYTHOS_4K",  "0")),
        "llama128": float(os.environ.get("QWEN35_9B_LLAMA_128", "0")),
        "llama512": float(os.environ.get("QWEN35_9B_LLAMA_512", "0")),
        "llama4k":  float(os.environ.get("QWEN35_9B_LLAMA_4K",  "0")),
    }

    # --- Polaris verifiable compute ---
    # Two modes, auto-selected:
    #   TDX (preferred):  POLARIS_API_KEY is set → scoring runs inside Intel TDX enclave
    #   Ed25519 (legacy): SPARKINFER_POLARIS_PRIVATE_KEY is set → bot signs attestations
    # The private key NEVER touches the eval box — the bot signs/submits here on the bot host.
    POLARIS_PRIVKEY = None
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
            print(f">> Polaris TDX enabled — scoring will run inside Intel TDX enclave")
        else:
            _key_b64 = os.environ.get("SPARKINFER_POLARIS_PRIVATE_KEY", "")
            if _key_b64:
                try:
                    POLARIS_PRIVKEY = _b64.b64decode(_key_b64)
                    print(f">> Polaris Ed25519 enabled — receipts will be signed")
                except Exception as e:
                    print(f">> Polaris key load failed: {e} — attestations will NOT be signed")
            else:
                print(">> Polaris enabled but no POLARIS_API_KEY or SPARKINFER_POLARIS_PRIVATE_KEY set — "
                      "attestations will be collected but NOT signed")

    dash = load_dash()
    frontier = dash["status"]["frontier_tps"] if dash else args.frontier   # live ledger frontier
    # OLDEST-FIRST: evaluate ascending by PR number so the original of any duplicate is seen before
    # its copy, and the earliest submitter is graded first (fairness + copycat attribution).
    prs = json.loads(gh(["pr", "list", "-R", args.repo, "--state", "open",
                         "--json", "number,headRefName,headRefOid,title,isCrossRepository,labels,isDraft"]).stdout or "[]")
    prs.sort(key=lambda p: p["number"])
    if args.only_pr:
        prs = [p for p in prs if p["number"] == args.only_pr]
        if not prs:
            print(f"PR #{args.only_pr} not open"); return
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
            if earlier in logged_copycats:
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
        # Gate 2 — copycat: tiered containment vs earlier PRs (open/closed/merged).
        original, copy_c = find_copycat_match(num)
        if original is not None:
            author = pr_author.get(num, "?")
            _, added = fps.get(num, (set(), set()))
            if skip_copycat_scoring(added, copy_c):
                print(f"PR #{num}: copycat-like #{original} at {copy_c:.0%} but too few added lines — allow eval")
            elif copy_c >= COPYCAT_BLOCK:
                print(f"PR #{num}: COPYCAT ≥85% of #{original} by {pr_author.get(original,'?')} "
                      f"(author {author}) — block, no eval")
                if not args.dry_run and num not in logged_copycats:
                    flag_copycat(args.repo, num, original, author)
                    copy_log.append({"pr": num, "author": author, "original": original,
                                     "date": datetime.date.today().isoformat(), "blocked": True})
                    logged_copycats.add(num); state_changed = True
                    if author.lower() not in load_denylist():
                        block_account(author, f"#{num} ≥85% copycat of #{original} ({copy_c:.0%})")
                        close_blocked_pr(args.repo, num, {author})
                continue
            else:
                print(f"PR #{num}: COPYCAT WARN {copy_c:.0%} of #{original} by {pr_author.get(original,'?')} "
                      f"(author {author}) — warn, skip eval")
                if not args.dry_run and num not in logged_copycats:
                    warn_strikes = sum(1 for e in copy_log
                                       if e.get("author") == author and not e.get("blocked", True))
                    strike = warn_strikes + 1
                    will_block = warn_copycat(args.repo, num, original, author, strike, copy_c)
                    copy_log.append({"pr": num, "author": author, "original": original,
                                     "date": datetime.date.today().isoformat(), "blocked": False,
                                     "strike": strike, "containment": round(copy_c, 3)})
                    logged_copycats.add(num); state_changed = True
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

    # Reuse the pinned stable box first (cached model, good download speed). Skip when on bare metal.
    if PINNED_INSTANCE and not ssh_box_enabled():
        with open(INSTANCE_FILE, "w") as f: f.write(PINNED_INSTANCE)

    # --- Same-box baseline -------------------------------------------------------------------------
    base_iid = current_instance(args.instance) if args.instance else 0
    bcmd = [sys.executable, os.path.join(HERE, "vast_eval.py"),
            *_vast_eval_transport_args(args.instance),
            "--ref", "origin/main", "--frontier", "0", "--ceiling", str(args.ceiling),
            "--eval-mode", "longctx", "--keep"]
    if args.bidir:
        bcmd += ["--bidir", "--primary-quant", args.primary_quant]
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
                if PINNED_INSTANCE: _write_pin(nid)
                print(f"  (instance updated to {nid}{'; re-pinned' if PINNED_INSTANCE else ''})")
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
    if not args.bidir and (not bres.get("pass") or not bres.get("tps")):
        log = (br.stdout + br.stderr)[-1200:]
        print(f">> same-box baseline (origin/main) failed ({bres.get('label','no result')}) — "
              f"aborting; no PRs graded.\n{log}"); return
    if args.bidir:
        run_baseline = float((bres.get("score_qwen36") or {}).get("tps") or bres.get("tps") or 0)
        s36 = bres.get("score_qwen36") or {}
        run_guard_128 = float(s36.get("ctx_128_tps") or run_baseline)
        run_guard_512 = float(s36.get("ctx_512_tps") or 0)
        run_guard_4k = float(s36.get("ctx_4096_tps") or 0)
        run_guard_16k = float(s36.get("ctx_16384_tps") or run_baseline)
        run_guard_32k = float(s36.get("ctx_32768_tps") or 0)
        score_ctx = int(s36.get("score_context") or bres.get("score_context") or 128)
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

    # Bidir mode: bench both model mains on the box (Qwen3.5 at 128/512/4k only).
    if args.bidir and _vast_sh:
        ssh_ep = ssh_box_endpoint()
        if ssh_ep:
            host, port = ssh_ep
        elif _vast_endpoint and _vast_info_of:
            import vastai
            v = vastai.VastAI()
            info = _vast_info_of(v, base_iid)
            host, port = _vast_endpoint(info) if info else (None, None)
        else:
            host = port = None

        def _ctx_sweep(host, port, gguf, label, store, ctx_list=None):
            B = "/root/sparkinfer/build/runtime/qwen3_gguf_bench"
            tps = {}
            sweep = ctx_list or [("128", 0), ("512", 512), ("4k", 4096),
                                 ("16k", 16384), ("32k", 32768)]
            for lbl, ctx in sweep:
                cmd = f"export PATH=/usr/local/cuda/bin:$PATH; {B} '{gguf}' 128 {ctx}"
                r = _vast_sh(host, port, cmd, timeout=600)
                m = re.search(r"decode\s+tg\s*:\s*([0-9.]+)", r.stdout + r.stderr)
                tps[lbl] = float(m.group(1)) if m else 0.0
                print(f"    [{label}] ctx={lbl} tps={tps[lbl]}")
            if tps.get("128", 0) > 0:
                store["128"] = tps["128"]
                store["512"] = tps["512"] if tps["512"] > 0 else round(tps["128"] * 0.98, 2)
                store["4k"]  = tps["4k"]  if tps["4k"]  > 0 else round(tps["128"] * 0.93, 2)
                store["16k"] = tps["16k"] if tps["16k"] > 0 else store.get("16k", 0)
                store["32k"] = tps["32k"] if tps["32k"] > 0 else store.get("32k", 0)
                print(f"  {label} same-box main: 128={store['128']} 512={store['512']} "
                      f"4k={store['4k']} 16k={store['16k']} 32k={store['32k']} tok/s")
            else:
                print(f"  {label} bench failed — using config defaults")

        if host and port:
            _ctx_sweep(host, port, "/workspace/models36/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                       "Qwen3.6", QWEN36_BASE)
            qmap = {
                "Q4_K_M": "Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf",
                "Q8_0":   "Qwythos-9B-Claude-Mythos-5-1M-Q8_0.gguf",
                "BF16":   "Qwythos-9B-Claude-Mythos-5-1M-BF16.gguf",
            }
            qfile = qmap[args.primary_quant]
            _ctx_sweep(host, port, f"/workspace/models35/{qfile}",
                       f"Qwythos ({args.primary_quant})", QWYTHOS_BASE,
                       ctx_list=[("128", 0), ("512", 512), ("4k", 4096)])
        else:
            print(f"  could not reach eval box for model bench — using config defaults")

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
        cur_iid = current_instance(args.instance) if args.instance else 0
        cmd = [sys.executable, os.path.join(HERE, "vast_eval.py"),
               *_vast_eval_transport_args(args.instance),
               "--ref", ref,
               "--frontier", "0", "--ceiling", str(args.ceiling),
               "--eval-mode", "longctx", "--guard-128-baseline", str(run_guard_128),
               "--guard-512-baseline", str(run_guard_512),
               "--guard-4k-baseline", str(run_guard_4k),
               "--guard-16k-baseline", str(run_guard_16k),
               "--guard-32k-baseline", str(run_guard_32k),
               "--keep"]
        if args.bidir:
            cmd[cmd.index("--keep"):cmd.index("--keep")] = [
                "--bidir",
                "--primary-quant", args.primary_quant,
                "--p35-guard-128-baseline", str(QWYTHOS_BASE["128"]),
                "--p35-guard-512-baseline", str(QWYTHOS_BASE["512"]),
                "--p35-guard-4k-baseline",  str(QWYTHOS_BASE["4k"]),
                "--p-guard-128-baseline", str(QWEN36_BASE["128"]),
                "--p-guard-512-baseline", str(QWEN36_BASE["512"]),
                "--p-guard-4k-baseline",  str(QWEN36_BASE["4k"]),
                "--p-guard-16k-baseline", str(QWEN36_BASE["16k"]),
                "--p-guard-32k-baseline", str(QWEN36_BASE["32k"]),
                "--g36-guard-128-baseline", str(QWEN36_BASE["128"]),
                "--g36-guard-512-baseline", str(QWEN36_BASE["512"]),
                "--g36-guard-4k-baseline",  str(QWEN36_BASE["4k"]),
                "--g36-guard-16k-baseline", str(QWEN36_BASE["16k"]),
                "--g36-guard-32k-baseline", str(QWEN36_BASE["32k"]),
                "--g35-guard-128-baseline", str(QWYTHOS_BASE["128"]),
                "--g35-guard-512-baseline", str(QWYTHOS_BASE["512"]),
                "--g35-guard-4k-baseline",  str(QWYTHOS_BASE["4k"]),
                "--p-llama-128-baseline", str(QWEN36_BASE["llama128"]),
                "--p-llama-512-baseline", str(QWEN36_BASE["llama512"]),
                "--p-llama-4k-baseline",  str(QWEN36_BASE["llama4k"])]
        if PINNED_INSTANCE and not ssh_box_enabled() and str(cur_iid) == PINNED_INSTANCE:
            cmd.append("--pinned")  # never destroy the pin; retry-then-fallback on bring-up failure
        if args.polaris:
            cmd.insert(cmd.index("--keep"), "--polaris")
        pinned = "--pinned" in cmd
        box_label = ssh_box_arg() if ssh_box_enabled() else f"instance {cur_iid}"
        print(f"PR #{num} @ {oid}: evaluating '{ref}' (vs same-box main) on {box_label}"
              f"{' [pinned]' if pinned else ''} ...")
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

            # --- Polaris: parse unsigned attestation from eval box, attest it, upload with eval log ---
            polaris_line = next((l for l in r.stdout.splitlines()
                                 if l.startswith("POLARIS_ATTESTATION ")), None)
            polaris_bundle = None
            if polaris_line and res:
                try:
                    attestation = json.loads(polaris_line[len("POLARIS_ATTESTATION "):])
                    receipt = None

                    if POLARIS_API_KEY:
                        # --- TDX path: submit scoring to Polaris Intel TDX enclave ---
                        from eval.polaris.receipt import build_polaris_receipt
                        from eval.polaris.client import PolarisClient

                        # Nonce binds the attestation to this specific eval
                        nonce_input = (
                            attestation.get("code", {}).get("commit", "") +
                            attestation.get("references", {}).get("model_sha256", "") +
                            attestation.get("references", {}).get("eval_seed", "")
                        ).encode("utf-8")
                        nonce = hashlib.sha256(nonce_input).hexdigest()[:64]

                        client = PolarisClient(POLARIS_API_KEY)
                        polaris_resp = client.attest_scoring(
                            attestation.get("measurements", {}),
                            nonce,
                            POLARIS_PUBKEY,
                        )
                        receipt = build_polaris_receipt(polaris_resp, attestation)
                        print(f">> Polaris TDX: Intel verified={polaris_resp.get('tee_attestation', {}).get('verification', {}).get('intel_verified')}")

                    elif POLARIS_PRIVKEY:
                        # --- Ed25519 path: sign attestation with SparkInfer private key ---
                        from eval.polaris.receipt import build_receipt
                        receipt = build_receipt(attestation, POLARIS_PRIVKEY)
                        print(f">> Polaris Ed25519: signed with SparkInfer key")

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
            for lab in {l for l in cur if l.startswith("eval:") or l.startswith("eval-qwen")}:
                remove_label(args.repo, num, lab)
            add_label(args.repo, num, f"eval:{label}")
            if res and res.get("mode") == "bidir":
                if res.get("label_qwen35"):
                    add_label(args.repo, num, f"eval-qwen35:{res['label_qwen35']}")
                if res.get("label_qwen36"):
                    add_label(args.repo, num, f"eval-qwen36:{res['label_qwen36']}")
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
            proof = upload_eval_log(args.repo, num, pr.get("title", ""), oid, res,
                                    r.stdout + r.stderr, run_baseline, polaris=polaris_bundle)
            if proof and polaris_bundle:
                res["polaris_receipt_url"] = proof
            update_dashboard(args.repo, pr, areas, res, proof_url=proof)
            # auto-close on REJECT is DISABLED — merge-first is the only automated action.
            # Rejected PRs stay open so authors can rebase and re-submit.
        # NB: run_baseline is NOT ratcheted here — every PR is graded against merged origin/main, so
        # independent optimizations each get their true gain (the frontier advances on MERGE, not eval).

    # Per-round merge workflow: among the PRs graded this round, label the biggest verified speedup
    # `merge-first` and the rest `needs-rebase`; if a prior winner merged, flag its rivals `re-evaluate`.
    if not args.dry_run:
        reconcile_merge_labels(args.repo)

    # Stop vast instance after all PRs (bare-metal SSH boxes are left running).
    if not ssh_box_enabled():
        final_iid = current_instance(args.instance)
        if final_iid:
            print(f">> stopping instance {final_iid} — model cache persists for next run")
            subprocess.run(["vastai", "stop", "instance", str(final_iid)], capture_output=True)
    print("done — no merges (manual).")

if __name__ == "__main__":
    main()
