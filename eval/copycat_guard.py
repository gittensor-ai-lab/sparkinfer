#!/usr/bin/env python3
"""Real-time copycat guard — triggered by pull_request_target (opened).

Multi-layer copycat detection, fires the instant a PR is opened:

  LAYER 1 (containment): ≥80% → instant block + close (zero tolerance)
  LAYER 2 (containment): 70–79% → copycat-warn label + warning comment
  LAYER 3 (structural) : 40–69% with lev≥0.70+cos≥0.60 → bump to WARN
  2 warning strikes (any layer) → block + close (just like ≥80%)

Self-resubmissions (same author iterating on own earlier PR) are excluded.
Invoked by .github/workflows/copycat-guard.yml via:  PR_NUM=<num> python3 eval/copycat_guard.py
"""
import json, os, subprocess, sys
from datetime import date
from pathlib import Path

REPO = os.environ.get("EVAL_REPO", "gittensor-ai-lab/sparkinfer")
ROOT = Path(__file__).resolve().parents[1]
COPYCAT_LOG = ROOT / ".github" / "copycats.json"
DENYLIST_FILE = ROOT / ".github" / "blocked-contributors.txt"
FLAG_FILE = ROOT / ".github" / "FLAGGED.md"
FLAG_LABEL = "flagged:gaming"

# Detection thresholds
COPYCAT_CONTAINMENT = 0.80   # layer 1: ≥80% → instant block + close
COPYCAT_WARN         = 0.70   # layer 2: 70–79% → warning
MAX_WARNINGS         = 2      # block on this many warnings across any PRs
LEV_THRESH           = 0.70   # layer 3: token Levenshtein ratio ≥ this
BIGRAM_COSINE_THRESH = 0.60   # layer 3: bigram cosine similarity ≥ this
STRUCT_MIN            = 0.40   # layer 3: containment must be at least this to trigger
# Layer 4: LLM judge (DeepSeek) — catches per-function copycats diluted by larger PR
LLM_FUNC_MIN          = 0.30   # per-function containment must be ≥30% to escalate
LLM_CONFIDENCE_MIN    = 0.70   # LLM must be ≥70% confident to flag
DEEPSEEK_API          = "https://api.deepseek.com/v1/chat/completions"


def gh(args):
    return subprocess.run(["gh"] + args, capture_output=True, text=True)


# ---- fingerprinting (both layers) ----

def pr_fingerprint(repo, num):
    """(changed files, normalized non-empty added lines) — for layer 1/2 containment."""
    diff = gh(["pr", "diff", str(num), "-R", repo]).stdout or ""
    files, added = set(), set()
    for line in diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            p = line[4:].strip()
            if p.startswith(("a/", "b/")): p = p[2:]
            if p and p != "/dev/null": files.add(p)
        elif line.startswith("+") and not line.startswith("+++"):
            s = line[1:].strip()
            if s and not s.startswith(("//", "#", "/*", "*")): added.add(s)
    return files, added


def containment(copy_added, orig_added):
    if not copy_added: return 0.0
    return len(copy_added & orig_added) / len(copy_added)


def pr_added_tokens(repo, num):
    """All non-comment tokens from added lines — for layer 3 structural comparison."""
    diff = gh(["pr", "diff", str(num), "-R", repo]).stdout or ""
    tokens = []
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            s = line[1:].strip()
            if s and not s.startswith(("//", "#", "/*", "*")):
                tokens.extend(t for t in s.split() if len(t) > 1 and not t.startswith("//"))
    return tokens


# ---- layer 3: structural similarity ----

def levenshtein_ratio(tokens_a, tokens_b):
    """Token-level edit similarity (0–1) via difflib SequenceMatcher."""
    if not tokens_a or not tokens_b:
        return 0.0
    import difflib
    return difflib.SequenceMatcher(None, tokens_a, tokens_b).ratio()


def bigram_cosine(tokens_a, tokens_b):
    """Cosine similarity (0–1) of bigram frequency vectors."""
    from collections import Counter
    if len(tokens_a) < 2 or len(tokens_b) < 2:
        return 0.0
    bg_a = Counter(zip(tokens_a, tokens_a[1:]))
    bg_b = Counter(zip(tokens_b, tokens_b[1:]))
    common = set(bg_a) & set(bg_b)
    if not common:
        return 0.0
    dot = sum(bg_a[k] * bg_b[k] for k in common)
    norm_a = sum(v * v for v in bg_a.values()) ** 0.5
    norm_b = sum(v * v for v in bg_b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def structural_similarity(repo, copy_num, orig_num, containment_pct):
    """Compute lev + bigram-cos for a borderline candidate. Returns (lev, cos, fired).
    fired=True when containment >= 40% AND lev >= 0.70 AND cos >= 0.60."""
    if containment_pct < STRUCT_MIN:
        return 0.0, 0.0, False
    tok_copy = pr_added_tokens(repo, copy_num)
    tok_orig = pr_added_tokens(repo, orig_num)
    if not tok_copy or not tok_orig:
        return 0.0, 0.0, False
    lev = levenshtein_ratio(tok_copy, tok_orig)
    cos = bigram_cosine(tok_copy, tok_orig)
    fired = lev >= LEV_THRESH and cos >= BIGRAM_COSINE_THRESH
    return lev, cos, fired


# ---- layer 4: function-block containment + LLM judge (catches dilution) ----

def split_into_blocks(repo, num):
    """Split a PR's added lines into logical CUDA code blocks (kernel functions, device
    functions, and other named scopes). Filters out trivial if/else conditionals that
    happen to match across PRs but aren't actual functions (minimum 10 tokens)."""
    diff = gh(["pr", "diff", str(num), "-R", repo]).stdout or ""
    blocks = []
    current_sig = None; current_body = []
    for line in diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        s = line[1:].strip()
        if not s or s.startswith(("//", "#", "/*", "*")):
            continue
        # CUDA function boundaries ONLY — not flow control (if/for/while). Detection uses
        # __global__/__device__/__host__/template or a return-type signature (void/float/etc name(...)).
        is_cuda = any(kw in s for kw in (
            "__global__", "__device__", "__host__", "template <", "template<"))
        # Skip flow-control keywords — "if (", "for (", "while (", "else if" are NOT functions.
        is_flow = any(s.strip().startswith(kw) for kw in (
            "if ", "if(", "for ", "for(", "while ", "while(", "else ", "switch ", "switch(",
            "} else ", "} else if"))
        is_func_sig = not is_flow and s.endswith("{") and "(" in s
        if is_cuda or is_func_sig:
            if current_sig and current_body:
                tokens = [t for l in current_body for t in l.split() if len(t)>1]
                if len(tokens) >= 10:
                    blocks.append((current_sig, "\n".join(current_body)))
            current_sig = s
            current_body = []
        elif current_sig is not None:
            current_body.append(line[1:])
    if current_sig and current_body:
        tokens = [t for l in current_body for t in l.split() if len(t)>1]
        if len(tokens) >= 10:
            blocks.append((current_sig, "\n".join(current_body)))
    return blocks


def per_function_containment(repo, copy_num, orig_num):
    """Return the HIGHEST per-function containment across all shared blocks. If the original
    PR has one function (focused change) and the copy PR adds it inside a larger PR, this
    will catch it even when PR-level containment is low."""
    copy_blocks = split_into_blocks(repo, copy_num)
    orig_blocks = split_into_blocks(repo, orig_num)
    if not copy_blocks or not orig_blocks:
        return 0.0, "", ""
    best = 0.0; best_copy_sig = ""; best_orig_sig = ""
    for csig, cb in copy_blocks:
        ctokens = set(cb.split())
        if len(ctokens) < 5:
            continue
        for osig, ob in orig_blocks:
            otokens = set(ob.split())
            if len(otokens) < 5:
                continue
            if not (ctokens & otokens):
                continue
            c = len(ctokens & otokens) / len(ctokens)
            if c > best:
                best = c; best_copy_sig = csig; best_orig_sig = osig
    return best, best_copy_sig, best_orig_sig


def llm_judge_copycat(copy_func_body, orig_func_body, copy_sig, orig_sig):
    """Call DeepSeek API to judge whether two code blocks are substantially the same.
    Returns (is_copy: bool, confidence: float, explanation: str)."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return False, 0.0, "no API key configured"
    import urllib.request
    prompt = (
        "You are a code-copycat detector for a GPU kernel optimization contest. "
        "Two contributors submitted CUDA kernels. Determine if the COPY function is "
        "substantially derived from the ORIGINAL function (same computation, same memory "
        "access pattern, same numerical method) even if variable names or minor structure differ.\n\n"
        f"ORIGINAL function signature: {orig_sig}\n"
        f"```cpp\n{orig_func_body[:3000]}\n```\n\n"
        f"COPY function signature: {copy_sig}\n"
        f"```cpp\n{copy_func_body[:3000]}\n```\n\n"
        "Answer in this exact format:\n"
        "COPYCAT: YES|NO\n"
        "CONFIDENCE: 0.XX\n"
        "REASON: one sentence explaining why")
    try:
        req = urllib.request.Request(
            DEEPSEEK_API,
            data=json.dumps({
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0, "max_tokens": 200
            }).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        reply = body["choices"][0]["message"]["content"].strip()
        # parse
        is_copy = "COPYCAT: YES" in reply.upper()
        conf = 0.0
        for line in reply.splitlines():
            if "CONFIDENCE:" in line.upper():
                try: conf = float(line.split(":")[-1].strip())
                except: pass
        reason = ""
        for line in reply.splitlines():
            if "REASON:" in line.upper():
                reason = line.split(":", 1)[-1].strip()
                break
        return is_copy, conf, f"LLM judge: {reply[:180]}"
    except Exception as e:
        return False, 0.0, f"API error: {str(e)[:120]}"


# ---- policy state management ----

def load_denylist():
    try:
        out = set()
        for line in open(DENYLIST_FILE):
            s = line.split("#", 1)[0].strip().lower()
            if s: out.add(s)
        return out
    except Exception: return set()

def load_copycat_log():
    try: return json.load(open(COPYCAT_LOG))
    except Exception: return []

def save_copycat_log(log):
    COPYCAT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(COPYCAT_LOG, "w") as f: json.dump(log, f, indent=2)

def block_account(login, reason):
    cur = load_denylist()
    if login.lower() not in cur:
        with open(DENYLIST_FILE, "a") as f: f.write(f"\n{login}\n")
    with open(FLAG_FILE, "a") as f:
        f.write(f"\n## {date.today().isoformat()} — `{login}` (auto-blocked)\n\n{reason}\n")

def push_policy_files():
    subprocess.run(["git", "-C", str(ROOT), "add",
                    ".github/copycats.json", ".github/blocked-contributors.txt", ".github/FLAGGED.md"],
                   capture_output=True)
    if subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"]).returncode != 0:
        subprocess.run(["git", "-C", str(ROOT), "commit", "-q",
                        "-m", "copycat-guard: policy file update"], capture_output=True)
        subprocess.run(["git", "-C", str(ROOT), "pull", "-q", "--rebase", "origin", "main"],
                       capture_output=True)
        subprocess.run(["git", "-C", str(ROOT), "push", "-q", "origin", "main"], capture_output=True)


# ---- PR actions ----

def flag_copycat(repo, num, original, author):
    """Layer 1: ≥80% containment → block + close."""
    subprocess.run(["gh", "pr", "edit", str(num), "-R", repo, "--add-label", "copycat"], capture_output=True)
    body = (f"<!-- sparkinfer-copycat -->\n## 🐈 Flagged: copycat (real-time guard)\n\n"
            f"This PR re-submits substantially the same diff as the earlier #{original} by "
            f"a different author. Duplicating another contributor's work is treated as gaming "
            f"the SN74 emission mechanism. The account has been **blocked** and this PR "
            f"**closed** — zero tolerance, no warning.\n\n"
            f"See [`.github/COPYCATS.md`](../blob/main/.github/COPYCATS.md).")
    subprocess.run(["gh", "pr", "comment", str(num), "-R", repo, "--body", body], capture_output=True)


def warn_copycat(repo, num, original, author, strike_count, containment_pct, structural=False, llm_conf=0.0):
    """Layer 2/3/4: 70–79%, structural flip, or LLM verdict → warning. Block on 2nd strike."""
    subprocess.run(["gh", "pr", "edit", str(num), "-R", repo, "--add-label", "copycat-warn"], capture_output=True)
    will_block = bool(strike_count >= MAX_WARNINGS)
    if llm_conf > 0:
        head = (f"AI semantic analysis (DeepSeek) identified a function in this PR as "
                f"substantially copied from #{original} (confidence {llm_conf:.0%}). "
                f"Combined per-function containment: {containment_pct:.0%}.")
    elif structural and containment_pct >= COPYCAT_WARN:
        head = (f"A specific function in this PR ({containment_pct:.0%} per-function containment) "
                f"is substantially contained in #{original} by a different author — the PR-level "
                f"containment is low because the copied function is embedded inside a larger diff.")
    elif structural:
        head = (f"**{containment_pct:.0f}% containment** + structural similarity "
                "(Levenshtein + bigram cosine both above threshold vs this PR's code shape)")
    else:
        head = f"**{containment_pct:.0f}% containment** in the earlier #{original}"
    if will_block:
        tail = ("\n\nThis is the **second** copycat-like submission — the account is now "
                "**blocked** and the PR closed.")
    else:
        tail = (f"\n\n⚠️ Warning (strike {strike_count}/{MAX_WARNINGS}). A second copycat-like "
                "submission will result in an automatic block. If this is a legitimate independent "
                "implementation, comment on this PR and a maintainer will review.")
    body = (f"<!-- sparkinfer-copycat-warn -->\n## 🐈 Copycat warning (real-time guard)\n\n"
            f"{head} by a different author.{tail}")
    subprocess.run(["gh", "pr", "comment", str(num), "-R", repo, "--body", body], capture_output=True)
    if will_block:
        close_blocked_pr(repo, num, {author})
    return will_block


def close_blocked_pr(repo, num, hits):
    subprocess.run(["gh", "pr", "edit", str(num), "-R", repo, "--add-label", FLAG_LABEL], capture_output=True)
    who = ", ".join(f"`{h}`" for h in sorted(hits))
    body = ("<!-- sparkinfer-flagged -->\n## 🚩 Flagged: eval-gaming\n\n"
            f"Blocked account(s) for gaming the SN74 emission mechanism (sybil / coordinated "
            f"duplicate farming): {who}. The PR is **not evaluated, scored, or merged**.\n\n"
            f"See [`.github/FLAGGED.md`](../blob/main/.github/FLAGGED.md).")
    subprocess.run(["gh", "pr", "comment", str(num), "-R", repo, "--body", body], capture_output=True)
    return subprocess.run(["gh", "pr", "close", str(num), "-R", repo]).returncode == 0


def pr_author_login(repo, num):
    info = json.loads(gh(["pr", "view", str(num), "-R", repo, "--json", "author"]).stdout or "{}")
    return (info.get("author") or {}).get("login", "")


# ---- main ----

def main():
    pr_num = int(os.environ.get("PR_NUM") or 0)
    if not pr_num:
        print("PR_NUM not set — nothing to guard"); return
    author = pr_author_login(REPO, pr_num)
    print(f"copycat-guard: PR #{pr_num} by {author} — scanning for copycat ...")

    denylist = load_denylist()
    if author.lower() in denylist:
        print(f"  author {author} already in denylist — skip"); return

    files, added = pr_fingerprint(REPO, pr_num)
    if not added:
        print("  no added lines to scan — not a copycat"); return

    open_prs = json.loads(gh(["pr", "list", "-R", REPO, "--state", "open",
                               "--json", "number,author,isDraft", "--limit", "100"]).stdout or "[]")
    log = load_copycat_log()
    blocked_prs = {e["pr"] for e in log}
    earlier_nums = sorted(p["number"] for p in open_prs if p["number"] < pr_num and not p["isDraft"])
    print(f"  {len(earlier_nums)} earlier open non-draft PRs to check")

    original = None; orig_author = None; best_containment = 0.0
    best_lev = 0.0; best_cos = 0.0; structural_fired = False

    for e_num in earlier_nums:
        e_author = next((p["author"]["login"] for p in open_prs if p["number"] == e_num), "")
        if not e_author or e_author == author: continue
        if e_author.lower() in denylist: continue
        if e_num in blocked_prs: continue
        ef, ea = pr_fingerprint(REPO, e_num)
        if not (files & ef): continue
        c = containment(added, ea)
        if c > best_containment:
            original = e_num; orig_author = e_author; best_containment = c
        if c >= COPYCAT_CONTAINMENT:
            break
        if c >= STRUCT_MIN and c < COPYCAT_WARN and not structural_fired:
            lev, cos, hit = structural_similarity(REPO, pr_num, e_num, c)
            if hit:
                structural_fired = True; best_lev = lev; best_cos = cos
                if not original or best_containment < COPYCAT_WARN:
                    original = e_num; orig_author = e_author; best_containment = c
                print(f"  structural copycat: lev={lev:.2f} cos={cos:.2f} c={c:.2f} vs #{e_num}")

        # 4b) Layer 4: per-function containment + LLM judge. The copier's evasion tactic is to
        #     embed a verbatim-copied kernel inside a much larger PR to dilute the PR-level ratio.
        #     4b-i:  ANY single function >=70% contained → deterministic bump to WARN (no API).
        #     4b-ii: 30-69% → escalate to LLM judge for semantic verification.
        if c < COPYCAT_WARN and not structural_fired:
            func_c, func_csig, func_osig = per_function_containment(REPO, pr_num, e_num)
            if func_c >= COPYCAT_WARN:
                # Deterministic: a single copied function is >=70% contained even though the PR
                # as a whole appears different (dilution). Bump to WARN — same tier as layer 2,
                # but the comment surfaces the function-level evidence.
                structural_fired = True; best_lev = func_c; best_cos = -1.0  # cos=-1 = "func-level"
                best_containment = max(best_containment, func_c, COPYCAT_WARN)
                if not original or func_c > (best_containment if original==e_num else 0):
                    original = e_num; orig_author = e_author
                print(f"  per-function bump: {func_csig[:60]}... is {func_c:.0%} contained in #{e_num} -> WARN")
            elif func_c >= LLM_FUNC_MIN:
                print(f"  layer 4: per-function containment={func_c:.1%} (vs #{e_num}) -> escalating to LLM")
                cb = next((b for s,b in split_into_blocks(REPO,pr_num) if s==func_csig), "")
                ob = next((b for s,b in split_into_blocks(REPO,e_num) if s==func_osig), "")
                is_copy, llm_conf, reason = llm_judge_copycat(cb, ob, func_csig, func_osig)
                print(f"  LLM: copycat={is_copy} confidence={llm_conf:.2f} reason={reason[:120]}")
                if is_copy and llm_conf >= LLM_CONFIDENCE_MIN:
                    structural_fired = True; best_lev = llm_conf; best_cos = 0.0
                    best_containment = max(best_containment, func_c, COPYCAT_WARN)
                    if not original or func_c > (best_containment if original==e_num else 0):
                        original = e_num; orig_author = e_author
                    print(f"  LLM verdict: COPYCAT CONFIRMED -> bumping to WARN vs #{e_num}")

    if original is None or (best_containment < COPYCAT_WARN and not structural_fired):
        print("  no copycat detected — clean"); return

    is_block = (best_containment >= COPYCAT_CONTAINMENT)
    if structural_fired and not is_block:
        best_containment = max(best_containment, COPYCAT_WARN)
        print(f"  bumping to WARN: structural lev={best_lev:.2f} cos={best_cos:.2f}")

    if is_block:
        print(f"  COPYCAT ≥80%: #{pr_num} is {best_containment:.1%} contained in #{original} by {orig_author}")
        flag_copycat(REPO, pr_num, original, author)
        log.append({"pr": pr_num, "author": author, "original": original,
                    "date": date.today().isoformat(), "blocked": True})
        save_copycat_log(log)
        block_account(author, f"#{pr_num} ≥80% copycat of #{original} ({best_containment:.0%})")
        close_blocked_pr(REPO, pr_num, {author})
        print("  block + close done")
    else:
        warn_strikes = sum(1 for e in log if e.get("author") == author and not e.get("blocked", True))
        strike = warn_strikes + 1
        tag = "LLM" if (structural_fired and best_cos == 0.0) else ("structural" if structural_fired else "containment")
        print(f"  COPYCAT WARN ({tag}): #{pr_num} vs #{original} (strike {strike}/{MAX_WARNINGS})")
        is_llm = structural_fired and best_cos == 0.0      # LLM judge (cos=0 sentinel)
        is_func = structural_fired and best_cos == -1.0     # per-function deterministic (cos=-1)
        llm_conf_val = best_lev if is_llm else 0.0
        will_block = warn_copycat(REPO, pr_num, original, author, strike, best_containment, structural_fired, llm_conf_val)
        log.append({"pr": pr_num, "author": author, "original": original,
                    "date": date.today().isoformat(), "blocked": False,
                    "penalty_days": 0, "strike": strike,
                    "containment": round(best_containment, 3)})
        save_copycat_log(log)
        if will_block:
            block_account(author, f"2nd copycat strike: #{pr_num} ({best_containment:.0%} of #{original})")
            close_blocked_pr(REPO, pr_num, {author})

    push_policy_files()


if __name__ == "__main__":
    main()
