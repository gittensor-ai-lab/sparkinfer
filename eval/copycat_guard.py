#!/usr/bin/env python3
"""Real-time copycat guard — triggered by pull_request_target (opened).

Multi-layer copycat detection, fires the instant a PR is opened:

  LAYER 1 (containment): ≥85% → block + close
  LAYER 2 (containment): 75–84% → copycat-warn label + warning comment
  LAYER 4 (per-function) : ≥92% single-function containment → warn only (never block alone)
  3 warning strikes (any layer) → block + close

Structural similarity and LLM auto-warn are disabled (too many FPs on independent
contributors landing similar optimizations). Self-resubmissions are excluded.

Invoked by .github/workflows/copycat-guard.yml via:  PR_NUM=<num> python3 eval/copycat_guard.py
"""
import json, os, re, subprocess, sys
from datetime import date
from pathlib import Path

from copycat_policy import (
    COPYCAT_BLOCK, COPYCAT_WARN, COPYCAT_CONTAINMENT, MAX_WARNINGS,
    MIN_ADDED_LINES, LITERAL_BLOCK, FUNC_BLOCK_WARN,
    STRUCTURAL_ENABLED, LLM_ENABLED, skip_copycat_scoring,
    COPYCAT_REFERENCE_STATE,
)

REPO = os.environ.get("EVAL_REPO", "gittensor-ai-lab/sparkinfer")
ROOT = Path(__file__).resolve().parents[1]
COPYCAT_LOG = ROOT / ".github" / "copycats.json"
DENYLIST_FILE = ROOT / ".github" / "blocked-contributors.txt"
FLAG_FILE = ROOT / ".github" / "FLAGGED.md"
FLAG_LABEL = "flagged:gaming"

# Layer 3 legacy constants (only used when STRUCTURAL_ENABLED)
LEV_THRESH           = 0.70
BIGRAM_COSINE_THRESH = 0.60
STRUCT_MIN            = 0.40

# Layer 4 LLM (off by default; set COPYCAT_LLM_ENABLED=1 to re-enable)
LLM_FUNC_MIN          = 0.60
LLM_CONFIDENCE_MIN    = 0.85
LLM_BODY_MAX_CHARS    = 2000
LLM_MAX_TOKENS        = 150

def _llm_enabled():
    if os.environ.get("COPYCAT_LLM_ENABLED", "").strip().lower() in ("1", "true", "yes"):
        return True
    return LLM_ENABLED

# Provider defaults (override via COPYCAT_LLM_PROVIDER / COPYCAT_LLM_MODEL env):
#   openai   → gpt-4o-mini (default — cheap, stdlib-only, works in GHA)
#   cursor   → composer-2.5 standard tier
#   deepseek → legacy deepseek-chat
PROVIDER_DEFAULTS = {
    "cursor":   {"model": "composer-2.5", "api": ""},
    "openai":   {"model": "gpt-4o-mini",  "api": "https://api.openai.com/v1/chat/completions"},
    "deepseek": {"model": "deepseek-chat", "api": "https://api.deepseek.com/v1/chat/completions"},
}


def gh(args):
    return subprocess.run(["gh"] + args, capture_output=True, text=True)


# CUDA launch / template-instantiation one-liners converge across independent GQA PRs.
_BOILERPLATE_RE = re.compile(
    r'<<<|cudaLaunch|\w+_kernel\s*<|^\s*\w+<\d',
    re.I,
)


def is_boilerplate_block(sig, body):
    """Skip per-function scoring on shared launch/dispatch boilerplate."""
    text = f"{sig} {body}".strip()
    if _BOILERPLATE_RE.search(text) and len(body.splitlines()) <= 3:
        return True
    s = sig.strip()
    if s.startswith(("if ", "if(", "} else", "else if")):
        return True
    return False


def pr_has_label(repo, num, label):
    info = json.loads(gh(["pr", "view", str(num), "-R", repo, "--json", "labels"]).stdout or "{}")
    return any(l.get("name") == label for l in info.get("labels", []))


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
        if is_boilerplate_block(csig, cb):
            continue
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


def _llm_provider():
    """Pick LLM backend from env. Default: openai (gpt-4o-mini)."""
    explicit = os.environ.get("COPYCAT_LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    if os.environ.get("CURSOR_API_KEY", "").strip():
        return "cursor"
    if os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return "deepseek"
    return "openai"


def _llm_api_key(provider):
    keys = {
        "cursor": "CURSOR_API_KEY",
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    env = keys.get(provider, "")
    return os.environ.get(env, "").strip() if env else ""


def _llm_model(provider):
    return os.environ.get(
        "COPYCAT_LLM_MODEL",
        PROVIDER_DEFAULTS.get(provider, {}).get("model", "gpt-4o-mini"),
    ).strip()


def _build_judge_prompt(copy_func_body, orig_func_body, copy_sig, orig_sig):
    cap = LLM_BODY_MAX_CHARS
    return (
        "You are a code-copycat detector for a GPU kernel optimization contest. "
        "Reply with ONLY the three lines below — no tools, no extra text.\n\n"
        "Determine if the COPY function is substantially derived from the ORIGINAL "
        "(same computation, memory access pattern, numerical method) even if names differ.\n\n"
        f"ORIGINAL signature: {orig_sig}\n"
        f"```cpp\n{orig_func_body[:cap]}\n```\n\n"
        f"COPY signature: {copy_sig}\n"
        f"```cpp\n{copy_func_body[:cap]}\n```\n\n"
        "COPYCAT: YES|NO\n"
        "CONFIDENCE: 0.XX\n"
        "REASON: one sentence")


def _parse_judge_reply(reply):
    is_copy = "COPYCAT: YES" in reply.upper()
    conf = 0.0
    for line in reply.splitlines():
        if "CONFIDENCE:" in line.upper():
            try:
                conf = float(line.split(":")[-1].strip())
            except ValueError:
                pass
    return is_copy, conf, f"LLM judge: {reply[:180]}"


def _chat_complete_openai(api_url, api_key, model, prompt):
    import urllib.request
    req = urllib.request.Request(
        api_url,
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": LLM_MAX_TOKENS,
        }).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"].strip()


def _chat_complete_cursor(api_key, model, prompt):
    """Cursor Composer via SDK — standard tier (fast=false) for lower cost."""
    from cursor_sdk import Agent, AgentOptions, LocalAgentOptions, ModelSelection, ModelParameterValue

    selection = ModelSelection(id=model)
    if model.startswith("composer"):
        selection = ModelSelection(
            id=model,
            params=[ModelParameterValue(id="fast", value="false")],
        )
    result = Agent.prompt(
        prompt,
        AgentOptions(
            api_key=api_key,
            model=selection,
            local=LocalAgentOptions(cwd=str(ROOT)),
        ),
    )
    if result.status == "error":
        raise RuntimeError(f"cursor run failed: {getattr(result, 'id', '?')}")
    return (result.result or "").strip()


def llm_judge_copycat(copy_func_body, orig_func_body, copy_sig, orig_sig):
    """LLM judge for borderline per-function copycats. Returns (is_copy, confidence, note)."""
    provider = _llm_provider()
    api_key = _llm_api_key(provider)
    if not api_key:
        return False, 0.0, "no LLM API key configured (set OPENAI_API_KEY)"
    model = _llm_model(provider)
    prompt = _build_judge_prompt(copy_func_body, orig_func_body, copy_sig, orig_sig)
    try:
        if provider == "cursor":
            reply = _chat_complete_cursor(api_key, model, prompt)
        elif provider in ("openai", "deepseek"):
            api_url = os.environ.get(
                "COPYCAT_LLM_API",
                PROVIDER_DEFAULTS[provider]["api"],
            )
            reply = _chat_complete_openai(api_url, api_key, model, prompt)
        else:
            return False, 0.0, f"unknown COPYCAT_LLM_PROVIDER={provider!r}"
        return _parse_judge_reply(reply)
    except Exception as e:
        return False, 0.0, f"API error ({provider}/{model}): {str(e)[:120]}"


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
    """Layer 1: ≥85% containment → block + close."""
    subprocess.run(["gh", "pr", "edit", str(num), "-R", repo, "--add-label", "copycat"], capture_output=True)
    body = (f"<!-- sparkinfer-copycat -->\n## 🐈 Flagged: copycat (real-time guard)\n\n"
            f"This PR re-submits substantially the same diff (≥85% line overlap) as the earlier "
            f"#{original} by a different author. Duplicating another contributor's work is treated "
            f"as gaming the SN74 emission mechanism. The account has been **blocked** and this PR "
            f"**closed**.\n\n"
            f"See [`.github/COPYCATS.md`](../blob/main/.github/COPYCATS.md).")
    subprocess.run(["gh", "pr", "comment", str(num), "-R", repo, "--body", body], capture_output=True)


def warn_copycat(repo, num, original, author, strike_count, containment_pct, structural=False, llm_conf=0.0):
    """Layer 2/4: 75–84%, per-function, or LLM verdict → warning. Block on 3rd strike."""
    subprocess.run(["gh", "pr", "edit", str(num), "-R", repo, "--add-label", "copycat-warn"], capture_output=True)
    will_block = bool(strike_count >= MAX_WARNINGS)
    if llm_conf > 0:
        head = (f"AI semantic analysis identified a function in this PR as "
                f"substantially copied from #{original} (confidence {llm_conf:.0%}). "
                f"Combined per-function containment: {containment_pct:.0%}.")
    elif structural and containment_pct >= FUNC_BLOCK_WARN:
        head = (f"A specific function in this PR ({containment_pct:.0%} per-function containment) "
                f"is substantially contained in #{original} by a different author — the PR-level "
                f"containment is low because the copied function is embedded inside a larger diff.")
    elif structural:
        head = (f"**{containment_pct:.0f}% containment** + structural similarity "
                "(Levenshtein + bigram cosine both above threshold vs this PR's code shape)")
    else:
        head = f"**{containment_pct:.0f}% containment** in the earlier #{original}"
    if will_block:
        tail = (f"\n\nThis is the **{MAX_WARNINGS}rd** copycat-like submission — the account is now "
                "**blocked** and the PR closed.")
    else:
        tail = (f"\n\n⚠️ Warning (strike {strike_count}/{MAX_WARNINGS}). "
                f"{MAX_WARNINGS} copycat-like submissions will result in an automatic block. "
                "If this is a legitimate independent implementation, comment on this PR and a "
                "maintainer will review (label `copycat-cleared` removes the strike).")
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


def list_reference_prs(repo, limit=300):
    """PRs eligible as copycat originals — open non-draft only (not closed/merged)."""
    raw = json.loads(gh(["pr", "list", "-R", repo, "--state", COPYCAT_REFERENCE_STATE,
                         "--json", "number,author,isDraft", "--limit", str(limit)]).stdout or "[]")
    return [p for p in raw if not p.get("isDraft")]


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
    if pr_has_label(REPO, pr_num, "copycat-cleared"):
        print("  copycat-cleared label — skip"); return

    files, added = pr_fingerprint(REPO, pr_num)
    if not added:
        print("  no added lines to scan — not a copycat"); return

    open_prs = list_reference_prs(REPO)
    log = load_copycat_log()
    blocked_prs = {e["pr"] for e in log if e.get("blocked", True)}
    cleared_prs = {e["pr"] for e in log if e.get("blocked") is False}
    if pr_num in cleared_prs:
        print("  cleared in copycats.json — skip"); return
    pr_author = {p["number"]: p["author"]["login"] for p in open_prs}
    earlier_nums = sorted(p["number"] for p in open_prs if p["number"] < pr_num)
    print(f"  {len(earlier_nums)} earlier open non-draft PRs to check")

    original = None; orig_author = None; best_containment = 0.0
    pr_level_containment = 0.0
    best_lev = 0.0; best_cos = 0.0; structural_fired = False

    for e_num in earlier_nums:
        e_author = pr_author.get(e_num, "")
        if not e_author or e_author == author: continue
        if e_author.lower() in denylist: continue
        if e_num in blocked_prs: continue
        ef, ea = pr_fingerprint(REPO, e_num)
        if not (files & ef): continue
        c = containment(added, ea)
        if c > pr_level_containment:
            pr_level_containment = c
        if c > best_containment:
            original = e_num; orig_author = e_author; best_containment = c
        if c >= COPYCAT_BLOCK:
            break
        if STRUCTURAL_ENABLED and c >= STRUCT_MIN and c < COPYCAT_WARN and not structural_fired:
            lev, cos, hit = structural_similarity(REPO, pr_num, e_num, c)
            if hit:
                structural_fired = True; best_lev = lev; best_cos = cos
                if not original or best_containment < COPYCAT_WARN:
                    original = e_num; orig_author = e_author; best_containment = c
                print(f"  structural copycat: lev={lev:.2f} cos={cos:.2f} c={c:.2f} vs #{e_num}")

        # Layer 4: per-function containment (near-verbatim kernel inside larger PR).
        if c < COPYCAT_WARN and not structural_fired:
            func_c, func_csig, func_osig = per_function_containment(REPO, pr_num, e_num)
            if func_c >= FUNC_BLOCK_WARN:
                structural_fired = True; best_lev = func_c; best_cos = -1.0
                best_containment = max(best_containment, COPYCAT_WARN)
                original = e_num; orig_author = e_author
                print(f"  per-function bump: {func_csig[:60]}... is {func_c:.0%} contained in #{e_num} -> WARN")
            elif _llm_enabled() and func_c >= LLM_FUNC_MIN:
                print(f"  layer 4 LLM: per-function containment={func_c:.1%} (vs #{e_num})")
                cb = next((b for s, b in split_into_blocks(REPO, pr_num) if s == func_csig), "")
                ob = next((b for s, b in split_into_blocks(REPO, e_num) if s == func_osig), "")
                is_copy, llm_conf, reason = llm_judge_copycat(cb, ob, func_csig, func_osig)
                print(f"  LLM: copycat={is_copy} confidence={llm_conf:.2f} reason={reason[:120]}")
                if is_copy and llm_conf >= LLM_CONFIDENCE_MIN:
                    structural_fired = True; best_lev = llm_conf; best_cos = 0.0
                    best_containment = max(best_containment, func_c, COPYCAT_WARN)
                    if not original or func_c > (best_containment if original == e_num else 0):
                        original = e_num; orig_author = e_author
                    print(f"  LLM verdict: COPYCAT CONFIRMED -> bumping to WARN vs #{e_num}")

    if original is None or (best_containment < COPYCAT_WARN and not structural_fired):
        print("  no copycat detected — clean"); return

    if skip_copycat_scoring(added, best_containment):
        print(f"  only {len(added)} added lines (<{MIN_ADDED_LINES}) and "
              f"containment {best_containment:.0%} < {LITERAL_BLOCK:.0%} — skip"); return

    is_block = (pr_level_containment >= COPYCAT_BLOCK)
    if structural_fired and not is_block:
        best_containment = max(best_containment, COPYCAT_WARN)
        print(f"  bumping to WARN: per-function/structural (PR-level {pr_level_containment:.0%})")

    if is_block:
        print(f"  COPYCAT ≥85%: #{pr_num} is {best_containment:.1%} contained in #{original} by {orig_author}")
        flag_copycat(REPO, pr_num, original, author)
        log.append({"pr": pr_num, "author": author, "original": original,
                    "date": date.today().isoformat(), "blocked": True})
        save_copycat_log(log)
        block_account(author, f"#{pr_num} ≥85% copycat of #{original} ({best_containment:.0%})")
        close_blocked_pr(REPO, pr_num, {author})
        print("  block + close done")
    else:
        warn_strikes = sum(1 for e in log
                           if e.get("author") == author and not e.get("blocked", True)
                           and int(e.get("penalty_days", PENALTY_DAYS)) != 0)
        strike = warn_strikes + 1
        tag = "LLM" if (structural_fired and best_cos == 0.0) else ("func" if structural_fired else "containment")
        print(f"  COPYCAT WARN ({tag}): #{pr_num} vs #{original} (strike {strike}/{MAX_WARNINGS})")
        is_llm = structural_fired and best_cos == 0.0
        llm_conf_val = best_lev if is_llm else 0.0
        will_block = warn_copycat(REPO, pr_num, original, author, strike, best_containment, structural_fired, llm_conf_val)
        log.append({"pr": pr_num, "author": author, "original": original,
                    "date": date.today().isoformat(), "blocked": False,
                    "penalty_days": 0, "strike": strike,
                    "containment": round(best_containment, 3)})
        save_copycat_log(log)
        if will_block:
            block_account(author, f"{MAX_WARNINGS} copycat strikes: #{pr_num} ({best_containment:.0%} of #{original})")
            close_blocked_pr(REPO, pr_num, {author})

    push_policy_files()


if __name__ == "__main__":
    main()
