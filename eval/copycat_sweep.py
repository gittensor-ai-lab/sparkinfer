#!/usr/bin/env python3
"""One-shot copycat sweep — run the 4-layer detection against ALL existing open PRs.

Reads every open non-draft PR, fingerprints each, and for every pair of different
authors touching shared files, runs the full layer 1/2/3/4 pipeline. Reports
findings to stdout; labels/comments/blocks only if --apply is passed.

Usage: python3 eval/copycat_sweep.py [--apply]
"""
import json, os, subprocess, sys
from datetime import date
from copycat_guard import *

REPO = os.environ.get("EVAL_REPO", "gittensor-ai-lab/sparkinfer")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


def sweep(apply=False):
    """Run full copycat detection across all open non-draft PRs."""
    # 1) Fetch all open non-draft PRs
    open_prs = json.loads(gh(["pr", "list", "-R", REPO, "--state", "open",
                               "--json", "number,author,isDraft,title", "--limit", "100"]).stdout or "[]")
    open_prs = [p for p in open_prs if not p["isDraft"]]
    all_nums = sorted(p["number"] for p in open_prs)
    pr_author = {p["number"]: p["author"]["login"] for p in open_prs}
    pr_title  = {p["number"]: p["title"] for p in open_prs}

    print(f"copycat-sweep: {len(open_prs)} open non-draft PRs ({min(all_nums)}–{max(all_nums)})")
    print(f"  checking {len(all_nums)*(len(all_nums)-1)//2} pairs across {len(set(pr_author.values()))} authors")
    print()

    denylist = load_denylist()
    log = load_copycat_log()
    already_flagged = {e["pr"] for e in log if e.get("blocked", True)}
    warned_prs = {e["pr"] for e in log if not e.get("blocked", True)}

    findings = []

    # 2) For each PR (newer), compare against all EARLIER PRs by different authors
    for pr_num in all_nums:
        author = pr_author.get(pr_num, "?").lower()
        if author in denylist:
            continue  # already blocked
        files, added = pr_fingerprint(REPO, pr_num)
        if not added:
            continue

        for e_num in [n for n in all_nums if n < pr_num]:
            e_author = pr_author.get(e_num, "?").lower()
            if not e_author or e_author == author:
                continue
            if e_author in denylist or e_num in already_flagged:
                continue
            ef, ea = pr_fingerprint(REPO, e_num)
            if not (files & ef):
                continue

            c = containment(added, ea)
            detection = None  # which layer caught it
            confidence = c
            details = {}

            # L1/L2: containment
            if c >= COPYCAT_CONTAINMENT:
                detection = "L1: ≥80% containment"
            elif c >= COPYCAT_WARN:
                detection = "L2: 70-79% containment"

            # L3: structural
            if not detection and c >= STRUCT_MIN:
                lev, cos, hit = structural_similarity(REPO, pr_num, e_num, c)
                if hit:
                    detection = f"L3: structural (lev={lev:.2f} cos={cos:.2f})"
                    details = {"lev": round(lev,3), "cos": round(cos,3)}

            # L4: per-function
            if not detection:
                func_c, func_csig, func_osig = per_function_containment(REPO, pr_num, e_num)
                if func_c >= COPYCAT_WARN:
                    detection = f"L4a: per-function {func_c:.0%} (PR-level only {c:.0%})"
                    details = {"func_c": round(func_c,3), "func_sig": func_csig[:80]}
                elif func_c >= LLM_FUNC_MIN:
                    cb = next((b for s,b in split_into_blocks(REPO,pr_num) if s==func_csig), "")
                    ob = next((b for s,b in split_into_blocks(REPO,e_num) if s==func_osig), "")
                    is_copy, llm_c, reason = llm_judge_copycat(cb, ob, func_csig, func_osig)
                    if is_copy and llm_c >= LLM_CONFIDENCE_MIN:
                        detection = f"L4b: LLM judge (confidence={llm_c:.0%}, func_c={func_c:.0%})"
                        details = {"func_c": round(func_c,3), "llm_conf": round(llm_c,3), "reason": reason[:120]}

            if detection:
                is_block = detection.startswith("L1")  # >=80% = block, rest = warn
                findings.append((pr_num, e_num, author, e_author, c, detection, is_block, details))
                break  # one finding per PR — the strongest match

    # 3) Report
    if not findings:
        print("No copycats detected across any open PR pair.")
        return

    blocks = [f for f in findings if f[6]]
    warns  = [f for f in findings if not f[6]]
    print(f"{'=' * 70}")
    print(f"FOUND: {len(findings)} copycat(s) — {len(blocks)} block, {len(warns)} warn")
    print(f"{'=' * 70}")

    for pr, orig, auth, oauth, c, det, is_block, dets in findings:
        icon = "BLOCK" if is_block else "WARN"
        print(f"\n  #{pr} ({auth}) vs #{orig} ({oauth}) — {det}")
        print(f"    #{pr}: {pr_title[pr][:90]}")
        print(f"    #{orig}: {pr_title[orig][:90]}")
        if dets.get("func_sig"): print(f"    function: {dets['func_sig']}")
        if dets.get("reason"):   print(f"    LLM: {dets['reason']}")
        already = "already warned" if pr in warned_prs else "not yet flagged"
        print(f"    action: {icon} ({already})")

    if not apply:
        print(f"\n--- dry-run: re-run with --apply to label/comment/block/close ---")
        return

    # 4) Apply actions
    print(f"\n--- applying actions ---")
    for pr, orig, auth, oauth, c, det, is_block, dets in findings:
        if pr in already_flagged or pr in warned_prs:
            print(f"  #{pr}: already flagged — skip"); continue

        strike = sum(1 for e in log if e.get("author","").lower() == auth and not e.get("blocked",True)) + 1

        if is_block:
            print(f"  #{pr} >=80% -> block+close")
            flag_copycat(REPO, pr, orig, auth)
            log.append({"pr":pr,"author":auth,"original":orig,"date":date.today().isoformat(),"blocked":True})
            save_copycat_log(log)
            block_account(auth, f"Sweep: #{pr} >=80% copycat of #{orig} ({c:.0%})")
            close_blocked_pr(REPO, pr, {auth})
        else:
            llm_conf = dets.get("llm_conf", 0.0)
            is_struct = "L3" in det or "L4" in det
            print(f"  #{pr} -> copycat-warn (strike {strike})")
            will_block = warn_copycat(REPO, pr, orig, auth, strike, c, is_struct, llm_conf)
            log.append({"pr":pr,"author":auth,"original":orig,"date":date.today().isoformat(),
                        "blocked":False,"penalty_days":0,"strike":strike,"containment":round(c,3)})
            save_copycat_log(log)
            if will_block:
                block_account(auth, f"2nd copycat strike (sweep): #{pr} (vs #{orig})")
                close_blocked_pr(REPO, pr, {auth})
                print(f"    -> 2nd strike -> block+close")

    push_policy_files()
    print("done")


if __name__ == "__main__":
    apply_flag = "--apply" in sys.argv
    sweep(apply=apply_flag)
