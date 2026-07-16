"""Shared copycat detection thresholds — used by copycat_guard, pr_eval_bot, copycat_sweep."""

# PR-level containment (shared file + added-line overlap required)
COPYCAT_BLOCK = 0.85   # ≥85% → block + close
COPYCAT_WARN = 0.75    # 75–84% → copycat-warn (no close on first hits)
MAX_WARNINGS = 3       # warns before auto-block

# Small PR guard — avoids ratio explosions on tiny diffs
MIN_ADDED_LINES = 15
LITERAL_BLOCK = 0.98   # still block tiny PRs if ≥98% literal

# Per-function dilution catch (near-verbatim kernel embedded in larger PR).
# WARN only — never escalates to block on its own (PR-level containment must be ≥85%).
FUNC_BLOCK_WARN = 0.92

# Structural (Levenshtein + bigram) layer disabled — too many FPs on independent
# contributors converging on the same optimization pattern.
STRUCTURAL_ENABLED = False

# LLM semantic judge disabled by default — re-enable with COPYCAT_LLM_ENABLED=1.
LLM_ENABLED = False

# Back-compat alias
COPYCAT_CONTAINMENT = COPYCAT_BLOCK

# Copycat reference pool: only still-open PRs (excludes closed and merged).
COPYCAT_REFERENCE_STATE = "open"


def skip_copycat_scoring(added_lines, containment):
    """Skip borderline scoring on tiny PRs unless near-literal copy."""
    n = len(added_lines)
    if n >= MIN_ADDED_LINES:
        return False
    if n >= 3 and containment >= LITERAL_BLOCK:
        return False
    return True
