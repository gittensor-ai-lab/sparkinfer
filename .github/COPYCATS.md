# Copycat detection & history

A **copycat** PR re-submits substantially the same diff as an earlier PR (often an already-merged
one) to farm credit for someone else's work. Detection runs in two places:

- **Real-time guard** (`eval/copycat_guard.py` + `.github/workflows/copycat-guard.yml`) — fires when a PR is opened.
- **Eval bot** (`eval/pr_eval_bot.py`) — re-checks during scheduled evaluation.

Shared thresholds live in `eval/copycat_policy.py`.

## How it works

- PRs are compared **oldest-first** (ascending PR number), so the original is always seen before any copy.
- Fingerprint = (changed files, normalized non-comment added lines).
- Compared against **every earlier open PR** by a **different author** that touches the same file(s)
  (closed and merged PRs are **not** used as copycat references).
- **Self-resubmissions** (same author iterating on their own earlier PR) are **not** copycats.

### Tiered policy

| Containment | Action |
|-------------|--------|
| **≥ 85%** | Label [`copycat`](../../labels/copycat), **block account**, **close PR**, skip eval |
| **75–84%** | Label [`copycat-warn`](../../labels/copycat-warn), warning comment, **skip eval** |
| **&lt; 75%** | Pass (no copycat label) |

- **3 warnings** (`copycat-warn`, any PR) within the log → auto-block + close.
- Maintainers can clear a false positive with label `copycat-cleared` (manual). Cleared PRs are logged in
  `copycats.json` with `"blocked": false, "penalty_days": 0` and are **skipped** by both guards.
- **Tiny PRs** (&lt; 15 added lines) are skipped unless **≥ 98%** literal overlap.
- **Per-function check**: a single CUDA function ≥ **92%** contained in an earlier PR → **warn only**
  (never block on per-function alone). CUDA launch / template-instantiation boilerplate is excluded.
  **Block requires PR-level containment ≥ 85%.**
- **Structural similarity** and **LLM auto-warn** are **disabled** by default (too many false positives when independent contributors land similar optimizations).

Blocked accounts are listed in [`blocked-contributors.txt`](./blocked-contributors.txt) and logged in [`FLAGGED.md`](./FLAGGED.md).

The machine-readable log is [`copycats.json`](./copycats.json) — one entry per detection
`{pr, author, original, date, blocked?, strike?, containment?}`.

## History

| date | copycat PR | author | copied from | note |
|------|-----------|--------|-------------|------|
| 2026-06-25 | #14 | `glorysr1209-png` | #4 (`galuis116`) | flash_prefill mask; identical 1-line diff. Account already blocked as sybil. |
| 2026-06-25 | #9  | `glorysr1209-png` | #6 (`galuis116`) | gguf metadata desync; 7/8 added lines identical. Account already blocked as sybil. |
| 2026-06-25 | #54 | `kiannidev` | #53 (`James-CUDA`) | maintainer-flagged: same decode change as #53. Below auto threshold at the time; strike 1. **2-day penalty** (first-time contributor leniency). |
| 2026-07-10 | #326 | `inference2026` | #195 (`fansilas`) | **false positive (cleared).** Per-function template-launch line matched at 100%; PR-level overlap only 27%. `blocked: false`, `penalty_days: 0`. |
| 2026-07-10 | #338 | `inference2026` | #318 (`Paral1995`) | **false positive (cleared).** 3-line GQA-4 `g2` build fix; 100% literal overlap with #318 call-site boilerplate. Same diff as #336 (merged first). `blocked: false`, `penalty_days: 0`. |

> `glorysr1209-png` also opened #13 and #15 (same bug-clusters as #11 / #12) but with different
> code — not literal copies, so not logged here; they were closed under the sybil block instead.
