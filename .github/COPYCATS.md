# Copycat detection & history

A **copycat** PR re-submits substantially the same diff as an earlier PR (often an already-merged
one) to farm credit for someone else's work. The eval bot detects and records them automatically.

## How it works (`eval/pr_eval_bot.py`)

- PRs are evaluated **oldest-first** (ascending PR number), so the original is always seen before
  any copy and the earliest submitter is graded first.
- For each PR, the bot fingerprints the diff = (changed files, normalized non-comment added lines)
  and compares against **every earlier PR** (open, closed, or merged).
- A PR is a **copycat** when it shares a changed file with an earlier PR **and** ≥80% of its added
  lines already appear in that earlier PR's diff (`COPYCAT_CONTAINMENT = 0.80`). This catches literal
  copies while leaving genuinely different fixes of the same bug alone.
- A copycat is labeled [`copycat`](../../labels/copycat), commented (citing the original PR), and
  **not evaluated or scored**.
- **First strike → 5-day eval penalty.** From the *first* copycat, the author is frozen for **5
  days**: during the window the bot will **not** greenlight or evaluate *any* of their PRs — it
  applies a [`penalty`](../../labels/penalty) label and skips them (already-scored PRs keep their
  result). The window runs 5 days from the most recent strike.
  - *Per-strike leniency:* a strike entry may set `"penalty_days"` to shorten the freeze — e.g. a
    **first-time contributor's first mistake** who is otherwise a genuine contributor (kiannidev's
    #54 → **2 days**). Default is 5.
- **Two strikes → block.** The 2nd copycat by the same author auto-adds them to
  [`blocked-contributors.txt`](./blocked-contributors.txt) with a reason (logged in
  [`FLAGGED.md`](./FLAGGED.md)); from then on their PRs are auto-closed and never evaluated.

The machine-readable log is [`copycats.json`](./copycats.json) — one entry per detected copycat
`{pr, author, original, date}`. It is append-only and maintained by the bot.

## History

| date | copycat PR | author | copied from | note |
|------|-----------|--------|-------------|------|
| 2026-06-25 | #14 | `glorysr1209-png` | #4 (`galuis116`) | flash_prefill mask; identical 1-line diff. Account already blocked as sybil. |
| 2026-06-25 | #9  | `glorysr1209-png` | #6 (`galuis116`) | gguf metadata desync; 7/8 added lines identical. Account already blocked as sybil. |
| 2026-06-25 | #54 | `kiannidev` | #53 (`James-CUDA`) | maintainer-flagged: same "default split-K down + PDL" decode change as #53 (which scored `eval:none`). Below the auto containment threshold (0.50) but a duplicate attempt — strike 1 of 2. **2-day penalty** (first-time contributor, first mistake; genuine contributor of #44/#52). |

> `glorysr1209-png` also opened #13 and #15 (same bug-clusters as #11 / #12) but with different
> code — not literal copies, so not logged here; they were closed under the sybil block instead.
