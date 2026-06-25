# Flagged accounts — eval-gaming / sybil log

Record of GitHub accounts blocked from sparkinfer for gaming the SN74 merged-PR
emission mechanism (sybil accounts, coordinated duplicate submissions, low-effort
PR farming). Enforcement is automated: see [`blocked-contributors.txt`](./blocked-contributors.txt),
read by `eval/pr_eval_bot.py`, which auto-labels (`flagged:gaming`), comments, closes,
and skips evaluation for any PR involving a listed account.

This is an append-only audit trail. Each entry states the accounts, the evidence, and
the action taken.

---

## 2026-06-25 — `glorysr1209-png` + `seekmistar01` (sybil pair)

**Accounts:** `glorysr1209-png`, `seekmistar01`

**Evidence — shared git identity across two accounts (concrete):**
On PRs **#15, #14, #13** the commit is *authored* by `glorysr1209-png` but *committed*
by `seekmistar01`. PR **#11** (merged) was opened and committed by `seekmistar01`.
A differing committer that is itself another contributor's account — repeated across
several PRs — indicates one operator pushing from a single git environment under two
GitHub identities. Neither account is a repo collaborator (only `ai-hpc` has push),
so this is **not** a maintainer rebasing a contributor's work.

**Evidence — duplicate low-effort farming:**
`glorysr1209-png` filed a near-duplicate of every bug-cluster also submitted by other
accounts (flash_prefill mask, gguf metadata desync, n_dims bound, shared-expert absent),
shadowing them to maximize merged-PR count rather than contributing distinct work.

**PRs involved:**
| PR | account | state at flag |
|----|---------|---------------|
| #15 | glorysr1209-png (committed by seekmistar01) | open → closed |
| #14 | glorysr1209-png (committed by seekmistar01) | open → closed |
| #13 | glorysr1209-png (committed by seekmistar01) | open → closed |
| #9  | glorysr1209-png | open → closed |
| #11 | seekmistar01 | already merged (labeled for record) |

**Action:** both accounts added to `blocked-contributors.txt`. Open PRs labeled
`flagged:gaming` and closed. Future PRs from either account are auto-flagged, closed,
and not evaluated by the bot. Merged PR #11's emission cannot be reversed; logged here.

## 2026-06-25 — `kiannidev` (auto-blocked)

Auto-blocked after 2 copycat PRs (#54, #57).
