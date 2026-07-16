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

## 2026-06-25 — `kiannidev` (auto-blocked, later overridden)

Auto-blocked after 2 copycat strikes (#54, #57). #57←#56 is a near-verbatim copy
(109/110 added lines identical); #54←#53 was a weaker maintainer-flagged duplicate.

## 2026-06-26 — `kiannidev` block overridden by maintainer decision

Maintainer decision: lift the block and drop the **#57** strike, keeping only the #54
strike. Mechanics: `copycat` / `flagged:gaming` cleared from #57, #57 **kept closed**,
#57 removed from `copycats.json`, and `kiannidev` removed from `blocked-contributors.txt`.
The 5-day eval penalty from the #54 strike still stands (window ends 2026-06-30).

Note: #57 is objectively a copy, so the auto-detector **re-flags it whenever the PR is open**
(this is exactly what happened on a re-run after it was briefly reopened). For this override to
hold, #57 must stay **closed** (or be added to an explicit copycat exception). This entry
supersedes the auto-block records above.

## 2026-06-29 — `ai-engram` (auto-blocked)

Auto-blocked: copycat of #83 (fansilas) re-submitted as PR #88. Zero-tolerance copycat policy.

## 2026-06-30 — `carlh7777` + `thomasbaker9010251` (sybil pair)

**Accounts:** `carlh7777` (PR opener), `thomasbaker9010251` (commit author/committer).

**Evidence — shared git identity across two accounts (concrete):**
PR **#112** (`perf(moe): fuse down-projection Q8_1 quantize into gate/up`) was *opened* by
`carlh7777`, but every commit on it is *authored AND committed* by `thomasbaker9010251`
(`Thomas B <thomas.b.901025@gmail.com>`). Opener ≠ committer, both non-collaborators (only
`ai-hpc` has push) — one operator pushing from a single git environment under two GitHub
identities. `thomasbaker9010251` opens no PRs of its own; it appears only as the commit author
behind `carlh7777`'s submissions. Same signal class as the glorysr1209-png/seekmistar01 pair above.

**History:** `carlh7777` — 3 PRs, none merged (#49, #99 closed `not-tested`; #112 a known
self-reported **−4.7%** regression submitted as a "perf" PR).

**Action:** both accounts added to `blocked-contributors.txt`; future PRs from either auto-flag,
close, and skip evaluation.

**NB:** an earlier draft of this entry wrongly cited `andriypolanski`/#105 as the twin because #112
and #105 shared an identical −4.7% result. That was coincidence — `andriypolanski`'s commits are
self-consistent (own login + email), so andriypolanski is **not** part of this sybil and is **not**
blocked. The real link is the carlh7777 ↔ thomasbaker9010251 opener/committer split above.

## 2026-07-01 — `bohdansolovie` + `kiannidev` (coordinated copycat ring + evidence-tampering)

**Accounts:** `bohdansolovie` (bohdansolovie@gmail.com), `kiannidev` (kiannidev@gmail.com) — distinct git identities, coordinated (not a shared-identity sybil).

**Cross-copying each other's work to double merged-PR count:**
- **#57 (kiannidev) ← #56 (bohdansolovie):** near-verbatim, 109/110 added lines identical (prior auto-block; later maintainer-overridden, #54 strike kept).
- **#108 (bohdansolovie) ← #104/#109 (kiannidev):** #108's original title was byte-identical to kiannidev's open #109 (`perf(moe): router→gu2 PDL chain + fused gate/up MMVQ`); body said "Supersedes closed #104 (kiannidev)".

**Active cover-up (decisive):** after #108 was closed, `bohdansolovie` **renamed it to "Hello" (17:33 UTC) then force-pushed the branch to `main` (17:35 UTC)** to erase the diff. #56 was force-pushed twice as well. Legitimate work is not retitled and force-erased after being flagged.

**Action:** both added to `blocked-contributors.txt`; PRs #108/#109/#104/#56 labeled `flagged:gaming`; future PRs from either auto-flag, close, skip eval. Already-merged PRs (bohdansolovie #65; kiannidev #52/#44/#23/#22/#21) predate this and cannot be reversed. kiannidev's earlier #54/#57 override is superseded — the pattern continued.

## 2026-07-02 — `Daedalus-Icarus` (auto-blocked)

Auto-blocked after 1 copycat PRs (#132).

## 2026-07-03 — `devmixa702` (auto-blocked)

Auto-blocked after 1 copycat PRs (#191).

## 2026-07-04 — `jony376` (copycat) · `fansilas` UN-blocked (false attribution — reversed)

The auto-block of **`fansilas`** over #221 was a **misattribution and has been reversed.**
`fa_split_gqa_mma_i8_kernel` is fansilas's own kernel (#195, `eval:XL`, merged 03:47); #221 (fansilas,
commit **05:23**) is their legitimate follow-up trimming its shared int32→float round-trip. #209
(`jony376`) is a **verbatim-identical** copy of that change — same code *and* the same multi-line
comments — but its commit is dated **11:18, ~6h after** fansilas's 05:23 (#209 was *opened* at 02:47 as an
empty placeholder, then the copied code was force-pushed in at 11:18). The detector attributed "original"
by earliest PR-*open* time, which inverted the real authorship.

**Action:** `fansilas` removed from `blocked-contributors.txt`; #221 reopened + un-flagged. Strike
reassigned to **`jony376`** — #209 labeled `flagged:gaming`/`copycat` + closed, `jony376` added to
`blocked-contributors.txt`. fansilas's merged record (#195/#122/#86/#83) stands.

## 2026-07-10 — `inference2026` (auto-blocked, then cleared)

#326 was auto-blocked as ≥85% copycat of #195 — **false positive**: per-function
containment matched a 7-token `fa_split_gqa_kernel<...>` template instantiation line
(100% boilerplate overlap); actual PR-level containment vs #195 was 27%. Maintainer
cleared: removed from `blocked-contributors.txt`, label `copycat-cleared`, #326 reopened
for eval.

#338 was auto-blocked as ≥85% copycat of #318 — **false positive**: 3-line build fix
routing the GQA-4 hd256 branch through `fa_launch_combine_dispatch_hd256` after #300
removed `g2`; added lines matched #318's existing call-site pattern at 100% (tiny-PR
literal rule). Independent convergent fix with #336 (merged ~14 min earlier).
Maintainer cleared: label `copycat-cleared`, #338 reopened. Account was not on
`blocked-contributors.txt`.

## 2026-07-15 — `jimcody1995` (auto-blocked)

#436 ≥85% copycat of #403 (90%). Maintainer confirmed block on 2026-07-16.

## 2026-07-09 — `jsdevninja` (auto-blocked)

Real-time copycat guard closed #298 (≥85% containment vs #121). Block comment posted
but entry never landed on `main`. Maintainer confirmed block on 2026-07-16.

## 2026-07-04 — `james3773` (rename of blocked `jony376`)

`jony376` was blocked for verbatim copycat of fansilas's #221 via #209. GitHub login
`jony376` no longer exists (404); flagged PRs #209/#142 are under **`james3773`**. Added
`james3773` to `blocked-contributors.txt` on 2026-07-16; kept `jony376` for historical
commit-author matching.
