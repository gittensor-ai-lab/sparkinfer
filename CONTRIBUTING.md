# Contributing to sparkinfer

sparkinfer is the engineering arm of **SN74 on Gittensor**. Contributions are rewarded
for **real, verified inference-speed engineering** — not benchmark gaming. This guide is
how to make a contribution that counts.

## Built through Gittensor

Gittensor helps power SPARKINFER through SN74: the project receives subnet emissions,
contributors submit source PRs, the evaluator rebuilds those PRs on real RTX 5090 hardware,
and rewards are assigned from verified marginal speedups that keep correctness intact.
You do not need to be in Discord or understand the subnet internals to contribute, but the
source of the incentive loop is clear: SPARKINFER is built through **SN74 on Gittensor**.

## Principles

- **Source-required & reproducible.** The validator builds your PR from source. No
  opaque prebuilt images — the shipped prebuilt binaries are a *run* convenience, not a
  submission format.
- **Correctness first.** A faster kernel that changes the model's output is worth zero.
  Every change is gated against a frozen reference (see *Accuracy gate* below).
- **General, not overfit.** Optimizations must hold across the basket and across shapes; a win
  on one model but a regression on another is overfitting, and the guards will catch it. The
  live basket is **Qwen3.8-27B** (dense, the scored target) and **Qwen3.6-35B-A3B** (MoE, a
  no-regression guard). They share `qwen35.cpp`, `qwen35_prefill.cpp` and the kernels, so a
  change aimed at one routinely lands in the other's path.
- **Blackwell only, by design.** Targets `sm_120` (RTX 5090, RTX PRO 6000) and `sm_121`
  (RTX Spark / Jetson Thor). CUDA 12.8+ (13 works). Not `sm_100`.

## Before you open a PR

```bash
# 1. build + tests (all ctest targets must pass)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j && ctest --test-dir build

# 2. speed — does it actually go faster?
bench/scripts/bench.sh --download            # and --compare for the llama.cpp gap

# 3. accuracy — did it stay correct?  (this is the gate that blocks regressions)
bench/scripts/accuracy.sh --download
```

**Accuracy gate.** Run `bench/scripts/accuracy.sh` (or `qwen3_gguf_score`) on the build
*before* and *after* your change. A correct optimization must keep:
- **top-1 token agreement within the current eval threshold** vs the previous build, and
- **low mean KL** (the next-token distributions should barely move).

(`accuracy.sh` also compares against llama.cpp; the implementation bar there is ≥ 90%
top-1, currently met at ~96–99%.) If `compute-sanitizer` is available, your kernels
must be clean (0 errors).

**Speculative-decode work is held to a stricter bar.** While the DSpark scope is active, the eval
bot additionally requires **exact token equality** against the same build with the draft disabled —
not distributional agreement. If you're touching the draft or the verify path, check that locally
(`runtime/examples/dspark_tau_check.cpp` reports `LOSSLESS`) before opening the PR, and check it
over several repeats: a single lossless run is weak evidence, and the bot runs repeats for exactly
that reason. See the gate table in *What gets evaluated, reviewed, or closed*.

## How rewards work (SN74 on Gittensor)

**Speedup-only.** You're paid for the **verified marginal speedup** your PR adds over the
current best ("frontier"), not your rank — so "copy the leader + ε" pays ≈ ε. Both **current
`main` and your PR are built and benchmarked on the same RTX 5090** in one run and scored on the
delta between them, so speed differences between eval machines can't inflate or hide your result.

**Competing PRs (per-round merge workflow).** A run grades every queued PR against the *same*
`main`, so two independent optimizations each get their true gain. The bot then labels the round's
biggest one [`merge-first`](../../labels/merge-first) and the rest
[`needs-rebase`](../../labels/needs-rebase). The `merge-first` winner is **auto-merged** once it
clears every guard — verified speedup, clean CI, no conflicts, author in good standing, and it
touches only `kernels`/`runtime`/`moe` (never the maintainer-owned paths); a maintainer can stop
that with a `hold` label. Once the `merge-first` PR is merged, the others **stay `needs-rebase`** —
**rebase your branch onto the new `main`** and push; the bot then re-runs your eval against the new
frontier (briefly tagging [`re-evaluate`](../../labels/re-evaluate) during the re-grade), so you're
credited for the **marginal** gain on top of what merged (independent wins stack and keep scoring; a
change the merge already captured drops to `none`). A `needs-rebase` PR can't win the next round
until you actually rebase + it re-evals. Keep your branch rebased on `main`. The eval loop
labels each PR **XL / L / M / S / XS** from the measured delta (or **BASELINE** for the first
verified entry on a new model/target) — never by hand — and that tier is the payout. A speedup
is scored the same wherever it lands (`kernels/`, `runtime/`, `moe/`); there is **no
per-subsystem budget**. Tiers are bands of **% speedup over the frontier** — `XS` 2–3.5%, `S`
3.5–6%, `M` 6–10%, `L` 10–18%, `XL` >18% (a gain under 2% is within measurement noise → `none`).
Because they scale with the frontier, every tier stays reachable as decode speed grows.

**Non-speedup PRs are welcome — but score 0.** Bug fixes, refactors, tests, benchmarks, docs,
and tooling are appreciated and we'll review good ones, but SN74 emits only for verified
speedups, so they earn no reward. They are reviewed by hand, not by the eval bot — see
*What gets evaluated, reviewed, or closed* below for how that lane works. (The eval/scoring
harness is maintainer-owned — see *Maintainer-owned paths*.)

**Speedups are scored against the active evaluation scope, which is narrower than "anything
faster."** At any time the eval bot scores **one** dimension on one model pair. A genuine,
correct speedup somewhere that dimension doesn't measure will score `none` — not because the
work is bad, but because the harness isn't pointed at it. Check the scope below *before* you
invest in an optimization.

**Evaluation is opt-in and proof-gated.** The RTX 5090 eval runs only when **both** hold: you tick
**`- [x] Tested on RTX 5090`** *and* fill the template's **decode tok/s** table with a real
end-to-end improvement (`after > before`, from `bench/scripts/bench.sh` — not an isolated-kernel
microbenchmark). Then the bot greenlights it (**`test-on-5090`**) and evaluates on the next poll.
- Box ticked but the decode table empty / placeholder / no gain → **`needs-benchmark`**, not evaluated
  (fill in real numbers and it greenlights automatically).
- Box not ticked → **auto-closed** (same as `rtx5090-required` CI). Tick the box, fill tables, and reopen to submit.
There is **no override** — every PR is evaluated on a real RTX 5090 only after it legitimately
passes the gate (box ticked + real before<after decode numbers).

> ⚠️ Tick that box **only if you actually ran it on an RTX 5090** and pasted the benchmark log.
> Checking it without testing is false attestation — it is treated as gaming and the account will
> be **blocked** (added to the denylist), the same as sybil farming.

## What gets evaluated, reviewed, or closed

Three lanes. Which one your PR lands in depends on **what it changes**, not on how good it is —
so read this before you start, and say in the PR description which lane you're aiming for.

### The active evaluation scope

The eval bot scores **one dimension at a time**, and it moves as the optimization target moves.
The authoritative statement is the `SCOPE` block at the top of the bot itself
([`eval/pr_dspark_bot.py`](eval/pr_dspark_bot.py)) — that file is the source of truth, this
table is a summary and can lag it.

| | Current (since 2026-09-03) |
|---|---|
| **Scored dimensions** | DSpark decode **and** DSpark-enabled batched prefill at **ctx=4k / 16k / 32k**, plus target-model prefill **and** decode at **ctx=256k** — eight dimensions, any one of which can earn the tier |
| **Model pair** | target = Qwen3.8-27B ModelOpt NVFP4; draft = released DSpark checkpoint |
| **Harness** | `runtime/examples/dspark_tau_check.cpp` for 4k/16k/32k — both legs in one process, one model load. The 256k rows come from a one-row `qwen3_gguf_bench` sweep. |
| **256k is target-only** | No draft at that context: a 262,144-token KV cache needs the VRAM the draft would occupy, so `dspark-decode@256k` does not exist. Both 256k rows measure the served target model. |
| **Not measured at all** | ctx=128 (any length below `kEngageMinSeq`=1024), batched-prefill parity |
| **Explicitly out of scope** | replacing the DSpark drafter — including the checkpoint's MTP head. See below. |

Nothing at ctx=128 has coverage any more. That was deliberate: below `kEngageMinSeq` the batched
verify never arms, so the token loop runs one target forward per *kept* token — exactly what AR
runs — and the only way to move the metric was to speculate less. A metric whose optimum is
"turn the feature off" measures the wrong thing.

#### Replacing the drafter is out of scope — MTP included

The work wanted here is **optimizing the DSpark drafter**. Swapping it for a different drafter is
not in scope, however well it measures, and that specifically includes drafting with the
checkpoint's own MTP head. Such a PR will be closed on scope, not on merit.

This is a real precedent, not a hypothetical. #912 did exactly that — MTP head recursed to depth 2
— and it was **merged and then reverted**. Its numbers were not the problem; they were the best
any PR has posted: `eval-dspark:XL`, +24.4% (dspark 149.29 against main's 119.99), τ 1.969, AR
flat, byte-lossless, `top1=1.0 kl=0.0`, both no-regression guards green.

It was reverted because of what it did to the *measurement*. `SPARKINFER_DSPARK_MTP` defaulted ON
wherever a checkpoint ships `mtp.*` tensors, which on the scored Qwen3.8 ModelOpt build meant
DSpark silently stopped being the drafter — while the bot, the metric name, the harness and the
labels all still said "dspark". Every later round would have reported `dspark-decode@4k` while
measuring MTP. Contributors optimizing that number would have been tuning the wrong component,
and genuine DSpark drafter improvements would have scored as regressions against an MTP baseline.

The general rule this stands for: **a change that redefines what the scored dimension measures is
out of scope even when it improves the number.** If you believe the target itself should move,
open an issue and argue it — do not land it inside a perf PR.

### Lane 1 — evaluated and scored

Changes to `kernels/`, `runtime/`, or `moe/` that move the scored dimension, with the opt-in gate
satisfied (box ticked + real decode numbers). These earn a tier and can auto-merge. To earn one,
a PR must clear **every** gate below — the first failure stops that PR and the bot moves on, so
a rejection comment names one gate, not all of them:

| Gate | Bar | Why |
|---|---|---|
| **Losslessness** | exact token equality vs the same build with the draft disabled | A speculative decoder that is fast because it emits unverified tokens is wrong, not fast. Fail-closed: missing or unparseable reads as fail. |
| **Differential accuracy** | PR vs `main` token distributions must agree | Catches a PR that changes what the *target* produces — which losslessness alone cannot, since DSpark would faithfully reproduce a broken AR. |
| **AR no-regression** | ≥ 0.98× `main` | Stops a "speedup" bought by slowing the AR baseline the two legs share, or by trading away ordinary serving decode. |
| **Acceptance (τ) floor** | ≥ 0.95× `main` | Stops buying throughput by accepting fewer tokens. |
| **Long-context guards** | ≥ 0.98× `main`, decode **and** prefill @ ctx=16k, on Qwen3.8 **and** Qwen3.6 | DSpark work lands in `qwen35.cpp` / shared kernels, which Qwen3.6 also uses — the exact surface through which #775 regressed a model nobody was scoring at the time. |

Tiers are bands of % speedup over the frontier (`XS` 2–3.5% … `XL` >18%; under 2% is noise →
`none`). A `none` or a failed gate is **auto-closed** — reopen after a fix and it re-evaluates.

### Lane 2 — manually reviewed, not scored

Correctness fixes, refactors, tests, benchmarks, docs, tooling — **including work on code the
current scope doesn't measure.** These score 0 by design (SN74 emits only for verified speedups),
but scoring 0 is not the same as being unwanted, and being outside the eval scope is **not**
grounds for closing a correctness fix. A bug is a bug whether or not the harness is currently
pointed at it.

**Getting into this lane without tripping the auto-close.** An unticked RTX 5090 box is only safe
for a PR that touches neither `runtime/` nor the PR template's checkbox — a docs-only change
outside `runtime/`, say, is left open. Anything touching `runtime/` with the box unticked is
auto-closed, *whatever it is*, because runtime changes need the greenlight to enter the eval
queue. So for a correctness fix in `runtime/`:

- **open it as a draft** — drafts are exempt from the 5090 auto-close — and say in the description
  that it's a correctness fix not seeking evaluation; or
- **ask for the [`hold`](../../labels/hold) label**, which exempts it from both the 5090
  auto-close and the stale-close, then mark it ready.

Maintainers, members and collaborators are exempt automatically. If your fix gets auto-closed
anyway, that's the gate misfiring on intent — reopen as a draft and say so; it will not count
against you.

> **This is a correction of past practice, not just a description of it.** When the scope
> narrowed to DSpark, open PRs unrelated to DSpark were closed in bulk to clear the eval queue,
> and at least one genuine fix — [#885](../../pull/885), a Muse Glimmer GEMV correctness fix —
> was closed for scope rather than on its merits. That was a queue-management action applied too
> broadly. Out-of-scope *optimizations* will still be closed (they cannot be scored, and an open
> PR that cannot be scored is a promise the harness can't keep); out-of-scope *correctness fixes*
> should not be, and if yours was, reopen it and say so.

### Lane 3 — closed

- **Opt-in gate not met.** Box unticked → auto-closed (same as the `rtx5090-required` CI check)
  if the PR touches `runtime/` **or** carries the template checkbox unticked. A docs-only PR
  outside `runtime/` with no checkbox is left open. Drafts, `hold`, and
  maintainers/members/collaborators are exempt. Box ticked but the decode table empty or showing
  no gain → `needs-benchmark`, held rather than closed; fill in real numbers and it greenlights
  automatically.
- **Stale.** No new commits for over a day while queued → auto-closed to keep the eval queue
  clean. This is not a judgment on the work: push a commit or reopen and it's picked straight
  back up on the next cycle. `hold` and the current round winner are exempt.
- **Repeated `none`/REJECT.** A third consecutive unscored result auto-closes; `hold` and
  `merge-first` are exempt.
- **Out-of-scope optimizations**, per Lane 2 above — including anything that replaces the
  DSpark drafter rather than optimizing it (MTP head and equivalents), and more generally any
  change that redefines what the scored dimension measures. See *Replacing the drafter is out
  of scope* above for the #912 precedent.
- **Maintainer-owned paths** (below) — cannot merge regardless of content.
- **Gaming** — copycatting, sybil farming; see *Anti-gaming*.

### Anti-gaming (how submissions are kept honest)

The bot evaluates PRs **oldest-first** and fingerprints each diff, so gaming is caught automatically:

- **Copycatting.** Re-submitting an earlier PR's diff — *even with a few extra lines bolted on to
  look original or slip past the evaluator* — is flagged by diff-containment fingerprint. A first
  copycat strike **freezes all your evaluations for 5 days** (`penalty` label, skipped; PRs already
  scored keep their result); a **second strike blocks** the account. Logged in
  [`.github/copycats.json`](.github/copycats.json) / [`COPYCATS.md`](.github/COPYCATS.md).
- **Sybil / duplicate-account farming** (one operator pushing under multiple GitHub identities, or
  shadowing others' work) is blocked outright; evidence is recorded in [`.github/FLAGGED.md`](.github/FLAGGED.md).
- **No override.** There is no way to force-evaluate around the gate — not even for a maintainer.
  Real, original, frontier-advancing work is the only thing that scores.

## Maintainer-owned paths (eval, scoring & governance)

The evaluation harness and scoring config are **maintainer-owned** and must not be changed
in a contributor PR. They decide labels and emissions and are the trust anchor validators
rely on — so a change here, however well-intentioned, can't ride in on the same PR it would
score. These paths are protected:

| Path | What |
|---|---|
| `eval/` | the PR-evaluation bot + GPU runner |
| `bench/scripts/` | the on-box scoring harness (`evaluate.sh`, `label.py`, `accuracy*`, `_common.sh`, the eval prompts) |
| `runtime/examples/dspark_tau_check.cpp` | **the measuring instrument for the current scope** — lives under `runtime/`, but is harness, not contributor surface |
| `.gittensor/` | intra-repo emission weights |
| `sparkinfer-web` `public/dashboard/data.json` | the live frontier ledger (eval bot pushes here; in-repo `dashboard/` is legacy) |
| `.github/` | CI, `CODEOWNERS`, and this guard |

⚠️ **Two of these sit inside paths you're otherwise invited to edit.** `runtime/examples/dspark_tau_check.cpp`
and the eval prompt corpora under `bench/scripts/` belong to the harness even though `runtime/` is
a contributor path. A PR touching any harness path is **skipped, not closed** — the bot comments
saying so and never spends GPU time on it, because a number produced by a modified instrument
can't be accepted either way. Split the harness change out and the rest evaluates normally.

**Enforcement.** A required **`sensitive-paths-guard`** check automatically fails any PR from a
non-maintainer that touches these paths, and `CODEOWNERS` requires maintainer review — so such
PRs **cannot merge**, regardless of content. The evaluator also grades with the harness pinned
to the protected branch, so editing it in a PR never affects that PR's own score.

**Improving the harness is welcome — just not via a direct PR.** Open an issue or discussion
describing the change; if a maintainer agrees, they'll land it (with credit). Keep your own PRs
scoped to `kernels/`, `runtime/`, and `moe/` — that's the rewarded optimization work.

## Style & scope

- Match the surrounding code (portable CUDA is the production path; CuTe/tensor-core is
  the opt-in ceiling). Keep kernels readable and commented where non-obvious.
- Reference the bench + accuracy numbers in your PR description (before → after).
- Keep changes focused; one optimization per PR makes the measured delta attributable.

By contributing you agree your work is licensed under the repository's [MIT License](LICENSE).
