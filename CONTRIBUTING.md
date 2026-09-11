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
  on one model but a regression on another is overfitting, and the guards will catch it. The live
  basket is **Muse Glimmer** and **Qwen3.8-27B** (both scored), with **Qwen3.6-35B-A3B** (MoE) and
  **Qwen3.8 ModelOpt** as no-regression guards. They share `qwen35.cpp`, `qwen35_prefill.cpp` and
  the kernels, so a change aimed at one routinely lands in another's path — which is exactly why
  a win is scored wherever it lands and a regression anywhere is caught.
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

**Speculative-decode work is held to a stricter bar.** If your PR touches speculative decoding, the
eval bot additionally requires **exact token equality** against the same build with the draft disabled —
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

**There is no target optimization — optimize anything.** There is no single blessed model,
kernel, context length or subsystem. Decode, prefill, attention, FFN, quantization, KV cache,
sampling, batching, memory traffic, load time — if you can make inference genuinely faster
without changing what it produces, that is the work, and it is scored the same wherever it lands.

**If no evaluation measures your optimization yet, ask for one.** Several bots score in parallel
(see the table below) and between them they cover a lot, but they cannot cover everything. A real
speedup on an axis nobody measures would otherwise score `none` — not because the work is bad, but
because nothing is pointed at it. That is a gap in the harness, not a verdict on your PR, and the
fix is to close the gap:

- **Open an issue** describing the axis — model, metric, context length, and the command that
  measures it — with your before/after numbers. Say why the existing axes miss it.
- A maintainer adds it to the harness (harness paths are maintainer-owned — see *Maintainer-owned
  paths*), and your PR is then evaluated against it on the next poll.
- You can open the PR at the same time; ask for the [`hold`](../../labels/hold) label so it is not
  auto-closed while the axis is being added, and link the issue from the PR.

Requesting an axis is a normal, welcome contribution — most of the current axes exist because
somebody asked. What is *not* welcome is redefining what an existing axis measures in order to
move it; see *Do not redefine what a scored dimension measures* below.

**Declare which model(s) your PR targets.** The template has a **Target model(s)** block next to
the RTX 5090 box — tick every model your change is meant to speed up. It decides which bots spend
GPU on it:

- a bot whose model you **did not** tick may skip your PR rather than spend a ~20-minute round
  proving a change it cannot move;
- **tick nothing and every bot evaluates it**, exactly as before this existed — the declaration can
  only ever remove work you have said is pointless, never cause a PR to go unevaluated by accident;
- **tick `Shared / both` when unsure**, or when you touched shared code (`qwen35.cpp`,
  `qwen35_prefill.cpp`, most of `kernels/`) — most optimizations land there and genuinely help more
  than one model. Over-ticking costs eval time; under-ticking can cost you a tier a bot would have
  awarded.

This exists because it was being paid for in wasted GPU: [#1025](../../pull/1025)
(`perf(muse)`, a Muse Glimmer prefill change) consumed a full DSpark round to conclude `+0.1%` →
`eval-dspark:none`, and [#1018](../../pull/1018) and [#1023](../../pull/1023) did the same before it.

> **Declaring a model you did not target, to dodge a guard, is gaming.** The no-regression guards
> exist because the code is shared — a change aimed at one model regularly lands in another's path.
> Mis-declaring to route around one is treated like false attestation on the 5090 box.

**Evaluation is opt-in and proof-gated.** The RTX 5090 eval runs only when **both** hold: you tick
**`- [x] Tested on RTX 5090`** *and* fill **either** the template's **decode tok/s** table **or**
its **prefill pp tok/s** table with a real end-to-end improvement (`after > before`, from
`bench/scripts/bench.sh` — not an isolated-kernel microbenchmark). Either table alone is enough:
a prefill-only optimization with flat decode greenlights on its prefill numbers, and vice versa.
Then the bot greenlights it (**`test-on-5090`**) and evaluates on the next poll.
- **Keep the template's row labels.** The greenlight parser reads the tables by their row labels,
  so leave `before prefill (main)` / `after prefill (this PR)` (and `before (main)` /
  `after (this PR)`) intact and just fill in the numbers. Renaming a row — dropping the word
  `prefill` from it, say — makes that number invisible to the gate, and a prefill-only PR then
  looks like a decode PR with no gain and is skipped.
- Box ticked but neither table has a real gain → **`needs-benchmark`**, not evaluated
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

### What is currently measured

Several bots run in parallel, each on its own hourly slot, and **any** of them can earn your PR a
tier. A PR is evaluated by every bot whose greenlight it passes, and the tier you are paid is the
**best** result across them — a change that helps one model and is merely neutral on another is
not penalized for the neutral result. (A bot that does not measure your model reports `none`, which
is an absence of evidence, not a negative finding.)

**A regression is not the same as a neutral result.** If any bot fails your PR on a gate — accuracy,
losslessness, or a no-regression guard — that `REJECT` takes precedence over any tier another bot
awarded. You cannot buy a win on one model with a regression on another.

The bots themselves are the source of truth (their `SCOPE` blocks); this table is a summary and
can lag them.

| Bot | Model | Scored axes |
|---|---|---|
| [`eval/pr_museglimmer_bot.py`](eval/pr_museglimmer_bot.py) | Muse Glimmer | prefill **and** decode at **ctx 128 / 512 / 4k / 16k / 32k / 64k** — twelve axes, any one of which can earn the tier |
| [`eval/pr_dspark_bot.py`](eval/pr_dspark_bot.py) | Qwen3.8-27B ModelOpt NVFP4 + DSpark draft | DSpark decode and DSpark-enabled batched prefill at **ctx 4k / 16k / 32k**, plus target prefill and decode at **ctx 256k** |

Both bots additionally run **no-regression guards** on models they are not scoring, because the
code is shared — see the gate table in *Lane 1*:

| Bot | Guards it runs |
|---|---|
| Muse Glimmer bot | Qwen3.6 **and** ModelOpt (Qwen3.8) @ ctx 32k |
| DSpark bot | Qwen3.6 **and** Qwen3.8 @ ctx 16k |

**Which bots honour your `Target model(s)` declaration follows from that table.** The DSpark bot
skips a PR declared Muse-only, because the Muse bot's own run still guards Qwen3.8 and Qwen3.6 — so
nothing goes unchecked. The Muse bot does **not** yet skip a PR declared Qwen3.8-only, because no
bot currently guards Muse Glimmer from the other side; skipping there would let a Muse regression
land unnoticed. That asymmetry is a gap in the harness, not a policy: it closes when a Muse guard
is added to the DSpark bot.

**This list is not a menu, and it is not exhaustive.** It is what happens to be wired up today.
If your optimization is real and lands somewhere none of these axes reach, that is a reason to
[ask for an axis](#how-rewards-work-sn74-on-gittensor), not a reason to abandon the work.

#### Do not redefine what a scored dimension measures

"Optimize anything" means anything that makes inference genuinely faster. It does **not** mean
changing what a scored axis measures so the number moves. A metric has to keep meaning the same
thing across rounds or nobody's score means anything — including yours.

The canonical case: on the DSpark axis, the work wanted is **optimizing the DSpark drafter**.
Swapping it for a different drafter is not the same thing, however well it measures, and that
specifically includes drafting with the checkpoint's own MTP head. Such a PR is closed on that
basis, not on merit — and note this is a rule about *substituting the thing being measured*, not
about which subsystems you may touch.

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

The general rule this stands for: **a change that redefines what a scored axis measures is out of
scope even when it improves the number.** That is the one real limit on "optimize anything" — and
it has an outlet: if you think an axis should measure something different, or that a new axis
should exist, open an issue and argue it. Do not land it inside a perf PR.

### Lane 1 — evaluated and scored

Changes to `kernels/`, `runtime/`, or `moe/` that move any measured axis, with the opt-in gate
satisfied (box ticked + a real gain in the decode **or** prefill table). These earn a tier and can
auto-merge. To earn one, a PR must clear **every** gate the bot that scores it applies — the first
failure stops that PR and the bot moves on, so a rejection comment names one gate, not all of them.

The gates below are the DSpark bot's, the strictest set (speculative decoding needs losslessness
and an acceptance floor that ordinary decode work does not). Other bots apply the subset that
makes sense for what they measure — every bot runs accuracy and no-regression guards; only the
speculative axes carry the losslessness and τ gates:

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

Correctness fixes, refactors, tests, benchmarks, docs, tooling — **including work on code no
current axis measures.** These score 0 by design (SN74 emits only for verified speedups), but
scoring 0 is not the same as being unwanted, and not being measured is **not** grounds for closing
anything. A bug is a bug whether or not a harness is pointed at it.

**An optimization nobody measures yet is not a Lane 2 PR — it is a Lane 1 PR waiting for an axis.**
Do not quietly downgrade real speed work to "unscored" because the harness has a gap. Open the
issue asking for the axis (see *How rewards work*), ask for [`hold`](../../labels/hold) so the PR
survives while it is added, and it is evaluated like any other once the axis lands.

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

> **This is a correction of past practice, not just a description of it.** When the scope was
> narrowed to a single target, open PRs unrelated to it were closed in bulk to clear the eval
> queue, and at least one genuine fix — [#885](../../pull/885), a Muse Glimmer GEMV correctness
> fix — was closed for scope rather than on its merits. That was a queue-management action applied
> too broadly, and the narrow-scope rule it enforced is gone: **there is no target optimization
> now, so "out of scope" is no longer a reason to close anything.** A correctness fix stays open on
> its merits; an optimization no axis reaches yet gets an axis requested for it. The only
> scope-shaped close left is a change that redefines what an existing axis measures. If your PR was
> closed under the old rule, reopen it and say so.

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
  `merge-first` are exempt. **If you are waiting on a requested axis, get the
  [`hold`](../../labels/hold) label** — otherwise a PR that only scores `none` because nothing
  measures it yet will be closed by this rule before the axis lands.
- **Changes that redefine what a scored axis measures** — including anything that replaces the
  DSpark drafter rather than optimizing it (MTP head and equivalents). See *Do not redefine what
  a scored dimension measures* above for the #912 precedent. Note this is **not** "your
  optimization is off-target": there is no target, and an optimization no axis reaches yet gets an
  axis requested for it rather than a close.
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
| `runtime/examples/dspark_tau_check.cpp` | **a measuring instrument** (the DSpark axes) — lives under `runtime/`, but is harness, not contributor surface |
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
