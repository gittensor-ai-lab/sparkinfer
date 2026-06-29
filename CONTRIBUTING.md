# Contributing to sparkinfer

sparkinfer is the engineering arm of **SN74 on Gittensor**. Contributions are rewarded
for **real, verified inference-speed engineering** — not benchmark gaming. This guide is
how to make a contribution that counts.

## Principles

- **Source-required & reproducible.** The validator builds your PR from source. No
  opaque prebuilt images — the shipped prebuilt binaries are a *run* convenience, not a
  submission format.
- **Correctness first.** A faster kernel that changes the model's output is worth zero.
  Every change is gated against a frozen reference (see *Accuracy gate* below).
- **General, not overfit.** Optimizations must hold across the basket — **Qwen3-MoE and
  Gemma 4** — and across shapes. A win on one model/shape but not the other is overfitting.
- **Blackwell only, by design.** Targets `sm_120` (RTX 5090, RTX PRO 6000) and `sm_121`
  (RTX Spark / Jetson Thor). CUDA 12.8+ (13 works). Not `sm_100`.

## Before you open a PR

```bash
# 1. build + tests (must be 5/5)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j && ctest --test-dir build

# 2. speed — does it actually go faster?
bench/scripts/bench.sh --download            # and --compare for the llama.cpp gap

# 3. accuracy — did it stay correct?  (this is the gate that blocks regressions)
bench/scripts/accuracy.sh --download
```

**Accuracy gate.** Run `bench/scripts/accuracy.sh` (or `qwen3_gguf_score`) on the build
*before* and *after* your change. A correct optimization must keep:
- **≥ 99–100% top-1 token agreement** vs the previous build, and
- **mean KL ≈ 0** (the next-token distributions barely move).

(`accuracy.sh` also compares against llama.cpp; the implementation bar there is ≥ 90%
top-1, currently met at ~96–99%.) If `compute-sanitizer` is available, your kernels
must be clean (0 errors).

## How rewards work (SN74)

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
Because they scale with the frontier, every tier stays reachable as decode speed grows. See the
[org reward model](https://github.com/gittensor-ai-lab).

**Non-speedup PRs are welcome — but score 0.** Bug fixes, refactors, tests, benchmarks, docs,
and tooling are appreciated and we'll review and merge good ones, but SN74 emits only for
verified speedups, so they earn no reward. (The eval/scoring harness is maintainer-owned — see
*Maintainer-owned paths* below.)

**Evaluation is opt-in and proof-gated.** The RTX 5090 eval runs only when **both** hold: you tick
**`- [x] Tested on RTX 5090`** *and* fill the template's **decode tok/s** table with a real
end-to-end improvement (`after > before`, from `bench/scripts/bench.sh` — not an isolated-kernel
microbenchmark). Then the bot greenlights it (**`test-on-5090`**) and evaluates on the next poll.
- Box ticked but the decode table empty / placeholder / no gain → **`needs-benchmark`**, not evaluated
  (fill in real numbers and it greenlights automatically).
- Box not ticked → **`not-tested`**, not evaluated.
There is **no override** — every PR is evaluated on a real RTX 5090 only after it legitimately
passes the gate (box ticked + real before<after decode numbers).

> ⚠️ Tick that box **only if you actually ran it on an RTX 5090** and pasted the benchmark log.
> Checking it without testing is false attestation — it is treated as gaming and the account will
> be **blocked** (added to the denylist), the same as sybil farming.

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
| `bench/scripts/` | the on-box scoring harness (`evaluate.sh`, `label.py`, `accuracy*`, `_common.sh`, the eval prompt) |
| `.gittensor/` | intra-repo emission weights |
| `dashboard/data.json` | the live frontier ledger |
| `.github/` | CI, `CODEOWNERS`, and this guard |

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
