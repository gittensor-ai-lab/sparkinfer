# sparkinfer Miner Guide

This guide is for SN74 miners and contributors who want to earn rewards by
improving sparkinfer. The rule is simple: rewards come from verified speedups on
real code, not from claims, formatting, or duplicated ideas.

## What Scores

A PR can score when it does all of the following:

- Builds from source on the evaluator's RTX 5090.
- Preserves correctness against the frozen llama.cpp reference.
- Improves at least one measured context by **2% or more**.
- Avoids unacceptable regressions in the guarded contexts.
- Changes code that is actually used by the benchmark path.

The measured contexts are:

| context | target |
|---:|---|
| 128 | short decode, no prefill context |
| 512 | 512-context decode, 128 generated tokens |
| 4k | 4k-context decode, 128 generated tokens |
| 16k | 16k-context decode, 128 generated tokens |
| 32k | 32k-context decode, 128 generated tokens |

Small gains are not aggregated across contexts. For example, +1% at 128 and +1%
at 4k is still not a scoring +2% improvement.

## What Does Not Score

These changes may be useful, but they do not earn a speed label unless they also
produce a verified frontier speedup:

- Documentation-only changes.
- Refactors with no benchmark improvement.
- Test-only changes.
- Benchmark harness changes that do not improve runtime speed.
- Copying an already merged optimization without a new measurable improvement.
- Changes that improve one synthetic path but are unused by the eval target.

## Correctness Gate

The evaluator compares sparkinfer against llama.cpp. A fast PR is rejected if it
changes the model output too much.

The gate checks:

- Token agreement / top-1 match.
- KL divergence against the reference.
- Stable decode behavior on held-out prompts.

Do not trade accuracy for speed. Accuracy is part of the benchmark.

## Regression Labels

A PR can improve one context and regress another. The bot makes this explicit
with context-specific labels:

| label | meaning |
|---|---|
| `regression-128` | 128-token decode regressed |
| `regression-512` | 512-context decode regressed |
| `regression-4k` | 4k-context decode regressed |
| `regression-16k` | 16k-context decode regressed |
| `regression-32k` | 32k-context decode regressed |

If no context improves by at least 2% and any guarded context regresses, the PR
is rejected and may be auto-closed.

## Speed Labels

The reward label is based on the strongest verified context improvement over the
current live frontier:

| label | meaning |
|---|---|
| `eval:XL` | very large verified speedup |
| `eval:L` | large verified speedup |
| `eval:M` | medium verified speedup |
| `eval:S` | small verified speedup |
| `eval:XS` | minimum accepted verified speedup |
| `eval:none` | correct, but no significant improvement |
| `eval:REJECT` | correctness failure, build failure, or unacceptable regression |

The exact label is deterministic from the evaluator output. The bot does not use
AI judgment to decide rewards.

## Local Checklist Before Opening A PR

Run these from the repo root on a Blackwell GPU box when possible:

```bash
# Decode speed
bench/scripts/bench.sh --download

# Same-GPU llama.cpp comparison
bench/scripts/bench.sh --download --compare

# Correctness gate
bench/scripts/accuracy.sh --download
```

For local model files:

```bash
bench/scripts/bench.sh /path/to/model.gguf --tokens 128
bench/scripts/accuracy.sh /path/to/model.gguf
```

If your change targets long context, also record the context you expect to move:

```bash
# Examples; use the benchmark flags supported by the current bench script.
bench/scripts/bench.sh /path/to/model.gguf --ctx 512 --tokens 128
bench/scripts/bench.sh /path/to/model.gguf --ctx 4096 --tokens 128
bench/scripts/bench.sh /path/to/model.gguf --ctx 16384 --tokens 128
bench/scripts/bench.sh /path/to/model.gguf --ctx 32768 --tokens 128
```

## PR Requirements

A good PR includes:

- A short description of the bottleneck.
- The files and kernels changed.
- Local speed numbers, including GPU model and CUDA version.
- Accuracy numbers or a clear statement that `accuracy.sh` passed.
- Any expected context-specific effect: `128`, `512`, `4k`, `16k`, or `32k`.

Keep PRs narrow. A small kernel PR with a clear benchmark is easier to verify and
merge than a broad rewrite.

## Current Target

The current frontier is Qwen3-MoE Q4_K_M decode on RTX 5090 / consumer Blackwell.
The project is especially interested in:

- Long-context flash decode at 16k and beyond.
- KV read efficiency and staging.
- Quantized MoE expert throughput.
- Removing launch overhead from the decode path.
- Improvements that hold correctness against llama.cpp.

## Do Not Game The Eval

The evaluator uses held-out prompts, same-box baseline comparisons, immutable
logs, and path-aware labels. Attempts to tune for the harness instead of the
model path can be rejected or ignored.

The best way to earn is to make the shipped runtime faster and keep it correct.
