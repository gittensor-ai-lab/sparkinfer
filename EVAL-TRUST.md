# Eval trust & verifiability

How sparkinfer's SN74 evaluation is made trustworthy — what you can verify **today**, what's on the
**roadmap**, and the honest boundaries. We'd rather state this plainly than over-claim.

## TL;DR

- **Today:** every result is **reproducible from source** — anyone can rebuild `main` and the PR on
  an RTX 5090 and recover the same same-box delta. The frontier has already been **independently
  reproduced** by a community member on a rented 5090.
- **Roadmap:** **attested, multi-source eval** — the deterministic scoring runs in a CPU TEE (Intel
  TDX) that returns a hardware-signed receipt; run logs are committed immutably; and independent
  validators re-measure for consensus.
- **Honest boundary:** a consumer RTX 5090 has **no GPU Confidential Computing**, so the **speed
  number itself can't be sealed in a GPU enclave.** We make it trustworthy through **reproduction +
  consensus**, not a cryptographic proof of tok/s — and we keep it that way **on purpose** (below).

## What the evaluator does

For each PR commit, on one RTX 5090, in one run:
1. **Build `main` and the PR from source** on the same box.
2. **Warm up** the GPU to saturate clocks (cold clocks otherwise skew the first measurement).
3. **Bench decode** for both, **interleaved**, and score the **same-box delta %** — so box-to-box
   hardware variance (~2%) cancels and the score is hardware-independent.
4. **Gate correctness**: top-1 token agreement ≥ 0.90 and **KL ≤ 0.20** (preferred ≤ 0.15) vs
   llama.cpp on the same GGUF. A speedup that erodes accuracy is `REJECT`ed regardless of how fast it
   is — accuracy is the moat.
5. **Label** = a **deterministic function of the measurements** (`XS … XL`, `none`, `BASELINE`,
   `REJECT`), so independent validators converge on the same verdict.

The bot runs the same code you can: [`eval/`](eval) (`pr_eval_bot.py`, `vast_eval.py`), with the
scoring math in [`bench/scripts/label.py`](bench/scripts/label.py).

## Reproduce it yourself

```bash
# on any RTX 5090 (sm_120, CUDA 12.8+)
git clone https://github.com/gittensor-ai-lab/sparkinfer && cd sparkinfer
bench/scripts/bench.sh --download            # sparkinfer decode tok/s
bench/scripts/bench.sh --download --compare  # head-to-head vs llama.cpp (same GGUF)
bench/scripts/accuracy.sh --download         # top-1 / KL / perplexity vs llama.cpp
```

> A note on the head-to-head: build llama.cpp with full CUDA optimization — an under-tuned llama
> build will *understate* its tok/s and *overstate* our lead. Our published figure (**+4.5% at
> 128-tok**, ~parity at long context) is against a fully-built llama.cpp; please compare apples to
> apples.

## Deterministic vs non-deterministic (why the trust model is split)

| | **Correctness** (top-1 / KL) | **Speed** (tok/s) |
|---|---|---|
| nature | deterministic — fixed model + inputs + greedy ⇒ exact | non-deterministic — clocks/thermal/box vary |
| verify by | **re-running** → identical numbers (or a TEE proof) | **reproduction + consensus** within tolerance |

There is no cryptographic proof of a *benchmark* number — only agreement among independent measurers.
We design for that honestly.

## Roadmap to attested eval

0. **Reproducible from source + published protocol** ← *today.*
1. **Immutable public run log** — each eval's full log + result committed to a public `eval-log` repo
   (GitHub-timestamped), one verifiable page per run.
2. **Attested scoring** — run `label.py` + the accuracy gate inside **Intel TDX** (e.g. via a CPU-TEE
   provider); the receipt binds the exact code + commit + inputs + verdict, checkable offline. Kills
   "did they edit the harness or the numbers."
3. **Multi-validator consensus** — independent validators re-measure the same-box delta %;
   stake-weighted trimmed median, commit-reveal. Removes the single scorer.

## Why RTX 5090 ≠ Confidential Computing — and why we keep it that way

GPU Confidential Computing (encrypted-VRAM TEE + attestation) exists only on **datacenter** parts
(H100/H200, B200/GB200), **not** consumer RTX. So we *could* chase cryptographic GPU attestation by
moving to datacenter GPUs — but we won't, for three reasons:

1. **It's our whole point.** sparkinfer optimizes the **consumer/edge Blackwell** GPUs people actually
   own (RTX 5090, RTX PRO 6000, RTX Spark, Jetson Thor) — the gap the datacenter engines leave.
   Optimizing H100/B200 means a different product in a different market.
2. **The eval must run on the target hardware.** Measuring H100 speed isn't measuring 5090 speed.
3. **Datacenter TEE wouldn't even prove the real number.** CC mode adds overhead and changes the
   performance it's trying to attest — you'd get a hardware-signed receipt of a *CC-degraded* speed,
   not the production one.

So the right design for *our* hardware is **reproduction + consensus + attested (CPU) scoring** — not
a GPU enclave. That's a more honest claim, and it matches the machines our users run.

## The honest claim

> Every result is **reproducible from source**, the scoring is **deterministic and (on the roadmap)
> hardware-attested**, and the speed is corroborated by **independent reproduction + consensus** — a
> stake-backed, stabilized, openly-checkable protocol. **Not** a cryptographic proof of GPU tok/s
> (no consumer GPU can give one) — and we'd rather say so.

Found a hole in this? Open an issue — adversarial review of the eval is exactly what makes it
trustworthy.
