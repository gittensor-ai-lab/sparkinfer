# Changelog

Notable changes to sparkinfer. Format loosely follows [Keep a Changelog](https://keepachangelog.com);
versions track the GitHub [releases](https://github.com/gittensor-ai-lab/sparkinfer/releases).

## [0.4.0] — 2026-07-11

This release adds **Qwen3.5-9B (Qwythos)** as a first-class target and locks in **three-model** decode:
**Qwen3-MoE**, **Qwen3.6-35B-A3B**, and **Qwen3.5-9B** — all beating llama.cpp on the same RTX 5090
box. Qwen3.5 landed in under a day and is already **20%+ faster than llama.cpp** at 128/512/4k context;
Qwen3.6 holds the **30%+ long-context lead** from v0.3.8 with no regression.

### 🆕 Qwen3.5 (Qwythos-9B) — +24% past llama.cpp in one day

Dense hybrid Gated-DeltaNet + full-attention · 9B · hd256 · Q4_K_M. Same RTX 5090, 128 generated tokens,
`qwen3_gguf_bench` (3-rep median):

| context | sparkinfer | llama.cpp | delta |
|---|---:|---:|---:|
| **128-token decode** | **279.8 tok/s** | 224.91 tok/s | **+24%** |
| **512-context decode** | **277.9 tok/s** | 225.10 tok/s | **+23%** |
| **4k-context decode** | **270.1 tok/s** | 224.68 tok/s | **+20%** |

Landed in a single sprint: dense-hybrid loader, FFN down Q6→Q4 requant at load (#323), split-K + int8
graph capture (#324), GQA-4 shared-KV tiles (#326), and bidirectional eval against Qwen3.6 guards.
**`SPARKINFER_DOWN_REQUANT_Q4K` now defaults ON** (set `=0` to keep native Q6_K reads).

### 🏁 Three models — all ahead of llama.cpp (128-token decode)

| model | sparkinfer (128 tok/s) | llama.cpp | delta |
|---|---:|---:|---:|
| **Qwen3-MoE (30B-A3B)** | **480.7 tok/s** | 365.85 tok/s | **+31%** |
| **Qwen3.6-35B-A3B** | **424.9 tok/s** | 275.81 tok/s | **+54%** |
| **Qwen3.5-9B (Qwythos)** | **279.8 tok/s** | 224.91 tok/s | **+24%** |

Qwen3.6 long-context ladder unchanged vs v0.3.8 (post-#300 MMA correctness rebench):

| context | sparkinfer | llama.cpp | delta |
|---|---:|---:|---:|
| **128-token decode** | **424.9 tok/s** | 275.81 tok/s | **+54%** |
| **512-context decode** | **420.1 tok/s** | 275.61 tok/s | **+52%** |
| **4k-context decode** | **403.1 tok/s** | 276.30 tok/s | **+46%** |
| **16k-context decode** | **386.4 tok/s** | 280.66 tok/s | **+38%** |
| **32k-context decode** | **364.3 tok/s** | 279.83 tok/s | **+30%** |

### Performance — landed since v0.3.8

- **#318** (`eval:M`) — quantized dense FFN + GDN fusions + hd256 32k combine
- **#323** (`eval:S`) — requantize dense FFN down Q6_K→Q4_K at load (~5% Qwythos decode)
- **#324** (`eval:M`) — tune dense split-K and int8 graph capture for Qwen3.5
- **#326** (`eval:XS`) — GQA-4 shared-KV tile for Qwythos dense attention
- **#331** (`eval:XS`) — complete MoE gate_up→quant_h→down PDL chain for bs=1 decode
- **#300** — hd256 MMA correctness fix in flash-decode split (no perf regression on release rebench)

### Eval — Polaris Ed25519 fallback

When Intel TDX is unavailable (Polaris API timeout/404), the eval bot falls back to **Ed25519-signed
receipts** if `SPARKINFER_POLARIS_PRIVATE_KEY` is set — eval logs still ship a verifiable receipt.

### The proof, in four layers

1. **Speed** — three models, each **20–54%** over llama.cpp on the same GPU/GGUF; Qwen3.6 long-context lead held.
2. **Correctness** — top-1 **≥ 0.95**, KL **≈ 0.01** vs llama.cpp on held-out prompts.
3. **Same-box baseline** — bidirectional Qwen3.5 + Qwen3.6 eval with per-model no-regression guards.
4. **Polaris** — TDX receipts when available; Ed25519 fallback when the enclave API is down.

**Verified:** RTX 5090 · Qwen3.5 **280 tok/s** (128-tok) · Qwen3.6 **425 tok/s** (128-tok) · Qwen3-MoE **481 tok/s** (128-tok).

### Contributors

- **@James-CUDA** — #323 (Qwen3.5 FFN down Q6→Q4 requant at load)
- **@9876543210-tc-0123456789** — #324 (Qwen3.5 split-K + int8 graph capture)
- **@inference2026** — #326 (GQA-4 shared-KV tile for Qwythos hd256)
- **@claytonlin1110** — #331 (MoE gate_up→quant_h→down PDL chain)
- **@Paral1995** — #318 (dense FFN + GDN fusions + hd256 32k combine)
- **@reyanthony062001-ops** — #300 (hd256 int8-MMA flash-decode correctness)
- **@skyrocket2026** — Qwen3.5 bidir eval infra (#315–#317, #322), Polaris Ed25519 fallback, v0.4.0 release

## [0.3.8] — 2026-07-09

This release adds **hardware-rooted trust** to the eval pipeline and locks in the Qwen3.6 speed story
across the full context ladder. Every graded run can now ship an **Intel TDX attestation** (Polaris) that
third parties verify offline — and Qwen3.6 stays **30%+ faster than llama.cpp at every tracked context,
128 through 32k**, on the same RTX 5090 and UD-Q4_K_M GGUF.

### 🔐 Polaris — verifiable eval receipts (Intel TDX)

The benchmark loop is no longer "trust us, we ran it on a GPU." The bot can emit a **Polaris receipt**
that binds code commit, model SHA256, eval seed, and measured tok/s to an **Intel DCAP quote**:

- **#295** — Polaris TDX integration: `judge.py` assembles attestations on the eval box; the bot submits
  scoring to Polaris and uploads signed receipts alongside eval logs
- **#301** — production fixes: correct API wiring, Qwen3.6 model SHA pinning, stdout forwarding so
  `POLARIS_ATTESTATION` survives SSH capture, TDX verify/hash fixes; smoke + test helper scripts

Anyone can run `eval/polaris/verify.py receipt.json` — no GPU, no trust in the operator.

### 🏁 Qwen3.6 — 30%+ past llama.cpp at every context

Same RTX 5090, Qwen3.6-35B-A3B UD-Q4_K_M, 128 generated tokens, warm & interleaved vs `llama-bench`:

| context | sparkinfer | llama.cpp | delta |
|---|---:|---:|---:|
| **128-token decode** | **426.0 tok/s** | 275.81 tok/s | **+54%** |
| **512-context decode** | **419.2 tok/s** | 275.61 tok/s | **+52%** |
| **4k-context decode** | **402.4 tok/s** | 276.30 tok/s | **+46%** |
| **16k-context decode** | **385.2 tok/s** | 280.66 tok/s | **+37%** |
| **32k-context decode** | **363.5 tok/s** | 279.83 tok/s | **+30%** |

The frontier climbed **23 → 426 tok/s** in under a week; the long-context tail no longer lags.

### Performance — landed since v0.3.7

- **#282** (`eval:XL`) — fused router GEMV + bitonic top-k (grid-completion decode) — **426 tok/s at 128-ctx** (@fansilas)
- **#279** — partial-RoPE KV fuse, GDN conv-L2, Q5 S=8, Q8_0 MMVQ, addnorm3 (@jimcody1995)
- **#284** — int8 KV + tensor-core flash-decode for the hd256 full-attn layers (@nickmopen)

### The proof, in four layers

v0.3.8 stacks proof the way v0.3.6 stacked speed + correctness + quality — now with **hardware attestation**:

1. **Speed** — Qwen3.6 +30–54% over llama.cpp at 128/512/4k/16k/32k on the same box/GGUF.
2. **Correctness** — top-1 **≥ 0.95**, KL **≈ 0.01** vs llama.cpp on held-out prompts.
3. **Same-box baseline** — every PR graded against origin/main measured in-session (no hardware lottery).
4. **Polaris TDX** — Intel-verified receipt per eval; offline verification without re-running the GPU job.

**Verified:** RTX 5090, Qwen3.6 **426 tok/s** (128-tok), top-1 **0.95**, Polaris `intel_verified=True`.

## [0.3.6] — 2026-07-04

This release breaks the long-context deficit wide open and adds a new axis of proof. sparkinfer now
beats the llama.cpp Q4_K_M baseline by **30–36% from 128 to 16k, and ~30% (+29.8%) at 32k** on the same RTX 5090
and GGUF — the 16k lead jumped from **+8.4% (v0.3.5) to +31%** — driven by moving long-context attention
onto the tensor cores. And it ships the first **LLM-quality benchmark suite**, so the frontier is proven
on real task capability, not only speed and token-agreement.

### Performance — ~30% or more past llama.cpp at every context, out to 32k

Same RTX 5090, same Qwen3-MoE Q4_K_M GGUF, 128 generated tokens, warm and **interleaved** (per-round
A/B so GPU-clock drift cancels):

| context | sparkinfer | llama.cpp | delta |
|---|---:|---:|---:|
| **128-token decode** | **489.86 tok/s** | 363.15 tok/s | **+34.9%** |
| **512-context decode** | **471.09 tok/s** | 346.45 tok/s | **+36.0%** |
| **4k-context decode** | **393.49 tok/s** | 295.35 tok/s | **+33.2%** |
| **16k-context decode** | **327.31 tok/s** | 249.18 tok/s | **+31.4%** |
| **32k-context decode** | **260.30 tok/s** | 200.52 tok/s | **+29.8%** |

### Added — int8 tensor-core long-context flash decode (#195, #221)

The 16k/32k gains come from the first use of **tensor cores in the decode path**:

- **#195** (`eval:XL`) — a "gutted-dot" experiment showed long-context flash decode is **compute-bound**
  on the per-token QK dot + warp reduction, not bandwidth-bound. So it batches the 8 GQA q-heads of a
  kv-head as the M dimension: `S = Q·Kᵀ` and `O = P·V` become small **int8 `wmma` matmuls** (2× throughput,
  int32 accumulate), and K/V are stored **int8 (Q8-style)** to **halve the KV read**. Context-adaptive
  (engages only ≥8k) and template-specialized on a compile-time flag, so 128/512/4k stay **byte-identical**.
- **#221** (`eval:XL`) — trims the kernel's shared-memory round-trips and raises occupancy from 4 to 5
  resident blocks/SM (register + shared-memory limited).

Together they took **16k decode 266 → 330 tok/s**, correctness held (top-1 ≥ 0.94, KL ≤ 0.04).

### Added — LLM quality benchmark suite (#192)

Speed and token-agreement don't prove the model still *answers well*. [`bench/quality`](bench/quality)
scores five standard capabilities on **real data** — **IFEval, GSM8K, MMLU-Pro, HumanEval, BFCL** — with
deterministic, stdlib-only scorers (constraint checks, final-answer extraction, unit-test `pass@1`,
tool-call matching). A spot-check of the current frontier: **GSM8K 100%, IFEval 78%**, overall ~69% on a
real-data subset. Because sparkinfer matches llama.cpp at **96% top-1 / KL 0.017**, these scores are at
**parity with llama.cpp by construction** — the suite proves the optimizations preserved *capability*,
not just the token distribution.

### The proof, in three layers

v0.3.6 is deliberately not "fast only." Each frontier claim now stands on three independent checks:

1. **Speed** — +31–36% over llama.cpp from 128 to 16k and +29.8% at 32k, warm & interleaved on the same GPU/GGUF.
2. **Correctness** — every kernel gated at **top-1 ≥ 0.90 / KL ≤ 0.20** vs llama.cpp (currently ~0.96 / ~0.02),
   reproducible from source and immutably logged.
3. **Quality** — real-task benchmarks (IFEval/GSM8K/MMLU-Pro/HumanEval/BFCL) confirm the model still
   solves the tasks, at parity with the reference.

### Momentum

v0.3.5 first pushed past llama.cpp across the tracked context ladder (16k barely, +8%). v0.3.6 turns that
into a **decisive ~30%+ lead at every length through 32k** and proves it holds on quality. The next frontier:
deeper 32k+ work, KV-cache quantization beyond attention, and running the full quality suite in the eval gate.

Thanks to everyone keeping the benchmark loop fast, public, correctness-gated — and now quality-gated.

## [0.3.5] — 2026-07-03

This release lands the long-context follow-through: sparkinfer now beats the llama.cpp Q4_K_M baseline
at every tracked dashboard context size — **128, 512, 4k, and 16k** — on the same RTX 5090 and same
GGUF. The headline is no longer only the short decode frontier; the 16k path is now ahead too.

![sparkinfer v0.3.5 all tracked contexts pass llama.cpp](docs/releases/v0.3.5.png)

### Performance — all tracked context sizes are past llama.cpp

Same RTX 5090, same Qwen3-MoE Q4_K_M GGUF, 128 generated tokens:

| context target | sparkinfer | llama.cpp | delta |
|---|---:|---:|---:|
| **128-token decode** | **493.56 tok/s** | 365.85 tok/s | **+34.9%** |
| **512-context decode** | **469.58 tok/s** | 342.59 tok/s | **+37.1%** |
| **4k-context decode** | **392.65 tok/s** | 292.99 tok/s | **+34.0%** |
| **16k-context decode** | **266.14 tok/s** | 245.53 tok/s | **+8.4%** |

### Changed — the benchmark surface is now context-aware

The evaluation loop now treats **128, 512, 4k, and 16k** as first-class guard surfaces. A PR can earn
credit for improving any one context by at least 2%, without aggregating small gains across unrelated
contexts. Regressions are labeled by context (`regression-128`, `regression-512`, `regression-4k`,
`regression-16k`) so contributors can see exactly where a change helped or hurt.

The dashboard was updated to show the full context comparison directly against llama.cpp, including
both card summaries and paired horizontal bars. This keeps the public frontier easy to scan while the
project moves from short-decode wins into long-context competition.

### Momentum

v0.3.4 proved the first short-decode optimization round. v0.3.5 proves the next step: the same
optimization loop can push past llama.cpp across the visible context ladder, including 16k. The next
frontier remains deeper long-context work: 16k/32k stability, paged/KV read efficiency, KV staging, and
continued decode-kernel occupancy work.

Thanks to all contributors and reviewers keeping the benchmark loop fast, public, and competitive.

## [0.3.4] — 2026-07-02

This release closes the **first round of decode optimization** and marks it as a working proof of
concept: contributors can move the RTX 5090 Qwen3-MoE frontier quickly, the eval loop can verify it,
and the dashboard can carry the public proof trail. The headline 128-token frontier is now
**484.79 tok/s** on Qwen3-30B-A3B Q4_K_M — **32.5% faster than llama.cpp** on the same RTX 5090
128-token decode target — with top-1 **0.9612** and KL **0.0175** vs llama.cpp.

![sparkinfer v0.3.4 RTX 5090 decode frontier](docs/releases/v0.3.4.png)

### Performance — first decode-optimization round lands at 484.79 tok/s
The round merged the final short-context decode pass:
- **#121** — optimize Qwen decode kernels; evaluated at **468.10 tok/s** (`eval:none`) and merged as
  useful implementation groundwork.
- **#122** — fuse QK-norm + RoPE + KV append and emit Q8_1 attention data in the flash combine path;
  evaluated at **479.83 tok/s** (`eval:L`) and advanced the public frontier.

After merging, a final `origin/main` benchmark on the cached RTX 5090 measured **484.79 tok/s** at the
same 128-token decode target, versus llama.cpp at **365.85 tok/s**: **+32.5% faster than llama.cpp**.
This is the last optimization of the first short-decode round: enough to prove the path, not the end
of the project.

### Next — compete at long context
The published milestone remains the next focus: **16k and 32k context**. v0.3.3 showed the long-context
proof of concept; v0.3.4 finishes the short-decode momentum and points contributors at the next
competition surface: long-context flash decode, paged/KV read efficiency, KV quantization, and stable
eval dimensions for 16k/32k.

### Thanks
Thanks to everyone who contributed, evaluated, reviewed, and kept the loop moving with momentum.

## [0.3.3] — 2026-07-01

Two things this release: scoring that **rewards late-game effort** (so it stays worth optimizing as the
frontier matures), and a **long-context proof of concept** that finds — and largely fixes — the biggest
open opportunity, to point contributors at where the real headroom is. The 128-tok frontier is
unchanged at **453.70 tok/s**.

### Changed — difficulty-compensated scoring (#113): reward late-game effort
As the frontier pulls past llama.cpp, each further % gain gets much harder (near the roofline the
easy headroom is gone), so a fixed %-band scale under-rewards late-game work — a hard +4% now took
more than an easy +20% at cold start. `label.py` now scales the **label tier** by a difficulty
multiplier `D = 1 + K·max(0, frontier/ref − 1)` (K=8, ref = llama.cpp 365.85, capped at 4×): a gain
scores like the effort it took relative to a mature baseline. Safeguards: the boost multiplies the
**label only** — `pct_over_frontier` still reports the true measured speedup, the significance gate
stays on the **raw** delta (noise is never boosted), and the cold-start era (frontier ≤ ref) is
untouched (D=1, no retroactive inflation). Applied from new evals onward. On the real history #83/#89/#86
move S→M/L; everything below llama is unchanged. Governance-tunable (`SPARKINFER_DIFFICULTY_{BOOST,K,REF,MAX}`);
replay with [`eval/sim_difficulty.py`](eval/sim_difficulty.py).

### Added — long-context decode: the deficit found, and a first fix (#115) — proof of concept for miners
Our headline "+24% past llama.cpp" is measured at 128-tok; at real KV **depth** the story reverses.
A same-box depth sweep (sparkinfer vs `llama-bench -d`) found sparkinfer's decode **collapses** with
context — **5.2× behind llama at 32k** (37 vs 193 tok/s), running ~6× *below* the memory roofline. Root
cause: the flash-decode split count was **fixed** (`n_splits=32`), so at 32k each split streamed a
~1024-long serial online-softmax chunk on ≤1024 blocks (latency-bound, SMs idle).

**#115 makes `n_splits` depth-adaptive** (scale with `seq_len`, target ~256 KV/split, powers of two from
32, capped 256) so the grid fills the SMs at depth; the decode CUDA graph is re-captured only ~log₂
times per generation. **Correctness-preserving by construction** — the online-softmax combine is an
*exact* reduction, bit-identical for any split count (top-1/KL unchanged). Short context is untouched
(adaptive holds 32 below ~8k), so the frontier is unaffected. Tune via `SPARKINFER_SPLIT_CHUNK`; pin a
fixed value via `SPARKINFER_NSPLITS`.

**Long-context speedup — RTX 5090, decode tok/s at KV depth:**

| KV depth | before (fixed 32) | after (adaptive) | speedup | gap to llama.cpp |
|---|---|---|---|---|
| 128 | 442.8 | 442.7 | 1.00× | unchanged (no short-context regression) |
| 4,096 | 194.0 | 193.8 | 1.00× | unchanged |
| **16,384** | 70.8 | **166.2** | **2.35×** | 3.4× → **1.44×** behind |
| **32,768** | 38.5 | **110.7** | **2.88×** | 5.0× → **1.74×** behind |

This is a **proof of concept, not the finish line** — it's here to guide contributors: long-context
flash-decode (KV-split scaling, paged-KV read efficiency, KV quantization) is where the headroom is, and
one config fix already closed most of a 5× gap. The 128-tok eval doesn't measure it yet — a long-context
eval dimension is the natural next step.

## [0.3.2] — 2026-06-30

The lead over llama.cpp **doubles to ~24%**, and the evaluation that proves it is **substantially
hardened** — held-out prompts, reference quarantine, clock-recorded runs, an immutable frontier
ledger, and a corrected KL metric.

### Performance — RTX 5090 frontier 410.85 → 453.70 tok/s (+10.4%); now **24% past llama.cpp**
Two verified kernel optimizations merged (top-1 0.97):
- **#89** — run the Q/K/V projections on **concurrent CUDA streams**, overlapping latency-bound bs=1 GEMVs → 435.41, byte-identical (@James-CUDA)
- **#86** — **single-pass MoE top-k** (one parallel rank-count vs 8 serial arg-max passes) + fused RoPE/KV-append → 453.70 (@fansilas)

Same RTX 5090, same Q4_K_M GGUF, warm & interleaved vs `llama-bench`:

| decode length | sparkinfer | llama.cpp |   | vs v0.3.1 |
|---|---|---|---|---|
| **128 tok** | **453.70** | 365.85 | **+24.0%** | was +12.1% |
| **256 tok** | **443.53** | 364.90 | **+21.6%** | was +10.0% |
| **512 tok** | **425.23** | 361.64 | **+17.6%** | was +6.7% |

The lead grew at **every** length — the recent decode-path work cut the per-token overhead that used
to shrink the long-context lead.

### Added — trust-hardened evaluation pipeline (#102)
Closes the remaining gaming/poisoning vectors from the eval trust-model audit:
- **Held-out prompts (H1)** — each eval scores a fresh, unpredictable per-seed window of a diverse
  corpus, so a submission can't overfit a fixed prompt; the seed is logged for reproduction.
- **Reference quarantine (C2)** — the baseline weights (sha256-pinned) and llama.cpp (commit-pinned)
  are verified/rebuilt each run, so a tampered persisted copy can't skew a verdict.
- **Clock record (M1)** — the graphics clock each number was produced at is pinned where the box
  permits and **always recorded**, so the absolute tok/s is reproducible.
- **Immutable frontier ledger (H2)** — every frontier advance appends a GitHub-timestamped line
  `(date, PR, author, commit, Δ%, prev→new)` to the public eval-log; auditable line-by-line.
- **Provenance** (clock, seed, reference pins) is written into every verdict and immutable log.

### Fixed — the KL accuracy metric (honest, strict gate kept)
The held-out KL looked high (0.27 on hard text) — investigation found a **measurement artifact**: the
gate dumped only sparkinfer's top-20 and floored llama's tail, over-penalizing KL on flat
distributions. The fix dumps a deeper top-k so llama's mass is covered; the **true divergence is ~0.02**
(top-1 0.97). Proven honest — a sensitivity test reads KL 18 on a deliberately broken build, and a
12-prompt sweep holds KL 0.007–0.022. So the **strict `KL ≤ 0.20` gate is kept**: it holds on held-out
text because the measurement is correct, not because it was loosened.

### Verified
- **RTX 5090** frontier **453.70 tok/s** (128-tok), top-1 **0.97** vs llama.cpp — **+24.0% @128 /
  +21.6% @256 / +17.6% @512** over a fully-built CUDA llama.cpp, same-box, warm, interleaved.

### Contributors
- **@James-CUDA** — #89 (concurrent Q/K/V CUDA streams)
- **@fansilas** — #86 (single-pass MoE top-k + fused RoPE/KV-append)

## [0.3.1] — 2026-06-29

The lead over llama.cpp widens to **double digits — and now holds at every context length** — and the
evaluation becomes **publicly verifiable**: a hardware trust model plus an immutable, per-run public log.

### Performance — RTX 5090 frontier 388.68 → 410.85 tok/s (+5.7%); now **10%+ past llama.cpp**
Two verified kernel optimizations merged (top-1 0.97, KL ≈ 0.14):
- **#72** — split-K the router projection GEMV for decode occupancy → 394.45 (@Dexterity104)
- **#83** — emit Q8_1 from the residual RMSNorm, dropping the per-layer activation quantize → 410.85 (@fansilas)

Same RTX 5090, same Q4_K_M GGUF, warm & interleaved vs `llama-bench`:

| decode length | sparkinfer | llama.cpp |   |
|---|---|---|---|
| **128 tok** | **410.2** | 366.0 | **+12.1%** |
| **256 tok** | **402.2** | 365.8 | **+10.0%** |
| **512 tok** | **386.6** | 362.5 | **+6.7%** |

sparkinfer is now **ahead at every length** — v0.3.0 was ~parity at 512; the recent decode-path work
(residual Q8_1, router split-K) lifted the long-context number too.

### Added — trustless, publicly-verifiable evaluation
- **[`EVAL-TRUST.md`](EVAL-TRUST.md)** — the eval trust model: **reproducible from source today**, the
  attested-eval roadmap (CPU-TEE scoring receipts → multi-validator consensus), and the honest boundary
  (a consumer RTX 5090 has **no GPU Confidential Computing**, so the speed number is trusted via
  **reproduction + consensus**, not a GPU enclave — by design, since we optimize the hardware people own).
- **[sparkinfer-log](https://github.com/gittensor-ai-lab/sparkinfer-log)** — every eval is now committed
  **immutably** to a public repo (raw `log.txt` + `result.json`, host IPs scrubbed) and rendered at a
  **unique, verifiable URL per run** (GitHub Pages). The dashboard links each verdict to its proof.

### Changed — accuracy gate tightened
- **KL hard-reject at 0.20** (preferred ≤ 0.15): a speedup that erodes parity with llama.cpp now
  `REJECT`s regardless of tok/s. In practice #83 first regressed KL to 0.21 → `REJECT`, the author
  reworked it to KL 0.14 → clean `S` → merged. The gate forced a better PR.

### Fixed — eval stability
- **Warm-up before the baseline**, **fresh same-box checkout** on reused boxes (`FETCH_HEAD`, not a
  stale `origin/main`), and a **baseline sanity guard** — so cold clocks and stale builds can't skew a
  verdict.

### Verified
- **RTX 5090** frontier **410.85 tok/s** (128-tok), top-1 **0.97** vs llama.cpp (KL ≈ 0.14) —
  **+12.1% @128 / +10.0% @256 / +6.7% @512** over llama.cpp, same-box, warm, interleaved.

### Contributors
- **@fansilas** — #83 (emit Q8_1 from the residual RMSNorm)
- **@Dexterity104** — #72 (split-K router projection GEMV)

## [0.3.0] — 2026-06-28

The milestone release: sparkinfer's CUDA kernels **overtake llama.cpp** on Qwen3-MoE single-stream
decode — at the **kernel level**, same model, same Q4_K_M precision, same greedy `bs=1` decode. No
speculative decoding (EAGLE-3 / Medusa), no draft model, no flash-decoding accuracy trade — just
faster kernels. Plus the first **production-readiness** feature: a thermal-safe inference governor.

### Performance — RTX 5090 frontier 313.14 → 388.68 tok/s (+24%)
Four verified kernel optimizations merged (top-1 0.95–0.98 vs llama.cpp, KL ≈ 0.145):
- **#71** — int8 dp4a MMVQ for the Q4_K MoE down projection → 333.75 (@Dexterity104)
- **#74** — split-K MMVQ down for M-tier decode occupancy → 339.59 (@jaso0n0818)
- **#76** — fuse per-head Q/K-norm + Q/K rope into single kernels → 371.27 (@James-CUDA)
- **#73** — skip the unused per-expert token-count pass in single-token decode → 388.68 (@Dexterity104)

### 🏁 First to beat llama.cpp — at the kernel level
Same RTX 5090, same Qwen3-30B-A3B Q4_K_M GGUF, head-to-head vs `llama-bench`, warm & controlled:

| decode length | sparkinfer | llama.cpp |   |
|---|---|---|---|
| **128 tok** | **388.7** | 372.0 | **+4.5%** |
| 256 tok | 381.5 | 371.7 | +2.6% |
| 512 tok | 367.3 | 368.6 | ~parity |

A **genuine kernel win** — identical weights, precision, and greedy single-stream decode; the
speedup lives in the CUDA kernels (fused quantized MoE FFN, int8 dp4a MMVQ across every decode GEMV,
split-K occupancy, fused attention norms), **not** in algorithmic shortcuts. The lead is largest at
short generations and narrows to parity at long context — the per-token attention/KV path is the
next frontier.

### Added — production-readiness: thermal-safe inference (#77, @ai-hpc)
- **`ThermalGovernor`** — a DVFS-style decode governor that throttles **throughput** when the GPU
  runs hot (turbo / balanced / safe / emergency tiers, predictive), **preserving correctness
  exactly**: it only paces token emission and never touches weights, precision, logits, or sampling,
  so output is **bit-identical** to an un-paced run. Opt-in; zero overhead when off. Forcing the
  tiers on a real RTX 5090 traded throughput for power **309 W → 87 W (3.5×)** with *identical token
  ids* across every mode.
- **GPU observability** — engine-level `query_gpu_stats()` / `Runtime::gpu_stats()` (heat, VRAM,
  power, SM clock via NVML, mapped to the CUDA device by PCI bus id).

### Changed — evaluation hardened against thermal & caching effects
- **Warm-up before the baseline.** The from-source build leaves the GPU idle for minutes, so the
  first timed build (the same-box baseline) was read on **cold clocks** and inflated every PR's
  delta. The bench now spins clocks to boost before timing.
- **Fresh same-box baseline on reused boxes.** The baseline checkout ran `git fetch origin origin/main`
  — which silently fails (the branch is `main`) — and on a **reused** box left a *stale* checkout, so
  it built **pre-merge** code and a just-merged gain was double-counted into the next PRs. Now it
  fetches the real branch and checks out `FETCH_HEAD` (guaranteed fresh).
- **Baseline sanity guard.** A run aborts if the same-box `main` baseline reads < 90 % of the known
  frontier (cold / throttling / degraded box) instead of grading against a bogus-low baseline.

### Verified
- **RTX 5090** frontier **388.68 tok/s** (128-tok decode), top-1 **0.98** vs llama.cpp (KL ≈ 0.145),
  **21.4 GB** resident — **+4.5 % over llama.cpp** at 128-tok, ~parity at 512-tok. Same-box, warm,
  llama-anchored, controlled measurement.

### Contributors
- **@Dexterity104** — #71 (int8 dp4a Q4_K MoE down), #73 (skip per-expert token count)
- **@jaso0n0818** — #74 (split-K MMVQ down)
- **@James-CUDA** — #76 (fuse Q/K-norm + Q/K rope)
- **@ai-hpc** — #77 (thermal governor + GPU observability)

## [0.2.3] — 2026-06-26

A performance jump **and** a fairer, more trustworthy evaluation: every PR is now measured against
`main` on the **same GPU**, scored on the same-box delta, and worked through a per-round merge
workflow that can auto-merge the winner.

### Performance — RTX 5090 frontier 285.32 → 313.14 tok/s (+9.7%)
Two verified MMVQ int8 quantized-read optimizations merged (top-1 0.99 vs llama.cpp, KL ≈ 0.15):
- **#65** — int8 dp4a MMVQ for the Q6_K MoE down projection → 291.58 (@bohdansolovie)
- **#70** — int8 MMVQ for the last fp32-path GEMVs (attn-V + LM head + gate/up) → 313.14 (@James-CUDA)

The llama.cpp gap closed to **0.86×** (313.14 vs 365.73 tok/s).

### Changed — fairer, hardware-independent scoring
- **Same-box baseline.** Each eval builds **current `main` and the PR on the same RTX 5090** and
  scores the **delta between them**, so speed differences between eval machines can't inflate or
  hide a result. (Previously a PR's tok/s was compared to a frontier measured on a *different* box.)
- **No within-run ratchet — independent PRs each score.** Every queued PR is graded against `main`,
  not against the other PRs in the run. Before, whichever PR was graded first ratcheted the frontier
  and made the next — a *different* optimization — look like `eval:none`.
- **Label tiers are now bands of % over the frontier** (`XS` 2–3.5% … `XL` >18%), so all five stay
  reachable as decode speed grows (the old fraction-of-headroom rule collapsed the small tiers).

### Added — per-round merge workflow (+ guarded auto-merge)
- A round grades the whole queue against the same `main`, labels the biggest verified speedup
  **`merge-first`** and the rest **`needs-rebase`**. After the winner merges, rivals **rebase onto
  the new `main`** and the bot re-evaluates them for their *marginal* gain on top — so independent
  wins stack and an overlapping one correctly drops to `none` (`re-evaluate` tags the re-grade).
- **Auto-merge (opt-in, heavily guarded).** The `merge-first` winner auto-merges only with a verified
  speedup, no `copycat`/`flagged:gaming`/`penalty`/`hold`, author in good standing, changes confined
  to `kernels`/`runtime`/`moe`, clean CI, and no conflicts. A `hold` label or `SPARKINFER_AUTOMERGE=0`
  stops it; branch protection is still enforced.

### Fixed
- **Dashboard journey is merged-only.** The frontier and the optimization journey advance only when a
  PR is **merged** (by its measured tok/s), not on eval — so unmerged or losing-rival evals no longer
  pollute the chart.
- **Self-healing eval box.** Stopped vast.ai boxes get reclaimed, so the pinned box can vanish between
  runs; the bot now reuses it if it survived, else provisions a fresh one (Google Drive model fetch)
  immediately and re-pins — no wasted retries.

### Verified
- **RTX 5090** frontier **313.14 tok/s**, top-1 0.99 vs llama.cpp (KL ≈ 0.15 nats), 21.4 GB resident.
  Auto-evaluation runs on a 2-hour schedule.

### Contributors
- **@James-CUDA** — #70 (int8 MMVQ for the fp32-path GEMVs)
- **@bohdansolovie** — #65 (int8 dp4a MMVQ for the Q6_K MoE down)

## [0.2.2] — 2026-06-26

A day of rapid frontier progress (**+52% decode**), a copycat caught gaming the eval, and a
hardened auto-eval pipeline that now runs reliably on a 30-minute schedule.

### Performance — RTX 5090 frontier 187.61 → 285.32 tok/s (+52%) in a day
Five verified speedups landed since v0.2.0, each paid only for its **marginal gain over the
previous frontier** (correctness-gated, top-1 ≥ 96% vs llama.cpp throughout):

| PR | optimization | → frontier | label |
|----|--------------|-----------:|:-----:|
| #44 | vectorized fused RMSNorm (128-bit bf16×8 loads) | 197.22 | `M` |
| #50 | decode dp4a (MMVQ) default + argmax widen | 240.11 | `XL` |
| #52 | two-pass multi-block decode argmax (1 SM → all SMs) | 262.17 | `L` |
| #59 | llama.cpp Q4_K `mul_mat_vec_q` for attention GEMVs | 279.11 | `L` |
| #63 | parallelized flash-decode combine + `n_splits=32` | 285.32 | `M` |

The llama.cpp gap closed to **0.78×** (285.32 vs 365.73 tok/s).

### Security (anti-gaming)
- **Copycat-to-bypass capture + 5-day penalty.** Caught a PR that re-submitted an earlier
  author's diff with a few extra lines bolted on to look original and slip past the eval — the
  diff-containment fingerprint flags these even with cosmetic additions. A first copycat strike
  now **freezes the author's evaluations for 5 days** (`penalty` label, skipped; already-scored
  PRs keep their result); a **2nd strike auto-blocks**. Logged in `.github/copycats.json` /
  `COPYCATS.md`.
- **No manual eval override.** Removed the `force-eval` bypass entirely — every PR is evaluated
  on a real RTX 5090 **only** after it legitimately passes the gate (box ticked **and** a real
  before<after decode table). Nothing skips the benchmark.

### Fixed — stabilized 30-minute auto-evaluation
- **Google Drive model source.** HuggingFace was throttling the 18.6 GB GGUF to ~0.2–5 KB/s on
  many vast.ai hosts (effectively stalled). The eval now fetches it from Google Drive via `gdown`
  (measured **20–74 MB/s**), with HF/curl as fallback — the model lands in minutes, not never.
- **Pinned stable instance (reuse-first, never destroy).** The eval reuses one known-good box
  with the cached model by default instead of provisioning fresh each run. On bring-up failure it
  retries on the next run (~30 min) up to twice before provisioning a new box — and **never
  destroys the pinned one**. Eliminates the re-download / re-provision churn between runs.
- **Dud-host skip-list + cron lock.** Blacklist hosts whose entire network is dead (not just HF);
  a `flock` lock prevents overlapping cron ticks. Together these make the 30-minute auto-eval reliable.
- **Dashboard.** Optimization-journey x-axis labels rotated 45° so the (now 12) bars no longer collide.

### Changed
- **Label tiers are now bands of % speedup over the frontier** (`XS` 2–3.5%, `S` 3.5–6%, `M` 6–10%,
  `L` 10–18%, `XL` >18%; <2% is within noise → `none`) — same denominator as the significance gate.
  The previous *fraction-of-headroom* rule collapsed `XS`/`S` once the frontier neared the ceiling
  (the 2% noise floor alone exceeded their headroom bands); the new bands keep all five tiers
  reachable and scale with decode speed.

### Verified
- **RTX 5090** frontier **285.32 tok/s**, top-1 0.96 vs llama.cpp (KL ≈ 0.14 nats), 21.4 GB resident.

### Contributors
- **@James-CUDA** — #50 (`XL`), #59 (`L`), #63 (`M`)
- **@kiannidev** — #44 (`M`), #52 (`L`)

## [0.2.0] — 2026-06-25

Evaluation-pipeline hardening, anti-gaming controls, and the live frontier dashboard.

### Added
- **Opt-in RTX 5090 evaluation** — the PR auto-eval bot runs the on-device eval only after the
  PR template's *Tested on RTX 5090* box is ticked (auto-applies `test-on-5090`) or a maintainer
  greenlights it; otherwise the PR is labeled `not-tested` and skipped (no GPU). Falsely ticking
  the box is treated as gaming.
- **Live optimization-journey chart** on the [dashboard](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/)
  — recorded passes (history) plus optimizations that have **landed** on the frontier; the bot
  appends each frontier-advancing merge automatically. Accuracy (token-match / KL) now tracks the
  frontier instead of a stale manual value.
- **Community safety hardening** (merged PRs) — input/scratch bounds guards across the MoE expert
  FFN, decode runner, and router kernel; GGUF load-time validation (reject unsupported GGML types,
  clamp invalid `general.alignment`, bounds-check tensor regions vs file size).

### Security (anti-gaming)
- **Sensitive-path merge gate** — `CODEOWNERS` + a `sensitive-paths-guard` status check + branch
  protection block any non-maintainer PR touching the eval/scoring/governance paths (`eval/`,
  `bench/scripts/`, `.gittensor/`, `dashboard/data.json`, `.github/`). The bot also grades with
  `bench/scripts` pinned to `origin/main`, so a PR cannot grade itself.
- **Contributor denylist + auto-block** — `.github/blocked-contributors.txt` (+ `FLAGGED.md`
  evidence log); the bot flags, comments, closes, and skips eval for any PR whose opener or commit
  author/committer is blocked. First entry: a 2-account sybil pair sharing one git identity.
- **Copycat detection** — diff-fingerprint each PR against earlier ones; ≥80% containment of a
  *different* author's earlier diff → `copycat` label, skipped eval, logged to `.github/copycats.json`;
  2 strikes auto-blocks the author.

### Changed
- PRs are evaluated **oldest-first**, so the original of any duplicate is graded before its copy.
- Dashboard: removed the obsolete **emission-weights** panel (scoring is speedup-only — there is no
  per-subsystem budget).

### Fixed (evaluation pipeline)
- Provisioning self-heals: abandon phantom-`running` hosts in ~2 min, retry across hosts, blacklist
  repeat offenders, and survive SSH drops during the 17 GB model download (nohup + resumable fetch).
- Build: pin `g++-12` as the CUDA host compiler (nvcc vs Ubuntu 24.04 GCC 13.3 `cstdio` break);
  cap `-j2` to avoid OOM on 64 GB eval boxes.
- A submission that does not compile now yields a clean `eval:REJECT` instead of an infra error.
- **Force-clean per-PR checkout** — each PR builds its own commit (a stale-checkout bug had graded
  several PRs against the wrong code).
- Labels/comments applied via the GitHub REST API (the GraphQL path silently failed on a
  deprecation warning).

### Verified
- **RTX 5090** frontier ratcheted to **187.61 tok/s** (PDL decode; #8, `eval:L`), **top-1 98%**
  token agreement vs llama.cpp (KL ≈ 0.14 nats).

### Contributors
First community contributors — thank you! 🎉
[@galuis116](https://github.com/galuis116), [@jaso0n0818](https://github.com/jaso0n0818),
[@kiannidev](https://github.com/kiannidev), [@philluiz2323](https://github.com/philluiz2323).

> A fifth early account was removed for sybil / eval-gaming (one git identity across two logins,
> farming merged-PR emissions) — see **Security** above and `.github/FLAGGED.md`.

[0.2.0]: https://github.com/gittensor-ai-lab/sparkinfer/releases/tag/v0.2.0

## [0.1.0] — 2026-06-22

First release of the consolidated **sparkinfer** monorepo (kernels + MoE engine + runtime + benchmarks).

### Added
- **Native GGUF loading** — mmap parser + on-GPU **byte-exact Q4_K / Q6_K dequant**;
  expert weights kept quantized resident (Q4_K_M-sized footprint, not bf16).
- **Qwen3-MoE runtime** — embed → RMSNorm → QKV → per-head QK-norm → RoPE → paged GQA
  flash-decode → routed top-k MoE (+ optional shared expert) → LM head → greedy decode.
- **Kernels** — flash-decode (hd128/256/512), **flash-decoding (KV-split)** attention,
  **fused quantized MoE expert FFN** (dequant only the routed experts on-read), decode
  GEMV (coalesced `[out,in]`), GEMM, fused RMSNorm, RoPE.
- **CUDA-graph decode** — the per-token compute is captured once and replayed.
- **Turnkey harness** — `bench/scripts/bench.sh` (decode tok/s, `--compare` vs llama.cpp)
  and `accuracy.sh` (token-match / KL / perplexity); auto-detect arch, fetch model.
- **Accuracy gate** — `qwen3_gguf_score` teacher-forced scorer (per-position argmax +
  top-k logprobs + perplexity), for regression-checking optimizations.
- **Prebuilt binaries** attached to this release (sm_120 / CUDA 13 / glibc 2.39), with
  automatic **source-build fallback** when incompatible.

### Verified
- **RTX 5090** (sm_120, CUDA 13): `ctest` 5/5, compute-sanitizer 0 errors,
  **163.88 tok/s** decode, **100% top-1 token agreement** with llama.cpp (KL ≈ 0.14 nats),
  21.4 GB resident.
- **RTX PRO 6000** (sm_120, CUDA 12.8): **0.60 → 134 tok/s** decode across 6 source-verifiable
  optimization passes.

### Fixed (during RTX 5090 / CUDA 13 bring-up)
- CUDA 13 removed `cudaDeviceProp::memoryClockRate` / `memoryBusWidth` → query via
  `cudaDeviceGetAttribute` (portable across CUDA 12.x / 13).
- Flash-decode scratch (`fa_*`) was NULL on the non-GGUF path (allocated only in
  `load_gguf`) → moved to the constructor (caught by compute-sanitizer).
- Top-level superbuild was missing `enable_testing()` → `ctest` found no tests.

[0.1.0]: https://github.com/gittensor-ai-lab/sparkinfer/releases/tag/v0.1.0
