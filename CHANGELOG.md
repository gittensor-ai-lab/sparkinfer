# Changelog

Notable changes to sparkinfer. Format loosely follows [Keep a Changelog](https://keepachangelog.com);
versions track the GitHub [releases](https://github.com/gittensor-ai-lab/sparkinfer/releases).

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
