# Automatic evaluation (vast.ai or fixed SSH box)

Provision (or reuse) a Blackwell GPU on vast.ai, **or** use a fixed bare-metal box via SSH.
Build a sparkinfer submission, gate it for **correctness**, measure its **speed**, and assign an
eval-loop **label** — automatically.

## Transport

| `EVAL_TRANSPORT` | Behavior |
|------------------|----------|
| `vast` (default) | Reuse a pinned `--reuse` instance; left running after eval (no auto-rent) |
| `ssh` | Fixed box via `EVAL_SSH_HOST` + `EVAL_SSH_PORT`; vast.ai is not contacted |

Copy `.env.eval.example` → `.env.eval` for local/cron config. Legacy: `EVAL_USE_VAST=0` also
selects SSH when `EVAL_SSH_HOST` is set.

```bash
# fixed box (no vast billing):
export EVAL_TRANSPORT=ssh EVAL_SSH_HOST=91.224.44.227 EVAL_SSH_PORT=50200
python eval/vast_eval.py --ref main --frontier 285 --ceiling 366

# vast.ai (default):
export EVAL_TRANSPORT=vast
python eval/vast_eval.py --reuse <instance_id> --ref main --frontier 285 --ceiling 366
```

```
submission (git ref) ─► build from source ─► correctness gate (token-match / KL vs llama.cpp)
                     ─► 128 / 512 / 4k / 16k / 32k guards ─► strongest context speed score ─► LABEL
```

The numeric label is a **deterministic function of measurements** (`bench/scripts/label.py`) so
independent validators converge on it; the orchestrator only drives the box.

## Setup (one-time)

```bash
pip install --upgrade vastai
vastai set api-key <YOUR_KEY>            # or: export VAST_API_KEY=...
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
```

## Run

```bash
# reuse a box (started if stopped) — evaluate, left running after (default):
python eval/vast_eval.py --reuse <instance_id> --frontier 164 --ceiling 366 --ref main

# stop after eval, or destroy (frees the disk):
python eval/vast_eval.py --reuse <instance_id> --ref <git-ref> --frontier 164 --ceiling 366 --stop
python eval/vast_eval.py --reuse <instance_id> --ref <git-ref> --frontier 164 --ceiling 366 --destroy
```

**The instance is LEFT RUNNING after every eval by default** — pass `--stop` to pause billing while
the disk and cached weights (`/workspace/models`) persist. `--destroy` frees the disk.
Auto-rent is **off**; pass `--allow-provision` only if you want legacy destroy-and-recreate behavior.

`--frontier` = current best tok/s for the scored target · `--ceiling` = roofline/reference display
value. Reuse mode assumes the weights are cached at `/workspace/models`.

The default eval target is now multi-context decode:
- **128-token, 512-context, 4k-context, 16k-context, and 32k-context decode** are all no-regression guards. A PR must keep at least 98% of same-box `origin/main` speed at every measured context.
- The **strongest single context improvement** becomes the scored target for `eval:<label>`. Improvements are never aggregated across contexts; two sub-2% gains do not combine into a score.
- The bot also applies a UI-only context label (`128-context`, `512-context`, `4k-context`, `16k-context`, or `32k-context`) for the context that improved most. This does not change the score.
- If a PR has both a real context win and a regression elsewhere, it is not rejected automatically; the bot adds `regression-128`, `regression-512`, `regression-4k`, `regression-16k`, and/or `regression-32k` labels for the regressed contexts. Regression labels block auto-merge and require maintainer judgment.
- If no single context clears the 2% significance gate and any context regresses, the bot returns `eval:REJECT` and auto-closes the PR.
- Difficulty compensation uses the selected context's llama.cpp baseline, so late-game improvements past the mature reference get the same multiplier logic at every context.

Each context is sampled once by default (`SPARKINFER_GUARD_*_REPS=1`, `SPARKINFER_SCORE_REPS=1`) to keep eval cost bounded.

Set `SPARKINFER_EVAL_MODE=short` or pass `--eval-mode short` to keep the legacy 128-token scoring path.

## Bidirectional scoring: Qwen3.5 + Qwen3.6 (default)

`--bidir` (or `BIDIR=1` / legacy `TRIPLE=1` in `.env.eval`) scores **both directions** in one build:

```
build once ─► score_qwen35  Qwythos-9B : 128/4k/32k/64k speed + prefill pp at 4k/32k/64k/128k ─► eval-qwen35:<LABEL>
           │              guard Qwen3.6  : 5 contexts ─► must NOT regress
           └► score_qwen36  Qwen3.6      : 128/512/4k/16k/32k decode + prefill pp at all 5 contexts ─► eval-qwen36:<LABEL>
                          guard Qwen3.5  : 128/512/4k ─► must NOT regress
```

- **Qwen3.5** (Qwythos-9B) is measured at **128, 512, 4k only** — not 16k/32k.
- **Qwen3.6** runs the full **5-context** decode sweep (128/512/4k/16k/32k) and **5-context prefill** pp at the same lengths.
- Each direction gets its own label: `eval-qwen35:<tier>` and `eval-qwen36:<tier>`.
- Headline `eval:<label>` is the best verified tier among passing directions.
- Qwen3-30B is **no longer** part of the eval pipeline.
- `PRIMARY_QUANT` selects the Qwen3.5 GGUF: `Q4_K_M` (default), `Q8_0`, or `BF16`.
- Models: `/workspace/models35` (Qwythos), `/workspace/models36` (Qwen3.6).
- Orchestrator: `bench/scripts/evaluate_bidir.sh`.

```bash
python eval/vast_eval.py --ssh HOST:PORT --bidir --primary-quant Q4_K_M --ref main
./eval/run_bot.sh --bidir
```

## Polaris TDX receipts (default)

Eval runs through **Polaris** by default (`POLARIS=1`). The GPU box collects an unsigned
attestation via `eval/polaris/judge.py`; the bot host submits it to Polaris for Intel TDX
verification and uploads the signed receipt with the eval log. When TDX is unavailable (API
timeout, 404, etc.), the bot falls back to **Ed25519** signing if
`SPARKINFER_POLARIS_PRIVATE_KEY` is set.

```bash
# .env.eval
POLARIS=1
POLARIS_API_KEY=pi_sk_...
SPARKINFER_POLARIS_PRIVATE_KEY=...   # base64, 32 bytes — Ed25519 fallback
POLARIS_API_BASE=https://polaris.computer

./eval/run_bot.sh              # Polaris on (default)
./eval/run_bot.sh --no-polaris # legacy unsigned path
./eval/run_polaris_test.sh     # end-to-end smoke test
./eval/run_polaris_smoke.sh    # TDX or Ed25519 smoke from saved attestation
```

Set `POLARIS=0` in `.env.eval` or pass `--no-polaris` to disable.

## Legacy dual/triple modes

`--dual` and `--triple` are aliases for `--bidir`. The old Qwen3-30B guard paths
(`evaluate_dual.sh`, `evaluate_triple.sh`) are retained for reference but no longer used by the bot.

## Verdict (stdout)

```json
{ "commit": "abc1234", "tps": 165.2, "top1": 1.0, "kl": 0.14, "frontier_tps": 164,
  "pass": true, "label": "none", "delta_tps": 1.2, "pct_over_frontier": 0.7 }
```
Labels: **REJECT** (failed correctness or a no-regression guard) · **none** (within the significance gate) ·
**XS · S · M · L · XL** (verified speedup bucket, by fraction of remaining headroom closed).

Policy tests:
```bash
python3 bench/scripts/test_label.py
```

## PR auto-evaluation bot (retired from cron)

> **Status: scoring is DFlash-only now.** `pr_eval_bot.py`'s casual bidir (Qwen3.5/Qwen3.6) speed
> scoring is **no longer scheduled** — the every-2-hours cron entry has been removed. Standalone
> Qwen3.5/Qwen3.6 optimization PRs no longer get an automatic `eval:<LABEL>`; only DFlash PRs are
> scored (see "DFlash PR auto-evaluation bot" below), gated by a no-regression check on Qwen3.5/3.6
> decode + prefill. The script and `--labels-only` reconcile path (greenlight / stale-PR closing)
> still work for manual/ad-hoc runs — it's just not on a timer.

`pr_eval_bot.py` polls open PRs and, for any PR with a **new head commit**, runs the evaluation,
applies an `eval:<LABEL>` label, and posts the result as a PR comment. **It never merges** — merge
manually after review. Idempotent: each commit is evaluated once (tracked by a hidden marker in the
bot's comment), so it only spins the GPU when there's new work.

Each bot run also **closes open PRs with no GitHub activity for 2+ days** (`updatedAt` — commits,
comments, reviews, label changes). **Draft PRs** are closed after **4+ days in draft status**
(`createdAt` or latest `converted_to_draft`; activity does not reset the clock). PRs labeled
`hold` or `merge-first` are skipped. Override with
`SPARKINFER_STALE_PR_DAYS=0` / `SPARKINFER_DRAFT_STALE_DAYS=0` to disable, or set different thresholds.

```bash
eval/setup_labels.sh                                   # one-time: create the eval:* labels
python eval/pr_eval_bot.py --instance 42134865 --frontier 164 --ceiling 366   # one poll
python eval/pr_eval_bot.py --instance 42134865 --dry-run                       # eval but don't post
```

Formerly scheduled every 2 hours via `eval/run_bot_cron.sh`; that crontab entry has been removed
(see status note above). To run it by hand instead:
```bash
python eval/pr_eval_bot.py --instance 42134865 --frontier 164 --ceiling 366   # one poll
./eval/run_bot_cron.sh --labels-only  # greenlight/stale-PR reconcile only, no GPU
```
Each run: **always uses the pinned GPU** (`VAST_DEFAULT_INSTANCE` /
`~/.sparkinfer_pinned_instance`). **Never rents** a new one. If the pin is already running and
SSH works → full eval of new PR commits. If the pin is stopped/unreachable → `--labels-only`
(greenlight / needs-benchmark / merge-first reconcile, no GPU). Needs
`gh` authenticated and `VAST_INSTANCE` / `VAST_DEFAULT_INSTANCE` in `.env.eval`
(`VAST_NO_AUTO_PROVISION=1`).

## Qwen3.8-27B PR auto-evaluation bots

> **On cron since 2026-09-15:** `pr_qwen38_bot.py` hourly at `:30` (`eval/run_qwen38_cron.sh`),
> beside `pr_museglimmer_bot.py` at `:00`. `pr_dspark_bot.py` held `:30` until then and is paused;
> it still runs by hand. Each bot's module docstring is the authority, not this README.
>
> Only one bot may hold the shared `/tmp/sparkinfer_bot.lock` at a time, and they all drive the
> same single pinned GPU, so two bots in one slot would contend. The crontab is host state, not
> repo state — check with `crontab -l` on the eval host (the machine running the bot, **not** the
> GPU box it SSHes into) rather than trusting any schedule written down here.

### `pr_qwen38_bot.py` (on cron, hourly at `:30`)

Scores **same-box PR vs `origin/main`** on the upstream **`unsloth/Qwen3.8-27B-NVFP4`** checkpoint
(a compressed-tensors directory: NVFP4 FFN, FP8 attention and Gated-DeltaNet projections). It
applies `eval-qwen38:{XL,L,M,S,XS,none,REJECT}`, derives the generic `eval:*` tier, picks
`qwen38-merge-first`, and auto-merges that PR when `SPARKINFER_QWEN38_AUTOMERGE=1`. A `none` or
`REJECT` closes the PR.

1. **Speed.** The tier is the best measured delta among:
   - `prefill@16k`;
   - **concurrent decode `cb-decode@c2/c4/c8/c16/c32`** (issue #1080): aggregate tok/s with N
     requests in flight through `ContinuousBatchEngine`, measured as `pr_dspark_bot.py` measures
     the ModelOpt checkpoint:

     ```bash
     SPARKINFER_QWEN38_PREFILL_NVFP4=1 SPARKINFER_QWEN38_DECODE_NVFP4=1 SPARKINFER_KV_INT8=1 \
       build/runtime/qwen3_gguf_cb_bench <checkpoint> N 256 256 512
     ```

   - **long-context decode `modelopt-decode@256k`** (issue #1113): one 262144-token row on the
     ModelOpt NVFP4 checkpoint — the checkpoint the axis was defined on in `pr_dspark_bot.py`, and
     the only one that fits a 32 GB card at that context (30.0 GB peak; the unsloth weights this
     bot otherwise scores are 22 GB before any KV). It runs as its own sweep tier because
     `bench_sweep_run` applies one rep count per call:

     ```bash
     bench_sweep_run "$MODELOPT_GUARD_MODEL_DIR" 128 262144 3
     ```

     A checkpoint that is absent, or a sweep that fails, leaves the axis unscored for that round —
     it is never a rejection.

   `decode@128`, `prefill@128`, `modelopt-prefill@256k` and `cb-decode@c1` are measured as floors.
   Any measured dimension below 98% of `main` is a hard REJECT.

   This checkpoint's packed decode step runs FP8 and Q4_K kernels that the ModelOpt checkpoint never
   uses, and no other bot measures concurrency on it. `main` (`507017b`), aggregate tok/s:

   | requests | 1 | 2 | 4 | 8 | 16 | 32 |
   |---|--:|--:|--:|--:|--:|--:|
   | tok/s | 83.0 | 155.9 | 259.0 | 421.9 | 314.1 | ~249 |

   Throughput falls past 8 requests, and at 32, two of the 33 requests fail to open (out of
   memory). Each width is the **median of three complete runs**. A run counts as complete only if
   every request either finished or failed outright; a run where requests stopped part-way is
   re-run, up to five attempts. One such run on `main` read 293 tok/s at c32 instead of ~249. Single
   runs of identical code at c32 differ by up to 2.7%, which is outside the 2% reject band.

2. **Accuracy gate — differential (PR vs main), not absolute.** llama.cpp cannot read a
   compressed-tensors directory, so there is no same-weights external reference. Instead both
   builds score the same token stream and the two distributions are compared
   (`bench/scripts/accuracy_compare_pair.py`): **top1 ≥ 0.99, KL ≤ 0.01**. Failure is a hard
   REJECT regardless of speed. *Limitation:* it catches newly introduced divergence only — never a
   bug already present on `main`.

   The batched-prefill parity gate (`bench/scripts/prefill_parity_check.py`) is **not run**. It is
   absolute, and `main` fails it on this checkpoint: a common prefix of 3–4 of 24 tokens at n=32
   and n=128, against a 0.75 bar. Running it would reject every PR. `pr_dspark_bot.py` turned it
   off on the ModelOpt checkpoint for the same reason.

3. **No-regression guards on the other models**, 0.98 tolerance. Each is a hard REJECT regardless
   of Qwen3.8's own result:
   - Qwen3.6: decode + prefill at ctx 0/512/4k/16k/32k;
   - the ModelOpt Qwen3.8 checkpoint and Muse Glimmer: decode + prefill at 32k, the same guards
     `pr_museglimmer_bot.py` runs, plus concurrent decode at 16 and 32 requests. Each concurrency
     guard is the median of three complete runs, with each model run the way its own bot runs it.
     Those guards exist because this bot's PRs mostly change packed decode, which a single-request
     guard never enters. Two sessions on `main` agreed within 0.7%.

   The models share `qwen35.cpp`, `inference_engine.cpp` and the packed-decode kernels. The Muse
   bot skips PRs declared for Qwen3.8 alone, so these guards are the only check such PRs get
   against Muse Glimmer. A checkpoint missing from the box is skipped and reported as skipped.

   The reverse also holds: this bot skips PRs declared for Muse Glimmer alone. So `pr_museglimmer_bot.py`
   guards the unsloth checkpoint too, with decode + prefill at 32k beside its ModelOpt and Qwen3.6
   guards.
   - Ternary-Bonsai-2-27B: decode + prefill at 128 and 32k on the PTQ1_0 GGUF, as
     `pr_bonsai_bot.py` runs it (added 2026-09-24; `pr_museglimmer_bot.py` runs the same guard).
     That bot skips PRs declared for Qwen3.8 or Muse Glimmer alone. 128 is included because the
     dense-GGUF prefill work on that model lives at short prompts (#1139: 1.94× at 128, flat at 4k).
     A guard run that measures nothing is retried next round, never read as a regression.

**Not evaluated:**
- PRs whose template declares a different target model (#1027).
- PRs that change the measuring harness: `qwen3_gguf_bench.cpp`, `qwen3_gguf_cb_bench.cpp`,
  `qwen_checkpoint.h`, `qwen3_gguf_config.h`, `eval/` or `bench/scripts/`.

Every ref, `main` included, is built with `main`'s copy of those files.

```bash
python eval/pr_qwen38_bot.py --repo gittensor-ai-lab/sparkinfer   # one poll
python eval/pr_qwen38_bot.py --only-prs 636 --reeval              # re-score one PR
python eval/pr_qwen38_bot.py --labels-only                        # no GPU, reconcile labels only
```

Box paths are `QWEN38_*` in `.env.eval`. `QWEN38_MODEL_DIR` defaults to
`/root/workspace/models_qwen38`. On the current box `QWEN38_REMOTE_REPO` shares the DSpark bot's
clone, `/workspace/eval/bot_repo`, because the disk has no room for a second checkout.

### `pr_dspark_bot.py` (paused 2026-09-15, run by hand)

Scores DSpark decode and DSpark-enabled batched prefill at **4k, 16k, and 32k**, target-model
prefill and decode at **256k**, and concurrent decode at c2–c32, all on the ModelOpt checkpoint
(`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`). Every measured axis is also a no-regression
floor; decode is lossless against same-process AR and acceptance cannot regress materially. The
Qwen3.6 / Qwen3.8 shared-path guards remain mandatory. To resume it, put
`eval/run_dspark_cron.sh` back on `:30` and move `run_qwen38_cron.sh` off that slot first.

`pr_modelopt_bot.py` and `pr_dflash_bot.py` are kept for reference and run by hand only.

## Ternary-Bonsai-2-27B PR auto-evaluation bot

### `pr_bonsai_bot.py` (on cron every two hours at `:15`)

Added 2026-09-24 for issue #1138. Ternary-Bonsai-2-27B is on `main` (#1124), but until this bot
existed a speedup on it scored `none` on the other bots, which measure other models. #1139 was the
first such PR.

Scores **same-box PR vs `origin/main`** on `prism-ml/Ternary-Bonsai-2-27B-gguf`
(`Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5,946,648,928 bytes), default folded loader —
`SPARKINFER_BONSAI_NATIVE` unset, as the server runs it. It applies
`eval-bonsai:{XL,L,M,S,XS,none,REJECT}` and derives the generic `eval:*` tier. It picks
`bonsai-merge-first`, but **neither auto-merges nor auto-closes** unless
`SPARKINFER_BONSAI_AUTOMERGE=1` / `SPARKINFER_BONSAI_AUTOCLOSE=1` (REJECT only) are set. It
evaluates every PR not declared for a different model, most of which are not aimed at this one, so
`none` never closes anything here.

1. **Speed.** Decode and prefill at ctx 128/512/4k/16k/32k (one `bench_sweep_run`, reps 5), plus
   concurrent decode at c2–c32 (median of three complete runs per width, as in `pr_qwen38_bot.py`).
   The tier is the best measured delta; any axis below 98% of `main` is a REJECT. A concurrency
   width that fails to run is dropped, never scored as zero.

2. **Accuracy — three gates.** llama.cpp cannot read PTQ1_0, so there is no external reference.
   - *Differential score:* `qwen3_gguf_score` on the PR build and on `main` over
     `bench/scripts/eval_corpus.txt` (~1,200 tokens), compared by `accuracy_compare_pair.py`:
     **top1 ≥ 0.93, KL ≤ 0.03, PPL ≤ 1.02× main**. The folded path is not bit-deterministic across
     processes, so the bars come from a measured main-against-main spread (below), not from the
     Qwen3.8 bot's 0.99/0.01.
   - *Prefill path:* `qwen3_gguf_score` never enters batched prefill. `qwen3_gguf_prefill_check`
     compares batched prefill with the token loop at prefix 128 (inside the fused quantized-B
     GEMM's M ≤ 512 window) and 1024 (outside it), 64 teacher-forced positions, three runs a side.
     The PR's mean must stay within `max(3× main's KL, main + 0.05)` and 0.10 of main's top-1.
   - *`eval/bonsai_regression.py`* (tensors, score, generate, serve) on the PR build. It is
     absolute, so it rejects only when `main` passes it in the same round.

3. **No-regression guards @ 32k, decode + prefill:** Qwen3.6-35B-A3B, the ModelOpt and unsloth
   Qwen3.8-27B checkpoints, and Muse Glimmer. The Muse and Qwen3.8 bots skip PRs declared for
   Ternary-Bonsai-2-27B alone, so these guards are the only check such PRs get against those models.
   In the other direction, both of those bots guard Ternary-Bonsai-2-27B at 128 and 32k. A guard
   that measures nothing is retried next round; an absent checkpoint is skipped and reported.

**Failures.** A fault on the box — GPU not drained, a failed fetch, a missing model, an SSH drop,
an OOM kill — is retried next round with nothing posted. A PR that fails to build or crashes gets
`eval-bonsai:REJECT` once for that commit.

**Not evaluated:** PRs declared for other models only; PRs that change the harness (the files
`pr_qwen38_bot.py` pins, plus `qwen3_gguf_score.cpp`, `qwen3_gguf_generate.cpp`,
`qwen3_gguf_prefill_check.cpp` and `bonsai_inspect.cpp`). Every ref is measured as the PR merged
into `main` when GitHub publishes that ref, with `main`'s harness.

```bash
python eval/pr_bonsai_bot.py --dry-run                            # what would be evaluated, no GPU
python eval/pr_bonsai_bot.py --only-prs 1139 --reeval --no-post   # measure one PR, post nothing
python eval/pr_bonsai_bot.py --labels-only                        # no GPU, reconcile labels only
```

Box paths: `BONSAI_REMOTE_REPO` (default `/root/sparkinfer_bonsai`, its own clone because it builds
with `-DBUILD_SERVER=ON`; not under `/workspace`, whose overlay filesystem makes cargo fail to write
the server's `tokenizers-c` archive with `Bad address (os error 14)`), `BONSAI_GGUF`,
`BONSAI_TOKENIZER_DIR` (the GGUF carries no `tokenizer.json`; the model shares Qwen3.8's).

Measured on the eval box when the bot was added (2026-09-24, `main` `bb67474`, RTX 5090):

| ctx | decode tok/s | prefill tok/s |
|---|--:|--:|
| 128 | 99.1 | 2,090 |
| 512 | 98.4 | 3,882 |
| 4k | 97.0 | 8,699 |
| 16k | 93.7 | 8,407 |
| 32k | 88.7 | 8,081 |

Concurrent decode c2/c4/c8/c16/c32: 163 / 263 / 498 / 783 / 1,061 tok/s.

- **`main` against itself:** every speed axis within ±0.9%, every gate passing, `none`. Two loads
  of one build do not agree exactly on this model: over eight pairs of processes, top-1 0.959–0.978,
  KL 0.0084–0.0133, PPL ratio 0.995–1.005. `SPARKINFER_DETERMINISTIC=1` does not change that. So the
  score gate is **top1 ≥ 0.93, KL ≤ 0.03, PPL ≤ 1.02× main** — outside that spread, with the PPL
  ratio as the sharp edge — rather than the Qwen3.8 bot's 0.99/0.01, which rejected `main` against
  itself in the first validation round.
- **#1139 against it:** `eval-bonsai:XL` from prefill@128, 2,090 → 4,186 tok/s (+100.3%); every
  other axis within −0.7% / +1.5%; top-1 0.964, KL 0.0102, PPL ×1.0002; the prefill path,
  `bonsai_regression.py` and all four guards passing.
- **Cost:** ~20 minutes of GPU per ref (build 2–4 min, speed 1.5, concurrency 7, score 0.5, prefill
  check 3, `bonsai_regression.py` 2.3, guards 2.7), so a round with one pending PR holds the box for
  ~40 minutes. A round with nothing to evaluate never touches the GPU.

## DFlash PR auto-evaluation bot (retired from cron)

> **Status: retired from cron** — superseded by the Qwen3.8-27B bot above. Still runnable by hand.

`pr_dflash_bot.py` evaluates PRs that touch DFlash paths (`dflash*`, `qwen3_gguf_dflash_*`,
`dflash_accuracy.sh`). It scores **same-box PR DFlash tok/s vs `origin/main` DFlash tok/s**,
applies `eval-dflash:{XL,L,M,S,XS,none,REJECT}`, picks `dflash-merge-first`, and can auto-merge
(`SPARKINFER_AUTOMERGE=1`) when accuracy passes (SPEC_AGREE) and the tier is a verified speedup.

**Qwen3.5/3.6 no-regression guard.** On the *same PR build* used for the DFlash bench, the bot
also runs the standard decode + prefill sweep for Qwen3.6 (128/512/4k/16k/32k) and Qwythos/Qwen3.5
(128/4k/32k/64k decode, 4k/32k/64k/128k prefill) — once for the PR ref, once for `origin/main` —
and compares every context pairwise (same 0.98 no-regression tolerance as the AR bot's guards). If
*either* model regresses on *either* metric at *any* context, the DFlash tier is overridden to
`eval-dflash:REJECT` regardless of how large the DFlash speedup was — the PR comment lists exactly
which context/metric failed. This means DFlash PRs now need `/workspace/models35` (Qwythos) and
`/workspace/models36` (Qwen3.6) present on the eval box; the bot downloads/verifies them itself
(same `_common.sh` `ensure_model`/`ensure_tokenizer` helpers as the AR bot) if missing. Expect
longer per-PR eval time than before (two extra model sweeps, PR + main).

```bash
eval/setup_labels.sh                                  # creates eval-dflash:* + dflash-merge-*
./eval/run_dflash_bot.sh                               # one poll
./eval/run_dflash_bot.sh --only-prs 636 --reeval       # force one PR
```

**Schedule every hour, on the hour** (shares `/tmp/sparkinfer_bot.lock` with the AR bot):
```bash
crontab -l 2>/dev/null; echo "0 * * * * $PWD/eval/run_dflash_cron.sh >> /tmp/sparkinfer_dflash_bot.log 2>&1" | crontab -
```
Pinned GPU only; never rents. If the pin is down → `--labels-only` reconcile.

**Dashboard.** Eval verdicts and frontier updates are committed to
[`gittensor-ai-lab/sparkinfer-web`](https://github.com/gittensor-ai-lab/sparkinfer-web)
(`public/dashboard/data.json`), not to this repo's `dashboard/`. Override with
`SPARKINFER_WEB_REPO` / `SPARKINFER_WEB_DIR` / `SPARKINFER_WEB_BRANCH` (default branch:
`feat/landing-page`).

**Dashboard merge-sync (no GPU).** The heavy eval cron may not run for hours, and `record_merge()`
only used to fire for merged PRs that still had `merge-first`. Run `run_sync_cron.sh` every 15 min
alongside it — it syncs **any recently merged PR** that has dashboard eval data onto the
frontier/journey and reconciles round labels (never evaluates, never merges), sharing the eval lock
so the two never overlap:
```bash
crontab -l 2>/dev/null; echo "*/15 * * * * $PWD/eval/run_sync_cron.sh >> /tmp/sparkinfer_sync.log 2>&1" | crontab -
```

(For a Claude-agent flavor instead of system cron — e.g. to add LLM anti-gaming triage of the diff
before labeling — schedule a recurring agent that shells out to `pr_eval_bot.py`; the numeric label
still comes from the deterministic evaluator so validators converge.)

## Status / notes

- The **on-instance evaluator** (`bench/scripts/evaluate.sh` + `label.py`) reuses the tested
  `bench.sh` / `accuracy.sh`. The **vast lifecycle** (search/create/ssh/destroy) needs *your* key
  to run — validate the vast-specific calls (offer query, `--image`, instance field names) on the
  first run and adjust if your account's defaults differ.
- First eval on a fresh box builds llama.cpp (~10–15 min); it persists at `/workspace/.llamacpp`.
- Correctness gates vs **llama.cpp** for GGUF models. The Qwen3.8-27B bot instead gates
  **PR vs main** (score-vs-baseline: ~100% top-1 + KL≈0), which is the extension suggested here —
  necessary there because llama.cpp cannot read a compressed-tensors checkpoint at all.
- Anti-gaming (an LLM/KDA agent reading the diff for benchmark-special-casing, weakened tolerances,
  harness edits) is a layer *on top* — it flags, it doesn't set the numeric label.
