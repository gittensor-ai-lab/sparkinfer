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

> **The scored path is `pr_dspark_bot.py`** (`eval/run_dspark_cron.sh`, hourly at `:00`), and as
> of 2026-08-21 it is the ONLY bot on cron. It scores DSpark decode and DSpark-enabled batched
> prefill at **4k, 16k, and 32k**, plus target-model prefill at **256k**, on the ModelOpt checkpoint
> (`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`). Every measured axis is also a no-regression
> floor; decode is lossless against same-process AR and acceptance cannot regress materially.
> The Qwen3.6 / Qwen3.8 shared-path guards remain mandatory. See that file's module docstring —
> it is the authority, not this README.
>
> The 256k row uses `qwen3_gguf_bench` sweep mode with one exact 262,144-token context,
> checkpoint-native NVFP4 prefill/decode representations, and INT8 KV. Main currently uses the sequential prefill
> fallback there, so one main or PR measurement takes roughly 65 minutes on the pinned RTX 5090.
> The hourly lock therefore skips overlapping ticks; it never starts concurrent GPU evaluations.
>
> Every other bot here — `pr_qwen38_bot.py`, `pr_modelopt_bot.py`, `pr_museglimmer_bot.py`,
> `pr_dflash_bot.py` — is kept for reference and run by hand only. The sections below describe
> them as they were when they held the scored slot; each bot's own docstring is current, this
> README is not.
>
> Only one bot may hold the shared `/tmp/sparkinfer_bot.lock` at a time, and they all drive the
> same single pinned GPU, so two on cron would contend. The crontab is host state, not repo
> state — check with `crontab -l` on the eval host (the machine running the bot, **not** the GPU
> box it SSHes into) rather than trusting any schedule written down here.

### `pr_qwen38_bot.py` (superseded, hand-run only)

Scores **same-box PR vs `origin/main`** on three axes, applies
`eval-qwen38:{XL,L,M,S,XS,none,REJECT}`, mirrors it to `eval:*` (SN74 scoring reads `eval:*`),
picks `qwen38-merge-first`, and can auto-merge (`SPARKINFER_QWEN38_AUTOMERGE=1`, off by default).

1. **Speed — decode @ ctx=128**, median of 5 reps. The scored checkpoint is the HuggingFace
   **compressed-tensors NVFP4 directory** (`unsloth/Qwen3.8-27B-NVFP4`), *not* a GGUF — that is
   what `sparkinfer-server` actually serves. `qwen3_gguf_bench`/`qwen3_gguf_score` read directories
   via `runtime/examples/qwen_checkpoint.h`, shared with the server so the benchmark and the
   server cannot disagree about how a checkpoint is configured.

   Prefill is deliberately *not* scored: at ctx=128 the batched prefill path declines (it needs
   int8 KV, enabled only at ctx≥4096), so the number would measure the sequential fallback.

2. **Accuracy gate — differential (PR vs main), not absolute.** llama.cpp cannot read a
   compressed-tensors directory, so there is no same-weights external reference. Instead both
   builds score the same token stream and the two distributions are compared
   (`bench/scripts/accuracy_compare_pair.py`): **top1 ≥ 0.99, KL ≤ 0.01**. Tight, because two
   builds of the same model on the same box should agree almost exactly. Failure is a hard
   REJECT regardless of speed.

   *Limitation:* a differential gate catches newly introduced divergence only — never a bug
   already present on `main`.

3. **Qwen3.6 no-regression guard** — decode + prefill at ctx 0/512/4k/16k/32k, 0.98 tolerance.
   Qwen3.8 and Qwen3.6 share `qwen35.cpp`/`inference_engine.cpp`; a regression there is a hard
   REJECT regardless of Qwen3.8's own result.

```bash
python eval/pr_qwen38_bot.py --repo gittensor-ai-lab/sparkinfer   # one poll
python eval/pr_qwen38_bot.py --only-prs 636 --reeval              # re-score one PR
python eval/pr_qwen38_bot.py --labels-only                        # no GPU, reconcile labels only
```

Box paths are `QWEN38_*` in `.env.eval` (`QWEN38_MODEL_DIR` defaults to
`/root/workspace/models_qwen38`).

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
