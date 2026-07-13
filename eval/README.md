# Automatic evaluation (vast.ai or fixed SSH box)

Provision (or reuse) a Blackwell GPU on vast.ai, **or** use a fixed bare-metal box via SSH.
Build a sparkinfer submission, gate it for **correctness**, measure its **speed**, and assign an
eval-loop **label** — automatically.

## Transport

| `EVAL_TRANSPORT` | Behavior |
|------------------|----------|
| `vast` (default) | Full vast.ai rent / reuse / stop logic — unchanged |
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
# reuse a box (started if stopped) — evaluate, then STOP it again (the default):
python eval/vast_eval.py --reuse <instance_id> --frontier 164 --ceiling 366 --ref main

# evaluate then DESTROY (frees the disk), or --keep to leave it running:
python eval/vast_eval.py --ref <git-ref> --frontier 164 --ceiling 366 --destroy
```

**The instance is STOPPED after every eval by default** — compute billing pauses while the disk
and cached weights (`/workspace/models`) persist, so the next `--reuse` run starts fast.
`--keep` leaves it running; `--destroy` frees the disk too.

`--frontier` = current best tok/s for the scored target · `--ceiling` = roofline/reference display
value. Reuse mode assumes the weights are cached at `/workspace/models`.

The default eval target is now multi-context decode:
- **128-token, 512-context, 4k-context, 16k-context, and 32k-context decode** are all no-regression guards. A PR must keep at least 98% of same-box `origin/main` speed at every measured context.
- The **strongest single context improvement** becomes the scored target for `eval:<label>`. Improvements are never aggregated across contexts; two sub-2% gains do not combine into a score.
- The bot also applies a UI-only context label (`128-context`, `512-context`, `4k-context`, `16k-context`, or `32k-context`) for the context that improved most. This does not change the score.
- If a PR has both a real context win and a regression elsewhere, it is not rejected automatically; the bot adds `regression-128`, `regression-512`, `regression-4k`, `regression-16k`, and/or `regression-32k` labels for the regressed contexts. Regression labels block auto-merge and require maintainer judgment.
- If no single context clears the 2% significance gate and any context regresses, the bot returns `eval:REJECT` and auto-closes the PR.
- Difficulty compensation uses the selected context's llama.cpp baseline, so late-game improvements past the mature reference get the same multiplier logic at every context.

32k is intentionally sampled once by default (`SPARKINFER_GUARD_32K_REPS=1`) to keep the eval cost bounded while still making long-context regressions and wins visible.

Set `SPARKINFER_EVAL_MODE=short` or pass `--eval-mode short` to keep the legacy 128-token scoring path.

## Bidirectional scoring: Qwen3.5 + Qwen3.6 (default)

`--bidir` (or `BIDIR=1` / legacy `TRIPLE=1` in `.env.eval`) scores **both directions** in one build:

```
build once ─► score_qwen35  Qwythos-9B : 128/4k/32k/64k/128k speed + accuracy ─► eval-qwen35:<LABEL>
           │              guard Qwen3.6  : 5 contexts ─► must NOT regress
           └► score_qwen36  Qwen3.6      : 128/512/4k/16k/32k ─► eval-qwen36:<LABEL>
                          guard Qwen3.5  : 128/512/4k ─► must NOT regress
```

- **Qwen3.5** (Qwythos-9B) is measured at **128, 512, 4k only** — not 16k/32k.
- **Qwen3.6** runs the full **5-context** sweep (128/512/4k/16k/32k).
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

## PR auto-evaluation bot

`pr_eval_bot.py` polls open PRs and, for any PR with a **new head commit**, runs the evaluation,
applies an `eval:<LABEL>` label, and posts the result as a PR comment. **It never merges** — merge
manually after review. Idempotent: each commit is evaluated once (tracked by a hidden marker in the
bot's comment), so it only spins the GPU when there's new work.

```bash
eval/setup_labels.sh                                   # one-time: create the eval:* labels
python eval/pr_eval_bot.py --instance 42134865 --frontier 164 --ceiling 366   # one poll
python eval/pr_eval_bot.py --instance 42134865 --dry-run                       # eval but don't post
```

**Schedule it every 2 hours** (the wrapper gives cron a sane env + refreshes the evaluator):
```bash
crontab -l 2>/dev/null; echo "0 */2 * * * $PWD/eval/run_bot_cron.sh >> /tmp/sparkinfer_bot.log 2>&1" | crontab -
```
Each run: reuse the pinned instance if it survived, else provision fresh (Google Drive model) →
evaluate new PR commits → **stop it again** → label + comment. Disable with `crontab -e`. Needs `gh` authenticated and the vast key saved (`vastai set api-key`).

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
- Correctness currently gates vs **llama.cpp**. For an optimization PR, also gate vs the **previous
  frontier build** (score-vs-baseline: ~100% top-1 + KL≈0) — a small extension to `evaluate.sh`.
- Anti-gaming (an LLM/KDA agent reading the diff for benchmark-special-casing, weakened tolerances,
  harness edits) is a layer *on top* — it flags, it doesn't set the numeric label.
