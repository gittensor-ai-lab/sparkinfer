# Automatic evaluation (vast.ai)

Provision (or reuse) a Blackwell GPU on vast.ai, build a sparkinfer submission, gate it for
**correctness**, measure its **speed**, and assign an eval-loop **label** — automatically.

```
submission (git ref) ─► build from source ─► correctness gate (token-match / KL vs llama.cpp)
                     ─► 2k no-regression gate ─► 16k speed score ─► LABEL
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

The default eval target is now long-context decode:
- **2k context**: no-regression guard. A PR must keep at least 98% of same-box `origin/main` 2k speed.
- **16k context**: scored frontier. Labels are based on verified 16k speedup over same-box `origin/main`.
- **32k context**: telemetry only for now. It is published in the verdict/log but does not score yet.

Set `SPARKINFER_EVAL_MODE=short` or pass `--eval-mode short` to keep the legacy 128-token scoring path.

## Verdict (stdout)

```json
{ "commit": "abc1234", "tps": 165.2, "top1": 1.0, "kl": 0.14, "frontier_tps": 164,
  "pass": true, "label": "none", "delta_tps": 1.2, "pct_over_frontier": 0.7 }
```
Labels: **REJECT** (failed correctness or the 2k no-regression gate) · **none** (within the significance gate) ·
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

**Dashboard merge-sync (no GPU).** The heavy eval cron records a merge only on its next tick, so a
*manual* merge leaves the dashboard stale while it's paused. Run `run_sync_cron.sh` every 15 min
alongside it — it just records merged `merge-first` PRs onto the frontier/journey and reconciles
labels (never evaluates, never merges), sharing the eval lock so the two never overlap:
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
