#!/usr/bin/env bash
# Cron wrapper for the sparkinfer PR auto-eval bot — gives cron a sane env, refreshes the
# evaluator from main, and runs one poll. Schedule it every 30 min:
#
#   */30 * * * * /home/speedy/gittensor-ai-lab/sparkinfer/eval/run_bot_cron.sh >> /tmp/sparkinfer_bot.log 2>&1
#
# Override params via env (or edit the defaults):  VAST_INSTANCE, FRONTIER, CEILING, REPO.
export HOME="${HOME:-/home/speedy}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"

# Single-run lock: a full run can exceed the 30-min cron interval (model download, multi-PR queue).
# Without this, the next tick starts a SECOND bot on top of the first — overlapping runs that each
# provision boxes => duplicate work and instance churn ("keep restarting"). Non-blocking: if a run
# is still active, skip this tick.
exec 9>/tmp/sparkinfer_bot.lock
flock -n 9 || { echo "[$(date -u +%FT%TZ)] previous bot run still active — skipping this tick"; exit 0; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1
git pull -q origin main 2>/dev/null || true     # keep the bot + evaluator current
echo "[$(date -u +%FT%TZ)] sparkinfer PR bot run"
python3 eval/pr_eval_bot.py \
  --instance "${VAST_INSTANCE:-42134865}" \
  --frontier "${FRONTIER:-164}" \
  --ceiling  "${CEILING:-366}" \
  --repo     "${REPO:-gittensor-ai-lab/sparkinfer}"
