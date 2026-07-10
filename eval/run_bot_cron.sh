#!/usr/bin/env bash
# Cron wrapper for the sparkinfer PR auto-eval bot — gives cron a sane env, refreshes the
# evaluator from main, and runs one poll. Schedule it every 2 hours:
#
#   0 */2 * * * /home/speedy/gittensor-ai-lab/sparkinfer/eval/run_bot_cron.sh >> /tmp/sparkinfer_bot.log 2>&1
#
# Transport (set in .env.eval or env):
#   EVAL_TRANSPORT=vast  — rent/reuse vast.ai (default)
#   EVAL_TRANSPORT=ssh   — fixed GPU box via EVAL_SSH_HOST / EVAL_SSH_PORT (no vast billing)
export HOME="${HOME:-/home/speedy}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export SPARKINFER_AUTOMERGE=1   # auto-merge the round's merge-first winner (guarded). Set 0 to disable.

exec 9>/tmp/sparkinfer_bot.lock
flock -n 9 || { echo "[$(date -u +%FT%TZ)] previous bot run still active — skipping this tick"; exit 0; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1

# Load local eval secrets + transport config (not committed).
if [ -f "$REPO_DIR/.env.eval" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_DIR/.env.eval"
  set +a
fi

git pull -q origin main 2>/dev/null || true
echo "[$(date -u +%FT%TZ)] sparkinfer PR bot run (EVAL_TRANSPORT=${EVAL_TRANSPORT:-vast})"

BOT_ARGS=(
  --frontier "${FRONTIER:-285}"
  --ceiling  "${CEILING:-366}"
  --repo     "${REPO:-gittensor-ai-lab/sparkinfer}"
)
# vast.ai path still needs --instance; SSH path ignores it.
if [ "${EVAL_TRANSPORT:-vast}" != "ssh" ]; then
  BOT_ARGS+=(--instance "${VAST_INSTANCE:-42682383}")
fi
[ "${BIDIR:-${TRIPLE:-1}}" != "0" ] && BOT_ARGS+=(--bidir --primary-quant "${PRIMARY_QUANT:-Q4_K_M}")
if [ "${POLARIS:-1}" = "0" ]; then
  BOT_ARGS+=(--no-polaris)
else
  BOT_ARGS+=(--polaris)
fi

python3 eval/pr_eval_bot.py "${BOT_ARGS[@]}"
