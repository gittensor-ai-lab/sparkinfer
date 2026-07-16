#!/usr/bin/env bash
# Lightweight dashboard merge-sync (NO GPU). Records recently merged PRs onto the dashboard
# (frontier + optimization journey) and reconciles the round labels, so a MANUAL merge shows up
# within minutes — even while the heavy 2-hour eval cron is paused for manual work. It never
# evaluates and never auto-merges (it just reflects what's already merged).
#
# Schedule it every 15 min, alongside (and independent of) run_bot_cron.sh:
#   */15 * * * * /home/speedy/gittensor-ai-lab/sparkinfer/eval/run_sync_cron.sh >> /tmp/sparkinfer_sync.log 2>&1
export HOME="${HOME:-/home/speedy}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
unset SPARKINFER_AUTOMERGE          # sync NEVER merges — only records merges + labels

# Share the eval lock so a sync can never overlap an eval run (or another sync). Non-blocking:
# if an eval/sync is active, skip this tick (the next one picks it up).
exec 9>/tmp/sparkinfer_bot.lock
flock -n 9 || exit 0

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1
git pull -q origin main 2>/dev/null || true        # keep the bot + dashboard current
echo "[$(date -u +%FT%TZ)] sparkinfer dashboard sync"
python3 -c "import sys; sys.path.insert(0,'eval'); import pr_eval_bot as b; r='${REPO:-gittensor-ai-lab/sparkinfer}'; b.close_exhausted_eval_prs(r); b.reconcile_merge_labels(r)"
