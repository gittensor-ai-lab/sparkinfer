#!/usr/bin/env bash
# Lightweight dashboard merge-sync (NO GPU). Records recently merged PRs onto the dashboard
# (frontier + optimization journey) and reconciles the round labels, so a MANUAL merge shows up
# within minutes — even while the heavy 2-hour eval cron is paused for manual work. It never
# evaluates and never auto-merges (it just reflects what's already merged).
#
# Schedule it every 15 min, alongside (and independent of) the bot wrappers:
#   */15 * * * * GH_TOKEN="$(gh auth token -u <bot account>)" /path/to/sparkinfer/eval/run_sync_cron.sh >> /tmp/sparkinfer_sync.log 2>&1
# Without GH_TOKEN every tick is refused (cron_common.sh).
export HOME="${HOME:-/home/speedy}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
unset SPARKINFER_AUTOMERGE          # sync NEVER merges — only records merges + labels

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=eval/cron_common.sh
source "$REPO_DIR/eval/cron_common.sh" || exit 1
# The bots' ticks fire on these same minutes and wait LOCK_WAIT_S for the lock: the sync's own run
# (its --kill-after included) stays well inside that; only a git fetch GitHub is slow to answer
# can add to it.
BOT_TIMEOUT_S="${SPARKINFER_SYNC_TIMEOUT_S:-180}"

# Share the eval lock so a sync can never overlap an eval run (or another sync). Non-blocking:
# if an eval/sync is active, skip this tick (the next one picks it up).
exec 9>"$LOCK_FILE"
flock -n 9 || exit 0

cd "$REPO_DIR" || { note_refused sync "cannot enter $REPO_DIR"; exit 1; }
# .env.eval too (SPARKINFER_BOT_LOGIN, SPARKINFER_BOT_TREE): the sync closes PRs as well. It never
# merges, whatever .env.eval says.
if [ -f "$REPO_DIR/.env.eval" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_DIR/.env.eval"
  set +a
fi
unset SPARKINFER_AUTOMERGE
require_bot_token sync || exit 1
prepare_bot_tree || { note_refused sync "$TREE_WHY"; exit 1; }   # never REPO_DIR's checkout
cd "$BOT_TREE" || { note_refused sync "cannot enter $BOT_TREE"; exit 1; }
note_ran sync
echo "[$(date -u +%FT%TZ)] sparkinfer dashboard sync"
run_bot sync python3 -c "import sys; sys.path.insert(0,'eval'); import pr_eval_bot as b; r='${REPO:-gittensor-ai-lab/sparkinfer}'; b.close_exhausted_eval_prs(r); b.reconcile_merge_labels(r)"
