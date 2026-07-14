#!/usr/bin/env bash
# Close open PRs with no GitHub activity for 2+ days (same policy as close-stale-prs.yml).
#
# Schedule daily alongside the eval bot:
#   0 6 * * * /path/to/sparkinfer/eval/run_stale_pr_cron.sh >> /tmp/sparkinfer_stale.log 2>&1
set -euo pipefail
export HOME="${HOME:-/home/speedy}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

exec 9>/tmp/sparkinfer_bot.lock
flock -n 9 || { echo "[$(date -u +%FT%TZ)] bot lock held — skip stale PR sweep"; exit 0; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1
git pull -q origin main 2>/dev/null || true

echo "[$(date -u +%FT%TZ)] close stale PRs (threshold=${SPARKINFER_STALE_PR_DAYS:-2}d)"
python3 -c "
import os, sys
sys.path.insert(0, 'eval')
import pr_eval_bot as bot
repo = os.environ.get('REPO', 'gittensor-ai-lab/sparkinfer')
days = int(os.environ.get('SPARKINFER_STALE_PR_DAYS', '2'))
closed = bot.close_stale_prs(repo, days=days)
print('closed:', sorted(closed))
"
