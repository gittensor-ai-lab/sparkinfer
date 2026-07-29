#!/usr/bin/env bash
# Run the sparkinfer DFlash PR eval bot once (interactive).
#
#   ./eval/run_dflash_bot.sh
#   ./eval/run_dflash_bot.sh --only-prs 636 --reeval
#   ./eval/run_dflash_bot.sh --dry-run
#   EVAL_SKIP_GIT_SYNC=1 ./eval/run_dflash_bot.sh   # use local eval/ changes
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ "${EVAL_SKIP_GIT_SYNC:-0}" != "1" ]; then
  git fetch -q origin main 2>/dev/null || true
  if git rev-parse --verify origin/main >/dev/null 2>&1; then
    git checkout -q main 2>/dev/null || true
    git reset --hard origin/main
  fi
fi

if [ -f "$REPO_DIR/.env.eval" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_DIR/.env.eval"
  set +a
else
  echo "!! missing $REPO_DIR/.env.eval — copy .env.eval.example and fill in secrets" >&2
  exit 1
fi

export SSH_KEY="${SSH_KEY:-$HOME/.ssh/speedy}"
export PATH="/usr/local/bin:/usr/bin:/bin:${HOME}/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export SPARKINFER_AUTOMERGE="${SPARKINFER_AUTOMERGE:-1}"
export SPARKINFER_AUTOMERGE_ADMIN="${SPARKINFER_AUTOMERGE_ADMIN:-1}"
export VAST_NO_AUTO_PROVISION=1

BOT_ARGS=(--repo "${REPO:-gittensor-ai-lab/sparkinfer}")
if [ "${EVAL_TRANSPORT:-vast}" != "ssh" ]; then
  BOT_ARGS+=(--instance "${VAST_INSTANCE:-${VAST_DEFAULT_INSTANCE:-0}}")
fi

echo "[$(date -u +%FT%TZ)] dflash eval bot (EVAL_TRANSPORT=${EVAL_TRANSPORT:-vast}, AUTOMERGE=${SPARKINFER_AUTOMERGE:-0})"
exec python3 eval/pr_dflash_bot.py "${BOT_ARGS[@]}" "$@"
