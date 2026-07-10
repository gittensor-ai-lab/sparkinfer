#!/usr/bin/env bash
# Run the sparkinfer PR eval bot once (interactive). Sources .env.eval for transport + secrets.
#
#   ./eval/run_bot.sh              # full run on SSH box (or vast if EVAL_TRANSPORT=vast)
#   ./eval/run_bot.sh --dry-run    # poll PRs + print plan, no GPU eval
#   ./eval/run_bot.sh --dual       # force dual-model eval (Qwen3.6 + Qwen3-30B guard)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

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

BOT_ARGS=(
  --frontier "${FRONTIER:-285}"
  --ceiling  "${CEILING:-366}"
  --repo     "${REPO:-gittensor-ai-lab/sparkinfer}"
)
if [ "${EVAL_TRANSPORT:-vast}" != "ssh" ]; then
  BOT_ARGS+=(--instance "${VAST_INSTANCE:-42682383}")
fi
if [ -n "${DUAL:-}" ] || printf '%s\n' "$@" | grep -qx -- '--dual'; then
  BOT_ARGS+=(--dual)
fi
if [ -n "${POLARIS:-}" ] || printf '%s\n' "$@" | grep -qx -- '--polaris'; then
  BOT_ARGS+=(--polaris)
fi

echo "[$(date -u +%FT%TZ)] eval bot (EVAL_TRANSPORT=${EVAL_TRANSPORT:-vast}, SSH_KEY=$SSH_KEY)"
exec python3 eval/pr_eval_bot.py "${BOT_ARGS[@]}" "$@"
