#!/usr/bin/env bash
# Run the sparkinfer PR eval bot once (interactive). Sources .env.eval for transport + secrets.
#
#   ./eval/run_bot.sh              # full run on SSH box (Polaris TDX on by default)
#   ./eval/run_bot.sh --dry-run    # poll PRs + print plan, no GPU eval
#   ./eval/run_bot.sh --bidir      # Qwen3.5 + Qwen3.6 bidirectional eval (default)
#   ./eval/run_bot.sh --no-polaris # skip Polaris TDX receipts
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Harness + eval code must match origin/main (vast_eval syncs bench/scripts from there).
git fetch -q origin main 2>/dev/null || true
if git rev-parse --verify origin/main >/dev/null 2>&1; then
  git checkout -q main 2>/dev/null || true
  git reset --hard origin/main
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

BOT_ARGS=(
  --frontier "${FRONTIER:-285}"
  --ceiling  "${CEILING:-366}"
  --repo     "${REPO:-gittensor-ai-lab/sparkinfer}"
)
if [ "${EVAL_TRANSPORT:-vast}" != "ssh" ]; then
  BOT_ARGS+=(--instance "${VAST_INSTANCE:-42682383}")
fi
if printf '%s\n' "$@" | grep -qx -- '--bidir' || \
   [ -n "${BIDIR:-${TRIPLE:-1}}" ] && [ "${BIDIR:-${TRIPLE:-1}}" != "0" ] || \
   [ -n "${DUAL:-}" ] || printf '%s\n' "$@" | grep -qxE -- '--triple|--dual'; then
  BOT_ARGS+=(--bidir --primary-quant "${PRIMARY_QUANT:-Q4_K_M}")
fi
if printf '%s\n' "$@" | grep -qx -- '--no-polaris' || [ "${POLARIS:-1}" = "0" ]; then
  BOT_ARGS+=(--no-polaris)
else
  BOT_ARGS+=(--polaris)
fi

echo "[$(date -u +%FT%TZ)] eval bot (EVAL_TRANSPORT=${EVAL_TRANSPORT:-vast}, POLARIS=${POLARIS:-1}, SSH_KEY=$SSH_KEY)"
exec python3 eval/pr_eval_bot.py "${BOT_ARGS[@]}" "$@"
