#!/usr/bin/env bash
# Cron wrapper for the sparkinfer PR auto-eval bot — one poll every 2 hours:
#
#   0 */2 * * * /home/autotiny/Desktop/sparkinfer/eval/run_bot_cron.sh >> /tmp/sparkinfer_bot.log 2>&1
#
# Policy:
#   • Never rent a new vast.ai GPU (pinned instance only; VAST_NO_AUTO_PROVISION=1).
#   • If the pinned box is running AND SSH-reachable → full eval + auto-merge.
#   • If the box is down/stopped/unreachable → labels only (greenlight / needs-benchmark /
#     merge-first reconcile). No start_instance, no provision.
#
# Transport (set in .env.eval or env):
#   EVAL_TRANSPORT=vast  — reuse pinned vast.ai instance (default)
#   EVAL_TRANSPORT=ssh   — fixed GPU box via EVAL_SSH_HOST / EVAL_SSH_PORT
export HOME="${HOME:-/home/autotiny}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export SPARKINFER_AUTOMERGE="${SPARKINFER_AUTOMERGE:-1}"
export SPARKINFER_AUTOMERGE_ADMIN="${SPARKINFER_AUTOMERGE_ADMIN:-1}"
export VAST_NO_AUTO_PROVISION=1

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
# Re-assert after .env.eval in case a stale file omitted them.
export SPARKINFER_AUTOMERGE="${SPARKINFER_AUTOMERGE:-1}"
export SPARKINFER_AUTOMERGE_ADMIN="${SPARKINFER_AUTOMERGE_ADMIN:-1}"
export VAST_NO_AUTO_PROVISION=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

git pull -q origin main 2>/dev/null || true

BOT_ARGS=(
  --frontier "${FRONTIER:-285}"
  --ceiling  "${CEILING:-366}"
  --repo     "${REPO:-gittensor-ai-lab/sparkinfer}"
)
if [ "${EVAL_TRANSPORT:-vast}" != "ssh" ]; then
  BOT_ARGS+=(--instance "${VAST_INSTANCE:-${VAST_DEFAULT_INSTANCE:-0}}")
fi
[ "${BIDIR:-${TRIPLE:-1}}" != "0" ] && BOT_ARGS+=(--bidir --primary-quant "${PRIMARY_QUANT:-Q4_K_M}")
if [ "${POLARIS:-1}" = "0" ]; then
  BOT_ARGS+=(--no-polaris)
else
  BOT_ARGS+=(--polaris)
fi

# Returns 0 iff the eval GPU is usable right now (no rent, no start).
gpu_ready() {
  local key="${SSH_KEY:-$HOME/.ssh/speedy}"
  if [ "${EVAL_TRANSPORT:-vast}" = "ssh" ]; then
    local host="${EVAL_SSH_HOST:-}" port="${EVAL_SSH_PORT:-22}"
    [ -n "$host" ] || return 1
    ssh -i "$key" -o BatchMode=yes -o ConnectTimeout=10 \
        -o StrictHostKeyChecking=accept-new -p "$port" "root@$host" 'true' 2>/dev/null
    return $?
  fi
  local iid="${VAST_INSTANCE:-${VAST_DEFAULT_INSTANCE:-}}"
  [ -n "$iid" ] && [ "$iid" != "0" ] || return 1
  command -v vastai >/dev/null 2>&1 || return 1
  local raw st ip port
  raw="$(vastai show instance "$iid" --raw 2>/dev/null)" || return 1
  read -r st ip port < <(python3 -c "
import json, sys
d = json.loads(sys.stdin.read() or '{}')
st = d.get('actual_status') or ''
ip = d.get('public_ipaddr') or ''
ports = d.get('ports') or {}
p = ((ports.get('22/tcp') or [{}])[0] or {}).get('HostPort') or ''
print(st, ip, p)
" <<<"$raw")
  [ "$st" = "running" ] && [ -n "$ip" ] && [ -n "$port" ] || return 1
  ssh -i "$key" -o BatchMode=yes -o ConnectTimeout=10 \
      -o StrictHostKeyChecking=accept-new -p "$port" "root@$ip" 'true' 2>/dev/null
}

TS="$(date -u +%FT%TZ)"
if gpu_ready; then
  echo "[$TS] sparkinfer PR bot — GPU up — full eval (AUTOMERGE=${SPARKINFER_AUTOMERGE}, no-rent)"
  python3 eval/pr_eval_bot.py "${BOT_ARGS[@]}"
else
  echo "[$TS] sparkinfer PR bot — GPU down/unreachable — labels only (no rent, no start)"
  python3 eval/pr_eval_bot.py "${BOT_ARGS[@]}" --labels-only
fi
