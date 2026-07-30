#!/usr/bin/env bash
# Cron wrapper for the sparkinfer DFlash PR eval bot — offset from AR bot:
#
#   0 1-23/2 * * * /home/autotiny/Desktop/sparkinfer/eval/run_dflash_cron.sh >> /tmp/sparkinfer_dflash_bot.log 2>&1
#   (odd hours :00 — 01:00, 03:00, …; AR bot stays on even hours)
#
# Policy (same as AR bot):
#   • Pinned vast.ai GPU only; never rent / never start from cron when down.
#   • Shares /tmp/sparkinfer_bot.lock with run_bot_cron.sh (no overlap).
#   • GPU up → full dflash eval + auto-merge; GPU down → --labels-only.
export HOME="${HOME:-/home/autotiny}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export SPARKINFER_AUTOMERGE="${SPARKINFER_AUTOMERGE:-1}"
export SPARKINFER_AUTOMERGE_ADMIN="${SPARKINFER_AUTOMERGE_ADMIN:-1}"
export VAST_NO_AUTO_PROVISION=1

exec 9>/tmp/sparkinfer_bot.lock
flock -n 9 || { echo "[$(date -u +%FT%TZ)] previous bot run still active — skipping dflash tick"; exit 0; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1

if [ -f "$REPO_DIR/.env.eval" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_DIR/.env.eval"
  set +a
fi
export SPARKINFER_AUTOMERGE="${SPARKINFER_AUTOMERGE:-1}"
export SPARKINFER_AUTOMERGE_ADMIN="${SPARKINFER_AUTOMERGE_ADMIN:-1}"
export VAST_NO_AUTO_PROVISION=1
# 512 gen crosses the KV-split knee; 128-gen DFlash PRs scored eval-dflash:none.
export DFLASH_BENCH_TOKENS="${DFLASH_BENCH_TOKENS:-512}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

git pull -q origin main 2>/dev/null || true

PIN_FILE="${VAST_PIN_FILE:-$HOME/.sparkinfer_pinned_instance}"
INSTANCE_FILE="${VAST_INSTANCE_FILE:-$HOME/.sparkinfer_vast_instance}"
resolve_pin() {
  local v=""
  if [ -f "$PIN_FILE" ]; then
    v="$(tr -d '[:space:]' <"$PIN_FILE" 2>/dev/null || true)"
  fi
  if [ -z "$v" ] || [ "$v" = "0" ]; then
    v="${VAST_DEFAULT_INSTANCE:-${VAST_INSTANCE:-}}"
  fi
  printf '%s' "$v"
}
PINNED_ID="$(resolve_pin)"
if [ "${EVAL_TRANSPORT:-vast}" != "ssh" ] && [ -n "$PINNED_ID" ] && [ "$PINNED_ID" != "0" ]; then
  export VAST_INSTANCE="$PINNED_ID"
  export VAST_DEFAULT_INSTANCE="$PINNED_ID"
  printf '%s\n' "$PINNED_ID" >"$PIN_FILE"
  printf '%s\n' "$PINNED_ID" >"$INSTANCE_FILE"
fi

BOT_ARGS=(--repo "${REPO:-gittensor-ai-lab/sparkinfer}")
if [ "${EVAL_TRANSPORT:-vast}" != "ssh" ]; then
  BOT_ARGS+=(--instance "${VAST_INSTANCE:-0}")
fi

gpu_ready() {
  local key="${SSH_KEY:-$HOME/.ssh/speedy}"
  if [ "${EVAL_TRANSPORT:-vast}" = "ssh" ]; then
    local host="${EVAL_SSH_HOST:-}" port="${EVAL_SSH_PORT:-22}"
    [ -n "$host" ] || return 1
    ssh -i "$key" -o BatchMode=yes -o ConnectTimeout=10 \
        -o StrictHostKeyChecking=accept-new -p "$port" "root@$host" 'true' 2>/dev/null
    return $?
  fi
  local iid="${VAST_INSTANCE:-}"
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
  echo "[$TS] sparkinfer DFlash bot — pinned GPU ${VAST_INSTANCE:-ssh} up — full eval (AUTOMERGE=${SPARKINFER_AUTOMERGE})"
  python3 eval/pr_dflash_bot.py "${BOT_ARGS[@]}"
else
  echo "[$TS] sparkinfer DFlash bot — pinned GPU ${VAST_INSTANCE:-?} down — labels only"
  python3 eval/pr_dflash_bot.py "${BOT_ARGS[@]}" --labels-only
fi
