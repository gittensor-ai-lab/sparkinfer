#!/usr/bin/env bash
# Cron wrapper for the sparkinfer DFlash PR eval bot:
#
#   0 * * * * /home/autotiny/Desktop/sparkinfer/eval/run_dflash_cron.sh >> /tmp/sparkinfer_dflash_bot.log 2>&1
#   (every hour, on the hour)
#
# Policy (same as AR bot):
#   • Pinned vast.ai GPU only; never rent / never start from cron when down.
#   • Shares /tmp/sparkinfer_bot.lock with run_bot_cron.sh and the sparkinfer-web dashboard-sync
#     crons (no overlap). The dashboard sync runs every 15 min (*/15 * * * *), which lands on
#     this cron's own :00 slot every single tick — a bare `flock -n` would race that fast,
#     low-stakes sync job and could lose an entire eval window if it happened to grab the lock
#     first. Wait a bounded amount instead: long enough to outlast a quick dashboard git-sync,
#     short enough to still bail if something is genuinely stuck.
#   • GPU up → full dflash eval + auto-merge; GPU down → --labels-only.
export HOME="${HOME:-/home/autotiny}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export SPARKINFER_AUTOMERGE="${SPARKINFER_AUTOMERGE:-1}"
export SPARKINFER_AUTOMERGE_ADMIN="${SPARKINFER_AUTOMERGE_ADMIN:-1}"
export VAST_NO_AUTO_PROVISION=1

exec 9>/tmp/sparkinfer_bot.lock
flock -w 120 9 || { echo "[$(date -u +%FT%TZ)] lock held 120s+ — previous bot run still active, skipping dflash tick"; exit 0; }

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
    local host="${EVAL_SSH_HOST:-}" port="${EVAL_SSH_PORT:-22}" user="${EVAL_SSH_USER:-root}"
    [ -n "$host" ] || return 1
    # IdentitiesOnly=yes: without it ssh also offers every identity in a running ssh-agent
    # before the explicit -i key -- fine interactively (an agent can paper over a box where the
    # -i key was never actually added to authorized_keys) but cron has no agent, so that gap
    # surfaces as a flat "Permission denied" with zero indication why. This is what actually
    # caused six straight silent "down — labels only" ticks on 2026-08-09: the box's
    # authorized_keys never had SSH_KEY's public half, only the interactive session's agent key
    # -- fixed by adding SSH_KEY's public key to the box directly, and pinned here so it can't
    # silently regress (an agent that happens to have a WORKING key can also eat into
    # MaxAuthTries before the right key is tried, on top of the missing-key case).
    # ConnectTimeout=20 (was 10): cheap extra headroom, kept even though it wasn't the root cause.
    local err rc
    err="$(ssh -i "$key" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 \
        -o StrictHostKeyChecking=accept-new -p "$port" "$user@$host" 'true' 2>&1)"
    rc=$?
    [ "$rc" -eq 0 ] || echo "gpu_ready: ssh to $user@$host:$port failed (exit=$rc): $err" >&2
    return "$rc"
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

# Cosmetic-only: VAST_INSTANCE is unconditionally set from .env.eval's legacy vast.ai section
# even in EVAL_TRANSPORT=ssh mode (never cleared), so it was showing "pinned GPU 46396637" in
# every ssh-mode log line regardless of the real target — misleading when debugging. Report the
# actual target being checked instead.
GPU_LABEL="${EVAL_SSH_HOST:-ssh}"
[ "${EVAL_TRANSPORT:-vast}" = "ssh" ] || GPU_LABEL="${VAST_INSTANCE:-?}"

TS="$(date -u +%FT%TZ)"
if gpu_ready; then
  echo "[$TS] sparkinfer DFlash bot — pinned GPU $GPU_LABEL up — full eval (AUTOMERGE=${SPARKINFER_AUTOMERGE})"
  python3 eval/pr_dflash_bot.py "${BOT_ARGS[@]}"
else
  echo "[$TS] sparkinfer DFlash bot — pinned GPU $GPU_LABEL down — labels only"
  python3 eval/pr_dflash_bot.py "${BOT_ARGS[@]}" --labels-only
fi
