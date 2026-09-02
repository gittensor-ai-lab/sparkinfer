#!/usr/bin/env bash
# Cron wrapper for the sparkinfer DSpark speculative-decode PR eval bot:
#
#   0 * * * * /home/autotiny/Desktop/sparkinfer/eval/run_dspark_cron.sh >> /tmp/sparkinfer_dspark_bot.log 2>&1
#
#   Hourly, taking over the :00 slot run_modelopt_cron.sh held. DSpark decode replaces ModelOpt
#   long-context as the scored dimension (2026-08-18, explicit user decision), so the two are NOT
#   meant to run together — both drive the one pinned GPU, and both want the same lock.
#
#   The scored dimensions are DSpark decode + prefill at 4k/16k/32k and target prefill at 256k
#   (SCORING_DIMS in the bot). The 256k baseline currently uses sequential prefill and takes about
#   65 minutes per ref, so a main+PR round intentionally spans multiple hourly ticks.
#
#   The Qwen3.6 and Qwen3.8 no-regression guards ARE run — check_q36_guard / check_q38_guard — and
#   a round bails if either produces no MAIN baseline. The 256k row dominates wall time until a
#   memory-safe chunked batched-prefill implementation replaces the sequential fallback.
#
#   If a round ever DOES overrun the interval (a large PR backlog), the flock below makes the
#   overlapping tick exit 0 after 120s rather than piling up — the effect is a skipped tick, not a
#   queue. Do NOT raise flock's -w to try to queue ticks: that would serialize a backlog of stale
#   rounds against one GPU.
#
# Policy (same as the sibling bots):
#   • Pinned eval box only; never rent / never start from cron when down.
#   • Shares /tmp/sparkinfer_bot.lock with every other bot and the sparkinfer-web dashboard-sync
#     crons — CRITICAL: this MUST be the SAME lock file, since all of them drive the ONE pinned GPU
#     on the SAME box and would otherwise race for it if a cron tick overlaps.
#   • GPU up → full DSpark 4k/16k/32k eval + 256k prefill + accuracy/losslessness gates;
#     GPU down → --labels-only (no GPU, no ssh, pure label reconciliation).
#   • Auto-merge stays OFF unless SPARKINFER_DSPARK_AUTOMERGE=1 is explicitly set. Deliberately NOT
#     forced to 1 here and deliberately NOT inherited from SPARKINFER_MODELOPT_AUTOMERGE: this bot
#     scores a different dimension through a code path (speculative decode) that has been lossless
#     for a matter of days, so the first rounds should be read by a human before anything merges
#     itself. Set it in .env.eval when you are satisfied the verdicts are right.
export HOME="${HOME:-/home/autotiny}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export VAST_NO_AUTO_PROVISION=1

exec 9>/tmp/sparkinfer_bot.lock
flock -w 120 9 || { echo "[$(date -u +%FT%TZ)] lock held 120s+ — previous bot run still active, skipping dspark tick"; exit 0; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1

if [ -f "$REPO_DIR/.env.eval" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_DIR/.env.eval"
  set +a
fi
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
    # IdentitiesOnly=yes: cron has no ssh-agent, so without it a box whose authorized_keys only has
    # SSH_KEY's public half (not some agent identity) fails outright with "Permission denied".
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

GPU_LABEL="${EVAL_SSH_HOST:-ssh}"
[ "${EVAL_TRANSPORT:-vast}" = "ssh" ] || GPU_LABEL="${VAST_INSTANCE:-?}"

TS="$(date -u +%FT%TZ)"
if gpu_ready; then
  echo "[$TS] sparkinfer DSpark bot — pinned GPU $GPU_LABEL up — full eval (AUTOMERGE=${SPARKINFER_DSPARK_AUTOMERGE:-0})"
  python3 eval/pr_dspark_bot.py "${BOT_ARGS[@]}"
else
  echo "[$TS] sparkinfer DSpark bot — pinned GPU $GPU_LABEL down — labels only"
  python3 eval/pr_dspark_bot.py "${BOT_ARGS[@]}" --labels-only
fi
