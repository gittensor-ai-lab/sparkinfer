#!/usr/bin/env bash
# Cron wrapper for the sparkinfer Ternary-Bonsai-2-27B PR eval bot (eval/pr_bonsai_bot.py):
#
#   15 * * * * GH_TOKEN="$(gh auth token -u <bot account>)" /path/to/sparkinfer/eval/run_bonsai_pr_cron.sh >> /tmp/sparkinfer_bonsai_bot.log 2>&1
#
#   Without GH_TOKEN every tick is refused. The bot runs origin/main from its own worktree; this
#   script is /path/to/sparkinfer's copy, so keep that checkout on main (cron_common.sh).
#
#   Hourly at :15 (2026-09-24; started every two hours, moved to hourly the same day), between the
#   Muse Glimmer bot's :00 and the Qwen3.8 bot's :30. A round holds the one GPU for ~40 min per
#   pending PR, so when it overlaps a sibling round this tick defers on the shared lock and retries
#   next hour -- that is expected, not an error (cron_common.sh makes a long run of skips loud). A
#   tick with nothing to evaluate never touches the GPU.
#
# Policy (same as the sibling wrappers):
#   • Pinned eval box only; never rent / never start from cron when down.
#   • Shares /tmp/sparkinfer_bot.lock with every other bot -- they all drive the ONE pinned GPU.
#   • Runs origin/main from its own worktree, and only with a GH_TOKEN (cron_common.sh).
#   • GPU up → full eval; GPU down → --labels-only.
#   • Auto-merge follows SPARKINFER_BONSAI_AUTOMERGE (=1 in .env.eval). Auto-close is ON unless
#     SPARKINFER_BONSAI_AUTOCLOSE=0: a measured REJECT closes, and `none` closes only a PR declared
#     for Ternary-Bonsai alone (pr_bonsai_bot.py). This script sets neither.
export HOME="${HOME:-/home/autotiny}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export VAST_NO_AUTO_PROVISION=1

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=eval/cron_common.sh
source "$REPO_DIR/eval/cron_common.sh" || exit 1

open_lock bonsai || exit 1
flock -w "$LOCK_WAIT_S" 9 || { note_lock_skip bonsai; exit 0; }
note_lock_taken bonsai
cd "$REPO_DIR" || { note_refused bonsai "cannot enter $REPO_DIR"; exit 1; }

keep_cron_token
if [ -f "$REPO_DIR/.env.eval" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_DIR/.env.eval"
  set +a
fi
restore_cron_token
export VAST_NO_AUTO_PROVISION=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

require_bot_token bonsai || exit 1
# The bot runs exactly origin/main, from its own worktree -- not whatever REPO_DIR has checked out.
prepare_bot_tree || { note_refused bonsai "$TREE_WHY"; exit 1; }
cd "$BOT_TREE" || { note_refused bonsai "cannot enter $BOT_TREE"; exit 1; }
note_ran bonsai

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
    # IdentitiesOnly=yes: same rationale as run_dflash_cron.sh's gpu_ready — cron has no
    # ssh-agent, so without it a box whose authorized_keys only has SSH_KEY's public half (not
    # some agent identity) fails outright with "Permission denied".
    local err rc
    err="$(timeout 60 ssh -i "$key" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 \
        -o StrictHostKeyChecking=accept-new -p "$port" "$user@$host" 'true' 2>&1 9>&-)"
    rc=$?
    [ "$rc" -eq 0 ] || echo "gpu_ready: ssh to $user@$host:$port failed (exit=$rc): $err" >&2
    return "$rc"
  fi
  local iid="${VAST_INSTANCE:-}"
  [ -n "$iid" ] && [ "$iid" != "0" ] || return 1
  command -v vastai >/dev/null 2>&1 || return 1
  local raw st ip port
  raw="$(timeout 60 vastai show instance "$iid" --raw 2>/dev/null 9>&-)" || return 1
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
  timeout 60 ssh -i "$key" -o BatchMode=yes -o ConnectTimeout=10 \
      -o StrictHostKeyChecking=accept-new -p "$port" "root@$ip" 'true' 2>/dev/null 9>&-
}

GPU_LABEL="${EVAL_SSH_HOST:-ssh}"
[ "${EVAL_TRANSPORT:-vast}" = "ssh" ] || GPU_LABEL="${VAST_INSTANCE:-?}"

TS="$(date -u +%FT%TZ)"

# A dead box degrades the tick to --labels-only; cron_common.sh's note_gpu_down makes the outage loud.
DOWN_FILE="${BONSAI_DOWN_FILE:-$HOME/.sparkinfer_bonsai_gpu_down}"

if gpu_ready; then
  note_gpu_up "$DOWN_FILE"
  echo "[$TS] sparkinfer Ternary-Bonsai bot — pinned GPU $GPU_LABEL up — full eval (AUTOMERGE=${SPARKINFER_BONSAI_AUTOMERGE:-0} AUTOCLOSE=${SPARKINFER_BONSAI_AUTOCLOSE:-1})"
  run_bot bonsai python3 eval/pr_bonsai_bot.py "${BOT_ARGS[@]}"
else
  DOWN_N="$(note_gpu_down "$DOWN_FILE" BONSAI | tail -1)"
  echo "[$TS] sparkinfer Ternary-Bonsai bot — pinned GPU $GPU_LABEL down (tick $DOWN_N) — labels only"
  run_bot bonsai python3 eval/pr_bonsai_bot.py "${BOT_ARGS[@]}" --labels-only
fi
