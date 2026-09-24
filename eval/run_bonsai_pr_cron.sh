#!/usr/bin/env bash
# Cron wrapper for the sparkinfer Ternary-Bonsai-2-27B PR eval bot (eval/pr_bonsai_bot.py):
#
#   15 */2 * * * GH_TOKEN="$(gh auth token -u <bot account>)" /path/to/sparkinfer/eval/run_bonsai_pr_cron.sh >> /tmp/sparkinfer_bonsai_bot.log 2>&1
#
#   Every two hours at :15 (decision 2026-09-24), between the Muse Glimmer bot's :00 and the Qwen3.8
#   bot's :30. A round with a pending PR holds the one GPU for well over half an hour, so an hourly
#   third bot would keep finding the lock busy. A tick with nothing to evaluate never touches the GPU.
#
# Policy (same as the sibling wrappers):
#   • Pinned eval box only; never rent / never start from cron when down.
#   • Shares /tmp/sparkinfer_bot.lock with every other bot -- they all drive the ONE pinned GPU.
#   • GPU up → full eval; GPU down → --labels-only.
#   • Auto-merge and auto-close stay OFF unless SPARKINFER_BONSAI_AUTOMERGE=1 /
#     SPARKINFER_BONSAI_AUTOCLOSE=1 are set explicitly. Neither is exported here.
export HOME="${HOME:-/home/autotiny}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export VAST_NO_AUTO_PROVISION=1

exec 9>/tmp/sparkinfer_bot.lock
flock -w 120 9 || { echo "[$(date -u +%FT%TZ)] lock held 120s+ — previous bot run still active, skipping bonsai tick"; exit 0; }

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
    # IdentitiesOnly=yes: same rationale as run_dflash_cron.sh's gpu_ready — cron has no
    # ssh-agent, so without it a box whose authorized_keys only has SSH_KEY's public half (not
    # some agent identity) fails outright with "Permission denied".
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

# Outage escalation, ported verbatim from run_museglimmer_cron.sh (itself from run_dspark_cron.sh).
# A dead box does NOT stop the tick — it degrades it to --labels-only, which
# logs one ordinary-looking line and exits 0. On 2026-09-04/05 that ran 14 consecutive times and
# nothing got louder, so the outage was only found by someone asking why there were no new
# numbers; on 2026-09-08 the counter below is what made a one-tick outage visible immediately.
# Silent degradation is the right RUNTIME behaviour — never rent, never fail a tick — but it must
# not be silent to a reader.
DOWN_FILE="${BONSAI_DOWN_FILE:-$HOME/.sparkinfer_bonsai_gpu_down}"
note_gpu_up() { rm -f "$DOWN_FILE"; }
note_gpu_down() {
  local n first
  n=0; first="$TS"
  if [ -f "$DOWN_FILE" ]; then
    n="$(sed -n 1p "$DOWN_FILE" 2>/dev/null | tr -dc '0-9')"; n="${n:-0}"
    first="$(sed -n 2p "$DOWN_FILE" 2>/dev/null)"; first="${first:-$TS}"
  fi
  n=$((n + 1))
  printf '%s\n%s\n' "$n" "$first" >"$DOWN_FILE"
  # Loud at the 3rd consecutive miss, then once every 6 after that, so a long outage keeps
  # reappearing in the log instead of scrolling away behind identical one-liners.
  # Banner to STDERR, count to STDOUT. The caller reads the count through a $(...) substitution,
  # which would otherwise swallow the banner entirely -- the exact bug this block exists to prevent.
  # Cron's `>> log 2>&1` puts stderr in the same log, so the banner still lands next to the tick.
  if [ "$n" -ge 3 ] && { [ "$n" -eq 3 ] || [ $((n % 6)) -eq 0 ]; }; then
    {
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      echo "!! BONSAI EVAL DEGRADED: $n consecutive ticks with NO GPU."
      echo "!! Pinned box $GPU_LABEL unreachable since $first."
      echo "!! Nothing has been measured, scored or auto-merged since then."
      echo "!! Fix: point EVAL_SSH_HOST/EVAL_SSH_PORT in .env.eval at a live RTX 5090."
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    } >&2
  fi
  echo "$n"
}

if gpu_ready; then
  note_gpu_up
  echo "[$TS] sparkinfer Ternary-Bonsai bot — pinned GPU $GPU_LABEL up — full eval (AUTOMERGE=${SPARKINFER_BONSAI_AUTOMERGE:-0} AUTOCLOSE=${SPARKINFER_BONSAI_AUTOCLOSE:-0})"
  python3 eval/pr_bonsai_bot.py "${BOT_ARGS[@]}"
else
  DOWN_N="$(note_gpu_down | tail -1)"
  echo "[$TS] sparkinfer Ternary-Bonsai bot — pinned GPU $GPU_LABEL down (tick $DOWN_N) — labels only"
  python3 eval/pr_bonsai_bot.py "${BOT_ARGS[@]}" --labels-only
fi
