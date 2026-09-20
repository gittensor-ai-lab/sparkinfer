#!/usr/bin/env bash
# Cron wrapper for the Ternary-Bonsai-2-27B regression guard (eval/bonsai_regression.py):
#
#   45 4 * * * /home/autotiny/Desktop/sparkinfer/eval/run_bonsai_cron.sh >> /tmp/sparkinfer_bonsai_guard.log 2>&1
#
#   DAILY, not hourly, and not on :00 or :30. This is a correctness guard on one checkpoint, not a
#   PR gate: nothing here votes on a pull request, so it buys nothing by running twelve times
#   between two commits, and the two PR bots that DO gate (Muse Glimmer at :00, Qwen3.8 at :30)
#   should not have to queue behind it on the one pinned GPU.
#
#   It watches `feat/ternary-bonsai-2-27b` rather than main because that is where the support
#   lives. Point BONSAI_REF at main once it merges.
#
# What a tick costs: an incremental build plus ~3m40s of GPU. The `serve` check starts a server
# per path, which is most of that and is the point -- the worst defect this model turned up was
# invisible to every single-request check.
#
# Policy (same as the sibling bots):
#   • Pinned eval box only; never rent, never provision, never start from cron when down.
#   • Shares /tmp/sparkinfer_bot.lock with every other bot, BOTH sides: the wrapper holds the local
#     lock and the remote script holds the box's own, since the bots drive the ONE pinned GPU.
#   • GPU down → report and exit 0. A guard that cannot reach the GPU has found nothing, and
#     failing the tick would be indistinguishable from the model regressing.
#   • Restores the box checkout to its parked commit afterwards, pass or fail, so the next bot
#     round does not find the tree on a feature branch.
export HOME="${HOME:-/home/autotiny}"
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export VAST_NO_AUTO_PROVISION=1

exec 9>/tmp/sparkinfer_bot.lock
flock -w 120 9 || { echo "[$(date -u +%FT%TZ)] lock held 120s+ — another bot run is active, skipping bonsai tick"; exit 0; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR" || exit 1
if [ -f "$REPO_DIR/.env.eval" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_DIR/.env.eval"
  set +a
fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

BONSAI_REF="${BONSAI_REF:-feat/ternary-bonsai-2-27b}"
BONSAI_PARK="${BONSAI_PARK:-a8f53e6}"      # what the box tree is left on between jobs
BOX_REPO="${BONSAI_BOX_REPO:-/workspace/sparkinfer}"
KEY="${SSH_KEY:-$HOME/.ssh/speedy}"
TS="$(date -u +%FT%TZ)"

# IdentitiesOnly=yes: cron has no ssh-agent, so without it a box whose authorized_keys carries
# only this key's public half fails outright. Same rationale as the sibling wrappers.
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new
          -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=40
          -p "${EVAL_SSH_PORT:-22}")
BOX="${EVAL_SSH_USER:-root}@${EVAL_SSH_HOST:-}"

if [ -z "${EVAL_SSH_HOST:-}" ] || ! ssh "${SSH_OPTS[@]}" "$BOX" 'true' 2>/dev/null; then
  echo "[$TS] bonsai guard — pinned GPU ${EVAL_SSH_HOST:-?} unreachable — nothing checked"
  exit 0
fi

echo "[$TS] bonsai guard — pinned GPU ${EVAL_SSH_HOST} up — ref $BONSAI_REF"
ssh "${SSH_OPTS[@]}" "$BOX" BONSAI_REF="$BONSAI_REF" BONSAI_PARK="$BONSAI_PARK" BOX_REPO="$BOX_REPO" 'bash -s' <<'REMOTE'
set -u
exec 9>/tmp/sparkinfer_bot.lock
flock -w 600 9 || { echo "  box lock busy — skipping"; exit 0; }
# Identify servers by /proc/<pid>/exe, never pgrep -f: a pattern matches this script too.
sweep() {
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    [ "$p" = "$$" ] && continue
    e=$(readlink "/proc/$p/exe" 2>/dev/null) || continue
    case "$e" in *sparkinfer_server*) echo "  reaping stray server pid $p"; kill -9 "$p" 2>/dev/null ;; esac
  done
}
sweep
cd "$BOX_REPO" || exit 1
git fetch -q origin "$BONSAI_REF" && git checkout -q FETCH_HEAD || { echo "  checkout failed"; exit 1; }
echo "  at $(git log --oneline -1)"
cd build && cmake . >/dev/null 2>&1
nice -n 5 make -j"$(nproc)" sparkinfer_server bonsai_inspect qwen3_gguf_score qwen3_gguf_generate 2>&1 \
  | grep -E ' error|Error [0-9]' | head -5
export LD_LIBRARY_PATH="$BOX_REPO/build/runtime:$BOX_REPO/build/kernels:$BOX_REPO/build/moe:${LD_LIBRARY_PATH:-}"
cd "$BOX_REPO"
python3 eval/bonsai_regression.py
rc=$?
# Park the tree again whatever happened, so the next bot round does not build a feature branch.
git checkout -q "$BONSAI_PARK" 2>/dev/null
sweep
exit $rc
REMOTE
rc=$?
if [ "$rc" -eq 0 ]; then
  echo "[$(date -u +%FT%TZ)] bonsai guard — OK"
else
  echo "[$(date -u +%FT%TZ)] bonsai guard — FAILED (exit $rc)"
fi
exit "$rc"
