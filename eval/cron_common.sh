# Shared by the eval cron wrappers (run_bonsai_pr_cron.sh, run_museglimmer_cron.sh,
# run_qwen38_cron.sh, run_sync_cron.sh). Sourced, never run; the wrapper sets REPO_DIR first (the
# working copy it lives in, which holds .env.eval and the git objects).
#
# LOCK_FILE, LOCK_WAIT_S, BOT_TIMEOUT_S and STEP_TIMEOUT_S below come from the environment cron gives
# the wrapper, not from .env.eval (they are read first). SPARKINFER_LOCK_FILE exists for the tests:
# every bot, and the wrappers not converted to this file, lock /tmp/sparkinfer_bot.lock.

# The lock every bot and the dashboard sync share: one GPU, one round at a time.
LOCK_FILE="${SPARKINFER_LOCK_FILE:-/tmp/sparkinfer_bot.lock}"

# How long a model bot's tick waits for that lock. The dashboard sync fires on the same minutes as
# every bot tick (*/15); at 120 s a slow sync cost a bot its whole hour. Still far short of queueing
# a missed tick into the next one.
LOCK_WAIT_S="${SPARKINFER_LOCK_WAIT_S:-300}"

# Longest a bot run may hold the lock. Every GPU step has its own timeout, so this only catches a
# run that hangs anyway -- which would otherwise hold the lock, and starve every bot, forever. A
# long backlog round (Bonsai: ~40 min per PR) fits inside; a killed round loses nothing that the
# next one does not pick back up, and the next round reaps what it left on the box.
BOT_TIMEOUT_S="${SPARKINFER_BOT_TIMEOUT_S:-28800}"

# Each git or ssh step a wrapper runs before the bot, all while holding the lock.
STEP_TIMEOUT_S="${SPARKINFER_STEP_TIMEOUT_S:-120}"

# Loud at the 3rd consecutive occurrence, then once every 6 after that, so a long run of them keeps
# reappearing in the log instead of scrolling away behind identical one-liners. $1 = count.
_loud_now() { [ "$1" -ge 3 ] && { [ "$1" -eq 3 ] || [ $(($1 % 6)) -eq 0 ]; }; }

_bump() {  # $1 = counter file; prints the new consecutive count
  local n
  n="$(sed -n 1p "$1" 2>/dev/null | tr -dc '0-9')"
  n=$(( ${n:-0} + 1 ))
  printf '%s\n' "$n" >"$1"
  echo "$n"
}

# Consecutive ticks a bot skipped on the lock. A skip is ordinary when rounds overlap, but a run
# that never lets go looks exactly like that in the log, one quiet line an hour.
note_lock_skip() {
  local bot="$1" n
  n="$(_bump "$HOME/.sparkinfer_${bot}_lock_skips")"
  echo "[$(date -u +%FT%TZ)] lock held ${LOCK_WAIT_S}s+ — another bot's round is still running, skipping $bot tick ($n in a row)"
  if _loud_now "$n"; then
    {
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      echo "!! ${bot} EVAL STARVED: $n consecutive ticks skipped on $LOCK_FILE."
      echo "!! Holding it now:"
      pgrep -af 'eval/pr_[a-z0-9]*_bot\.py|pr_eval_bot' 2>/dev/null | sed 's/^/!!   /' || true
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    } >&2
  fi
}

# Once the lock is ours. The bot inherits it: arb.hold_bot_lock must not wait for it again.
note_lock_taken() {
  rm -f "$HOME/.sparkinfer_${1}_lock_skips"
  export SPARKINFER_BOT_LOCK_HELD=1
}

# Consecutive ticks refused before the bot could run (no bot account, no origin/main tree). Every
# bot stops outright then -- not even labels-only -- so a run of them must not stay quiet.
note_refused() {  # $1 = bot, $2 = why
  local n
  n="$(_bump "$HOME/.sparkinfer_${1}_refused")"
  echo "[$(date -u +%FT%TZ)] $2 — skipping $1 tick ($n in a row)" >&2
  if _loud_now "$n"; then
    {
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      echo "!! ${1} EVAL NOT RUNNING: $n consecutive ticks refused before the bot started."
      echo "!! Last reason: $2"
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    } >&2
  fi
}
note_ran() { rm -f "$HOME/.sparkinfer_${1}_refused"; }

# The bots post, label, close and merge as whichever account `gh` resolves. The crontab passes the
# bot account's token as GH_TOKEN; when that lookup fails GH_TOKEN is empty, and gh falls back to
# the machine's active account -- someone else's -- for every one of those actions. With
# SPARKINFER_BOT_LOGIN set (.env.eval), the token must also belong to that account -- when GitHub
# answers the lookup: if it does not, the tick goes ahead (every gh call would fail anyway).
require_bot_token() {
  local login
  if [ -z "${GH_TOKEN:-}" ]; then
    note_refused "$1" "GH_TOKEN is empty — refusing to act as this machine's active gh account"
    return 1
  fi
  [ -n "${SPARKINFER_BOT_LOGIN:-}" ] || return 0
  # A failed lookup prints GitHub's error body; only a successful one is a login.
  login="$(timeout 30 "${SPARKINFER_GH:-gh}" api user --jq .login 2>/dev/null)" || login=""
  if [ -n "$login" ] && [ "$login" != "$SPARKINFER_BOT_LOGIN" ]; then
    note_refused "$1" "GH_TOKEN belongs to $login, not the bot account $SPARKINFER_BOT_LOGIN"
    return 1
  fi
  echo "acting as GitHub account: ${login:-unknown (GitHub did not answer)}"
}

# The bots run from BOT_TREE, a detached worktree reset to origin/main every tick -- never from
# REPO_DIR's checkout. REPO_DIR is a working copy: a branch left checked out there, a local edit or a
# `git pull` that failed quietly ran that code against every open PR (2026-09-26: a feature branch
# checked out there ran a Qwen3.8 round). The wrapper script itself is still REPO_DIR's copy (cron
# starts it there), so keep REPO_DIR on main and do development in another worktree.
# Sets BOT_TREE. Returns 1 (skip the tick) when no origin/main tree can be had: running unreviewed
# code is worse than a skipped hour.
#
# The tree is the wrappers' own, marked so in its git admin dir when they make it. Only a tree
# carrying that mark is ever reset or removed; the leftover of one, a worktree of this repository
# whose admin dir is gone, is moved aside. SPARKINFER_BOT_TREE pointed at anyone else's checkout is
# refused.
_TREE_MARK=sparkinfer-bot-tree
_common_dir() { git -C "$1" rev-parse --path-format=absolute --git-common-dir 2>/dev/null; }
_tree_state() {  # ours | orphan | absent | foreign
  local gd ptr
  if [ ! -e "$BOT_TREE" ] || { [ -d "$BOT_TREE" ] && [ -z "$(ls -A "$BOT_TREE" 2>/dev/null)" ]; }; then
    echo absent; return
  fi
  if [ ! -f "$BOT_TREE/.git" ]; then echo foreign; return; fi
  gd="$(git -C "$BOT_TREE" rev-parse --absolute-git-dir 2>/dev/null)"
  if [ -n "$gd" ]; then
    if [ -f "$gd/$_TREE_MARK" ] && [ "$(_common_dir "$BOT_TREE")" = "$(_common_dir "$REPO_DIR")" ]; then
      echo ours
    else
      echo foreign
    fi
    return
  fi
  # Its admin dir is gone (REPO_DIR re-cloned, say): ours only if it pointed into this repository.
  ptr="$(sed -n 's/^gitdir: //p' "$BOT_TREE/.git" 2>/dev/null)"
  case "$ptr" in
    "$(_common_dir "$REPO_DIR")"/worktrees/*) echo orphan ;;
    *) echo foreign ;;
  esac
}
_reset_tree() {
  timeout "$STEP_TIMEOUT_S" git -C "$BOT_TREE" checkout -q -f --detach origin/main \
    && timeout "$STEP_TIMEOUT_S" git -C "$BOT_TREE" clean -qfd
}
_make_tree() {
  # -f: a tree whose directory was deleted is still registered, and would otherwise refuse the path.
  # Marked before anything is checked out, so a tree these wrappers made always carries the mark.
  timeout "$STEP_TIMEOUT_S" git -C "$REPO_DIR" worktree add -q -f --no-checkout --detach "$BOT_TREE" origin/main >&2 \
    && : >"$(git -C "$BOT_TREE" rev-parse --absolute-git-dir)/$_TREE_MARK"
}
_drop_tree() {
  timeout "$STEP_TIMEOUT_S" git -C "$REPO_DIR" worktree remove -f -f "$BOT_TREE" 2>/dev/null \
    || rm -rf -- "$BOT_TREE"
}
prepare_bot_tree() {
  local state
  BOT_TREE="${SPARKINFER_BOT_TREE:-$HOME/.sparkinfer_bot_tree}"
  BOT_TREE="${BOT_TREE%/}"
  TREE_WHY="no origin/main bot tree"   # the wrappers' refusal reason when this returns 1
  case "$BOT_TREE" in
    /?*) ;;
    *) TREE_WHY="SPARKINFER_BOT_TREE must be an absolute path (got '$BOT_TREE')"; echo "$TREE_WHY" >&2; return 1 ;;
  esac
  if [ "$(realpath -m "$BOT_TREE")" = "$(realpath -m "$HOME")" ] \
      || [ "$(realpath -m "$BOT_TREE")" = "$(realpath -m "$REPO_DIR")" ]; then
    TREE_WHY="SPARKINFER_BOT_TREE must be a directory of its own (got '$BOT_TREE')"
    echo "$TREE_WHY" >&2
    return 1
  fi
  if ! timeout "$STEP_TIMEOUT_S" git -C "$REPO_DIR" fetch -q origin main 2>/dev/null; then
    echo "WARN: git fetch origin main failed — running the last fetched origin/main" >&2
  fi
  if ! git -C "$REPO_DIR" rev-parse -q --verify origin/main >/dev/null; then
    TREE_WHY="origin/main does not resolve in $REPO_DIR"
    echo "$TREE_WHY — the bot tree is left as it is" >&2
    return 1
  fi
  # REPO_DIR still follows main when it is on main, so the wrappers themselves stay current.
  if [ "$(git -C "$REPO_DIR" symbolic-ref -q --short HEAD 2>/dev/null)" = main ]; then
    timeout "$STEP_TIMEOUT_S" git -C "$REPO_DIR" merge -q --ff-only origin/main >/dev/null 2>&1 \
      || echo "WARN: $REPO_DIR could not fast-forward to origin/main (local changes?) — the bot runs origin/main regardless" >&2
  else
    echo "NOTE: $REPO_DIR is not on main — the bot runs origin/main from $BOT_TREE regardless, but this wrapper is that checkout's copy" >&2
  fi
  state="$(_tree_state)"
  case "$state" in
    foreign)
      TREE_WHY="$BOT_TREE is not the bot tree these wrappers made — refusing to reset or remove it; move it away or set SPARKINFER_BOT_TREE"
      echo "$TREE_WHY" >&2
      return 1 ;;
    ours)
      _reset_tree && { echo "bot tree: $BOT_TREE @ $(git -C "$BOT_TREE" rev-parse --short HEAD) (origin/main)"; return 0; }
      echo "WARN: $BOT_TREE could not be reset (a checkout killed part-way?) — making it again" >&2
      _drop_tree ;;
    orphan)
      # Moved aside, not deleted: its admin dir is gone, so its mark cannot be checked.
      echo "WARN: $BOT_TREE lost its git admin dir (the working copy re-cloned?) — moved to $BOT_TREE.orphan-$$, making it again" >&2
      mv -- "$BOT_TREE" "$BOT_TREE.orphan-$$" || { TREE_WHY="could not move $BOT_TREE aside"; return 1; } ;;
  esac
  if ! { _make_tree && _reset_tree; }; then
    TREE_WHY="could not make $BOT_TREE an origin/main worktree of $REPO_DIR"
    echo "$TREE_WHY" >&2
    return 1
  fi
  echo "bot tree: $BOT_TREE @ $(git -C "$BOT_TREE" rev-parse --short HEAD) (origin/main, new)"
}

# Outage escalation, from run_dspark_cron.sh (2026-09-09). A dead box does NOT stop a tick -- it
# degrades it to --labels-only, which logs one ordinary-looking line and exits 0. On 2026-09-04/05
# that ran 14 consecutive times and nothing got louder, so the outage was only found by someone
# asking why there were no new numbers. Silent degradation is the right RUNTIME behaviour -- never
# rent, never fail a tick -- but it must not be silent to a reader.
note_gpu_up() { rm -f "$1"; }
note_gpu_down() {  # $1 = counter file, $2 = bot name; prints the consecutive count (reads TS, GPU_LABEL)
  local f="$1" n first
  n=0; first="$TS"
  if [ -f "$f" ]; then
    n="$(sed -n 1p "$f" 2>/dev/null | tr -dc '0-9')"; n="${n:-0}"
    first="$(sed -n 2p "$f" 2>/dev/null)"; first="${first:-$TS}"
  fi
  n=$((n + 1))
  printf '%s\n%s\n' "$n" "$first" >"$f"
  # Banner to STDERR, count to STDOUT. The caller reads the count through a $(...) substitution,
  # which would otherwise swallow the banner entirely -- the exact bug this block exists to prevent.
  # Cron's `>> log 2>&1` puts stderr in the same log, so the banner still lands next to the tick.
  if _loud_now "$n"; then
    {
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      echo "!! $2 EVAL DEGRADED: $n consecutive ticks with NO GPU."
      echo "!! Pinned box $GPU_LABEL unreachable since $first."
      echo "!! Nothing has been measured, scored or auto-merged since then."
      echo "!! Fix: point EVAL_SSH_HOST/EVAL_SSH_PORT in .env.eval at a live RTX 5090."
      echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    } >&2
  fi
  echo "$n"
}

# Runs one bot pass under BOT_TIMEOUT_S: run_bot <bot> <command...>. timeout signals its whole
# process group, so the bot's ssh children go with it. A bot that fails every tick (a broken commit
# on main, say) gets the same escalation as a dead box.
run_bot() {
  local bot="$1" rc start=$SECONDS n
  shift
  timeout --kill-after=60 "$BOT_TIMEOUT_S" "$@"
  rc=$?
  if [ "$rc" -eq 124 ] || { [ "$rc" -eq 137 ] && [ $((SECONDS - start)) -ge "$BOT_TIMEOUT_S" ]; }; then
    echo "!! [$(date -u +%FT%TZ)] bot run stopped after ${BOT_TIMEOUT_S}s — the next tick picks the round back up" >&2
  elif [ "$rc" -ge 128 ]; then
    echo "!! [$(date -u +%FT%TZ)] bot run killed by signal $((rc - 128))" >&2
  fi
  if [ "$rc" -eq 0 ]; then
    rm -f "$HOME/.sparkinfer_${bot}_failed_runs"
  else
    n="$(_bump "$HOME/.sparkinfer_${bot}_failed_runs")"
    if _loud_now "$n"; then
      {
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!! ${bot} BOT FAILING: $n consecutive runs ended with exit $rc — see the lines above."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      } >&2
    fi
  fi
  return "$rc"
}
