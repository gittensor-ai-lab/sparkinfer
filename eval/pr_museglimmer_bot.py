#!/usr/bin/env python3
"""sparkinfer Muse Glimmer PR auto-evaluator.

THE SCORED BOT as of 2026-09-09, by explicit instruction. It replaces pr_dspark_bot.py on the
hourly cron; only one bot may hold /tmp/sparkinfer_bot.lock and they all drive the same single
pinned GPU, so they must not run concurrently.

NOTE: the crontab is machine state, not repo state -- merging a commit that changes this note
does NOT change what runs. Whichever `eval/run_*_cron.sh` line is in the eval host's crontab is
the bot actually scoring PRs.

Sibling of pr_dflash_bot.py. Scope as of 2026-09-09 (widened from decode@0 + prefill@128):
Muse Glimmer's plain AR decode AND prefill across five contexts, plus two cross-model
no-regression guards at 32k. The earlier narrow scope was deliberate -- Muse Glimmer was a very
young architecture (4 real correctness bugs found and fixed in one bring-up session, 6d911d4 and
preceding) -- but it had become the wrong tradeoff: every dimension sat at or below 128 tokens,
so the gate was blind to long-context work. PR #1006 ("40x prefill@4k", int8 KV honoured at
ctx >= 4096) moves nothing the old matrix measured.

Scoring, same-box PR-vs-main on a single pinned GPU:
  1. Speed  — decode AND prefill at ctx 128/512/4k/16k/32k (SCORED_CTXS), ten axes, one Muse
              Glimmer model load via
              bench_sweep_run, PR vs a freshly-measured origin/main, same box, same run. Same
              tier buckets as the AR and DFlash bots (BUCKETS/SIG/REGRESS_TOL below — copied,
              not reinvented). EVERY axis is also a no-regression floor — REGRESS_TOL failing on
              ANY ONE of the ten is a hard REJECT, so a PR cannot buy a headline win at one
              context by giving away another — but otherwise the tier is the BEST measured delta
              across the set: a PR that improves one axis with the rest merely flat (not
              regressed) still earns credit for the real improvement it made, e.g. an XL
              prefill@4k win paired with flat decode still reports XL. Originally two dimensions,
              added 2026-08-12 (pt. 4 below) because Muse Glimmer's
              `batched_prefill_enabled()` always returns false (its SWA/NoPE per-layer pattern
              has no batched kernel yet), so it *always* pays the slow token-loop prefill path —
              a decode-only gate could never see a prefill-specific regression.
  2. Accuracy gate — teacher-forced qwen3_gguf_score vs a live llama-server reference on the
              SAME GGUF, exactly the methodology validated by hand this session (commit
              6d911d4's message): qwen3_gguf_score dumps sparkinfer's per-position distribution,
              llama-server answers /completion (n_probs + cache_prompt + temperature=0) for the
              same positions, accuracy_compare.py reports top1/KL. Bar: top1 >= 0.90, KL <= 0.10
              (this session achieved 0.980 / 0.0029 on the exact eval_text.txt corpus reused
              here). A PR that fails this bar is REJECTed regardless of speed — speed is
              meaningless if a PR silently breaks a still-fragile architecture's correctness.

Applies `eval-museglimmer:<TIER>` AND mirrors it to the generic `eval:<TIER>` label (SN74
scoring reads eval:* tiers) — explicit user decision, 2026-08-11; originally NOT mirrored given
Muse Glimmer's youth at the time, see git history on apply_result() for that reasoning. Auto-
close on none/REJECT and auto-merge on a verified speedup are both live (SPARKINFER_MUSEGLIMMER_
AUTOMERGE=1 in .env.eval) — same policy as pr_dflash_bot.py, also an explicit user decision
after this bot's first live run wrongly auto-closed an unrelated PR (#768, reopened +
apologized); the user was told the risk directly and chose to accept it rather than narrow the
evaluation scope.

  3. Cross-model no-regression guards @ 32k — decode + prefill on TWO models, same box, same PR
              build, vs a freshly-measured origin/main:
                * Qwen3.6-35B-A3B (Q36_GUARD_*), and
                * the ModelOpt Qwen3.8-27B NVFP4 checkpoint (MODELOPT_MODEL_DIR) -- the one
                  pr_dspark_bot.py scores, so a shared-code regression is caught here at Muse-PR
                  time instead of surfacing later as a mystery in that bot's numbers.
              Narrowed from the previous five-context Qwen3.6 sweep to 32k only: those extra
              points cost a model load each on models this bot does not score, and 32k is where
              shared prefill/KV code actually breaks. Both guards share one implementation
              (_check_model_guard) so they cannot drift apart. Reuses pr_dflash_bot.py's GUARD36
              sweep mechanism (bench_sweep_run, REGRESS_TOL=0.98) rather than reinventing it.
              A regression here is a hard REJECT regardless of Muse Glimmer's own speed/accuracy
              result — same discipline as the accuracy gate above. This reverses an earlier,
              explicit scope decision to leave cross-model guarding to a separate pushed backup
              branch (backup/qwen-model-optimization-20260811 @ 6d911d4) instead of a per-PR
              check; the LMCache integration (PR #775) touched shared code
              (qwen35.cpp/inference_engine.cpp) used by both Muse Glimmer and Qwen3.6, making the
              backup-branch safety net insufficient on its own — a per-PR guard catches this class
              of regression before merge, not after. Scoped to Qwen3.6 only (not Qwythos/Qwen3.5)
              per explicit instruction, 2026-08-12 — narrower than the DFlash bot's guard, in
              keeping with this bot's own "stay small and strict" design goal above.

  4. 128-ctx prefill scoring — see pt. 1's shared-floor/best-of-the-rest description. Reverses
              the earlier "not prefill" scope decision at the top of this docstring; unlike the
              Qwen3.6 guard
              (pt. 3, a pass/fail gate on a DIFFERENT model), this is Muse Glimmer's own second
              first-class scored dimension. EVAL_SCHEMA_VERSION bumped to v3-prefill128 — a PR
              evaluated before this existed must not keep a decode-only-scored label forever.

  python eval/pr_museglimmer_bot.py --instance 46074104
  python eval/pr_museglimmer_bot.py --only-prs 636 --reeval

Never rents a GPU. Shares the pinned box with the AR and DFlash bots via flock in the cron
wrapper (run_museglimmer_cron.sh) — all three MUST share /tmp/sparkinfer_bot.lock.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ssh_box import ssh_box_enabled, ssh_box_endpoint, ssh_box_user  # noqa: E402

# Reuse shared helpers from the AR bot (labels, greenlight, denylist, gh() wrapper, Polaris
# signing, stale-close primitives, …) — same import pattern as pr_dflash_bot.py.
import pr_eval_bot as arb  # noqa: E402

SPEEDUP_LABELS = {"XL", "L", "M", "S", "XS"}
# Same tier-bucketing constants as pr_dflash_bot.py / pr_eval_bot.py — copied verbatim, not
# reinvented, per explicit instruction.
SIG = 0.02
REGRESS_TOL = 0.98
BUCKETS = [(0.18, "XL"), (0.10, "L"), (0.06, "M"), (0.035, "S"), (SIG, "XS")]

# Accuracy gate bars — validated by hand this session (6d911d4: top1=0.980, KL=0.0029 achieved
# on the 99-token eval_text.txt corpus; bars themselves are the short-pass bars this codebase
# already uses elsewhere, see accuracy_compare.py's own docstring bar for top-1).
ACC_TOP1_BAR = float(os.environ.get("MUSEGLIMMER_ACC_TOP1_BAR", "0.90"))
ACC_KL_BAR = float(os.environ.get("MUSEGLIMMER_ACC_KL_BAR", "0.10"))

EVAL_PREFIX = "eval-museglimmer:"
MUSEGLIMMER_MERGE_FIRST = "museglimmer-merge-first"
MUSEGLIMMER_NEEDS_REBASE = "museglimmer-needs-rebase"
# Bumped for the Qwen3.6 no-regression guard (see module docstring, pt. 3), then again for
# 128-ctx prefill scoring (pt. 4) — same reasoning as pr_dflash_bot.py's own v2-qwenguard bump: a
# PR evaluated before a scoring change existed must not keep a stale-scored label/score forever.
# v4 (2026-09-09): scoring matrix widened from {decode@0, prefill@128} to decode AND prefill at
# 128/512/4k/16k/32k, Qwen3.6 guard narrowed to 32k, ModelOpt Qwen3.8 32k guard added. A PR
# scored under v3 must not keep a two-dimension label forever, so the version changes and every
# open PR is re-evaluated.
EVAL_SCHEMA_VERSION = "v4-ctx5-prefill-decode-modelopt-guard"
MARKER_RE = re.compile(
    r"<!-- sparkinfer-museglimmer-eval:" + re.escape(EVAL_SCHEMA_VERSION) + r":([0-9a-f]+)(?:\s+(\{.*?\}))? -->",
    re.DOTALL,
)

# --- box paths (see .env.eval's MUSEGLIMMER_* block) ---
# Deliberately a SEPARATE clone from DFLASH_REMOTE_REPO/pr_eval_bot's /root/sparkinfer: this
# session validated Muse Glimmer support against /root/sparkinfer_mg specifically, while
# /root/sparkinfer carries unrelated uncommitted work from a different task that must not be
# touched or built against.
# Defaults are the paths that actually exist on the current eval box, NOT historical ones.
#
# Both of these were stale and only worked because .env.eval overrode them -- and .env.eval is
# gitignored, so a box rebuilt from the repo alone, or a restored copy of that file, silently
# broke the bot. MUSEGLIMMER_GGUF pointed at "muse-glimmer-30B-kquant-17gb.gguf" while the file on
# disk is "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf", and REMOTE_REPO at /root/sparkinfer_mg,
# which does not exist on this box. The env vars still override; they are no longer load-bearing.
REMOTE_REPO = os.environ.get("MUSEGLIMMER_REMOTE_REPO", "/workspace/eval/bot_repo")
MUSEGLIMMER_MODELS_DIR = os.environ.get("MUSEGLIMMER_MODELS_DIR",
                                        "/root/workspace/models_muse_glimmer")
DEFAULT_GGUF = os.environ.get(
    "MUSEGLIMMER_GGUF",
    os.path.join(MUSEGLIMMER_MODELS_DIR, "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"),
)
DEFAULT_MODELS_DIR = os.environ.get("MUSEGLIMMER_MODELS_DIR", "/root/workspace/models_muse_glimmer")
# Shared llama.cpp checkout used by every eval bot on this box (persists outside any repo
# checkout so `git clean -qfd` in the remote script's checkout step can never delete it).
LLAMACPP_DIR = os.environ.get("LLAMACPP_DIR", "/root/workspace/.llamacpp")
BENCH_TOKENS = int(os.environ.get("MUSEGLIMMER_BENCH_TOKENS", "128"))
# Repeats per context; qwen3_gguf_bench's sweep mode takes the median internally. 5, not 1 --
# see the two long incident comments in _remote_script (PR #785's bogus XL and PR #790's bogus
# guard REJECT, both single-sample artefacts on a box where GPU clocks cannot be pinned).
BENCH_REPS = 5
ACC_TOPK = int(os.environ.get("MUSEGLIMMER_ACC_TOPK", "128"))
# Fixed local port for the reference llama-server this bot starts/stops per run. Distinct from
# accuracy.sh's interactive default (8081) purely so a manual accuracy.sh run on the same box
# can't collide with a bot tick (the flock in the cron wrapper already prevents two bot ticks
# from overlapping with each other).
LLAMA_SERVER_PORT = int(os.environ.get("MUSEGLIMMER_LLAMA_PORT", "8097"))
EVAL_TEXT = "bench/scripts/eval_text.txt"  # the exact known-good corpus from this session's fixes

# Qwen3.6 no-regression guard (module docstring, pt. 3) — same env var names as pr_dflash_bot.py's
# Q36_GUARD_* (PRIMARY36_MODEL_REPO/PRIMARY36_TOK_REPO) so one .env.eval entry covers both bots.
# Defaults point at this box's actual layout (confirmed 2026-08-12: /root/workspace/models36),
# distinct from the DFlash bot's vast.ai-box convention (/workspace/models36).
Q36_GUARD_MODELS_DIR = os.environ.get("Q36_GUARD_MODELS_DIR", "/root/workspace/models36")
Q36_GUARD_MODEL_FILE = os.environ.get("Q36_GUARD_MODEL_FILE", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
Q36_GUARD_MODEL_REPO = os.environ.get("PRIMARY36_MODEL_REPO", "unsloth/Qwen3.6-35B-A3B-GGUF")
Q36_GUARD_TOK_REPO = os.environ.get("PRIMARY36_TOK_REPO", "Qwen/Qwen3.6-35B-A3B")
GUARD_CTX_LABEL = {0: "128", 512: "512", 4096: "4k", 16384: "16k", 32768: "32k"}

# Scored contexts, BOTH phases at every one (2026-09-09). Previously this bot scored exactly two
# numbers: decode at ctx=0 and prefill at ctx=128. That could not see anything a long-context PR
# did -- PR #1006 (int8 KV honoured, "40x prefill@4k") moves nothing this bot measured, because
# every dimension it had sat at or below 128 tokens.
#
# 128 rather than 0 for the short point: qwen3_gguf_bench only emits a prefill number when ctx > 0
# (see print_bench_block), so a 0 context can contribute decode but never prefill. Using 128 gives
# both phases at every scored context and makes the matrix uniform.
#
# 64k added 2026-09-09. prefill_single_pass_max_tokens() defaults to 32768, so a prompt LONGER than
# that is ingested in windows with pos0 > 0 -- a different code path that no scored context reached.
# Measured on main (post-#1006) at that boundary:
#
#     ctx=32768   prefill 2080.89 pp tok/s      (single pass)
#     ctx=49152   prefill   89.11 pp tok/s      (windowed -> refused -> token loop)
#
# A 23x cliff the moment windowing engages, completely invisible to a matrix that stops at 32k.
# 64k rather than 48k because it is the round number the work in this area is reported against and
# sits further past the boundary; VRAM is flat across the two (28.0 GB at 32k, 28.4 GB at 48k --
# the prefill arena SHRINKS as windowing bounds it, offsetting the KV growth), and Muse's own
# context_length is 131072, so 64k is well inside the model's range.
#
# 64k gets its OWN tier, not a seat in the long tier, and that is a correctness requirement rather
# than a cost decision. Contexts inside one bench_sweep_run share a session: KV is sized once for
# the session's MAXIMUM context, while each context's prefill arena is sized for itself. Putting
# 64k beside 32k therefore measures 32k under 64k-sized KV pressure, and 32k is the largest
# single-pass context (prefill_single_pass_max_tokens = 32768), i.e. the most arena-hungry point.
# Measured on the same main, same box, same day:
#
#     32k prefill, long tier ending at 32k    2080.89 pp tok/s
#     32k prefill, long tier ending at 64k     939.13 pp tok/s     -55%
#
# 4k and 16k were unaffected (4128.99 / 2860.47, matching earlier rounds), so this is specifically
# the big-arena-meets-big-KV interaction. A PR-vs-main delta stayed fair either way -- both legs saw
# the same pressure -- but the 32k axis stopped measuring 32k serving, and the session also peaked
# at 97.4% of VRAM. One extra model load (~1 min) buys back both.
SCORED_CTXS = [128, 512, 4096, 16384, 32768, 65536]
# Repeats PER CONTEXT, not one number for all five. The reps=5 rule recorded below exists because
# a SHORT measurement is dominated by launch/dispatch jitter on a box where GPU clocks cannot be
# pinned -- prefill@128 completes in ~1s, so a single sample is meaningless there. That argument
# does not transfer to long contexts: measured on main 2026-09-09, Muse prefill runs ~100 pp tok/s
# at EVERY context (the sequential token-loop path), so one 32k prefill pass already takes ~356
# seconds. A measurement that long has averaged over its own jitter; repeating it five times buys
# nothing and costs half an hour.
#
#   ctx     decode tok/s   prefill pp tok/s   one prefill pass
#   128        105.50          111.65             ~1 s
#   512        103.63          110.30             ~5 s
#   4k          95.22          103.52            ~40 s
#   16k         88.93           97.26           ~168 s
#   32k         80.92           92.29           ~356 s
#
# 5/5/3/2/1 keeps the jitter defence exactly where it matters and brings one sweep from ~47 min to
# ~14 min, which is what makes an hourly round possible at all (the round sweeps TWICE, main and
# PR, before the two guards and the accuracy gate). Revisit once PR #1006-style work lands: when
# prefill stops being ~100 pp/s everywhere, long contexts get cheap and reps can go back up.
# Grouped as (contexts, reps) TIERS, not a per-context dict, because bench_sweep_run
# (bench/scripts/_eval_speed.sh) accepts "<ctx> <reps>" pairs but then collapses them:
#
#     [ "$reps" -gt "$max_reps" ] && max_reps="$reps"
#     export SPARKINFER_BENCH_SWEEP_REPS="$max_reps"
#
# One reps value is applied to EVERY context in the call -- the maximum. Every other bot passes
# the same reps for every context, so this never mattered until this bot became the first to pass
# differing values, and the pair syntax made per-context reps look supported. Passing
# "128 5 ... 32768 1" therefore ran 32k FIVE times (~27 min for that context alone) instead of
# once, turning a ~14 min sweep into ~45 min and the round into ~2.5 h.
#
# So: one bench_sweep_run call per tier. That costs one extra model load (~1 min) and is the only
# way to get 5 samples where a measurement is ~1s and 1 sample where it is ~330s.
SCORED_REPS_TIERS = [([128, 512], 5), ([4096, 16384, 32768], 1), ([65536], 1)]
SCORED_REPS = {c: r for ctxs, r in SCORED_REPS_TIERS for c in ctxs}
# The GUARDS deliberately keep BENCH_REPS (5) even at 32k, and that is not an inconsistency with
# SCORED_REPS above. Repeat count should follow how long ONE measurement takes, and that is a
# property of the model, not of the context: Muse prefills 32k in ~356s (self-averaging, reps=1 is
# fine), while Qwen3.6 and the ModelOpt checkpoint prefill the same 32k in seconds -- short,
# jitter-prone, and exactly the shape that made PR #790's guard REJECT on a single bogus 6501
# reading when two re-runs of the same binary both returned ~8477. A guard that can hard-REJECT a
# real PR is the last place to economise on samples.
SCORED_CTX_LABEL = {128: "128", 512: "512", 4096: "4k", 16384: "16k", 32768: "32k",
                    65536: "64k"}
# Order matters only for display; every entry is both a scored dimension AND a no-regression
# floor, so a PR cannot buy a win at one context by giving one away at another.
SCORING_DIMS = [f"muse-{phase}@{SCORED_CTX_LABEL[c]}"
                for c in SCORED_CTXS for phase in ("decode", "prefill")]
SCORING_DIM = SCORING_DIMS[0]

# Cross-model no-regression guards, decode AND prefill, at 32k only (2026-09-09). Narrowed from
# the previous 0/512/4k/16k/32k Qwen3.6 sweep: those five points cost five model loads per run on
# a model this bot does not score, and 32k is where shared prefill/KV code actually breaks. The
# time that buys is spent on Muse's own five scored contexts instead.
Q36_GUARD_CTXS = [32768]
MODELOPT_GUARD_CTXS = [32768]
# The ModelOpt Qwen3.8-27B NVFP4 checkpoint -- the one pr_dspark_bot.py SCORES. Guarding it here
# means a Muse PR that regresses shared code (qwen35.cpp, the prefill/KV kernels) is caught by
# this bot before it lands, rather than showing up later as a mystery regression in the DSpark
# bot's own numbers. Same env var name as that bot uses, so one .env.eval entry serves both.
MODELOPT_GUARD_MODEL_DIR = os.environ.get("MODELOPT_MODEL_DIR", "/root/workspace/models_q38_modelopt")

# Auto-merge is wired (mirrors pr_dflash_bot.py's auto_merge_ok_dflash/try_auto_merge_dflash
# shape) but OFF unless this exact env var is set — NOT set in .env.eval, so it stays fully
# inert until a human deliberately flips it on. Single-line change to enable later.
AUTO_MERGE = os.environ.get("SPARKINFER_MUSEGLIMMER_AUTOMERGE") == "1"
AUTOMERGE_BLOCK = {
    "copycat", "copycat-warn", "flagged:gaming", "penalty", "needs-benchmark",
    MUSEGLIMMER_NEEDS_REBASE, arb.REEVALUATE_LABEL, arb.HOLD_LABEL, *arb.REGRESSION_LABELS,
}

SCORES_FILE = os.path.expanduser(
    os.environ.get("MUSEGLIMMER_SCORES_FILE", "~/.sparkinfer_museglimmer_scores.json")
)

# Polaris verifiable-compute receipts — same policy/keys as the AR and DFlash bots (on by
# default; TDX via POLARIS_API_KEY when configured, else Ed25519 fallback). Wired through
# judge.py's --from-stdin generic RESULT_JSON path (NOT --dflash, which hardcodes a
# DFlash-shaped measurement block and eval_mode="dflash" — reusing it here would produce a
# mislabeled, semantically wrong attestation). SPARKINFER_EVAL_MODE is set explicitly below so
# the attestation correctly records "museglimmer-128", not the AR bot's "longctx" default.
POLARIS_ENABLED = os.environ.get("POLARIS", "1") != "0"
POLARIS_API_KEY = os.environ.get("POLARIS_API_KEY", "")
_POLARIS_PUBKEY_FILE = os.path.join(HERE, "polaris", "sparkinfer_eval.pub")


def _load_polaris_pubkey():
    try:
        with open(_POLARIS_PUBKEY_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    except Exception:
        pass
    return ""


def _load_scores():
    try:
        return json.load(open(SCORES_FILE))
    except Exception:
        return {}


def _save_scores(data):
    try:
        with open(SCORES_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f">> museglimmer scores save skipped: {e}")


def tier_from_gain(pr_tps: float, main_tps: float, metric: str = "decode"):
    """Return (label, delta_pct, pass_ok, reason). Identical logic to pr_dflash_bot.py's
    tier_from_gain — same bucket thresholds, same no-regression floor. `metric` only affects the
    reason text — lets prefill@128 scoring reuse this verbatim instead of duplicating it (this
    codebase's "copied, not reinvented" convention) while still reporting which dimension a
    regression/improvement actually came from."""
    if main_tps <= 0:
        return "REJECT", 0.0, False, f"main {metric} baseline is 0"
    if pr_tps < REGRESS_TOL * main_tps:
        pct = 100.0 * (pr_tps - main_tps) / main_tps
        return "REJECT", round(pct, 1), False, (
            f"{metric} regression: {pr_tps:.2f} < {100 * REGRESS_TOL:.0f}% of main {main_tps:.2f}"
        )
    g = (pr_tps - main_tps) / main_tps
    pct = round(100.0 * g, 1)
    if g < SIG:
        return "none", pct, True, f"within significance gate — not a verified {metric} improvement"
    for thr, name in BUCKETS:
        if g >= thr:
            return name, pct, True, "ok"
    return "none", pct, True, "ok"


_TIER_RANK = {"REJECT": -1, "none": 0, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5}


def _check_model_guard(pr: dict, main: dict, key: str, model: str, tol: float = REGRESS_TOL):
    """No-regression check for ONE guarded model: PR vs same-box main, decode + prefill, every
    measured context. Adapted from pr_dflash_bot.py's check_qwen_guard.

    Parameterised by `key` (the _parse_remote dict key holding that model's per-context numbers)
    and `model` (its display name) so the Qwen3.6 and ModelOpt guards are the SAME code rather
    than two copies that can drift apart -- the failure mode of a duplicated guard is that one of
    them quietly stops guarding and nobody notices, which is exactly how the DSpark bot's Qwen3.8
    guard spent its whole life benching the scored checkpoint against itself.

    Returns (ok, [human-readable regression/failure strings])."""
    problems = []
    if pr.get(f"{key}_failed") or main.get(f"{key}_failed") or not pr.get(key) or not main.get(key):
        problems.append(f"{model} guard measurement unavailable")
    pr_ctxs, main_ctxs = pr.get(key) or {}, main.get(key) or {}
    # Iterate over MAIN's contexts (the reference set) — a PR build that crashes partway through
    # its own sweep must not make that context silently uncheckable. Fail closed: a real main
    # baseline (base > 0) with a missing/zero PR measurement (cur <= 0) is a regression, not a skip.
    for ctx, main_vals in main_ctxs.items():
        label = GUARD_CTX_LABEL.get(ctx, str(ctx))
        pr_vals = pr_ctxs.get(ctx) or {}
        for metric in ("decode", "prefill"):
            base = main_vals.get(metric, 0)
            if base <= 0:
                continue  # main itself has no baseline for this metric/ctx — not comparable
            cur = pr_vals.get(metric, 0)
            if cur <= 0:
                problems.append(
                    f"{model} {metric}@{label}: PR measurement missing/zero "
                    f"(main {base:.1f}) — treated as regression"
                )
                continue
            if cur < base * tol:
                pct = 100.0 * (cur - base) / base
                problems.append(
                    f"{model} {metric}@{label}: {cur:.1f} < {100 * tol:.0f}% of main "
                    f"{base:.1f} ({pct:+.1f}%)"
                )
    return (len(problems) == 0, problems)


def check_q36_guard(pr: dict, main: dict, tol: float = REGRESS_TOL):
    """Qwen3.6 no-regression guard (module docstring pt. 3), decode + prefill @ 32k."""
    return _check_model_guard(pr, main, "guard36", "qwen3.6", tol)


def check_modelopt_guard(pr: dict, main: dict, tol: float = REGRESS_TOL):
    """ModelOpt Qwen3.8-27B NVFP4 no-regression guard, decode + prefill @ 32k. Same discipline as
    the Qwen3.6 guard and the same hard-REJECT consequence -- this is the checkpoint the DSpark
    bot scores, so a Muse PR that regresses it via shared code must not land."""
    return _check_model_guard(pr, main, "guardmo", "modelopt", tol)


def museglimmer_evaluated_commits(repo, num):
    """Head commits that already have a REAL scoring verdict posted — mirrors
    dflash_evaluated_commits: infra/transport failures (label:null in the marker) don't count."""
    r = arb.gh(["pr", "view", str(num), "-R", repo, "--json", "comments"])
    done = set()
    for c in json.loads(r.stdout or "{}").get("comments", []):
        body = c.get("body") or ""
        m = MARKER_RE.search(body)
        if not m or "sparkinfer museglimmer auto-eval" not in body:
            continue
        meta_raw = m.group(2)
        try:
            meta = json.loads(meta_raw) if meta_raw else {}
        except json.JSONDecodeError:
            meta = {}
        if meta.get("label") is None:
            continue
        done.add(m.group(1))
    return done


def strip_museglimmer_eval_labels(repo, num):
    for lab in list(arb.labels_on(repo, num)):
        if lab.startswith(EVAL_PREFIX):
            arb.remove_label(repo, num, lab)


STALE_DAYS = float(os.environ.get("MUSEGLIMMER_STALE_DAYS", "1"))


def _pr_last_activity_ts(repo, num):
    """Last real author activity (most recent commit's committedDate), not PR updatedAt — same
    rationale as pr_dflash_bot.py's copy of this helper (bot comments/labels bump updatedAt)."""
    r = arb.gh(["pr", "view", str(num), "-R", repo, "--json", "commits,createdAt"])
    try:
        info = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return None
    dates = [c.get("committedDate") for c in (info.get("commits") or []) if c.get("committedDate")]
    ts_str = max(dates) if dates else info.get("createdAt")
    if not ts_str:
        return None
    try:
        return time.mktime(time.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return None


def close_stale_museglimmer_prs(repo, prs, dry_run=False):
    """Close open PRs with no author commit activity in STALE_DAYS+ days. HOLD_LABEL and the
    current museglimmer-merge-first winner are exempt."""
    closed = set()
    now = time.time()
    for pr in prs:
        num = pr["number"]
        if pr.get("isDraft"):
            continue
        labs = {l["name"] for l in pr.get("labels", [])}
        if arb.HOLD_LABEL in labs or MUSEGLIMMER_MERGE_FIRST in labs:
            continue
        ts = _pr_last_activity_ts(repo, num)
        if ts is None:
            continue
        age_days = (now - ts) / 86400
        if age_days < STALE_DAYS:
            continue
        print(f"PR #{num}: stale ({age_days:.1f}d since last commit, threshold {STALE_DAYS}d) — closing")
        closed.add(num)
        if dry_run:
            continue
        body = (
            "<!-- sparkinfer-museglimmer-auto-close-stale -->\n"
            f"## Closed: stale — no commits in {age_days:.1f} days\n\n"
            f"This PR has had no new commits in over {STALE_DAYS:g} days — closing automatically "
            "to keep the Muse Glimmer eval queue clean. Reopen (or push a new commit / open a "
            "fresh PR) whenever you're ready to continue; it'll be picked back up on the next "
            "eval cycle."
        )
        arb.gh(["pr", "comment", str(num), "-R", repo, "--body", body])
        arb.gh(["pr", "close", str(num), "-R", repo])
    return closed


def resolve_ssh(instance_id: int):
    """Return (host, port) for the pinned box. Same logic as pr_dflash_bot.py's resolve_ssh."""
    if ssh_box_enabled():
        ep = ssh_box_endpoint()
        if not ep:
            raise RuntimeError("EVAL_TRANSPORT=ssh but EVAL_SSH_HOST unset")
        return ep
    key = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/speedy"))
    os.environ.setdefault("SSH_KEY", key)
    iid = arb.current_instance(instance_id) or instance_id
    raw = subprocess.run(
        ["vastai", "show", "instance", str(iid), "--raw"],
        capture_output=True, text=True, timeout=60,
    )
    if raw.returncode != 0 or not (raw.stdout or "").strip():
        raise RuntimeError(f"vastai show instance {iid} failed: {(raw.stderr or '')[:200]}")
    info = json.loads(raw.stdout)
    ip = (info.get("public_ipaddr") or "").strip()
    ports = info.get("ports") or {}
    m = ports.get("22/tcp") or [{}]
    port = int((m[0] or {}).get("HostPort") or 0)
    if info.get("actual_status") != "running" or not ip or not port:
        raise RuntimeError(
            f"pinned instance {iid} not SSH-ready (status={info.get('actual_status')})"
        )
    return ip, port


def ssh_run(host, port, cmd, timeout=7200, stdin_data=None, via_stdin=False):
    """Same shape as pr_dflash_bot.py's ssh_run — via_stdin=True avoids MAX_ARG_STRLEN limits
    for the (much smaller here, but still nontrivial) remote script text."""
    key = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/speedy"))
    user = ssh_box_user() if ssh_box_enabled() else "root"
    remote = ["bash", "-s"] if via_stdin else [cmd]
    return subprocess.run(
        [
            "ssh", "-i", key,
            "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=40",
            "-p", str(port), f"{user}@{host}", *remote,
        ],
        capture_output=True, text=True, timeout=timeout,
        input=cmd if via_stdin else stdin_data,
    )


_EXPLICIT_FAIL_MARKERS = ("BUILD_FAILED", "LLAMACPP_CONFIGURE_FAILED", "LLAMACPP_BUILD_FAILED")


def _crash_reason(*outputs: str) -> str | None:
    """Same ERR-trap diagnostic extraction as pr_dflash_bot.py, PLUS the remote script's own
    explicit *_FAILED markers (BUILD_FAILED, LLAMACPP_CONFIGURE_FAILED, LLAMACPP_BUILD_FAILED).
    Those markers are `exit 1` inside an `|| { ...; exit 1; }` handler, not a bare failing
    command under `set -e` -- bash's ERR trap does NOT fire for an explicit `exit`, so a
    perfectly-diagnosed compile error (e.g. #777's `launch_muse_sandwich_tail_q8` undefined)
    was falling through to `_looks_like_hard_kill`'s "no diagnostic captured" bucket, getting
    misreported as an ambiguous hard kill AND triggering a pointless retry of a build that will
    deterministically fail again. Found by hand-running _remote_script directly against #777
    after the bot mislabeled it twice."""
    combined = "\n".join(o or "" for o in outputs)
    lines = combined.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("REMOTE_SCRIPT_FAILED "):
            extra = lines[i + 1].strip() if i + 1 < len(lines) else ""
            return line.strip() + (f" | gpu: {extra}" if extra else "")
    for i, line in enumerate(lines):
        marker = next((m for m in _EXPLICIT_FAIL_MARKERS if line.startswith(m)), None)
        if not marker:
            continue
        # The single most actionable line is usually the compiler/linker's own "error:" —
        # prefer that over the marker's generic "tail of build.log:" header.
        for follow in lines[i + 1:i + 60]:
            if "error:" in follow or "Error " in follow:
                return f"{marker}: {follow.strip()}"
        tail = " | ".join(l.strip() for l in lines[i + 1:i + 3] if l.strip())
        return marker + (f": {tail}" if tail else "")
    return None


def _looks_like_hard_kill(stdout: str, stderr: str) -> bool:
    """Same hard-kill heuristic as pr_dflash_bot.py (no ERR-trap diagnostic captured at all)."""
    combined = (stdout or "") + "\n" + (stderr or "")
    if _crash_reason(stdout, stderr):
        return False
    return "ACCURACY_STAGE_DONE" not in combined  # never reached even the accuracy checkpoint


def _ssh_run_resilient(host, port, script: str, label: str):
    """One automatic retry on an apparent hard kill — same insurance pr_dflash_bot.py added
    after #684/#690 (heavy model-reload boundaries silently killing the whole remote shell)."""
    r = ssh_run(host, port, script, via_stdin=True)
    if r.returncode != 0 and _looks_like_hard_kill(r.stdout, r.stderr):
        print(f">> {label}: looks like a hard kill (no ERR-trap diagnostic, no accuracy-stage "
              f"checkpoint reached) — retrying once")
        r = ssh_run(host, port, script, via_stdin=True)
    return r


def _remote_script(ref: str) -> str:
    """Bash run on the eval box: checkout ref, build, 128-decode speed bench, accuracy gate vs
    a live llama-server reference. Run once per ref (PR, then "main") — identical script both
    times so the two measurements are directly comparable."""
    repo = shlex.quote(REMOTE_REPO)
    gguf = shlex.quote(DEFAULT_GGUF)
    llamacpp_dir = shlex.quote(LLAMACPP_DIR)
    ref_q = shlex.quote(ref)
    ntok = BENCH_TOKENS
    topk = ACC_TOPK
    port = LLAMA_SERVER_PORT
    eval_text = shlex.quote(EVAL_TEXT)
    q36_dir = shlex.quote(Q36_GUARD_MODELS_DIR)
    q36_file = shlex.quote(Q36_GUARD_MODEL_FILE)
    q36_repo = shlex.quote(Q36_GUARD_MODEL_REPO)
    q36_tok = shlex.quote(Q36_GUARD_TOK_REPO)
    mo_dir = shlex.quote(MODELOPT_GUARD_MODEL_DIR)
    # bench_sweep_run takes alternating "<ctx> <reps>" pairs; the ctx-only list drives the shell
    # for-loop that reads the results back out. Both are derived from the same SCORED_CTXS /
    # *_GUARD_CTXS constants so the sweep and the read-back can never disagree about which
    # contexts were measured.
    #
    # reps=5 everywhere, for the reason the long comments below record: this box cannot pin GPU
    # clocks ("current user does not have permission to change clocks"), so median-of-N is the
    # only defence against a single noisy sample hard-REJECTing a real PR.
    # One shell block per reps tier; see SCORED_REPS_TIERS for why this cannot be a single call.
    scored_blocks = []
    for ctxs, reps in SCORED_REPS_TIERS:
        args = " ".join(f"{c} {reps}" for c in ctxs)
        lst = " ".join(str(c) for c in ctxs)
        scored_blocks.append(
            f'wait_gpu_clear\n'
            f'if bench_sweep_run "$GGUF" "$NTOK" {args}; then\n'
            f'  for ctx in {lst}; do\n'
            f'    echo "MUSE $ctx $(_bench_sweep_get $ctx decode_tps) $(_bench_sweep_get $ctx prefill_pp)"\n'
            f'    if [ "$ctx" = "128" ]; then\n'
            f'      BC_DECODE=$(_bench_sweep_get 128 decode_tps)\n'
            f'      BC_PREFILL=$(_bench_sweep_get 128 prefill_pp)\n'
            f'    fi\n'
            f'  done\n'
            f'else\n'
            f'  MUSE_OK=0\n'
            f'fi')
    scored_sweep_blocks = "\n".join(scored_blocks)
    q36_sweep_args = " ".join(f"{c} {BENCH_REPS}" for c in Q36_GUARD_CTXS)
    q36_ctx_list = " ".join(str(c) for c in Q36_GUARD_CTXS)
    mo_sweep_args = " ".join(f"{c} {BENCH_REPS}" for c in MODELOPT_GUARD_CTXS)
    mo_ctx_list = " ".join(str(c) for c in MODELOPT_GUARD_CTXS)
    return f"""
set -euo pipefail
# Surface *why* a crash happened instead of dying silently — same diagnostic trap as
# pr_dflash_bot.py's _remote_script (a REJECT from an infra crash should carry a real cause).
trap 'rc=$?; ln=$LINENO; reason=""; \\
  case $rc in \\
    137) reason="likely OOM-killed (SIGKILL)" ;; \\
    139) reason="likely segfault (SIGSEGV)" ;; \\
    134) reason="likely abort (SIGABRT)" ;; \\
    124) reason="likely timeout" ;; \\
  esac; \\
  echo "REMOTE_SCRIPT_FAILED line=$ln exit=$rc reason=$reason" >&2; \\
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader >&2 2>/dev/null || true' ERR

# One run does several back-to-back multi-GB model load/unload cycles (sparkinfer speed bench,
# sparkinfer score dump, llama-server) — poll GPU memory down to near-empty before each heavy
# load instead of assuming the previous process's exit already freed it (same lesson as
# pr_dflash_bot.py's wait_gpu_clear, #684/#690).
wait_gpu_clear() {{
  local tries=0 used
  while [ "$tries" -lt 30 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -lt 1024 ] 2>/dev/null && return 0
    sleep 1
    tries=$((tries + 1))
  done
  echo "WARN: GPU memory still ${{used:-unknown}} MiB after ${{tries}}s wait — proceeding anyway" >&2
}}

export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
export CUDA_HOME=${{CUDA_HOME:-/usr/local/cuda-13.0}}
REPO={repo}
GGUF={gguf}
NTOK={ntok}
TOPK={topk}
PORT={port}
EVAL_TEXT={eval_text}
Q36_GUARD_MODELS_DIR={q36_dir}
Q36_GUARD_MODEL_FILE={q36_file}
Q36_GUARD_MODEL_REPO={q36_repo}
Q36_GUARD_TOK_REPO={q36_tok}
MODELOPT_GUARD_MODEL_DIR={mo_dir}

cd "$REPO"
git remote set-url origin https://github.com/gittensor-ai-lab/sparkinfer.git 2>/dev/null || true
git fetch -q origin {ref_q}
git reset -q --hard
git clean -qfd
git checkout -qf FETCH_HEAD
HEAD=$(git rev-parse --short HEAD)
echo "REMOTE_HEAD $HEAD"

test -f "$GGUF" || {{ echo "FAIL missing GGUF $GGUF"; exit 1; }}

# Build sparkinfer's speed-bench + teacher-forced-score binaries. Always reconfigure (cheap,
# idempotent) — skipping it on an existing CMakeCache left stale generated Makefiles pointing at
# a DIFFERENT PR branch's files once the checkout switched underneath it (pr_dflash_bot.py
# #693/#694 hit exactly this).
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/mg_cmake.log 2>&1
cmake --build build --target qwen3_gguf_bench qwen3_gguf_score -j"$(nproc)" >/tmp/mg_build.log 2>&1 || {{
  echo "BUILD_FAILED — tail of /tmp/mg_build.log:" >&2
  tail -80 /tmp/mg_build.log >&2
  exit 1
}}
test -x build/runtime/qwen3_gguf_bench
test -x build/runtime/qwen3_gguf_score

# --- 128-token decode speed (ctx=0, n=128 — the SAME "128-context, no prefill" convention the
# rest of this codebase's decode buckets use, e.g. pr_eval_bot.py's CTX_SERIES[128]) PLUS
# 128-ctx prefill throughput, from ONE Muse Glimmer model load via the same JSON-sweep mechanism
# the Qwen3.6 guard below already uses — sourced here (not just before the guard) so both share
# it instead of reloading the ~17GB GGUF twice for two separate single-context bench calls.
#
# reps=5 (median), not 1: found 2026-08-13 after PR #785 scored a bogus eval-museglimmer:XL
# (207% "prefill improvement" it never touched — its own diff never goes near a prefill kernel).
# Two things were true at once, confirmed live on the eval box with a per-rep debug build:
#   1. reps=1 (no averaging) is fragile now that prefill@128 is fast: qwen3_gguf_bench's sweep
#      mode medians over `reps` internally (one shared SPARKINFER_BENCH_SWEEP_REPS across every
#      ctx point, see run_sweep() in qwen3_gguf_bench.cpp), but this call passed reps=1, i.e. a
#      single uncorroborated sample. That was fine while Muse Glimmer's prefill@128 ran the slow
#      token-loop path (~1.1s — dispatch jitter is noise-floor); #787's batched-GEMM prefill (the
#      same round) cut that to ~0.1s, where the same jitter is a much larger fraction of a shorter
#      measurement.
#   2. PR #785 itself was NOT just an unlucky single sample: on current main (#785 reverted) all
#      5/5 reps land in a tight 1105-1115 pp tok/s band, but on #785's own branch every rep was
#      wildly inflated (reps=1 -> 3395, reps=5 median -> 84723, i.e. taking more samples made it
#      WORSE, not better). #785's Q3A weight requantization runs once at load, before any prefill
#      code, yet destabilized every subsequent prefill measurement on that binary -- a real bug in
#      that PR, not benchmark noise. Left as a mystery for whoever resubmits it; reps=5 defends
#      this bot against a repeat regardless of root cause on the PR side.
source bench/scripts/_common.sh
source bench/scripts/_eval_speed.sh
SI_BIN="$PWD/build/runtime"; SI_LD=""
# Fail loudly and immediately on a missing checkpoint. Without this the sweep just returns no
# rows, the baseline comes back empty, and the round reports a measurement problem several
# minutes later with no hint that the PATH was the issue -- which is exactly how a stale
# MUSEGLIMMER_GGUF hid for as long as it did.
if [ ! -f "$GGUF" ]; then
  echo "FAIL Muse Glimmer checkpoint not found at $GGUF" >&2
  echo "     (set MUSEGLIMMER_GGUF, or place the model in MUSEGLIMMER_MODELS_DIR)" >&2
  ls -la "$(dirname "$GGUF")" 2>&1 | head -10 >&2
  exit 1
fi
wait_gpu_clear
# Five contexts, decode AND prefill at each, in ONE model load (bench_sweep_run sweeps all the
# ctx points per load). Emitting one MUSE line per context rather than two fixed RESULT_ lines
# keeps the wire format the same shape as the guards' and lets SCORED_CTXS change without
# touching the parser.
MUSE_OK=1
BC_DECODE=0
BC_PREFILL=0
{scored_sweep_blocks}
[ "$MUSE_OK" = "1" ] || echo "MUSE_FAILED"
# Back-compat: decode@128 / prefill@128 also go out under their old names. The PR comment
# renderer, the Polaris payload and the published dashboard all read these two keys, and none of
# them should have to change because the scoring matrix grew.
# Captured during the tier that measured 128 -- _bench_sweep_get reads the LAST sweep's JSON, and
# 128 is not in the last tier.
echo "RESULT_DECODE_TPS ${{BC_DECODE:-0}}"
echo "RESULT_PREFILL128_PP ${{BC_PREFILL:-0}}"

# --- accuracy gate: sparkinfer teacher-forced score vs a live llama-server reference, same
# GGUF, same eval_text.txt corpus this session already validated by hand (6d911d4) ---
#
# Deliberately NOT bench/scripts/_common.sh's ensure_llamacpp()/reference.lock here: reference.lock
# pins LLAMACPP_COMMIT to a July-2026 commit that predates llama.cpp's native muse-glimmer
# architecture support entirely (merged ~2026-08-10). ensure_llamacpp() tamper-checks the
# checkout's HEAD against that pin and — on mismatch — fetches and resets to the OLD pinned
# commit, which cannot even load a general.architecture=muse-glimmer GGUF. Calling it here would
# silently wreck the box's already-working llama.cpp checkout on the very first tick and break
# every subsequent accuracy gate. reference.lock is shared with the Qwen3.5/Qwen3.6/Qwythos evals
# (pr_eval_bot.py/pr_dflash_bot.py) — bumping it to a bleeding-edge commit to fix this would risk
# changing THEIR baseline numbers, which is out of scope here. Build straight off whatever's
# already checked out at LLAMACPP_DIR instead, unpinned, isolated to this bot only.
LLAMACPP_DIR={llamacpp_dir}
if [ ! -d "$LLAMACPP_DIR/.git" ]; then
  echo "FAIL: $LLAMACPP_DIR missing or not a git checkout. Muse Glimmer's accuracy gate needs a" >&2
  echo "llama.cpp build with native muse-glimmer support (src/models/muse-glimmer.cpp) -- clone" >&2
  echo "https://github.com/ggml-org/llama.cpp fresh into this path before the bot's first run." >&2
  exit 1
fi
mkdir -p "$LLAMACPP_DIR/build"
cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \\
  -DCMAKE_BUILD_TYPE=Release >/tmp/mg_llamacpp_cmake.log 2>&1 || {{
  echo "LLAMACPP_CONFIGURE_FAILED — tail:" >&2
  tail -60 /tmp/mg_llamacpp_cmake.log >&2
  exit 1
}}
cmake --build "$LLAMACPP_DIR/build" -j"$(nproc)" --target llama-server llama-tokenize \\
  >/tmp/mg_llamacpp_build.log 2>&1 || {{
  echo "LLAMACPP_BUILD_FAILED — tail:" >&2
  tail -80 /tmp/mg_llamacpp_build.log >&2
  exit 1
}}
test -x "$LLAMACPP_DIR/build/bin/llama-tokenize"
test -x "$LLAMACPP_DIR/build/bin/llama-server"

# Tokenize the known-good corpus with llama-tokenize (not gen_eval_prompt.py's HF-tokenizer
# path) — guarantees byte-identical tokenization to the llama-server reference queried below,
# same approach already proven this session.
TOKOUT=$("$LLAMACPP_DIR/build/bin/llama-tokenize" -m "$GGUF" -f "$EVAL_TEXT" --ids)
IDS=$(echo "$TOKOUT" | grep -oE '[0-9]+' | tr '\\n' ' ')
echo "$IDS" > /tmp/mg_eval_ids.txt
test -n "$IDS" || {{ echo "FAIL empty tokenized IDS from llama-tokenize"; exit 1; }}
echo "TOKEN_COUNT $(echo "$IDS" | wc -w)"

wait_gpu_clear
build/runtime/qwen3_gguf_score "$GGUF" "$TOPK" $IDS > /tmp/mg_score.txt
grep -q '^PPL' /tmp/mg_score.txt || echo "WARN: qwen3_gguf_score produced no PPL line" >&2

wait_gpu_clear
"$LLAMACPP_DIR/build/bin/llama-server" -m "$GGUF" -ngl 99 -c 2048 --port "$PORT" --no-jinja \\
  >/tmp/mg_llama_srv.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 2
done
echo "ACCURACY_STAGE_DONE"

# /dev/null as the tokenizer-path arg is safe: accuracy_compare.py's 3rd positional arg is a
# file of already-tokenized space-separated ids (produced above), so its all-digit check skips
# the HF-tokenizer-load code path entirely — the tokenizer path is never opened.
ACCOUT=$(python3 bench/scripts/accuracy_compare.py /tmp/mg_score.txt /dev/null /tmp/mg_eval_ids.txt \\
         "http://localhost:$PORT" "$TOPK")
echo "$ACCOUT"
kill $SRV 2>/dev/null || true
wait $SRV 2>/dev/null || true
trap - EXIT

METRIC_LINE=$(echo "$ACCOUT" | grep '^METRIC ' | tail -1)
TOP1=$(echo "$METRIC_LINE" | sed -E 's/.*top1=([0-9.]+).*/\\1/')
KL=$(echo "$METRIC_LINE" | sed -E 's/.*kl=([0-9.]+).*/\\1/')
PPLS=$(echo "$METRIC_LINE" | sed -E 's/.*ppl_spark=([0-9.]+).*/\\1/')
PPLL=$(echo "$METRIC_LINE" | sed -E 's/.*ppl_llama=([0-9.]+).*/\\1/')
echo "RESULT_TOP1 ${{TOP1:-0}}"
echo "RESULT_KL ${{KL:-99}}"
echo "RESULT_PPL_SPARK ${{PPLS:-0}}"
echo "RESULT_PPL_LLAMA ${{PPLL:-0}}"

# --- Qwen3.6 no-regression guard (decode + prefill, ctx 0/512/4k/16k/32k) — same qwen3_gguf_bench
# binary already built above, same bench_sweep_run mechanism pr_dflash_bot.py's GUARD36 uses
# (module docstring, pt. 3). A separate GGUF/model load from Muse Glimmer's own — a shared-code
# regression that only shows up on Qwen3.6's architecture would otherwise slip past this bot
# entirely, as it did for the LMCache integration (PR #775) until checked by hand.
# _common.sh/_eval_speed.sh/SI_BIN already sourced above for the decode+prefill128 sweep — reused
# here, not re-sourced.
#
# reps=5 (median) per context, not 1: found 2026-08-13 investigating PR #790, which the guard
# REJECTed on a single-sample "qwen3.6 prefill@512: 6501.2 < 98% of main 8431.0" reading. Two
# independent re-runs of this exact sweep shape on the SAME PR #790 binary both landed at
# 8475-8479 -- matching main's own baseline, not remotely close to 6501. The 6501 reading was
# itself the noise, not a real regression (pin_clocks() is unavailable on this box -- "current
# user does not have permission to change clocks", a container/virtualization restriction, so
# there's no way to remove the underlying GPU clock variance at the source; median-of-N over
# independent samples is the only mitigation available here). Same root cause and same fix as the
# reps=1->5 change above for Muse Glimmer's own prefill128 -- this sweep just hadn't been touched
# yet. Costs a few more seconds per context but a guard that can hard-REJECT a real PR on a single
# unaveraged sample is worse than the extra runtime.
export MODELS_DIR="$Q36_GUARD_MODELS_DIR" MODEL_REPO="$Q36_GUARD_MODEL_REPO" \\
       MODEL_FILE="$Q36_GUARD_MODEL_FILE" TOK_REPO="$Q36_GUARD_TOK_REPO"
export MODEL_SHA256="${{QWEN36_MODEL_SHA256:-}}"
( ensure_model && ensure_tokenizer ) || echo "WARN: qwen3.6 guard model setup failed" >&2
Q36_GGUF="$Q36_GUARD_MODELS_DIR/$Q36_GUARD_MODEL_FILE"

echo "GUARD_START"
wait_gpu_clear
if bench_sweep_run "$Q36_GGUF" 128 {q36_sweep_args}; then
  for ctx in {q36_ctx_list}; do
    echo "GUARD36 $ctx $(_bench_sweep_get $ctx decode_tps) $(_bench_sweep_get $ctx prefill_pp)"
  done
else
  echo "GUARD36_FAILED"
fi

# --- ModelOpt Qwen3.8-27B NVFP4 no-regression guard (decode + prefill @ 32k) ---
# A compressed-tensors DIRECTORY, not a GGUF -- qwen3_gguf_bench accepts either (see its usage
# line). This is the checkpoint pr_dspark_bot.py scores; guarding it here catches a shared-code
# regression at Muse-PR time instead of leaving the other bot to discover it after the merge.
# Skipped, not failed, when the checkpoint is absent: a box without it must not turn every PR
# into a REJECT, and check_modelopt_guard reports SKIPPED so "guarded" is never claimed when
# nothing was.
if [ -d "$MODELOPT_GUARD_MODEL_DIR" ]; then
  wait_gpu_clear
  if bench_sweep_run "$MODELOPT_GUARD_MODEL_DIR" 128 {mo_sweep_args}; then
    for ctx in {mo_ctx_list}; do
      echo "GUARDMO $ctx $(_bench_sweep_get $ctx decode_tps) $(_bench_sweep_get $ctx prefill_pp)"
    done
  else
    echo "GUARDMO_FAILED"
  fi
else
  echo "GUARDMO_UNAVAILABLE"
fi
echo "GUARD_END"
"""


def _at(res: dict, ctx: int, phase: str) -> float:
    """One measurement out of a parsed run's matrix, 0.0 when that context/phase is missing.
    Every consumer reads through this rather than indexing, so a sweep that dropped one context
    degrades to a zero (which tier_from_gain scores as a regression) instead of a KeyError that
    would fail the whole round."""
    return float(((res.get("muse") or {}).get(ctx) or {}).get(phase, 0.0) or 0.0)


def _parse_remote(stdout: str) -> dict:
    out = {}
    guard36 = {}
    guardmo = {}
    muse = {}
    for line in (stdout or "").splitlines():
        if line.startswith("REMOTE_HEAD "):
            out["head"] = line.split()[1]
        elif line.startswith("RESULT_DECODE_TPS "):
            try:
                out["decode_tps"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_PREFILL128_PP "):
            try:
                out["prefill128_pp"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_TOP1 "):
            try:
                out["top1"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_KL "):
            try:
                out["kl"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_PPL_SPARK "):
            try:
                out["ppl_spark"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_PPL_LLAMA "):
            try:
                out["ppl_llama"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("TOKEN_COUNT "):
            try:
                out["token_count"] = int(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("MUSE "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    muse[int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.strip() == "MUSE_FAILED":
            out["muse_failed"] = True
        elif line.startswith("GUARD36 "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    guard36[int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.strip() == "GUARD36_FAILED":
            out["guard36_failed"] = True
        elif line.startswith("GUARDMO "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    guardmo[int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.strip() == "GUARDMO_FAILED":
            out["guardmo_failed"] = True
        elif line.strip() == "GUARDMO_UNAVAILABLE":
            out["guardmo_unavailable"] = True
    out["guard36"] = guard36
    out["guardmo"] = guardmo
    out["muse"] = muse
    return out


def push_eval_polaris(host, port):
    """Sync eval/polaris/ (judge.py + receipt.py) from a TRUSTED source (origin/main) before
    running attestation — same protection as pr_dflash_bot.py's push_eval_polaris (a PR's own
    checkout must never supply the code that produces its own attestation)."""
    use_local = os.environ.get("SPARKINFER_USE_LOCAL_POLARIS", "").strip().lower() in ("1", "true", "yes")
    tar_data = None
    source = "local checkout"
    extract_root = os.path.join(REMOTE_REPO, "eval")
    if not use_local:
        subprocess.run(["git", "fetch", "-q", "origin", "main"], cwd=ROOT, capture_output=True, timeout=120)
        arch = subprocess.run(
            ["git", "archive", "--format=tar.gz", "origin/main", "eval/polaris"],
            cwd=ROOT, capture_output=True, timeout=120,
        )
        if arch.returncode == 0 and arch.stdout:
            tar_data = arch.stdout
            source = "origin/main"
            extract_root = REMOTE_REPO
        else:
            print(">> WARN: origin/main eval/polaris fetch failed — refusing to fall back to the "
                  "local working tree for a real run — attestation unavailable this run")
            return False
    else:
        polaris_dir = os.path.join(HERE, "polaris")
        if os.path.isdir(polaris_dir):
            tar = subprocess.run(["tar", "-C", HERE, "-czf", "-", "polaris"],
                                  capture_output=True, timeout=120)
            if tar.returncode == 0 and tar.stdout:
                tar_data = tar.stdout
                source = "local checkout"
                extract_root = os.path.join(REMOTE_REPO, "eval")
    if tar_data is None:
        print(">> WARN: no eval/polaris archive — attestation unavailable this run")
        return False
    key = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/speedy"))
    user = ssh_box_user() if ssh_box_enabled() else "root"
    import tempfile
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".tgz", delete=False) as tmp:
            tmp.write(tar_data)
            tmp_path = tmp.name
        scp = subprocess.run(
            ["scp", "-P", str(port), "-i", key, "-o", "IdentitiesOnly=yes",
             "-o", "StrictHostKeyChecking=accept-new",
             "-o", "BatchMode=yes", tmp_path, f"{user}@{host}:/tmp/si_polaris_mg.tgz"],
            capture_output=True, text=True, timeout=120,
        )
        if scp.returncode != 0:
            print(f">> WARN: eval/polaris scp failed (rc={scp.returncode}): {scp.stderr[-500:]}")
            return False
        extract = (
            f"mkdir -p {shlex.quote(extract_root)} && "
            f"tar -xzf /tmp/si_polaris_mg.tgz -C {shlex.quote(extract_root)} && "
            "rm -f /tmp/si_polaris_mg.tgz"
        )
        r = ssh_run(host, port, extract, timeout=60)
        if r.returncode != 0:
            print(f">> WARN: eval/polaris extract failed (rc={r.returncode}): {(r.stdout + r.stderr)[-500:]}")
            return False
        print(f">> eval/polaris synced from {source} (trusted attestation code)")
        return True
    except subprocess.TimeoutExpired:
        print(">> WARN: eval/polaris sync timed out — attestation unavailable this run")
        return False
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def collect_polaris_attestation(host, port, res: dict, pr_ref: str):
    """Run judge.py --from-stdin (a RESULT_JSON-prefixed line, the AR bot's own generic single-
    model shape) to assemble+sign an attestation. NOT --dflash — that mode hardcodes a
    DFlash-shaped measurement block and eval_mode="dflash", which would be a factually wrong
    label for a plain-AR Muse Glimmer eval. Never raises — a Polaris failure must not block the
    verdict itself, only omit its receipt."""
    if not POLARIS_ENABLED:
        return None
    checkout_cmd = (
        f"cd {shlex.quote(REMOTE_REPO)} && "
        f"git fetch -q origin {shlex.quote(pr_ref)} && git checkout -qf FETCH_HEAD"
    )
    r0 = ssh_run(host, port, checkout_cmd, timeout=60)
    if r0.returncode != 0:
        print(f">> Polaris: could not checkout {pr_ref} for attestation (rc={r0.returncode}): "
              f"{(r0.stderr or '')[-300:]}")
        return None
    if not push_eval_polaris(host, port):
        return None
    result_json = {
        "model": "museglimmer-128",
        "label": res.get("label"),
        "pass": res.get("pass"),
        "tps": res.get("pr_decode_tps"),
        "delta_tps": (res.get("pr_decode_tps") or 0) - (res.get("main_decode_tps") or 0),
        "pct_over_frontier": res.get("delta_pct"),
        "score_context": 128,
        "best_context_label": "128",
        "ctx_128_tps": res.get("pr_decode_tps"),
        "top1": res.get("pr_top1"),
        "kl": res.get("pr_kl"),
    }
    eval_seed = f"museglimmer-{int(time.time() * 1000)}"  # unique nonce per attestation
    stdin_payload = "RESULT_JSON " + json.dumps(result_json)
    cmd = (
        f"cd {shlex.quote(REMOTE_REPO)} && "
        f"SPARKINFER_EVAL_MODE=museglimmer-128 SPARKINFER_DECODE_TOKENS={BENCH_TOKENS} "
        f"SPARKINFER_EVAL_SEED={shlex.quote(eval_seed)} python3 eval/polaris/judge.py --from-stdin "
        f"--model-file {shlex.quote(DEFAULT_GGUF)} "
        f"--build-dir {shlex.quote(REMOTE_REPO)}/build/runtime "
        f"--sparkinfer-root {shlex.quote(REMOTE_REPO)}"
    )
    try:
        r = ssh_run(host, port, cmd, timeout=60, stdin_data=stdin_payload)
    except Exception as e:
        print(f">> Polaris judge SSH failed: {e}")
        return None
    if r.returncode != 0:
        print(f">> Polaris judge failed (rc={r.returncode}): {(r.stderr or '')[-500:]}")
        return None
    polaris_line = next((l for l in (r.stdout or "").splitlines()
                         if l.startswith("POLARIS_ATTESTATION ")), None)
    if not polaris_line:
        print(">> Polaris judge produced no attestation")
        return None
    try:
        attestation = json.loads(polaris_line[len("POLARIS_ATTESTATION "):])
    except json.JSONDecodeError as e:
        print(f">> Polaris attestation JSON parse failed: {e}")
        return None
    privkey = arb._load_polaris_privkey()
    if not POLARIS_API_KEY and not privkey:
        print(">> Polaris: attestation collected but NOT signed (no key configured)")
        return {"attestation": attestation}
    try:
        receipt = arb.build_polaris_receipt_from_attestation(
            attestation, api_key=POLARIS_API_KEY, privkey=privkey, pubkey=_load_polaris_pubkey(),
        )
        return {"attestation": attestation, "receipt": receipt}
    except Exception as e:
        print(f">> Polaris signing failed: {e}")
        return {"attestation": attestation}


def measure_main_baseline(host, port):
    """Measure main's decode+accuracy+Qwen3.6-guard baseline ONCE per round, not once per PR —
    main's code can't change mid-round (the only merge in this bot's own flow is
    try_auto_merge_museglimmer, called from reconcile_museglimmer_merge_labels AFTER every
    pending PR has already been individually evaluated in main()'s loop, see call ordering
    there), so every PR in the round comparing against a freshly-remeasured main was pure
    redundant GPU/build time. Returns {"ok": True, **parsed} or {"ok": False, "reason", "log"}."""
    r = _ssh_run_resilient(host, port, _remote_script("main"), "main run")
    if r.returncode != 0:
        tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-2000:]
        crash = _crash_reason(r.stdout, r.stderr)
        reason = "main run failed" + (f" — {crash}" if crash else " (no crash diagnostic captured, possible hard kill — retried once)")
        return {"ok": False, "reason": reason, "log": tail}
    main = _parse_remote(r.stdout or "")
    # Fail the ROUND, not the PR, when the baseline is incomplete: comparing a PR against a
    # partial baseline silently turns a missing context into a "regression".
    missing = [SCORED_CTX_LABEL[c] for c in SCORED_CTXS
               if not ((main.get("muse") or {}).get(c) or {}).get("decode")]
    if main.get("muse_failed") or missing:
        return {"ok": False,
                "reason": "main bench missing Muse Glimmer measurements at ctx " + ",".join(missing or ["(sweep failed)"]),
                "log": (r.stdout or "")[-1500:]}
    main["ok"] = True
    return main


def eval_museglimmer_on_box(host, port, pr_ref: str, main: dict):
    """Run the PR ref's speed+accuracy script on the same box and compare against `main`, an
    already-measured baseline shared across every PR in the round (see measure_main_baseline)."""
    print(f">> Muse Glimmer eval on box: PR ref={pr_ref}")
    r = _ssh_run_resilient(host, port, _remote_script(pr_ref), "PR run")
    if r.returncode != 0:
        tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-2000:]
        crash = _crash_reason(r.stdout, r.stderr)
        reason = "PR speed/accuracy run failed" + (f" — {crash}" if crash else " (no crash diagnostic captured, possible hard kill — retried once)")
        return {"ok": False, "reason": reason, "log": tail}
    pr = _parse_remote(r.stdout or "")
    # A missing PR-side context is NOT treated as an infra failure here -- it is scored as a
    # regression by tier_from_gain (cur=0 against a real main baseline), which is the fail-closed
    # direction. Only a wholesale sweep failure is reported as a run failure.
    if pr.get("muse_failed") or not (pr.get("muse") or {}):
        return {"ok": False, "reason": "PR bench produced no Muse Glimmer measurements",
                "log": (r.stdout or "")[-1500:]}
    if "top1" not in pr or "kl" not in pr:
        return {"ok": False, "reason": "PR run missing accuracy METRIC line", "log": (r.stdout or "")[-1500:]}
    for ctx in SCORED_CTXS:
        pv = (pr.get("muse") or {}).get(ctx) or {}
        mv = (main.get("muse") or {}).get(ctx) or {}
        print(f">> PR @{SCORED_CTX_LABEL[ctx]:>4}: decode {pv.get('decode', 0):9.2f} "
              f"(main {mv.get('decode', 0):9.2f})   prefill {pv.get('prefill', 0):10.2f} "
              f"(main {mv.get('prefill', 0):10.2f})")
    print(f">> PR accuracy top1={pr.get('top1', 0):.3f} kl={pr.get('kl', 99):.4f}")

    # Ten scored axes: decode and prefill at each of SCORED_CTXS. Every one is ALSO a
    # no-regression floor -- any single axis regressing is a hard REJECT, so a PR cannot buy a
    # headline win at one context by giving away another. Otherwise the tier is the best measured
    # delta across the whole set, which preserves the previous behaviour that a PR improving one
    # phase with the other merely flat still earns credit for the real improvement it made.
    #
    # Iterating SCORED_CTXS rather than restating the contexts keeps this in step with the
    # constant; the DSpark bot's equivalent comment once named a dimension its list did not hold.
    scored = []
    for ctx in SCORED_CTXS:
        clabel = SCORED_CTX_LABEL[ctx]
        pr_vals = (pr.get("muse") or {}).get(ctx) or {}
        main_vals = (main.get("muse") or {}).get(ctx) or {}
        for phase, unit in (("decode", "decode"), ("prefill", "prefill")):
            name = f"muse-{unit}@{clabel}"
            lab, dlt, ok, why = tier_from_gain(
                pr_vals.get(phase, 0.0), main_vals.get(phase, 0.0), metric=name)
            scored.append({"dim": name, "label": lab, "delta": dlt, "passed": ok, "reason": why})
    by_dim = {x["dim"]: x for x in scored}

    regressed = [x for x in scored if x["label"] == "REJECT"]
    if regressed:
        worst = min(regressed, key=lambda x: x["delta"])
        label, delta_pct, passed = "REJECT", worst["delta"], False
        speed_reason = " | ".join(x["reason"] for x in regressed)
    else:
        # max() over measured deltas, not over tier letters: two axes can share a bucket while
        # one is clearly the larger win, and the delta is what the tier came from anyway.
        best = max((by_dim[d] for d in SCORING_DIMS if d in by_dim),
                   key=lambda x: x["delta"], default=by_dim[SCORING_DIM])
        label, delta_pct, passed, speed_reason = (
            best["label"], best["delta"], best["passed"], best["reason"])

    # Back-compat names for the report/dashboard/Polaris payload, taken from the 128 axes that
    # used to be the only two dimensions this bot had.
    decode_delta_pct = by_dim["muse-decode@128"]["delta"]
    decode_label = by_dim["muse-decode@128"]["label"]
    prefill_delta_pct = by_dim["muse-prefill@128"]["delta"]
    prefill_label = by_dim["muse-prefill@128"]["label"]

    pr_top1 = pr.get("top1", 0.0)
    pr_kl = pr.get("kl", 99.0)
    accuracy_ok = pr_top1 >= ACC_TOP1_BAR and pr_kl <= ACC_KL_BAR
    reason = speed_reason
    if not accuracy_ok:
        # Accuracy gate is a hard REJECT regardless of speed — same discipline as
        # pr_dflash_bot.py's SPEC_AGREE veto: a fast-but-wrong PR is worthless on a still-fragile
        # architecture, and speed alone can't prove correctness.
        acc_reason = (f"accuracy gate failed: top1={pr_top1:.3f} (bar >={ACC_TOP1_BAR}) "
                      f"kl={pr_kl:.4f} (bar <={ACC_KL_BAR})")
        reason = f"{acc_reason} | speed: {speed_reason}"
        label = "REJECT"
        passed = False

    mo_ok, mo_problems = check_modelopt_guard(pr, main)
    if pr.get("guardmo_unavailable") or main.get("guardmo_unavailable"):
        # Absent checkpoint is a SKIP, not a REJECT -- but say so, so a round that guarded
        # nothing never reads as a round that guarded successfully.
        mo_ok, mo_problems = True, []
        print(">> modelopt guard SKIPPED — checkpoint not installed (MODELOPT_MODEL_DIR)")
    if not mo_ok:
        mo_reason = "modelopt no-regression guard failed: " + "; ".join(mo_problems[:6])
        reason = f"{mo_reason} | {reason}"
        label = "REJECT"
        passed = False

    q36_ok, q36_problems = check_q36_guard(pr, main)
    if not q36_ok:
        # Same hard-REJECT discipline as the accuracy gate: a Muse Glimmer PR that silently
        # regresses Qwen3.6 via shared code (qwen35.cpp/inference_engine.cpp) is unmergeable
        # regardless of its own speed/accuracy result — see module docstring pt. 3.
        q36_reason = "qwen3.6 no-regression guard failed: " + "; ".join(q36_problems[:6])
        reason = f"{q36_reason} | {reason}"
        label = "REJECT"
        passed = False

    res = {
        "ok": True,
        "label": label,
        "pass": passed and label != "REJECT",
        "reason": reason,
        "delta_pct": delta_pct,
        "pr_decode_tps": _at(pr, 128, "decode"),
        "main_decode_tps": _at(main, 128, "decode"),
        "decode_delta_pct": decode_delta_pct,
        "decode_regressed": decode_label == "REJECT",
        "speedup_vs_main": (round(_at(pr, 128, "decode") / _at(main, 128, "decode"), 3)
                            if _at(main, 128, "decode") else 0),
        "pr_prefill128_pp": _at(pr, 128, "prefill"),
        "main_prefill128_pp": _at(main, 128, "prefill"),
        "prefill_delta_pct": prefill_delta_pct,
        "prefill_regressed": prefill_label == "REJECT",
        "prefill_speedup_vs_main": (round(_at(pr, 128, "prefill") / _at(main, 128, "prefill"), 3)
                                    if _at(main, 128, "prefill") else 0),
        # Every scored axis, so the PR comment and the published log can show the whole matrix
        # rather than just the tier-winning row.
        "scored_dims": scored,
        "best_dim": (best["dim"] if not regressed else worst["dim"]),
        "modelopt_guard_ok": mo_ok,
        "modelopt_guard_problems": mo_problems,
        "muse_pr": (pr.get("muse") or {}),
        "muse_main": (main.get("muse") or {}),
        "guardmo_skipped": bool(pr.get("guardmo_unavailable") or main.get("guardmo_unavailable")),
        "pr_top1": pr_top1,
        "pr_kl": pr_kl,
        "pr_ppl_spark": pr.get("ppl_spark"),
        "pr_ppl_llama": pr.get("ppl_llama"),
        "main_top1": main.get("top1"),
        "main_kl": main.get("kl"),
        "accuracy_ok": accuracy_ok,
        "q36_guard_ok": q36_ok,
        "q36_guard_problems": q36_problems,
        "q36_guard": pr.get("guard36"),
        "q36_guard_main": main.get("guard36"),
        "pr_head": pr.get("head"),
        "main_head": main.get("head"),
    }
    polaris = collect_polaris_attestation(host, port, res, pr_ref)
    if polaris:
        res["polaris"] = polaris
    return res


def _ctx_list_str() -> str:
    """"128/512/4k/16k/32k/64k" from SCORED_CTXS. Derived rather than written out: the PR comment
    twice named a context list that had gone stale behind the constant."""
    return "/".join(SCORED_CTX_LABEL[c] for c in SCORED_CTXS)


def _matrix_table(res: dict) -> str:
    """The full PR-vs-main matrix. Rendered from res["scored_dims"] rather than from named keys so
    it stays correct when SCORED_CTXS changes -- the previous comment hard-coded two rows and would
    have silently kept showing two after the matrix grew to ten."""
    dims = res.get("scored_dims") or []
    if not dims:
        return ""
    pr_m, main_m = res.get("muse_pr") or {}, res.get("muse_main") or {}
    rows = ["| ctx | phase | main | PR | delta |", "|---|---|---|---|---|"]
    for ctx in SCORED_CTXS:
        for phase in ("decode", "prefill"):
            name = f"muse-{phase}@{SCORED_CTX_LABEL[ctx]}"
            d = next((x for x in dims if x["dim"] == name), None)
            if not d:
                continue
            mv = float((main_m.get(ctx) or {}).get(phase, 0) or 0)
            pv = float((pr_m.get(ctx) or {}).get(phase, 0) or 0)
            flag = " **REJECT**" if d["label"] == "REJECT" else ""
            rows.append(f"| {SCORED_CTX_LABEL[ctx]} | {phase} | {mv:.2f} | {pv:.2f} | "
                        f"{d['delta']:+.1f}%{flag} |")
    return "\n".join(rows) + "\n\n"


def format_comment(commit: str, res: dict) -> str:
    meta = {
        "label": res.get("label"),
        "delta_pct": res.get("delta_pct"),
        "pr_decode_tps": res.get("pr_decode_tps"),
        "main_decode_tps": res.get("main_decode_tps"),
        "pr_prefill128_pp": res.get("pr_prefill128_pp"),
        "main_prefill128_pp": res.get("main_prefill128_pp"),
        "pass": res.get("pass"),
        "accuracy_ok": res.get("accuracy_ok"),
        "q36_guard_ok": res.get("q36_guard_ok"),
        "modelopt_guard_ok": res.get("modelopt_guard_ok"),
        "modelopt_guard_skipped": res.get("guardmo_skipped"),
        # WHICH axis produced delta_pct. Necessary now that the tier comes from ten axes while the
        # marker still carries only the 128 numbers for the dashboard: without this a reader sees
        # a headline delta that does not match either number next to it (e.g. +3900% from
        # prefill@4k printed beside a flat decode@128).
        "best_dim": res.get("best_dim"),
        # The whole matrix, so a consumer that wants more than the 128 pair does not have to
        # re-scrape the rendered table.
        "dims": {d["dim"]: {"delta": d["delta"], "label": d["label"]}
                 for d in (res.get("scored_dims") or [])},
    }
    marker = (
        f"<!-- sparkinfer-museglimmer-eval:{EVAL_SCHEMA_VERSION}:{commit} "
        f"{json.dumps(meta, separators=(',', ':'))} -->"
    )
    if not res.get("ok"):
        return (
            f"{marker}\n## sparkinfer museglimmer auto-eval — error\n\n"
            f"**reason:** `{res.get('reason')}`\n\n"
            f"<details><summary>log tail</summary>\n\n```\n{(res.get('log') or '')[:1800]}\n```\n</details>\n"
        )
    lab = res["label"]
    if res.get("accuracy_ok"):
        acc_row = (f"| accuracy gate | ✅ top1={res.get('pr_top1', 0):.3f} "
                    f"(bar >={ACC_TOP1_BAR}) · KL={res.get('pr_kl', 0):.4f} (bar <={ACC_KL_BAR}) |\n")
    else:
        acc_row = (f"| accuracy gate | ❌ **FAILED** — top1={res.get('pr_top1', 0):.3f} "
                    f"(bar >={ACC_TOP1_BAR}) · KL={res.get('pr_kl', 0):.4f} (bar <={ACC_KL_BAR}) — "
                    "**verdict forced to REJECT regardless of speed** |\n")
    main_acc_note = ""
    if res.get("main_top1") is not None and (res.get("main_top1", 1) < ACC_TOP1_BAR or (res.get("main_kl") or 0) > ACC_KL_BAR):
        main_acc_note = (f"| ⚠️ same-box main accuracy | top1={res.get('main_top1'):.3f} "
                          f"kl={res.get('main_kl'):.4f} — main ALSO misses the bar (informational; "
                          "not gated on main, but check the box/corpus if this persists) |\n")
    if res.get("q36_guard_ok"):
        q36_row = "| qwen3.6 guard | ✅ no regression (decode+prefill @ 32k) |\n"
    else:
        problems = "; ".join((res.get("q36_guard_problems") or [])[:4])
        q36_row = (f"| qwen3.6 guard | ❌ **FAILED** — {problems} — "
                    "**verdict forced to REJECT regardless of speed/accuracy** |\n")
    if res.get("guardmo_skipped"):
        # Say SKIPPED explicitly. A guard that silently reports nothing reads identical to one
        # that passed, which is how an unnoticed vacuous guard survives for months.
        mo_row = ("| modelopt guard | ⚠️ SKIPPED — checkpoint not installed on the box "
                  "(`MODELOPT_MODEL_DIR`); shared-code regressions on Qwen3.8 were NOT checked |\n")
    elif res.get("modelopt_guard_ok"):
        mo_row = "| modelopt guard | ✅ no regression (decode+prefill @ 32k, Qwen3.8-27B NVFP4) |\n"
    else:
        mo_problems = "; ".join((res.get("modelopt_guard_problems") or [])[:4])
        mo_row = (f"| modelopt guard | ❌ **FAILED** — {mo_problems} — "
                  "**verdict forced to REJECT regardless of speed/accuracy** |\n")
    polaris = res.get("polaris") or {}
    receipt = polaris.get("receipt")
    if receipt:
        rtype = "TDX (Intel hardware attestation)" if receipt.get("attestation_type") == "tdx-quote" \
            else "Ed25519 (SparkInfer key)"
        polaris_row = f"| Polaris receipt | `{receipt.get('receipt_id', '?')[:16]}…` — {rtype} |\n"
    elif polaris.get("attestation"):
        polaris_row = "| Polaris receipt | collected, not signed (no key configured) |\n"
    else:
        polaris_row = ""
    return (
        f"{marker}\n## sparkinfer museglimmer auto-eval — `eval-museglimmer:{lab}`\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| **label** | `eval-museglimmer:{lab}` |\n"
        f"| scored at | decode + prefill @ {_ctx_list_str()} — every axis is also a regression floor, label is the best |\n"
        f"| tier from | `{res.get('best_dim') or '?'}` ({res.get('delta_pct', 0):+.1f}%) |\n"
        f"{acc_row}"
        f"{main_acc_note}"
        f"{q36_row}"
        f"{mo_row}"
        f"| PPL sparkinfer / llama.cpp | {res.get('pr_ppl_spark') or '?'} / {res.get('pr_ppl_llama') or '?'} |\n"
        f"{polaris_row}"
        f"| commit | `{commit[:9]}` |\n\n"
        f"{_matrix_table(res)}"
        f"{res.get('reason') or ''}\n\n"
        f"<sub>Scored on the pinned eval box vs same-box `origin/main` — AR decode AND prefill at "
        f"ctx {_ctx_list_str()}; ANY axis regressing is a hard REJECT, but "
        f"otherwise the reported label is the **best** measured delta across the "
        f"{len(SCORING_DIMS)} — "
        "a PR that improves just one, with the rest flat, still earns credit for that. "
        "Cross-model no-regression guards run at 32k on Qwen3.6 and the ModelOpt Qwen3.8-27B "
        "NVFP4 checkpoint. This is informational, not a judgment on your PR: a `none` label just "
        "means no measurable Muse Glimmer speedup was verified on either metric, which is expected "
        "and fine if that isn't what your change is about. "
        "Correctness gated against a live llama.cpp reference on the same GGUF. Also gated on two "
        "cross-model no-regression guards at 32k (decode+prefill, same box vs main): "
        "Qwen3.6-35B-A3B and the ModelOpt Qwen3.8-27B NVFP4 checkpoint — "
        "Muse Glimmer PRs can touch code shared with other models. "
        "Automated — **not merged**; merge manually after review.</sub>\n"
    )


def auto_merge_ok_museglimmer(repo, num):
    info = json.loads(arb.gh([
        "pr", "view", str(num), "-R", repo, "--json",
        "state,isDraft,labels,author,mergeable,files",
    ]).stdout or "{}")
    if info.get("state") != "OPEN" or info.get("isDraft"):
        return False, "not an open, non-draft PR"
    labs = {l["name"] for l in info.get("labels", [])}
    tiers = {l.split(":", 1)[1] for l in labs if l.startswith(EVAL_PREFIX)}
    if not (tiers & SPEEDUP_LABELS):
        return False, "no verified eval-museglimmer:speedup label"
    if MUSEGLIMMER_MERGE_FIRST not in labs:
        return False, "not museglimmer-merge-first"
    blocked = labs & AUTOMERGE_BLOCK
    if blocked:
        return False, f"blocking label(s): {', '.join(sorted(blocked))}"
    author = (info.get("author") or {}).get("login", "")
    if author.lower() in arb.load_denylist():
        return False, f"author {author} is blocked"
    if arb.author_penalty_until(author):
        return False, f"author {author} is under penalty"
    sens = [f["path"] for f in info.get("files", [])
            if any(f["path"].startswith(p) for p in arb.AUTOMERGE_SENSITIVE)]
    if sens:
        return False, f"touches protected paths: {', '.join(sens[:3])}"
    if arb.pr_merge_conflict(info.get("mergeable")):
        return False, "merge conflict with base"
    if info.get("mergeable") != "MERGEABLE":
        return False, f"not cleanly mergeable ({info.get('mergeable')})"
    return True, "ok"


def try_auto_merge_museglimmer(repo, num):
    ok, reason = auto_merge_ok_museglimmer(repo, num)
    if not ok:
        print(f">> museglimmer auto-merge SKIP #{num}: {reason}")
        return False
    r = arb.gh(["pr", "merge", str(num), "-R", repo, "--squash"])
    if r.returncode != 0 and os.environ.get("SPARKINFER_AUTOMERGE_ADMIN", "1") == "1":
        err = ((r.stderr or "") + (r.stdout or "")).lower()
        if "not mergeable" in err or "branch policy" in err or "required" in err or "prohibited" in err:
            print(">> museglimmer auto-merge: branch policy blocked — retrying with --admin")
            r = arb.gh(["pr", "merge", str(num), "-R", repo, "--squash", "--admin"])
    if r.returncode == 0:
        print(f">> MUSEGLIMMER AUTO-MERGED #{num} (museglimmer-merge-first)")
        arb.gh(["pr", "comment", str(num), "-R", repo, "--body",
                "<!-- sparkinfer-museglimmer-automerge -->\n"
                "Auto-merged as the round's `museglimmer-merge-first` winner — verified same-box "
                "128-token decode speedup over `main`, accuracy-gated vs llama.cpp."])
        return True
    print(f">> museglimmer auto-merge BLOCKED #{num}: {(r.stderr or r.stdout or '')[:200]}")
    return False


def reconcile_museglimmer_merge_labels(repo, dry_run=False):
    scores = _load_scores()
    open_prs = json.loads(arb.gh([
        "pr", "list", "-R", repo, "--state", "open",
        "--json", "number,labels", "--limit", "80",
    ]).stdout or "[]")
    open_labels = {p["number"]: {l["name"] for l in p["labels"]} for p in open_prs}

    merged = json.loads(arb.gh([
        "pr", "list", "-R", repo, "--state", "merged", "--label", MUSEGLIMMER_MERGE_FIRST,
        "--json", "number", "--limit", "10",
    ]).stdout or "[]")
    for m in merged:
        if not dry_run:
            arb.remove_label(repo, m["number"], MUSEGLIMMER_MERGE_FIRST)

    scored = []
    for num, labs in open_labels.items():
        if MUSEGLIMMER_NEEDS_REBASE in labs:
            continue
        tiers = {l.split(":", 1)[1] for l in labs if l.startswith(EVAL_PREFIX)}
        tier = next((t for t in tiers if t in SPEEDUP_LABELS), None)
        if not tier:
            continue
        entry = scores.get(str(num)) or {}
        if entry.get("label") not in SPEEDUP_LABELS:
            if tier not in SPEEDUP_LABELS:
                continue
            entry = {"label": tier, "delta_pct": entry.get("delta_pct") or 0}
        scored.append((num, float(entry.get("delta_pct") or 0), entry.get("label") or tier))

    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        print(">> museglimmer round: no verified speedup PRs")
        return
    winner = scored[0][0]
    print(f">> museglimmer round: merge-first #{winner}; rebase {[n for n,_,_ in scored[1:]] or 'none'}")
    if dry_run:
        return
    arb.add_label(repo, winner, MUSEGLIMMER_MERGE_FIRST)
    arb.remove_label(repo, winner, MUSEGLIMMER_NEEDS_REBASE)
    for num, _, _ in scored[1:]:
        arb.add_label(repo, num, MUSEGLIMMER_NEEDS_REBASE)
        arb.remove_label(repo, num, MUSEGLIMMER_MERGE_FIRST)
    if AUTO_MERGE:
        try_auto_merge_museglimmer(repo, winner)


def upload_museglimmer_eval_log(repo, num, title, oid, res):
    """Commit the eval result (+ Polaris receipt/attestation) to sparkinfer-log, mirroring
    pr_dflash_bot.py's upload_dflash_eval_log with a museglimmer-prefixed run id."""
    try:
        rid = f"museglimmer-{int(num):04d}-{oid[:7]}"
        arb._ensure_log_repo()
        rundir = os.path.join(arb.LOG_DIR, "runs", rid)
        os.makedirs(rundir, exist_ok=True)
        polaris = res.get("polaris") or {}
        receipt = polaris.get("receipt")
        result = {
            "id": rid, "pr": int(num), "title": title,
            "url": f"https://github.com/{repo}/pull/{num}", "commit": oid[:7],
            "eval_mode": "museglimmer-128",
            "label": res.get("label"), "pass": res.get("pass"), "reason": res.get("reason"),
            "delta_pct": res.get("delta_pct"),
            "pr_decode_tps": res.get("pr_decode_tps"), "main_decode_tps": res.get("main_decode_tps"),
            "speedup_vs_main": res.get("speedup_vs_main"),
            "pr_prefill128_pp": res.get("pr_prefill128_pp"), "main_prefill128_pp": res.get("main_prefill128_pp"),
            "prefill_speedup_vs_main": res.get("prefill_speedup_vs_main"),
            "pr_top1": res.get("pr_top1"), "pr_kl": res.get("pr_kl"),
            "accuracy_ok": res.get("accuracy_ok"),
            "q36_guard_ok": res.get("q36_guard_ok"), "q36_guard_problems": res.get("q36_guard_problems"),
            "gpu": "pinned eval box", "date": arb.datetime.date.today().isoformat(),
        }
        if receipt:
            result["polaris"] = True
            result["polaris_receipt_id"] = receipt.get("receipt_id")
        json.dump(result, open(os.path.join(rundir, "result.json"), "w"), indent=2)
        if polaris.get("attestation"):
            json.dump(polaris["attestation"], open(os.path.join(rundir, "attestation.json"), "w"), indent=2)
        if receipt:
            json.dump(receipt, open(os.path.join(rundir, "receipt.json"), "w"), indent=2)
        ipath = os.path.join(arb.LOG_DIR, "index.json")
        idx = json.load(open(ipath)) if os.path.exists(ipath) else []
        idx = [e for e in idx if e.get("id") != rid]
        idx_entry = {"id": rid, "pr": int(num), "title": title, "label": res.get("label"),
                     "delta_pct": res.get("delta_pct"), "eval_mode": "museglimmer-128", "date": result["date"]}
        if receipt:
            idx_entry["polaris"] = True
            idx_entry["polaris_receipt_id"] = receipt.get("receipt_id", "")[:16]
        idx.append(idx_entry)
        idx.sort(key=lambda x: x["id"])
        json.dump(idx, open(ipath, "w"), indent=2)
        subprocess.run(["git", "-C", arb.LOG_DIR, "add", "-A"], check=True)
        msg = f"museglimmer-eval: #{num} {oid[:7]} -> eval-museglimmer:{res.get('label')}"
        if receipt:
            msg += f" + polaris {receipt.get('receipt_id', '?')[:16]}"
        commit = subprocess.run(["git", "-C", arb.LOG_DIR, "commit", "-q", "-m", msg], check=False)
        if commit.returncode != 0:
            print(">> museglimmer eval-log upload skipped: nothing to commit")
            return None
        push = subprocess.run(["git", "-C", arb.LOG_DIR, "push", "-q"], check=False)
        if push.returncode != 0:
            print(f">> museglimmer eval-log push failed (rc={push.returncode})")
            return None
        url = arb.LOG_PAGE + rid
        print(f">> museglimmer eval log: {url}")
        return url
    except Exception as e:
        print(f">> museglimmer eval-log upload failed: {e}")
        return None


def apply_result(repo, num, commit, res, title="", dry_run=False):
    body = format_comment(commit, res)
    label = res.get("label") if res.get("ok") else "REJECT"
    if not res.get("ok"):
        label = "REJECT"
    print(f"PR #{num}: eval-museglimmer:{label}  "
          f"decode PR={res.get('pr_decode_tps')} main={res.get('main_decode_tps')}  "
          f"prefill128 PR={res.get('pr_prefill128_pp')} main={res.get('main_prefill128_pp')}  "
          f"delta={res.get('delta_pct')}%  accuracy_ok={res.get('accuracy_ok')}  "
          f"q36_guard_ok={res.get('q36_guard_ok')}")
    if dry_run:
        print(body[:500])
        return
    strip_museglimmer_eval_labels(repo, num)
    if label in SPEEDUP_LABELS:
        # A fresh, valid speedup score for the CURRENT head commit means this PR is caught up
        # with main and deserves a fair shot at winning the next merge-first reconciliation --
        # clear any stale needs-rebase from a round it lost (or an old conflict that's since been
        # resolved). Found 2026-08-13: reconcile_museglimmer_merge_labels() filters candidates on
        # `MUSEGLIMMER_NEEDS_REBASE not in labs` (this bot's own "who's eligible to win" gate) but
        # the ONLY place that ever removed the label was the winner-selection branch itself --
        # a PR that lost one round, or ever hit a transient merge conflict, could never be
        # reconsidered again even after a completely clean re-evaluation confirmed its score,
        # since it was filtered out of candidacy before scoring was ever compared. Hit #790 and
        # #791 both losing merge-first to a strictly worse score for exactly this reason.
        arb.remove_label(repo, num, MUSEGLIMMER_NEEDS_REBASE)
    arb.add_label(repo, num, f"{EVAL_PREFIX}{label}")
    # Mirrored to the generic `eval:*` label, same as pr_dflash_bot.py -- SN74 scoring reads
    # eval:* tiers, so this makes Muse Glimmer submissions count toward that live incentive
    # mechanism. Explicit user decision, 2026-08-11 (originally deliberately NOT mirrored, given
    # Muse Glimmer's youth at the time -- see git history on this line for that reasoning).
    # Derived from every per-bot `eval-<model>:<tier>` label rather than overwritten with this
    # bot's own verdict -- this bot only measures Muse Glimmer, so writing the generic label
    # directly let a `none` here erase another model's real tier depending purely on which
    # staggered cron ran last. See arb.sync_generic_eval_label().
    arb.sync_generic_eval_label(repo, num)
    arb.gh(["pr", "comment", str(num), "-R", repo, "--body", body])
    if res.get("ok"):
        upload_museglimmer_eval_log(repo, num, title, commit, res)
    if res.get("ok") and res.get("delta_pct") is not None:
        scores = _load_scores()
        scores[str(num)] = {
            "commit": commit,
            "label": label,
            "delta_pct": res.get("delta_pct"),
            "pr_decode_tps": res.get("pr_decode_tps"),
            "main_decode_tps": res.get("main_decode_tps"),
            "pr_prefill128_pp": res.get("pr_prefill128_pp"),
            "main_prefill128_pp": res.get("main_prefill128_pp"),
            "pass": res.get("pass"),
            "accuracy_ok": res.get("accuracy_ok"),
            "q36_guard_ok": res.get("q36_guard_ok"),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _save_scores(scores)
        # Auto-close on "REJECT" ONLY (narrowed 2026-09-09; was none/REJECT). Re-enabled 2026-08-11
        # after an explicit, informed decision: the very first supervised run of this bot closed a
        # real external contributor's unrelated PR (#768) this exact same way, since
        # arb.greenlight_status() is generic (any PR with a checked "tested" box + a decode/
        # prefill before/after table) and matches essentially any performance PR in the repo, not
        # just Muse-Glimmer-relevant ones -- "none" is the expected, non-judgmental outcome for
        # most PRs this bot evaluates, not a rejection of the PR's actual purpose. That PR was
        # reopened + apologized for. The user was told this risk explicitly and chose to accept it
        # (broad scope, matching pr_dflash_bot.py) rather than narrow evaluation to only
        # Muse-Glimmer-relevant PRs. If this causes another wrongful close, reopen + apologize the
        # same way, and reconsider the scope-narrowing alternative that was declined here.
        # AUTO-CLOSE ONLY ON REJECT, NEVER ON "none" (changed 2026-09-09).
        #
        # "none" means THIS bot measured no change on ITS axes. For a PR aimed at a different
        # model that is the expected, uninformative outcome -- not a verdict on the PR. Closing on
        # it destroys good work: this bot is currently the only one on cron, so every PR in the
        # repo is scored against one model's metrics, and a genuine improvement to another model
        # measures "none" here by construction. PR #1008 (a 15x Muse prefill win) was minutes away
        # from being auto-closed by the DSpark bot for exactly this reason and had to be caught by
        # hand; the same trap now points the other way.
        #
        # "REJECT" is different in kind and still closes: it means a measured REGRESSION on a
        # scored axis, an accuracy-gate failure, or a cross-model guard failure. That is real,
        # attributable harm and is worth acting on no matter what the PR was aiming at.
        #
        # Abandoned PRs are still handled -- by the age-based stale close, which is about
        # inactivity rather than about a measurement.
        if label == "REJECT":
            if not res.get("q36_guard_ok", True):
                fail_clause = "and regressed the Qwen3.6 no-regression guard (decode/prefill on shared code)"
            elif not res.get("accuracy_ok"):
                fail_clause = "and failed the accuracy gate"
            elif res.get("prefill_regressed"):
                fail_clause = "and regressed 128-ctx prefill throughput specifically"
            elif res.get("decode_regressed"):
                fail_clause = "(decode regression)"
            elif label == "none":   # unreachable: see the REJECT-only guard above
                fail_clause = "with no verified improvement on both decode and prefill"
            else:
                fail_clause = "(regression)"
            close_body = (
                "<!-- sparkinfer-museglimmer-auto-close -->\n"
                f"## Closed: sparkinfer museglimmer auto-eval — `eval-museglimmer:{label}`\n\n"
                f"This PR's Muse Glimmer 128-decode/prefill speed measured **{res.get('delta_pct')}%** "
                f"vs main, {fail_clause} "
                "— closing automatically. This bot evaluates every eligible PR in the repo "
                "against Muse Glimmer's decode AND prefill@128 speed specifically, regardless of "
                "what the PR is actually about — a close here isn't a judgment on the PR's purpose, "
                "just that it didn't move these particular metrics. Reopen (or open a fresh PR) if "
                "you have a fix or a different approach."
            )
            arb.gh(["pr", "comment", str(num), "-R", repo, "--body", close_body])
            arb.gh(["pr", "close", str(num), "-R", repo])
            print(f">> auto-closed PR #{num} (eval-museglimmer:{label})")


def main():
    ap = argparse.ArgumentParser(description="Muse Glimmer 128-decode PR eval bot")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--repo", default="gittensor-ai-lab/sparkinfer")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reeval", action="store_true")
    ap.add_argument("--labels-only", action="store_true",
                    help="reconcile museglimmer-merge-first only — no GPU")
    ap.add_argument("--only-prs", default="",
                    help="comma-separated PR numbers (bypass greenlight)")
    args = ap.parse_args()

    only = {int(x) for x in args.only_prs.split(",") if x.strip().isdigit()}

    print(f">> museglimmer eval transport: "
          f"{'ssh' if ssh_box_enabled() else f'vast.ai (instance {arb.current_instance(args.instance) or args.instance})'}")
    print(f">> AUTOMERGE={int(AUTO_MERGE)}")

    if args.labels_only:
        reconcile_museglimmer_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — museglimmer labels only (no GPU).")
        return

    prs = json.loads(arb.gh([
        "pr", "list", "-R", args.repo, "--state", "open",
        "--json", "number,title,labels,isDraft,headRefOid,headRefName,mergeable,author,body",
        "--limit", "80",
    ]).stdout or "[]")
    prs.sort(key=lambda p: p["number"])

    stale_closed = close_stale_museglimmer_prs(args.repo, prs, dry_run=args.dry_run) if not only else set()

    denylist = arb.load_denylist()
    pending = []
    for pr in prs:
        num = pr["number"]
        if num in stale_closed:
            continue
        if only and num not in only:
            continue
        if pr.get("isDraft"):
            continue
        # Gate — blocked contributor: never spend GPU on a flagged/sybil PR. Checks the opener
        # AND every commit's author/committer (arb.pr_involved_logins), not just the PR's own
        # author field — same "Gate 1" pattern pr_eval_bot.py's AR bot already uses, reused
        # verbatim rather than reinvented.
        hits = arb.pr_involved_logins(args.repo, num) & denylist
        if hits:
            print(f"PR #{num}: BLOCKED (denylisted: {', '.join(sorted(hits))}) — flag + close, no eval")
            if not args.dry_run:
                arb.close_blocked_pr(args.repo, num, hits)
            continue
        labs = {l["name"] for l in pr.get("labels", [])}
        if arb.HOLD_LABEL in labs:
            print(f"PR #{num}: hold — skip")
            continue
        head = (pr.get("headRefOid") or "")[:40]
        short = head[:9]
        if not args.reeval and head and head in museglimmer_evaluated_commits(args.repo, num):
            print(f"PR #{num} @ {short}: already museglimmer-evaluated — skip")
            continue
        if arb.pr_merge_conflict(pr.get("mergeable")):
            print(f"PR #{num}: merge conflict — museglimmer-needs-rebase")
            if not args.dry_run:
                arb.add_label(args.repo, num, MUSEGLIMMER_NEEDS_REBASE)
            continue

        if not only:
            status, why = arb.greenlight_status(args.repo, num, labs)
            if status != "ok":
                print(f"PR #{num}: not greenlit ({why}) — skip museglimmer eval")
                continue
            print(f"PR #{num}: greenlit ({why})")
        else:
            print(f"PR #{num}: --only-prs targeted")

        ref = f"pull/{num}/head"
        pending.append((num, head, short, ref, pr.get("title", "")))

    if not pending:
        reconcile_museglimmer_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — no museglimmer PRs to evaluate.")
        return

    if args.dry_run:
        print("--- dry-run would evaluate: " + ", ".join(f"#{n}" for n, *_ in pending))
        return

    pin = arb.PINNED_INSTANCE
    if pin and not ssh_box_enabled():
        with open(arb.INSTANCE_FILE, "w") as f:
            f.write(str(pin))

    try:
        host, port = resolve_ssh(args.instance)
    except Exception as e:
        print(f">> GPU unavailable: {e}")
        reconcile_museglimmer_merge_labels(args.repo, dry_run=False)
        print("done — museglimmer labels only (GPU down).")
        return

    _ssh_user = ssh_box_user() if ssh_box_enabled() else "root"
    print(f">> SSH {_ssh_user}@{host}:{port}")

    print(">> measuring main baseline (once for this round, shared across all pending PRs) …")
    main_result = measure_main_baseline(host, port)
    if not main_result.get("ok"):
        # No usable baseline -> nothing in this round can be scored. Bail out here rather than
        # burning GPU time building N different PR branches against a baseline we already know
        # is broken, and rather than posting a misleading per-PR "main run failed" on every
        # pending PR for what is really one shared infra problem.
        print(f">> main baseline measurement failed: {main_result.get('reason')} — skipping round")
        reconcile_museglimmer_merge_labels(args.repo, dry_run=False)
        print("done — museglimmer round skipped (main baseline unusable).")
        return
    print(f">> main baseline: decode={main_result['decode_tps']:.2f} "
          f"prefill128={main_result['prefill128_pp']:.2f} "
          f"top1={main_result.get('top1', 0):.3f} kl={main_result.get('kl', 99):.4f}")

    for num, head, short, ref, title in pending:
        print(f"PR #{num} @ {short}: evaluating Muse Glimmer '{ref}' …")
        try:
            res = eval_museglimmer_on_box(host, port, ref, main_result)
        except Exception as e:
            res = {"ok": False, "reason": f"exception: {e}"}
        apply_result(args.repo, num, head or short, res, title=title, dry_run=False)

    reconcile_museglimmer_merge_labels(args.repo, dry_run=False)
    print("done — museglimmer eval pass complete.")


if __name__ == "__main__":
    main()
