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
  1. Speed  — decode AND prefill at ctx 128/512/4k/16k/32k/64k (SCORED_CTXS, one Muse Glimmer
              model load via bench_sweep_run) plus concurrent decode at c2-c32 (CB_CONCS):
              seventeen axes (SCORING_DIMS), PR vs a freshly-measured origin/main, same box,
              same run. Same
              tier buckets as the AR and DFlash bots (BUCKETS/SIG/REGRESS_TOL below — copied,
              not reinvented). EVERY axis is also a no-regression floor — REGRESS_TOL failing on
              ANY ONE of them is a hard REJECT, so a PR cannot buy a headline win at one
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
evaluation scope. Narrowed 2026-09-26, with the sibling bots: a REJECT closes, but a `none` closes
only a PR declared for Muse Glimmer alone that no other bot scored a speedup or made merge-first
(arb.none_may_close), and nothing closes over a head that moved after the measured commit. A PR is
measured merged onto the round's main commit (arb.merged_checkout_script), as the siblings do.

  3. Cross-model no-regression guards @ 32k — decode + prefill on FOUR models, same box, same PR
              build, vs a freshly-measured origin/main:
                * Qwen3.6-35B-A3B (Q36_GUARD_*), and
                * the ModelOpt Qwen3.8-27B NVFP4 checkpoint (MODELOPT_MODEL_DIR) -- the one
                  pr_dspark_bot.py scores, so a shared-code regression is caught here at Muse-PR
                  time instead of surfacing later as a mystery in that bot's numbers, and
                * the unsloth Qwen3.8-27B NVFP4 checkpoint (QWEN38_MODEL_DIR) -- the one
                  pr_qwen38_bot.py scores (added 2026-09-15). That bot skips PRs declared for
                  Muse Glimmer alone, so without this guard nothing checked them against it.
                * Ternary-Bonsai-2-27B (BONSAI_GGUF, added 2026-09-24) -- the one pr_bonsai_bot.py
                  scores, at 128 as well as 32k, for the same reason. A guard main measured
                  nothing for skips the round (measure_main_baseline), and a guard sweep the OOM
                  killer took is retried as the box's; a guard only the PR build failed to measure
                  is a regression (fail-closed, as on pr_qwen38_bot.py). A new guard must not add a
                  way to auto-close a PR over an infrastructure fault.
              Narrowed from the previous five-context Qwen3.6 sweep to 32k only: those extra
              points cost a model load each on models this bot does not score, and 32k is where
              shared prefill/KV code actually breaks. All the guards share one implementation
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
import calendar
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
# v6 (2026-09-15): unsloth Qwen3.8 32k guard added beside the ModelOpt and Qwen3.6 guards.
# v7 (2026-09-24): Ternary-Bonsai-2-27B guard added (decode + prefill @ 128 and 32k).
EVAL_SCHEMA_VERSION = "v7-ctx6-prefill-decode-cbdecode-modelopt-unsloth-bonsai-guards"
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

# The measuring instrument. Unlike the Qwen3.8 and Bonsai bots, this bot pins no harness: the PR is
# built merged onto the round's main, so a PR that edits these files would be measured with its own
# copy of the ruler it is scored by. Such a PR is not evaluated, as on those bots (2026-09-26).
HARNESS_PATHS = (
    "runtime/examples/qwen3_gguf_bench.cpp",
    "runtime/examples/qwen3_gguf_cb_bench.cpp",
    "runtime/examples/qwen3_gguf_score.cpp",
    "runtime/examples/qwen_checkpoint.h",
    "runtime/examples/qwen3_gguf_config.h",
    "eval/",
    "bench/scripts/",
)

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
# So: one bench_sweep_run call per tier, which is how differing reps were possible at all.
#
# 2026-09-14: every context is back to reps=5, which collapses the tiers to ONE call (and so one
# model load instead of three -- this is now cheaper, not more expensive).
#
# The reps=1 tier existed because a 32k prefill took ~330s when it was written. After this week's
# prefill work it takes 2.47s; 16k takes 1.13s. reps=5 across all six contexts costs about 48s of
# prefill in total, so the reason to economise is gone -- and economising cost real damage:
#
#   #1064 scored eval-museglimmer:XL on muse-prefill@16k +96.1% and AUTO-MERGED as merge-first,
#   on a single unaveraged baseline sample of 7442.44 pp/s. Every other round measured main at
#   14283-14551. The PR's own 16k reading (14597) was normal, every one of its other 16 axes was
#   flat, and the axis it actually claims (cb-decode@c8) measured -0.7%. The entire tier came from
#   one bad baseline sample.
#
# This is the second time: #785 was merged and reverted for exactly this, which is why 128/512
# went to reps=5 (596e4ed). That fix was applied to the tier that had just misfired instead of to
# the practice, so the next context to get a bad sample repeated it.
SCORED_REPS_TIERS = [([128, 512, 4096, 16384, 32768, 65536], 5)]
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
# Concurrent-decode axes (issue #1026, requested by FranDev132). Every one of the twelve axes
# above drives a SINGLE request -- bench_decode runs one sequence, the prefill sweep ingests one
# prompt -- so nothing in that set can observe a change that only moves multi-request throughput.
# Muse gets no concurrency scaling at all on main (aggregate tok/s is flat from c=1 to c=16,
# because each concurrent request re-reads all ~17 GB of weights on its own), which is exactly the
# kind of headroom no scored axis was pointed at.
#
# Same binary, invocation shape and parser pr_dspark_bot.py already uses for its own cb-decode
# axes; only the checkpoint differs.
CB_CONCS = [2, 4, 8, 16, 32]
# 256 prompt / 256 generated, matching pr_dspark_bot.py. Its comment records why this
# cannot be shortened to save GPU: at shorter generations the run spends its time
# measuring its own startup, and c=4/c=8 can land several percent low on UNCHANGED code
# -- a spurious-rejection generator on an axis that is also a no-regression floor.
CB_TOKENS = 256
CB_DIM_FOR = {c: f"muse-cb-decode@c{c}" for c in CB_CONCS}
CB_DIMS = [CB_DIM_FOR[c] for c in CB_CONCS]

# Order matters only for display. SCORING_DIM must stay muse-decode@128 (the back-compat
# dimension the report/dashboard/Polaris payload read), so the concurrency axes append.
SCORING_DIMS = [f"muse-{phase}@{SCORED_CTX_LABEL[c]}"
                for c in SCORED_CTXS for phase in ("decode", "prefill")] + CB_DIMS
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
# The unsloth Qwen3.8-27B NVFP4 checkpoint (NVFP4 FFN + FP8 attention/Gated-DeltaNet projections) --
# the one pr_qwen38_bot.py scores. Its FP8 and Q4_K code paths are not the ModelOpt checkpoint's, so
# the ModelOpt guard above does not cover it. pr_qwen38_bot.py skips PRs declared for Muse Glimmer
# alone, so this guard is the only check they get against it. Same env var as that bot uses.
UNSLOTH_GUARD_MODEL_DIR = os.environ.get("QWEN38_MODEL_DIR", "/root/workspace/models_qwen38")
UNSLOTH_GUARD_CTXS = [32768]
# Ternary-Bonsai-2-27B, the model pr_bonsai_bot.py scores, which skips PRs declared for Muse Glimmer
# alone. 128 as well as 32k: the dense-GGUF prefill work on that model lives at short prompts
# (#1139: 1.94x at 128, flat at 4k). One model load either way.
BONSAI_GUARD_GGUF = os.environ.get(
    "BONSAI_GGUF", "/root/workspace/models_bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
BONSAI_GUARD_CTXS = [128, 32768]

# Auto-merge (the shape of pr_dflash_bot.py's auto_merge_ok_dflash/try_auto_merge_dflash) is OFF
# unless this exact env var is "1". The eval host's .env.eval sets it (explicit decision; see the
# module docstring); the wrappers never force it.
AUTO_MERGE = os.environ.get("SPARKINFER_MUSEGLIMMER_AUTOMERGE") == "1"
AUTOMERGE_BLOCK = {
    "copycat", "copycat-warn", "flagged:gaming", "penalty", "needs-benchmark",
    MUSEGLIMMER_NEEDS_REBASE, arb.REEVALUATE_LABEL, arb.HOLD_LABEL, *arb.REGRESSION_LABELS,
}

SCORES_FILE = os.path.expanduser(
    os.environ.get("MUSEGLIMMER_SCORES_FILE", "~/.sparkinfer_museglimmer_scores.json")
)
# Box faults per PR and commit (arb.record_strike): one recurring at a commit is charged to the PR.
STRIKES_FILE = os.path.expanduser(
    os.environ.get("MUSEGLIMMER_STRIKES_FILE", "~/.sparkinfer_museglimmer_strikes.json")
)
# When each PR began waiting on its author, per head: the stale close's clock (arb.AuthorWaitClock).
AUTHOR_WAIT_FILE = os.path.expanduser(
    os.environ.get("MUSEGLIMMER_AUTHOR_WAIT_FILE", "~/.sparkinfer_museglimmer_author_wait.json")
)
# PRs the bot gave up on this run (its own errors): the run then exits 3, so they are not silent.
GAVE_UP = set()
# A REJECT that may be the box's (a concurrent width only the PR build failed) repeats this many rounds.
STRIKES_TO_REJECT = 2

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
    arb.write_json_atomic(SCORES_FILE, data)


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
    # A measurement wildly above main is a broken forward pass, not a speedup. #1037 reported
    # muse-decode@128 at 352,667 tok/s against main's 106 (3,300x) while emitting garbage
    # (top-1 0.000, KL 11.29): a pass that stops computing also stops taking time. Scoring that
    # as XL would be the worst possible outcome, so it fails closed with a reason that says what
    # it is. The bound is a RATIO against the same-box baseline, not an absolute ceiling, so it
    # keeps working as the model gets faster.
    if pr_tps > IMPLAUSIBLE_GAIN * main_tps:
        return "REJECT", round(100.0 * (pr_tps - main_tps) / main_tps, 1), False, (
            f"{metric} implausible: {pr_tps:.2f} is {pr_tps / main_tps:.0f}x main {main_tps:.2f} "
            f"— treated as a broken measurement, not a speedup"
        )
    g = (pr_tps - main_tps) / main_tps
    pct = round(100.0 * g, 1)
    if g < SIG:
        return "none", pct, True, f"within significance gate — not a verified {metric} improvement"
    for thr, name in BUCKETS:
        if g >= thr:
            return name, pct, True, "ok"
    return "none", pct, True, "ok"


# Above this multiple of the same-box main baseline a measurement is treated as broken rather
# than fast. 20x is far outside anything a real optimization has produced here (the largest to
# date is ~2.0x on a concurrency axis) and far below #1037's 3,300x.
IMPLAUSIBLE_GAIN = float(os.environ.get("MUSEGLIMMER_IMPLAUSIBLE_GAIN", "20"))

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


def check_unsloth_guard(pr: dict, main: dict, tol: float = REGRESS_TOL):
    """Unsloth Qwen3.8-27B NVFP4 no-regression guard, decode + prefill @ 32k. Same discipline and the
    same hard REJECT as the ModelOpt guard -- this is the checkpoint pr_qwen38_bot.py scores."""
    return _check_model_guard(pr, main, "guardun", "unsloth qwen3.8", tol)


def check_bonsai_guard(pr: dict, main: dict, tol: float = REGRESS_TOL):
    """Ternary-Bonsai-2-27B no-regression guard, decode + prefill @ 128 and 32k -- the model
    pr_bonsai_bot.py scores."""
    return _check_model_guard(pr, main, "guardbn", "ternary-bonsai", tol)


def museglimmer_evaluated_commits(repo, num):
    """Head commits that already have a REAL scoring verdict posted — mirrors
    dflash_evaluated_commits: infra/transport failures (label:null in the marker) don't count."""
    return arb.evaluated_commits_from(repo, num, MARKER_RE, "sparkinfer museglimmer auto-eval")


def strip_museglimmer_eval_labels(repo, num):
    arb.strip_own_tier_labels(repo, num, EVAL_PREFIX)


STALE_DAYS = float(os.environ.get("MUSEGLIMMER_STALE_DAYS", "1"))


def _pr_last_activity_ts(repo, num):
    """Last real author activity (most recent commit's committedDate), not PR updatedAt — same
    rationale as pr_dflash_bot.py's copy of this helper (bot comments/labels bump updatedAt)."""
    r = arb.gh(["pr", "view", str(num), "-R", repo, "--json", "commits,createdAt"])
    try:
        info = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return None
    # The PR's own opening counts as activity too: a PR opened from commits made days earlier used
    # to be closed as stale before it was ever evaluated. calendar.timegm, not time.mktime: these
    # are UTC, and mktime read them as local time (2 h early on a CEST controller).
    dates = [c.get("committedDate") for c in (info.get("commits") or []) if c.get("committedDate")]
    dates += [info["createdAt"]] if info.get("createdAt") else []
    if not dates:
        return None
    try:
        return calendar.timegm(time.strptime(max(dates), "%Y-%m-%dT%H:%M:%SZ"))
    except ValueError:
        return None


def _waits_for_the_winner(pr, labs, head, main_now):
    """A verified speedup at this head, sent to museglimmer-needs-rebase only because another PR won
    merge-first. Until that PR merges and main moves there is nothing to rebase onto, so the wait
    is the merge's, not the author's; from then on the rebase is the author's (CONTRIBUTING), and
    the stale clock starts. A conflict, or any other block label, is the author's at once. Kept
    too when today's main is unknown."""
    entry = _load_scores().get(str(pr["number"])) or {}
    if (MUSEGLIMMER_NEEDS_REBASE not in labs or labs & (AUTOMERGE_BLOCK - {MUSEGLIMMER_NEEDS_REBASE})
            or entry.get("commit") != head or entry.get("label") not in SPEEDUP_LABELS or not entry.get("pass")
            or arb.pr_merge_conflict(pr.get("mergeable"))
            or arb.strike_count(STRIKES_FILE, pr["number"], head, "conflict")):
        return False
    return not main_now or not arb.scored_against_stale_main(entry, main_now)


def close_stale_museglimmer_prs(repo, prs, dry_run=False):
    """Close open PRs routed to Muse Glimmer with no author commit activity in STALE_DAYS+ days.
    Drafts, `hold`, other models' PRs and any bot's merge-first are exempt
    (arb.stale_close_skip_reason) -- this used to close every idle PR in the repo.
    Idle means waiting on its author for STALE_DAYS as well (arb.AuthorWaitClock): a PR the bot kept
    waiting is not closed the round it is handed back. STALE_DAYS <= 0 turns the stale close off."""
    closed = set()
    if arb.stale_close_disabled(STALE_DAYS):
        return closed
    now = time.time()
    main_now = None
    clock = arb.AuthorWaitClock(AUTHOR_WAIT_FILE, [p["number"] for p in prs], now, record=not dry_run)
    for pr in prs:
        num = pr["number"]
        if arb.stale_close_skip_reason(pr, "muse", EVAL_PREFIX):
            clock.forget(num)                  # a maintainer's, a merge's or another bot's wait
            continue
        ts = _pr_last_activity_ts(repo, num)
        if ts is None:
            continue
        # A commit dated in the future (a skewed or forged clock) proves no recent work: only the bot's
        # own clock counts for it. Taken as-is, it kept the PR from ever going stale.
        future = ts > now
        age_days = 0.0 if future else (now - ts) / 86400
        # Every PR is classified, not only one idle for STALE_DAYS: the clock must start the round a
        # PR is handed back, or it would start only once the commit is that old -- twice the period.
        note = print if age_days >= STALE_DAYS else (lambda *_: None)     # the log names idle PRs only
        head = (pr.get("headRefOid") or "")[:40]
        labs = {l["name"] for l in pr.get("labels", [])}
        if main_now is None:
            main_now = arb.current_main_sha(repo)
        if arb.gave_up(STRIKES_FILE, num, head) and not _unmeasurable_reason(repo, pr, labs, count_gave_up=False):
            # The bot failed on it itself (loudly: the run exits 3). Not the author's to lose it for.
            note(f"PR #{num}: idle {age_days:.1f}d, but the bot gave up on its head after its own errors — kept open")
            clock.forget(num)
            continue
        owed = _remeasure_state(repo, num, head, labs, main_now)
        if owed is None or (owed and not _unmeasurable_reason(repo, pr, labs)):
            # A verified speedup the bot owes a re-measure onto today's main (or GitHub did not say):
            # the wait is the bot's.
            note(f"PR #{num}: idle {age_days:.1f}d but owed a re-measure onto the new main — kept open")
            if owed:
                clock.forget(num)
            continue
        if _waits_for_the_winner(pr, labs, head, main_now):
            note(f"PR #{num}: idle {age_days:.1f}d, a verified speedup waiting for the merge-first PR to merge — kept open")
            if main_now:                       # not on an unanswered main read: unknown keeps the clock
                clock.forget(num)
            continue
        evaluated = museglimmer_evaluated_commits(repo, num)
        if evaluated is None:
            note(f"PR #{num}: idle {age_days:.1f}d; GitHub did not return its comments — kept open")
            continue
        greenlit = None
        if head not in evaluated:
            greenlit = arb.greenlight_status(repo, num, labs)[0]
            if greenlit not in ("ok", "unknown"):
                # Never measured at this head and not asking to be (docs, tooling, a ticked box with
                # no numbers): not in this bot's queue, and "not being measured is not grounds for
                # closing" (CONTRIBUTING). The daily close-stale-prs Action closes one left idle.
                note(f"PR #{num}: idle {age_days:.1f}d, never measured here and not greenlit — left to the daily stale close")
                clock.forget(num)
                continue
        if not arb.strike_count(STRIKES_FILE, num, head, "harness") and arb.waiting_for_first_verdict(
                repo, pr, evaluated, never_paths=HARNESS_PATHS,
                box_conflict=bool(arb.strike_count(STRIKES_FILE, num, head, "conflict")), greenlit=greenlit):
            # Greenlit and still waiting for this bot to measure its head: the wait is the bot's.
            note(f"PR #{num}: idle {age_days:.1f}d but still waiting for its first museglimmer verdict — kept open")
            clock.forget(num)
            continue
        # Waiting on its author -- counted from the first round the bot saw that, not from the commit.
        waited = (now - max(0.0 if future else ts, clock.since(num, head))) / 86400
        if waited < STALE_DAYS:
            note(f"PR #{num}: idle {age_days:.1f}d, waiting on its author for {waited:.1f}d "
                 f"(closed at {STALE_DAYS:g}d) — kept open")
            continue
        age_days = max(age_days, waited)       # (the same, unless the commit was dated in the future)
        print(f"PR #{num}: stale ({age_days:.1f}d since last commit, {waited:.1f}d waiting on its author, "
              f"threshold {STALE_DAYS:g}d) — closing")
        closed.add(num)
        clock.forget(num)
        if dry_run:
            continue
        body = (
            "<!-- sparkinfer-museglimmer-auto-close-stale -->\n"
            f"## Closed: stale — no commits in {age_days:.1f} days\n\n"
            f"This PR has had no new commits in over {STALE_DAYS:g} days — closing automatically "
            "to keep the Muse Glimmer eval queue clean. Nothing is wrong with it for that reason "
            "alone: push your latest work and reopen it (or open a fresh PR) whenever you're "
            "ready, and it is picked back up on the next eval cycle."
        )
        arb.gh(["pr", "comment", str(num), "-R", repo, "--body", body])
        arb.gh(["pr", "close", str(num), "-R", repo])
    clock.save()
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
        # errors="replace": a stray non-UTF-8 byte in a build log raised after the whole run, and the
        # PR was retried every round with nothing posted.
        capture_output=True, text=True, errors="replace", timeout=timeout,
        input=cmd if via_stdin else stdin_data,
    )


_EXPLICIT_FAIL_MARKERS = ("HARNESS_TOUCHED", "BUILD_FAILED", "LLAMACPP_CONFIGURE_FAILED", "LLAMACPP_BUILD_FAILED",
                          "MERGE_CONFLICT", "ACCURACY_COMPARE_FAILED", "BASE_AHEAD")


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
        # The single most actionable line is usually the compiler/linker's own error -- prefer
        # that over the marker's header and over make's `*** Error N` (arb.first_build_error).
        err = arb.first_build_error(lines[i + 1:i + 80])
        if err:
            return f"{marker}: {err}"
        tail = " | ".join(l.strip() for l in lines[i + 1:i + 3] if l.strip())
        return marker + (f": {tail}" if tail else "")
    return None


def _is_box_fault(stdout: str, stderr: str) -> bool:
    """A failed run that is the box's rather than the PR's -- a GPU that did not drain, a failed
    fetch, a compiler killed for memory, or a shell that died with no diagnostic before the run's
    end (GUARD_END). Nothing is posted and no label changes; the next round measures again (as
    pr_bonsai_bot.py does), and a fault recurring at one commit is charged to the PR after
    arb.BOX_FAULT_STRIKES rounds (apply_result). Before, it posted `eval-museglimmer:REJECT` every round until the box
    recovered, which also set the generic `eval:REJECT` that SN74 scoring reads."""
    combined = (stdout or "") + "\n" + (stderr or "")
    if "RETRYABLE_INFRA_FAILURE" in combined:
        return True
    if "exit=137" in (_crash_reason(stdout, stderr) or ""):
        # SIGKILL: the host OOM killer, whose trigger may be anything on the box (pr_bonsai_bot.py's
        # rule). Recurring at one commit, it is charged to the PR after BOX_FAULT_STRIKES rounds.
        return True
    return _looks_like_hard_kill(stdout, stderr)


def _looks_like_hard_kill(stdout: str, stderr: str) -> bool:
    """No ERR-trap diagnostic, and the run never reached its end. GUARD_END, not
    ACCURACY_STAGE_DONE: all four guard model loads run after the accuracy stage, so a shell killed
    during them (an ssh drop, the OOM killer, the box rebooting) used to count as the PR's failure --
    and, recorded for its commit, as a REJECT the PR kept until its next push. pr_qwen38_bot.py
    made the same change for the same reason."""
    combined = (stdout or "") + "\n" + (stderr or "")
    if _crash_reason(stdout, stderr):
        return False
    return "GUARD_END" not in combined


def _ssh_run_resilient(host, port, script: str, label: str):
    """One automatic retry on an apparent hard kill — same insurance pr_dflash_bot.py added
    after #684/#690 (heavy model-reload boundaries silently killing the whole remote shell)."""
    r = ssh_run(host, port, script, via_stdin=True)
    if r.returncode != 0:
        combined = (r.stdout or "") + "\n" + (r.stderr or "")
        # An explicitly classified infrastructure fault must never be charged to the PR: the
        # GPU not draining between stages says nothing about the change being measured.
        if "RETRYABLE_INFRA_FAILURE" in combined:
            print(f">> {label}: transient infrastructure failure — retrying the entire "
                  "measurement once")
            r = ssh_run(host, port, script, via_stdin=True)
        elif _looks_like_hard_kill(r.stdout, r.stderr):
            print(f">> {label}: looks like a hard kill (no ERR-trap diagnostic, the run never "
                  f"reached GUARD_END) — retrying once")
            r = ssh_run(host, port, script, via_stdin=True)
    return r


def _remote_script(ref: str, role: str = "pr", onto: str | None = None) -> str:
    """Bash run on the eval box: checkout ref, build, 128-decode speed bench, accuracy gate vs
    a live llama-server reference. Run once per ref (PR, then "main") — identical script both
    times so the two measurements are directly comparable.

    A PR (`ref` = pull/<n>/head) is measured MERGED onto `onto`, the exact main commit the round's
    baseline measured (arb.merged_checkout_script), as the Qwen3.8 and Bonsai bots do since #1145.
    Measuring the branch tip itself compared a branch behind main against a newer main: every
    speedup merged since the branch point read as the PR's own regression -- a REJECT and a close
    -- and the branch built its own older copy of the harness."""
    if role == "pr":
        checkout = arb.merged_checkout_script(ref, onto or "origin/main", HARNESS_PATHS)
    else:
        checkout = (f'timeout 600 git fetch -q origin {shlex.quote(ref)} || {{ echo "RETRYABLE_INFRA_FAILURE git fetch {ref} failed" >&2; exit 1; }}\n'
                    'find .git -maxdepth 1 -name index.lock -mmin +10 -delete 2>/dev/null || true\n'
                    'git reset -q --hard || { echo "RETRYABLE_INFRA_FAILURE git reset failed" >&2; exit 1; }\n'
                    'git clean -qfd || { echo "RETRYABLE_INFRA_FAILURE git clean failed" >&2; exit 1; }\n'
                    'git checkout -qf FETCH_HEAD || { echo "RETRYABLE_INFRA_FAILURE git checkout failed" >&2; exit 1; }\n'
                    'echo "REMOTE_HEAD $(git rev-parse --short HEAD)"\n'
                    'echo "REMOTE_SHA $(git rev-parse HEAD)"\n')
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
    cb_concs = " ".join(str(c) for c in CB_CONCS)
    cb_tokens = CB_TOKENS
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
            f'  MUSE_RC=${{_BENCH_SWEEP_RC:-1}}\n'
            f'fi')
    scored_sweep_blocks = "\n".join(scored_blocks)
    q36_sweep_args = " ".join(f"{c} {BENCH_REPS}" for c in Q36_GUARD_CTXS)
    q36_ctx_list = " ".join(str(c) for c in Q36_GUARD_CTXS)
    mo_sweep_args = " ".join(f"{c} {BENCH_REPS}" for c in MODELOPT_GUARD_CTXS)
    mo_ctx_list = " ".join(str(c) for c in MODELOPT_GUARD_CTXS)
    un_dir = shlex.quote(UNSLOTH_GUARD_MODEL_DIR)
    un_sweep_args = " ".join(f"{c} {BENCH_REPS}" for c in UNSLOTH_GUARD_CTXS)
    un_ctx_list = " ".join(str(c) for c in UNSLOTH_GUARD_CTXS)
    bn_gguf = shlex.quote(BONSAI_GUARD_GGUF)
    bn_sweep_args = " ".join(f"{c} {BENCH_REPS}" for c in BONSAI_GUARD_CTXS)
    bn_ctx_list = " ".join(str(c) for c in BONSAI_GUARD_CTXS)
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
  # 30s was not enough once the concurrent-decode axes landed (3b42b7a): the c=32 run holds
  # ~26 GB across 33 sequences and does not always release inside half a minute, so the
  # accuracy stage started anyway and OOM'd loading the model. That surfaced as
  # REMOTE_SCRIPT_FAILED line=232 and was charged to the PR -- it skipped the whole round on
  # the 20:00 main baseline and put eval:REJECT on #1059, whose own measurements were fine.
  while [ "$tries" -lt 180 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -lt 1024 ] 2>/dev/null && return 0
    sleep 1
    tries=$((tries + 1))
  done
  # Do NOT proceed into a load that is now certain to OOM and get blamed on the PR. Name it
  # as infrastructure so _ssh_run_resilient retries the measurement instead.
  echo "RETRYABLE_INFRA_FAILURE GPU still holding ${{used:-unknown}} MiB after ${{tries}}s — refusing to start a load that would OOM" >&2
  return 1
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
UNSLOTH_GUARD_MODEL_DIR={un_dir}
BONSAI_GUARD_GGUF={bn_gguf}

{arb.round_guard_sh("museglimmer")}
cd "$REPO"
git remote set-url origin https://github.com/gittensor-ai-lab/sparkinfer.git 2>/dev/null || true
# A network failure is the box's, never the PR's: RETRYABLE, not the ERR trap. A PR is merged onto
# the baseline commit (PR_TIP, MERGED_ONTO); main reports its commit (REMOTE_SHA).
{checkout}
test -f "$GGUF" || {{ echo "FAIL missing GGUF $GGUF"; exit 1; }}

# Build sparkinfer's speed-bench + teacher-forced-score binaries. Always reconfigure (cheap,
# idempotent) — skipping it on an existing CMakeCache left stale generated Makefiles pointing at
# a DIFFERENT PR branch's files once the checkout switched underneath it (pr_dflash_bot.py
# #693/#694 hit exactly this).
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/mg_cmake.log 2>&1 || {{
  echo "BUILD_FAILED — cmake configure; tail:" >&2
  tail -40 /tmp/mg_cmake.log >&2
  exit 1
}}
{arb.BUILD_FAILURE_SH}
build_targets() {{
  cmake --build build --target qwen3_gguf_bench qwen3_gguf_score qwen3_gguf_cb_bench -j"$1" >/tmp/mg_build.log 2>&1
}}
# A compiler killed for memory, a full disk or the overlayfs EFAULT is the box: rebuild once at -j4,
# then call it infra. Anything else is the PR's build error, errors first (arb.BUILD_FAILURE_SH).
if ! build_targets "$(nproc)"; then
  if FAULT=$(build_box_fault /tmp/mg_build.log); then
    echo "build hit a box-side fault ($FAULT) -- rebuilding with -j4" >&2
    if ! build_targets 4; then
      if FAULT=$(build_box_fault /tmp/mg_build.log); then
        echo "RETRYABLE_INFRA_FAILURE build: $FAULT" >&2
        exit 1
      fi
      report_build_failure /tmp/mg_build.log
      exit 1
    fi
  else
    report_build_failure /tmp/mg_build.log
    exit 1
  fi
fi
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
MUSE_RC=0
BC_DECODE=0
BC_PREFILL=0
{scored_sweep_blocks}
# rc=137 is SIGKILL (the host OOM killer): the box's, not the PR's (eval_museglimmer_on_box).
[ "$MUSE_OK" = "1" ] || echo "MUSE_FAILED rc=${{MUSE_RC:-1}}"
# Back-compat: decode@128 / prefill@128 also go out under their old names. The PR comment
# renderer, the Polaris payload and the published dashboard all read these two keys, and none of
# them should have to change because the scoring matrix grew.
# Captured during the tier that measured 128 -- _bench_sweep_get reads the LAST sweep's JSON, and
# 128 is not in the last tier.
echo "RESULT_DECODE_TPS ${{BC_DECODE:-0}}"
echo "RESULT_PREFILL128_PP ${{BC_PREFILL:-0}}"

# --- concurrent decode (issue #1026) --------------------------------------------------------
# Aggregate tok/s with N requests in flight. One model load per concurrency point, so this is
# the expensive half of the round -- but it is the only thing here that observes the packed
# multi-row forward, which no single-request axis enters.
#
# A failed or zero point emits MUSECB_FAILED <c> rc=<exit> and is never read as a regression to
# zero. A width main could not measure is dropped for the round; one main measured and the PR build
# could not is a REJECT judged over two rounds on the same commit (eval_museglimmer_on_box,
# pr_bonsai_bot.py's rule); rc=137, the OOM killer, is the box's.
for CC in {cb_concs}; do
  CB_OUT=/tmp/mg_cb_$CC.txt
  wait_gpu_clear
  CB_RC=0
  if timeout 1800 build/runtime/qwen3_gguf_cb_bench "$GGUF" "$CC" {cb_tokens} {cb_tokens} 512 > "$CB_OUT" 2>&1 || {{ CB_RC=$?; false; }}; then
    CB_AGG=$(sed -n 's/.*agg_tok_s=\\([0-9.]*\\).*/\\1/p' "$CB_OUT" | tail -1)
    if [ -n "${{CB_AGG:-}}" ] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" "${{CB_AGG:-0}}"; then
      echo "MUSECB $CC $CB_AGG"
    else
      echo "MUSECB_FAILED $CC rc=0"
      echo "concurrent decode produced no positive metric at c=$CC" >&2
      tail -10 "$CB_OUT" >&2 || true
    fi
  else
    # rc=137 is SIGKILL (the host OOM killer): the box's (eval_museglimmer_on_box).
    echo "MUSECB_FAILED $CC rc=$CB_RC"
    echo "concurrent-decode harness exited $CB_RC at c=$CC" >&2
    tail -10 "$CB_OUT" >&2 || true
  fi
done
# The concurrency sweep is the heaviest stage in the run (five model loads, the last holding
# ~26 GB across 33 sequences). Drain before the accuracy gate rather than letting that gate's
# own wait absorb it, so a slow release is reported here instead of as an accuracy failure.
wait_gpu_clear || exit 1

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
  # The reference is the box's own llama.cpp checkout, which no PR can touch: infra, not the PR.
  echo "RETRYABLE_INFRA_FAILURE llama.cpp reference configure failed (the box's checkout)" >&2
  exit 1
}}
cmake --build "$LLAMACPP_DIR/build" -j"$(nproc)" --target llama-server llama-tokenize \\
  >/tmp/mg_llamacpp_build.log 2>&1 || {{
  echo "LLAMACPP_BUILD_FAILED — errors, then tail:" >&2
  grep -E '(^|[^A-Za-z_])(error|fatal error)( |:)|undefined reference' /tmp/mg_llamacpp_build.log | head -20 >&2 || true
  tail -40 /tmp/mg_llamacpp_build.log >&2
  echo "RETRYABLE_INFRA_FAILURE llama.cpp reference build failed (the box's checkout)" >&2
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
  curl -s --max-time 10 "http://localhost:$PORT/health" 2>/dev/null | grep -q '"ok"' && break
  sleep 2
done
# The reference is the box's own llama.cpp, which no PR touches: if it never came up, that is infra
# (it used to fall through to a failed accuracy compare, charged to the PR).
if ! curl -s --max-time 10 "http://localhost:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
  echo "RETRYABLE_INFRA_FAILURE llama.cpp reference server never became healthy (the box's reference)" >&2
  tail -20 /tmp/mg_llama_srv.log >&2 || true
  exit 1
fi
echo "ACCURACY_STAGE_DONE"

# /dev/null as the tokenizer-path arg is safe: accuracy_compare.py's 3rd positional arg is a
# file of already-tokenized space-separated ids (produced above), so its all-digit check skips
# the HF-tokenizer-load code path entirely — the tokenizer path is never opened.
# Bounded: the compare only waits on the box's own llama-server, and a server that stops answering
# used to hang the run to the ssh limit.
ACC_RC=0
ACCOUT=$(timeout 1800 python3 bench/scripts/accuracy_compare.py /tmp/mg_score.txt /dev/null /tmp/mg_eval_ids.txt \\
         "http://localhost:$PORT" "$TOPK") || ACC_RC=$?
if [ "$ACC_RC" != 0 ]; then
  echo "$ACCOUT"
  # The reference is the box's own llama-server: if it went down or stopped answering mid-compare,
  # that is infra. If it is still up, the compare failed on the PR's score dump.
  # 3: accuracy_compare.py's REFERENCE_FAILED -- the server answered /health but not /completion.
  if [ "$ACC_RC" = 124 ] || [ "$ACC_RC" = 3 ] || ! curl -s --max-time 10 "http://localhost:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
    echo "RETRYABLE_INFRA_FAILURE llama.cpp reference server went down during the accuracy compare" >&2
    tail -20 /tmp/mg_llama_srv.log >&2 || true
    exit 1
  fi
  echo "ACCURACY_COMPARE_FAILED -- the PR's score dump could not be compared with the reference" >&2
  exit 1
fi
echo "$ACCOUT"
kill $SRV 2>/dev/null || true
wait $SRV 2>/dev/null || true
trap - EXIT

METRIC_LINE=$(echo "$ACCOUT" | grep '^METRIC ' | tail -1)
TOP1=$(echo "$METRIC_LINE" | sed -E 's/.*top1=([0-9.]+).*/\\1/')
KL=$(echo "$METRIC_LINE" | sed -E 's/.*kl=([0-9.]+).*/\\1/')
PPLS=$(echo "$METRIC_LINE" | sed -E 's/.*ppl_spark=([0-9.]+).*/\\1/')
PPLL=$(echo "$METRIC_LINE" | sed -E 's/.*ppl_llama=([0-9.]+).*/\\1/')
ACCN=$(echo "$METRIC_LINE" | sed -nE 's/.* n=([0-9]+).*/\\1/p')
ACCX=$(echo "$METRIC_LINE" | sed -nE 's/.* n_expected=([0-9]+).*/\\1/p')
echo "RESULT_ACC_POSITIONS ${{ACCN:-?}} ${{ACCX:-?}}"
echo "RESULT_TOP1 ${{TOP1:-0}}"
echo "RESULT_KL ${{KL:-99}}"
echo "RESULT_PPL_SPARK ${{PPLS:-0}}"
echo "RESULT_PPL_LLAMA ${{PPLL:-0}}"

# --- Qwen3.6 no-regression guard (decode + prefill @ 32k) — same qwen3_gguf_bench
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
  echo "GUARD36_FAILED rc=${{_BENCH_SWEEP_RC:-1}}"
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
    echo "GUARDMO_FAILED rc=${{_BENCH_SWEEP_RC:-1}}"
  fi
else
  echo "GUARDMO_UNAVAILABLE"
fi

# --- Unsloth Qwen3.8-27B NVFP4 no-regression guard (decode + prefill @ 32k) ---
# The checkpoint pr_qwen38_bot.py scores, run the way that bot runs it (no env pins). Skipped, not
# failed, when the checkpoint is absent, exactly like the ModelOpt guard above.
if [ -d "$UNSLOTH_GUARD_MODEL_DIR" ]; then
  wait_gpu_clear
  if bench_sweep_run "$UNSLOTH_GUARD_MODEL_DIR" 128 {un_sweep_args}; then
    for ctx in {un_ctx_list}; do
      echo "GUARDUN $ctx $(_bench_sweep_get $ctx decode_tps) $(_bench_sweep_get $ctx prefill_pp)"
    done
  else
    echo "GUARDUN_FAILED rc=${{_BENCH_SWEEP_RC:-1}}"
  fi
else
  echo "GUARDUN_UNAVAILABLE"
fi

# --- Ternary-Bonsai-2-27B no-regression guard (decode + prefill @ 128 and 32k) ---
# The GGUF pr_bonsai_bot.py scores, run the way that bot runs it (no env pins). Skipped, not failed,
# when the GGUF is absent.
if [ -f "$BONSAI_GUARD_GGUF" ]; then
  wait_gpu_clear
  if bench_sweep_run "$BONSAI_GUARD_GGUF" 128 {bn_sweep_args}; then
    for ctx in {bn_ctx_list}; do
      echo "GUARDBN $ctx $(_bench_sweep_get $ctx decode_tps) $(_bench_sweep_get $ctx prefill_pp)"
    done
  else
    echo "GUARDBN_FAILED rc=${{_BENCH_SWEEP_RC:-1}}"
  fi
else
  echo "GUARDBN_UNAVAILABLE"
fi
echo "GUARD_END"
"""


def _at(res: dict, ctx: int, phase: str) -> float:
    """One measurement out of a parsed run's matrix, 0.0 when that context/phase is missing.
    Every consumer reads through this rather than indexing, so a sweep that dropped one context
    degrades to a zero (which tier_from_gain scores as a regression) instead of a KeyError that
    would fail the whole round."""
    return float(((res.get("muse") or {}).get(ctx) or {}).get(phase, 0.0) or 0.0)


def _cb(res: dict, conc: int):
    """One concurrent-decode measurement, or None when that concurrency point was not measured.

    None (not 0.0) on purpose: a point the harness could not produce must DROP its axis, never be
    scored as a regression to zero. _at() returns 0.0 for a missing single-request context because
    those come from one sweep that either ran or did not; the concurrency points are independent
    runs where one can fail on its own."""
    v = (res.get("muse_cb") or {}).get(conc)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _parse_remote(stdout: str) -> dict:
    out = {}
    guard36 = {}
    guardmo = {}
    guardun = {}
    guardbn = {}
    muse = {}
    muse_cb = {}
    cb_failed = set()
    cb_failed_box = set()
    for line in (stdout or "").splitlines():
        if line.startswith("REMOTE_HEAD "):
            out["head"] = line.split()[1]
        elif line.startswith("REMOTE_SHA ") and len(line.split()) >= 2:
            out["sha"] = line.split()[1]            # main's commit (role main) / the local merge
        elif line.startswith("PR_TIP ") and len(line.split()) >= 2:
            out["pr_tip"] = line.split()[1]         # the PR commit fetched and built (full SHA)
        elif line.startswith("MERGED_ONTO ") and len(line.split()) >= 2:
            out["merged_onto"] = line.split()[1]
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
        elif line.startswith("RESULT_ACC_POSITIONS "):
            parts = line.split()
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                out["acc_n"], out["acc_expected"] = int(parts[1]), int(parts[2])
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
        elif line.split()[:1] == ["MUSE_FAILED"]:
            out["muse_failed"] = True
            if arb.failed_rc(line) == 137:
                out["muse_failed_box"] = True
        elif line.startswith("MUSECB "):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    muse_cb[int(parts[1])] = float(parts[2])
                except ValueError:
                    pass
        elif line.startswith("MUSECB_FAILED"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    cb_failed.add(int(parts[1]))
                    if arb.failed_rc(line) == 137:
                        cb_failed_box.add(int(parts[1]))
                except ValueError:
                    pass
        elif line.startswith("GUARD36 "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    guard36[int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.split()[:1] == ["GUARD36_FAILED"]:
            out["guard36_failed"] = True
            if arb.failed_rc(line) == 137:
                out["guard36_failed_box"] = True
        elif line.startswith("GUARDMO "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    guardmo[int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.split()[:1] == ["GUARDMO_FAILED"]:
            out["guardmo_failed"] = True
            if arb.failed_rc(line) == 137:
                out["guardmo_failed_box"] = True
        elif line.strip() == "GUARDMO_UNAVAILABLE":
            out["guardmo_unavailable"] = True
        elif line.startswith("GUARDUN "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    guardun[int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.split()[:1] == ["GUARDUN_FAILED"]:
            out["guardun_failed"] = True
            if arb.failed_rc(line) == 137:
                out["guardun_failed_box"] = True
        elif line.strip() == "GUARDUN_UNAVAILABLE":
            out["guardun_unavailable"] = True
        elif line.startswith("GUARDBN "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    guardbn[int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.split()[:1] == ["GUARDBN_FAILED"]:
            out["guardbn_failed"] = True
            if arb.failed_rc(line) == 137:
                out["guardbn_failed_box"] = True
        elif line.strip() == "GUARDBN_UNAVAILABLE":
            out["guardbn_unavailable"] = True
    out["guard36"] = guard36
    out["guardmo"] = guardmo
    out["guardun"] = guardun
    out["guardbn"] = guardbn
    out["muse"] = muse
    out["muse_cb"] = muse_cb
    out["cb_failed"] = sorted(cb_failed)
    out["cb_failed_box"] = sorted(cb_failed_box)
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
    # The tip that was measured, not whatever pull/<n>/head points at by now.
    tip = res.get("pr_tip") or ""
    target = shlex.quote(tip) if arb._FULL_SHA_RE.match(tip) else "FETCH_HEAD"
    checkout_cmd = (
        f"cd {shlex.quote(REMOTE_REPO)} && "
        f"git fetch -q origin {shlex.quote(pr_ref)} && git checkout -qf {target}"
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
    r = _ssh_run_resilient(host, port, _remote_script("main", role="main"), "main run")
    if r.returncode != 0:
        tail = arb.failure_excerpt(r.stdout, r.stderr, _EXPLICIT_FAIL_MARKERS)
        crash = arb.failure_cause(_crash_reason(r.stdout, r.stderr), r.stdout, r.stderr)
        reason = "main run failed" + (f" — {crash}" if crash else " (no crash diagnostic captured, possible hard kill — retried once)")
        return {"ok": False, "reason": reason, "log": tail}
    main = _parse_remote(r.stdout or "")
    # Fail the ROUND, not the PR, when the baseline is incomplete: comparing a PR against a
    # partial baseline silently turns a missing context into a "regression".
    # Decode AND prefill at every scored context: a zero on main would REJECT (and close) every PR.
    missing = [SCORED_CTX_LABEL[c] for c in SCORED_CTXS
               if not (((main.get("muse") or {}).get(c) or {}).get("decode")
                       and ((main.get("muse") or {}).get(c) or {}).get("prefill"))]
    if main.get("muse_failed") or missing:
        return {"ok": False,
                "reason": "main bench missing Muse Glimmer measurements at ctx " + ",".join(missing or ["(sweep failed)"]),
                "log": (r.stdout or "")[-1500:]}
    # Every guard must have measured something unless its checkpoint is absent. Otherwise every PR
    # in the round is measured in full only to be deferred for the missing guard.
    unguarded = [key for key in ("guard36", "guardmo", "guardun", "guardbn")
                 if not main.get(f"{key}_unavailable") and not arb.guard_measured(main.get(key))]
    if unguarded:
        return {"ok": False, "reason": "main measured nothing for guard(s) " + ", ".join(unguarded),
                "log": (r.stdout or "")[-1500:]}
    # The accuracy gate is absolute (against llama.cpp). If main misses it, the box, the reference
    # or main itself is off, and every PR would be REJECTed -- and closed -- for it: skip the round.
    if (main.get("top1") is None or main.get("kl") is None
            or main["top1"] < ACC_TOP1_BAR or main["kl"] > ACC_KL_BAR
            or (main.get("acc_n") is not None and main["acc_n"] < main["acc_expected"])):
        return {"ok": False, "reason": f"main misses its own accuracy gate against llama.cpp "
                                       f"(top1={main.get('top1')} kl={main.get('kl')} "
                                       f"positions={main.get('acc_n')}/{main.get('acc_expected')})",
                "log": (r.stdout or "")[-1500:]}
    if not main.get("sha"):
        return {"ok": False, "reason": "main run did not report its commit", "log": (r.stdout or "")[-1500:]}
    main["ok"] = True
    return main


def eval_museglimmer_on_box(host, port, pr_ref: str, main: dict):
    """Run the PR ref's speed+accuracy script on the same box and compare against `main`, an
    already-measured baseline shared across every PR in the round (see measure_main_baseline)."""
    print(f">> Muse Glimmer eval on box: PR ref={pr_ref}")
    r = _ssh_run_resilient(host, port, _remote_script(pr_ref, role="pr", onto=main.get("sha")), "PR run")
    if r.returncode != 0:
        touched = arb.harness_touched_line(r.stdout, r.stderr)
        if touched:
            # The tip the box fetched edits the measuring harness (pushed after this round listed the
            # PR's files): not measured, nothing posted; the next round's file check skips it.
            return {"ok": False, "harness": True, "reason": touched,
                    "pr_tip": _parse_remote(r.stdout or "").get("pr_tip")}
        conflict = arb.merge_conflict_line(r.stdout, r.stderr)
        if conflict:
            # Does not merge onto the main this round measured: a rebase, not a verdict.
            return {"ok": False, "conflict": True, "reason": conflict,
                    "pr_tip": arb.merge_conflict_tip(r.stdout, r.stderr)}
        ahead = arb.base_ahead_line(r.stdout, r.stderr)
        if ahead:
            # Rebased onto a main newer than this round's baseline: measured next round, onto it.
            return {"ok": False, "retry": True, "log": "", "reason": ahead}
        tail = arb.failure_excerpt(r.stdout, r.stderr, _EXPLICIT_FAIL_MARKERS)
        crash = arb.failure_cause(_crash_reason(r.stdout, r.stderr), r.stdout, r.stderr)
        reason = "PR speed/accuracy run failed" + (f" — {crash}" if crash else " (no crash diagnostic captured, possible hard kill — retried once)")
        # "box": a fault recurring at one commit is charged to the PR after BOX_FAULT_STRIKES rounds.
        return {"ok": False, "retry": _is_box_fault(r.stdout, r.stderr), "strike_key": "box", "reason": reason,
                "log": tail, "pr_tip": _parse_remote(r.stdout or "").get("pr_tip")}
    pr = _parse_remote(r.stdout or "")
    # A missing PR-side context is NOT treated as an infra failure here -- it is scored as a
    # regression by tier_from_gain (cur=0 against a real main baseline), which is the fail-closed
    # direction. Only a wholesale sweep failure is reported as a run failure.
    t1, kl = pr.get("top1"), pr.get("kl")
    output_wrong = t1 is not None and kl is not None and (t1 < ACC_TOP1_BAR or kl > ACC_KL_BAR)
    if pr.get("muse_failed_box") and not output_wrong:
        # The speed sweep was SIGKILLed (the host OOM killer): the box's, like a build the compiler
        # cannot finish. Charged to the PR after BOX_FAULT_STRIKES rounds at one commit.
        return {"ok": False, "pr_tip": pr.get("pr_tip"), "retry": True, "strike_key": "sweep-box", "log": "",
                "reason": "the PR's Muse Glimmer speed sweep was killed (exit 137) — infra"}
    if pr.get("muse_failed") or not (pr.get("muse") or {}):
        # Lead with the accuracy result when it was measured and failed. A sweep that dropped
        # contexts is usually the SYMPTOM; "your change makes the model emit garbage" is the
        # cause, and it is already in hand. #1037 was told only "produced no measurements" while
        # the same log carried top-1 0.000 / KL 11.29 — sending its author hunting for a harness
        # problem instead of a correctness bug in their own diff.
        why = "PR bench produced no Muse Glimmer measurements"
        t1, kl = pr.get("top1"), pr.get("kl")
        if t1 is not None and kl is not None and (t1 < ACC_TOP1_BAR or kl > ACC_KL_BAR):
            why = (f"PR output is incorrect — top-1 {t1:.3f} (bar >={ACC_TOP1_BAR}), "
                   f"KL {kl:.4f} (bar <={ACC_KL_BAR}); the incomplete speed sweep is a symptom "
                   f"of that, not a harness fault")
        return {"ok": False, "pr_tip": pr.get("pr_tip"), "reason": why, "log": (r.stdout or "")[-1500:]}
    if "top1" not in pr or "kl" not in pr:
        return {"ok": False, "pr_tip": pr.get("pr_tip"), "reason": "PR run missing accuracy METRIC line", "log": (r.stdout or "")[-1500:]}
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

    # Concurrent decode (issue #1026). An axis is scored only when BOTH sides produced a positive
    # measurement; a point either run could not produce is DROPPED, not scored as a regression.
    # These are five independent processes, any one of which can fail on its own (OOM at the
    # widest concurrency is the expected case -- at c=32 the KV pool for 33 sequences leaves the
    # card with almost nothing, and that is a property of the box, not of the PR). Reading a
    # missing run as 0.0 would hard-REJECT an innocent PR, since every axis is also a
    # no-regression floor.
    cb_skipped = []
    for conc in CB_CONCS:
        name = CB_DIM_FOR[conc]
        pr_v, main_v = _cb(pr, conc), _cb(main, conc)
        if pr_v is None or main_v is None:
            cb_skipped.append(f"{name} (pr={'-' if pr_v is None else f'{pr_v:.1f}'} "
                              f"main={'-' if main_v is None else f'{main_v:.1f}'})")
            continue
        lab, dlt, ok, why = tier_from_gain(pr_v, main_v, metric=name)
        scored.append({"dim": name, "label": lab, "delta": dlt, "passed": ok, "reason": why})
    if cb_skipped:
        print(f">> concurrent-decode axes not scored this round (no paired measurement; a width main "
              f"measured and the PR did not is judged below): {', '.join(cb_skipped)}")

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
    # Every position of the stream, not just the ones the PR's dump has: the compare skips the rest.
    covered = pr.get("acc_n") is None or pr["acc_n"] >= pr["acc_expected"]
    accuracy_ok = pr_top1 >= ACC_TOP1_BAR and pr_kl <= ACC_KL_BAR and covered
    reason = speed_reason
    if not accuracy_ok:
        # Accuracy gate is a hard REJECT regardless of speed — same discipline as
        # pr_dflash_bot.py's SPEC_AGREE veto: a fast-but-wrong PR is worthless on a still-fragile
        # architecture, and speed alone can't prove correctness.
        acc_reason = (f"accuracy gate failed: top1={pr_top1:.3f} (bar >={ACC_TOP1_BAR}) "
                      f"kl={pr_kl:.4f} (bar <={ACC_KL_BAR})"
                      + ("" if covered else f"; the PR's score dump covers {pr['acc_n']} of "
                                            f"{pr['acc_expected']} positions"))
        reason = f"{acc_reason} | speed: {speed_reason}"
        label = "REJECT"
        passed = False

    killed = [k for k in ("guard36", "guardmo", "guardun", "guardbn")
              if pr.get(f"{k}_failed_box") and main.get(k)]
    killed += [f"cb-decode@c{c}" for c in (pr.get("cb_failed_box") or [])
               if _cb(main, c) is not None and _cb(pr, c) is None]
    if killed and accuracy_ok:
        # A guard sweep SIGKILLed on the PR build (the host OOM killer): the box's, not a regression.
        # Beside a failed accuracy gate, which a busy box cannot fake, the REJECT is posted instead.
        return {"ok": False, "pr_tip": pr.get("pr_tip"), "retry": True, "strike_key": "guard-box", "log": "",
                "reason": f"{', '.join(killed)} was killed on the PR build (exit 137, the OOM killer) — infra"}
    # Beside that failed accuracy gate, a guard the OOM killer took measured nothing: it is reported
    # as not measured, not as a regression (the close comment used to name it as the failure).
    guards_killed = []

    def _killed_only(key, ok, problems):
        return (not ok and key in killed and bool(problems)
                and all(p.endswith("measurement unavailable") or "PR measurement missing/zero" in p
                        for p in problems))

    mo_ok, mo_problems = check_modelopt_guard(pr, main)
    if pr.get("guardmo_unavailable") or main.get("guardmo_unavailable"):
        # Absent checkpoint is a SKIP, not a REJECT -- but say so, so a round that guarded
        # nothing never reads as a round that guarded successfully.
        mo_ok, mo_problems = True, []
        print(">> modelopt guard SKIPPED — checkpoint not installed (MODELOPT_MODEL_DIR)")
    un_ok, un_problems = check_unsloth_guard(pr, main)
    if pr.get("guardun_unavailable") or main.get("guardun_unavailable"):
        # Same SKIP-not-REJECT handling as the ModelOpt guard, and the same announcement.
        un_ok, un_problems = True, []
        print(">> unsloth qwen3.8 guard SKIPPED — checkpoint not installed (QWEN38_MODEL_DIR)")
    q36_ok, q36_problems = check_q36_guard(pr, main)
    if _killed_only("guardmo", mo_ok, mo_problems):
        guards_killed.append("modelopt")
        mo_ok, mo_problems = True, []
    if _killed_only("guardun", un_ok, un_problems):
        guards_killed.append("unsloth")
        un_ok, un_problems = True, []
    if _killed_only("guard36", q36_ok, q36_problems):
        guards_killed.append("qwen3.6")
        q36_ok, q36_problems = True, []
    # A guard that measured NOTHING is infra, not a regression: no verdict, re-evaluated next
    # round. Only the Ternary-Bonsai guard below did this; the other three REJECTed -- and, a REJECT
    # being a closing verdict, closed the PR -- over a measurement that never happened (the Qwen3.8
    # bot closed #1112 and #1114 that way on 2026-09-18).
    for g_ok, g_problems in ((mo_ok, mo_problems), (un_ok, un_problems), (q36_ok, q36_problems)):
        unavailable = [p for p in g_problems if p.endswith("measurement unavailable")]
        if not g_ok and unavailable and len(unavailable) == len(g_problems) and label != "REJECT":
            # (Beside a REJECT already decided, it is posted with that REJECT instead.)
            return {"ok": False, "pr_tip": pr.get("pr_tip"), "retry": True, "strike_key": "guard-unmeasured", "log": "",
                    "reason": "; ".join(unavailable) + " — infra, not a regression; the PR is "
                              "re-evaluated next round rather than rejected"}
    if not mo_ok:
        mo_reason = "modelopt no-regression guard failed: " + "; ".join(mo_problems[:6])
        reason = f"{mo_reason} | {reason}"
        label = "REJECT"
        passed = False

    if not un_ok:
        un_reason = "unsloth qwen3.8 no-regression guard failed: " + "; ".join(un_problems[:6])
        reason = f"{un_reason} | {reason}"
        label = "REJECT"
        passed = False

    bn_ok, bn_problems = check_bonsai_guard(pr, main)
    bn_skipped = bool(pr.get("guardbn_unavailable") or main.get("guardbn_unavailable"))
    if bn_skipped:
        bn_ok, bn_problems = True, []
        print(">> ternary-bonsai guard SKIPPED — GGUF not installed (BONSAI_GGUF)")
    if _killed_only("guardbn", bn_ok, bn_problems):
        guards_killed.append("bonsai")
        bn_ok, bn_problems = True, []
    if not bn_ok:
        # Measured nothing -> infra: no verdict, re-evaluated next round (module docstring pt. 3).
        unavailable = [p for p in bn_problems if p.endswith("measurement unavailable")]
        if unavailable and len(unavailable) == len(bn_problems) and label != "REJECT":
            # retry: nothing is posted. Without it apply_result still wrote eval-museglimmer:REJECT.
            return {"ok": False, "pr_tip": pr.get("pr_tip"), "retry": True, "strike_key": "guard-unmeasured", "log": "",
                    "reason": "; ".join(unavailable) + " — infra, not a regression; the PR is "
                              "re-evaluated next round rather than rejected"}
        reason = "ternary-bonsai no-regression guard failed: " + "; ".join(bn_problems[:6]) + f" | {reason}"
        label = "REJECT"
        passed = False

    if not q36_ok:
        # Same hard-REJECT discipline as the accuracy gate: a Muse Glimmer PR that silently
        # regresses Qwen3.6 via shared code (qwen35.cpp/inference_engine.cpp) is unmergeable
        # regardless of its own speed/accuracy result — see module docstring pt. 3.
        q36_reason = "qwen3.6 no-regression guard failed: " + "; ".join(q36_problems[:6])
        reason = f"{q36_reason} | {reason}"
        label = "REJECT"
        passed = False

    # A concurrent width main measured this round that the PR build could not complete. It used to be
    # dropped as "no paired measurement" -- so a PR that crashed the packed multi-row decode at every
    # width, the path these axes exist to watch, could still score XL and auto-merge. Alone it is a
    # REJECT judged over two rounds on the same commit (STRIKES_TO_REJECT), as in pr_bonsai_bot.py.
    cb_pr_missing = [c for c in CB_CONCS if _cb(main, c) is not None and _cb(pr, c) is None]
    cb_strike = None
    if cb_pr_missing:
        why = (f"concurrent decode at {','.join(f'c{c}' for c in cb_pr_missing)} did not complete on the PR "
               f"build, while main measured it this round")
        if label != "REJECT":          # after every guard: alone, it is the soft case
            cb_strike = "cb"
        reason = f"{why} | {reason}"
        label = "REJECT"
        passed = False

    res = {
        "ok": True,
        "strike_key": cb_strike,
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
        "cb_pr": (pr.get("muse_cb") or {}),
        "cb_main": (main.get("muse_cb") or {}),
        "guardmo_skipped": bool(pr.get("guardmo_unavailable") or main.get("guardmo_unavailable")),
        "unsloth_guard_ok": un_ok,
        "unsloth_guard_problems": un_problems,
        "guardun_skipped": bool(pr.get("guardun_unavailable") or main.get("guardun_unavailable")),
        "bonsai_guard_ok": bn_ok,
        "bonsai_guard_problems": bn_problems,
        "guardbn_skipped": bn_skipped,
        "pr_top1": pr_top1,
        "pr_kl": pr_kl,
        "pr_ppl_spark": pr.get("ppl_spark"),
        "pr_ppl_llama": pr.get("ppl_llama"),
        "main_top1": main.get("top1"),
        "main_kl": main.get("kl"),
        "accuracy_ok": accuracy_ok,
        "q36_guard_ok": q36_ok,
        "q36_guard_problems": q36_problems,
        "guards_killed": guards_killed,
        "q36_guard": pr.get("guard36"),
        "q36_guard_main": main.get("guard36"),
        "pr_head": pr.get("head"),
        "main_head": main.get("head"),
        "pr_tip": pr.get("pr_tip"),
        "merged_onto": pr.get("merged_onto"),
        "onto": main.get("sha"),       # the full main commit this verdict was measured against
    }
    # A receipt for a measurement that has happened, never a gate on it: an exception here must not
    # discard `res` (the Qwen3.8 bot lost a measured +27.7% on #832 to exactly that).
    try:
        polaris = collect_polaris_attestation(host, port, res, pr_ref)
        if polaris:
            res["polaris"] = polaris
    except Exception as e:
        print(f">> Polaris attestation failed ({type(e).__name__}: {e}) — keeping the measurement")
    return res


def _ctx_list_str() -> str:
    """"128/512/4k/16k/32k/64k" from SCORED_CTXS. Derived rather than written out: the PR comment
    twice named a context list that had gone stale behind the constant."""
    return "/".join(SCORED_CTX_LABEL[c] for c in SCORED_CTXS)


def _cb_list_str() -> str:
    """'c2/c4/c8/c16/c32' — the concurrency points, formatted like _ctx_list_str()'s contexts."""
    return "/".join(f"c{c}" for c in CB_CONCS)


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
    out = "\n".join(rows) + "\n\n"

    # Concurrent decode, its own table -- different units (aggregate tok/s over N in-flight
    # requests) and a different x-axis (concurrency, not context), so folding it into the ctx
    # table above would mislabel both. Rendered from scored_dims for the reason the docstring
    # gives: a hard-coded row list silently goes stale the next time the matrix grows.
    cb_pr, cb_main = res.get("cb_pr") or {}, res.get("cb_main") or {}
    cb_rows = []
    for conc in CB_CONCS:
        d = next((x for x in dims if x["dim"] == CB_DIM_FOR[conc]), None)
        if not d:
            continue
        mv, pv = cb_main.get(conc), cb_pr.get(conc)
        flag = " **REJECT**" if d["label"] == "REJECT" else ""
        cb_rows.append(f"| c{conc} | {float(mv or 0):.2f} | {float(pv or 0):.2f} | "
                       f"{d['delta']:+.1f}%{flag} |")
    if cb_rows:
        out += ("**Concurrent decode** — aggregate tok/s with N requests in flight\n\n"
                "| concurrency | main | PR | delta |\n|---|---|---|---|\n"
                + "\n".join(cb_rows) + "\n\n")
    # Name the axes that were requested but produced no paired measurement, so a reader is never
    # left wondering why c32 is missing rather than zero.
    missing = [f"c{c}" for c in CB_CONCS
               if not any(x["dim"] == CB_DIM_FOR[c] for x in dims)]
    if missing:
        out += (f"<sub>Concurrency {', '.join(missing)} not scored this round — no paired "
                f"measurement. A width main could not measure is dropped; one only the PR build "
                f"could not complete is named in the verdict above.</sub>\n\n")
    return out


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
        "unsloth_guard_ok": res.get("unsloth_guard_ok"),
        "unsloth_guard_skipped": res.get("guardun_skipped"),
        "bonsai_guard_ok": res.get("bonsai_guard_ok"),
        "bonsai_guard_skipped": res.get("guardbn_skipped"),
        # WHICH axis produced delta_pct. Necessary now that the tier comes from many axes while the
        # marker still carries only the 128 numbers for the dashboard: without this a reader sees
        # a headline delta that does not match either number next to it (e.g. +3900% from
        # prefill@4k printed beside a flat decode@128).
        "best_dim": res.get("best_dim"),
        # The whole matrix, so a consumer that wants more than the 128 pair does not have to
        # re-scrape the rendered table.
        "dims": {d["dim"]: {"delta": d["delta"], "label": d["label"]}
                 for d in (res.get("scored_dims") or [])},
    }
    if not res.get("ok"):
        # A failed run that reaches here is the PR's (box faults return earlier, posting nothing):
        # recorded for this commit, so it is not rebuilt, re-run and re-posted every round.
        meta["label"] = "REJECT"
    marker = (
        f"<!-- sparkinfer-museglimmer-eval:{EVAL_SCHEMA_VERSION}:{commit} "
        f"{json.dumps(meta, separators=(',', ':'))} -->"
    )
    if not res.get("ok"):
        return (
            f"{marker}\n## sparkinfer museglimmer auto-eval — error\n\n"
            f"**reason:** `{res.get('reason')}`\n\n"
            f"<details><summary>log tail</summary>\n\n```\n{(res.get('log') or '')[:1800]}\n```\n</details>\n\n"
            f"<sub>The PR is built merged onto the round's `main`, not on its own branch: an error in a file the PR does not touch usually means it needs a rebase onto current `main`. Recorded for this commit; push a fix and it is evaluated again.</sub>\n"
        )
    lab = res["label"]
    if res.get("accuracy_ok"):
        acc_row = (f"| accuracy gate | ✅ top1={res.get('pr_top1', 0):.3f} "
                    f"(bar >={ACC_TOP1_BAR}) · KL={res.get('pr_kl', 0):.4f} (bar <={ACC_KL_BAR}) |\n")
    else:
        acc_row = (f"| accuracy gate | ❌ **FAILED** — top1={res.get('pr_top1', 0):.3f} "
                    f"(bar >={ACC_TOP1_BAR}) · KL={res.get('pr_kl', 0):.4f} (bar <={ACC_KL_BAR}) — "
                    "**verdict forced to REJECT regardless of speed** |\n")
    killed = res.get("guards_killed") or []
    if "qwen3.6" in killed:
        q36_row = "| qwen3.6 guard | ⚠️ NOT MEASURED — its sweep was killed on the PR build (exit 137, the host OOM killer); the REJECT is the accuracy gate's |\n"
    elif res.get("q36_guard_ok"):
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
    elif "modelopt" in killed:
        mo_row = "| modelopt guard | ⚠️ NOT MEASURED — its sweep was killed on the PR build (exit 137, the host OOM killer); the REJECT is the accuracy gate's |\n"
    elif res.get("modelopt_guard_ok"):
        mo_row = "| modelopt guard | ✅ no regression (decode+prefill @ 32k, Qwen3.8-27B NVFP4) |\n"
    else:
        mo_problems = "; ".join((res.get("modelopt_guard_problems") or [])[:4])
        mo_row = (f"| modelopt guard | ❌ **FAILED** — {mo_problems} — "
                  "**verdict forced to REJECT regardless of speed/accuracy** |\n")
    if res.get("guardun_skipped"):
        un_row = ("| unsloth qwen3.8 guard | ⚠️ SKIPPED — checkpoint not installed on the box "
                  "(`QWEN38_MODEL_DIR`); shared-code regressions on it were NOT checked |\n")
    elif "unsloth" in killed:
        un_row = "| unsloth qwen3.8 guard | ⚠️ NOT MEASURED — its sweep was killed on the PR build (exit 137, the host OOM killer); the REJECT is the accuracy gate's |\n"
    elif res.get("unsloth_guard_ok"):
        un_row = "| unsloth qwen3.8 guard | ✅ no regression (decode+prefill @ 32k, unsloth Qwen3.8-27B NVFP4) |\n"
    else:
        un_problems = "; ".join((res.get("unsloth_guard_problems") or [])[:4])
        un_row = (f"| unsloth qwen3.8 guard | ❌ **FAILED** — {un_problems} — "
                  "**verdict forced to REJECT regardless of speed/accuracy** |\n")
    if res.get("guardbn_skipped"):
        bn_row = ("| ternary-bonsai guard | ⚠️ SKIPPED — GGUF not installed on the box "
                  "(`BONSAI_GGUF`); shared-code regressions on it were NOT checked |\n")
    elif "bonsai" in killed:
        bn_row = "| ternary-bonsai guard | ⚠️ NOT MEASURED — its sweep was killed on the PR build (exit 137, the host OOM killer); the REJECT is the accuracy gate's |\n"
    elif res.get("bonsai_guard_ok"):
        bn_row = "| ternary-bonsai guard | ✅ no regression (decode+prefill @ 128 and 32k, Ternary-Bonsai-2-27B) |\n"
    else:
        bn_problems = "; ".join((res.get("bonsai_guard_problems") or [])[:4])
        bn_row = (f"| ternary-bonsai guard | ❌ **FAILED** — {bn_problems} — "
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
        f"| scored at | decode + prefill @ {_ctx_list_str()} · concurrent decode @ "
        f"{_cb_list_str()} — {len(SCORING_DIMS)} axes, each also a regression floor; the label is the best |\n"
        f"| tier from | `{res.get('best_dim') or '?'}` ({res.get('delta_pct', 0):+.1f}%) |\n"
        f"{acc_row}"
        f"{q36_row}"
        f"{mo_row}"
        f"{un_row}"
        f"{bn_row}"
        f"| PPL sparkinfer / llama.cpp | {res.get('pr_ppl_spark') or '?'} / {res.get('pr_ppl_llama') or '?'} |\n"
        f"{polaris_row}"
        f"| commit | `{commit[:9]}`"
        + (f", measured merged onto `main` `{res['merged_onto']}` (this round's baseline)"
           if res.get("merged_onto") else "")
        + " |\n\n"
        f"{_matrix_table(res)}"
        f"{res.get('reason') or ''}\n\n"
        f"<sub>Scored on the pinned eval box vs same-box `origin/main` — AR decode AND prefill at "
        f"ctx {_ctx_list_str()}, plus concurrent decode (aggregate tok/s) at "
        f"{_cb_list_str()}; ANY axis regressing is a hard REJECT, but "
        f"otherwise the reported label is the **best** measured delta across all "
        f"{len(SCORING_DIMS)} — "
        "a PR that improves just one, with the rest flat, still earns credit for that. "
        "This is informational, not a judgment on your PR: a `none` label just means no "
        "measurable Muse Glimmer speedup was verified on any scored axis, which is expected and "
        "fine if that isn't what your change is about. "
        "Correctness gated against a live llama.cpp reference on the same GGUF. Also gated on "
        "cross-model no-regression guards (decode+prefill, same box vs main): Qwen3.6-35B-A3B, "
        "the ModelOpt and unsloth Qwen3.8-27B NVFP4 checkpoints at 32k, and Ternary-Bonsai-2-27B "
        "at 128 and 32k — "
        "Muse Glimmer PRs can touch code shared with other models. "
        "Automated. The round's best-scoring PR may be auto-merged as `merge-first` once every "
        "gate above passes; a separate comment says so explicitly when that happens.</sub>\n"
    )


def auto_merge_ok_museglimmer(repo, num, require_merge_first=True, ranking_loss_ok=False):
    """Can this PR be merged now? With require_merge_first=False: may it be MADE merge-first -- the
    same test minus that label, so a winner whose merge would be refused cannot hold merge-first
    while every other speedup PR is pushed to needs-rebase (pr_bonsai_bot.py, 2026-09-26).
    ranking_loss_ok: the bot's own museglimmer-needs-rebase does not count -- reconcile's call for a PR
    sent there only for losing an earlier ranking (_waits_for_the_winner)."""
    try:
        info = json.loads(arb.gh([
            "pr", "view", str(num), "-R", repo, "--json",
            "state,isDraft,labels,author,mergeable,files,changedFiles,headRefOid,baseRefName",
        ]).stdout or "{}")
    except json.JSONDecodeError:
        info = None
    if not isinstance(info, dict) or not info:
        return False, arb.PR_UNREADABLE
    if info.get("state") != "OPEN" or info.get("isDraft"):
        return False, "not an open, non-draft PR"
    labs = {l["name"] for l in info.get("labels", [])}
    tiers = {l.split(":", 1)[1] for l in labs if l.startswith(EVAL_PREFIX)}
    if not (tiers & SPEEDUP_LABELS):
        return False, "no verified eval-museglimmer:speedup label"
    if require_merge_first and MUSEGLIMMER_MERGE_FIRST not in labs:
        return False, "not museglimmer-merge-first"
    # Only the exact commit this bot scored. The labels survive a push made after the verdict, and a
    # re-measurement that fails on the box side posts nothing -- so the label alone could merge
    # unmeasured code (pr_bonsai_bot.py has checked this since it was written).
    head = info.get("headRefOid") or ""
    scored = _load_scores().get(str(num)) or {}
    if not head or scored.get("commit") != head:
        return False, (f"head {head[:9] or '?'} is not the commit last scored "
                       f"({(scored.get('commit') or 'none')[:9]})")
    if scored.get("label") not in SPEEDUP_LABELS or not scored.get("pass"):
        return False, f"recorded verdict for {head[:9]} is {scored.get('label')} (pass={scored.get('pass')})"
    # A REJECT from any other bot is a measured harm on another model.
    if any(l.endswith((":REJECT", ":REJECT" + arb.NOISE_PARK_SUFFIX)) for l in labs if l.startswith("eval")):
        return False, "carries a REJECT from another eval bot"
    blocked = labs & (AUTOMERGE_BLOCK - ({MUSEGLIMMER_NEEDS_REBASE} if ranking_loss_ok else set()))
    if blocked:
        return False, f"blocking label(s): {', '.join(sorted(blocked))}"
    author = (info.get("author") or {}).get("login", "")
    if author.lower() in arb.load_denylist():
        return False, f"author {author} is blocked"
    if arb.author_penalty_until(author):
        return False, f"author {author} is under penalty"
    if (info.get("baseRefName") or "main") != "main":
        return False, f"based on {info.get('baseRefName')}, not main"
    listed = info.get("files") or []
    if (info.get("changedFiles") or 0) > len(listed):
        return False, f"changes {info.get('changedFiles')} files, more than GitHub lists ({len(listed)})"
    harness = [f["path"] for f in listed if any(f["path"].startswith(h) for h in HARNESS_PATHS)]
    if harness:
        # It could never have been measured with its own ruler (the box stops on HARNESS_TOUCHED);
        # whatever verdict it carries, a harness change merges by hand.
        return False, f"touches the eval harness: {', '.join(harness[:3])}"
    sens = [f["path"] for f in info.get("files", [])
            if any(f["path"].startswith(p) for p in arb.AUTOMERGE_SENSITIVE)]
    if sens:
        return False, f"touches protected paths: {', '.join(sens[:3])}"
    if arb.pr_merge_conflict(info.get("mergeable")):
        return False, "merge conflict with base"
    # GitHub reports UNKNOWN for a while after main moves; that must not cost a PR the ranking.
    if info.get("mergeable") != "MERGEABLE" and require_merge_first:
        return False, f"not cleanly mergeable ({info.get('mergeable')})"
    # Last: measured against the main that is there now. Once any bot has merged something else,
    # this PR merged onto the new main is a combination nobody measured, so it is re-measured
    # first (main()'s selection). Reconcile keeps a PR refused for this alone in the running
    # (arb.refused_only_for_stale_main), which is why no other refusal may come after it.
    return arb.fresh_against_main(repo, scored)


def try_auto_merge_museglimmer(repo, num):
    ok, reason = auto_merge_ok_museglimmer(repo, num)
    if not ok:
        print(f">> museglimmer auto-merge SKIP #{num}: {reason}")
        return False
    # Pinned to the SCORED commit, which auto_merge_ok_museglimmer just found equal to the head,
    # --admin included. A second head lookup here would pin whatever a push had made the head since.
    head = (_load_scores().get(str(num)) or {}).get("commit") or ""
    if not arb._FULL_SHA_RE.match(head):
        print(f">> museglimmer auto-merge SKIP #{num}: no scored commit to pin the merge to")
        return False
    args = ["pr", "merge", str(num), "-R", repo, "--squash", "--match-head-commit", head]
    r = arb.gh(args)
    if r.returncode != 0 and os.environ.get("SPARKINFER_AUTOMERGE_ADMIN", "1") == "1":
        err = ((r.stderr or "") + (r.stdout or "")).lower()
        if "not mergeable" in err or "branch policy" in err or "required" in err or "prohibited" in err:
            print(">> museglimmer auto-merge: branch policy blocked — retrying with --admin")
            r = arb.gh(args + ["--admin"])
    if r.returncode == 0:
        print(f">> MUSEGLIMMER AUTO-MERGED #{num} (museglimmer-merge-first)")
        arb.gh(["pr", "comment", str(num), "-R", repo, "--body",
                "<!-- sparkinfer-museglimmer-automerge -->\n"
                "Auto-merged as the round's `museglimmer-merge-first` winner — a verified same-box "
                "Muse Glimmer speedup over `main` on the axis the verdict names, accuracy-gated vs "
                "llama.cpp, with every cross-model guard passing."])
        return True
    print(f">> museglimmer auto-merge BLOCKED #{num}: {(r.stderr or r.stdout or '')[:200]}")
    return False


def _unmeasurable_reason(repo, pr, labs, count_gave_up=True):
    """Why main()'s selection would not measure this PR now, or None: the same filters it applies
    (keep them in step). Reconcile keeps a PR refused only for a moved main in the running solely
    when the selection will re-measure it; one it never re-measures -- no longer greenlit, now
    declared for another model, conflicting -- would hold its place, merge-first included, for ever
    while fresher PRs waited behind it."""
    if pr.get("isDraft"):
        return "draft"
    if (pr.get("baseRefName") or "main") != "main":
        return f"based on {pr.get('baseRefName')}, not main"
    if arb.HOLD_LABEL in labs:
        return "hold"
    head = (pr.get("headRefOid") or "")[:40]
    if arb.strike_count(STRIKES_FILE, pr["number"], head, "harness"):
        return "edits the eval harness (found on the box)"
    if arb.strike_count(STRIKES_FILE, pr["number"], head, "conflict"):
        # The selection skips it until a push: kept "in the running" it held merge-first for ever.
        return "does not merge onto main on the box (needs a rebase)"
    if pr.get("changedFiles") and pr["changedFiles"] > len(pr.get("files") or []):
        return "changes more files than GitHub lists"
    if count_gave_up and arb.gave_up(STRIKES_FILE, pr["number"], (pr.get("headRefOid") or "")[:40]):
        return "the bot gave up on this commit after its own errors"
    skip = arb.model_skip_reason(pr.get("body") or "", "muse")
    if skip:
        return skip
    touched = [f.get("path", "") for f in (pr.get("files") or [])]
    if any(t.startswith(h) for t in touched for h in HARNESS_PATHS):
        return "touches the eval harness"
    if arb.pr_merge_conflict(pr.get("mergeable")):
        return "merge conflict"
    status, why = arb.greenlight_status(repo, pr["number"], labs)
    # "unknown" (GitHub did not answer) keeps: demoting on a non-answer takes a real winner's place.
    return None if status in ("ok", "unknown") else f"not greenlit ({why})"


def reconcile_museglimmer_merge_labels(repo, dry_run=False):
    scores = _load_scores()
    open_prs = arb.open_prs_or_none(repo, "number,labels,isDraft,body,files,mergeable,headRefOid,baseRefName")
    if open_prs is None:
        print(">> museglimmer round: GitHub did not return the open PRs — labels left as they are")
        return
    open_labels = {p["number"]: {l["name"] for l in p["labels"]} for p in open_prs}
    open_by_num = {p["number"]: p for p in open_prs}

    merged = json.loads(arb.gh([
        "pr", "list", "-R", repo, "--state", "merged", "--label", MUSEGLIMMER_MERGE_FIRST,
        "--json", "number", "--limit", "10",
    ]).stdout or "[]")
    for m in merged:
        if not dry_run:
            arb.remove_label(repo, m["number"], MUSEGLIMMER_MERGE_FIRST)

    scored = []
    stale_first = []   # carries merge-first but can no longer win it
    stale_main = set()   # in the running, but its merge waits for a re-measure onto today's main
    main_now = None      # read once, for a needs-rebase that may only mean a lost ranking
    for num, labs in open_labels.items():
        if not dry_run:
            labs = open_labels[num] = arb.repair_own_tier(
                repo, num, labs, EVAL_PREFIX, scores.get(str(num)), (open_by_num[num].get("headRefOid") or "")[:40],
                lambda: museglimmer_evaluated_commits(repo, num))
        # A sync GitHub did not answer when this bot posted its verdict, healed -- on this bot's PRs
        # only: the retired AR bot's labels derive the generic one by another rule (the failing side).
        if not dry_run and any(l.startswith(EVAL_PREFIX) for l in labs) and arb.generic_label_out_of_sync(labs):
            arb.sync_generic_eval_label(repo, num)
        # A PR that cannot be merged -- hold, needs-rebase, penalty, any other AUTOMERGE_BLOCK
        # label, or anything else auto-merge would refuse -- must not take merge-first and push the
        # others to needs-rebase for a merge that never happens (pr_bonsai_bot.py, #1154).
        lost_only = False
        if (labs & AUTOMERGE_BLOCK) == {MUSEGLIMMER_NEEDS_REBASE}:
            # Sent to needs-rebase only for losing an earlier ranking, with its verdict still standing
            # on today's main: it stays in the running. Left out, a worse PR merged first once the
            # winner was re-measured lower, and nothing merged at all once the winner was closed or
            # held. After main moves, the rebase is its author's (CONTRIBUTING).
            if main_now is None:
                main_now = arb.current_main_sha(repo) or ""
            pr = open_by_num[num]
            lost_only = bool(main_now) and _waits_for_the_winner(pr, labs, (pr.get("headRefOid") or "")[:40], main_now)
        if labs & AUTOMERGE_BLOCK and not lost_only:
            if MUSEGLIMMER_MERGE_FIRST in labs:
                stale_first.append(num)
            continue
        tiers = {l.split(":", 1)[1] for l in labs if l.startswith(EVAL_PREFIX)}
        tier = next((t for t in tiers if t in SPEEDUP_LABELS), None)
        if not tier:
            # No speedup tier (any more): its head moved, or a re-measure found none. A merge-first
            # left here exempted it from every close and could sit beside the next winner's.
            if MUSEGLIMMER_MERGE_FIRST in labs:
                stale_first.append(num)
            continue
        ok, why = auto_merge_ok_museglimmer(repo, num, require_merge_first=False,
                                            ranking_loss_ok=lost_only)
        if not ok and why == arb.PR_UNREADABLE:
            # Not an answer: demoting on it would take merge-first from the real holder.
            print(f">> museglimmer round: GitHub did not return #{num} — labels left as they are")
            return
        if not ok and arb.refused_only_for_stale_main(why):
            # Nothing but a moved main stands in the way: it keeps its place and is re-measured
            # (main()'s selection); only the merge waits for that. Demoting it here let the stale
            # close shut a verified winner the bot itself owed a measurement. Only if the
            # selection will re-measure it, though.
            blocker = _unmeasurable_reason(repo, open_by_num[num], labs)
            if blocker:
                ok, why = False, f"{why}; not re-measured: {blocker}"
            else:
                print(f">> museglimmer round: #{num} stays in the running, merge waits ({why})")
                stale_main.add(num)
        if not ok and num not in stale_main:
            print(f">> museglimmer round: #{num} cannot be merge-first ({why})")
            if MUSEGLIMMER_MERGE_FIRST in labs:
                stale_first.append(num)
            continue
        entry = scores.get(str(num)) or {}
        if entry.get("label") not in SPEEDUP_LABELS:
            if tier not in SPEEDUP_LABELS:
                continue
            entry = {"label": tier, "delta_pct": entry.get("delta_pct") or 0}
        scored.append((num, float(entry.get("delta_pct") or 0), entry.get("label") or tier))

    # A PR that can merge now outranks one still waiting for its re-measure.
    scored.sort(key=lambda x: (x[0] not in stale_main, x[1]), reverse=True)
    if not dry_run:
        for num in stale_first:
            arb.remove_label(repo, num, MUSEGLIMMER_MERGE_FIRST)
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
        # Nothing merges this round while the winner waits for its re-measure: no one needs a rebase.
        if winner not in stale_main:
            arb.add_label(repo, num, MUSEGLIMMER_NEEDS_REBASE)
        arb.remove_label(repo, num, MUSEGLIMMER_MERGE_FIRST)
    if AUTO_MERGE:
        try_auto_merge_museglimmer(repo, winner)


def upload_museglimmer_eval_log(repo, num, title, oid, res):
    """Commit the eval result (+ Polaris receipt/attestation) to sparkinfer-log, mirroring
    pr_dflash_bot.py's upload_dflash_eval_log with a museglimmer-prefixed run id."""
    try:
        arb._ensure_log_repo()
        rid = arb.eval_log_run_id(f"museglimmer-{int(num):04d}-{oid[:7]}", res.get("onto"))
        rundir = os.path.join(arb.LOG_DIR, "runs", rid)
        os.makedirs(rundir, exist_ok=True)
        polaris = res.get("polaris") or {}
        receipt = polaris.get("receipt")
        result = {
            "id": rid, "pr": int(num), "title": title,
            "url": f"https://github.com/{repo}/pull/{num}", "commit": oid[:7],
            # What was built: this commit merged onto that main (the Polaris receipt, whose schema
            # has no field for it, names the PR commit alone).
            "measured_onto": res.get("onto"),
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
            "modelopt_guard_ok": res.get("modelopt_guard_ok"), "unsloth_guard_ok": res.get("unsloth_guard_ok"),
            "unsloth_guard_problems": res.get("unsloth_guard_problems"),
            "bonsai_guard_ok": res.get("bonsai_guard_ok"),
            "bonsai_guard_problems": res.get("bonsai_guard_problems"),
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
        push = subprocess.run(["git", "-C", arb.LOG_DIR, "push", "-q"], check=False, timeout=300)
        if push.returncode != 0:
            print(f">> museglimmer eval-log push failed (rc={push.returncode})")
            return None
        url = arb.LOG_PAGE + rid
        print(f">> museglimmer eval log: {url}")
        return url
    except Exception as e:
        print(f">> museglimmer eval-log upload failed: {e}")
        return None


def _remeasure_state(repo, num, head, labs, main_now):
    """Is an already-evaluated head owed a re-measure onto today's main? True only when auto_merge_ok_museglimmer
    refuses it for a moved main and nothing else: a passing speedup verdict for this head, measured
    against a main other than today's. Merging it now would ship a combination nobody measured. A PR
    the merge gate refuses for anything else (a block label, another bot's REJECT, a protected path,
    a penalty, a conflict) is not re-measured every time main moves, since it could not merge anyway.
    None when GitHub did not answer: unknown, which must neither re-measure nor close."""
    entry = _load_scores().get(str(num)) or {}
    if (entry.get("commit") != head or entry.get("label") not in SPEEDUP_LABELS
            or not entry.get("pass") or labs & AUTOMERGE_BLOCK):
        return False
    if not main_now:
        return None
    if not arb.scored_against_stale_main(entry, main_now):
        return False
    ok, why = auto_merge_ok_museglimmer(repo, num, require_merge_first=False)
    if not ok and why == arb.PR_UNREADABLE:
        return None
    return not ok and arb.refused_only_for_stale_main(why)


def _remeasure_against_new_main(repo, num, head, labs, main_now):
    """_remeasure_state, for the selection: only a definite yes spends GPU on it."""
    return _remeasure_state(repo, num, head, labs, main_now) is True


def apply_result(repo, num, commit, res, title="", dry_run=False, pr_body=""):
    if not res.get("ok") and res.get("harness"):
        print(f"PR #{num}: {res.get('reason')} — edits the eval harness, not evaluated")
        if not dry_run:
            # Remembered for this head: GitHub lists at most 100 files, so the selection's own check
            # can miss the edit and the PR would be measured (and stopped) again every round.
            arb.record_strike(STRIKES_FILE, num, commit, "harness")
        return
    if not res.get("ok") and res.get("conflict"):
        # No verdict and no REJECT label: the PR does not merge onto the main this round measured,
        # which says nothing about its change. Same outcome as the pre-GPU merge-conflict check.
        print(f"PR #{num}: {res.get('reason')} — museglimmer-needs-rebase, no verdict")
        if not dry_run:
            arb.add_label(repo, num, MUSEGLIMMER_NEEDS_REBASE)
            # Remembered for this head: GitHub may keep calling it mergeable, and the label would be
            # dropped and the PR measured again every round (arb.strip_stale_verdict_labels).
            arb.record_strike(STRIKES_FILE, num, commit, "conflict")
        return
    if not res.get("ok") and res.get("retry"):
        # The box's fault (_is_box_fault, an unmeasured guard, an exception): nothing is posted and
        # no label changes; the next round measures again -- except a fault of the PR's own run
        # that recurs at one commit (while main's run passed each time), which after
        # BOX_FAULT_STRIKES rounds is posted as a failed run. Retried for ever, such a PR never got a
        # verdict, and being greenlit and unmeasured it is never stale either.
        n = (arb.record_strikes(STRIKES_FILE, num, commit, res["strike_key"])
             if res.get("strike_key") and not dry_run else 0)
        if n < arb.BOX_FAULT_STRIKES:
            print(f"PR #{num}: museglimmer eval deferred — {res.get('reason')} "
                  f"({'the bot itself' if res.get('strike_key') == 'error' else 'infra'}; re-evaluated next "
                  f"round{f', strike {n} of {arb.BOX_FAULT_STRIKES}' if n else ''})")
            return
        if res.get("strike_key") == "error":
            # The bot's own failure, not the PR's: never charged. It stops measuring this commit
            # (arb.gave_up) instead of retrying it every round for ever.
            print(f"!! PR #{num}: the bot itself failed on {commit[:9]} {n} rounds in a row "
                  f"({res.get('reason')}) — not measured again until a push; nothing posted")
            GAVE_UP.add(num)
            return
        res = dict(res, retry=False,
                   reason=f"{res.get('reason')} — {n} rounds at this commit, so it is charged to the PR")
    if res.get("ok") and res.get("strike_key") and not dry_run:
        # A REJECT whose only cause may be the box (a width the PR build could not complete): judged
        # over STRIKES_TO_REJECT rounds on the same commit; the first time nothing is posted.
        n = arb.record_strikes(STRIKES_FILE, num, commit, res["strike_key"])
        if n < STRIKES_TO_REJECT:
            print(f"PR #{num}: {res.get('reason', '').split(' | ')[0]} — strike {n} of {STRIKES_TO_REJECT}; "
                  f"nothing posted, measured again next round")
            return
    if not dry_run:
        arb.clear_strikes(STRIKES_FILE, num)
    body = format_comment(commit, res)
    label = res.get("label") if res.get("ok") else "REJECT"
    if not res.get("ok"):
        label = "REJECT"
    print(f"PR #{num}: eval-museglimmer:{label}  "
          f"decode PR={res.get('pr_decode_tps')} main={res.get('main_decode_tps')}  "
          f"prefill128 PR={res.get('pr_prefill128_pp')} main={res.get('main_prefill128_pp')}  "
          f"delta={res.get('delta_pct')}%  accuracy_ok={res.get('accuracy_ok')}  "
          f"q36_guard_ok={res.get('q36_guard_ok')}  bonsai_guard_ok={res.get('bonsai_guard_ok')}")
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
    arb.gh(["pr", "comment", str(num), "-R", repo, "--body", arb.fit_comment(body)])
    # Scores first: a run that dies in the (network) log upload must not leave a posted verdict the
    # bot can neither merge nor re-measure.
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
            "unsloth_guard_ok": res.get("unsloth_guard_ok"),
            "bonsai_guard_ok": res.get("bonsai_guard_ok"),
            "onto": res.get("onto"),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _save_scores(scores)
    else:
        # A failed run, or no delta: recorded all the same (arb.record_posted_verdict).
        arb.record_posted_verdict(_load_scores, _save_scores, num, commit, label, res)
    if res.get("ok"):
        upload_museglimmer_eval_log(repo, num, title, commit, res)
    if res.get("ok") and res.get("delta_pct") is not None:
        # AUTO-CLOSE POLICY (user decision, 2026-09-11): a `none` closes too, not only a REJECT.
        #
        # `none` means no verified speedup on any axis this bot measures. It is NOT a finding of
        # harm, so the close comment must not read like one -- and the two legitimate ways to hold
        # a PR open through it are named there: the `hold` label (for work waiting on a requested
        # evaluation axis, see CONTRIBUTING "If no evaluation measures your optimization yet") and
        # reopening with new numbers.
        #
        # The risk is on record: this bot's first live run wrongly auto-closed an unrelated PR
        # (#768, reopened + apologised) precisely because `none` is the label an off-axis PR gets.
        # `hold` and drafts are never closed -- filtered before evaluation, and read again just
        # before closing (arb.verdict_close_blocker) -- so those remain the escape hatches.
        #
        # "REJECT" is different in kind: a measured REGRESSION, an accuracy failure, or a guard
        # failure. Real, attributable harm, worth acting on whatever the PR was aiming at.
        #
        # Abandoned PRs are still handled by the age-based stale close, which is about inactivity
        # rather than about a measurement.
        #
        # Narrowed 2026-09-26: `none` closes only a PR declared for Muse Glimmer alone that no other
        # bot scored a speedup or made merge-first (arb.none_may_close). An undeclared or "Shared"
        # PR aimed at another model got this bot's `none` and was closed before that model's bot
        # measured it. A REJECT is evidence of harm and still closes.
        if label == "REJECT" or (label == "none" and arb.none_may_close(
                pr_body, arb.labels_on_or_none(repo, num), "muse", EVAL_PREFIX)):
            if not res.get("q36_guard_ok", True):
                fail_clause = "and regressed the Qwen3.6 no-regression guard (decode/prefill on shared code)"
            elif not res.get("modelopt_guard_ok", True):
                fail_clause = "and regressed the ModelOpt Qwen3.8 no-regression guard (decode/prefill @ 32k)"
            elif not res.get("unsloth_guard_ok", True):
                fail_clause = "and regressed the unsloth Qwen3.8 no-regression guard (decode/prefill @ 32k)"
            elif not res.get("bonsai_guard_ok", True):
                fail_clause = "and regressed the Ternary-Bonsai-2-27B no-regression guard (decode/prefill @ 128 or 32k)"
            elif not res.get("accuracy_ok"):
                fail_clause = "and failed the accuracy gate"
            elif res.get("prefill_regressed"):
                fail_clause = "and regressed 128-ctx prefill throughput specifically"
            elif res.get("decode_regressed"):
                fail_clause = "(decode regression)"
            elif label == "none":
                fail_clause = f"showing no verified improvement on any of the {len(SCORING_DIMS)} scored axes"
            else:
                fail_clause = "(regression)"

            # A `none` is an absence of evidence, a REJECT is evidence of harm. Saying the same
            # thing for both is how a contributor whose work simply is not measured yet reads a
            # close as an accusation -- and #768 is the precedent for getting that wrong.
            if label == "none":
                close_body = (
                    "<!-- sparkinfer-museglimmer-auto-close -->\n"
                    "## Closed: no verified speedup — `eval-museglimmer:none`\n\n"
                    f"Measured on the pinned RTX 5090 against the same-box `origin/main`: "
                    f"**{res.get('delta_pct')}%** on the best of "
                    f"{len(SCORING_DIMS)} scored axes (decode + prefill @ "
                    f"{_ctx_list_str()}, concurrent decode @ {_cb_list_str()}).\n\n"
                    "**This is not a finding that anything is wrong with your PR.** Nothing "
                    "regressed, and every correctness gate passed — the change just did not move "
                    "a number this bot measures. The queue is closed rather than left open so it "
                    "reflects work that can still be scored.\n\n"
                    "Three ways forward, depending on which applies:\n\n"
                    "- **Nothing here measures your optimization yet?** That is a gap in the "
                    "harness, not a verdict. Open an issue describing the axis you need — model, "
                    "metric, context/concurrency, and the command that measures it — with your "
                    "before/after numbers, then reopen this PR and ask for the "
                    "[`hold`](../../labels/hold) label so it stays open while the axis is added. "
                    "That path is real: `cb-decode@c2..c32` exists because #1026 asked for it.\n"
                    "- **Correctness fix, refactor, test or docs?** Those are welcome and reviewed "
                    "by hand; they score 0 by design. Reopen as a **draft**, or ask for `hold`, "
                    "and say so — being unscored is not grounds for closing that work.\n"
                    "- **Expected a speedup?** Re-measure against current `main` (it moves fast), "
                    "push the change as a new commit and reopen: the new commit is evaluated on "
                    "the next poll (reopening alone does not re-run a commit that already has its "
                    "verdict)."
                )
            else:
                close_body = (
                    "<!-- sparkinfer-museglimmer-auto-close -->\n"
                    f"## Closed: regression or failed gate — `eval-museglimmer:{label}`\n\n"
                    f"Measured **{res.get('delta_pct')}%** vs the same-box `origin/main`, "
                    f"{fail_clause} — closing automatically.\n\n"
                    "Every scored axis is also a no-regression floor, and the correctness and "
                    "cross-model guards are hard gates, so one failure closes the PR whatever it "
                    "was aiming at. The verdict comment above names which axis or gate failed. "
                    "Push a fix and reopen: the new commit is evaluated on the next poll "
                    "(reopening alone does not re-run a commit that already has its verdict)."
                )
            # Not over a commit the author has already replaced (a push while it was measured),
            # nor over a `hold` or a draft made while the round ran.
            why = arb.verdict_close_blocker(repo, num, commit)
            if why:
                print(f">> PR #{num}: not closed — {why}")
                return
            arb.gh(["pr", "comment", str(num), "-R", repo, "--body", close_body])
            arb.gh(["pr", "close", str(num), "-R", repo])
            print(f">> auto-closed PR #{num} (eval-museglimmer:{label})")


def _exit_if_gave_up():
    """Exit 3 when the selection gave up on a PR after the bot's own errors (GAVE_UP), so the
    wrapper's failed-run banner shows it -- on every way out of a run, one that then measured
    nothing (the GPU down, the lock busy) included."""
    if GAVE_UP:
        print(f"!! museglimmer: gave up on {', '.join(f'#{n}' for n in sorted(GAVE_UP))} after the bot's own "
              f"errors — see above; exiting 3 so the wrapper's failed-run banner shows it")
        sys.exit(3)


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

    ok, login = arb.acting_account_ok()
    if not ok:
        print(f"!! gh acts as {login}, not SPARKINFER_BOT_LOGIN={os.environ.get('SPARKINFER_BOT_LOGIN')} — nothing done")
        sys.exit(3)
    if args.labels_only:
        reconcile_museglimmer_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — museglimmer labels only (no GPU).")
        return

    prs = arb.open_prs_or_none(args.repo, "number,title,labels,isDraft,headRefOid,headRefName,baseRefName,"
                                          "mergeable,author,body,files,changedFiles")
    if prs is None:
        # Not an empty queue: GitHub did not answer (an expired token, an outage). Exit non-zero so the
        # wrapper's failed-run banner shows a bot that has stopped seeing PRs.
        print("!! GitHub did not return the open PRs — nothing done this run")
        sys.exit(3)
    prs.sort(key=lambda p: p["number"])

    stale_closed = close_stale_museglimmer_prs(args.repo, prs, dry_run=args.dry_run) if not only else set()
    main_now = arb.current_main_sha(args.repo)

    denylist = arb.load_denylist()
    pending = []
    for pr in prs:
        num = pr["number"]
        if num in stale_closed:
            continue
        if only and num not in only:
            continue
        # A tier measured on an older head no longer describes this PR: dropped for drafts and held
        # PRs too, which the checks below skip before the selection's own drop further down.
        labs0, head0 = {l["name"] for l in pr.get("labels", [])}, (pr.get("headRefOid") or "")[:40]
        removed = arb.strip_foreign_stale_labels(args.repo, num, labs0, head0, EVAL_PREFIX) if not args.dry_run else set()
        if removed:
            print(f"PR #{num} @ {head0[:9]}: dropped {', '.join(sorted(removed))} (no verdict on this head backs them)")
            labs0 = labs0 - removed
            pr["labels"] = [{"name": l} for l in sorted(labs0)]
        if (not args.dry_run and (pr.get("isDraft") or arb.HOLD_LABEL in labs0)
                and any(l.startswith(EVAL_PREFIX) or l == MUSEGLIMMER_NEEDS_REBASE for l in labs0)
                and arb.strip_stale_verdict_labels(args.repo, num, labs0, EVAL_PREFIX, head0,
                                                   museglimmer_evaluated_commits(args.repo, num), MUSEGLIMMER_NEEDS_REBASE,
                                                   arb.pr_merge_conflict(pr.get("mergeable"))
                or bool(arb.strike_count(STRIKES_FILE, num, (pr.get("headRefOid") or "")[:40], "conflict")))):
            print(f"PR #{num} @ {head0[:9]}: no museglimmer verdict for this head yet — dropped the old eval-museglimmer label")
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
        remeasure = False
        evaluated = museglimmer_evaluated_commits(args.repo, num)
        if evaluated is None:
            # Not "no verdict yet": re-measuring, or dropping its labels, on a failed read is wrong.
            print(f"PR #{num} @ {short}: GitHub did not return its comments — skipped this round")
            continue
        # An eval-museglimmer tier measured on an older head no longer describes this PR.
        if not args.dry_run and arb.strip_stale_verdict_labels(
                args.repo, num, labs, EVAL_PREFIX, head, evaluated, MUSEGLIMMER_NEEDS_REBASE,
                arb.pr_merge_conflict(pr.get("mergeable"))
                or bool(arb.strike_count(STRIKES_FILE, num, (pr.get("headRefOid") or "")[:40], "conflict"))):
            print(f"PR #{num} @ {short}: no museglimmer verdict for this head yet — dropped the old eval-museglimmer label")
        if (pr.get("baseRefName") or "main") != "main":
            # Measured merged onto main, it would be credited with its base branch's commits, and a
            # merge would land in that branch, not main.
            print(f"PR #{num}: based on {pr.get('baseRefName')}, not main — not evaluated")
            continue
        if not args.reeval and arb.strike_count(STRIKES_FILE, num, head, "harness"):
            print(f"PR #{num} @ {short}: edits the eval harness (found on the box) — not evaluated until a push")
            continue
        if (pr.get("changedFiles") or 0) > len(pr.get("files") or []):
            # More files than GitHub lists: the harness check cannot see them all.
            print(f"PR #{num}: changes {pr.get('changedFiles')} files, more than GitHub lists — not evaluated")
            continue
        if not args.reeval and arb.strike_count(STRIKES_FILE, num, head, "conflict"):
            print(f"PR #{num} @ {short}: does not merge onto main on the box — museglimmer-needs-rebase until a push")
            continue
        if not args.reeval and head and head in evaluated:
            if not _remeasure_against_new_main(args.repo, num, head, labs, main_now):
                print(f"PR #{num} @ {short}: already museglimmer-evaluated — skip")
                continue
            remeasure = True
        # Did the author declare a DIFFERENT target model (#1027)? Checked before any GPU time.
        #
        # This bot used to evaluate every PR regardless, because no other bot guards Muse Glimmer.
        # That cost a real PR: #1082 declared Qwen3.8-27B only and claimed +35% at cb-decode@c32 on
        # the unsloth checkpoint -- an axis pr_qwen38_bot.py scores -- and this bot scored it
        # eval-museglimmer:none and auto-closed it before the Qwen3.8 bot polled it. Explicit
        # decision 2026-09-15: skip such PRs. The cost is that a PR declared for Qwen3.8 alone is
        # no longer checked against Muse Glimmer by any bot. arb.model_skip_reason() fails open:
        # an absent, ambiguous or "shared" declaration still evaluates here.
        skip_why = arb.model_skip_reason(pr.get("body") or "", "muse")
        if skip_why:
            print(f"PR #{num}: {skip_why} — skip museglimmer eval")
            continue
        # Does it edit the measuring instrument (HARNESS_PATHS)? The merged build would carry the
        # PR's edit, so a number measured with a changed ruler cannot be accepted either way.
        touched = [f.get("path", "") for f in (pr.get("files") or [])]
        harness_hits = [t for t in touched if any(t.startswith(h) for h in HARNESS_PATHS)]
        if harness_hits:
            print(f"PR #{num}: touches the eval harness ({', '.join(harness_hits[:3])}) — not evaluated")
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

        # Measured merged onto the exact main the round's baseline measured (_remote_script `onto`).
        ref = f"pull/{num}/head"
        # Last, so only a PR the bot would otherwise measure keeps its run exiting 3.
        if not args.reeval and arb.gave_up(STRIKES_FILE, num, head):
            GAVE_UP.add(num)
            print(f"PR #{num} @ {short}: the bot failed on this commit {arb.BOX_FAULT_STRIKES} rounds in a row — "
                  f"not measured again until a push")
            continue
        if remeasure:
            print(f"PR #{num} @ {short}: a merge candidate scored against an older main — re-measuring")
        pending.append((num, head, short, ref, pr.get("title", ""), pr.get("body") or ""))

    if not pending:
        reconcile_museglimmer_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — no museglimmer PRs to evaluate.")
        _exit_if_gave_up()
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
        _exit_if_gave_up()
        return

    _ssh_user = ssh_box_user() if ssh_box_enabled() else "root"
    print(f">> SSH {_ssh_user}@{host}:{port}")

    if not arb.hold_bot_lock():
        print(">> the shared bot lock stayed busy — nothing measured this run")
        _exit_if_gave_up()
        return
    print(">> measuring main baseline (once for this round, shared across all pending PRs) …")
    try:
        main_result = measure_main_baseline(host, port)
    except Exception as e:   # ssh timeout or transport failure: the round, not a PR
        main_result = {"ok": False, "reason": f"exception: {type(e).__name__}: {e}"}
    if not main_result.get("ok"):
        # No usable baseline -> nothing in this round can be scored. Bail out here rather than
        # burning GPU time building N different PR branches against a baseline we already know
        # is broken, and rather than posting a misleading per-PR "main run failed" on every
        # pending PR for what is really one shared infra problem.
        print(f">> main baseline measurement failed: {main_result.get('reason')} — skipping round")
        reconcile_museglimmer_merge_labels(args.repo, dry_run=False)
        print("done — museglimmer round skipped (main baseline unusable).")
        # Non-zero: a main that stays unusable stops this bot measuring anything, the PR fixing it
        # included -- run_bot (cron_common.sh) makes a run of these loud.
        sys.exit(3)
    print(f">> main baseline: decode={main_result['decode_tps']:.2f} "
          f"prefill128={main_result['prefill128_pp']:.2f} "
          f"top1={main_result.get('top1', 0):.3f} kl={main_result.get('kl', 99):.4f}")

    for num, head, short, ref, title, pr_body in pending:
        print(f"PR #{num} @ {short}: evaluating Muse Glimmer '{ref}' …")
        try:
            res = eval_museglimmer_on_box(host, port, ref, main_result)
        except Exception as e:
            # A transport failure is retried with nothing posted (it used to be a REJECT); so is a run
            # killed at the 2 h ssh limit, charged as a failed run once it recurs (arb.exception_result).
            res = arb.exception_result(e)
        # Recorded against the tip the box built (arb.measured_commit): a mid-round push must not
        # leave a verdict naming a commit that was never measured (#1167).
        commit, moved = arb.measured_commit(head, res)
        if moved:
            print(f">> PR #{num} moved during the round: listed {short}, measured {commit[:9]}")
        apply_result(args.repo, num, commit or short, res, title=title, dry_run=False, pr_body=pr_body)

    reconcile_museglimmer_merge_labels(args.repo, dry_run=False)
    print("done — museglimmer eval pass complete.")
    _exit_if_gave_up()


if __name__ == "__main__":
    main()
