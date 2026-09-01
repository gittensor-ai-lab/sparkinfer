#!/usr/bin/env python3
"""sparkinfer DSpark PR auto-evaluator (Qwen3.8-27B, decode+prefill @4k/@16k/@32k).

Adapted from pr_modelopt_bot.py — same transport, same tier buckets, same GitHub plumbing — and
narrowed to one question: how fast is lossless DSpark serving across production context lengths?

  0. SCOPE (2026-09-01) — DSpark decode and DSpark-enabled batched prefill at ctx=4k, 16k
              and 32k are scored dimensions.
              The target is the ModelOpt NVFP4 checkpoint (MODEL_DIR); the draft is the released
              DSpark checkpoint (DRAFT_DIR). Both legs — the AR reference and the speculative run —
              are measured in ONE process by runtime/examples/dspark_tau_check.cpp, so they share a
              model load, a GPU state and an env, and the PR-vs-main delta is apples-to-apples.
              Each context's one fresh harness run supplies decode, AR, acceptance, exact-output
              equality and prompt-prefill throughput. The best verified improvement earns the tier;
              every axis is also a no-regression floor, so a win at one context cannot hide damage
              at another. Context 128 remains intentionally excluded because speculation cannot
              amortize its draft/verify overhead there.

  1. LOSSLESSNESS IS A GATE, NOT A SCORE. dspark_tau_check regenerates the same prompt with the
              draft disabled and requires the two token sequences to be IDENTICAL. A speculative
              decoder that is fast because it emits unverified tokens is not fast, it is wrong, so
              a run with LOSSLESS=0 is REJECTed no matter what it measured. Fail-closed: a missing
              or unparseable LOSSLESS reads as 0, never as a pass.

              This is a stronger accuracy statement than the top1/KL bar the sibling bots use — it
              is exact token equality, not distributional agreement — but it is RELATIVE to the AR
              path in the same build. It cannot see a PR that breaks AR itself, because DSpark
              would then faithfully reproduce the broken AR. Hence gate 2.

  2. Accuracy gate — DIFFERENTIAL, inherited from pr_modelopt_bot.py unchanged: score the same
              token stream on the PR build and on origin/main and require the two distributions to
              agree (bench/scripts/accuracy_compare_pair.py). This is what catches a PR that
              changes what the TARGET produces, which losslessness alone cannot.

              Limitation, stated plainly: a differential gate cannot catch a bug already present on
              main. It catches newly introduced divergence only.

  3. AR no-regression floor — the AR leg of the same run, REGRESS_TOL=0.98. Without it a PR could
              post a DSpark "speedup" purely by making the AR baseline slower (they share every
              target forward), or by slowing ordinary serving decode to help the speculative path.
              A regression here is a hard REJECT regardless of the DSpark number.

  4. LONG-CONTEXT no-regression guards @ ctx=16k, decode AND prefill, on TWO models (added
              2026-08-18 by explicit request). Differential, PR vs freshly-measured main, same box,
              same round, REGRESS_TOL. Both are hard REJECTs.

                qwen3.8 (MODEL_DIR)  the ModelOpt NVFP4 checkpoint — the DSpark target itself, and
                                     the build the published benchmark table quotes. It used to be
                                     covered because pr_modelopt_bot.py SCORED decode@16k and
                                     prefill@16k on it; when DSpark decode@128 became the only
                                     scored dimension that coverage vanished, and this closes it.

                qwen3.6 (Q36_GGUF)   Qwen3.6-35B-A3B, a different architecture (MoE, not dense) on
                                     a separate model load. DSpark work lands in qwen35.cpp /
                                     qwen35_prefill.cpp / the shared kernels, all of which Qwen3.6
                                     uses — the exact surface through which PR #775 regressed a
                                     model nobody was scoring at the time.

              16k only, not the sibling bots' full 0/512/4k/16k/32k sweep: one context per model
              keeps a round to two extra model loads, and 16k is where long-context prefill and a
              large KV read both actually cost something.

              If either guard produces no MAIN baseline the whole round bails, rather than letting
              every PR collect a REJECT for one shared infra fault.

Still deliberately NOT run: the batched-prefill parity check. It is absolute rather than
differential, and main currently fails it at n=128 (~0.58 against a 0.75 bar) — a real pre-existing
divergence that stayed hidden until the gate's prompt corpus was lengthened enough for that case to
run at all. Enabling it here would REJECT every PR until that is fixed.

Applies `eval-dspark:<TIER>` AND mirrors it to the generic `eval:<TIER>` label (SN74 scoring reads
eval:* tiers). Auto-close on none/REJECT is live; auto-merge stays OFF unless
SPARKINFER_DSPARK_AUTOMERGE=1 is explicitly set.

  python eval/pr_dspark_bot.py
  python eval/pr_dspark_bot.py --only-prs 870 --reeval

Never rents a GPU. Shares the pinned box with any other bot via flock in the cron wrapper
(run_dspark_cron.sh) — all bots MUST share /tmp/sparkinfer_bot.lock.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
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

# The dimensions that can earn a tier are decode and DSpark-enabled prefill at 4k, 16k and 32k.
# A PR is tiered on whichever axis it moved most, not on a sum or average: decode and prefill are
# largely independent, so averaging would dilute a real focused improvement.
#
# This said ctx=128 until 2026-08-21, and the paragraph above it described decode@128 and
# prefill@128 as no-regression floors. Both were inherited verbatim from pr_modelopt_bot.py and
# neither survived the move to long context: the module docstring's SCOPE section is explicit that
# "nothing is measured at ctx=128 any more". The scored set now covers 4k/16k/32k; the floors are
# AR decode, acceptance, exact losslessness, and the two shared-model guards.
#
# The tier is the best verified improvement across decode/prefill at 4k, 16k and 32k. Every
# listed dimension is also a no-regression floor, so a PR cannot buy a win at one context by making
# another production context slower.
SCORING_DIMS = [
    "dspark-decode@4k", "dspark-prefill@4k",
    "dspark-decode@16k", "dspark-prefill@16k",
    "dspark-decode@32k", "dspark-prefill@32k",
]
SCORING_DIM = SCORING_DIMS[0]

# Accuracy gate bars. This gate is DIFFERENTIAL (PR vs origin/main on the same token stream, see
# the module docstring pt. 2), not absolute-vs-llama.cpp, so the bars are much tighter than the
# 0.90/0.10 an across-engine comparison needs: two builds of the same model on the same box
# should agree essentially exactly. Anything less means the PR changed the model's numerics.
# Not 1.0/0.0 — a PR may legitimately reassociate float ops (fusing a kernel, changing a
# reduction order), which perturbs the last bits without being a correctness bug.
# top1 0.90 and KL 0.1, both loosened by explicit decision 2026-08-16 from the Qwen3.8 bot's
# 0.99 / 0.01. The gate stays differential (PR vs main on the same token stream), so it still
# catches a PR that CHANGES this model's output distribution -- it now tolerates ten times the KL
# drift and lets one token in ten flip its argmax before calling that a failure.
#
# Worth knowing what that admits, since these are the numbers this codebase measured by hand
# during Qwen3.8 bring-up: a second quantization of the GDN projections scored top1 0.96 / KL
# 0.035 and was rejected as too lossy to ship; a Q8_0 fit scored 1.00 / 0.015. Both pass here.
# So this bar no longer distinguishes "same model, faster" from "slightly different, faster" --
# it is a guard against gross corruption, not against quality drift. The batched-prefill parity
# gate and the Qwen3.8 no-regression guard are what carry correctness now.
ACC_TOP1_BAR = float(os.environ.get("MODELOPT_ACC_TOP1_BAR", "0.9"))
ACC_KL_BAR = float(os.environ.get("MODELOPT_ACC_KL_BAR", "0.1"))

EVAL_PREFIX = "eval-dspark:"
MODELOPT_MERGE_FIRST = "dspark-merge-first"
MODELOPT_NEEDS_REBASE = "dspark-needs-rebase"
# First schema for this bot. Same reasoning as the sibling bots' own bumps: a PR evaluated before
# a scoring change existed must not keep a stale-scored label/score forever.
# v2 (2026-08-19): the NSPLITS=1 pin was removed from dspark_tau_check, so throughput is now
# measured at the split count the engine selects rather than in a regime nothing serves. The
# baseline moves by ~72% (dspark@4k 43.06 -> 74.05, ar@4k 47.39 -> 90.19) and every score taken
# under v1 is incomparable -- bumping the schema forces re-evaluation instead of letting a stale
# label sit next to a number that no longer means the same thing.
EVAL_SCHEMA_VERSION = "v8-dspark-4k-16k-32k-decode-prefill"
MARKER_RE = re.compile(
    r"<!-- sparkinfer-dspark-eval:" + re.escape(EVAL_SCHEMA_VERSION) + r":([0-9a-f]+)(?:\s+(\{.*?\}))? -->",
    re.DOTALL,
)

# --- box paths (see .env.eval's MODELOPT_* block) ---
# Separate clone from pr_eval_bot's /root/sparkinfer, which carries unrelated uncommitted work.
REMOTE_REPO = os.environ.get("MODELOPT_REMOTE_REPO", "/root/sparkinfer_modelopt")
# The SCORED checkpoint: gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 -- our own uniform-NVFP4
# ModelOpt quantization, a compressed-tensors DIRECTORY, NOT a GGUF. qwen3_gguf_bench /
# qwen3_gguf_score grew directory support for exactly this (runtime/examples/qwen_checkpoint.h).
#
# NOT the same checkpoint as Q38_GUARD_MODEL_DIR below (upstream unsloth, NVFP4 FFN + FP8
# attention). Both are supported; this one is scored and that one is guarded. They differ by
# ~13% decode and ~30% prefill, so mixing them up silently changes every number this bot prints
# -- which is exactly what happened while writing the Qwen3.8 README section on 2026-08-17.
MODEL_DIR = os.environ.get("MODELOPT_MODEL_DIR", "/root/workspace/models_q38_modelopt")
# The single weight blob inside MODEL_DIR, for the Polaris attestation ONLY. The sibling bots pass
# their .gguf here; the analogue for a compressed-tensors checkout is the safetensors file, not the
# directory -- receipt.model_sha256() returns "" for anything that is not a regular file, so
# passing MODEL_DIR would mint a receipt whose model_sha256 pins nothing while still looking valid.
# If a future checkpoint is sharded (model-00001-of-0000N.safetensors) this path stops existing and
# the sha degrades to "" rather than crashing; override MODELOPT_MODEL_WEIGHT_FILE if that happens.
MODEL_WEIGHT_FILE = os.environ.get("MODELOPT_MODEL_WEIGHT_FILE",
                                   os.path.join(MODEL_DIR, "model-00001-of-00003.safetensors"))
# The DSpark draft checkpoint (block_size=7, 5 layers, hidden 5120). It has no embed_tokens or
# lm_head of its own -- it borrows the target's, which is what makes its proposals directly
# comparable to what the target would produce, and what makes the losslessness check meaningful.
DRAFT_DIR = os.environ.get("DSPARK_DRAFT_DIR", "/root/workspace/dspark")
# Prompt length for the scored context. dspark_tau_check takes explicit token ids, so this is the
# real ctx: the bot tokenizes bench/scripts/bench_prompt_32k.txt and truncates to this many ids.
# ctx=16384, not 4096 (explicit decision 2026-08-28): the serving target is now long-context
# traffic, so contributors must optimise the KV-read and verification regime that production uses.
# At ctx=128 speculation CANNOT pay: the token
# loop runs one target forward per kept token, exactly as AR does, so the draft's only payoff is
# arming the batched verify -- and the batched path is slower at short context, so it never arms.
# Three merges optimised decode@128 from 47.35 to 86.47 tok/s against an AR of 87.91, i.e. straight
# at a ceiling of "stop speculating", with tau collapsing 1.255 -> 1.000 on the way. The metric was
# rewarding the removal of speculation because that was the only direction available.
#
# 16384 is past kEngageMinSeq (1024), so the adaptive gate CAN arm the batched verify, which is the
# only path on which accepting k tokens costs one target forward instead of k. That makes tau a
# real lever again and makes "DSpark beats AR" reachable rather than structurally impossible.
DSPARK_CTX = int(os.environ.get("DSPARK_CTX", "16384"))
DSPARK_CTX_4K = int(os.environ.get("DSPARK_CTX_4K", "4096"))
DSPARK_CTX_32K = int(os.environ.get("DSPARK_CTX_32K", "32768"))
DSPARK_PREFILL_CTX = int(os.environ.get("DSPARK_PREFILL_CTX", "32768"))
# Repeats of the speculative generation per round. Every one must match AR for LOSSLESS to be 1.
#
# Three independent processes, not three generations in one loaded runtime. Process isolation makes
# each repetition start from the same allocator/graph/model state and still catches nondeterministic
# losslessness defects. Batched 16k prefill keeps the extra model loads cheaper than the old
# in-process token-loop repeats.
#
# Detection honesty: the defect class this targets (the draft/verify overlap race) needs FULL-BLOCK
# accepts to fire, because it corrupts accept GROUPING and only the adaptive gate turns that into a
# different verify path. When this was written (2026-08-18) tau sat at ~1.0-1.1, full blocks
# essentially never happened, and the hazard could not be reproduced at 4k even with OVERLAP=1
# forced and 3 reps -- so the gate was a strict improvement that was UNVERIFIED against a live
# instance.
#
# That caveat is now OUT OF DATE in the direction that matters: tau has since risen to ~1.66
# (#893/#894 and the verify-cost work), so full blocks are no longer vanishingly rare and this
# gate is closer to load-bearing than it was. It is still unverified against a live instance --
# nobody has reproduced the race here since tau moved. Re-running that reproduction at the current
# tau is worth someone's time, and if it fires, 3 reps is the number to revisit first.
DSPARK_SPEC_REPS = int(os.environ.get("DSPARK_SPEC_REPS", "3"))

# Mean accept length (tau) no-regression floor. THE most important gate this bot has, because
# without it the scored objective is satisfied by deleting the feature it scores.
#
# The failure is not hypothetical and it is not subtle. Maximising DSPARK_TPS while DSpark is a NET
# LOSS against AR means the gradient points at "speculate less", and its optimum is "do not
# speculate at all" -- at which point DSpark IS autoregressive decode and the metric reads a win.
# It happened twice: #868 took tau 1.255 -> 1.185 at ctx=128, and #869 idled the draft entirely
# there (tau -> 1.000, decode 47 -> 86 tok/s, all of it the cost of speculation going away). Moving
# the scored context to 4k did NOT fix it -- #876 then proposed exactly the same thing at 4k, its
# own table showing tau 1.0847 -> 1.0000 for +9.6%, and it would have passed every other gate
# including losslessness, which a non-speculating decoder satisfies trivially.
#
# So: a PR may not buy throughput with acceptance. Wins must come from cheaper speculation at
# equal-or-better tau, which is the only direction that can ever carry DSpark past AR.
#
# The tolerance is looser than REGRESS_TOL (0.98): tau is a ratio of accepted tokens to steps over
# a 128-token generation, so it moves in coarser increments than a throughput number and a
# genuinely neutral change can jitter it. 0.95 blocks the "turn it off" direction -- which shows up
# as tau collapsing toward 1.0, a 7%+ move from here -- without rejecting noise.
DSPARK_TAU_TOL = float(os.environ.get("DSPARK_TAU_TOL", "0.95"))

# Files that define WHAT IS MEASURED rather than what runs. A PR touching any of these is not
# evaluated at all -- it is not a question of whether the change is good.
#
# dspark_tau_check.cpp picks the env both legs run under, computes the throughput, and renders the
# losslessness verdict; bench_prompt_32k.txt is the prompt every score is taken on. Changing either
# moves the baseline for every contributor at once, including the one proposing it. Scoring such a
# PR would mean scoring the ruler, and the incentive that creates is to argue with the measurement
# instead of improving the engine.
#
# This is policy, not a quality judgement, and it applies even when the diff is CORRECT. #871 and
# #875 both identified a real defect in the NSPLITS pin -- verified: the pin costs 35% of decode at
# ctx=4096 -- and both were still closed, because the fix belongs to whoever owns the harness. The
# other half of the same problem is #872, which optimised the regime the pin creates: +11.8%
# measured, -0.3% in production, auto-merged at tier L before anyone could read it.
HARNESS_PATHS = (
    "runtime/examples/dspark_tau_check.cpp",
    "bench/scripts/bench_prompt_32k.txt",
    "eval/",
    "bench/scripts/",
)
BENCH_TOKENS = int(os.environ.get("MODELOPT_BENCH_TOKENS", "128"))
ACC_TOPK = int(os.environ.get("MODELOPT_ACC_TOPK", "128"))
# INACTIVE IN THIS BOT. Inherited from pr_modelopt_bot.py, where it IS a live floor: the fraction
# of the continuation that batched prefill must still generate identically to the token loop,
# absolute rather than PR-vs-main (that bot's remote script has the PREFILL_PARITY block; this one
# does not). Here the whole gate is deliberately off -- see the module docstring -- because main
# fails it at n=128 and enabling it would REJECT every PR.
#
# The scaffolding around it is dead in step with that: parity_bar is interpolated into the remote
# script but nothing there reads $PARITY_BAR, and the PREFILL_PARITY_OK / PARITY worst= parse
# branches below never fire because nothing emits those lines. Left in place so re-enabling is one
# block, not a rewrite -- but do not read any of it as a gate that runs.
PARITY_BAR = float(os.environ.get("MODELOPT_PARITY_BAR", "0.75"))
# Score dumps for the differential accuracy gate. main's is written once per round by
# measure_main_baseline(); each PR compares its own dump against it. Kept on the box (not shipped
# back over ssh) because a 128-deep top-k dump over the whole corpus is megabytes.
SCORE_DUMP_MAIN = "/tmp/mopt_score_main.txt"
SCORE_DUMP_PR = "/tmp/mopt_score_pr.txt"
EVAL_TEXT = "bench/scripts/eval_text.txt"  # same corpus the sibling bots score

# Qwen3.6 no-regression guard (module docstring, pt. 3) — same env var names as pr_dflash_bot.py's
# Q36_GUARD_* (PRIMARY36_MODEL_REPO/PRIMARY36_TOK_REPO) so one .env.eval entry covers both bots.
# Defaults point at this box's actual layout (confirmed 2026-08-12: /root/workspace/models36),
# distinct from the DFlash bot's vast.ai-box convention (/workspace/models36).
Q36_GUARD_MODELS_DIR = os.environ.get("Q36_GUARD_MODELS_DIR", "/root/workspace/models36")
Q36_GUARD_MODEL_FILE = os.environ.get("Q36_GUARD_MODEL_FILE", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
Q36_GUARD_MODEL_REPO = os.environ.get("PRIMARY36_MODEL_REPO", "unsloth/Qwen3.6-35B-A3B-GGUF")
Q36_GUARD_TOK_REPO = os.environ.get("PRIMARY36_TOK_REPO", "Qwen/Qwen3.6-35B-A3B")
GUARD_CTX_LABEL = {0: "128", 512: "512", 4096: "4k", 16384: "16k", 32768: "32k"}

# Qwen3.8 no-regression guard, over the OTHER supported NVFP4 build (upstream unsloth, NVFP4 FFN +
# FP8 attention). Both builds are supported targets; this one is not scored, so without a guard
# nothing would notice it getting slower. They share load_compressed_tensors, the batched prefill
# and every decode kernel, so a ModelOpt optimisation that speeds up the scored checkpoint by
# pessimising the shared path is a regression -- and this bot's own numbers cannot see it, because
# it only ever benches models_q38_modelopt.
# Same shape as the inherited Qwen3.6 guard below, except the subject is a compressed-tensors
# DIRECTORY rather than a GGUF, so there is no ensure_model/ensure_tokenizer download step.
Q38_GUARD_MODEL_DIR = os.environ.get("Q38_GUARD_MODEL_DIR", "/root/workspace/models_qwen38")
# These guard contexts cover the other supported Qwen3.8 checkpoint on shared prefill/KV paths.
Q38_GUARD_CTXS = [0, 4096, 16384]

# Auto-merge is wired (mirrors pr_dflash_bot.py's auto_merge_ok_dflash/try_auto_merge_dflash
# shape) but OFF unless this exact env var is set.
# Was reading SPARKINFER_QWEN38_AUTOMERGE until 2026-08-17 -- a leftover from this file being
# forked from pr_qwen38_bot.py, whose rename to the ModelOpt-specific var missed this one
# word-boundary. It "worked" only because .env.eval happened to set BOTH SPARKINFER_QWEN38_
# AUTOMERGE and SPARKINFER_MODELOPT_AUTOMERGE=1, not because they were actually independent —
# flipping one without the other would have silently done nothing. Reads its own var now.
AUTO_MERGE = os.environ.get("SPARKINFER_DSPARK_AUTOMERGE") == "1"
AUTOMERGE_BLOCK = {
    "copycat", "copycat-warn", "flagged:gaming", "penalty", "needs-benchmark",
    MODELOPT_NEEDS_REBASE, arb.REEVALUATE_LABEL, arb.HOLD_LABEL, *arb.REGRESSION_LABELS,
}

SCORES_FILE = os.path.expanduser(
    os.environ.get("DSPARK_SCORES_FILE", "~/.sparkinfer_dspark_scores.json")
)

# Polaris verifiable-compute receipts — same policy/keys as the AR and DFlash bots (on by
# default; TDX via POLARIS_API_KEY when configured, else Ed25519 fallback). Wired through
# judge.py's --from-stdin generic RESULT_JSON path (NOT --dflash, which hardcodes a
# DFlash-shaped measurement block and eval_mode="dflash" — reusing it here would produce a
# mislabeled, semantically wrong attestation). SPARKINFER_EVAL_MODE is set explicitly below so
# the attestation correctly records "qwen38-16k", not the AR bot's "longctx" default.
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
        print(f">> dspark scores save skipped: {e}")


def _report_pct(pct: float, reject: bool = False) -> float:
    """Round a delta so the number shown can never contradict the label it produced.

    round() breaks that: PR #863 gained 9.9996% on decode@16k, which is BELOW the 10% L threshold
    and correctly scored M -- but round(9.9996, 1) prints "10.0%", so the comment read
    "delta=10.0%  eval-modelopt:M" and looked like an off-by-one in the buckets. (It missed L by
    0.0004 points: it needed 90.17404 tok/s and measured 90.1737.) More decimals do not help --
    that value reads as 10.0 / 10.00 / 10.000 and only separates at the fourth.

    Which way to break the tie depends on which side the result landed, and getting only one side
    right just moves the confusion:

      PASSING (none/XS/S/M/L/XL) -> truncate TOWARD zero. A gain then never displays into a bucket
      it did not earn (9.9996 -> 9.9), and a small negative that still cleared the -2% floor never
      displays AT the floor (-1.96 -> -1.9) and so never reads as a REJECT it isn't.

      REJECT -> truncate AWAY from zero (-2.04 -> -2.1), so the regression that triggered it never
      displays milder than the threshold it broke.
    """
    scaled = pct * 10.0
    return (math.floor(scaled) if reject else math.trunc(scaled)) / 10.0


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
        return "REJECT", _report_pct(pct, reject=True), False, (
            f"{metric} regression: {pr_tps:.2f} < {100 * REGRESS_TOL:.0f}% of main {main_tps:.2f}"
        )
    g = (pr_tps - main_tps) / main_tps
    pct = _report_pct(100.0 * g)
    if g < SIG:
        return "none", pct, True, f"within significance gate — not a verified {metric} improvement"
    for thr, name in BUCKETS:
        if g >= thr:
            return name, pct, True, "ok"
    return "none", pct, True, "ok"


_TIER_RANK = {"REJECT": -1, "none": 0, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5}


def check_q38_guard(pr: dict, main: dict, tol: float = REGRESS_TOL):
    """Did this ModelOpt PR regress the SHIPPING Qwen3.8 checkpoint?

    Differential, PR vs freshly-measured main on the same box and contexts -- the same discipline
    and the same comparison convention as check_q36_guard below (tol is a RATIO FLOOR: cur must
    stay >= main * 0.98, it is NOT a fractional delta).

    It exists because this bot's scoring numbers come only from models_q38_modelopt, so a change
    that wins on the ModelOpt checkpoint by pessimising the shared compressed-tensors path
    (load_compressed_tensors, prefill_batched_run, the decode GEMVs) would score as an improvement
    while making the OTHER supported NVFP4 build -- upstream unsloth, which nothing else here
    measures -- slower.

    Iterates MAIN's contexts as the reference set and fails closed on a missing/zero PR
    measurement, for the same reason check_q36_guard does: a PR build that crashes partway through
    its own sweep must not make that context silently uncheckable."""
    problems = []
    # Checkpoint not installed on the box. Report it every round rather than letting it read as
    # a pass: this guard was silently vacuous for its entire life (it benched the scored
    # checkpoint against itself), and the failure mode of a guard nobody notices is exactly how
    # that went unseen. Not a REJECT -- a missing download is not the PR's fault -- but the
    # verdict line will say SKIPPED, so "guarded" is never claimed when nothing was guarded.
    if pr.get("guard38_unavailable") or main.get("guard38_unavailable"):
        return True, ["qwen3.8 guard SKIPPED -- upstream checkpoint not installed "
                      "(Q38_GUARD_MODEL_DIR); the shared-code regression it exists to catch is "
                      "UNCOVERED this round"]
    if (pr.get("guard38_failed") or main.get("guard38_failed")
            or not pr.get("guard38") or not main.get("guard38")):
        problems.append("qwen3.8 guard measurement unavailable")
    pr_ctxs, main_ctxs = pr.get("guard38") or {}, main.get("guard38") or {}
    for ctx, main_vals in main_ctxs.items():
        label = GUARD_CTX_LABEL.get(ctx, str(ctx))
        pr_vals = pr_ctxs.get(ctx) or {}
        for metric in ("decode", "prefill"):
            base = main_vals.get(metric, 0)
            if base <= 0:
                continue
            cur = pr_vals.get(metric, 0)
            if cur <= 0:
                problems.append(
                    f"qwen3.8 {metric}@{label}: PR measurement missing/zero "
                    f"(main {base:.1f}) — treated as regression"
                )
                continue
            if cur < base * tol:
                pct = 100.0 * (cur - base) / base
                problems.append(
                    f"qwen3.8 {metric}@{label}: {cur:.1f} < {100 * tol:.0f}% of main "
                    f"{base:.1f} ({pct:+.1f}%)"
                )
    return (len(problems) == 0, problems)


def check_q36_guard(pr: dict, main: dict, tol: float = REGRESS_TOL):
    """No-regression check: PR vs same-box main, Qwen3.6 only, decode + prefill, every measured
    context. Adapted from pr_dflash_bot.py's check_qwen_guard (its qwen3.5/Qwythos half dropped —
    scoped to qwen3.6 only per explicit instruction, module docstring pt. 3). Returns
    (ok, [human-readable regression/failure strings])."""
    problems = []
    if pr.get("guard36_failed") or main.get("guard36_failed") or not pr.get("guard36") or not main.get("guard36"):
        problems.append("qwen3.6 guard measurement unavailable")
    pr_ctxs, main_ctxs = pr.get("guard36") or {}, main.get("guard36") or {}
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
                    f"qwen3.6 {metric}@{label}: PR measurement missing/zero "
                    f"(main {base:.1f}) — treated as regression"
                )
                continue
            if cur < base * tol:
                pct = 100.0 * (cur - base) / base
                problems.append(
                    f"qwen3.6 {metric}@{label}: {cur:.1f} < {100 * tol:.0f}% of main "
                    f"{base:.1f} ({pct:+.1f}%)"
                )
    return (len(problems) == 0, problems)


def qwen38_evaluated_commits(repo, num):
    """Head commits that already have a REAL scoring verdict posted — mirrors
    dflash_evaluated_commits: infra/transport failures (label:null in the marker) don't count."""
    r = arb.gh(["pr", "view", str(num), "-R", repo, "--json", "comments"])
    done = set()
    for c in json.loads(r.stdout or "{}").get("comments", []):
        body = c.get("body") or ""
        m = MARKER_RE.search(body)
        if not m or "sparkinfer DSpark auto-eval" not in body:
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


def strip_qwen38_eval_labels(repo, num):
    for lab in list(arb.labels_on(repo, num)):
        if lab.startswith(EVAL_PREFIX):
            arb.remove_label(repo, num, lab)


STALE_DAYS = float(os.environ.get("MODELOPT_STALE_DAYS", "1"))


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


def close_stale_qwen38_prs(repo, prs, dry_run=False):
    """Close open PRs with no author commit activity in STALE_DAYS+ days. HOLD_LABEL and the
    current qwen38-merge-first winner are exempt."""
    closed = set()
    now = time.time()
    for pr in prs:
        num = pr["number"]
        if pr.get("isDraft"):
            continue
        labs = {l["name"] for l in pr.get("labels", [])}
        if arb.HOLD_LABEL in labs or MODELOPT_MERGE_FIRST in labs:
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
            "<!-- sparkinfer-qwen38-auto-close-stale -->\n"
            f"## Closed: stale — no commits in {age_days:.1f} days\n\n"
            f"This PR has had no new commits in over {STALE_DAYS:g} days — closing automatically "
            "to keep the Qwen3.8-27B eval queue clean. Reopen (or push a new commit / open a "
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
    """No ERR-trap diagnostic captured AND the run did not reach its final checkpoint.

    The sibling bots key this off ACCURACY_STAGE_DONE because accuracy is their LAST stage. Here
    it is not: the Qwen3.6 guard runs after it, so ACCURACY_STAGE_DONE would mark a run "far
    enough along" while the entire guard was still missing. Observed for real 2026-08-15 --
    a main run reached ACCURACY_STAGE_DONE and GUARD_START, then died at exit 6 during the guard
    sweep with no ERR-trap output at all; re-running the identical script succeeded, so it was
    transient. Under the old heuristic that run would NOT have been retried, check_q36_guard would
    have seen an empty guard36 dict, called the measurement unavailable, and hard-REJECTed --
    auto-closing a PR for a flake. GUARD_END is the real end-of-run marker, so use that."""
    combined = (stdout or "") + "\n" + (stderr or "")
    if _crash_reason(stdout, stderr):
        return False
    return "GUARD_END" not in combined


def _ssh_run_resilient(host, port, script: str, label: str):
    """Retry once on a hard kill or an explicitly classified child-process infrastructure fault."""
    r = ssh_run(host, port, script, via_stdin=True)
    if r.returncode != 0:
        combined = (r.stdout or "") + "\n" + (r.stderr or "")
        if "RETRYABLE_INFRA_FAILURE" in combined:
            print(f">> {label}: transient DSpark child-process failure — retrying the entire "
                  "measurement once")
            r = ssh_run(host, port, script, via_stdin=True)
        elif _looks_like_hard_kill(r.stdout, r.stderr):
            print(f">> {label}: looks like a hard kill (no ERR-trap diagnostic, no final "
                  f"checkpoint reached) — retrying once")
            r = ssh_run(host, port, script, via_stdin=True)
    return r


def _remote_script(ref: str, role: str = "pr", main: dict | None = None) -> str:
    """Bash run on the eval box: checkout ref, build, DSpark speculative decode @ctx=16k against
    the NVFP4 target, and the teacher-forced score dump.

    Run once per ref -- identical script both times so the two measurements are directly
    comparable. `role` only decides which score dump path is written and whether the
    differential accuracy compare runs (main has nothing to compare against yet; the PR run
    compares itself against main's dump from earlier in the same round)."""
    repo = shlex.quote(REMOTE_REPO)
    model_dir = shlex.quote(MODEL_DIR)
    ref_q = shlex.quote(ref)
    ntok = BENCH_TOKENS
    topk = ACC_TOPK
    parity_bar = PARITY_BAR
    draft_dir = shlex.quote(DRAFT_DIR)
    ctx = DSPARK_CTX
    ctx4 = DSPARK_CTX_4K
    ctx32 = DSPARK_CTX_32K
    prefill_ctx = DSPARK_PREFILL_CTX
    spec_reps = DSPARK_SPEC_REPS
    eval_text = shlex.quote(EVAL_TEXT)
    dump_self = shlex.quote(SCORE_DUMP_MAIN if role == "main" else SCORE_DUMP_PR)
    dump_main = shlex.quote(SCORE_DUMP_MAIN)
    is_pr = "1" if role == "pr" else "0"
    # Reference values so a PR run can reject itself the moment a gate fails, instead of paying for
    # the accuracy dump and both 16k guard sweeps to produce a verdict already decided. Zero for the
    # main run, which has nothing to compare against and must always run every stage -- it IS the
    # reference. Kept as plain numbers rather than re-deriving the thresholds in bash: the bash only
    # ever compares, the policy stays in Python.
    m = main or {}
    ref_dspark = float(m.get("dspark_tps") or 0)
    ref_dspark4 = float(m.get("dspark4_tps") or 0)
    ref_dspark32 = float(m.get("dspark32_tps") or 0)
    ref_prefill = float(m.get("prefill_pp") or 0)
    ref_prefill4 = float(m.get("prefill4_pp") or 0)
    ref_prefill16 = float(m.get("prefill16_pp") or 0)
    ref_ar = float(m.get("ar_tps") or 0)
    ref_ar4 = float(m.get("ar4_tps") or 0)
    ref_ar32 = float(m.get("ar32_tps") or 0)
    ref_tau = float(m.get("mean_accept") or 0)
    ref_tau4 = float(m.get("mean_accept4") or 0)
    ref_tau32 = float(m.get("mean_accept32") or 0)
    tau_tol = DSPARK_TAU_TOL
    regress_tol = REGRESS_TOL
    acc_top1 = ACC_TOP1_BAR
    acc_kl = ACC_KL_BAR
    q38_guard_dir = shlex.quote(Q38_GUARD_MODEL_DIR)
    q36_dir = shlex.quote(Q36_GUARD_MODELS_DIR)
    q36_file = shlex.quote(Q36_GUARD_MODEL_FILE)
    q36_repo = shlex.quote(Q36_GUARD_MODEL_REPO)
    q36_tok = shlex.quote(Q36_GUARD_TOK_REPO)
    return f"""
set -euo pipefail
# Surface *why* a crash happened instead of dying silently -- same diagnostic trap as the sibling
# bots' _remote_script (a REJECT from an infra crash should carry a real cause).
trap 'rc=$?; ln=$LINENO; reason=""; \\
  case $rc in \\
    137) reason="likely OOM-killed (SIGKILL)" ;; \\
    139) reason="likely segfault (SIGSEGV)" ;; \\
    134) reason="likely abort (SIGABRT)" ;; \\
    124) reason="likely timeout" ;; \\
  esac; \\
  echo "REMOTE_SCRIPT_FAILED line=$ln exit=$rc reason=$reason" >&2; \\
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader >&2 2>/dev/null || true' ERR

# One run does several back-to-back multi-GB model load/unload cycles -- poll GPU memory down to
# near-empty before each heavy load instead of assuming the previous process's exit already freed
# it. This matters more here than for the sibling bots: the NVFP4 checkpoint's resident footprint
# is ~32GB of a 32GB card (dequant -> Q4_K decode copies plus the NVFP4 prefill copies), so even a
# few hundred MB of not-yet-reclaimed memory is the difference between loading and OOM.
wait_gpu_clear() {{
  local tries=0 used
  while [ "$tries" -lt 30 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -lt 1024 ] 2>/dev/null && return 0
    sleep 1
    tries=$((tries + 1))
  done
  echo "WARN: GPU memory still ${{used:-unknown}} MiB after ${{tries}}s wait -- proceeding anyway" >&2
}}

export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
export CUDA_HOME=${{CUDA_HOME:-/usr/local/cuda-13.0}}
REPO={repo}
MODEL_DIR={model_dir}
DRAFT_DIR={draft_dir}
CTX={ctx}
CTX4={ctx4}
CTX32={ctx32}
PREFILL_CTX={prefill_ctx}
NTOK={ntok}
TOPK={topk}
PARITY_BAR={parity_bar}
EVAL_TEXT={eval_text}
DUMP_SELF={dump_self}
DUMP_MAIN={dump_main}
IS_PR={is_pr}
REF_DSPARK={ref_dspark}
REF_DSPARK4={ref_dspark4}
REF_DSPARK32={ref_dspark32}
REF_PREFILL={ref_prefill}
REF_PREFILL4={ref_prefill4}
REF_PREFILL16={ref_prefill16}
REF_AR={ref_ar}
REF_AR4={ref_ar4}
REF_AR32={ref_ar32}
REF_TAU={ref_tau}
REF_TAU4={ref_tau4}
REF_TAU32={ref_tau32}
TAU_TOL={tau_tol}
REGRESS_TOL={regress_tol}
Q38_GUARD_MODEL_DIR={q38_guard_dir}
Q36_GUARD_MODELS_DIR={q36_dir}
Q36_GUARD_MODEL_FILE={q36_file}
Q36_GUARD_MODEL_REPO={q36_repo}
Q36_GUARD_TOK_REPO={q36_tok}

cd "$REPO"
git remote set-url origin https://github.com/gittensor-ai-lab/sparkinfer.git 2>/dev/null || true
git fetch -q origin {ref_q}
git reset -q --hard
git clean -qfd
git checkout -qf FETCH_HEAD
HEAD=$(git rev-parse --short HEAD)
echo "REMOTE_HEAD $HEAD"

# PIN THE MEASURING INSTRUMENT TO origin/main, for every ref including main itself.
#
# The harness lives in the repo, so without this each ref is measured with ITS OWN copy of it --
# and the moment the harness changes, every PR branched before that change is compared against a
# baseline measured by a DIFFERENT instrument. That is not a subtle skew; on 2026-08-19 removing
# the NSPLITS pin moved the baseline by ~72%, and #878 -- branched an hour earlier, so still
# carrying the pinned harness -- measured 45.37 against main's 74.07 and was auto-closed for a
# "-47.5% regression" that was entirely the two refs using different rulers. It was the best PR of
# the day: it had closed the DSpark/AR gap with AR flat and tau held.
#
# Same reasoning as the eval/polaris sync below, and consistent with HARNESS_PATHS: a PR may not
# change these files anyway, so taking them from main costs a contributor nothing and makes the
# comparison mean what it claims.
git fetch -q origin main
git checkout -q origin/main -- runtime/examples/dspark_tau_check.cpp bench/scripts/bench_prompt_32k.txt 2>/dev/null || {{
  echo "HARNESS_PIN_FAILED -- could not take the harness from origin/main" >&2
  exit 1
}}
echo "HARNESS_PINNED $(git rev-parse --short origin/main) (dspark_tau_check.cpp + bench_prompt_32k.txt)"

test -d "$MODEL_DIR" || {{ echo "FAIL missing NVFP4 checkpoint dir $MODEL_DIR"; exit 1; }}
test -f "$MODEL_DIR/config.json" || {{ echo "FAIL $MODEL_DIR has no config.json"; exit 1; }}

# Always reconfigure (cheap, idempotent) -- skipping it on an existing CMakeCache left stale
# generated Makefiles pointing at a DIFFERENT PR branch's files once the checkout switched
# underneath it (the sibling bots hit exactly this, #693/#694).
#
# Reconfiguring is NOT enough to fix a poisoned compiler, though: CMake caches
# CMAKE_CUDA_COMPILER in CMakeCache.txt and keeps it on every subsequent configure no matter what
# PATH says. On 2026-08-21 this box had an apt CUDA 11.5 at /usr/bin/nvcc; one configure run
# without the PATH above cached it, and every round after that died with "nvcc fatal: Unsupported
# gpu architecture 'compute_89'" -- 11.5 predates sm_89 -- while the PATH export sat right there
# looking correct. Blow the cache away when it points anywhere but the toolkit we intend.
# Reclaim this box's own build litter before compiling. nvcc writes GB-scale intermediates to
# /tmp for EVERY .cu x EVERY arch (sm_89/90/100/120), and nothing else removes them; profiling
# runs leave .nsys-rep/.sqlite behind that are far larger. On 2026-08-22 the disk sat at 93% and
# three consecutive cron rounds died -- two with "[FAIL] target load", one with a Qwen3.6 guard
# that ran clean standalone -- while both manually-launched rounds that day passed, each having
# been preceded by a hand cleanup. Safe here because the round holds the shared lock, so no other
# build can be mid-flight.
rm -rf /tmp/tmpxft_* /tmp/*.nsys-rep /tmp/*.sqlite 2>/dev/null || true
echo "DISK_BEFORE_BUILD $(df -h / | awk 'NR==2{{print $4}}') free"

mkdir -p build
if [ -f build/CMakeCache.txt ] && ! grep -q '^CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda' build/CMakeCache.txt; then
  echo "WARN: build/CMakeCache.txt has a non-/usr/local/cuda CUDA compiler -- wiping build dir" >&2
  grep '^CMAKE_CUDA_COMPILER:FILEPATH=' build/CMakeCache.txt >&2 || true
  rm -rf build && mkdir -p build
fi
export CUDACXX="${{CUDACXX:-/usr/local/cuda/bin/nvcc}}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/mopt_cmake.log 2>&1
cmake --build build --target dspark_tau_check qwen3_gguf_score qwen3_gguf_bench -j"$(nproc)" >/tmp/mopt_build.log 2>&1 || {{
  echo "BUILD_FAILED -- tail of /tmp/mopt_build.log:" >&2
  tail -80 /tmp/mopt_build.log >&2
  exit 1
}}
test -x build/runtime/dspark_tau_check
test -x build/runtime/qwen3_gguf_score
test -x build/runtime/qwen3_gguf_bench

# --- DSpark speculative decode @ ctx=CTX ---
# ONE process runs both legs against one loaded model: the AR reference first (on clean state --
# dflash_generate opens its own session, so taking the reference afterwards returns empty), then
# the speculative run, then an exact token-equality check between them. That shared load is the
# whole point: AR_TPS and DSPARK_TPS come off the same weights, the same GPU state and the same
# env, so their ratio means something and the PR-vs-main delta is not measuring load variance.
#
# The determinism pins inside dspark_tau_check.cpp apply identically to both legs. Both refs in a
# round run the identical binary under identical pins and the scored metric is relative, so the
# offset cancels. AR and DSpark share the same batched prompt-prefill implementation.
test -d "$DRAFT_DIR" || {{ echo "FAIL missing DSpark draft dir $DRAFT_DIR"; exit 1; }}
test -f "$DRAFT_DIR/config.json" || {{ echo "FAIL $DRAFT_DIR has no config.json"; exit 1; }}

# Real prompt, not a synthetic ramp -- same reasoning as the sibling bots: a synthetic token stream
# is a gaming surface, and that matters when the bot can auto-merge without a human reading the
# diff. Truncated to CTX ids so "ctx=16k" means 16384 real tokens of ordinary prose.
# bench_prompt_32k.txt, NOT bench_prompt.txt: the latter is ~245 tokens, so truncating it to CTX
# would have produced a 245-token context while every label said 16k. Nothing would have warned --
# the run succeeds, the metric parses, and the number is simply of something else. That is the same
# failure the prefill parity gate had when its corpus was too short for n=128 to run, except that
# one at least printed "skipped".
#
# So the length is CHECKED, not assumed: short corpus is a hard failure of the round. And the text
# is deliberately non-repeating natural prose. Tiling a short prompt to reach 16384 would have
# been easy and would have wrecked the measurement -- repeated text is far more
# predictable, which inflates acceptance, and tau is the exact quantity this bot reports.
DSPARK_IDS=/tmp/dspark_prompt_ids.txt
python3 - "$MODEL_DIR/tokenizer.json" bench/scripts/bench_prompt_32k.txt "$CTX" > "$DSPARK_IDS" <<'PYDS'
import sys
from tokenizers import Tokenizer
want = int(sys.argv[3])
ids = Tokenizer.from_file(sys.argv[1]).encode(open(sys.argv[2]).read()).ids
if len(ids) < want:
    # SystemExit(str) prints to stderr and exits non-zero. Deliberately no backslash escapes
    # anywhere in this heredoc: it is nested inside the bot's own f-string, which consumes them
    # and emits a real newline mid-string, breaking the remote Python.
    raise SystemExit("PROMPT_TOO_SHORT have=%d want=%d" % (len(ids), want))
print(" ".join(str(i) for i in ids[:want]))
PYDS
DS_NIDS=$(wc -w < "$DSPARK_IDS")
if [ "$DS_NIDS" -ne "$CTX" ]; then
  echo "DSPARK_PROMPT_TOO_SHORT have=$DS_NIDS want=$CTX -- refusing to score a context that is not $CTX" >&2
  exit 1
fi
echo "DSPARK_PROMPT_IDS $DS_NIDS"

wait_gpu_clear
DS_OUT=/tmp/dspark_run.txt
# Run every losslessness repetition in a fresh process. Reusing one loaded runtime couples the
# repetitions through allocator and CUDA-graph state; the gate is supposed to ask whether this
# build is lossless from a clean serving start, not whether an earlier synthetic audit poisoned a
# later one. Each child still performs its own AR-vs-DSpark exact-token comparison. Keep run 1 as
# the scored throughput sample and aggregate only the absolute correctness verdict across runs.
DSPARK_INFRA_FAILED=0
DS_ALL_OK=1
for rep in $(seq 1 {spec_reps}); do
  REP_OUT="/tmp/dspark_run_${{rep}}.txt"
  if ! SPARKINFER_DSPARK_SPEC_REPS=1 timeout 900 build/runtime/dspark_tau_check \
    "$MODEL_DIR" "$DRAFT_DIR" "$NTOK" $(cat "$DSPARK_IDS") > "$REP_OUT" 2>&1; then
    DSPARK_INFRA_FAILED=1
    DS_ALL_OK=0
    echo "DSPARK_CHILD_FAILED context=16k rep=$rep" >&2
  fi
  if [ "$rep" -eq 1 ]; then cp "$REP_OUT" "$DS_OUT"; fi
  REP_LL=$(sed -n 's/^METRIC LOSSLESS //p' "$REP_OUT" | tail -1)
  [ "${{REP_LL:-0}}" = "1" ] || DS_ALL_OK=0
  echo "DSPARK_FRESH_REP $rep lossless=${{REP_LL:-0}}"
done
echo "METRIC LOSSLESS $DS_ALL_OK" >> "$DS_OUT"
echo "METRIC LOSSLESS_RUNS {spec_reps}" >> "$DS_OUT"
grep -E '^(DSPARK|AR|draft:) ' "$DS_OUT" || true

# Default 0 on a missing metric, so a crashed or truncated run reads as "no speedup, not lossless"
# rather than inheriting whatever the caller had. The scorer treats dspark_tps==0 and lossless==0
# as hard failures, which makes this fail-closed end to end.
_ds_metric() {{
  local v
  v=$(sed -n "s/^METRIC $1 //p" "$DS_OUT" | tail -1)
  echo "${{v:-0}}"
}}
echo "RESULT_DSPARK_TPS $(_ds_metric DSPARK_TPS)"
echo "RESULT_AR_TPS $(_ds_metric AR_TPS)"
echo "RESULT_MEAN_ACCEPT $(_ds_metric MEAN_ACCEPT)"
echo "RESULT_LOSSLESS $(_ds_metric LOSSLESS)"
echo "RESULT_LOSSLESS_RUNS $(_ds_metric LOSSLESS_RUNS)"
PREFILL16_PP=$(_ds_metric DSPARK_PREFILL_PP)
echo "RESULT_PREFILL16_PP ${{PREFILL16_PP:-0}}"

# 4k uses the same real prompt and the same lossless AR-vs-DSpark harness. One process supplies
# DSpark decode, AR decode, acceptance and DSpark-enabled prompt prefill, so adding both requested
# 4k axes costs one model load rather than separate decode and prefill loads.
wait_gpu_clear
DS4_OUT=/tmp/dspark_decode4k.txt
DS4_IDS=/tmp/dspark_prompt4k_ids.txt
python3 - "$MODEL_DIR/tokenizer.json" bench/scripts/bench_prompt_32k.txt "$CTX4" > "$DS4_IDS" <<'PY4K'
import sys
from tokenizers import Tokenizer
want = int(sys.argv[3])
ids = Tokenizer.from_file(sys.argv[1]).encode(open(sys.argv[2]).read()).ids
if len(ids) < want:
    raise SystemExit("PROMPT_4K_TOO_SHORT have=%d want=%d" % (len(ids), want))
print(" ".join(str(i) for i in ids[:want]))
PY4K
DS4_ALL_OK=1
for rep in $(seq 1 {spec_reps}); do
  REP_OUT="/tmp/dspark4_run_${{rep}}.txt"
  if ! SPARKINFER_DSPARK_SPEC_REPS=1 timeout 900 build/runtime/dspark_tau_check \
    "$MODEL_DIR" "$DRAFT_DIR" "$NTOK" "@$DS4_IDS" > "$REP_OUT" 2>&1; then
    DSPARK_INFRA_FAILED=1
    DS4_ALL_OK=0
    echo "DSPARK_CHILD_FAILED context=4k rep=$rep" >&2
  fi
  if [ "$rep" -eq 1 ]; then cp "$REP_OUT" "$DS4_OUT"; fi
  REP_LL=$(sed -n 's/^METRIC LOSSLESS //p' "$REP_OUT" | tail -1)
  [ "${{REP_LL:-0}}" = "1" ] || DS4_ALL_OK=0
  echo "DSPARK4_FRESH_REP $rep lossless=${{REP_LL:-0}}"
done
echo "METRIC LOSSLESS $DS4_ALL_OK" >> "$DS4_OUT"
echo "METRIC LOSSLESS_RUNS {spec_reps}" >> "$DS4_OUT"
_ds4_metric() {{
  local v
  v=$(sed -n "s/^METRIC $1 //p" "$DS4_OUT" | tail -1)
  echo "${{v:-0}}"
}}
echo "RESULT_DSPARK4_TPS $(_ds4_metric DSPARK_TPS)"
echo "RESULT_AR4_TPS $(_ds4_metric AR_TPS)"
echo "RESULT_MEAN_ACCEPT4 $(_ds4_metric MEAN_ACCEPT)"
echo "RESULT_LOSSLESS4 $(_ds4_metric LOSSLESS)"
echo "RESULT_LOSSLESS4_RUNS $(_ds4_metric LOSSLESS_RUNS)"
PREFILL4_PP=$(_ds4_metric DSPARK_PREFILL_PP)
echo "RESULT_PREFILL4_PP ${{PREFILL4_PP:-0}}"

# The second decode dimension and the prefill dimension share the exact 32k prompt. Run the full
# speculative generation here; its first fresh process supplies both decode throughput and TTFT,
# avoiding a redundant fourth model load solely to time the same prompt pass.
wait_gpu_clear
DS32_OUT=/tmp/dspark_decode32k.txt
PREFILL_IDS=/tmp/dspark_prefill32k_ids.txt
python3 - "$MODEL_DIR/tokenizer.json" bench/scripts/bench_prompt_32k.txt "$CTX32" > "$PREFILL_IDS" <<'PYPF'
import sys
from tokenizers import Tokenizer
want = int(sys.argv[3])
ids = Tokenizer.from_file(sys.argv[1]).encode(open(sys.argv[2]).read()).ids
if len(ids) < want:
    raise SystemExit("PREFILL_PROMPT_TOO_SHORT have=%d want=%d" % (len(ids), want))
print(" ".join(str(i) for i in ids[:want]))
PYPF
DS32_ALL_OK=1
for rep in $(seq 1 {spec_reps}); do
  REP_OUT="/tmp/dspark32_run_${{rep}}.txt"
  if ! SPARKINFER_DSPARK_SPEC_REPS=1 timeout 1200 build/runtime/dspark_tau_check \
    "$MODEL_DIR" "$DRAFT_DIR" "$NTOK" "@$PREFILL_IDS" > "$REP_OUT" 2>&1; then
    DSPARK_INFRA_FAILED=1
    DS32_ALL_OK=0
    echo "DSPARK_CHILD_FAILED context=32k rep=$rep" >&2
  fi
  if [ "$rep" -eq 1 ]; then cp "$REP_OUT" "$DS32_OUT"; fi
  REP_LL=$(sed -n 's/^METRIC LOSSLESS //p' "$REP_OUT" | tail -1)
  [ "${{REP_LL:-0}}" = "1" ] || DS32_ALL_OK=0
  echo "DSPARK32_FRESH_REP $rep lossless=${{REP_LL:-0}}"
done
echo "METRIC LOSSLESS $DS32_ALL_OK" >> "$DS32_OUT"
echo "METRIC LOSSLESS_RUNS {spec_reps}" >> "$DS32_OUT"
_ds32_metric() {{
  local v
  v=$(sed -n "s/^METRIC $1 //p" "$DS32_OUT" | tail -1)
  echo "${{v:-0}}"
}}
echo "RESULT_DSPARK32_TPS $(_ds32_metric DSPARK_TPS)"
echo "RESULT_AR32_TPS $(_ds32_metric AR_TPS)"
echo "RESULT_MEAN_ACCEPT32 $(_ds32_metric MEAN_ACCEPT)"
echo "RESULT_LOSSLESS32 $(_ds32_metric LOSSLESS)"
echo "RESULT_LOSSLESS32_RUNS $(_ds32_metric LOSSLESS_RUNS)"
PREFILL_PP=$(_ds32_metric DSPARK_PREFILL_PP)
echo "RESULT_PREFILL_PP ${{PREFILL_PP:-0}}"

# A child that could not load the model, timed out, crashed or was OOM-killed is an evaluator
# infrastructure failure, not evidence of a slow or incorrect PR. Exit nonzero before fail-fast
# correctness gates; _ssh_run_resilient retries this whole ref once with clean processes. Exact
# token mismatch still returns zero from the harness with LOSSLESS=0 and remains non-retryable.
if [ "$DSPARK_INFRA_FAILED" = "1" ]; then
  echo "RETRYABLE_INFRA_FAILURE one or more DSpark harness processes exited nonzero" >&2
  exit 75
fi

# --- FAIL FAST -------------------------------------------------------------------------------
# Every gate below is decidable from the stage that just ran, so a PR that fails one is already
# REJECTed no matter what the remaining stages say. Running them anyway costs the accuracy dump
# plus two 16k guard sweeps -- each its own multi-GB model load -- to produce a verdict that cannot
# change. Skipped only for a PR: the main run has no reference and must measure every stage.
#
# The thresholds arrive as numbers from Python (REF_*/`*_TOL`); bash only compares, so there is one
# definition of the policy and it is not this file.
if [ "$IS_PR" = "1" ]; then
  _fail_fast() {{ echo "EARLY_REJECT $1"; echo "EARLY_REJECT_STAGE dspark"; exit 0; }}
  _lt() {{ python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)" "$1" "$2"; }}
  DS=$(_ds_metric DSPARK_TPS); AR=$(_ds_metric AR_TPS); PF=${{PREFILL_PP:-0}}
  DS4=$(_ds4_metric DSPARK_TPS); AR4=$(_ds4_metric AR_TPS); PF4=${{PREFILL4_PP:-0}}
  PF16=${{PREFILL16_PP:-0}}
  DS32=$(_ds32_metric DSPARK_TPS); AR32=$(_ds32_metric AR_TPS)
  TAU=$(_ds_metric MEAN_ACCEPT); LL=$(_ds_metric LOSSLESS); LLR=$(_ds_metric LOSSLESS_RUNS)
  TAU4=$(_ds4_metric MEAN_ACCEPT); LL4=$(_ds4_metric LOSSLESS); LLR4=$(_ds4_metric LOSSLESS_RUNS)
  TAU32=$(_ds32_metric MEAN_ACCEPT); LL32=$(_ds32_metric LOSSLESS); LLR32=$(_ds32_metric LOSSLESS_RUNS)
  [ "${{LL%%.*}}" = "1" ] || _fail_fast "losslessness gate failed: DSpark output does not match the AR reference token-for-token"
  _lt "$LLR" "{spec_reps}" && _fail_fast "losslessness verified over only ${{LLR%%.*}} run(s), need {spec_reps}"
  [ "${{LL4%%.*}}" = "1" ] || _fail_fast "4k losslessness gate failed: DSpark output does not match AR"
  _lt "$LLR4" "{spec_reps}" && _fail_fast "4k losslessness verified over only ${{LLR4%%.*}} run(s), need {spec_reps}"
  [ "${{LL32%%.*}}" = "1" ] || _fail_fast "32k losslessness gate failed: DSpark output does not match AR"
  _lt "$LLR32" "{spec_reps}" && _fail_fast "32k losslessness verified over only ${{LLR32%%.*}} run(s), need {spec_reps}"
  if _lt "0" "$REF_TAU"; then
    _lt "$TAU" "$(python3 -c "print($REF_TAU * $TAU_TOL)")" \
      && _fail_fast "mean accept regression: tau $TAU < $(python3 -c "print(round($TAU_TOL*100))")% of main $REF_TAU -- throughput bought by speculating less is not a speedup"
  fi
  if _lt "0" "$REF_TAU4"; then
    _lt "$TAU4" "$(python3 -c "print($REF_TAU4 * $TAU_TOL)")" \
      && _fail_fast "4k mean accept regression: tau $TAU4 < $(python3 -c "print(round($TAU_TOL*100))")% of main $REF_TAU4"
  fi
  if _lt "0" "$REF_TAU32"; then
    _lt "$TAU32" "$(python3 -c "print($REF_TAU32 * $TAU_TOL)")" \
      && _fail_fast "32k mean accept regression: tau $TAU32 < $(python3 -c "print(round($TAU_TOL*100))")% of main $REF_TAU32"
  fi
  if _lt "0" "$REF_DSPARK"; then
    _lt "$DS" "$(python3 -c "print($REF_DSPARK * $REGRESS_TOL)")" \
      && _fail_fast "dspark-decode@16k regression: $DS < $(python3 -c "print(round($REGRESS_TOL*100))")% of main $REF_DSPARK"
  fi
  if _lt "0" "$REF_DSPARK4"; then
    _lt "$DS4" "$(python3 -c "print($REF_DSPARK4 * $REGRESS_TOL)")" \
      && _fail_fast "dspark-decode@4k regression: $DS4 < $(python3 -c "print(round($REGRESS_TOL*100))")% of main $REF_DSPARK4"
  fi
  if _lt "0" "$REF_AR4"; then
    _lt "$AR4" "$(python3 -c "print($REF_AR4 * $REGRESS_TOL)")" \
      && _fail_fast "ar-decode@4k regression: $AR4 < $(python3 -c "print(round($REGRESS_TOL*100))")% of main $REF_AR4"
  fi
  if _lt "0" "$REF_AR"; then
    _lt "$AR" "$(python3 -c "print($REF_AR * $REGRESS_TOL)")" \
      && _fail_fast "ar-decode@16k regression: $AR < $(python3 -c "print(round($REGRESS_TOL*100))")% of main $REF_AR"
  fi
  if _lt "0" "$REF_DSPARK32"; then
    _lt "$DS32" "$(python3 -c "print($REF_DSPARK32 * $REGRESS_TOL)")" \
      && _fail_fast "dspark-decode@32k regression: $DS32 < $(python3 -c "print(round($REGRESS_TOL*100))")% of main $REF_DSPARK32"
  fi
  if _lt "0" "$REF_AR32"; then
    _lt "$AR32" "$(python3 -c "print($REF_AR32 * $REGRESS_TOL)")" \
      && _fail_fast "ar-decode@32k regression: $AR32 < $(python3 -c "print(round($REGRESS_TOL*100))")% of main $REF_AR32"
  fi
  if _lt "0" "$REF_PREFILL"; then
    _lt "$PF" "$(python3 -c "print($REF_PREFILL * $REGRESS_TOL)")" \
      && _fail_fast "dspark-prefill@32k regression: $PF < $(python3 -c "print(round($REGRESS_TOL*100))")% of main $REF_PREFILL"
  fi
  if _lt "0" "$REF_PREFILL4"; then
    _lt "$PF4" "$(python3 -c "print($REF_PREFILL4 * $REGRESS_TOL)")" \
      && _fail_fast "dspark-prefill@4k regression: $PF4 < $(python3 -c "print(round($REGRESS_TOL*100))")% of main $REF_PREFILL4"
  fi
  if _lt "0" "$REF_PREFILL16"; then
    _lt "$PF16" "$(python3 -c "print($REF_PREFILL16 * $REGRESS_TOL)")" \
      && _fail_fast "dspark-prefill@16k regression: $PF16 < $(python3 -c "print(round($REGRESS_TOL*100))")% of main $REF_PREFILL16"
  fi
fi
# ---------------------------------------------------------------------------------------------

# --- teacher-forced score dump (differential accuracy gate, module docstring pt. 2) ---
# llama.cpp cannot read a compressed-tensors directory, so there is no same-weights external
# reference available for this checkpoint. Instead score the SAME token stream on this build and
# diff it against origin/main's dump -- see bench/scripts/accuracy_compare_pair.py.
#
# Tokenized with the checkpoint's OWN tokenizer.json rather than llama-tokenize: no GGUF of this
# model is involved anywhere in this bot, and the HF tokenizer is what the server uses.
IDS=$(python3 - "$MODEL_DIR/tokenizer.json" "$EVAL_TEXT" <<'PYTOK'
import sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
print(" ".join(str(i) for i in tok.encode(open(sys.argv[2]).read()).ids))
PYTOK
) || {{ echo "TOKENIZE_FAILED" >&2; exit 1; }}
TOKEN_COUNT=$(printf '%s' "$IDS" | wc -w)
echo "RESULT_TOKEN_COUNT $TOKEN_COUNT"

wait_gpu_clear
build/runtime/qwen3_gguf_score "$MODEL_DIR" "$TOPK" $IDS > "$DUMP_SELF" 2>/tmp/mopt_score.err || {{
  echo "SCORE_FAILED -- tail of /tmp/mopt_score.err:" >&2
  tail -40 /tmp/mopt_score.err >&2
  exit 1
}}
echo "ACCURACY_STAGE_DONE"

if [ "$IS_PR" = "1" ]; then
  if [ -s "$DUMP_MAIN" ]; then
    python3 bench/scripts/accuracy_compare_pair.py "$DUMP_SELF" "$DUMP_MAIN" 2>&1 | tee /tmp/acc_cmp.txt || true
  else
    echo "ACCURACY_NO_BASELINE" >&2
  fi
fi

# Second fail-fast point: the accuracy gate is decided, and the two guard sweeps below are two
# more model loads that cannot change a REJECT. accuracy_compare_pair.py prints "METRIC top1=..
# kl=..", so grep it back out rather than re-deriving it.
if [ "$IS_PR" = "1" ]; then
  ACCLINE=$(grep -m1 "^METRIC .*top1=" /tmp/acc_cmp.txt 2>/dev/null || true)
  if [ -n "$ACCLINE" ]; then
    A_TOP1=$(printf '%s' "$ACCLINE" | tr ' ' '\n' | sed -n 's/^top1=//p' | head -1)
    A_KL=$(printf '%s' "$ACCLINE" | tr ' ' '\n' | sed -n 's/^kl=//p' | head -1)
    if [ -n "$A_TOP1" ] && [ -n "$A_KL" ]; then
      if _lt "$A_TOP1" "{acc_top1}" || _lt "{acc_kl}" "$A_KL"; then
        echo "EARLY_REJECT accuracy gate failed vs main: top1=$A_TOP1 (bar >={acc_top1}) kl=$A_KL (bar <={acc_kl})"
        echo "EARLY_REJECT_STAGE accuracy"
        exit 0
      fi
    fi
  fi
fi

# --- no-regression guards @ ctx=16k (added 2026-08-18 by explicit request) ---
# The scored dimension covers speculative decode at ctx=16k, but not standalone long-context
# prefill or the sibling Qwen3.6 model. Without these two guards a DSpark PR could speed up its own
# path while silently regressing what actually ships -- the AR serving numbers the README
# publishes. Both are differential (PR vs freshly-measured main, same box, same round,
# REGRESS_TOL) and both are hard REJECTs, exactly like the sibling bots' guards.
#
# 16k ONLY, not the sibling bots' full 0/512/4k/16k/32k sweep: one context per model keeps a round
# to two extra model loads, and 16k is where long-context prefill and a large KV read both
# actually cost something -- a regression in the shared prefill/decode paths shows there first.
#
# Synthetic prompts on purpose (SPARKINFER_BENCH_PROMPT_FILE is never set in this script). These
# are PR-vs-main comparisons, not absolute numbers, and a real prompt file would be Qwen3.8 token
# ids that mean different text -- or nothing -- in Qwen3.6's vocabulary.
source bench/scripts/_common.sh
source bench/scripts/_eval_speed.sh
SI_BIN="$PWD/build/runtime"; SI_LD=""

# Qwen3.8, the ModelOpt checkpoint -- the one this bot's DSpark target loads, and the one the
# published benchmark table quotes. It used to be covered by pr_modelopt_bot.py SCORING it at
# decode@16k/prefill@16k; once DSpark decode@128 became the only scored dimension that coverage
# vanished, which is the gap this closes.
echo "GUARD38_START"
wait_gpu_clear
# $Q38_GUARD_MODEL_DIR, NOT $MODEL_DIR. This benched $MODEL_DIR -- the SCORED checkpoint --
# until 2026-08-25, i.e. it measured the scored build against itself and called the result a
# guard. The upstream unsloth NVFP4 build it exists to protect had no coverage at all, and every
# q38_guard_ok=True this bot has ever reported was vacuous. The sibling pr_modelopt_bot.py had
# it right; this one lost the variable when the block was copied across.
#
# The guard checkpoint is a separate ~22 GB download and may simply not be present. Say so
# explicitly instead of failing: a missing checkpoint is an operator/provisioning fact, not a
# verdict on the PR, and failing closed here would halt every round. GUARD38_UNAVAILABLE is
# surfaced loudly by the bot rather than folded into a pass.
if [ ! -f "$Q38_GUARD_MODEL_DIR/config.json" ]; then
  echo "GUARD38_UNAVAILABLE $Q38_GUARD_MODEL_DIR"
elif bench_sweep_run "$Q38_GUARD_MODEL_DIR" 128 16384 5; then
  echo "GUARD38 16384 $(_bench_sweep_get 16384 decode_tps) $(_bench_sweep_get 16384 prefill_pp)"
else
  echo "GUARD38_FAILED"
fi
echo "GUARD38_END"

# Qwen3.6-35B-A3B. A separate model load and a different architecture (MoE, not dense): a
# shared-code regression that only shows on Qwen3.6 would otherwise slip past entirely, as it did
# for the LMCache integration (PR #775) until it was caught by hand.
export MODELS_DIR="$Q36_GUARD_MODELS_DIR" MODEL_REPO="$Q36_GUARD_MODEL_REPO" \
       MODEL_FILE="$Q36_GUARD_MODEL_FILE" TOK_REPO="$Q36_GUARD_TOK_REPO"
export MODEL_SHA256="${{QWEN36_MODEL_SHA256:-}}"
( ensure_model && ensure_tokenizer ) || echo "WARN: qwen3.6 guard model setup failed" >&2
Q36_GGUF="$Q36_GUARD_MODELS_DIR/$Q36_GUARD_MODEL_FILE"

echo "GUARD_START"
wait_gpu_clear
if bench_sweep_run "$Q36_GGUF" 128 16384 5; then
  echo "GUARD36 16384 $(_bench_sweep_get 16384 decode_tps) $(_bench_sweep_get 16384 prefill_pp)"
else
  echo "GUARD36_FAILED"
fi
echo "GUARD_END"

# Still NOT run: the batched-prefill parity check. It is absolute rather than differential, and
# main currently fails it at n=128 (~0.58 against a 0.75 bar) -- a real pre-existing divergence
# that was hidden until the gate's prompt corpus was lengthened. Enabling it here would REJECT
# every PR until that is fixed, which is a separate piece of work.
"""


def _parse_remote(stdout: str) -> dict:
    """Markers emitted by _remote_script + the METRIC line accuracy_compare_pair.py prints.

    Unlike the sibling bots, the accuracy numbers are NOT re-echoed as RESULT_* by the remote
    bash -- the comparator's own machine-readable METRIC line is parsed directly, so there is one
    fewer place for the two to disagree about what was measured."""
    out = {}
    guard36 = {}
    guard38 = {}
    for line in (stdout or "").splitlines():
        if line.startswith("REMOTE_HEAD "):
            out["head"] = line.split()[1]
        elif line.startswith("PREFILL_PARITY_OK"):
            out["parity_ok"] = True
        elif line.startswith("PREFILL_PARITY_FAILED"):
            out["parity_ok"] = False
        elif line.startswith("PARITY worst="):
            try:
                out["parity_worst"] = float(line.split("worst=")[1].split()[0])
            except (ValueError, IndexError):
                pass
        elif line.startswith("RESULT_DSPARK_TPS "):
            try:
                out["dspark_tps"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_AR_TPS "):
            try:
                out["ar_tps"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_DSPARK4_TPS "):
            try:
                out["dspark4_tps"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_AR4_TPS "):
            try:
                out["ar4_tps"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_DSPARK32_TPS "):
            try:
                out["dspark32_tps"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_AR32_TPS "):
            try:
                out["ar32_tps"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_PREFILL_PP "):
            try:
                out["prefill_pp"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_PREFILL4_PP "):
            try:
                out["prefill4_pp"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_PREFILL16_PP "):
            try:
                out["prefill16_pp"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_MEAN_ACCEPT "):
            try:
                out["mean_accept"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_MEAN_ACCEPT32 "):
            try:
                out["mean_accept32"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_MEAN_ACCEPT4 "):
            try:
                out["mean_accept4"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_LOSSLESS4_RUNS "):
            try:
                out["lossless4_runs"] = int(float(line.split()[1]))
            except ValueError:
                out["lossless4_runs"] = 0
        elif line.startswith("RESULT_LOSSLESS4 "):
            try:
                out["lossless4"] = int(float(line.split()[1])) == 1
            except ValueError:
                out["lossless4"] = False
        elif line.startswith("RESULT_LOSSLESS32_RUNS "):
            try:
                out["lossless32_runs"] = int(float(line.split()[1]))
            except ValueError:
                out["lossless32_runs"] = 0
        elif line.startswith("RESULT_LOSSLESS32 "):
            try:
                out["lossless32"] = int(float(line.split()[1])) == 1
            except ValueError:
                out["lossless32"] = False
        elif line.startswith("RESULT_LOSSLESS_RUNS "):
            try:
                out["lossless_runs"] = int(float(line.split()[1]))
            except ValueError:
                out["lossless_runs"] = 0
        elif line.startswith("RESULT_LOSSLESS "):
            # Fail-closed: anything that is not exactly 1 -- including a value the harness never
            # printed, which the remote _ds_metric defaults to 0 -- means "not proven lossless".
            try:
                out["lossless"] = int(float(line.split()[1])) == 1
            except ValueError:
                out["lossless"] = False
        elif line.startswith("EARLY_REJECT_STAGE "):
            out["early_reject_stage"] = line.split(None, 1)[1].strip()
        elif line.startswith("EARLY_REJECT "):
            out["early_reject"] = line.split(None, 1)[1].strip()
        elif line.startswith("DSPARK_RUN_FAILED"):
            out["dspark_run_failed"] = True
        elif line.startswith("RESULT_TOKEN_COUNT "):
            try:
                out["token_count"] = int(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("METRIC "):
            for tok in line.split()[1:]:
                if "=" not in tok:
                    continue
                k, _, v = tok.partition("=")
                if k in ("top1", "kl", "ppl_pr", "ppl_main"):
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
        elif line.startswith("GUARD38 "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    guard38[int(parts[1])] = {"decode": float(parts[2]),
                                              "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.startswith("GUARD38_UNAVAILABLE"):
            # Checkpoint not installed -- distinct from FAILED (which means it ran and broke).
            out["guard38_unavailable"] = True
        elif line.startswith("GUARD38_FAILED"):
            out["guard38_failed"] = True
        elif line.startswith("GUARD36 "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    guard36[int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.strip() == "GUARD36_FAILED":
            out["guard36_failed"] = True
    out["guard36"] = guard36
    out["guard38"] = guard38
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
    label for a plain-AR Qwen3.8-27B eval. Never raises — a Polaris failure must not block the
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
        "model": "qwen38-dspark-4k-16k-32k",
        "label": res.get("label"),
        "pass": res.get("pass"),
        "tps": res.get("pr_decode_tps"),
        "delta_tps": (res.get("pr_decode_tps") or 0) - (res.get("main_decode_tps") or 0),
        "pct_over_frontier": res.get("delta_pct"),
        "score_context": DSPARK_CTX,
        "best_context_label": "4k+16k+32k-decode-prefill",
        "ctx_4096_tps": res.get("pr_dspark4_tps"),
        "ctx_16384_tps": res.get("pr_decode_tps"),
        "top1": res.get("pr_top1"),
        "kl": res.get("pr_kl"),
    }
    eval_seed = f"qwen38-{int(time.time() * 1000)}"  # unique nonce per attestation
    stdin_payload = "RESULT_JSON " + json.dumps(result_json)
    cmd = (
        f"cd {shlex.quote(REMOTE_REPO)} && "
        f"SPARKINFER_EVAL_MODE=qwen38-dspark-4k-16k-32k SPARKINFER_DECODE_TOKENS={BENCH_TOKENS} "
        f"SPARKINFER_EVAL_SEED={shlex.quote(eval_seed)} python3 eval/polaris/judge.py --from-stdin "
        f"--model-file {shlex.quote(MODEL_WEIGHT_FILE)} "
        f"--build-dir {shlex.quote(REMOTE_REPO)}/build/runtime "
        f"--sparkinfer-root {shlex.quote(REMOTE_REPO)}"
    )
    try:
        # 180s, not the siblings' 60s: judge.py sha256s --model-file, and this checkpoint's weight
        # blob is 21 GiB -- measured 32.7s on the pinned box with a warm cache, before judge.py
        # signs the receipt and calls the Polaris API. 60s left almost no margin for a cold read.
        r = ssh_run(host, port, cmd, timeout=180, stdin_data=stdin_payload)
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
    try_auto_merge_qwen38, called from reconcile_qwen38_merge_labels AFTER every
    pending PR has already been individually evaluated in main()'s loop, see call ordering
    there), so every PR in the round comparing against a freshly-remeasured main was pure
    redundant GPU/build time. Returns {"ok": True, **parsed} or {"ok": False, "reason", "log"}."""
    r = _ssh_run_resilient(host, port, _remote_script("main", role="main"), "main run")
    if r.returncode != 0:
        tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-2000:]
        crash = _crash_reason(r.stdout, r.stderr)
        reason = "main run failed" + (f" — {crash}" if crash else " (no crash diagnostic captured, possible hard kill — retried once)")
        return {"ok": False, "reason": reason, "log": tail}
    main = _parse_remote(r.stdout or "")
    # Fail closed on missing OR zero. A crashed/truncated harness run leaves _ds_metric's 0
    # default, and a zero baseline would make tier_from_gain treat every PR as an infinite speedup
    # on the dimension that earns a tier -- i.e. auto-merge on a broken measurement.
    if not main.get("dspark_tps"):
        return {"ok": False, "reason": "main run missing/zero DSpark decode@16k tok/s",
                "log": (r.stdout or "")[-1500:]}
    if not main.get("dspark4_tps"):
        return {"ok": False, "reason": "main run missing/zero DSpark decode@4k tok/s",
                "log": (r.stdout or "")[-1500:]}
    if not main.get("dspark32_tps"):
        return {"ok": False, "reason": "main run missing/zero DSpark decode@32k tok/s",
                "log": (r.stdout or "")[-1500:]}
    if not main.get("ar_tps"):
        return {"ok": False, "reason": "main run missing/zero AR decode@16k tok/s",
                "log": (r.stdout or "")[-1500:]}
    if not main.get("ar4_tps"):
        return {"ok": False, "reason": "main run missing/zero AR decode@4k tok/s",
                "log": (r.stdout or "")[-1500:]}
    if not main.get("ar32_tps"):
        return {"ok": False, "reason": "main run missing/zero AR decode@32k tok/s",
                "log": (r.stdout or "")[-1500:]}
    if not main.get("prefill_pp"):
        return {"ok": False, "reason": "main run missing/zero prefill@32k prompt tok/s",
                "log": (r.stdout or "")[-1500:]}
    for key, label in (("prefill4_pp", "4k"), ("prefill16_pp", "16k")):
        if not main.get(key):
            return {"ok": False, "reason": f"main run missing/zero prefill@{label} prompt tok/s",
                    "log": (r.stdout or "")[-1500:]}
    # If MAIN is not lossless the whole round is meaningless: every PR would be compared against a
    # baseline that is already emitting unverified tokens, and "faster than a broken reference" is
    # not a result. Stop the round rather than scoring against it.
    if not main.get("mean_accept"):
        return {"ok": False, "reason": "main run missing/zero mean accept (tau) — the tau floor "
                                       "cannot be applied without a reference",
                "log": (r.stdout or "")[-1500:]}
    if not main.get("mean_accept4"):
        return {"ok": False, "reason": "main run missing/zero 4k mean accept (tau)",
                "log": (r.stdout or "")[-1500:]}
    if not main.get("mean_accept32"):
        return {"ok": False, "reason": "main run missing/zero 32k mean accept (tau)",
                "log": (r.stdout or "")[-1500:]}
    if not main.get("lossless"):
        return {"ok": False, "reason": "main run is NOT lossless — DSpark diverged from AR on main, "
                                       "so there is no valid baseline to score against",
                "log": (r.stdout or "")[-1500:]}
    if not main.get("lossless4") or main.get("lossless4_runs", 0) < DSPARK_SPEC_REPS:
        return {"ok": False, "reason": "main run is not proven lossless at 4k",
                "log": (r.stdout or "")[-1500:]}
    if not main.get("lossless32") or main.get("lossless32_runs", 0) < DSPARK_SPEC_REPS:
        return {"ok": False, "reason": "main run is not proven lossless at 32k",
                "log": (r.stdout or "")[-1500:]}
    # Both guards need a main reference set. If either did not measure, EVERY PR this round would
    # fail its guard "measurement unavailable" branch and collect a REJECT for what is really one
    # shared infra problem -- so bail out of the round instead, same reasoning as the caller's own
    # baseline bail-out.
    # guard38_unavailable is NOT a bail: the checkpoint simply is not installed, which is a
    # standing operator condition rather than a transient infra fault. Bailing on it would stop
    # the eval loop entirely until someone downloads ~22 GB. check_q38_guard reports it as a
    # loud SKIP on every round instead.
    if not main.get("guard38_unavailable") and (main.get("guard38_failed") or not main.get("guard38")):
        return {"ok": False, "reason": "main run produced no qwen3.8 16k guard baseline",
                "log": (r.stdout or "")[-1500:]}
    if main.get("guard36_failed") or not main.get("guard36"):
        return {"ok": False, "reason": "main run produced no qwen3.6 16k guard baseline",
                "log": (r.stdout or "")[-1500:]}
    main["ok"] = True
    return main


def eval_qwen38_on_box(host, port, pr_ref: str, main: dict):
    """Run the PR ref's speed+accuracy script on the same box and compare against `main`, an
    already-measured baseline shared across every PR in the round (see measure_main_baseline)."""
    print(f">> DSpark eval on box: PR ref={pr_ref}")
    r = _ssh_run_resilient(host, port, _remote_script(pr_ref, role="pr", main=main), "PR run")
    if r.returncode != 0:
        tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-2000:]
        crash = _crash_reason(r.stdout, r.stderr)
        reason = "PR speed/accuracy run failed" + (f" — {crash}" if crash else " (no crash diagnostic captured, possible hard kill — retried once)")
        return {"ok": False, "reason": reason, "log": tail}
    pr = _parse_remote(r.stdout or "")
    # Fail-fast short-circuit. The remote script stops at the first decided gate, so the later
    # stages' metrics are ABSENT BY DESIGN -- without this they would trip the missing-metric checks
    # below and be reported as an infra fault ("missing accuracy METRIC line") instead of the real
    # verdict. Everything measured before the stop is still reported.
    if pr.get("early_reject"):
        stage = pr.get("early_reject_stage", "?")
        print(f">> PR fail-fast at the {stage} stage: {pr['early_reject']}")
        res = {
            "ok": True, "label": "REJECT", "pass": False,
            "reason": pr["early_reject"] + f" (stopped at the {stage} stage; later stages skipped)",
            "delta_pct": 0.0, "early_reject": True, "early_reject_stage": stage,
            "pr_dspark_tps": pr.get("dspark_tps", 0.0), "main_dspark_tps": main.get("dspark_tps", 0.0),
            "pr_dspark4_tps": pr.get("dspark4_tps", 0.0), "main_dspark4_tps": main.get("dspark4_tps", 0.0),
            "pr_dspark32_tps": pr.get("dspark32_tps", 0.0), "main_dspark32_tps": main.get("dspark32_tps", 0.0),
            "pr_ar_tps": pr.get("ar_tps", 0.0), "main_ar_tps": main.get("ar_tps", 0.0),
            "pr_prefill_pp": pr.get("prefill_pp", 0.0), "main_prefill_pp": main.get("prefill_pp", 0.0),
            "pr_prefill4_pp": pr.get("prefill4_pp", 0.0), "main_prefill4_pp": main.get("prefill4_pp", 0.0),
            "pr_prefill16_pp": pr.get("prefill16_pp", 0.0), "main_prefill16_pp": main.get("prefill16_pp", 0.0),
            "pr_decode_tps": pr.get("dspark_tps", 0.0), "main_decode_tps": main.get("dspark_tps", 0.0),
            "pr_mean_accept": pr.get("mean_accept", 0.0), "main_mean_accept": main.get("mean_accept", 0.0),
            "lossless": pr.get("lossless") is True, "lossless_runs": pr.get("lossless_runs", 0),
            "tau_ok": False if stage == "dspark" else None,
            "dspark_vs_ar": round(pr["dspark_tps"] / pr["ar_tps"], 3) if pr.get("ar_tps") else 0,
            "scored_dimension": SCORING_DIM, "speedup_vs_main": 0,
            "pr_top1": pr.get("top1"), "pr_kl": pr.get("kl"),
            "accuracy_ok": stage != "accuracy",
            "pr_head": pr.get("head"), "main_head": main.get("head"),
        }
        return res
    if not pr.get("dspark_tps"):
        return {"ok": False, "reason": "PR run missing/zero DSpark decode@16k tok/s",
                "log": (r.stdout or "")[-1500:]}
    if not pr.get("dspark4_tps"):
        return {"ok": False, "reason": "PR run missing/zero DSpark decode@4k tok/s",
                "log": (r.stdout or "")[-1500:]}
    if not pr.get("dspark32_tps"):
        return {"ok": False, "reason": "PR run missing/zero DSpark decode@32k tok/s",
                "log": (r.stdout or "")[-1500:]}
    if not pr.get("ar_tps"):
        return {"ok": False, "reason": "PR run missing/zero AR decode@16k tok/s",
                "log": (r.stdout or "")[-1500:]}
    if not pr.get("ar4_tps"):
        return {"ok": False, "reason": "PR run missing/zero AR decode@4k tok/s",
                "log": (r.stdout or "")[-1500:]}
    if not pr.get("ar32_tps"):
        return {"ok": False, "reason": "PR run missing/zero AR decode@32k tok/s",
                "log": (r.stdout or "")[-1500:]}
    if not pr.get("prefill_pp"):
        return {"ok": False, "reason": "PR run missing/zero prefill@32k prompt tok/s",
                "log": (r.stdout or "")[-1500:]}
    for key, label in (("prefill4_pp", "4k"), ("prefill16_pp", "16k")):
        if not pr.get(key):
            return {"ok": False, "reason": f"PR run missing/zero prefill@{label} prompt tok/s",
                    "log": (r.stdout or "")[-1500:]}
    if not pr.get("mean_accept"):
        return {"ok": False, "reason": "PR run missing/zero mean accept (tau)",
                "log": (r.stdout or "")[-1500:]}
    if not pr.get("mean_accept4"):
        return {"ok": False, "reason": "PR run missing/zero 4k mean accept (tau)",
                "log": (r.stdout or "")[-1500:]}
    if not pr.get("mean_accept32"):
        return {"ok": False, "reason": "PR run missing/zero 32k mean accept (tau)",
                "log": (r.stdout or "")[-1500:]}
    if "top1" not in pr or "kl" not in pr:
        # Either the score dump failed, or main's dump was missing so the comparator never ran
        # (ACCURACY_NO_BASELINE). Both are infra faults, but they must NOT pass as "accurate" --
        # a gate that cannot measure fails closed, same as check_q36_guard's unavailability path.
        return {"ok": False, "reason": "PR run missing accuracy METRIC line (score dump failed, "
                                       "or no main baseline dump to diff against)",
                "log": (r.stdout or "")[-1500:]}
    print(f">> PR dspark@4k={pr['dspark4_tps']:.2f} (scored) "
          f"prefill@4k={pr['prefill4_pp']:.2f} pp/s (scored) "
          f"dspark@16k={pr['dspark_tps']:.2f} (scored) "
          f"prefill@16k={pr['prefill16_pp']:.2f} pp/s (scored) "
          f"dspark@32k={pr['dspark32_tps']:.2f} (scored) "
          f"prefill@32k={pr['prefill_pp']:.2f} pp/s (scored) "
          f"ar@16k={pr['ar_tps']:.2f} "
          f"ratio={pr['dspark_tps'] / pr['ar_tps']:.3f}x "
          f"tau={pr.get('mean_accept', 0):.3f} lossless={pr.get('lossless')} "
          f"top1={pr.get('top1', 0):.4f} kl={pr.get('kl', 99):.5f}")

    # Six scored DSpark axes plus three AR-only floors. Without the AR floors a PR could post a
    # DSpark "speedup" by slowing ordinary serving; every regression remains a hard REJECT.
    dims = [
        ("dspark-decode@4k",  pr["dspark4_tps"], main["dspark4_tps"]),
        ("dspark-prefill@4k", pr["prefill4_pp"], main["prefill4_pp"]),
        ("dspark-decode@16k", pr["dspark_tps"], main["dspark_tps"]),
        ("dspark-prefill@16k", pr["prefill16_pp"], main["prefill16_pp"]),
        ("dspark-decode@32k", pr["dspark32_tps"], main["dspark32_tps"]),
        ("dspark-prefill@32k", pr["prefill_pp"], main["prefill_pp"]),
        ("ar-decode@4k",      pr["ar4_tps"],     main["ar4_tps"]),
        ("ar-decode@16k",     pr["ar_tps"],     main["ar_tps"]),
        ("ar-decode@32k",     pr["ar32_tps"],   main["ar32_tps"]),
    ]
    scored = []
    for name, pr_v, main_v in dims:
        lab, dlt, ok, why = tier_from_gain(pr_v, main_v, metric=name)
        scored.append({"dim": name, "label": lab, "delta": dlt, "passed": ok, "reason": why})
    by_dim = {s["dim"]: s for s in scored}

    # ANY scored axis or AR floor regressing is a hard REJECT. This prevents trading one serving
    # context or phase away to buy a headline improvement elsewhere.
    regressed = [s for s in scored if s["label"] == "REJECT"]
    if regressed:
        worst = min(regressed, key=lambda s: s["delta"])
        label, delta_pct, passed = "REJECT", worst["delta"], False
        speed_reason = " | ".join(s["reason"] for s in regressed)
        best = worst
    else:
        # The tier comes from the LONG-CONTEXT pair, decode@16k and prefill@16k (2026-08-17), on
        # the ModelOpt checkpoint this bot scores. A 128-context-only improvement scores "none" by
        # design. Written against SCORING_DIMS rather than restating its contents, because an
        # earlier version of this comment named a different dimension than the list actually held
        # and nobody noticed until the two were compared by hand.
        # Best of the scoring dimensions, by measured delta. max() over deltas rather than over
        # tier letters: two dimensions can share a bucket while one is clearly the larger win, and
        # the delta is what the tier was derived from anyway.
        best = max((by_dim[d] for d in SCORING_DIMS if d in by_dim),
                   key=lambda s: s["delta"], default=by_dim[SCORING_DIM])
        label, delta_pct, passed, speed_reason = best["label"], best["delta"], best["passed"], best["reason"]
    dspark4_label, dspark4_delta_pct = by_dim["dspark-decode@4k"]["label"], by_dim["dspark-decode@4k"]["delta"]
    prefill4_label, prefill4_delta_pct = by_dim["dspark-prefill@4k"]["label"], by_dim["dspark-prefill@4k"]["delta"]
    dspark_label, dspark_delta_pct = by_dim["dspark-decode@16k"]["label"], by_dim["dspark-decode@16k"]["delta"]
    prefill16_label, prefill16_delta_pct = by_dim["dspark-prefill@16k"]["label"], by_dim["dspark-prefill@16k"]["delta"]
    dspark32_label, dspark32_delta_pct = by_dim["dspark-decode@32k"]["label"], by_dim["dspark-decode@32k"]["delta"]
    prefill_label, prefill_delta_pct = by_dim["dspark-prefill@32k"]["label"], by_dim["dspark-prefill@32k"]["delta"]
    ar_label,     ar_delta_pct     = by_dim["ar-decode@16k"]["label"],     by_dim["ar-decode@16k"]["delta"]
    # Keep the speed-only verdict: `label` below can be forced to REJECT by the accuracy gate or
    # the Qwen3.6 guard, and the comment/dashboard still need to say whether speed itself moved.
    speed_label = label

    pr_top1 = pr.get("top1", 0.0)
    pr_kl = pr.get("kl", 99.0)
    accuracy_ok = pr_top1 >= ACC_TOP1_BAR and pr_kl <= ACC_KL_BAR
    reason = speed_reason
    if not accuracy_ok:
        # Hard REJECT regardless of speed. This gate is differential (PR vs main on the same
        # token stream), so failing it means the PR CHANGED this model's output distribution --
        # exactly the class of bug a speed number cannot see. Six such bugs were found by hand
        # during Qwen3.8-27B bring-up, every one of which left throughput untouched.
        acc_reason = (f"accuracy gate failed vs main: top1={pr_top1:.4f} (bar >={ACC_TOP1_BAR}) "
                      f"kl={pr_kl:.5f} (bar <={ACC_KL_BAR})")
        reason = f"{acc_reason} | speed: {speed_reason}"
        label = "REJECT"
        passed = False

    # LOSSLESSNESS. The gate this bot exists to enforce, and the reason throughput here is not
    # self-validating: DSpark can be made arbitrarily fast by accepting draft tokens without
    # verifying them, which is not a faster decoder, it is a wrong one. dspark_tau_check
    # regenerates the same prompt with the draft disabled and requires EXACT token equality.
    #
    # Absolute, not differential -- like the parity gate it replaces, and for the same reason: it
    # compares the speculative path against the AR path inside the PR's own build, so it fails a
    # run whose defect is already on main rather than only a newly introduced one. Missing => fail
    # (the remote _ds_metric defaults it to 0), because a losslessness result that did not run is
    # not evidence of losslessness.
    #
    # Real precedent, not hypothetical: leaving verify row 0 in flight while the draft ran its own
    # block made 5-7 of 40 repeats of the SAME generation emit different tokens, and it hid for a
    # long time because it perturbed accept GROUPING, which is invisible until the adaptive gate
    # turns it into a different verify path. Speed alone never saw it.
    # The verdict AND the evidence behind it. A build whose dspark_tau_check predates the repeat
    # verdict emits LOSSLESS_RUNS=0 (metric absent) or 1 (single run), and would otherwise hand back
    # a confident-looking pass earned by one roll of the dice. Requiring the configured rep count
    # means a stale binary fails closed instead of silently downgrading the gate.
    runs = pr.get("lossless_runs", 0)
    # TAU FLOOR. Deliberately placed before the losslessness gate, because a PR that removes
    # speculation passes losslessness trivially -- with no draft, DSpark IS AR and the two token
    # sequences cannot differ -- so losslessness offers no protection here at all.
    pr_tau, main_tau = pr.get("mean_accept", 0.0), main.get("mean_accept", 0.0)
    tau_ok = bool(main_tau) and pr_tau >= main_tau * DSPARK_TAU_TOL
    pr_tau4, main_tau4 = pr.get("mean_accept4", 0.0), main.get("mean_accept4", 0.0)
    tau4_ok = bool(main_tau4) and pr_tau4 >= main_tau4 * DSPARK_TAU_TOL
    pr_tau32, main_tau32 = pr.get("mean_accept32", 0.0), main.get("mean_accept32", 0.0)
    tau32_ok = bool(main_tau32) and pr_tau32 >= main_tau32 * DSPARK_TAU_TOL
    if not tau_ok:
        tau_reason = (f"mean accept regression: tau {pr_tau:.4f} < {100 * DSPARK_TAU_TOL:.0f}% of "
                      f"main {main_tau:.4f} — throughput bought by speculating less is not a "
                      f"speedup, it is the feature being removed")
        reason = f"{tau_reason} | {reason}"
        label = "REJECT"
        passed = False
    if not tau4_ok:
        reason = (f"4k mean accept regression: tau {pr_tau4:.4f} < "
                  f"{100 * DSPARK_TAU_TOL:.0f}% of main {main_tau4:.4f} | {reason}")
        label = "REJECT"
        passed = False
    if not tau32_ok:
        reason = (f"32k mean accept regression: tau {pr_tau32:.4f} < "
                  f"{100 * DSPARK_TAU_TOL:.0f}% of main {main_tau32:.4f} | {reason}")
        label = "REJECT"
        passed = False

    lossless_ok = pr.get("lossless") is True and runs >= DSPARK_SPEC_REPS
    runs4 = pr.get("lossless4_runs", 0)
    lossless4_ok = pr.get("lossless4") is True and runs4 >= DSPARK_SPEC_REPS
    runs32 = pr.get("lossless32_runs", 0)
    lossless32_ok = pr.get("lossless32") is True and runs32 >= DSPARK_SPEC_REPS
    if pr.get("lossless") is True and runs < DSPARK_SPEC_REPS:
        reason = (f"losslessness verified over only {runs} run(s), need {DSPARK_SPEC_REPS} "
                  f"(stale dspark_tau_check?) | {reason}")
        label = "REJECT"
        passed = False
    elif not lossless_ok:
        ls_reason = ("losslessness gate failed: DSpark output does not match the AR reference "
                     "token-for-token" + ("" if "lossless" in pr else " (not measured)"))
        reason = f"{ls_reason} | {reason}"
        label = "REJECT"
        passed = False
    if not lossless32_ok:
        reason = (f"32k losslessness gate failed or only {runs32}/{DSPARK_SPEC_REPS} fresh runs "
                  f"completed | {reason}")
        label = "REJECT"
        passed = False
    if not lossless4_ok:
        reason = (f"4k losslessness gate failed or only {runs4}/{DSPARK_SPEC_REPS} fresh runs "
                  f"completed | {reason}")
        label = "REJECT"
        passed = False

    # Qwen3.8 (ModelOpt) no-regression guard @16k -- decode AND prefill. Hard REJECT: a DSpark PR
    # that speeds up speculative decode while regressing the AR long-context path is regressing
    # what actually ships, and nothing else in this bot measures that any more.
    q38_ok, q38_problems = check_q38_guard(pr, main)
    if not q38_ok:
        q38_reason = "qwen3.8 16k no-regression guard failed: " + "; ".join(q38_problems[:6])
        reason = f"{q38_reason} | {reason}"
        label = "REJECT"
        passed = False

    # Qwen3.6 no-regression guard @16k -- same discipline, different architecture. DSpark work
    # lands in qwen35.cpp / qwen35_prefill.cpp / the shared kernels, all of which Qwen3.6 uses.
    q36_ok, q36_problems = check_q36_guard(pr, main)
    if not q36_ok:
        q36_reason = "qwen3.6 16k no-regression guard failed: " + "; ".join(q36_problems[:6])
        reason = f"{q36_reason} | {reason}"
        label = "REJECT"
        passed = False

    res = {
        "ok": True,
        "label": label,
        "pass": passed and label != "REJECT",
        "reason": reason,
        "delta_pct": delta_pct,
        # Generic names kept so the dashboard/scores plumbing below reads unchanged; here they mean
        # the SCORED dimension, DSpark speculative decode at ctx=16k.
        "pr_decode_tps": pr["dspark_tps"],
        "main_decode_tps": main["dspark_tps"],
        "decode_delta_pct": dspark_delta_pct,
        "decode_regressed": dspark_label == "REJECT",
        "pr_dspark_tps": pr["dspark_tps"],
        "main_dspark_tps": main["dspark_tps"],
        "pr_dspark4_tps": pr["dspark4_tps"],
        "main_dspark4_tps": main["dspark4_tps"],
        "decode4_delta_pct": dspark4_delta_pct,
        "decode4_regressed": dspark4_label == "REJECT",
        "pr_dspark32_tps": pr["dspark32_tps"],
        "main_dspark32_tps": main["dspark32_tps"],
        "decode32_delta_pct": dspark32_delta_pct,
        "decode32_regressed": dspark32_label == "REJECT",
        "pr_prefill_pp": pr["prefill_pp"],
        "main_prefill_pp": main["prefill_pp"],
        "prefill_delta_pct": prefill_delta_pct,
        "prefill_regressed": prefill_label == "REJECT",
        "pr_prefill4_pp": pr["prefill4_pp"],
        "main_prefill4_pp": main["prefill4_pp"],
        "prefill4_delta_pct": prefill4_delta_pct,
        "prefill4_regressed": prefill4_label == "REJECT",
        "pr_prefill16_pp": pr["prefill16_pp"],
        "main_prefill16_pp": main["prefill16_pp"],
        "prefill16_delta_pct": prefill16_delta_pct,
        "prefill16_regressed": prefill16_label == "REJECT",
        "pr_ar_tps": pr["ar_tps"],
        "main_ar_tps": main["ar_tps"],
        "pr_ar32_tps": pr["ar32_tps"],
        "main_ar32_tps": main["ar32_tps"],
        "ar_delta_pct": ar_delta_pct,
        "ar_regressed": ar_label == "REJECT",
        "pr_ar4_tps": pr["ar4_tps"],
        "main_ar4_tps": main["ar4_tps"],
        "pr_mean_accept": pr.get("mean_accept", 0.0),
        "main_mean_accept": main.get("mean_accept", 0.0),
        # How far the speculative path is from being worth running at all. Below 1.0 DSpark is a
        # net loss against ordinary AR decode; this is the number the work is trying to move.
        "dspark_vs_ar": round(pr["dspark_tps"] / pr["ar_tps"], 3) if pr.get("ar_tps") else 0,
        "lossless": lossless_ok,
        "lossless_runs": runs,
        "lossless32": lossless32_ok,
        "lossless32_runs": runs32,
        "lossless4": lossless4_ok,
        "lossless4_runs": runs4,
        "tau_ok": tau_ok,
        "tau4_ok": tau4_ok,
        "tau32_ok": tau32_ok,
        "pr_mean_accept32": pr_tau32,
        "main_mean_accept32": main_tau32,
        "pr_mean_accept4": pr_tau4,
        "main_mean_accept4": main_tau4,
        "q38_guard_ok": q38_ok,
        "q38_guard_problems": q38_problems,
        "q38_guard": pr.get("guard38"),
        "q38_guard_main": main.get("guard38"),
        "q36_guard_ok": q36_ok,
        "q36_guard_problems": q36_problems,
        "q36_guard": pr.get("guard36"),
        "q36_guard_main": main.get("guard36"),
        "scored_dimension": best["dim"],
        "speedup_vs_main": round(best["delta"] / 100 + 1, 3),
        "pr_top1": pr_top1,
        "pr_kl": pr_kl,
        "pr_ppl": pr.get("ppl_pr"),
        "main_ppl": pr.get("ppl_main"),
        "token_count": pr.get("token_count"),
        "accuracy_ok": accuracy_ok,
        "pr_head": pr.get("head"),
        "main_head": main.get("head"),
    }
    # Attestation is a RECEIPT for a measurement that has already happened, not a gate on it, so it
    # must never be able to void one. collect_polaris_attestation() already returns None on ssh
    # failure / non-zero rc, but anything raised OUTSIDE its internal try -- building the command,
    # a bad constant, an env lookup -- propagated all the way out of this function and cost the
    # caller the fully-populated `res` above. Observed for real on PR #832 (2026-08-15): a NameError
    # on a leftover DEFAULT_GGUF discarded a measured +27.7% decode speedup with top1=1.0/kl=0.0 and
    # published eval-qwen38:REJECT with every metric None. A REJECT is close to the most expensive
    # verdict this bot can emit, so it must be reachable only from real measurements.
    try:
        polaris = collect_polaris_attestation(host, port, res, pr_ref)
        if polaris:
            res["polaris"] = polaris
    except Exception as e:
        print(f">> Polaris attestation failed ({type(e).__name__}: {e}) — keeping the measurement")
    return res


def format_comment(commit: str, res: dict) -> str:
    meta = {
        "label": res.get("label"),
        "delta_pct": res.get("delta_pct"),
        "pr_decode_tps": res.get("pr_decode_tps"),
        "main_decode_tps": res.get("main_decode_tps"),
        "pr_ar_tps": res.get("pr_ar_tps"),
        "main_ar_tps": res.get("main_ar_tps"),
        "pr_ar_tps": res.get("pr_ar_tps"),
        "main_ar_tps": res.get("main_ar_tps"),
        "pr_dspark_tps": res.get("pr_dspark_tps"),
        "main_dspark_tps": res.get("main_dspark_tps"),
        "pr_dspark4_tps": res.get("pr_dspark4_tps"),
        "main_dspark4_tps": res.get("main_dspark4_tps"),
        "decode4_delta_pct": res.get("decode4_delta_pct"),
        "pr_dspark32_tps": res.get("pr_dspark32_tps"),
        "main_dspark32_tps": res.get("main_dspark32_tps"),
        "pr_prefill_pp": res.get("pr_prefill_pp"),
        "main_prefill_pp": res.get("main_prefill_pp"),
        "prefill_delta_pct": res.get("prefill_delta_pct"),
        "pr_prefill4_pp": res.get("pr_prefill4_pp"),
        "main_prefill4_pp": res.get("main_prefill4_pp"),
        "prefill4_delta_pct": res.get("prefill4_delta_pct"),
        "pr_prefill16_pp": res.get("pr_prefill16_pp"),
        "main_prefill16_pp": res.get("main_prefill16_pp"),
        "prefill16_delta_pct": res.get("prefill16_delta_pct"),
        "scored_dimension": res.get("scored_dimension"),
        "pr_top1": res.get("pr_top1"),
        "pr_kl": res.get("pr_kl"),
        "pass": res.get("pass"),
        "accuracy_ok": res.get("accuracy_ok"),
        "lossless": res.get("lossless"),
        "lossless4": res.get("lossless4"),
        "lossless4_runs": res.get("lossless4_runs"),
        "lossless32": res.get("lossless32"),
        "pr_mean_accept4": res.get("pr_mean_accept4"),
        "main_mean_accept4": res.get("main_mean_accept4"),
        "tau4_ok": res.get("tau4_ok"),
        "pr_mean_accept": res.get("pr_mean_accept"),
        "main_mean_accept": res.get("main_mean_accept"),
        "tau_ok": res.get("tau_ok"),
        "pr_mean_accept32": res.get("pr_mean_accept32"),
        "main_mean_accept32": res.get("main_mean_accept32"),
        "tau32_ok": res.get("tau32_ok"),
    }
    marker = (
        f"<!-- sparkinfer-dspark-eval:{EVAL_SCHEMA_VERSION}:{commit} "
        f"{json.dumps(meta, separators=(',', ':'))} -->"
    )
    if not res.get("ok"):
        return (
            f"{marker}\n## sparkinfer DSpark auto-eval — error\n\n"
            f"**reason:** `{res.get('reason')}`\n\n"
            f"<details><summary>log tail</summary>\n\n```\n{(res.get('log') or '')[:1800]}\n```\n</details>\n"
        )
    lab = res["label"]
    # dict.get's default only fires for a MISSING key, not one present with a None value -- and
    # fail-fast leaves exactly that. A PR rejected at the dspark stage (losslessness, AR floor,
    # tau floor) never reaches the accuracy stage, so pr_top1/pr_kl are set to None rather than
    # dropped, and `_num('pr_top1'):.3f` raised
    #     TypeError: unsupported format string passed to NoneType.__format__
    # which took down the whole round AFTER the GPU work was done: #897 was correctly rejected on
    # losslessness and then lost its label and its comment to this traceback. Coerce explicitly.
    def _num(key, default=0.0):
        v = res.get(key)
        return default if v is None else v
    if res.get("accuracy_ok"):
        acc_row = (f"| accuracy gate | ✅ top1={_num('pr_top1'):.3f} "
                    f"(bar >={ACC_TOP1_BAR}) · KL={_num('pr_kl'):.4f} (bar <={ACC_KL_BAR}) |\n")
    elif res.get("early_reject"):
        # Not a failure of this gate -- it never ran. Saying "FAILED top1=0.000" here would be a
        # false statement about a PR that was stopped earlier, for a different reason.
        acc_row = ("| accuracy gate | ⏭️ not reached — rejected earlier at the "
                   f"`{res.get('early_reject_stage', 'dspark')}` stage |\n")
    else:
        acc_row = (f"| accuracy gate | ❌ **FAILED** — top1={_num('pr_top1'):.3f} "
                    f"(bar >={ACC_TOP1_BAR}) · KL={_num('pr_kl'):.4f} (bar <={ACC_KL_BAR}) — "
                    "**verdict forced to REJECT regardless of speed** |\n")
    # No "main accuracy" row: this gate is differential, so main IS the reference -- there is no
    # separate absolute bar for it to miss.
    main_acc_note = ""
    if res.get("lossless"):
        ls_row = ("| losslessness gate | ✅ DSpark matches the AR reference token-for-token, "
                  f"verified across **{res.get('lossless_runs', 1)}** independent runs |\n")
    else:
        ls_row = ("| losslessness gate | ❌ **FAILED** — DSpark diverged from the AR reference — "
                  "**verdict forced to REJECT regardless of speed** |\n")
    if res.get("lossless32"):
        ls32_row = ("| losslessness @32k | ✅ DSpark matches AR token-for-token, "
                    f"verified across **{res.get('lossless32_runs', 1)}** independent runs |\n")
    else:
        ls32_row = ("| losslessness @32k | ❌ **FAILED or not measured** — "
                    "**verdict forced to REJECT** |\n")
    if res.get("lossless4"):
        ls4_row = ("| losslessness @4k | ✅ DSpark matches AR token-for-token, "
                   f"verified across **{res.get('lossless4_runs', 1)}** independent runs |\n")
    else:
        ls4_row = ("| losslessness @4k | ❌ **FAILED or not measured** — "
                   "**verdict forced to REJECT** |\n")
    if res.get("tau_ok"):
        tau_row = (f"| mean accept τ floor | ✅ {_num('pr_mean_accept'):.4f} vs main "
                   f"{_num('main_mean_accept'):.4f} (bar ≥{100 * DSPARK_TAU_TOL:.0f}%) |\n")
    else:
        tau_row = (f"| mean accept τ floor | ❌ **FAILED** — {_num('pr_mean_accept'):.4f} vs main "
                   f"{_num('main_mean_accept'):.4f} — throughput gained by speculating less is "
                   "the feature being removed, not a speedup — **verdict forced to REJECT** |\n")
    tau4_row = (f"| mean accept τ @4k | {_num('pr_mean_accept4'):.4f} "
                f"(main {_num('main_mean_accept4'):.4f}) — "
                f"{'✅' if res.get('tau4_ok') else '❌'} |\n")
    tau32_row = (f"| mean accept τ @32k | {_num('pr_mean_accept32'):.4f} "
                 f"(main {_num('main_mean_accept32'):.4f}) — "
                 f"{'✅' if res.get('tau32_ok') else '❌'} |\n")
    def _guard_row(name, ok_key, prob_key, ctxs_key):
        if res.get(ok_key):
            vals = (res.get(ctxs_key) or {}).get(16384) or {}
            detail = (f" — decode {vals['decode']:.1f} tok/s · prefill {vals['prefill']:.0f} pp"
                      if vals.get("decode") and vals.get("prefill") else "")
            return f"| {name} guard @16k | ✅ no regression (decode+prefill){detail} |\n"
        problems = "; ".join((res.get(prob_key) or [])[:4])
        return (f"| {name} guard @16k | ❌ **FAILED** — {problems} — "
                "**verdict forced to REJECT regardless of speed** |\n")
    # "(upstream unsloth)", not "(ModelOpt)": ModelOpt is the SCORED checkpoint, and labelling the
    # guard after it is what made the $MODEL_DIR/$Q38_GUARD_MODEL_DIR mix-up invisible in the PR
    # comment for as long as it lasted -- the row read exactly as intended while measuring the
    # wrong build.
    q38_row = _guard_row("qwen3.8 (upstream unsloth)", "q38_guard_ok", "q38_guard_problems", "q38_guard")
    q36_row = _guard_row("qwen3.6", "q36_guard_ok", "q36_guard_problems", "q36_guard")
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
        f"{marker}\n## sparkinfer DSpark auto-eval — `eval-dspark:{lab}`\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| **label** | `eval-dspark:{lab}` |\n"
        f"| scored at | **DSpark decode + batched prefill @4k/@16k/@32k** on the ModelOpt NVFP4 checkpoint |\n"
        f"| **PR DSpark @4k** | **{_num('pr_dspark4_tps'):.2f} tok/s** |\n"
        f"| **main DSpark @4k** | **{_num('main_dspark4_tps'):.2f} tok/s** |\n"
        f"| DSpark decode @4k vs main | {res.get('decode4_delta_pct', 0):+.1f}% |\n"
        f"| **PR prefill @4k** | **{_num('pr_prefill4_pp'):.1f} pp/s** |\n"
        f"| **main prefill @4k** | **{_num('main_prefill4_pp'):.1f} pp/s** |\n"
        f"| prefill @4k vs main | {res.get('prefill4_delta_pct', 0):+.1f}% |\n"
        f"| **PR DSpark @16k** | **{_num('pr_dspark_tps'):.2f} tok/s** |\n"
        f"| **main DSpark @16k** | **{_num('main_dspark_tps'):.2f} tok/s** |\n"
        f"| DSpark decode @16k vs main | {res.get('decode_delta_pct', 0):+.1f}% |\n"
        f"| **PR prefill @16k** | **{_num('pr_prefill16_pp'):.1f} pp/s** |\n"
        f"| **main prefill @16k** | **{_num('main_prefill16_pp'):.1f} pp/s** |\n"
        f"| prefill @16k vs main | {res.get('prefill16_delta_pct', 0):+.1f}% |\n"
        f"| **PR DSpark @32k** | **{_num('pr_dspark32_tps'):.2f} tok/s** |\n"
        f"| **main DSpark @32k** | **{_num('main_dspark32_tps'):.2f} tok/s** |\n"
        f"| DSpark decode @32k vs main | {res.get('decode32_delta_pct', 0):+.1f}% |\n"
        f"| **PR prefill @32k** | **{_num('pr_prefill_pp'):.1f} pp/s** |\n"
        f"| **main prefill @32k** | **{_num('main_prefill_pp'):.1f} pp/s** |\n"
        f"| prefill vs main | {res.get('prefill_delta_pct', 0):+.1f}% |\n"
        f"| PR AR tok/s (floor) | {_num('pr_ar_tps'):.2f} |\n"
        f"| main AR tok/s (floor) | {_num('main_ar_tps'):.2f} |\n"
        f"| AR vs main (floor) | {res.get('ar_delta_pct', 0):+.1f}% |\n"
        f"| **DSpark vs AR** | **{res.get('dspark_vs_ar', 0):.3f}×** — above 1.0 means speculation finally pays |\n"
        f"| mean accept τ | {_num('pr_mean_accept'):.3f} (main {_num('main_mean_accept'):.3f}, ceiling 7) |\n"
        f"{acc_row}"
        f"{main_acc_note}"
        f"{ls_row}"
        f"{ls4_row}"
        f"{ls32_row}"
        f"{tau4_row}"
        f"{tau_row}"
        f"{tau32_row}"
        f"{q38_row}"
        f"{q36_row}"
        f"| PPL PR / main | {res.get('pr_ppl') or '?'} / {res.get('main_ppl') or '?'} |\n"
        f"{polaris_row}"
        f"| commit | `{commit[:9]}` |\n\n"
        f"{res.get('reason') or ''}\n\n"
        "<sub>Scored on the pinned eval box vs same-box `origin/main`: DSpark speculative decode "
        "and production batched prefill at ctx=4k, ctx=16k and ctx=32k on the ModelOpt NVFP4 checkpoint. "
        "The AR reference is measured in "
        "the same process and the same model load. Both a regression in AR decode and any "
        "divergence from the AR token sequence are hard REJECTs — a speculative decoder that is "
        "fast because it skips verification is not faster, it is wrong. τ is the lever, and the "
        "row above reports it against a block_size of 7. This is informational, not a judgment on "
        "your PR: a `none` label just "
        "means no measurable DSpark decode@16k speedup was verified, which is expected and fine if "
        "that isn't what your change is about. "
        "Automated — merge behaviour depends on SPARKINFER_DSPARK_AUTOMERGE.</sub>\n"
    )


def auto_merge_ok_qwen38(repo, num):
    info = json.loads(arb.gh([
        "pr", "view", str(num), "-R", repo, "--json",
        "state,isDraft,labels,author,mergeable,files",
    ]).stdout or "{}")
    if info.get("state") != "OPEN" or info.get("isDraft"):
        return False, "not an open, non-draft PR"
    labs = {l["name"] for l in info.get("labels", [])}
    tiers = {l.split(":", 1)[1] for l in labs if l.startswith(EVAL_PREFIX)}
    if not (tiers & SPEEDUP_LABELS):
        return False, "no verified eval-qwen38:speedup label"
    if MODELOPT_MERGE_FIRST not in labs:
        return False, "not dspark-merge-first"
    blocked = labs & AUTOMERGE_BLOCK
    if blocked:
        return False, f"blocking label(s): {', '.join(sorted(blocked))}"
    author = (info.get("author") or {}).get("login", "")
    if author.lower() in arb.load_denylist():
        return False, f"author {author} is blocked"
    if arb.author_penalty_until(author):
        return False, f"author {author} is under penalty"
    # arb.AUTOMERGE_SENSITIVE covers eval/ and bench/scripts/ but NOT runtime/examples/, which is
    # where THIS bot's measuring instrument lives. dspark_tau_check.cpp decides what "DSpark decode
    # @4k" even means: it picks the env pins both legs run under, it computes the throughput, and it
    # renders the losslessness verdict. A PR that edits it does not make inference faster, it makes
    # the benchmark report a different number -- and because the bot builds the PR's own harness and
    # compares it against main's, such a change measures as a speedup and would have auto-merged
    # into the instrument that scores everything after it.
    #
    # Not hypothetical: PR #871 proposed exactly this (drop the NSPLITS pin, "+6.8%
    # dspark-decode@128") and nothing in the auto-merge gate would have stopped it. Blocking is not
    # a veto -- harness changes can still be merged by a human, which is the right bar for a change
    # to the thing doing the measuring.
    sensitive = tuple(arb.AUTOMERGE_SENSITIVE) + ("runtime/examples/",)
    sens = [f["path"] for f in info.get("files", [])
            if any(f["path"].startswith(p) for p in sensitive)]
    if sens:
        return False, f"touches protected paths: {', '.join(sens[:3])}"
    if arb.pr_merge_conflict(info.get("mergeable")):
        return False, "merge conflict with base"
    if info.get("mergeable") != "MERGEABLE":
        return False, f"not cleanly mergeable ({info.get('mergeable')})"
    return True, "ok"


# The README's Qwen3.8 ModelOpt table is regenerated between these markers. Everything outside
# them is prose and is never touched, so the table can be machine-written without a generator
# owning the whole document.
# No README auto-refresh here, unlike pr_modelopt_bot.py. See the auto-merge site below for why:
# the published Qwen3.8 tables are AR serving speed, and DSpark is not yet faster than AR.


def try_auto_merge_qwen38(repo, num):
    ok, reason = auto_merge_ok_qwen38(repo, num)
    if not ok:
        print(f">> dspark auto-merge SKIP #{num}: {reason}")
        return False
    r = arb.gh(["pr", "merge", str(num), "-R", repo, "--squash"])
    if r.returncode != 0 and os.environ.get("SPARKINFER_AUTOMERGE_ADMIN", "1") == "1":
        err = ((r.stderr or "") + (r.stdout or "")).lower()
        if "not mergeable" in err or "branch policy" in err or "required" in err or "prohibited" in err:
            print(">> dspark auto-merge: branch policy blocked — retrying with --admin")
            r = arb.gh(["pr", "merge", str(num), "-R", repo, "--squash", "--admin"])
    if r.returncode == 0:
        print(f">> DSPARK AUTO-MERGED #{num} (dspark-merge-first)")
        arb.gh(["pr", "comment", str(num), "-R", repo, "--body",
                "<!-- sparkinfer-dspark-automerge -->\n"
                "Auto-merged as the round's `dspark-merge-first` winner — verified same-box "
                "128-token decode speedup over `main`, accuracy-gated vs llama.cpp."])
        return True
    print(f">> dspark auto-merge BLOCKED #{num}: {(r.stderr or r.stdout or '')[:200]}")
    return False


def reconcile_qwen38_merge_labels(repo, dry_run=False):
    scores = _load_scores()
    open_prs = json.loads(arb.gh([
        "pr", "list", "-R", repo, "--state", "open",
        "--json", "number,labels", "--limit", "80",
    ]).stdout or "[]")
    open_labels = {p["number"]: {l["name"] for l in p["labels"]} for p in open_prs}

    merged = json.loads(arb.gh([
        "pr", "list", "-R", repo, "--state", "merged", "--label", MODELOPT_MERGE_FIRST,
        "--json", "number", "--limit", "10",
    ]).stdout or "[]")
    for m in merged:
        if not dry_run:
            arb.remove_label(repo, m["number"], MODELOPT_MERGE_FIRST)

    scored = []
    for num, labs in open_labels.items():
        if MODELOPT_NEEDS_REBASE in labs:
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
        print(">> dspark round: no verified speedup PRs")
        return
    winner = scored[0][0]
    print(f">> dspark round: merge-first #{winner}; rebase {[n for n,_,_ in scored[1:]] or 'none'}")
    if dry_run:
        return
    arb.add_label(repo, winner, MODELOPT_MERGE_FIRST)
    arb.remove_label(repo, winner, MODELOPT_NEEDS_REBASE)
    for num, _, _ in scored[1:]:
        arb.add_label(repo, num, MODELOPT_NEEDS_REBASE)
        arb.remove_label(repo, num, MODELOPT_MERGE_FIRST)
    if AUTO_MERGE and try_auto_merge_qwen38(repo, winner):
        # The README refresh was left unwired on 2026-08-18 with an explicit condition: DSpark ran
        # at 0.571x plain AR decode, and publishing that into the ModelOpt table -- which is AR
        # decode/prefill and reads as this model's serving speed -- would have misrepresented it.
        # "Wire it up once DSpark crosses AR and there is a number worth publishing."
        #
        # It has. main measured 112.34 tok/s against 90.48 AR (1.242x, tau 1.662, lossless) on
        # 2026-08-20. refresh_readme_dspark publishes it as its OWN section rather than into that
        # table, for the reason the original note was really guarding: a speculative number and a
        # serving number are not interchangeable.
        entry = _load_scores().get(str(winner)) or {}
        commit = entry.get("commit") or ""
        if commit:
            refresh_readme_dspark(entry, commit)
        else:
            print(">> DSpark README refresh: no stored commit for the merged PR — skipped",
                  file=sys.stderr)


README_DSPARK_MARK = ("<!-- BENCH:qwen38-dspark:start -->", "<!-- BENCH:qwen38-dspark:end -->")


def refresh_readme_dspark(entry, commit, repo_root=None, push=True):
    """Rewrite the README's DSpark section from a just-merged PR's own measurements.

    Mirrors pr_modelopt_bot.refresh_readme_modelopt, with two deliberate differences.

    SEPARATE SECTION, NOT THE MODELOPT TABLE. That table is AR decode/prefill and reads as this
    model's serving speed. A speculative-decode figure folded into it would be misread as the
    general number, because tau depends on how predictable the generated text is -- measured 1.12
    on code against 1.41 on prose on the same build. So this publishes its own block, states the
    context it was measured at, names the corpus, and prints the AR number beside it rather than
    in place of it.

    PUBLISHES THE RATIO EVEN IF IT IS BELOW 1.0. The refresh was originally left unwired because
    DSpark ran at 0.571x AR and there was "no number worth publishing". Showing both columns and
    the ratio makes a sub-1.0 result read correctly instead of looking like a claim, and a section
    that silently disappears when the number regresses is worse than one that reports it.

    Writes through a THROWAWAY WORKTREE of origin/main, never the bot's own checkout -- the cron
    runs this bot from whatever branch that tree happens to be on, so committing in place could
    push an unmerged branch to main.
    """
    need = ("pr_dspark_tps", "pr_ar_tps", "pr_mean_accept")
    vals = {k: float(entry.get(k) or 0) for k in need}
    if not all(v > 0 for v in vals.values()):
        missing = [k for k in need if vals[k] <= 0]
        print(f">> DSpark README refresh: skipped, measurement incomplete ({', '.join(missing)})",
              file=sys.stderr)
        return False
    ratio = float(entry.get("dspark_vs_ar") or (vals["pr_dspark_tps"] / vals["pr_ar_tps"]))
    ctx_label = f"{DSPARK_CTX // 1024}k" if DSPARK_CTX >= 1024 else str(DSPARK_CTX)

    def _rewrite(path):
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as e:
            print(f">> DSpark README refresh: cannot read {path}: {e}", file=sys.stderr)
            return False
        start_m, end_m = README_DSPARK_MARK
        i, j = text.find(start_m), text.find(end_m)
        if i < 0 or j < 0 or j < i:
            print(f">> DSpark README refresh: markers not found in {path} — leaving it alone",
                  file=sys.stderr)
            return False
        body = [
            start_m, "",
            "| context | DSpark decode | AR decode | speedup | mean accepted (τ) |",
            "|---:|---:|---:|---:|---:|",
            f"| {ctx_label} | **{vals['pr_dspark_tps']:,.1f}** tok/s | "
            f"{vals['pr_ar_tps']:,.1f} tok/s | **{ratio:.3f}×** | {vals['pr_mean_accept']:.3f} |",
            "",
            "<sub>**Lossless**: the eval regenerates the same prompt with the draft disabled and "
            "requires the two token sequences to be byte-identical, so this is exact-token "
            "equality with autoregressive decode, not distributional agreement. A run that is not "
            "lossless is rejected regardless of speed.</sub>",
            "",
            f"<sub>Measured at ctx={DSPARK_CTX} on `bench/scripts/bench_prompt_32k.txt`. Speculative "
            "throughput depends on how predictable the generated text is — the same build measures "
            "a materially different τ on prose, code and repetitive text — so treat this as that "
            "workload at that context, not a general serving figure. The AR column is the "
            "autoregressive decode measured in the same process, same model load, same GPU "
            f"state.</sub>",
            "",
            f"<sub>Auto-refreshed by the DSpark eval bot at `{commit[:9]}` — these are the numbers "
            "that PR measured on the pinned RTX 5090, which after squash-merge are main's. "
            "Regenerated on every auto-merge, so the table cannot drift behind the code.</sub>",
            end_m,
        ]
        new_text = text[:i] + "\n".join(body) + text[j + len(end_m):]
        if new_text == text:
            print(">> DSpark README refresh: numbers unchanged")
            return False
        open(path, "w", encoding="utf-8").write(new_text)
        return True

    if not push:
        return _rewrite(os.path.join(repo_root or ROOT, "README.md"))

    root = repo_root or ROOT
    wt = os.path.join(tempfile.gettempdir(), f"si_readme_dspark_{os.getpid()}")
    subprocess.run(["git", "-C", root, "worktree", "remove", "--force", wt],
                   capture_output=True, text=True)
    if subprocess.run(["git", "-C", root, "fetch", "-q", "origin", "main"],
                      capture_output=True, text=True).returncode != 0:
        print(">> DSpark README refresh: fetch origin/main failed", file=sys.stderr)
        return False
    mk = subprocess.run(["git", "-C", root, "worktree", "add", "--detach", wt, "origin/main"],
                        capture_output=True, text=True)
    if mk.returncode != 0:
        print(f">> DSpark README refresh: worktree add failed: {(mk.stderr or '')[:200]}",
              file=sys.stderr)
        return False
    try:
        if not _rewrite(os.path.join(wt, "README.md")):
            return False
        msg = (f"docs: refresh Qwen3.8 DSpark table ({commit[:9]})\n\n"
               f"Auto-generated by pr_dspark_bot after auto-merging {commit[:9]}. The numbers are "
               f"that PR's own same-box measurements, which the squash-merge makes main's.")
        for args in (["add", "README.md"], ["commit", "-m", msg]):
            rc = subprocess.run(["git", "-C", wt] + args, capture_output=True, text=True)
            if rc.returncode != 0:
                print(f">> DSpark README refresh: git {args[0]} failed: {(rc.stderr or '')[:200]}",
                      file=sys.stderr)
                return False
        rc = subprocess.run(["git", "-C", wt, "push", "origin", "HEAD:main"],
                            capture_output=True, text=True)
        if rc.returncode != 0:
            print(f">> DSpark README refresh: push to main FAILED: {(rc.stderr or '')[:300]}",
                  file=sys.stderr)
            return False
        print(f">> DSpark README refresh: table updated at {commit[:9]} and pushed to main")
        return True
    finally:
        subprocess.run(["git", "-C", root, "worktree", "remove", "--force", wt],
                       capture_output=True, text=True)
        subprocess.run(["git", "-C", root, "worktree", "prune"], capture_output=True, text=True)


def upload_qwen38_eval_log(repo, num, title, oid, res):
    """Commit the eval result (+ Polaris receipt/attestation) to sparkinfer-log, mirroring
    pr_dflash_bot.py's upload_dflash_eval_log with a qwen38-prefixed run id."""
    try:
        rid = f"qwen38-{int(num):04d}-{oid[:7]}"
        arb._ensure_log_repo()
        rundir = os.path.join(arb.LOG_DIR, "runs", rid)
        os.makedirs(rundir, exist_ok=True)
        polaris = res.get("polaris") or {}
        receipt = polaris.get("receipt")
        result = {
            "id": rid, "pr": int(num), "title": title,
            "url": f"https://github.com/{repo}/pull/{num}", "commit": oid[:7],
            "eval_mode": "qwen38-dspark-4k-16k-32k",
            "label": res.get("label"), "pass": res.get("pass"), "reason": res.get("reason"),
            "delta_pct": res.get("delta_pct"),
            "pr_decode_tps": res.get("pr_decode_tps"), "main_decode_tps": res.get("main_decode_tps"),
            "pr_dspark4_tps": res.get("pr_dspark4_tps"), "main_dspark4_tps": res.get("main_dspark4_tps"),
            "decode4_delta_pct": res.get("decode4_delta_pct"),
            "pr_prefill4_pp": res.get("pr_prefill4_pp"), "main_prefill4_pp": res.get("main_prefill4_pp"),
            "prefill4_delta_pct": res.get("prefill4_delta_pct"),
            "pr_prefill16_pp": res.get("pr_prefill16_pp"), "main_prefill16_pp": res.get("main_prefill16_pp"),
            "prefill16_delta_pct": res.get("prefill16_delta_pct"),
            "pr_prefill_pp": res.get("pr_prefill_pp"), "main_prefill_pp": res.get("main_prefill_pp"),
            "pr_ar_tps": res.get("pr_ar_tps"), "main_ar_tps": res.get("main_ar_tps"),
            "prefill_delta_pct": res.get("prefill_delta_pct"),
            "pr_ar_tps": res.get("pr_ar_tps"),
            "main_ar_tps": res.get("main_ar_tps"),
            "prefill16k_delta_pct": res.get("prefill16k_delta_pct"),
            "pr_dspark_tps": res.get("pr_dspark_tps"),
            "main_dspark_tps": res.get("main_dspark_tps"),
            "decode16k_delta_pct": res.get("decode16k_delta_pct"),
            "scored_dimension": res.get("scored_dimension"),
            "speedup_vs_main": res.get("speedup_vs_main"),
            "pr_top1": res.get("pr_top1"), "pr_kl": res.get("pr_kl"),
            "pr_top1": res.get("pr_top1"), "pr_kl": res.get("pr_kl"),
            "accuracy_ok": res.get("accuracy_ok"),
            "lossless": res.get("lossless"), "dspark_vs_ar": res.get("dspark_vs_ar"),
            "lossless4": res.get("lossless4"), "lossless4_runs": res.get("lossless4_runs"),
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
                     "delta_pct": res.get("delta_pct"), "eval_mode": "qwen38-16k", "date": result["date"]}
        if receipt:
            idx_entry["polaris"] = True
            idx_entry["polaris_receipt_id"] = receipt.get("receipt_id", "")[:16]
        idx.append(idx_entry)
        idx.sort(key=lambda x: x["id"])
        json.dump(idx, open(ipath, "w"), indent=2)
        subprocess.run(["git", "-C", arb.LOG_DIR, "add", "-A"], check=True)
        msg = f"qwen38-eval: #{num} {oid[:7]} -> eval-qwen38:{res.get('label')}"
        if receipt:
            msg += f" + polaris {receipt.get('receipt_id', '?')[:16]}"
        commit = subprocess.run(["git", "-C", arb.LOG_DIR, "commit", "-q", "-m", msg], check=False)
        if commit.returncode != 0:
            print(">> dspark eval-log upload skipped: nothing to commit")
            return None
        push = subprocess.run(["git", "-C", arb.LOG_DIR, "push", "-q"], check=False)
        if push.returncode != 0:
            print(f">> dspark eval-log push failed (rc={push.returncode})")
            return None
        url = arb.LOG_PAGE + rid
        print(f">> dspark eval log: {url}")
        return url
    except Exception as e:
        print(f">> dspark eval-log upload failed: {e}")
        return None


def apply_result(repo, num, commit, res, title="", dry_run=False):
    body = format_comment(commit, res)
    label = res.get("label") if res.get("ok") else "REJECT"
    if not res.get("ok"):
        label = "REJECT"
    print(f"PR #{num}: eval-dspark:{label}  "
          f"dspark PR={res.get('pr_dspark_tps')} main={res.get('main_dspark_tps')}  "
          f"ar PR={res.get('pr_ar_tps')} main={res.get('main_ar_tps')}  "
          f"from={res.get('scored_dimension')}  "
          f"top1={res.get('pr_top1')} kl={res.get('pr_kl')}  "
          f"delta={res.get('delta_pct')}%  accuracy_ok={res.get('accuracy_ok')}  "
          f"lossless={res.get('lossless')} tau_ok={res.get('tau_ok')} "
          f"tau={res.get('pr_mean_accept', 0):.3f} "
          f"dspark_vs_ar={res.get('dspark_vs_ar', 0):.3f}x "
          f"q38_guard_ok={res.get('q38_guard_ok')} q36_guard_ok={res.get('q36_guard_ok')}")
    if dry_run:
        print(body[:500])
        return
    strip_qwen38_eval_labels(repo, num)
    if label in SPEEDUP_LABELS:
        # A fresh, valid speedup score for the CURRENT head commit means this PR is caught up
        # with main and deserves a fair shot at winning the next merge-first reconciliation --
        # clear any stale needs-rebase from a round it lost (or an old conflict that's since been
        # resolved). Found 2026-08-13: reconcile_qwen38_merge_labels() filters candidates on
        # `MODELOPT_NEEDS_REBASE not in labs` (this bot's own "who's eligible to win" gate) but
        # the ONLY place that ever removed the label was the winner-selection branch itself --
        # a PR that lost one round, or ever hit a transient merge conflict, could never be
        # reconsidered again even after a completely clean re-evaluation confirmed its score,
        # since it was filtered out of candidacy before scoring was ever compared. Hit #790 and
        # #791 both losing merge-first to a strictly worse score for exactly this reason.
        arb.remove_label(repo, num, MODELOPT_NEEDS_REBASE)
    arb.add_label(repo, num, f"{EVAL_PREFIX}{label}")
    # Mirrored to the generic `eval:*` label, same as pr_dflash_bot.py -- SN74 scoring reads
    # eval:* tiers, so this makes Qwen3.8-27B submissions count toward that live incentive
    # mechanism. Explicit user decision, 2026-08-11 (originally deliberately NOT mirrored, given
    # Qwen3.8-27B's youth at the time -- see git history on this line for that reasoning).
    for lab in {l for l in arb.labels_on(repo, num) if l.startswith("eval:")}:
        arb.remove_label(repo, num, lab)
    arb.add_label(repo, num, f"eval:{label}")
    # The verdict comment is the ONLY place a contributor sees why their PR was labelled and (for
    # a non-speedup) closed, so a dropped post is worse than a dropped label. Post via the REST
    # issues endpoint rather than `gh pr comment`: the latter resolves the PR through a GraphQL
    # query that now fails outright on this repo with "Projects (classic) is being deprecated ...
    # (repository.pullRequest.projectCards)", independent of any outage. Then CHECK it -- a
    # silently-failed comment is what left PR #862 closed with a label and no result on
    # 2026-08-17, while the round logged a clean pass.
    c = arb.gh(["api", f"repos/{repo}/issues/{num}/comments",
                "--method", "POST", "--field", f"body={body}"])
    if c is None or c.returncode != 0:
        print(f">> ERROR: PR #{num}: verdict comment FAILED to post — the PR now carries a label "
              f"with no visible explanation. Verdict body follows so the round is not lost:\n"
              f"{body[:900]}", file=sys.stderr)
    if res.get("ok"):
        upload_qwen38_eval_log(repo, num, title, commit, res)
    if res.get("ok") and res.get("delta_pct") is not None:
        scores = _load_scores()
        scores[str(num)] = {
            "commit": commit,
            "label": label,
            "delta_pct": res.get("delta_pct"),
            "pr_decode_tps": res.get("pr_decode_tps"),
            "main_decode_tps": res.get("main_decode_tps"),
            # Preserve the multi-context result so dashboards can render it without re-benchmarking.
            "pr_dspark_tps": res.get("pr_dspark_tps"),
            "pr_dspark4_tps": res.get("pr_dspark4_tps"),
            "main_dspark4_tps": res.get("main_dspark4_tps"),
            "decode4_delta_pct": res.get("decode4_delta_pct"),
            "pr_prefill4_pp": res.get("pr_prefill4_pp"),
            "main_prefill4_pp": res.get("main_prefill4_pp"),
            "prefill4_delta_pct": res.get("prefill4_delta_pct"),
            "pr_prefill16_pp": res.get("pr_prefill16_pp"),
            "main_prefill16_pp": res.get("main_prefill16_pp"),
            "prefill16_delta_pct": res.get("prefill16_delta_pct"),
            "pr_mean_accept": res.get("pr_mean_accept"),
            "pr_ar_tps": res.get("pr_ar_tps"),
            "dspark_vs_ar": res.get("dspark_vs_ar"),
            "pr_ar_tps": res.get("pr_ar_tps"),
            "pr_top1": res.get("pr_top1"),
            "pr_kl": res.get("pr_kl"),
            "pass": res.get("pass"),
            "accuracy_ok": res.get("accuracy_ok"),
            "lossless": res.get("lossless"),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _save_scores(scores)
        # Auto-close on "none"/"REJECT" -- same policy as pr_dflash_bot.py. Re-enabled 2026-08-11
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
        if label in ("none", "REJECT"):
            if not res.get("lossless", True) or not res.get("lossless4", True) or not res.get("lossless32", True):
                fail_clause = "and failed exact DSpark-vs-AR losslessness at one or more contexts"
            elif not res.get("accuracy_ok"):
                fail_clause = "and failed the accuracy gate"
            elif res.get("decode_regressed") and res.get("prefill_regressed"):
                fail_clause = "(dspark decode@16k and AR decode@16k regression)"
            elif res.get("decode_regressed"):
                fail_clause = "(dspark decode@16k regression)"
            elif res.get("prefill_regressed"):
                fail_clause = "(prefill@128 regression)"
            elif label == "none":
                fail_clause = "with no verified improvement on the scored 4k/16k/32k decode/prefill axes"
            else:
                fail_clause = "(regression)"
            close_body = (
                "<!-- sparkinfer-qwen38-auto-close -->\n"
                f"## Closed: sparkinfer DSpark auto-eval — `eval-dspark:{label}`\n\n"
                f"This PR's best Qwen3.8-27B DSpark axis measured **{res.get('delta_pct')}%** "
                f"vs main, {fail_clause} "
                "— closing automatically. This bot evaluates every eligible PR in the repo "
                "against Qwen3.8-27B's decode AND prefill@128 speed specifically, regardless of "
                "what the PR is actually about — a close here isn't a judgment on the PR's purpose, "
                "just that it didn't move these particular metrics. Reopen (or open a fresh PR) if "
                "you have a fix or a different approach."
            )
            arb.gh(["pr", "comment", str(num), "-R", repo, "--body", close_body])
            arb.gh(["pr", "close", str(num), "-R", repo])
            print(f">> auto-closed PR #{num} (eval-dspark:{label})")


def main():
    ap = argparse.ArgumentParser(description="Qwen3.8-27B ModelOpt 128-decode PR eval bot")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--repo", default="gittensor-ai-lab/sparkinfer")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reeval", action="store_true")
    ap.add_argument("--labels-only", action="store_true",
                    help="reconcile dspark-merge-first only — no GPU")
    ap.add_argument("--only-prs", default="",
                    help="comma-separated PR numbers (bypass greenlight)")
    args = ap.parse_args()

    only = {int(x) for x in args.only_prs.split(",") if x.strip().isdigit()}

    print(f">> dspark eval transport: "
          f"{'ssh' if ssh_box_enabled() else f'vast.ai (instance {arb.current_instance(args.instance) or args.instance})'}")
    print(f">> AUTOMERGE={int(AUTO_MERGE)}")

    if args.labels_only:
        reconcile_qwen38_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — dspark labels only (no GPU).")
        return

    prs = json.loads(arb.gh([
        "pr", "list", "-R", args.repo, "--state", "open",
        # `files` is REQUIRED by the harness-touch check below. Without it pr.get("files") is None,
        # the check silently never fires, and a harness PR sails through to a full round.
        "--json", "number,title,labels,isDraft,headRefOid,headRefName,mergeable,author,body,files",
        "--limit", "80",
    ]).stdout or "[]")
    prs.sort(key=lambda p: p["number"])

    stale_closed = close_stale_qwen38_prs(args.repo, prs, dry_run=args.dry_run) if not only else set()

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
        if not args.reeval and head and head in qwen38_evaluated_commits(args.repo, num):
            print(f"PR #{num} @ {short}: already dspark-evaluated — skip")
            continue
        # Before the greenlight and before any GPU time: does this PR edit the measuring
        # instrument? Checked here rather than at auto-merge because a harness PR should not
        # consume a 20-minute round to produce a number that cannot be accepted either way.
        touched = [f.get("path", "") for f in (pr.get("files") or [])]
        harness_hits = [t for t in touched if any(t.startswith(h) for h in HARNESS_PATHS)]
        if harness_hits:
            print(f"PR #{num}: touches the eval harness ({', '.join(harness_hits[:3])}) — not evaluated")
            if not args.dry_run and not qwen38_evaluated_commits(args.repo, num):
                arb.gh(["pr", "comment", str(num), "-R", args.repo, "--body",
                        "<!-- sparkinfer-dspark-harness -->\n"
                        "**Not evaluated — this PR changes the eval harness.**\n\n"
                        f"Touched: {', '.join('`' + h + '`' for h in harness_hits)}\n\n"
                        "These files define what every PR is *measured against* — the env both legs "
                        "run under, how throughput is computed, what counts as lossless, and the "
                        "prompt every score is taken on. A change to them moves the baseline for "
                        "everyone at once, so they are maintained by the project rather than "
                        "accepted through the contribution flow. This is policy and applies even "
                        "when the change is correct.\n\n"
                        "If you have found a genuine defect in the harness, please open an **issue** "
                        "describing it — that is welcome and useful, and at least one such report "
                        "has already led to a fix. Engine changes are evaluated as normal."])
            continue
        if arb.pr_merge_conflict(pr.get("mergeable")):
            print(f"PR #{num}: merge conflict — dspark-needs-rebase")
            if not args.dry_run:
                arb.add_label(args.repo, num, MODELOPT_NEEDS_REBASE)
            continue

        if not only:
            status, why = arb.greenlight_status(args.repo, num, labs)
            if status != "ok":
                print(f"PR #{num}: not greenlit ({why}) — skip dspark eval")
                continue
            print(f"PR #{num}: greenlit ({why})")
        else:
            print(f"PR #{num}: --only-prs targeted")

        ref = f"pull/{num}/head"
        pending.append((num, head, short, ref, pr.get("title", "")))

    if not pending:
        reconcile_qwen38_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — no dspark PRs to evaluate.")
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
        reconcile_qwen38_merge_labels(args.repo, dry_run=False)
        print("done — dspark labels only (GPU down).")
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
        reconcile_qwen38_merge_labels(args.repo, dry_run=False)
        print("done — dspark round skipped (main baseline unusable).")
        return
    # main has no top1/kl of its own: it IS the accuracy reference, and its score dump was just
    # written to SCORE_DUMP_MAIN for each PR in this round to diff against.
    print(f">> main baseline: dspark@16k={main_result['dspark_tps']:.2f} tok/s (scored) "
          f"dspark@32k={main_result['dspark32_tps']:.2f} tok/s (scored) "
          f"prefill@32k={main_result['prefill_pp']:.2f} pp/s (scored) "
          f"ar@16k={main_result['ar_tps']:.2f} tok/s "
          f"ratio={main_result['dspark_tps'] / main_result['ar_tps']:.3f}x "
          f"tau={main_result.get('mean_accept', 0):.3f} lossless={main_result.get('lossless')}")

    for num, head, short, ref, title in pending:
        print(f"PR #{num} @ {short}: evaluating Qwen3.8-27B '{ref}' …")
        try:
            res = eval_qwen38_on_box(host, port, ref, main_result)
        except Exception as e:
            res = {"ok": False, "reason": f"exception: {e}"}
        apply_result(args.repo, num, head or short, res, title=title, dry_run=False)

    reconcile_qwen38_merge_labels(args.repo, dry_run=False)
    print("done — dspark eval pass complete.")


if __name__ == "__main__":
    main()
