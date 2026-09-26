#!/usr/bin/env python3
"""sparkinfer Qwen3.8-27B PR auto-evaluator.

Sibling of pr_museglimmer_bot.py / pr_dflash_bot.py, and the ONLY scored bot once Qwen3.8-27B
becomes the eval scope (see eval/README.md). Narrowly scoped on purpose:

  1. Speed — PR vs a freshly-measured origin/main, same box, same round. The tier is the best of
              prefill@16k, concurrent decode @c2..c32 and ModelOpt decode@256k (SCORING_DIMS);
              decode@128, prefill@128, prefill@256k and concurrent decode @c1 are measured as
              no-regression floors only (the history below explains how the scored set grew).
              Same tier buckets as every other bot in this directory (BUCKETS/SIG/REGRESS_TOL
              below — copied, not reinvented).

              The scored checkpoint is the HuggingFace compressed-tensors DIRECTORY (mixed NVFP4
              FFN + FP8 attention/GDN projections, unsloth/Qwen3.8-27B-NVFP4), NOT a GGUF of the
              same model. That is deliberate: it is what sparkinfer-server actually serves, so it
              is what a PR's speed claim should be measured against. qwen3_gguf_bench and
              qwen3_gguf_score grew directory support for exactly this
              (runtime/examples/qwen_checkpoint.h, shared with the server so the two cannot
              disagree about how a checkpoint is configured).

              Prefill@128 was ALSO scored (a floor now), from the same sweep and the same model load (no extra
              GPU time -- prefill_pp was already computed alongside decode_tps and simply
              discarded). Combination rule is pr_museglimmer_bot.py's, verbatim: either dimension
              regressing is a hard REJECT, otherwise the better of the two tiers wins, so a pure
              prefill win with flat decode still scores.

              Read the ABSOLUTE prefill number with care. Measured 2026-08-15 on the pinned box,
              Qwen3.8-27B prefill@128 is ~86 pp -- roughly this model's DECODE speed (82.7 tok/s),
              which is the signature of a token-at-a-time path rather than one batched GEMM pass.
              llama.cpp ingests the same 128-token prompt at ~2651 t/s from a Q4_K_M GGUF of this
              model. That gap is ~31x and is not explained by quantization. Scoring the dimension
              does not fix it; it makes it visible and rewards whoever does.
              What scoring it IS valid for regardless: it is a PR-vs-main comparison on the same
              box and the same shape, so a PR that speeds up or regresses whatever path ctx=128
              actually takes is measured correctly even while the absolute number is unflattering.

  1b. Concurrent decode — cb-decode@c2/c4/c8/c16/c32 scored, cb-decode@c1 a floor (issue #1080).
  1c. Long-context decode — ModelOpt NVFP4 decode@256k scored, prefill@256k a floor (issue #1113).
      One 262144-token row, its own rep tier, on the checkpoint the axis was defined on: the
      unsloth weights this bot otherwise scores do not leave room for a 256K KV cache.
              Aggregate tok/s with N requests in flight through ContinuousBatchEngine
              (qwen3_gguf_cb_bench: N 256-token prompts, 256 tokens each, and one 512-token
              prefill injected mid-batch), measured exactly as pr_dspark_bot.py measures its
              ModelOpt rows. Every other dimension here drives ONE request, and this checkpoint's
              packed step takes FP8 and Q4_K paths the ModelOpt checkpoint never reaches, so a
              change to those paths moved nothing any bot measured. See CB_CONCS.

  2. Accuracy gate — DIFFERENTIAL, not absolute. llama.cpp cannot read a compressed-tensors
              directory, so the Muse Glimmer methodology (teacher-forced score vs a live
              llama-server on the SAME weights) is impossible for this checkpoint; comparing
              against a GGUF of the same model would compare two different quantizations and
              could never hold a tight bar. Instead: score the same token stream on the PR build
              and on origin/main, and require the two distributions to agree
              (bench/scripts/accuracy_compare_pair.py, top1 >= 0.99 / KL <= 0.01 — tight, because
              two builds of the same model on the same box should agree almost exactly).

              A PR that fails this bar is REJECTed regardless of speed. Qwen3.8-27B bring-up
              surfaced six separate silent correctness bugs (wrong weight-layout transpose,
              silu-vs-sigmoid gate, NVFP4 global-scale direction, GDN A_log transform, the
              1+weight norm convention, GDN v-head broadcast) — every one of which left
              throughput completely untouched. Speed alone cannot see that class of bug.

              Limitation, stated plainly: a differential gate cannot catch a bug already present
              on main. It catches newly introduced divergence only.

  3. Qwen3.6 no-regression guard — decode + prefill at ctx 0/512/4k/16k/32k, same box, same PR
              build, vs a freshly-measured origin/main, REGRESS_TOL=0.98. Reuses the sibling bots'
              GUARD36 sweep mechanism verbatim. A regression here is a hard REJECT regardless of
              Qwen3.8's own speed/accuracy result: Qwen3.8 and Qwen3.6 share qwen35.cpp /
              inference_engine.cpp, and that shared surface is exactly how PR #775 regressed a
              model nobody was scoring at the time.

  3b. ModelOpt Qwen3.8 and Muse Glimmer no-regression guards — decode + prefill at 32k and
              concurrent decode at c16/c32 on each, with the same hard REJECT for a regression. A
              concurrent width main measured that the PR build could not complete (a crash, a hang,
              runs cut short) is judged over rounds instead ("guard-cb"), as a scored width is; a
              32k guard the PR build failed to measure still REJECTs at once. A guard run the OOM
              killer took (every attempt, for a concurrent width) is the box's ("guard-box"),
              retried the same way. The 32k guards are
              the pair pr_museglimmer_bot.py runs; the concurrency guards cover the packed decode
              this bot's PRs mostly change, which no single-stream guard enters. The Muse bot skips
              PRs declared for Qwen3.8 alone (since #1082), so this bot is the only check those PRs
              get against either model. A checkpoint missing from the box is skipped and reported,
              never rejected.

  3c. Ternary-Bonsai-2-27B no-regression guard (2026-09-24) — decode + prefill at 128 and 32k on the
              PTQ1_0 GGUF, default loader, same hard REJECT. pr_bonsai_bot.py skips PRs declared
              for Qwen3.8 alone, so this is the only check they get against that model. 128 as
              well as 32k because the dense-GGUF prefill work on that model lives at short
              prompts (#1139: 1.94x at 128, flat at 4k); both come from one model load.

Applies `eval-qwen38:<TIER>` AND mirrors it to the generic `eval:<TIER>` label (SN74 scoring reads
eval:* tiers). Auto-close is live: a REJECT closes; a `none` closes only a PR declared for
Qwen3.8 alone that no other bot scored a speedup or made merge-first (arb.none_may_close,
2026-09-26 -- it used to close on every `none`, including PRs ticked "Shared"). Auto-merge follows
SPARKINFER_QWEN38_AUTOMERGE (=1 in .env.eval).

  python eval/pr_qwen38_bot.py --instance 46074104
  python eval/pr_qwen38_bot.py --only-prs 636 --reeval

Never rents a GPU. Shares the pinned box with any other bot via flock in the cron wrapper
(run_qwen38_cron.sh) — all bots MUST share /tmp/sparkinfer_bot.lock.
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

# Dimensions that can earn a tier: prefill@16k, and concurrent decode at 2 to 32 requests.
# decode@128, prefill@128 and cb-decode@c1 are measured too and act as no-regression floors (see
# eval_qwen38_on_box), so a PR that only improves a floor scores "none".
#
# prefill@16k has been scored since 2026-08-15. The concurrent-decode axes were added 2026-09-15 for
# issue #1080. This checkpoint's packed continuous-batch step does not take the ModelOpt paths:
#
#   Gated-DeltaNet qkv / z / out   FP8 W8A8 -> launch_gemv_fp8*      (ModelOpt: NVFP4 GEMM)
#   attention q / k / v / o         Q4_K, requantized at load -> MMVQ (ModelOpt: NVFP4)
#   MLP layers 56-63                Q4_K, requantized at load         (ModelOpt: NVFP4)
#
# pr_dspark_bot.py scores concurrency only on ModelOpt, and every other dimension here drives ONE
# request, so a change to those paths moved nothing any bot measured.
SCORING_DIM = "prefill@16k"
# Measured exactly like pr_dspark_bot.py's cb-decode rows -- same binary, same 256/256/512 shape,
# same env -- so the two checkpoints' numbers are comparable. c=1 is measured as a FLOOR, never
# scored: it is what stops a PR buying concurrency scaling by slowing the single-stream path.
#
# 256 tokens per request, not fewer. pr_dspark_bot.py measured why: at 64 tokens a run is under two
# seconds and partly measures its own startup, and identical code landed at -3.4% at c=4 -- inside
# the -2% reject band of an axis that is also a floor.
#
# Measured on main 507017b, aggregate tok/s, two runs each (c16 four, c32 ten):
#
#     c1 83.0 / 82.7   c2 155.9 / 155.3   c4 259.0 / 257.9   c8 421.9 / 420.3
#     c16 312.8-315.2   c32 247.9-252.6 (nine of ten runs)
#
# Throughput FALLS past c8, and at c32 two of the 33 requests fail to open (device out of memory)
# on every run -- the gap #1080 describes. Within one session identical code spreads 0.4% at c1-c8,
# 0.8% at c16 and 1.9% at c32, but a later round read c32 at 245.9: single runs of identical code
# at c32 can differ by 2.7%, outside the -2% reject band. Each width is therefore the median of
# CB_REPS complete runs, which costs about three extra minutes per ref.
#
# The tenth c32 run read 293.0: its requests stopped part-way (4,493 tokens instead of 7,936)
# without logging an error, so the aggregate was computed over a shorter wall time. A run like
# that would score an unchanged PR +18% or -15%. The ladder therefore accepts a run only when every
# request either completed or failed outright (cb_complete in _remote_script) and re-runs the width
# otherwise.
CB_CONCS = [1, 2, 4, 8, 16, 32]
CB_SCORED_CONCS = [2, 4, 8, 16, 32]
CB_TOKENS = 256
# Tokens the one injected 512-token prefill request generates. Fixed in qwen3_gguf_cb_bench.cpp
# (run_stream(long_prompt, 8)), which is harness and taken from main.
CB_LONG_TOKENS = 8
CB_REPS = 3
# Runs allowed per width before the round fails as infra: CB_REPS complete ones plus two partial.
CB_MAX_ATTEMPTS = 5
CB_DIM_FOR = {c: f"cb-decode@c{c}" for c in CB_CONCS}
# Long-context decode, its own sweep tier (#1113). bench_sweep_run applies ONE rep count to a whole
# call -- the max across its (ctx, reps) pairs -- so 256k cannot share the 5-rep guard call without
# costing five 60 s rows. It is measured on the ModelOpt NVFP4 checkpoint because that is the
# checkpoint the axis was defined on (pr_dspark_bot.py's target-decode@256k) and the only one that
# fits: 30.0 GB peak, against 22 GB of unsloth weights before any KV.
LONGCTX_CTX = 262144
LONGCTX_REPS = 3
LONGCTX_DECODE_DIM = "modelopt-decode@256k"
LONGCTX_PREFILL_DIM = "modelopt-prefill@256k"
SCORING_DIMS = [SCORING_DIM] + [CB_DIM_FOR[c] for c in CB_SCORED_CONCS] + [LONGCTX_DECODE_DIM]

# The measuring instrument. A PR touching any of these is not evaluated, and every ref -- main
# included -- is built with main's copy of them (see the HARNESS_PINNED block in _remote_script).
# Same policy and reasoning as pr_dspark_bot.py's HARNESS_PATHS: changing them moves the baseline
# for every contributor at once, and without the pin a PR branched before a harness change is
# measured with a different ruler than main (#878 was auto-closed for exactly that).
HARNESS_PATHS = (
    "runtime/examples/qwen3_gguf_bench.cpp",
    "runtime/examples/qwen3_gguf_cb_bench.cpp",
    # The accuracy gate's only input is this tool's dump: a PR editing it could print main's dump
    # (a public path) and pass any numerics. The Muse and Bonsai bots already treat it as harness.
    "runtime/examples/qwen3_gguf_score.cpp",
    "runtime/examples/qwen_checkpoint.h",
    "runtime/examples/qwen3_gguf_config.h",
    "eval/",
    "bench/scripts/",
)

# Accuracy gate bars. This gate is DIFFERENTIAL (PR vs origin/main on the same token stream, see
# the module docstring pt. 2), not absolute-vs-llama.cpp, so the bars are much tighter than the
# 0.90/0.10 an across-engine comparison needs: two builds of the same model on the same box
# should agree essentially exactly. Anything less means the PR changed the model's numerics.
# Not 1.0/0.0 — a PR may legitimately reassociate float ops (fusing a kernel, changing a
# reduction order), which perturbs the last bits without being a correctness bug.
ACC_TOP1_BAR = float(os.environ.get("QWEN38_ACC_TOP1_BAR", "0.99"))
ACC_KL_BAR = float(os.environ.get("QWEN38_ACC_KL_BAR", "0.01"))

EVAL_PREFIX = "eval-qwen38:"
QWEN38_MERGE_FIRST = "qwen38-merge-first"
QWEN38_NEEDS_REBASE = "qwen38-needs-rebase"
# First schema for this bot. Same reasoning as the sibling bots' own bumps: a PR evaluated before
# a scoring change existed must not keep a stale-scored label/score forever.
# v2 (2026-09-15): concurrent-decode axes added (issue #1080), and the harness is taken from main.
# v3 (2026-09-15): ModelOpt Qwen3.8 and Muse Glimmer no-regression guards added.
# v4 (2026-09-15): those guards also cover concurrent decode at c16/c32.
# v6 (2026-09-24): Ternary-Bonsai-2-27B no-regression guard added (pt. 3c).
EVAL_SCHEMA_VERSION = "v6-unsloth-concurrency-cross-model-guards-cb-longctx256k-bonsai"
MARKER_RE = re.compile(
    r"<!-- sparkinfer-qwen38-eval:" + re.escape(EVAL_SCHEMA_VERSION) + r":([0-9a-f]+)(?:\s+(\{.*?\}))? -->",
    re.DOTALL,
)

# --- box paths (see .env.eval's QWEN38_* block) ---
# Separate clone from pr_eval_bot's /root/sparkinfer, which carries unrelated uncommitted work.
REMOTE_REPO = os.environ.get("QWEN38_REMOTE_REPO", "/root/sparkinfer_qwen38")
# The scored checkpoint is a HuggingFace compressed-tensors DIRECTORY (mixed NVFP4 FFN + FP8
# attention/GDN projections, unsloth/Qwen3.8-27B-NVFP4) -- NOT a GGUF. That is what the server
# actually serves, so it is what gets benchmarked; qwen3_gguf_bench/qwen3_gguf_score grew
# directory support for exactly this (runtime/examples/qwen_checkpoint.h).
MODEL_DIR = os.environ.get("QWEN38_MODEL_DIR", "/root/workspace/models_qwen38")
# The single weight blob inside MODEL_DIR, for the Polaris attestation ONLY. The sibling bots pass
# their .gguf here; the analogue for a compressed-tensors checkout is the safetensors file, not the
# directory -- receipt.model_sha256() returns "" for anything that is not a regular file, so
# passing MODEL_DIR would mint a receipt whose model_sha256 pins nothing while still looking valid.
# If a future checkpoint is sharded (model-00001-of-0000N.safetensors) this path stops existing and
# the sha degrades to "" rather than crashing; override QWEN38_MODEL_WEIGHT_FILE if that happens.
MODEL_WEIGHT_FILE = os.environ.get("QWEN38_MODEL_WEIGHT_FILE",
                                   os.path.join(MODEL_DIR, "model.safetensors"))
BENCH_TOKENS = int(os.environ.get("QWEN38_BENCH_TOKENS", "128"))
ACC_TOPK = int(os.environ.get("QWEN38_ACC_TOPK", "128"))
# INACTIVE (2026-09-15). The batched-prefill parity gate this constant belonged to is not run; see
# the "batched-prefill parity: NOT RUN" block in _remote_script. Kept so re-enabling is one block.
PARITY_BAR = float(os.environ.get("QWEN38_PARITY_BAR", "0.75"))
# Score dumps for the differential accuracy gate. main's is written once per round by
# measure_main_baseline(); each PR compares its own dump against it. Kept on the box (not shipped
# back over ssh) because a 128-deep top-k dump over the whole corpus is megabytes.
SCORE_DUMP_MAIN = "/tmp/q38_score_main.txt"
SCORE_DUMP_PR = "/tmp/q38_score_pr.txt"
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

# Cross-model no-regression guards beside Qwen3.6, decode AND prefill at 32k (2026-09-15): the same
# two checkpoints and the same shape pr_museglimmer_bot.py guards. This bot scores packed-decode and
# prefill work on code those models share, and the Muse bot skips PRs declared for Qwen3.8 alone
# (#1082), so without these nothing checks such a PR against either model -- #1081, #1082 and #1083
# all auto-merged that way.
#
#   ModelOpt Qwen3.8-27B NVFP4   the checkpoint the release container serves and the DSpark bot scores
#   Muse Glimmer 30B (GGUF)      the model the Muse bot scores
#
# Same env var names as those bots, so one .env.eval entry serves all of them. A checkpoint absent
# from the box is SKIPPED and says so -- never a REJECT, never a silent pass.
MODELOPT_GUARD_MODEL_DIR = os.environ.get("MODELOPT_MODEL_DIR", "/root/workspace/models_q38_modelopt")
MODELOPT_GUARD_CTXS = [32768]
MUSE_GUARD_GGUF = os.environ.get(
    "MUSEGLIMMER_GGUF", "/root/workspace/models_muse_glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf")
MUSE_GUARD_CTXS = [32768]
# Ternary-Bonsai-2-27B (pt. 3c), the model pr_bonsai_bot.py scores, run the way that bot runs it.
BONSAI_GUARD_GGUF = os.environ.get(
    "BONSAI_GGUF", "/root/workspace/models_bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
BONSAI_GUARD_CTXS = [128, 32768]
# reps=5 (median), as for the Qwen3.6 guard: a guard that hard-REJECTs must not act on one sample.
GUARD_REPS = 5
# Concurrent-decode guards on the same two checkpoints. The 32k guards above run ONE request, but
# the PRs this bot scores mostly change the packed multi-row decode step, which a single request
# never enters -- so a PR could slow ModelOpt or Muse Glimmer under concurrency and pass every
# single-stream guard. Each width is the median of CB_REPS complete runs through cb_median, the same
# function the scored ladder uses, and each model is run the way its own bot runs it.
#
# Measured on main 9171513, two sessions of three runs, medians: ModelOpt c16 1078.3 / 1078.9 and
# c32 ~1598 / 1589.7; Muse Glimmer c16 1006.9 / 1005.6 and c32 1189.2 / 1181.3 tok/s. Spread 0.06-0.67%,
# far inside the -2% reject band. No request failed to open; one ModelOpt c32 run stopped part-way
# (7,970 of 8,200 tokens), which cb_complete rejects and re-runs. 8-12 s a run, so the four guards
# add about two minutes per ref.
CB_GUARD_CONCS = [16, 32]

# Auto-merge (the shape of pr_dflash_bot.py's auto_merge_ok_dflash/try_auto_merge_dflash) is OFF
# unless this exact env var is "1". The eval host's .env.eval sets it (explicit decision; see the
# module docstring); the wrappers never force it.
AUTO_MERGE = os.environ.get("SPARKINFER_QWEN38_AUTOMERGE") == "1"
AUTOMERGE_BLOCK = {
    "copycat", "copycat-warn", "flagged:gaming", "penalty", "needs-benchmark",
    QWEN38_NEEDS_REBASE, arb.REEVALUATE_LABEL, arb.HOLD_LABEL, *arb.REGRESSION_LABELS,
}

SCORES_FILE = os.path.expanduser(
    os.environ.get("QWEN38_SCORES_FILE", "~/.sparkinfer_qwen38_scores.json")
)
# Box faults per PR and commit (arb.record_strike): one recurring at a commit is charged to the PR.
STRIKES_FILE = os.path.expanduser(
    os.environ.get("QWEN38_STRIKES_FILE", "~/.sparkinfer_qwen38_strikes.json")
)
# When each PR began waiting on its author, per head: the stale close's clock (arb.AuthorWaitClock).
AUTHOR_WAIT_FILE = os.path.expanduser(
    os.environ.get("QWEN38_AUTHOR_WAIT_FILE", "~/.sparkinfer_qwen38_author_wait.json")
)
# PRs the bot gave up on this run (its own errors): the run then exits 3, so they are not silent.
GAVE_UP = set()

# Polaris verifiable-compute receipts — same policy/keys as the AR and DFlash bots (on by
# default; TDX via POLARIS_API_KEY when configured, else Ed25519 fallback). Wired through
# judge.py's --from-stdin generic RESULT_JSON path (NOT --dflash, which hardcodes a
# DFlash-shaped measurement block and eval_mode="dflash" — reusing it here would produce a
# mislabeled, semantically wrong attestation). SPARKINFER_EVAL_MODE is set explicitly below so
# the attestation correctly records "qwen38-128", not the AR bot's "longctx" default.
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
    g = (pr_tps - main_tps) / main_tps
    pct = round(100.0 * g, 1)
    if g < SIG:
        return "none", pct, True, f"within significance gate — not a verified {metric} improvement"
    for thr, name in BUCKETS:
        if g >= thr:
            return name, pct, True, "ok"
    return "none", pct, True, "ok"


_TIER_RANK = {"REJECT": -1, "none": 0, "XS": 1, "S": 2, "M": 3, "L": 4, "XL": 5}


def _cb_summary(d: dict) -> str:
    return " ".join(f"c{c}={float(d.get(f'cb{c}_agg') or 0):.1f}" for c in CB_CONCS)


def _cb_fields(pr: dict, main: dict, by_dim: dict) -> dict:
    """Per-width concurrent-decode numbers for the result, the PR comment and the sealed log."""
    out = {}
    for c in CB_CONCS:
        out[f"pr_cb{c}_agg"] = pr.get(f"cb{c}_agg")
        out[f"main_cb{c}_agg"] = main.get(f"cb{c}_agg")
        out[f"cb{c}_delta_pct"] = (by_dim.get(CB_DIM_FOR[c]) or {}).get("delta")
        out[f"pr_cb{c}_itl"] = pr.get(f"cb{c}_itl")
        out[f"main_cb{c}_itl"] = main.get(f"cb{c}_itl")
        out[f"pr_cb{c}_err"] = pr.get(f"cb{c}_err")
        out[f"main_cb{c}_err"] = main.get(f"cb{c}_err")
        out[f"pr_cb{c}_tok"] = pr.get(f"cb{c}_tok")
        out[f"main_cb{c}_tok"] = main.get(f"cb{c}_tok")
        out[f"pr_cb{c}_runs"] = pr.get(f"cb{c}_runs")
        out[f"main_cb{c}_runs"] = main.get(f"cb{c}_runs")
    return out


def _cb_table(res: dict) -> str:
    """The concurrent-decode block of the PR comment: its own table, because the x-axis is the
    number of requests in flight, not a context length."""
    rows = []
    for c in CB_CONCS:
        pv, mv = res.get(f"pr_cb{c}_agg"), res.get(f"main_cb{c}_agg")
        if pv is None or mv is None:
            continue
        d = res.get(f"cb{c}_delta_pct")
        role = "scored" if c in CB_SCORED_CONCS else "floor"
        delta = "?" if d is None else f"{d:+.1f}%"
        errs = f"{int(res.get(f'main_cb{c}_err') or 0)} / {int(res.get(f'pr_cb{c}_err') or 0)}"
        rows.append(f"| c{c} ({role}) | {mv:.1f} | {pv:.1f} | {delta} | {errs} |")
    if not rows:
        return ""
    return ("**Concurrent decode** — aggregate tok/s with N requests in flight "
            f"(`qwen3_gguf_cb_bench <checkpoint> N {CB_TOKENS} {CB_TOKENS} 512`)\n\n"
            "| requests | main | PR | delta | failed requests (main / PR) |\n"
            "|---|--:|--:|--:|--:|\n" + "\n".join(rows) + "\n\n")


def _check_model_guard(pr: dict, main: dict, key: str, model: str, tol: float = REGRESS_TOL,
                       metrics=("decode", "prefill"), label_for=None):
    """No-regression check for ONE guarded model: PR vs same-box main, decode + prefill, every
    measured context. `key` is the _parse_remote dict key holding that model's per-context numbers
    and `model` its display name, so every guard is the SAME code -- the pr_museglimmer_bot.py
    shape, which exists because a duplicated guard can quietly stop guarding. Returns
    (ok, [human-readable regression/failure strings])."""
    problems = []
    if pr.get(f"{key}_failed") or main.get(f"{key}_failed") or not pr.get(key) or not main.get(key):
        problems.append(f"{model} guard measurement unavailable")
    pr_ctxs, main_ctxs = pr.get(key) or {}, main.get(key) or {}
    # Iterate over MAIN's contexts (the reference set) — a PR build that crashes partway through
    # its own sweep must not make that context silently uncheckable. Fail closed: a real main
    # baseline (base > 0) with a missing/zero PR measurement (cur <= 0) is a regression, not a skip.
    for ctx, main_vals in main_ctxs.items():
        label = label_for(ctx) if label_for else GUARD_CTX_LABEL.get(ctx, str(ctx))
        pr_vals = pr_ctxs.get(ctx) or {}
        for metric in metrics:
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
    """Qwen3.6 no-regression guard (module docstring pt. 3), decode + prefill at ctx 0..32k."""
    return _check_model_guard(pr, main, "guard36", "qwen3.6", tol)


def check_modelopt_guard(pr: dict, main: dict, tol: float = REGRESS_TOL):
    """ModelOpt Qwen3.8-27B NVFP4 no-regression guard (pt. 3b), decode + prefill @ 32k.

    The 256k row shares the guardmo dict but is a scored, optional axis (#1113): a 256k sweep that
    fails on the PR (it peaks near the card's 32 GB) leaves that axis unscored. It used to be read
    here as "PR measurement missing -- treated as regression": a REJECT, and a close."""
    def no_longctx(d):
        return {**d, "guardmo": {c: v for c, v in (d.get("guardmo") or {}).items() if c != LONGCTX_CTX}}
    return _check_model_guard(no_longctx(pr), no_longctx(main), "guardmo", "modelopt", tol)


def check_muse_guard(pr: dict, main: dict, tol: float = REGRESS_TOL):
    """Muse Glimmer 30B no-regression guard (pt. 3b), decode + prefill @ 32k."""
    return _check_model_guard(pr, main, "guardmg", "muse glimmer", tol)


def check_bonsai_guard(pr: dict, main: dict, tol: float = REGRESS_TOL):
    """Ternary-Bonsai-2-27B no-regression guard (pt. 3c), decode + prefill @ 128 and 32k."""
    return _check_model_guard(pr, main, "guardbn", "ternary-bonsai", tol)


def check_modelopt_cb_guard(pr: dict, main: dict, tol: float = REGRESS_TOL):
    """ModelOpt concurrent-decode no-regression guard (pt. 3b), aggregate tok/s at CB_GUARD_CONCS."""
    return _check_model_guard(pr, main, "guardcbmo", "modelopt concurrent", tol,
                              metrics=("cb-decode",), label_for=lambda c: f"c{c}")


def check_muse_cb_guard(pr: dict, main: dict, tol: float = REGRESS_TOL):
    """Muse Glimmer concurrent-decode no-regression guard (pt. 3b), aggregate tok/s at CB_GUARD_CONCS."""
    return _check_model_guard(pr, main, "guardcbmg", "muse glimmer concurrent", tol,
                              metrics=("cb-decode",), label_for=lambda c: f"c{c}")


def qwen38_evaluated_commits(repo, num):
    """Head commits that already have a REAL scoring verdict posted — mirrors
    dflash_evaluated_commits: infra/transport failures (label:null in the marker) don't count."""
    return arb.evaluated_commits_from(repo, num, MARKER_RE, "sparkinfer qwen38 auto-eval")


def _verdict_heads(repo, num):
    """The heads this bot counts as measured: carrying its verdict marker AND recorded as the PR's
    latest verdict (arb.recorded_verdict_heads). None when GitHub did not answer."""
    return arb.recorded_verdict_heads(qwen38_evaluated_commits(repo, num), _load_scores().get(str(num)))


def strip_qwen38_eval_labels(repo, num):
    arb.strip_own_tier_labels(repo, num, EVAL_PREFIX)


STALE_DAYS = float(os.environ.get("QWEN38_STALE_DAYS", "1"))


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
    """A verified speedup at this head, sent to qwen38-needs-rebase only because another PR won
    merge-first. Until that PR merges and main moves there is nothing to rebase onto, so the wait
    is the merge's, not the author's; from then on the rebase is the author's (CONTRIBUTING), and
    the stale clock starts. A conflict, or any other block label, is the author's at once. Kept
    too when today's main is unknown."""
    entry = _load_scores().get(str(pr["number"])) or {}
    if (QWEN38_NEEDS_REBASE not in labs or labs & (AUTOMERGE_BLOCK - {QWEN38_NEEDS_REBASE})
            or entry.get("commit") != head or entry.get("label") not in SPEEDUP_LABELS or not entry.get("pass")
            or arb.pr_merge_conflict(pr.get("mergeable"))
            or arb.strike_count(STRIKES_FILE, pr["number"], head, "conflict")):
        return False
    return not main_now or not arb.scored_against_stale_main(entry, main_now)


def close_stale_qwen38_prs(repo, prs, dry_run=False):
    """Close open PRs routed to Qwen3.8-27B with no author commit activity in STALE_DAYS+ days.
    Drafts, `hold`, other models' PRs and any bot's merge-first are exempt
    (arb.stale_close_skip_reason) -- this used to close every idle PR in the repo, #1157 included.
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
        if arb.stale_close_skip_reason(pr, "qwen38", EVAL_PREFIX):
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
        evaluated = _verdict_heads(repo, num)
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
            note(f"PR #{num}: idle {age_days:.1f}d but still waiting for its first qwen38 verdict — kept open")
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
            "<!-- sparkinfer-qwen38-auto-close-stale -->\n"
            f"## Closed: stale — no commits in {age_days:.1f} days\n\n"
            f"This PR has had no new commits in over {STALE_DAYS:g} days — closing automatically "
            "to keep the Qwen3.8-27B eval queue clean. Nothing is wrong with it for that reason "
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


# SCORE_FAILED is the PR's own qwen3_gguf_score crashing: an explicit `exit 1`, so without it here
# _crash_reason found nothing and _is_box_fault read the run as a silent kill -- retried every round
# for ever with nothing posted, instead of the REJECT a crashing forward pass earns.
_EXPLICIT_FAIL_MARKERS = ("HARNESS_TOUCHED", "BUILD_FAILED", "LLAMACPP_CONFIGURE_FAILED", "LLAMACPP_BUILD_FAILED",
                          "MERGE_CONFLICT", "SCORE_FAILED", "BASE_AHEAD")


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
    for line in lines:
        if line.startswith("RETRYABLE_INFRA_FAILURE "):
            return line.strip()
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


# RETRYABLE_INFRA_FAILURE lines that name the box, not the ref. The concurrent-decode ladder's own
# RETRYABLE line is not here: a width the PR build cannot complete (main completed it this round) is
# judged over rounds instead, eval_qwen38_on_box's "cb" strike.
_BOX_FAULT_MARKERS = ("RETRYABLE_INFRA_FAILURE git ", "RETRYABLE_INFRA_FAILURE build:",
                      "RETRYABLE_INFRA_FAILURE GPU", "RETRYABLE_INFRA_FAILURE concurrent decode killed",
                      "RETRYABLE_INFRA_FAILURE score step killed")


def _is_box_fault(stdout: str, stderr: str) -> bool:
    """A failed run that is the box's rather than the PR's: nothing is posted and no label changes;
    the next round measures again (as pr_bonsai_bot.py does). Before, it posted an error comment and
    `eval-qwen38:REJECT` -- which also set the generic `eval:REJECT` that SN74 scoring reads -- every
    round until the box recovered."""
    combined = (stdout or "") + "\n" + (stderr or "")
    if any(m in combined for m in _BOX_FAULT_MARKERS):
        return True
    if "RETRYABLE_INFRA_FAILURE " in combined:
        return False
    crash = _crash_reason(stdout, stderr)
    if crash and "exit=137" in crash:
        # SIGKILL: the host OOM killer, whose trigger may be anything on the box (pr_bonsai_bot.py's
        # rule). Recurring at one commit, it is charged to the PR after BOX_FAULT_STRIKES rounds.
        return True
    return crash is None and "GUARD_END" not in combined


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
    # The concurrent-decode ladder names its own failures retryable: a model load at the widest
    # width can fail on memory a previous process has not returned yet.
    if "RETRYABLE_INFRA_FAILURE " in combined:
        return True
    if _crash_reason(stdout, stderr):
        return False
    return "GUARD_END" not in combined


def _ssh_run_resilient(host, port, script: str, label: str):
    """One automatic retry on an apparent hard kill — same insurance pr_dflash_bot.py added
    after #684/#690 (heavy model-reload boundaries silently killing the whole remote shell)."""
    r = ssh_run(host, port, script, via_stdin=True)
    if r.returncode != 0 and _looks_like_hard_kill(r.stdout, r.stderr):
        print(f">> {label}: looks like a hard kill (no ERR-trap diagnostic, the run never "
              f"reached GUARD_END) — retrying once")
        r = ssh_run(host, port, script, via_stdin=True)
    return r


def _remote_script(ref: str, role: str = "pr", onto: str | None = None) -> str:
    """Bash run on the eval box: checkout ref, build, decode+prefill@128 bench on the NVFP4
    checkpoint, teacher-forced score dump, and the Qwen3.6 no-regression guard.

    Run once per ref -- identical script both times so the two measurements are directly
    comparable. `role` only decides which score dump path is written and whether the
    differential accuracy compare runs (main has nothing to compare against yet; the PR run
    compares itself against main's dump from earlier in the same round).

    A PR (`ref` = pull/<n>/head) is measured MERGED onto `onto`, the exact main commit the round's
    baseline measured (arb.merged_checkout_script), with its harness pinned from that same commit.
    GitHub's own pull/<n>/merge is not used: it can be built on an older main than the baseline
    (#1145, 2026-09-24), which reads main's newer speedups as the PR's regressions."""
    if role == "pr":
        base = onto or "origin/main"
        checkout = arb.merged_checkout_script(ref, base, HARNESS_PATHS)
    else:
        base = "HEAD"   # main's harness is the commit just checked out, not a second fetch of main
        checkout = (f'timeout 600 git fetch -q origin {shlex.quote(ref)} || {{ echo "RETRYABLE_INFRA_FAILURE git fetch {ref} failed" >&2; exit 1; }}\n'
                    'find .git -maxdepth 1 -name index.lock -mmin +10 -delete 2>/dev/null || true\n'
                    'git reset -q --hard || { echo "RETRYABLE_INFRA_FAILURE git reset failed" >&2; exit 1; }\n'
                    'git clean -qfd || { echo "RETRYABLE_INFRA_FAILURE git clean failed" >&2; exit 1; }\n'
                    'git checkout -qf FETCH_HEAD || { echo "RETRYABLE_INFRA_FAILURE git checkout failed" >&2; exit 1; }\n'
                    'echo "REMOTE_HEAD $(git rev-parse --short HEAD)"\n'
                    'echo "REMOTE_SHA $(git rev-parse HEAD)"\n')
    base_q = shlex.quote(base)
    repo = shlex.quote(REMOTE_REPO)
    model_dir = shlex.quote(MODEL_DIR)
    ntok = BENCH_TOKENS
    topk = ACC_TOPK
    parity_bar = PARITY_BAR
    eval_text = shlex.quote(EVAL_TEXT)
    dump_self = shlex.quote(SCORE_DUMP_MAIN if role == "main" else SCORE_DUMP_PR)
    dump_main = shlex.quote(SCORE_DUMP_MAIN)
    is_pr = "1" if role == "pr" else "0"
    q36_dir = shlex.quote(Q36_GUARD_MODELS_DIR)
    q36_file = shlex.quote(Q36_GUARD_MODEL_FILE)
    q36_repo = shlex.quote(Q36_GUARD_MODEL_REPO)
    q36_tok = shlex.quote(Q36_GUARD_TOK_REPO)
    cb_concs = " ".join(str(c) for c in CB_CONCS)
    cb_tokens = CB_TOKENS
    cb_long_tokens = CB_LONG_TOKENS
    cb_reps = CB_REPS
    cb_max_attempts = CB_MAX_ATTEMPTS
    mo_dir = shlex.quote(MODELOPT_GUARD_MODEL_DIR)
    muse_gguf = shlex.quote(MUSE_GUARD_GGUF)
    mo_sweep_args = " ".join(f"{c} {GUARD_REPS}" for c in MODELOPT_GUARD_CTXS)
    mo_ctx_list = " ".join(str(c) for c in MODELOPT_GUARD_CTXS)
    lc_ctx, lc_reps = LONGCTX_CTX, LONGCTX_REPS
    mg_sweep_args = " ".join(f"{c} {GUARD_REPS}" for c in MUSE_GUARD_CTXS)
    mg_ctx_list = " ".join(str(c) for c in MUSE_GUARD_CTXS)
    bonsai_gguf = shlex.quote(BONSAI_GUARD_GGUF)
    bn_sweep_args = " ".join(f"{c} {GUARD_REPS}" for c in BONSAI_GUARD_CTXS)
    bn_ctx_list = " ".join(str(c) for c in BONSAI_GUARD_CTXS)
    cb_guard_concs = " ".join(str(c) for c in CB_GUARD_CONCS)
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
  # 180 s, then give up as infrastructure (the Muse bot's rule): 30 s and "proceeding anyway" loaded
  # the next model into a card the c=32 run had not released, and the OOM was charged to the PR.
  while [ "$tries" -lt 180 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -lt 1024 ] 2>/dev/null && return 0
    sleep 1
    tries=$((tries + 1))
  done
  echo "RETRYABLE_INFRA_FAILURE GPU still holding ${{used:-unknown}} MiB after ${{tries}}s — refusing to start a load that would OOM" >&2
  # exit, not return: cb_median runs inside `if !`, where set -e is suspended and a return would be
  # ignored. The whole run stops as infrastructure (_is_box_fault) wherever this is called from.
  exit 1
}}

export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH
export CUDA_HOME=${{CUDA_HOME:-/usr/local/cuda-13.0}}
{arb.round_guard_sh("qwen38")}
REPO={repo}
MODEL_DIR={model_dir}
NTOK={ntok}
TOPK={topk}
PARITY_BAR={parity_bar}
EVAL_TEXT={eval_text}
DUMP_SELF={dump_self}
DUMP_MAIN={dump_main}
IS_PR={is_pr}
Q36_GUARD_MODELS_DIR={q36_dir}
Q36_GUARD_MODEL_FILE={q36_file}
Q36_GUARD_MODEL_REPO={q36_repo}
Q36_GUARD_TOK_REPO={q36_tok}
MODELOPT_GUARD_MODEL_DIR={mo_dir}
MUSE_GUARD_GGUF={muse_gguf}
BONSAI_GUARD_GGUF={bonsai_gguf}

cd "$REPO"
git remote set-url origin https://github.com/gittensor-ai-lab/sparkinfer.git 2>/dev/null || true
{checkout}
# Pin the measuring instrument for every ref, main included (HARNESS_PATHS), from the main commit
# this ref is measured against. A PR that edits these files is skipped before it gets here, so
# this changes nothing for the PRs that are evaluated except that a branch older than a harness
# change is measured with main's ruler -- the same ruler as its baseline. A network failure is the
# box's, never the PR's: RETRYABLE, not the ERR trap (a sticky REJECT).
timeout 600 git fetch -q origin main || {{ echo "RETRYABLE_INFRA_FAILURE git fetch main failed" >&2; exit 1; }}
git checkout -q {base_q} -- runtime/examples/qwen3_gguf_bench.cpp \
  runtime/examples/qwen3_gguf_cb_bench.cpp runtime/examples/qwen3_gguf_score.cpp runtime/examples/qwen_checkpoint.h \
  runtime/examples/qwen3_gguf_config.h bench/scripts 2>/dev/null || {{
  echo "HARNESS_PIN_FAILED -- could not take the harness from {base}" >&2
  exit 1
}}
echo "HARNESS_PINNED $(git rev-parse --short {base_q})"

test -d "$MODEL_DIR" || {{ echo "FAIL missing NVFP4 checkpoint dir $MODEL_DIR"; exit 1; }}
test -f "$MODEL_DIR/config.json" || {{ echo "FAIL $MODEL_DIR has no config.json"; exit 1; }}

# Always reconfigure (cheap, idempotent) -- skipping it on an existing CMakeCache left stale
# generated Makefiles pointing at a DIFFERENT PR branch's files once the checkout switched
# underneath it (the sibling bots hit exactly this, #693/#694).
#
# Two more things pr_dspark_bot.py learned on this box, which this bot now shares a clone with:
# CMake keeps a cached CUDA compiler whatever PATH says (an apt CUDA 11.5 at /usr/bin/nvcc poisoned
# every round on 2026-08-21), and nvcc leaves GB-scale intermediates in /tmp that filled the disk
# on 2026-08-22. The round holds the shared lock, so no other build can be mid-flight.
rm -rf /tmp/tmpxft_* /tmp/*.nsys-rep /tmp/*.sqlite 2>/dev/null || true
echo "DISK_BEFORE_BUILD $(df -h / | awk 'NR==2{{print $4}}') free"
mkdir -p build
if [ -f build/CMakeCache.txt ] && ! grep -q '^CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda' build/CMakeCache.txt; then
  echo "WARN: build/CMakeCache.txt has a non-/usr/local/cuda CUDA compiler -- wiping build dir" >&2
  rm -rf build && mkdir -p build
fi
export CUDACXX="${{CUDACXX:-/usr/local/cuda/bin/nvcc}}"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/q38_cmake.log 2>&1 || {{
  echo "BUILD_FAILED -- cmake configure; tail:" >&2
  tail -40 /tmp/q38_cmake.log >&2
  exit 1
}}
{arb.BUILD_FAILURE_SH}
build_targets() {{
  cmake --build build --target qwen3_gguf_bench qwen3_gguf_score qwen3_gguf_generate qwen3_gguf_cb_bench -j"$1" >/tmp/q38_build.log 2>&1
}}
# A compiler killed for memory, a full disk or the overlayfs EFAULT is the box: rebuild once at -j4,
# then call it infra. Anything else is the PR's build error, errors first (arb.BUILD_FAILURE_SH).
if ! build_targets "$(nproc)"; then
  if FAULT=$(build_box_fault /tmp/q38_build.log); then
    echo "build hit a box-side fault ($FAULT) -- rebuilding with -j4" >&2
    if ! build_targets 4; then
      if FAULT=$(build_box_fault /tmp/q38_build.log); then
        echo "RETRYABLE_INFRA_FAILURE build: $FAULT" >&2
        exit 1
      fi
      report_build_failure /tmp/q38_build.log
      exit 1
    fi
  else
    report_build_failure /tmp/q38_build.log
    exit 1
  fi
fi
test -x build/runtime/qwen3_gguf_bench
test -x build/runtime/qwen3_gguf_score
test -x build/runtime/qwen3_gguf_cb_bench

# --- decode @ ctx=128 on the NVFP4 checkpoint ---
# The single scored dimension (module docstring pt. 1). Prefill is deliberately NOT scored here:
# at ctx=128 this model's batched prefill path declines anyway (it needs int8 KV, which the bench
# only turns on at ctx>=4096), so a prefill number at this context would measure the sequential
# fallback and move for reasons unrelated to the prefill kernels a PR touches.
#
# reps=5 (median), not 1: the sibling bots both learned this the hard way (PR #785's bogus XL,
# PR #790's false guard REJECT). pin_clocks() is unavailable on this box -- "current user does not
# have permission to change clocks" -- so median-of-N is the only mitigation for GPU clock
# variance available here.
source bench/scripts/_common.sh
source bench/scripts/_eval_speed.sh
SI_BIN="$PWD/build/runtime"; SI_LD=""

# Score the decode against a REAL prompt, not bench_decode's built-in synthetic ramp
# (ids[i] = 100 + i % 20000). Measured impact on this model is nil -- 82.61 vs 82.67 tok/s, inside
# run-to-run spread -- because Qwen3.8 is dense_ffn and dense decode is weight-bandwidth bound, so
# token content does not change the cost. The point is not the number, it is that a synthetic
# prompt is a gaming surface: an optimisation keyed on a repeating/ramping token stream would post
# a real-looking speedup here and nothing in production. That matters now that this bot AUTO-MERGES
# its winner without a human reading the diff. Both refs in a round use the same file, so the
# PR-vs-main comparison stays apples-to-apples either way.
# If the file is missing or too short, bench_decode logs and falls back to the ramp rather than
# padding -- a partly-synthetic prompt would be worse than an honestly synthetic one.
BENCH_PROMPT_IDS=/tmp/q38_bench_prompt_ids.txt
if python3 - "$MODEL_DIR/tokenizer.json" bench/scripts/bench_prompt.txt > "$BENCH_PROMPT_IDS" 2>/dev/null <<'PYBP'
import sys
from tokenizers import Tokenizer
ids = Tokenizer.from_file(sys.argv[1]).encode(open(sys.argv[2]).read()).ids
print(" ".join(str(i) for i in ids))
PYBP
then
  export SPARKINFER_BENCH_PROMPT_FILE="$BENCH_PROMPT_IDS"
  echo "BENCH_PROMPT_IDS $(wc -w < "$BENCH_PROMPT_IDS")"
else
  echo "BENCH_PROMPT_TOKENIZE_FAILED -- falling back to the synthetic prompt" >&2
fi

wait_gpu_clear
# 16384 joins the sweep for the long-context prefill dimension. It is the only context here that
# costs meaningful extra GPU time, and it is where prefill work actually lives: int8 KV turns on
# at ctx>=4096 so the BATCHED path is exercised, and a PR optimising long-context prefill shows
# nothing at 128. PR #834 was auto-closed for exactly that blind spot -- a real 1.29x prefill@4096
# win measured -0.0% against a 128-only metric.
if bench_sweep_run "$MODEL_DIR" "$NTOK" 128 5 16384 5; then
  DECODE128_TPS=$(_bench_sweep_get 128 decode_tps)
  # Same model load, same sweep, no extra GPU time for this one: prefill_pp is already computed
  # alongside decode_tps for this context, it was simply being discarded.
  PREFILL128_PP=$(_bench_sweep_get 128 prefill_pp)
  PREFILL16K_PP=$(_bench_sweep_get 16384 prefill_pp)
else
  DECODE128_TPS=0
  PREFILL128_PP=0
  PREFILL16K_PP=0
  # rc=137 is SIGKILL (the host OOM killer): the box's, not the PR's (eval_qwen38_on_box).
  echo "SWEEP_FAILED rc=${{_BENCH_SWEEP_RC:-1}}"
fi
echo "RESULT_DECODE128_TPS ${{DECODE128_TPS:-0}}"
echo "RESULT_PREFILL128_PP ${{PREFILL128_PP:-0}}"
echo "RESULT_PREFILL16K_PP ${{PREFILL16K_PP:-0}}"

# --- concurrent decode, cb-decode@c1..c32 (issue #1080) ---
# Aggregate tok/s with N requests in flight: the only stage here that enters the packed
# multi-row forward. One model load per width. Same invocation and env as pr_dspark_bot.py.
#
# A request the engine cannot open (out of memory at the widest width) is not a harness failure:
# qwen3_gguf_cb_bench logs "request error" and counts no tokens for it, so it shows up as lower
# aggregate throughput. The count is reported next to the number. A harness that exits nonzero or
# measures nothing IS infra, never a regression to zero -- scoring 0 would REJECT the PR for the
# harness's own failure.
# cb_complete C TOKENS ERRORS: did every request either finish or fail outright? C streams of
# {cb_tokens} tokens plus one long request of {cb_long_tokens}; a request that logged an error
# contributes nothing. Any other total means requests stopped part-way without an error, and the
# aggregate is then computed over a shortened wall time (see CB_CONCS in the bot).
cb_complete() {{
  local c=$1 tok=$2 err=$3 b a
  for b in 0 1; do
    a=$((err - b))
    [ "$a" -ge 0 ] && [ "$a" -le "$c" ] || continue
    [ "$tok" -eq $(( (c - a) * {cb_tokens} + (1 - b) * {cb_long_tokens} )) ] && return 0
  done
  return 1
}}

# cb_median CHECKPOINT C [ENV=VALUE ...]: aggregate tok/s with C requests in flight, the median of
# {cb_reps} complete runs (cb_complete). Sets CB_AGG, CB_ITL, CB_TOK, CB_ERR and CB_AGGS (the runs).
# A crashed or cut-short run is one failed attempt of {cb_max_attempts}; a hang (124) ends it at once.
# Returns 1, with the reason on stderr, when too few runs completed, a run hung, or nothing positive
# was measured; CB_RC is then 137 only if the OOM killer took every failed attempt, else the last
# other exit (0 when the others were runs cut short, or nothing positive was measured). The caller
# decides what that means: the scored ladder fails the round as infra; a guard width main measured
# is a fault of the PR's run judged over rounds (eval_qwen38_on_box's "guard-cb" strike).
cb_median() {{
  local ckpt=$1 cc=$2 out=/tmp/q38_cb.txt attempt=0 valid=0 a i all_killed=1 last_rc=0 other_rc=0
  shift 2
  CB_AGGS=""; CB_ITLS=""; CB_AGG=0; CB_ITL=0; CB_TOK=0; CB_ERR=0; CB_RC=0
  while [ "$valid" -lt {cb_reps} ]; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt {cb_max_attempts} ]; then
      echo "concurrent decode at c=$cc on $ckpt did not complete on $((attempt - 1 - valid)) of {cb_max_attempts} runs" >&2
      # 137 only when the OOM killer took every failed attempt (the box's); else the last other exit.
      if [ "$all_killed" = 1 ] && [ "$last_rc" = 137 ]; then CB_RC=137; else CB_RC=$other_rc; fi
      return 1
    fi
    wait_gpu_clear
    if timeout 900 env "$@" build/runtime/qwen3_gguf_cb_bench "$ckpt" "$cc" {cb_tokens} {cb_tokens} 512 > "$out" 2>&1; then
      :
    else
      # One failed attempt, like a run cut short -- it used to give up on the width at once, so
      # one transient crash failed a guard (a REJECT and a close). A hang is not retried: it repeats.
      last_rc=$?
      echo "concurrent-decode harness exited $last_rc at c=$cc on $ckpt (attempt $attempt)" >&2
      tail -20 "$out" >&2 || true
      if [ "$last_rc" != 137 ]; then all_killed=0; other_rc=$last_rc; fi
      if [ "$last_rc" = 124 ]; then CB_RC=124; return 1; fi
      continue
    fi
    CB_TOK=$(sed -n 's/.*decode_tokens=\\([0-9]*\\).*/\\1/p' "$out" | tail -1)
    CB_ERR=$(grep -c "request error" "$out" || true)
    if ! cb_complete "$cc" "${{CB_TOK:-0}}" "${{CB_ERR:-0}}"; then
      echo "CB_PARTIAL c=$cc attempt=$attempt decode_tokens=${{CB_TOK:-0}} request_errors=${{CB_ERR:-0}} ($ckpt)" >&2
      all_killed=0
      continue
    fi
    a=$(sed -n 's/.*agg_tok_s=\\([0-9.]*\\).*/\\1/p' "$out" | tail -1)
    i=$(sed -n 's/.*mean_itl_ms=\\([0-9.]*\\).*/\\1/p' "$out" | tail -1)
    CB_AGGS="$CB_AGGS ${{a:-0}}"; CB_ITLS="$CB_ITLS ${{i:-0}}"
    valid=$((valid + 1))
  done
  CB_AGG=$(python3 -c "import statistics, sys; print(statistics.median(float(x) for x in sys.argv[1:]))" $CB_AGGS)
  CB_ITL=$(python3 -c "import statistics, sys; print(statistics.median(float(x) for x in sys.argv[1:]))" $CB_ITLS)
  if ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" "${{CB_AGG:-0}}"; then
    echo "concurrent decode produced no positive metric at c=$cc on $ckpt" >&2
    return 1
  fi
}}

for CC in {cb_concs}; do
  if ! cb_median "$MODEL_DIR" "$CC" SPARKINFER_QWEN38_PREFILL_NVFP4=1 SPARKINFER_QWEN38_DECODE_NVFP4=1 SPARKINFER_KV_INT8=1; then
    if [ "${{CB_RC:-0}}" = 137 ]; then
      # SIGKILL (the host OOM killer): the box's (_BOX_FAULT_MARKERS), bounded by BOX_FAULT_STRIKES.
      echo "RETRYABLE_INFRA_FAILURE concurrent decode killed at c=$CC (exit 137)" >&2
      exit 75
    fi
    echo "RETRYABLE_INFRA_FAILURE concurrent decode failed at c=$CC (see above)" >&2
    exit 75
  fi
  echo "RESULT_CB${{CC}}_RUNS$CB_AGGS"
  echo "RESULT_CB${{CC}}_TOK ${{CB_TOK:-0}}"
  echo "RESULT_CB${{CC}}_AGG ${{CB_AGG:-0}}"
  echo "RESULT_CB${{CC}}_ITL ${{CB_ITL:-0}}"
  echo "RESULT_CB${{CC}}_ERR ${{CB_ERR:-0}}"
done

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
if build/runtime/qwen3_gguf_score "$MODEL_DIR" "$TOPK" $IDS > "$DUMP_SELF" 2>/tmp/q38_score.err; then
  :
else
  SCORE_RC=$?
  # SIGKILL is the host OOM killer, whose trigger may be anything on the box: infra (_BOX_FAULT_MARKERS).
  if [ "$SCORE_RC" = 137 ]; then
    echo "RETRYABLE_INFRA_FAILURE score step killed (exit 137)" >&2
    exit 1
  fi
  echo "SCORE_FAILED -- tail of /tmp/q38_score.err:" >&2
  tail -40 /tmp/q38_score.err >&2
  exit 1
fi
echo "ACCURACY_STAGE_DONE"

# --- batched-prefill parity: NOT RUN (2026-09-15) ---
# This bot used to hard-REJECT on bench/scripts/prefill_parity_check.py, which compares batched
# prefill with the token loop inside one build: absolute, not PR-vs-main. main 507017b fails it on
# this checkpoint at n=32 AND n=128 (common prefix 3-4 of 24 tokens against a 0.75 bar, three runs
# out of three), so with the gate on, every PR would be REJECTed and auto-closed. pr_dspark_bot.py
# turned the same gate off for the same reason on the ModelOpt checkpoint. Re-enable it once main
# passes.

if [ "$IS_PR" = "1" ]; then
  if [ -s "$DUMP_MAIN" ]; then
    python3 bench/scripts/accuracy_compare_pair.py "$DUMP_SELF" "$DUMP_MAIN" || true
  else
    echo "ACCURACY_NO_BASELINE" >&2
  fi
else
  # main against itself must read top-1 1.000 / KL 0 (pr_bonsai_bot.py's SELFCHECK): a main dump
  # the comparator cannot read would give every PR in the round top1=0 -- a REJECT, and a close.
  python3 bench/scripts/accuracy_compare_pair.py "$DUMP_SELF" "$DUMP_SELF" --metric-label SELFCHECK || true
fi

# --- Qwen3.6 no-regression guard (decode + prefill, ctx 0/512/4k/16k/32k) ---
# A separate model load from Qwen3.8's own: a shared-code regression that only shows up on
# Qwen3.6's architecture would otherwise slip past this bot entirely, as it did for the LMCache
# integration (PR #775) until checked by hand. reps=5 for the same clock-variance reason as above.
# _common.sh/_eval_speed.sh/SI_BIN already sourced above -- reused here, not re-sourced.
#
# The real-prompt file above is Qwen3.8 token ids and MUST NOT leak into this guard: Qwen3.6 is a
# different model with a different vocabulary, so those ids denote different text (or none). The
# guard also sweeps to ctx=32768, far past this prompt's length, which would fall back per-context
# anyway. Unset so the guard is unambiguously synthetic on both refs -- which is all it needs,
# since it is a PR-vs-main comparison, not an absolute number.
unset SPARKINFER_BENCH_PROMPT_FILE
export MODELS_DIR="$Q36_GUARD_MODELS_DIR" MODEL_REPO="$Q36_GUARD_MODEL_REPO" \\
       MODEL_FILE="$Q36_GUARD_MODEL_FILE" TOK_REPO="$Q36_GUARD_TOK_REPO"
export MODEL_SHA256="${{QWEN36_MODEL_SHA256:-}}"
( ensure_model && ensure_tokenizer ) || echo "WARN: qwen3.6 guard model setup failed" >&2
Q36_GGUF="$Q36_GUARD_MODELS_DIR/$Q36_GUARD_MODEL_FILE"

echo "GUARD_START"
wait_gpu_clear
if bench_sweep_run "$Q36_GGUF" 128 0 5 512 5 4096 5 16384 5 32768 5; then
  for ctx in 0 512 4096 16384 32768; do
    echo "GUARD36 $ctx $(_bench_sweep_get $ctx decode_tps) $(_bench_sweep_get $ctx prefill_pp)"
  done
else
  echo "GUARD36_FAILED rc=${{_BENCH_SWEEP_RC:-1}}"
fi

# --- ModelOpt Qwen3.8-27B NVFP4 no-regression guard (decode + prefill @ 32k) ---
# Same block as pr_museglimmer_bot.py's. A compressed-tensors directory; qwen3_gguf_bench reads
# either kind. Skipped, not failed, when the checkpoint is absent from the box.
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

# --- long-context decode, ModelOpt NVFP4 @ 256k (#1113) ---
# Its own sweep call, and its own rep count: one row fills the 262144-token context, ~60 s, 30.0 GB
# peak. A failure here is NOT a guard failure -- the axis simply goes unscored for this round.
if [ -d "$MODELOPT_GUARD_MODEL_DIR" ]; then
  wait_gpu_clear
  if bench_sweep_run "$MODELOPT_GUARD_MODEL_DIR" 128 {lc_ctx} {lc_reps}; then
    echo "GUARDMO {lc_ctx} $(_bench_sweep_get {lc_ctx} decode_tps) $(_bench_sweep_get {lc_ctx} prefill_pp)"
  else
    echo "LONGCTX_FAILED"
  fi
else
  echo "LONGCTX_UNAVAILABLE"
fi

# --- Muse Glimmer 30B no-regression guard (decode + prefill @ 32k) ---
# The model pr_museglimmer_bot.py scores, measured the way that bot measures it: qwen3_gguf_bench
# on the GGUF with no env pins. Skipped, not failed, when the GGUF is absent from the box.
if [ -f "$MUSE_GUARD_GGUF" ]; then
  wait_gpu_clear
  if bench_sweep_run "$MUSE_GUARD_GGUF" 128 {mg_sweep_args}; then
    for ctx in {mg_ctx_list}; do
      echo "GUARDMG $ctx $(_bench_sweep_get $ctx decode_tps) $(_bench_sweep_get $ctx prefill_pp)"
    done
  else
    echo "GUARDMG_FAILED rc=${{_BENCH_SWEEP_RC:-1}}"
  fi
else
  echo "GUARDMG_UNAVAILABLE"
fi

# --- Ternary-Bonsai-2-27B no-regression guard (decode + prefill @ 128 and 32k, pt. 3c) ---
# The model pr_bonsai_bot.py scores, run the way that bot runs it: qwen3_gguf_bench on the GGUF, no
# env pins. Skipped, not failed, when the GGUF is absent from the box.
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

# --- Concurrent-decode no-regression guards: ModelOpt and Muse Glimmer (pt. 3b) ---
# The PRs this bot scores mostly change packed decode, which the single-stream 32k guards above never
# enter. Each model runs the way its own bot runs it: ModelOpt with pr_dspark_bot.py's env, Muse
# Glimmer with none (pr_museglimmer_bot.py). A width the PR build could not complete prints *_FAILED;
# unlike the 32k guards it is judged over rounds (guard-cb), as a scored width is. An absent checkpoint
# skips both of its guards.
if [ -d "$MODELOPT_GUARD_MODEL_DIR" ]; then
  for CC in {cb_guard_concs}; do
    if cb_median "$MODELOPT_GUARD_MODEL_DIR" "$CC" SPARKINFER_QWEN38_PREFILL_NVFP4=1 SPARKINFER_QWEN38_DECODE_NVFP4=1 SPARKINFER_KV_INT8=1; then
      echo "GUARDCBMO $CC $CB_AGG"
    else
      echo "GUARDCBMO_FAILED $CC rc=${{CB_RC:-1}}"
    fi
  done
fi
if [ -f "$MUSE_GUARD_GGUF" ]; then
  for CC in {cb_guard_concs}; do
    if cb_median "$MUSE_GUARD_GGUF" "$CC"; then
      echo "GUARDCBMG $CC $CB_AGG"
    else
      echo "GUARDCBMG_FAILED $CC rc=${{CB_RC:-1}}"
    fi
  done
fi
echo "GUARD_END"
"""


def _parse_remote(stdout: str) -> dict:
    """Markers emitted by _remote_script + the METRIC line accuracy_compare_pair.py prints.

    Unlike the sibling bots, the accuracy numbers are NOT re-echoed as RESULT_* by the remote
    bash -- the comparator's own machine-readable METRIC line is parsed directly, so there is one
    fewer place for the two to disagree about what was measured."""
    out = {}
    guard36 = {}
    cross_guards = {"GUARDMO": {}, "GUARDMG": {}, "GUARDBN": {}}
    cb_guards = {"GUARDCBMO": {}, "GUARDCBMG": {}}
    for line in (stdout or "").splitlines():
        if line.startswith("REMOTE_HEAD "):
            out["head"] = line.split()[1]
        elif line.startswith("REMOTE_SHA "):
            out["sha"] = line.split()[1]
        elif line.startswith("PR_TIP "):
            out["pr_tip"] = line.split()[1]
        elif line.startswith("MERGED_ONTO "):
            out["merged_onto"] = line.split()[1]
        elif line.startswith("PREFILL_PARITY_OK"):
            out["parity_ok"] = True
        elif line.startswith("PREFILL_PARITY_FAILED"):
            out["parity_ok"] = False
        elif line.startswith("PARITY worst="):
            try:
                out["parity_worst"] = float(line.split("worst=")[1].split()[0])
            except (ValueError, IndexError):
                pass
        elif line.startswith("RESULT_DECODE128_TPS "):
            try:
                out["decode128_tps"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_PREFILL128_PP "):
            try:
                out["prefill128_pp"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_PREFILL16K_PP "):
            try:
                out["prefill16k_pp"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_CB") and line.split()[0].endswith(("_AGG", "_ITL", "_ERR", "_TOK", "_RUNS")):
            # RESULT_CB<N>_AGG / _ITL / _ERR / _TOK -> out["cb<N>_agg"] / ["cb<N>_itl"] / ...;
            # RESULT_CB<N>_RUNS -> out["cb<N>_runs"], the individual complete runs the median is of.
            key, _, val = line.partition(" ")
            n, _, kind = key[len("RESULT_CB"):].partition("_")
            try:
                if kind == "RUNS":
                    out[f"cb{int(n)}_runs"] = [float(x) for x in val.split()]
                else:
                    out[f"cb{int(n)}_{kind.lower()}"] = float(val.split()[0])
            except (ValueError, IndexError):
                pass
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
                if k in ("top1", "kl", "ppl_pr", "ppl_main", "n", "n_main"):
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
        elif line.startswith("SELFCHECK "):
            for tok in line.split()[1:]:
                k, _, v = tok.partition("=")
                if k in ("top1", "kl"):
                    try:
                        out[f"selfcheck_{k}"] = float(v)
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
        elif line.split(" ", 1)[0] in cb_guards:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    cb_guards[parts[0]][int(parts[1])] = {"cb-decode": float(parts[2])}
                except ValueError:
                    pass
        elif line.split(" ", 1)[0] in ("GUARDCBMO_FAILED", "GUARDCBMG_FAILED"):
            key = line.split(" ", 1)[0].split("_")[0].lower()
            out[key + "_failed"] = True
            rc = arb.failed_rc(line)
            if rc == 137:
                out[key + "_failed_box"] = True
            width = line.split()[1] if len(line.split()) > 2 else "?"
            out.setdefault(key + "_failed_at", []).append(
                f"c{width} " + {0: "too few usable runs", 124: "hung", 137: "killed"}.get(rc, f"crashed, rc={rc}"))
        elif line.split(" ", 1)[0] in cross_guards:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    cross_guards[parts[0]][int(parts[1])] = {"decode": float(parts[2]),
                                                             "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.split()[:1] in (["GUARDMO_FAILED"], ["GUARDMG_FAILED"], ["GUARDBN_FAILED"]):
            out[line.split()[0].split("_")[0].lower() + "_failed"] = True
            if arb.failed_rc(line) == 137:
                out[line.split()[0].split("_")[0].lower() + "_failed_box"] = True
        elif line.split()[:1] == ["SWEEP_FAILED"]:
            out["sweep_failed"] = True
            if arb.failed_rc(line) == 137:
                out["sweep_failed_box"] = True
        elif line.strip() in ("LONGCTX_FAILED", "LONGCTX_UNAVAILABLE"):
            # Soft: the 256k axis is not scored this round. Never a guard failure -- a checkpoint
            # that is absent, or a sweep that could not fit, must not reject a PR (#1113).
            out["longctx_unmeasured"] = True
        elif line.strip() in ("GUARDMO_UNAVAILABLE", "GUARDMG_UNAVAILABLE", "GUARDBN_UNAVAILABLE"):
            out[line.strip().split("_")[0].lower() + "_unavailable"] = True
    out["guard36"] = guard36
    out["guardmo"] = cross_guards["GUARDMO"]
    out["guardmg"] = cross_guards["GUARDMG"]
    out["guardbn"] = cross_guards["GUARDBN"]
    out["guardcbmo"] = cb_guards["GUARDCBMO"]
    out["guardcbmg"] = cb_guards["GUARDCBMG"]
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
        "model": "qwen38-128",
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
    eval_seed = f"qwen38-{int(time.time() * 1000)}"  # unique nonce per attestation
    stdin_payload = "RESULT_JSON " + json.dumps(result_json)
    cmd = (
        f"cd {shlex.quote(REMOTE_REPO)} && "
        f"SPARKINFER_EVAL_MODE=qwen38-128 SPARKINFER_DECODE_TOKENS={BENCH_TOKENS} "
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
        tail = arb.failure_excerpt(r.stdout, r.stderr, _EXPLICIT_FAIL_MARKERS)
        crash = _crash_reason(r.stdout, r.stderr)
        reason = "main run failed" + (f" — {crash}" if crash else " (no crash diagnostic captured, possible hard kill — retried once)")
        return {"ok": False, "reason": reason, "log": tail}
    main = _parse_remote(r.stdout or "")
    if main.get("sweep_failed"):
        return {"ok": False, "reason": "main speed sweep failed"
                                       + (" (killed, exit 137 — the box's)" if main.get("sweep_failed_box") else ""),
                "log": (r.stdout or "")[-1500:]}
    # Fail closed on a ZERO too, not just a missing line: the script prints 0 for a width it could not
    # measure (a 16k KV pool that fails to allocate yields 0 rather than an error), and a 0 baseline
    # makes every PR of the round a REJECT, closed.
    if not (main.get("decode128_tps") or 0) > 0:
        return {"ok": False, "reason": "main bench missing/zero decode@128 tok/s", "log": (r.stdout or "")[-1500:]}
    if not (main.get("prefill128_pp") or 0) > 0:
        return {"ok": False, "reason": "main bench missing/zero prefill@128 pp", "log": (r.stdout or "")[-1500:]}
    if not main.get("prefill16k_pp"):
        return {"ok": False, "reason": "main bench missing/zero prefill@16k pp (KV pool alloc?)",
                "log": (r.stdout or "")[-1500:]}
    missing_cb = [c for c in CB_CONCS if not main.get(f"cb{c}_agg")]
    if missing_cb:
        return {"ok": False, "reason": "main bench missing/zero concurrent decode at "
                                       + "/".join(f"c{c}" for c in missing_cb),
                "log": (r.stdout or "")[-1500:]}
    # Every guard must have measured something unless its checkpoint is absent. Otherwise every PR
    # in the round is measured in full only to be deferred for the missing guard.
    # The 256k row lives in guardmo but is a scored axis, not the guard (check_modelopt_guard):
    # main's 32k ModelOpt sweep failing must skip the round even when 256k was measured.
    def rows(key):
        r = main.get(key) or {}
        return {c: v for c, v in r.items() if c != LONGCTX_CTX} if key == "guardmo" else r
    unguarded = [key for key, flag in (("guard36", None), ("guardmo", "guardmo"), ("guardcbmo", "guardmo"),
                                       ("guardmg", "guardmg"), ("guardcbmg", "guardmg"),
                                       ("guardbn", "guardbn"))
                 if not (flag and main.get(f"{flag}_unavailable"))
                 and (not arb.guard_measured(rows(key)) or main.get(f"{key}_failed"))]
    if unguarded:
        return {"ok": False, "reason": "main measured nothing for guard(s) " + ", ".join(unguarded),
                "log": (r.stdout or "")[-1500:]}
    if main.get("selfcheck_top1") != 1.0 or (main.get("selfcheck_kl") or 0.0) > 1e-6:
        # The dump every PR's accuracy is compared with: one the comparator cannot read would
        # REJECT, and close, every PR in the round.
        return {"ok": False, "reason": "main score dump failed its self-comparison "
                                       f"(top1={main.get('selfcheck_top1')} kl={main.get('selfcheck_kl')})",
                "log": (r.stdout or "")[-1500:]}
    if not main.get("sha"):
        # Every verdict records it (`onto`); without it every PR would count as scored on an old main.
        return {"ok": False, "reason": "main run did not report its commit", "log": (r.stdout or "")[-1500:]}
    main["ok"] = True
    return main


def _guard_coverage(d: dict) -> str:
    """One line naming how many rows each guard produced -- a guard that measured nothing is the
    difference between a REJECT and a retry, so the round log should not make anyone guess."""
    return (f"modelopt {len(d.get('guardmo') or {})} ctx / {len(d.get('guardcbmo') or {})} cc · "
            f"muse {len(d.get('guardmg') or {})} ctx / {len(d.get('guardcbmg') or {})} cc · "
            f"qwen3.6 {len(d.get('guard36') or {})} ctx · "
            f"bonsai {len(d.get('guardbn') or {})} ctx")


# The guards' names in reasons and comments, by the key their remote lines use.
_GUARD_NAMES = {"guard36": "qwen3.6", "guardmo": "modelopt 32k", "guardmg": "muse glimmer 32k",
                "guardbn": "ternary-bonsai", "guardcbmo": "modelopt concurrent-decode",
                "guardcbmg": "muse glimmer concurrent-decode"}


def _cb_attempt_lines(r, pr, limit=1800):
    """What cb_median said (its stderr) about the concurrent guard widths that did not complete: not
    the scored ladder's lines, nor a guard width's that passed in the end. The end is kept."""
    failed = [(path, "c=" + at.split()[0][1:] + " ")
              for key, path in (("guardcbmo", MODELOPT_GUARD_MODEL_DIR), ("guardcbmg", MUSE_GUARD_GGUF))
              for at in (pr.get(f"{key}_failed_at") or [])]
    lines = [l for l in (r.stderr or "").splitlines()
             if l.startswith(("concurrent decode at c=", "concurrent-decode harness exited", "CB_PARTIAL",
                              "concurrent decode produced no positive metric"))
             and any(w in l and (f" on {path} " in l + " " or f"({path})" in l) for path, w in failed)]
    text = "\n".join(lines)
    return text if len(text) <= limit else text[len(text) - limit:].split("\n", 1)[-1]


def eval_qwen38_on_box(host, port, pr_ref: str, main: dict):
    """Run the PR ref's speed+accuracy script on the same box and compare against `main`, an
    already-measured baseline shared across every PR in the round (see measure_main_baseline)."""
    print(f">> Qwen3.8-27B eval on box: PR ref={pr_ref}")
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
        crash = _crash_reason(r.stdout, r.stderr)
        reason = "PR speed/accuracy run failed" + (f" — {crash}" if crash else " (no crash diagnostic captured, possible hard kill — retried once)")
        # A concurrency width the PR build could not complete, while main did this round: judged over
        # rounds, as the sibling bots judge it, not REJECTed on one run -- that label marks the commit
        # evaluated, so one flake stuck until a push. "box": any fault recurring at one commit is
        # charged to the PR after BOX_FAULT_STRIKES rounds.
        cb = "RETRYABLE_INFRA_FAILURE concurrent decode failed" in (r.stderr or "") + (r.stdout or "")
        box = _is_box_fault(r.stdout, r.stderr)
        return {"ok": False, "retry": box or cb, "strike_key": "box" if box or not cb else "cb", "reason": reason,
                "log": tail, "pr_tip": _parse_remote(r.stdout or "").get("pr_tip")}
    pr = _parse_remote(r.stdout or "")
    if "ACCURACY_NO_BASELINE" in (r.stderr or ""):
        # main's score dump was gone when this run compared against it: the box, not the PR.
        # Bounded all the same (a PR's own build could remove the dump).
        return {"ok": False, "pr_tip": pr.get("pr_tip"), "retry": True, "strike_key": "box",
                "reason": "main's score dump was gone when the PR run compared against it", "log": ""}
    t1, kl = pr.get("top1"), pr.get("kl")
    output_wrong = t1 is not None and kl is not None and (t1 < ACC_TOP1_BAR or kl > ACC_KL_BAR)
    if pr.get("sweep_failed_box") and not output_wrong:
        # The speed sweep was SIGKILLed (the host OOM killer): the box's, like a build the compiler
        # cannot finish. Its zeros must not read as a regression; charged after BOX_FAULT_STRIKES.
        return {"ok": False, "pr_tip": pr.get("pr_tip"), "retry": True, "strike_key": "sweep-box", "log": "",
                "reason": "the PR's Qwen3.8 speed sweep was killed (exit 137) — infra"}
    if "decode128_tps" not in pr:
        return {"ok": False, "pr_tip": pr.get("pr_tip"), "reason": "PR bench missing decode@128 tok/s", "log": (r.stdout or "")[-1500:]}
    if "prefill128_pp" not in pr:
        return {"ok": False, "pr_tip": pr.get("pr_tip"), "reason": "PR bench missing prefill@128 pp", "log": (r.stdout or "")[-1500:]}
    if not pr.get("prefill16k_pp"):
        return {"ok": False, "pr_tip": pr.get("pr_tip"), "reason": "PR bench missing/zero prefill@16k pp (KV pool alloc?)",
                "log": (r.stdout or "")[-1500:]}
    missing_cb = [c for c in CB_CONCS if not pr.get(f"cb{c}_agg")]
    if missing_cb:
        return {"ok": False, "pr_tip": pr.get("pr_tip"), "reason": "PR bench missing/zero concurrent decode at "
                                       + "/".join(f"c{c}" for c in missing_cb),
                "log": (r.stdout or "")[-1500:]}
    if "top1" not in pr or "kl" not in pr:
        # Either the score dump failed, or main's dump was missing so the comparator never ran
        # (ACCURACY_NO_BASELINE). Both are infra faults, but they must NOT pass as "accurate" --
        # a gate that cannot measure fails closed, same as check_q36_guard's unavailability path.
        return {"ok": False, "pr_tip": pr.get("pr_tip"), "reason": "PR run missing accuracy METRIC line (score dump failed, "
                                       "or no main baseline dump to diff against)",
                "log": (r.stdout or "")[-1500:]}
    print(f">> PR decode@128={pr['decode128_tps']:.2f} prefill@128={pr['prefill128_pp']:.2f} "
          f"prefill@16k={pr['prefill16k_pp']:.2f} cb {_cb_summary(pr)} "
          f"top1={pr.get('top1', 0):.4f} kl={pr.get('kl', 99):.5f}")
    print(f">> PR guard coverage: {_guard_coverage(pr)}")

    # THREE scored dimensions on the NVFP4 checkpoint, all from the one model load (module
    # docstring pt. 1). Same rule pr_museglimmer_bot.py uses for its two, generalised rather than
    # grown into a longer if/elif chain -- with three dimensions the hand-written cascade needs
    # 2^3 orderings to stay correct and silently mis-scores if one is missed:
    #   * ANY dimension regressing is a hard REJECT, regardless of the others.
    #   * Otherwise the BEST tier wins, so a PR that improves one axis with the others merely flat
    #     still scores for the real work it did (a long-context prefill PR is not expected to move
    #     decode@128, and vice versa).
    dims = [
        ("decode@128",  pr["decode128_tps"],   main["decode128_tps"]),
        ("prefill@128", pr["prefill128_pp"],   main["prefill128_pp"]),
        ("prefill@16k", pr["prefill16k_pp"],   main["prefill16k_pp"]),
    ] + [(CB_DIM_FOR[c], pr[f"cb{c}_agg"], main[f"cb{c}_agg"]) for c in CB_CONCS]
    # Long-context decode (#1113): scored, with its prefill a floor beside it, exactly as the other
    # tiers work. Both arms must have measured it -- a round where the sweep did not run scores the
    # PR on everything else rather than inventing a number.
    lc_pr = (pr.get("guardmo") or {}).get(LONGCTX_CTX) or {}
    lc_main = (main.get("guardmo") or {}).get(LONGCTX_CTX) or {}
    if lc_pr.get("decode") and lc_main.get("decode"):
        dims = dims + [(LONGCTX_DECODE_DIM, lc_pr["decode"], lc_main["decode"]),
                       (LONGCTX_PREFILL_DIM, lc_pr["prefill"], lc_main["prefill"])]
    scored = []
    for name, pr_v, main_v in dims:
        lab, dlt, ok, why = tier_from_gain(pr_v, main_v, metric=name)
        scored.append({"dim": name, "label": lab, "delta": dlt, "passed": ok, "reason": why})
    by_dim = {s["dim"]: s for s in scored}

    # ANY dimension regressing is still a hard REJECT. decode@128 and prefill@128 are NOT scoring
    # dimensions any more -- they cannot earn a tier -- but they remain no-regression FLOORS,
    # because without them a PR could trade decode throughput away to buy long-context prefill and
    # still auto-merge at XL. They cost nothing to keep: all three come from the one sweep.
    regressed = [s for s in scored if s["label"] == "REJECT"]
    if regressed:
        worst = min(regressed, key=lambda s: s["delta"])
        label, delta_pct, passed = "REJECT", worst["delta"], False
        speed_reason = " | ".join(s["reason"] for s in regressed)
        best = worst
    else:
        # The tier comes from the best of SCORING_DIMS by measured delta: prefill@16k or a
        # concurrent width. An improvement to a floor alone (decode@128, prefill@128, c1) scores
        # "none" by design. max() over deltas rather than tier letters, as pr_dspark_bot.py does:
        # two dimensions can share a bucket while one is clearly the larger win.
        best = max((by_dim[d] for d in SCORING_DIMS if d in by_dim), key=lambda s: s["delta"])
        label, delta_pct, passed, speed_reason = best["label"], best["delta"], best["passed"], best["reason"]
    decode_label,  decode_delta_pct  = by_dim["decode@128"]["label"],  by_dim["decode@128"]["delta"]
    prefill_label, prefill_delta_pct = by_dim["prefill@128"]["label"], by_dim["prefill@128"]["delta"]
    p16k_label,    p16k_delta_pct    = by_dim["prefill@16k"]["label"], by_dim["prefill@16k"]["delta"]
    # Keep the speed-only verdict: `label` below can be forced to REJECT by the accuracy gate or
    # the Qwen3.6 guard, and the comment/dashboard still need to say whether speed itself moved.
    speed_label = label

    pr_top1 = pr.get("top1", 0.0)
    pr_kl = pr.get("kl", 99.0)
    # Every position main's dump has, not just the ones the PR's dump has: the compare skips the rest.
    covered = pr.get("n_main") is None or pr.get("n", 0) >= pr["n_main"]
    accuracy_ok = pr_top1 >= ACC_TOP1_BAR and pr_kl <= ACC_KL_BAR and covered
    reason = speed_reason
    if not accuracy_ok:
        # Hard REJECT regardless of speed. This gate is differential (PR vs main on the same
        # token stream), so failing it means the PR CHANGED this model's output distribution --
        # exactly the class of bug a speed number cannot see. Six such bugs were found by hand
        # during Qwen3.8-27B bring-up, every one of which left throughput untouched.
        acc_reason = (f"accuracy gate failed vs main: top1={pr_top1:.4f} (bar >={ACC_TOP1_BAR}) "
                      f"kl={pr_kl:.5f} (bar <={ACC_KL_BAR})"
                      + ("" if covered else f"; the PR's score dump covers {int(pr.get('n', 0))} of "
                                            f"{int(pr['n_main'])} positions"))
        reason = f"{acc_reason} | speed: {speed_reason}"
        label = "REJECT"
        passed = False

    # No batched-prefill parity gate: main fails it on this checkpoint (see the remote script), and
    # an absolute gate that main fails would reject every PR. The differential accuracy gate above
    # and the Qwen3.6 guard below still apply.

    killed = [k for k in _GUARD_NAMES if pr.get(f"{k}_failed_box") and main.get(k)]
    if killed and accuracy_ok:
        # A guard sweep SIGKILLed on the PR build (the host OOM killer): the box's, not a regression.
        # Beside a failed accuracy gate, which a busy box cannot fake, the REJECT is posted instead.
        return {"ok": False, "pr_tip": pr.get("pr_tip"), "retry": True, "strike_key": "guard-box", "log": "",
                "reason": f"the {', '.join(_GUARD_NAMES[k] for k in killed)} guard was killed on the PR "
                          "build (exit 137) — infra"}
    # Beside that failed accuracy gate, a guard the OOM killer took measured nothing: it is reported
    # as not measured, not as a regression (the close comment used to name it as the failure).
    guards_killed = []

    def _killed_only(keys, ok, problems):
        return (not ok and any(k in killed for k in keys) and bool(problems)
                and all(p.endswith("measurement unavailable") or "PR measurement missing/zero" in p
                        for p in problems))

    q36_ok, q36_problems = check_q36_guard(pr, main)
    if _killed_only(("guard36",), q36_ok, q36_problems):
        guards_killed.append("qwen3.6")
        q36_ok, q36_problems = True, []
    if not q36_ok and all(p.endswith("measurement unavailable") for p in q36_problems) and label != "REJECT":
        # Measured nothing: infra, like the cross-model guards below -- not a regression to close on.
        return {"ok": False, "pr_tip": pr.get("pr_tip"), "retry": True, "strike_key": "guard-unmeasured", "log": "",
                "reason": "; ".join(q36_problems) + " — infra, not a regression; the PR is "
                          "re-evaluated next round rather than rejected"}
    if not q36_ok:
        # Same hard-REJECT discipline as the accuracy gate: a Qwen3.8-27B PR that silently
        # regresses Qwen3.6 via shared code (qwen35.cpp/inference_engine.cpp) is unmergeable
        # regardless of its own speed/accuracy result — see module docstring pt. 3.
        q36_reason = "qwen3.6 no-regression guard failed: " + "; ".join(q36_problems[:6])
        reason = f"{q36_reason} | {reason}"
        label = "REJECT"
        passed = False

    # ModelOpt and Muse Glimmer guards (pt. 3b): same discipline, same hard REJECT. An absent
    # checkpoint is a SKIP, reported as one, so a round that guarded nothing never reads as a pass.
    cross = {}
    cb_incomplete, cb_detail = [], []
    for key, name, checks in (("guardmo", "modelopt", (check_modelopt_guard, check_modelopt_cb_guard)),
                              ("guardmg", "muse glimmer", (check_muse_guard, check_muse_cb_guard)),
                              ("guardbn", "ternary-bonsai", (check_bonsai_guard,))):
        skipped = bool(pr.get(f"{key}_unavailable") or main.get(f"{key}_unavailable"))
        ok, problems = True, []
        cb_key = key.replace("guard", "guardcb")
        if not skipped:
            for check in checks:
                c_ok, c_problems = check(pr, main)
                if (not c_ok and check in (check_modelopt_cb_guard, check_muse_cb_guard)
                        and pr.get(f"{cb_key}_failed") and not pr.get(f"{cb_key}_failed_box") and main.get(cb_key)
                        and all(p.endswith("measurement unavailable") or "PR measurement missing/zero" in p
                                for p in c_problems)):
                    # A concurrent width the PR build could not complete (runs cut short, a crash, a
                    # hang), main having measured it: judged as a fault of the PR's run, over rounds
                    # (below), as the scored widths are -- not a regression to REJECT and close on.
                    cb_incomplete.append(name)
                    at = ", ".join(pr.get(f"{cb_key}_failed_at") or [])
                    cb_detail.append(f"{name} ({at})" if at else name)
                    c_ok, c_problems = True, []
                ok, problems = ok and c_ok, problems + c_problems
        if skipped:
            print(f">> {name} guard SKIPPED — checkpoint not installed on the box")
        if _killed_only((key, key.replace("guard", "guardcb")), ok, problems):
            guards_killed.append(name)
            ok, problems = True, []
        if not ok:
            # A guard that measured NOTHING is infra, not a regression. Take the same retry path a
            # failed bench run takes -- label null, no verdict posted, re-evaluated next round --
            # instead of closing a PR over a measurement that never happened. On 2026-09-18 one
            # round's missing Muse concurrent rows produced `muse glimmer concurrent guard
            # measurement unavailable` and auto-closed #1112 and #1114, both of which had just
            # measured +10% on the 256k axis; the guard itself ran fine by hand minutes later.
            unavailable = [p for p in problems if p.endswith("measurement unavailable")]
            if unavailable and len(unavailable) == len(problems) and label != "REJECT":
                # retry: nothing is posted. Without it apply_result still wrote eval-qwen38:REJECT.
                # (Beside a REJECT already decided, it is posted with that REJECT instead.)
                return {"ok": False, "pr_tip": pr.get("pr_tip"), "retry": True, "strike_key": "guard-unmeasured",
                        "reason": "; ".join(unavailable) + " — infra, not a regression; the PR is "
                                  "re-evaluated next round rather than rejected",
                        "log": ""}
            reason = f"{name} no-regression guard failed: " + "; ".join(problems[:6]) + f" | {reason}"
            label = "REJECT"
            passed = False
        cross[key] = (ok, problems, skipped)
    if cb_incomplete:
        why = (f"the {' and '.join(cb_detail)} concurrent-decode guard{'s' if len(cb_detail) > 1 else ''} "
               f"did not complete on the PR build while main's did")
        if label != "REJECT":
            # Retried; charged to the PR, as a failed run, after BOX_FAULT_STRIKES rounds at one commit.
            return {"ok": False, "pr_tip": pr.get("pr_tip"), "retry": True, "strike_key": "guard-cb",
                    "log": _cb_attempt_lines(r, pr), "reason": why + " — re-evaluated next round"}
        reason = f"{reason} | not measured this round: {why}"

    res = {
        "ok": True,
        "label": label,
        "pass": passed and label != "REJECT",
        "reason": reason,
        "delta_pct": delta_pct,
        "pr_decode_tps": pr["decode128_tps"],
        "main_decode_tps": main["decode128_tps"],
        "decode_delta_pct": decode_delta_pct,
        "decode_regressed": decode_label == "REJECT",
        "pr_decode256k_tps": (lc_pr.get("decode") or 0.0) or None,
        "main_decode256k_tps": (lc_main.get("decode") or 0.0) or None,
        "decode256k_delta_pct": by_dim[LONGCTX_DECODE_DIM]["delta"] if LONGCTX_DECODE_DIM in by_dim else None,
        "pr_prefill_pp": pr["prefill128_pp"],
        "main_prefill_pp": main["prefill128_pp"],
        "prefill_delta_pct": prefill_delta_pct,
        "prefill_regressed": prefill_label == "REJECT",
        "pr_prefill16k_pp": pr["prefill16k_pp"],
        "main_prefill16k_pp": main["prefill16k_pp"],
        "prefill16k_delta_pct": p16k_delta_pct,
        "prefill16k_regressed": p16k_label == "REJECT",
        "regressed_dims": [s["dim"] for s in scored if s["label"] == "REJECT"],
        **_cb_fields(pr, main, by_dim),
        # Which dimension the headline tier came from -- otherwise an XL on the comment is
        # ambiguous between a decode win and a prefill win.
        "scored_dimension": best["dim"],
        "speedup_vs_main": round(pr["decode128_tps"] / main["decode128_tps"], 3) if main.get("decode128_tps") else 0,
        "pr_top1": pr_top1,
        "pr_kl": pr_kl,
        "pr_ppl": pr.get("ppl_pr"),
        "main_ppl": pr.get("ppl_main"),
        "token_count": pr.get("token_count"),
        "accuracy_ok": accuracy_ok,
        "q36_guard_ok": q36_ok,
        "q36_guard_problems": q36_problems,
        "guards_killed": guards_killed,
        "guards_cb_incomplete": cb_incomplete,
        "q36_guard": pr.get("guard36"),
        "q36_guard_main": main.get("guard36"),
        "modelopt_guard_ok": cross["guardmo"][0],
        "modelopt_guard_problems": cross["guardmo"][1],
        "modelopt_guard_skipped": cross["guardmo"][2],
        "muse_guard_ok": cross["guardmg"][0],
        "muse_guard_problems": cross["guardmg"][1],
        "muse_guard_skipped": cross["guardmg"][2],
        "bonsai_guard_ok": cross["guardbn"][0],
        "bonsai_guard_problems": cross["guardbn"][1],
        "bonsai_guard_skipped": cross["guardbn"][2],
        "pr_head": pr.get("head"),
        "main_head": main.get("head"),
        "pr_tip": pr.get("pr_tip"),
        "merged_onto": pr.get("merged_onto"),
        "onto": main.get("sha"),       # the full main commit this verdict was measured against
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
        "pr_prefill_pp": res.get("pr_prefill_pp"),
        "main_prefill_pp": res.get("main_prefill_pp"),
        "pr_prefill16k_pp": res.get("pr_prefill16k_pp"),
        "main_prefill16k_pp": res.get("main_prefill16k_pp"),
        "scored_dimension": res.get("scored_dimension"),
        "pr_top1": res.get("pr_top1"),
        "pr_kl": res.get("pr_kl"),
        "pass": res.get("pass"),
        "accuracy_ok": res.get("accuracy_ok"),
        "q36_guard_ok": res.get("q36_guard_ok"),
        "modelopt_guard_ok": res.get("modelopt_guard_ok"),
        "muse_guard_ok": res.get("muse_guard_ok"),
        "bonsai_guard_ok": res.get("bonsai_guard_ok"),
    }
    if not res.get("ok"):
        # A failed run that reaches here is the PR's (box faults return earlier, posting nothing):
        # recorded for this commit, so it is not rebuilt, re-run and re-posted every round.
        meta["label"] = "REJECT"
    marker = (
        f"<!-- sparkinfer-qwen38-eval:{EVAL_SCHEMA_VERSION}:{commit} "
        f"{json.dumps(meta, separators=(',', ':'))} -->"
    )
    if not res.get("ok"):
        return (
            f"{marker}\n## sparkinfer qwen38 auto-eval — error\n\n"
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
    # No "main accuracy" row: this gate is differential, so main IS the reference -- there is no
    # separate absolute bar for it to miss.
    main_acc_note = ""
    killed = res.get("guards_killed") or []
    if "qwen3.6" in killed:
        q36_row = "| qwen3.6 guard | ⚠️ NOT MEASURED — its sweep was killed on the PR build (exit 137, the host OOM killer); the REJECT is the accuracy gate's |\n"
    elif res.get("q36_guard_ok"):
        q36_row = "| qwen3.6 guard | ✅ no regression (decode+prefill, ctx 0/512/4k/16k/32k) |\n"
    else:
        problems = "; ".join((res.get("q36_guard_problems") or [])[:4])
        q36_row = (f"| qwen3.6 guard | ❌ **FAILED** — {problems} — "
                    "**verdict forced to REJECT regardless of speed/accuracy** |\n")
    cross_rows = ""
    cb_note = f", concurrent decode @ {'/'.join(f'c{c}' for c in CB_GUARD_CONCS)}"
    bn_ctxs = "/".join("32k" if c == 32768 else str(c) for c in BONSAI_GUARD_CTXS)
    # A concurrent-decode guard the PR build could not complete is posted only beside another REJECT.
    cb_incomplete = res.get("guards_cb_incomplete") or []
    cb_missing = " · concurrent decode ⚠️ NOT MEASURED — the PR build did not complete it (main did)"
    for prefix, name, what, at in (
            ("modelopt", "modelopt guard", "Qwen3.8-27B NVFP4 (ModelOpt)", f"decode+prefill @ 32k{cb_note}"),
            ("muse", "muse glimmer guard", "Muse Glimmer 30B", f"decode+prefill @ 32k{cb_note}"),
            ("bonsai", "ternary-bonsai guard", "Ternary-Bonsai-2-27B", f"decode+prefill @ {bn_ctxs}")):
        short = name.replace(" guard", "")
        if res.get(f"{prefix}_guard_skipped"):
            # Say SKIPPED explicitly: a guard that reports nothing reads the same as one that passed.
            cross_rows += (f"| {name} | ⚠️ SKIPPED — checkpoint not installed on the box; "
                           f"shared-code regressions on {what} were NOT checked |\n")
        elif short in killed:
            cross_rows += f"| {name} | ⚠️ NOT MEASURED — a run of it was killed on the PR build (exit 137, the host OOM killer); the REJECT is the accuracy gate's |\n"
        elif res.get(f"{prefix}_guard_ok") and short in cb_incomplete:
            cross_rows += (f"| {name} | decode+prefill @ 32k ✅ no regression ({what}){cb_missing}; "
                           "the REJECT is another gate's |\n")
        elif res.get(f"{prefix}_guard_ok"):
            cross_rows += f"| {name} | ✅ no regression ({at}, {what}) |\n"
        else:
            probs = "; ".join((res.get(f"{prefix}_guard_problems") or [])[:4])
            cross_rows += (f"| {name} | ❌ **FAILED** — {probs} — "
                           "**verdict forced to REJECT regardless of speed/accuracy**"
                           f"{cb_missing if short in cb_incomplete else ''} |\n")
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
    # Long-context decode is only reported when both arms measured it (#1113).
    longctx_row = ""
    if res.get("pr_decode256k_tps") and res.get("main_decode256k_tps"):
        longctx_row = (f"| decode@256k (ModelOpt NVFP4) | PR {res['pr_decode256k_tps']:.2f} / "
                       f"main {res['main_decode256k_tps']:.2f} tok/s "
                       f"({res.get('decode256k_delta_pct', 0):+.1f}%) |\n")
    return (
        f"{marker}\n## sparkinfer qwen38 auto-eval — `eval-qwen38:{lab}`\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| **label** | `eval-qwen38:{lab}` |\n"
        f"| scored at | best of prefill@16k, concurrent decode @c2/c4/c8/c16/c32 and ModelOpt decode@256k; decode@128, prefill@128, prefill@256k and concurrent decode @c1 are no-regression floors |\n"
        f"| tier came from | `{res.get('scored_dimension', '?')}` |\n"
        f"| PR decode tok/s | {res['pr_decode_tps']:.2f} |\n"
        f"| main decode tok/s | {res['main_decode_tps']:.2f} |\n"
        f"| decode speedup vs main | **{res.get('speedup_vs_main', 0):.2f}×** ({res.get('decode_delta_pct', 0):+.1f}%) |\n"
        f"| PR prefill@128 pp | {res['pr_prefill_pp']:.2f} |\n"
        f"| main prefill@128 pp | {res['main_prefill_pp']:.2f} |\n"
        f"| prefill@128 vs main | {res.get('prefill_delta_pct', 0):+.1f}% |\n"
        f"| PR prefill@16k pp | {res['pr_prefill16k_pp']:.2f} |\n"
        f"| main prefill@16k pp | {res['main_prefill16k_pp']:.2f} |\n"
        f"| prefill@16k vs main | {res.get('prefill16k_delta_pct', 0):+.1f}% |\n"
        f"{longctx_row}"

        f"{acc_row}"
        f"{main_acc_note}"
        f"{q36_row}"
        f"{cross_rows}"
        f"| PPL PR / main | {res.get('pr_ppl') or '?'} / {res.get('main_ppl') or '?'} |\n"
        f"{polaris_row}"
        f"| commit | `{commit[:9]}`"
        + (f", measured merged onto `main` `{res['merged_onto']}` (this round's baseline)"
           if res.get("merged_onto") else "")
        + " |\n\n"
        f"{_cb_table(res)}"
        f"{res.get('reason') or ''}\n\n"
        "<sub>Scored on the pinned RTX 5090 against the same-box `origin/main`, on the upstream "
        "`unsloth/Qwen3.8-27B-NVFP4` checkpoint, with the measuring harness taken from `main`. "
        "Every measured dimension is also a no-regression floor; otherwise the label is the best "
        "tier among prefill@16k and concurrent decode @c2–c32. Accuracy is differential: this "
        "build and `main` score the same token stream and must agree. Also gated on no-regression "
        "guards for Qwen3.6 (decode+prefill, ctx 0/512/4k/16k/32k) and for the ModelOpt Qwen3.8 "
        "checkpoint and Muse Glimmer (decode+prefill @ 32k, concurrent decode @ c16/c32) and "
        "Ternary-Bonsai-2-27B (decode+prefill @ 128 and 32k), because Qwen3.8 PRs can touch code "
        "shared with other models. A `none` label means no measurable speedup on these axes, "
        "which is expected if that is not what your change is about.</sub>\n"
    )


def auto_merge_ok_qwen38(repo, num, require_merge_first=True, ranking_loss_ok=False):
    """Can this PR be merged now? With require_merge_first=False: may it be MADE merge-first -- the
    same test minus that label, so a winner whose merge would be refused cannot hold merge-first
    while every other speedup PR is pushed to needs-rebase (pr_bonsai_bot.py, 2026-09-26).
    ranking_loss_ok: the bot's own qwen38-needs-rebase does not count -- reconcile's call for a PR
    sent there only for losing an earlier ranking (_waits_for_the_winner)."""
    try:
        info = json.loads(arb.gh([
            "pr", "view", str(num), "-R", repo, "--json",
            "state,isDraft,labels,author,mergeable,files,changedFiles,headRefOid,baseRefName,comments",
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
        return False, "no verified eval-qwen38:speedup label"
    if require_merge_first and QWEN38_MERGE_FIRST not in labs:
        return False, "not qwen38-merge-first"
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
    # ... and a REJECT another bot measured for this very commit whose label is gone (arb.foreign_rejects).
    rejected = arb.foreign_rejects(info.get("comments"), info.get("headRefOid") or "", "qwen38")
    if rejected:
        return False, f"{', '.join(rejected)} measured this commit REJECT"
    blocked = labs & (AUTOMERGE_BLOCK - ({QWEN38_NEEDS_REBASE} if ranking_loss_ok else set()))
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


def try_auto_merge_qwen38(repo, num):
    ok, reason = auto_merge_ok_qwen38(repo, num)
    if not ok:
        print(f">> qwen38 auto-merge SKIP #{num}: {reason}")
        return False
    # Pinned to the SCORED commit, which auto_merge_ok_qwen38 just found equal to the head, --admin
    # included. A second head lookup here would pin whatever a push had made the head in between.
    head = (_load_scores().get(str(num)) or {}).get("commit") or ""
    if not arb._FULL_SHA_RE.match(head):
        print(f">> qwen38 auto-merge SKIP #{num}: no scored commit to pin the merge to")
        return False
    args = ["pr", "merge", str(num), "-R", repo, "--squash", "--match-head-commit", head]
    r = arb.gh(args)
    if r.returncode != 0 and os.environ.get("SPARKINFER_AUTOMERGE_ADMIN", "1") == "1":
        err = ((r.stderr or "") + (r.stdout or "")).lower()
        if "not mergeable" in err or "branch policy" in err or "required" in err or "prohibited" in err:
            print(">> qwen38 auto-merge: branch policy blocked — retrying with --admin")
            r = arb.gh(args + ["--admin"])
    if r.returncode == 0:
        print(f">> QWEN38 AUTO-MERGED #{num} (qwen38-merge-first)")
        arb.gh(["pr", "comment", str(num), "-R", repo, "--body",
                "<!-- sparkinfer-qwen38-automerge -->\n"
                "Auto-merged as the round's `qwen38-merge-first` winner — verified same-box "
                "speedup over `main` on the unsloth checkpoint, with every floor, the differential "
                "accuracy gate and the Qwen3.6, ModelOpt, Muse Glimmer and Ternary-Bonsai guards passing."])
        return True
    print(f">> qwen38 auto-merge BLOCKED #{num}: {(r.stderr or r.stdout or '')[:200]}")
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
    skip = arb.model_skip_reason(pr.get("body") or "", "qwen38")
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


def reconcile_qwen38_merge_labels(repo, dry_run=False):
    scores = _load_scores()
    open_prs = arb.open_prs_or_none(repo, "number,labels,isDraft,body,files,mergeable,headRefOid,baseRefName")
    if open_prs is None:
        print(">> qwen38 round: GitHub did not return the open PRs — labels left as they are")
        return
    open_labels = {p["number"]: {l["name"] for l in p["labels"]} for p in open_prs}
    open_by_num = {p["number"]: p for p in open_prs}

    merged = json.loads(arb.gh([
        "pr", "list", "-R", repo, "--state", "merged", "--label", QWEN38_MERGE_FIRST,
        "--json", "number", "--limit", "10",
    ]).stdout or "[]")
    for m in merged:
        if not dry_run:
            arb.remove_label(repo, m["number"], QWEN38_MERGE_FIRST)

    scored = []
    stale_first = []   # carries merge-first but can no longer win it
    stale_main = set()   # in the running, but its merge waits for a re-measure onto today's main
    main_now = None      # read once, for a needs-rebase that may only mean a lost ranking
    for num, labs in open_labels.items():
        if not dry_run:
            labs = open_labels[num] = arb.repair_own_tier(
                repo, num, labs, EVAL_PREFIX, scores.get(str(num)), (open_by_num[num].get("headRefOid") or "")[:40],
                lambda: _verdict_heads(repo, num))
        # A sync GitHub did not answer when this bot posted its verdict, healed -- on this bot's PRs
        # only: the retired AR bot's labels derive the generic one by another rule (the failing side).
        if not dry_run and any(l.startswith(EVAL_PREFIX) for l in labs) and arb.generic_label_out_of_sync(labs):
            arb.sync_generic_eval_label(repo, num)
        # A PR that cannot be merged -- hold, needs-rebase, penalty, any other AUTOMERGE_BLOCK
        # label, or anything else auto-merge would refuse -- must not take merge-first and push the
        # others to needs-rebase for a merge that never happens (pr_bonsai_bot.py, #1154).
        lost_only = False
        if (labs & AUTOMERGE_BLOCK) == {QWEN38_NEEDS_REBASE}:
            # Sent to needs-rebase only for losing an earlier ranking, with its verdict still standing
            # on today's main: it stays in the running. Left out, a worse PR merged first once the
            # winner was re-measured lower, and nothing merged at all once the winner was closed or
            # held. After main moves, the rebase is its author's (CONTRIBUTING).
            if main_now is None:
                main_now = arb.current_main_sha(repo) or ""
            pr = open_by_num[num]
            lost_only = bool(main_now) and _waits_for_the_winner(pr, labs, (pr.get("headRefOid") or "")[:40], main_now)
        if labs & AUTOMERGE_BLOCK and not lost_only:
            if QWEN38_MERGE_FIRST in labs:
                stale_first.append(num)
            continue
        tiers = {l.split(":", 1)[1] for l in labs if l.startswith(EVAL_PREFIX)}
        tier = next((t for t in tiers if t in SPEEDUP_LABELS), None)
        if not tier:
            # No speedup tier (any more): its head moved, or a re-measure found none. A merge-first
            # left here exempted it from every close and could sit beside the next winner's.
            if QWEN38_MERGE_FIRST in labs:
                stale_first.append(num)
            continue
        ok, why = auto_merge_ok_qwen38(repo, num, require_merge_first=False,
                                       ranking_loss_ok=lost_only)
        if not ok and why == arb.PR_UNREADABLE:
            # Not an answer: demoting on it would take merge-first from the real holder.
            print(f">> qwen38 round: GitHub did not return #{num} — labels left as they are")
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
                print(f">> qwen38 round: #{num} stays in the running, merge waits ({why})")
                stale_main.add(num)
        if not ok and num not in stale_main:
            print(f">> qwen38 round: #{num} cannot be merge-first ({why})")
            if QWEN38_MERGE_FIRST in labs:
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
            arb.remove_label(repo, num, QWEN38_MERGE_FIRST)
    if not scored:
        print(">> qwen38 round: no verified speedup PRs")
        return
    winner = scored[0][0]
    print(f">> qwen38 round: merge-first #{winner}; rebase {[n for n,_,_ in scored[1:]] or 'none'}")
    if dry_run:
        return
    arb.add_label(repo, winner, QWEN38_MERGE_FIRST)
    arb.remove_label(repo, winner, QWEN38_NEEDS_REBASE)
    for num, _, _ in scored[1:]:
        # Nothing merges this round while the winner waits for its re-measure: no one needs a rebase.
        if winner not in stale_main:
            arb.add_label(repo, num, QWEN38_NEEDS_REBASE)
        arb.remove_label(repo, num, QWEN38_MERGE_FIRST)
    if AUTO_MERGE:
        try_auto_merge_qwen38(repo, winner)


def upload_qwen38_eval_log(repo, num, title, oid, res):
    """Commit the eval result (+ Polaris receipt/attestation) to sparkinfer-log, mirroring
    pr_dflash_bot.py's upload_dflash_eval_log with a qwen38-prefixed run id."""
    try:
        arb._ensure_log_repo()
        rid = arb.eval_log_run_id(f"qwen38-{int(num):04d}-{oid[:7]}", res.get("onto"))
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
            "eval_mode": "qwen38-128",
            "label": res.get("label"), "pass": res.get("pass"), "reason": res.get("reason"),
            "delta_pct": res.get("delta_pct"),
            "pr_decode_tps": res.get("pr_decode_tps"), "main_decode_tps": res.get("main_decode_tps"),
            "pr_prefill_pp": res.get("pr_prefill_pp"), "main_prefill_pp": res.get("main_prefill_pp"),
            "prefill_delta_pct": res.get("prefill_delta_pct"),
            "pr_prefill16k_pp": res.get("pr_prefill16k_pp"),
            "main_prefill16k_pp": res.get("main_prefill16k_pp"),
            "prefill16k_delta_pct": res.get("prefill16k_delta_pct"),
            "scored_dimension": res.get("scored_dimension"),
            "regressed_dims": res.get("regressed_dims"),
            **{k: v for k, v in res.items()
               if k.startswith(("pr_cb", "main_cb")) or (k.startswith("cb") and k.endswith("_delta_pct"))},
            "speedup_vs_main": res.get("speedup_vs_main"),
            "pr_top1": res.get("pr_top1"), "pr_kl": res.get("pr_kl"),
            "accuracy_ok": res.get("accuracy_ok"),
            "q36_guard_ok": res.get("q36_guard_ok"), "q36_guard_problems": res.get("q36_guard_problems"),
            "modelopt_guard_ok": res.get("modelopt_guard_ok"), "modelopt_guard_problems": res.get("modelopt_guard_problems"),
            "modelopt_guard_skipped": res.get("modelopt_guard_skipped"),
            "muse_guard_ok": res.get("muse_guard_ok"), "muse_guard_problems": res.get("muse_guard_problems"),
            "muse_guard_skipped": res.get("muse_guard_skipped"),
            "bonsai_guard_ok": res.get("bonsai_guard_ok"), "bonsai_guard_problems": res.get("bonsai_guard_problems"),
            "bonsai_guard_skipped": res.get("bonsai_guard_skipped"),
            "guards_cb_incomplete": res.get("guards_cb_incomplete") or [],
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
                     "delta_pct": res.get("delta_pct"), "eval_mode": "qwen38-128", "date": result["date"]}
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
            print(">> qwen38 eval-log upload skipped: nothing to commit")
            return None
        push = subprocess.run(["git", "-C", arb.LOG_DIR, "push", "-q"], check=False, timeout=300)
        if push.returncode != 0:
            print(f">> qwen38 eval-log push failed (rc={push.returncode})")
            return None
        url = arb.LOG_PAGE + rid
        print(f">> qwen38 eval log: {url}")
        return url
    except Exception as e:
        print(f">> qwen38 eval-log upload failed: {e}")
        return None


def _remeasure_state(repo, num, head, labs, main_now):
    """Is an already-evaluated head owed a re-measure onto today's main? True only when auto_merge_ok_qwen38
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
    ok, why = auto_merge_ok_qwen38(repo, num, require_merge_first=False)
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
        print(f"PR #{num}: {res.get('reason')} — qwen38-needs-rebase, no verdict")
        if not dry_run:
            arb.add_label(repo, num, QWEN38_NEEDS_REBASE)
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
            print(f"PR #{num}: qwen38 eval deferred — {res.get('reason')} "
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
        res = dict(res, retry=False, reason=arb.charged_reason(res.get("reason"), n))
    if not dry_run:
        arb.clear_strikes(STRIKES_FILE, num)
    body = format_comment(commit, res)
    label = res.get("label") if res.get("ok") else "REJECT"
    if not res.get("ok"):
        label = "REJECT"
    print(f"PR #{num}: eval-qwen38:{label}  "
          f"decode PR={res.get('pr_decode_tps')} main={res.get('main_decode_tps')}  "
          f"prefill PR={res.get('pr_prefill_pp')} main={res.get('main_prefill_pp')}  "
          f"from={res.get('scored_dimension')}  "
          f"top1={res.get('pr_top1')} kl={res.get('pr_kl')}  "
          f"delta={res.get('delta_pct')}%  accuracy_ok={res.get('accuracy_ok')}  "
          f"q36_guard_ok={res.get('q36_guard_ok')}  modelopt_guard_ok={res.get('modelopt_guard_ok')}  "
          f"muse_guard_ok={res.get('muse_guard_ok')}  bonsai_guard_ok={res.get('bonsai_guard_ok')}")
    if dry_run:
        print(body[:500])
        return
    strip_qwen38_eval_labels(repo, num)
    if label in SPEEDUP_LABELS:
        # A fresh, valid speedup score for the CURRENT head commit means this PR is caught up
        # with main and deserves a fair shot at winning the next merge-first reconciliation --
        # clear any stale needs-rebase from a round it lost (or an old conflict that's since been
        # resolved). Found 2026-08-13: reconcile_qwen38_merge_labels() filters candidates on
        # `QWEN38_NEEDS_REBASE not in labs` (this bot's own "who's eligible to win" gate) but
        # the ONLY place that ever removed the label was the winner-selection branch itself --
        # a PR that lost one round, or ever hit a transient merge conflict, could never be
        # reconsidered again even after a completely clean re-evaluation confirmed its score,
        # since it was filtered out of candidacy before scoring was ever compared. Hit #790 and
        # #791 both losing merge-first to a strictly worse score for exactly this reason.
        arb.remove_label(repo, num, QWEN38_NEEDS_REBASE)
    arb.add_label(repo, num, f"{EVAL_PREFIX}{label}")
    # Mirrored to the generic `eval:*` label, same as pr_dflash_bot.py -- SN74 scoring reads
    # eval:* tiers, so this makes Qwen3.8-27B submissions count toward that live incentive
    # mechanism. Explicit user decision, 2026-08-11 (originally deliberately NOT mirrored, given
    # Qwen3.8-27B's youth at the time -- see git history on this line for that reasoning).
    # Derived from every per-bot `eval-<model>:<tier>` label rather than overwritten with this
    # bot's own verdict -- this bot only measures Qwen3.8-27B, so writing the generic label
    # directly let a `none` here erase another model's real tier depending purely on which
    # staggered cron ran last. See arb.sync_generic_eval_label().
    arb.sync_generic_eval_label(repo, num)
    # Whether the verdict posted: a close carries it instead when it did not (below).
    posted = getattr(arb.gh(["pr", "comment", str(num), "-R", repo, "--body", arb.fit_comment(body)]),
                     "returncode", 0) == 0
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
            "pr_top1": res.get("pr_top1"),
            "pr_kl": res.get("pr_kl"),
            "pass": res.get("pass"),
            "accuracy_ok": res.get("accuracy_ok"),
            "q36_guard_ok": res.get("q36_guard_ok"),
            "modelopt_guard_ok": res.get("modelopt_guard_ok"),
            "muse_guard_ok": res.get("muse_guard_ok"),
            "bonsai_guard_ok": res.get("bonsai_guard_ok"),
            "onto": res.get("onto"),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _save_scores(scores)
    else:
        # A failed run, or no delta: recorded all the same (arb.record_posted_verdict).
        arb.record_posted_verdict(_load_scores, _save_scores, num, commit, label, res)
    if res.get("ok"):
        upload_qwen38_eval_log(repo, num, title, commit, res)
    if res.get("ok") and res.get("delta_pct") is not None:
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
        # `none` closes only a PR declared for this model alone that no other bot scored a speedup
        # or made merge-first (arb.none_may_close); a REJECT is evidence of harm and closes.
        if label == "REJECT" or (label == "none" and arb.none_may_close(
                pr_body, arb.labels_on_or_none(repo, num), "qwen38", EVAL_PREFIX)):
            if not res.get("q36_guard_ok", True):
                fail_clause = "and regressed the Qwen3.6 no-regression guard (decode/prefill on shared code)"
            elif not res.get("modelopt_guard_ok", True):
                fail_clause = "and regressed the ModelOpt Qwen3.8 no-regression guard (decode/prefill @ 32k or concurrent decode)"
            elif not res.get("muse_guard_ok", True):
                fail_clause = "and regressed the Muse Glimmer no-regression guard (decode/prefill @ 32k or concurrent decode)"
            elif not res.get("bonsai_guard_ok", True):
                fail_clause = "and regressed the Ternary-Bonsai-2-27B no-regression guard (decode/prefill @ 128 or 32k)"
            elif not res.get("accuracy_ok"):
                fail_clause = "and failed the accuracy gate"
            elif res.get("regressed_dims"):
                fail_clause = f"({', '.join(res['regressed_dims'])} regression)"
            else:
                fail_clause = "(regression)"
            # A `none` is an absence of evidence, a REJECT is evidence of harm; saying the same
            # thing for both reads a close as an accusation (#768). Same split as pr_dspark_bot.py.
            if label == "none":
                close_body = (
                    "<!-- sparkinfer-qwen38-auto-close -->\n"
                    "## Closed: no verified speedup — `eval-qwen38:none`\n\n"
                    "Measured on the pinned RTX 5090 against the same-box `origin/main`, on the "
                    f"unsloth checkpoint: **{res.get('delta_pct')}%** on the best scored axis "
                    "(prefill@16k and concurrent decode @c2–c32).\n\n"
                    "**This is not a finding that anything is wrong with your PR.** Nothing "
                    "regressed and every correctness gate passed — the change just did not move a "
                    "number this bot measures.\n\n"
                    "- **Targeting a different model?** Tick it under **Target model(s)** in the "
                    "PR template; a PR declared for another model is skipped rather than scored.\n"
                    "- **Nothing here measures your optimization yet?** Open an issue describing "
                    "the axis you need, with your before/after numbers, then reopen and ask for "
                    "the `hold` label.\n"
                    "- **Correctness fix, refactor, test or docs?** Reopen as a **draft** or ask "
                    "for `hold`; those are reviewed by hand.\n"
                    "- **Expected a speedup?** Push the change as a new commit and reopen; the new "
                    "commit is evaluated on the next round (reopening alone does not re-run a commit "
                    "that already has its verdict)."
                )
            else:
                close_body = (
                    "<!-- sparkinfer-qwen38-auto-close -->\n"
                    f"## Closed: regression or failed gate — `eval-qwen38:{label}`\n\n"
                    f"Measured **{res.get('delta_pct')}%** vs the same-box `origin/main`, "
                    f"{fail_clause} — closing automatically.\n\n"
                    "Every measured axis is also a no-regression floor, and accuracy and the "
                    "Qwen3.6, ModelOpt, Muse Glimmer and Ternary-Bonsai guards are hard gates, so one failure closes the PR whatever it was "
                    "aiming at. The verdict comment above names which one. Push a fix and reopen: "
                    "the new commit is evaluated on the next poll (reopening alone does not re-run a "
                    "commit that already has its verdict)."
                )
            # Not over a commit the author has already replaced (a push while it was measured),
            # nor over a `hold` or a draft made while the round ran.
            if not posted:
                # The verdict travels with the close, marker included. Left open instead, its REJECT label
                # without a marker read to the other bots as a tier no verdict backs: they dropped it, and
                # one of them merged the PR.
                close_body = body + "\n\n---\n\n" + close_body
            why = arb.verdict_close_blocker(repo, num, commit)
            if why:
                print(f">> PR #{num}: not closed — {why}")
                return
            arb.gh(["pr", "comment", str(num), "-R", repo, "--body", arb.fit_comment(close_body)])
            arb.gh(["pr", "close", str(num), "-R", repo])
            print(f">> auto-closed PR #{num} (eval-qwen38:{label})")


def _exit_if_gave_up():
    """Exit 3 when the selection gave up on a PR after the bot's own errors (GAVE_UP), so the
    wrapper's failed-run banner shows it -- on every way out of a run, one that then measured
    nothing (the GPU down, the lock busy) included."""
    if GAVE_UP:
        print(f"!! qwen38: gave up on {', '.join(f'#{n}' for n in sorted(GAVE_UP))} after the bot's own "
              f"errors — see above; exiting 3 so the wrapper's failed-run banner shows it")
        sys.exit(3)


def main():
    ap = argparse.ArgumentParser(description="Qwen3.8-27B (unsloth checkpoint) PR eval bot")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--repo", default="gittensor-ai-lab/sparkinfer")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reeval", action="store_true")
    ap.add_argument("--labels-only", action="store_true",
                    help="reconcile qwen38-merge-first only — no GPU")
    ap.add_argument("--only-prs", default="",
                    help="comma-separated PR numbers (bypass greenlight)")
    args = ap.parse_args()

    only = {int(x) for x in args.only_prs.split(",") if x.strip().isdigit()}

    print(f">> qwen38 eval transport: "
          f"{'ssh' if ssh_box_enabled() else f'vast.ai (instance {arb.current_instance(args.instance) or args.instance})'}")
    print(f">> AUTOMERGE={int(AUTO_MERGE)}")

    ok, login = arb.acting_account_ok()
    if not ok:
        print(f"!! gh acts as {login}, not SPARKINFER_BOT_LOGIN={os.environ.get('SPARKINFER_BOT_LOGIN')} — nothing done")
        sys.exit(3)
    if args.labels_only:
        reconcile_qwen38_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — qwen38 labels only (no GPU).")
        return

    prs = arb.open_prs_or_none(args.repo, "number,title,labels,isDraft,headRefOid,headRefName,baseRefName,"
                                          "mergeable,author,body,files,changedFiles")
    if prs is None:
        # Not an empty queue: GitHub did not answer (an expired token, an outage). Exit non-zero so the
        # wrapper's failed-run banner shows a bot that has stopped seeing PRs.
        print("!! GitHub did not return the open PRs — nothing done this run")
        sys.exit(3)
    prs.sort(key=lambda p: p["number"])

    stale_closed = close_stale_qwen38_prs(args.repo, prs, dry_run=args.dry_run) if not only else set()
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
                and any(l.startswith(EVAL_PREFIX) or l == QWEN38_NEEDS_REBASE for l in labs0)
                and arb.strip_stale_verdict_labels(args.repo, num, labs0, EVAL_PREFIX, head0,
                                                   _verdict_heads(args.repo, num), QWEN38_NEEDS_REBASE,
                                                   arb.pr_merge_conflict(pr.get("mergeable"))
                or bool(arb.strike_count(STRIKES_FILE, num, (pr.get("headRefOid") or "")[:40], "conflict")))):
            print(f"PR #{num} @ {head0[:9]}: no qwen38 verdict for this head yet — dropped the old eval-qwen38 label")
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
        evaluated = _verdict_heads(args.repo, num)
        if evaluated is None:
            # Not "no verdict yet": re-measuring, or dropping its labels, on a failed read is wrong.
            print(f"PR #{num} @ {short}: GitHub did not return its comments — skipped this round")
            continue
        # An eval-qwen38 tier measured on an older head no longer describes this PR.
        if not args.dry_run and arb.strip_stale_verdict_labels(
                args.repo, num, labs, EVAL_PREFIX, head, evaluated, QWEN38_NEEDS_REBASE,
                arb.pr_merge_conflict(pr.get("mergeable"))
                or bool(arb.strike_count(STRIKES_FILE, num, (pr.get("headRefOid") or "")[:40], "conflict"))):
            print(f"PR #{num} @ {short}: no qwen38 verdict for this head yet — dropped the old eval-qwen38 label")
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
            print(f"PR #{num} @ {short}: does not merge onto main on the box — qwen38-needs-rebase until a push")
            continue
        if not args.reeval and head and head in evaluated:
            if not _remeasure_against_new_main(args.repo, num, head, labs, main_now):
                print(f"PR #{num} @ {short}: already qwen38-evaluated — skip")
                continue
            remeasure = True
        # Did the author declare a DIFFERENT target model (#1027)? A Muse Glimmer change cannot
        # move these axes, and proving that costs a full round. arb.model_skip_reason() fails
        # open: an absent or ambiguous declaration evaluates. Safe to skip because the Muse bot
        # runs ModelOpt (Qwen3.8) and Qwen3.6 guards on every PR it scores.
        skip_why = arb.model_skip_reason(pr.get("body") or "", "qwen38")
        if skip_why:
            print(f"PR #{num}: {skip_why} — skip qwen38 eval")
            continue
        # Does it edit the measuring instrument (HARNESS_PATHS)? Checked before any GPU time: a
        # number measured with a changed ruler cannot be accepted either way.
        touched = [f.get("path", "") for f in (pr.get("files") or [])]
        harness_hits = [t for t in touched if any(t.startswith(h) for h in HARNESS_PATHS)]
        if harness_hits:
            print(f"PR #{num}: touches the eval harness ({', '.join(harness_hits[:3])}) — not evaluated")
            continue
        if arb.pr_merge_conflict(pr.get("mergeable")):
            print(f"PR #{num}: merge conflict — qwen38-needs-rebase")
            if not args.dry_run:
                arb.add_label(args.repo, num, QWEN38_NEEDS_REBASE)
            continue

        if not only:
            status, why = arb.greenlight_status(args.repo, num, labs)
            if status != "ok":
                print(f"PR #{num}: not greenlit ({why}) — skip qwen38 eval")
                continue
            print(f"PR #{num}: greenlit ({why})")
        else:
            print(f"PR #{num}: --only-prs targeted")

        # Evaluate the PR MERGED INTO main, not its branch tip. The harness is pinned from main
        # (so a PR cannot rewrite the thing that measures it), and a branch that predates a struct
        # change on main therefore builds main's harness against its own older headers: on
        # 2026-09-18 every PR branched before 8d55f3f ("window_tokens" in KVCacheConfig) failed with
        # BUILD_FAILED and scored eval:REJECT with no measurements at all -- #1109 among them, one
        # commit behind. The merge is built on the box, onto the exact main commit the baseline
        # measured (_remote_script `onto`), not taken from GitHub's pull/<n>/merge, which can be
        # built on an older main (#1145, 2026-09-24).
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
        reconcile_qwen38_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — no qwen38 PRs to evaluate.")
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
        reconcile_qwen38_merge_labels(args.repo, dry_run=False)
        print("done — qwen38 labels only (GPU down).")
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
        reconcile_qwen38_merge_labels(args.repo, dry_run=False)
        print("done — qwen38 round skipped (main baseline unusable).")
        # Non-zero: a main that stays unusable stops this bot measuring anything, the PR fixing it
        # included -- run_bot (cron_common.sh) makes a run of these loud.
        sys.exit(3)
    # main has no top1/kl of its own: it IS the accuracy reference, and its score dump was just
    # written to SCORE_DUMP_MAIN for each PR in this round to diff against.
    print(f">> main baseline: decode@128={main_result['decode128_tps']:.2f} tok/s "
          f"prefill@128={main_result['prefill128_pp']:.2f} pp "
          f"prefill@16k={main_result['prefill16k_pp']:.2f} pp cb {_cb_summary(main_result)}")
    print(f">> main guard coverage: {_guard_coverage(main_result)}")

    for num, head, short, ref, title, pr_body in pending:
        print(f"PR #{num} @ {short}: evaluating Qwen3.8-27B '{ref}' …")
        try:
            res = eval_qwen38_on_box(host, port, ref, main_result)
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

    reconcile_qwen38_merge_labels(args.repo, dry_run=False)
    print("done — qwen38 eval pass complete.")
    _exit_if_gave_up()


if __name__ == "__main__":
    main()
