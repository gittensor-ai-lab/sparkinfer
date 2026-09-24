#!/usr/bin/env python3
"""sparkinfer Qwen3.8-27B PR auto-evaluator.

Sibling of pr_museglimmer_bot.py / pr_dflash_bot.py, and the ONLY scored bot once Qwen3.8-27B
becomes the eval scope (see eval/README.md). Narrowly scoped on purpose:

  1. Speed — decode @ ctx=128 on the NVFP4 checkpoint, PR vs a freshly-measured origin/main, same
              box, same round, median of 5 reps. Same tier buckets as every other bot in this
              directory (BUCKETS/SIG/REGRESS_TOL below — copied, not reinvented).

              The scored checkpoint is the HuggingFace compressed-tensors DIRECTORY (mixed NVFP4
              FFN + FP8 attention/GDN projections, unsloth/Qwen3.8-27B-NVFP4), NOT a GGUF of the
              same model. That is deliberate: it is what sparkinfer-server actually serves, so it
              is what a PR's speed claim should be measured against. qwen3_gguf_bench and
              qwen3_gguf_score grew directory support for exactly this
              (runtime/examples/qwen_checkpoint.h, shared with the server so the two cannot
              disagree about how a checkpoint is configured).

              Prefill@128 is ALSO scored, from the same sweep and the same model load (no extra
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
              concurrent decode at c16/c32 on each, with the same hard REJECT. The 32k guards are
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
eval:* tiers). Auto-close on none/REJECT is live; auto-merge stays OFF unless
SPARKINFER_QWEN38_AUTOMERGE=1 is explicitly set.

  python eval/pr_qwen38_bot.py --instance 46074104
  python eval/pr_qwen38_bot.py --only-prs 636 --reeval

Never rents a GPU. Shares the pinned box with any other bot via flock in the cron wrapper
(run_qwen38_cron.sh) — all bots MUST share /tmp/sparkinfer_bot.lock.
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

# Auto-merge is wired (mirrors pr_dflash_bot.py's auto_merge_ok_dflash/try_auto_merge_dflash
# shape) but OFF unless this exact env var is set — NOT set in .env.eval, so it stays fully
# inert until a human deliberately flips it on. Single-line change to enable later.
AUTO_MERGE = os.environ.get("SPARKINFER_QWEN38_AUTOMERGE") == "1"
AUTOMERGE_BLOCK = {
    "copycat", "copycat-warn", "flagged:gaming", "penalty", "needs-benchmark",
    QWEN38_NEEDS_REBASE, arb.REEVALUATE_LABEL, arb.HOLD_LABEL, *arb.REGRESSION_LABELS,
}

SCORES_FILE = os.path.expanduser(
    os.environ.get("QWEN38_SCORES_FILE", "~/.sparkinfer_qwen38_scores.json")
)

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
    try:
        with open(SCORES_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f">> qwen38 scores save skipped: {e}")


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
    """ModelOpt Qwen3.8-27B NVFP4 no-regression guard (pt. 3b), decode + prefill @ 32k."""
    return _check_model_guard(pr, main, "guardmo", "modelopt", tol)


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
    r = arb.gh(["pr", "view", str(num), "-R", repo, "--json", "comments"])
    done = set()
    for c in json.loads(r.stdout or "{}").get("comments", []):
        body = c.get("body") or ""
        m = MARKER_RE.search(body)
        if not m or "sparkinfer qwen38 auto-eval" not in body:
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


STALE_DAYS = float(os.environ.get("QWEN38_STALE_DAYS", "1"))


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
        if arb.HOLD_LABEL in labs or QWEN38_MERGE_FIRST in labs:
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


_EXPLICIT_FAIL_MARKERS = ("BUILD_FAILED", "LLAMACPP_CONFIGURE_FAILED", "LLAMACPP_BUILD_FAILED",
                          "MERGE_CONFLICT")


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
        print(f">> {label}: looks like a hard kill (no ERR-trap diagnostic, no accuracy-stage "
              f"checkpoint reached) — retrying once")
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
        checkout = arb.merged_checkout_script(ref, base)
    else:
        base = "origin/main"
        checkout = (f"git fetch -q origin {shlex.quote(ref)}\n"
                    "git reset -q --hard\n"
                    "git clean -qfd\n"
                    "git checkout -qf FETCH_HEAD\n"
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
# change is measured with main's ruler -- the same ruler as its baseline.
git fetch -q origin main
git checkout -q {base_q} -- runtime/examples/qwen3_gguf_bench.cpp \
  runtime/examples/qwen3_gguf_cb_bench.cpp runtime/examples/qwen_checkpoint.h \
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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/q38_cmake.log 2>&1
cmake --build build --target qwen3_gguf_bench qwen3_gguf_score qwen3_gguf_generate qwen3_gguf_cb_bench -j"$(nproc)" >/tmp/q38_build.log 2>&1 || {{
  echo "BUILD_FAILED -- tail of /tmp/q38_build.log:" >&2
  tail -80 /tmp/q38_build.log >&2
  exit 1
}}
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
# Returns 1, with the reason on stderr, when the harness exits nonzero, measures nothing, or stops
# requests part-way on too many runs. The caller decides what that means: the scored ladder fails
# the round, a guard fails closed.
cb_median() {{
  local ckpt=$1 cc=$2 out=/tmp/q38_cb.txt attempt=0 valid=0 a i
  shift 2
  CB_AGGS=""; CB_ITLS=""; CB_AGG=0; CB_ITL=0; CB_TOK=0; CB_ERR=0
  while [ "$valid" -lt {cb_reps} ]; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt {cb_max_attempts} ]; then
      echo "concurrent decode at c=$cc on $ckpt stopped requests part-way on $((attempt - 1 - valid)) of {cb_max_attempts} runs" >&2
      return 1
    fi
    wait_gpu_clear
    if ! timeout 900 env "$@" build/runtime/qwen3_gguf_cb_bench "$ckpt" "$cc" {cb_tokens} {cb_tokens} 512 > "$out" 2>&1; then
      echo "concurrent-decode harness exited nonzero at c=$cc on $ckpt" >&2
      tail -20 "$out" >&2 || true
      return 1
    fi
    CB_TOK=$(sed -n 's/.*decode_tokens=\\([0-9]*\\).*/\\1/p' "$out" | tail -1)
    CB_ERR=$(grep -c "request error" "$out" || true)
    if ! cb_complete "$cc" "${{CB_TOK:-0}}" "${{CB_ERR:-0}}"; then
      echo "CB_PARTIAL c=$cc attempt=$attempt decode_tokens=${{CB_TOK:-0}} request_errors=${{CB_ERR:-0}} ($ckpt)" >&2
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
build/runtime/qwen3_gguf_score "$MODEL_DIR" "$TOPK" $IDS > "$DUMP_SELF" 2>/tmp/q38_score.err || {{
  echo "SCORE_FAILED -- tail of /tmp/q38_score.err:" >&2
  tail -40 /tmp/q38_score.err >&2
  exit 1
}}
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
  echo "GUARD36_FAILED"
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
    echo "GUARDMO_FAILED"
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
    echo "GUARDMG_FAILED"
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
    echo "GUARDBN_FAILED"
  fi
else
  echo "GUARDBN_UNAVAILABLE"
fi

# --- Concurrent-decode no-regression guards: ModelOpt and Muse Glimmer (pt. 3b) ---
# The PRs this bot scores mostly change packed decode, which the single-stream 32k guards above never
# enter. Each model runs the way its own bot runs it: ModelOpt with pr_dspark_bot.py's env, Muse
# Glimmer with none (pr_museglimmer_bot.py). A failed measurement prints *_FAILED and the guard fails
# closed, as the 32k guards do; an absent checkpoint skips both of its guards.
if [ -d "$MODELOPT_GUARD_MODEL_DIR" ]; then
  for CC in {cb_guard_concs}; do
    if cb_median "$MODELOPT_GUARD_MODEL_DIR" "$CC" SPARKINFER_QWEN38_PREFILL_NVFP4=1 SPARKINFER_QWEN38_DECODE_NVFP4=1 SPARKINFER_KV_INT8=1; then
      echo "GUARDCBMO $CC $CB_AGG"
    else
      echo "GUARDCBMO_FAILED $CC"
    fi
  done
fi
if [ -f "$MUSE_GUARD_GGUF" ]; then
  for CC in {cb_guard_concs}; do
    if cb_median "$MUSE_GUARD_GGUF" "$CC"; then
      echo "GUARDCBMG $CC $CB_AGG"
    else
      echo "GUARDCBMG_FAILED $CC"
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
                if k in ("top1", "kl", "ppl_pr", "ppl_main"):
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
        elif line.startswith("GUARD36 "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    guard36[int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.strip() == "GUARD36_FAILED":
            out["guard36_failed"] = True
        elif line.split(" ", 1)[0] in cb_guards:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    cb_guards[parts[0]][int(parts[1])] = {"cb-decode": float(parts[2])}
                except ValueError:
                    pass
        elif line.split(" ", 1)[0] in ("GUARDCBMO_FAILED", "GUARDCBMG_FAILED"):
            out[line.split(" ", 1)[0].split("_")[0].lower() + "_failed"] = True
        elif line.split(" ", 1)[0] in cross_guards:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    cross_guards[parts[0]][int(parts[1])] = {"decode": float(parts[2]),
                                                             "prefill": float(parts[3])}
                except ValueError:
                    pass
        elif line.strip() in ("GUARDMO_FAILED", "GUARDMG_FAILED", "GUARDBN_FAILED"):
            out[line.strip().split("_")[0].lower() + "_failed"] = True
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
        tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-2000:]
        crash = _crash_reason(r.stdout, r.stderr)
        reason = "main run failed" + (f" — {crash}" if crash else " (no crash diagnostic captured, possible hard kill — retried once)")
        return {"ok": False, "reason": reason, "log": tail}
    main = _parse_remote(r.stdout or "")
    if "decode128_tps" not in main:
        return {"ok": False, "reason": "main bench missing decode@128 tok/s", "log": (r.stdout or "")[-1500:]}
    if "prefill128_pp" not in main:
        return {"ok": False, "reason": "main bench missing prefill@128 pp", "log": (r.stdout or "")[-1500:]}
    # Fail closed on a ZERO too, not just a missing line: a 16k KV pool that fails to allocate
    # yields 0 rather than an error, and a 0 baseline makes tier_from_gain REJECT every PR.
    if not main.get("prefill16k_pp"):
        return {"ok": False, "reason": "main bench missing/zero prefill@16k pp (KV pool alloc?)",
                "log": (r.stdout or "")[-1500:]}
    missing_cb = [c for c in CB_CONCS if not main.get(f"cb{c}_agg")]
    if missing_cb:
        return {"ok": False, "reason": "main bench missing/zero concurrent decode at "
                                       + "/".join(f"c{c}" for c in missing_cb),
                "log": (r.stdout or "")[-1500:]}
    main["ok"] = True
    return main


def _guard_coverage(d: dict) -> str:
    """One line naming how many rows each guard produced -- a guard that measured nothing is the
    difference between a REJECT and a retry, so the round log should not make anyone guess."""
    return (f"modelopt {len(d.get('guardmo') or {})} ctx / {len(d.get('guardcbmo') or {})} cc · "
            f"muse {len(d.get('guardmg') or {})} ctx / {len(d.get('guardcbmg') or {})} cc · "
            f"qwen3.6 {len(d.get('guard36') or {})} ctx · "
            f"bonsai {len(d.get('guardbn') or {})} ctx")


def eval_qwen38_on_box(host, port, pr_ref: str, main: dict):
    """Run the PR ref's speed+accuracy script on the same box and compare against `main`, an
    already-measured baseline shared across every PR in the round (see measure_main_baseline)."""
    print(f">> Qwen3.8-27B eval on box: PR ref={pr_ref}")
    r = _ssh_run_resilient(host, port, _remote_script(pr_ref, role="pr", onto=main.get("sha")), "PR run")
    if r.returncode != 0:
        combined = (r.stdout or "") + "\n" + (r.stderr or "")
        if "MERGE_CONFLICT" in combined:
            # Does not merge onto the main this round measured: a rebase, not a verdict.
            line = next((l for l in combined.splitlines() if l.startswith("MERGE_CONFLICT")), "")
            return {"ok": False, "conflict": True,
                    "reason": line or "PR does not merge cleanly onto the measured main"}
        tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-2000:]
        crash = _crash_reason(r.stdout, r.stderr)
        reason = "PR speed/accuracy run failed" + (f" — {crash}" if crash else " (no crash diagnostic captured, possible hard kill — retried once)")
        return {"ok": False, "reason": reason, "log": tail}
    pr = _parse_remote(r.stdout or "")
    if "decode128_tps" not in pr:
        return {"ok": False, "reason": "PR bench missing decode@128 tok/s", "log": (r.stdout or "")[-1500:]}
    if "prefill128_pp" not in pr:
        return {"ok": False, "reason": "PR bench missing prefill@128 pp", "log": (r.stdout or "")[-1500:]}
    if not pr.get("prefill16k_pp"):
        return {"ok": False, "reason": "PR bench missing/zero prefill@16k pp (KV pool alloc?)",
                "log": (r.stdout or "")[-1500:]}
    missing_cb = [c for c in CB_CONCS if not pr.get(f"cb{c}_agg")]
    if missing_cb:
        return {"ok": False, "reason": "PR bench missing/zero concurrent decode at "
                                       + "/".join(f"c{c}" for c in missing_cb),
                "log": (r.stdout or "")[-1500:]}
    if "top1" not in pr or "kl" not in pr:
        # Either the score dump failed, or main's dump was missing so the comparator never ran
        # (ACCURACY_NO_BASELINE). Both are infra faults, but they must NOT pass as "accurate" --
        # a gate that cannot measure fails closed, same as check_q36_guard's unavailability path.
        return {"ok": False, "reason": "PR run missing accuracy METRIC line (score dump failed, "
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

    # No batched-prefill parity gate: main fails it on this checkpoint (see the remote script), and
    # an absolute gate that main fails would reject every PR. The differential accuracy gate above
    # and the Qwen3.6 guard below still apply.

    q36_ok, q36_problems = check_q36_guard(pr, main)
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
    for key, name, checks in (("guardmo", "modelopt", (check_modelopt_guard, check_modelopt_cb_guard)),
                              ("guardmg", "muse glimmer", (check_muse_guard, check_muse_cb_guard)),
                              ("guardbn", "ternary-bonsai", (check_bonsai_guard,))):
        skipped = bool(pr.get(f"{key}_unavailable") or main.get(f"{key}_unavailable"))
        ok, problems = True, []
        if not skipped:
            for check in checks:
                c_ok, c_problems = check(pr, main)
                ok, problems = ok and c_ok, problems + c_problems
        if skipped:
            print(f">> {name} guard SKIPPED — checkpoint not installed on the box")
        if not ok:
            # A guard that measured NOTHING is infra, not a regression. Take the same retry path a
            # failed bench run takes -- label null, no verdict posted, re-evaluated next round --
            # instead of closing a PR over a measurement that never happened. On 2026-09-18 one
            # round's missing Muse concurrent rows produced `muse glimmer concurrent guard
            # measurement unavailable` and auto-closed #1112 and #1114, both of which had just
            # measured +10% on the 256k axis; the guard itself ran fine by hand minutes later.
            unavailable = [p for p in problems if p.endswith("measurement unavailable")]
            if unavailable and len(unavailable) == len(problems):
                return {"ok": False,
                        "reason": "; ".join(unavailable) + " — infra, not a regression; the PR is "
                                  "re-evaluated next round rather than rejected",
                        "log": ""}
            reason = f"{name} no-regression guard failed: " + "; ".join(problems[:6]) + f" | {reason}"
            label = "REJECT"
            passed = False
        cross[key] = (ok, problems, skipped)

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
    marker = (
        f"<!-- sparkinfer-qwen38-eval:{EVAL_SCHEMA_VERSION}:{commit} "
        f"{json.dumps(meta, separators=(',', ':'))} -->"
    )
    if not res.get("ok"):
        return (
            f"{marker}\n## sparkinfer qwen38 auto-eval — error\n\n"
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
    # No "main accuracy" row: this gate is differential, so main IS the reference -- there is no
    # separate absolute bar for it to miss.
    main_acc_note = ""
    if res.get("q36_guard_ok"):
        q36_row = "| qwen3.6 guard | ✅ no regression (decode+prefill, ctx 0/512/4k/16k/32k) |\n"
    else:
        problems = "; ".join((res.get("q36_guard_problems") or [])[:4])
        q36_row = (f"| qwen3.6 guard | ❌ **FAILED** — {problems} — "
                    "**verdict forced to REJECT regardless of speed/accuracy** |\n")
    cross_rows = ""
    cb_note = f", concurrent decode @ {'/'.join(f'c{c}' for c in CB_GUARD_CONCS)}"
    bn_ctxs = "/".join("32k" if c == 32768 else str(c) for c in BONSAI_GUARD_CTXS)
    for prefix, name, what, at in (
            ("modelopt", "modelopt guard", "Qwen3.8-27B NVFP4 (ModelOpt)", f"decode+prefill @ 32k{cb_note}"),
            ("muse", "muse glimmer guard", "Muse Glimmer 30B", f"decode+prefill @ 32k{cb_note}"),
            ("bonsai", "ternary-bonsai guard", "Ternary-Bonsai-2-27B", f"decode+prefill @ {bn_ctxs}")):
        if res.get(f"{prefix}_guard_skipped"):
            # Say SKIPPED explicitly: a guard that reports nothing reads the same as one that passed.
            cross_rows += (f"| {name} | ⚠️ SKIPPED — checkpoint not installed on the box; "
                           f"shared-code regressions on {what} were NOT checked |\n")
        elif res.get(f"{prefix}_guard_ok"):
            cross_rows += f"| {name} | ✅ no regression ({at}, {what}) |\n"
        else:
            probs = "; ".join((res.get(f"{prefix}_guard_problems") or [])[:4])
            cross_rows += (f"| {name} | ❌ **FAILED** — {probs} — "
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
    if QWEN38_MERGE_FIRST not in labs:
        return False, "not qwen38-merge-first"
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


def try_auto_merge_qwen38(repo, num):
    ok, reason = auto_merge_ok_qwen38(repo, num)
    if not ok:
        print(f">> qwen38 auto-merge SKIP #{num}: {reason}")
        return False
    r = arb.gh(["pr", "merge", str(num), "-R", repo, "--squash"])
    if r.returncode != 0 and os.environ.get("SPARKINFER_AUTOMERGE_ADMIN", "1") == "1":
        err = ((r.stderr or "") + (r.stdout or "")).lower()
        if "not mergeable" in err or "branch policy" in err or "required" in err or "prohibited" in err:
            print(">> qwen38 auto-merge: branch policy blocked — retrying with --admin")
            r = arb.gh(["pr", "merge", str(num), "-R", repo, "--squash", "--admin"])
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


def reconcile_qwen38_merge_labels(repo, dry_run=False):
    scores = _load_scores()
    open_prs = json.loads(arb.gh([
        "pr", "list", "-R", repo, "--state", "open",
        "--json", "number,labels", "--limit", "80",
    ]).stdout or "[]")
    open_labels = {p["number"]: {l["name"] for l in p["labels"]} for p in open_prs}

    merged = json.loads(arb.gh([
        "pr", "list", "-R", repo, "--state", "merged", "--label", QWEN38_MERGE_FIRST,
        "--json", "number", "--limit", "10",
    ]).stdout or "[]")
    for m in merged:
        if not dry_run:
            arb.remove_label(repo, m["number"], QWEN38_MERGE_FIRST)

    scored = []
    for num, labs in open_labels.items():
        if QWEN38_NEEDS_REBASE in labs:
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
        print(">> qwen38 round: no verified speedup PRs")
        return
    winner = scored[0][0]
    print(f">> qwen38 round: merge-first #{winner}; rebase {[n for n,_,_ in scored[1:]] or 'none'}")
    if dry_run:
        return
    arb.add_label(repo, winner, QWEN38_MERGE_FIRST)
    arb.remove_label(repo, winner, QWEN38_NEEDS_REBASE)
    for num, _, _ in scored[1:]:
        arb.add_label(repo, num, QWEN38_NEEDS_REBASE)
        arb.remove_label(repo, num, QWEN38_MERGE_FIRST)
    if AUTO_MERGE:
        try_auto_merge_qwen38(repo, winner)


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
            "pr_top1": res.get("pr_top1"), "pr_kl": res.get("pr_kl"),
            "accuracy_ok": res.get("accuracy_ok"),
            "q36_guard_ok": res.get("q36_guard_ok"), "q36_guard_problems": res.get("q36_guard_problems"),
            "modelopt_guard_ok": res.get("modelopt_guard_ok"), "modelopt_guard_problems": res.get("modelopt_guard_problems"),
            "modelopt_guard_skipped": res.get("modelopt_guard_skipped"),
            "muse_guard_ok": res.get("muse_guard_ok"), "muse_guard_problems": res.get("muse_guard_problems"),
            "muse_guard_skipped": res.get("muse_guard_skipped"),
            "bonsai_guard_ok": res.get("bonsai_guard_ok"), "bonsai_guard_problems": res.get("bonsai_guard_problems"),
            "bonsai_guard_skipped": res.get("bonsai_guard_skipped"),
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
        push = subprocess.run(["git", "-C", arb.LOG_DIR, "push", "-q"], check=False)
        if push.returncode != 0:
            print(f">> qwen38 eval-log push failed (rc={push.returncode})")
            return None
        url = arb.LOG_PAGE + rid
        print(f">> qwen38 eval log: {url}")
        return url
    except Exception as e:
        print(f">> qwen38 eval-log upload failed: {e}")
        return None


def apply_result(repo, num, commit, res, title="", dry_run=False):
    if not res.get("ok") and res.get("conflict"):
        # No verdict and no REJECT label: the PR does not merge onto the main this round measured,
        # which says nothing about its change. Same outcome as the pre-GPU merge-conflict check.
        print(f"PR #{num}: {res.get('reason')} — qwen38-needs-rebase, no verdict")
        if not dry_run:
            arb.add_label(repo, num, QWEN38_NEEDS_REBASE)
        return
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
    arb.gh(["pr", "comment", str(num), "-R", repo, "--body", body])
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
            "pr_top1": res.get("pr_top1"),
            "pr_kl": res.get("pr_kl"),
            "pass": res.get("pass"),
            "accuracy_ok": res.get("accuracy_ok"),
            "q36_guard_ok": res.get("q36_guard_ok"),
            "modelopt_guard_ok": res.get("modelopt_guard_ok"),
            "muse_guard_ok": res.get("muse_guard_ok"),
            "bonsai_guard_ok": res.get("bonsai_guard_ok"),
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
                    "for `hold`; those are reviewed by hand."
                )
            else:
                close_body = (
                    "<!-- sparkinfer-qwen38-auto-close -->\n"
                    f"## Closed: regression or failed gate — `eval-qwen38:{label}`\n\n"
                    f"Measured **{res.get('delta_pct')}%** vs the same-box `origin/main`, "
                    f"{fail_clause} — closing automatically.\n\n"
                    "Every measured axis is also a no-regression floor, and accuracy and the "
                    "Qwen3.6, ModelOpt, Muse Glimmer and Ternary-Bonsai guards are hard gates, so one failure closes the PR whatever it was "
                    "aiming at. The verdict comment above names which one. Reopen once it is "
                    "addressed and it re-evaluates on the next poll."
                )
            arb.gh(["pr", "comment", str(num), "-R", repo, "--body", close_body])
            arb.gh(["pr", "close", str(num), "-R", repo])
            print(f">> auto-closed PR #{num} (eval-qwen38:{label})")


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

    if args.labels_only:
        reconcile_qwen38_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — qwen38 labels only (no GPU).")
        return

    prs = json.loads(arb.gh([
        "pr", "list", "-R", args.repo, "--state", "open",
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
            print(f"PR #{num} @ {short}: already qwen38-evaluated — skip")
            continue
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
        pending.append((num, head, short, ref, pr.get("title", "")))

    if not pending:
        reconcile_qwen38_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — no qwen38 PRs to evaluate.")
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
        print("done — qwen38 round skipped (main baseline unusable).")
        return
    # main has no top1/kl of its own: it IS the accuracy reference, and its score dump was just
    # written to SCORE_DUMP_MAIN for each PR in this round to diff against.
    print(f">> main baseline: decode@128={main_result['decode128_tps']:.2f} tok/s "
          f"prefill@128={main_result['prefill128_pp']:.2f} pp "
          f"prefill@16k={main_result['prefill16k_pp']:.2f} pp cb {_cb_summary(main_result)}")
    print(f">> main guard coverage: {_guard_coverage(main_result)}")

    for num, head, short, ref, title in pending:
        print(f"PR #{num} @ {short}: evaluating Qwen3.8-27B '{ref}' …")
        try:
            res = eval_qwen38_on_box(host, port, ref, main_result)
        except Exception as e:
            res = {"ok": False, "reason": f"exception: {e}"}
        apply_result(args.repo, num, head or short, res, title=title, dry_run=False)

    reconcile_qwen38_merge_labels(args.repo, dry_run=False)
    print("done — qwen38 eval pass complete.")


if __name__ == "__main__":
    main()
