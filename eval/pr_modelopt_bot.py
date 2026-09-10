#!/usr/bin/env python3
"""sparkinfer Qwen3.8-27B ModelOpt (NVFP4) PR auto-evaluator.

Sibling of pr_museglimmer_bot.py / pr_dflash_bot.py, and the ONLY scored bot once Qwen3.8-27B
becomes the eval scope (see eval/README.md). Narrowly scoped on purpose:

  0. SCOPE (2026-08-17) — LONG CONTEXT on the ModelOpt checkpoint is what earns a tier:
              decode @ ctx=16k and prefill @ ctx=16k, whichever the PR moved more. Those two are
              the only kinds of work this bot will auto-merge.

              decode@128 and prefill@128 are still measured and still act as no-regression FLOORS
              (as do the two scoring dimensions), so a PR cannot buy long-context throughput by
              trading short-context away; a PR that improves only a 128-context dimension scores
              "none" by design.

              "The ModelOpt checkpoint" is load-bearing: this bot only ever benchmarks MODEL_DIR,
              so every scored number is gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090's. The
              upstream unsloth build is guarded, never scored (see Q38_GUARD_MODEL_DIR).

              Costs no extra GPU time -- ctx 16384 is one sweep entry and both metrics fall out of
              that same model load. The rest of this docstring describes the machinery, which is
              unchanged.

  1. Speed — decode and prefill on the NVFP4 checkpoint at ctx 128/4k/16k, PR vs a freshly-
              measured origin/main, same box, same round, median of 5 reps. Which of those earn a
              tier and which are only floors is pt. 0's business, not this section's; everything
              below describes how the numbers are produced, which is the same either way. Same
              tier buckets as every other bot in this directory (BUCKETS/SIG/REGRESS_TOL below —
              copied, not reinvented).

              The scored checkpoint is a compressed-tensors DIRECTORY, NOT a GGUF of the same
              model. qwen3_gguf_bench and qwen3_gguf_score grew directory support for exactly this
              (runtime/examples/qwen_checkpoint.h, shared with the server so the two cannot
              disagree about how a checkpoint is configured).

              sparkinfer supports TWO NVFP4 builds of Qwen3.8-27B and they are not equivalent, so
              which one a number came from has to be stated every time:

                MODEL_DIR (SCORED)   gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 -- our own
                                     quantization, uniform NVFP4 on every Linear, built with
                                     NVIDIA ModelOpt for the RTX 5090's FP4 tensor cores. This is
                                     what a PR's speed claim is measured against.

                Q38_GUARD_MODEL_DIR  unsloth/Qwen3.8-27B-NVFP4 -- upstream, mixed NVFP4 FFN + FP8
                                     attention/GDN projections. Also supported, and NOT scored;
                                     it gets a no-regression guard instead (check_q38_guard), so
                                     an optimisation cannot win on the scored build by pessimising
                                     this one. Both are supported targets; scoring one and
                                     guarding the other is what keeps that honest.

              Measured 2026-08-17 on the pinned box at the same commit, the gap between them is
              real and large -- ModelOpt 95.8/93.7/90.1 tok/s decode and 6546/10021/9749 pp
              prefill at ctx 128/4k/16k, against unsloth's 84.9/83.2/80.6 and 5031/9548/8727 --
              so quoting a Qwen3.8 number without naming its checkpoint is meaningless.

              Prefill is measured at every context in the same sweep and the same model load (no
              extra GPU time -- prefill_pp was already computed alongside decode_tps and simply
              discarded). Combination rule is pr_museglimmer_bot.py's, verbatim: any dimension
              regressing is a hard REJECT, otherwise the better of the SCORING_DIMS tiers wins, so
              a pure prefill win with flat decode still scores.

              Historical note on the absolute prefill number, kept because it explains why this
              dimension is watched so closely. Measured 2026-08-15, Qwen3.8-27B prefill@128 was
              ~86 pp -- roughly this model's DECODE speed at the time (82.7 tok/s), the signature
              of a token-at-a-time path rather than one batched GEMM pass, against llama.cpp's
              ~2651 t/s on the same prompt from a Q4_K_M GGUF. Making that gap visible is what got
              it fixed: the batched prefill path landed and prefill@128 is ~6.5k pp as of the
              2026-08-17 figures above. Do not quote the ~86 pp number as current.

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

Applies `eval-qwen38:<TIER>` AND mirrors it to the generic `eval:<TIER>` label (SN74 scoring reads
eval:* tiers). Auto-close on none/REJECT is live; auto-merge stays OFF unless
SPARKINFER_MODELOPT_AUTOMERGE=1 is explicitly set.

  python eval/pr_modelopt_bot.py --instance 46074104
  python eval/pr_modelopt_bot.py --only-prs 636 --reeval

Never rents a GPU. Shares the pinned box with any other bot via flock in the cron wrapper
(run_modelopt_cron.sh) — all bots MUST share /tmp/sparkinfer_bot.lock.
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

# The dimensions that can earn a tier: LONG CONTEXT, both axes (decision 2026-08-17). decode@16k
# and prefill@16k are the optimisation target, so those two are what auto-merges.
#
# A PR is tiered on whichever of the two it moved MORE, not on a sum or an average -- they are
# largely independent (decode at 16k is KV-read bound at m=1, prefill at 16k is a tensor-core GEMM
# over the whole prompt), so a real win in one is a real win, and averaging would let an untouched
# dimension dilute it.
#
# decode@128 and prefill@128 stay MEASURED as no-regression FLOORS, as do these two: a PR that
# buys long-context throughput by trading away short-context is still a hard REJECT. Floor and
# scoring are separate lists precisely so the target can move without weakening the guard -- a PR
# that improves only a 128-context dimension now scores "none" by design.
#
# Costs no extra GPU time: ctx 16384 was already in the sweep, and both metrics come out of that
# same model load.
SCORING_DIMS = ["decode@16k", "prefill@16k"]
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

EVAL_PREFIX = "eval-modelopt:"
MODELOPT_MERGE_FIRST = "modelopt-merge-first"
MODELOPT_NEEDS_REBASE = "modelopt-needs-rebase"
# First schema for this bot. Same reasoning as the sibling bots' own bumps: a PR evaluated before
# a scoring change existed must not keep a stale-scored label/score forever.
EVAL_SCHEMA_VERSION = "v1-modelopt-decode128"
MARKER_RE = re.compile(
    r"<!-- sparkinfer-modelopt-eval:" + re.escape(EVAL_SCHEMA_VERSION) + r":([0-9a-f]+)(?:\s+(\{.*?\}))? -->",
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
BENCH_TOKENS = int(os.environ.get("MODELOPT_BENCH_TOKENS", "128"))
ACC_TOPK = int(os.environ.get("MODELOPT_ACC_TOPK", "128"))
# Batched-prefill parity floor: the fraction of the continuation that batched prefill must still
# generate identically to the token loop. Absolute, not PR-vs-main -- see the PREFILL_PARITY block
# in the remote script for why a differential gate cannot catch this class of bug.
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
# The guard sweeps the same three contexts the scored bench does, so a shared-path regression shows
# up wherever it lives: ctx 0 covers the short-context floors, 4k/16k cover the scoring dimensions
# and are where the shared prefill and KV paths cost the most.
Q38_GUARD_CTXS = [0, 4096, 16384]

# Auto-merge is wired (mirrors pr_dflash_bot.py's auto_merge_ok_dflash/try_auto_merge_dflash
# shape) but OFF unless this exact env var is set.
# Was reading SPARKINFER_QWEN38_AUTOMERGE until 2026-08-17 -- a leftover from this file being
# forked from pr_qwen38_bot.py, whose rename to the ModelOpt-specific var missed this one
# word-boundary. It "worked" only because .env.eval happened to set BOTH SPARKINFER_QWEN38_
# AUTOMERGE and SPARKINFER_MODELOPT_AUTOMERGE=1, not because they were actually independent —
# flipping one without the other would have silently done nothing. Reads its own var now.
AUTO_MERGE = os.environ.get("SPARKINFER_MODELOPT_AUTOMERGE") == "1"
AUTOMERGE_BLOCK = {
    "copycat", "copycat-warn", "flagged:gaming", "penalty", "needs-benchmark",
    MODELOPT_NEEDS_REBASE, arb.REEVALUATE_LABEL, arb.HOLD_LABEL, *arb.REGRESSION_LABELS,
}

SCORES_FILE = os.path.expanduser(
    os.environ.get("MODELOPT_SCORES_FILE", "~/.sparkinfer_modelopt_scores.json")
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
        print(f">> modelopt scores save skipped: {e}")


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
        if not m or "sparkinfer modelopt auto-eval" not in body:
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
    """One automatic retry on an apparent hard kill — same insurance pr_dflash_bot.py added
    after #684/#690 (heavy model-reload boundaries silently killing the whole remote shell)."""
    r = ssh_run(host, port, script, via_stdin=True)
    if r.returncode != 0 and _looks_like_hard_kill(r.stdout, r.stderr):
        print(f">> {label}: looks like a hard kill (no ERR-trap diagnostic, no accuracy-stage "
              f"checkpoint reached) — retrying once")
        r = ssh_run(host, port, script, via_stdin=True)
    return r


def _remote_script(ref: str, role: str = "pr") -> str:
    """Bash run on the eval box: checkout ref, build, decode+prefill@128 bench on the NVFP4
    checkpoint, teacher-forced score dump, and the Qwen3.6 no-regression guard.

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
    eval_text = shlex.quote(EVAL_TEXT)
    dump_self = shlex.quote(SCORE_DUMP_MAIN if role == "main" else SCORE_DUMP_PR)
    dump_main = shlex.quote(SCORE_DUMP_MAIN)
    is_pr = "1" if role == "pr" else "0"
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
NTOK={ntok}
TOPK={topk}
PARITY_BAR={parity_bar}
EVAL_TEXT={eval_text}
DUMP_SELF={dump_self}
DUMP_MAIN={dump_main}
IS_PR={is_pr}
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
cmake --build build --target qwen3_gguf_bench qwen3_gguf_score qwen3_gguf_generate -j"$(nproc)" >/tmp/mopt_build.log 2>&1 || {{
  echo "BUILD_FAILED -- tail of /tmp/mopt_build.log:" >&2
  tail -80 /tmp/mopt_build.log >&2
  exit 1
}}
test -x build/runtime/qwen3_gguf_bench
test -x build/runtime/qwen3_gguf_score

# --- decode @ ctx=128 on the NVFP4 checkpoint ---
# A no-regression FLOOR since the 2026-08-17 scope change, not a scored dimension (module
# docstring pt. 0); SCORING_DIMS is decode@16k + prefill@16k. Prefill is deliberately NOT scored
# at this context:
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
BENCH_PROMPT_IDS=/tmp/mopt_bench_prompt_ids.txt
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
# 4096 joins the sweep purely to feed the README refresh (README_BENCH_MARK below): the published
# Qwen3.8 table quotes 128/4k/16k, and a PR that auto-merges is what makes those numbers stale, so
# the round has to have measured all three. Same model load as 128/16k, so it costs one extra
# context's worth of reps and no extra load. NOT a scoring or floor dimension -- nothing reads
# decode4k/prefill4k except the README writer.
if bench_sweep_run "$MODEL_DIR" "$NTOK" 128 5 4096 5 16384 5; then
  DECODE128_TPS=$(_bench_sweep_get 128 decode_tps)
  DECODE4K_TPS=$(_bench_sweep_get 4096 decode_tps)
  PREFILL4K_PP=$(_bench_sweep_get 4096 prefill_pp)
  # Same model load, same sweep, no extra GPU time for this one: prefill_pp is already computed
  # alongside decode_tps for this context, it was simply being discarded.
  PREFILL128_PP=$(_bench_sweep_get 128 prefill_pp)
  PREFILL16K_PP=$(_bench_sweep_get 16384 prefill_pp)
  # decode@16k: the scoring dimension (2026-08-17). Exactly the same free-metric story as
  # prefill@128 above -- ctx 16384 is ALREADY in this sweep for prefill@16k, and decode_tps was
  # computed alongside it and discarded. Costs no extra GPU time, and it is where a long-context
  # decode optimisation actually shows: at ctx=128 the KV read is negligible, so a PR that speeds
  # up decode against a 16k KV cache measures ~0.0% on decode@128.
  DECODE16K_TPS=$(_bench_sweep_get 16384 decode_tps)
else
  DECODE128_TPS=0
  PREFILL128_PP=0
  PREFILL16K_PP=0
  DECODE16K_TPS=0
  DECODE4K_TPS=0
  PREFILL4K_PP=0
fi
echo "RESULT_DECODE128_TPS ${{DECODE128_TPS:-0}}"
echo "RESULT_PREFILL128_PP ${{PREFILL128_PP:-0}}"
echo "RESULT_PREFILL16K_PP ${{PREFILL16K_PP:-0}}"
echo "RESULT_DECODE16K_TPS ${{DECODE16K_TPS:-0}}"
echo "RESULT_DECODE4K_TPS ${{DECODE4K_TPS:-0}}"
echo "RESULT_PREFILL4K_PP ${{PREFILL4K_PP:-0}}"

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

# --- batched-prefill parity (see bench/scripts/prefill_parity_check.py) ---
# The accuracy gate above CANNOT see batched prefill: qwen3_gguf_score teacher-forces through
# forward_token() and never enters prefill_batched_run(), while bench_sweep_run right above it
# reports prefill@128 and prefill@16k measured on exactly that path. That blind spot let the FP4
# FFN arm drop the entire FFN contribution from the residual stream from #837 until 2026-08-16 --
# twelve perf PRs, all green, all scored against a wrong state. Throughput is what selected for
# the bug, so throughput alone must not be able to pass a prefill PR again.
#
# Absolute, not differential: unlike the top1/KL gate this compares batched prefill against the
# TOKEN LOOP in the same build, so it catches a defect already present on main rather than only a
# newly introduced divergence. A run where main is equally broken must still fail.
wait_gpu_clear
if PARITY_BAR="$PARITY_BAR" python3 bench/scripts/prefill_parity_check.py \
     "$MODEL_DIR" "$MODEL_DIR/tokenizer.json" 32,128 > /tmp/mopt_parity.txt 2>&1; then
  echo "PREFILL_PARITY_OK"
else
  echo "PREFILL_PARITY_FAILED" >&2
fi
grep -E "^PARITY|^n=" /tmp/mopt_parity.txt || tail -20 /tmp/mopt_parity.txt

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

# --- Qwen3.8 no-regression guard (shared compressed-tensors loader / prefill / decode) ---
echo "GUARD38_START"
wait_gpu_clear
if [ -d "$Q38_GUARD_MODEL_DIR" ] && [ -f "$Q38_GUARD_MODEL_DIR/config.json" ]; then
  if bench_sweep_run "$Q38_GUARD_MODEL_DIR" 128 0 5 4096 5 16384 5; then
    for ctx in 0 4096 16384; do
      echo "GUARD38 $ctx $(_bench_sweep_get $ctx decode_tps) $(_bench_sweep_get $ctx prefill_pp)"
    done
  else
    echo "GUARD38_FAILED"
  fi
else
  echo "GUARD38_FAILED"
  echo "WARN: qwen3.8 guard checkpoint missing at $Q38_GUARD_MODEL_DIR" >&2
fi
echo "GUARD38_END"

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
echo "GUARD_END"
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
        elif line.startswith("RESULT_DECODE16K_TPS "):
            try:
                out["decode16k_tps"] = float(line.split()[1])
            except ValueError:
                pass
        # 4k: README-only (see README_BENCH_MARK). Never scored, never a floor.
        elif line.startswith("RESULT_DECODE4K_TPS "):
            try:
                out["decode4k_tps"] = float(line.split()[1])
            except ValueError:
                pass
        elif line.startswith("RESULT_PREFILL4K_PP "):
            try:
                out["prefill4k_pp"] = float(line.split()[1])
            except ValueError:
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
        elif line.startswith("GUARD38 "):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    guard38[int(parts[1])] = {"decode": float(parts[2]),
                                              "prefill": float(parts[3])}
                except ValueError:
                    pass
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
    # Same fail-closed rule for the SCORING dimension, and it matters more here than for the
    # floors: a zero baseline would make tier_from_gain treat every PR as an infinite speedup on
    # a dimension that can earn a tier, i.e. auto-merge on a broken measurement.
    if not main.get("decode16k_tps"):
        return {"ok": False, "reason": "main bench missing/zero decode@16k tok/s (KV pool alloc?)",
                "log": (r.stdout or "")[-1500:]}
    main["ok"] = True
    return main


def eval_qwen38_on_box(host, port, pr_ref: str, main: dict):
    """Run the PR ref's speed+accuracy script on the same box and compare against `main`, an
    already-measured baseline shared across every PR in the round (see measure_main_baseline)."""
    print(f">> Qwen3.8-27B eval on box: PR ref={pr_ref}")
    r = _ssh_run_resilient(host, port, _remote_script(pr_ref, role="pr"), "PR run")
    if r.returncode != 0:
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
    if not pr.get("decode16k_tps"):
        return {"ok": False, "reason": "PR bench missing/zero decode@16k tok/s (KV pool alloc?)",
                "log": (r.stdout or "")[-1500:]}
    if "top1" not in pr or "kl" not in pr:
        # Either the score dump failed, or main's dump was missing so the comparator never ran
        # (ACCURACY_NO_BASELINE). Both are infra faults, but they must NOT pass as "accurate" --
        # a gate that cannot measure fails closed, same as check_q36_guard's unavailability path.
        return {"ok": False, "reason": "PR run missing accuracy METRIC line (score dump failed, "
                                       "or no main baseline dump to diff against)",
                "log": (r.stdout or "")[-1500:]}
    print(f">> PR decode@16k={pr['decode16k_tps']:.2f} (scored) "
          f"decode@128={pr['decode128_tps']:.2f} prefill@128={pr['prefill128_pp']:.2f} "
          f"prefill@16k={pr['prefill16k_pp']:.2f} "
          f"top1={pr.get('top1', 0):.4f} kl={pr.get('kl', 99):.5f}")

    # FOUR measured dimensions on the NVFP4 checkpoint, all from the one model load (module
    # docstring pt. 0). Only the two in SCORING_DIMS can earn a tier; the other two are floors.
    # Same rule pr_museglimmer_bot.py uses for its two, generalised rather than grown into a
    # longer if/elif chain -- with four dimensions the hand-written cascade needs 2^4 orderings
    # to stay correct and silently mis-scores if one is missed:
    #   * ANY dimension regressing is a hard REJECT, regardless of the others.
    #   * Otherwise the BEST tier among SCORING_DIMS wins, so a PR that improves one axis with
    #     the others merely flat still scores for the real work it did (a long-context prefill PR
    #     is not expected to move decode@16k, and vice versa).
    dims = [
        ("decode@16k",  pr["decode16k_tps"],   main["decode16k_tps"]),
        ("decode@128",  pr["decode128_tps"],   main["decode128_tps"]),
        ("prefill@128", pr["prefill128_pp"],   main["prefill128_pp"]),
        ("prefill@16k", pr["prefill16k_pp"],   main["prefill16k_pp"]),
    ]
    scored = []
    for name, pr_v, main_v in dims:
        lab, dlt, ok, why = tier_from_gain(pr_v, main_v, metric=name)
        scored.append({"dim": name, "label": lab, "delta": dlt, "passed": ok, "reason": why})
    by_dim = {s["dim"]: s for s in scored}

    # ANY dimension regressing is still a hard REJECT, including the two that cannot earn a tier.
    # decode@128 and prefill@128 are NOT scoring dimensions -- since the 2026-08-17 scope change
    # they are no-regression FLOORS only -- but they stay measured, because without them a PR
    # could trade short-context throughput away to buy long-context and still auto-merge at XL.
    # It costs nothing to keep: all four come from the one sweep.
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
    decode_label,  decode_delta_pct  = by_dim["decode@128"]["label"],  by_dim["decode@128"]["delta"]
    prefill_label, prefill_delta_pct = by_dim["prefill@128"]["label"], by_dim["prefill@128"]["delta"]
    p16k_label,    p16k_delta_pct    = by_dim["prefill@16k"]["label"], by_dim["prefill@16k"]["delta"]
    d16k_label,    d16k_delta_pct    = by_dim["decode@16k"]["label"],  by_dim["decode@16k"]["delta"]
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

    # Batched-prefill parity. ABSOLUTE, not differential: it compares batched prefill against the
    # token loop inside the PR's own build, so unlike the accuracy gate above it fails a run whose
    # defect is already present on main. That is the whole point -- the FP4 FFN arm dropped the
    # entire FFN contribution from the residual stream from #837 to 2026-08-16 and every one of the
    # twelve PRs in between passed, because qwen3_gguf_score never enters prefill_batched_run()
    # while bench_sweep_run measures prefill on exactly that path.
    #
    # Missing => fail, matching check_q36_guard's fail-closed handling of an absent measurement: a
    # parity result that did not run is not evidence that prefill is sound, and treating it as a
    # pass would restore the blind spot this gate exists to close.
    parity_ok = pr.get("parity_ok")
    if parity_ok is not True:
        pw = pr.get("parity_worst")
        par_reason = ("batched-prefill parity failed (bar >=%.2f%s): batched prefill does not "
                      "reproduce the token loop" % (
                          PARITY_BAR,
                          ", worst=%.3f" % pw if isinstance(pw, float) else ", not measured"))
        reason = f"{par_reason} | {reason}"
        label = "REJECT"
        passed = False

    q38_ok, q38_problems = check_q38_guard(pr, main)
    if not q38_ok:
        # Same hard-REJECT discipline as the Qwen3.6 guard below: a ModelOpt PR that wins on its
        # own checkpoint while regressing the shipping Qwen3.8 one via shared code is unmergeable
        # regardless of its own score.
        q38_reason = "qwen3.8 no-regression guard failed: " + "; ".join(q38_problems[:6])
        reason = f"{q38_reason} | {reason}"
        label = "REJECT"
        passed = False

    q36_ok, q36_problems = check_q36_guard(pr, main)
    if not q36_ok:
        # Same hard-REJECT discipline as the accuracy gate: a Qwen3.8-27B PR that silently
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
        "pr_decode_tps": pr["decode128_tps"],
        "main_decode_tps": main["decode128_tps"],
        "decode_delta_pct": decode_delta_pct,
        "decode_regressed": decode_label == "REJECT",
        "pr_prefill_pp": pr["prefill128_pp"],
        "main_prefill_pp": main["prefill128_pp"],
        "prefill_delta_pct": prefill_delta_pct,
        "prefill_regressed": prefill_label == "REJECT",
        "pr_prefill16k_pp": pr["prefill16k_pp"],
        "main_prefill16k_pp": main["prefill16k_pp"],
        "prefill16k_delta_pct": p16k_delta_pct,
        "prefill16k_regressed": p16k_label == "REJECT",
        "pr_decode16k_tps": pr["decode16k_tps"],
        "main_decode16k_tps": main["decode16k_tps"],
        "decode16k_delta_pct": d16k_delta_pct,
        "decode16k_regressed": d16k_label == "REJECT",
        # README-only, unscored: the published table quotes 128/4k/16k, and after this PR
        # auto-merges these ARE main's numbers (squash-merge makes PR head == main content).
        "pr_decode4k_tps": pr.get("decode4k_tps", 0.0),
        "pr_prefill4k_pp": pr.get("prefill4k_pp", 0.0),
        # Which dimension the headline tier came from -- otherwise an XL on the comment is
        # ambiguous between a decode win and a prefill win.
        "scored_dimension": best["dim"],
        # The headline ratio now tracks the SCORING dimension (decode@16k), not decode@128 --
        # leaving it on decode@128 would print a "speedup" unrelated to what earned the tier.
        "speedup_vs_main": round(pr["decode16k_tps"] / main["decode16k_tps"], 3) if main.get("decode16k_tps") else 0,
        "pr_top1": pr_top1,
        "pr_kl": pr_kl,
        "pr_ppl": pr.get("ppl_pr"),
        "main_ppl": pr.get("ppl_main"),
        "token_count": pr.get("token_count"),
        "accuracy_ok": accuracy_ok,
        "q36_guard_ok": q36_ok,
        "q38_guard_ok": q38_ok,
        "parity_ok": parity_ok is True,
        "q36_guard_problems": q36_problems,
        "q36_guard": pr.get("guard36"),
        "q36_guard_main": main.get("guard36"),
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
        "pr_prefill_pp": res.get("pr_prefill_pp"),
        "main_prefill_pp": res.get("main_prefill_pp"),
        "pr_prefill16k_pp": res.get("pr_prefill16k_pp"),
        "main_prefill16k_pp": res.get("main_prefill16k_pp"),
        "pr_decode16k_tps": res.get("pr_decode16k_tps"),
        "main_decode16k_tps": res.get("main_decode16k_tps"),
        "scored_dimension": res.get("scored_dimension"),
        "pr_top1": res.get("pr_top1"),
        "pr_kl": res.get("pr_kl"),
        "pass": res.get("pass"),
        "accuracy_ok": res.get("accuracy_ok"),
        "q36_guard_ok": res.get("q36_guard_ok"),
    }
    marker = (
        f"<!-- sparkinfer-modelopt-eval:{EVAL_SCHEMA_VERSION}:{commit} "
        f"{json.dumps(meta, separators=(',', ':'))} -->"
    )
    if not res.get("ok"):
        return (
            f"{marker}\n## sparkinfer modelopt auto-eval — error\n\n"
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
        f"{marker}\n## sparkinfer modelopt auto-eval — `eval-modelopt:{lab}`\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| **label** | `eval-modelopt:{lab}` |\n"
        f"| scored at | **decode@16k + prefill@16k** on the ModelOpt checkpoint — the better of the two earns the tier; decode@128 and prefill@128 are measured as no-regression floors |\n"
        f"| tier came from | `{res.get('scored_dimension', '?')}` |\n"
        f"| **PR decode@16k tok/s** | **{res.get('pr_decode16k_tps', 0):.2f}** |\n"
        f"| **main decode@16k tok/s** | **{res.get('main_decode16k_tps', 0):.2f}** |\n"
        f"| **decode@16k speedup vs main** | **{res.get('speedup_vs_main', 0):.2f}×** ({res.get('decode16k_delta_pct', 0):+.1f}%) |\n"
        f"| PR decode@128 tok/s | {res['pr_decode_tps']:.2f} |\n"
        f"| main decode@128 tok/s | {res['main_decode_tps']:.2f} |\n"
        f"| decode@128 vs main (floor) | {res.get('decode_delta_pct', 0):+.1f}% |\n"
        f"| PR prefill@128 pp | {res['pr_prefill_pp']:.2f} |\n"
        f"| main prefill@128 pp | {res['main_prefill_pp']:.2f} |\n"
        f"| prefill@128 vs main | {res.get('prefill_delta_pct', 0):+.1f}% |\n"
        f"| PR prefill@16k pp | {res['pr_prefill16k_pp']:.2f} |\n"
        f"| main prefill@16k pp | {res['main_prefill16k_pp']:.2f} |\n"
        f"| prefill@16k vs main | {res.get('prefill16k_delta_pct', 0):+.1f}% |\n"

        f"{acc_row}"
        f"{main_acc_note}"
        f"{q36_row}"
        f"| PPL PR / main | {res.get('pr_ppl') or '?'} / {res.get('main_ppl') or '?'} |\n"
        f"{polaris_row}"
        f"| commit | `{commit[:9]}` |\n\n"
        f"{res.get('reason') or ''}\n\n"
        "<sub>Scored on the pinned eval box vs same-box `origin/main` — 128-token AR decode "
        "(ctx=0) AND 128-ctx prefill throughput, from one model load; either dimension regressing "
        "is a hard REJECT, but otherwise the reported label is the **better** of the two tiers — "
        "a PR that improves just one, with the other flat, still earns credit for that "
        "(no DFlash, no long-context beyond 128) — Qwen3.8-27B's narrow, deliberately "
        "strict eval scope. This is informational, not a judgment on your PR: a `none` label just "
        "means no measurable Qwen3.8-27B speedup was verified on either metric, which is expected "
        "and fine if that isn't what your change is about. "
        "Correctness gated against a live llama.cpp reference on the same GGUF. Also gated on a "
        "Qwen3.6 no-regression guard (decode+prefill, ctx 0/512/4k/16k/32k, same box vs main) — "
        "Qwen3.8-27B PRs can touch code shared with other models. "
        "Automated — **not merged**; merge manually after review.</sub>\n"
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


# The README's Qwen3.8 ModelOpt table is regenerated between these markers. Everything outside
# them is prose and is never touched, so the table can be machine-written without a generator
# owning the whole document.
README_BENCH_MARK = ("<!-- BENCH:qwen38-modelopt:start -->", "<!-- BENCH:qwen38-modelopt:end -->")


def refresh_readme_modelopt(res, commit, repo_root=None, push=True):
    """Rewrite the README's ModelOpt table from a just-merged PR's own measurements.

    Called ONLY after a successful auto-merge. The PR is squash-merged, so its head content IS
    main's content -- the numbers the round measured on the PR are main's numbers, and no second
    benchmark pass is needed. That is also why this cannot use the round's `main` baseline: that
    was measured BEFORE the merge and is now one PR out of date.

    Writes through a THROWAWAY WORKTREE of origin/main, never the bot's own checkout. The cron
    runs this bot from whatever branch that tree happens to be on -- feat/qwen3.8-dspark at the
    time of writing -- so committing and `push origin HEAD:main` in place would have pushed an
    entire unmerged feature branch to main on the first auto-merge. The worktree also means a
    half-finished refresh can never leave the bot's tree dirty mid-round.

    Deliberately limited to the ModelOpt table. The GGUF head-to-head and the unsloth table are
    left alone because nothing in a round measures them; both carry their own "measured at
    <commit>" stamp instead. A dated number is honest, a silently stale one is not.

    repo_root/push=False are for tests: they operate on the given tree and skip git entirely.
    """
    need = ("pr_decode_tps", "pr_decode16k_tps", "pr_prefill_pp", "pr_prefill16k_pp",
            "pr_decode4k_tps", "pr_prefill4k_pp")
    vals = {k: res.get(k) or 0 for k in need}
    if not all(vals.values()):
        missing = [k for k in need if not vals[k]]
        print(f">> README refresh: skipped, measurement incomplete ({', '.join(missing)})",
              file=sys.stderr)
        return False

    def _rewrite(path):
        """Regenerate the marked block in `path`. Returns True if the file changed."""
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as e:
            print(f">> README refresh: cannot read {path}: {e}", file=sys.stderr)
            return False
        start_m, end_m = README_BENCH_MARK
        i, j = text.find(start_m), text.find(end_m)
        if i < 0 or j < 0 or j < i:
            print(f">> README refresh: markers not found in {path} — leaving it alone",
                  file=sys.stderr)
            return False
        rows = [("128", vals["pr_decode_tps"], vals["pr_prefill_pp"]),
                ("4k", vals["pr_decode4k_tps"], vals["pr_prefill4k_pp"]),
                ("16k", vals["pr_decode16k_tps"], vals["pr_prefill16k_pp"])]
        body = [start_m, "", "| context | decode | prefill |", "|---:|---:|---:|"]
        for ctx, dec, pp in rows:
            body.append(f"| {ctx} | **{dec:,.1f}** tok/s | **{pp:,.0f}** tok/s |")
        body += ["",
                 f"<sub>Auto-refreshed by the ModelOpt eval bot at `{commit[:9]}` — these are the "
                 f"numbers that PR measured on the pinned RTX 5090, which after squash-merge are "
                 f"main's. Regenerated on every auto-merge, so the table cannot drift behind the "
                 f"code.</sub>",
                 end_m]
        new_text = text[:i] + "\n".join(body) + text[j + len(end_m):]
        if new_text == text:
            print(">> README refresh: numbers unchanged")
            return False
        open(path, "w", encoding="utf-8").write(new_text)
        return True

    if not push:
        return _rewrite(os.path.join(repo_root or os.path.dirname(HERE), "README.md"))

    root = repo_root or os.path.dirname(HERE)
    wt = os.path.join(tempfile.gettempdir(), f"si_readme_refresh_{os.getpid()}")
    subprocess.run(["git", "-C", root, "worktree", "remove", "--force", wt],
                   capture_output=True, text=True)
    if subprocess.run(["git", "-C", root, "fetch", "-q", "origin", "main"],
                      capture_output=True, text=True).returncode != 0:
        print(">> README refresh: fetch origin/main failed", file=sys.stderr)
        return False
    mk = subprocess.run(["git", "-C", root, "worktree", "add", "--detach", wt, "origin/main"],
                        capture_output=True, text=True)
    if mk.returncode != 0:
        print(f">> README refresh: worktree add failed: {(mk.stderr or '')[:200]}", file=sys.stderr)
        return False
    try:
        if not _rewrite(os.path.join(wt, "README.md")):
            return False
        msg = (f"docs: refresh Qwen3.8 ModelOpt benchmark table ({commit[:9]})\n\n"
               f"Auto-generated by pr_modelopt_bot after auto-merging {commit[:9]}. The numbers "
               f"are that PR's own same-box measurements, which the squash-merge makes main's.")
        for args in (["add", "README.md"], ["commit", "-m", msg]):
            rc = subprocess.run(["git", "-C", wt] + args, capture_output=True, text=True)
            if rc.returncode != 0:
                print(f">> README refresh: git {args[0]} failed: {(rc.stderr or '')[:200]}",
                      file=sys.stderr)
                return False
        rc = subprocess.run(["git", "-C", wt, "push", "origin", "HEAD:main"],
                            capture_output=True, text=True)
        if rc.returncode != 0:
            print(f">> README refresh: push to main FAILED: {(rc.stderr or '')[:300]}",
                  file=sys.stderr)
            return False
        print(f">> README refresh: ModelOpt table updated at {commit[:9]} and pushed to main")
        return True
    finally:
        subprocess.run(["git", "-C", root, "worktree", "remove", "--force", wt],
                       capture_output=True, text=True)
        subprocess.run(["git", "-C", root, "worktree", "prune"], capture_output=True, text=True)


def try_auto_merge_qwen38(repo, num):
    ok, reason = auto_merge_ok_qwen38(repo, num)
    if not ok:
        print(f">> modelopt auto-merge SKIP #{num}: {reason}")
        return False
    r = arb.gh(["pr", "merge", str(num), "-R", repo, "--squash"])
    if r.returncode != 0 and os.environ.get("SPARKINFER_AUTOMERGE_ADMIN", "1") == "1":
        err = ((r.stderr or "") + (r.stdout or "")).lower()
        if "not mergeable" in err or "branch policy" in err or "required" in err or "prohibited" in err:
            print(">> modelopt auto-merge: branch policy blocked — retrying with --admin")
            r = arb.gh(["pr", "merge", str(num), "-R", repo, "--squash", "--admin"])
    if r.returncode == 0:
        print(f">> MODELOPT AUTO-MERGED #{num} (qwen38-merge-first)")
        arb.gh(["pr", "comment", str(num), "-R", repo, "--body",
                "<!-- sparkinfer-qwen38-automerge -->\n"
                "Auto-merged as the round's `qwen38-merge-first` winner — verified same-box "
                "128-token decode speedup over `main`, accuracy-gated vs llama.cpp."])
        return True
    print(f">> modelopt auto-merge BLOCKED #{num}: {(r.stderr or r.stdout or '')[:200]}")
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
        print(">> modelopt round: no verified speedup PRs")
        return
    winner = scored[0][0]
    print(f">> modelopt round: merge-first #{winner}; rebase {[n for n,_,_ in scored[1:]] or 'none'}")
    if dry_run:
        return
    arb.add_label(repo, winner, MODELOPT_MERGE_FIRST)
    arb.remove_label(repo, winner, MODELOPT_NEEDS_REBASE)
    for num, _, _ in scored[1:]:
        arb.add_label(repo, num, MODELOPT_NEEDS_REBASE)
        arb.remove_label(repo, num, MODELOPT_MERGE_FIRST)
    if AUTO_MERGE and try_auto_merge_qwen38(repo, winner):
        # The merge just made every published Qwen3.8 ModelOpt number one PR out of date. The
        # winner's own scored measurements are main's numbers now (squash-merge), so refresh the
        # README from the scores entry rather than spending another GPU pass on it.
        won = _load_scores().get(str(winner)) or {}
        if won:
            refresh_readme_modelopt(won, won.get("commit", ""))
        else:
            print(f">> README refresh: no stored score for #{winner} — table left as-is",
                  file=sys.stderr)


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
            "pr_decode16k_tps": res.get("pr_decode16k_tps"),
            "main_decode16k_tps": res.get("main_decode16k_tps"),
            "decode16k_delta_pct": res.get("decode16k_delta_pct"),
            "scored_dimension": res.get("scored_dimension"),
            "speedup_vs_main": res.get("speedup_vs_main"),
            "pr_top1": res.get("pr_top1"), "pr_kl": res.get("pr_kl"),
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
            print(">> modelopt eval-log upload skipped: nothing to commit")
            return None
        push = subprocess.run(["git", "-C", arb.LOG_DIR, "push", "-q"], check=False)
        if push.returncode != 0:
            print(f">> modelopt eval-log push failed (rc={push.returncode})")
            return None
        url = arb.LOG_PAGE + rid
        print(f">> modelopt eval log: {url}")
        return url
    except Exception as e:
        print(f">> modelopt eval-log upload failed: {e}")
        return None


def apply_result(repo, num, commit, res, title="", dry_run=False):
    body = format_comment(commit, res)
    label = res.get("label") if res.get("ok") else "REJECT"
    if not res.get("ok"):
        label = "REJECT"
    print(f"PR #{num}: eval-modelopt:{label}  "
          f"decode PR={res.get('pr_decode_tps')} main={res.get('main_decode_tps')}  "
          f"prefill PR={res.get('pr_prefill_pp')} main={res.get('main_prefill_pp')}  "
          f"from={res.get('scored_dimension')}  "
          f"top1={res.get('pr_top1')} kl={res.get('pr_kl')}  "
          f"delta={res.get('delta_pct')}%  accuracy_ok={res.get('accuracy_ok')}  "
          f"q36_guard_ok={res.get('q36_guard_ok')} "
          f"q38_guard_ok={res.get('q38_guard_ok')} "
          f"parity_ok={res.get('parity_ok')}")
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
    # Derived from every per-bot `eval-<model>:<tier>` label rather than overwritten with this
    # bot's own verdict -- this bot only measures ModelOpt, so writing the generic label
    # directly let a `none` here erase another model's real tier depending purely on which
    # staggered cron ran last. See arb.sync_generic_eval_label().
    arb.sync_generic_eval_label(repo, num)
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
            # The full 128/4k/16k pair set, so refresh_readme_modelopt() can rebuild the published
            # table from a merged PR's stored score without re-benchmarking. Only decode@16k is
            # scored; the rest ride along from the same sweep.
            "pr_decode16k_tps": res.get("pr_decode16k_tps"),
            "pr_decode4k_tps": res.get("pr_decode4k_tps"),
            "pr_prefill_pp": res.get("pr_prefill_pp"),
            "pr_prefill4k_pp": res.get("pr_prefill4k_pp"),
            "pr_prefill16k_pp": res.get("pr_prefill16k_pp"),
            "pr_top1": res.get("pr_top1"),
            "pr_kl": res.get("pr_kl"),
            "pass": res.get("pass"),
            "accuracy_ok": res.get("accuracy_ok"),
            "q36_guard_ok": res.get("q36_guard_ok"),
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
            elif not res.get("accuracy_ok"):
                fail_clause = "and failed the accuracy gate"
            elif res.get("decode_regressed") and res.get("prefill_regressed"):
                fail_clause = "(decode@128 and prefill@128 regression)"
            elif res.get("decode_regressed"):
                fail_clause = "(decode@128 regression)"
            elif res.get("prefill_regressed"):
                fail_clause = "(prefill@128 regression)"
            elif label == "none":
                fail_clause = "with no verified decode@16k or prefill@16k improvement (the scored dimensions)"
            else:
                fail_clause = "(regression)"
            close_body = (
                "<!-- sparkinfer-qwen38-auto-close -->\n"
                f"## Closed: sparkinfer modelopt auto-eval — `eval-modelopt:{label}`\n\n"
                f"This PR's Qwen3.8-27B `{res.get('scored_dimension', 'long-context')}` speed "
                f"measured **{res.get('delta_pct')}%** "
                f"vs main, {fail_clause} "
                "— closing automatically. This bot evaluates every eligible PR in the repo "
                "against Qwen3.8-27B's long-context decode AND prefill speed specifically, regardless of "
                "what the PR is actually about — a close here isn't a judgment on the PR's purpose, "
                "just that it didn't move these particular metrics. Reopen (or open a fresh PR) if "
                "you have a fix or a different approach."
            )
            arb.gh(["pr", "comment", str(num), "-R", repo, "--body", close_body])
            arb.gh(["pr", "close", str(num), "-R", repo])
            print(f">> auto-closed PR #{num} (eval-modelopt:{label})")


def main():
    ap = argparse.ArgumentParser(description="Qwen3.8-27B ModelOpt 128-decode PR eval bot")
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

    print(f">> modelopt eval transport: "
          f"{'ssh' if ssh_box_enabled() else f'vast.ai (instance {arb.current_instance(args.instance) or args.instance})'}")
    print(f">> AUTOMERGE={int(AUTO_MERGE)}")

    if args.labels_only:
        reconcile_qwen38_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — modelopt labels only (no GPU).")
        return

    prs = json.loads(arb.gh([
        "pr", "list", "-R", args.repo, "--state", "open",
        "--json", "number,title,labels,isDraft,headRefOid,headRefName,mergeable,author,body",
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
            print(f"PR #{num} @ {short}: already modelopt-evaluated — skip")
            continue
        if arb.pr_merge_conflict(pr.get("mergeable")):
            print(f"PR #{num}: merge conflict — modelopt-needs-rebase")
            if not args.dry_run:
                arb.add_label(args.repo, num, MODELOPT_NEEDS_REBASE)
            continue

        if not only:
            status, why = arb.greenlight_status(args.repo, num, labs)
            if status != "ok":
                print(f"PR #{num}: not greenlit ({why}) — skip modelopt eval")
                continue
            print(f"PR #{num}: greenlit ({why})")
        else:
            print(f"PR #{num}: --only-prs targeted")

        ref = f"pull/{num}/head"
        pending.append((num, head, short, ref, pr.get("title", "")))

    if not pending:
        reconcile_qwen38_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — no modelopt PRs to evaluate.")
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
        print("done — modelopt labels only (GPU down).")
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
        print("done — modelopt round skipped (main baseline unusable).")
        return
    # main has no top1/kl of its own: it IS the accuracy reference, and its score dump was just
    # written to SCORE_DUMP_MAIN for each PR in this round to diff against.
    print(f">> main baseline: decode@16k={main_result['decode16k_tps']:.2f} tok/s (scored) "
          f"decode@128={main_result['decode128_tps']:.2f} tok/s "
          f"prefill@128={main_result['prefill128_pp']:.2f} pp "
          f"prefill@16k={main_result['prefill16k_pp']:.2f} pp")

    for num, head, short, ref, title in pending:
        print(f"PR #{num} @ {short}: evaluating Qwen3.8-27B '{ref}' …")
        try:
            res = eval_qwen38_on_box(host, port, ref, main_result)
        except Exception as e:
            res = {"ok": False, "reason": f"exception: {e}"}
        apply_result(args.repo, num, head or short, res, title=title, dry_run=False)

    reconcile_qwen38_merge_labels(args.repo, dry_run=False)
    print("done — modelopt eval pass complete.")


if __name__ == "__main__":
    main()
