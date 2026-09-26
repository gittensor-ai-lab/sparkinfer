#!/usr/bin/env python3
"""sparkinfer Ternary-Bonsai-2-27B PR auto-evaluator.

Added 2026-09-24 (issue #1138). Ternary-Bonsai-2-27B landed on main in #1124, but nothing measured
it: a PR that sped it up scored `none` on the Muse Glimmer and Qwen3.8 bots -- which measure other
models -- and was auto-closed by them. #1139 (1.94x prefill@128 on this model) was the first such PR.

Sibling of pr_museglimmer_bot.py (the GGUF context sweep and concurrency ladder) and
pr_qwen38_bot.py (the differential accuracy gate, merge-ref evaluation and harness pinning). Both
are copied rather than reinvented, so the three bots score the same way.

Scoring, same-box PR-vs-main on the single pinned GPU, default (folded) loader arm -- the one the
server uses, `SPARKINFER_BONSAI_NATIVE` unset:
  1. Speed -- decode AND prefill at ctx 128/512/4k/16k/32k (SCORED_CTXS, #1138's list), plus
              concurrent decode at c2..c32. Every axis is also a no-regression floor at
              REGRESS_TOL; otherwise the tier is the best measured delta. Same buckets as the
              sibling bots.
  2. Accuracy, three gates, because this model's quirks make each of them blind to something:
       a. Differential teacher-forced score -- qwen3_gguf_score on the PR build and on main over the
          same tokens, compared by bench/scripts/accuracy_compare_pair.py. llama.cpp cannot read
          PTQ1_0 (GGML type 143), so there is no external reference; this catches a PR that changes
          the model's numerics. The folded path is NOT bit-deterministic across processes (PPL on
          one passage moved 9.50-9.68 between loads of one build), so the bars are set from a
          measured main-against-main spread, not copied from the Qwen3.8 bot's 0.99/0.01.
       b. Prefill path -- qwen3_gguf_score drives forward_token and never enters batched prefill,
          so a prefill change is invisible to (a). qwen3_gguf_prefill_check compares batched
          prefill against the token loop inside one build; it is run PF_RUNS times per prefix on
          each side, because on this model one build reads 13-16/16 top-1 across runs. Gated on the
          PR's mean falling clearly outside main's, never on a single reading.
       c. eval/bonsai_regression.py -- tensors, score, generate, serve: absolute broken-path checks
          (trit order, the v-head regrouping, native against folded, a row outliving its batch).
          Gated per check: a check main also fails in the same round is reported but cannot
          reject every PR, and the other checks still gate. The serve check decides on 2 of up
          to 3 trials, because one trial fails about one time in five on a sound build.
  3. Cross-model no-regression guards @ 32k, decode + prefill: Qwen3.6-35B-A3B, the ModelOpt and
     unsloth Qwen3.8-27B NVFP4 checkpoints, and Muse Glimmer. The Muse and Qwen3.8 bots skip a PR
     declared for Ternary-Bonsai-2-27B alone (arb.model_skip_reason), so these guards are the only
     check such a PR gets against those models. Absent checkpoint -> SKIPPED and said so;
     unmeasured -> infra, retried next round; measured regression -> REJECT.

Policy, by explicit decision 2026-09-24: tiers mirror to the generic `eval:*` label (as the sibling
bots do); cron hourly at :15, between Muse's :00 and Qwen3.8's :30. Auto-merge is
SPARKINFER_BONSAI_AUTOMERGE=1 in .env.eval, turned on the same day once #1139 had validated the bot,
matching the sibling bots' live policy -- with one extra guard they lacked: it merges only the exact
head commit this bot scored (auto_merge_ok_bonsai).

Closing, by explicit decision 2026-09-26, so this bot works like its siblings:
  * a measured REJECT closes the PR (SPARKINFER_BONSAI_AUTOCLOSE=0 turns closing off);
  * a measured `none` closes it only when the PR declares Ternary-Bonsai-2-27B and nothing else,
    and no other bot scored it a speedup or made it merge-first (arb.none_may_close, shared with
    the siblings since 2026-09-26). This bot also evaluates every undeclared PR, and most of those
    are aimed at another model and legitimately score `none` here -- #768 and #1082 were closed by
    a sibling bot for exactly that;
  * nothing closes over a head that moved after the commit that was measured, nor over a `hold`
    or a draft made while the round ran (arb.verdict_close_blocker);
  * a PR routed to this model with no commits for BONSAI_STALE_DAYS, which has also waited on its
    author that long (arb.AuthorWaitClock), closes as stale, as the siblings' own stale close does
    for theirs (arb.stale_close_skip_reason) -- unless it is greenlit and still waiting for this
    bot's first verdict on its head, a verified speedup waiting for the merge-first PR to merge,
    never measured here and not greenlit (left to the daily Action), or another bot's verified
    speedup;
  * a run that FAILED (build error, crash) never closes: its verdict comment says why.

  python eval/pr_bonsai_bot.py --only-prs 1139 --reeval --no-post

Never rents a GPU. Shares /tmp/sparkinfer_bot.lock with the sibling bots via its cron wrapper
(run_bonsai_pr_cron.sh).
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import shlex
import statistics
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

import pr_eval_bot as arb  # noqa: E402

SPEEDUP_LABELS = {"XL", "L", "M", "S", "XS"}
# Same tier-bucketing constants as the sibling bots -- copied verbatim, not reinvented.
SIG = 0.02
REGRESS_TOL = 0.98
BUCKETS = [(0.18, "XL"), (0.10, "L"), (0.06, "M"), (0.035, "S"), (SIG, "XS")]
IMPLAUSIBLE_GAIN = float(os.environ.get("BONSAI_IMPLAUSIBLE_GAIN", "20"))

EVAL_PREFIX = "eval-bonsai:"
BONSAI_MERGE_FIRST = "bonsai-merge-first"
BONSAI_NEEDS_REBASE = "bonsai-needs-rebase"
EVAL_SCHEMA_VERSION = "v1-ctx5-prefill-decode-cbdecode-diffacc-pfcheck-guards"
MARKER_RE = re.compile(
    r"<!-- sparkinfer-bonsai-eval:" + re.escape(EVAL_SCHEMA_VERSION) + r":([0-9a-f]+)(?:\s+(\{.*?\}))? -->",
    re.DOTALL,
)

# --- box paths ---
# Its own clone: this bot builds with -DBUILD_SERVER=ON (bonsai_regression.py's serve check), and
# the sibling bots' clones must not inherit that cache or have their build dirs switched under them.
# Under /root, not /workspace: on this box /workspace is Docker's overlayfs, where cargo cannot write
# the server's tokenizers-c archive -- "failed to build archive ... Bad address (os error 14)", every
# time (2026-09-24). The same build passes on /root.
REMOTE_REPO = os.environ.get("BONSAI_REMOTE_REPO", "/root/sparkinfer_bonsai")
BONSAI_GGUF = os.environ.get("BONSAI_GGUF",
                             "/root/workspace/models_bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
# prism-ml/Ternary-Bonsai-2-27B-gguf. Checked by size on every run (a sha256 of 6 GB costs a
# minute): a truncated download loads, and measures, as a different model.
BONSAI_GGUF_BYTES = int(os.environ.get("BONSAI_GGUF_BYTES", "5946648928"))
# The GGUF carries no tokenizer.json. The model is derived from Qwen3.8-27B and shares its
# vocabulary, and bonsai_regression.py already tokenizes with that checkpoint's tokenizer.
BONSAI_TOKENIZER_DIR = os.environ.get("BONSAI_TOKENIZER_DIR", "/root/workspace/models_qwen38")
# The checkpoint bonsai_regression.py's tensors check decodes weights against.
BONSAI_REFERENCE_DIR = os.environ.get("BONSAI_REFERENCE_DIR", "/root/workspace/models_qwen38")
BENCH_TOKENS = int(os.environ.get("BONSAI_BENCH_TOKENS", "128"))
BENCH_REPS = 5

SCORED_CTXS = [128, 512, 4096, 16384, 32768]
SCORED_CTX_LABEL = {128: "128", 512: "512", 4096: "4k", 16384: "16k", 32768: "32k"}
# Concurrent decode, same binary and invocation as the sibling bots. Median of CB_REPS complete
# runs per point (pr_qwen38_bot.py's cb_median): the single-sample version is how #1064 took an XL
# from one bad baseline reading.
CB_CONCS = [2, 4, 8, 16, 32]
CB_TOKENS = 256
CB_LONG_TOKENS = 8
CB_REPS = 3
CB_MAX_ATTEMPTS = 5
CB_DIM_FOR = {c: f"bonsai-cb-decode@c{c}" for c in CB_CONCS}
SCORING_DIMS = ([f"bonsai-{phase}@{SCORED_CTX_LABEL[c]}"
                 for c in SCORED_CTXS for phase in ("decode", "prefill")]
                + [CB_DIM_FOR[c] for c in CB_CONCS])
SCORING_DIM = SCORING_DIMS[0]

# The measuring instrument. A PR touching any of these is not evaluated, and every ref -- main
# included -- is built with main's copy (the HARNESS_PINNED block in _remote_script). Same policy
# as pr_qwen38_bot.py, widened to the accuracy tools this bot gates on.
HARNESS_PATHS = (
    "runtime/examples/qwen3_gguf_bench.cpp",
    "runtime/examples/qwen3_gguf_cb_bench.cpp",
    "runtime/examples/qwen3_gguf_score.cpp",
    "runtime/examples/qwen3_gguf_generate.cpp",
    "runtime/examples/qwen3_gguf_prefill_check.cpp",
    "runtime/examples/bonsai_inspect.cpp",
    "runtime/examples/qwen_checkpoint.h",
    "runtime/examples/qwen3_gguf_config.h",
    "eval/",
    "bench/scripts/",
)
_HARNESS_PIN = [p for p in HARNESS_PATHS if p != "eval/"] + ["eval/bonsai_regression.py"]

# (a) Differential score gate. Two loads of ONE build do not agree on this model: measured on the
# eval box 2026-09-24, main bb67474 against itself over eval_corpus.txt, eight pairs of processes:
#
#     top-1  0.959 - 0.978      KL  0.0084 - 0.0133      PPL ratio  0.995 - 1.005
#
# SPARKINFER_DETERMINISTIC=1 does not change that (a pair under it: top-1 0.976, PPL 3.700 / 3.691),
# so the spread is not the batched-prefill atomics that mode removes. The Qwen3.8 bot's 0.99/0.01
# would therefore reject main against itself -- it did, in the first validation round. The bars sit
# outside the measured spread with margin; the PPL ratio is the sharp one (4x the worst pair's
# deviation), since a broken path reads PPL in the hundreds on this model (the v-head bug: 139).
ACC_TOP1_BAR = float(os.environ.get("BONSAI_ACC_TOP1_BAR", "0.93"))
ACC_KL_BAR = float(os.environ.get("BONSAI_ACC_KL_BAR", "0.03"))
# PPL of the PR's dump over main's, same positions. Far tighter than top-1 across the pairs above, so
# a real quality loss shows here before it clears the top-1/KL noise.
ACC_PPL_RATIO = float(os.environ.get("BONSAI_ACC_PPL_RATIO", "1.02"))
ACC_TOPK = int(os.environ.get("BONSAI_ACC_TOPK", "128"))
# ~1,200 tokens rather than the siblings' 99-token eval_text.txt: with run-to-run noise in the
# forward pass, one flipped argmax out of 99 is a full percentage point of top-1.
ACC_TEXT = "bench/scripts/eval_corpus.txt"
SCORE_DUMP_MAIN = "/tmp/bonsai_score_main.txt"
SCORE_DUMP_PR = "/tmp/bonsai_score_pr.txt"

# (b) Prefill-path gate. 128 sits inside the fused quantized-B GEMM's M <= 512 window and 1024
# outside it, so both prefill GEMM arms are exercised. 64 continuation positions rather than the
# tool's default 16, so one flipped argmax moves top-1 by 1.6 points instead of 6.
PF_PREFIXES = [128, 1024]
PF_CONT = 64
PF_RUNS = int(os.environ.get("BONSAI_PF_RUNS", "3"))
PF_TRIES = 3   # attempts per run before that run counts as failed (a crash, not a hang)
PF_TEXT = "bench/scripts/bench_prompt_4k.txt"
# The PR's mean over PF_RUNS against main's mean over PF_RUNS. A broken batched prefill reads KL
# in whole nats and top-1 near zero; these margins sit well inside that and well outside the
# measured run-to-run spread.
PF_TOP1_DROP = float(os.environ.get("BONSAI_PF_TOP1_DROP", "0.10"))
PF_KL_RATIO = float(os.environ.get("BONSAI_PF_KL_RATIO", "3.0"))
PF_KL_ABS = float(os.environ.get("BONSAI_PF_KL_ABS", "0.05"))

# Cross-model guards, same env var names as the sibling bots so one .env.eval entry serves all.
Q36_GUARD_MODELS_DIR = os.environ.get("Q36_GUARD_MODELS_DIR", "/root/workspace/models36")
Q36_GUARD_MODEL_FILE = os.environ.get("Q36_GUARD_MODEL_FILE", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
Q36_GUARD_MODEL_REPO = os.environ.get("PRIMARY36_MODEL_REPO", "unsloth/Qwen3.6-35B-A3B-GGUF")
Q36_GUARD_TOK_REPO = os.environ.get("PRIMARY36_TOK_REPO", "Qwen/Qwen3.6-35B-A3B")
MODELOPT_GUARD_MODEL_DIR = os.environ.get("MODELOPT_MODEL_DIR", "/root/workspace/models_q38_modelopt")
UNSLOTH_GUARD_MODEL_DIR = os.environ.get("QWEN38_MODEL_DIR", "/root/workspace/models_qwen38")
MUSE_GUARD_GGUF = os.environ.get(
    "MUSEGLIMMER_GGUF",
    "/root/workspace/models_muse_glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf")
GUARD_CTXS = [32768]
GUARD_CTX_LABEL = {128: "128", 512: "512", 4096: "4k", 16384: "16k", 32768: "32k"}
# key in _parse_remote's output -> (wire tag, display name, what to do when absent)
GUARDS = (
    ("guard36", "GUARD36", "qwen3.6"),
    ("guardmo", "GUARDMO", "modelopt qwen3.8"),
    ("guardun", "GUARDUN", "unsloth qwen3.8"),
    ("guardmg", "GUARDMG", "muse glimmer"),
)

AUTO_MERGE = os.environ.get("SPARKINFER_BONSAI_AUTOMERGE") == "1"
# On unless switched off (module docstring, "Closing"): the siblings close unconditionally.
AUTO_CLOSE = os.environ.get("SPARKINFER_BONSAI_AUTOCLOSE", "1") != "0"
STALE_DAYS = float(os.environ.get("BONSAI_STALE_DAYS", "1"))
AUTOMERGE_BLOCK = {
    "copycat", "copycat-warn", "flagged:gaming", "penalty", "needs-benchmark",
    BONSAI_NEEDS_REBASE, arb.REEVALUATE_LABEL, arb.HOLD_LABEL, *arb.REGRESSION_LABELS,
}

SCORES_FILE = os.path.expanduser(
    os.environ.get("BONSAI_SCORES_FILE", "~/.sparkinfer_bonsai_scores.json")
)
# A concurrency width that completes on main but not on the PR, after every in-run attempt, is
# probably the PR's doing -- but one round is not enough to say so. The first time, nothing is
# posted and the next round measures again; the same failure on the same commit then REJECTs.
STRIKES_FILE = os.path.expanduser(
    os.environ.get("BONSAI_STRIKES_FILE", "~/.sparkinfer_bonsai_strikes.json")
)
# When each PR began waiting on its author, per head: the stale close's clock (arb.AuthorWaitClock).
AUTHOR_WAIT_FILE = os.path.expanduser(
    os.environ.get("BONSAI_AUTHOR_WAIT_FILE", "~/.sparkinfer_bonsai_author_wait.json")
)
# PRs the bot gave up on this run (its own errors): the run then exits 3, so they are not silent.
GAVE_UP = set()
STRIKES_TO_REJECT = 2
BOX_FAULT_STRIKES = arb.BOX_FAULT_STRIKES   # a box-shaped failure recurring at one commit

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


def _load_strikes():
    return arb._load_strikes(STRIKES_FILE)


def record_strike(num, commit, keys):
    """One more round in which each `+`-joined key failed for this PR at this exact commit; returns
    the highest count. Counted per key (arb.record_strike), and a new commit starts again at 1."""
    return arb.record_strikes(STRIKES_FILE, num, commit, keys)


def clear_strikes(num):
    arb.clear_strikes(STRIKES_FILE, num)


def tier_from_gain(pr_tps: float, main_tps: float, metric: str = "decode"):
    """Return (label, delta_pct, pass_ok, reason). Identical to the sibling bots' tier_from_gain."""
    if main_tps <= 0:
        return "REJECT", 0.0, False, f"main {metric} baseline is 0"
    if pr_tps < REGRESS_TOL * main_tps:
        pct = 100.0 * (pr_tps - main_tps) / main_tps
        return "REJECT", round(pct, 1), False, (
            f"{metric} regression: {pr_tps:.2f} < {100 * REGRESS_TOL:.0f}% of main {main_tps:.2f}"
        )
    # A pass that stops computing also stops taking time (#1037: 3,300x "faster", emitting garbage).
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


def _check_model_guard(pr: dict, main: dict, key: str, model: str, tol: float = REGRESS_TOL):
    """No-regression check for ONE guarded model, decode + prefill at every context main measured.
    The sibling bots' implementation, unchanged. Returns (ok, [problems])."""
    problems = []
    if pr.get(f"{key}_failed") or main.get(f"{key}_failed") or not pr.get(key) or not main.get(key):
        problems.append(f"{model} guard measurement unavailable")
    pr_ctxs, main_ctxs = pr.get(key) or {}, main.get(key) or {}
    for ctx, main_vals in main_ctxs.items():
        label = GUARD_CTX_LABEL.get(ctx, str(ctx))
        pr_vals = pr_ctxs.get(ctx) or {}
        for metric in ("decode", "prefill"):
            base = main_vals.get(metric, 0)
            if base <= 0:
                continue
            cur = pr_vals.get(metric, 0)
            if cur <= 0:
                problems.append(f"{model} {metric}@{label}: PR measurement missing/zero "
                                f"(main {base:.1f}) — treated as regression")
                continue
            if cur < base * tol:
                pct = 100.0 * (cur - base) / base
                problems.append(f"{model} {metric}@{label}: {cur:.1f} < {100 * tol:.0f}% of main "
                                f"{base:.1f} ({pct:+.1f}%)")
    return (len(problems) == 0, problems)


def check_prefill_path(pr: dict, main: dict):
    """Gate 2b: batched prefill against the token loop, the PR's mean against main's.

    Returns (ok, problems, rows). A prefix main could not measure is reported as unavailable (infra,
    the caller retries); a PR run that failed where main's did not is a problem, since
    qwen3_gguf_prefill_check only fails when batched prefill refuses or crashes."""
    problems, rows = [], []
    for p in PF_PREFIXES:
        pr_runs = (pr.get("pfcheck") or {}).get(p) or []
        main_runs = (main.get("pfcheck") or {}).get(p) or []
        pr_fail = (pr.get("pfcheck_failed") or {}).get(p, 0)
        if not main_runs:
            problems.append(f"prefill-path check @{p}: main measurement unavailable")
            continue
        m_t = statistics.mean(t for t, _ in main_runs)
        m_k = statistics.mean(k for _, k in main_runs)
        t_bar = m_t - PF_TOP1_DROP
        k_bar = max(m_k * PF_KL_RATIO, m_k + PF_KL_ABS)
        pr_killed = (pr.get("pfcheck_failed_box") or {}).get(p, 0)
        # Beside runs that completed wrong, a killed one changes nothing: a box fault cannot fake a
        # wrong result, so the PR is judged on the runs it has (below).
        completed_fail = bool(pr_runs) and (
            statistics.mean(k for _, k in pr_runs) > k_bar or statistics.mean(t for t, _ in pr_runs) < t_bar)
        if pr_killed and not pr_fail and not completed_fail:
            # Runs the OOM killer took on every attempt, whose trigger may be anything on the box:
            # judged like any box fault of the PR's run (eval_bonsai_on_box's "pf-box" strike). Not
            # REJECTed, and not dropped either -- the gate judged on the runs left used to pass.
            problems.append(f"prefill-path check @{p}: {pr_killed} of {pr_killed + len(pr_runs)} runs "
                            f"killed on the PR build (exit 137)")
            rows.append({"prefix": p, "pr_runs": len(pr_runs), "main_runs": len(main_runs), "pr_top1": None,
                         "pr_kl": None, "main_top1": m_t, "main_kl": m_k,
                         "top1_bar": t_bar, "kl_bar": k_bar})
            continue
        if not pr_runs:
            problems.append(f"prefill-path check @{p}: every PR run failed "
                            f"(batched prefill refused or crashed)")
            rows.append({"prefix": p, "pr_runs": 0, "main_runs": len(main_runs), "pr_top1": None,
                         "pr_kl": None, "main_top1": m_t, "main_kl": m_k,
                         "top1_bar": t_bar, "kl_bar": k_bar})
            continue
        p_t = statistics.mean(t for t, _ in pr_runs)
        p_k = statistics.mean(k for _, k in pr_runs)
        rows.append({"prefix": p, "pr_runs": len(pr_runs), "main_runs": len(main_runs),
                     "pr_top1": p_t, "pr_kl": p_k, "main_top1": m_t, "main_kl": m_k,
                     "top1_bar": t_bar, "kl_bar": k_bar})
        if pr_fail:
            problems.append(f"prefill-path check @{p}: {pr_fail} of {pr_fail + pr_killed + len(pr_runs)} PR "
                            f"runs failed (batched prefill refused or crashed)")
        if p_k > k_bar:
            problems.append(f"prefill-path check @{p}: mean KL {p_k:.4f} > bar {k_bar:.4f} "
                            f"(main {m_k:.4f})")
        if p_t < t_bar:
            problems.append(f"prefill-path check @{p}: mean top-1 {p_t:.3f} < bar {t_bar:.3f} "
                            f"(main {m_t:.3f})")
    return (len(problems) == 0, problems, rows)


def bonsai_evaluated_commits(repo, num):
    """Head commits that already carry a real verdict. A marker with label null does not count."""
    return arb.evaluated_commits_from(repo, num, MARKER_RE, "sparkinfer bonsai auto-eval")


def _verdict_heads(repo, num):
    """The heads this bot counts as measured: carrying its verdict marker AND recorded as the PR's
    latest verdict (arb.recorded_verdict_heads). None when GitHub did not answer."""
    return arb.recorded_verdict_heads(bonsai_evaluated_commits(repo, num), _load_scores().get(str(num)))


def strip_bonsai_eval_labels(repo, num):
    arb.strip_own_tier_labels(repo, num, EVAL_PREFIX)


def _pr_last_activity_ts(repo, num):
    """Last real author activity (most recent commit's committedDate), not PR updatedAt: bot
    comments and labels bump updatedAt. The sibling bots' helper."""
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


def _remeasure_state(repo, num, head, labs, main_now):
    """Is an already-evaluated head owed a re-measure onto today's main? True only when auto_merge_ok_bonsai
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
    ok, why = auto_merge_ok_bonsai(repo, num, require_merge_first=False)
    if not ok and why == arb.PR_UNREADABLE:
        return None
    return not ok and arb.refused_only_for_stale_main(why)


def _remeasure_against_new_main(repo, num, head, labs, main_now):
    """_remeasure_state, for the selection: only a definite yes spends GPU on it."""
    return _remeasure_state(repo, num, head, labs, main_now) is True


def _waits_for_the_winner(pr, labs, head, main_now):
    """A verified speedup at this head, sent to bonsai-needs-rebase only because another PR won
    merge-first. Until that PR merges and main moves there is nothing to rebase onto, so the wait
    is the merge's, not the author's; from then on the rebase is the author's (CONTRIBUTING), and
    the stale clock starts. A conflict, or any other block label, is the author's at once. Kept
    too when today's main is unknown."""
    entry = _load_scores().get(str(pr["number"])) or {}
    if (BONSAI_NEEDS_REBASE not in labs or labs & (AUTOMERGE_BLOCK - {BONSAI_NEEDS_REBASE})
            or entry.get("commit") != head or entry.get("label") not in SPEEDUP_LABELS or not entry.get("pass")
            or arb.pr_merge_conflict(pr.get("mergeable"))
            or arb.strike_count(STRIKES_FILE, pr["number"], head, "conflict")):
        return False
    return not main_now or not arb.scored_against_stale_main(entry, main_now)


def close_stale_bonsai_prs(repo, prs, dry_run=False):
    """Close PRs routed to this model with no author commit in STALE_DAYS+ days -- the siblings'
    stale close, limited like theirs now are to the bot's own PRs (arb.stale_close_skip_reason).
    Idle means waiting on its author for STALE_DAYS as well (arb.AuthorWaitClock): a PR the bot kept
    waiting is not closed the round it is handed back. STALE_DAYS <= 0 turns the stale close off, and
    so does SPARKINFER_BONSAI_AUTOCLOSE=0, which turns every close off (module docstring)."""
    closed = set()
    if arb.stale_close_disabled(STALE_DAYS) or not AUTO_CLOSE:
        return closed
    now = time.time()
    main_now = None
    clock = arb.AuthorWaitClock(AUTHOR_WAIT_FILE, [p["number"] for p in prs], now, record=not dry_run)
    for pr in prs:
        num = pr["number"]
        if arb.stale_close_skip_reason(pr, "bonsai", EVAL_PREFIX):
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
            note(f"PR #{num}: idle {age_days:.1f}d but still waiting for its first bonsai verdict — kept open")
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
        arb.gh(["pr", "comment", str(num), "-R", repo, "--body",
                "<!-- sparkinfer-bonsai-auto-close-stale -->\n"
                f"## Closed: stale — no commits in {age_days:.1f} days\n\n"
                f"This PR has had no new commits in over {STALE_DAYS:g} days, so it is closed to "
                "keep the Ternary-Bonsai-2-27B eval queue to work that is moving. Nothing is wrong "
                "with it for that reason alone: push your latest work and reopen it whenever you "
                "are ready, and it is evaluated again on the next round."])
        arb.gh(["pr", "close", str(num), "-R", repo])
    clock.save()
    return closed


def resolve_ssh(instance_id: int):
    """Return (host, port) for the pinned box. Same logic as the sibling bots."""
    if ssh_box_enabled():
        ep = ssh_box_endpoint()
        if not ep:
            raise RuntimeError("EVAL_TRANSPORT=ssh but EVAL_SSH_HOST unset")
        return ep
    key = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/speedy"))
    os.environ.setdefault("SSH_KEY", key)
    iid = arb.current_instance(instance_id) or instance_id
    raw = subprocess.run(["vastai", "show", "instance", str(iid), "--raw"],
                         capture_output=True, text=True, timeout=60)
    if raw.returncode != 0 or not (raw.stdout or "").strip():
        raise RuntimeError(f"vastai show instance {iid} failed: {(raw.stderr or '')[:200]}")
    info = json.loads(raw.stdout)
    ip = (info.get("public_ipaddr") or "").strip()
    m = (info.get("ports") or {}).get("22/tcp") or [{}]
    port = int((m[0] or {}).get("HostPort") or 0)
    if info.get("actual_status") != "running" or not ip or not port:
        raise RuntimeError(f"pinned instance {iid} not SSH-ready (status={info.get('actual_status')})")
    return ip, port


def ssh_run(host, port, cmd, timeout=7200, stdin_data=None, via_stdin=False):
    key = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/speedy"))
    user = ssh_box_user() if ssh_box_enabled() else "root"
    remote = ["bash", "-s"] if via_stdin else [cmd]
    return subprocess.run(
        ["ssh", "-i", key, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new",
         "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=40",
         "-p", str(port), f"{user}@{host}", *remote],
        # errors="replace": a stray non-UTF-8 byte in a build log raised after the whole run, and the
        # PR was retried every round with nothing posted.
        capture_output=True, text=True, errors="replace", timeout=timeout,
        input=cmd if via_stdin else stdin_data,
    )


_EXPLICIT_FAIL_MARKERS = ("HARNESS_TOUCHED", "BUILD_FAILED", "SCORE_FAILED", "TOKENIZE_FAILED", "HARNESS_PIN_FAILED",
                          "MODEL_CHECK_FAILED", "MERGE_CONFLICT", "BASE_AHEAD")
# Markers naming the box rather than the ref: missing model files, the box's tokenizer, a fetch.
_INFRA_MARKERS = ("RETRYABLE_INFRA_FAILURE", "MODEL_CHECK_FAILED", "TOKENIZE_FAILED",
                  "HARNESS_PIN_FAILED")


def _crash_reason(*outputs: str) -> str | None:
    """The ERR-trap diagnostic, or one of the script's explicit *_FAILED markers (an explicit
    `exit 1` does not fire the ERR trap). Same extraction as the sibling bots."""
    lines = "\n".join(o or "" for o in outputs).splitlines()
    for i, line in enumerate(lines):
        if line.startswith("REMOTE_SCRIPT_FAILED "):
            extra = lines[i + 1].strip() if i + 1 < len(lines) else ""
            return line.strip() + (f" | gpu: {extra}" if extra else "")
    for i, line in enumerate(lines):
        marker = next((m for m in _EXPLICIT_FAIL_MARKERS if line.startswith(m)), None)
        if not marker:
            continue
        err = arb.first_build_error(lines[i + 1:i + 80])
        if err:
            return f"{marker}: {err}"
        tail = " | ".join(l.strip() for l in lines[i + 1:i + 3] if l.strip())
        return marker + (f": {tail}" if tail else "")
    return None


def _is_infra_failure(stdout: str, stderr: str) -> bool:
    """Is a failed run the box's fault rather than the ref's? Those are retried next round with
    nothing posted: an infrastructure fault must never be charged to a PR.

      RETRYABLE_INFRA_FAILURE  the GPU did not drain (another tenant holds it), a git fetch failed
      *_CHECK/TOKENIZE/PIN     the box is missing a model or its tokenizer, or could not pin
      exit=137                 SIGKILL -- the host OOM killer, whose trigger may be another tenant
      no diagnostic at all     the ssh session or the shell died underneath the script
    """
    combined = (stdout or "") + "\n" + (stderr or "")
    if any(m in combined for m in _INFRA_MARKERS):
        return True
    crash = _crash_reason(stdout, stderr)
    if crash is None:
        return True
    return "exit=137" in crash


def _ssh_run_resilient(host, port, script: str, label: str):
    """One automatic retry on an infrastructure-shaped failure, as the sibling bots do."""
    r = ssh_run(host, port, script, via_stdin=True)
    if r.returncode != 0 and _is_infra_failure(r.stdout, r.stderr):
        print(f">> {label}: infrastructure-shaped failure — retrying the whole measurement once")
        r = ssh_run(host, port, script, via_stdin=True)
    return r


def _remote_script(ref: str, role: str = "pr", onto: str | None = None) -> str:
    """Bash run on the eval box: checkout, pin the harness, build, then every measurement.
    Identical for main and a PR; `role` only picks the score dump path and whether the
    differential compare runs (main is the reference it compares against).

    A PR (`ref` = pull/<n>/head) is measured MERGED onto `onto`, the exact main commit the round's
    baseline measured (arb.merged_checkout_script), and its harness is pinned from that same
    commit -- so the PR and the baseline differ by the PR's change and nothing else, even if main
    moves mid-round or GitHub's own merge ref is stale."""
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
    gguf = shlex.quote(BONSAI_GGUF)
    tok_dir = shlex.quote(BONSAI_TOKENIZER_DIR)
    ref_dir = shlex.quote(BONSAI_REFERENCE_DIR)
    dump_self = shlex.quote(SCORE_DUMP_MAIN if role == "main" else SCORE_DUMP_PR)
    dump_main = shlex.quote(SCORE_DUMP_MAIN)
    is_pr = "1" if role == "pr" else "0"
    harness_pin = " ".join(shlex.quote(p.rstrip("/")) for p in _HARNESS_PIN)
    sweep_args = " ".join(f"{c} {BENCH_REPS}" for c in SCORED_CTXS)
    ctx_list = " ".join(str(c) for c in SCORED_CTXS)
    guard_args = " ".join(f"{c} {BENCH_REPS}" for c in GUARD_CTXS)
    guard_list = " ".join(str(c) for c in GUARD_CTXS)
    cb_concs = " ".join(str(c) for c in CB_CONCS)
    pf_prefixes = " ".join(str(p) for p in PF_PREFIXES)
    return f"""
set -euo pipefail
trap 'rc=$?; ln=$LINENO; reason=""; \\
  case $rc in \\
    137) reason="likely OOM-killed (SIGKILL)" ;; \\
    139) reason="likely segfault (SIGSEGV)" ;; \\
    134) reason="likely abort (SIGABRT)" ;; \\
    124) reason="likely timeout" ;; \\
  esac; \\
  echo "REMOTE_SCRIPT_FAILED line=$ln exit=$rc reason=$reason" >&2; \\
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader >&2 2>/dev/null || true' ERR

# Poll GPU memory down to near-empty before each heavy load (pr_museglimmer_bot.py, 180 s). A GPU
# that will not drain is someone else's work, never the PR's: say so and let the bot retry.
wait_gpu_clear() {{
  local tries=0 used
  while [ "$tries" -lt 180 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "$used" ] && [ "$used" -lt 1024 ] 2>/dev/null && return 0
    sleep 1
    tries=$((tries + 1))
  done
  echo "RETRYABLE_INFRA_FAILURE GPU still holding ${{used:-unknown}} MiB after ${{tries}}s — refusing to start a load that would OOM" >&2
  return 1
}}

# bonsai_regression.py's serve check starts servers from this build. Reap only those -- matched by
# /proc/<pid>/exe under this clone, never by name: other sparkinfer_server processes on the box
# are not ours.
reap_our_servers() {{
  local p e
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    [ "$p" = "$$" ] && continue
    e=$(readlink "/proc/$p/exe" 2>/dev/null) || continue
    case "$e" in "$REPO"/build/*) echo "reaping stray $e (pid $p)" >&2; kill -9 "$p" 2>/dev/null || true ;; esac
  done
}}

export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:/usr/local/bin:$HOME/.cargo/bin:$PATH
export CUDA_HOME=${{CUDA_HOME:-/usr/local/cuda-13.0}}
{arb.round_guard_sh("bonsai")}
REPO={repo}
GGUF={gguf}
TOK_DIR={tok_dir}
REF_DIR={ref_dir}
DUMP_SELF={dump_self}
DUMP_MAIN={dump_main}
IS_PR={is_pr}
Q36_GUARD_MODELS_DIR={shlex.quote(Q36_GUARD_MODELS_DIR)}
Q36_GUARD_MODEL_FILE={shlex.quote(Q36_GUARD_MODEL_FILE)}
Q36_GUARD_MODEL_REPO={shlex.quote(Q36_GUARD_MODEL_REPO)}
Q36_GUARD_TOK_REPO={shlex.quote(Q36_GUARD_TOK_REPO)}
MODELOPT_GUARD_MODEL_DIR={shlex.quote(MODELOPT_GUARD_MODEL_DIR)}
UNSLOTH_GUARD_MODEL_DIR={shlex.quote(UNSLOTH_GUARD_MODEL_DIR)}
MUSE_GUARD_GGUF={shlex.quote(MUSE_GUARD_GGUF)}

# The box, not the ref: a missing or truncated model fails the round as infrastructure.
if [ ! -f "$GGUF" ] || [ "$(stat -c %s "$GGUF")" != "{BONSAI_GGUF_BYTES}" ]; then
  echo "MODEL_CHECK_FAILED Ternary-Bonsai-2-27B GGUF missing or not {BONSAI_GGUF_BYTES} bytes: $GGUF" >&2
  exit 1
fi
test -f "$TOK_DIR/tokenizer.json" || {{ echo "MODEL_CHECK_FAILED no tokenizer.json in $TOK_DIR" >&2; exit 1; }}
# main is the accuracy reference for every PR in the round; never let a stale dump stand in for it.
[ "$IS_PR" = "1" ] || rm -f "$DUMP_MAIN"
reap_our_servers

# A network failure is the box's, never the PR's: RETRYABLE, not the ERR trap.
if [ ! -d "$REPO/.git" ]; then
  git clone -q https://github.com/gittensor-ai-lab/sparkinfer.git "$REPO" || {{
    echo "RETRYABLE_INFRA_FAILURE git clone failed" >&2; exit 1; }}
fi
cd "$REPO"
git remote set-url origin https://github.com/gittensor-ai-lab/sparkinfer.git 2>/dev/null || true
{checkout}
echo "STAGE start $(date +%s)"

# Pin the measuring instrument for every ref, main included (HARNESS_PATHS) -- from the main commit
# this ref is measured against, so a PR and its baseline always share one ruler.
timeout 600 git fetch -q origin main || {{ echo "RETRYABLE_INFRA_FAILURE git fetch main failed" >&2; exit 1; }}
git checkout -q {base_q} -- {harness_pin} 2>/dev/null || {{
  echo "HARNESS_PIN_FAILED -- could not take the harness from {base}" >&2
  exit 1
}}
echo "HARNESS_PINNED $(git rev-parse --short {base_q})"

# A PRIVATE TMPDIR for nvcc. Its tmpxft_* intermediates in a shared /tmp get deleted by whatever
# else on the box tidies /tmp, which fails the build in files the PR never touched.
export TMPDIR="$REPO/.tmp-bonsai-bot"
mkdir -p "$TMPDIR" && rm -rf "${{TMPDIR:?}}"/* 2>/dev/null || true
echo "DISK_BEFORE_BUILD $(df -h "$REPO" | awk 'NR==2{{print $4}}') free"
mkdir -p build
if [ -f build/CMakeCache.txt ] && ! grep -q '^CMAKE_CUDA_COMPILER:FILEPATH=/usr/local/cuda' build/CMakeCache.txt; then
  echo "WARN: build/CMakeCache.txt has a non-/usr/local/cuda CUDA compiler -- wiping build dir" >&2
  rm -rf build && mkdir -p build
fi
export CUDACXX="${{CUDACXX:-/usr/local/cuda/bin/nvcc}}"
# Always reconfigure: a cache from a different ref's checkout keeps generated Makefiles pointing
# at files that no longer exist (#693/#694).
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SERVER=ON >"$TMPDIR/cmake.log" 2>&1 || {{
  echo "BUILD_FAILED — cmake configure; tail:" >&2
  tail -40 "$TMPDIR/cmake.log" >&2
  exit 1
}}
{arb.BUILD_FAILURE_SH}
build_targets() {{
  cmake --build build --target qwen3_gguf_bench qwen3_gguf_cb_bench qwen3_gguf_score \\
    qwen3_gguf_generate qwen3_gguf_prefill_check bonsai_inspect sparkinfer_server \\
    -j"$1" >"$TMPDIR/build.log" 2>&1
}}
# A compiler killed for memory, a full disk or the overlayfs EFAULT is the box, not the PR: rebuild
# once with less parallelism, and if that dies the same way, retry the round rather than blame the
# PR. Anything else is the PR's own build error, reported with its error lines first.
if ! build_targets "$(nproc)"; then
  if FAULT=$(build_box_fault "$TMPDIR/build.log"); then
    echo "build hit a box-side fault ($FAULT) -- rebuilding with -j4" >&2
    if ! build_targets 4; then
      if FAULT=$(build_box_fault "$TMPDIR/build.log"); then
        echo "RETRYABLE_INFRA_FAILURE build: $FAULT" >&2
        exit 1
      fi
      report_build_failure "$TMPDIR/build.log"
      exit 1
    fi
  else
    report_build_failure "$TMPDIR/build.log"
    exit 1
  fi
fi
for b in qwen3_gguf_bench qwen3_gguf_cb_bench qwen3_gguf_score qwen3_gguf_generate qwen3_gguf_prefill_check bonsai_inspect; do
  test -x "build/runtime/$b" || {{ echo "BUILD_FAILED — build/runtime/$b missing" >&2; exit 1; }}
done
export LD_LIBRARY_PATH="$REPO/build/runtime:$REPO/build/kernels:$REPO/build/moe:${{LD_LIBRARY_PATH:-}}"
# STAGE <name> <epoch>: where a round's GPU time goes, readable straight from the log.
echo "STAGE built $(date +%s)"

# tok_ids FILE: the file's token ids, space-separated, with the Qwen3.8 tokenizer the model shares.
tok_ids() {{
  python3 - "$TOK_DIR/tokenizer.json" "$1" <<'PYTOK'
import sys
from tokenizers import Tokenizer
print(" ".join(str(i) for i in Tokenizer.from_file(sys.argv[1]).encode(open(sys.argv[2]).read()).ids))
PYTOK
}}

source bench/scripts/_common.sh
source bench/scripts/_eval_speed.sh
SI_BIN="$PWD/build/runtime"; SI_LD=""

# --- 1. speed: decode + prefill at every scored context, one model load ---
# A real prompt rather than the synthetic ramp for the contexts it covers, as pr_qwen38_bot.py
# does: an optimisation keyed on a repeating token stream must not post a speedup here.
BENCH_PROMPT_IDS="$TMPDIR/bench_prompt_ids.txt"
if tok_ids bench/scripts/bench_prompt.txt > "$BENCH_PROMPT_IDS" 2>/dev/null; then
  export SPARKINFER_BENCH_PROMPT_FILE="$BENCH_PROMPT_IDS"
else
  echo "BENCH_PROMPT_TOKENIZE_FAILED -- falling back to the synthetic prompt" >&2
fi
wait_gpu_clear
if bench_sweep_run "$GGUF" {BENCH_TOKENS} {sweep_args}; then
  for ctx in {ctx_list}; do
    echo "BONSAI $ctx $(_bench_sweep_get $ctx decode_tps) $(_bench_sweep_get $ctx prefill_pp)"
  done
else
  echo "BONSAI_FAILED ${{_BENCH_SWEEP_RC:-1}}"
fi
echo "STAGE speed $(date +%s)"

# --- 1b. concurrent decode, median of {CB_REPS} complete runs per width (pr_qwen38_bot.py) ---
# cb_complete C TOKENS ERRORS: every request either finished or failed outright. C streams of
# {CB_TOKENS} tokens plus one long request of {CB_LONG_TOKENS}; any other total means requests
# stopped part-way, and the aggregate would be computed over a shortened wall time.
cb_complete() {{
  local c=$1 tok=$2 err=$3 b a
  for b in 0 1; do
    a=$((err - b))
    [ "$a" -ge 0 ] && [ "$a" -le "$c" ] || continue
    [ "$tok" -eq $(( (c - a) * {CB_TOKENS} + (1 - b) * {CB_LONG_TOKENS} )) ] && return 0
  done
  return 1
}}
cb_median() {{
  local ckpt=$1 cc=$2 out="$TMPDIR/cb.txt" attempt=0 valid=0 a rc
  CB_AGGS=""; CB_AGG=0; CB_WHY=run
  while [ "$valid" -lt {CB_REPS} ]; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt {CB_MAX_ATTEMPTS} ]; then
      echo "concurrent decode at c=$cc did not complete on $((attempt - 1 - valid)) of {CB_MAX_ATTEMPTS} runs" >&2
      return 1
    fi
    # The card not draining is the box's fault, never the width's: say so (BONSAICB_FAILED c gpu).
    wait_gpu_clear || {{ CB_WHY=gpu; return 1; }}
    # A nonzero exit is one failed attempt, like a partial run -- it used to give up on the width
    # at once, so one crash dropped the axis for the whole round. A timeout still gives up: a hang
    # repeats, and five of them would cost over an hour.
    if timeout 900 build/runtime/qwen3_gguf_cb_bench "$ckpt" "$cc" {CB_TOKENS} {CB_TOKENS} 512 > "$out" 2>&1; then
      rc=0
    else
      rc=$?
    fi
    if [ "$rc" != 0 ]; then
      echo "CB_EXIT c=$cc attempt=$attempt exit=$rc" >&2
      tail -10 "$out" >&2 || true
      if [ "$rc" = 124 ]; then return 1; fi
      continue
    fi
    local tok err
    tok=$(sed -n 's/.*decode_tokens=\\([0-9]*\\).*/\\1/p' "$out" | tail -1)
    err=$(grep -c "request error" "$out" || true)
    if ! cb_complete "$cc" "${{tok:-0}}" "${{err:-0}}"; then
      echo "CB_PARTIAL c=$cc attempt=$attempt decode_tokens=${{tok:-0}} request_errors=${{err:-0}}" >&2
      continue
    fi
    a=$(sed -n 's/.*agg_tok_s=\\([0-9.]*\\).*/\\1/p' "$out" | tail -1)
    CB_AGGS="$CB_AGGS ${{a:-0}}"
    valid=$((valid + 1))
  done
  CB_AGG=$(python3 -c "import statistics, sys; print(statistics.median(float(x) for x in sys.argv[1:]))" $CB_AGGS)
  python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" "${{CB_AGG:-0}}"
}}
# A width that fails is reported (BONSAICB_FAILED <c> run|gpu), never read as a regression to zero.
# A width main fails is dropped for the round, loudly (measure_main_baseline does not require every
# width); a width only the PR fails is judged in eval_bonsai_on_box, over two rounds (STRIKES_TO_REJECT).
for CC in {cb_concs}; do
  if cb_median "$GGUF" "$CC"; then
    echo "BONSAICB $CC $CB_AGG$CB_AGGS"
  else
    echo "BONSAICB_FAILED $CC ${{CB_WHY:-run}}"
  fi
done
echo "STAGE concurrency $(date +%s)"
wait_gpu_clear

# --- 2a. differential teacher-forced score ---
IDS=$(tok_ids {shlex.quote(ACC_TEXT)}) || {{ echo "TOKENIZE_FAILED (score corpus)" >&2; exit 1; }}
echo "RESULT_TOKEN_COUNT $(printf '%s' "$IDS" | wc -w)"
if build/runtime/qwen3_gguf_score "$GGUF" {ACC_TOPK} $IDS > "$DUMP_SELF" 2>"$TMPDIR/score.err"; then
  :
else
  rc=$?
  # SIGKILL is the host OOM killer, whose trigger may be anything on the box: infra, not the PR.
  if [ "$rc" = 137 ]; then
    echo "RETRYABLE_INFRA_FAILURE score step killed (exit 137)" >&2
    exit 1
  fi
  echo "SCORE_FAILED (exit $rc) -- tail of the score log:" >&2
  tail -40 "$TMPDIR/score.err" >&2
  exit 1
fi
echo "SCORE_DONE $(grep -c '^S ' "$DUMP_SELF" || true)"
echo "ACCURACY_STAGE_DONE"
if [ "$IS_PR" = "0" ]; then
  # main against itself must read top-1 1.000 / KL 0: proves the dump every PR is compared with
  # is readable, so a PR run whose compare prints nothing is the PR's own dump.
  python3 bench/scripts/accuracy_compare_pair.py "$DUMP_SELF" "$DUMP_SELF" --metric-label SELFCHECK || true
else
  if [ -s "$DUMP_MAIN" ]; then
    python3 bench/scripts/accuracy_compare_pair.py "$DUMP_SELF" "$DUMP_MAIN" || true
  else
    echo "ACCURACY_NO_BASELINE"
  fi
fi
echo "STAGE score $(date +%s)"

# --- 2b. batched prefill against the token loop, {PF_RUNS} runs per prefix ---
PF_IDS=$(tok_ids {shlex.quote(PF_TEXT)}) || {{ echo "TOKENIZE_FAILED (prefill-check corpus)" >&2; exit 1; }}
PF_OUT="$TMPDIR/pfcheck.txt"
for P in {pf_prefixes}; do
  NEED=$((P + {PF_CONT}))
  # cut, not head: head closes the pipe early and pipefail turns the SIGPIPE into an exit.
  IDS_P=$(printf '%s' "$PF_IDS" | cut -d' ' -f1-"$NEED")
  for R in $(seq 1 {PF_RUNS}); do
    # Up to {PF_TRIES} attempts per run: one crashed process out of {PF_RUNS} used to REJECT the PR on
    # the spot. A timeout is not retried -- a hang repeats.
    GOT=0; KILLED=1; FAIL_RC=1
    for TRY in $(seq 1 {PF_TRIES}); do
      wait_gpu_clear
      if timeout 900 build/runtime/qwen3_gguf_prefill_check "$GGUF" "$P" {PF_CONT} $IDS_P > "$PF_OUT" 2>&1; then
        rc=0
      else
        rc=$?
      fi
      if [ "$rc" = 0 ]; then
        # A value is a separate word that starts with a digit: glibc prints NaN as "-nan", which
        # used to be dropped with its run (2 NaN runs of 3 still passed). The mandatory space keeps
        # "TOP1 9/10 -nan" from backtracking into "9/1" and capturing the "0".
        T=$(sed -n 's/^TOP1 *[0-9]*\\/[0-9]*  *\\([0-9][0-9.]*\\).*/\\1/p' "$PF_OUT" | tail -1)
        K=$(sed -n 's/^KL  *\\([0-9][0-9.eE+-]*\\).*/\\1/p' "$PF_OUT" | tail -1)
        if [ -n "$T" ] && [ -n "$K" ]; then
          echo "PFCHECK $P $R $T $K"
          GOT=1
          break
        fi
      fi
      if [ "$rc" != 137 ]; then KILLED=0; FAIL_RC=$rc; fi
      echo "PFCHECK_RETRY $P $R attempt=$TRY exit=$rc" >&2
      tail -8 "$PF_OUT" >&2 || true
      if [ "$rc" = 124 ]; then break; fi
    done
    # rc=137 only when the host OOM killer took every attempt (the box's); else the last other exit.
    if [ "$GOT" != 1 ]; then
      if [ "$KILLED" = 1 ]; then echo "PFCHECK_FAILED $P $R rc=137"; else echo "PFCHECK_FAILED $P $R rc=$FAIL_RC"; fi
    fi
  done
done

echo "STAGE prefill_check $(date +%s)"

# --- 2c. bonsai_regression.py: tensors, score, generate, serve ---
wait_gpu_clear
REG_OUT="$TMPDIR/bonsai_regression.txt"
# -u: a run killed part-way (its timeout, the OOM killer) keeps everything it printed. 2700 s: the
# serve check can now run up to five trials per path.
if timeout 2700 python3 -u eval/bonsai_regression.py --model "$GGUF" --reference "$REF_DIR" \\
     --tokenizer "$TOK_DIR" --build "$REPO/build/runtime" > "$REG_OUT" 2>&1; then
  REG_RC=0
else
  REG_RC=$?
fi
if [ "$REG_RC" = 0 ]; then
  echo "BONSAIREG_OK"
else
  echo "BONSAIREG_FAILED"
  echo "BONSAIREG_EXIT $REG_RC"
  if grep -q '^FAILED:' "$REG_OUT"; then
    sed -n '/^FAILED:/,$p' "$REG_OUT" | grep -E '^ +- ' | sed 's/^ *- /BONSAIREG_WHY /' | head -20 || true
  else
    echo "BONSAIREG_WHY did not complete (exit $REG_RC): $(grep -v '^\\s*$' "$REG_OUT" | tail -1)"
  fi
  tail -30 "$REG_OUT" >&2
fi
# Inconclusive serve trials and similar: shown in the verdict, never gated.
grep '^NOTE: ' "$REG_OUT" | sed 's/^NOTE: /BONSAIREG_NOTE /' | head -8 || true
reap_our_servers
echo "STAGE bonsai_regression $(date +%s)"

# --- 3. cross-model no-regression guards @ 32k ---
# The real-prompt file is Qwen3.8 token ids; the guards run synthetic, as the sibling bots do.
unset SPARKINFER_BENCH_PROMPT_FILE
export MODELS_DIR="$Q36_GUARD_MODELS_DIR" MODEL_REPO="$Q36_GUARD_MODEL_REPO" \\
       MODEL_FILE="$Q36_GUARD_MODEL_FILE" TOK_REPO="$Q36_GUARD_TOK_REPO"
export MODEL_SHA256="${{QWEN36_MODEL_SHA256:-}}"
( ensure_model && ensure_tokenizer ) || echo "WARN: qwen3.6 guard model setup failed" >&2
echo "GUARD_START"
guard() {{  # guard TAG CHECKPOINT
  local tag=$1 ckpt=$2
  if [ ! -e "$ckpt" ]; then echo "${{tag}}_UNAVAILABLE"; return 0; fi
  # "gpu": the card never drained -- the box's fault, never the PR's (eval_bonsai_on_box).
  wait_gpu_clear || {{ echo "${{tag}}_FAILED gpu"; return 0; }}
  if bench_sweep_run "$ckpt" 128 {guard_args}; then
    for ctx in {guard_list}; do
      echo "$tag $ctx $(_bench_sweep_get $ctx decode_tps) $(_bench_sweep_get $ctx prefill_pp)"
    done
  else
    echo "${{tag}}_FAILED rc=${{_BENCH_SWEEP_RC:-1}}"
  fi
}}
guard GUARD36 "$Q36_GUARD_MODELS_DIR/$Q36_GUARD_MODEL_FILE"
guard GUARDMO "$MODELOPT_GUARD_MODEL_DIR"
guard GUARDUN "$UNSLOTH_GUARD_MODEL_DIR"
guard GUARDMG "$MUSE_GUARD_GGUF"
echo "STAGE guards $(date +%s)"
echo "GUARD_END"
"""


def _parse_remote(stdout: str) -> dict:
    out = {"bonsai": {}, "bonsai_cb": {}, "cb_runs": {}, "cb_failed": [], "cb_failed_gpu": [],
           "pfcheck": {}, "pfcheck_failed": {}, "bonsaireg_why": [], "bonsaireg_notes": []}
    for key, _tag, _name in GUARDS:
        out[key] = {}
    tags = {tag: key for key, tag, _name in GUARDS}
    for line in (stdout or "").splitlines():
        parts = line.split()
        if not parts:
            continue
        head = parts[0]
        try:
            if head == "REMOTE_HEAD" and len(parts) >= 2:
                out["head"] = parts[1]
            elif head == "REMOTE_SHA" and len(parts) >= 2:
                out["sha"] = parts[1]
            elif head == "PR_TIP" and len(parts) >= 2:
                out["pr_tip"] = parts[1]
            elif head == "MERGED_ONTO" and len(parts) >= 2:
                out["merged_onto"] = parts[1]
            elif head == "BONSAI" and len(parts) >= 4:
                out["bonsai"][int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
            elif head == "STAGE" and len(parts) >= 3:
                out.setdefault("stages", []).append((parts[1], int(parts[2])))
            elif head == "BONSAI_FAILED":
                out["bonsai_failed"] = True
                if len(parts) >= 2 and parts[1].lstrip("-").isdigit():
                    out["bonsai_failed_rc"] = int(parts[1])
            elif head == "BONSAICB" and len(parts) >= 3:
                out["bonsai_cb"][int(parts[1])] = float(parts[2])
                out["cb_runs"][int(parts[1])] = [float(x) for x in parts[3:]]
            elif head == "BONSAICB_FAILED" and len(parts) >= 2:
                out["cb_failed"].append(int(parts[1]))
                if len(parts) >= 3 and parts[2] == "gpu":
                    out["cb_failed_gpu"].append(int(parts[1]))
            elif head == "RESULT_TOKEN_COUNT" and len(parts) >= 2:
                out["token_count"] = int(parts[1])
            elif head == "SCORE_DONE":
                out["score_done"] = True
                out["score_positions"] = int(parts[1]) if len(parts) >= 2 else 0
            elif head == "SELFCHECK":
                for tok in parts[1:]:
                    k, _, v = tok.partition("=")
                    if k in ("top1", "kl", "n"):
                        out[f"selfcheck_{k}"] = float(v)
            elif head == "ACCURACY_NO_BASELINE":
                out["accuracy_no_baseline"] = True
            elif head == "METRIC":
                for tok in parts[1:]:
                    k, _, v = tok.partition("=")
                    if k in ("top1", "kl", "ppl_pr", "ppl_main", "n", "n_main"):
                        out[k] = float(v)
            elif head == "PFCHECK" and len(parts) >= 5:
                p = int(parts[1])
                try:
                    t, k = float(parts[3]), float(parts[4])
                except ValueError:
                    t = k = float("nan")
                if t != t or k != k:      # NaN / unreadable: a failed run, never a dropped one
                    out["pfcheck_failed"][p] = out["pfcheck_failed"].get(p, 0) + 1
                else:
                    out["pfcheck"].setdefault(p, []).append((t, k))
            elif head == "PFCHECK_FAILED" and len(parts) >= 2:
                p = int(parts[1])
                key = "pfcheck_failed_box" if "rc=137" in parts[2:] else "pfcheck_failed"
                out.setdefault(key, {})
                out[key][p] = out[key].get(p, 0) + 1
            elif head == "BONSAIREG_OK":
                out["bonsaireg_ok"] = True
            elif head == "BONSAIREG_FAILED":
                out["bonsaireg_ok"] = False
            elif head == "BONSAIREG_WHY":
                out["bonsaireg_why"].append(line.split(" ", 1)[1].strip() if " " in line else "")
            elif head == "BONSAIREG_NOTE" and " " in line:
                out["bonsaireg_notes"].append(line.split(" ", 1)[1].strip())
            elif head == "BONSAIREG_EXIT" and len(parts) >= 2:
                out["bonsaireg_exit"] = int(parts[1])
            elif head in tags and len(parts) >= 4:
                out[tags[head]][int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
            elif head.endswith("_FAILED") and head[:-len("_FAILED")] in tags:
                key = tags[head[:-len("_FAILED")]]
                out[key + "_failed"] = True
                # "gpu" (never drained) or rc=137 (the OOM killer): the box, not the PR.
                if len(parts) >= 2 and (parts[1] == "gpu" or parts[1] == "rc=137"):
                    out[key + "_failed_box"] = True
            elif head.endswith("_UNAVAILABLE") and head[:-len("_UNAVAILABLE")] in tags:
                out[tags[head[:-len("_UNAVAILABLE")]] + "_unavailable"] = True
        except (ValueError, IndexError):
            continue
    return out


def _at(res: dict, ctx: int, phase: str) -> float:
    """One single-request measurement, 0.0 when missing -- which tier_from_gain scores as a
    regression against a real baseline (fail closed)."""
    return float(((res.get("bonsai") or {}).get(ctx) or {}).get(phase, 0.0) or 0.0)


def _cb(res: dict, conc: int):
    """One concurrent-decode median, or None when that width was not measured (the axis drops)."""
    v = (res.get("bonsai_cb") or {}).get(conc)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _guard_coverage(d: dict) -> str:
    return " · ".join(f"{name} {len(d.get(key) or {})} ctx" for key, _tag, name in GUARDS)


REG_CHECKS = ("tensors", "score", "generate", "serve")


def _reg_check_of(why: str) -> str:
    """The bonsai_regression.py check a FAILED line belongs to: "tensors: ..." -> "tensors". The
    serve check is split by path ("serve: folded ..." -> "serve:folded"), so a native-path flake on
    main does not un-gate a PR that breaks the folded path."""
    head, _, rest = (why or "").partition(":")
    head = head.strip()
    if head not in REG_CHECKS:
        return "*"
    path = (rest.split() or [""])[0]
    return f"{head}:{path}" if head == "serve" and path in ("folded", "native") else head


def _reg_expand(checks) -> set:
    """A serve failure with no path (the check raised) stands for both paths, so it cancels or is
    cancelled by a per-path failure on the other side instead of never matching it."""
    out = set()
    for c in checks:
        out |= {"serve:folded", "serve:native"} if c == "serve" else {c}
    return out


def _reg_failed_checks(ok, why_lines) -> set:
    """Which checks failed: {} on a pass, {"*"} when the script failed without naming one (it
    crashed, or was killed by its timeout) -- which then stands for every check."""
    if ok:
        return set()
    return _reg_expand({_reg_check_of(w) for w in (why_lines or [])}) or {"*"}


def push_eval_polaris(host, port):
    """Sync eval/polaris/ from origin/main before attesting: a PR's own checkout must never supply
    the code that produces its own attestation. Same as the sibling bots."""
    use_local = os.environ.get("SPARKINFER_USE_LOCAL_POLARIS", "").strip().lower() in ("1", "true", "yes")
    tar_data, source, extract_root = None, "local checkout", os.path.join(REMOTE_REPO, "eval")
    if not use_local:
        subprocess.run(["git", "fetch", "-q", "origin", "main"], cwd=ROOT, capture_output=True, timeout=120)
        arch = subprocess.run(["git", "archive", "--format=tar.gz", "origin/main", "eval/polaris"],
                              cwd=ROOT, capture_output=True, timeout=120)
        if arch.returncode != 0 or not arch.stdout:
            print(">> WARN: origin/main eval/polaris fetch failed — attestation unavailable this run")
            return False
        tar_data, source, extract_root = arch.stdout, "origin/main", REMOTE_REPO
    else:
        tar = subprocess.run(["tar", "-C", HERE, "-czf", "-", "polaris"], capture_output=True, timeout=120)
        if tar.returncode == 0 and tar.stdout:
            tar_data = tar.stdout
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
             "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
             tmp_path, f"{user}@{host}:/tmp/si_polaris_bonsai.tgz"],
            capture_output=True, text=True, timeout=120)
        if scp.returncode != 0:
            print(f">> WARN: eval/polaris scp failed (rc={scp.returncode}): {scp.stderr[-500:]}")
            return False
        r = ssh_run(host, port,
                    f"mkdir -p {shlex.quote(extract_root)} && "
                    f"tar -xzf /tmp/si_polaris_bonsai.tgz -C {shlex.quote(extract_root)} && "
                    "rm -f /tmp/si_polaris_bonsai.tgz", timeout=60)
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
    """Assemble and sign an attestation through judge.py --from-stdin (the generic single-model
    shape). Never raises into the verdict: a receipt is optional, the measurement is not."""
    if not POLARIS_ENABLED:
        return None
    # The tip that was measured, not whatever pull/<n>/head points at by now: a third fetch of the
    # ref could attest a commit pushed after the measurement.
    tip = res.get("pr_tip") or ""
    target = shlex.quote(tip) if arb._FULL_SHA_RE.match(tip) else "FETCH_HEAD"
    r0 = ssh_run(host, port, f"cd {shlex.quote(REMOTE_REPO)} && git fetch -q origin "
                             f"{shlex.quote(pr_ref)} && git checkout -qf {target}", timeout=120)
    if r0.returncode != 0:
        print(f">> Polaris: could not checkout {pr_ref} for attestation: {(r0.stderr or '')[-300:]}")
        return None
    if not push_eval_polaris(host, port):
        return None
    result_json = {
        "model": "bonsai-128",
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
    cmd = (f"cd {shlex.quote(REMOTE_REPO)} && SPARKINFER_EVAL_MODE=bonsai-128 "
           f"SPARKINFER_DECODE_TOKENS={BENCH_TOKENS} "
           f"SPARKINFER_EVAL_SEED={shlex.quote(f'bonsai-{int(time.time() * 1000)}')} "
           f"python3 eval/polaris/judge.py --from-stdin --model-file {shlex.quote(BONSAI_GGUF)} "
           f"--build-dir {shlex.quote(REMOTE_REPO)}/build/runtime "
           f"--sparkinfer-root {shlex.quote(REMOTE_REPO)}")
    r = ssh_run(host, port, cmd, timeout=300, stdin_data="RESULT_JSON " + json.dumps(result_json))
    if r.returncode != 0:
        print(f">> Polaris judge failed (rc={r.returncode}): {(r.stderr or '')[-500:]}")
        return None
    line = next((l for l in (r.stdout or "").splitlines() if l.startswith("POLARIS_ATTESTATION ")), None)
    if not line:
        print(">> Polaris judge produced no attestation")
        return None
    try:
        attestation = json.loads(line[len("POLARIS_ATTESTATION "):])
    except json.JSONDecodeError as e:
        print(f">> Polaris attestation JSON parse failed: {e}")
        return None
    privkey = arb._load_polaris_privkey()
    if not POLARIS_API_KEY and not privkey:
        print(">> Polaris: attestation collected but NOT signed (no key configured)")
        return {"attestation": attestation}
    try:
        receipt = arb.build_polaris_receipt_from_attestation(
            attestation, api_key=POLARIS_API_KEY, privkey=privkey, pubkey=_load_polaris_pubkey())
        return {"attestation": attestation, "receipt": receipt}
    except Exception as e:
        print(f">> Polaris signing failed: {e}")
        return {"attestation": attestation}


def _run_failure(r, what: str) -> dict:
    tail = arb.failure_excerpt(r.stdout, r.stderr, _EXPLICIT_FAIL_MARKERS)
    crash = _crash_reason(r.stdout, r.stderr)
    infra = _is_infra_failure(r.stdout, r.stderr)
    crash = arb.failure_cause(crash, r.stdout, r.stderr)             # the text only
    reason = f"{what} failed" + (f" — {crash}" if crash else " (no crash diagnostic captured — retried once)")
    # The tip the box fetched, when it got that far: a build failure is recorded against the
    # commit that failed to build (arb.measured_commit).
    tip = _parse_remote(r.stdout or "").get("pr_tip")
    out = {"ok": False, "retry": infra, "reason": reason, "log": tail, "pr_tip": tip}
    if infra:
        # Almost always the box -- but a PR that really does exhaust the compiler, get its score
        # step OOM-killed or take the box down would otherwise be retried for ever with nothing
        # posted, so each kind is charged to the PR after BOX_FAULT_STRIKES rounds (apply_result).
        blob = (r.stderr or "") + (r.stdout or "")
        out["strike_key"] = ("build-box" if "RETRYABLE_INFRA_FAILURE build:" in blob
                             else "score-box" if "RETRYABLE_INFRA_FAILURE score step" in blob
                             else "box")
    return out


def measure_main_baseline(host, port):
    """Measure main ONCE per round; every PR in the round compares against it. Fails the ROUND,
    not a PR, when anything the PR comparison needs is missing."""
    r = _ssh_run_resilient(host, port, _remote_script("main", role="main"), "main run")
    if r.returncode != 0:
        return _run_failure(r, "main run")
    main = _parse_remote(r.stdout or "")
    log = (r.stdout or "")[-1500:]
    missing = [SCORED_CTX_LABEL[c] for c in SCORED_CTXS
               if not (_at(main, c, "decode") > 0 and _at(main, c, "prefill") > 0)]
    if main.get("bonsai_failed") or missing:
        return {"ok": False, "reason": "main bench missing Ternary-Bonsai-2-27B measurements at ctx "
                                       + ",".join(missing or ["(sweep failed)"]), "log": log}
    # A width main cannot measure is dropped for the round, loudly (eval_bonsai_on_box, the verdict
    # table), not a reason to skip the round: if main itself broke at a width, skipping would stall
    # this bot for every PR, the one fixing it included. Every guard, on the other hand, must have
    # measured something unless its checkpoint is absent: otherwise every PR in the round would be
    # measured in full only to be deferred for the missing guard.
    unguarded = [name for key, _t, name in GUARDS
                 if not main.get(f"{key}_unavailable") and not arb.guard_measured(main.get(key))]
    if unguarded:
        return {"ok": False, "reason": "main measured nothing for the " + ", ".join(unguarded)
                                       + " guard", "log": log}
    if not main.get("score_done") or main.get("score_positions", 0) < 100:
        return {"ok": False, "reason": f"main score dump missing or short "
                                       f"({main.get('score_positions', 0)} positions)", "log": log}
    if main.get("selfcheck_n") is not None and main["selfcheck_n"] < 100:
        # The rows the comparator could read (a NaN row is not one): every PR is judged on these.
        return {"ok": False, "reason": f"main score dump has only {int(main['selfcheck_n'])} readable "
                                       f"positions", "log": log}
    if main.get("selfcheck_top1") != 1.0 or (main.get("selfcheck_kl") or 0.0) > 1e-6:
        return {"ok": False, "reason": "main score dump failed its self-comparison "
                                       f"(top1={main.get('selfcheck_top1')} kl={main.get('selfcheck_kl')})",
                "log": log}
    short = [p for p in PF_PREFIXES if len((main.get("pfcheck") or {}).get(p) or []) < PF_RUNS]
    if short:
        return {"ok": False, "reason": "main prefill-path check incomplete at prefix "
                                       + ",".join(str(p) for p in short), "log": log}
    if "bonsaireg_ok" not in main:
        return {"ok": False, "reason": "main bonsai_regression.py did not run", "log": log}
    if not main.get("sha"):
        # Every verdict records it (`onto`); without it every PR would count as scored on an old main.
        return {"ok": False, "reason": "main run did not report its commit", "log": log}
    main["ok"] = True
    return main


def eval_bonsai_on_box(host, port, pr_ref: str, main: dict):
    """Run the PR ref's script and score it against `main`, the round's shared baseline."""
    print(f">> Ternary-Bonsai-2-27B eval on box: PR ref={pr_ref}")
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
            return {"ok": False, "retry": True, "conflict": True, "log": "", "reason": conflict,
                    "pr_tip": arb.merge_conflict_tip(r.stdout, r.stderr)}
        ahead = arb.base_ahead_line(r.stdout, r.stderr)
        if ahead:
            # Rebased onto a main newer than this round's baseline: measured next round, onto it.
            return {"ok": False, "retry": True, "log": "", "reason": ahead}
        return _run_failure(r, "PR speed/accuracy run")
    pr = _parse_remote(r.stdout or "")
    log = (r.stdout or "")[-1500:]
    tip = pr.get("pr_tip")   # every verdict below is recorded against it (arb.measured_commit)
    if pr.get("accuracy_no_baseline"):
        # Almost always the box; bounded all the same (a PR's own build could remove the dump).
        return {"ok": False, "retry": True, "log": log, "pr_tip": tip, "strike_key": "box",
                "reason": "main's score dump was gone when the PR run compared against it"}
    if "top1" not in pr or "kl" not in pr:
        # main's dump passed its self-comparison this round, so an unreadable pair is the PR's dump.
        return {"ok": False, "retry": False, "log": log, "pr_tip": tip,
                "reason": "accuracy_compare_pair.py could not read the PR's score dump"}
    pr_top1, pr_kl = pr["top1"], pr["kl"]
    ppl_ratio = (pr.get("ppl_pr") or 0) / pr["ppl_main"] if pr.get("ppl_main") else None
    # Every position main's dump has, not just the ones the PR's dump has: the compare skips the rest.
    covered = pr.get("n_main") is None or pr.get("n", 0) >= pr["n_main"]
    accuracy_ok = (pr_top1 >= ACC_TOP1_BAR and pr_kl <= ACC_KL_BAR
                   and ppl_ratio is not None and ppl_ratio <= ACC_PPL_RATIO and covered)

    # Checks the box kept from running this round: (strike key, why). They defer the verdict --
    # unless a gate that a contended or failing box cannot fake (accuracy, the prefill path, the
    # regression script's checks: `hard` below) already REJECTs the PR, which is then posted.
    # Throughput can be faked by the same contention, so a speed REJECT does not override them.
    deferred = []

    # The hard gates, first: known before anything is deferred.
    hard = []
    if not accuracy_ok:
        hard.append(f"accuracy gate failed vs main: top1={pr_top1:.4f} (bar >={ACC_TOP1_BAR}) "
                    f"kl={pr_kl:.5f} (bar <={ACC_KL_BAR}) ppl x{ppl_ratio or 0:.4f} (bar <={ACC_PPL_RATIO})"
                    + ("" if covered else f"; the PR's score dump covers {int(pr.get('n', 0))} of "
                                          f"{int(pr['n_main'])} positions"))
    pf_ok, pf_problems, pf_rows = check_prefill_path(pr, main)
    if not pf_ok:
        unavailable = [p for p in pf_problems if p.endswith("measurement unavailable")]
        killed = [p for p in pf_problems if p.endswith("killed on the PR build (exit 137)")]
        if len(unavailable) + len(killed) == len(pf_problems):
            if unavailable:
                deferred.append((None, "; ".join(unavailable)))   # main's, not the PR's: no strike
            if killed:
                deferred.append(("pf-box", "; ".join(killed)))    # the box's, bounded by strikes
        else:
            hard.append("prefill-path accuracy gate failed: " + "; ".join(pf_problems[:4]))

    # Gate 2c is absolute, so a check may only reject a PR when main passes THAT check in the same
    # round. Per check, not all-or-nothing: a serve flake on main used to switch off the tensors,
    # score and generate checks too, for every PR in the round.
    reg_main_ok = bool(main.get("bonsaireg_ok"))
    reg_pr_ok = pr.get("bonsaireg_ok")
    reg_main_failed = _reg_failed_checks(reg_main_ok, main.get("bonsaireg_why"))
    if reg_pr_ok is None:
        deferred.append(("reg-box", "PR bonsai_regression.py did not run"))
        reg_pr_failed = set()
    else:
        reg_pr_failed = _reg_failed_checks(reg_pr_ok, pr.get("bonsaireg_why"))
    reg_gated_fail = sorted(reg_pr_failed - reg_main_failed) if "*" not in reg_main_failed else []
    reg_soft_why, serve_soft_why = None, None
    if reg_gated_fail:
        why = [w for w in (pr.get("bonsaireg_why") or [])
               if "*" in reg_gated_fail or _reg_expand({_reg_check_of(w)}) & set(reg_gated_fail)]
        reg_exit = pr.get("bonsaireg_exit")
        if reg_gated_fail == ["*"] and reg_exit in (124, 137):
            # Killed without naming a check -- its time limit or the OOM killer. The box's fault as
            # often as the PR's, so like a width only the PR fails it is judged over two rounds.
            reg_soft_why = ("bonsai_regression.py did not complete on the PR build ("
                            + ("timed out" if reg_exit == 124 else "killed") + f", exit {reg_exit})")
        elif all(c.startswith("serve") for c in reg_gated_fail):
            # The serve check alone: 2 of up to 3 trials already, but still a few percent per round
            # on a sound build (one trial fails ~1 in 8), so it too must repeat before it REJECTs.
            serve_soft_why = ("bonsai_regression.py's serve check failed (main passes it): "
                              + "; ".join(why[:2] or ["see log"]))
        else:
            hard.append("bonsai_regression.py failed (main passes that check): "
                        + "; ".join(why[:3] or ["see log"]))

    if pr.get("bonsai_failed_rc") == 137 and not hard:
        # The speed sweep was SIGKILLed -- the host OOM killer, whose trigger may be anything. Like a
        # build the compiler cannot finish, it is charged to the PR after BOX_FAULT_STRIKES rounds.
        # Not when a hard gate already failed: that is the PR's, and said first (below).
        return {"ok": False, "retry": True, "log": log, "pr_tip": tip, "strike_key": "sweep-box",
                "reason": "the PR's speed sweep was killed (exit 137) — infra"}
    if pr.get("bonsai_failed") or not pr.get("bonsai"):
        # Lead with a failed gate: a pass that emits garbage also drops measurements, and the
        # author should be sent to their numerics, not to the harness (#1037).
        why = "PR bench produced no Ternary-Bonsai-2-27B measurements"
        if hard:
            why = f"{hard[0]}; the failed speed sweep is a symptom"
        return {"ok": False, "retry": False, "reason": why, "log": log, "pr_tip": tip}

    for ctx in SCORED_CTXS:
        print(f">> PR @{SCORED_CTX_LABEL[ctx]:>4}: decode {_at(pr, ctx, 'decode'):9.2f} "
              f"(main {_at(main, ctx, 'decode'):9.2f})   prefill {_at(pr, ctx, 'prefill'):10.2f} "
              f"(main {_at(main, ctx, 'prefill'):10.2f})")
    print(f">> PR accuracy vs main: top1={pr_top1:.4f} kl={pr_kl:.5f}")
    print(f">> PR guard coverage: {_guard_coverage(pr)}")

    scored = []
    for ctx in SCORED_CTXS:
        for phase in ("decode", "prefill"):
            name = f"bonsai-{phase}@{SCORED_CTX_LABEL[ctx]}"
            lab, dlt, ok, why = tier_from_gain(_at(pr, ctx, phase), _at(main, ctx, phase), metric=name)
            scored.append({"dim": name, "label": lab, "delta": dlt, "passed": ok, "reason": why})
    cb_skipped, cb_pr_missing = [], []
    for conc in CB_CONCS:
        name = CB_DIM_FOR[conc]
        pr_v, main_v = _cb(pr, conc), _cb(main, conc)
        if main_v is None:
            # main could not measure this width this round: the axis is dropped for the round, and
            # the verdict table says so (measure_main_baseline does not require every width).
            cb_skipped.append(name)
            continue
        if pr_v is None:
            cb_pr_missing.append(conc)
            continue
        lab, dlt, ok, why = tier_from_gain(pr_v, main_v, metric=name)
        scored.append({"dim": name, "label": lab, "delta": dlt, "passed": ok, "reason": why})
    if cb_skipped:
        print(f">> concurrent-decode axes not scored (no main measurement): {', '.join(cb_skipped)}")
    gpu_missing = [c for c in cb_pr_missing if c in (pr.get("cb_failed_gpu") or [])]
    if gpu_missing:
        # The card never drained before that width: the box's fault, measured again next round.
        deferred.append(("cb-box", "GPU did not drain before concurrent decode at "
                                   + ",".join(f"c{c}" for c in gpu_missing) + " — infra"))
        cb_pr_missing = [c for c in cb_pr_missing if c not in gpu_missing]
    by_dim = {x["dim"]: x for x in scored}

    regressed = [x for x in scored if x["label"] == "REJECT"]
    if regressed:
        best = min(regressed, key=lambda x: x["delta"])
        label, delta_pct, passed = "REJECT", best["delta"], False
        reason = " | ".join(x["reason"] for x in regressed)
    else:
        best = max((by_dim[d] for d in SCORING_DIMS if d in by_dim), key=lambda x: x["delta"])
        label, delta_pct, passed, reason = best["label"], best["delta"], best["passed"], best["reason"]
    speed_label = label

    def reject(why: str):
        nonlocal label, passed, reason
        label, passed = "REJECT", False
        reason = f"{why} | {reason}"

    for why in hard:
        reject(why)

    guard_results, guard_soft = {}, []
    for key, _tag, name in GUARDS:
        skipped = bool(pr.get(f"{key}_unavailable") or main.get(f"{key}_unavailable"))
        if not skipped and pr.get(f"{key}_failed_box"):
            # The PR run's guard never started because the GPU did not drain, or was SIGKILLed.
            # Charged to the PR after BOX_FAULT_STRIKES rounds at one commit, like sweep-box.
            why = f"the {name} guard could not run on the PR build (GPU not drained, or killed) — infra"
            deferred.append(("guard-box", why))
            guard_results[key] = {"ok": None, "problems": [why], "skipped": False,
                                  "not_run": "could not run on the PR build this round (box); not checked"}
            continue
        ok, problems = (True, []) if skipped else _check_model_guard(pr, main, key, name)
        if skipped:
            print(f">> {name} guard SKIPPED — checkpoint not installed on the box")
        if not ok:
            # A guard that measured NOTHING is infra, not a regression (pr_qwen38_bot.py, #1112/#1114).
            unavailable = [p for p in problems if p.endswith("measurement unavailable")]
            if unavailable and len(unavailable) == len(problems):
                return {"ok": False, "retry": True, "log": log, "pr_tip": tip, "strike_key": "guard-unmeasured",
                        "reason": "; ".join(unavailable) + " — infra, not a regression; "
                                  "re-evaluated next round rather than rejected"}
            if (pr.get(f"{key}_failed") or not pr.get(key)) and main.get(key):
                # The guard's sweep failed on the PR build only: judged over two rounds, like a
                # concurrency width only the PR fails, not closed on one run.
                guard_soft.append(name)
            else:
                reject(f"{name} no-regression guard failed: " + "; ".join(problems[:6]))
        guard_results[key] = {"ok": ok, "problems": problems, "skipped": skipped}

    # Last, so it is known whether they are the ONLY failures: a width the PR build could not
    # complete while main did, and a regression script killed part-way. Alone they are judged over
    # two rounds on the same commit (STRIKES_TO_REJECT, apply_result); beside any other failure they
    # are just more reasons. The strike key names the kind, not the exact widths, so a PR failing
    # c32 one round and c16,c32 the next still reaches its second strike.
    if deferred and not hard:
        # No hard gate rejects it, and something did not run: measured again next round.
        return {"ok": False, "retry": True, "log": log, "pr_tip": tip,
                "strike_key": "+".join(dict.fromkeys(k for k, _w in deferred if k)) or None,
                "reason": " | ".join(w for _k, w in deferred) + " — re-evaluated next round"}
    if deferred:
        reason = f"{reason} | not run this round: " + "; ".join(w for _k, w in deferred)
    soft = []
    if cb_pr_missing:
        widths = ",".join(f"c{c}" for c in cb_pr_missing)
        soft.append(("cb", f"concurrent decode at {widths} did not complete on the PR build in "
                           f"{CB_MAX_ATTEMPTS} attempts, while main measured it this round"))
    if reg_soft_why:
        soft.append(("reg", reg_soft_why))
    if serve_soft_why:
        soft.append(("serve", serve_soft_why))
    if guard_soft:
        soft.append(("guard", f"the {', '.join(guard_soft)} guard could not be measured on the PR "
                              f"build while main measured it this round"))
    strike_key = None
    if soft:
        whys = " | ".join(w for _k, w in soft)
        if label == "REJECT":
            # After the REJECT's own reason: the close comment quotes the first clause, and a check
            # that merely could not run is not what closed the PR.
            reason = f"{reason} | {whys}"
        else:
            reject(whys)
            strike_key = "+".join(k for k, _w in soft)

    res = {
        "ok": True,
        "label": label,
        "speed_label": speed_label,
        "pass": passed and label != "REJECT",
        "reason": reason,
        "delta_pct": delta_pct,
        "best_dim": best["dim"],
        "scored_dims": scored,
        "pr_decode_tps": _at(pr, 128, "decode"),
        "main_decode_tps": _at(main, 128, "decode"),
        "pr_prefill128_pp": _at(pr, 128, "prefill"),
        "main_prefill128_pp": _at(main, 128, "prefill"),
        "bonsai_pr": pr.get("bonsai") or {},
        "bonsai_main": main.get("bonsai") or {},
        "cb_pr": pr.get("bonsai_cb") or {},
        "cb_main": main.get("bonsai_cb") or {},
        "pr_top1": pr_top1,
        "pr_kl": pr_kl,
        "pr_ppl": pr.get("ppl_pr"),
        "main_ppl": pr.get("ppl_main"),
        "ppl_ratio": ppl_ratio,
        "token_count": pr.get("token_count"),
        "accuracy_ok": accuracy_ok,
        "prefill_path_ok": pf_ok,
        "prefill_path_rows": pf_rows,
        "prefill_path_problems": pf_problems,
        "bonsaireg_pr_ok": reg_pr_ok,
        "bonsaireg_main_ok": reg_main_ok,
        "bonsaireg_why": pr.get("bonsaireg_why") or [],
        "bonsaireg_main_failed": sorted(reg_main_failed),
        "bonsaireg_pr_failed": sorted(reg_pr_failed),
        "bonsaireg_gated_fail": reg_gated_fail,
        "bonsaireg_notes": pr.get("bonsaireg_notes") or [],
        "cb_pr_missing": cb_pr_missing,
        "strike_key": strike_key,
        "guards": guard_results,
        "guards_pr": {key: pr.get(key) for key, _t, _n in GUARDS},
        "guards_main": {key: main.get(key) for key, _t, _n in GUARDS},
        "pr_head": pr.get("head"),
        "main_head": main.get("head"),
        "pr_tip": pr.get("pr_tip"),
        "merged_onto": pr.get("merged_onto"),
        "onto": main.get("sha"),       # the full main commit this verdict was measured against
    }
    try:
        polaris = collect_polaris_attestation(host, port, res, pr_ref)
        if polaris:
            res["polaris"] = polaris
    except Exception as e:
        print(f">> Polaris attestation failed ({type(e).__name__}: {e}) — keeping the measurement")
    return res


def _ctx_list_str() -> str:
    return "/".join(SCORED_CTX_LABEL[c] for c in SCORED_CTXS)


def _cb_list_str() -> str:
    return "/".join(f"c{c}" for c in CB_CONCS)


def _matrix_table(res: dict) -> str:
    dims = {d["dim"]: d for d in (res.get("scored_dims") or [])}
    if not dims:
        return ""
    pr_m, main_m = res.get("bonsai_pr") or {}, res.get("bonsai_main") or {}
    rows = ["| ctx | phase | main | PR | delta |", "|---|---|---|---|---|"]
    for ctx in SCORED_CTXS:
        for phase in ("decode", "prefill"):
            d = dims.get(f"bonsai-{phase}@{SCORED_CTX_LABEL[ctx]}")
            if not d:
                continue
            mv = float((main_m.get(ctx) or {}).get(phase, 0) or 0)
            pv = float((pr_m.get(ctx) or {}).get(phase, 0) or 0)
            flag = " **REJECT**" if d["label"] == "REJECT" else ""
            rows.append(f"| {SCORED_CTX_LABEL[ctx]} | {phase} | {mv:.2f} | {pv:.2f} | {d['delta']:+.1f}%{flag} |")
    out = "\n".join(rows) + "\n\n"
    cb_pr, cb_main = res.get("cb_pr") or {}, res.get("cb_main") or {}
    cb_rows = []
    for conc in CB_CONCS:
        d = dims.get(CB_DIM_FOR[conc])
        if not d:
            continue
        flag = " **REJECT**" if d["label"] == "REJECT" else ""
        cb_rows.append(f"| c{conc} | {float(cb_main.get(conc) or 0):.2f} | "
                       f"{float(cb_pr.get(conc) or 0):.2f} | {d['delta']:+.1f}%{flag} |")
    if cb_rows:
        out += ("**Concurrent decode** — aggregate tok/s with N requests in flight, median of "
                f"{CB_REPS} runs\n\n| concurrency | main | PR | delta |\n|---|---|---|---|\n"
                + "\n".join(cb_rows) + "\n\n")
    pr_missing = [f"c{c}" for c in (res.get("cb_pr_missing") or [])]
    missing = [f"c{c}" for c in CB_CONCS if CB_DIM_FOR[c] not in dims and f"c{c}" not in pr_missing]
    if pr_missing:
        out += (f"<sub>Concurrency {', '.join(pr_missing)} did not complete on the PR build in "
                f"{CB_MAX_ATTEMPTS} attempts, while main measured it this round.</sub>\n\n")
    if missing:
        out += (f"<sub>Concurrency {', '.join(missing)} not scored this round — main has no "
                f"measurement for it.</sub>\n\n")
    pf_rows = res.get("prefill_path_rows") or []
    if pf_rows:
        out += (f"**Prefill path** — batched prefill against the token loop, {PF_CONT} teacher-forced "
                f"positions, mean of {PF_RUNS} runs per side\n\n"
                "| prefix | main top-1 | PR top-1 | top-1 bar | main KL | PR KL | KL bar |\n"
                "|---|---|---|---|---|---|---|\n")
        for r in pf_rows:
            pt = "—" if r["pr_top1"] is None else f"{r['pr_top1']:.3f}"
            pk = "—" if r["pr_kl"] is None else f"{r['pr_kl']:.4f}"
            out += (f"| {r['prefix']} | {r['main_top1']:.3f} | {pt} | ≥{r['top1_bar']:.3f} | "
                    f"{r['main_kl']:.4f} | {pk} | ≤{r['kl_bar']:.4f} |\n")
        out += "\n"
    return out


def _policy_note() -> str:
    """What the bot will do with the verdict, read from the live switches -- never stated from
    memory, so the PR comment cannot claim a policy the running config does not have."""
    merge = ("The round's best passing speedup is auto-merged as `bonsai-merge-first` (only at the "
             "exact commit scored); a separate comment says so." if AUTO_MERGE
             else "This bot does not auto-merge.")
    close = ("A measured `REJECT` closes the PR, and so does `none` on a PR that declares "
             "Ternary-Bonsai-2-27B alone." if AUTO_CLOSE
             else "It does not close PRs on a verdict.")
    return f"{merge} {close}"


def _reg_row(res: dict) -> str:
    """The bonsai_regression.py row: which checks gated, which main also failed, and any note."""
    main_failed = res.get("bonsaireg_main_failed") or []
    notes = res.get("bonsaireg_notes") or []
    note = (" · note: " + "; ".join(notes[:2])) if notes else ""
    if res.get("bonsaireg_gated_fail"):
        why = "; ".join((res.get("bonsaireg_why") or [])[:3]) or "see log"
        return f"| bonsai_regression.py | ❌ **FAILED** — {why} — **verdict forced to REJECT** |\n"
    if "*" in main_failed:
        state = "passes" if res.get("bonsaireg_pr_ok") else "fails"
        return (f"| bonsai_regression.py | ⚠️ not gated — it did not complete on main this round (the "
                f"PR {state} it); a failure already on main cannot reject a PR{note} |\n")
    if res.get("bonsaireg_pr_ok"):
        extra = f" (main fails {', '.join(main_failed)} this round)" if main_failed else ""
        return f"| bonsai_regression.py | ✅ tensors · score · generate · serve{extra}{note} |\n"
    both = ", ".join(res.get("bonsaireg_pr_failed") or [])
    return (f"| bonsai_regression.py | ⚠️ not gated for {both} — main fails it too this round, so it "
            f"cannot reject a PR; every other check passed{note} |\n")


def _gate_row(name: str, ok, detail_ok: str, detail_fail: str, skipped: str = "") -> str:
    if skipped:
        return f"| {name} | ⚠️ SKIPPED — {skipped} |\n"
    if ok:
        return f"| {name} | ✅ {detail_ok} |\n"
    return f"| {name} | ❌ **FAILED** — {detail_fail} — **verdict forced to REJECT** |\n"


def format_comment(commit: str, res: dict) -> str:
    label = res.get("label") if res.get("ok") else "REJECT"
    meta = {
        "label": label,
        "delta_pct": res.get("delta_pct"),
        "best_dim": res.get("best_dim"),
        "pr_decode_tps": res.get("pr_decode_tps"),
        "main_decode_tps": res.get("main_decode_tps"),
        "pr_prefill128_pp": res.get("pr_prefill128_pp"),
        "main_prefill128_pp": res.get("main_prefill128_pp"),
        "pass": res.get("pass"),
        "accuracy_ok": res.get("accuracy_ok"),
        "prefill_path_ok": res.get("prefill_path_ok"),
        "bonsaireg_ok": res.get("bonsaireg_pr_ok"),
        "guards_ok": {k: v.get("ok") for k, v in (res.get("guards") or {}).items()},
        "dims": {d["dim"]: {"delta": d["delta"], "label": d["label"]}
                 for d in (res.get("scored_dims") or [])},
    }
    marker = (f"<!-- sparkinfer-bonsai-eval:{EVAL_SCHEMA_VERSION}:{commit} "
              f"{json.dumps(meta, separators=(',', ':'))} -->")
    if not res.get("ok"):
        return (
            f"{marker}\n## sparkinfer bonsai auto-eval — `eval-bonsai:REJECT` (run failed)\n\n"
            f"The PR's build or run on Ternary-Bonsai-2-27B failed on the pinned eval box, where "
            f"the same-round `origin/main` run succeeded.\n\n"
            f"**reason:** `{res.get('reason')}`\n\n"
            f"<details><summary>log tail</summary>\n\n```\n{(res.get('log') or '')[:1800]}\n```\n</details>\n\n"
            f"<sub>The PR is built merged onto the round's `main`, not on its own branch: an error in a file the PR does not touch usually means it needs a rebase onto current `main`. Recorded for this commit; a new push is evaluated again.</sub>\n"
        )
    acc = (f"top-1 {res.get('pr_top1', 0):.4f} (bar ≥{ACC_TOP1_BAR}) · KL {res.get('pr_kl', 0):.5f} "
           f"(bar ≤{ACC_KL_BAR}) · PPL ×{res.get('ppl_ratio') or 0:.4f} of main (bar ≤{ACC_PPL_RATIO})")
    acc_row = _gate_row("accuracy vs main (teacher-forced)", res.get("accuracy_ok"),
                        f"{acc} over {res.get('token_count') or '?'} tokens", acc)
    pf_row = _gate_row(
        "prefill path vs main", res.get("prefill_path_ok"),
        f"batched prefill within main's spread at prefix {'/'.join(str(p) for p in PF_PREFIXES)}",
        "; ".join((res.get("prefill_path_problems") or [])[:3]))
    reg_row = _reg_row(res)
    guard_rows = ""
    for key, _tag, name in GUARDS:
        g = (res.get("guards") or {}).get(key) or {}
        guard_rows += _gate_row(
            f"{name} guard", g.get("ok"), "no regression (decode + prefill @ 32k)",
            "; ".join((g.get("problems") or [])[:4]),
            skipped=("checkpoint not installed on the box; not checked" if g.get("skipped")
                     else g.get("not_run") or ""))
    polaris = res.get("polaris") or {}
    receipt = polaris.get("receipt")
    if receipt:
        rtype = ("TDX (Intel hardware attestation)" if receipt.get("attestation_type") == "tdx-quote"
                 else "Ed25519 (SparkInfer key)")
        polaris_row = f"| Polaris receipt | `{receipt.get('receipt_id', '?')[:16]}…` — {rtype} |\n"
    elif polaris.get("attestation"):
        polaris_row = "| Polaris receipt | collected, not signed (no key configured) |\n"
    else:
        polaris_row = ""
    return (
        f"{marker}\n## sparkinfer bonsai auto-eval — `eval-bonsai:{label}`\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| **label** | `eval-bonsai:{label}` |\n"
        f"| model | Ternary-Bonsai-2-27B PTQ1_0 GGUF, default (folded) loader |\n"
        f"| scored at | decode + prefill @ {_ctx_list_str()} · concurrent decode @ {_cb_list_str()} — "
        f"{len(SCORING_DIMS)} axes, each also a regression floor; the label is the best |\n"
        f"| tier from | `{res.get('best_dim') or '?'}` ({res.get('delta_pct', 0):+.1f}%) |\n"
        f"{acc_row}{pf_row}{reg_row}{guard_rows}"
        f"| PPL PR / main | {res.get('pr_ppl') or '?'} / {res.get('main_ppl') or '?'} |\n"
        f"{polaris_row}"
        f"| commit | `{commit[:9]}`"
        + (f", measured merged onto `main` `{res['merged_onto']}` (this round's baseline)"
           if res.get("merged_onto") else "")
        + " |\n\n"
        f"{_matrix_table(res)}"
        f"{res.get('reason') or ''}\n\n"
        f"<sub>Measured on the pinned RTX 5090 against a same-box `origin/main` from the same round. "
        f"Any axis regressing below {100 * REGRESS_TOL:.0f}% of main is a REJECT; otherwise the label "
        f"is the best measured delta. `none` only means no Ternary-Bonsai-2-27B speedup was "
        f"measured, which is expected for a change aimed at another model. {_policy_note()}</sub>\n"
    )


def auto_merge_ok_bonsai(repo, num, require_merge_first=True, ranking_loss_ok=False):
    """Can this PR be merged now? With require_merge_first=False it is the reconcile's question
    instead -- may this PR be MADE merge-first -- which must be the same test minus that one label:
    a winner whose merge would then be refused holds merge-first round after round while every
    other speedup PR is pushed to needs-rebase (#1154, 2026-09-24, for a hold; any refusal below
    did the same until 2026-09-26).
    ranking_loss_ok: the bot's own bonsai-needs-rebase does not count -- reconcile's call for a PR
    sent there only for losing an earlier ranking (_waits_for_the_winner)."""
    try:
        info = json.loads(arb.gh(["pr", "view", str(num), "-R", repo, "--json",
                                  "state,isDraft,labels,author,mergeable,files,changedFiles,headRefOid,baseRefName"]).stdout or "{}")
    except json.JSONDecodeError:
        info = None
    if not isinstance(info, dict) or not info:
        return False, arb.PR_UNREADABLE
    if info.get("state") != "OPEN" or info.get("isDraft"):
        return False, "not an open, non-draft PR"
    labs = {l["name"] for l in info.get("labels", [])}
    tiers = {l.split(":", 1)[1] for l in labs if l.startswith(EVAL_PREFIX)}
    if not (tiers & SPEEDUP_LABELS):
        return False, "no verified eval-bonsai speedup label"
    if require_merge_first and BONSAI_MERGE_FIRST not in labs:
        return False, "not bonsai-merge-first"
    # Merge only the exact commit this bot scored. The label and bonsai-merge-first survive a push
    # made after the verdict, and if the next round's re-measurement fails on the box side nothing
    # is posted and the stale label stays -- so gating on the label alone could merge unmeasured
    # code. The sibling bots do not check this; a fresh bot with auto-merge on should.
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
    blocked = labs & (AUTOMERGE_BLOCK - ({BONSAI_NEEDS_REBASE} if ranking_loss_ok else set()))
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
    # GitHub reports UNKNOWN for a while after main moves; that must not cost a PR the ranking
    # (the merge itself re-checks). A conflict does.
    if info.get("mergeable") != "MERGEABLE" and (require_merge_first
                                                 or info.get("mergeable") == "CONFLICTING"):
        return False, f"not cleanly mergeable ({info.get('mergeable')})"
    # Last: measured against the main that is there now. Once any bot has merged something else,
    # this PR merged onto the new main is a combination nobody measured, so it is re-measured
    # first (main()'s selection). Reconcile keeps a PR refused for this alone in the running
    # (arb.refused_only_for_stale_main), which is why no other refusal may come after it.
    return arb.fresh_against_main(repo, scored)


def try_auto_merge_bonsai(repo, num):
    ok, reason = auto_merge_ok_bonsai(repo, num)
    if not ok:
        print(f">> bonsai auto-merge SKIP #{num}: {reason}")
        return False
    # Pin the merge to the SCORED commit, which auto_merge_ok_bonsai just found equal to the head. A
    # second head lookup here would pin whatever a push had made the head in between -- unscored.
    head = (_load_scores().get(str(num)) or {}).get("commit") or ""
    if not arb._FULL_SHA_RE.match(head):
        print(f">> bonsai auto-merge SKIP #{num}: no scored commit to pin the merge to")
        return False
    args = ["pr", "merge", str(num), "-R", repo, "--squash", "--match-head-commit", head]
    r = arb.gh(args)
    if r.returncode != 0 and os.environ.get("SPARKINFER_AUTOMERGE_ADMIN", "1") == "1":
        # Same branch-policy retry as the sibling bots: a required check that is "expected" but never
        # runs on this repo would otherwise block every auto-merge. --admin still honours
        # --match-head-commit, so it cannot push through a commit other than the scored one.
        err = ((r.stderr or "") + (r.stdout or "")).lower()
        if "not mergeable" in err or "branch policy" in err or "required" in err or "prohibited" in err:
            print(">> bonsai auto-merge: branch policy blocked — retrying with --admin")
            r = arb.gh(args + ["--admin"])
    if r.returncode == 0:
        print(f">> BONSAI AUTO-MERGED #{num} (bonsai-merge-first)")
        arb.gh(["pr", "comment", str(num), "-R", repo, "--body",
                "<!-- sparkinfer-bonsai-automerge -->\n"
                "Auto-merged as the round's `bonsai-merge-first` winner — verified same-box "
                "Ternary-Bonsai-2-27B speedup over `main`, with every accuracy gate and "
                "cross-model guard passing."])
        return True
    print(f">> bonsai auto-merge BLOCKED #{num}: {(r.stderr or r.stdout or '')[:200]}")
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
    skip = arb.model_skip_reason(pr.get("body") or "", "bonsai")
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


def reconcile_bonsai_merge_labels(repo, dry_run=False):
    scores = _load_scores()
    open_prs = arb.open_prs_or_none(repo, "number,labels,isDraft,body,files,mergeable,headRefOid,baseRefName")
    if open_prs is None:
        print(">> bonsai round: GitHub did not return the open PRs — labels left as they are")
        return
    merged = json.loads(arb.gh(["pr", "list", "-R", repo, "--state", "merged", "--label",
                                BONSAI_MERGE_FIRST, "--json", "number", "--limit", "10"]).stdout or "[]")
    if not dry_run:
        for m in merged:
            arb.remove_label(repo, m["number"], BONSAI_MERGE_FIRST)
    scored = []
    stale_first = []   # carries merge-first but can no longer win it
    stale_main = set()   # in the running, but its merge waits for a re-measure onto today's main
    main_now = None      # read once, for a needs-rebase that may only mean a lost ranking
    for p in open_prs:
        labs = {l["name"] for l in p["labels"]}
        if not dry_run:
            labs = arb.repair_own_tier(repo, p["number"], labs, EVAL_PREFIX, scores.get(str(p["number"])),
                                       (p.get("headRefOid") or "")[:40],
                                       lambda: _verdict_heads(repo, p["number"]))
        # A sync GitHub did not answer when this bot posted its verdict, healed -- on this bot's PRs
        # only: the retired AR bot's labels derive the generic one by another rule (the failing side).
        if not dry_run and any(l.startswith(EVAL_PREFIX) for l in labs) and arb.generic_label_out_of_sync(labs):
            arb.sync_generic_eval_label(repo, p["number"])
        # A PR that cannot be merged -- `hold`, needs-rebase, a penalty or copycat flag, any other
        # AUTOMERGE_BLOCK label -- must not take merge-first either. It used to: on 2026-09-24
        # #1154 was held for review with the round's best score, won merge-first, had its merge
        # refused, and pushed the next-best PR to needs-rebase for a merge that never happened --
        # every round, for as long as the hold lasted.
        lost_only = False
        if (labs & AUTOMERGE_BLOCK) == {BONSAI_NEEDS_REBASE}:
            # Sent to needs-rebase only for losing an earlier ranking, with its verdict still standing
            # on today's main: it stays in the running. Left out, a worse PR merged first once the
            # winner was re-measured lower, and nothing merged at all once the winner was closed or
            # held. After main moves, the rebase is its author's (CONTRIBUTING).
            if main_now is None:
                main_now = arb.current_main_sha(repo) or ""
            lost_only = bool(main_now) and _waits_for_the_winner(p, labs, (p.get("headRefOid") or "")[:40], main_now)
        if labs & AUTOMERGE_BLOCK and not lost_only:
            if BONSAI_MERGE_FIRST in labs:
                stale_first.append(p["number"])
            continue
        tier = next((l.split(":", 1)[1] for l in labs
                     if l.startswith(EVAL_PREFIX) and l.split(":", 1)[1] in SPEEDUP_LABELS), None)
        if not tier:
            # No speedup tier (any more): its head moved, or a re-measure found none. A merge-first
            # left here exempted it from every close and could sit beside the next winner's.
            if BONSAI_MERGE_FIRST in labs:
                stale_first.append(p["number"])
            continue
        # Everything else auto-merge would refuse -- a head that moved past the scored commit, a
        # REJECT from another bot, a penalty, a protected path, a conflict -- must not win either.
        ok, why = auto_merge_ok_bonsai(repo, p["number"], require_merge_first=False,
                                       ranking_loss_ok=lost_only)
        if not ok and why == arb.PR_UNREADABLE:
            # Not an answer: demoting on it would take merge-first from the real holder.
            print(f">> bonsai round: GitHub did not return #{p['number']} — labels left as they are")
            return
        if not ok and arb.refused_only_for_stale_main(why):
            # Nothing but a moved main stands in the way: it keeps its place and is re-measured
            # (main()'s selection); only the merge waits for that. Demoting it here let the stale
            # close shut a verified winner the bot itself owed a measurement. Only if the
            # selection will re-measure it, though.
            blocker = _unmeasurable_reason(repo, p, labs)
            if blocker:
                ok, why = False, f"{why}; not re-measured: {blocker}"
            else:
                print(f">> bonsai round: #{p['number']} stays in the running, merge waits ({why})")
                stale_main.add(p["number"])
        if not ok and p["number"] not in stale_main:
            print(f">> bonsai round: #{p['number']} cannot be merge-first ({why})")
            if BONSAI_MERGE_FIRST in labs:
                stale_first.append(p["number"])
            continue
        entry = scores.get(str(p["number"])) or {}
        scored.append((p["number"], float(entry.get("delta_pct") or 0)))
    # A PR that can merge now outranks one still waiting for its re-measure.
    scored.sort(key=lambda x: (x[0] not in stale_main, x[1]), reverse=True)
    if not dry_run:
        for num in stale_first:
            arb.remove_label(repo, num, BONSAI_MERGE_FIRST)
    if not scored:
        print(">> bonsai round: no verified speedup PRs")
        return
    winner = scored[0][0]
    print(f">> bonsai round: merge-first #{winner}; rebase {[n for n, _ in scored[1:]] or 'none'}")
    if dry_run:
        return
    arb.add_label(repo, winner, BONSAI_MERGE_FIRST)
    arb.remove_label(repo, winner, BONSAI_NEEDS_REBASE)
    for num, _ in scored[1:]:
        # Nothing merges this round while the winner waits for its re-measure: no one needs a rebase.
        if winner not in stale_main:
            arb.add_label(repo, num, BONSAI_NEEDS_REBASE)
        arb.remove_label(repo, num, BONSAI_MERGE_FIRST)
    if AUTO_MERGE:
        try_auto_merge_bonsai(repo, winner)


def _axis_matrix(res: dict) -> dict:
    """{scored axis: {"pr", "main", "delta", "label"}} for every axis in res["scored_dims"].

    The PR/main readings come from the same parsed matrices the verdict was computed from, so the
    two can never disagree. An axis a run did not measure is simply absent."""
    vals = {}
    pr_m, main_m = res.get("bonsai_pr") or {}, res.get("bonsai_main") or {}
    for ctx in SCORED_CTXS:
        for phase in ("decode", "prefill"):
            pv = (pr_m.get(ctx) or {}).get(phase)
            mv = (main_m.get(ctx) or {}).get(phase)
            vals[f"bonsai-{phase}@{SCORED_CTX_LABEL[ctx]}"] = (pv, mv)
    cb_pr, cb_main = res.get("cb_pr") or {}, res.get("cb_main") or {}
    for conc in CB_CONCS:
        vals[CB_DIM_FOR[conc]] = (cb_pr.get(conc), cb_main.get(conc))
    out = {}
    for d in res.get("scored_dims") or []:
        pv, mv = vals.get(d["dim"], (None, None))
        out[d["dim"]] = {"pr": pv, "main": mv, "delta": d["delta"], "label": d["label"]}
    return out


def upload_bonsai_eval_log(repo, num, title, oid, res):
    """Commit the result (+ Polaris receipt) to sparkinfer-log, as the sibling bots do."""
    try:
        arb._ensure_log_repo()
        rid = arb.eval_log_run_id(f"bonsai-{int(num):04d}-{oid[:7]}", res.get("onto"))
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
            "eval_mode": "bonsai-128",
            "label": res.get("label"), "pass": res.get("pass"), "reason": res.get("reason"),
            "delta_pct": res.get("delta_pct"), "best_dim": res.get("best_dim"),
            "dims": {d["dim"]: d["delta"] for d in (res.get("scored_dims") or [])},
            # Raw PR / main reading behind every scored axis. `dims` alone carries only the
            # delta, which is not enough for a consumer that charts absolute throughput (the
            # sparkinfer-web Ternary-Bonsai track plots the prefill@16k journey from these).
            "matrix": _axis_matrix(res),
            "pr_decode_tps": res.get("pr_decode_tps"), "main_decode_tps": res.get("main_decode_tps"),
            "pr_prefill128_pp": res.get("pr_prefill128_pp"),
            "main_prefill128_pp": res.get("main_prefill128_pp"),
            "pr_top1": res.get("pr_top1"), "pr_kl": res.get("pr_kl"),
            "accuracy_ok": res.get("accuracy_ok"), "prefill_path_ok": res.get("prefill_path_ok"),
            "bonsaireg_ok": res.get("bonsaireg_pr_ok"),
            "guards_ok": {k: v.get("ok") for k, v in (res.get("guards") or {}).items()},
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
        entry = {"id": rid, "pr": int(num), "title": title, "label": res.get("label"),
                 "delta_pct": res.get("delta_pct"), "eval_mode": "bonsai-128", "date": result["date"]}
        if receipt:
            entry["polaris"] = True
            entry["polaris_receipt_id"] = receipt.get("receipt_id", "")[:16]
        idx.append(entry)
        idx.sort(key=lambda x: x["id"])
        json.dump(idx, open(ipath, "w"), indent=2)
        subprocess.run(["git", "-C", arb.LOG_DIR, "add", "-A"], check=True)
        msg = f"bonsai-eval: #{num} {oid[:7]} -> eval-bonsai:{res.get('label')}"
        if subprocess.run(["git", "-C", arb.LOG_DIR, "commit", "-q", "-m", msg], check=False).returncode != 0:
            print(">> bonsai eval-log upload skipped: nothing to commit")
            return None
        if subprocess.run(["git", "-C", arb.LOG_DIR, "push", "-q"], check=False, timeout=300).returncode != 0:
            print(">> bonsai eval-log push failed")
            return None
        print(f">> bonsai eval log: {arb.LOG_PAGE + rid}")
        return arb.LOG_PAGE + rid
    except Exception as e:
        print(f">> bonsai eval-log upload failed: {e}")
        return None


def _closes_on_none(body: str, labels=()) -> bool:
    """`none` closes only a PR that declares this model and no other, and that no other bot scored a
    speedup or made merge-first (module docstring; arb.none_may_close, shared with the siblings)."""
    return arb.none_may_close(body, labels, "bonsai", EVAL_PREFIX)


def apply_result(repo, num, commit, res, title="", dry_run=False, body=""):
    if not res.get("ok") and res.get("harness"):
        print(f"PR #{num}: {res.get('reason')} — edits the eval harness, not evaluated")
        if not dry_run:
            # Remembered for this head: GitHub lists at most 100 files, so the selection's own check
            # can miss the edit and the PR would be measured (and stopped) again every round.
            arb.record_strike(STRIKES_FILE, num, commit, "harness")
        return
    if not res.get("ok") and res.get("conflict"):
        print(f"PR #{num}: {res.get('reason')} — bonsai-needs-rebase, no verdict")
        if not dry_run:
            arb.add_label(repo, num, BONSAI_NEEDS_REBASE)
            # Remembered for this head: GitHub may keep calling it mergeable, and the label would be
            # dropped and the PR measured again every round (arb.strip_stale_verdict_labels).
            arb.record_strike(STRIKES_FILE, num, commit, "conflict")
        return
    if not res.get("ok") and res.get("retry"):
        # Infrastructure: nothing is posted and no label changes. The next round measures again --
        # except a box-shaped fault that keeps recurring at one commit (a build that exhausts the
        # compiler even at -j4), which after BOX_FAULT_STRIKES rounds is posted as a failed run.
        n = record_strike(num, commit, res["strike_key"]) if res.get("strike_key") and not dry_run else 0
        if n < BOX_FAULT_STRIKES:
            print(f"PR #{num}: bonsai eval deferred — {res.get('reason')} "
                  f"({'the bot itself' if res.get('strike_key') == 'error' else 'infra'}; re-evaluated next "
                  f"round{f', strike {n} of {BOX_FAULT_STRIKES}' if n else ''})")
            return
        if res.get("strike_key") == "error":
            # The bot's own failure, not the PR's: never charged. It stops measuring this commit
            # (arb.gave_up) instead of retrying it every round for ever.
            print(f"!! PR #{num}: the bot itself failed on {commit[:9]} {n} rounds in a row "
                  f"({res.get('reason')}) — not measured again until a push; nothing posted")
            GAVE_UP.add(num)
            return
        res = dict(res, retry=False,
                   reason=f"{res.get('reason')} — {n} rounds in a row at this commit, so it is "
                          f"charged to the PR")
    if res.get("ok") and res.get("strike_key"):
        # A REJECT whose only cause is a concurrency width the PR build could not complete. Judged
        # over two rounds on the same commit: the first time nothing is posted.
        if dry_run:
            print(f"PR #{num}: would count a strike for {res['strike_key']} "
                  f"(REJECT on strike {STRIKES_TO_REJECT})")
        else:
            n = record_strike(num, commit, res["strike_key"])
            if n < STRIKES_TO_REJECT:
                print(f"PR #{num}: {res.get('reason', '').split(' | ')[0]} — strike {n} of "
                      f"{STRIKES_TO_REJECT}; nothing posted, measured again next round")
                return
    label = res.get("label") if res.get("ok") else "REJECT"
    comment = format_comment(commit, res)
    print(f"PR #{num}: eval-bonsai:{label}  from={res.get('best_dim')} delta={res.get('delta_pct')}%  "
          f"accuracy_ok={res.get('accuracy_ok')} prefill_path_ok={res.get('prefill_path_ok')} "
          f"bonsaireg_ok={res.get('bonsaireg_pr_ok')} "
          f"guards_ok={ {k: v.get('ok') for k, v in (res.get('guards') or {}).items()} }")
    if dry_run:
        print(comment)
        return
    clear_strikes(num)
    strip_bonsai_eval_labels(repo, num)
    if label in SPEEDUP_LABELS:
        # A fresh speedup for the current head makes this PR eligible for merge-first again
        # (the sibling bots' #790/#791 fix).
        arb.remove_label(repo, num, BONSAI_NEEDS_REBASE)
    arb.add_label(repo, num, f"{EVAL_PREFIX}{label}")
    # Mirrored to the generic eval:* label (explicit decision 2026-09-24), derived from every
    # per-bot label so a `none` here cannot erase another model's real tier.
    arb.sync_generic_eval_label(repo, num)
    # Whether the verdict posted: a close carries it instead when it did not (below).
    posted = getattr(arb.gh(["pr", "comment", str(num), "-R", repo, "--body", arb.fit_comment(comment)]),
                     "returncode", 0) == 0
    if not res.get("ok"):
        arb.record_posted_verdict(_load_scores, _save_scores, num, commit, label, res)
        return
    # Scores first: a run that dies in the (network) log upload must not leave a posted verdict the
    # bot can neither merge nor re-measure.
    if res.get("delta_pct") is not None:
        scores = _load_scores()
        scores[str(num)] = {
            "commit": commit, "label": label, "delta_pct": res.get("delta_pct"),
            "best_dim": res.get("best_dim"), "pass": res.get("pass"),
            "accuracy_ok": res.get("accuracy_ok"), "prefill_path_ok": res.get("prefill_path_ok"),
            "onto": res.get("onto"),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _save_scores(scores)
    else:
        arb.record_posted_verdict(_load_scores, _save_scores, num, commit, label, res)
    upload_bonsai_eval_log(repo, num, title, commit, res)
    # Closing (module docstring): a measured REJECT closes; `none` closes only a PR declared for
    # this model alone -- this bot also scores every undeclared PR, most of them aimed elsewhere.
    if not AUTO_CLOSE:
        return
    if label == "REJECT":
        first = (res.get("reason") or "").split(" | ")[0]
        close_body = (
            "<!-- sparkinfer-bonsai-auto-close -->\n"
            "## Closed: regression or failed gate — `eval-bonsai:REJECT`\n\n"
            f"Measured on the pinned RTX 5090 against the same-box `origin/main`: {first}.\n\n"
            "Every scored axis is also a no-regression floor, and the accuracy gates and "
            "cross-model guards are hard gates, so one failure closes the PR whatever it was "
            "aiming at. The verdict comment above has the full table. Push a fix and reopen: the "
            "new commit is evaluated on the next round (reopening alone does not re-run a commit "
            "that already has its verdict).")
    elif label == "none" and _closes_on_none(body, arb.labels_on_or_none(repo, num)):
        close_body = (
            "<!-- sparkinfer-bonsai-auto-close -->\n"
            "## Closed: no verified speedup — `eval-bonsai:none`\n\n"
            f"Measured on the pinned RTX 5090 against the same-box `origin/main`: "
            f"**{res.get('delta_pct')}%** on the best of {len(SCORING_DIMS)} scored axes "
            f"(decode + prefill @ {_ctx_list_str()}, concurrent decode @ {_cb_list_str()}).\n\n"
            "**This is not a finding that anything is wrong with your PR.** Nothing regressed and "
            "every correctness gate passed; the change just did not move a number this bot "
            "measures.\n\n"
            "- **Nothing here measures your optimization yet?** Open an issue describing the axis "
            "you need, with your before/after numbers, then reopen and ask for the `hold` label.\n"
            "- **Correctness fix, refactor, test or docs?** Reopen as a **draft** or ask for "
            "`hold`; those are reviewed by hand.\n"
            "- **Expected a speedup?** Re-measure against current `main`, push the change as a new "
            "commit (a rebase counts) and reopen with the new numbers; the new commit is evaluated "
            "on the next round.")
    else:
        return
    # Not over a commit the author has already replaced (a push landing while this PR was being
    # measured gets its own evaluation next round), nor over a `hold` or a draft made meanwhile.
    if not posted:
        # The verdict travels with the close, marker included. Left open instead, its REJECT label
        # without a marker read to the other bots as a tier no verdict backs: they dropped it, and
        # one of them merged the PR.
        close_body = comment + "\n\n---\n\n" + close_body
    why = arb.verdict_close_blocker(repo, num, commit)
    if why:
        print(f">> PR #{num}: not closed — {why}")
        return
    arb.gh(["pr", "comment", str(num), "-R", repo, "--body", arb.fit_comment(close_body)])
    arb.gh(["pr", "close", str(num), "-R", repo])
    print(f">> auto-closed PR #{num} (eval-bonsai:{label})")


def _exit_if_gave_up():
    """Exit 3 when the selection gave up on a PR after the bot's own errors (GAVE_UP), so the
    wrapper's failed-run banner shows it -- on every way out of a run, one that then measured
    nothing (the GPU down, the lock busy) included."""
    if GAVE_UP:
        print(f"!! bonsai: gave up on {', '.join(f'#{n}' for n in sorted(GAVE_UP))} after the bot's own "
              f"errors — see above; exiting 3 so the wrapper's failed-run banner shows it")
        sys.exit(3)


def main():
    ap = argparse.ArgumentParser(description="Ternary-Bonsai-2-27B PR eval bot")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--repo", default="gittensor-ai-lab/sparkinfer")
    ap.add_argument("--dry-run", action="store_true", help="list what would be evaluated; no GPU")
    ap.add_argument("--no-post", action="store_true",
                    help="evaluate on the GPU and print the verdicts, but change nothing on GitHub")
    ap.add_argument("--reeval", action="store_true")
    ap.add_argument("--labels-only", action="store_true", help="reconcile bonsai-merge-first only — no GPU")
    ap.add_argument("--only-prs", default="",
                    help="comma-separated PR numbers (bypasses greenlight; with --no-post also "
                         "the hold and draft filters, since nothing is posted)")
    args = ap.parse_args()
    only = {int(x) for x in args.only_prs.split(",") if x.strip().isdigit()}
    quiet = args.dry_run or args.no_post
    # A held or draft PR is never acted on. A report-only run may still measure one by name.
    look_only = bool(only) and args.no_post

    print(f">> bonsai eval transport: "
          f"{'ssh' if ssh_box_enabled() else f'vast.ai (instance {arb.current_instance(args.instance) or args.instance})'}")
    print(f">> AUTOMERGE={int(AUTO_MERGE)} AUTOCLOSE={int(AUTO_CLOSE)} NO_POST={int(args.no_post)}")

    ok, login = arb.acting_account_ok()
    if not ok:
        print(f"!! gh acts as {login}, not SPARKINFER_BOT_LOGIN={os.environ.get('SPARKINFER_BOT_LOGIN')} — nothing done")
        sys.exit(3)
    if args.labels_only:
        reconcile_bonsai_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — bonsai labels only (no GPU).")
        return

    prs = arb.open_prs_or_none(args.repo, "number,title,labels,isDraft,headRefOid,headRefName,baseRefName,"
                                          "mergeable,author,body,files,changedFiles")
    if prs is None:
        # Not an empty queue: GitHub did not answer (an expired token, an outage). Exit non-zero so the
        # wrapper's failed-run banner shows a bot that has stopped seeing PRs.
        print("!! GitHub did not return the open PRs — nothing done this run")
        sys.exit(3)
    prs.sort(key=lambda p: p["number"])

    stale_closed = close_stale_bonsai_prs(args.repo, prs, dry_run=quiet) if not only else set()
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
        removed = arb.strip_foreign_stale_labels(args.repo, num, labs0, head0, EVAL_PREFIX) if not quiet else set()
        if removed:
            print(f"PR #{num} @ {head0[:9]}: dropped {', '.join(sorted(removed))} (no verdict on this head backs them)")
            labs0 = labs0 - removed
            pr["labels"] = [{"name": l} for l in sorted(labs0)]
        if (not quiet and (pr.get("isDraft") or arb.HOLD_LABEL in labs0)
                and any(l.startswith(EVAL_PREFIX) or l == BONSAI_NEEDS_REBASE for l in labs0)
                and arb.strip_stale_verdict_labels(args.repo, num, labs0, EVAL_PREFIX, head0,
                                                   _verdict_heads(args.repo, num), BONSAI_NEEDS_REBASE,
                                                   arb.pr_merge_conflict(pr.get("mergeable"))
                or bool(arb.strike_count(STRIKES_FILE, num, (pr.get("headRefOid") or "")[:40], "conflict")))):
            print(f"PR #{num} @ {head0[:9]}: no bonsai verdict for this head yet — dropped the old eval-bonsai label")
        if pr.get("isDraft") and not look_only:
            continue
        hits = arb.pr_involved_logins(args.repo, num) & denylist
        if hits:
            print(f"PR #{num}: BLOCKED (denylisted: {', '.join(sorted(hits))}) — flag + close, no eval")
            if not quiet:
                arb.close_blocked_pr(args.repo, num, hits)
            continue
        labs = {l["name"] for l in pr.get("labels", [])}
        if arb.HOLD_LABEL in labs and not look_only:
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
        # An eval-bonsai tier measured on an older head no longer describes this PR.
        if not quiet and arb.strip_stale_verdict_labels(
                args.repo, num, labs, EVAL_PREFIX, head, evaluated, BONSAI_NEEDS_REBASE,
                arb.pr_merge_conflict(pr.get("mergeable"))
                or bool(arb.strike_count(STRIKES_FILE, num, (pr.get("headRefOid") or "")[:40], "conflict"))):
            print(f"PR #{num} @ {short}: no bonsai verdict for this head yet — dropped the old eval-bonsai label")
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
            print(f"PR #{num} @ {short}: does not merge onto main on the box — bonsai-needs-rebase until a push")
            continue
        if not args.reeval and head and head in evaluated:
            if not _remeasure_against_new_main(args.repo, num, head, labs, main_now):
                print(f"PR #{num} @ {short}: already bonsai-evaluated — skip")
                continue
            remeasure = True
        # Skipped only when the author ticked specific other models and not this one or shared
        # (fails open). The sibling bots guard Ternary-Bonsai-2-27B on the PRs this skips.
        skip_why = arb.model_skip_reason(pr.get("body") or "", "bonsai")
        if skip_why:
            print(f"PR #{num}: {skip_why} — skip bonsai eval")
            continue
        touched = [f.get("path", "") for f in (pr.get("files") or [])]
        harness_hits = [t for t in touched if any(t.startswith(h) for h in HARNESS_PATHS)]
        if harness_hits:
            print(f"PR #{num}: touches the eval harness ({', '.join(harness_hits[:3])}) — not evaluated")
            continue
        if arb.pr_merge_conflict(pr.get("mergeable")):
            print(f"PR #{num}: merge conflict — bonsai-needs-rebase")
            if not quiet:
                arb.add_label(args.repo, num, BONSAI_NEEDS_REBASE)
            continue
        if not only:
            status, why = arb.greenlight_status(args.repo, num, labs)
            if status != "ok":
                print(f"PR #{num}: not greenlit ({why}) — skip bonsai eval")
                continue
            print(f"PR #{num}: greenlit ({why})")
        else:
            print(f"PR #{num}: --only-prs targeted")
        # Measured as its tip merged onto the round's main baseline commit, built on the box
        # (arb.merged_checkout_script) -- not GitHub's pull/<n>/merge, which can be stale (#1145).
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
        reconcile_bonsai_merge_labels(args.repo, dry_run=quiet)
        print("done — no bonsai PRs to evaluate.")
        _exit_if_gave_up()
        return
    if args.dry_run:
        print("--- dry-run would evaluate: " + ", ".join(f"#{p[0]} ({p[3]})" for p in pending))
        return

    pin = arb.PINNED_INSTANCE
    if pin and not ssh_box_enabled():
        with open(arb.INSTANCE_FILE, "w") as f:
            f.write(str(pin))
    try:
        host, port = resolve_ssh(args.instance)
    except Exception as e:
        print(f">> GPU unavailable: {e}")
        reconcile_bonsai_merge_labels(args.repo, dry_run=quiet)
        print("done — bonsai labels only (GPU down).")
        _exit_if_gave_up()
        return
    print(f">> SSH {ssh_box_user() if ssh_box_enabled() else 'root'}@{host}:{port}")

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
        print(f">> main baseline measurement failed: {main_result.get('reason')} — skipping round")
        print((main_result.get("log") or "")[-1500:])
        reconcile_bonsai_merge_labels(args.repo, dry_run=quiet)
        print("done — bonsai round skipped (main baseline unusable).")
        # Non-zero: a main that stays unusable stops this bot measuring anything, the PR fixing it
        # included -- run_bot (cron_common.sh) makes a run of these loud.
        sys.exit(3)
    print(">> main baseline: " + "  ".join(
        f"@{SCORED_CTX_LABEL[c]} {_at(main_result, c, 'decode'):.1f}/{_at(main_result, c, 'prefill'):.1f}"
        for c in SCORED_CTXS) + f"  cb {main_result.get('bonsai_cb')}")
    print(f">> main prefill-path: {main_result.get('pfcheck')}  bonsai_regression "
          f"{'OK' if main_result.get('bonsaireg_ok') else 'FAILED ' + str(main_result.get('bonsaireg_why'))}")
    print(f">> main guard coverage: {_guard_coverage(main_result)}")

    for num, head, short, ref, title, body in pending:
        print(f"PR #{num} @ {short}: evaluating Ternary-Bonsai-2-27B '{ref}' …")
        try:
            res = eval_bonsai_on_box(host, port, ref, main_result)
        except Exception as e:
            res = arb.exception_result(e)   # retried; a 2 h hang charged after BOX_FAULT_STRIKES rounds
        # Recorded against the tip the box built, which a mid-round push can make differ from
        # the listed head (arb.measured_commit, #1167).
        commit, moved = arb.measured_commit(head, res)
        if moved:
            print(f">> PR #{num} moved during the round: listed {short}, measured {commit[:9]} — "
                  f"recording {commit[:9]}")
        apply_result(args.repo, num, commit or short, res, title=title, dry_run=args.no_post,
                     body=body)

    reconcile_bonsai_merge_labels(args.repo, dry_run=quiet)
    print("done — bonsai eval pass complete.")
    _exit_if_gave_up()


if __name__ == "__main__":
    main()
