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
          Gated only when main passes it in the same round, so a failure already on main is
          reported but cannot reject every PR.
  3. Cross-model no-regression guards @ 32k, decode + prefill: Qwen3.6-35B-A3B, the ModelOpt and
     unsloth Qwen3.8-27B NVFP4 checkpoints, and Muse Glimmer. The Muse and Qwen3.8 bots skip a PR
     declared for Ternary-Bonsai-2-27B alone (arb.model_skip_reason), so these guards are the only
     check such a PR gets against those models. Absent checkpoint -> SKIPPED and said so;
     unmeasured -> infra, retried next round; measured regression -> REJECT.

Policy, by explicit decision 2026-09-24: tiers mirror to the generic `eval:*` label (as the sibling
bots do); cron hourly at :15, between Muse's :00 and Qwen3.8's :30. Auto-merge is
SPARKINFER_BONSAI_AUTOMERGE=1 in .env.eval, turned on the same day once #1139 had validated the bot,
matching the sibling bots' live policy -- with one extra guard they lack: it merges only the exact
head commit this bot scored (auto_merge_ok_bonsai). Auto-close is OFF
(SPARKINFER_BONSAI_AUTOCLOSE=1 enables it, for REJECT only). This bot
evaluates every PR that does not declare a different model, most of which are not aimed at this
model and legitimately score `none` here, so `none` never closes anything -- #768 and #1082 were
both closed by a sibling bot for exactly that.

  python eval/pr_bonsai_bot.py --only-prs 1139 --reeval --no-post

Never rents a GPU. Shares /tmp/sparkinfer_bot.lock with the sibling bots via its cron wrapper
(run_bonsai_pr_cron.sh).
"""
from __future__ import annotations

import argparse
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
AUTO_CLOSE = os.environ.get("SPARKINFER_BONSAI_AUTOCLOSE") == "1"
AUTOMERGE_BLOCK = {
    "copycat", "copycat-warn", "flagged:gaming", "penalty", "needs-benchmark",
    BONSAI_NEEDS_REBASE, arb.REEVALUATE_LABEL, arb.HOLD_LABEL, *arb.REGRESSION_LABELS,
}

SCORES_FILE = os.path.expanduser(
    os.environ.get("BONSAI_SCORES_FILE", "~/.sparkinfer_bonsai_scores.json")
)

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
        print(f">> bonsai scores save skipped: {e}")


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
            problems.append(f"prefill-path check @{p}: {pr_fail} of {pr_fail + len(pr_runs)} PR "
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
    r = arb.gh(["pr", "view", str(num), "-R", repo, "--json", "comments"])
    done = set()
    for c in json.loads(r.stdout or "{}").get("comments", []):
        body = c.get("body") or ""
        m = MARKER_RE.search(body)
        if not m or "sparkinfer bonsai auto-eval" not in body:
            continue
        try:
            meta = json.loads(m.group(2)) if m.group(2) else {}
        except json.JSONDecodeError:
            meta = {}
        if meta.get("label") is None:
            continue
        done.add(m.group(1))
    return done


def strip_bonsai_eval_labels(repo, num):
    for lab in list(arb.labels_on(repo, num)):
        if lab.startswith(EVAL_PREFIX):
            arb.remove_label(repo, num, lab)


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
        capture_output=True, text=True, timeout=timeout,
        input=cmd if via_stdin else stdin_data,
    )


_EXPLICIT_FAIL_MARKERS = ("BUILD_FAILED", "SCORE_FAILED", "TOKENIZE_FAILED", "HARNESS_PIN_FAILED",
                          "MODEL_CHECK_FAILED", "MERGE_CONFLICT")
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
        for follow in lines[i + 1:i + 60]:
            if "error:" in follow or "Error " in follow:
                return f"{marker}: {follow.strip()}"
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
        checkout = arb.merged_checkout_script(ref, base)
    else:
        base = "origin/main"
        checkout = (f'git fetch -q origin {shlex.quote(ref)} || {{ echo "RETRYABLE_INFRA_FAILURE git fetch {ref} failed" >&2; exit 1; }}\n'
                    'git reset -q --hard\n'
                    'git clean -qfd\n'
                    'git checkout -qf FETCH_HEAD\n'
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
git fetch -q origin main || {{ echo "RETRYABLE_INFRA_FAILURE git fetch main failed" >&2; exit 1; }}
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
cmake --build build --target qwen3_gguf_bench qwen3_gguf_cb_bench qwen3_gguf_score \\
  qwen3_gguf_generate qwen3_gguf_prefill_check bonsai_inspect sparkinfer_server \\
  -j"$(nproc)" >"$TMPDIR/build.log" 2>&1 || {{
  echo "BUILD_FAILED — tail of the build log:" >&2
  tail -80 "$TMPDIR/build.log" >&2
  exit 1
}}
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
  echo "BONSAI_FAILED"
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
  local ckpt=$1 cc=$2 out="$TMPDIR/cb.txt" attempt=0 valid=0 a
  CB_AGGS=""; CB_AGG=0
  while [ "$valid" -lt {CB_REPS} ]; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt {CB_MAX_ATTEMPTS} ]; then
      echo "concurrent decode at c=$cc stopped requests part-way on $((attempt - 1 - valid)) of {CB_MAX_ATTEMPTS} runs" >&2
      return 1
    fi
    wait_gpu_clear || return 1
    if ! timeout 900 build/runtime/qwen3_gguf_cb_bench "$ckpt" "$cc" {CB_TOKENS} {CB_TOKENS} 512 > "$out" 2>&1; then
      echo "concurrent-decode harness exited nonzero at c=$cc" >&2
      tail -10 "$out" >&2 || true
      return 1
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
# A width that fails is DROPPED from scoring (BONSAICB_FAILED), never read as a regression to zero:
# a harness that did not run is not a slowdown.
for CC in {cb_concs}; do
  if cb_median "$GGUF" "$CC"; then
    echo "BONSAICB $CC $CB_AGG$CB_AGGS"
  else
    echo "BONSAICB_FAILED $CC"
  fi
done
echo "STAGE concurrency $(date +%s)"
wait_gpu_clear

# --- 2a. differential teacher-forced score ---
IDS=$(tok_ids {shlex.quote(ACC_TEXT)}) || {{ echo "TOKENIZE_FAILED (score corpus)" >&2; exit 1; }}
echo "RESULT_TOKEN_COUNT $(printf '%s' "$IDS" | wc -w)"
build/runtime/qwen3_gguf_score "$GGUF" {ACC_TOPK} $IDS > "$DUMP_SELF" 2>"$TMPDIR/score.err" || {{
  echo "SCORE_FAILED -- tail of the score log:" >&2
  tail -40 "$TMPDIR/score.err" >&2
  exit 1
}}
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
    wait_gpu_clear
    if timeout 900 build/runtime/qwen3_gguf_prefill_check "$GGUF" "$P" {PF_CONT} $IDS_P > "$PF_OUT" 2>&1; then
      T=$(sed -n 's/^TOP1 *[0-9]*\\/[0-9]* *\\([0-9.]*\\).*/\\1/p' "$PF_OUT" | tail -1)
      K=$(sed -n 's/^KL *\\([0-9.eE+-]*\\).*/\\1/p' "$PF_OUT" | tail -1)
      if [ -n "$T" ] && [ -n "$K" ]; then
        echo "PFCHECK $P $R $T $K"
        continue
      fi
    fi
    echo "PFCHECK_FAILED $P $R"
    tail -8 "$PF_OUT" >&2 || true
  done
done

echo "STAGE prefill_check $(date +%s)"

# --- 2c. bonsai_regression.py: tensors, score, generate, serve ---
wait_gpu_clear
REG_OUT="$TMPDIR/bonsai_regression.txt"
if timeout 1800 python3 eval/bonsai_regression.py --model "$GGUF" --reference "$REF_DIR" \\
     --tokenizer "$TOK_DIR" --build "$REPO/build/runtime" > "$REG_OUT" 2>&1; then
  echo "BONSAIREG_OK"
else
  echo "BONSAIREG_FAILED"
  if grep -q '^FAILED:' "$REG_OUT"; then
    sed -n '/^FAILED:/,$p' "$REG_OUT" | grep -E '^ +- ' | sed 's/^ *- /BONSAIREG_WHY /' | head -8 || true
  else
    echo "BONSAIREG_WHY $(grep -v '^\\s*$' "$REG_OUT" | tail -1)"
  fi
  tail -30 "$REG_OUT" >&2
fi
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
  wait_gpu_clear || {{ echo "${{tag}}_FAILED"; return 0; }}
  if bench_sweep_run "$ckpt" 128 {guard_args}; then
    for ctx in {guard_list}; do
      echo "$tag $ctx $(_bench_sweep_get $ctx decode_tps) $(_bench_sweep_get $ctx prefill_pp)"
    done
  else
    echo "${{tag}}_FAILED"
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
    out = {"bonsai": {}, "bonsai_cb": {}, "cb_runs": {}, "cb_failed": [], "pfcheck": {},
           "pfcheck_failed": {}, "bonsaireg_why": []}
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
            elif head == "BONSAICB" and len(parts) >= 3:
                out["bonsai_cb"][int(parts[1])] = float(parts[2])
                out["cb_runs"][int(parts[1])] = [float(x) for x in parts[3:]]
            elif head == "BONSAICB_FAILED" and len(parts) >= 2:
                out["cb_failed"].append(int(parts[1]))
            elif head == "RESULT_TOKEN_COUNT" and len(parts) >= 2:
                out["token_count"] = int(parts[1])
            elif head == "SCORE_DONE":
                out["score_done"] = True
                out["score_positions"] = int(parts[1]) if len(parts) >= 2 else 0
            elif head == "SELFCHECK":
                for tok in parts[1:]:
                    k, _, v = tok.partition("=")
                    if k in ("top1", "kl"):
                        out[f"selfcheck_{k}"] = float(v)
            elif head == "ACCURACY_NO_BASELINE":
                out["accuracy_no_baseline"] = True
            elif head == "METRIC":
                for tok in parts[1:]:
                    k, _, v = tok.partition("=")
                    if k in ("top1", "kl", "ppl_pr", "ppl_main"):
                        out[k] = float(v)
            elif head == "PFCHECK" and len(parts) >= 5:
                out["pfcheck"].setdefault(int(parts[1]), []).append((float(parts[3]), float(parts[4])))
            elif head == "PFCHECK_FAILED" and len(parts) >= 2:
                p = int(parts[1])
                out["pfcheck_failed"][p] = out["pfcheck_failed"].get(p, 0) + 1
            elif head == "BONSAIREG_OK":
                out["bonsaireg_ok"] = True
            elif head == "BONSAIREG_FAILED":
                out["bonsaireg_ok"] = False
            elif head == "BONSAIREG_WHY":
                out["bonsaireg_why"].append(line.split(" ", 1)[1].strip() if " " in line else "")
            elif head in tags and len(parts) >= 4:
                out[tags[head]][int(parts[1])] = {"decode": float(parts[2]), "prefill": float(parts[3])}
            elif head.endswith("_FAILED") and head[:-len("_FAILED")] in tags:
                out[tags[head[:-len("_FAILED")]] + "_failed"] = True
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
    r0 = ssh_run(host, port, f"cd {shlex.quote(REMOTE_REPO)} && git fetch -q origin "
                             f"{shlex.quote(pr_ref)} && git checkout -qf FETCH_HEAD", timeout=120)
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
    tail = ((r.stdout or "") + "\n" + (r.stderr or ""))[-2000:]
    crash = _crash_reason(r.stdout, r.stderr)
    infra = _is_infra_failure(r.stdout, r.stderr)
    reason = f"{what} failed" + (f" — {crash}" if crash else " (no crash diagnostic captured — retried once)")
    return {"ok": False, "retry": infra, "reason": reason, "log": tail}


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
    if not main.get("score_done") or main.get("score_positions", 0) < 100:
        return {"ok": False, "reason": f"main score dump missing or short "
                                       f"({main.get('score_positions', 0)} positions)", "log": log}
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
    main["ok"] = True
    return main


def eval_bonsai_on_box(host, port, pr_ref: str, main: dict):
    """Run the PR ref's script and score it against `main`, the round's shared baseline."""
    print(f">> Ternary-Bonsai-2-27B eval on box: PR ref={pr_ref}")
    r = _ssh_run_resilient(host, port, _remote_script(pr_ref, role="pr", onto=main.get("sha")), "PR run")
    if r.returncode != 0:
        combined = (r.stdout or "") + "\n" + (r.stderr or "")
        if "MERGE_CONFLICT" in combined:
            # Does not merge onto the main this round measured: a rebase, not a verdict.
            line = next((l for l in combined.splitlines() if l.startswith("MERGE_CONFLICT")), "")
            return {"ok": False, "retry": True, "conflict": True, "log": "",
                    "reason": line or "PR does not merge cleanly onto the measured main"}
        return _run_failure(r, "PR speed/accuracy run")
    pr = _parse_remote(r.stdout or "")
    log = (r.stdout or "")[-1500:]
    if pr.get("accuracy_no_baseline"):
        return {"ok": False, "retry": True, "log": log,
                "reason": "main's score dump was gone when the PR run compared against it"}
    if "top1" not in pr or "kl" not in pr:
        # main's dump passed its self-comparison this round, so an unreadable pair is the PR's dump.
        return {"ok": False, "retry": False, "log": log,
                "reason": "accuracy_compare_pair.py could not read the PR's score dump"}
    pr_top1, pr_kl = pr["top1"], pr["kl"]
    ppl_ratio = (pr.get("ppl_pr") or 0) / pr["ppl_main"] if pr.get("ppl_main") else None
    accuracy_ok = (pr_top1 >= ACC_TOP1_BAR and pr_kl <= ACC_KL_BAR
                   and ppl_ratio is not None and ppl_ratio <= ACC_PPL_RATIO)
    if pr.get("bonsai_failed") or not pr.get("bonsai"):
        # Lead with accuracy when it failed: a pass that emits garbage also drops measurements,
        # and the author should be sent to their numerics, not to the harness (#1037).
        why = "PR bench produced no Ternary-Bonsai-2-27B measurements"
        if not accuracy_ok:
            why = (f"PR output diverges from main — top-1 {pr_top1:.3f} (bar >={ACC_TOP1_BAR}), "
                   f"KL {pr_kl:.4f} (bar <={ACC_KL_BAR}), PPL x{ppl_ratio or 0:.3f} of main "
                   f"(bar <={ACC_PPL_RATIO}); the failed speed sweep is a symptom")
        return {"ok": False, "retry": False, "reason": why, "log": log}

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
    cb_skipped = []
    for conc in CB_CONCS:
        name = CB_DIM_FOR[conc]
        pr_v, main_v = _cb(pr, conc), _cb(main, conc)
        if pr_v is None or main_v is None:
            cb_skipped.append(name)
            continue
        lab, dlt, ok, why = tier_from_gain(pr_v, main_v, metric=name)
        scored.append({"dim": name, "label": lab, "delta": dlt, "passed": ok, "reason": why})
    if cb_skipped:
        print(f">> concurrent-decode axes not scored (no paired measurement): {', '.join(cb_skipped)}")
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

    if not accuracy_ok:
        reject(f"accuracy gate failed vs main: top1={pr_top1:.4f} (bar >={ACC_TOP1_BAR}) "
               f"kl={pr_kl:.5f} (bar <={ACC_KL_BAR}) ppl x{ppl_ratio or 0:.4f} (bar <={ACC_PPL_RATIO})")

    pf_ok, pf_problems, pf_rows = check_prefill_path(pr, main)
    if not pf_ok:
        if all(p.endswith("measurement unavailable") for p in pf_problems):
            return {"ok": False, "retry": True, "log": log, "reason": "; ".join(pf_problems)}
        reject("prefill-path accuracy gate failed: " + "; ".join(pf_problems[:4]))

    # Gate 2c is absolute, so it may only reject a PR when main passes it in the same round.
    reg_main_ok = bool(main.get("bonsaireg_ok"))
    reg_pr_ok = pr.get("bonsaireg_ok")
    reg_gated = reg_main_ok
    if reg_pr_ok is None:
        return {"ok": False, "retry": True, "log": log, "reason": "PR bonsai_regression.py did not run"}
    if reg_gated and not reg_pr_ok:
        reject("bonsai_regression.py failed (main passes it): "
               + "; ".join(pr.get("bonsaireg_why")[:3] or ["see log"]))

    guard_results = {}
    for key, _tag, name in GUARDS:
        skipped = bool(pr.get(f"{key}_unavailable") or main.get(f"{key}_unavailable"))
        ok, problems = (True, []) if skipped else _check_model_guard(pr, main, key, name)
        if skipped:
            print(f">> {name} guard SKIPPED — checkpoint not installed on the box")
        if not ok:
            # A guard that measured NOTHING is infra, not a regression (pr_qwen38_bot.py, #1112/#1114).
            unavailable = [p for p in problems if p.endswith("measurement unavailable")]
            if unavailable and len(unavailable) == len(problems):
                return {"ok": False, "retry": True, "log": log,
                        "reason": "; ".join(unavailable) + " — infra, not a regression; "
                                  "re-evaluated next round rather than rejected"}
            reject(f"{name} no-regression guard failed: " + "; ".join(problems[:6]))
        guard_results[key] = {"ok": ok, "problems": problems, "skipped": skipped}

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
        "guards": guard_results,
        "guards_pr": {key: pr.get(key) for key, _t, _n in GUARDS},
        "guards_main": {key: main.get(key) for key, _t, _n in GUARDS},
        "pr_head": pr.get("head"),
        "main_head": main.get("head"),
        "pr_tip": pr.get("pr_tip"),
        "merged_onto": pr.get("merged_onto"),
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
    missing = [f"c{c}" for c in CB_CONCS if CB_DIM_FOR[c] not in dims]
    if missing:
        out += (f"<sub>Concurrency {', '.join(missing)} not scored this round — no paired "
                f"measurement (a width that fails to run is dropped, never counted as a "
                f"regression).</sub>\n\n")
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
    close = "A `REJECT` closes the PR." if AUTO_CLOSE else "It does not close PRs."
    return f"{merge} {close}"


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
            f"<sub>Recorded for this commit; a new push is evaluated again.</sub>\n"
        )
    acc = (f"top-1 {res.get('pr_top1', 0):.4f} (bar ≥{ACC_TOP1_BAR}) · KL {res.get('pr_kl', 0):.5f} "
           f"(bar ≤{ACC_KL_BAR}) · PPL ×{res.get('ppl_ratio') or 0:.4f} of main (bar ≤{ACC_PPL_RATIO})")
    acc_row = _gate_row("accuracy vs main (teacher-forced)", res.get("accuracy_ok"),
                        f"{acc} over {res.get('token_count') or '?'} tokens", acc)
    pf_row = _gate_row(
        "prefill path vs main", res.get("prefill_path_ok"),
        f"batched prefill within main's spread at prefix {'/'.join(str(p) for p in PF_PREFIXES)}",
        "; ".join((res.get("prefill_path_problems") or [])[:3]))
    if res.get("bonsaireg_main_ok"):
        reg_row = _gate_row("bonsai_regression.py", res.get("bonsaireg_pr_ok"),
                            "tensors · score · generate · serve",
                            "; ".join((res.get("bonsaireg_why") or [])[:3]) or "see log")
    else:
        state = "passes" if res.get("bonsaireg_pr_ok") else "fails"
        reg_row = (f"| bonsai_regression.py | ⚠️ not gated — main fails it this round (the PR {state} "
                   f"it); a failure already on main cannot reject a PR |\n")
    guard_rows = ""
    for key, _tag, name in GUARDS:
        g = (res.get("guards") or {}).get(key) or {}
        guard_rows += _gate_row(
            f"{name} guard", g.get("ok"), "no regression (decode + prefill @ 32k)",
            "; ".join((g.get("problems") or [])[:4]),
            skipped="checkpoint not installed on the box; not checked" if g.get("skipped") else "")
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


def auto_merge_ok_bonsai(repo, num):
    info = json.loads(arb.gh(["pr", "view", str(num), "-R", repo, "--json",
                              "state,isDraft,labels,author,mergeable,files,headRefOid"]).stdout or "{}")
    if info.get("state") != "OPEN" or info.get("isDraft"):
        return False, "not an open, non-draft PR"
    labs = {l["name"] for l in info.get("labels", [])}
    tiers = {l.split(":", 1)[1] for l in labs if l.startswith(EVAL_PREFIX)}
    if not (tiers & SPEEDUP_LABELS):
        return False, "no verified eval-bonsai speedup label"
    if BONSAI_MERGE_FIRST not in labs:
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
    if any(l.endswith(":REJECT") for l in labs if l.startswith("eval")):
        return False, "carries a REJECT from another eval bot"
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
    if info.get("mergeable") != "MERGEABLE":
        return False, f"not cleanly mergeable ({info.get('mergeable')})"
    return True, "ok"


def try_auto_merge_bonsai(repo, num):
    ok, reason = auto_merge_ok_bonsai(repo, num)
    if not ok:
        print(f">> bonsai auto-merge SKIP #{num}: {reason}")
        return False
    # Pin the merge to the commit auto_merge_ok_bonsai just checked, so a push landing in the gap
    # between the check and the merge cannot be what gets merged (--match-head-commit).
    head = (json.loads(arb.gh(["pr", "view", str(num), "-R", repo, "--json", "headRefOid"]).stdout
                       or "{}").get("headRefOid") or "")
    args = ["pr", "merge", str(num), "-R", repo, "--squash"]
    if head:
        args += ["--match-head-commit", head]
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


def reconcile_bonsai_merge_labels(repo, dry_run=False):
    scores = _load_scores()
    open_prs = json.loads(arb.gh(["pr", "list", "-R", repo, "--state", "open",
                                  "--json", "number,labels", "--limit", "80"]).stdout or "[]")
    merged = json.loads(arb.gh(["pr", "list", "-R", repo, "--state", "merged", "--label",
                                BONSAI_MERGE_FIRST, "--json", "number", "--limit", "10"]).stdout or "[]")
    if not dry_run:
        for m in merged:
            arb.remove_label(repo, m["number"], BONSAI_MERGE_FIRST)
    scored = []
    stale_first = []   # carries merge-first but can no longer win it
    for p in open_prs:
        labs = {l["name"] for l in p["labels"]}
        # A PR that cannot be merged -- `hold`, needs-rebase, a penalty or copycat flag, any other
        # AUTOMERGE_BLOCK label -- must not take merge-first either. It used to: on 2026-09-24
        # #1154 was held for review with the round's best score, won merge-first, had its merge
        # refused, and pushed the next-best PR to needs-rebase for a merge that never happened --
        # every round, for as long as the hold lasted.
        if labs & AUTOMERGE_BLOCK:
            if BONSAI_MERGE_FIRST in labs:
                stale_first.append(p["number"])
            continue
        tier = next((l.split(":", 1)[1] for l in labs
                     if l.startswith(EVAL_PREFIX) and l.split(":", 1)[1] in SPEEDUP_LABELS), None)
        if not tier:
            continue
        entry = scores.get(str(p["number"])) or {}
        scored.append((p["number"], float(entry.get("delta_pct") or 0)))
    scored.sort(key=lambda x: x[1], reverse=True)
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
        rid = f"bonsai-{int(num):04d}-{oid[:7]}"
        arb._ensure_log_repo()
        rundir = os.path.join(arb.LOG_DIR, "runs", rid)
        os.makedirs(rundir, exist_ok=True)
        polaris = res.get("polaris") or {}
        receipt = polaris.get("receipt")
        result = {
            "id": rid, "pr": int(num), "title": title,
            "url": f"https://github.com/{repo}/pull/{num}", "commit": oid[:7],
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
        if subprocess.run(["git", "-C", arb.LOG_DIR, "push", "-q"], check=False).returncode != 0:
            print(">> bonsai eval-log push failed")
            return None
        print(f">> bonsai eval log: {arb.LOG_PAGE + rid}")
        return arb.LOG_PAGE + rid
    except Exception as e:
        print(f">> bonsai eval-log upload failed: {e}")
        return None


def apply_result(repo, num, commit, res, title="", dry_run=False):
    if not res.get("ok") and res.get("conflict"):
        print(f"PR #{num}: {res.get('reason')} — bonsai-needs-rebase, no verdict")
        if not dry_run:
            arb.add_label(repo, num, BONSAI_NEEDS_REBASE)
        return
    if not res.get("ok") and res.get("retry"):
        # Infrastructure: nothing is posted and no label changes. The next round measures again.
        print(f"PR #{num}: bonsai eval deferred — {res.get('reason')} (infra; re-evaluated next round)")
        return
    label = res.get("label") if res.get("ok") else "REJECT"
    body = format_comment(commit, res)
    print(f"PR #{num}: eval-bonsai:{label}  from={res.get('best_dim')} delta={res.get('delta_pct')}%  "
          f"accuracy_ok={res.get('accuracy_ok')} prefill_path_ok={res.get('prefill_path_ok')} "
          f"bonsaireg_ok={res.get('bonsaireg_pr_ok')} "
          f"guards_ok={ {k: v.get('ok') for k, v in (res.get('guards') or {}).items()} }")
    if dry_run:
        print(body)
        return
    strip_bonsai_eval_labels(repo, num)
    if label in SPEEDUP_LABELS:
        # A fresh speedup for the current head makes this PR eligible for merge-first again
        # (the sibling bots' #790/#791 fix).
        arb.remove_label(repo, num, BONSAI_NEEDS_REBASE)
    arb.add_label(repo, num, f"{EVAL_PREFIX}{label}")
    # Mirrored to the generic eval:* label (explicit decision 2026-09-24), derived from every
    # per-bot label so a `none` here cannot erase another model's real tier.
    arb.sync_generic_eval_label(repo, num)
    arb.gh(["pr", "comment", str(num), "-R", repo, "--body", body])
    if not res.get("ok"):
        return
    upload_bonsai_eval_log(repo, num, title, commit, res)
    if res.get("delta_pct") is not None:
        scores = _load_scores()
        scores[str(num)] = {
            "commit": commit, "label": label, "delta_pct": res.get("delta_pct"),
            "best_dim": res.get("best_dim"), "pass": res.get("pass"),
            "accuracy_ok": res.get("accuracy_ok"), "prefill_path_ok": res.get("prefill_path_ok"),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _save_scores(scores)
    # `none` never closes here: most PRs this bot evaluates are aimed at another model (module
    # docstring). A measured REJECT closes only when the operator has turned auto-close on.
    if AUTO_CLOSE and label == "REJECT":
        arb.gh(["pr", "comment", str(num), "-R", repo, "--body",
                "<!-- sparkinfer-bonsai-auto-close -->\n"
                "## Closed: regression or failed gate — `eval-bonsai:REJECT`\n\n"
                "The verdict comment above names the axis or gate that failed on "
                "Ternary-Bonsai-2-27B. Every scored axis is a no-regression floor, and the accuracy "
                "gates and cross-model guards are hard gates. Reopen once it is addressed and it "
                "re-evaluates on the next poll."])
        arb.gh(["pr", "close", str(num), "-R", repo])
        print(f">> auto-closed PR #{num} (eval-bonsai:REJECT)")


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
                    help="comma-separated PR numbers (bypasses greenlight, and the draft filter)")
    args = ap.parse_args()
    only = {int(x) for x in args.only_prs.split(",") if x.strip().isdigit()}
    quiet = args.dry_run or args.no_post

    print(f">> bonsai eval transport: "
          f"{'ssh' if ssh_box_enabled() else f'vast.ai (instance {arb.current_instance(args.instance) or args.instance})'}")
    print(f">> AUTOMERGE={int(AUTO_MERGE)} AUTOCLOSE={int(AUTO_CLOSE)} NO_POST={int(args.no_post)}")

    if args.labels_only:
        reconcile_bonsai_merge_labels(args.repo, dry_run=args.dry_run)
        print("done — bonsai labels only (no GPU).")
        return

    prs = json.loads(arb.gh([
        "pr", "list", "-R", args.repo, "--state", "open",
        "--json", "number,title,labels,isDraft,headRefOid,headRefName,mergeable,author,body,files",
        "--limit", "80",
    ]).stdout or "[]")
    prs.sort(key=lambda p: p["number"])

    denylist = arb.load_denylist()
    pending = []
    for pr in prs:
        num = pr["number"]
        if only and num not in only:
            continue
        if pr.get("isDraft") and num not in only:
            continue
        hits = arb.pr_involved_logins(args.repo, num) & denylist
        if hits:
            print(f"PR #{num}: BLOCKED (denylisted: {', '.join(sorted(hits))}) — flag + close, no eval")
            if not quiet:
                arb.close_blocked_pr(args.repo, num, hits)
            continue
        labs = {l["name"] for l in pr.get("labels", [])}
        if arb.HOLD_LABEL in labs and num not in only:
            print(f"PR #{num}: hold — skip")
            continue
        head = (pr.get("headRefOid") or "")[:40]
        short = head[:9]
        if not args.reeval and head and head in bonsai_evaluated_commits(args.repo, num):
            print(f"PR #{num} @ {short}: already bonsai-evaluated — skip")
            continue
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
        pending.append((num, head, short, ref, pr.get("title", "")))

    if not pending:
        reconcile_bonsai_merge_labels(args.repo, dry_run=quiet)
        print("done — no bonsai PRs to evaluate.")
        return
    if args.dry_run:
        print("--- dry-run would evaluate: " + ", ".join(f"#{n} ({ref})" for n, _h, _s, ref, _t in pending))
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
        return
    print(f">> SSH {ssh_box_user() if ssh_box_enabled() else 'root'}@{host}:{port}")

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
        return
    print(">> main baseline: " + "  ".join(
        f"@{SCORED_CTX_LABEL[c]} {_at(main_result, c, 'decode'):.1f}/{_at(main_result, c, 'prefill'):.1f}"
        for c in SCORED_CTXS) + f"  cb {main_result.get('bonsai_cb')}")
    print(f">> main prefill-path: {main_result.get('pfcheck')}  bonsai_regression "
          f"{'OK' if main_result.get('bonsaireg_ok') else 'FAILED ' + str(main_result.get('bonsaireg_why'))}")
    print(f">> main guard coverage: {_guard_coverage(main_result)}")

    for num, head, short, ref, title in pending:
        print(f"PR #{num} @ {short}: evaluating Ternary-Bonsai-2-27B '{ref}' …")
        try:
            res = eval_bonsai_on_box(host, port, ref, main_result)
        except Exception as e:
            res = {"ok": False, "retry": True, "reason": f"exception: {type(e).__name__}: {e}"}
        apply_result(args.repo, num, head or short, res, title=title, dry_run=args.no_post)

    reconcile_bonsai_merge_labels(args.repo, dry_run=quiet)
    print("done — bonsai eval pass complete.")


if __name__ == "__main__":
    main()
