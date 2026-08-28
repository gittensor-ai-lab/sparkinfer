#!/usr/bin/env python3
"""TPE auto-tuner for DSpark speculative decode.

Defaults to ctx=128 (DSPARK_CTX). This tuner uses short independent train/holdout prompts and does
not reproduce the evaluator's fixed 32k corpus; set DSPARK_CTX only for exploratory sweeps where
the supplied prompt actually contains that many tokens.

Searches the DFlash/DSpark runtime knobs for the configuration that maximises DSpark decode
throughput, subject to a hard losslessness constraint. Modelled on p-e-w/heretic, which co-optimises
(refusal rate, KL divergence) with Optuna TPE rather than hand-tuning ablation parameters -- the
same shape as the problem here, with (throughput, exactness) in place of (refusals, KL).

WHY THIS EXISTS
    Every DSpark win merged so far was a human running a grid by hand. PR #867 was worth 1.36x and
    the whole change was a proposal depth of 3 instead of 5 -- and the 5 it replaced was itself
    carefully hand-tuned, with a comment citing accept lengths of 5.33 / 5.95 / 6.43. Those are
    Qwen3.6 DFlash numbers. Qwen3.8's tau was 1.25 when this was written and is ~1.66 now, so
    re-read any constant here before trusting it -- that is the entire point. The constant was
    optimal for a different model and stayed wrong by 1.36x until somebody re-ran the grid. There
    are ~280 SPARKINFER_* knobs read via getenv in the hot paths; this searches twelve of them.

    Crucially they are read at RUNTIME, so a trial is a process launch, not a rebuild. That is what
    makes this cheap: ~30s per trial, almost all of it the model load.

THE CONSTRAINT IS THE WHOLE DESIGN
    A naive throughput search over these knobs will find configurations that are fast because they
    are WRONG -- speculative decoding can always be made faster by verifying less. So every trial
    runs dspark_tau_check, which regenerates the same prompt with the draft disabled and requires
    exact token equality, and a trial that is not lossless scores zero no matter how fast it was.
    SPARKINFER_DFLASH_OVERLAP is deliberately left in the search space as a live test of that: it
    is a real speedup and it is known to break losslessness, so a healthy run should try it, score
    it zero, and learn to avoid it. Check that in the summary before trusting any result.

OVERFITTING
    Heretic computes directions on train[:400] and scores on a held-out test[:100]. Same discipline
    here: the search optimises against one prompt, and the winner is re-measured on a DIFFERENT
    prompt it never saw, against the default config on that same held-out prompt. A configuration
    that only wins on the tuning prompt is an artifact, and the summary will show it.

SHARING THE BOX
    The lock is taken PER TRIAL, not for the whole study, so the hourly eval cron can interleave
    between trials instead of being starved for the length of the run. A trial is ~30s, well inside
    the cron wrapper's 120s flock wait.

    python3 eval/tune_dspark.py --trials 120
    python3 eval/tune_dspark.py --trials 60 --resume      # continue an existing study
    python3 eval/tune_dspark.py --baseline-only           # just measure the current default
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

MODEL_DIR = os.environ.get("MODELOPT_MODEL_DIR", "/root/workspace/models_q38_modelopt")
DRAFT_DIR = os.environ.get("DSPARK_DRAFT_DIR", "/root/workspace/dspark")
REMOTE_REPO = os.environ.get("MODELOPT_REMOTE_REPO", "/root/sparkinfer_modelopt")
NTOK = int(os.environ.get("DSPARK_TUNE_NTOK", "128"))
CTX = int(os.environ.get("DSPARK_CTX", "128"))
LOCK = "/tmp/sparkinfer_bot.lock"

# Two prompts: one to optimise against, one held out to validate the winner on. Different text, not
# a different slice of the same text -- the point is to catch a config that keys on this particular
# token stream, and adjacent slices of one document are not independent enough to show that.
TRAIN_PROMPT = (
    "Artificial intelligence is a field of computer science that builds systems able to perform "
    "tasks which normally require human intelligence, such as understanding language, recognising "
    "images, and making decisions under uncertainty. Researchers have pursued this goal since the "
    "middle of the twentieth century, and progress has come in waves rather than steadily. Early "
    "efforts focused on symbolic reasoning and hand-written rules, which worked well in narrow, "
    "well-defined domains but struggled to generalise to the messiness of everyday perception."
)
HOLDOUT_PROMPT = (
    "The harbour at dawn was the colour of weak tea, and the fishing boats came in one after "
    "another with their engines throttled down to a mutter. Maria counted them from the seawall as "
    "she had every morning for thirty years, and when the count came up one short she did not move "
    "for a long time. The gulls went on shrieking over the gutting tables. Somebody's radio was "
    "playing a song about a woman who waits, which struck her as a poor joke, and she turned up "
    "her collar against a wind that had come a long way over cold water to reach her."
)


def sh(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def load_env():
    """Source .env.eval the same way the cron wrappers do, so the box coordinates come from one
    place and cannot drift out of sync with the eval bot's."""
    path = os.path.join(ROOT, ".env.eval")
    if not os.path.exists(path):
        sys.exit(f"missing {path}")
    out = sh(["bash", "-c", f"set -a && . {shlex.quote(path)} && set +a && env"]).stdout
    for line in out.splitlines():
        k, _, v = line.partition("=")
        if k.startswith(("EVAL_SSH_", "SSH_KEY", "MODELOPT_", "DSPARK_")):
            os.environ.setdefault(k, v)
    for k in ("EVAL_SSH_HOST", "EVAL_SSH_PORT", "SSH_KEY"):
        if not os.environ.get(k):
            sys.exit(f"{k} not set in .env.eval")


def ssh_run(cmd: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return sh([
        "ssh", "-i", os.environ["SSH_KEY"], "-p", os.environ["EVAL_SSH_PORT"],
        "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20",
        f"{os.environ.get('EVAL_SSH_USER', 'root')}@{os.environ['EVAL_SSH_HOST']}", cmd,
    ], timeout=timeout)


def _locked(cmd: str, timeout: int) -> subprocess.CompletedProcess:
    """Run one ssh command holding the shared bot lock, and only for that command's duration.

    Taking the lock for the whole study would starve the hourly eval cron for an hour or more;
    taking it per trial means a cron tick waits at most one trial (~30s) against its 120s budget.
    Exit 75 means the lock could not be had in time -- the caller prunes that trial rather than
    recording a zero, since it measured nothing."""
    script = (
        f"exec 9>{LOCK}\n"
        f"flock -w 600 9 || {{ echo 'LOCK_TIMEOUT' >&2; exit 75; }}\n"
        + " ".join([
            "ssh", "-i", shlex.quote(os.environ["SSH_KEY"]),
            "-p", shlex.quote(os.environ["EVAL_SSH_PORT"]),
            "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20",
            shlex.quote(f"{os.environ.get('EVAL_SSH_USER', 'root')}@{os.environ['EVAL_SSH_HOST']}"),
            shlex.quote(cmd),
        ])
    )
    return sh(["bash", "-c", script], timeout=timeout)


METRIC_RE = re.compile(r"^METRIC (\w+) ([-\d.eE+]+)$", re.M)


def parse_metrics(out: str) -> dict:
    return {m.group(1): float(m.group(2)) for m in METRIC_RE.finditer(out or "")}


def tokenize(prompt: str, n: int) -> list[int]:
    """Tokenize on the BOX, with the checkpoint's own tokenizer -- the same one the harness and the
    server use. Doing it locally would need the tokenizer file here and risk a different version."""
    # The prompt is embedded as a repr rather than piped on stdin: ssh_run() sends no stdin, and a
    # silently empty prompt tokenizes to zero ids and then fails much later as a missing metric.
    py = (
        "import json\n"
        "from tokenizers import Tokenizer\n"
        f"t=Tokenizer.from_file({MODEL_DIR!r}+'/tokenizer.json')\n"
        f"print(json.dumps(t.encode({prompt!r}).ids[:{n}]))\n"
    )
    r = ssh_run(f"cd {shlex.quote(REMOTE_REPO)} && python3 -c {shlex.quote(py)}", timeout=120)
    if r.returncode != 0:
        sys.exit(f"tokenize failed: {r.stderr[-500:]}")
    line = [l for l in (r.stdout or "").splitlines() if l.startswith("[")]
    if not line:
        sys.exit(f"tokenize produced no ids: {(r.stdout or '')[-300:]}")
    return json.loads(line[-1])


def build(ref: str):
    cmd = (
        f"cd {shlex.quote(REMOTE_REPO)} && git fetch -q origin && "
        f"git reset -q --hard && git clean -qfd && git checkout -qf {shlex.quote(ref)} && "
        f"echo HEAD $(git rev-parse --short HEAD) && "
        f"cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/tmp/tune_cmake.log 2>&1 && "
        f"cmake --build build --target dspark_tau_check -j$(nproc) >/tmp/tune_build.log 2>&1 && "
        f"echo BUILD_OK"
    )
    r = _locked(cmd, timeout=1800)
    if "BUILD_OK" not in (r.stdout or ""):
        sys.exit(f"build failed:\n{(r.stdout or '')[-800:]}\n{(r.stderr or '')[-800:]}")
    head = re.search(r"HEAD (\w+)", r.stdout)
    return head.group(1) if head else "?"


def measure(env: dict, ids: list[int], timeout: int = 900) -> dict:
    """One trial: run the harness under `env` and return its metrics.

    Note what is NOT set here. dspark_tau_check pins NSPLITS / PREFILL_I8 / PREFILL_SKINNY_SPLITK /
    QWEN38_PREFILL_NVFP4 / PREFILL_BATCHED itself with setenv(..., overwrite=0), i.e. an externally
    supplied value would WIN. Those pins are what make the AR reference and the speculative run
    start from the same state, so overriding one would silently invalidate the losslessness check
    that this whole search is constrained on. Only DFLASH_* knobs go in `env`."""
    prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in sorted(env.items()))
    cmd = (
        f"cd {shlex.quote(REMOTE_REPO)} && {prefix} timeout 600 "
        f"build/runtime/dspark_tau_check {shlex.quote(MODEL_DIR)} {shlex.quote(DRAFT_DIR)} "
        f"{NTOK} {' '.join(str(i) for i in ids)} 2>&1"
    )
    r = _locked(cmd, timeout=timeout)
    if r.returncode == 75:
        return {"_lock_timeout": 1.0}
    return parse_metrics(r.stdout)


def suggest(trial) -> dict:
    """The search space: twelve knobs that shape DSpark's short-context decode.

    SPARKINFER_DFLASH_OVERLAP is in here on purpose. It is a genuine ~4% speedup and it is known to
    break losslessness (5-7 of 40 repeats of the same generation diverged; that is why it defaults
    off). Leaving it in gives the run a built-in positive control: if the constraint is working,
    trials with overlap=1 should appear and should score zero."""
    env = {
        # Primary. #867 moved this 5 -> 3 for 1.36x; the search should rediscover that, which is
        # also a sanity check that the harness is measuring what we think it is.
        "SPARKINFER_DFLASH_PROPOSALS": trial.suggest_int("proposals", 1, 7),
        # 0 = derive from depth (round up over {4,8,16}); the explicit widths let the optimizer
        # decouple block width from depth, which the derived rule cannot express.
        "SPARKINFER_DFLASH_BLOCK_WIDTH": trial.suggest_categorical("block_width", [0, 4, 8, 16]),
        # 0 = never batch-verify, 1 = always, 2 = adaptive (default).
        "SPARKINFER_DFLASH_COMPACT_VERIFY": trial.suggest_categorical("compact_verify", [0, 1, 2]),
        "SPARKINFER_DFLASH_BLOCK_SCORE": trial.suggest_int("block_score", 1, 8),
        "SPARKINFER_DFLASH_COMPACT_MAX_SEQ": trial.suggest_int("compact_max_seq", 0, 2048, step=64),
        # The engage threshold in EIGHTHS. Was SPARKINFER_DFLASH_ENGAGE_KEEP (whole tokens, 1..8)
        # until #890 replaced that knob; the old var is no longer read by the runtime at all, so
        # every trial that "tuned" it after #890 was searching a dimension with no effect. Range
        # 8..64 is the same 1..8 tokens expressed in eighths, plus the sub-token band (8..16) that
        # is the whole point of the finer domain -- the shipped default is 10, i.e. 1.25 tokens,
        # which the old whole-token knob could not express.
        "SPARKINFER_DFLASH_ENGAGE_KEEP_EIGHTHS": trial.suggest_int("engage_keep_eighths", 8, 64),
        "SPARKINFER_DFLASH_SHARED_STREAM": trial.suggest_categorical("shared_stream", [0, 1]),
        "SPARKINFER_DFLASH_CTX_GEMM": trial.suggest_categorical("ctx_gemm", [0, 1]),
        "SPARKINFER_DFLASH_CTX_TRIM": trial.suggest_categorical("ctx_trim", [0, 1]),
        "SPARKINFER_DFLASH_HEAD_I4": trial.suggest_categorical("head_i4", [0, 1]),
        "SPARKINFER_DFLASH_OVERLAP": trial.suggest_categorical("overlap", [0, 1]),
    }
    # Conditional, because the code treats PRESENCE as the on-switch: getenv(...) != nullptr arms
    # the gate, so setting it to 0 turns it ON at threshold 0 rather than leaving it off. The
    # comment at that site says there is "no calibration data yet for this checkpoint's actual
    # accept-rate distribution" -- this search is exactly that calibration.
    if trial.suggest_categorical("confidence_gate_on", [False, True]):
        env["SPARKINFER_DFLASH_CONFIDENCE_GATE"] = round(
            trial.suggest_float("confidence_gate", -6.0, 6.0), 3)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--startup", type=int, default=30, help="random exploration trials before TPE")
    ap.add_argument("--ref", default="origin/main")
    ap.add_argument("--study", default=os.path.expanduser("~/.sparkinfer_dspark_tuner.db"))
    ap.add_argument("--name", default="dspark-decode128")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()

    load_env()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f">> box {os.environ['EVAL_SSH_HOST']}:{os.environ['EVAL_SSH_PORT']}  ref={args.ref}")
    head = build(args.ref)
    print(f">> built dspark_tau_check at {head}")

    train_ids = tokenize(TRAIN_PROMPT, CTX)
    hold_ids = tokenize(HOLDOUT_PROMPT, CTX)
    print(f">> prompts: train={len(train_ids)} tok, holdout={len(hold_ids)} tok")

    base_train = measure({}, train_ids)
    base_hold = measure({}, hold_ids)
    for label, m in (("train", base_train), ("holdout", base_hold)):
        if not m.get("DSPARK_TPS"):
            sys.exit(f"baseline on {label} produced no DSPARK_TPS -- aborting rather than "
                     f"optimising against a broken reference")
        print(f">> baseline[{label}] dspark={m['DSPARK_TPS']:.2f} ar={m.get('AR_TPS', 0):.2f} "
              f"tau={m.get('MEAN_ACCEPT', 0):.4f} lossless={int(m.get('LOSSLESS', 0))}")
    if args.baseline_only:
        return

    t0 = time.time()
    state = {"n": 0, "rejected": 0, "best": base_train["DSPARK_TPS"]}

    def objective(trial):
        env = suggest(trial)
        m = measure(env, train_ids)
        state["n"] += 1
        if m.get("_lock_timeout"):
            raise optuna.TrialPruned("could not acquire the box lock")
        tps = m.get("DSPARK_TPS", 0.0)
        lossless = int(m.get("LOSSLESS", 0)) == 1
        for k, v in m.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("env", env)
        # The constraint. A fast-but-wrong configuration is not a result, so it scores zero rather
        # than being merely penalised -- TPE then learns the region is dead instead of trading
        # exactness off against throughput at some exchange rate we never agreed to.
        if not lossless or tps <= 0:
            state["rejected"] += 1
            return 0.0
        if tps > state["best"]:
            state["best"] = tps
            print(f"   trial {state['n']:>3}: {tps:7.2f} tok/s  tau={m.get('MEAN_ACCEPT', 0):.4f}  "
                  f"NEW BEST (+{100 * (tps / base_train['DSPARK_TPS'] - 1):.1f}% vs default)")
        return tps

    study = optuna.create_study(
        study_name=args.name,
        storage=f"sqlite:///{args.study}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=args.startup, seed=1337),
        load_if_exists=args.resume,
    )
    study.enqueue_trial({  # start from the shipping default, so TPE has one known-good anchor
        "proposals": 3, "block_width": 0, "compact_verify": 2, "block_score": 1,
        "compact_max_seq": 384, "engage_keep_eighths": 10, "shared_stream": 1, "ctx_gemm": 1,
        "ctx_trim": 1, "head_i4": 1, "overlap": 0, "confidence_gate_on": False,
    }, skip_if_exists=True)
    study.optimize(objective, n_trials=args.trials)

    ok = [t for t in study.trials if t.value and t.value > 0]
    mins = (time.time() - t0) / 60
    print(f"\n>> {len(study.trials)} trials in {mins:.0f} min — {len(ok)} lossless, "
          f"{state['rejected']} rejected by the constraint")

    overlap_trials = [t for t in study.trials if t.params.get("overlap") == 1]
    overlap_passed = [t for t in overlap_trials if t.value and t.value > 0]
    print(f">> constraint control: {len(overlap_trials)} trials tried OVERLAP=1, "
          f"{len(overlap_passed)} of them passed losslessness"
          + ("  <-- expected 0; investigate" if overlap_passed else "  (as expected)"))

    if not ok:
        print(">> no lossless configuration beat zero — nothing to report")
        return

    best = study.best_trial
    env = best.user_attrs.get("env", {})
    print(f"\n>> BEST on train: {best.value:.2f} tok/s "
          f"(+{100 * (best.value / base_train['DSPARK_TPS'] - 1):.1f}% vs default {base_train['DSPARK_TPS']:.2f})")
    for k, v in sorted(env.items()):
        print(f"     {k}={v}")

    # Held-out validation. This is the number that decides whether the win is real: same config,
    # a prompt the search never optimised against, compared to the default on that same prompt.
    print("\n>> validating on the held-out prompt …")
    hv = measure(env, hold_ids)
    if not hv.get("DSPARK_TPS"):
        print(">> holdout run produced no metrics — treat the result as UNVALIDATED")
        return
    gain_t = 100 * (best.value / base_train["DSPARK_TPS"] - 1)
    gain_h = 100 * (hv["DSPARK_TPS"] / base_hold["DSPARK_TPS"] - 1)
    print(f"   holdout: {hv['DSPARK_TPS']:.2f} vs default {base_hold['DSPARK_TPS']:.2f} "
          f"= {gain_h:+.1f}%   (train said {gain_t:+.1f}%)")
    print(f"   holdout lossless={int(hv.get('LOSSLESS', 0))}  tau={hv.get('MEAN_ACCEPT', 0):.4f}  "
          f"ar={hv.get('AR_TPS', 0):.2f}")
    if int(hv.get("LOSSLESS", 0)) != 1:
        print("   REJECTED: not lossless on the held-out prompt")
    elif gain_h < 0.5 * gain_t:
        print("   WARNING: the held-out gain is less than half the tuning gain — likely overfit "
              "to the tuning prompt. Do not ship this without re-measuring more widely.")
    else:
        print("   VALIDATED: the gain holds on a prompt the search never saw.")
        print("\n>> to reproduce, run dspark_tau_check (or the server) with:")
        print("   " + " ".join(f"{k}={v}" for k, v in sorted(env.items())))


if __name__ == "__main__":
    main()
