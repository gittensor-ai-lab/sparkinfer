#!/usr/bin/env python3
"""Regression guard for Ternary-Bonsai-2 (PTQ1_0) support.

Three checks, because every bug found bringing this model up was invisible to at least one of
them:

  tensors    Decode a weight out of its ternary blocks, undo the stored Hadamard rotation, and
             compare against the un-quantized checkpoint the model was derived from. Ternary
             quantisation lands at cosine ~0.88; anything far below that means the format is
             being read wrong. This is what caught the trit order, the v-head regrouping and the
             fp16 subnormal flush -- none of which a round-trip test can see, because a
             round-trip packs with whatever order it unpacks.

  score      Teacher-forced perplexity on a fixed passage, folded and native. Catches accuracy
             regressions that still produce fluent text -- the GDN v-head convention was worth
             PPL 139 against 8, and read as "the model is just weak" until measured.

  generate   Greedy token ids, folded against native. These are two routes to the same
             arithmetic and must agree exactly. NOTE this runs generate as well as score on
             purpose: qwen3_gguf_score drives forward_token per position and never touches
             prefill's seed-argmax path, so two of this model's bugs were invisible to scoring.

  serve      The same prompt at temperature 0, asked alone and then again inside a batch that
             DECAYS to one row (three short requests beside it). Everything above this line runs
             one request at a time and is blind to the whole continuous-batch path -- which is
             where the worst bug this model turned up lived, and it was not even ternary-specific:
             decode_packed compacts a session's recurrent state to bf16, and a row that outlives
             its batch was served from that state read at the wrong width.

             The baseline is asked TWICE, sequentially, and the batched answer has to match one of
             them. Not pedantry: measured on the NVFP4 Qwen3.8-27B checkpoint, three sequential
             asks with no concurrency anywhere returned 354, 354 and 362 characters, so a single
             baseline would have failed this check nightly on a model that was working. Two
             baselines that disagree say the checkpoint is non-deterministic at temperature 0 and
             the comparison cannot discriminate on it: that trial is INCONCLUSIVE, reported as a
             NOTE and never counted as a failure.

             One trial is not a verdict. On the eval box (2026-09-25) a build equal to main failed
             this check in 2 runs of 8, so a single failure forced a REJECT on a sound PR about one
             time in five. Each path now re-runs a failed trial, on a fresh server, and fails only
             when 2 of up to 3 conclusive trials fail (SERVE_FAILS_TO_FAIL of
             SERVE_CONCLUSIVE_TRIALS); a first trial that passes is still a pass. Costs a server
             start per trial, so it is the slow check.

Every check runs, and every failure is listed under `FAILED:` with its check's name first
(`tensors:`, `score:`, `generate:`, `serve:`) -- the bot gates each check separately against main.
A check that raises is a failure of that check, not a crash of the whole script. Thresholds are
deliberately loose -- they are there to catch a broken path, not to police the third decimal place.
"""
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# A passage long enough that perplexity means something. 6 positions of noise once read as
# "516 against 304" and said nothing at all.
PASSAGE = (
    "Machine learning is a field of study in artificial intelligence concerned with the "
    "development and study of statistical algorithms that can learn from data and generalise "
    "to unseen data, and thus perform tasks without explicit instructions. Within a "
    "subdiscipline in machine learning, advances in the field of deep learning have allowed "
    "neural networks to surpass many previous approaches in performance."
)
PROMPT = "The capital of France is Paris. The capital of Japan is"

# Tensors worth checking, one per input width, so a sign vector that is mis-sliced shows up.
TENSORS = ["token_embd.weight", "blk.0.ffn_gate.weight", "blk.31.ffn_down.weight"]
HF_FOR = {
    "token_embd.weight": ("model.language_model.embed_tokens.weight", "plain"),
    "blk.0.ffn_gate.weight": ("model.language_model.layers.0.mlp.gate_proj", "nvfp4"),
    "blk.31.ffn_down.weight": ("model.language_model.layers.31.mlp.down_proj", "nvfp4"),
}


def run(cmd, env=None, timeout=1800):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, env=e, timeout=timeout).stdout


def tokenize(tokenizer_dir, text):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    return [str(i) for i in tok(text, add_special_tokens=False).input_ids]


def check_tensors(a, failures):
    """Decoded-and-un-rotated weights against the checkpoint this model was quantised from."""
    import numpy as np
    from safetensors import safe_open
    import torch

    f = safe_open(os.path.join(a.reference, "model.safetensors"), framework="pt")
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    rows = 256
    for name in TENSORS:
        hf, kind = HF_FOR[name]
        out = f"/tmp/bonsai_reg_{name.replace('.', '_')}.f32"
        txt = run([a.inspect, a.model, "--dump", name, "1", str(rows), out])
        m = re.search(r"wrote \d+ rows x (\d+)", txt)
        if not m:
            failures.append(f"tensors: could not dump {name}: {txt.strip()[:120]}")
            continue
        width = int(m.group(1))
        mine = np.fromfile(out, dtype=np.float32).reshape(rows, width)
        if kind == "plain":
            ref = f.get_slice(hf)[0:rows].float().numpy()
        else:
            p = f.get_slice(hf + ".weight_packed")[0:rows].numpy().view(np.uint8)
            sc = f.get_slice(hf + ".weight_scale")[0:rows].float().numpy()
            nib = np.empty((p.shape[0], p.shape[1] * 2), dtype=np.uint8)
            nib[:, 0::2], nib[:, 1::2] = p & 0xF, p >> 4
            val = lut[nib & 7] * np.where(nib & 8, -1, 1)
            ref = (val.reshape(rows, -1, 16) * sc[:, :, None]).reshape(rows, -1)
        cos = ((mine * ref).sum(1) /
               np.maximum(np.linalg.norm(mine, axis=1) * np.linalg.norm(ref, axis=1), 1e-30)).mean()
        print(f"  {name:28s} cosine {cos:+.4f} (want >= {a.min_cos})")
        if cos < a.min_cos:
            failures.append(f"tensors: {name} cosine {cos:.4f} below {a.min_cos}")
        os.unlink(out)


def check_score(a, ids, failures):
    """Perplexity on both paths."""
    ppl = {}
    for label, native in (("folded", ""), ("native", "all")):
        txt = run([a.score, a.model, "1"] + ids, env={"SPARKINFER_BONSAI_NATIVE": native})
        m = re.search(r"^PPL ([0-9.]+)", txt, re.M)
        if not m:
            failures.append(f"score: {label} produced no PPL")
            continue
        ppl[label] = float(m.group(1))
        print(f"  {label:8s} PPL {ppl[label]:.3f} (want <= {a.max_ppl})")
        if ppl[label] > a.max_ppl:
            failures.append(f"score: {label} PPL {ppl[label]:.3f} above {a.max_ppl}")
    # The native path reads the trits as stored rather than refitting them, so it should never be
    # the worse of the two by more than run-to-run drift.
    if len(ppl) == 2 and ppl["native"] > ppl["folded"] * 1.05:
        failures.append(f"score: native {ppl['native']:.3f} worse than folded {ppl['folded']:.3f}")


def check_generate(a, ids, failures):
    """The two routes to the same arithmetic must produce the same tokens."""
    got = {}
    for label, native in (("folded", ""), ("native", "all")):
        txt = run([a.generate, a.model, "16"] + ids, env={"SPARKINFER_BONSAI_NATIVE": native})
        m = re.search(r"^OUTPUT_IDS: (.+)$", txt, re.M)
        if not m:
            failures.append(f"generate: {label} produced no output (crash?)")
            continue
        got[label] = m.group(1).split()
        print(f"  {label:8s} {' '.join(got[label][:10])} ...")
    if len(got) == 2 and got["folded"] != got["native"]:
        failures.append("generate: native and folded disagree on greedy tokens")


SERVE_LONG = "List the first 40 prime numbers, separated by commas. Answer directly."


FAILED_PREFIX = "<request failed: "


def _chat(port, prompt, max_tokens, out):
    body = json.dumps({"model": "b", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            m = json.load(r)["choices"][0]["message"]
            out.append(m.get("content") or m.get("reasoning") or "")
    except Exception as e:                                    # noqa: BLE001 - reported, not raised
        out.append(f"{FAILED_PREFIX}{e}>")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# A trial is conclusive when its two alone-baselines agree. Fail on SERVE_FAILS_TO_FAIL failing
# trials out of up to SERVE_CONCLUSIVE_TRIALS; stop after SERVE_MAX_TRIALS whatever happened.
SERVE_CONCLUSIVE_TRIALS = 3
SERVE_FAILS_TO_FAIL = 2
SERVE_MAX_TRIALS = 5


def _wait_gpu_clear(limit_s=120):
    """Poll until the previous server's memory is back (the bot's wait_gpu_clear, 1 GiB). Starting
    the next server into a card still being freed is one way a trial goes wrong for no code reason."""
    for _ in range(limit_s):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            if int(out.split()[0]) < 1024:
                return
        except Exception:                                     # noqa: BLE001 - no nvidia-smi: don't wait
            return
        time.sleep(1)


def _serve_trial(a, label, native):
    """One fresh server, two alone-baselines, one decayed batch.

    Returns ("pass" | "fail" | "inconclusive", detail). "fail" covers a server that never became
    healthy and a request that produced nothing, as well as a row served differently."""
    port = _free_port()
    env = dict(os.environ)
    env["SPARKINFER_BONSAI_NATIVE"] = native
    _wait_gpu_clear()
    # Its own process group, so a hung server is killed with its children rather than left
    # holding the GPU for whatever runs next.
    srv = subprocess.Popen([a.server, "-m", a.model, "--ctx", "16384", "--port", str(port),
                            "--tokenizer", os.path.join(a.tokenizer, "tokenizer.json")],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           env=env, start_new_session=True)
    try:
        up = False
        for _ in range(200):
            if srv.poll() is not None:
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                up = True
                break
            except Exception:                                 # noqa: BLE001 - still starting
                time.sleep(4)
        if not up:
            return "fail", "server never became healthy"

        # Two sequential baselines, so non-determinism at temperature 0 is told apart from a
        # batching fault rather than being reported as one.
        alone = []
        _chat(port, SERVE_LONG, 220, alone)
        _chat(port, SERVE_LONG, 220, alone)

        # Three short rows finish early and leave the long one decoding by itself, which is
        # when the unbatched kernel takes over a state the packed path has already compacted.
        decayed = []
        threads = [threading.Thread(target=_chat, args=(port, SERVE_LONG, 220, decayed))]
        shorts = [[] for _ in range(3)]
        for i, box in enumerate(shorts):
            threads.append(threading.Thread(
                target=_chat, args=(port, f"Write one sentence about the number {i + 1}.", 12, box)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if len(alone) < 2 or not decayed:
            return "fail", "produced no completion"
        # A request that failed is a failure, never an answer to compare: a server that dies on its
        # first request returned two identical "connection refused" strings, which matched.
        failed = [x for x in alone + decayed + [s[0] for s in shorts if s]
                  if x.startswith(FAILED_PREFIX)]
        if failed or srv.poll() is not None:
            return "fail", ("a request failed: " + failed[0][len(FAILED_PREFIX):-1][:120] if failed
                            else "the server exited during the check")
        stable = alone[0] == alone[1]
        same = decayed[0] in alone
        print(f"  {label:8s} baselines {len(alone[0])}/{len(alone[1])} chars, after a decayed "
              f"batch {len(decayed[0])} -- {'matches a baseline' if same else 'MATCHES NEITHER'}"
              f"{'' if stable else '  (baselines disagree: checkpoint is non-deterministic)'}")
        if same:
            return "pass", ""
        if not stable:
            return "inconclusive", "baselines differ at temperature 0"
        print(f"      baseline: {alone[0][:120]!r}")
        print(f"      decayed : {decayed[0][:120]!r}")
        return "fail", "row served differently alone and after a batch"
    finally:
        try:
            os.killpg(os.getpgid(srv.pid), signal.SIGKILL)
        except Exception:                                     # noqa: BLE001 - already gone
            srv.kill()
        srv.wait(timeout=60)
        time.sleep(5)


def serve_verdict(trial):
    """Run trials of one path until they decide. Returns ("pass" | "fail" | "inconclusive", detail).

    A first conclusive trial that passes is a pass, as before. A failure is re-run, and the path
    fails only on SERVE_FAILS_TO_FAIL failing conclusive trials; a pass after a failure needs a
    third trial to settle it. Inconclusive trials (non-deterministic baselines) are re-run too but
    can never convict: if no trial settles the question, the result is inconclusive."""
    fails, passes, details = 0, 0, []
    for _ in range(SERVE_MAX_TRIALS):
        outcome, detail = trial()
        if outcome == "inconclusive":
            details.append(detail)
            continue
        if outcome == "pass":
            passes += 1
            if fails == 0:
                return "pass", ""
        else:
            fails += 1
            details.append(detail)
        if fails >= SERVE_FAILS_TO_FAIL:
            return "fail", f"{details[-1]} ({fails} of {fails + passes} conclusive trials)"
        if fails + passes >= SERVE_CONCLUSIVE_TRIALS:
            return "pass", f"passed {passes} of {fails + passes} conclusive trials"
    if fails + passes == 0:
        return "inconclusive", f"no conclusive trial in {SERVE_MAX_TRIALS}: {details[-1] if details else ''}"
    return "inconclusive", (f"{fails} failing and {passes} passing conclusive trials in "
                            f"{SERVE_MAX_TRIALS} -- not enough to decide")


def check_serve(a, failures, notes=None):
    """A row that outlives its batch must decode exactly as it would have alone."""
    for label, native in (("folded", ""), ("native", "all")):
        outcome, detail = serve_verdict(lambda: _serve_trial(a, label, native))
        if outcome == "fail":
            failures.append(f"serve: {label} {detail}")
        elif outcome == "inconclusive" and notes is not None:
            notes.append(f"serve: {label} inconclusive -- {detail}")
        elif detail and notes is not None:
            notes.append(f"serve: {label} {detail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/root/workspace/models_bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
    ap.add_argument("--reference", default="/root/workspace/models_qwen38",
                    help="the un-quantised checkpoint this model was derived from")
    ap.add_argument("--tokenizer", default="/root/workspace/models_qwen38")
    ap.add_argument("--build", default="/workspace/sparkinfer/build/runtime")
    ap.add_argument("--inspect", default=None,
                    help="bonsai_inspect binary; defaults to the one in --build")
    ap.add_argument("--max-ppl", type=float, default=11.0,
                    help="this passage measures 9.30 native / 9.56 folded; a broken-path gate, not "
                         "a tolerance. Do not copy a threshold from a run on a DIFFERENT passage -- "
                         "the first version of this gate did, set 9.5 from an 8.11 measurement, and "
                         "failed on a model that was working perfectly.")
    ap.add_argument("--min-cos", type=float, default=0.80,
                    help="correct ternary decode lands at ~0.88")
    ap.add_argument("--server", default=None,
                    help="sparkinfer_server binary; defaults to the one beside --build")
    ap.add_argument("--skip", default="",
                    help="comma-separated: tensors,score,generate,serve")
    a = ap.parse_args()
    if not a.inspect:
        a.inspect = os.path.join(a.build, "bonsai_inspect")
    a.score = os.path.join(a.build, "qwen3_gguf_score")
    a.generate = os.path.join(a.build, "qwen3_gguf_generate")
    if not a.server:
        a.server = os.path.join(os.path.dirname(a.build.rstrip("/")), "server", "sparkinfer_server")

    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    failures, notes = [], []
    checks = (
        ("tensors", "decoded weights against the un-quantised checkpoint",
         lambda: check_tensors(a, failures)),
        ("score", "perplexity, folded against native",
         lambda: check_score(a, tokenize(a.tokenizer, PASSAGE), failures)),
        ("generate", "greedy tokens, folded against native",
         lambda: check_generate(a, tokenize(a.tokenizer, PROMPT), failures)),
        ("serve", "a row that outlives its batch, against the same row asked alone",
         lambda: check_serve(a, failures, notes)),
    )
    for name, what, run_check in checks:
        if name in skip:
            continue
        print(f"{name}: {what}")
        try:
            run_check()
        except Exception as e:                                # noqa: BLE001 - reported per check
            # A missing module on the box (2026-09-25: safetensors) or a hung binary: a failure of
            # THIS check, named as such, so the bot can tell it apart from the others -- main fails
            # it too when the cause is the box, and a check main also fails is not gated.
            failures.append(f"{name}: raised {type(e).__name__}: {str(e)[:160]}")

    for n in notes:
        print(f"NOTE: {n}")
    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nbonsai_regression: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
