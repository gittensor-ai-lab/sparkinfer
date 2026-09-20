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
             DECAYS to one row (three short requests beside it). Both must come back identical.
             Everything above this line runs one request at a time and is blind to the whole
             continuous-batch path -- which is where the worst bug this model turned up lived,
             and it was not even ternary-specific: decode_packed compacts a session's recurrent
             state to bf16, and a row that outlives its batch was served from that state read at
             the wrong width. Costs a server start per path, so it is the slow check.

Exits non-zero on the first failed check. Thresholds are deliberately loose -- they are there to
catch a broken path, not to police the third decimal place.
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
        out.append(f"<request failed: {e}>")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def check_serve(a, failures):
    """A row that outlives its batch must decode exactly as it would have alone."""
    for label, native in (("folded", ""), ("native", "all")):
        port = _free_port()
        env = dict(os.environ)
        env["SPARKINFER_BONSAI_NATIVE"] = native
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
                except Exception:                             # noqa: BLE001 - still starting
                    time.sleep(4)
            if not up:
                failures.append(f"serve: {label} server never became healthy")
                continue

            alone = []
            _chat(port, SERVE_LONG, 220, alone)

            # Three short rows finish early and leave the long one decoding by itself, which is
            # when the unbatched kernel takes over a state the packed path has already compacted.
            decayed, threads = [], []
            threads.append(threading.Thread(target=_chat, args=(port, SERVE_LONG, 220, decayed)))
            shorts = [[] for _ in range(3)]
            for i, box in enumerate(shorts):
                threads.append(threading.Thread(
                    target=_chat, args=(port, f"Write one sentence about the number {i + 1}.", 12, box)))
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            if not alone or not decayed:
                failures.append(f"serve: {label} produced no completion")
                continue
            same = alone[0] == decayed[0]
            print(f"  {label:8s} alone {len(alone[0]):4d} chars, after a decayed batch "
                  f"{len(decayed[0]):4d} -- {'identical' if same else 'DIFFERENT'}")
            if not same:
                print(f"      alone  : {alone[0][:120]!r}")
                print(f"      decayed: {decayed[0][:120]!r}")
                failures.append(f"serve: {label} row served differently alone and after a batch")
        finally:
            try:
                os.killpg(os.getpgid(srv.pid), signal.SIGKILL)
            except Exception:                                 # noqa: BLE001 - already gone
                srv.kill()
            srv.wait(timeout=60)
            time.sleep(5)


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
    failures = []

    if "tensors" not in skip:
        print("tensors: decoded weights against the un-quantised checkpoint")
        check_tensors(a, failures)
    if "score" not in skip:
        print("score: perplexity, folded against native")
        check_score(a, tokenize(a.tokenizer, PASSAGE), failures)
    if "generate" not in skip:
        print("generate: greedy tokens, folded against native")
        check_generate(a, tokenize(a.tokenizer, PROMPT), failures)
    if "serve" not in skip:
        print("serve: a row that outlives its batch, against the same row asked alone")
        check_serve(a, failures)

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nbonsai_regression: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
