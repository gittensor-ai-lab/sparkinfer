#!/usr/bin/env python3
"""Greedy speculation past 16K tokens of context is lossless: the same long prompts, one at a time,
against a launch that speculates and one that does not (SPARKINFER_SPECULATIVE=0, the drafter
loaded), both SPARKINFER_DETERMINISTIC=1. Every completion must be identical, and the speculating
launch must actually have speculated (sparkinfer_speculative_runs_total).

usage: spec_long_ctx_lossless.py <server-bin> <model-dir> <draft-dir> <repo-root> [ENV=VAL ...]
"""
import glob, json, os, signal, subprocess, sys, time, urllib.request

server, model, draft, root = sys.argv[1:5]
extra = dict(kv.split("=", 1) for kv in sys.argv[5:])
PORT = 18159
LENGTHS = [int(x) for x in os.environ.get("SLL_LENGTHS", "20000,40000").split(",")]


def corpus(patterns):
    out = []
    for p in patterns:
        for f in sorted(glob.glob(os.path.join(root, p), recursive=True)):
            out.append(open(f, encoding="utf-8", errors="ignore").read())
    return "\n\n".join(out)


TEXTS = [(corpus(["docs/**/*.md", "*.md"]), "Continue the document above with a new section."),
         (corpus(["runtime/src/**/*.cpp"]), "Continue the source file above with the next function.")]


def launch(spec):
    env = dict(os.environ, **extra, SPARKINFER_DETERMINISTIC="1")
    if not spec:
        env["SPARKINFER_SPECULATIVE"] = "0"
    p = subprocess.Popen([server, "-m", model, "--tokenizer", os.path.join(model, "tokenizer.json"),
                          "--model-name", "q", "--ctx", "131072", "--draft-model", draft,
                          "--host", "127.0.0.1", "--port", str(PORT)],
                         stdout=open(f"/tmp/sll_{int(spec)}.log", "w"), stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)
    for _ in range(360):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5)
            return p
        except Exception:
            if p.poll() is not None:
                sys.exit("server exited")
            time.sleep(5)
    sys.exit("server did not come up")


def stop(p):
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(90)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)


def complete(text):
    body = {"model": "q", "messages": [{"role": "user", "content": text}], "max_tokens": 256,
            "temperature": 0.0}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"], out["usage"], time.time() - t


def runs():
    text = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10).read().decode()
    return sum(float(l.split()[-1]) for l in text.splitlines()
               if l.startswith("sparkinfer_speculative_runs_total"))


def run(spec):
    p = launch(spec)
    try:
        res = []
        for body, instruction in TEXTS:
            for n in LENGTHS:
                seg = body[len(body) // 3:][: int(n * 3.6)]
                res.append(complete(seg + "\n\n" + instruction))
        return res, runs()
    finally:
        stop(p)


spec, spec_runs = run(True)
plain, _ = run(False)
ok = spec_runs > 0
for (a, ua, ta), (b, ub, tb) in zip(spec, plain):
    same = a == b
    ok = ok and same
    print(f"prompt {ua['prompt_tokens']:6d} tokens: identical {same}; wall {ta:.1f} s speculating, "
          f"{tb:.1f} s plain")
print(f"speculative runs on the speculating launch: {spec_runs:.0f}")
print("SPEC_LONG_CTX_LOSSLESS", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
