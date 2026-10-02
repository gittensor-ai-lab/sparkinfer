#!/usr/bin/env python3
"""Smoke test: C long prompts (~20K tokens each) at once against a running server with a drafter.
Every request must complete with its full max_tokens, and the server must have speculated.
usage: spec_long_ctx_conc.py <port> <repo-root> [C]"""
import glob, json, os, sys, threading, time, urllib.request

port, root = sys.argv[1:3]
C = int(sys.argv[3]) if len(sys.argv) > 3 else 4
text = "\n\n".join(open(f, encoding="utf-8", errors="ignore").read()
                   for f in sorted(glob.glob(os.path.join(root, "runtime/src/**/*.cpp"), recursive=True)))


def metric(name):
    t = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10).read().decode()
    return sum(float(l.split()[-1]) for l in t.splitlines() if l.startswith(name))


res = [None] * C


def one(i):
    seg = text[i * 90000:][:72000]
    body = {"model": "q", "messages": [{"role": "user", "content": seg + "\n\nContinue the file."}],
            "max_tokens": 256, "temperature": 0.7, "seed": i, "ignore_eos": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        res[i] = json.loads(urllib.request.urlopen(req, timeout=1800).read())["usage"]
    except Exception as e:
        res[i] = {"error": str(e)}


before = metric("sparkinfer_speculative_tokens_total")
t0 = time.time()
th = [threading.Thread(target=one, args=(i,)) for i in range(C)]
for t in th:
    t.start()
for t in th:
    t.join()
spec = metric("sparkinfer_speculative_tokens_total") - before
toks = sum(r.get("completion_tokens", 0) for r in res)
print(f"c{C}: prompts {[r.get('prompt_tokens') for r in res]}, {toks} tokens in {time.time() - t0:.1f} s, "
      f"speculated tokens {spec:.0f}")
ok = all("error" not in r and r.get("completion_tokens") == 256 for r in res) and spec > 0
print("SPEC_LONG_CTX_CONC", "PASS" if ok else "FAIL", "" if ok else res)
