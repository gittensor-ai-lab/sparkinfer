#!/usr/bin/env python3
"""End-to-end check for mixed steps (SPARKINFER_MIXED_CHUNK) under concurrent load.

Two launches from the same binary run the same load, mixing on and off:
  wave 1   24 chats at once, one shared system prompt, user turns from a few words to ~1500
           tokens: greedy, sampled (seeded) and logprobs requests
  wave 2   a follow-up turn for 16 of those conversations, at once: their prompts start with
           wave 1's, so they hit the prefix cache, which only has entries if the checkpoints were
           taken while the prompts were prefilled
Checks, per launch: every request succeeds, stops at its max_tokens or end of turn, and a logprobs
request has one entry per completion token. Across the launches: wave 2 takes as many prefix-cache
hits with mixing as without (the checkpoints were taken), and how far each greedy answer agrees.
The two launches run different pass shapes, so near ties may flip late; a layout slip (a prompt's
tokens or state in another's slot) shows up as answers that disagree from the first tokens.

usage: mixed_conc_check.py <server-bin> <model-dir-or-gguf> [ENV=VAL ...]   (extra env for both)
MCC_TOKENIZER=<tokenizer.json> for a model that does not ship one beside it (a GGUF file).
MCC_ON / MCC_OFF=<VAR=VAL[,VAR=VAL]> replace the toggle (default SPARKINFER_MIXED_CHUNK=1024 / 0),
e.g. SPARKINFER_PACKED_DECODE=1 / 0 to check packed decode against one forward per request.
"""
import json, os, signal, subprocess, sys, threading, time, urllib.request

server, model = sys.argv[1:3]
extra = dict(kv.split("=", 1) for kv in sys.argv[3:])
PORT = 18141
WORDS = ("river stone lantern copper meadow signal harbor quiet engine paper orbit garden window "
         "thunder ladder silver market candle forest bridge").split()


def toggle(spec):
    return dict(kv.split("=", 1) for kv in spec.split(",") if kv)


def launch(mixed):
    env = dict(os.environ, **extra, SPARKINFER_PREFIX_CACHE="1")
    env.update(toggle(os.environ.get("MCC_ON", "SPARKINFER_MIXED_CHUNK=1024") if mixed
                      else os.environ.get("MCC_OFF", "SPARKINFER_MIXED_CHUNK=0")))
    tok = os.environ.get("MCC_TOKENIZER") or os.path.join(model, "tokenizer.json")
    p = subprocess.Popen([server, "-m", model, "--tokenizer", tok,
                          "--model-name", "q", "--ctx", "32768", "--host", "127.0.0.1", "--port", str(PORT)],
                         stdout=open(f"/tmp/mcc_{int(mixed)}.log", "w"), stderr=subprocess.STDOUT, env=env,
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
        p.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


def metric(name):
    text = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10).read().decode()
    return sum(float(l.split()[-1]) for l in text.splitlines() if l.startswith(name))


SYSTEM = "You are a careful assistant. " + " ".join(WORDS[i % len(WORDS)] for i in range(400))


def user_text(i):
    n = [4, 30, 120, 300, 700, 1200][i % 6]
    return f"Request {i}: " + " ".join(WORDS[(i * 7 + k) % len(WORDS)] for k in range(n)) + \
        ". Summarize the words above in a few sentences."


def kind(i):
    return "logprobs" if i % 6 == 5 else ("sampled" if i % 6 == 4 else "greedy")


def body(i, messages):
    b = {"model": "q", "messages": messages, "max_tokens": 48 + (i % 5) * 24, "temperature": 0.0}
    if kind(i) == "sampled":
        b.update(temperature=0.8, seed=1000 + i)
    if kind(i) == "logprobs":
        b.update(logprobs=True, top_logprobs=2)
    return b


def wave(bodies):
    out = [None] * len(bodies)
    def run(k):
        out[k] = post(bodies[k])
    th = [threading.Thread(target=run, args=(k,)) for k in range(len(bodies))]
    for t in th:
        t.start()
        time.sleep(0.05)
    for t in th:
        t.join()
    return out


def run_launch(mixed):
    p = launch(mixed)
    try:
        conv = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text(i)}]
                for i in range(24)]
        r1 = wave([body(i, conv[i]) for i in range(24)])
        hits0 = metric("sparkinfer_prefix_cache_hits")
        b2 = []
        for i in range(16):
            ans = r1[i].get("choices", [{}])[0].get("message", {}).get("content", "")
            conv[i] = conv[i] + [{"role": "assistant", "content": ans},
                                 {"role": "user", "content": f"Now list three of those words, request {i}."}]
            b2.append(body(i, conv[i]))
        r2 = wave(b2)
        hits = metric("sparkinfer_prefix_cache_hits") - hits0
        return r1, r2, hits
    finally:
        stop(p)


def problems(results, bodies_max):
    bad = []
    for i, r in enumerate(results):
        if "error" in r or not r.get("choices"):
            bad.append(f"{i}: {r.get('error', 'no choices')}")
            continue
        c = r["choices"][0]
        toks = r.get("usage", {}).get("completion_tokens", 0)
        if c.get("finish_reason") not in ("stop", "length"):
            bad.append(f"{i}: finish_reason {c.get('finish_reason')}")
        if c.get("finish_reason") == "length" and toks != bodies_max[i]:
            bad.append(f"{i}: length stop at {toks} of {bodies_max[i]}")
        if kind(i) == "logprobs":
            # completion_tokens counts an end-of-turn token; logprobs carry no entry for it.
            n = len((c.get("logprobs") or {}).get("content") or [])
            want = toks - 1 if c.get("finish_reason") == "stop" else toks
            if n != want:
                bad.append(f"{i}: {n} logprobs for {toks} tokens ({c.get('finish_reason')})")
    return bad


def agree(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


on, off = run_launch(True), run_launch(False)
ok = True
for name, (r1, r2, hits) in (("mixed", on), ("plain", off)):
    bad = problems(r1, [48 + (i % 5) * 24 for i in range(24)]) + \
          [f"wave2 {x}" for x in problems(r2, [48 + (i % 5) * 24 for i in range(16)])]
    print(f"{name}: {len(bad)} problems, wave-2 prefix-cache hits {hits:.0f}/16")
    for b in bad[:10]:
        print("   ", b)
    ok = ok and not bad
ok = ok and on[2] >= off[2] and on[2] > 0
rows = []
for w, (a, b) in enumerate(((on[0], off[0]), (on[1], off[1]))):
    for i, (x, y) in enumerate(zip(a, b)):
        if kind(i) == "sampled" or "error" in x or "error" in y:
            continue
        cx = x["choices"][0]["message"]["content"]
        cy = y["choices"][0]["message"]["content"]
        rows.append((w + 1, i, agree(cx, cy), min(len(cx), len(cy))))
early = [r for r in rows if r[2] < min(20, r[3])]
full = sum(1 for r in rows if r[2] == r[3])
print(f"greedy answers: {len(rows)}, identical {full}, diverging within 20 chars {len(early)}")
for w, i, n, ln in early:
    print(f"    wave {w} request {i}: agree {n} of {ln} chars")
# A few early flips can be near-ties; most diverging at once cannot.
ok = ok and len(early) * 4 <= len(rows)
print("MIXED_CONC_CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
