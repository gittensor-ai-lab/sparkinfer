#!/usr/bin/env python3
"""Single-stream speculative decode speed by context length and kind of text.

Prompts are built from this repository: prose from its Markdown docs, code from its C++ sources,
cut to about 1K / 4K / 16K tokens (each from two different offsets), with an instruction to
continue the text. Greedy, one request at a time, streamed. Decode tok/s per request =
(tokens - 1) / (last token time - first token time). Run it against a server with the drafter and
against one with SPARKINFER_SPECULATIVE=0 (the drafter loaded, plain decode) to get the speedup.

usage: spec_long_ctx.py <port> <label> <repo-root> [max_tokens]
"""
import glob, json, os, sys, time, urllib.request

port, label, root = sys.argv[1:4]
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 384
LENGTHS = [int(x) for x in os.environ.get("SLC_LENGTHS", "1000,4000,16000").split(",")]
CHARS_PER_TOKEN = 3.6   # rough; the server reports the real prompt length


def corpus(patterns):
    text = []
    for p in patterns:
        for f in sorted(glob.glob(os.path.join(root, p), recursive=True)):
            try:
                text.append(open(f, encoding="utf-8", errors="ignore").read())
            except OSError:
                pass
    return "\n\n".join(text)


KINDS = {
    "prose": (corpus(["docs/**/*.md", "*.md"]),
              "Continue the document above with a new section of several paragraphs, in the same style."),
    "code": (corpus(["runtime/src/**/*.cpp"]),
             "Continue the source file above with the next function, in the same style."),
}


def one(text, instruction):
    body = {"model": "q", "messages": [{"role": "user", "content": text + "\n\n" + instruction}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t_first = t_last = None
    usage = None
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            l = raw.decode().strip()
            if not l.startswith("data:") or l[5:].strip() == "[DONE]":
                continue
            ev = json.loads(l[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    now = time.time()
                    t_first = t_first or now
                    t_last = now
    n = (usage or {}).get("completion_tokens", 0)
    rate = (n - 1) / max(1e-6, t_last - t_first) if t_first and t_last and n > 1 else 0.0
    return (usage or {}).get("prompt_tokens", 0), n, rate


one(KINDS["prose"][0][:2000], KINDS["prose"][1])   # warm-up
for kind, (text, instruction) in KINDS.items():
    for length in LENGTHS:
        chars = int(length * CHARS_PER_TOKEN)
        rates, ptoks = [], []
        for start in (0, len(text) // 2):
            seg = text[start:start + chars]
            if len(seg) < chars:
                seg = text[:chars]
            p, n, r = one(seg, instruction)
            rates.append(r)
            ptoks.append(p)
        print(f"{label} {kind:5s} ~{length // 1000}K (prompt {min(ptoks)}-{max(ptoks)} tokens): "
              f"{sum(rates) / len(rates):.1f} tok/s  ({' '.join('%.0f' % r for r in rates)})", flush=True)
