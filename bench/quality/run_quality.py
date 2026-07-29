#!/usr/bin/env python3
"""Run the LLM-quality benchmark suite against a model backend and score it.

Quality (does the model still answer well?) is orthogonal to sparkinfer's existing
top-1/KL gate (does the kernel match llama.cpp's distribution?). Run both engines here
and the score diff is your **quality parity** - proof that an optimization kept capability,
not just token agreement.

Usage:
  # offline pipeline check (no GPU/model needed):
  python3 run_quality.py --backend oracle
  python3 run_quality.py --backend mock
  python3 run_quality.py --self-test-llama-server   # mock HTTP server, no GPU

  # sparkinfer:
  python3 run_quality.py --backend sparkinfer \\
      --model /workspace/models/Qwen3-30B-A3B-Q4_K_M.gguf \\
      --bin ./build/qwen3_gguf_generate --tokenizer /workspace/models/tokenizer.json

  # llama.cpp server (start llama-server first — preferred A/B path):
  python3 run_quality.py --backend llama-server --llama-url http://127.0.0.1:8082

  # llama.cpp one-shot CLI (reloads model each item — slow):
  python3 run_quality.py --backend llama --llama-cli /workspace/.llamacpp/build/bin/llama-cli \\
      --model /workspace/models/Qwen3-30B-A3B-Q4_K_M.gguf

  # only some benchmarks / fewer items:
  python3 run_quality.py --backend sparkinfer --benchmarks gsm8k,humaneval --limit 20 ...

  # named quality tiers:
  python3 run_quality.py --backend sparkinfer --tier development ...  # ~10%, 78 items
  python3 run_quality.py --backend sparkinfer --tier benchmark ...    # ~25%, 196 items
"""
import argparse, json, os, subprocess, sys, glob, urllib.error, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorers

HERE = os.path.dirname(os.path.abspath(__file__))
# Qwen3 chat template (thinking disabled) - identical to runtime/tools/run_qwen3.py.
CHAT = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MAXNEW = {"gsm8k": 320, "humaneval": 320, "mmlu_pro": 96, "ifeval": 200, "bfcl": 160}
TIERS = {
    # Fast pre-merge/dev check over every suite.
    "development": {"bfcl": 30, "gsm8k": 13, "humaneval": 2, "ifeval": 3, "mmlu_pro": 30},
    # Heavier benchmark check used for release/frontier quality claims.
    "benchmark": {"bfcl": 75, "gsm8k": 33, "humaneval": 5, "ifeval": 8, "mmlu_pro": 75},
}


# - per-benchmark prompt construction -

def build_prompt(item):
    b = item["benchmark"]
    if b == "gsm8k":
        return (item["prompt"] + "\n\nSolve it step by step. On the last line, write the "
                "final answer as: #### <number>")
    if b == "mmlu_pro":
        opts = "\n".join(f"{chr(65+i)}) {c}" for i, c in enumerate(item["choices"]))
        return (item["prompt"] + "\n\n" + opts +
                "\n\nRespond with just the letter of the correct answer.")
    if b == "ifeval":
        return item["prompt"]
    if b == "humaneval":
        return ("Complete the following Python function. Respond with only the function "
                "code in a code block.\n\n" + item["prompt"])
    if b == "bfcl":
        tools = json.dumps(item["tools"], indent=2)
        return ("You can call these tools:\n" + tools + "\n\nUser: " + item["prompt"] +
                '\n\nRespond with ONLY a JSON object: {"tool": "<name>", "arguments": {...}}')
    raise ValueError(b)


# - backends: generate(prompt_text, max_new) -> text -

class OracleBackend:
    """Derives a known-passing answer from the gold - verifies the runner+scorers end-to-end
    (works on real data too, for the answer-derivable benchmarks)."""
    def __init__(self, items): pass
    def gen_for(self, item):
        b = item["benchmark"]
        if b == "gsm8k":    return f"... #### {item['target']}"
        if b == "mmlu_pro": return f"The answer is {item['answer']}."
        if b == "bfcl":
            return json.dumps({"tool": item["target"]["name"], "arguments": item["target"]["arguments"]})
        # ifeval / humaneval have no gold text/solution in the data -> oracle can't synthesize one
        return ""

class MockBackend:
    def gen_for(self, item): return "I don't know."

class SparkinferBackend:
    def __init__(self, model, binary, tokenizer):
        from tokenizers import Tokenizer
        self.model, self.binary = model, binary
        self.tok = Tokenizer.from_file(tokenizer)
    def gen_for(self, item):
        ids = self.tok.encode(CHAT.format(p=build_prompt(item))).ids
        n = MAXNEW.get(item["benchmark"], 200)
        cmd = [self.binary, self.model, str(n)] + [str(i) for i in ids]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        out = []
        for line in r.stdout.splitlines():
            if line.startswith("OUTPUT_IDS:"):
                out = [int(x) for x in line.split(":", 1)[1].split()]
        return self.tok.decode(out)

class LlamaBackend:
    def __init__(self, model, llama_cli):
        self.model, self.cli = model, llama_cli
    def gen_for(self, item):
        n = MAXNEW.get(item["benchmark"], 200)
        prompt = CHAT.format(p=build_prompt(item))
        cmd = [self.cli, "-m", self.model, "-p", prompt, "-n", str(n),
               "-ngl", "99", "--no-display-prompt", "-no-cnv", "--temp", "0"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return r.stdout


class LlamaServerBackend:
    """llama.cpp `llama-server` HTTP backend (keeps the model resident — the A/B path).

    Tries native `/completion` first, then OpenAI-compatible `/v1/completions`.
    """

    def __init__(self, base_url, timeout=600):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _post_json(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _content_from_completion(body):
        # llama.cpp /completion: {"content": "..."} or {"content": ["..."]}
        if isinstance(body.get("content"), str):
            return body["content"]
        if isinstance(body.get("content"), list) and body["content"]:
            return "".join(str(x) for x in body["content"])
        # OpenAI /v1/completions
        choices = body.get("choices") or []
        if choices:
            c0 = choices[0]
            if isinstance(c0.get("text"), str):
                return c0["text"]
            msg = c0.get("message") or {}
            if isinstance(msg.get("content"), str):
                return msg["content"]
        raise ValueError("llama-server response missing completion text: " + json.dumps(body)[:200])

    def gen_for(self, item):
        n = MAXNEW.get(item["benchmark"], 200)
        prompt = CHAT.format(p=build_prompt(item))
        # Prefer native llama-server /completion (temp 0 = greedy, matches CLI path).
        try:
            body = self._post_json("/completion", {
                "prompt": prompt,
                "n_predict": n,
                "temperature": 0.0,
                "stream": False,
            })
            return self._content_from_completion(body)
        except urllib.error.HTTPError as e:
            if e.code not in (404, 405):
                raise
        body = self._post_json("/v1/completions", {
            "prompt": prompt,
            "max_tokens": n,
            "temperature": 0.0,
            "stream": False,
        })
        return self._content_from_completion(body)


def make_backend(args, items):
    if args.backend == "oracle": return OracleBackend(items)
    if args.backend == "mock":   return MockBackend()
    if args.backend == "sparkinfer":
        return SparkinferBackend(args.model, args.bin, args.tokenizer)
    if args.backend == "llama":
        return LlamaBackend(args.model, args.llama_cli)
    if args.backend == "llama-server":
        return LlamaServerBackend(args.llama_url)
    raise SystemExit("unknown backend " + args.backend)


# - driver -

def load(benchmarks, limit, tier=""):
    items = []
    tier_counts = TIERS.get(tier, {})
    for path in sorted(glob.glob(os.path.join(HERE, "data", "*.jsonl"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if benchmarks and name not in benchmarks: continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        n = limit or tier_counts.get(name, 0)
        items += rows[:n] if n else rows
    return items


def self_test_llama_server():
    """Spin a tiny stdlib HTTP mock of llama-server; no GPU required."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            payload = json.loads(raw.decode("utf-8") or "{}")
            prompt = payload.get("prompt", "")
            text = "mock-ok"
            if "Solve it step by step" in prompt:
                text = "reasoning\n#### 42"
            body = json.dumps({"content": text}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        be = LlamaServerBackend(f"http://127.0.0.1:{port}", timeout=5)
        item = {
            "id": "self0",
            "benchmark": "gsm8k",
            "prompt": "What is 6*7?",
            "target": "42",
        }
        out = be.gen_for(item)
        r = scorers.SCORERS["gsm8k"](item, out)
        if not r["pass"]:
            raise SystemExit(f"self-test failed: out={out!r} detail={r['detail']}")
        print("llama-server self-test OK  (mock HTTP /completion + gsm8k scorer)")
    finally:
        httpd.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["oracle", "mock", "sparkinfer", "llama", "llama-server"],
                    help="generation backend")
    ap.add_argument("--model"); ap.add_argument("--bin"); ap.add_argument("--tokenizer")
    ap.add_argument("--llama-cli")
    ap.add_argument("--llama-url", default="http://127.0.0.1:8082",
                    help="llama-server base URL (for --backend llama-server)")
    ap.add_argument("--benchmarks", default="", help="comma list; default all")
    ap.add_argument("--limit", type=int, default=0, help="items per benchmark (0=all)")
    ap.add_argument("--tier", choices=sorted(TIERS),
                    help="named per-suite sample: development ~=10%, benchmark ~=25%")
    ap.add_argument("--out", default="", help="write per-item JSONL results here")
    ap.add_argument("--self-test-llama-server", action="store_true",
                    help="CPU-only mock of llama-server HTTP; verifies the backend wiring")
    args = ap.parse_args()

    if args.self_test_llama_server:
        self_test_llama_server()
        return
    if not args.backend:
        ap.error("--backend is required (or pass --self-test-llama-server)")

    benchmarks = set(b for b in args.benchmarks.split(",") if b)
    items = load(benchmarks, args.limit, args.tier or "")
    backend = make_backend(args, items)

    agg, results = {}, []
    for it in items:
        b = it["benchmark"]
        try:
            out = backend.gen_for(it)
            r = scorers.SCORERS[b](it, out)
        except Exception as e:  # keep long benchmark tiers running after malformed outputs
            r = {"score": 0.0, "pass": False, "detail": "ERROR: " + repr(e)}
        agg.setdefault(b, []).append(r["score"])
        results.append({"id": it["id"], "benchmark": b, "score": r["score"],
                        "pass": r["pass"], "detail": r["detail"]})
        print(f"  {b:10s} {it['id']:6s} {'PASS' if r['pass'] else 'fail':4s}  {r['detail'][:70]}")

    label = f"{args.backend}" + (f" / {args.tier}" if args.tier else "")
    print("\n" + "=" * 46 + f"\nQUALITY REPORT  -  backend={label}\n" + "=" * 46)
    total = []
    for b in sorted(agg):
        s = agg[b]; total += s
        print(f"  {b:12s}  {sum(s)/len(s)*100:5.1f}%   ({sum(1 for x in s if x==1.0)}/{len(s)})")
    if total:
        print("  " + "-" * 30 + f"\n  {'OVERALL':12s}  {sum(total)/len(total)*100:5.1f}%   ({len(total)} items)")

    if args.out:
        with open(args.out, "w") as f:
            for r in results: f.write(json.dumps(r) + "\n")
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()
