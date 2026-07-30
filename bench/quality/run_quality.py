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

  # sparkinfer:
  python3 run_quality.py --backend sparkinfer \
      --model /workspace/models/Qwen3-30B-A3B-Q4_K_M.gguf \
      --bin ./build/qwen3_gguf_generate --tokenizer /workspace/models/tokenizer.json

  # llama.cpp reference:
  python3 run_quality.py --backend llama --llama-cli /workspace/.llamacpp/build/bin/llama-cli \
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
    """llama.cpp `llama-server` over HTTP - keeps the model resident across items.

    `--backend llama` reloads the GGUF per item via llama-cli, which is impractical
    for the benchmark tier; this talks to an already-running server instead. Prefers
    the native `/completion` route and falls back to the OpenAI-compatible
    `/v1/completions` when a build doesn't expose it. Greedy (temperature 0) so the
    reference matches the other backends.
    """
    def __init__(self, url, timeout=600):
        self.base = url.rstrip("/")
        self.timeout = timeout
        self.openai_route = False   # latched after one fallback, so we don't re-probe per item

    def _post(self, path, payload):
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def gen_for(self, item):
        n = MAXNEW.get(item["benchmark"], 200)
        prompt = CHAT.format(p=build_prompt(item))
        if not self.openai_route:
            try:
                return self._post("/completion", {"prompt": prompt, "n_predict": n,
                                                  "temperature": 0.0})["content"]
            except urllib.error.HTTPError as e:
                if e.code not in (404, 405, 501):
                    raise
                self.openai_route = True
        d = self._post("/v1/completions", {"prompt": prompt, "max_tokens": n, "temperature": 0.0})
        return d["choices"][0]["text"]


def make_backend(args, items):
    if args.backend == "oracle": return OracleBackend(items)
    if args.backend == "mock":   return MockBackend()
    if args.backend == "sparkinfer":
        return SparkinferBackend(args.model, args.bin, args.tokenizer)
    if args.backend == "llama":
        return LlamaBackend(args.model, args.llama_cli)
    if args.backend == "llama-server":
        if not args.llama_url:
            raise SystemExit("--backend llama-server requires --llama-url")
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
    """Drive LlamaServerBackend against a stub server - no GPU, no model, no llama.cpp.

    Checks the request shape both endpoints receive, that the native route is used when
    present, that a 404 latches the OpenAI fallback, and that what comes back still
    scores through scorers.SCORERS. Returns a process exit code.
    """
    import http.server, threading

    seen = []

    def make_server(native_ok):
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass          # keep the harness output clean
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((self.path, body))
                if self.path == "/completion":
                    if not native_ok:
                        self.send_response(404); self.end_headers(); return
                    payload = {"content": "... #### 42"}
                elif self.path == "/v1/completions":
                    payload = {"choices": [{"text": "... #### 42"}]}
                else:
                    self.send_response(404); self.end_headers(); return
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    item = {"benchmark": "gsm8k", "id": "selftest", "prompt": "2+2?", "target": "42"}
    failures = []

    def check(label, cond):
        print(f"  {'ok  ' if cond else 'FAIL'}  {label}")
        if not cond: failures.append(label)

    print("self-test: llama-server backend")
    for native_ok, want_path, want_key in ((True, "/completion", "n_predict"),
                                           (False, "/v1/completions", "max_tokens")):
        seen.clear()
        srv = make_server(native_ok)
        try:
            be = LlamaServerBackend(f"http://127.0.0.1:{srv.server_address[1]}")
            out = be.gen_for(item)
        finally:
            srv.shutdown()
        tag = "native" if native_ok else "fallback"
        check(f"{tag}: request reached {want_path}", seen[-1][0] == want_path)
        check(f"{tag}: sent {want_key}", want_key in seen[-1][1])
        check(f"{tag}: greedy (temperature 0)", seen[-1][1].get("temperature") == 0.0)
        check(f"{tag}: prompt uses the chat template", "<|im_start|>" in seen[-1][1]["prompt"])
        check(f"{tag}: generation returned", out.strip().endswith("42"))
        check(f"{tag}: scorer accepts the output", scorers.SCORERS["gsm8k"](item, out)["pass"])
        if not native_ok:
            check("fallback: probed /completion first", seen[0][0] == "/completion")

    print("\nself-test: %s" % ("PASS" if not failures else "FAIL (%d)" % len(failures)))
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["oracle", "mock", "sparkinfer", "llama", "llama-server"])
    ap.add_argument("--model"); ap.add_argument("--bin"); ap.add_argument("--tokenizer")
    ap.add_argument("--llama-cli")
    ap.add_argument("--llama-url", help="base URL of a running llama-server, e.g. http://localhost:8082")
    ap.add_argument("--self-test-llama-server", action="store_true",
                    help="verify the llama-server wiring against a stub (no GPU/model)")
    ap.add_argument("--benchmarks", default="", help="comma list; default all")
    ap.add_argument("--limit", type=int, default=0, help="items per benchmark (0=all)")
    ap.add_argument("--tier", choices=sorted(TIERS),
                    help="named per-suite sample: development ~=10%%, benchmark ~=25%%")
    ap.add_argument("--out", default="", help="write per-item JSONL results here")
    args = ap.parse_args()

    if args.self_test_llama_server:
        sys.exit(self_test_llama_server())
    if not args.backend:
        ap.error("--backend is required (or use --self-test-llama-server)")

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
