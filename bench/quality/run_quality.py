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
"""
import argparse, json, os, subprocess, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorers

HERE = os.path.dirname(os.path.abspath(__file__))
# Qwen3 chat template (thinking disabled) - identical to runtime/tools/run_qwen3.py.
CHAT = "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MAXNEW = {"gsm8k": 320, "humaneval": 320, "mmlu_pro": 96, "ifeval": 200, "bfcl": 160}


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


def make_backend(args, items):
    if args.backend == "oracle": return OracleBackend(items)
    if args.backend == "mock":   return MockBackend()
    if args.backend == "sparkinfer":
        return SparkinferBackend(args.model, args.bin, args.tokenizer)
    if args.backend == "llama":
        return LlamaBackend(args.model, args.llama_cli)
    raise SystemExit("unknown backend " + args.backend)


# - driver -

def load(benchmarks, limit):
    items = []
    for path in sorted(glob.glob(os.path.join(HERE, "data", "*.jsonl"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if benchmarks and name not in benchmarks: continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        items += rows[:limit] if limit else rows
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["oracle", "mock", "sparkinfer", "llama"])
    ap.add_argument("--model"); ap.add_argument("--bin"); ap.add_argument("--tokenizer")
    ap.add_argument("--llama-cli")
    ap.add_argument("--benchmarks", default="", help="comma list; default all")
    ap.add_argument("--limit", type=int, default=0, help="items per benchmark (0=all)")
    ap.add_argument("--out", default="", help="write per-item JSONL results here")
    args = ap.parse_args()

    benchmarks = set(b for b in args.benchmarks.split(",") if b)
    items = load(benchmarks, args.limit)
    backend = make_backend(args, items)

    agg, results = {}, []
    for it in items:
        b = it["benchmark"]
        out = backend.gen_for(it)
        r = scorers.SCORERS[b](it, out)
        agg.setdefault(b, []).append(r["score"])
        results.append({"id": it["id"], "benchmark": b, "score": r["score"],
                        "pass": r["pass"], "detail": r["detail"]})
        print(f"  {b:10s} {it['id']:6s} {'PASS' if r['pass'] else 'fail':4s}  {r['detail'][:70]}")

    print("\n" + "=" * 46 + f"\nQUALITY REPORT  -  backend={args.backend}\n" + "=" * 46)
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
