#!/usr/bin/env bash
# Compare API decode throughput (C++ tokenizer) vs native qwen3_gguf_bench.
# Usage: bench_api_vs_native.sh [model.gguf] [api_url]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GGUF="${1:-$ROOT/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
API="${2:-http://127.0.0.1:8080}"
BENCH="${ROOT}/build/runtime/qwen3_gguf_bench"
DECODE_N=128
CONTEXTS=(128 512 4096 16384)

if [ ! -x "$BENCH" ]; then
  echo "!! missing $BENCH — build with -DBUILD_SERVER=ON" >&2
  exit 1
fi

python3 <<'PY'
import json, subprocess, sys, time, urllib.request
from tokenizers import Tokenizer

ROOT = sys.argv[1]
GGUF = sys.argv[2]
API = sys.argv[3]
BENCH = sys.argv[4]
DECODE_N = int(sys.argv[5])
CONTEXTS = [int(x) for x in sys.argv[6].split()]

tok_path = f"{ROOT}/models/tokenizer.json"
tok = Tokenizer.from_file(tok_path)
IM_END = ""
THINK_OPEN = ""
THINK_CLOSE = ""

def wrap_user(text: str) -> str:
    return (
        f"<|im_start|>user\n{text}{IM_END}\n"
        f"<|im_start|>assistant\n{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"
    )

def make_prompt(target_tokens: int) -> str:
    # Grow filler until wrapped prompt reaches target token count.
    filler = "The quick brown fox jumps over the lazy dog. "
    text = "bench"
    while True:
        wrapped = wrap_user(text)
        n = len(tok.encode(wrapped).ids)
        if n >= target_tokens:
            return text
        text += filler

def native_decode_tps(ctx: int) -> float:
    out = subprocess.check_output([BENCH, GGUF, str(DECODE_N), str(ctx)], text=True, stderr=subprocess.STDOUT)
    for line in out.splitlines():
        if "decode tg" in line:
            return float(line.split("decode tg")[1].split("tok/s")[0].strip().lstrip(":").strip())
    raise RuntimeError(f"native bench parse failed for ctx={ctx}\n{out}")

def api_bench(ctx: int):
    prompt = make_prompt(ctx)
    body = json.dumps({
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": DECODE_N,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{API}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if obj.get("error"):
                raise RuntimeError(obj["error"])
            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta:
                chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
    wall = time.perf_counter() - t0
    decode_s = max(wall - (ttft or 0), 1e-6)
    decode_tps = chunks / decode_s if chunks else 0.0
    return {
        "prompt_tokens": len(tok.encode(wrap_user(prompt)).ids),
        "ttft_s": ttft or 0.0,
        "decode_tokens": chunks,
        "decode_tps": decode_tps,
        "wall_s": wall,
    }

print(f"model: {GGUF}")
print(f"api: {API}")
print(f"decode_n={DECODE_N}")
print()
print(f"{'ctx':>6}  {'native tok/s':>12}  {'api tok/s':>12}  {'api/native':>10}  {'ttft(s)':>8}  {'prompt tok':>10}")
print("-" * 70)

for ctx in CONTEXTS:
    native = native_decode_tps(ctx)
    # Stop API server during native bench to avoid VRAM contention
    subprocess.run(["bash", "-c", "kill $(cat /tmp/sparkinfer_server.pid) 2>/dev/null || true"], check=False)
    time.sleep(2)
    api = api_bench(ctx)
  # restart server between API runs if killed - caller should manage; rerun server after native
    print(
        f"{ctx:6d}  {native:12.2f}  {api['decode_tps']:12.2f}  {api['decode_tps']/native*100:9.1f}%  "
        f"{api['ttft_s']:8.2f}  {api['prompt_tokens']:10d}"
    )
PY
"$ROOT" "$GGUF" "$API" "$BENCH" "$DECODE_N" "${CONTEXTS[*]}"
