#!/usr/bin/env python3
"""API vs native sparkinfer decode benchmark through 16k context.

Uses tokenizers only to size prompts (harness). Serving path is C++ sparkinfer_server.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

IM_END = "<|" + "im_end|>"
THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"
DECODE_N = 128
CONTEXTS = [128, 512, 4096, 16384, 32768]


def wrap_user(text: str) -> str:
    return (
        f"<|im_start|>user\n{text}{IM_END}\n"
        f"<|im_start|>assistant\n{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"
    )


def make_prompt(tok: Tokenizer, target_tokens: int) -> str:
    filler = "The quick brown fox jumps over the lazy dog. "
    text = "bench"
    while len(tok.encode(wrap_user(text)).ids) < target_tokens:
        text += filler
    return text


def native_decode_tps(bench: Path, gguf: Path, ctx: int) -> float:
    out = subprocess.check_output(
        [str(bench), str(gguf), str(DECODE_N), str(ctx)],
        text=True,
        stderr=subprocess.STDOUT,
    )
    for line in out.splitlines():
        if "decode tg" in line:
            return float(line.split("decode tg", 1)[1].split("tok/s", 1)[0].strip().lstrip(":"))
    raise RuntimeError(f"failed to parse native bench ctx={ctx}\n{out[-2000:]}")


def api_stream_bench(api: str, tok: Tokenizer, ctx: int) -> dict:
    prompt = make_prompt(tok, ctx)
    prompt_tokens = len(tok.encode(wrap_user(prompt)).ids)
    body = json.dumps(
        {
            "model": "qwen3.6-35b-a3b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": DECODE_N,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{api}/v1/chat/completions",
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
            if "error" in obj:
                raise RuntimeError(obj["error"])
            delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta:
                chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - t0
    wall = time.perf_counter() - t0
    decode_s = max(wall - (ttft or 0.0), 1e-6)
    return {
        "prompt_tokens": prompt_tokens,
        "ttft_s": ttft or 0.0,
        "decode_tokens": chunks,
        "decode_tps": chunks / decode_s,
        "wall_s": wall,
    }


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--api-only"]
    api_only = "--api-only" in sys.argv
    root = Path(args[0]) if args else Path("/workspace/sparkinfer-server")
    api = args[1] if len(args) > 1 else "http://127.0.0.1:8080"
    gguf = root / "models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    bench = root / "build/runtime/qwen3_gguf_bench"
    tok = Tokenizer.from_file(str(root / "models/tokenizer.json"))

    print(f"model: {gguf}")
    print(f"api:   {api}")
    print(f"decode_n={DECODE_N}")
    print()

    native: dict[int, float] = {}
    if not api_only:
        print(">> native sparkinfer (qwen3_gguf_bench, bs=1)")
        for ctx in CONTEXTS:
            native[ctx] = native_decode_tps(bench, gguf, ctx)
            print(f"  ctx={ctx:5d}  {native[ctx]:.2f} tok/s")
        print()

    if api_only:
        # Paste prior native numbers or re-measure; default from last PRO 6000 run.
        native = {
            128: 436.18,
            512: 430.09,
            4096: 416.01,
            16384: 412.83,
            32768: 393.15,
        }

    # Ensure API is reachable
    try:
        urllib.request.urlopen(f"{api}/health", timeout=5)
    except Exception as e:
        print(f"!! API not reachable at {api}: {e}", file=sys.stderr)
        return 1

    print(">> API sparkinfer_server (C++ tokenizer, stream=true)")
    api_rows: dict[int, dict] = {}
    for ctx in CONTEXTS:
        api_rows[ctx] = api_stream_bench(api, tok, ctx)
        r = api_rows[ctx]
        print(
            f"  ctx={ctx:5d}  {r['decode_tps']:.2f} tok/s  "
            f"ttft={r['ttft_s']:.2f}s  prompt={r['prompt_tokens']} tok"
        )

    print()
    print(f"{'ctx':>6}  {'native':>10}  {'api':>10}  {'api/native':>10}  {'overhead':>10}  {'ttft(s)':>8}")
    print("-" * 62)
    for ctx in CONTEXTS:
        n = native[ctx]
        a = api_rows[ctx]["decode_tps"]
        oh = (1.0 - a / n) * 100.0 if n > 0 else 0.0
        print(
            f"{ctx:6d}  {n:10.2f}  {a:10.2f}  {a/n*100:9.1f}%  {oh:9.1f}%  "
            f"{api_rows[ctx]['ttft_s']:8.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
