#!/usr/bin/env python3
"""Sweep max_tokens and context to find optimal SparkInfer API settings."""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

IM_END = "<|" + "im_end|>"
THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"

MAX_TOKENS_SWEEP = [256, 512, 1024, 2048, 4096]
PROMPT_CTX_SWEEP = [512, 4096, 8192, 16384, 30000, 32768]
SERVER_CTX = 36864  # PRO 6000: 32k input + 4k output (see changelog-pro6000.md)


def wrap_user(text: str, thinking: bool) -> str:
    tail = "" if thinking else f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"
    return f"<|im_start|>user\n{text}{IM_END}\n<|im_start|>assistant\n{tail}"


def make_prompt(tok: Tokenizer, target_tokens: int, thinking: bool) -> tuple[str, int]:
    filler = "Analyze inference throughput and KV-cache behavior. "
    text = "bench"
    while len(tok.encode(wrap_user(text, thinking)).ids) < target_tokens:
        text += filler
    ids = tok.encode(wrap_user(text, thinking)).ids
    return text, len(ids)


@dataclass
class RunResult:
    max_tokens: int
    prompt_tokens: int
    thinking: bool
    ok: bool
    error: str
    out_tokens: int
    reasoning_tokens: int
    ttft_s: float
    wall_s: float
    decode_tps: float


def stream_bench(
    api: str,
    tok: Tokenizer,
    prompt_tokens_target: int,
    max_tokens: int,
    thinking: bool,
) -> RunResult:
    prompt, prompt_tokens = make_prompt(tok, prompt_tokens_target, thinking)
    body = json.dumps(
        {
            "model": "qwen3.6-35b-a3b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "enable_thinking": thinking,
        }
    ).encode()
    req = urllib.request.Request(
        f"{api.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    content_chunks = 0
    reasoning_chunks = 0
    err = ""
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                if "error" in obj:
                    err = str(obj["error"])
                    break
                delta = obj.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    content_chunks += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                if delta.get("reasoning_content"):
                    reasoning_chunks += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")[:200]
    except Exception as e:
        err = str(e)

    wall = time.perf_counter() - t0
    out_tokens = content_chunks + reasoning_chunks
    decode_s = max(wall - (ttft or 0.0), 1e-6)
    return RunResult(
        max_tokens=max_tokens,
        prompt_tokens=prompt_tokens,
        thinking=thinking,
        ok=not err,
        error=err,
        out_tokens=out_tokens,
        reasoning_tokens=reasoning_chunks,
        ttft_s=ttft or 0.0,
        wall_s=wall,
        decode_tps=out_tokens / decode_s if out_tokens else 0.0,
    )


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/sparkinfer-server")
    api = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8080"
    tok = Tokenizer.from_file(str(root / "models/tokenizer.json"))

    print(f"SparkInfer max_tokens sweep — server ctx={SERVER_CTX}")
    print(f"api: {api}")
    print()

    # 1) max_tokens sweep at moderate prompt
    print("=== max_tokens sweep (prompt~512, instruct mode) ===")
    print(f"{'max_tok':>8}  {'out':>6}  {'tps':>8}  {'wall_s':>8}  {'ttft_s':>8}  status")
    print("-" * 58)
    for mt in MAX_TOKENS_SWEEP:
        r = stream_bench(api, tok, 512, mt, thinking=False)
        status = "OK" if r.ok else f"ERR {r.error[:40]}"
        print(
            f"{mt:8d}  {r.out_tokens:6d}  {r.decode_tps:8.1f}  {r.wall_s:8.2f}  "
            f"{r.ttft_s:8.2f}  {status}"
        )

    print()
    print("=== max_tokens sweep (prompt~512, think mode) ===")
    print(f"{'max_tok':>8}  {'reason':>6}  {'answer':>6}  {'tps':>8}  {'wall_s':>8}  status")
    print("-" * 58)
    for mt in MAX_TOKENS_SWEEP:
        r = stream_bench(api, tok, 512, mt, thinking=True)
        answer = r.out_tokens - r.reasoning_tokens
        status = "OK" if r.ok else f"ERR {r.error[:40]}"
        print(
            f"{mt:8d}  {r.reasoning_tokens:6d}  {answer:6d}  {r.decode_tps:8.1f}  "
            f"{r.wall_s:8.2f}  {status}"
        )

    print()
    print("=== context pressure (max_tokens=4096, instruct) ===")
    print(f"{'prompt':>8}  {'headroom':>10}  {'out':>6}  {'tps':>8}  status")
    print("-" * 52)
    for pt in PROMPT_CTX_SWEEP:
        headroom = SERVER_CTX - pt
        if headroom < 256:
            print(f"{pt:8d}  {headroom:10d}  {'skip':>6}  {'—':>8}  skip (<256 headroom)")
            continue
        mt = min(4096, headroom)
        r = stream_bench(api, tok, pt, mt, thinking=False)
        status = "OK" if r.ok else f"ERR {r.error[:40]}"
        print(
            f"{pt:8d}  {headroom:10d}  {r.out_tokens:6d}  {r.decode_tps:8.1f}  {status}"
        )

    print()
    print("=== recommendations ===")
    print(f"- Server --ctx:        {SERVER_CTX}")
    print("- API max_tokens cap:  4096 (hardcoded in sparkinfer_server)")
    print("- Optimal instruct:    max_tokens=1024–2048 (speed vs length sweet spot)")
    print("- Optimal think mode:  max_tokens=2048–4096 (reasoning eats budget)")
    print("- Long chats:          32k input + 4k output needs --ctx 36864 (PRO 6000)")
    print("- Template overhead:   ~6 tokens; full 4k at 32k needs prompt <32760 user tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
