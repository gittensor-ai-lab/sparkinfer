#!/usr/bin/env python3
"""Tokenize / detokenize for sparkinfer-server (Jan/OpenAI chat payloads).

  encode: read JSON {"messages":[...]} from stdin -> {"ids":[...]}
  decode: read JSON {"ids":[...]} from stdin -> {"text":"..."}

Requires: pip install tokenizers
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

DEFAULT_TOK_URL = "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main/tokenizer.json"


def load_tokenizer(path: str):
    from tokenizers import Tokenizer
    if not os.path.isfile(path):
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        url = os.environ.get("SPARKINFER_TOKENIZER_URL", DEFAULT_TOK_URL)
        print(f"downloading tokenizer -> {path}", file=sys.stderr)
        urllib.request.urlretrieve(url, path)
    return Tokenizer.from_file(path)


def apply_chat_template(tok, messages):
    # Qwen3 chat template (thinking disabled) — matches bench/quality/run_quality.py
    im_end = "<|im_end|>"
    think_open = "<think>"
    think_close = "</think>"
    parts = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}{im_end}\n")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}{im_end}\n")
        else:
            parts.append(f"<|im_start|>{role}\n{content}{im_end}\n")
    parts.append(f"<|im_start|>assistant\n{think_open}\n\n{think_close}\n\n")
    return "".join(parts)


def cmd_encode(tok_path: str):
    body = json.load(sys.stdin)
    messages = body.get("messages") or []
    if not messages and body.get("prompt"):
        text = body["prompt"]
    else:
        text = apply_chat_template(load_tokenizer(tok_path), messages)
    ids = load_tokenizer(tok_path).encode(text).ids
    json.dump({"ids": ids, "prompt_chars": len(text)}, sys.stdout)


def cmd_decode(tok_path: str):
    body = json.load(sys.stdin)
    ids = body.get("ids") or []
    text = load_tokenizer(tok_path).decode(ids)
    json.dump({"text": text}, sys.stdout)


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    op, tok_path = sys.argv[1], sys.argv[2]
    if op == "encode":
        cmd_encode(tok_path)
    elif op == "decode":
        cmd_decode(tok_path)
    else:
        print(f"unknown op {op!r}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
