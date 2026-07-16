#!/usr/bin/env python3
"""Tokenize a prompt, run the sparkinfer Qwen3 GGUF generator, detokenize output.

  pip install tokenizers
  python run_qwen3.py <weight_dir> <binary> "<prompt>" [max_new] [--raw]

Downloads tokenizer.json from the Qwen3-30B-A3B base repo on first run. By default
wraps the prompt in the Qwen3 chat template (thinking disabled); --raw does a plain
completion instead.
"""
import sys, os, subprocess, urllib.request

TOK_URL = "https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/main/tokenizer.json"


def main():
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(1)
    wdir, binary, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
    max_new = int(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else 64
    raw = "--raw" in sys.argv

    tok_path = os.path.join(os.path.dirname(wdir.rstrip("/")) or ".", "tokenizer.json")
    if not os.path.exists(tok_path):
        print("downloading tokenizer.json ...")
        urllib.request.urlretrieve(TOK_URL, tok_path)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)

    if raw:
        text = prompt
    else:
        text = ("<|im_start|>user\n" + prompt + "<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    ids = tok.encode(text).ids
    print(f"prompt tokens ({len(ids)}):", ids[:32], "..." if len(ids) > 32 else "")

    cmd = [binary, wdir, str(max_new)] + [str(i) for i in ids]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout, end="")
    if res.returncode != 0:
        print("STDERR:", res.stderr[-2000:]); sys.exit(1)

    out_ids = []
    for line in res.stdout.splitlines():
        if line.startswith("OUTPUT_IDS:"):
            out_ids = [int(x) for x in line.split(":", 1)[1].split()]
    print("\n===== GENERATED TEXT =====")
    print(tok.decode(out_ids))


if __name__ == "__main__":
    main()
