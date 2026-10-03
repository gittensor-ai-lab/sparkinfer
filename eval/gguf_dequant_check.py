#!/usr/bin/env python3
"""Check sparkinfer's GPU GGUF dequantizers against gguf-py's reference.

Picks the first tensor of each requested ggml type in a GGUF file, dequantizes it with
build/runtime/gguf_dequant_check, and compares the bf16 result with gguf.quants.dequantize
rounded to bf16. Pass when every element is within one bf16 ulp of the reference.

  GGUF_PY=/path/to/llama.cpp/gguf-py python3 eval/gguf_dequant_check.py model.gguf \
      [--bin build/runtime/gguf_dequant_check] [--types Q3_K,IQ4_NL,IQ3_S,IQ4_XS]

Every requested type must be present in the file; a missing one fails the check.
"""
import argparse, os, subprocess, sys, tempfile

if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])
import numpy as np
import gguf
from gguf.quants import dequantize


def to_bf16_bits(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)   # round to nearest even


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--bin", default="build/runtime/gguf_dequant_check")
    ap.add_argument("--types", default="Q3_K,IQ4_NL,IQ3_S,IQ4_XS,Q4_K,Q5_K,Q6_K,Q8_0")
    a = ap.parse_args()
    want = a.types.split(",")
    r = gguf.GGUFReader(a.gguf)
    picked = {}
    for t in r.tensors:
        name = t.tensor_type.name
        if name in want and name not in picked:
            picked[name] = t
    missing = [w for w in want if w not in picked]
    if missing:
        # A type that was asked for and not checked is a failure, not a skip: otherwise a typo in
        # --types or the wrong file checks nothing and still exits 0.
        print("FAIL not in this file:", ",".join(missing))
    ok = not missing and bool(picked)
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([a.bin, a.gguf, d] + [t.name for t in picked.values()], check=True)
        for ty, t in picked.items():
            ref = to_bf16_bits(dequantize(t.data, t.tensor_type).reshape(-1))
            got = np.fromfile(os.path.join(d, t.name + ".bf16"), dtype=np.uint16)
            if got.size != ref.size:
                print(f"FAIL {ty:7s} {t.name}: size {got.size} vs {ref.size}")
                ok = False
                continue
            # bf16 bit patterns of same-sign values are monotone, so the ulp distance is a difference
            gi = got.astype(np.int32)
            ri = ref.astype(np.int32)
            ulp = np.abs(gi - ri)
            sign_flip = ((gi ^ ri) & 0x8000) != 0
            zero = ((gi & 0x7FFF) == 0) & ((ri & 0x7FFF) == 0)
            bad = (ulp > 1) & ~zero | (sign_flip & ~zero & (ulp > 1))
            n_bad = int(bad.sum())
            print(f"{'ok  ' if n_bad == 0 else 'FAIL'} {ty:7s} {t.name:28s} values {ref.size:>10d} "
                  f"exact {float((ulp == 0).mean()) * 100:6.2f}%  >1ulp {n_bad}")
            ok &= n_bad == 0
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
