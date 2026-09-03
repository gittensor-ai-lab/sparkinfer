#!/usr/bin/env python3
"""Diff vision_resize_check's smart_resize output against the transformers original.

Exact integer logic on both sides, so this is a strict equality check -- no tolerance. An
off-by-one changes the patch grid, which changes how many image placeholder tokens the prompt
must carry, which silently misaligns the whole embedding splice.

Usage:  vision_resize_check <candidate.txt>
"""
import math, re, sys

def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Verbatim from transformers/models/qwen2_vl/image_processing_qwen2_vl.py."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

lines = open(sys.argv[1]).read().splitlines()
# Match the PARAMETER header specifically: the candidate also prints "# <check> ok" lines
# for the placeholder-expansion checks, and taking the first "#" line grabbed one of those.
hdr = next(l for l in lines if l.startswith("#") and "factor=" in l)
factor = int(re.search(r"factor=(\d+)", hdr).group(1))
minp = int(re.search(r"min_pixels=(\d+)", hdr).group(1))
maxp = int(re.search(r"max_pixels=(\d+)", hdr).group(1))

n = bad = 0
for l in lines:
    if l.startswith("#") or not l.strip():
        continue
    m = re.match(r"(\d+) (\d+) -> (ERR|(\d+) (\d+))", l)
    h, w = int(m.group(1)), int(m.group(2))
    try:
        want = smart_resize(h, w, factor, minp, maxp)
    except ValueError:
        want = None
    got = None if m.group(3) == "ERR" else (int(m.group(4)), int(m.group(5)))
    n += 1
    if got != want:
        bad += 1
        if bad <= 10:
            print(f"  MISMATCH {h}x{w}: candidate {got}, reference {want}")
print(f"{n} cases, {bad} mismatches")
print("VERDICT:", "MATCH -- smart_resize is a faithful port" if bad == 0 else "MISMATCH")
sys.exit(0 if bad == 0 else 1)
