#!/usr/bin/env python3
"""PIL/HF reference for image preprocessing, to judge vision_preprocess_check's dump.

Written from transformers' Qwen2VLImageProcessor, not from sparkinfer's C++, for the same reason
vision_ref.py is: a differential written by mirroring the thing under test validates arithmetic,
not convention.

The ordering subtlety this exists to measure: HF resizes the **uint8** image through PIL and only
then rescales to [0,1] and normalizes. sparkinfer rescales to float first and resizes in float.
Resampling is linear, so the scaling itself commutes -- but PIL's uint8 path ROUNDS to integers
and CLAMPS bicubic's overshoot at high-contrast edges, and neither of those commutes. So this
reports our dump against both orders:

  hf_uint8  -- what transformers actually computes (the bar that matters)
  float32   -- PIL's own resampler run in float, isolating rounding/clamping from resampler bugs

If we match float32 closely but not hf_uint8, the resampler is correct and the gap is purely the
rescale-before-resize ordering. If we match neither, the resampler itself is wrong.

Usage: vision_preprocess_ref.py <ckpt_dir> <image> [--compare FILE]
"""
import argparse, json, math, os, sys
import numpy as np
from PIL import Image


def smart_resize(height, width, factor, min_pixels, max_pixels):
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    rnd = lambda n: int(round(n / factor)) * factor      # noqa: E731  (Python's banker's rounding)
    flr = lambda n: int(math.floor(n / factor)) * factor  # noqa: E731
    cel = lambda n: int(math.ceil(n / factor)) * factor   # noqa: E731
    h_bar, w_bar = max(factor, rnd(height)), max(factor, rnd(width))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar, w_bar = flr(height / beta), flr(width / beta)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar, w_bar = cel(height * beta), cel(width * beta)
    return h_bar, w_bar


def patchify(arr, gh, gw, C, T, P):
    """[C,H,W] -> [gh*gw, C*T*P*P], each patch laid out (C,T,P,P), still image repeated over T."""
    x = arr.reshape(C, gh, P, gw, P).transpose(1, 3, 0, 2, 4)      # [gh,gw,C,P,P]
    x = np.repeat(x[:, :, :, None], T, axis=3)                     # [gh,gw,C,T,P,P]
    return x.reshape(gh * gw, C * T * P * P).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("image")
    ap.add_argument("--compare", default="")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))["vision_config"]
    P, T, C = cfg["patch_size"], cfg["temporal_patch_size"], cfg["in_channels"]
    merge = cfg["spatial_merge_size"]
    factor = P * merge
    mean = np.array(cfg.get("image_mean", [0.5, 0.5, 0.5]), np.float32).reshape(3, 1, 1)
    std = np.array(cfg.get("image_std", [0.5, 0.5, 0.5]), np.float32).reshape(3, 1, 1)
    min_px, max_px = 65536, 16777216

    im = Image.open(a.image).convert("RGB")
    w, h = im.size
    h_bar, w_bar = smart_resize(h, w, factor, min_px, max_px)
    gh, gw = h_bar // P, w_bar // P
    print(f"image {w}x{h} -> resized {w_bar}x{h_bar} -> patch grid {gh}x{gw} = {gh*gw} patches")
    if (h_bar, w_bar) == (h, w):
        print("  NOTE: smart_resize is the IDENTITY here -- this image does not exercise the resampler")

    # (a) exactly what transformers does: PIL resize on uint8, then rescale, then normalize
    u8 = np.asarray(im.resize((w_bar, h_bar), Image.BICUBIC), dtype=np.uint8)
    hf = (u8.astype(np.float32).transpose(2, 0, 1) / 255.0 - mean) / std

    # (b) PIL's same resampler in float, per channel ('F' mode is single-channel), no round/clamp
    src = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    fl = np.stack([np.asarray(Image.fromarray(src[c], mode="F").resize((w_bar, h_bar), Image.BICUBIC),
                              dtype=np.float32) for c in range(C)])
    fl = (fl - mean) / std

    # (c) torchvision's bicubic: identical separable/antialias structure, but a = -0.75 rather
    # than PIL's -0.5. preprocessor_config.json declares Qwen2VLImageProcessorFast (torchvision),
    # while processor_config.json declares the slow PIL one, so which is "the" reference is
    # genuinely ambiguous -- this quantifies what the choice costs.
    def tv_resize(src_u8, oh, ow, a=-0.75):
        def cubic(x):
            x = np.abs(x)
            return np.where(x < 1, ((a + 2) * x - (a + 3)) * x * x + 1,
                   np.where(x < 2, (((x - 5) * x + 8) * x - 4) * a, 0.0))
        def axis(arr, out_len, ax):
            in_len = arr.shape[ax]
            sc = in_len / out_len
            fs = max(sc, 1.0)
            o = np.arange(out_len)
            ctr = (o + 0.5) * sc
            lo = np.maximum(0, np.floor(ctr - 2 * fs + 0.5).astype(int))
            hi = np.minimum(in_len, np.floor(ctr + 2 * fs + 0.5).astype(int))
            out = np.zeros(tuple(out_len if i == ax else d for i, d in enumerate(arr.shape)), np.float64)
            for k in range(out_len):
                idx = np.arange(lo[k], hi[k])
                w = cubic(((idx + 0.5) - ctr[k]) / fs)
                w = w / w.sum()
                sl = np.take(arr, idx, axis=ax)
                sh = [1] * arr.ndim; sh[ax] = len(idx)
                np.copyto(np.take(out, [k], axis=ax), 0)
                acc = (sl * w.reshape(sh)).sum(axis=ax)
                if ax == 0: out[k] = acc
                else: out[:, k] = acc
            return out
        x = axis(src_u8.astype(np.float64), ow, 1)
        x = axis(x, oh, 0)
        return np.clip(np.floor(x + 0.5), 0, 255)

    src_u8 = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)
    tv = np.stack([tv_resize(src_u8[c], h_bar, w_bar) for c in range(C)]).astype(np.float32)
    tv = (tv / 255.0 - mean) / std

    refs = {"hf_uint8": patchify(hf, gh, gw, C, T, P), "float32": patchify(fl, gh, gw, C, T, P),
            "tv_uint8": patchify(tv, gh, gw, C, T, P)}
    print(f"  hf_uint8 vs float32 differ by max {np.abs(refs['hf_uint8']-refs['float32']).max():.6f}"
          f"  (this is the rounding/clamping the orders disagree on)")

    if not a.compare:
        return
    got = np.fromfile(a.compare, dtype=np.float32)
    if got.size != refs["hf_uint8"].size:
        print(f"[FAIL] size mismatch: reference {refs['hf_uint8'].size}, candidate {got.size}")
        sys.exit(1)
    got = got.reshape(refs["hf_uint8"].shape)
    print(f"\ncompare vs {a.compare}")
    results = {}
    for name, ref in refs.items():
        d = np.abs(got - ref)
        den = max(float(np.abs(ref).max()), 1e-9)
        cos = float((got.ravel() @ ref.ravel()) / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
        results[name] = (float(d.max()), float(d.max()) / den, cos, float(d.mean()))
        print(f"  vs {name:9s} max|diff|={d.max():.6f}  rel={d.max()/den:.3e}  "
              f"mean|diff|={d.mean():.6f}  cosine={cos:.8f}")

    # The bar is agreement with what transformers ACTUALLY computes (hf_uint8), and it is tight
    # on purpose: an earlier version of this script passed a preprocessor that was off by 0.127,
    # because it judged on cosine, which barely moves when a few edge pixels are wrong. A gate
    # that passes before and after a real fix is not a gate.
    #
    # The irreducible floor is one uint8 LSB -- PIL accumulates its 8-bit passes in fixed point,
    # so exact .5 ties can break either way. Anything above that is a genuine divergence: wrong
    # rescale/resize order, a missing intermediate round, or the wrong cubic coefficient.
    lsb = (1.0 / 255.0) / float(min(std.ravel()))
    max_hf, _, cos_hf, mean_hf = results["hf_uint8"]
    print(f"  one uint8 LSB in normalized units = {lsb:.6f}")
    print()
    if max_hf <= 1.01 * lsb and mean_hf <= 0.01 * lsb:
        print(f"VERDICT: MATCH -- within one uint8 LSB of transformers "
              f"(max {max_hf:.6f} <= {lsb:.6f}, mean {mean_hf:.6f})")
    else:
        print(f"VERDICT: MISMATCH -- max {max_hf:.6f} vs 1 LSB {lsb:.6f}, mean {mean_hf:.6f}")
        print("  vs float32 closer than vs hf_uint8 => rescale/resize ORDER or intermediate "
              "rounding is wrong; vs tv_uint8 closer => cubic coefficient is wrong")
        sys.exit(1)


if __name__ == "__main__":
    main()
