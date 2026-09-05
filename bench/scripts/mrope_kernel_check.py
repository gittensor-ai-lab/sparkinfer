#!/usr/bin/env python3
"""Diff the CUDA prefill RoPE kernel's rotated Q against transformers' OWN rotary embedding.

mrope_ref_check.py proves we COMPUTE the reference's position ids. This proves the kernel APPLIES
them the way the reference does -- which is where the from-scratch logic lives (the interleaved
axis selection, pf_mrope_axis). The two checks are not redundant: correct positions applied to the
wrong frequency bands would pass the first and fail this one.

Uses transformers' Qwen3_5TextRotaryEmbedding (its real apply_interleaved_mrope and cos/sin) plus
the standard NeoX rotate-half, and reproduces only the surrounding RMSNorm, which is ordinary and
not what is under test.
"""
import argparse, json, sys, types

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as M


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--tol", type=float, default=None,
                    help="absolute tolerance; default is derived from the measured bf16 floor")
    args = ap.parse_args()
    d = json.load(open(args.dump))

    N, HQ, HD = d["n_tokens"], d["n_q_heads"], d["head_dim"]
    rot, theta, eps = d["rotary_dim"], d["theta"], d["eps"]

    q_in = torch.tensor(d["q_in"], dtype=torch.float32).view(N, HQ, HD)
    qw = torch.tensor(d["q_norm_w"], dtype=torch.float32)
    got = torch.tensor(d["q_out"], dtype=torch.float32).view(N, HQ, HD)
    pos = torch.tensor(d["positions"], dtype=torch.long).view(N, 3).T.unsqueeze(1)  # (3,1,N)

    # Per-head RMSNorm, then bf16-rounded exactly as the kernel stores it.
    var = q_in.pow(2).mean(-1, keepdim=True)
    xn = (q_in * torch.rsqrt(var + eps)) * qw
    xn = xn.to(torch.bfloat16).to(torch.float32)

    # transformers' own rotary: builds inv_freq from the config, applies interleaved MRoPE, and
    # returns cos/sin. Driven through the real class, not reimplemented.
    cfg = types.SimpleNamespace(
        rope_parameters={"rope_type": "default", "rope_theta": theta,
                         "partial_rotary_factor": rot / HD,
                         "mrope_section": d["mrope_section"], "mrope_interleaved": True},
        head_dim=HD, hidden_size=HD, num_attention_heads=1,
        max_position_embeddings=262144)
    rope = M.Qwen3_5TextRotaryEmbedding(cfg)
    cos, sin = rope(torch.zeros(1, N, HD, dtype=torch.float32), pos)   # (1, N, rot)
    cos, sin = cos[0], sin[0]

    # NeoX rotate-half over the first rotary_dim dims only (partial rotary); the tail passes through.
    xr = xn[..., :rot]
    c = cos.unsqueeze(1)   # (N,1,rot)
    s = sin.unsqueeze(1)
    want = xn.clone()
    want[..., :rot] = xr * c + rotate_half(xr) * s

    diff = (got - want).abs()
    # The kernel stores bf16 and uses __cosf/__sinf, so agreement is bounded by bf16 resolution at
    # this magnitude, not by float equality. Derive the bar from the data instead of inventing one.
    floor = (want.to(torch.bfloat16).to(torch.float32) - want).abs().max().item()
    tol = args.tol if args.tol is not None else max(floor * 4, 2e-3)

    print(f"tokens={N} heads={HQ} head_dim={HD} rotary_dim={rot}")
    print(f"max|kernel - transformers| = {diff.max().item():.6g}")
    print(f"bf16 storage floor         = {floor:.6g}   tolerance = {tol:.6g}")

    # A pure pass/fail on a tolerance would also pass if MRoPE were silently ignored, because the
    # text tokens agree either way. Show that the vision rows genuinely differ from a 1D baseline.
    pos1d = pos[0].expand(3, -1, -1)
    cos1, sin1 = rope(torch.zeros(1, N, HD, dtype=torch.float32), pos1d)
    want1d = xn.clone()
    want1d[..., :rot] = xr * cos1[0].unsqueeze(1) + rotate_half(xr) * sin1[0].unsqueeze(1)
    sep = (want - want1d).abs().max().item()
    print(f"max|mrope - 1d baseline|   = {sep:.6g}   (must exceed the tolerance, or the test is vacuous)")

    ok = diff.max().item() <= tol
    if sep <= tol:
        print("\nFAILED -- the chosen positions do not distinguish MRoPE from 1D RoPE; test is vacuous")
        return 1
    print("\nPASSED -- kernel matches transformers" if ok else "\nFAILED -- kernel diverges from transformers")
    if not ok:
        bad = (diff > tol).nonzero()[:8]
        for b in bad:
            t, h, dd = b.tolist()
            print(f"  tok={t} head={h} dim={dd}  ours={got[t,h,dd]:.6f}  ref={want[t,h,dd]:.6f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
