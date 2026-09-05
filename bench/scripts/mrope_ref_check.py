#!/usr/bin/env python3
"""Diff runtime MRoPE position ids against transformers' OWN get_rope_index.

Deliberately calls the REAL method off Qwen3_5ForConditionalGeneration rather than
reimplementing it here. A hand-written Python reference is worth very little for this kind of
check: in the vision-tower work the local NumPy reference reproduced the very omission it was
meant to catch (both forgot rotary entirely) and agreed with the CUDA path to 0.99915 while both
were wrong. Only code we did not write can falsify code we did.

Reads the C++ side's output from a JSON file produced by mrope_positions_dump, and compares
elementwise. Exact integer equality is required -- these are position INDICES, so "close" is
meaningless.
"""
import argparse, json, sys, types

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as M


def reference_positions(token_ids, image_token_id, video_token_id, spans, merge, start_pos):
    """Drive transformers' own get_rope_index on a single sequence."""
    # get_rope_index needs only config.vision_config.spatial_merge_size off `self`, plus
    # mm_token_type_ids marking text(0)/image(1)/video(2). Build the smallest object that
    # satisfies that rather than instantiating a 27B model.
    cfg = types.SimpleNamespace(vision_config=types.SimpleNamespace(spatial_merge_size=merge))
    stub = types.SimpleNamespace(config=cfg)
    # Both methods hang off Qwen3_5Model, NOT Qwen3_5ForConditionalGeneration -- the generation
    # wrapper does not carry them.
    stub.get_vision_position_ids = types.MethodType(
        M.Qwen3_5Model.get_vision_position_ids, stub)

    ids = torch.tensor([token_ids], dtype=torch.long)
    mm = torch.zeros_like(ids)
    for i, t in enumerate(token_ids):
        if t == image_token_id:
            mm[0, i] = 1
        elif t == video_token_id:
            mm[0, i] = 2

    # image_grid_thw / video_grid_thw are (num, 3) in PRE-merge patch units.
    img = [s for s in spans if s["kind"] == "image"]
    vid = [s for s in spans if s["kind"] == "video"]
    image_grid_thw = torch.tensor([[s["t"], s["h"], s["w"]] for s in img], dtype=torch.long) if img else None
    video_grid_thw = torch.tensor([[s["t"], s["h"], s["w"]] for s in vid], dtype=torch.long) if vid else None

    pos, _ = M.Qwen3_5Model.get_rope_index(
        stub, ids, mm, image_grid_thw, video_grid_thw, None)
    # (3, 1, seq) -> [[t,h,w], ...]
    pos = pos[:, 0, :]
    return [[int(pos[0, i]), int(pos[1, i]), int(pos[2, i])] for i in range(pos.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="JSON written by mrope_positions_dump")
    args = ap.parse_args()
    d = json.load(open(args.dump))

    got = [d["positions"][i:i + 3] for i in range(0, len(d["positions"]), 3)]
    want = reference_positions(d["token_ids"], d["image_token_id"], d["video_token_id"],
                               d["spans"], d["spatial_merge"], d.get("start_pos", 0))

    if len(got) != len(want):
        print(f"FAIL: length {len(got)} vs reference {len(want)}")
        return 1

    bad = [(i, g, w) for i, (g, w) in enumerate(zip(got, want)) if g != w]
    print(f"tokens: {len(got)}   mismatches: {len(bad)}")
    for i, g, w in bad[:12]:
        print(f"  [{i}] tok={d['token_ids'][i]}  ours={g}  reference={w}")
    if bad:
        print("\nFAILED -- position ids differ from transformers")
        return 1
    print("\nPASSED -- position ids are elementwise identical to transformers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
