#!/usr/bin/env python3
"""Diff our CUDA vision tower against transformers' own Qwen3_5VisionModel.

This exists because bench/scripts/vision_ref.py is NOT independent enough. It is a re-derivation
written by hand from the architecture, and a re-derivation can miss exactly what the C++ missed --
which is what happened: both omitted the vision tower's rotary position embeddings, so they agreed
to cosine 0.999 while both being wrong. Only the real implementation can catch that class of error.

Both sides are fed the SAME pixel tensor, so this tests the tower alone. Ours consumes row-major
patch order; transformers consumes merge-block-major, so the pixels are permuted for it here.

Usage: vision_hf_check.py <ckpt_dir> --pixels ROWMAJOR.bin --gh GH --gw GW [--compare OURS.bin]
"""
import argparse, glob, os, sys
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("--pixels", required=True)
    ap.add_argument("--gh", type=int, required=True); ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--compare", default=""); ap.add_argument("--dump", default="")
    a = ap.parse_args()

    cfg = AutoConfig.from_pretrained(a.model_dir).vision_config
    m = cfg.spatial_merge_size
    gh, gw = a.gh, a.gw
    patch_in = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size ** 2

    pix = np.fromfile(a.pixels, dtype=np.float32).reshape(gh * gw, patch_in)

    # row-major -> merge-block-major. perm[k] is the row-major index of the k-th patch in the
    # order transformers expects: blocks in raster order, and within a block its m*m patches.
    perm = np.arange(gh * gw).reshape(gh // m, m, gw // m, m).transpose(0, 2, 1, 3).reshape(-1)
    pix_mbm = pix[perm]

    # float32 throughout: the point is to isolate an ALGORITHMIC difference, and bf16 noise on a
    # 27-block tower is large enough (cosine ~0.9993 at 2304 patches) to hide or mimic one.
    model = Qwen3_5VisionModel._from_config(cfg).to(torch.float32).eval()
    sd = {}
    for f in sorted(glob.glob(os.path.join(a.model_dir, "*.safetensors"))):
        for k, v in load_file(f).items():
            if k.startswith("model.visual."):
                sd[k[len("model.visual."):]] = v.to(torch.float32)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "inv_freq" not in k]   # buffers, recomputed not stored
    print(f"loaded vision tower: {len(sd)} tensors, missing={missing[:4]} unexpected={unexpected[:4]}")
    if missing:
        print("[FAIL] missing required weights"); sys.exit(1)

    grid_t = torch.tensor([[1, gh, gw]], dtype=torch.long)
    with torch.no_grad():
        out = model(torch.from_numpy(pix_mbm.copy()), grid_t).pooler_output
    ref = out.float().numpy()

    # Run the SAME reference in bf16 to measure what precision alone costs at THIS size, instead
    # of hardcoding a floor. Attention error accumulates with sequence length, so a bar that is
    # right at 784 patches is wrong at 6144 -- and a fixed bar would either pass a real bug at
    # large grids or fail correct code. The floor is measured per invocation for that reason.
    with torch.no_grad():
        ref_bf16 = model.to(torch.bfloat16)(
            torch.from_numpy(pix_mbm.copy()).to(torch.bfloat16), grid_t).pooler_output.float().numpy()
    model.to(torch.float32)
    floor = float((ref_bf16.ravel() @ ref.ravel()) /
                  (np.linalg.norm(ref_bf16) * np.linalg.norm(ref) + 1e-30))
    print(f"  bf16 noise floor at this size: cosine={floor:.8f}  "
          f"(reference against itself in bf16 -- our best possible)")
    print(f"transformers: grid {gh}x{gw} -> {ref.shape[0]} merged embeddings of {ref.shape[1]}")
    print(f"  mean={ref.mean():+.6f} std={ref.std():.6f} absmax={np.abs(ref).max():.4f}")
    if a.dump:
        ref.astype(np.float32).tofile(a.dump); print(f"  dumped -> {a.dump}")

    if not a.compare:
        return
    got = np.fromfile(a.compare, dtype=np.float32)
    if got.size != ref.size:
        print(f"[FAIL] size mismatch: reference {ref.size}, candidate {got.size}"); sys.exit(1)
    got = got.reshape(ref.shape)
    d = np.abs(got - ref)
    cos = float((got.ravel() @ ref.ravel()) / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
    print(f"\ncompare vs {a.compare}")
    print(f"  max|diff|={d.max():.6f}  mean|diff|={d.mean():.6f}  rel={d.max()/max(np.abs(ref).max(),1e-9):.3e}")
    print(f"  cosine={cos:.8f}   (candidate absmax {np.abs(got).max():.4f} vs reference {np.abs(ref).max():.4f})")
    # Judged against the MEASURED floor, not a guessed constant: we compute in bf16, so agreeing
    # with an fp32 reference better than bf16 itself does is not possible. The margin covers our
    # kernels rounding at different points than torch does, not a different algorithm.
    print()
    if cos >= floor - 0.002:
        print(f"VERDICT: MATCH -- cosine {cos:.6f} is at the bf16 floor {floor:.6f}")
    else:
        print(f"VERDICT: MISMATCH -- cosine {cos:.6f} vs bf16 floor {floor:.6f}; "
              f"the gap ({floor - cos:.6f}) is algorithmic, not precision")
        sys.exit(1)


if __name__ == "__main__":
    main()
