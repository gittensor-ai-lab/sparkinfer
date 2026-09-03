#!/usr/bin/env python3
"""NumPy reference for Qwen3.8-27B's vision tower, read straight from checkpoint bytes.

This is GROUND TRUTH for the CUDA implementation, so it is deliberately written from the
architecture (config.json + HF's Qwen3-VL definition) rather than by mirroring sparkinfer's C++.
A differential written by copying the thing under test validates arithmetic, not convention --
that mistake cost real time on the DSpark draft, where a q-only YaRN scale matched our own code
and was wrong against HF.

No torch, no safetensors package: just mmap + the flat JSON header + numpy.

Usage: vision_ref.py <checkpoint_dir> [--h 64 --w 64] [--dump out.npz]
"""
import argparse, glob, json, mmap, os, struct, sys
import numpy as np

# ---------- checkpoint reader ----------
class ST:
    def __init__(self, d):
        self.maps, self.index = [], {}
        for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            fh = open(f, "rb")
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            base = 8 + n
            self.maps.append((fh, mm))
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                self.index[k] = (mm, base + v["data_offsets"][0], base + v["data_offsets"][1],
                                 v["dtype"], tuple(v["shape"]))

    def get(self, name):
        if name not in self.index:
            raise KeyError(name)
        mm, a, b, dt, shape = self.index[name]
        raw = mm[a:b]
        if dt == "BF16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            # bf16 -> f32 is a pure left shift into the high half; no table, no rounding.
            out = (u16.astype(np.uint32) << 16).view(np.float32)
        elif dt == "F32":
            out = np.frombuffer(raw, dtype=np.float32)
        elif dt == "F16":
            out = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        else:
            raise ValueError(f"{name}: unhandled dtype {dt}")
        return out.reshape(shape)

# ---------- ops ----------
# eps is NOT in the checkpoint's vision_config. It does not matter: measured on the released
# weights, 1e-6 vs torch's 1e-5 default differ by rel 1.5e-05 / cosine 0.9999999995 on the final
# merged embeddings -- far below what bf16 CUDA arithmetic will contribute. The norms carry both
# `weight` and `bias`, which settles the other half of the question: LayerNorm, not RMSNorm.
def layernorm(x, w, b, eps=1e-6):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b

def gelu_tanh(x):
    # gelu_pytorch_tanh: the tanh approximation, NOT the erf form. They differ by ~1e-3 at the
    # knee, which is far above the tolerance a 27-block tower needs.
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))

def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)

# ---------- tower ----------
def run_tower(st, cfg, pixels, grid_h, grid_w, trace=None):
    """pixels: [n_patches, in_ch * temporal * patch * patch] already patchified.
    Returns merged embeddings [n_patches // merge^2, out_hidden]."""
    H, heads = cfg["hidden_size"], cfg["num_heads"]
    hd = H // heads
    depth, inter = cfg["depth"], cfg["intermediate_size"]
    merge = cfg["spatial_merge_size"]

    # patch embed: the Conv3d is a dense matmul over each flattened patch.
    pw = st.get("model.visual.patch_embed.proj.weight")          # [H, C, T, P, P]
    pb = st.get("model.visual.patch_embed.proj.bias")            # [H]
    x = pixels @ pw.reshape(H, -1).T + pb                        # [N, H]
    if trace is not None: trace["after_patch_embed"] = x.copy()

    # learned position embedding, bilinearly resampled from the 48x48 table to this grid --
    # Qwen3-VL interpolates rather than truncating, so a non-48 grid is not a slice.
    pos = st.get("model.visual.pos_embed.weight")                # [2304, H]
    side = int(round(pos.shape[0] ** 0.5))
    pg = pos.reshape(side, side, H)
    ys = np.linspace(0, side - 1, grid_h); xs = np.linspace(0, side - 1, grid_w)
    y0 = np.clip(np.floor(ys).astype(int), 0, side - 1); y1 = np.clip(y0 + 1, 0, side - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, side - 1); x1 = np.clip(x0 + 1, 0, side - 1)
    wy = (ys - y0)[:, None, None]; wx = (xs - x0)[None, :, None]
    p = (pg[np.ix_(y0, x0)] * (1 - wy) * (1 - wx) + pg[np.ix_(y1, x0)] * wy * (1 - wx)
         + pg[np.ix_(y0, x1)] * (1 - wy) * wx + pg[np.ix_(y1, x1)] * wy * wx)
    x = x + p.reshape(grid_h * grid_w, H)
    if trace is not None: trace["after_pos_embed"] = x.copy()

    N = x.shape[0]
    for b in range(depth):
        pfx = f"model.visual.blocks.{b}."
        h = layernorm(x, st.get(pfx + "norm1.weight"), st.get(pfx + "norm1.bias"))
        qkv = h @ st.get(pfx + "attn.qkv.weight").T + st.get(pfx + "attn.qkv.bias")   # [N, 3H]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(N, heads, hd).transpose(1, 0, 2)
        k = k.reshape(N, heads, hd).transpose(1, 0, 2)
        v = v.reshape(N, heads, hd).transpose(1, 0, 2)
        # FULL bidirectional attention over all patches -- no causal mask. A causal mask here is
        # the classic silent bug: it still produces embeddings, just wrong ones.
        att = softmax((q @ k.transpose(0, 2, 1)) / np.sqrt(hd), axis=-1)
        o = (att @ v).transpose(1, 0, 2).reshape(N, H)
        x = x + o @ st.get(pfx + "attn.proj.weight").T + st.get(pfx + "attn.proj.bias")
        h = layernorm(x, st.get(pfx + "norm2.weight"), st.get(pfx + "norm2.bias"))
        h = gelu_tanh(h @ st.get(pfx + "mlp.linear_fc1.weight").T + st.get(pfx + "mlp.linear_fc1.bias"))
        x = x + h @ st.get(pfx + "mlp.linear_fc2.weight").T + st.get(pfx + "mlp.linear_fc2.bias")
        if trace is not None and b in (0, depth // 2, depth - 1):
            trace[f"after_block_{b}"] = x.copy()

    # merger: norm per patch, then group each 2x2 spatial block into one row.
    x = layernorm(x, st.get("model.visual.merger.norm.weight"), st.get("model.visual.merger.norm.bias"))
    gh, gw = grid_h // merge, grid_w // merge
    x = (x.reshape(gh, merge, gw, merge, H).transpose(0, 2, 1, 3, 4).reshape(gh * gw, merge * merge * H))
    x = gelu_tanh(x @ st.get("model.visual.merger.linear_fc1.weight").T
                  + st.get("model.visual.merger.linear_fc1.bias"))
    x = x @ st.get("model.visual.merger.linear_fc2.weight").T + st.get("model.visual.merger.linear_fc2.bias")
    if trace is not None: trace["merged"] = x.copy()
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--w", type=int, default=64); ap.add_argument("--dump", default="")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))["vision_config"]
    P, T, C, merge = cfg["patch_size"], cfg["temporal_patch_size"], cfg["in_channels"], cfg["spatial_merge_size"]
    gran = P * merge
    if a.h % gran or a.w % gran:
        sys.exit(f"image dims must be multiples of {gran}")
    gh, gw = a.h // P, a.w // P

    # Deterministic synthetic image in [-1, 1], so the C++ side can reproduce the exact input
    # without an image decoder existing yet. A smooth ramp, not noise: a wrong patch order or a
    # transposed grid shows up as structure, where noise would just look like noise.
    yy, xx = np.meshgrid(np.arange(a.h), np.arange(a.w), indexing="ij")
    img = np.stack([(yy / max(a.h - 1, 1)) * 2 - 1,
                    (xx / max(a.w - 1, 1)) * 2 - 1,
                    ((yy + xx) / max(a.h + a.w - 2, 1)) * 2 - 1]).astype(np.float32)  # [C,H,W]
    # patchify: [C,H,W] -> [gh*gw, C*T*P*P]; a still image repeats across the temporal axis.
    pt = img.reshape(C, gh, P, gw, P).transpose(1, 3, 0, 2, 4)           # [gh,gw,C,P,P]
    pt = np.repeat(pt[:, :, :, None], T, axis=3)                         # [gh,gw,C,T,P,P]
    pixels = pt.reshape(gh * gw, C * T * P * P).astype(np.float32)

    st = ST(a.model_dir)
    trace = {}
    out = run_tower(st, cfg, pixels, gh, gw, trace)
    print(f"image {a.h}x{a.w} -> patch grid {gh}x{gw} = {gh*gw} patches -> {out.shape[0]} merged embeddings")
    print(f"output shape {out.shape} (expect [{gh//merge*gw//merge}, {cfg['out_hidden_size']}])")
    for k, v in trace.items():
        print(f"  {k:22s} shape={str(v.shape):16s} mean={v.mean():+.6f} std={v.std():.6f} "
              f"absmax={np.abs(v).max():.4f} sum={v.sum():+.6f}")
    if a.dump:
        np.savez(a.dump, pixels=pixels, out=out, **trace)
        print(f"dumped -> {a.dump}")

if __name__ == "__main__":
    main()
