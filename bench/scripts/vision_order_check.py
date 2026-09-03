#!/usr/bin/env python3
"""Does patch ORDER change the merged embeddings?

vision_ref.py feeds patches row-major and regroups into 2x2 merge blocks at the merger.
HF's Qwen2VLImageProcessor feeds them merge-block-major from the start:

    patches.reshape(b, C, gh//m, m, P, gw//m, m, P).permute(0, 2, 5, 3, 6, 1, 4, 7)

Attention here is full and bidirectional, so it is permutation-EQUIVARIANT, and position
embeddings travel with their own patch in both schemes. The two orderings should therefore cancel
and produce identical merged rows. That is an argument, not a result -- so run both and diff.

If they disagree, vision_ref.py is emitting image regions in the wrong sequence: correctly shaped,
correctly normed, confidently wrong, and invisible to every check that is not this one.
"""
import importlib.util, json, sys
import numpy as np

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vr", os.path.join(_HERE, "vision_ref.py"))
vr = importlib.util.module_from_spec(spec); sys.modules["vr"] = vr; spec.loader.exec_module(vr)

D = sys.argv[1] if len(sys.argv) > 1 else "/workspace/eval/models/qwen38-target"
cfg = json.load(open(D + "/config.json"))["vision_config"]
P, T, C = cfg["patch_size"], cfg["temporal_patch_size"], cfg["in_channels"]
m, H = cfg["spatial_merge_size"], cfg["hidden_size"]
st = vr.ST(D)

h = w = 64
gh, gw = h // P, w // P                      # full patch grid
bh, bw = gh // m, gw // m                    # merge-block grid
yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
img = np.stack([(yy / (h - 1)) * 2 - 1, (xx / (w - 1)) * 2 - 1,
                ((yy + xx) / (h + w - 2)) * 2 - 1]).astype(np.float32)   # [C,H,W]

# --- A: row-major, exactly what vision_ref.py does ---
a = img.reshape(C, gh, P, gw, P).transpose(1, 3, 0, 2, 4)
a = np.repeat(a[:, :, :, None], T, axis=3).reshape(gh * gw, C * T * P * P).astype(np.float32)
out_a = vr.run_tower(st, cfg, a, gh, gw)

# --- B: HF's exact reshape+permute, merge-block-major ---
b = img.reshape(C, bh, m, P, bw, m, P).transpose(1, 4, 2, 5, 0, 3, 6)   # [bh,bw,m,m,C,P,P]
b = np.repeat(b[..., None, :, :], T, axis=-3) if False else b           # keep layout explicit
b = np.stack([b] * T, axis=-3).reshape(gh * gw, C * T * P * P).astype(np.float32) \
    if False else np.repeat(b.reshape(gh * gw, C, P, P)[:, :, None], T, axis=2).reshape(gh * gw, C * T * P * P).astype(np.float32)

# B needs its position embeddings in the SAME merge-block-major sequence. Build the row-major
# tower's pos grid, then permute it identically so each patch keeps its own spatial embedding.
perm = np.arange(gh * gw).reshape(bh, m, bw, m).transpose(0, 2, 1, 3).reshape(-1)  # rowmajor idx per B slot
def run_B():
    pos = st.get("model.visual.pos_embed.weight")
    side = int(round(pos.shape[0] ** 0.5))
    pg = pos.reshape(side, side, H)
    ys = np.linspace(0, side - 1, gh); xs = np.linspace(0, side - 1, gw)
    y0 = np.clip(np.floor(ys).astype(int), 0, side-1); y1 = np.clip(y0+1, 0, side-1)
    x0 = np.clip(np.floor(xs).astype(int), 0, side-1); x1 = np.clip(x0+1, 0, side-1)
    wy = (ys - y0)[:, None, None]; wx = (xs - x0)[None, :, None]
    p = (pg[np.ix_(y0,x0)]*(1-wy)*(1-wx) + pg[np.ix_(y1,x0)]*wy*(1-wx)
         + pg[np.ix_(y0,x1)]*(1-wy)*wx + pg[np.ix_(y1,x1)]*wy*wx).reshape(gh*gw, H)
    pw = st.get("model.visual.patch_embed.proj.weight"); pb = st.get("model.visual.patch_embed.proj.bias")
    x = b @ pw.reshape(H, -1).T + pb
    x = x + p[perm]                                   # each B slot gets ITS OWN spatial embedding
    N = x.shape[0]; heads = cfg["num_heads"]; hd = H // heads
    for i in range(cfg["depth"]):
        pfx = f"model.visual.blocks.{i}."
        hh = vr.layernorm(x, st.get(pfx+"norm1.weight"), st.get(pfx+"norm1.bias"))
        qkv = hh @ st.get(pfx+"attn.qkv.weight").T + st.get(pfx+"attn.qkv.bias")
        q,k,v = np.split(qkv,3,axis=-1)
        q=q.reshape(N,heads,hd).transpose(1,0,2); k=k.reshape(N,heads,hd).transpose(1,0,2); v=v.reshape(N,heads,hd).transpose(1,0,2)
        att = vr.softmax((q@k.transpose(0,2,1))/np.sqrt(hd), axis=-1)
        o=(att@v).transpose(1,0,2).reshape(N,H)
        x = x + o @ st.get(pfx+"attn.proj.weight").T + st.get(pfx+"attn.proj.bias")
        hh = vr.layernorm(x, st.get(pfx+"norm2.weight"), st.get(pfx+"norm2.bias"))
        hh = vr.gelu_tanh(hh @ st.get(pfx+"mlp.linear_fc1.weight").T + st.get(pfx+"mlp.linear_fc1.bias"))
        x = x + hh @ st.get(pfx+"mlp.linear_fc2.weight").T + st.get(pfx+"mlp.linear_fc2.bias")
    x = vr.layernorm(x, st.get("model.visual.merger.norm.weight"), st.get("model.visual.merger.norm.bias"))
    # already merge-block-major: consecutive m*m rows ARE one block, no transpose needed
    x = x.reshape(bh*bw, m*m*H)
    x = vr.gelu_tanh(x @ st.get("model.visual.merger.linear_fc1.weight").T + st.get("model.visual.merger.linear_fc1.bias"))
    return x @ st.get("model.visual.merger.linear_fc2.weight").T + st.get("model.visual.merger.linear_fc2.bias")

out_b = run_B()
d = np.abs(out_a - out_b)
den = max(np.abs(out_a).max(), 1e-9)
cos = float((out_a.ravel()@out_b.ravel())/(np.linalg.norm(out_a)*np.linalg.norm(out_b)))
print(f"A row-major (vision_ref) : {out_a.shape}  absmax={np.abs(out_a).max():.4f}")
print(f"B merge-block-major (HF) : {out_b.shape}  absmax={np.abs(out_b).max():.4f}")
print(f"max|A-B| = {d.max():.3e}   rel = {d.max()/den:.3e}   cosine = {cos:.10f}")
ok = d.max()/den < 1e-4
print("VERDICT:", "orderings AGREE -- vision_ref is correct" if ok
      else "orderings DISAGREE -- vision_ref emits regions in the wrong sequence")
sys.exit(0 if ok else 1)
