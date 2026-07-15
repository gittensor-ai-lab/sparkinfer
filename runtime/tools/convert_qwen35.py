#!/usr/bin/env python3
"""Convert a Hugging Face Qwen3.5-MoE checkpoint to the sparkinfer weight format.

Output: one raw bf16 .bin per tensor in <out_dir>, with the names Qwen35Model::
load_weights expects. All matrices are written pre-transposed so the runtime's
row-major  C[M,N] = A[M,K] @ B[K,N]  GEMMs consume them directly.

HF stores Linear weights as [out, in] (y = W @ x). The runtime computes
y = x @ B, so every projection is transposed to [in, out] here.

Layout written per layer i (layer_i.<name>.bin):
  input_norm[H], wq[H, nq*hd], wk[H, nkv*hd], wv[H, nkv*hd], wo[nq*hd, H],
  q_norm[hd], k_norm[hd], post_attn_norm[H], router_w[H, E],
  gate[E, H, F], up[E, H, F], down[E, F, H],
  shared_gate[H, F], shared_up[H, F], shared_down[F, H]
plus embed_tokens[vocab, H], final_norm[H], lm_head[H, vocab].

Usage:
  pip install safetensors numpy
  python convert_qwen35.py /path/to/hf_qwen3.5_moe  ./qwen35_weights
"""
import sys, os, glob, json
import numpy as np
from safetensors import safe_open


def to_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 (round-to-nearest-even), returned as uint16."""
    f = x.astype(np.float32).view(np.uint32)
    # round to nearest even
    rounded = (f + 0x7FFF + ((f >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)

    # Index all tensors across shards.
    tensors = {}
    for st in sorted(glob.glob(os.path.join(src, "*.safetensors"))):
        with safe_open(st, framework="numpy") as f:
            for k in f.keys():
                tensors[k] = (st, k)

    cfg = json.load(open(os.path.join(src, "config.json")))
    n_layers = cfg["num_hidden_layers"]
    print(f"converting {n_layers} layers, {len(tensors)} tensors -> {out}")

    def get(name):
        st, k = tensors[name]
        with safe_open(st, framework="numpy") as f:
            return f.get_tensor(k).astype(np.float32)

    def write(name, arr):
        to_bf16(np.ascontiguousarray(arr)).tofile(os.path.join(out, name + ".bin"))

    # globals
    write("embed_tokens", get("model.embed_tokens.weight"))          # [vocab, H]
    write("final_norm",   get("model.norm.weight"))                  # [H]
    # Tied-embedding checkpoints ship no lm_head.weight; fall back to the input
    # embedding, which has the same [vocab, H] layout (as convert_gguf.py does).
    lm = "lm_head.weight" if "lm_head.weight" in tensors else "model.embed_tokens.weight"
    write("lm_head",      get(lm).T)                                 # [vocab,H] -> [H,vocab]

    for i in range(n_layers):
        p = f"model.layers.{i}."
        o = f"layer_{i}."
        write(o + "input_norm",     get(p + "input_layernorm.weight"))
        write(o + "wq", get(p + "self_attn.q_proj.weight").T)        # [nq*hd,H] -> [H,nq*hd]
        write(o + "wk", get(p + "self_attn.k_proj.weight").T)
        write(o + "wv", get(p + "self_attn.v_proj.weight").T)
        write(o + "wo", get(p + "self_attn.o_proj.weight").T)        # [H,nq*hd] -> [nq*hd,H]
        write(o + "q_norm", get(p + "self_attn.q_norm.weight"))
        write(o + "k_norm", get(p + "self_attn.k_norm.weight"))
        write(o + "post_attn_norm", get(p + "post_attention_layernorm.weight"))
        write(o + "router_w", get(p + "mlp.gate.weight").T)          # [E,H] -> [H,E]

        # routed experts: HF gate/up_proj [F,H], down_proj [H,F]
        E = cfg["num_experts"]
        gate = np.stack([get(p + f"mlp.experts.{e}.gate_proj.weight").T for e in range(E)])  # [E,H,F]
        up   = np.stack([get(p + f"mlp.experts.{e}.up_proj.weight").T   for e in range(E)])  # [E,H,F]
        down = np.stack([get(p + f"mlp.experts.{e}.down_proj.weight").T for e in range(E)])  # [E,F,H]
        write(o + "gate", gate); write(o + "up", up); write(o + "down", down)

        # shared expert (Qwen MoE naming may vary; adjust if absent)
        write(o + "shared_gate", get(p + "mlp.shared_expert.gate_proj.weight").T)  # [H,F]
        write(o + "shared_up",   get(p + "mlp.shared_expert.up_proj.weight").T)    # [H,F]
        write(o + "shared_down", get(p + "mlp.shared_expert.down_proj.weight").T)  # [F,H]
        print(f"  layer {i} done")

    print("done. point Qwen35Model::load_weights at", out)


if __name__ == "__main__":
    main()
