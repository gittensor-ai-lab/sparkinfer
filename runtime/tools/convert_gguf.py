#!/usr/bin/env python3
"""Convert a Qwen3-MoE GGUF (any quant: Q4_K_M, Q8_0, ...) to the sparkinfer
weight format used by Qwen35Model::load_weights.

The `gguf` package dequantizes every ggml type (Q4_K, Q6_K, ...) to float32 with
the reference implementation, so we don't need a C++ k-quant decoder. GGUF stores
linear weights as PyTorch [out, in] (same bytes as HF safetensors), so the same
transposes as convert_qwen35.py apply. Writes bf16 .bin files + a config.txt.

  pip install gguf numpy
  python convert_gguf.py model.gguf ./weights_dir
"""
import sys, os
import numpy as np
import gguf
import gguf.quants as Q

GT = gguf.GGUFValueType


def to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    r = gguf.GGUFReader(path)

    def meta(k):
        f = r.fields.get(k)
        if not f: return None
        if f.types[0] == GT.STRING: return bytes(f.parts[f.data[0]]).decode()
        return f.parts[f.data[-1]][0]

    A = "qwen3moe."
    arch = meta("general.architecture")
    assert arch == "qwen3moe", f"expected qwen3moe arch, got {arch}"
    cfg = dict(
        vocab=int(meta(A + "vocab_size") or 151936),
        hidden=int(meta(A + "embedding_length")),
        n_layers=int(meta(A + "block_count")),
        n_q_heads=int(meta(A + "attention.head_count")),
        n_kv_heads=int(meta(A + "attention.head_count_kv")),
        head_dim=int(meta(A + "attention.key_length")),
        n_experts=int(meta(A + "expert_count")),
        top_k=int(meta(A + "expert_used_count")),
        n_shared=0,
        moe_ffn=int(meta(A + "expert_feed_forward_length")),
        rope_theta=float(meta(A + "rope.freq_base")),
        rms_eps=float(meta(A + "attention.layer_norm_rms_epsilon")),
        eos_id=int(meta("tokenizer.ggml.eos_token_id") or 151645),
    )
    # vocab from token_embd if metadata missing
    ten = {t.name: t for t in r.tensors}
    if "token_embd.weight" in ten:
        cfg["vocab"] = int(ten["token_embd.weight"].shape[-1]) if False else cfg["vocab"]

    with open(os.path.join(out, "config.txt"), "w") as f:
        for k, v in cfg.items():
            f.write(f"{k}={v}\n")
    print("config:", cfg)

    def deq(name):
        t = ten[name]
        try:
            return Q.dequantize(t.data, t.tensor_type).astype(np.float32)
        except Exception:
            return np.asarray(t.data, dtype=np.float32)

    def write(name, arr):
        to_bf16(arr).tofile(os.path.join(out, name + ".bin"))

    # globals
    write("embed_tokens", deq("token_embd.weight"))          # [vocab, hidden] (as-is)
    write("final_norm",   deq("output_norm.weight"))
    lm = "output.weight" if "output.weight" in ten else "token_embd.weight"  # tied fallback
    write("lm_head", deq(lm).T)                               # [vocab,hidden] -> [hidden,vocab]

    for i in range(cfg["n_layers"]):
        b = f"blk.{i}."
        o = f"layer_{i}."
        write(o + "input_norm",     deq(b + "attn_norm.weight"))
        write(o + "wq", deq(b + "attn_q.weight").T)           # [out,in] -> [in,out]
        write(o + "wk", deq(b + "attn_k.weight").T)
        write(o + "wv", deq(b + "attn_v.weight").T)
        write(o + "wo", deq(b + "attn_output.weight").T)
        write(o + "q_norm", deq(b + "attn_q_norm.weight"))    # [head_dim]
        write(o + "k_norm", deq(b + "attn_k_norm.weight"))
        write(o + "post_attn_norm", deq(b + "ffn_norm.weight"))
        write(o + "router_w", deq(b + "ffn_gate_inp.weight").T)            # [E,H] -> [H,E]
        write(o + "gate", deq(b + "ffn_gate_exps.weight").transpose(0, 2, 1))  # [E,F,H]->[E,H,F]
        write(o + "up",   deq(b + "ffn_up_exps.weight").transpose(0, 2, 1))
        write(o + "down", deq(b + "ffn_down_exps.weight").transpose(0, 2, 1))  # [E,H,F]->[E,F,H]
        print(f"  layer {i}/{cfg['n_layers']}", end="\r", flush=True)
    print("\ndone ->", out)


if __name__ == "__main__":
    main()
