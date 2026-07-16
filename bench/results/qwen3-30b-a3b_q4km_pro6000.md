# Qwen3-30B-A3B Q4_K_M — sparkinfer vs llama.cpp

**Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 96 GB, CUDA 12.8)
**Model:** Qwen3-30B-A3B, Q4_K_M GGUF (17.3 GiB), 48 layers, 128 experts top-8
**Setting:** single stream, batch = 1, greedy decode

| Engine | Decode (tg) | VRAM | gap to llama.cpp |
|---|---:|---:|---:|
| llama.cpp (CUDA) | **240.5 tok/s** | 17.3 GB | 1.0× |
| sparkinfer — pass 5 (current) | **133 tok/s** | 21.7 GB | **1.8×** |
| sparkinfer — pass 4 (vec GEMV) | 119 tok/s | 21.7 GB | 2.0× |
| sparkinfer — pass 3 (CUDA graph) | 118.7 tok/s | 21.7 GB | 2.0× |
| sparkinfer — pass 2 (decode GEMV) | 84.4 tok/s | 21.7 GB | 2.85× |
| sparkinfer — pass 1 (fused experts) | 32.7 tok/s | 21.7 GB | 7.3× |
| sparkinfer — baseline | 0.60 tok/s | 23.3 GB | 400× |

**5 optimization passes: 0.60 → 133 tok/s (≈220×), gap to llama.cpp 400× → 1.8×.**

Passes: (1) fused selected-expert quantized GEMV — dequant only the 8 routed
experts on-read; (2) decode GEMV for dense projections (coalesced [out,in],
replacing M=1 tiled GEMM); (3) CUDA-graph the whole decode step (capture once,
replay per token); (4) 128-bit vectorized GEMV loads; (5) flash-decoding
(KV-split) attention — one block per (seq, q_head, split) + combine, for SM
occupancy at decode and long-context scaling.

### Profile of the remaining 7.5 ms/token (ablation, no NCU)
| Component | Cost | Share |
|---|---:|---:|
| Attention block (QKV/O GEMV + RoPE + KV-append + flash-decode) | 2.5 ms | 33% |
| LM head + argmax | 0.6 ms | 8% |
| MoE (router + fused experts) | 0.6 ms | 8% |
| Long tail (rmsnorm/residual/rope/kv-append, ~770 kernels/token) | ~3.8 ms | 51% |

At batch-1 every kernel is tiny and **latency-bound**.

### Pass 6 + plateau finding
Fusing residual+RMSNorm (pass 6) gave only 133 → 134 tok/s (~1%), and vectorizing
GEMV loads (pass 4) gave ~0. Together these show the remaining gap is **not** the
norms/residuals or memory bandwidth — it's the **latency-bound GEMVs** at bs=1
(Q/K/V/O projections dominate the attention block). We've hit that floor at ~1.8×.

Closing it further means either deep GEMV micro-tuning (matching llama.cpp's
`mul_mat_vec` — high effort, diminishing returns) or a different regime:
- **long-context decode** — flash-decoding (pass 5) scales with KV length across
  many blocks, so the gap should *narrow* at long context (the project's real
  target: edge long-context MoE). Worth benchmarking at 8K–32K.
- **prefill + tensor cores** — bs>1 / prompt processing is where MMA matters; the
  current prefill is token-by-token.

**Bottom line: 0.60 → 134 tok/s (223×) in 6 passes, within 1.8× of llama.cpp on
single-stream short-context decode, hand-written kernels, verified correct.**

(historical first-pass note below.)

**Optimization pass 1: 0.60 → 32.7 tok/s (~55×), gap to llama.cpp 400× → 7.3×.**

The win came from the fused selected-expert kernel (`expert_ffn_q4k.cu`): dequantize
only the 8 routed experts (not all 128) on-read inside a warp-per-output-row GEMV,
with per-tensor Q4_K/Q6_K dispatch (Q4_K_M mixes the two per layer). Output is
correct ("The capital of France is Paris.").

The remaining ~7× to llama.cpp is the next pass:

## Optimization roadmap

1. ✅ **DONE — Fused selected-expert GEMV.** Dequantize only the 8 routed experts
   (not all 128), on-read inside a warp-per-row GEMV. Was ~16× wasted dequant +
   ~16 GB redundant traffic/token; now gone. **0.60 → 32.7 tok/s.** Experts are
   ~90% of FLOPs, so this was the single biggest lever.

2. **No tensor cores.** Our GEMV/GEMM are scalar CUDA-core kernels. For decode (bs=1)
   the experts are memory-bound so this matters less there, but the dense
   projections (Q/K/V/O, router, LM head) still run as tiled GEMM with M=1 (very
   inefficient). Fix: a proper GEMV for the dense bf16 projections; dp4a/MMA for
   prefill.

3. **Dense projections via M=1 tiled GEMM.** The Q/K/V/O/router/LM-head GEMMs use a
   16×16 tiled GEMM with one row → ~16× wasted threads. A dedicated GEMV would help.

4. **Per-token host sync, no CUDA graph.** `forward_token` syncs the stream and
   copies the argmax to host every token; ~48 layers × many tiny kernel launches,
   uncaptured. Fix: CUDA-graph the decode step, keep sampling on-device (the
   runtime is already sync-free up to the final argmax).

5. **Naive paged attention** (scalar gqa8, re-reads KV each step).

The #1 fix alone (fused selected-expert quantized GEMV) attacks both the 16×
dequant waste *and* the tensor-core gap on the dominant compute.

## Notes
- Correctness is established separately: sparkinfer produces the same answers as
  the model ("The capital of France is Paris." + EOS); this run measures speed only.
- VRAM: sparkinfer is higher (23.3 vs 17.3 GB) because dense weights are stored
  bf16 (~3 GB extra), plus the per-layer expert dequant scratch (~1.6 GB) and a
  bf16 KV cache. The fused path removes the scratch and can quantize dense too.

## Reproduce
```bash
scripts/bench_vs_llamacpp.sh \
  Qwen3-30B-A3B-Q4_K_M.gguf \
  /path/to/llama.cpp/build/bin/llama-bench \
  /path/to/sparkinfer/build/qwen3_bench
```
