# sparkinfer-runtime

Edge AI inference runtime for **NVIDIA RTX Spark** and RTX 5090-class GPUs.

Part of [gittensor-ai-lab](https://github.com/orgs/gittensor-ai-lab) — SN74 decentralized kernel and runtime optimization network.

---

## Primary Target: NVIDIA RTX Spark

RTX Spark is NVIDIA's Blackwell-based AI superchip: 20-core ARM CPU + Blackwell GPU + **128 GB unified LPDDR5X memory** in a single package. It runs 120B-parameter models locally and delivers ~1 PFLOP of AI compute — making it the first consumer-grade platform where datacenter-scale MoE inference is native.

- [NVIDIA RTX Spark × MediaTek — Architecture Overview](https://www.mediatek.com/products/personal-computing/nvidia-rtx-spark)
- [RTX Spark laptops launching Fall 2026 — Tom's Guide](https://www.tomsguide.com/computing/gaming-laptops/all-8-laptops-launching-with-nvidia-rtx-spark-this-fall-and-what-they-can-do)
- [NVIDIA RTX AI Garage: Local Agents on RTX Spark](https://blogs.nvidia.com/blog/rtx-ai-garage-computex-spark-local-agents/)

**Why RTX Spark changes the runtime problem:**

| Before (cloud/datacenter era) | RTX Spark era |
|---|---|
| Optimize for TP=8, batch=1024 | Optimize for single-node, batch=1–64 |
| Memory bandwidth = HBM (3+ TB/s) | Memory bandwidth = LPDDR5X (~273 GB/s) |
| Expert eviction = rare | Expert scheduling = critical |
| KV cache = secondary concern | KV cache = primary memory pressure |
| vLLM / SGLang sufficient | New runtime layer required |

---

## Hardware Targets

| Device | Memory | Bandwidth | Arch | Status |
|---|---|---|---|---|
| **RTX Spark** (GB10) | 128 GB unified | ~273 GB/s | sm_121 | **Primary** |
| RTX PRO 6000 (GB202) | 96 GB GDDR7 ECC | ~1.79 TB/s | sm_120 | Supported |
| RTX 5090 (GB202) | 32 GB GDDR7 | 1.79 TB/s | sm_120 | Supported |
| Jetson Thor | unified | — | sm_121 | Planned |

> Consumer Blackwell is `sm_120` (RTX 5090) / `sm_121` (RTX Spark, Jetson Thor) —
> **not** `sm_100`, which is datacenter Blackwell (B200/GB200) and binary-incompatible.
> Builds target `89;90;100;120;121`; requires CUDA Toolkit 12.8+.

---

## What This Runtime Does

Standard serving stacks (vLLM, SGLang) are designed for distributed datacenter inference — tensor parallelism across multiple GPUs, large batch throughput, HBM-class bandwidth. On RTX Spark, those assumptions break:

- No TP — everything runs on one chip
- Unified memory means CPU and GPU share the same pool — expert weights don't need PCIe transfer
- LPDDR5X bandwidth (~273 GB/s) is 6–7× lower than GDDR7 — memory layout and scheduling matter more than raw kernel FLOPS
- Interactive latency targets (< 50 ms TTFT) require a latency-first scheduler, not a throughput-first one

This runtime is built specifically for that environment.

---

## Components

```
include/sparkinfer/
├── runtime.h         — Runtime lifecycle, device query (SMs, bandwidth, cc)
├── scheduler.h       — Continuous batching, priority preemption
├── kv_cache.h        — Paged KV block allocator, per-layer sub-pools
├── kv_ops.h          — KV append + residual add
├── decode.h          — DecodeRunner: generic MoE decode-layer wiring
└── models/qwen35.h   — Qwen3.5-35B-A3B model: config, weights, generate()

src/
├── runtime.cpp       — device setup + capability query
├── scheduler.cpp     — priority continuous-batching scheduler
├── kv_cache.cpp      — paged block allocator + device block tables
├── decode_layer.cpp  — DecodeRunner (norm→QKV→attn→O→residual→MoE→residual)
├── models/qwen35.cpp — full Qwen3.5 forward + greedy generate loop
└── csrc/cuda/kv_ops.cu — paged KV append + residual add kernels

tools/convert_qwen35.py  — HF safetensors → sparkinfer weight format
examples/qwen35_generate.cpp — greedy generation demo (token ids in/out)
```

---

## Target Models

| Model | Total / Active | Q4_K_M Size | Notes |
|---|---|---|---|
| Qwen3.5-35B-A3B | 35B / 3B | ~20 GB | 256 experts, 2 KV heads |
| Gemma 4 26B-A4B | 26B / 4B | ~14.6 GB | Interleaved local/global attn, head_dim=512 |

Both fit in RTX Spark's 128 GB unified memory with significant headroom for KV cache.

---

## Related Repos

| Repo | Purpose |
|---|---|
| [sparkinfer-kernels](https://github.com/gittensor-ai-lab/sparkinfer-kernels) | Native CUDA + CuTe DSL kernels: flash decode, RoPE, MoE router/FFN, GEMM, RMSNorm |
| [sparkinfer-moe](https://github.com/gittensor-ai-lab/sparkinfer-moe) | Sync-free MoE engine: router GEMM → top-k → SwiGLU expert FFN |
| [sparkinfer-bench](https://github.com/gittensor-ai-lab/sparkinfer-bench) | Reproducible benchmarks for RTX Spark, RTX 5090, Jetson Thor |
| [sparkinfer-agent](https://github.com/gittensor-ai-lab/sparkinfer-agent) | Kernel design agents: NCU report parsing, auto-tuning loops |

---

## Build

The runtime is the integrator — it pulls in the sibling `../kernels` and `../moe`
checkouts. Needs CUDA Toolkit 12.8+.

```bash
bash scripts/build.sh            # configures sm_120, builds, runs ctest
# or manually:
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="120"   # 121 for RTX Spark
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

---

## Running Qwen3.5-35B-A3B

The runtime runs Qwen3.5 end-to-end on token IDs (embed → 40 layers → LM head →
greedy sample). The full layer is implemented: RMSNorm, Q/K/V projection,
**per-head QK-norm**, **RoPE**, paged GQA flash-decode (8:1, head_dim=128),
O-projection, residuals, **256-expert top-8 routed MoE + 1 shared expert**, all
sync-free (token counts stay on-device, CUDA-graph capturable).

**1. Convert weights** (one-time, CPU, needs the HF checkpoint):

```bash
pip install safetensors numpy
python tools/convert_qwen35.py /path/to/hf/Qwen3.5-35B-A3B ./qwen35_weights
```

**2. Generate** (on an RTX 5090 / RTX Spark). Tokenize with the HF tokenizer to
get input IDs, then:

```bash
./build/qwen35_generate ./qwen35_weights 64  <id0> <id1> ...   # 64 new tokens
```

It prints the generated token IDs; decode them with the HF tokenizer.

```cpp
// or from C++:
sparkinfer::Qwen35Config cfg;                       // 35B-A3B defaults
sparkinfer::KVCacheManager kv(kvc, 512ull<<20);
auto engine = sparkinfer::moe::MoEEngine::create(mc);
sparkinfer::Qwen35Model model(cfg, &kv, engine.get());
model.load_weights("./qwen35_weights");
auto out = model.generate(prompt_ids, 64);          // greedy
```

---

## Verification

- **Targets the 5090/Spark** — every `.cu` compiles through NVRTC → `ptxas
  -arch=sm_120`/`sm_121` to a real cubin.
- **Correct math** — `tests/qwen35_cpu_test` runs the *entire* Qwen3.5 forward
  (embed, QK-norm, RoPE, multi-layer GQA, routed + shared MoE, LM head)
  autoregressively and matches a double-precision reference to ~1e-8;
  `tests/decode_layer_cpu_test` checks the layer composition; the kernel math is
  covered in sparkinfer-kernels' `cpu_reference_test`.
- **On hardware** — `tests/decode_runner_gpu_test` and `examples/qwen35_generate`
  run the real device path (both self-skip cleanly when no GPU is present).

---

## SN74 / Gittensor

This repository is part of **SN74** on [Gittensor](https://github.com/gittensor-ai-lab) — a decentralized network where human engineers and AI agents co-design and continuously improve kernels, memory systems, and MoE routing for edge AI hardware. All optimizations are reproducible: source-required, benchmark-verified, no hidden training or closed binaries.
