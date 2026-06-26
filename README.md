# sparkinfer

**Blackwell-native MoE/LLM inference runtime.** The engineering arm of [SN74 on Gittensor](https://github.com/gittensor-ai-lab) — reproducible, hardware-level inference-speed gains for NVIDIA Blackwell consumer/edge GPUs: RTX Spark (`sm_121`), RTX 5090 & RTX PRO 6000 (`sm_120`), Jetson Thor (`sm_121`).

## Proven

Qwen3-30B-A3B (Q4_K_M GGUF) runs end-to-end on an RTX PRO 6000 (sm_120), decode optimized **0.60 → 134 tok/s (≈220×)** across 6 source-verifiable passes, output verified correct, **21.7 GB** resident (experts kept quantized). Independently verified on an **RTX 5090** (CUDA 13): community optimizations have ratcheted the decode frontier to **285.32 tok/s** (≈0.78× llama.cpp) at **≥96% top-1 token agreement** with llama.cpp (KL ≈ 0.14 nats) — see the live [dashboard](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/), [accuracy](bench/results/accuracy_qwen3-30b-a3b_q4km.md), and [RTX 5090](bench/results/qwen3-30b-a3b_q4km_rtx5090.md) results.

## Why a custom engine

The datacenter engines (vLLM, SGLang, TensorRT-LLM) optimize **throughput** for multi-user serving and are *retrofitting* consumer Blackwell (`sm_120/121`); llama.cpp is the portable foundation but specializes for no architecture. sparkinfer targets the gap they leave — single-stream (`bs=1`) latency for **on-device agents**, not datacenter batch throughput:

- **Open + hackable.** A small, readable CUDA codebase — every kernel is auditable and forkable, not a closed graph compiler. Gains are source-required, reproducible, and rewarded per *verified frontier-delta* on SN74.
- **Newest-architecture-first.** We ship Blackwell kernels for the latest models before the big engines do — e.g. Gemma 4's `head_dim=512` global attention has **no public implementation** in FlashInfer, vLLM, or llama.cpp; we have one.
- **Deeper MoE-bandwidth specialization, not generality.** Rather than run everything everywhere, we go deep on **single-device MoE decode** — experts kept quantized-resident, dequantized on-read, minimizing **bytes-per-token** — where the generalist engines leave performance on the table.

## Quickstart

On an NVIDIA Blackwell box (CUDA 12.8+) — the scripts auto-detect your GPU arch, fetch **prebuilt binaries** (or build from source if incompatible), and download the model:

```bash
# decode throughput (fetches Qwen3-30B-A3B Q4_K_M on first run)
bench/scripts/bench.sh --download

# head-to-head vs llama.cpp on the same GGUF + GPU
bench/scripts/bench.sh --download --compare

# accuracy gate — token-match / KL / perplexity vs llama.cpp
bench/scripts/accuracy.sh --download
```

Your own model: `bench/scripts/bench.sh /path/to/model.gguf --tokens 256`. All options: [`bench/scripts/README.md`](bench/scripts/README.md).

## Layout & scoring

| Path | What |
|---|---|
| [`kernels/`](kernels) | CUDA kernels — flash-decode (hd128/256/512), decode GEMV, fused quantized MoE expert FFN, GEMM, RMSNorm, RoPE, GGUF dequant |
| [`runtime/`](runtime) | scheduler, paged KV cache, CUDA-graph decode, native GGUF loading, model forward |
| [`moe/`](moe) | sync-free MoE router + expert dispatch (on-device counts, CUDA-graph-ready) |
| [`bench/`](bench) | reproducible benchmarks + eval harness (the eval/scoring scripts are maintainer-owned) |

**Scoring is speedup-only.** SN74 pays each merged PR for its **verified frontier-delta speedup**, labeled **XL / L / M / S / XS** by the deterministic eval loop (or **BASELINE** for the first verified entry on a new model/target). A speedup is scored the same wherever it lands — there is no per-subsystem budget. **Non-speedup PRs — tooling, bench, docs, refactors — are welcome but score 0.** See [`.gittensor/weights.json`](.gittensor/weights.json) and the [org reward model](https://github.com/gittensor-ai-lab).

## Build

Requires **CUDA Toolkit 12.8+** (first toolkit with `sm_120` / `sm_121` codegen).

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120   # or 121 for RTX Spark / Jetson Thor
cmake --build build -j
ctest --test-dir build
```

The top-level `CMakeLists.txt` is a superbuild (`kernels → moe → runtime`); each subsystem also builds standalone (the sibling `../kernels` references resolve within the monorepo). A direct `nvcc` build from the repo root works too — see [`bench/scripts`](bench/scripts).

## Targets

**Blackwell only, by design:** `sm_120` (RTX 5090, RTX PRO 6000) and `sm_121` (RTX Spark / GB10, Jetson Thor). **Not** `sm_100` (datacenter B200/GB200 — binary-incompatible).

## Roadmap

**Milestone 1 — RTX 5090 proof-of-concept (now).** Qwen3-MoE + Gemma 4 decode on `sm_120` (1.8 TB/s): the kernel set (fused quantized MoE expert FFN, flash-decode incl. Gemma `hd512`, on-read dp4a MMVQ, split-K occupancy) and the source-verified, correctness-gated eval loop vs llama.cpp. The 5090 is where we prove the kernels and the loop.

**Milestone 2 — MoE on low-bandwidth unified memory (next).** Migrate the engineering to the **RTX Spark / GB10 class** (`sm_121`, 128 GB unified, **~273 GB/s — ≈6.5× less bandwidth than the 5090**). There, dense models hit a hard bandwidth floor (e.g. Llama-70B ≈ 2.7 tok/s) and **MoE is the only viable path** (≈10× fewer bytes/token). The optimization target shifts from warp occupancy to **bytes-per-token**: NVFP4 experts, expert residency / caching / prefetch, and eliminating redundant weight reads — the single-device-unified-memory MoE specialization the throughput-tuned datacenter engines don't have. This is the hardware NVIDIA ships as "personal AI," and making big MoE fast on it is where sparkinfer is designed to win.

## Contributing

Source-required and reproducible — the validator builds your PR from source (the
prebuilt binaries are a run convenience, not a submission format). Before a PR, run
`bench/scripts/bench.sh` (speed) and `bench/scripts/accuracy.sh` (accuracy must hold:
~100% top-1 + KL ≈ 0 vs the prior build). Contributions are rewarded on SN74 by the
**verified marginal speedup** added over the live frontier, correctness-gated against a
frozen reference, validated on both basket models (Qwen + Gemma). See
[CONTRIBUTING.md](CONTRIBUTING.md) and the [org reward model](https://github.com/gittensor-ai-lab).

## Automated evaluation

Open a PR and a bot evaluates it automatically (polls every ~30 min). For each new commit it
builds your branch **from source** on an RTX 5090, gates **correctness** (token-match / KL vs
llama.cpp), benchmarks **decode speed**, and posts a comment with an **`eval:<label>`** verdict:

| label | meaning |
|---|---|
| `XL · L · M · S · XS` | verified speedup over the live frontier, by **% gain** (`XS` 2–3.5% … `XL` >18%) |
| `none` | correct, but no verified improvement (within the significance gate) |
| `REJECT` | failed the correctness gate — the output changed |
| `BASELINE` | first verified entry; establishes the frontier |

The label is a **deterministic function of the measurements**, so it's reproducible across
validators. The bot also tags the PR's **subsystem** — `area:kernels` / `runtime` / `moe` /
`bench` — from its changed paths (categorization only — scoring is speedup-only; deterministic, no AI).
The bot **never merges** — merging is manual after review. Runs the same evaluator you can run
yourself: [`eval/`](eval) (`vast_eval.py`, `pr_eval_bot.py`).

## License

[MIT](LICENSE) · [Changelog](CHANGELOG.md)
