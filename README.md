![sparkinfer banner](docs/sparkinfer.png)

# SP⚡RKINFER · Powered by SN74

**Agentic AI inference. Optimized for every Blackwell GPU.**

**+71% faster** than llama.cpp with Blackwell-native **Custom CUDA kernels** (Qwen3.6-35B-A3B SOTA, RTX 5090, v0.4.1). SparkInfer is the runtime layer for **Private AI** agents — optimized MoE/LLM decoding from desk-side RTX to workstation PRO 6000. Continuously optimized by competition at **[SN74 on Gittensor](https://gittensor.io/miners/repository?name=gittensor-ai-lab%2Fsparkinfer)** and **Kernel Design Agents**.

**Why fastest?** Faster inference means more intelligence, more responsive agents, and more efficient compute.

| | |
|---|---|
| **[Live dashboard](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/)** | Frontier tok/s, optimization journey, evaluated PRs |
| **[SparkInfer Chat](https://github.com/gittensor-ai-lab/sparkinfer-web)** | Next.js UI — streaming, think mode, OpenAI-compatible proxy |
| **[OpenAI API server](server/)** | `sparkinfer-server` on `dev` — `/v1/chat/completions`, `/v1/info`, `/v1/tokenize` |

> **Fewer models. Deeper optimization. Faster evolution.**

## Frontier models

SparkInfer focuses on the models driving the future of AI — not thousands of legacy architectures.

| Model | Role |
|---|---|
| [**Qwen3.6-35B-A3B**](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | Primary SOTA — hybrid Gated-DeltaNet + full-attention MoE |
| [**Qwythos 9B**](https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF) | Mythos-level reasoning · long-context guard |
| [**SparkDistill**](https://github.com/gittensor-model-hub/SparkDistill/) | Fable 5 / OpenAI 5.6-level CoT *(coming soon)* |
| [**MiniMax M3**](https://huggingface.co/MiniMaxAI/MiniMax-M3) | Open MoE frontier *(next)* |

## Blackwell native

**Consumer → Workstation → Datacenter** — built for NVIDIA Blackwell from the beginning (`sm_120` + `sm_121`, not datacenter `sm_100`).

| GPU | Arch | Target |
|---|---|---|
| [RTX Spark GB10](https://nvidianews.nvidia.com/news/nvidia-microsoft-windows-pcs-agents-rtx-spark) | `sm_121` | Personal AI PC · desk-side agents |
| [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) | `sm_121` | AI workstation |
| [RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/) | `sm_120` | Consumer Blackwell · current dev platform |
| [RTX PRO 6000](https://www.nvidia.com/en-us/products/workstations/) | `sm_120` | 96 GB workstation · 32k/4k API profile |

## Benchmark · Qwen3.6-35B-A3B SOTA

RTX 5090 · same `UD-Q4_K_M` GGUF · greedy bs=1 decode · warm interleaved · **v0.4.1** frontier.

| context | SparkInfer | llama.cpp | Δ |
|---:|---:|---:|---:|
| 128 | **473** tok/s | 276 tok/s | **+71%** |
| 512 | **481** tok/s | 276 tok/s | +74% |
| 4k | **460** tok/s | 276 tok/s | +66% |
| 16k | **450** tok/s | 281 tok/s | +60% |
| 32k | **428** tok/s | 280 tok/s | +53% |

Quality parity vs llama.cpp: top-1 **0.953** · KL **0.031** · IFEval **83%** · BFCL **75%**.

Full competitor matrix (vLLM, SGLang, TensorRT-LLM) and quality tables:
[`bench/competitors/latest-results.md`](bench/competitors/latest-results.md) ·
[`bench/quality/README.md`](bench/quality/README.md).

Runtime footprint (excluding model weights):

| runtime | size | vs sparkinfer |
|---|---:|---:|
| sparkinfer native binary | **2.5 MB** | 1× |
| llama.cpp CUDA | 80 MB | 33× larger |
| vLLM | 605 MB | 243× larger |

## Powered by SN74 — moving at the speed of ⚡

Contributors submit PRs; the bot verifies correctness and speed on real RTX 5090 hardware; SN74 rewards verified marginal speedups. **15 releases in 3 weeks** — from first llama.cpp beat to **+71% on Qwen3.6 SOTA**.

1. Pick a narrow bottleneck in the Blackwell decode path.
2. Submit a PR with source changes and benchmark evidence.
3. The bot builds `main` and the PR on the same RTX 5090.
4. Correctness vs llama.cpp; guards at 128 / 512 / 4k / 16k / 32k decode.
5. Strongest context improvement scores; regressions get `regression-*` labels.
6. Frontier merges; the [dashboard](https://gittensor-ai-lab.github.io/sparkinfer/dashboard/) updates.

Miner workflow: [`docs/miner-guide.md`](docs/miner-guide.md).

## Roadmap

### Milestone 1 · Now — Fast on every Blackwell edge GPU

*Fastest = cost-effective inference* — more tokens per dollar on Blackwell edge first.

- Qwen3.6 SOTA: **+71%** vs llama.cpp on RTX 5090 (473 tok/s @ 128 ctx)
- RTX PRO 6000 — **32k input + 4k output**, full MoE resident
- RTX Spark + DGX Spark `sm_121` bring-up for desk-side agents
- Fastest AI runtime at the edge · desktop app, RAG, memory

### Milestone 2 · Next — Trustable AI on confidential compute

Attested builds and sealed execution on PRO 6000 server and B200.

- TDX + NVIDIA CC attestation for `sparkinfer-server` workloads
- Source-verified binaries — same eval loop, inside the enclave
- Privacy guardrails and end-to-end encryption
- Domain-specific models via [SparkDistill](https://github.com/gittensor-model-hub/SparkDistill/)
- Licensed on-prem runtime for regulated enterprise

## Quickstart

On NVIDIA Blackwell (CUDA 12.8+) — scripts auto-detect GPU arch, fetch prebuilt binaries (or build from source), and download the model:

```bash
# decode throughput (fetches Qwen3-30B-A3B Q4_K_M on first run)
bench/scripts/bench.sh --download

# head-to-head vs llama.cpp on the same GGUF + GPU
bench/scripts/bench.sh --download --compare

# accuracy gate — token-match / KL vs llama.cpp
bench/scripts/accuracy.sh --download
```

Your own model: `bench/scripts/bench.sh /path/to/model.gguf --tokens 256`. Options: [`bench/scripts/README.md`](bench/scripts/README.md).

**OpenAI-compatible server** (`dev` branch): [`server/README.md`](server/README.md)

```bash
./server/run.sh --download   # http://127.0.0.1:8080/v1/chat/completions
```

## Layout & scoring

| Path | What |
|---|---|
| [`kernels/`](kernels) | CUDA kernels — flash-decode, decode GEMV, fused MoE FFN, GEMM, RMSNorm, RoPE, GGUF dequant |
| [`runtime/`](runtime) | scheduler, paged KV cache, CUDA-graph decode, native GGUF loading, model forward |
| [`moe/`](moe) | sync-free MoE router + expert dispatch |
| [`bench/`](bench) | reproducible benchmarks + eval harness |
| [`dashboard/`](dashboard) | static frontier dashboard (GitHub Pages) |
| [`server/`](server) | OpenAI-compatible HTTP API (`BUILD_SERVER=ON`) |

**Scoring is speedup-only.** SN74 pays verified marginal speedups labeled **XL / L / M / S / XS**. Sub-2% gains are never aggregated across contexts. See [`.gittensor/weights.json`](.gittensor/weights.json).

## Build

Requires **CUDA Toolkit 12.8+** (`sm_120` / `sm_121` codegen).

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120   # or 121 for RTX Spark / Jetson Thor
cmake --build build -j
ctest --test-dir build
```

## Automated evaluation

Open a PR — a bot evaluates every ~30 min: source build on RTX 5090, correctness gate vs llama.cpp, no-regression guards, **`eval:<label>`** verdict. The bot **never auto-merges**. Details: [`eval/`](eval) · **[EVAL-TRUST.md](EVAL-TRUST.md)** (Polaris TDX receipts, reproducible from source today).

| label | meaning |
|---|---|
| `XL · L · M · S · XS` | verified speedup over frontier, by % gain |
| `none` | correct, no verified improvement |
| `REJECT` | failed correctness or regression |
| `BASELINE` | first verified frontier entry |

## Contributing

Source-required and reproducible. Before a PR: `bench/scripts/bench.sh` + `bench/scripts/accuracy.sh`. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) · [Changelog](CHANGELOG.md)
