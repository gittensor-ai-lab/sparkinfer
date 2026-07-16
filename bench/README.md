# sparkinfer-bench

Reproducible benchmark suite for edge MoE inference on **NVIDIA RTX Spark**, RTX 5090, and Jetson Thor.

Part of [gittensor-ai-lab](https://github.com/orgs/gittensor-ai-lab) — SN74.

---

## Design principles

**Source-required builds.** All kernels are compiled from source. No pre-built binaries, no Docker image swapping. A benchmark result is only valid if it includes the commit hash and build flags that produced it.

**Frozen model weights.** Benchmarks use publicly available Q4_K_M quants pinned by SHA256. Weight substitution to game routing efficiency is detectable and excluded.

**Hardware-level metrics.** Primary metrics are measured memory bandwidth (GB/s), TFLOPS utilization, and latency at batch size 1. Token throughput is a derived metric, not the target.

---

## Target models

| Model | Quant | Size | Benchmark targets |
|---|---|---|---|
| Qwen3.5-35B-A3B | Q4_K_M | ~20 GB | 130 tok/s @ RTX 5090, bs=1, ctx=2K |
| Gemma 4 26B-A4B | Q4_K_M | ~14.6 GB | 256K context on RTX 5090 without OOM |

---

## Structure

```
benchmarks/
├── attention/
│   └── flash_decode_bench.py     # flash decode latency vs head_dim, seqlen, GQA ratio
├── moe/
│   └── moe_routing_bench.py      # routing latency, expert dispatch, CUDA graph speedup
├── gemm/                         # grouped GEMM throughput vs expert count, token count
├── memory/                       # KV cache bandwidth, expert weight fetch latency
└── e2e/                          # end-to-end generation, TTFT, TBT

configs/
├── models/
│   ├── qwen35_35b_a3b.yaml       # arch spec, derived memory requirements
│   └── gemma4_26b_a4b.yaml       # interleaved 5L:1G attention, hd256/hd512
├── targets/
│   ├── qwen35_q4km_rtx5090.yaml  # baseline 80 tok/s, target 130 tok/s
│   ├── qwen35_q4km_rtx_spark.yaml
│   ├── gemma4_q4km_rtx5090.yaml  # 20.2 GB @ 256K ctx
│   └── gemma4_q4km_rtx_spark.yaml
├── rtx5090.yaml                  # hardware spec: 32 GB, 1.79 TB/s, sm_120
└── rtx_spark.yaml                # hardware spec: 128 GB, 273 GB/s, sm_121

scripts/
└── run_all.sh                    # run all benchmarks and emit results/ JSON
```

---

## Running

```bash
# Single benchmark
python benchmarks/attention/flash_decode_bench.py \
    --model configs/models/qwen35_35b_a3b.yaml \
    --target configs/targets/qwen35_q4km_rtx5090.yaml

# Full suite
bash scripts/run_all.sh
```

Results land in `results/` as JSON with kernel commit hash, build flags, hardware UUID, and driver version embedded.
