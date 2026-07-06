# Qwen3-30B-A3B RTX 5090 competitor run

Date: 2026-07-03

Node: `91.224.44.222:40030`

Hardware: RTX 5090, driver `595.71.05`, CUDA `13.0`

Benchmark shape: batch size 1, 128 generated tokens, contexts `128`, `512`, `4096`, `16384`.

## Headline results

| Context | SGLang GPTQ Int4 FP8 KV | vLLM GPTQ Int4 FP8 KV | vLLM AWQ FP8 KV | cyankiwi 4-bit auto-quant FP8 KV | sparkinfer GGUF Q4_K_M |
|---:|---:|---:|---:|---:|---:|
| 128 | 241.21 tok/s | 280.83 tok/s | 278.08 tok/s | 274.39 tok/s | 475.14 tok/s |
| 512 | 239.82 tok/s | 270.86 tok/s | 267.64 tok/s | 264.57 tok/s | 472.01 tok/s |
| 4,096 | 234.67 tok/s | 202.65 tok/s | 199.57 tok/s | 199.30 tok/s | 389.10 tok/s |
| 16,384 | 226.12 tok/s | 81.89 tok/s | 80.41 tok/s | 80.56 tok/s | 261.98 tok/s |

## Models

- vLLM AWQ: `cognitivecomputations/Qwen3-30B-A3B-AWQ`
- vLLM GPTQ Int4: `Qwen/Qwen3-30B-A3B-GPTQ-Int4`
- SGLang GPTQ Int4: `Qwen/Qwen3-30B-A3B-GPTQ-Int4`
- cyankiwi 4-bit: `cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit`
- sparkinfer: `Qwen/Qwen3-30B-A3B-GGUF`, `Qwen3-30B-A3B-Q4_K_M.gguf`

The cyankiwi repo name says AWQ-4bit, but vLLM detects the checkpoint quantization as `compressed-tensors`. The saved failed log for the explicit `--quantization awq` attempt is kept in `vllm_awq_cyankiwi/cyankiwi_instruct2507_awq4bit_fp8_ctx128_out128.log`.

## vLLM GPTQ Int4 FP8 KV

Command family: `vllm bench latency`, vLLM auto-detected `auto_gptq`, `--kv-cache-dtype fp8_e4m3`, graph mode.

Raw files: `vllm_gptq_int4/`

| Context | Avg latency (s) | Output tok/s |
|---:|---:|---:|
| 128 | 0.455800 | 280.83 |
| 512 | 0.472565 | 270.86 |
| 4,096 | 0.631616 | 202.65 |
| 16,384 | 1.562994 | 81.89 |

## SGLang GPTQ Int4 FP8 KV

Command family: `python -m sglang.bench_one_batch`, SGLang `0.5.14`, `--quantization gptq_marlin`, `--kv-cache-dtype fp8_e4m3`, batch size 1.

Raw files: `sglang_gptq_int4/`

| Context | Median decode latency (s) | Median decode tok/s | Status |
|---:|---:|---:|---|
| 128 | 0.004146 | 241.21 | pass |
| 512 | 0.004170 | 239.82 | pass |
| 4,096 | 0.004261 | 234.67 | pass with fixed prefill config |
| 16,384 | 0.004422 | 226.12 | pass with fixed prefill config |

The default 4k run fails before producing a JSON result with `RuntimeError: The size of tensor a (...) must match the size of tensor b (4096)`. The working 4k/16k runs disable chunked prefill (`--chunked-prefill-size -1`) and set `--max-prefill-tokens` to the full prompt length. The 16k run also needs `--mem-fraction-static 0.85`; without it SGLang auto-selects too small a static memory fraction for KV allocation.

Fixed-run raw files: `sglang_gptq_int4_fix/`

## vLLM AWQ FP8 KV

Command family: `vllm bench latency`, `--quantization awq`, `--kv-cache-dtype fp8_e4m3`, graph mode.

Raw files: `vllm_awq_fp8_only/`

| Context | Avg latency (s) | Output tok/s |
|---:|---:|---:|
| 128 | 0.460296 | 278.08 |
| 512 | 0.478263 | 267.64 |
| 4,096 | 0.641379 | 199.57 |
| 16,384 | 1.591767 | 80.41 |

## vLLM AWQ default KV

Raw files: `vllm_awq_variants/default_kv_*`

| Context | Avg latency (s) | Output tok/s |
|---:|---:|---:|
| 128 | 0.517197 | 247.49 |
| 512 | 0.532640 | 240.31 |
| 4,096 | 0.698626 | 183.22 |
| 16,384 | 1.783694 | 71.76 |

## vLLM AWQ FP8 KV with `--enforce-eager`

Raw files: `vllm_awq_variants/fp8_enforce_eager_*`

| Context | Avg latency (s) | Output tok/s |
|---:|---:|---:|
| 128 | 4.461926 | 28.69 |
| 512 | 4.398002 | 29.10 |
| 4,096 | 4.589706 | 27.89 |
| 16,384 | 5.618173 | 22.78 |

`--enforce-eager` disables torch.compile and CUDA graphs in vLLM and is not competitive for this comparison.

## cyankiwi 4-bit auto-quant FP8 KV

Command family: `vllm bench latency`, no explicit `--quantization`, `--kv-cache-dtype fp8_e4m3`, graph mode.

Raw files: `vllm_awq_cyankiwi/cyankiwi_instruct2507_autoquant_fp8_*`

| Context | Avg latency (s) | Output tok/s |
|---:|---:|---:|
| 128 | 0.466496 | 274.39 |
| 512 | 0.483811 | 264.57 |
| 4,096 | 0.642246 | 199.30 |
| 16,384 | 1.588904 | 80.56 |

## sparkinfer GGUF Q4_K_M

Raw files: `sparkinfer/`

| Context | Output tok/s |
|---:|---:|
| 128 | 475.14 |
| 512 | 472.01 |
| 4,096 | 389.10 |
| 16,384 | 261.98 |

## TensorRT-LLM GPTQ/AWQ probes

TensorRT-LLM `1.2.1` and TensorRT `10.14.1.48.post1` were installed in a separate virtualenv after adding the system MPI runtime. `trtllm-bench` loads successfully.

Raw files: `trtllm_gptq_int4/`, `trtllm_awq/`, `probes/`

| Target | TRT-LLM quantization | Result |
|---|---|---|
| `Qwen/Qwen3-30B-A3B-GPTQ-Int4` | `W4A16_GPTQ` | build rejected the HF `quantization_config` as unsupported |
| `cognitivecomputations/Qwen3-30B-A3B-AWQ` | `W4A16_AWQ` | build rejected the HF `quantization_config` as unsupported |

No TensorRT-LLM latency number was produced because no engine was built from the downloaded pre-quantized HF checkpoints.

## TensorRT Edge-LLM probe

`tensorrt-edge-llm` and `tensorrt_edge_llm` were not available as PyPI packages on this host. NVIDIA's TensorRT Edge-LLM flow is documented for supported Thor edge platforms; this RTX 5090 node is an experimental/non-primary target for that stack.

## Saved artifacts

- `scripts/`: benchmark runner scripts copied from the node. The HF token environment file was intentionally not copied.
- `vllm_awq/`: original vLLM AWQ local run and setup logs.
- `vllm_awq_fp8_only/`: clean FP8-KV vLLM AWQ rerun.
- `vllm_awq_variants/`: default-KV and enforce-eager variant runs.
- `vllm_awq_cyankiwi/`: cyankiwi 4-bit compressed-tensors run and failed explicit-AWQ attempt.
- `vllm_gptq_int4/`: Qwen GPTQ Int4 vLLM run.
- `sglang_gptq_int4/`: SGLang GPTQ Int4 run and 4k failure variants.
- `sglang_gptq_int4_fix/`: fixed SGLang GPTQ Int4 4k/16k runs with full-length prefill buffers.
- `trtllm_gptq_int4/`: TensorRT-LLM GPTQ build attempts.
- `trtllm_awq/`: TensorRT-LLM AWQ build attempt.
- `probes/`: TensorRT-LLM install log, MPI install log, and TensorRT Edge-LLM package probe.
- `sparkinfer/`: sparkinfer GGUF benchmark logs.
