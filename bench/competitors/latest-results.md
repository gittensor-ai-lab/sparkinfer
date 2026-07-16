# Latest Competitor Results

Last run: 2026-07-03 on RTX 5090 node `91.224.44.222:40030`.

Raw artifacts: `bench/competitors/results/qwen3_awq_rtx5090_20260703/`

## vLLM AWQ vs sparkinfer GGUF

vLLM engine: `0.23.1rc1.dev758+g978de8335`

vLLM model: `cognitivecomputations/Qwen3-30B-A3B-AWQ`

sparkinfer commit: `5d1a80d9ca65585eec70f6ddd02edc1c6a9a5313`

sparkinfer model: `Qwen/Qwen3-30B-A3B-GGUF`, `Qwen3-30B-A3B-Q4_K_M.gguf`

Hardware: RTX 5090, driver `595.71.05`, CUDA `13.0`

Benchmark shape: batch size 1, 128 generated tokens, 2 warmup iterations, 8 measured iterations.

vLLM command shape:

```bash
vllm bench latency \
  --model /workspace/qwen_awq_bench/models/qwen3-awq \
  --trust-remote-code \
  --quantization awq \
  --input-len <ctx> \
  --output-len 128 \
  --batch-size 1 \
  --num-iters-warmup 2 \
  --num-iters 8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8_e4m3 \
  --disable-detokenize
```

Important caveat: this is a competitor/runtime comparison, not an identical quant-format comparison. vLLM is running AWQ safetensors with FP8 KV cache; sparkinfer is running its native GGUF `Q4_K_M` path.

| Context | vLLM AWQ FP8 KV | sparkinfer GGUF Q4_K_M |
|---:|---:|---:|
| 128 | 278.08 tok/s | 475.14 tok/s |
| 512 | 267.64 tok/s | 472.01 tok/s |
| 4,096 | 199.57 tok/s | 389.10 tok/s |
| 16,384 | 80.41 tok/s | 261.98 tok/s |

vLLM FP8-KV latency details:

| Context | Avg latency (s) | Output tok/s |
|---:|---:|---:|
| 128 | 0.460296 | 278.08 |
| 512 | 0.478263 | 267.64 |
| 4,096 | 0.641379 | 199.57 |
| 16,384 | 1.591767 | 80.41 |

Run notes:

- vLLM selected `Qwen3MoeForCausalLM`, `AutoAWQMarlinLinearMethod`, Marlin WNA16 MoE, and FlashInfer attention for the FP8-KV run.
- Removing `--kv-cache-dtype fp8_e4m3` made vLLM slower and switched the selected attention path for the default-KV run:

| Context | vLLM AWQ default KV |
|---:|---:|
| 128 | 247.49 tok/s |
| 512 | 240.31 tok/s |
| 4,096 | 183.22 tok/s |
| 16,384 | 71.76 tok/s |

- `--enforce-eager` was not competitive on this setup, dropping to roughly `23-29 tok/s`, so it should not be used for the headline vLLM comparison.

## cyankiwi 4-bit AWQ-labeled checkpoint

Model: `cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit`

Result: vLLM detects this checkpoint as `compressed-tensors`, not `awq`. Explicit `--quantization awq` fails with a quantization-method mismatch, so the successful run uses vLLM auto-quantization with `--kv-cache-dtype fp8_e4m3`.

| Context | Avg latency (s) | Output tok/s |
|---:|---:|---:|
| 128 | 0.466496 | 274.39 |
| 512 | 0.483811 | 264.57 |
| 4,096 | 0.642246 | 199.30 |
| 16,384 | 1.588904 | 80.56 |

## Qwen GPTQ Int4

Model: `Qwen/Qwen3-30B-A3B-GPTQ-Int4`

Result: vLLM detects this checkpoint as `auto_gptq`. The run used `--kv-cache-dtype fp8_e4m3`, graph mode, batch size 1, and 128 generated tokens.

| Context | Avg latency (s) | Output tok/s |
|---:|---:|---:|
| 128 | 0.455800 | 280.83 |
| 512 | 0.472565 | 270.86 |
| 4,096 | 0.631616 | 202.65 |
| 16,384 | 1.562994 | 81.89 |

## SGLang GPTQ Int4

Engine: SGLang `0.5.14`

Model: `Qwen/Qwen3-30B-A3B-GPTQ-Int4`

Result: SGLang requires `--quantization gptq_marlin` for this MoE GPTQ checkpoint. The successful runs used `python -m sglang.bench_one_batch`, `--kv-cache-dtype fp8_e4m3`, batch size 1, and 128 generated tokens.

| Context | Median decode latency (s) | Median decode tok/s | Status |
|---:|---:|---:|---|
| 128 | 0.004146 | 241.21 | pass |
| 512 | 0.004170 | 239.82 | pass |
| 4,096 | 0.004261 | 234.67 | pass with fixed prefill config |
| 16,384 | 0.004422 | 226.12 | pass with fixed prefill config |

The default 4k context failed before producing a result JSON with a SGLang internal buffer mismatch: `The size of tensor a (...) must match the size of tensor b (4096)`. The fix was to disable chunked prefill (`--chunked-prefill-size -1`) and set `--max-prefill-tokens` to the full prompt length. The 16k run also needs `--mem-fraction-static 0.85` so SGLang leaves enough static memory for KV allocation. Raw logs are in `bench/competitors/results/qwen3_awq_rtx5090_20260703/sglang_gptq_int4/` and `bench/competitors/results/qwen3_awq_rtx5090_20260703/sglang_gptq_int4_fix/`.

## TensorRT-LLM GPTQ/AWQ

Engine: TensorRT-LLM `1.2.1`, TensorRT `10.14.1.48.post1`

Result: `trtllm-bench` installed and loaded after adding the system MPI runtime, but direct engine builds from the downloaded pre-quantized HF checkpoints failed before engine generation.

| Target | TRT-LLM quantization | Result |
|---|---|---|
| `Qwen/Qwen3-30B-A3B-GPTQ-Int4` | `W4A16_GPTQ` | rejected unsupported HF `quantization_config` |
| `cognitivecomputations/Qwen3-30B-A3B-AWQ` | `W4A16_AWQ` | rejected unsupported HF `quantization_config` |

No TensorRT-LLM latency number was produced. The next viable path is an NVIDIA-supported conversion/export flow or an NGC image with a supported checkpoint format rather than direct `trtllm-bench build` from these HF AWQ/GPTQ directories.

## TensorRT Edge-LLM

Result: no RTX 5090 benchmark number.

`tensorrt-edge-llm` and `tensorrt_edge_llm` had no matching PyPI distribution on this host. NVIDIA's TensorRT Edge-LLM documentation lists Thor edge platforms as the official tested targets, with other NVIDIA GPUs experimental. For this RTX 5090 node, the practical path is to follow the Edge-LLM export/runtime flow on a supported edge target rather than direct pip install and benchmark.

## TensorRT-LLM Docker NVFP4

Engine: NVIDIA TensorRT-LLM NGC container `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20`

Model: `nvidia/Qwen3-30B-A3B-NVFP4`

Result: TensorRT-LLM AutoDeploy runs the NVFP4 checkpoint on RTX 5090 for short contexts, but failed in an internal FP4 layernorm kernel at 4k context. The successful runs used `trtllm-bench latency`, `_autodeploy`, batch size 1, 6 requests, 1 warmup request, 128 generated tokens, CUDA graph decode mode, and KV cache FP8.

| Context | Output tok/s | Avg request latency (ms) | Avg TTFT (ms) | Avg TPOT (ms) | Status |
|---:|---:|---:|---:|---:|---|
| 128 | 99.00 | 1292.76 | 400.70 | 7.024 | pass |
| 512 | 98.59 | 1298.18 | 400.48 | 7.069 | pass |
| 4,096 | — | — | — | — | failed: `ws_layernorm_fp4_traits.cu:92` CUDA invalid argument |
| 16,384 | — | — | — | — | not run after 4k failure |

An aligned total-sequence probe (`3968` input + `128` output = `4096` max tokens) failed with the same `ws_layernorm_fp4_traits.cu:92` CUDA invalid argument, so this does not appear to be only a `4096 + 128 = 4224` sequence-length issue. Raw logs are in `bench/competitors/results/qwen3_awq_rtx5090_20260703/trtllm_docker_nvfp4_ctx_sweep/` and `bench/competitors/results/qwen3_awq_rtx5090_20260703/trtllm_docker_nvfp4_aligned_sweep/`.

## Previous vLLM NVFP4

Run: 2026-07-03 on Vast RTX 5090 instance `43698008` (destroyed after run).

Engine: vLLM `0.23.1rc1.dev757+ga14f57a3a`

Model: `nvidia/Qwen3-30B-A3B-NVFP4`

Hardware: RTX 5090, driver `580.159.03`, CUDA `13.0`

Benchmark shape: `vllm bench latency`, batch size 1, 128 generated tokens, 2 warmup iterations, 8 measured iterations, `kv_cache_dtype=fp8_e4m3`.

Important caveat: this is not the same weight format as sparkinfer's GGUF `Q4_K_M` path. It is an HF NVFP4 checkpoint running through vLLM FlashInfer/CUTLASS kernels.

| Context | Avg latency (s) | Output tok/s |
|---:|---:|---:|
| 128 | 0.574194 | 222.92 |
| 512 | 0.578056 | 221.43 |
| 4,096 | 0.662612 | 193.17 |
| 16,384 | 1.348394 | 94.93 |

Run notes:

- `HF_HUB_DISABLE_XET=1` was required because HF Xet token requests failed with TLS handshake EOF on this Vast host.
- Cold checkpoint download took `3432.46s` for a 16.85 GiB checkpoint.
- `datasets` was upgraded from `1.1.1` to `5.0.0` to fix incompatibility with `pyarrow==24`.
- 16k context was run with `MAX_MODEL_LEN=32768` because vLLM requires `max_model_len >= input_len + output_len`.

## TensorRT-LLM NVFP4

Target: `nvidia/Qwen3-30B-A3B-NVFP4`, `trtllm-bench`, 128 context, 128 generated tokens.

Result: see the Docker NVFP4 section above for the successful NGC-container run.

Earlier setup on a separate Vast CUDA 13 image did not complete before the instance was stopped. On the later RTX 5090 node, direct TensorRT-LLM Python install was useful for AWQ/GPTQ probes, but the successful NVFP4 path used NVIDIA's TensorRT-LLM NGC container.

Better next path: isolate the 4k failure in a smaller TensorRT-LLM reproducer or retest with the next TensorRT-LLM container release.
