# Latest Competitor Results

Last run: 2026-07-03 on Vast RTX 5090 instance `43698008` (destroyed after run).

## vLLM NVFP4

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

Result: no benchmark number.

TensorRT-LLM was not preinstalled on the Vast CUDA 13 image. The setup reached apt dependencies for `libopenmpi-dev` and then PyTorch/CUDA wheel installation in a TensorRT-LLM virtualenv. After about 11 minutes it was still preparing/downloading PyTorch/CUDA wheels and had not reached `pip install tensorrt_llm` or `trtllm-bench`, so the attempt was stopped to avoid continuing paid GPU time on setup-only work.

Better next path: run NVIDIA's TensorRT-LLM NGC container directly, or prebuild a Vast image with TensorRT-LLM installed, then run `trtllm-bench` against the same NVFP4 checkpoint and contexts.
