# vLLM NVFP4 RTX 5090 Benchmark — 2026-07-03

Engine: vLLM `0.23.1rc1.dev757+ga14f57a3a`

Model: `nvidia/Qwen3-30B-A3B-NVFP4`

Hardware: RTX 5090, driver `580.159.03`, CUDA `13.0`

Command shape: `vllm bench latency`, batch size 1, 128 generated tokens, 2 warmup iterations, 8 measured iterations, `kv_cache_dtype=fp8_e4m3`.

Important caveat: this is not the same weight format as sparkinfer's GGUF `Q4_K_M` path. It is an HF NVFP4 checkpoint running through vLLM FlashInfer/CUTLASS kernels.

| Context | Avg latency (s) | Output tok/s |
|---:|---:|---:|
| 128 | 0.574194 | 222.92 |
| 512 | 0.578056 | 221.43 |
| 4,096 | 0.662612 | 193.17 |
| 16,384 | 1.348394 | 94.93 |

Notes:

- The first cold 128-context run spent `3432.46s` downloading the 16.85 GiB checkpoint from Hugging Face on this Vast host.
- `HF_HUB_DISABLE_XET=1` was required; HF Xet token requests failed with TLS handshake EOF on this host.
- `datasets` had to be upgraded from `1.1.1` to `5.0.0` because `datasets==1.1.1` is incompatible with the installed `pyarrow==24`.
- The 16k run requires `--max-model-len >= input_len + output_len`; it was rerun with `MAX_MODEL_LEN=32768`.
