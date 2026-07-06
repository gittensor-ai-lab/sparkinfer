# RTX 5090 Competitor Benchmarks

This folder is for competitor engine runs that are not apples-to-apples with
sparkinfer's native GGUF path unless explicitly marked as the same GGUF file.

sparkinfer's public baseline uses Qwen3-30B-A3B `Q4_K_M` GGUF. vLLM and
TensorRT-LLM normally benchmark Hugging Face quantized checkpoints such as FP8,
NVFP4, or GPTQ. Record the exact weight format with every result.

## Run

```bash
bench/competitors/run_vllm_trtllm_rtx5090.sh
```

Useful overrides:

```bash
OUT_DIR=/root/competitor-results \
VLLM_MODEL=nvidia/Qwen3-30B-A3B-NVFP4 \
TRTLLM_MODEL=nvidia/Qwen3-30B-A3B-NVFP4 \
CONTEXTS="128 512 4096 16384" \
bench/competitors/run_vllm_trtllm_rtx5090.sh
```

The harness tries vLLM first, then TensorRT-LLM. If TensorRT-LLM does not
support the selected checkpoint or the wheel stack on the host, the failure log
is kept in `OUT_DIR/trtllm/`.
