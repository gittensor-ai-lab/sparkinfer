# TensorRT-LLM NVFP4 RTX 5090 Attempt — 2026-07-03

Target model: `nvidia/Qwen3-30B-A3B-NVFP4`

Target benchmark: `trtllm-bench`, 128 context, 128 generated tokens.

Result: no benchmark number.

What happened:

- TensorRT-LLM was not preinstalled on the Vast CUDA 13 image.
- The setup reached apt dependency installation for `libopenmpi-dev` and then PyTorch/CUDA wheel installation in a new TensorRT-LLM virtualenv.
- After ~11 minutes the process was still preparing/downloading PyTorch/CUDA wheels and had not reached `pip install tensorrt_llm` or `trtllm-bench`.
- The attempt was stopped to avoid continuing paid GPU time on setup-only work.

Next better path:

- Use NVIDIA's TensorRT-LLM NGC container directly on a host that supports that image, or prebuild a Vast image with TensorRT-LLM installed.
- Then run `trtllm-bench` against the same NVFP4 checkpoint and contexts.
