#!/usr/bin/env bash
set -uo pipefail
cd /workspace/qwen_awq_bench
source /workspace/qwen_awq_bench/hf_env.sh
source /workspace/qwen_awq_bench/trtllm-venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
mkdir -p results/trtllm_gptq_int4
LOG=results/trtllm_gptq_int4/build_w4a16_gptq_retry.log
exec > >(tee -a "$LOG") 2>&1
trtllm-bench \
  --model Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  --model_path /workspace/qwen_awq_bench/models/qwen3-gptq-int4 \
  --workspace /workspace/qwen_awq_bench/results/trtllm_gptq_int4 \
  --log_level info \
  build \
  --quantization W4A16_GPTQ \
  --max_seq_len 16512 \
  --max_batch_size 1 \
  --max_num_tokens 16512 \
  --trust_remote_code true
code=$?
echo "$code" > results/trtllm_gptq_int4/build_w4a16_gptq_retry_exitcode.txt
echo ">> exit=$code $(date -u)"
