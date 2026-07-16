#!/usr/bin/env bash
set -uo pipefail
cd /workspace/qwen_awq_bench
mkdir -p results/trtllm_docker_nvfp4
IMG=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20
DATA=results/trtllm_docker_nvfp4/dataset_ctx128_out128_max512.json
LOG=results/trtllm_docker_nvfp4/latency_ctx128_out128_max512.log
OUTER=results/trtllm_docker_nvfp4/latency_ctx128_out128_max512_outer.log
REPORT=results/trtllm_docker_nvfp4/latency_ctx128_out128_max512.json
ITER=results/trtllm_docker_nvfp4/latency_ctx128_out128_max512_iter.jsonl
rm -f "$LOG" "$REPORT" "$ITER" "$OUTER"
docker run --rm \
  --gpus 'all,"capabilities=compute,utility"' \
  --ipc=host \
  --ulimit stack=67108864 \
  -v /workspace/qwen_awq_bench:/workspace/qwen_awq_bench \
  -w /workspace/qwen_awq_bench \
  "$IMG" bash -lc "
set -uo pipefail
trtllm-bench \\
  --model nvidia/Qwen3-30B-A3B-NVFP4 \\
  --model_path /workspace/qwen_awq_bench/models/qwen3-nvfp4 \\
  --workspace /workspace/qwen_awq_bench/results/trtllm_docker_nvfp4 \\
  prepare-dataset --output /workspace/qwen_awq_bench/$DATA --trust-remote-code token-unif-dist \\
  --num-requests 1 --input-min 128 --input-max 128 --output-min 128 --output-max 128
trtllm-bench \\
  --model nvidia/Qwen3-30B-A3B-NVFP4 \\
  --model_path /workspace/qwen_awq_bench/models/qwen3-nvfp4 \\
  --workspace /workspace/qwen_awq_bench/results/trtllm_docker_nvfp4 \\
  latency --backend _autodeploy --dataset /workspace/qwen_awq_bench/$DATA --max_seq_len 512 \\
  --num_requests 1 --warmup 0 --concurrency 1 \\
  --report_json /workspace/qwen_awq_bench/$REPORT --iteration_log /workspace/qwen_awq_bench/$ITER 2>&1 | tee /workspace/qwen_awq_bench/$LOG
exit \\${PIPESTATUS[0]}
" > "$OUTER" 2>&1
code=$?
echo "$code" > results/trtllm_docker_nvfp4/latency_ctx128_out128_max512_exitcode.txt
echo exit=$code
