#!/usr/bin/env bash
set -uo pipefail
cd /workspace/qwen_awq_bench
IMG=nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20
OUTDIR=results/trtllm_docker_nvfp4_aligned_sweep
mkdir -p "$OUTDIR"
run_case() {
  local label="$1"
  local ctx="$2"
  local out=128
  local seq=$((ctx + out))
  local tag="${label}_in${ctx}_out${out}"
  local data="$OUTDIR/dataset_${tag}.json"
  local cfg="$OUTDIR/config_${tag}.yaml"
  local log="$OUTDIR/latency_${tag}.log"
  local outer="$OUTDIR/latency_${tag}_outer.log"
  local report="$OUTDIR/latency_${tag}.json"
  local iter="$OUTDIR/latency_${tag}_iter.jsonl"
  local exitfile="$OUTDIR/latency_${tag}_exitcode.txt"
  cat > "$cfg" <<CFG
max_batch_size: 1
max_num_tokens: ${seq}
cuda_graph_config:
  mode: decode
  batch_sizes: [1]
  max_batch_size: 1
  enable_padding: false
CFG
  rm -f "$data" "$log" "$outer" "$report" "$iter" "$exitfile"
  echo "[aligned] start label=$label input=$ctx out=$out max_seq_len=$seq" | tee -a "$OUTDIR/sweep.log"
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
  --workspace /workspace/qwen_awq_bench/$OUTDIR \\
  prepare-dataset --output /workspace/qwen_awq_bench/$data --trust-remote-code token-unif-dist \\
  --num-requests 6 --input-min $ctx --input-max $ctx --output-min $out --output-max $out
trtllm-bench \\
  --model nvidia/Qwen3-30B-A3B-NVFP4 \\
  --model_path /workspace/qwen_awq_bench/models/qwen3-nvfp4 \\
  --workspace /workspace/qwen_awq_bench/$OUTDIR \\
  latency --backend _autodeploy --config /workspace/qwen_awq_bench/$cfg --dataset /workspace/qwen_awq_bench/$data --max_seq_len $seq \\
  --num_requests 6 --warmup 1 --concurrency 1 \\
  --report_json /workspace/qwen_awq_bench/$report --iteration_log /workspace/qwen_awq_bench/$iter 2>&1 | tee /workspace/qwen_awq_bench/$log
exit \\${PIPESTATUS[0]}
" > "$outer" 2>&1
  local code=$?
  echo "$code" > "$exitfile"
  echo "[aligned] done label=$label code=$code" | tee -a "$OUTDIR/sweep.log"
  return "$code"
}
: > "$OUTDIR/sweep.log"
status=0
run_case total4096 3968 || status=$?
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$OUTDIR/sweep.log"
sleep 5
run_case total16384 16256 || status=$?
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$OUTDIR/sweep.log"
exit "$status"
