#!/usr/bin/env bash
set -euo pipefail
cd /workspace/qwen_awq_bench
mkdir -p results/vllm_awq models/qwen3-awq
source /workspace/qwen_awq_bench/hf_env.sh
. vllm-awq-venv/bin/activate
{
  echo "explicit awq download via hf filenames $(date -u)"
  python - <<PY
import os
print("auth", bool(os.environ.get("HF_TOKEN")))
PY
} | tee -a results/vllm_awq/local_run.log
hf download cognitivecomputations/Qwen3-30B-A3B-AWQ \
  model-00001-of-00004.safetensors \
  model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors \
  model-00004-of-00004.safetensors \
  model.safetensors.index.json \
  config.json tokenizer.json tokenizer_config.json generation_config.json special_tokens_map.json vocab.json merges.txt \
  --local-dir /workspace/qwen_awq_bench/models/qwen3-awq \
  2>&1 | tee -a results/vllm_awq/local_run.log
ls -lh /workspace/qwen_awq_bench/models/qwen3-awq | tee -a results/vllm_awq/local_run.log
for ctx in 128 512 4096 16384; do
  echo ">> local awq ctx=$ctx $(date -u)" | tee -a results/vllm_awq/local_run.log
  set +e
  timeout 1800 vllm bench latency \
    --model /workspace/qwen_awq_bench/models/qwen3-awq \
    --trust-remote-code \
    --quantization awq \
    --input-len "$ctx" \
    --output-len 128 \
    --batch-size 1 \
    --num-iters-warmup 2 \
    --num-iters 8 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --kv-cache-dtype fp8_e4m3 \
    --disable-detokenize \
    --output-json "results/vllm_awq/vllm_awq_local_ctx${ctx}_out128.json" \
    > "results/vllm_awq/vllm_awq_local_ctx${ctx}_out128.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "results/vllm_awq/vllm_awq_local_ctx${ctx}_exitcode.txt"
  tail -100 "results/vllm_awq/vllm_awq_local_ctx${ctx}_out128.log" | tee -a results/vllm_awq/local_run.log || true
  python - <<PY | tee -a results/vllm_awq/local_run.log || true
import json, pathlib
p=pathlib.Path("results/vllm_awq/vllm_awq_local_ctx${ctx}_out128.json")
if p.exists():
    d=json.loads(p.read_text())
    lat=d.get("avg_latency") or d.get("latency_avg") or d.get("mean_latency")
    print("RESULT", ${ctx}, "latency", lat, "tok_s", (128/lat if lat else None))
PY
done
echo "done local vllm $(date -u)" | tee -a results/vllm_awq/local_run.log
