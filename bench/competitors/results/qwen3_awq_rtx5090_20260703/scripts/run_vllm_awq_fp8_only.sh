#!/usr/bin/env bash
set -euo pipefail
cd /workspace/qwen_awq_bench
source /workspace/qwen_awq_bench/hf_env.sh
source /workspace/qwen_awq_bench/vllm-awq-venv/bin/activate
mkdir -p results/vllm_awq_fp8_only
MODEL=/workspace/qwen_awq_bench/models/qwen3-awq
COMMON=(--model "$MODEL" --trust-remote-code --quantization awq --output-len 128 --batch-size 1 --num-iters-warmup 2 --num-iters 8 --gpu-memory-utilization 0.90 --max-model-len 32768 --kv-cache-dtype fp8_e4m3 --disable-detokenize)
for ctx in 128 512 4096 16384; do
  out="results/vllm_awq_fp8_only/fp8_ctx${ctx}_out128"
  echo ">> fp8 ctx=$ctx $(date -u)"
  timeout 1800 vllm bench latency "${COMMON[@]}" --input-len "$ctx" --output-json "${out}.json" 2>&1 | tee "${out}.log"
  code=${PIPESTATUS[0]}
  echo "$code" > "${out}_exitcode.txt"
  python3 - "$ctx" "${out}.json" <<PY
import json, sys
ctx=sys.argv[1]; p=sys.argv[2]
d=json.load(open(p)); lat=d["avg_latency"]
print(f"RESULT ctx={ctx} latency={lat:.9f} tok_s={128/lat:.2f}")
PY
done
echo "done fp8 $(date -u)"
