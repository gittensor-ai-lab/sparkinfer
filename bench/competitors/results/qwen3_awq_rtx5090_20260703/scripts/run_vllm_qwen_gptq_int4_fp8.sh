#!/usr/bin/env bash
set -euo pipefail
cd /workspace/qwen_awq_bench
source /workspace/qwen_awq_bench/hf_env.sh
source /workspace/qwen_awq_bench/vllm-awq-venv/bin/activate
mkdir -p models/qwen3-gptq-int4 results/vllm_gptq_int4
MODEL_ID=Qwen/Qwen3-30B-A3B-GPTQ-Int4
MODEL=/workspace/qwen_awq_bench/models/qwen3-gptq-int4
RUNLOG=results/vllm_gptq_int4/run.log
exec > >(tee -a "$RUNLOG") 2>&1
if [ ! -s "$MODEL/config.json" ] || ! ls "$MODEL"/*.safetensors >/dev/null 2>&1; then
  echo ">> downloading $MODEL_ID $(date -u)"
  hf download "$MODEL_ID" --local-dir "$MODEL"
fi
ls -lh "$MODEL" | sed -n 1,80p
COMMON=(--model "$MODEL" --trust-remote-code --output-len 128 --batch-size 1 --num-iters-warmup 2 --num-iters 8 --gpu-memory-utilization 0.90 --max-model-len 32768 --kv-cache-dtype fp8_e4m3 --disable-detokenize)
for ctx in 128 512 4096 16384; do
  out="results/vllm_gptq_int4/qwen_gptq_int4_fp8_ctx${ctx}_out128"
  echo ">> qwen gptq int4 fp8 ctx=$ctx $(date -u)"
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
echo "done qwen gptq int4 fp8 $(date -u)"
