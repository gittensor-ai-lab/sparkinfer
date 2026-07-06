#!/usr/bin/env bash
set -euo pipefail
cd /workspace/qwen_awq_bench
source /workspace/qwen_awq_bench/hf_env.sh
source /workspace/qwen_awq_bench/sglang-venv/bin/activate
mkdir -p results/sglang_gptq_int4
MODEL=/workspace/qwen_awq_bench/models/qwen3-gptq-int4
RUNLOG=results/sglang_gptq_int4/run.log
exec > >(tee -a "$RUNLOG") 2>&1
COMMON=(--model-path "$MODEL" --trust-remote-code --context-length 32768 --quantization gptq_marlin --kv-cache-dtype fp8_e4m3 --batch-size 1 --output-len 128 --warmups 2 --log-level warning)
for ctx in 128 512 4096 16384; do
  out="results/sglang_gptq_int4/sglang_qwen_gptq_int4_fp8_ctx${ctx}_out128"
  echo ">> sglang gptq int4 fp8 ctx=$ctx $(date -u)"
  timeout 2400 python -m sglang.bench_one_batch "${COMMON[@]}" --input-len "$ctx" --result-filename "${out}.json" 2>&1 | tee "${out}.log"
  code=${PIPESTATUS[0]}
  echo "$code" > "${out}_exitcode.txt"
done
echo "done sglang gptq int4 fp8 $(date -u)"
