#!/usr/bin/env bash
set -uo pipefail
cd /workspace/qwen_awq_bench
source /workspace/qwen_awq_bench/hf_env.sh
source /workspace/qwen_awq_bench/sglang-venv/bin/activate
mkdir -p results/sglang_gptq_int4_fix
MODEL=/workspace/qwen_awq_bench/models/qwen3-gptq-int4
out=results/sglang_gptq_int4_fix/try_maxprefill_nochunk_ctx4096_out128
python -m sglang.bench_one_batch \
  --model-path "$MODEL" \
  --trust-remote-code \
  --context-length 32768 \
  --quantization gptq_marlin \
  --kv-cache-dtype fp8_e4m3 \
  --batch-size 1 \
  --input-len 4096 \
  --output-len 128 \
  --warmups 2 \
  --max-prefill-tokens 4096 \
  --chunked-prefill-size -1 \
  --cuda-graph-backend-prefill disabled \
  --cuda-graph-backend-decode full \
  --log-level warning \
  --result-filename "${out}.json" 2>&1 | tee "${out}.log"
code=${PIPESTATUS[0]}
echo "$code" > "${out}_exitcode.txt"
echo exit=$code
