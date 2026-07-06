#!/usr/bin/env bash
set -euo pipefail
cd /workspace/qwen_awq_bench/sparkinfer
mkdir -p /workspace/qwen_awq_bench/results/sparkinfer models
exec > >(tee -a /workspace/qwen_awq_bench/results/sparkinfer/run_auth.log) 2>&1
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
GGUF=/workspace/qwen_awq_bench/sparkinfer/models/Qwen3-30B-A3B-Q4_K_M.gguf
URL=https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf
echo "auth gguf download $(date -u)"
curl --netrc -fL -C - --progress-bar "$URL" -o "$GGUF"
ls -lh "$GGUF"
while pgrep -f "vllm bench latency" >/dev/null; do
  echo "waiting for vllm to finish before sparkinfer GPU bench $(date -u)"
  sleep 30
done
for ctx in 128 512 4096 16384; do
  echo ">> sparkinfer ctx=$ctx $(date -u)"
  bench/scripts/bench.sh "$GGUF" --tokens 128 --ctx "$ctx" 2>&1 | tee "/workspace/qwen_awq_bench/results/sparkinfer/sparkinfer_ctx${ctx}_out128.log"
done
echo "done sparkinfer $(date -u)"
