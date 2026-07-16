#!/usr/bin/env bash
# Benchmark vLLM and TensorRT-LLM on an RTX 5090 with quantized Qwen MoE weights.
#
# This intentionally lives outside sparkinfer's runtime benchmark scripts because
# these engines usually use HF FP8/NVFP4/GPTQ weights rather than sparkinfer's
# GGUF Q4_K_M file. Keep the model format in the output when reporting results.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/bench/competitors/results/$(date -u +%Y%m%dT%H%M%SZ)}"
CONTEXTS="${CONTEXTS:-128 512 4096 16384}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
VLLM_MODEL="${VLLM_MODEL:-nvidia/Qwen3-30B-A3B-NVFP4}"
TRTLLM_MODEL="${TRTLLM_MODEL:-nvidia/Qwen3-30B-A3B-NVFP4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
SKIP_VLLM="${SKIP_VLLM:-0}"
SKIP_TRTLLM="${SKIP_TRTLLM:-0}"

mkdir -p "$OUT_DIR"/{vllm,trtllm,system}

log() { printf '\n>> %s\n' "$*" | tee -a "$OUT_DIR/run.log"; }
run_logged() {
  local name="$1"; shift
  log "$name"
  ("$@") 2>&1 | tee "$OUT_DIR/$name.log"
}

python_bin() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
  else
    command -v python
  fi
}

PYTHON="$(python_bin)"

log "writing system metadata to $OUT_DIR/system"
{
  date -u
  uname -a
  "$PYTHON" --version || true
} > "$OUT_DIR/system/host.txt"
nvidia-smi > "$OUT_DIR/system/nvidia-smi.txt" 2>&1 || true
nvcc --version > "$OUT_DIR/system/nvcc.txt" 2>&1 || true

install_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    "$PYTHON" -m pip install --upgrade pip
    "$PYTHON" -m pip install --retries 20 --timeout 120 uv
  fi
}

uv_in_venv() {
  local venv="$1"; shift
  VIRTUAL_ENV="$venv" PATH="$venv/bin:$PATH" uv pip "$@"
}

install_vllm() {
  install_uv
  if [ ! -x "$OUT_DIR/vllm-env/bin/python" ]; then
    uv venv "$OUT_DIR/vllm-env"
  fi
  # Nightly wheels are preferred for Blackwell/NVFP4 support. If that fails,
  # fall back to the public release so the log records the compatibility state.
  if ! uv_in_venv "$OUT_DIR/vllm-env" show vllm >/dev/null 2>&1; then
    uv_in_venv "$OUT_DIR/vllm-env" install \
      --extra-index-url https://wheels.vllm.ai/nightly \
      "vllm[bench]" || \
    uv_in_venv "$OUT_DIR/vllm-env" install \
      "vllm[bench]"
  fi
  uv_in_venv "$OUT_DIR/vllm-env" freeze > "$OUT_DIR/vllm/pip-freeze.txt"
}

run_vllm() {
  install_vllm
  local vllm_bin="$OUT_DIR/vllm-env/bin/vllm"
  [ -x "$vllm_bin" ] || { echo "vLLM executable not found after install: $vllm_bin" >&2; return 1; }
  "$vllm_bin" bench latency --help > "$OUT_DIR/vllm/bench-latency-help.txt" 2>&1 || true

  for ctx in $CONTEXTS; do
    local json="$OUT_DIR/vllm/vllm_ctx${ctx}_out${OUTPUT_TOKENS}.json"
    local log_file="$OUT_DIR/vllm/vllm_ctx${ctx}_out${OUTPUT_TOKENS}.log"
    log "vLLM latency: model=$VLLM_MODEL ctx=$ctx out=$OUTPUT_TOKENS"
    set +e
    "$vllm_bin" bench latency \
      --model "$VLLM_MODEL" \
      --trust-remote-code \
      --input-len "$ctx" \
      --output-len "$OUTPUT_TOKENS" \
      --batch-size 1 \
      --num-iters-warmup 2 \
      --num-iters 8 \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --disable-detokenize \
      --output-json "$json" \
      > "$log_file" 2>&1
    local rc=$?
    set -e
    echo "$rc" > "$OUT_DIR/vllm/vllm_ctx${ctx}_exitcode.txt"
    tail -80 "$log_file" || true
  done
}

install_trtllm() {
  install_uv
  if [ ! -x "$OUT_DIR/trtllm-env/bin/python" ]; then
    uv venv "$OUT_DIR/trtllm-env"
  fi
  if ! uv_in_venv "$OUT_DIR/trtllm-env" show tensorrt_llm >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y libopenmpi-dev git-lfs
    uv_in_venv "$OUT_DIR/trtllm-env" install \
      torch torchvision --index-url https://download.pytorch.org/whl/cu130
    uv_in_venv "$OUT_DIR/trtllm-env" install \
      tensorrt_llm
  fi
  uv_in_venv "$OUT_DIR/trtllm-env" freeze > "$OUT_DIR/trtllm/pip-freeze.txt"
}

run_trtllm() {
  install_trtllm
  local trt_bench="$OUT_DIR/trtllm-env/bin/trtllm-bench"
  "$trt_bench" --help > "$OUT_DIR/trtllm/trtllm-bench-help.txt" 2>&1 || true

  for ctx in $CONTEXTS; do
    local run_dir="$OUT_DIR/trtllm/ctx${ctx}_out${OUTPUT_TOKENS}"
    mkdir -p "$run_dir"
    log "TensorRT-LLM bench: model=$TRTLLM_MODEL ctx=$ctx out=$OUTPUT_TOKENS"
    set +e
    "$trt_bench" \
      --model "$TRTLLM_MODEL" \
      throughput \
      --backend pytorch \
      --dataset "token-norm-dist://input_mean=${ctx},input_stdev=0,output_mean=${OUTPUT_TOKENS},output_stdev=0,num_requests=16" \
      --output_dir "$run_dir" \
      > "$run_dir/trtllm.log" 2>&1
    local rc=$?
    set -e
    echo "$rc" > "$run_dir/exitcode.txt"
    tail -100 "$run_dir/trtllm.log" || true
  done
}

if [ "$SKIP_VLLM" != 1 ]; then
  run_vllm || true
fi

if [ "$SKIP_TRTLLM" != 1 ]; then
  run_trtllm || true
fi

log "done: $OUT_DIR"
