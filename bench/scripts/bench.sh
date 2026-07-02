#!/usr/bin/env bash
# Turnkey decode benchmark for sparkinfer on NVIDIA Blackwell.
#
#   bench/scripts/bench.sh [--download | <model.gguf>] [--tokens N] [--ctx N] [--compare]
#
#   --download   fetch Qwen3-30B-A3B Q4_K_M from Hugging Face (default if no GGUF given)
#   --tokens N   decode tokens to time (default 128)
#   --ctx N      prefill to this KV context depth before timing decode (default 0)
#   --compare    also build llama.cpp and run llama-bench on the same GGUF
#
# Auto-detects the GPU's CUDA arch, builds sparkinfer if needed, and prints tok/s.
# Env overrides: MODELS_DIR, MODEL_REPO, MODEL_FILE, ARCH, LLAMACPP_DIR.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

GGUF=""; TOKENS=128; CTX=0; COMPARE=0
while [ $# -gt 0 ]; do case "$1" in
  --download) GGUF="$MODELS_DIR/$MODEL_FILE" ;;
  --tokens)   shift; TOKENS="$1" ;;
  --ctx)      shift; CTX="$1" ;;
  --compare)  COMPARE=1 ;;
  -h|--help)  sed -n '2,9p' "$0"; exit 0 ;;
  *)          GGUF="$1" ;;
esac; shift; done
[ -z "$GGUF" ] && GGUF="$MODELS_DIR/$MODEL_FILE"

ARCH="$(detect_arch)"; echo ">> GPU arch: sm_$ARCH"
resolve_runner "$ARCH"     # prebuilt binaries if available, else build from source
[ "$GGUF" = "$MODELS_DIR/$MODEL_FILE" ] && ensure_model
[ -f "$GGUF" ] || { echo "!! GGUF not found: $GGUF  (pass a path or use --download)"; exit 1; }

echo; echo "=== sparkinfer — decode (n=$TOKENS, ctx=$CTX, bs=1) ==="
out="$(si_run qwen3_gguf_bench "$GGUF" "$TOKENS" "$CTX" 2>&1)" || true
echo "$out"
if ! echo "$out" | grep -q "decode tg"; then   # prebuilt incompatible (arch/driver/glibc) -> rebuild
  fallback_build "$ARCH"
  si_run qwen3_gguf_bench "$GGUF" "$TOKENS" "$CTX"
fi

if [ "$COMPARE" = 1 ]; then
  ensure_llamacpp "$ARCH"
  echo; echo "=== llama.cpp — decode (same GGUF, same GPU) ==="
  "$LLAMACPP_DIR/build/bin/llama-bench" -m "$GGUF" -p "$CTX" -n "$TOKENS" -ngl 99
fi
