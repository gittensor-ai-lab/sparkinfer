#!/usr/bin/env bash
# Shared helpers for the sparkinfer bench / accuracy scripts.
# Sourced by bench.sh and accuracy.sh. Everything auto-detects / auto-builds so a
# contributor can run a single command on a fresh Blackwell box.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root (bench/scripts -> root)
MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-30B-A3B-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3-30B-A3B-Q4_K_M.gguf}"
TOK_REPO="${TOK_REPO:-Qwen/Qwen3-30B-A3B}"
LLAMACPP_DIR="${LLAMACPP_DIR:-$ROOT/.llamacpp}"   # override to reuse an existing checkout

# compute capability -> CUDA arch (12.0 -> 120). RTX 5090 / PRO 6000 = 120, Spark/Thor = 121.
detect_arch() {
  local cc; cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
  echo "${ARCH:-${cc:-120}}"
}

# nvcc 12.8 fails against Ubuntu 24.04's GCC 13.3 libstdc++ (cstdio / __gnu_cxx errors). Pin the
# CUDA host compiler to g++-12 (a fully supported combo) when it's available.
CUDA_HOST_FLAG=""
[ -x /usr/bin/g++-12 ] && CUDA_HOST_FLAG="-DCMAKE_CUDA_HOST_COMPILER=g++-12"

ensure_sparkinfer() {  # $1 = arch
  [ -x "$ROOT/build/runtime/qwen3_gguf_bench" ] && [ -x "$ROOT/build/runtime/qwen3_gguf_score" ] && return
  echo ">> building sparkinfer (sm_$1) ..." >&2
  cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_CUDA_ARCHITECTURES="$1" -DCMAKE_BUILD_TYPE=Release $CUDA_HOST_FLAG >/dev/null
  # Cap at 2 parallel jobs — cc1plus for sm_120 uses ~2-3 GB RAM each; -j4 OOMs on 64GB eval boxes.
  cmake --build "$ROOT/build" -j2 >/dev/null
}

ensure_model() {
  [ -f "$MODELS_DIR/$MODEL_FILE" ] && return
  echo ">> downloading $MODEL_REPO/$MODEL_FILE -> $MODELS_DIR (~17 GB) ..." >&2
  mkdir -p "$MODELS_DIR"
  # Try three download methods in order; fall back to plain curl (no HF tools needed).
  HF_HUB_DISABLE_XET=1 hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODELS_DIR" >&2 || \
  python3 -c "from huggingface_hub import hf_hub_download as d; d('$MODEL_REPO','$MODEL_FILE',local_dir='$MODELS_DIR')" >&2 || \
  curl -fL --progress-bar "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}" \
       -o "$MODELS_DIR/$MODEL_FILE" >&2
}

ensure_tokenizer() {
  [ -f "$MODELS_DIR/tokenizer.json" ] && return
  echo ">> downloading tokenizer.json ..." >&2
  mkdir -p "$MODELS_DIR"
  HF_HUB_DISABLE_XET=1 hf download "$TOK_REPO" tokenizer.json --local-dir "$MODELS_DIR" >&2 || \
  python3 -c "from huggingface_hub import hf_hub_download as d; d('$TOK_REPO','tokenizer.json',local_dir='$MODELS_DIR')" >&2 || \
  curl -fL --progress-bar "https://huggingface.co/${TOK_REPO}/resolve/main/tokenizer.json" \
       -o "$MODELS_DIR/tokenizer.json" >&2
}

ensure_llamacpp() {  # $1 = arch ; builds llama-bench + llama-server (one-time, slow)
  [ -x "$LLAMACPP_DIR/build/bin/llama-bench" ] && [ -x "$LLAMACPP_DIR/build/bin/llama-server" ] && return
  echo ">> building llama.cpp (CUDA sm_$1) — one-time, several minutes ..." >&2
  [ -d "$LLAMACPP_DIR/.git" ] || git clone --depth=1 https://github.com/ggml-org/llama.cpp "$LLAMACPP_DIR" >&2
  cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$1" \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF $CUDA_HOST_FLAG >/dev/null 2>&1
  cmake --build "$LLAMACPP_DIR/build" -j4 --target llama-bench llama-server >/dev/null 2>&1
}

# ---- prebuilt binaries (GitHub release) with source-build fallback ----
PREBUILT_TAG="${PREBUILT_TAG:-v0.2.0}"
PREBUILT_TGZ="${PREBUILT_TGZ:-sparkinfer-v0.2.0-linux-x86_64-cuda13-sm120.tar.gz}"
PREBUILT_URL="${PREBUILT_URL:-https://github.com/gittensor-ai-lab/sparkinfer/releases/download/$PREBUILT_TAG/$PREBUILT_TGZ}"
PREBUILT_DIR="$ROOT/.prebuilt/sparkinfer-bin"
SI_BIN=""; SI_LD=""   # set by resolve_runner: binary dir + LD_LIBRARY_PATH

try_prebuilt() {   # download+extract the release bundle; sets SI_BIN/SI_LD; returns 1 if unavailable
  [ "${NO_PREBUILT:-0}" = 1 ] && return 1
  if [ ! -x "$PREBUILT_DIR/bin/qwen3_gguf_bench" ]; then
    command -v curl >/dev/null || return 1
    echo ">> fetching prebuilt $PREBUILT_TGZ ..." >&2
    mkdir -p "$ROOT/.prebuilt"
    curl -fsSL "$PREBUILT_URL" -o "$ROOT/.prebuilt/$PREBUILT_TGZ" 2>/dev/null || { echo ">> prebuilt download failed" >&2; return 1; }
    tar xzf "$ROOT/.prebuilt/$PREBUILT_TGZ" -C "$ROOT/.prebuilt" 2>/dev/null || return 1
  fi
  SI_BIN="$PREBUILT_DIR/bin"; SI_LD="$PREBUILT_DIR/lib"; return 0
}

resolve_runner() {   # $1=arch. Prefer an existing local build, else prebuilt, else build from source.
  if [ -x "$ROOT/build/runtime/qwen3_gguf_bench" ]; then SI_BIN="$ROOT/build/runtime"; SI_LD=""; echo ">> using local build" >&2; return; fi
  if try_prebuilt; then echo ">> using prebuilt binaries (will fall back to source build if incompatible)" >&2; return; fi
  ensure_sparkinfer "$1"; SI_BIN="$ROOT/build/runtime"; SI_LD=""
}

fallback_build() {   # $1=arch. Switch the runner to a fresh source build (prebuilt didn't work here).
  echo ">> prebuilt unusable on this box — building from source ..." >&2
  ensure_sparkinfer "$1"; SI_BIN="$ROOT/build/runtime"; SI_LD=""
}

si_run() {   # si_run <tool> <args...>  — run a sparkinfer binary with the resolved lib path
  if [ -n "$SI_LD" ]; then LD_LIBRARY_PATH="$SI_LD:${LD_LIBRARY_PATH:-}" "$SI_BIN/$1" "${@:2}"
  else "$SI_BIN/$1" "${@:2}"; fi
}
