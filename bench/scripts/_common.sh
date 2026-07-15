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

# C2 (reference quarantine): pin the baseline artifacts so a tampered persisted copy can't skew a
# verdict. reference.lock carries MODEL_SHA256 + LLAMACPP_COMMIT; empty = warn-only until pinned.
_HERE_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_HERE_COMMON/reference.lock" ] && source "$_HERE_COMMON/reference.lock"
LLAMACPP_REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp}"
sha256_of() { sha256sum "$1" 2>/dev/null | awk '{print $1}'; }

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

_download_model() {
  echo ">> downloading $MODEL_REPO/$MODEL_FILE -> $MODELS_DIR (~17 GB) ..." >&2
  mkdir -p "$MODELS_DIR"
  # Try three download methods in order; fall back to plain curl (no HF tools needed).
  HF_HUB_DISABLE_XET=1 hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODELS_DIR" >&2 || \
  python3 -c "from huggingface_hub import hf_hub_download as d; d('$MODEL_REPO','$MODEL_FILE',local_dir='$MODELS_DIR')" >&2 || \
  curl -fL --progress-bar "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}" \
       -o "$MODELS_DIR/$MODEL_FILE" >&2
}

ensure_model() {
  [ -f "$MODELS_DIR/$MODEL_FILE" ] || _download_model
  verify_model
}

# C2: the persisted baseline weights must be pristine — a malicious root build could corrupt the GGUF
# to depress llama's score and inflate its own relative gain. Verify against the pinned sha each eval.
verify_model() {
  local f="$MODELS_DIR/$MODEL_FILE" got
  got="$(sha256_of "$f")"
  if [ -z "${MODEL_SHA256:-}" ]; then
    echo ">> model sha256 (not pinned, warn-only): $got" >&2; return 0
  fi
  if [ "$got" = "$MODEL_SHA256" ]; then echo ">> model sha256 OK" >&2; return 0; fi
  echo ">> WARN: model sha256 MISMATCH (got ${got:-none}, want $MODEL_SHA256) — re-fetching clean baseline" >&2
  rm -f "$f"; _download_model
  got="$(sha256_of "$f")"
  [ "$got" = "$MODEL_SHA256" ] || { echo ">> FATAL: model sha still wrong after re-download ($got)" >&2; return 1; }
  echo ">> model sha256 OK after re-fetch" >&2
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

# Reuse the persisted llama.cpp only if it's the pinned commit, a clean tree, and the binary still
# matches the hash recorded when it was built (a root PR build can't have swapped it). Else rebuild.
_llamacpp_clean() {  # $1=llama-bench  $2=sentinel
  if [ -n "${LLAMACPP_COMMIT:-}" ]; then
    [ "$(git -C "$LLAMACPP_DIR" rev-parse HEAD 2>/dev/null)" = "$(git -C "$LLAMACPP_DIR" rev-parse "$LLAMACPP_COMMIT^{commit}" 2>/dev/null)" ] || return 1
    [ -z "$(git -C "$LLAMACPP_DIR" status --porcelain --untracked-files=no 2>/dev/null)" ] || return 1
  fi
  [ -f "$2" ] && [ "$(sha256_of "$1")" = "$(cat "$2" 2>/dev/null)" ]
}

_llamacpp_binary_ok() {  # headless llama-bench is multi-MB; partial links are ~17KB
  local f="$1" sz
  [ -x "$f" ] || return 1
  sz="$(stat -c%s "$f" 2>/dev/null || wc -c <"$f" 2>/dev/null || echo 0)"
  [ "${sz:-0}" -gt 500000 ]
}

_llamacpp_purge_stale_build() {  # $1=bdir — drop UI tree when headless server build is required
  local bdir="$1"
  if [ -d "$bdir/tools/ui" ]; then
    echo ">> llama.cpp purge stale UI build tree (headless server) ..." >&2
    rm -rf "$bdir"
    return 0
  fi
  if [ -f "$bdir/CMakeCache.txt" ] && ! grep -q 'LLAMA_BUILD_UI:BOOL=OFF' "$bdir/CMakeCache.txt" 2>/dev/null; then
    echo ">> llama.cpp reconfigure (LLAMA_BUILD_UI=OFF for headless server) ..." >&2
    rm -rf "$bdir"
  fi
}

ensure_llamacpp() {  # $1 = arch ; builds llama-bench + llama-server, pinned + tamper-checked (C2)
  local bench="$LLAMACPP_DIR/build/bin/llama-bench" srv="$LLAMACPP_DIR/build/bin/llama-server"
  local sentinel="$LLAMACPP_DIR/.si_refhash"
  local arch="$1" bdir="$LLAMACPP_DIR/build"
  [ -x "$bench" ] && ! _llamacpp_binary_ok "$bench" && { echo ">> WARN: llama-bench looks truncated — rebuild" >&2; rm -f "$bench"; }
  # Stale partial trees (UI assets, old CMakeCache) leave llama-server broken while llama-bench exists.
  if [ ! -x "$srv" ] && { [ -d "$bdir/tools/ui" ] || [ -f "$bdir/CMakeCache.txt" ]; }; then
    echo ">> llama.cpp llama-server missing — purging stale build tree ..." >&2
    rm -rf "$bdir"
    rm -f "$sentinel"
  fi
  [ -x "$bench" ] && [ -x "$srv" ] && _llamacpp_clean "$bench" "$sentinel" && return 0
  echo ">> (re)building llama.cpp reference (CUDA sm_$arch) ..." >&2
  if [ -n "${LLAMACPP_COMMIT:-}" ]; then
    local head_ok=0
    if [ -d "$LLAMACPP_DIR/.git" ]; then
      [ "$(git -C "$LLAMACPP_DIR" rev-parse HEAD 2>/dev/null)" = \
        "$(git -C "$LLAMACPP_DIR" rev-parse "$LLAMACPP_COMMIT^{commit}" 2>/dev/null)" ] && head_ok=1
    fi
    if [ "$head_ok" != 1 ]; then
      rm -rf "$LLAMACPP_DIR"; mkdir -p "$LLAMACPP_DIR"
      git -C "$LLAMACPP_DIR" init -q
      git -C "$LLAMACPP_DIR" remote add origin "$LLAMACPP_REPO"
      git -C "$LLAMACPP_DIR" fetch -q --depth 1 origin "$LLAMACPP_COMMIT" >&2 || {
        echo ">> FATAL: cannot fetch pinned llama commit $LLAMACPP_COMMIT" >&2; return 1; }
      git -C "$LLAMACPP_DIR" checkout -q FETCH_HEAD
      rm -rf "$bdir"
    fi
  else
    [ -d "$LLAMACPP_DIR/.git" ] || git clone --depth=1 "$LLAMACPP_REPO" "$LLAMACPP_DIR" >&2
    echo ">> llama.cpp NOT pinned (warn-only) — HEAD $(git -C "$LLAMACPP_DIR" rev-parse --short HEAD 2>/dev/null); set LLAMACPP_COMMIT in reference.lock" >&2
    rm -rf "$bdir"
  fi
  _llamacpp_purge_stale_build "$bdir"
  if [ ! -f "$bdir/CMakeCache.txt" ]; then
    : > /tmp/llama_build.log
    if ! cmake -S "$LLAMACPP_DIR" -B "$bdir" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$arch" \
          -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF \
          $CUDA_HOST_FLAG >&2; then
      echo ">> FATAL: llama.cpp cmake configure failed" >&2; return 1
    fi
  fi
  # Build targets sequentially — parallel llama-bench + llama-server races on shared ggml objects.
  if [ ! -x "$bench" ]; then
    echo ">> building llama-bench ..." >&2
    if ! cmake --build "$bdir" -j2 --target llama-bench 2>&1 \
          | tee -a /tmp/llama_build.log | awk 'NR<=5 || NR%40==0 || /Built target|error:|FAILED|fatal error/' >&2; then
      echo ">> FATAL: llama-bench build failed" >&2
      grep -iE 'error:|fatal error:' /tmp/llama_build.log | tail -20 >&2 || tail -40 /tmp/llama_build.log >&2
      return 1
    fi
  fi
  if [ ! -x "$srv" ]; then
    echo ">> building llama-server (single-threaded) ..." >&2
    if ! cmake --build "$bdir" -j1 --target llama-server 2>&1 \
          | tee -a /tmp/llama_build.log | awk 'NR<=5 || NR%20==0 || /Built target|error:|FAILED|fatal error/' >&2; then
      echo ">> FATAL: llama-server build failed" >&2
      grep -iE 'error:|fatal error:' /tmp/llama_build.log | tail -20 >&2 || tail -40 /tmp/llama_build.log >&2
      return 1
    fi
  fi
  [ -x "$bench" ] && [ -x "$srv" ] || { echo ">> FATAL: llama binaries missing after build" >&2; return 1; }
  sha256_of "$bench" > "$sentinel" 2>/dev/null || true
  echo ">> llama.cpp reference ready ($(git -C "$LLAMACPP_DIR" rev-parse --short HEAD 2>/dev/null))" >&2
}

# ---- prebuilt binaries (GitHub release) with source-build fallback ----
PREBUILT_REPO="${PREBUILT_REPO:-gittensor-ai-lab/sparkinfer}"
PREBUILT_TAG="${PREBUILT_TAG:-latest}"  # latest = newest release that carries a matching binary asset
PREBUILT_TGZ="${PREBUILT_TGZ:-}"
PREBUILT_URL="${PREBUILT_URL:-}"
PREBUILT_DIR=""
SI_BIN=""; SI_LD=""   # set by resolve_runner: binary dir + LD_LIBRARY_PATH

resolve_prebuilt_release() {   # $1=arch. Sets PREBUILT_TAG/PREBUILT_TGZ/PREBUILT_URL/PREBUILT_DIR.
  local arch="$1" tag="$PREBUILT_TAG" asset="" url="" cache_tag=""

  if [ -n "$PREBUILT_URL" ]; then
    asset="${PREBUILT_TGZ:-$(basename "${PREBUILT_URL%%\?*}")}"
    cache_tag="${tag:-custom}"
  elif [ -n "$PREBUILT_TGZ" ]; then
    asset="$PREBUILT_TGZ"
    if [ "$tag" = "latest" ]; then
      if [[ "$asset" =~ ^sparkinfer-(v[0-9][^-]*)- ]]; then
        tag="${BASH_REMATCH[1]}"
        url="https://github.com/$PREBUILT_REPO/releases/download/$tag/$asset"
      else
        url="https://github.com/$PREBUILT_REPO/releases/latest/download/$asset"
      fi
    else
      url="https://github.com/$PREBUILT_REPO/releases/download/$tag/$asset"
    fi
    cache_tag="$tag"
  elif [ "$tag" = "latest" ]; then
    command -v python3 >/dev/null || return 1
    local resolved
    resolved="$(python3 - "$PREBUILT_REPO" "$arch" <<'PY'
import json
import sys
import urllib.request

repo, arch = sys.argv[1], sys.argv[2]
needle = f"linux-x86_64-cuda13-sm{arch}"
api = f"https://api.github.com/repos/{repo}/releases?per_page=20"
req = urllib.request.Request(
    api,
    headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "sparkinfer-bench-scripts",
    },
)
with urllib.request.urlopen(req, timeout=20) as resp:
    releases = json.load(resp)

for release in releases:
    if release.get("draft") or release.get("prerelease"):
        continue
    tag = release.get("tag_name", "")
    for asset in release.get("assets", []):
        name = asset.get("name", "")
        if name.startswith("sparkinfer-") and needle in name and name.endswith(".tar.gz"):
            print(f"{tag}\t{name}\t{asset['browser_download_url']}")
            raise SystemExit(0)

raise SystemExit(f"no sparkinfer prebuilt asset found for {needle}")
PY
)" || { echo ">> prebuilt release lookup failed" >&2; return 1; }
    IFS=$'\t' read -r tag asset url <<< "$resolved"
    cache_tag="$tag"
  else
    asset="sparkinfer-$tag-linux-x86_64-cuda13-sm${arch}.tar.gz"
    url="https://github.com/$PREBUILT_REPO/releases/download/$tag/$asset"
    cache_tag="$tag"
  fi

  [ -n "$url" ] || url="$PREBUILT_URL"
  [ -n "$asset" ] && [ -n "$url" ] || return 1
  PREBUILT_TAG="$tag"
  PREBUILT_TGZ="$asset"
  PREBUILT_URL="$url"
  PREBUILT_DIR="$ROOT/.prebuilt/${cache_tag}-sm${arch}/sparkinfer-bin"
}

try_prebuilt() {   # download+extract the release bundle; sets SI_BIN/SI_LD; returns 1 if unavailable
  local arch="$1" cache_root tgz_path
  [ "${NO_PREBUILT:-0}" = 1 ] && return 1
  resolve_prebuilt_release "$arch" || return 1
  cache_root="$(dirname "$PREBUILT_DIR")"
  tgz_path="$cache_root/$PREBUILT_TGZ"
  if [ ! -x "$PREBUILT_DIR/bin/qwen3_gguf_bench" ]; then
    command -v curl >/dev/null || return 1
    echo ">> fetching prebuilt $PREBUILT_TGZ from $PREBUILT_TAG ..." >&2
    mkdir -p "$cache_root"
    curl -fsSL "$PREBUILT_URL" -o "$tgz_path" 2>/dev/null || { echo ">> prebuilt download failed" >&2; return 1; }
    tar xzf "$tgz_path" -C "$cache_root" 2>/dev/null || return 1
  fi
  SI_BIN="$PREBUILT_DIR/bin"; SI_LD="$PREBUILT_DIR/lib"; return 0
}

resolve_runner() {   # $1=arch. Prefer an existing local build, else prebuilt, else build from source.
  if [ -x "$ROOT/build/runtime/qwen3_gguf_bench" ]; then SI_BIN="$ROOT/build/runtime"; SI_LD=""; echo ">> using local build" >&2; return; fi
  if try_prebuilt "$1"; then echo ">> using prebuilt binaries (will fall back to source build if incompatible)" >&2; return; fi
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

# ---- M1: pin the GPU clocks so tok/s is REPRODUCIBLE run-to-run (not merely same-box-cancelled) ----
# Warmup fixes the cold-clock artifact but leaves boost free to wander with temperature/power, so an
# absolute tok/s isn't reproducible by a third party. Locking the graphics clock to a fixed,
# sustainable value makes the number repeatable; the same-box delta% is unaffected either way. The
# pinned value is reported in the verdict + log so a verifier reproduces at the same clock.
# Best-effort: needs root (eval boxes are root); if the box forbids -lgc, fall back to warmup-only.
GPU_CLOCKS_PINNED=0; PINNED_GCLK=""
_supported_gclks() { nvidia-smi -q -d SUPPORTED_CLOCKS 2>/dev/null | sed -n 's/.*Graphics *: *\([0-9][0-9]*\) MHz.*/\1/p'; }
pin_clocks() {
  command -v nvidia-smi >/dev/null || return 0
  nvidia-smi -pm 1 >/dev/null 2>&1 || true                      # persistence mode (best-effort)
  local tgt="${SPARKINFER_PIN_GCLK:-}" cap="${SPARKINFER_PIN_GCLK_CAP:-2550}"
  if [ -z "$tgt" ]; then                                        # highest supported clock <= cap
    tgt=$(_supported_gclks | sort -n | awk -v c="$cap" '$1<=c{v=$1} END{print v}')
  fi
  [ -z "$tgt" ] && { echo ">> WARN: no supported graphics clocks found — clocks NOT pinned" >&2; return 0; }
  if nvidia-smi -lgc "$tgt,$tgt" >/dev/null 2>&1; then
    GPU_CLOCKS_PINNED=1; PINNED_GCLK="$tgt"
    echo ">> GPU graphics clock pinned to ${tgt} MHz (reproducible tok/s)" >&2
  else
    echo ">> WARN: could not lock GPU clocks (no permission?) — falling back to warmup-only" >&2
  fi
}
unpin_clocks() {
  [ "$GPU_CLOCKS_PINNED" = 1 ] || return 0
  nvidia-smi -rgc >/dev/null 2>&1 || true
}
