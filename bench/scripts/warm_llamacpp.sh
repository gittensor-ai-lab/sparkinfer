#!/usr/bin/env bash
# Pre-build pinned llama.cpp on the eval box (once per session). vast_eval calls this after setup.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$HERE/_common.sh"
export LLAMACPP_DIR="${LLAMACPP_DIR:-/workspace/.llamacpp}"
ARCH="$(detect_arch)"
ensure_llamacpp "$ARCH"
echo ">> warm_llamacpp: ready (sm_$ARCH)"
