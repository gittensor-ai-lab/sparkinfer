#!/usr/bin/env bash
# Probe max context for all Qwythos quants (llama.cpp reference on RTX 5090).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for q in Q4_K_M Q8_0 BF16; do
  echo ">> === quant $q ===" >&2
  PRIMARY_QUANT="$q" "$HERE/probe_max_ctx_llama.sh"
done
