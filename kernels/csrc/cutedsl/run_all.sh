#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${CUTEDSL_VENV:-${ROOT}/.venv}"
OUT="${CUTEDSL_OUT:-${ROOT}/cutedsl_bench_qwen_shape.json}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[cutedsl] python3 not found" >&2
  exit 1
fi

python3 -m venv "${VENV}"
# shellcheck disable=SC1091
. "${VENV}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${ROOT}/requirements-cu13.txt"

python "${ROOT}/run_smoke.py" --require-dsl
python "${ROOT}/bench_cutedsl.py" \
  --hidden 2048 \
  --experts 256 \
  --ffn 768 \
  --iters 30 \
  --warmup 5 \
  --output "${OUT}"

echo "[cutedsl] benchmark written to ${OUT}"

