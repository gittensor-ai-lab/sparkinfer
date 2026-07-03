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

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"

pip_install() {
  local attempt
  for attempt in 1 2 3 4 5; do
    if python -m pip install \
      --retries 20 \
      --timeout "${PIP_DEFAULT_TIMEOUT}" \
      --trusted-host pypi.org \
      --trusted-host files.pythonhosted.org \
      "$@"; then
      return 0
    fi
    echo "[cutedsl] pip install attempt ${attempt} failed; retrying ..." >&2
    sleep $((attempt * 5))
  done
  return 1
}

if [ "${CUTEDSL_UPGRADE_PIP:-0}" = "1" ]; then
  pip_install --upgrade pip setuptools wheel
fi
pip_install -r "${ROOT}/requirements-cu13.txt"

python "${ROOT}/run_smoke.py" --require-dsl
python "${ROOT}/bench_cutedsl.py" \
  --hidden 2048 \
  --experts 256 \
  --ffn 768 \
  --iters 30 \
  --warmup 5 \
  --output "${OUT}"

echo "[cutedsl] benchmark written to ${OUT}"
