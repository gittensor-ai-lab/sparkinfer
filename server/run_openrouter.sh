#!/usr/bin/env bash
# Production-oriented SparkInfer profile for an OpenRouter provider endpoint.
set -euo pipefail

require_env() {
    if [ -z "${!1:-}" ]; then
        echo "error: $1 is required for an OpenRouter deployment" >&2
        exit 2
    fi
}

require_env SPARKINFER_HF_MODEL_ID
require_env SPARKINFER_MODEL_CREATED
require_env SPARKINFER_QUANTIZATION
require_env SPARKINFER_PROMPT_PRICE_USD
require_env SPARKINFER_COMPLETION_PRICE_USD
require_env MODEL_NAME
require_env API_KEY

# OpenRouter includes provider-side queue time in its public throughput measurement and explicitly
# prefers an early 429 over a long queue. Six is the qualified single-5090 concurrency shape; an
# operator may override it after measuring their own card and workload.
export SPARKINFER_MAX_QUEUE_DEPTH="${SPARKINFER_MAX_QUEUE_DEPTH:-6}"
export SPARKINFER_REQUEST_TIMEOUT_S="${SPARKINFER_REQUEST_TIMEOUT_S:-300}"
export SPARKINFER_SSE_KEEPALIVE_SECONDS="${SPARKINFER_SSE_KEEPALIVE_SECONDS:-15}"
export SPARKINFER_ALWAYS_STREAM_USAGE="${SPARKINFER_ALWAYS_STREAM_USAGE:-1}"
export SPARKINFER_OPENROUTER_PROVIDER=1
export SPARKINFER_READ_TIMEOUT_S="${SPARKINFER_READ_TIMEOUT_S:-300}"
export SPARKINFER_WRITE_TIMEOUT_S="${SPARKINFER_WRITE_TIMEOUT_S:-300}"

positive_integer() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || {
        echo "error: $1 must be a positive integer (got '$2')" >&2
        exit 2
    }
}
positive_integer SPARKINFER_MODEL_CREATED "$SPARKINFER_MODEL_CREATED"
positive_integer SPARKINFER_MAX_QUEUE_DEPTH "$SPARKINFER_MAX_QUEUE_DEPTH"
positive_integer SPARKINFER_REQUEST_TIMEOUT_S "$SPARKINFER_REQUEST_TIMEOUT_S"

if [ "${1:-}" = "--check-config" ]; then
    printf '%s\n' "OpenRouter profile OK: model=$MODEL_NAME queue=$SPARKINFER_MAX_QUEUE_DEPTH timeout=${SPARKINFER_REQUEST_TIMEOUT_S}s"
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run.sh" "$@"
