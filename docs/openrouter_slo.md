# OpenRouter operational SLO

This document is an operational target for a SparkInfer deployment, not a contractual service
level agreement. OpenRouter measures reliability and performance from requests at its own edge;
server-reported timing values are diagnostic only.

## Target

- Maintain **at least 95% measured uptime** after the first 100 requests. This is OpenRouter's
  threshold for normal routing. Authentication failures, payment failures, model-not-found
  responses, 5xx responses, mid-stream failures, and error finish reasons reduce that measurement.
- Reject excess load immediately with `429` instead of allowing queueing to inflate TTFT and lower
  externally measured throughput. `400`, `403`, `413`, and `429` do not count as uptime failures,
  although OpenRouter tracks rate limiting separately.
- Stream each token as soon as it is available and emit an SSE comment at least every 15 seconds
  while queueing or prefilling.
- Keep the model document accurate: model ID, creation time, prices, context/output limits,
  readiness, and declared request/token/concurrency capacity must describe the live deployment.

## Required deployment profile

Use `server/run_openrouter.sh`. It refuses to start until the model identity, creation timestamp,
and prompt/completion prices are configured, and supplies bounded production defaults:

```bash
export SPARKINFER_HF_MODEL_ID=gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090
export SPARKINFER_MODEL_CREATED=REPLACE_WITH_UPSTREAM_UNIX_TIMESTAMP
export SPARKINFER_QUANTIZATION=nvfp4
export SPARKINFER_PROMPT_PRICE_USD=0.0000001
export SPARKINFER_COMPLETION_PRICE_USD=0.0000003
export MODEL_NAME=your-org/your-model
export API_KEY="$OPENROUTER_PROVIDER_KEY"

# Optional declared limits for OpenRouter's schema v2.4 model monitor.
export SPARKINFER_REQUESTS_PER_MINUTE=120
export SPARKINFER_PROMPT_TOKENS_PER_MINUTE=500000
export SPARKINFER_COMPLETION_TOKENS_PER_MINUTE=60000

server/run_openrouter.sh /path/to/model-or-checkpoint-directory
```

Validate the profile without loading a model or opening a port with
`server/run_openrouter.sh --check-config`. The API key is checked for presence but never printed.

`SPARKINFER_MAX_QUEUE_DEPTH` defaults to `6` in this profile. Tune it from external concurrent
request measurements; it is both enforced by admission control and published as concurrency
capacity. `SPARKINFER_REQUEST_TIMEOUT_S` defaults to 300 seconds. Neither default applies when the
ordinary `server/run.sh` launcher is used.

## Monitoring and alerting

Poll `/health` for process/device health, `/v1/models` for launch readiness, `/v1/capacity` for
admission headroom, and `/metrics` for request outcomes and token totals. Alert before OpenRouter's
routing threshold is crossed:

- rolling provider success rate below 97%;
- any sustained 5xx or mid-stream failures;
- sustained `429` responses (not an uptime failure, but they reduce useful traffic);
- CUDA health failure or `is_ready: false`;
- TTFT or throughput regression measured outside the server.

OpenRouter's provider guide is the source of truth for its schema and routing calculations:
<https://openrouter.ai/docs/guides/community/for-providers>.
