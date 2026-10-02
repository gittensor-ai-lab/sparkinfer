# sparkinfer-server

OpenAI-compatible HTTP API for local GGUF inference — backend for sparkinfer.com.

Enable with `-DBUILD_SERVER=ON` when building this repo (`main`).

## Build

Requires **Rust/cargo** (build-time only) for HuggingFace `tokenizer.json` via [tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp).

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_SERVER=ON
cmake --build build -j$(nproc) --target sparkinfer_server
```

Pull-request and release CI packages the Linux server with its runtime libraries and includes it
in GitHub Artifact Attestations. After downloading the bundle, verify the server provenance with:

```bash
gh attestation verify sparkinfer-bin/bin/sparkinfer_server -R gittensor-ai-lab/sparkinfer
```

Or reuse the bench harness build root:

```bash
bench/scripts/_common.sh  # optional: sets ARCH
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_SERVER=ON
cmake --build build --target sparkinfer_server
```

## Run

```bash
export SPARKINFER_ROOT="$(pwd)"
# Native C++ tokenizer (tokenizers-cpp). Requires rustc/cargo at build time only.

# download model on first bench run, or:
# bench/scripts/bench.sh --download

# Default: unsloth/Qwen3.6-35B-A3B-GGUF UD-Q4_K_M (~22 GB)
# https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
./server/run.sh --download
# or:
./build/server/sparkinfer_server -m models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --port 8080
```

### Serve Qwen3.8 with speculative decoding

The release container speculates by default: it downloads the target and z-lab's DFlash2 drafter
and starts the OpenAI-compatible server with both (`-e SPEC_DRAFT=none` serves without one;
`serve-dspark` uses the DSpark drafter instead):

```bash
docker run --gpus all -p 8080:8080 -v qwen38:/models \
  ghcr.io/gittensor-ai-lab/sparkinfer-qwen38:latest
```

For a source build, pass the drafter directory explicitly:

```bash
./build/server/sparkinfer_server \
  -m models/Qwen3.8-27B-NVFP4-RTX5090 \
  --tokenizer models/Qwen3.8-27B-NVFP4-RTX5090/tokenizer.json \
  --draft-model models/Qwen3.8-27B-DFlash2 \
  --ctx 131072 --host 0.0.0.0 --port 8080
```

The KV pool is sized for the whole `--ctx` before the drafter loads. On a 32 GB card, 262,144 tokens
leaves no device memory for the drafter, which is why this example and the container use 131,072.

- **A drafter you ask for is required:** a missing or incompatible checkpoint, or one that does
  not fit in device memory, terminates the server rather than quietly changing performance. The
  container's default drafter is the exception: if it cannot be downloaded the container serves
  without it.
- **What speculates:** up to `SPARKINFER_SPEC_GROUP` (8) fresh requests at a time, greedy or
  sampled. A request speculates from its start up to the end of the drafter's context
  (`SPARKINFER_DSPARK_MAX_CTX`): the whole `--ctx` for DFlash2, whose layers all attend a sliding
  window, and 16,384 positions for DSpark. A member that reaches it stays in its group and
  verifies one token a step without a draft. With `SPARKINFER_SPEC_GROUP=1` a request whose prompt
  plus `max_tokens` exceeds that context does not speculate at all.
- **What does not:** requests with tools, `response_format` JSON schemas or other constraints,
  vision, penalties, logit bias, logprobs, forced tokens or a prefix session take the ordinary
  path, as do all requests while more are live than a group takes.
- **Same tokens:** a request speculating alone gives the tokens ordinary decode gives (greedy, or
  sampled with a seed). A group verifies with batch arithmetic, as packed decode does.
- **Observability:** `/metrics` exposes `sparkinfer_speculative_runs_total`,
  `sparkinfer_speculative_tokens_total` and `sparkinfer_speculative_handoffs_total`.

### Serve a GGUF instead of NVFP4

The server also loads Qwen3.8-27B from a GGUF, for example unsloth's
[`Qwen3.8-27B-UD-Q4_K_M.gguf`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (16.5 GB). Point
`-m` at the file. GGUF repos do not ship a `tokenizer.json`, so pass the one from any Qwen3.8 HF
checkpoint:

```bash
./build/server/sparkinfer_server \
  -m models/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --tokenizer models/Qwen3.8-27B-NVFP4-RTX5090/tokenizer.json \
  --ctx 32768 --host 0.0.0.0 --port 8080
```

Compared with the NVFP4 checkpoints:

- **Text only.** The vision tower loads from an HF checkpoint directory, so a GGUF target returns
  `400` for image and video input.
- **Slower prefill on short prompts.** Reading Q4_K means dequantizing it into the matrix-multiply
  operand on every prefill pass. That fixed cost dominates at 128 tokens and is amortized by 4k; see
  [Same weights, GGUF on both sides](../README.md#same-weights-gguf-on-both-sides) for the numbers.
- **DSpark is only measured with the NVFP4 target.**
- **The release container serves the NVFP4 checkpoint.** A GGUF target needs a source build.

## API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | `{"status":"ok"}` |
| `GET /v1/models` | OpenAI model list plus OpenRouter provider schema v2.4 capabilities (includes live `context_length`) |
| `GET /v1/info` | Model limits (`max_context`, `max_output_tokens`) — live values, not build-time constants |
| `GET /v1/capacity` | This worker's live occupancy: `active_requests`, `free_kv_blocks`, `max_queue_depth`, `accepting_requests`. Single-process only — not fleet-wide. |
| `GET /metrics` | Prometheus text-exposition counters/gauges: request totals by outcome (`ok`/`client_error`/`overloaded`/`timeout`/`cancelled`/`server_error`), prompt/completion token totals, active requests, free KV blocks, uptime. |
| `POST /v1/tokenize` | Token count for a chat request body |
| `POST /v1/completions` | Legacy OpenAI text completion (`prompt` string, `echo`, integer `logprobs`). `echo` prepends the prompt TEXT; it does not report per-prompt-token logprobs — use `/v1/score` for that. |
| `POST /v1/score` | **Teacher-forced scoring.** Per-token logprobs of a *supplied* continuation, no generation. See below. |
| `POST /v1/chat/completions` | Chat (JSON `messages`, optional `tools`, `tool_choice`, `stream`, `enable_thinking`, `reasoning`, or `reasoning_effort`). Responses include OpenAI `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`) plus additive GPU timing fields (`ttft_ms`, `generation_ms`, `decode_tps`) that standard OpenAI SDKs ignore. Streaming sends a final chunk with `choices:[]` + `usage` before `[DONE]` by default. A streaming client that disconnects mid-response cancels generation (checked via `DataSink::is_writable()`) instead of running to completion for nobody. Overload (no queue capacity) returns `429`; a request that exceeds `SPARKINFER_REQUEST_TIMEOUT_S` returns `504`. |
| `POST /v1/messages` | **Anthropic Messages API**, streaming and non-streaming: text, base64 images, tool use, thinking. A translation onto `/v1/chat/completions`, so it shares that route's generation, tool validation and sampling. Auth: `x-api-key` or `Authorization: Bearer`. See [Anthropic Messages and OpenAI Responses](#anthropic-messages-and-openai-responses). |
| `POST /v1/messages/count_tokens` | Anthropic token count for a messages body, including `system` and `tools`. Image content returns `400`, as on `/v1/tokenize`. |
| `POST /v1/responses` | **OpenAI Responses API**, stateless: streaming and non-streaming text, images, function tools and reasoning. Nothing is stored, so `previous_response_id` and `conversation` return `400` and `GET /v1/responses/{id}` returns `404`; send the whole conversation in `input`. |

### OpenRouter provider configuration

`/v1/models` exposes the typed v2.4 text capabilities OpenRouter consumes. Set the deployment
identity and per-token USD prices explicitly; unconfigured prices are omitted rather than reported
as zero:

```bash
export SPARKINFER_HF_MODEL_ID=gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090
export SPARKINFER_QUANTIZATION=nvfp4
export SPARKINFER_PROMPT_PRICE_USD=0.0000001
export SPARKINFER_COMPLETION_PRICE_USD=0.0000003
```

For a production provider endpoint, use `server/run_openrouter.sh` and follow the
[OpenRouter operational SLO](../docs/openrouter_slo.md). The profile requires stable model
metadata, bounds admission at six concurrent requests by default, enables a request deadline and
keeps streaming heartbeats/usage enabled. It also enables strict provider-schema mode, omitting
OpenAI's legacy `object`, `owned_by`, and root `context_length` fields because OpenRouter v2.4
rejects unknown properties. These are deployment defaults, not a contractual uptime guarantee.

SSE comment heartbeats are emitted every 15 seconds during queueing and long prefill. Configure
that with `SPARKINFER_SSE_KEEPALIVE_SECONDS` (`0` disables it). Streaming usage is always emitted
by default, even when the caller omits `stream_options.include_usage`; set
`SPARKINFER_ALWAYS_STREAM_USAGE=0` only for legacy deployments that require opt-in usage chunks.
`tool_choice` accepts `auto`, `none`, `required`, and OpenAI's named-function object form;
`parallel_tool_calls=false` enforces at most one returned call. Tools and structured output are
supported independently, but their combination is rejected because it has no unambiguous output
grammar in this server.

### Timing fields

The additive fields in `usage` measure work inside the inference engine, not end-to-end HTTP
latency:

| field | definition |
|---|---|
| `ttft_ms` | Time from successful engine admission (after request parsing, tokenisation and session/KV setup) until the first token is ready for the HTTP layer. |
| `generation_ms` | Time from successful engine admission until generation finishes or is cancelled. |
| `decode_tps` | `completion_tokens / (generation_ms - ttft_ms)`, with the denominator floored at 1 ms. This includes the first token in the numerator and is therefore an engine compatibility metric, not the conventional post-first-token rate. |

These values exclude request parsing, tokenisation, admission setup, SSE writes, network transit and
client backpressure. They are useful for operator telemetry, but a remote service can report
arbitrary values: gateways, auditors and billing systems must measure latency and throughput at
their own trust boundary rather than using these fields for rewards or accounting.

### Streaming logprobs

With `stream: true` and `logprobs: true`, each chunk's `logprobs.content` holds the entries for the
tokens that produced *that chunk's* text, so concatenating them across the stream yields exactly one
entry per generated token — the same array the non-streaming response returns, in the same order.
Reasoning deltas carry their own tokens' entries (a sparkinfer extension: OpenAI has no
`reasoning_content`), and if generation ends with entries whose tokens produced no text of their
own, a final `content: ""` delta carries them rather than dropping them.

Two earlier defects here are fixed: entries for tokens routed to `reasoning_content` used to be
held back and attached to the next *content* chunk (so a chunk reported logprobs for tokens whose
text it did not contain), and any entries still pending at end of generation were dropped outright.

### Teacher-forced scoring (`POST /v1/score`)

Given a prompt and a completion you supply, returns the per-token logprob of each completion token
under the model — without generating anything. Use it to score text the model did not produce:
another server's answer, a perturbed reference, recorded traffic. Verifying a runtime by comparing
its *own* greedy output only ever probes one trajectory; scoring probes any of them, so there is no
fixed answer set to memorise.

```bash
curl -s http://127.0.0.1:8080/v1/score -H 'Content-Type: application/json' -d '{
  "model": "qwen3.6-35b-a3b",
  "messages": [{"role":"user","content":"What is the capital of France?"}],
  "completion": "The capital of France is Paris.",
  "top_logprobs": 2
}'
```

| field | |
|---|---|
| `messages` **or** `prompt` | exactly one. `messages` applies the chat template exactly as `/v1/chat/completions` does; `prompt` is raw text, no template. |
| `completion` **or** `completion_token_ids` | exactly one; ids win if both are given and bypass tokenisation entirely. |
| `top_logprobs` | optional, 0–20 alternatives per position. |
| `enable_thinking` | optional, same meaning as chat completions. |

Response: `tokens`, `token_ids`, `bytes` (raw UTF-8 bytes of each scored token), `logprobs` (natural log, generation order), `sum_logprob`,
optional `top_logprobs`, and the same `usage` block every other endpoint returns.

The numbers are **identical** to what `/v1/chat/completions` reports for the same token at the same
position — same logits, same fp32 log-sum-exp — because scoring runs the same forward pass and the
same extraction, substituting your token for the sampler's pick. Scoring costs about what
generating that completion would; it batches, 429s and reports timings like any other request. The
last completion token costs no forward pass (nothing is predicted after it).

### Determinism (`SPARKINFER_DETERMINISTIC=1`)

By default sparkinfer is **not** run-to-run reproducible: repeat the same greedy request and the
token sequence can fork and per-token logprobs move by a few tenths of a nat. Set
`SPARKINFER_DETERMINISTIC=1` for bit-reproducible output.

Measured on an RTX 5090, Qwen3.6-35B-A3B UD-Q4_K_M, 12 prompts × 3 repeats at 48 tokens:

| | default | `SPARKINFER_DETERMINISTIC=1` |
|---|---|---|
| bit-identical runs | 1/36 | **36/36** |
| token-sequence forks | 13/36 | **0/36** |
| mean logprob drift | 0.44 nats | **0** |
| max logprob drift | 1.43 nats | **0** |

It is also batch-invariant on the qualified configuration: a request returns the same token IDs and
logprobs whether the server is idle or serving concurrent traffic, so a server can be audited while
it is in use. This promise is scoped to the exact runtime build, model and tokenizer artifacts,
runtime configuration, and GPU model. Changing any of those requires requalification; it does not
promise equality across different GPU models or future commits.

The nondeterminism is not in decode — the decode path has no float atomics at all. It is in batched
prompt prefill, in two fp32 `atomicAdd` accumulations whose operand order is decided by hardware
arbitration: the routed MoE down-projection combine, and the split-K reduction in the narrow-N GEMM
used by the Gated-DeltaNet gate projections. Both differ by only a few ULP, but they feed discrete
top-k expert *routing* and int8 activation requant at the next layer, so over 40+ layers one
occasionally flips an expert and moves the logits enough to change the argmax — which is why the
symptom looks far larger than the cause, and why it is seeded by the prompt pass rather than decode.
A third, smaller source is the reported logprob's own normalizer, read off a CUB inclusive scan
whose decoupled look-back folds a timing-dependent number of tile aggregates. See
`kernels/include/sparkinfer/kernels/deterministic.h`.

Verified bit-identical at every prompt length up to a full 8k context (117 / 782 / 1922 / 2492 /
3822 / 7622 tokens, 3 repeats each).

Cost, same hardware and model: decode throughput is unchanged (decode is untouched); TTFT is
**+2% at short prompts and +8% at 2k**. Above ~2048 tokens the mode also turns the GQA-fused int8
MMA prefill attention off (`SPARKINFER_PREFILL_ATTN_GQA_RQH=1`), which costs some long-context
prefill throughput — see the warning below for why. Bit-exactness holds for a fixed qualified
configuration, including concurrent batches. A few launch geometries elsewhere are chosen from the
device's SM count, so two *different* GPU models are not promised to agree even in this mode.

> **Known defect, independent of this mode: GQA-fused int8 prefill attention above ~2048 tokens.**
> With int8 KV (which the server enables whenever `--ctx` ≥ 4096) and a prompt of 2048 tokens or
> more, the GQA-fused MMA prefill attention disagrees sharply with the token-loop reference, and
> not reproducibly. `qwen3_gguf_prefill_check` against that reference, Qwen3.6-35B-A3B on an RTX
> 5090, mean KL over 16 teacher-forced positions:
>
> | prefix | 1500 | 2000 | **2100** | 3000 | 4000 |
> |---|---|---|---|---|---|
> | default (fused) | 0.00043 | 0.00022 | **0.18672** | 0.20657 | 0.23978 |
> | `…GQA_RQH=1` | ~0.0001 | ~0.0001 | ~0.0001 | ~0.0001 | 0.00008 |
>
> The cliff is exactly at 2048 and it is not the RQH=3 tier's own `n_tokens >= 2048` gate: RQH=2 is
> chosen both below and above it and is equally wrong above, so the length dependence is inside
> `launch_attn_gqa`. Only `RQH=1`, which drops to the per-q-head fallback, is correct there.
> Reproduce with:
>
> ```bash
> SPARKINFER_KV_INT8=1 ./build/runtime/qwen3_gguf_prefill_check <model.gguf> 4000 16 <4000+ real token ids>
> ```
>
> This is left ON by default rather than silently switched off, because disabling it moves the
> long-context prefill throughput the eval scores against — that call belongs with whoever owns the
> kernel. Deterministic mode simply declines to build on top of it.

### Anthropic Messages and OpenAI Responses

`/v1/messages` and `/v1/responses` are translation layers, like the LM Studio (`/api/v0`) and
Ollama (`/api/*`) routes: each request is rewritten into the chat-completions shape, served by the
same handler as `/v1/chat/completions`, and the response or stream is rewritten back. Tool calling,
images, reasoning and every sampling control therefore behave exactly as on `/v1`.

Point a client at the server:

- **Anthropic SDKs and Claude Code**: base URL is the server root, e.g. `http://host:8080`. Claude
  Code reads `ANTHROPIC_BASE_URL`, and `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` for the key.
- **OpenAI SDKs and Responses clients**: base URL is `http://host:8080/v1`.

Streams use each API's own named events (`message_start` … `message_stop`; `response.created` …
`response.completed`) and end without `[DONE]`. Both include usage.

| | Anthropic `/v1/messages` | OpenAI `/v1/responses` |
|---|---|---|
| Reasoning on/off | `thinking: {type: enabled \| adaptive \| disabled}` | `reasoning.effort` (`none` turns it off) |
| Reasoning effort | `output_config.effort` | `reasoning.effort` |
| Reasoning output | `thinking` blocks, empty `signature` | `reasoning` items with `reasoning_text` content, empty `summary` |
| Structured output | `output_config.format` (`json_schema`) | `text.format` (`json_object`, `json_schema`) |
| Tool choice | `auto`, `any`, `tool`, `none`; `disable_parallel_tool_use` | `auto`, `required`, `none`, `{type: function, name}` |

When reasoning is not specified, the model's own default applies, the same as on `/v1`.

Refused with `400` rather than silently degraded:

- **Anthropic**: server tools (`web_search`, `bash`, `code_execution`, ...), which Anthropic runs
  itself; `document` and `search_result` blocks; images inside a `tool_result`; image `file` sources.
- **Responses**: `previous_response_id`, `conversation` and `item_reference`, since nothing is stored;
  `background` mode; non-function tools; `input_file` and `file_id` images.

Not carried across: Anthropic `stop_reason` is `end_turn` for both EOS and a stop sequence, because
the chat handler's `finish_reason: "stop"` does not tell them apart. A `tool_result` with
`is_error: true` reaches the model as content prefixed `Error: `.

### Function tools

Qwen3.6 accepts OpenAI function definitions in `tools`, assistant `tool_calls` history, and
matching `role: "tool"` results. Both streaming and non-streaming responses expose native model
calls as `message.tool_calls` / `delta.tool_calls` with `finish_reason: "tool_calls"`; native XML
control markup is validated against the offered schema and is never exposed to clients.

`tool_choice` accepts omitted, `"auto"`, `"none"`, `"required"`, and OpenAI's named-function
object form. `"required"` and a named function are enforced, not just requested. With thinking
off, the assistant turn starts inside a tool call: `<tool_call>` plus `<function=NAME>` for a named
function or a single offered one, and `<tool_call>` plus `<function=` for `"required"` over several.
With thinking on, the model reasons first. If the attempt has no call to an offered function (the
model answered in prose, or wrote a name that is not offered), the server generates once more from
the model's reasoning with the call already opened on an offered function. For `"required"` over
several, that function is the offered name the model itself ranks highest, chosen one token at a
time with `logit_bias` restricted to the offered names. The retry spends what is left of
`max_tokens` and adds its tokens to `completion_tokens`. Calls to functions other than a named one
are dropped. With `parallel_tool_calls: false`, the first call is returned when the model emits
several. `sparkinfer_tool_calls_forced_total{step="retry"|"pick_function"}` in `/metrics` counts
how often that happened.
Tool calls use the native Qwen XML protocol (Qwen3.6, Qwen3.8); Muse Glimmer uses a different
tool protocol.

Tool calls are **constrained**. Every token of a tool-calling turn is sampled under a grammar that
only admits output the server's parser accepts:

- reasoning and content free of protocol markup;
- calls only to offered functions: at least one for `"required"`, only the named one for a named
  choice, and at most one with `parallel_tool_calls: false`;
- every argument in the template's exact framing, with a value its schema allows. That covers
  enums, consts, numeric ranges, string lengths and patterns, nested objects and arrays, `$ref`,
  and `anyOf`/`oneOf`/`allOf`.

So a well-formed request cannot get an invalid tool call back. The one exception is running out of
`max_tokens` mid-call, which returns `finish_reason: "length"`. The grammar is compiled once per
distinct tool set, and each token's mask costs microseconds. `sparkinfer_tool_calls_constrained_total`
counts constrained generations. `SPARKINFER_TOOL_GRAMMAR=0` turns the grammar off and falls back to the
forced call opening and retry described above.

The protocol puts a few limits on arguments:

- Argument text, including strings inside JSON-typed arguments, can contain `<`, just never a `<`
  that starts protocol markup (`<tool_call>`, `</parameter>`, `<think>`, `<|im_end|>`, …). That
  markup would cut the call short for any Qwen tool parser.
- A string argument with `minLength` or `maxLength` cannot contain `<`. Its characters are counted,
  and the grammar cannot count them and exclude markup at the same time.
- Numbers have at most 18 integer digits and a two-digit exponent, the range a double holds.

Where the grammar cannot enforce a schema exactly, the server logs the approximation, and the
argument is still validated after generation. This applies to:

- `oneOf` branches that can overlap;
- a `pattern` inside a JSON value;
- a non-integer or unbounded `multipleOf`;
- a free-form object (no `properties`, or `additionalProperties` allowed), where no grammar can stop
  a key from repeating.

`response_format` output is constrained the same way. `json_schema` output is exactly a value its
schema accepts; with thinking on, the JSON cannot contain a `<think>` or `</think>` marker, which
would split it. `json_object` output is always a JSON object. Keys can in principle repeat, so that
case keeps the validation and single retry. `sparkinfer_structured_output_constrained_total` counts
these generations.

Muse requests containing tool definitions or tool-call history return `400`, including when
`tool_choice` is `"none"`, so unsupported protocol data cannot be silently dropped.
JSON Schema `pattern` uses the safe, linear-time RE2 syntax; unsupported expressions are rejected.
Supported validation keywords are `type`, `properties`, `required`, `additionalProperties`,
`items`, `prefixItems`, `enum`, `const`, `anyOf`, `oneOf`, `allOf`, numeric bounds and
`multipleOf`, item/string length bounds, `pattern`, and local `$ref` pointers into
`$defs`/`definitions`, resolved against the tool's whole `parameters` schema. External
references are refused. Unsupported validation keywords return `400` rather than being silently
ignored. Annotation keywords `description`, `default`, and `title` are retained in the model
prompt; `format`, `$schema`, `$comment`, and `x-*` vendor extensions are accepted as annotations.
Qwen's native XML leaves string values unquoted. For a mixed string/non-string union, a value
that is valid JSON is interpreted as its JSON type first (for example, `1` becomes an integer);
avoid such unions when JSON-looking text must remain a string.

### Graceful shutdown

`SIGTERM`/`SIGINT` stop accepting new connections and new `/v1/chat/completions` requests
(`503`) immediately, then let in-flight requests finish naturally before the process exits —
no hard-killed streams. The process exits as soon as the engine has nothing running and nothing
waiting for capacity, so an idle instance exits within milliseconds rather than sitting out a
grace period; what is drained is generation, not idle keep-alive sockets.

`SPARKINFER_DRAIN_GRACE_S` (default `30`, previously `SPARKINFER_SHUTDOWN_GRACE_S`, still read)
bounds the wait: after it, the process force-exits with requests still in flight. Set it to `0`
to wait for in-flight work as long as it takes and let an orchestrator's own window do the
killing — a 4096-token completion at ~93 tok/s needs ~45 s, more than the default grace.

### RTX PRO 6000 deploy (32k / 4k)

See [`bench/results/qwen3-30b-a3b_q4km_pro6000.md`](../bench/results/qwen3-30b-a3b_q4km_pro6000.md) for the full 5090→PRO 6000 migration
notes and benchmark table.

```bash
export CTX=36864          # 32k prompt + 4k completion KV pool
export HOST=0.0.0.0
./server/run.sh --download
curl -s http://127.0.0.1:8080/v1/info
# {"model":"qwen3.6-35b-a3b","max_context":32768,"max_output_tokens":4096}
```

On RTX 5090 (32 GB) use a smaller `--ctx` (8k–16k) or `CTX=0` for GGUF defaults — the
same binary, different memory budget.

### Example

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"sparkinfer","messages":[{"role":"user","content":"Say hi in one word."}],"max_tokens":16}'
```

With API key (optional):

```bash
./build/server/sparkinfer_server -m model.gguf --api-key secret
curl ... -H 'Authorization: Bearer secret'
```

## Request isolation & continuous batching

Each `/v1/chat/completions` call is submitted to `ContinuousBatchEngine`, which:

- Allocates a **per-request `seq_id`** with **right-sized KV** (`prompt + max_tokens + headroom`, not `max_seq`)
- Runs **vLLM V1-style iteration-level scheduling**: each step packs pending decode
  requests first (up to `SPARKINFER_BATCH_TOKENS`), then admits prefills into the remaining
  budget, oldest first (one per step once the decode batch is wide; more while it is still
  filling). While decode is in flight, at most `SPARKINFER_LONG_PREFILLS_PER_STEP` (1) of the
  prefills larger than `SPARKINFER_PREFILL_MIX_MAX` run per step: hybrid batched prefill is
  atomic, so each 8k pass admitted mid-decode stalls the decoding requests for its duration,
  but holding them until decode drains starved them under steady load
- Under `chunked` (or when decode is waiting under `continuous`), non-batched models
  advance prefills in chunks of `SPARKINFER_PREFILL_CHUNK_TOKENS` before yielding
- Decodes the packed requests in **one forward** (`decode_packed`) rather than one per request.
  Greedy and sampled requests (`temperature`, `top_k`, `top_p`) both pack: a sampled row is drawn
  from its own logits with the single-request sampler, seed and step, so it gets the same token it
  would alone. A request with `presence_penalty`/`frequency_penalty`, `logit_bias`, a tool or JSON
  constraint, or `logprobs` decodes on its own. Sampled requests used to decode on their own too,
  and a request that sets no sampler takes the checkpoint's `generation_config` (temperature 1.0 on
  Qwen3.8), so concurrent throughput stayed at single-stream speed
- Frees KV blocks when the request finishes (no cross-request KV leakage)
- Uses per-request hybrid Gated-DeltaNet recurrent buffers when the model is hybrid

Shared prefix cache still works: when the chat prompt starts with configured prefix tokens,
`cache_prefix()` warms session 0 and only the suffix is prefilled per request.

### Automatic prefix cache

Chat and agent clients resend the whole conversation on every turn. With the prefix cache on (the
default), a request whose prompt starts with an earlier request's prompt reuses that work: its KV
blocks are shared, not copied, and only the rest of the prompt is prefilled. Responses report the
reused count as `usage.prompt_tokens_details.cached_tokens`.

Measured on an RTX 5090, Qwen3.8-27B NVFP4, `--ctx 32768`, an 11.5K-token system prompt:

| | cache on | cache off |
|---|--:|--:|
| time to first token, turns 2-4 of one conversation | **249-313 ms** | 834-1,592 ms |
| 12 concurrent conversations on that system prompt | **12/12 served** | 2/12 (10 x `429`) |

The second row is the KV pool: a conversation on a cached system prompt allocates blocks only for
its own suffix, so twelve fit where two did before.

How an entry is made. KV can be cut at any block, but Qwen3.8's 48 Gated-DeltaNet layers carry a
recurrent state that exists only at the position the sequence is at. So prefill stops at up to two
**checkpoints**, each rounded down to a 16-token KV block, and copies that state to pinned host
memory there:

- the **first** `<|im_start|>` at least `SPARKINFER_PREFIX_CACHE_MIN_TOKENS` in -- for a chat, the end
  of the system prompt (and tool definitions), which every conversation using it shares;
- the **last** `<|im_start|>` -- the start of the final assistant turn, where the same conversation's
  next request stops matching.

A checkpoint the request's own cache hit already covers is skipped, so a system prompt is
snapshotted once, not per request. When the request completes, each prefix and its snapshot become
a cache entry. Sharing the system prompt's blocks also raises how many long-prompt requests fit in
the KV pool at once: a conversation on a cached system prompt needs blocks only for its own suffix.

What is never cached: `/v1/score` (its numbers must not depend on another request) and requests
with images or video (the cache keys on token ids, and every image's placeholder tokens are the
same ids).

Memory. Entries hold KV blocks (up to three quarters of the pool) and snapshots in host RAM
(`SPARKINFER_PREFIX_CACHE_HOST_MB`). When a new request cannot get KV blocks, least-recently-used
entries are evicted before it is refused. `/metrics` reports `sparkinfer_prefix_cache_*` hits,
reused tokens, evictions, entries, blocks and host bytes.

A cached request prefills in two passes -- up to the checkpoint and after it -- instead of one. The
second pass sits as close to the token-by-token reference as the single pass does (tail lengths
15-4,111 tokens, same top-1 token throughout). Outputs still vary by the few tenths of a nat any two
default-mode runs vary by (see **Determinism**); under `SPARKINFER_DETERMINISTIC=1` the cache is off,
because a request's output must not depend on what earlier requests cached.
`runtime/examples/prefix_resume_check.cpp` measures both, and checks a cache hit reproduces an
uncached resume bit for bit in deterministic mode.


| Variable | Default | Purpose |
|----------|---------|---------|
| `SPARKINFER_BATCH_TOKENS` | `64` | Scheduler token budget per step (decode packing) |
| `SPARKINFER_SCHED_POLICY` | `continuous` | `continuous` (pack+mix), `chunked` (CHUNKED_PREFILL), or `priority` (exclusive prefill) |
| `SPARKINFER_PREFILL_CHUNK_TOKENS` | `512` | Token-loop prefill yield size when batched prefill is unavailable (`0` = unlimited). Hybrid models always use full batched GEMM prefill. |
| `SPARKINFER_PREFILL_MIX_MAX` | `2048` | Prompts longer than this are metered while decode is in flight (`0` = no prompt counts as long). |
| `SPARKINFER_LONG_PREFILLS_PER_STEP` | `1` | How many of those long prefills one step admits while decode is in flight. `0` holds them until decode drains, the old rule, which starves them under steady load (8K prompts at 16 concurrent requests: TTFT p99 127 s). |

Prior requests cannot leak decode context into later ones (KV is freed after each completion).

## Env

| Variable | Default | Purpose |
|----------|---------|---------|
| `SPARKINFER_ROOT` | `.` | Repo root (tokenizer script path) |
| `CTX` | `36864` (PRO 6000) / `0` (5090) | KV pool size passed as `--ctx` |
| `SPARKINFER_KV_INT8` | model-dependent | Same as `qwen3_gguf_generate` |
| `SPARKINFER_TOKENIZER_URL` | Qwen3.6-35B-A3B tokenizer | Override tokenizer download |
| `SPARKINFER_SERVER_PREFIX_TOKEN_FILE` | — | JSON `[id,...]` warmed via `cache_prefix` each request |
| `SPARKINFER_SERVER_PREFIX_TOKEN_IDS` | — | Comma-separated token ids (same as above) |
| `SPARKINFER_PREFIX_CACHE` | `1` | Automatic prefix cache (see **Automatic prefix cache**). `0` disables; `SPARKINFER_DETERMINISTIC=1` also disables it. |
| `SPARKINFER_PREFIX_CACHE_ENTRIES` | `256` | Most cached prefixes held at once; least-recently-used is evicted. In practice the memory limits below bind first. |
| `SPARKINFER_PREFIX_CACHE_KV_PCT` | `75` | Share of the KV pool cached prefixes may hold. A request that needs the room evicts least-recently-used entries first, and `/v1/capacity` / `sparkinfer_free_kv_blocks` count blocks only the cache holds as free. |
| `SPARKINFER_PREFIX_CACHE_HOST_MB` | a quarter of RAM, 8192-32768 | Pinned host memory for recurrent-state snapshots (~205 MB each on Qwen3.8-27B; none on Muse Glimmer). Up to 8 more snapshot buffers are kept pinned for reuse; `SPARKINFER_SNAPSHOT_POOL=0` allocates and frees one per snapshot instead. |
| `SPARKINFER_PREFIX_CACHE_MIN_TOKENS` | `1024` | Shortest prompt position a request checkpoints at. Shorter prompts still reuse cached prefixes but do not create one. |
| `SPARKINFER_PREFILL_BATCHED` | `1` | Batched prefill in `cache_prefix` / cold prompts |
| `SPARKINFER_PREFILL_ALIGN8_MIN` | `8` | On a model with NVFP4 prefill, a pass whose length is not a multiple of 8 prefills its aligned body in one pass and the last 1–7 tokens in one more forward, because the NVFP4 GEMMs take multiples of 8 rows (up to 3x faster for a short prompt, 1.9x for a long one). The smallest body split this way; `0` turns it off. |
| `SPARKINFER_DETERMINISTIC` | `0` | `1` = bit-reproducible output (see **Determinism** above). Decode speed unchanged; TTFT +2–8%. |
| `SPARKINFER_MAX_OUTPUT_TOKENS` | `4096` (container: `16384`) | Per-request generation cap. A request without `max_tokens` generates until the model stops, up to this cap or the room its prompt leaves in the context; a larger `max_tokens` is clamped to this cap. Each request reserves KV blocks for its prompt plus `max_tokens` when it is admitted, so a cap near the full context lets one long request hold the whole pool while other requests wait for it (see `SPARKINFER_ADMISSION_WAIT_S`). |
| `SPARKINFER_DRAFT_MODEL` | — | Drafter directory (DFlash2 or DSpark), same as `--draft-model`. The server exits if the drafter cannot be loaded, including when it does not fit in device memory. |
| `SPARKINFER_DSPARK_MAX_CTX` | `--ctx` for a drafter whose layers all attend a sliding window (DFlash2), else `16384` | Context a request can speculate through (the drafter's positions), capped at `--ctx`. |
| `SPARKINFER_DRAFT_OFFLOAD_MS` | `1000` | With a drafter loaded, how long live requests must stay more than a speculation group takes before the drafter's device memory (~3 GB for DFlash2) moves to pinned host memory. Nothing reads it while no group can form, and on a 32 GB card it is the headroom concurrent prefill and decode need. It comes back (~0.1 s) at the same addresses when a group can form again and the device has room for it. A prefill or a new session that runs out of memory beside an idle drafter moves it at once, whatever this is set to. `-1` turns off only the timed move; `SPARKINFER_DRAFT_OFFLOAD=0` keeps the drafter on the device always. It needs pinned host memory of ~1.25x the drafter (~3.8 GB for DFlash2). |
| `SPARKINFER_DRAFT_OFFLOAD` | `1` | `0` allocates the drafter with plain `cudaMalloc`, so it can never leave the device (the behaviour before the offload). |
| `SPARKINFER_DRAFT_RESTORE_HEADROOM_MB` | `1024` | Free device memory, beyond the drafter itself, required to bring it back. Below that, requests decode without speculation until there is room. |
| `SPARKINFER_SPEC_JOIN_ROOM` | `1` | A prompt joins a speculation group only if the device has room for its prefill, its captured context, its draft slot and the verify, checked before anything is allocated; otherwise it prefills on the ordinary path. `0` skips the check. |
| `SPARKINFER_SPEC_GROUP` | `8` | With a drafter loaded, how many live requests speculate together: up to this many fresh prompts each draft their own block (one batched draft pass with DFlash2), and one forward verifies all the blocks -- past four members each verifies a shorter block, to fit the 32-row verify. A request speculation cannot take (logprobs, penalties, a constraint, an image) or one more than this hands every member over to ordinary decode at its committed position. `1` speculates only a request that is alone, and stops at the next arrival. |
| `SPARKINFER_MAX_QUEUE_DEPTH` | `0` (unlimited) | Admission-time cap on the total active continuous-batch set (running and waiting between scheduler steps). Beyond it, new requests are rejected as `429` before KV allocation. Requests waiting for KV capacity count toward it. Production services that promise bounded admission should set this explicitly; `0` does not satisfy such a promise. |
| `SPARKINFER_SAMPLING_DEFAULTS` | `generation_config` | What a request that omits `temperature`, `top_k` or `top_p` gets. `generation_config` uses the checkpoint's `generation_config.json` (Qwen3.8: temperature 1.0, top_k 20, top_p 0.95), as vLLM does; greedy decoding makes a thinking model loop on long agent tasks. `greedy` restores greedy decoding for those requests. A checkpoint without the file, and `SPARKINFER_DETERMINISTIC=1`, stay greedy. An explicit value, including `temperature: 0`, always wins. DSpark speeds up greedy and sampled requests alike (`SPARKINFER_SPEC_SAMPLED=0` limits it to greedy ones). |
| `SPARKINFER_ADMISSION_WAIT_S` | `300` | How long a request that finds no free KV capacity waits for it before `429`, oldest first. Set `0` to refuse immediately instead of waiting, which is what a gateway that promises "reserve or `429`, never queue" wants (pair it with `SPARKINFER_MAX_QUEUE_DEPTH`). A request whose session memory cannot be allocated while other requests are running waits the same way, since their prefill scratch and session state come back as they progress; with nothing else running it gets `503` at once. `0` rejects immediately, the earlier behaviour. When `SPARKINFER_REQUEST_TIMEOUT_S` is set and shorter, the wait ends there and returns `504`. `/metrics` reports `sparkinfer_waiting_requests`, `sparkinfer_admission_waits_total` and `sparkinfer_admission_wait_timeouts_total`; `/v1/capacity` reports `waiting_requests`. |
| `SPARKINFER_SPARSE_GQA6` | `0` | `1` makes Qwen3.8 decode attend only the first KV block and the last 4096 tokens once a sequence reaches 16384 tokens. Faster long-context decode, but the model can no longer read anything in between: a long agent session loses its earlier tool results and instructions. Leave it off unless every request is known to need only recent context. |
| `SPARKINFER_PRESERVE_THINKING` | `1` | `0` replays a previous assistant turn's reasoning only for the turns since the last user message, instead of the whole history. The checkpoint's own chat template preserves all of it (llama.cpp calls this `--reasoning-preserve`); a request's `chat_template_kwargs.preserve_thinking` overrides either way. Reasoning only reaches the model if the client sends it back — as `reasoning_content`, as `reasoning` (the alias the server also emits), or left inside `<think>` tags in `content`. An assistant message may be appended verbatim: every field the server emits is accepted back, along with the `refusal` / `annotations` / `audio` / `function_call` nulls the OpenAI SDKs carry. |
| `SPARKINFER_DRAIN_GRACE_S` | `30` | How long `SIGTERM`/`SIGINT` waits for in-flight requests before force-exiting; `0` waits indefinitely. The process exits at once when nothing is in flight. `SPARKINFER_SHUTDOWN_GRACE_S` is the old name and is still read. |
| `SPARKINFER_REQUEST_TIMEOUT_S` | `0` (disabled) | Per-request wall-clock deadline from submission to finish; exceeding it returns `504`. Left disabled by default — a cold 32k-context prefill alone has been measured taking ~90s of TTFT, so an aggressive default would misfire on legitimate long-context requests. |
| `SPARKINFER_READ_TIMEOUT_S` / `SPARKINFER_WRITE_TIMEOUT_S` | `300` | Transport-level socket timeouts (httplib). Reset on each byte transferred, so a slow-but-progressing stream doesn't trip them. |
| `SPARKINFER_HTTP_THREADS` | `256` | HTTP worker threads. A streaming request (and a keep-alive connection) holds one for its whole life, so this is the most requests served at once; httplib's own default is one per CPU. |
| `SPARKINFER_MODEL_CREATED` | `0` | Model creation time as a Unix timestamp for OpenRouter schema v2.4. `run_openrouter.sh` requires a real value. |
| `SPARKINFER_REQUESTS_PER_MINUTE` | — | Optional request/minute capacity published by `/v1/models`. Enforcement belongs at the gateway; this declaration must match it. |
| `SPARKINFER_PROMPT_TOKENS_PER_MINUTE` | — | Optional prompt-token/minute capacity published by `/v1/models`. |
| `SPARKINFER_COMPLETION_TOKENS_PER_MINUTE` | — | Optional completion-token/minute capacity published by `/v1/models`. |
| `SPARKINFER_OPENROUTER_PROVIDER` | `0` | `1` emits the strict, closed OpenRouter v2.4 model document. The OpenRouter launcher sets this automatically. |

### Release container settings

`ghcr.io/gittensor-ai-lab/sparkinfer-qwen38` reads the variables below as well as every variable
above. Pass them with `-e NAME=value`. Server flags appended after the image name, or after
`serve-dspark`, are added to the server command line and take precedence (`--ctx 65536`).

| Variable | Default | Purpose |
|----------|---------|---------|
| `CTX` | `131072` | Context length, passed as `--ctx`. `262144` fits the model's full context but leaves ~3 GB on a 32 GB card, too little for concurrent requests to batch (the server warns at startup below 5 GiB free); use it for one long conversation at a time. |
| `SPARKINFER_MAX_OUTPUT_TOKENS` | `16384` | Per-request generation cap (see the table above) |
| `SPARKINFER_NO_DOWNLOAD` | `0` | `1` never downloads: the weights must already be in `MODEL_DIR` (and `DRAFT_DIR` for `serve-dspark`). A missing checkpoint fails immediately with what to mount, instead of attempting an egress the box may not have. A missing default DFlash2 drafter is not an error: the server runs without speculation and says so. |
| `MODEL_REPO` / `MODEL_DIR` | `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` / `/models/qwen38-nvfp4` | Target checkpoint, downloaded on first run |
| `SPEC_DRAFT` | `dflash2` | The default `serve` mode's drafter: `dflash2` downloads and loads `DFLASH2_REPO`, `none` serves without one. Ignored by `serve-dspark` and when `SPARKINFER_DRAFT_MODEL` or `--draft-model` is given. Past a 131,072-token context the default skips the drafter (no room for it on a 32 GB card). |
| `DFLASH2_REPO` / `DFLASH2_DIR` | `z-lab/Qwen3.8-27B-DFlash2` / `/models/qwen38-dflash2` | The default drafter |
| `DRAFT_REPO` / `DRAFT_DIR` | `gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4` / `/models/qwen38-dspark` | DSpark drafter, used by `serve-dspark` |
| `MODEL_NAME` | `qwen38-nvfp4` | Model id the API advertises |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | Listen address inside the container |

### Concurrency diagnostic

To investigate host-dependent scaling, run the stdlib-only diagnostic against a live server:

```bash
python3 server/scripts/diagnose_concurrency.py \
  --base-url http://127.0.0.1:8080 \
  --container sparkinfer \
  --concurrency 6,8,10,12 \
  --max-tokens 64 \
  --repeats 3
```

It writes one JSON report containing the exact payloads and token counts, external start/end/TTFT
measurements, server-reported usage, `/v1/capacity` samples, relevant container limits and
`SPARKINFER_*` variables, plus GPU clocks, utilization, power, temperature, memory and negotiated
PCIe link state throughout each burst. Use `external_aggregate_completion_tps` for capacity
comparisons; `usage.decode_tps` is retained only to diagnose differences in internal engine time.
