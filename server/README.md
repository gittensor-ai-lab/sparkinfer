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
no hard-killed streams. Bounded by `SPARKINFER_SHUTDOWN_GRACE_S` (default `30`): a client that
vanishes without a clean TCP close can otherwise leave the process waiting up to the read
timeout, so after the grace period the process force-exits regardless.

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
  requests first (up to `SPARKINFER_BATCH_TOKENS`), then admits at most one prefill into
  the remaining budget. Prefills larger than `SPARKINFER_PREFILL_MIX_MAX` wait until
  decode drains (hybrid batched prefill is atomic — mixing an 8k pass mid-decode would
  spike ITL by hundreds of ms)
- Under `chunked` (or when decode is waiting under `continuous`), non-batched models
  advance prefills in chunks of `SPARKINFER_PREFILL_CHUNK_TOKENS` before yielding
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

Memory. Entries hold KV blocks (capped at half the pool) and snapshots in host RAM
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
| `SPARKINFER_PREFILL_MIX_MAX` | `2048` | Max prompt tokens allowed to mix with decode in one step (`0` = always mix). Larger atomic prefills wait until decode drains to avoid ITL spikes. |

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
| `SPARKINFER_PREFIX_CACHE_ENTRIES` | `32` | Most cached prefixes held at once; least-recently-used is evicted. |
| `SPARKINFER_PREFIX_CACHE_HOST_MB` | `8192` | Pinned host memory for recurrent-state snapshots (~205 MB each on Qwen3.8-27B; none on Muse Glimmer). |
| `SPARKINFER_PREFIX_CACHE_MIN_TOKENS` | `1024` | Shortest prompt position a request checkpoints at. Shorter prompts still reuse cached prefixes but do not create one. |
| `SPARKINFER_PREFILL_BATCHED` | `1` | Batched prefill in `cache_prefix` / cold prompts |
| `SPARKINFER_DETERMINISTIC` | `0` | `1` = bit-reproducible output (see **Determinism** above). Decode speed unchanged; TTFT +2–8%. |
| `SPARKINFER_MAX_OUTPUT_TOKENS` | `4096` | Per-request generation cap (independent of context length, which is checked separately against the live `--ctx`) |
| `SPARKINFER_MAX_QUEUE_DEPTH` | `0` (unlimited) | Admission-time cap on the total active continuous-batch set (running and waiting between scheduler steps). Beyond it, new requests are rejected as `429` before KV allocation. Production services that promise bounded admission should set this explicitly; `0` does not satisfy such a promise. |
| `SPARKINFER_REQUEST_TIMEOUT_S` | `0` (disabled) | Per-request wall-clock deadline from submission to finish; exceeding it returns `504`. Left disabled by default — a cold 32k-context prefill alone has been measured taking ~90s of TTFT, so an aggressive default would misfire on legitimate long-context requests. |
| `SPARKINFER_READ_TIMEOUT_S` / `SPARKINFER_WRITE_TIMEOUT_S` | `300` | Transport-level socket timeouts (httplib). Reset on each byte transferred, so a slow-but-progressing stream doesn't trip them. |
| `SPARKINFER_MODEL_CREATED` | `0` | Model creation time as a Unix timestamp for OpenRouter schema v2.4. `run_openrouter.sh` requires a real value. |
| `SPARKINFER_REQUESTS_PER_MINUTE` | — | Optional request/minute capacity published by `/v1/models`. Enforcement belongs at the gateway; this declaration must match it. |
| `SPARKINFER_PROMPT_TOKENS_PER_MINUTE` | — | Optional prompt-token/minute capacity published by `/v1/models`. |
| `SPARKINFER_COMPLETION_TOKENS_PER_MINUTE` | — | Optional completion-token/minute capacity published by `/v1/models`. |
| `SPARKINFER_OPENROUTER_PROVIDER` | `0` | `1` emits the strict, closed OpenRouter v2.4 model document. The OpenRouter launcher sets this automatically. |

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
