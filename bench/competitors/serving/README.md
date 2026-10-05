# Serving head-to-head against vLLM

The scripts behind the serving numbers in the top-level README. Each starts one server per engine
on the same GPU, drives it with [AIPerf](https://github.com/ai-dynamo/aiperf) (streaming chat
completions), and prints one line per cell: output tok/s, time to first token (p50 / p90),
inter-token latency (p50) and mean output length.

| script | what it measures |
|---|---|
| `cells.sh` | the standard cells: chat 1024 → 256 tokens at 4 / 16 / 32 requests, 8K prompts at 4 / 16; `DECODE_ONLY=1` for pure decode (32 → 256 at 16 / 32) |
| `spec.sh` | speculative decoding with a DFlash draft on both engines at 1 / 2 / 4 / 8 requests, sampled (T=0.7); `DATASET=sharegpt` for real prompts |
| `longctx.sh` | one request at a time at 32K / 64K / 120K-token prompts, prefix caching off on both engines |

Every cell sets `ignore_eos`, so each answer has exactly the requested length (except ShareGPT,
whose dataset sets its own answer lengths), and `UNIQ=1` gives each cell its own prompt set, so
neither engine's prefix cache sees an earlier cell's prompts.

## Running

Needs a built `sparkinfer_server` (default `build/server/sparkinfer_server`), `aiperf` and `vllm`
on `PATH` (or `SI_BIN`, `AIPERF`, `VLLM`), and the model directories. Results go to
`./results/<script>_<date>/` (`OUT_ROOT` moves them); each run holds `/tmp/sparkinfer_bot.lock`
so two GPU jobs never overlap (`LOCK=` disables it).

```bash
# Qwen3.8-27B, the ModelOpt NVFP4 checkpoint on both engines
SI_M=/models/Qwen3.8-27B-NVFP4 UNIQ=1 bench/competitors/serving/cells.sh

# Qwen3.6-35B-A3B: the UD-Q4_K_M GGUF on sparkinfer, nvidia's NVFP4 checkpoint on vLLM
SI_M=/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf VL_M=/models/Qwen3.6-35B-A3B-NVFP4 \
TOK=/models/Qwen3.6-35B-A3B-NVFP4 UNIQ=1 bench/competitors/serving/cells.sh

# Speculation: z-lab/Qwen3.8-27B-DFlash2 on both engines
SI_M=/models/Qwen3.8-27B-NVFP4 DRAFT=/models/Qwen3.8-27B-DFlash2 UNIQ=1 bench/competitors/serving/spec.sh
DATASET=sharegpt SI_M=... DRAFT=... UNIQ=1 bench/competitors/serving/spec.sh

# Long prompts
SI_M=/models/Qwen3.8-27B-NVFP4 bench/competitors/serving/longctx.sh
```

`WHICH="sparkinfer"` (or `"vllm"`) runs one engine; `ONLYCHAT=1` / `ONLY8K=1` one cell family;
`SI_EXTRA="--draft-model DIR"` loads a draft into the `cells.sh` server.

## Reference results

RTX 5090, sparkinfer 0.6.29–0.6.30 against vLLM 0.30.0, both engines measured the same day on the
same box. Output tok/s, **sparkinfer** / vLLM.

`cells.sh`, no draft:

| model | chat c4 | chat c16 | chat c32 | 8K c4 | 8K c16 |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-35B-A3B | **976** / 669 | **1,769** / 1,695 | **2,405** / 2,364 | **588** / 452 | **674** / 654 |
| Qwen3.8-27B | **323** / 272 | **914** / 867 | **1,288** / 1,246 | **193** / 189 | **312** / 308 |

`spec.sh` (Qwen3.8 + DFlash2 on both engines), 1 / 2 / 4 / 8 requests:

| prompts | sparkinfer + draft | vLLM + draft | sparkinfer, no draft |
|---|---|---|---|
| synthetic | **227 / 385 / 578 / 785** | 196 / 313 / 381 / 426 | 96 / 183 / 322 / 561 |
| ShareGPT | **193 / 354 / 581 / 533** | 163 / 237 / 338 / 346 | 100 / 186 / 313 / 435 |

`longctx.sh`, time to first token, 32K / 64K / 120K prompts:

| model | sparkinfer | vLLM |
|---|---|---|
| Qwen3.8-27B | **2.49 / 6.05 / 14.4 s** | 3.33 / 9.25 / 25.1 s |
| Qwen3.6-35B-A3B | **1.31 / 3.33 / 8.28 s** | 1.39 / 3.67 / 9.46 s |

The 8K c16 cell moves about ±4% between runs of the same build; the others about ±2%.

## Notes

- vLLM needs `MAX_JOBS=4` (set in `common.sh`): flashinfer's JIT otherwise exhausts host memory on
  first start.
- With a draft loaded, sparkinfer's KV pool is sized beside the resident draft, so 8K prompts at 16
  requests queue for KV (~270 tok/s against ~310 without a draft).
- For lossless checks (draft vs no draft), run both servers with `SPARKINFER_DETERMINISTIC=1` and
  `SPARKINFER_PREFIX_CACHE=0`: batched prefill is not bit-reproducible across launches otherwise.
