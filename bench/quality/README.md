# bench/quality - LLM-quality benchmark suite

Measures whether the model still **answers well**, across five standard capabilities.
This is orthogonal to sparkinfer's existing `accuracy.sh` gate:

| | measures | question it answers |
|---|---|---|
| `accuracy.sh` (top-1 / KL) | distribution match vs llama.cpp | *"does the kernel compute the same thing?"* |
| **`bench/quality`** (this) | task capability | *"does the model still solve the task?"* |

Run **both engines** and the score difference is your **quality parity** - hard proof that an
optimization preserved capability, not just token agreement. Headline claim it unlocks:
*"same IFEval/GSM8K/HumanEval as llama.cpp, at +33% decode speed."*

## Benchmarks

| Suite | Measures | Scorer |
|---|---|---|
| **IFEval** | instruction following | programmatic constraint checks (bullets, length, keywords, JSON, etc.) |
| **GSM8K** | grade-school math reasoning | final-number extraction |
| **MMLU-Pro** | knowledge + reasoning (MCQ) | chosen-letter match |
| **HumanEval** | code generation | runs hidden unit tests (`pass@1`) |
| **BFCL** | tool / function calling | JSON tool-call name + argument match |

## Layout

```
scorers.py             deterministic, stdlib-only scoring (one fn per benchmark) + self-test
run_quality.py         driver: prompt-build -> backend generate -> score -> report (+ A/B)
fetch.py               pull real datasets from HuggingFace, ~10% stratified dev sample -> data/
data/*.jsonl           real ~10% dev samples (784 items) - regenerate with fetch.py
seed_hand_authored/    the original hand-written seed (offline fallback / reference)
```

## Quick start

```bash
# 1. verify the scorers + runner with no GPU/model:
python3 scorers.py                     # 10/10 scorer self-tests
python3 run_quality.py --backend oracle  # gold/scorer sanity for answer-derivable tasks
python3 run_quality.py --backend mock    # ~0%  - floor

# 2. score sparkinfer on a GPU box:
python3 run_quality.py --backend sparkinfer \
    --model /workspace/models/Qwen3-30B-A3B-Q4_K_M.gguf \
    --bin ./build/qwen3_gguf_generate \
    --tokenizer /workspace/models/tokenizer.json

# 3. score the llama.cpp reference (start llama-server first, then):
python3 run_quality.py --backend llama-server --llama-url http://localhost:8082 \
    --model /workspace/models/Qwen3-30B-A3B-Q4_K_M.gguf

# or one-shot llama-cli (reloads model each item — slow for benchmark tier):
python3 run_quality.py --backend llama \
    --model /workspace/models/Qwen3-30B-A3B-Q4_K_M.gguf \
    --llama-cli /workspace/.llamacpp/build/bin/llama-cli

# 4. subset / limit while iterating:
python3 run_quality.py --backend sparkinfer --benchmarks gsm8k,humaneval --limit 20 ...

# 5. standard tiers:
python3 run_quality.py --backend sparkinfer --tier development ...  # ~10%, 78 items
python3 run_quality.py --backend sparkinfer --tier benchmark ...    # ~25%, 196 items
```

**Quality parity** = run steps 2 and 3 (optionally `--out spark.jsonl` / `--out llama.jsonl`) and
diff the per-benchmark percentages. Equal scores mean the optimization is quality-neutral.

## Standard tiers

| Tier | Purpose | Items |
|---|---|--:|
| `--tier development` | fast pre-merge capability check | 78 |
| `--tier benchmark` | heavier release/frontier quality check | 196 |

Current `benchmark` tier comparison on RTX 5090, greedy decode (`bench/quality/run_quality.py --tier benchmark`):

### Qwen3-30B-A3B (Q4_K_M GGUF)

Model: `Qwen/Qwen3-30B-A3B-GGUF` · `Qwen3-30B-A3B-Q4_K_M.gguf`

| Backend | BFCL | GSM8K | HumanEval | IFEval | MMLU-Pro | Overall |
|---|---:|---:|---:|---:|---:|---:|
| sparkinfer GGUF | 73.33% | 84.85% | 80.00% | 77.08% | 44.00% | 64.37% |
| llama.cpp GGUF | 72.00% | 90.91% | 80.00% | 64.58% | 48.00% | 65.90% |
| vLLM AWQ | 76.00% | 84.85% | 80.00% | 77.08% | 48.00% | 66.92% |

vLLM uses `cognitivecomputations/Qwen3-30B-A3B-AWQ` (HF AWQ, not GGUF).

### Qwen3.6-35B-A3B (UD-Q4_K_M GGUF)

Model: `unsloth/Qwen3.6-35B-A3B-GGUF` · `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`  
Orchestrator: `bench/quality/run_qwen36_benchmark.sh` (sparkinfer first on exclusive GPU, then llama-server).

| Backend | BFCL | GSM8K | HumanEval | IFEval | MMLU-Pro | Overall |
|---|---:|---:|---:|---:|---:|---:|
| sparkinfer GGUF | 74.67% | 63.64% | 60.00% | 83.33% | 64.00% | 68.72% |
| llama.cpp GGUF | 74.67% | 57.58% | 60.00% | 87.50% | 62.67% | 67.30% |

Measured 2026-07-10 on RTX 5090 (`/tmp/qwen36_quality_rerun/`). sparkinfer and llama.cpp rows
use the same GGUF. Do **not** run both engines resident on GPU simultaneously on 32 GB cards.

## Data - real ~10% dev samples

`data/*.jsonl` holds a **real, ~10% stratified sample** of each benchmark (784 items) - small
enough to iterate, real enough to be meaningful. Regenerate / resize with `fetch.py`:

```bash
pip install datasets
python3 fetch.py                                   # all 5, ~10% each (seed 42, stratified)
python3 fetch.py --fraction 0.25 --max-items 800   # bigger sample
python3 fetch.py --benchmarks gsm8k,humaneval      # just some
```

| Benchmark | Source | Items | Notes |
|---|---|--:|---|
| gsm8k | `openai/gsm8k` (test) | 132 | 10% of 1319 |
| mmlu_pro | `TIGER-Lab/MMLU-Pro` (test) | 298 | stratified by subject, capped at `--max-items` |
| humaneval | `openai/openai_humaneval` | 20 | floored to `--min-items` (full set is only 164) |
| ifeval | `google/IFEval` | 34 | only rows whose constraints map to our checkers (`_ifeval_map` in `fetch.py`) |
| bfcl | `NousResearch/hermes-function-calling-v1` | 300 | real tool calls w/ `<tool_call>` ground truth; credential-bearing rows are filtered; true BFCL answers aren't on HF, xLAM-60k is gated |

Sampling is seeded (`--seed 42`) so runs are reproducible. `seed_hand_authored/` keeps the original
tiny hand-written seed as an offline fallback.

## Schema (one JSON object per line)

```jsonc
// gsm8k
{"id","benchmark":"gsm8k","prompt": "<question>", "target": <number>}
// mmlu_pro
{"id","benchmark":"mmlu_pro","prompt","choices":["..."],"answer":"B"}
// ifeval  (instructions are machine-checkable; see _IFEVAL_CHECKS in scorers.py)
{"id","benchmark":"ifeval","prompt","instructions":[{"type":"bullet_count","count":3}, ...]}
// humaneval
{"id","benchmark":"humaneval","prompt":"<def stub+docstring>","entry_point":"fn","test":"def check(candidate): ..."}
// bfcl
{"id","benchmark":"bfcl","prompt","tools":[{...}],"target":{"name":"fn","arguments":{...}}}
```

## Adding a benchmark

1. Add a `score_<name>(item, output) -> {"score","pass","detail"}` to `scorers.py` and register it
   in `SCORERS`; add a self-test case.
2. Add a prompt builder branch in `run_quality.py:build_prompt` and an `MAXNEW` entry.
3. Drop a `data/<name>.jsonl` seed set (and a `fetch.py` adapter if a canonical source exists).

## Safety note

HumanEval **executes model-generated code** to run the unit tests. It does so in a separate
`python -c` subprocess with a timeout, but it is not a hardened sandbox - run the suite on a
throwaway GPU box (like the eval instances), not a machine you care about.
