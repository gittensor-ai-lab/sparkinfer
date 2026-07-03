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

# 3. score the llama.cpp reference the same way:
python3 run_quality.py --backend llama \
    --model /workspace/models/Qwen3-30B-A3B-Q4_K_M.gguf \
    --llama-cli /workspace/.llamacpp/build/bin/llama-cli

# 4. subset / limit while iterating:
python3 run_quality.py --backend sparkinfer --benchmarks gsm8k,humaneval --limit 20 ...
```

**Quality parity** = run steps 2 and 3 (optionally `--out spark.jsonl` / `--out llama.jsonl`) and
diff the per-benchmark percentages. Equal scores mean the optimization is quality-neutral.

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
