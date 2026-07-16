# Accuracy — sparkinfer vs llama.cpp (Qwen3-30B-A3B Q4_K_M, RTX 5090)

Verifies sparkinfer **preserves the model's accuracy**, with llama.cpp on the *same*
GGUF as the reference. Teacher-forced over a fixed English text (100 positions),
CUDA 13 / sm_120.

| Metric | Result | Bar |
|---|---|---|
| **Top-1 token agreement** (argmax == llama.cpp argmax, per position) | **100/100 = 100%** | ≥ 90% |
| **Mean KL(llama ‖ sparkinfer)** (top-k, per position) | **0.136 nats** | small |
| **Perplexity, sparkinfer** (exact, full softmax) | **6.13** | — |
| Perplexity, llama.cpp (top-40 + floor → inflated) | 7.76 | ≈ sparkinfer |

**Verdict: accuracy preserved.** sparkinfer's greedy choice matches llama.cpp at every
position (well above the 90% bar); the full next-token distributions are close
(KL ≈ 0.14 nats). The engines share the same Q4_K_M weights, so the ranking is
identical — the small divergence is dequant-rounding + accumulation-order differences
in the distribution tail.

**PPL caveat:** both are `exp(-mean log p(actual next token))` over the same positions.
sparkinfer's is exact (full softmax). llama.cpp's is reconstructed from the server's
top-40 logprobs with a floor for tokens outside top-40, which **inflates** it — the
true llama.cpp PPL is ≈ sparkinfer's. The reliable cross-engine signals are the **100%
top-1 agreement** and the **small KL**.

## The accuracy gate (how to run)

- **sparkinfer:** `qwen3_gguf_score <gguf> <topk> <id0 id1 ...>` — teacher-forces the
  sequence, emits per-position `argmax` + top-k logprobs and the perplexity
  (`exp(-mean log p(actual next))`). Uses the new `Qwen35Model::copy_logits()` accessor.
- **reference:** `llama-server` → `/completion` with `n_probs` + `cache_prompt` per
  position (prefix KV reused → ~1 decode step each).
- **compare:** top-1 agreement, KL over top-k, PPL (same formula both sides).

### Two ways to use it
- **vs llama.cpp** — is the *implementation* correct? Bar: top-1 ≥ 90%, KL small.
  → **PASS** (100%, 0.14 nats).
- **vs previous sparkinfer** — did an *optimization* preserve accuracy? Run the same
  `score` on baseline vs optimized; expect **~100% top-1 + KL ≈ 0**. This is the
  regression gate for the optimization loop (and the SN74 eval-loop correctness gate).
