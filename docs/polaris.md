# Polaris — Verifiable Eval Receipts

Polaris produces cryptographically verifiable receipts for every SparkInfer
evaluation. A receipt binds an eval result to its full provenance — code commit,
model SHA256, eval seed, build hash, GPU clock — so anyone can verify that a
benchmark score came from a real RTX 5090 running the claimed code, without
re-running the GPU job themselves.

## Why

The eval bot runs on a cloud GPU box that SparkInfer controls. Without
attestation, the bot's word is the only evidence that a benchmark ran correctly.
Polaris receipts make eval results **third-party verifiable**:

- A miner can prove their speedup is real.
- A validator can audit results offline.
- A dashboard viewer can click through to see the cryptographic evidence.

## Two attestation modes

| | Ed25519 | Polaris TDX |
|---|---|---|
| Root of trust | SparkInfer private key | Intel DCAP hardware root |
| Where scoring runs | Bot host (software) | Intel TDX enclave |
| Key management | Must protect private key | No keys needed |
| Verification | Check Ed25519 signature | Verify DCAP quote + recompute bindings |
| Cost per eval | Free | ~$0.001–0.002 |
| Status | Legacy fallback | Preferred |

The bot auto-selects: if `POLARIS_API_KEY` is set, it uses TDX. Otherwise, if
`SPARKINFER_POLARIS_PRIVATE_KEY` is set, it falls back to Ed25519. If neither is
set, attestations are collected but not attested.

## Architecture

```
GPU Eval Box                    Bot Host                       Polaris Cloud
─────────────                   ────────                       ─────────────
evaluate_dual.sh                pr_eval_bot.py                 api.polaris.computer
  │                               │                               │
  ├─ benchmark Qwen3.6            │                               │
  ├─ benchmark Qwen3-30B          │                               │
  └─ output RESULT_JSON ────────►│                               │
                                  │                               │
judge.py                          │                               │
  ├─ collect provenance           │                               │
  │  (git commit, model SHA,      │                               │
  │   build hash, GPU clocks)     │                               │
  └─ output POLARIS_ATTESTATION ─►│                               │
                                  │                               │
                         ┌────────┤                               │
                         │ IF POLARIS_API_KEY:                    │
                         │   scoring.py ────base64──┐             │
                         │   result.json ───base64──┤             │
                         │   POST /v1/attest ──────►│  TDX enclave│
                         │                          │  ┌──────────┤
                         │                          │  │ score.py │
                         │                          │  │ DCAP quote│
                         │   ◄── TDX receipt ───────┤  └──────────┤
                         │   build_polaris_receipt()│             │
                         │                          │             │
                         │ ELSE IF PRIVKEY:                       │
                         │   build_ed25519_receipt()              │
                         └────────┤                               │
                                  │                               │
                         _upload_polaris_receipt()                │
                         → sparkinfer-log repo                    │
```

### judge.py (eval box)

Runs on the GPU box after the benchmark completes. Collects provenance data that
only the eval box can observe:

- **Code**: repo URL, git commit, build hash, scoring scripts commit
- **References**: model SHA256, model filename, guard model SHA256, llama.cpp
  commit, eval seed
- **Environment**: GPU name, architecture, compute capability, CUDA version,
  driver version, clock info (pinned frequency, measured MHz, spread)
- **Measurements**: primary model TPS at all contexts, top1 accuracy, KL
  divergence, guard context pass/fail
- **Verdict**: label, pass/fail, delta TPS, frontier %, context gains

The judge does **not** sign — it only assembles the unsigned attestation. The
private key never touches the eval box.

### Polaris TDX (bot host → Polaris Cloud)

When `POLARIS_API_KEY` is set, the bot submits the **scoring step** to a Polaris
Intel TDX enclave:

1. `scoring.py` (self-contained, stdlib only) and `result.json` are base64-encoded
   and sent to Polaris as files.
2. Polaris spawns a fresh TDX enclave, mounts the files, and runs
   `python3 /submission/score.py`.
3. The enclave produces a **DCAP quote** — a hardware-signed attestation from the
   Intel CPU proving the exact code + inputs + output.
4. The DCAP quote, collateral chain, and stdout are returned.

The scoring logic replicates `evaluate_dual.sh` lines 148–212:

- **Correctness gate**: top1 ≥ 0.90, KL ≤ 0.20
- **No-regression guard**: all 5 contexts (128, 512, 4k, 16k, 32k) must pass
- **Guard model**: Qwen3-30B speed + accuracy must hold
- **Label computation**: none / slight / modest / good / large

**Nonce binding**: `sha256(commit || model_sha256 || eval_seed)[:64]` — binds
the TDX quote to the exact eval, preventing replay attacks across different
commits or models.

### Ed25519 (bot host, legacy)

When `SPARKINFER_POLARIS_PRIVATE_KEY` is set but `POLARIS_API_KEY` is not, the
bot signs the attestation locally with Ed25519. The public key is committed at
`eval/polaris/sparkinfer_eval.pub`.

## Receipt structure

```json
{
  "polaris_version": 1,
  "receipt_id": "a1b2c3d4...",
  "chain": {
    "prev_receipt_hash": null,
    "chain_index": 0
  },
  "attestation": {
    "code": {
      "repo": "https://github.com/gittensor-ai-lab/sparkinfer",
      "commit": "01aef983347de4...",
      "build_hash": "sha256 of qwen3_gguf_bench binary",
      "scoring_scripts_commit": "origin/main commit at eval time"
    },
    "references": {
      "model_sha256": "...",
      "model_file": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
      "guard_model_sha256": "...",
      "guard_model_file": "Qwen3-30B-A3B-Q4_K_M.gguf",
      "llamacpp_commit": "...",
      "eval_seed": "..."
    },
    "environment": {
      "eval_mode": "longctx",
      "decode_tokens": 128,
      "gpu_name": "NVIDIA GeForce RTX 5090",
      "gpu_arch": "sm_120",
      "clocks_pinned": true,
      "clock_mhz": 2235,
      "clock_spread_mhz": 15,
      "pin_target_mhz": 2250,
      "cuda_version": "12.8",
      "driver_version": "570.124.06"
    },
    "measurements": {
      "primary": { "ctx_128_tps": 403.5, "top1": 0.95, "kl": 0.12, "..." : "..." },
      "guard": { "ctx_128_tps": 391.2, "top1": 0.97, "kl": 0.08, "..." : "..." }
    },
    "verdict": {
      "label": "good",
      "pass": true,
      "tps": 413.3,
      "delta_tps": 9.8,
      "pct_over_frontier": 5.2,
      "reason": "all gates passed"
    },
    "timestamp_utc": "2026-07-09T01:42:00Z"
  },
  "attestation_type": "tdx-quote",
  "tdx": {
    "quote_b64": "...",
    "collateral_b64": "...",
    "nonce": "sha256(commit||model_sha256||eval_seed)[:64]",
    "e2e_pubkey_b64": "Wf+zAcqX5IswqbCDDgiwEi0H+9Tj5jdGsqDhhvKgMy8=",
    "bound_digest": "...",
    "result_sha256": "sha256 of scoring.py stdout",
    "stdout_b64": "base64 of scoring.py output",
    "files_sha256": "sha256 of input files",
    "workload_sha256": "sha256 of scoring.py source",
    "verification": {
      "intel_verified": true,
      "report_data_match": true
    },
    "cost_usd": 0.0015
  }
}
```

## Verification

### CLI verifier

```bash
python3 eval/polaris/verify.py receipt.json
```

Outputs checkmarks for each validation step:

| Check | TDX | Ed25519 |
|---|---|---|
| Intel DCAP quote | ✓ quote verified by Intel | — |
| DCAP report_data | ✓ bindings match | — |
| Result hash | ✓ stdout matches result_sha256 | — |
| E2E pubkey | ✓ matches trusted key | — |
| Schema | ✓ valid | ✓ valid |
| Hash integrity | ✓ receipt_id matches | ✓ receipt_id matches |
| Signature | — | ✓ Ed25519 valid |
| Public key | — | ✓ matches trusted key |
| Internal consistency | ✓ | ✓ |
| Gate re-checks | ✓ correctness + guard | ✓ correctness + guard |

### Strict mode (TDX)

```bash
python3 eval/polaris/verify.py receipt.json --strict
```

Additionally verifies that the Intel verification collateral chain URLs match
expected values.

### Programmatic

```python
from eval.polaris.receipt import ReceiptValidator

receipt = json.load(open("receipt.json"))
validator = ReceiptValidator(receipt)
passed, results = validator.verify(public_key_b64="Wf+zAcqX5IswqbCDDgiwEi0H+9Tj5jdGsqDhhvKgMy8=")

for line in results:
    print(line)
print(f"Overall: {'PASS' if passed else 'FAIL'}")
```

The `verify()` method auto-detects receipt type — if `attestation_type` is
`"tdx-quote"` or a `"tdx"` key is present, it dispatches to TDX verification;
otherwise it uses Ed25519.

## Trust model

### What is attested

- The eval ran on a real RTX 5090 at known clocks.
- The code was the exact commit claimed.
- The model file was the exact SHA256 claimed.
- The scoring (correctness gate, guard gates, label) was computed correctly.

### What is NOT attested

- The GPU speed numbers themselves — consumer RTX 5090s have no GPU Confidential
  Computing, so the TPS values cannot be cryptographically sealed at the hardware
  level.
- The eval seed randomness — the seed is provided to the eval box, which could
  theoretically ignore it.

### Threat model

| Attack | Mitigation |
|---|---|
| Bot operator fakes a speedup | judge.py pins bench/scripts to origin/main; scoring runs in TDX enclave |
| Attacker replays an old receipt | Nonce binds each receipt to a specific (commit, model, seed) tuple |
| Attacker tampers with receipt | Tampering breaks DCAP quote or Ed25519 signature; hash mismatch |
| Polaris is compromised | Verification is fully offline — Intel DCAP collateral chain is checked, no trust in Polaris required |

## Running the eval bot with Polaris

### TDX mode (preferred)

```bash
export POLARIS_API_KEY=pi_sk_...
python3 eval/pr_eval_bot.py --instance <id> --dual --polaris
```

### Ed25519 mode (legacy)

```bash
export SPARKINFER_POLARIS_PRIVATE_KEY=<base64-encoded-32-byte-seed>
python3 eval/pr_eval_bot.py --instance <id> --dual --polaris
```

### No attestation (unsigned)

```bash
python3 eval/pr_eval_bot.py --instance <id> --dual
```

Without `--polaris`, `judge.py` never runs and no attestation is generated.

## Source files

| File | Purpose |
|---|---|
| `eval/polaris/judge.py` | Runs on eval box; collects provenance, outputs unsigned attestation |
| `eval/polaris/receipt.py` | Schema, canonicalization, Ed25519 sign/verify, TDX receipt assembly, ReceiptValidator |
| `eval/polaris/client.py` | Polaris API client; submits scoring to TDX enclave |
| `eval/polaris/scoring.py` | Self-contained scoring script that runs inside TDX enclave |
| `eval/polaris/verify.py` | Standalone CLI verifier — no GPU required |
| `eval/polaris/sparkinfer_eval.pub` | SparkInfer Ed25519 public key (trust anchor, also used as TDX e2e pubkey) |
| `eval/polaris/test_receipt.py` | 29 unit tests for receipt building, signing, verification, and tamper detection |
| `eval/pr_eval_bot.py` | Eval bot; Polaris dispatch logic at lines ~1181–1231 |
| `eval/vast_eval.py` | SSH plumbing; passes `--polaris` flag to eval box, invokes judge.py |

## Cost

- TDX enclave time: ~30–40 seconds per eval
- Cost: ~$0.001–0.002 per attestation
- At ~10 PRs/day, a $25 Polaris balance lasts 3–7 years
