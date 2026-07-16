## Summary

<!-- What this PR adds or changes, and why. One or two lines. -->


## Proof of speedup

> ⚠️ **The on-device eval runs only when BOTH are true:** (1) the box below is ticked, and
> (2) at least one **decode tok/s** or **Qwythos prefill pp** table shows a **real end-to-end
> improvement** (`after > before`, filled from `bench/scripts/bench.sh` — *not* an isolated-kernel
> microbenchmark). A ticked box with empty/placeholder tables (or no claimed gain on either metric)
> gets `needs-benchmark` and is **not** evaluated.
>
> Tick the box **only if you actually ran it on an RTX 5090**. False attestation is treated as
> gaming — the account is **blocked** ([`.github/blocked-contributors.txt`](blocked-contributors.txt)),
> same as copycatting or sybil farming.

- [ ] Tested on **RTX 5090** (`sm_120`)

**Decode tok/s** (end-to-end, from `bench/scripts/bench.sh` — fill if this PR targets decode):

| | decode tok/s |
|---|--:|
| before (main) |  |
| after (this PR) |  |

**Prefill pp tok/s** (Qwythos / Qwen3.5 — fill if this PR targets prefill; use `--ctx 4096`, `32768`,
`65536`, or `131072` and copy the `prefill pp` line — report your best context):

| | prefill pp tok/s |
|---|--:|
| before prefill (main) |  |
| after prefill (this PR) |  |

<!-- Paste the bench output backing the numbers above (baseline -> this PR). Isolated-kernel
     microbenchmarks are welcome as extra evidence but do NOT count as before/after. -->

```text
# paste bench/scripts/bench.sh output here (before -> after)
```

<!-- More checklist items will be added here later. -->
