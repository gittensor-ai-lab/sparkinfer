## Summary

<!-- What this PR adds or changes, and why. One or two lines. -->


## Proof of speedup

> ⚠️ **The on-device eval runs only when BOTH are true:** (1) the box below is ticked, and
> (2) the **decode tok/s** table shows a **real end-to-end improvement** (`after > before`,
> filled from `bench/scripts/bench.sh` — *not* an isolated-kernel microbenchmark). A ticked box
> with an empty/placeholder table gets `needs-benchmark` and is **not** evaluated (no point spending
> a GPU when there's no claimed decode gain).
>
> Tick the box **only if you actually ran it on an RTX 5090**. False attestation is treated as
> gaming — the account is **blocked** ([`.github/blocked-contributors.txt`](blocked-contributors.txt)),
> same as copycatting or sybil farming.

- [ ] Tested on **RTX 5090** (`sm_120`)

**Decode tok/s** (end-to-end, from `bench/scripts/bench.sh` — required for evaluation):

| | decode tok/s |
|---|--:|
| before (main) |  |
| after (this PR) |  |

<!-- Paste the bench output backing the numbers above (baseline -> this PR). Isolated-kernel
     microbenchmarks are welcome as extra evidence but do NOT count as the decode before/after. -->

```text
# paste bench/scripts/bench.sh output here (before -> after)
```

<!-- More checklist items will be added here later. -->
