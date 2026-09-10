## Summary

<!-- What this PR adds or changes, and why. One or two lines. -->


## Proof of speedup

> **There is no target optimization — optimize anything.** Decode, prefill, attention, FFN,
> quantization, KV cache, sampling, batching, memory traffic, load time: any genuine speedup that
> keeps output identical counts, and is scored the same wherever it lands.
>
> ⚠️ **The on-device eval runs only when BOTH are true:** (1) the box below is ticked, and
> (2) at least one of the **decode tok/s** / **prefill pp tok/s** tables shows a **real end-to-end
> improvement** (`after > before`, filled from `bench/scripts/bench.sh` — *not* an isolated-kernel
> microbenchmark). Either table alone is enough — a prefill-only PR with flat decode greenlights on
> its prefill numbers. A ticked box with empty/placeholder tables (or no claimed gain on either
> metric) gets `needs-benchmark` and is **not** evaluated.
>
> 📋 **Fill the tables in place — don't rename the rows.** The greenlight reads them by their row
> labels, so keep `before prefill (main)` / `after prefill (this PR)` exactly as written and just
> add the numbers. Dropping the word `prefill` from a row hides that number from the gate, and a
> prefill-only PR then looks like a decode PR with no gain and is skipped.
>
> 🆕 **Nothing measures your optimization yet?** That's a gap in the harness, not a verdict on your
> work. Open an issue describing the axis you need (model, metric, context length, and the command
> that measures it) with your before/after numbers, link it here, and ask for the `hold` label so
> this PR isn't auto-closed while the axis is added. See CONTRIBUTING.md → *How rewards work*.
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

**Prefill pp tok/s** (fill if this PR targets prefill; use `--ctx 4096`, `32768`, `65536`, or
`131072` and copy the `prefill pp` line — report your best context. Keep the row labels as-is):

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
