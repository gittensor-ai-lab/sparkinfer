#!/usr/bin/env python3
"""One summary line from an AIPerf artifact directory: output tok/s, TTFT p50/p90, ITL p50, OSL."""
import json
import os
import sys

d, tag, c = sys.argv[1], sys.argv[2], sys.argv[3]
f = os.path.join(d, "profile_export_aiperf.json")
if not os.path.exists(f):
    print(f"{tag:24s} c{c:3s} NO RESULT")
    sys.exit()
j = json.load(open(f))


def g(k, s="avg"):
    return (j.get(k) or {}).get(s) or 0.0


print(f"{tag:24s} c{c:3s} out_tok/s {g('output_token_throughput'):8.1f}  "
      f"TTFT p50 {g('time_to_first_token', 'p50'):8.1f} p90 {g('time_to_first_token', 'p90'):8.1f} ms  "
      f"ITL p50 {g('inter_token_latency', 'p50'):6.2f} ms  OSL {g('output_sequence_length'):.0f}")
