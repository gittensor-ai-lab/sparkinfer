#!/usr/bin/env bash
# Run the full benchmark suite and collect results to results/

set -euo pipefail

RESULTS_DIR="results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=== Blackwell-Infer Benchmark Suite ==="
echo "Device: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "Output: $RESULTS_DIR"
echo

echo "--- Flash Decode Attention ---"
python3 benchmarks/attention/flash_decode_bench.py \
    --output "$RESULTS_DIR/flash_decode.json"

echo
echo "--- MoE Routing ---"
python3 benchmarks/moe/moe_routing_bench.py \
    --output "$RESULTS_DIR/moe_routing.json"

echo
echo "=== Done. Results in $RESULTS_DIR ==="
