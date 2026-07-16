#!/usr/bin/env bash
# Superbuild for sparkinfer on RTX 5090 (sm_120).
# The runtime repo is the integrator: it pulls in sibling ../kernels and ../moe.
#
# Requirements on the build machine:
#   - CUDA Toolkit 12.8+ (nvcc; 12.8 is the first with sm_120 support)
#   - CMake 3.20+
#   - An RTX 5090 to run the GPU integration test (build works without one)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"

cmake -S "$HERE" -B "$HERE/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="120" \
    "$@"
cmake --build "$HERE/build" -j"$(nproc)"

# Run tests (CPU tests always; GPU test self-skips when no device present).
ctest --test-dir "$HERE/build" --output-on-failure
