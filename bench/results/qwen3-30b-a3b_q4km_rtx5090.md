# Qwen3-30B-A3B Q4_K_M — RTX 5090 verification

**Hardware:** NVIDIA GeForce RTX 5090 (sm_120, 32 GB GDDR7, ~1792 GB/s, driver 580.142), vast.ai
**Toolchain:** CUDA 13.0, gcc 13, CMake 3.28
**Model:** Qwen3-30B-A3B Q4_K_M GGUF (18.5 GiB, official `Qwen/Qwen3-30B-A3B-GGUF`)
**Setting:** single stream, batch = 1, greedy decode

## Result — runs end-to-end, correct, fits in 32 GB

| Check | Value |
|---|---|
| Build (CMake superbuild, sm_120) | ✓ clean |
| `ctest` | ✓ **5/5 pass** |
| compute-sanitizer (memcheck) | ✓ **0 errors** |
| VRAM resident (experts quantized) | **21.4 GB** |
| Decode throughput | **163.88 tok/s** (6.1 ms/token, n=128) |
| Correctness | "What is the capital of France?" → **"The capital of France is Paris."** ✓ |

## vs llama.cpp — same card, same CUDA (apples-to-apples)

Built llama.cpp (CUDA, `sm_120`) on the *same* 5090 / CUDA 13 and ran `llama-bench` on the *same* GGUF:

| Engine | decode tg128 | gap |
|---|--:|--:|
| llama.cpp (CUDA) | **365.73 ± 2.06 tok/s** | 1.0× |
| sparkinfer | 163.88 tok/s | **2.23×** |

This is the clean comparison. The earlier **1.8×** figure was on the PRO 6000 (134 vs 240.5); on the consumer 5090 — our flagship target — the gap is **wider: 2.23×**.

### Both engines scale to the 5090 — llama.cpp more

| Engine | PRO 6000 (CUDA 12.8) | RTX 5090 (CUDA 13) | gain |
|---|--:|--:|--:|
| sparkinfer | 134 | 163.88 | **+22%** |
| llama.cpp | 240.5 | 365.73 | **+52%** |

The 5090/CUDA-13 environment is faster for **both** engines (so the speedup is hardware/toolchain, not sparkinfer-specific — this answers the "is it the CUDA version?" question: the newer card + toolkit helps, but it helps *both*). The telling part is that llama.cpp gains **+52%** vs our **+22%**.

**Why:** our bs=1 decode is latency-bound on **launch overhead + ~770 tiny kernels/token** (established in the PRO 6000 profiling). Launch / CPU-side overhead does **not** scale with GPU clock, so a faster card barely helps that fraction — while llama.cpp's fewer, fatter kernels are more on-GPU-compute-bound and ride the clock up. The 5090 data therefore **reinforces the fusion thesis**: collapsing the tiny per-token kernels (fuse QKV, residual+norm, multi-output GEMV) is the lever, and it would also recover the clock-scaling we're currently leaving on the table.

## Bugs found & fixed during 5090 bring-up

Building + testing on this second GPU with a newer toolchain surfaced three real issues a single-box history had hidden (all fixed in `sparkinfer`):

1. **CUDA 13 removed** `cudaDeviceProp::memoryClockRate` / `memoryBusWidth` → query via `cudaDeviceGetAttribute`.
2. **flash-decode scratch (`fa_*`) was NULL on the non-GGUF path** (allocated only in `load_gguf`) → moved to the constructor; caught by compute-sanitizer.
3. **top-level superbuild lacked `enable_testing()`** → `ctest` found no tests.

## Reproduce

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120 && cmake --build build -j
HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --local-dir models
./build/runtime/qwen3_gguf_bench models/Qwen3-30B-A3B-Q4_K_M.gguf 128
python3 runtime/tools/run_qwen3.py models/Qwen3-30B-A3B-Q4_K_M.gguf \
  ./build/runtime/qwen3_gguf_generate "What is the capital of France?" 16
```
