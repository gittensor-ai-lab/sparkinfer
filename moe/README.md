# sparkinfer-moe

MoE (Mixture-of-Experts) engine for **NVIDIA RTX Spark**, RTX PRO 6000, RTX 5090, and Jetson Thor.

Part of [gittensor-ai-lab](https://github.com/orgs/gittensor-ai-lab) — SN74.

---

## What this is

The MoE dispatch and routing layer that sits between sparkinfer-runtime and sparkinfer-kernels. It handles:

- **Expert routing** — top-k selection with softmax, per-expert sigmoid (DeepSeek-V2 style), or expert-choice
- **Sync-free token dispatch** — token counts tracked in GPU memory only, no CPU readback, enabling end-to-end CUDA graph capture
- **Expert residency** — YAML `expert_cache:` knobs document intended eviction/prefetch policy; the current engine keeps layered weights resident once `set_layer_weights` runs

---

## Sync-free design

The key constraint for CUDA graph compatibility: `tokens_per_expert[num_experts]` must live on the GPU throughout the entire MoE forward pass. No synchronization barrier, no `cudaMemcpy` to CPU, no host-side branching on expert load.

```cpp
// Wrong — breaks CUDA graph
int counts[256];
cudaMemcpy(counts, d_tokens_per_expert, sizeof(counts), cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();  // ← graph capture stops here

// Right — sync-free
router.route(logits, expert_ids, expert_weights, d_tokens_per_expert, num_tokens, stream);
launch_group_gemm_swiglu(d_tokens_per_expert, ...);  // reads counts on GPU
```

This is the primary latency lever on RTX Spark: a captured graph eliminates kernel launch overhead and allows the driver to overlap memory transfers with compute across MoE layers.

---

## Models

| Model | Experts | top-k | Shared | RTX Spark | RTX 5090 |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B | 256 | 8 | 1 | all fit (128 GB) | eviction needed |
| Gemma 4 26B-A4B | 128 | 8 | 1 | all fit | all fit |

---

## Stack

```
include/sparkinfer/moe/
├── engine.h   # MoEEngine interface, MoEConfig, sync_free flag
└── router.h   # Router, RouterType, RouterConfig

src/
├── engine.cpp # MoEEngine::create / forward
└── router.cpp # routing kernel dispatch
```

Kernel implementation lives in-tree under [`kernels/`](../kernels/) (`csrc/cute/moe_swiglu/`, `csrc/cute/moe_gemm/`).

---

## Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build build -j$(nproc)
```

---

## Namespace

```cpp
#include "sparkinfer/moe/engine.h"
#include "sparkinfer/moe/router.h"

sparkinfer::moe::MoEConfig cfg{...};
auto engine = sparkinfer::moe::MoEEngine::create(cfg);
```
