#pragma once
// Batched-prefill entry point, kept in its own translation unit (qwen35_prefill.cpp) so the
// orchestration touches no other file's code. It takes an explicit context struct instead of
// reaching into Qwen35Model::Impl, so Impl stays private to qwen35.cpp — qwen35.cpp builds this
// struct from its Impl and calls prefill_batched_run().

#include "sparkinfer/models/qwen_config.h"
#include "sparkinfer/models/qwen35.h"   // Qwen35Weights
#include "sparkinfer/kv_cache.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sparkinfer {

// Persistent bf16 cache of the batched-prefill projection weights. The Qwen3.6 MoE prefill runs its
// Q/K/V/O + GDN + shared-expert projections as bf16 GEMMs (its int8 proj path is off), so every pass
// re-dequantizes the same static GGUF weights to bf16 into a shared scratch buffer. Caching each
// weight's bf16 form on first use (keyed by the quantized source pointer) turns that per-pass
// dequant into a one-time cost. Owned by the model, so it survives across prefill calls; the bf16
// values are exactly what the scratch dequant produced, so results are unchanged.
struct PrefillWeightCache {
    std::unordered_map<const void*, void*> map;   // quantized weight ptr -> persistent bf16 buffer
    std::vector<void*> owned;                      // buffers to cudaFree at model teardown
    size_t bytes = 0;                              // total cached bytes (capped as a VRAM backstop)
    bool disabled = false;                         // set once an alloc fails or the cap is hit
};

struct Qwen35PrefillCtx {
    const Qwen35Config&  cfg;
    const Qwen35Weights& w;
    KVCacheManager*      kv;
    cudaStream_t         stream;
    cudaStream_t         stream_k;         // reuse decode side streams for MoE overlap
    cudaStream_t         stream_v;
    uint64_t             seq_id;
    float*               lin_state;        // Gated-DeltaNet recurrent state (per layer)
    void*                lin_conv_state;   // bf16 causal-conv window (per layer)
    float*               logits;           // vocab scratch for the seed argmax
    int*                 d_out_id;         // device argmax slot
    int*                 h_out_id;         // pinned host argmax slot
    bool                 gguf;             // native GGUF load (quantized weights)
    int                  qdim, kvdim;                       // full-attn q / kv dims
    int                  linear_qdim, linear_vdim, linear_qkvdim;  // GDN dims
    PrefillWeightCache*  wcache;   // persistent bf16 projection cache, or null to dequant per pass
};

// Fill the paged KV cache + Gated-DeltaNet state for positions 0..n-1 in one batched pass.
// Returns the argmax at the last prompt position (seed for the first decode step), or -1 if the
// batched path is unsupported for this model/config (caller falls back to the token loop).
int prefill_batched_run(const Qwen35PrefillCtx& s, const int* prompt_ids, int n);

} // namespace sparkinfer
