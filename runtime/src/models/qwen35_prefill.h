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

namespace sparkinfer {

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
    // Per-row int8 scales of the routed expert weights, [layer][expert * rows], precomputed at
    // load. Non-null enables the fused quantized-B MoE GEMM (no per-layer int8 materialize).
    const float*         moe_rs_gate;
    const float*         moe_rs_up;
    const float*         moe_rs_down;
    // Slot for the model-owned cache of the fp8/int8 conversions of the STATIC projection weights
    // (see PfWeightCache in qwen35_prefill.cpp). Owned by the caller so it lives exactly as long as
    // the weights do; null disables caching.
    void**               weight_cache;
};

// Fill Qwen35PrefillCtx::weight_cache at load time so no prefill call converts a static weight.
// Only the ctx fields describing the weights are read (cfg/w/stream/gguf/dims/weight_cache).
void prefill_weight_cache_warm(const Qwen35PrefillCtx& s);

// Release a cache created through Qwen35PrefillCtx::weight_cache. Safe on null.
void prefill_weight_cache_free(void* cache);

// Fill the paged KV cache + Gated-DeltaNet state for positions 0..n-1 in one batched pass.
// Returns the argmax at the last prompt position (seed for the first decode step), or -1 if the
// batched path is unsupported for this model/config (caller falls back to the token loop).
int prefill_batched_run(const Qwen35PrefillCtx& s, const int* prompt_ids, int n);

} // namespace sparkinfer
