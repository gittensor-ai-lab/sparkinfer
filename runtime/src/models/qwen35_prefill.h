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
    const void*          emb_norm_ones;    // Muse Glimmer: constant-1.0 bf16 weight for the
                                           // unweighted embedding RMSNorm (nullptr for other models)
    int                  qdim, kvdim;                       // full-attn q / kv dims
    int                  linear_qdim, linear_vdim, linear_qkvdim;  // GDN dims
    // Per-row int8 scales of the routed expert weights, [layer][expert * rows], precomputed at
    // load. Non-null enables the fused quantized-B MoE GEMM (no per-layer int8 materialize).
    const float*         moe_rs_gate;
    const float*         moe_rs_up;
    const float*         moe_rs_down;
    int                  n_splits;
    // Optional DSpark prompt capture. When present, each selected layer copies all prompt rows
    // into capture_dst laid out as [token, capture_slot, hidden], matching dflash_context.
    const int*           capture_layers;
    int                  n_capture;
    void*                capture_dst;
};

// Fill the paged KV cache + Gated-DeltaNet state for positions 0..n-1 in one batched pass.
// Returns the argmax at the last prompt position (seed for the first decode step), or -1 if the
// batched path is unsupported for this model/config (caller falls back to the token loop).
int prefill_batched_run(const Qwen35PrefillCtx& s, const int* prompt_ids, int n);

// Exact short-block DFlash verifier. It evaluates all candidate rows from the live hybrid state,
// commits only the accepted prefix, and leaves rejected KV rows outside the logical sequence.
// Returns the number of consumed rows, or -1 when the exact fast path is unsupported.
// capture_only builds (and instantiates) the replay graph without launching it and without
// touching any model state -- stream capture records kernels instead of running them. Call it once
// during session setup so the ~4.9 ms of graph construction does not land on a decode step.
int dflash_verify_short_run(const Qwen35PrefillCtx& s, const int* token_ids, int n, int start_pos,
                            const int* capture_layers, int n_capture, void* capture_dst,
                            int* out_argmax, bool capture_only = false);

} // namespace sparkinfer
