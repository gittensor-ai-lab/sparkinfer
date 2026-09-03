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
    int                  capture_start;
    // Optional image input. MUST STAY LAST: every Qwen35PrefillCtx is built with positional
    // aggregate initialization, so a field inserted mid-struct silently shifts every later value
    // -- putting these after n_splits made capture_layers land in vision_emb. At the end, the
    // existing initializers simply omit them and they value-initialize to null/0, which is
    // exactly the text-only default.
    //
    // Null (always so for a text-only request) means the vision path is not merely skipped but
    // never referenced -- the splice site in prefill_batched_run is guarded on this pointer.
    //   vision_emb: [vision_n, hidden] bf16 on device, the tower's merged embeddings
    //   vision_pos: [vision_n] int32 on device, prompt positions carrying image_token_id
    // The caller validates vision_n against the placeholder count BEFORE building this, so by the
    // time prefill sees it the two are known to agree.
    const void*          vision_emb = nullptr;
    const int*           vision_pos = nullptr;
    int                  vision_n   = 0;
};

// Fill the paged KV cache + Gated-DeltaNet state for positions 0..n-1 in one batched pass.
// Returns the argmax at the last prompt position (seed for the first decode step), or -1 if the
// batched path is unsupported for this model/config (caller falls back to the token loop).
// pos0: where this pass's tokens start in the sequence (0 = whole prompt in one pass).
int prefill_batched_run(const Qwen35PrefillCtx& s, const int* prompt_ids, int n, int pos0 = 0);

// Exact short-block DFlash verifier. It evaluates all candidate rows from the live hybrid state,
// commits only the accepted prefix, and leaves rejected KV rows outside the logical sequence.
// Returns the number of consumed rows, or -1 when the exact fast path is unsupported.
// capture_only builds (and instantiates) the replay graph without launching it and without
// touching any model state -- stream capture records kernels instead of running them. Call it once
// during session setup so the ~4.9 ms of graph construction does not land on a decode step.
int dflash_verify_short_run(const Qwen35PrefillCtx& s, const int* token_ids, int n, int start_pos,
                            const int* capture_layers, int n_capture, void* capture_dst,
                            int* out_argmax, bool capture_only = false);

// Release request-scoped verify graphs and their device arena. Call after a speculative
// generation so the next long prefill sees the same free-VRAM budget as the first one.
void dflash_release_verify_cache();

} // namespace sparkinfer
