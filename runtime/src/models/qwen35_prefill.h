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
    uint64_t             seq_id;
    float*               lin_state;        // Gated-DeltaNet recurrent state (per layer)
    void*                lin_conv_state;   // bf16 causal-conv window (per layer)
    float*               logits;           // vocab scratch for the seed argmax
    int*                 d_out_id;         // device argmax slot
    int*                 h_out_id;         // pinned host argmax slot
    int*                 d_scalars;        // packed tok/pos/writepos/seqlen (decode scalars)
    int*                 d_seqlen;         // device seq len (aliases d_scalars[3])
    float*               fa_m;             // flash-decode partials (reuse decode buffers for parity)
    float*               fa_l;
    float*               fa_acc;
    bool                 gguf;             // native GGUF load (quantized weights)
    int                  qdim, kvdim;                       // full-attn q / kv dims
    int                  linear_qdim, linear_vdim, linear_qkvdim;  // GDN dims
};

// Fill the paged KV cache + Gated-DeltaNet state for positions 0..n-1 in one batched pass.
// Returns the argmax at the last prompt position (seed for the first decode step), or -1 if the
// batched path is unsupported for this model/config (caller falls back to the token loop).
int prefill_batched_run(const Qwen35PrefillCtx& s, const int* prompt_ids, int n);

// Phase-1 MoE hybrid: batched Q/K/V/O + GDN + int8 KV attention; per-token routed MoE FFN.
// SPARKINFER_PREFILL_MOE_BATCHED=0 disables. Returns -1 when unsupported.
int prefill_batched_moe_run(const Qwen35PrefillCtx& s, const int* prompt_ids, int n);

} // namespace sparkinfer
