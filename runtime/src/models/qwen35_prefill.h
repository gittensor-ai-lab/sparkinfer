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

// Gated-DeltaNet recurrent state is held for the LINEAR-ATTENTION layers only, packed one slot per
// such layer. It used to be sized and indexed by n_layers, so every full-attention layer carried a
// head_dim^2 * v_heads block of state it never reads: on Qwen3.8-27B that is 16 of 64 layers, 25% of
// each session's largest buffer -- ~50 MB a session, ~1.65 GB across 33 concurrent requests, which
// is the margin between a 32-row packed step fitting in 32 GB and running out of it.
//
// The layer rule is is_linear_layer()'s: a layer is linear unless (L+1) % full_attn_interval == 0.
// A linear layer's slot is therefore L minus the attention layers before it, L - L/interval, and a
// stack of n layers holds n - n/interval slots. With no interval (a stack that has no linear layers,
// or one gated back to the flag) both fall back to the layer index, i.e. exactly the old layout.
inline int gdn_state_slots(const Qwen35Config& c) {
    if (!c.hybrid || c.full_attn_interval <= 0) return c.n_layers;
    return c.n_layers - c.n_layers / c.full_attn_interval;
}
inline int gdn_state_slot(const Qwen35Config& c, int layer) {
    if (!c.hybrid || c.full_attn_interval <= 0) return layer;
    return layer - layer / c.full_attn_interval;
}

// Per-row scales of one layer's Bonsai decode-shadow legs (launch_ptq1_rows_i8's scale), the one
// thing the fused prefill GEMM's PTQ1 arm needs beyond the ternary blocks themselves. A member is
// null where that leg stayed folded.
struct BonsaiShadowRs {
    const float* wq = nullptr;
    const float* wk = nullptr;
    const float* wv = nullptr;
    const float* wo = nullptr;
    const float* ssm_out = nullptr;
    const float* down = nullptr;
    const float* wqkv = nullptr;
    const float* wqkv_gate = nullptr;
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
    const void*          emb_norm_ones;    // Muse Glimmer: constant-1.0 bf16 weight for the
                                           // unweighted embedding RMSNorm (nullptr for other models)
    // Ternary-Bonsai-2's native PTQ1_0 path. When set, w.embed_tokens is the checkpoint's own
    // ternary table rather than a bf16 expansion, so the lookup decodes a row and takes the
    // stored rotation back off it. Null/false for every other model.
    bool                 bonsai_embed_native;
    const void*          bonsai_sign_hidden;  // int8[hidden] on device; embedding + head
    // int8[moe_ffn] on device, or null. The dense FFN's down leg is the one ternary projection
    // whose input is not the residual width, so dq() cannot reach it with the vector above.
    const void*          bonsai_sign_ffn;
    int                  bonsai_block;
    void*                bonsai_rot;          // scratch for one rotated activation, or null
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

    // Interleaved-MRoPE rotary positions: [n_tokens * 3] int32 on device, laid out
    // [t0,h0,w0, t1,h1,w1, ...]. Null means the ordinary 1D ramp (pos0 + row), which is what every
    // text-only request uses and what the kernels compile to when this is absent.
    //
    // Indexed by the row within THIS PASS, so a windowed prefill must hand over the slice for its
    // own window rather than the whole prompt -- the same relationship `tok` already has to pos0.
    const int*           mrope_pos  = nullptr;

    // PACKED CONTINUOUS-BATCH DECODE. Non-null `packed_rows` turns dflash_verify_short_run's N
    // rows from "N consecutive positions of ONE sequence" into "N INDEPENDENT sequences, one
    // decode token each" -- which is the same forward, since every stage below the GDN block
    // already works per row: the projections and FFN take R rows through a single weight read, the
    // paged attention takes num_seqs with a per-row block table, and the KV-append kernels are
    // already instantiated for a per-row table (SINGLE_SEQUENCE=false).
    //
    // Every one of these is a DEVICE array of N entries, refreshed by the caller before each step
    // and never baked into the capture, so ONE packed graph per row count serves any set of
    // sessions. That is the whole reason they are pointer arrays rather than a base plus stride:
    // each session's lin_state / lin_conv_state / block table is its own allocation.
    //   packed_rows        [N] block-table pointers, one per row's sequence
    //   packed_lin_state   [N] per-session GDN recurrent-state bases (layer selected by offset)
    //   packed_lin_conv    [N] per-session GDN conv-window bases
    //   packed_pos         [N] HOST array of each row's absolute position in its own sequence
    const int*           packed_pos       = nullptr;
    const int* const*    packed_rows      = nullptr;
    // Ring tables for windowed (sliding-window) KV slices, one per row, same order as
    // packed_rows. Equal to packed_rows when the pool has no windowed slices, so a layer can
    // always take the table its own attention contract asks for (see KVCacheManager::windowed()).
    const int* const*    packed_rows_win  = nullptr;
    float* const*        packed_lin_state = nullptr;
    void* const*         packed_lin_conv  = nullptr;
    // The packed rows' recurrent state is the compacted bf16 form (see Qwen35Model::decode_packed).
    bool                 packed_state_b16 = false;
    // The Bonsai decode shadow's layers (n_layers entries), or null. A packed step reads its FFN
    // and its attention q/k/v and output projections from their ternary legs through the
    // arithmetic single-row decode runs on them, so every row decodes bit-identically batched or
    // alone; everything else still comes from `w`.
    const Qwen35LayerWeights* bonsai_dec_layers = nullptr;
    // int8[qdim] on device, or null: the sign vector attn_output reads at, and ssm_out too when
    // the GDN value width is the same.
    const void* bonsai_sign_out = nullptr;
    // The shadow's ternary LM head, or null: read through the int8 rows kernel, as single-row
    // decode reads it through the int8 GEMV.
    const void* bonsai_dec_head = nullptr;
    // Batched prefill's view of the same shadow: its layers, and per layer the row scales of each
    // ternary leg (n_layers entries). Null when the shadow is off or released.
    const Qwen35LayerWeights* bonsai_pf_layers = nullptr;
    const BonsaiShadowRs*     bonsai_pf_rs     = nullptr;
    // One row of single-row decode's int8 activation (q, per-block scales, sums), for the seed
    // token's pass through the ternary head exactly as a decode step makes it.
    signed char* bonsai_hq  = nullptr;
    float*       bonsai_hqd = nullptr;
    int*         bonsai_hqs = nullptr;

    // PACKED PROMPT PREFILL. multi_n > 0 turns the pass's N rows from ONE prompt into multi_n
    // FRESH prompts laid end to end: prompt i is rows [multi_off[i], multi_off[i] + multi_len[i])
    // at positions 0.. of its own session multi_seq_ids[i]. Everything row-wise -- the norms, the
    // projections, the FFN -- runs once over all N rows; only what belongs to a sequence (the
    // recurrent-state reset, the Gated-DeltaNet conv and scan, the KV write and attention, and
    // the seed) runs per prompt on its own slice. HOST arrays of multi_n entries; multi_seed
    // receives each prompt's argmax seed. See Qwen35Model::ingest_prompts_packed.
    int                  multi_n          = 0;
    const int*           multi_off        = nullptr;
    const int*           multi_len        = nullptr;
    const uint64_t*      multi_seq_ids    = nullptr;
    float* const*        multi_lin_state  = nullptr;
    void* const*         multi_lin_conv   = nullptr;
    int*                 multi_seed       = nullptr;

    // Set to true (never back to false) when this call declines because a scratch allocation
    // could not get its VRAM -- as opposed to every other reason prefill_batched_run returns -1,
    // which is an unsupported model/config that retrying cannot fix. Left however the caller set
    // it otherwise, so a caller that can free something (the Bonsai decode shadow, see #1154's
    // rejection) can tell "worth retrying" from "give up now". Null is fine; nothing is recorded.
    bool*                scratch_oom_out  = nullptr;
};

// Fill the paged KV cache + Gated-DeltaNet state for positions 0..n-1 in one batched pass.
// Returns the argmax at the last prompt position (seed for the first decode step), or -1 if the
// batched path is unsupported for this model/config (caller falls back to the token loop).
// pos0: where this pass's tokens start in the sequence (0 = whole prompt in one pass).
// Keep the batched-prefill scratch resident after the next pass even when it exceeds the
// keep-resident budget: set while a windowed prompt has more windows to run.
void prefill_hold_arena(bool hold);
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
