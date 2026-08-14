// Qwen MoE single-sequence greedy decoder.
//
// Per token: embed -> [40x Qwen layer] -> final RMSNorm -> LM head -> argmax.
// Qwen full-attention layer: RMSNorm -> Q/K/V -> per-head QK-norm -> RoPE ->
//             KV append -> GQA flash decode -> O-proj -> residual -> RMSNorm ->
//             routed top-8 MoE (+ shared expert) -> residual.
// Qwen3.5/Qwen3.6 hybrid layers replace full attention with a single-token
// Gated DeltaNet recurrent update on the 3-of-4 linear-attention layers.
// All steps run on one stream; only the sampled id is copied to the host, which
// autoregressive greedy decoding fundamentally requires.

#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/models/dflash_draft.h"
#include "sparkinfer/models/dflash_kernels.h"
#include "qwen35_prefill.h"
#include "sparkinfer/thermal_governor.h"
#include "sparkinfer/kv_ops.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/kernels/attention.h"
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/fused.h"
#include "sparkinfer/kernels/moe.h"
#include "sparkinfer/kernels/quant.h"
#include "sparkinfer/kernels/proj_requant.h"
#include "sparkinfer/kernels/prefill_nvfp4.h"
#include "sparkinfer/lmcache_bridge_client.h"
#include "sparkinfer/lmcache_staging.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <atomic>
#include <unordered_map>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <climits>
#include <limits>
#include <algorithm>

namespace sparkinfer {

namespace {
inline void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fprintf(stderr, "[qwen35] %s: %s\n", what, cudaGetErrorString(e));
}
// SPARKINFER_MUSE_FUSE_TAIL=0 splits Muse Glimmer's sandwich-norm tail back into the original
// launch_norm_then_add + launch_rmsnorm pair (the two produce bit-identical output).
inline bool muse_fuse_tail() {
    static const bool on = [] {
        const char* e = getenv("SPARKINFER_MUSE_FUSE_TAIL");
        return !(e && e[0] == '0');
    }();
    return on;
}
using bf16 = unsigned short;

// forward_token() sentinel: the decode graph was enqueued but not yet collected. Never a valid
// token id, so it cannot be confused with a real argmax.
constexpr int kDFlashDeferred = INT_MIN;

// Qwen35Model::set_logit_bias's sparse (id,val) scratch cap -- sparkinfer's own implementation
// bound for scratch-buffer sizing, NOT an OpenAI-documented limit.
constexpr int kMaxLogitBiasEntries = 1024;

// launch_gguf_dequant only implements F32/F16/Q8_0/Q4_K/Q6_K. Reject anything
// else at load time so Q5_K (etc.) cannot silently fall through as F32.
bool ggml_dequant_supported(int ggml_type) {
    switch (ggml_type) {
        case 0:  // F32
        case 1:  // F16
        case 8:  // Q8_0
        case 12: // Q4_K
        case 13: // Q5_K (UD / dynamic quants mix this in)
        case 14: // Q6_K
            return true;
        default:
            return false;
    }
}

long qwen_moe_meta_int(const GGUF& g, const std::string& key, long def) {
    const long missing = std::numeric_limits<long>::min();
    long v = g.meta_int("qwen35." + key, missing);
    if (v != missing) return v;
    v = g.meta_int("qwen35moe." + key, missing);
    if (v != missing) return v;
    v = g.meta_int("qwen3moe." + key, missing);
    if (v != missing) return v;
    v = g.meta_int("qwen3_5_moe." + key, missing);
    return v != missing ? v : def;
}

bool is_qwen35_or_qwen36_hybrid_moe(const GGUF& g) {
    const std::string name = g.meta_str("general.name");
    if (name.find("Qwen3.5-35B-A3B") != std::string::npos ||
        name.find("Qwen3.6-35B-A3B") != std::string::npos)
        return true;

    if (g.tensor("blk.0.attn_qkv.weight") != nullptr &&
        g.tensor("blk.3.attn_q.weight") != nullptr)
        return true;

    const GGUFTensor* emb = g.tensor("token_embd.weight");
    const long vocab = emb ? emb->dims[1] : qwen_moe_meta_int(g, "vocab_size", -1);
    const bool hybrid_tensor_layout =
        g.tensor("blk.0.attn_q.weight") == nullptr &&
        g.tensor("blk.3.attn_q.weight") != nullptr;
    return qwen_moe_meta_int(g, "block_count", -1) == 40 &&
           qwen_moe_meta_int(g, "embedding_length", -1) == 2048 &&
           qwen_moe_meta_int(g, "attention.head_count", -1) == 16 &&
           qwen_moe_meta_int(g, "attention.head_count_kv", -1) == 2 &&
           qwen_moe_meta_int(g, "attention.key_length", -1) == 256 &&
           qwen_moe_meta_int(g, "expert_count", -1) == 256 &&
           qwen_moe_meta_int(g, "expert_used_count", -1) == 8 &&
           qwen_moe_meta_int(g, "expert_feed_forward_length", -1) == 512 &&
           vocab == 248320 &&
           hybrid_tensor_layout;
}

bool is_linear_layer(const Qwen35Config& c, int layer) {
    return c.hybrid && c.full_attn_interval > 0 && ((layer + 1) % c.full_attn_interval) != 0;
}
}

struct SessionBuffers {
    float* lin_state = nullptr;
    bf16* lin_conv_state = nullptr;
    // Per-request running count of how many times each vocab id has appeared in THIS session's
    // generated completion so far -- for presence_penalty/frequency_penalty. Unlike lin_state/
    // lin_conv_state (hybrid-architecture-only), this exists for EVERY model. vocab-sized, alloc'd
    // once per session slot; see Qwen35Model::reset_penalty_counts for why it must ALSO be
    // explicitly re-zeroed once per REQUEST (not just once per session-slot creation) whenever a
    // session is reused across multiple requests (seq_id 0, the shared prefix session).
    int* penalty_counts = nullptr;
    // Per-request static per-vocab-id additive bias for logit_bias -- unlike penalty_counts (which
    // starts at zero and is incremented device-side every decode step), this is SET ONCE per
    // request (Qwen35Model::set_logit_bias) and stays constant for the rest of the request's
    // decode. Exists for EVERY model, same as penalty_counts, and needs the identical per-REQUEST
    // (not per-session-slot) re-zero-then-set discipline when a session is reused (seq_id 0).
    float* logit_bias = nullptr;
};

struct Qwen35Model::Impl {
    Qwen35Config cfg;
    KVCacheManager* kv;
    moe::MoEEngine* engine;
    Qwen35Weights w;
    cudaStream_t stream{};
    cudaStream_t stream_k{}, stream_v{};         // side streams for concurrent K/V projection
    cudaStream_t stream_pf{};                    // side stream for the L2 weight prefetch
    cudaEvent_t ev_pf_fork{}, ev_pf_done{};      // prefetch fork/join (kept inside the decode graph)
    cudaEvent_t ev_qkv{}, ev_k{}, ev_v{};        // fork/join events (captured into the decode graph)
    cudaEvent_t ev_pipe_fork{}, ev_gdn_z{}, ev_gdn_ab{};
    cudaEvent_t ev_sx_gate{}, ev_sx_done{};
    uint64_t active_seq_id = 0;
    std::atomic<uint64_t> next_session_id{1};
    std::unordered_map<uint64_t, SessionBuffers> sessions;
    int qdim, kvdim;
    int linear_qdim = 0, linear_vdim = 0, linear_qkvdim = 0;
    bool gguf = false;   // true after load_gguf: dense weights are native [out,in], use GEMV
    // CUDA-graph capture of the decode compute (captured once, replayed each token)
    cudaGraph_t cu_graph{};
    cudaGraphExec_t cu_exec{};
    bool graph_ready = false;
    cudaGraph_t cu_prefill_graph{};
    cudaGraphExec_t cu_prefill_exec{};
    bool graph_prefill_ready = false;
    int graph_prefill_attn_mode = -1;
    bool bench_feedback_graph = false;
    int graph_attn_mode = -1;  // host-side flash-decode dispatch class captured in cu_graph
    // Separate decode graph for DFlash verify (sample=true, dflash_capture=true). The normal
    // cu_graph/cu_exec can't be reused here because it carries the extra per-layer hidden
    // captures. Their destination row varies per call, which is handled the same way the graph
    // already handles token_id/position: cap_row is packed into d_scalars and read on device.
    cudaGraph_t cu_dflash_graph{};
    cudaGraphExec_t cu_dflash_exec{};
    bool dflash_graph_ready = false;
    int dflash_graph_attn_mode = -1;
    bool dflash_graph_sparse = false;

    // scratch (bf16)
    bf16 *x, *xn, *q, *k, *v, *attn, *ao, *h, *hn, *routed, *shared;
    bf16 *qraw = nullptr, *qgate = nullptr;
    bf16 *dbg_xn_dump = nullptr;   // DEBUG ONLY (SPARKINFER_MG_STAGE_DEBUG): [n_layers, H] xn snapshot
    bf16 *lin_qkv = nullptr, *lin_q = nullptr, *lin_k = nullptr, *lin_v = nullptr;
    bf16 *lin_z = nullptr, *lin_alpha = nullptr, *lin_beta = nullptr;
    bf16 *lin_gdn = nullptr, *lin_norm = nullptr, *shared_gate_tmp = nullptr;
    bf16 *sh_gate = nullptr, *sh_up = nullptr, *sh_h = nullptr;   // shared-expert GEMV scratch [moe_ffn]
    bf16 *lin_conv_state = nullptr;
    float* lin_state = nullptr;
    // "Current" session's penalty_counts (swapped by activate_session(), unconditionally, for
    // every model -- see SessionBuffers). penalty_counts_default backs session 0's entry, alloc'd
    // once at load time; sessions[seq_id].penalty_counts for every other session is alloc'd by
    // open_session().
    int* penalty_counts = nullptr;
    int* penalty_counts_default = nullptr;
    // "Current" session's logit_bias (swapped by activate_session(), unconditionally, for every
    // model -- mirrors penalty_counts exactly). logit_bias_default backs session 0's entry.
    float* logit_bias = nullptr;
    float* logit_bias_default = nullptr;
    // Transient scratch for Qwen35Model::set_logit_bias's sparse (id,val) -> device scatter. NOT
    // session-scoped (purely transient staging, safe to share across requests since submit_locked
    // -- the only caller -- always runs with the engine mutex held). Fixed size (kMaxLogitBiasEntries).
    int* h_logit_bias_ids = nullptr; float* h_logit_bias_vals = nullptr;
    int* d_logit_bias_ids = nullptr; float* d_logit_bias_vals = nullptr;
    float* logits;
    int *d_scalars, *d_tok, *d_out_id, *d_pos, *d_seqlen, *d_writepos, *d_shared_ids;
    int *d_cap_row = nullptr;   // dflash capture row, packed into d_scalars[4]
    int *h_scalars = nullptr, *h_out_id = nullptr;
    // Temperature-sampling params, refreshed via cudaMemcpyAsync before every forward_token call
    // (same pattern as h_scalars/d_scalars above) so a captured decode graph can safely replay
    // across separate requests/sessions with different temperature/seed -- see fused.h /
    // launch_temperature_sample's doc comment for why these can't be plain kernel arguments.
    float* h_sample_temp = nullptr;
    unsigned long long *h_sample_seed = nullptr, *h_sample_step = nullptr;
    float* d_sample_temp = nullptr;
    unsigned long long *d_sample_seed = nullptr, *d_sample_step = nullptr;
    // top_k/top_p params, same refresh-before-every-call discipline as the sample params above.
    int* h_sample_top_k = nullptr;
    float* h_sample_top_p = nullptr;
    int* d_sample_top_k = nullptr;
    float* d_sample_top_p = nullptr;
    // presence_penalty/frequency_penalty params, same refresh-before-every-call discipline.
    float* h_sample_presence_penalty = nullptr;
    float* h_sample_frequency_penalty = nullptr;
    float* d_sample_presence_penalty = nullptr;
    float* d_sample_frequency_penalty = nullptr;
    // top_k/top_p truncation scratch: allocated ONCE at load time (vocab-sized), fixed address,
    // never reallocated per-call -- see kernels::launch_topk_topp_mask's doc comment for why this
    // must always process the full vocab regardless of top_k/top_p (CUB's num_items is a host
    // constant, not a per-call device value).
    int* d_vocab_iota = nullptr;
    float* d_sorted_logits = nullptr;
    int* d_sorted_idx = nullptr;
    float* d_topk_exp = nullptr;
    float* d_topk_cumsum = nullptr;
    void* d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    void* d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;
    // logprobs/top_logprobs scratch -- same alloc-once-at-load-time discipline as the topk/topp
    // block above. d_rank_by_id[id] = sorted rank of vocab entry `id` (inverse permutation,
    // scattered by topk_topp_exp_kernel); d_chosen_logit = the winning token's raw logit, written
    // by launch_extract_chosen_logit. See Qwen35Model::last_token_logprobs's doc comment.
    int* d_rank_by_id = nullptr;
    float* d_chosen_logit = nullptr;
    float* d_shared_w;
    std::vector<void*> owned;   // device buffers from load_weights / load_gguf
    // GGUF fused-expert decode scratch (allocated by load_gguf)
    float *mf_logits = nullptr, *mf_weights = nullptr, *mf_h = nullptr, *mf_out = nullptr;
    float *sx_h = nullptr;   // pipelined shared-expert h_scratch (avoids racing routed mf_h)
    void  *sx_q8 = nullptr;  // pipelined shared-expert Q8_1(h) for down (avoids racing aq81)
    int   *mf_ids = nullptr, *mf_counts = nullptr;
    unsigned int *mf_rc = nullptr;   // fused-router grid-completion counter (persistent, zero-init)
    // Per-row int8 scales of the routed expert weights, for the batched prefill's fused
    // quantized-B MoE GEMM (prefill_moe_q.cu). Laid out [layer][expert * rows]; populated
    // EAGERLY here at load, never lazily -- the scored sweep times each context exactly once
    // (512 first), so a lazy fill would land inside the very pass it is meant to speed up.
    float *moe_rs_gate = nullptr, *moe_rs_up = nullptr, *moe_rs_down = nullptr;
    // Muse Glimmer dense prefill: one pool of per-output-row int8 scales for the native Q4_K/Q5_K
    // attn + FFN gate/up weights (Qwen35LayerWeights::*_rs point into it). See load().
    float *muse_rs = nullptr;
    // flash-decoding (KV-split) attention partials
    static constexpr int MAX_NSPLITS = 256;   // partials sized for this; adaptive n_splits <= this
    int n_splits = 32;
    bool adaptive_splits = true;              // scale n_splits with seq_len (decode graph re-captured on change)
    int split_chunk = 256;                    // target serial KV per split (SPARKINFER_SPLIT_CHUNK)
    float *fa_m = nullptr, *fa_l = nullptr, *fa_acc = nullptr;
    // Sink + sliding-window sparse-KV. Default on; SPARKINFER_SPARSE_KV=0 disables. Per-kv_head block list.
    int*   sparse_sel = nullptr;
    int    sparse_budget = 0;      // max sel slots = 1 + window
    int    sparse_window = 256;    // recent window in KV blocks (16 tokens/block)
    int    sparse_min_ctx = 8192;
    bool   graph_sparse = false;
    // GQA-8 (Qwen3.6 full-attn) sparse rides the dense int8-MMA kernel over a compacted
    // paged-KV view instead of a dedicated sparse walker (issue #559). Decode steps only.
    int*   sparse_vtbl = nullptr;  // compact view block table [sparse_budget]
    int*   sparse_vlen = nullptr;  // compact view seq_len (device scalar)
    int    sparse_vsplits = 128;   // KV splits over the view (sized so MMA chunks >= 2 blocks)
    // Muse Glimmer: pure sliding-window compact view for swa-flagged layers (no sink,
    // mandatory every step -- see fa_kv_compact_view_pure). Built once per decode step,
    // shared by every swa layer that step; global (non-swa) layers use the full btable.
    int*   swa_vtbl = nullptr;     // compact view block table [swa_budget]
    int*   swa_vlen = nullptr;     // compact view seq_len (device scalar)
    int    swa_budget = 0;         // sliding_window tokens / block_size, rounded up
    int    swa_vsplits = 32;       // KV splits over the (small, fixed-size) swa view
    bf16*  emb_norm_ones = nullptr; // [hidden] all-1.0, for the unweighted post-embedding RMSNorm
    // pre-quantized Q8_1 activation (computed once per projection input, shared across Q/K/V)
    signed char* aq8 = nullptr; float *aq8_d = nullptr, *aq8_s = nullptr;
    bool use_pq = true;   // SPARKINFER_PQ=0 disables the pre-quantized GEMV path
    void* aq81 = nullptr; // block_q8_1 activation for the faithful llama mmvq port
    bool use_llama = true; // default ON: faithful llama mmvq for Q4_K attn GEMVs (+9.7%, top1 0.99). =0 disables
    bool use_q6mmvq = true;  // default ON: int8 Q6_K mmvq for attn-V upgrades + LM head. =0 disables
    bool use_qkvstream = true; // default ON: run Q/K/V projections on concurrent streams. =0 disables
    bool use_qkfuse = true;// default ON: fused per-head Q-norm + K-norm (1 kernel). =0 disables
    bool use_ropekv = true;// default ON: fused RoPE + KV-append (1 kernel vs 2). =0 disables
    bool use_attnin = true;// default ON: single fused QK-norm+RoPE+KV-append (1 kernel vs qkfuse+ropekv=2). =0 disables
    bool use_fnq = true;   // default ON: post-MoE add_rmsnorm2 also emits Q8_1(xn), deleting the
                           // next layer's standalone QKV-input quantize node. =0 disables
    bool use_gdn_pipe = true;   // default ON: overlap GDN gate/scalar projections on side streams. =0 disables
    bool use_gdn_quad = false;  // default OFF: one-grid GDN Q4_K quad (H=2048). =1 enables
    bool use_attn_qkv = true;   // default ON: one-grid full-attn QKV MMVQ (Q4_K, H=2048). =0 disables
    bool use_shexp_pipe = true; // default ON: overlap shared expert with routed MoE. =0 disables
    bool use_addnorm3 = true;   // default ON: fold routed+shared residual_add into post-MoE add_rmsnorm. =0 disables
    bool use_router_fused = true; // default ON (256-expert path): fuse the router GEMV + bitonic top-k
                                  // into one kernel (grid-completion), dropping the top-k launch. =0 disables

    // Prefix KV reuse (Genie-style warm prompt): cache_prefix() retains KV + GDN state.
    std::vector<int> prefix_tokens;
    int prefix_len = 0;
    int prefix_next = -1;
    bool prefix_active = false;

    // Optional external KV cache tier (docs/lmcache_bridge_protocol.md). Null = disabled (the
    // default) -- every lookup/store call site below is a no-op when this is null, so nothing
    // about existing behavior changes unless a caller explicitly opts in via
    // set_lmcache_bridge().
    BridgeClient* lmcache_bridge = nullptr;

    // DFlash speculative decoding (target-side primitives).
    DFlashDraftModel* dflash_draft = nullptr;
    // Preserve the native Q6_K head for the draft's multi-row MMVQ while the
    // target uses its faster, narrower Q4_K requantized copy.
    const void* dflash_lm_head = nullptr;
    int dflash_lm_head_type = 0;
    bool dflash_capture = false;
    // DFlash verify token 0: enqueue the captured decode graph and collect it later, so the
    // draft block can be issued in between (see dflash_generate).
    bool defer_decode_sync = false;
    bool decode_pending = false;
    std::vector<int> dflash_layer_ids;
    int dflash_n_cap = 0;
    int dflash_max_rows = 16;
    int dflash_cap_row = 0;
    int final_seqlen_hint = -1;   // set by generate()/dflash_generate() before their prefill loop
    int dflash_ctx_len = 0;
    int dflash_ctx_cap = 0;
    bf16* dflash_hidden = nullptr;    // [max_rows, n_cap * H]
    bf16* dflash_context = nullptr;   // [ctx_cap, n_cap * H]
    float* spec_lin_snap = nullptr;
    bf16* spec_conv_snap = nullptr;

    template <class T> T* alloc(size_t n) { void* p=nullptr; cu(cudaMalloc(&p, n*sizeof(T)), "malloc"); return (T*)p; }
};

Qwen35Model::Qwen35Model(const Qwen35Config& cfg, KVCacheManager* kv, moe::MoEEngine* engine)
    : p_(new Impl()) {
    p_->cfg = cfg; p_->kv = kv; p_->engine = engine;
    // Flash-decode KV-split count is occupancy tuning only (math is identical for any
    // value — empty splits contribute zero), and it's baked into the decode CUDA graph
    // at construction. 16 over-subscribes the GPU for short context (32 q_heads * 16 =
    // 512 single-warp blocks); SPARKINFER_NSPLITS lets the scored regime be tuned/swept
    // without a rebuild. Clamp to [1, 64]; buffers below are sized from it.
    if (const char* ns = getenv("SPARKINFER_NSPLITS")) {
        int v = atoi(ns); if (v < 1) v = 1; if (v > Impl::MAX_NSPLITS) v = Impl::MAX_NSPLITS; p_->n_splits = v;
        p_->adaptive_splits = false;   // fixed n_splits (A/B/sweeps)
        fprintf(stderr, "[nsplits] flash-decode splits = %d (fixed env override)\n", v);
    }
    if (const char* c = getenv("SPARKINFER_SPLIT_CHUNK")) { int v = atoi(c); if (v > 0) p_->split_chunk = v; }
    p_->qdim = cfg.n_q_heads * cfg.head_dim;
    p_->kvdim = cfg.n_kv_heads * cfg.head_dim;
    p_->linear_qdim = cfg.linear_q_heads * cfg.linear_head_dim;
    p_->linear_vdim = cfg.linear_v_heads * cfg.linear_head_dim;
    p_->linear_qkvdim = 2 * p_->linear_qdim + p_->linear_vdim;
    cudaStreamCreate(&p_->stream);
    cudaStreamCreate(&p_->stream_k); cudaStreamCreate(&p_->stream_v);
    // Prefetch stream/events exist ONLY for Muse Glimmer, so every other architecture keeps
    // byte-for-byte the stream and event set it had before this change.
    if (cfg.muse_glimmer) {
        cudaStreamCreate(&p_->stream_pf);
        cudaEventCreateWithFlags(&p_->ev_pf_fork, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&p_->ev_pf_done, cudaEventDisableTiming);
    }
    cudaEventCreateWithFlags(&p_->ev_qkv, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&p_->ev_k, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&p_->ev_v, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&p_->ev_pipe_fork, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&p_->ev_gdn_z, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&p_->ev_gdn_ab, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&p_->ev_sx_gate, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&p_->ev_sx_done, cudaEventDisableTiming);
    const int H = cfg.hidden;
    p_->x=p_->alloc<bf16>(H); p_->xn=p_->alloc<bf16>(H);
    p_->q=p_->alloc<bf16>(p_->qdim); p_->k=p_->alloc<bf16>(p_->kvdim); p_->v=p_->alloc<bf16>(p_->kvdim);
    p_->attn=p_->alloc<bf16>(p_->qdim); p_->ao=p_->alloc<bf16>(H);
    p_->h=p_->alloc<bf16>(H); p_->hn=p_->alloc<bf16>(H);
    p_->routed=p_->alloc<bf16>(H); p_->shared=p_->alloc<bf16>(H);
    if (cfg.hybrid) {
        p_->qraw=p_->alloc<bf16>(p_->qdim * 2);
        p_->qgate=p_->alloc<bf16>(p_->qdim);
        p_->lin_qkv=p_->alloc<bf16>(p_->linear_qkvdim);
        p_->lin_q=p_->alloc<bf16>(p_->linear_qdim);
        p_->lin_k=p_->alloc<bf16>(p_->linear_qdim);
        p_->lin_v=p_->alloc<bf16>(p_->linear_vdim);
        p_->lin_z=p_->alloc<bf16>(p_->linear_vdim);
        p_->lin_alpha=p_->alloc<bf16>(cfg.linear_v_heads);
        p_->lin_beta=p_->alloc<bf16>(cfg.linear_v_heads);
        p_->lin_gdn=p_->alloc<bf16>(p_->linear_vdim);
        p_->lin_norm=p_->alloc<bf16>(p_->linear_vdim);
        p_->lin_conv_state=p_->alloc<bf16>((size_t)cfg.n_layers * (cfg.linear_conv_kernel - 1) * p_->linear_qkvdim);
        p_->lin_state=p_->alloc<float>((size_t)cfg.n_layers * cfg.linear_v_heads * cfg.linear_head_dim * cfg.linear_head_dim);
        p_->shared_gate_tmp=p_->alloc<bf16>(1);
    }
    p_->logits=p_->alloc<float>(cfg.vocab);
    // presence_penalty/frequency_penalty per-vocab running count, session-0's own buffer (every
    // OTHER session's own count buffer is alloc'd by open_session()) -- unconditional, every
    // model, unlike lin_state/lin_conv_state above (hybrid-architecture-only).
    p_->penalty_counts_default=p_->alloc<int>(cfg.vocab);
    p_->penalty_counts=p_->penalty_counts_default;
    cu(cudaMemsetAsync(p_->penalty_counts_default, 0, (size_t)cfg.vocab * sizeof(int), p_->stream),
       "penalty_counts_default zero");
    // logit_bias per-vocab additive bias, session-0's own buffer -- mirrors penalty_counts_default
    // exactly (unconditional, every model).
    p_->logit_bias_default=p_->alloc<float>(cfg.vocab);
    p_->logit_bias=p_->logit_bias_default;
    cu(cudaMemsetAsync(p_->logit_bias_default, 0, (size_t)cfg.vocab * sizeof(float), p_->stream),
       "logit_bias_default zero");
    p_->d_logit_bias_ids=p_->alloc<int>(kMaxLogitBiasEntries);
    p_->d_logit_bias_vals=p_->alloc<float>(kMaxLogitBiasEntries);
    cu(cudaHostAlloc(&p_->h_logit_bias_ids, kMaxLogitBiasEntries * sizeof(int), cudaHostAllocDefault),
       "host logit_bias ids");
    cu(cudaHostAlloc(&p_->h_logit_bias_vals, kMaxLogitBiasEntries * sizeof(float), cudaHostAllocDefault),
       "host logit_bias vals");
    p_->d_scalars=p_->alloc<int>(5);
    p_->d_tok=p_->d_scalars + 0; p_->d_pos=p_->d_scalars + 1;
    p_->d_writepos=p_->d_scalars + 2; p_->d_seqlen=p_->d_scalars + 3;
    p_->d_cap_row=p_->d_scalars + 4;
    p_->d_out_id=p_->alloc<int>(1);
    cu(cudaHostAlloc(&p_->h_scalars, 5 * sizeof(int), cudaHostAllocDefault), "host scalars");
    cu(cudaHostAlloc(&p_->h_out_id, sizeof(int), cudaHostAllocDefault), "host out id");
    p_->d_sample_temp=p_->alloc<float>(1);
    p_->d_sample_seed=p_->alloc<unsigned long long>(1);
    p_->d_sample_step=p_->alloc<unsigned long long>(1);
    cu(cudaHostAlloc(&p_->h_sample_temp, sizeof(float), cudaHostAllocDefault), "host sample temp");
    cu(cudaHostAlloc(&p_->h_sample_seed, sizeof(unsigned long long), cudaHostAllocDefault), "host sample seed");
    cu(cudaHostAlloc(&p_->h_sample_step, sizeof(unsigned long long), cudaHostAllocDefault), "host sample step");
    *p_->h_sample_temp = 0.f;
    p_->d_sample_top_k=p_->alloc<int>(1);
    p_->d_sample_top_p=p_->alloc<float>(1);
    cu(cudaHostAlloc(&p_->h_sample_top_k, sizeof(int), cudaHostAllocDefault), "host sample top_k");
    cu(cudaHostAlloc(&p_->h_sample_top_p, sizeof(float), cudaHostAllocDefault), "host sample top_p");
    *p_->h_sample_top_k = 0;
    *p_->h_sample_top_p = 1.f;
    p_->d_sample_presence_penalty=p_->alloc<float>(1);
    p_->d_sample_frequency_penalty=p_->alloc<float>(1);
    cu(cudaHostAlloc(&p_->h_sample_presence_penalty, sizeof(float), cudaHostAllocDefault), "host sample presence_penalty");
    cu(cudaHostAlloc(&p_->h_sample_frequency_penalty, sizeof(float), cudaHostAllocDefault), "host sample frequency_penalty");
    *p_->h_sample_presence_penalty = 0.f;
    *p_->h_sample_frequency_penalty = 0.f;
    // top_k/top_p truncation scratch -- allocated once here (vocab-sized, fixed address), never
    // reallocated per-call. Sizing the two CUB temp-storage buffers is a load-time-only call, no
    // kernel launch involved.
    p_->d_vocab_iota=p_->alloc<int>(cfg.vocab);
    p_->d_sorted_logits=p_->alloc<float>(cfg.vocab);
    p_->d_sorted_idx=p_->alloc<int>(cfg.vocab);
    p_->d_topk_exp=p_->alloc<float>(cfg.vocab);
    p_->d_topk_cumsum=p_->alloc<float>(cfg.vocab);
    kernels::launch_vocab_iota_init(p_->d_vocab_iota, cfg.vocab);
    p_->sort_temp_bytes = kernels::topk_sort_temp_storage_bytes(cfg.vocab);
    p_->scan_temp_bytes = kernels::topk_scan_temp_storage_bytes(cfg.vocab);
    cu(cudaMalloc(&p_->d_sort_temp, p_->sort_temp_bytes), "topk sort temp");
    cu(cudaMalloc(&p_->d_scan_temp, p_->scan_temp_bytes), "topk scan temp");
    // logprobs/top_logprobs scratch, same discipline as above. d_rank_by_id needs no memset --
    // topk_topp_exp_kernel fully overwrites it every decode step before it is ever read.
    p_->d_rank_by_id=p_->alloc<int>(cfg.vocab);
    p_->d_chosen_logit=p_->alloc<float>(1);
    p_->d_shared_ids=p_->alloc<int>(1); p_->d_shared_w=p_->alloc<float>(1);
    int zero=0; float one=1.f;
    cu(cudaMemcpy(p_->d_shared_ids,&zero,sizeof(int),cudaMemcpyHostToDevice),"shared ids");
    cu(cudaMemcpy(p_->d_shared_w,&one,sizeof(float),cudaMemcpyHostToDevice),"shared w");
    // Fused-expert + flash-decoding decode scratch (batch 1). Allocated here so
    // EVERY load path (set_weights / load_weights / load_gguf) has it — not just
    // GGUF. (fa_* NULL here is what crashed flash_decode_split on the non-GGUF path.)
    p_->mf_logits  = p_->alloc<float>(std::max(1, cfg.n_experts));
    p_->mf_ids     = p_->alloc<int>(std::max(1, cfg.top_k));
    p_->mf_weights = p_->alloc<float>(std::max(1, cfg.top_k));
    p_->mf_counts  = p_->alloc<int>(std::max(1, cfg.n_experts));
    p_->mf_rc      = p_->alloc<unsigned int>(1);
    cu(cudaMemset(p_->mf_rc, 0, sizeof(unsigned int)), "mf_rc zero");   // grid-completion counter starts at 0
    p_->mf_h       = p_->alloc<float>((size_t)std::max(1, cfg.top_k) * cfg.moe_ffn);
    p_->mf_out     = p_->alloc<float>(cfg.hidden);
    if (cfg.dense_ffn && cfg.top_k > 0) {
        cu(cudaMemcpy(p_->mf_ids, &zero, sizeof(int), cudaMemcpyHostToDevice), "dense expert id");
        cu(cudaMemcpy(p_->mf_weights, &one, sizeof(float), cudaMemcpyHostToDevice), "dense expert w");
    }
    if (cfg.n_shared > 0) {
        p_->sx_h  = p_->alloc<float>(cfg.moe_ffn);
        p_->sx_q8 = p_->alloc<char>(kernels::llama_q8_1_bytes(cfg.moe_ffn));
    }
    const size_t fa_n = (size_t)cfg.n_q_heads * Impl::MAX_NSPLITS;   // sized for the adaptive max
    p_->fa_m   = p_->alloc<float>(fa_n);
    p_->fa_l   = p_->alloc<float>(fa_n);
    p_->fa_acc = p_->alloc<float>(fa_n * cfg.head_dim);
    // Sink + sliding-window sparse KV: default ON for Qwythos GQA-4 hd256 (int8 KV).
    // SPARKINFER_SPARSE_KV=0 restores dense full-context flash-decode.
    bool sparse_enable = true;
    if (const char* se = getenv("SPARKINFER_SPARSE_KV")) sparse_enable = (se[0] != '0');
    const bool sparse_gqa4 = cfg.head_dim == 256 && cfg.n_kv_heads > 0 &&
                             cfg.n_q_heads == cfg.n_kv_heads * 4;
    // GQA-8 hd256 (Qwen3.6 full-attn layers), issue #559: same sink+window policy, but
    // realized as a compacted paged-KV view fed to the unmodified dense int8-MMA kernel.
    const bool sparse_gqa8 = cfg.head_dim == 256 && cfg.n_kv_heads > 0 &&
                             cfg.n_q_heads == cfg.n_kv_heads * 8;
    if (sparse_enable && (sparse_gqa4 || sparse_gqa8)) {
        p_->sparse_window = 256;
        if (const char* w = getenv("SPARKINFER_SPARSE_WINDOW")) { int v = atoi(w); if (v > 0) p_->sparse_window = v; }
        // Legacy aliases from the Quest prototype (blocks, not tokens).
        if (const char* rw = getenv("SPARKINFER_SPARSE_RECENT")) { int v = atoi(rw); if (v > 0) p_->sparse_window = v; }
        if (const char* b = getenv("SPARKINFER_SPARSE_BUDGET")) {
            int v = atoi(b); if (v > 1) p_->sparse_window = v - 1;   // budget included sink
        }
        // GQA-8: the dense hd256 path is already int8 tensor-core, so the O(window) read
        // only clears the dense cost decisively from ~16k context up (bot-measured on this
        // shape: sparse under 16k is a wash-to-regression, 16k/32k are wins). Engaging at
        // 16384 also keeps every mid-length scoring probe on the exact dense path.
        if (sparse_gqa8) p_->sparse_min_ctx = 16384;
        if (const char* mc = getenv("SPARKINFER_SPARSE_MIN_CTX")) { int v = atoi(mc); if (v > 0) p_->sparse_min_ctx = v; }
        p_->sparse_budget = 1 + p_->sparse_window;
        if (sparse_gqa8) {
            p_->sparse_vtbl = p_->alloc<int>(p_->sparse_budget);
            p_->sparse_vlen = p_->alloc<int>(1);
            // Split the compact view so each MMA split still covers >= 2 KV blocks (the
            // dense launcher's tensor-core engagement condition), capped at MAX_NSPLITS.
            // window=256 -> 4112-token view -> 128 splits (x8 kv heads = 1024 CTAs).
            int vs = (p_->sparse_budget * kv->block_size()) / 32;
            if (vs > Impl::MAX_NSPLITS) vs = Impl::MAX_NSPLITS;
            if (vs < 1) vs = 1;
            p_->sparse_vsplits = vs;
        } else {
            p_->sparse_sel = p_->alloc<int>((size_t)cfg.n_kv_heads * p_->sparse_budget);
        }
        fprintf(stderr, "[sparse-kv] sliding-window (default on): gqa=%d window=%d blocks (%d tokens) min_ctx=%d%s\n",
                sparse_gqa8 ? 8 : 4, p_->sparse_window, p_->sparse_window * kv->block_size(),
                p_->sparse_min_ctx, sparse_gqa8 ? " (compact-view, decode-only)" : "");
    }
    // Muse Glimmer: mandatory pure sliding-window view for swa-flagged layers, every step,
    // regardless of context length (architectural, not a long-context approximation).
    if (cfg.muse_glimmer && cfg.sliding_window > 0) {
        p_->swa_budget = (cfg.sliding_window + kv->block_size() - 1) / kv->block_size();
        p_->swa_vtbl = p_->alloc<int>(p_->swa_budget);
        p_->swa_vlen = p_->alloc<int>(1);
        int vs = (p_->swa_budget * kv->block_size()) / 32;
        if (vs > Impl::MAX_NSPLITS) vs = Impl::MAX_NSPLITS;
        if (vs < 1) vs = 1;
        p_->swa_vsplits = vs;
        fprintf(stderr, "[muse-glimmer] sliding-window: %d tokens (%d blocks), every-4th-layer global/NoPE\n",
                cfg.sliding_window, p_->swa_budget);
        // Unweighted RMSNorm applied to the token embedding before layer 0 (no learned
        // per-channel weight -- launch_rmsnorm always takes one, so fill a constant-1.0
        // buffer once at load time and reuse it as that "weight").
        p_->emb_norm_ones = p_->alloc<bf16>(H);
        std::vector<bf16> ones(H, (bf16)0x3F80u);   // bf16 bit pattern for 1.0f
        cudaMemcpy(p_->emb_norm_ones, ones.data(), (size_t)H * sizeof(bf16), cudaMemcpyHostToDevice);
    }
    const int kmax = (p_->qdim > H) ? p_->qdim : H;          // largest projection input dim
    p_->aq8   = p_->alloc<signed char>(kmax);
    p_->aq8_d = p_->alloc<float>(kmax >> 5);
    p_->aq8_s = p_->alloc<float>(kmax >> 5);
    p_->aq81  = p_->alloc<char>(kernels::llama_q8_1_bytes(kmax));
    if (const char* e = getenv("SPARKINFER_PQ"))    p_->use_pq    = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_LLAMA")) p_->use_llama = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_Q6MMVQ")) p_->use_q6mmvq = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_QKFUSE")) p_->use_qkfuse = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_ROPEKV")) p_->use_ropekv = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_FNQ"))    p_->use_fnq   = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_QKVSTREAM")) p_->use_qkvstream = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_ATTNIN")) p_->use_attnin = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_GDN_PIPE")) p_->use_gdn_pipe = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_GDN_QUAD")) p_->use_gdn_quad = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_ATTN_QKV")) p_->use_attn_qkv = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_SHEXP_PIPE")) p_->use_shexp_pipe = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_ADDNORM3")) p_->use_addnorm3 = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_ROUTER_FUSED")) p_->use_router_fused = !(e[0] == '0');
    {
        SessionBuffers d;
        if (cfg.hybrid) {
            d.lin_state = p_->lin_state;
            d.lin_conv_state = p_->lin_conv_state;
        }
        d.penalty_counts = p_->penalty_counts_default;   // unconditional -- every model
        d.logit_bias = p_->logit_bias_default;           // unconditional -- every model
        p_->sessions[0] = d;
    }
}

Qwen35Model::~Qwen35Model() {
    for (void* b : p_->owned) cudaFree(b);
    cudaFree(p_->x); cudaFree(p_->xn); cudaFree(p_->q); cudaFree(p_->k); cudaFree(p_->v);
    cudaFree(p_->attn); cudaFree(p_->ao); cudaFree(p_->h); cudaFree(p_->hn);
    cudaFree(p_->routed); cudaFree(p_->shared); cudaFree(p_->logits);
    cudaFree(p_->penalty_counts_default);
    cudaFree(p_->logit_bias_default);
    cudaFree(p_->d_logit_bias_ids); cudaFree(p_->d_logit_bias_vals);
    cudaFreeHost(p_->h_logit_bias_ids); cudaFreeHost(p_->h_logit_bias_vals);
    // main's packed decode scalars (d_tok/d_pos/d_seqlen/d_writepos alias into d_scalars — not freed separately)
    cudaFree(p_->d_scalars); cudaFree(p_->d_out_id);
    cudaFreeHost(p_->h_scalars); cudaFreeHost(p_->h_out_id);
    cudaFree(p_->d_sample_temp); cudaFree(p_->d_sample_seed); cudaFree(p_->d_sample_step);
    cudaFreeHost(p_->h_sample_temp); cudaFreeHost(p_->h_sample_seed); cudaFreeHost(p_->h_sample_step);
    cudaFree(p_->d_sample_top_k); cudaFree(p_->d_sample_top_p);
    cudaFreeHost(p_->h_sample_top_k); cudaFreeHost(p_->h_sample_top_p);
    cudaFree(p_->d_sample_presence_penalty); cudaFree(p_->d_sample_frequency_penalty);
    cudaFreeHost(p_->h_sample_presence_penalty); cudaFreeHost(p_->h_sample_frequency_penalty);
    cudaFree(p_->d_vocab_iota); cudaFree(p_->d_sorted_logits); cudaFree(p_->d_sorted_idx);
    cudaFree(p_->d_topk_exp); cudaFree(p_->d_topk_cumsum);
    cudaFree(p_->d_sort_temp); cudaFree(p_->d_scan_temp);
    cudaFree(p_->d_rank_by_id); cudaFree(p_->d_chosen_logit);
    cudaFree(p_->d_shared_ids); cudaFree(p_->d_shared_w);
    // Qwen3.6 Gated-DeltaNet buffers (allocated only for the hybrid model)
    cudaFree(p_->qraw); cudaFree(p_->qgate);
    cudaFree(p_->dbg_xn_dump);
    cudaFree(p_->lin_qkv); cudaFree(p_->lin_q); cudaFree(p_->lin_k); cudaFree(p_->lin_v);
    cudaFree(p_->lin_z); cudaFree(p_->lin_alpha); cudaFree(p_->lin_beta);
    cudaFree(p_->lin_gdn); cudaFree(p_->lin_norm); cudaFree(p_->lin_conv_state); cudaFree(p_->lin_state);
    cudaFree(p_->shared_gate_tmp);
    cudaFree(p_->mf_logits); cudaFree(p_->mf_weights); cudaFree(p_->mf_h); cudaFree(p_->mf_out);
    cudaFree(p_->sx_h); cudaFree(p_->sx_q8);
    cudaFree(p_->mf_ids); cudaFree(p_->mf_counts); cudaFree(p_->mf_rc);
    cudaFree(p_->moe_rs_gate); cudaFree(p_->moe_rs_up); cudaFree(p_->moe_rs_down);
    cudaFree(p_->muse_rs);
    cudaFree(p_->fa_m); cudaFree(p_->fa_l); cudaFree(p_->fa_acc);
    cudaFree(p_->sparse_sel);
    cudaFree(p_->sparse_vtbl); cudaFree(p_->sparse_vlen);
    cudaFree(p_->swa_vtbl); cudaFree(p_->swa_vlen); cudaFree(p_->emb_norm_ones);
    cudaFree(p_->aq8); cudaFree(p_->aq8_d); cudaFree(p_->aq8_s); cudaFree(p_->aq81);
    cudaFree(p_->dflash_hidden); cudaFree(p_->dflash_context);
    // spec_lin_snap / spec_conv_snap are in owned[] (allocated via Impl::alloc)
    for (auto& kv : p_->sessions) {
        if (kv.first == 0) continue;
        if (kv.second.lin_state) cudaFree(kv.second.lin_state);
        if (kv.second.lin_conv_state) cudaFree(kv.second.lin_conv_state);
    }
    if (p_->graph_ready) { cudaGraphExecDestroy(p_->cu_exec); cudaGraphDestroy(p_->cu_graph); }
    if (p_->graph_prefill_ready) { cudaGraphExecDestroy(p_->cu_prefill_exec); cudaGraphDestroy(p_->cu_prefill_graph); }
    if (p_->dflash_graph_ready) { cudaGraphExecDestroy(p_->cu_dflash_exec); cudaGraphDestroy(p_->cu_dflash_graph); }
    if (p_->ev_pf_fork) cudaEventDestroy(p_->ev_pf_fork);
    if (p_->ev_pf_done) cudaEventDestroy(p_->ev_pf_done);
    cudaEventDestroy(p_->ev_qkv); cudaEventDestroy(p_->ev_k); cudaEventDestroy(p_->ev_v);
    cudaEventDestroy(p_->ev_pipe_fork); cudaEventDestroy(p_->ev_gdn_z); cudaEventDestroy(p_->ev_gdn_ab);
    cudaEventDestroy(p_->ev_sx_gate); cudaEventDestroy(p_->ev_sx_done);
    if (p_->stream_pf) cudaStreamDestroy(p_->stream_pf);
    cudaStreamDestroy(p_->stream_v); cudaStreamDestroy(p_->stream_k);
    cudaStreamDestroy(p_->stream);
    delete p_;
}

void Qwen35Model::set_weights(const Qwen35Weights& w) { p_->w = w; }
const Qwen35Config& Qwen35Model::config() const { return p_->cfg; }

void Qwen35Model::copy_logits(float* host_logits) const {
    // p_->logits holds the last step's lm-head output; forward_token() syncs the
    // stream before returning, so it is valid to read here.
    cudaMemcpy(host_logits, p_->logits, (size_t)p_->cfg.vocab * sizeof(float), cudaMemcpyDeviceToHost);
}

Qwen35Model::TokenLogprob Qwen35Model::last_token_logprobs(int top_n) const {
    Impl& s = *p_;
    top_n = std::max(0, std::min(top_n, kMaxTopLogprobs));

    TokenLogprob out;
    out.token_id = *s.h_out_id;   // already synced by forward_token() before it returned

    float denom = 1.f, chosen_logit = 0.f;
    cudaMemcpy(&denom, s.d_topk_cumsum + (s.cfg.vocab - 1), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&chosen_logit, s.d_chosen_logit, sizeof(float), cudaMemcpyDeviceToHost);

    float row_max = 0.f;
    std::vector<int> ids;
    std::vector<float> logits;
    if (top_n > 0) {
        ids.resize(top_n);
        logits.resize(top_n);
        cudaMemcpy(ids.data(), s.d_sorted_idx, (size_t)top_n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(logits.data(), s.d_sorted_logits, (size_t)top_n * sizeof(float), cudaMemcpyDeviceToHost);
        row_max = logits[0];   // rank 0 is always the row max, free from the descending sort
    } else {
        cudaMemcpy(&row_max, s.d_sorted_logits, sizeof(float), cudaMemcpyDeviceToHost);
    }

    const float logsumexp = row_max + logf(denom);
    out.logprob = chosen_logit - logsumexp;
    out.top_alternatives.reserve((size_t)top_n);
    for (int i = 0; i < top_n; i++) out.top_alternatives.emplace_back(ids[i], logits[i] - logsumexp);
    return out;
}

void Qwen35Model::dflash_maybe_capture_layer(int layer) {
    Impl& s = *p_;
    if (!s.dflash_capture || !s.dflash_hidden || s.dflash_n_cap <= 0) return;
    int slot = -1;
    for (int i = 0; i < s.dflash_n_cap; i++) {
        if (s.dflash_layer_ids[i] == layer) { slot = i; break; }
    }
    if (slot < 0) return;
    const int H = s.cfg.hidden;
    // Writes dflash_hidden[cap_row][slot] directly. cap_row is read from d_scalars[4] on the
    // device, so this node is graph-capturable even though the row changes every verify token --
    // which retires the staging buffer and the extra out-of-graph flush memcpy that every DFlash
    // verify token paid on top of a plain decode forward.
    dflash_kernels::launch_capture_row(s.x, s.dflash_hidden, s.d_cap_row, slot, H,
                                       s.dflash_n_cap * H, s.dflash_max_rows, s.stream);
}

// Depth-adaptive KV-split count for a given seqlen: 32 (short) -> 128 (mid) -> 256 (long), plus
// the hd256/GQA occupancy correction. Extracted so both forward_token()'s per-token adaptation
// and dflash_generate()'s one-time pre-capture initialization compute the exact same value —
// see forward_token() below for why dflash_generate needs its own call to this.
int Qwen35Model::adaptive_nsplits_for(int seqlen) const {
    const Impl& s = *p_;
    const Qwen35Config& c = s.cfg;
    int want = 32;
    if ((long)seqlen > 2L * s.split_chunk) want = 128;
    if ((long)seqlen > 28L * s.split_chunk && (long)seqlen <= 48L * s.split_chunk)
        want = Impl::MAX_NSPLITS;
    if ((long)seqlen > 64L * s.split_chunk) want = Impl::MAX_NSPLITS;
    if (want > Impl::MAX_NSPLITS) want = Impl::MAX_NSPLITS;
    // hd256/GQA-8 occupancy correction (Qwen3.6 full-attention shape specifically) — see the
    // #707-era measurement notes this replaces for the exact tuning rationale (flat 160 through
    // 32k for GQA-8; GQA-4 promotes further at 64k/128k).
    if (c.head_dim == 256 && c.n_kv_heads > 0 && want >= 128) {
        if (c.n_q_heads == c.n_kv_heads * 8)
            want = 160;
        else if (c.n_q_heads == c.n_kv_heads * 4) {
            if ((long)seqlen > 98304L)           want = 128;  // 128k decode (seqlen ~131k)
            else if ((long)seqlen > 65536L)      want = 192;  // 64k decode band
            else                                 want = 160;
        }
    }
    return want;
}

int Qwen35Model::forward_token(int token_id, int position, bool sample, float temperature,
                               unsigned long long seed, unsigned long long sample_step,
                               int top_k, float top_p,
                               float presence_penalty, float frequency_penalty) {
    Impl& s = *p_;
    const Qwen35Config& c = s.cfg;
    const int H = c.hidden;
    kernels::GemmConfig gc{};
    int seqlen = position + 1;
    cudaStream_t st = s.stream;
    const bool dflash_cap = s.dflash_capture;

    // DEBUG ONLY (Muse Glimmer bring-up bisection): SPARKINFER_MG_STAGE_DEBUG=1 dumps
    // L2 norm + first 3 values of the residual stream at each pipeline stage, for every
    // forward_token call. Forces a fresh capture+launch every step (never replays an old
    // graph) so the "step" label baked into each debug kernel's launch params is always
    // accurate -- acceptable perf hit for a short debug run only. Remove once the bug hunt
    // concludes; harmless no-op (env unset) otherwise.
    static int mg_dbg = -1;
    if (mg_dbg < 0) { const char* e = getenv("SPARKINFER_MG_STAGE_DEBUG"); mg_dbg = (e && e[0] == '1') ? 1 : 0; }
    const bool mgd = c.muse_glimmer && mg_dbg;
    if (mgd && s.graph_ready) {
        cudaGraphExecDestroy(s.cu_exec); cudaGraphDestroy(s.cu_graph);
        s.cu_exec = nullptr; s.cu_graph = nullptr; s.graph_ready = false;
    }
    auto dbg_bf16 = [&](const void* p, int n, int tag, int layer) {
        if (mgd) kernels::launch_mg_debug_bf16(p, n, tag, layer, position, st);
    };
    auto dbg_f32 = [&](const float* p, int n, int tag, int layer) {
        if (mgd) kernels::launch_mg_debug_f32(p, n, tag, layer, position, st);
    };
    // DEBUG ONLY: SPARKINFER_MG_DUMP_STEP=<position> additionally raw-dumps tag=10 (this
    // layer's pre-attn-norm xn) for EVERY layer at that one decode step into a [n_layers,H]
    // bf16 device buffer, D2H-copied and written to SPARKINFER_MG_DUMP_FILE (default
    // /tmp/mg_xn_dump.bin) right after this step's graph launch is synced. Lets a later-layer
    // hypothesis be checked against a from-scratch Python reference seeded with sparkinfer's
    // OWN xn for that layer, without needing to replicate every earlier layer.
    static int dump_step = -2;
    if (dump_step < -1) { const char* e = getenv("SPARKINFER_MG_DUMP_STEP"); dump_step = e ? atoi(e) : -1; }
    const bool mgdump = mgd && dump_step == position;
    if (mgdump && !s.dbg_xn_dump)
        cu(cudaMalloc(&s.dbg_xn_dump, (size_t)(c.n_layers + 1) * H * sizeof(bf16)), "dbg_xn_dump alloc");
    auto dbg_xn_snapshot = [&](const void* p, int layer) {
        if (mgdump) cu(cudaMemcpyAsync(s.dbg_xn_dump + (size_t)layer * H, p, (size_t)H * sizeof(bf16),
                                       cudaMemcpyDeviceToDevice, st), "dbg_xn_dump copy");
    };

    s.h_scalars[0] = token_id;
    s.h_scalars[1] = position;
    s.h_scalars[2] = position;
    s.h_scalars[3] = seqlen;
    s.h_scalars[4] = s.dflash_cap_row;
    cu(cudaMemcpyAsync(s.d_scalars, s.h_scalars, 5 * sizeof(int), cudaMemcpyHostToDevice, st), "decode scalars");
    // Refreshed on every call (both capture and replay paths) so the decode graph's sampling
    // kernel -- always launched unconditionally inside the captured region, see below -- picks
    // up THIS call's temperature/seed/step rather than whatever was baked in at capture time.
    *s.h_sample_temp = temperature;
    *s.h_sample_seed = seed;
    *s.h_sample_step = sample_step;
    cu(cudaMemcpyAsync(s.d_sample_temp, s.h_sample_temp, sizeof(float), cudaMemcpyHostToDevice, st), "sample temp");
    cu(cudaMemcpyAsync(s.d_sample_seed, s.h_sample_seed, sizeof(unsigned long long), cudaMemcpyHostToDevice, st), "sample seed");
    cu(cudaMemcpyAsync(s.d_sample_step, s.h_sample_step, sizeof(unsigned long long), cudaMemcpyHostToDevice, st), "sample step");
    *s.h_sample_top_k = top_k;
    *s.h_sample_top_p = top_p;
    cu(cudaMemcpyAsync(s.d_sample_top_k, s.h_sample_top_k, sizeof(int), cudaMemcpyHostToDevice, st), "sample top_k");
    cu(cudaMemcpyAsync(s.d_sample_top_p, s.h_sample_top_p, sizeof(float), cudaMemcpyHostToDevice, st), "sample top_p");
    *s.h_sample_presence_penalty = presence_penalty;
    *s.h_sample_frequency_penalty = frequency_penalty;
    cu(cudaMemcpyAsync(s.d_sample_presence_penalty, s.h_sample_presence_penalty, sizeof(float), cudaMemcpyHostToDevice, st), "sample presence_penalty");
    cu(cudaMemcpyAsync(s.d_sample_frequency_penalty, s.h_sample_frequency_penalty, sizeof(float), cudaMemcpyHostToDevice, st), "sample frequency_penalty");

    // Depth-adaptive KV-split (see adaptive_nsplits_for() above for the tiers/occupancy math).
    // DFlash DECODE: freeze n_splits once an actual dflash verify graph is captured. Adapting it
    // while that graph is live (e.g. the 32->160 jump at seqlen>2*split_chunk ~= 512) invalidates
    // + re-captures it mid-stream, corrupting the compact-verify state (spurious token 0, then
    // repeat). Gating on dflash_capture alone (rather than dflash_capture && dflash_graph_ready)
    // was wrong: dflash_capture is set true before prefill even starts, so that blanket guard also
    // froze n_splits during PREFILL, where sample=false never captures any graph at all and there
    // is nothing to protect. Prefill then ran every position (short depths included) at whatever
    // single value happened to be inherited from framework state predating this call, instead of
    // ramping 32->128->160 with depth the way the reference (AR / pre-freeze) path does — a small
    // per-split floating-point rounding difference in the quantized-KV reduction that compounds
    // over thousands of positions into hidden states the draft model no longer agrees with
    // (verified: SPEC_AGREE collapses at 4k-ctx even though the final prefill logits/argmax token
    // are unaffected — only DFlash's own stashed hidden-state capture diverges). Gating on
    // dflash_graph_ready specifically lets prefill keep adapting exactly like the non-DFlash path,
    // and only starts freezing once there is a live graph that a change would actually corrupt.
    if (s.adaptive_splits && !(s.dflash_capture && s.dflash_graph_ready)) {
        // DFlash's decode graph freezes n_splits the instant it is captured (below) and never
        // revisits it again -- so if the freeze happens to land on a call where the CURRENT
        // position is still in a lower tier than where the rest of the generation will run (e.g.
        // a prompt just under 512 tokens, whose decode phase crosses seqlen>512 partway through),
        // the frozen value is permanently wrong for the back half of the run. AR's own graph
        // doesn't have this problem in isolation -- it re-adapts every step, safely recapturing
        // as seqlen grows -- but that means AR and a frozen DFlash graph would legitimately use
        // DIFFERENT split counts for the SAME position (AR=32, DFlash=160, both individually
        // correct for their own path but mismatched against each other), which is its own
        // source of divergence. So both sides peek ahead at the very moment their own first
        // decode-graph capture happens (final_seqlen_hint, set by generate()/dflash_generate()
        // before their prefill loop) and settle on the SAME tier for the whole decode phase,
        // rather than one side transitioning mid-run and the other starting there already.
        // Elsewhere (prefill positions, and any later step once a graph is already ready) this
        // branch doesn't fire, so prefill keeps adapting exactly as before.
        int want = adaptive_nsplits_for(seqlen);
        // DFlash only needs the hint applied once, right as its graph freezes (below) -- after
        // that this whole outer branch stops running (dflash_graph_ready gates it off), so the
        // frozen value simply stays. AR has no such freeze: it re-derives want from the CURRENT
        // seqlen on every single decode step and recaptures whenever that differs from s.n_splits
        // (by design, so it can safely track a long prefill's own adaptation). Applying the hint
        // to AR only once would just get overwritten back down on the very next step once seqlen
        // no longer matches the hinted tier -- it has to be applied on every decode step so AR
        // settles on and stays at the same tier DFlash is frozen at, instead of legitimately
        // transitioning mid-decode while DFlash cannot follow.
        const bool about_to_freeze_dflash = s.dflash_capture && !s.dflash_graph_ready;
        const bool ar_decode_step = sample && !s.dflash_capture;
        if (((sample && about_to_freeze_dflash) || ar_decode_step) && s.final_seqlen_hint > seqlen) {
            const int want_final = adaptive_nsplits_for(s.final_seqlen_hint);
            if (want_final > want) want = want_final;
        }
        if (want != s.n_splits) {                       // changed -> invalidate the captured graph
            s.n_splits = want;
            if (s.graph_ready) {
                cudaGraphExecDestroy(s.cu_exec); cudaGraphDestroy(s.cu_graph); s.graph_ready = false;
            }
            if (s.graph_prefill_ready) {
                cudaGraphExecDestroy(s.cu_prefill_exec); cudaGraphDestroy(s.cu_prefill_graph);
                s.cu_prefill_exec = nullptr; s.cu_prefill_graph = nullptr;
                s.graph_prefill_ready = false; s.graph_prefill_attn_mode = -1;
            }
            if (s.dflash_graph_ready) {
                cudaGraphExecDestroy(s.cu_dflash_exec); cudaGraphDestroy(s.cu_dflash_graph);
                s.cu_dflash_exec = nullptr; s.cu_dflash_graph = nullptr;
                s.dflash_graph_ready = false; s.dflash_graph_attn_mode = -1;
            }
        }
    }
    // launch_flash_decode_split chooses its scalar-vs-MMA implementation on the host
    // while the graph is captured. If int8 KV is enabled for a long-context run, a graph
    // captured at a short seqlen would otherwise keep replaying the scalar int8 path after
    // the sequence is large enough for the tensor-core path. Recapture at that mode change.
    static int famma_graph = -1;
    if (famma_graph < 0) {
        const char* e = getenv("SPARKINFER_FAMMA");
        famma_graph = (e && e[0] == '0') ? 0 : 1;
    }
    static int famma4_graph = -1;
    if (famma4_graph < 0) {
        const char* e = getenv("SPARKINFER_FAMMA4");
        famma4_graph = (e && e[0] == '0') ? 0 : 1;
    }
    int attn_graph_mode = 0;
    if (famma_graph && s.kv->int8_kv() && s.kv->block_size() == 16 &&
        c.n_kv_heads > 0 && c.n_q_heads == c.n_kv_heads * 8) {
        const int mma_chunk = (s.n_splits > 0) ? (seqlen + s.n_splits - 1) / s.n_splits : 0;
        attn_graph_mode = (seqlen > 512 && mma_chunk >= 32) ? 2 : 1;
    } else if (famma4_graph && s.kv->int8_kv() && s.kv->block_size() == 16 &&
               c.n_kv_heads > 0 && c.n_q_heads == c.n_kv_heads * 4) {
        const int mma_chunk = (s.n_splits > 0) ? (seqlen + s.n_splits - 1) / s.n_splits : 0;
        attn_graph_mode = (seqlen > 512 && mma_chunk >= 32) ? 3 : 1;
    }
    if (s.graph_ready && attn_graph_mode != s.graph_attn_mode) {
        cu(cudaGraphExecDestroy(s.cu_exec), "graph recapture destroy exec");
        cu(cudaGraphDestroy(s.cu_graph), "graph recapture destroy graph");
        s.cu_exec = nullptr;
        s.cu_graph = nullptr;
        s.graph_ready = false;
    }
    if (s.dflash_graph_ready && attn_graph_mode != s.dflash_graph_attn_mode) {
        cu(cudaGraphExecDestroy(s.cu_dflash_exec), "dflash graph recapture destroy exec");
        cu(cudaGraphDestroy(s.cu_dflash_graph), "dflash graph recapture destroy graph");
        s.cu_dflash_exec = nullptr;
        s.cu_dflash_graph = nullptr;
        s.dflash_graph_ready = false;
    }
    const bool sparse_avail = s.sparse_budget > 0 && s.kv->int8_kv() &&
                              c.head_dim == 256 && c.n_q_heads == c.n_kv_heads * 4;
    // GQA-8 compact-view sparse is decode-only (`sample`): prefill and teacher-forced
    // scoring always run the exact dense path, and the windowed view is never baked into
    // the prefill graph.
    const bool sparse_view_avail = s.sparse_vtbl != nullptr && sample && s.kv->int8_kv() &&
                                   c.head_dim == 256 && c.n_q_heads == c.n_kv_heads * 8;
    const bool sparse_on = (sparse_avail || sparse_view_avail) && seqlen >= s.sparse_min_ctx;
    if (s.graph_ready && s.graph_sparse != sparse_on) {
        cu(cudaGraphExecDestroy(s.cu_exec), "sparse recapture destroy exec");
        cu(cudaGraphDestroy(s.cu_graph), "sparse recapture destroy graph");
        s.cu_exec = nullptr; s.cu_graph = nullptr; s.graph_ready = false;
    }
    if (s.dflash_graph_ready && s.dflash_graph_sparse != sparse_on) {
        cu(cudaGraphExecDestroy(s.cu_dflash_exec), "dflash sparse recapture destroy exec");
        cu(cudaGraphDestroy(s.cu_dflash_graph), "dflash sparse recapture destroy graph");
        s.cu_dflash_exec = nullptr; s.cu_dflash_graph = nullptr; s.dflash_graph_ready = false;
    }
    if (s.graph_prefill_ready && attn_graph_mode != s.graph_prefill_attn_mode) {
        cu(cudaGraphExecDestroy(s.cu_prefill_exec), "prefill graph recapture destroy exec");
        cu(cudaGraphDestroy(s.cu_prefill_graph), "prefill graph recapture destroy graph");
        s.cu_prefill_exec = nullptr;
        s.cu_prefill_graph = nullptr;
        s.graph_prefill_ready = false;
        s.graph_prefill_attn_mode = -1;
    }
    if (c.hybrid && position == 0) {
        cu(cudaMemsetAsync(s.lin_state, 0,
                           (size_t)c.n_layers * c.linear_v_heads * c.linear_head_dim * c.linear_head_dim * sizeof(float), st),
           "linear state reset");
        cu(cudaMemsetAsync(s.lin_conv_state, 0,
                           (size_t)c.n_layers * (c.linear_conv_kernel - 1) * s.linear_qkvdim * sizeof(bf16), st),
           "linear conv reset");
    }

    // Prefill graph: embed→layers→final norm (no LM head). Decode graph: full path + argmax.
    // DFlash's per-layer hidden capture used to require the eager path entirely (a captured
    // graph bakes fixed destinations, and dflash_hidden[cap_row] varies call to call) -- the
    // capture kernel reads cap_row from device memory instead, so DFlash verify tokens
    // (sample=true) get graph replay with no post-replay fixup. The prefill/prompt path
    // (sample=false) still falls back to eager under capture; not on the scored decode path.
    if (!sample && s.graph_prefill_ready && !dflash_cap) {
        cu(cudaGraphLaunch(s.cu_prefill_exec, st), "prefill graph launch");
        cu(cudaStreamSynchronize(st), "prefill graph sync");
        return token_id;
    }
    if (sample && s.graph_ready && !dflash_cap) {
        cu(cudaGraphLaunch(s.cu_exec, st), "graph launch");
        cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "out_id");
        cu(cudaStreamSynchronize(st), "sync");
        return *s.h_out_id;
    }
    if (sample && dflash_cap && s.dflash_graph_ready) {
        cu(cudaGraphLaunch(s.cu_dflash_exec, st), "dflash graph launch");
        cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "out_id");
        // Deferred collect (DFlash verify token 0 only): return without blocking so the caller
        // can issue the draft block behind it. The target forward is then already running on the
        // GPU while the host is still issuing the draft's ~100 launches -- and it costs no
        // std::thread spawn/join per decode step.
        if (s.defer_decode_sync) { s.decode_pending = true; return kDFlashDeferred; }
        cu(cudaStreamSynchronize(st), "sync");
        return *s.h_out_id;
    }
    if (sample && s.graph_prefill_ready) {
        cu(cudaGraphExecDestroy(s.cu_prefill_exec), "drop prefill graph for decode");
        cu(cudaGraphDestroy(s.cu_prefill_graph), "drop prefill graph for decode");
        s.cu_prefill_exec = nullptr;
        s.cu_prefill_graph = nullptr;
        s.graph_prefill_ready = false;
        s.graph_prefill_attn_mode = -1;
    }
    // Reaching here means: no ready graph could be replayed above. For sample=true this is
    // always safe to (re)capture -- either the plain decode graph (dflash_cap false) or the
    // dflash decode graph (dflash_cap true, via the row-independent stage buffer); only the
    // prefill/prompt path (sample=false) still avoids capturing while dflash_cap is on.
    const bool capturing_graph = sample || !dflash_cap;
    if (capturing_graph)
        cu(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), sample ? "begin decode capture" : "begin prefill capture");

    kernels::launch_embedding(s.d_tok, s.w.embed_tokens, s.x, 1, H, st);
    dbg_bf16(s.x, H, 0, -1);   // tag 0: post-embedding, pre emb_norm
    if (c.muse_glimmer && s.emb_norm_ones)
        kernels::launch_rmsnorm(s.x, s.emb_norm_ones, s.x, 1, H, c.rms_eps, st);
    dbg_bf16(s.x, H, 1, -1);   // tag 1: post emb_norm

    int* btable = s.kv->block_table(s.active_seq_id);
    // GQA-8 sparse: materialize the sink+window compact view once per decode step (the
    // logical->physical block map and seq_len are shared by all full-attn layers; the
    // per-layer K/V pool rows for this token are appended before each layer's attention
    // as usual). Inside the capture so replays track the growing sequence.
    if (sparse_on && s.sparse_vtbl)
        kernels::launch_fa_kv_compact_view(s.d_seqlen, btable, s.sparse_vtbl, s.sparse_vlen,
                                           s.kv->block_size(), s.sparse_window, s.sparse_budget, st);
    // Muse Glimmer: pure sliding-window view for swa-flagged layers, every step (mandatory,
    // not gated by context length like the sparse-kv approximation above).
    if (c.muse_glimmer && s.swa_vtbl)
        kernels::launch_fa_kv_compact_view_pure(s.d_seqlen, btable, s.swa_vtbl, s.swa_vlen,
                                                s.kv->block_size(), s.swa_budget, s.swa_budget, st);
    // Prime: xn = RMSNorm(x, layer0.input_norm). Each layer's tail then fuses the
    // post-MoE residual with the NEXT layer's input norm (or final_norm), so the
    // per-layer input RMSNorm + two residual-adds collapse into two fused kernels.
    kernels::launch_rmsnorm(s.x, s.w.layers[0].input_norm, s.xn, 1, H, c.rms_eps, st);

    // When the fused norm+quant path is on, each layer's post-MoE add_rmsnorm2 also emits
    // Q8_1(xn) into aq81, so the next layer's QKV-input quantize (and the LM-head quantize)
    // are already done. Only the prime norm above still needs the standalone quant (layer 0).
    const bool fnq = s.gguf && s.use_fnq && s.use_pq && s.use_llama;

    // ---- L2 weight prefetch into the sandwich-norm tail's idle window ----
    // Decode is DRAM-bound but only ~86% bus-utilized; the gap is time the bus idles inside a
    // latency-bound kernel. Muse Glimmer's two single-CTA sandwich tails per layer are the largest
    // such window (~3.4 us each on one of 170 SMs). Fork a prefetch of the weight matrix the next
    // big GEMV will stream, so it overlaps the tail instead of leaving the bus idle. Each join sits
    // a full block after its fork (FFN between fork1/join1, attention between fork2/join2), so the
    // prefetch never serializes against the main stream. Prefetch has no side effects, so this is
    // bit-identical -- only where a byte is served from changes.
    // SPARKINFER_MG_L2PF_MB=0 disables (one-binary A/B control).
    static int l2pf_mb = -1;
    if (l2pf_mb < 0) {
        const char* e = getenv("SPARKINFER_MG_L2PF_MB");
        l2pf_mb = e ? atoi(e) : 5;   // flat optimum over 3-5 MB
        if (l2pf_mb < 0) l2pf_mb = 0;
    }
    // Each window is forked immediately before a latency-bound stretch and joined immediately
    // before the matrix it prefetched is streamed, so the prefetch always has the full window to
    // land and never stalls the main stream. Forking earlier than this is WORSE, not better: a
    // variant that spanned the whole attention block measured +0.54% against this schedule's
    // +0.99%, because attention streams ~47 MB through a 96 MB L2 and evicts the prefetch before
    // the FFN reads it. Sizing is a flat optimum over 3-6 MB; past ~24 MB the prefetch outruns its
    // window and the join serializes, which costs more than half the step.
    // Window 1 gets its own size: its runway is the whole attention block (~35 us), an order of
    // magnitude longer than the two ~4 us sandwich-tail windows, so it can absorb far more of
    // w.wo than they can of gate/up. Sizing them together undershoots window 1 and overshoots 2/3.
    static int l2pf_wo_mb = -1;
    if (l2pf_wo_mb < 0) {
        const char* e = getenv("SPARKINFER_MG_L2PF_WO_MB");
        l2pf_wo_mb = e ? atoi(e) : 8;   // swept 8/12/16: 101.57 / 101.55 / 101.46 tok/s
        if (l2pf_wo_mb < 0) l2pf_wo_mb = 0;
    }
    const bool l2pf = c.muse_glimmer && s.gguf && l2pf_mb > 0 && s.stream_pf != nullptr;
    const size_t pf_bytes = (size_t)l2pf_mb << 20;
    const size_t pf_wo_bytes = (size_t)l2pf_wo_mb << 20;
    bool pf_outstanding = false;
    auto pf_join = [&]() {
        if (!pf_outstanding) return;
        cu(cudaStreamWaitEvent(st, s.ev_pf_done, 0), "l2 prefetch join");
        pf_outstanding = false;
    };
    auto pf_fork_n = [&](const void* a, const void* b, size_t nbytes) {
        if (!l2pf || !nbytes || (!a && !b)) return;
        cu(cudaEventRecord(s.ev_pf_fork, st), "l2 prefetch fork");
        cu(cudaStreamWaitEvent(s.stream_pf, s.ev_pf_fork, 0), "l2 prefetch fork wait");
        if (a) kernels::launch_l2_prefetch(a, nbytes, s.stream_pf);
        if (b) kernels::launch_l2_prefetch(b, nbytes, s.stream_pf);
        cu(cudaEventRecord(s.ev_pf_done, s.stream_pf), "l2 prefetch done");
        pf_outstanding = true;
    };
    auto pf_fork = [&](const void* a, const void* b) { pf_fork_n(a, b, pf_bytes); };
    // A single fork per layer issuing every prefetch back-to-back was tried and is WORSE than
    // doing nothing (98.37 vs 99.44 tok/s): the side stream then hammers DRAM through the QKV
    // projections it is supposed to hide behind, and gate/up is evicted long before the FFN reads
    // it. Each window must be forked immediately before the stretch it covers.
    // Window mask: 1 = w.wo across the attention block, 2 = gate/up across the post-attn tail,
    // 4 = next layer's wq across the post-FFN tail. A window only earns its place if it beats the
    // ~0.89%/step its own fork+join event nodes cost.
    static int pf_win = -1;
    if (pf_win < 0) { const char* e = getenv("SPARKINFER_MG_L2PF_WIN"); pf_win = e ? atoi(e) : 7; }

    for (int L = 0; L < c.n_layers; L++) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        // Window 3 lands: the previous layer's post-FFN prefetch covered its sandwich tail.
        pf_join();
        // Window 1: the QKV projections are latency-bound, not bandwidth-bound (~32 MB over ~29 us
        // = ~1.1 TB/s of a ~1.66 TB/s bus), so the whole attention block has spare bandwidth. w.wo
        // is streamed at the end of it and only has to survive QKV's ~32 MB through a 96 MB L2, so
        // it is prefetchable across that entire runway -- unlike gate/up, which would have to
        // survive attention's full ~47 MB and measured worse when tried that way.
        if (pf_win & 1) pf_fork_n(w.wo, nullptr, pf_wo_bytes);
        dbg_bf16(s.xn, H, 10, L);   // tag 10: pre-attn-norm output (this layer's normed input)
        dbg_xn_snapshot(s.xn, L);
        // xn_q8_ready assumes the PREVIOUS layer's tail already emitted Q8_1(this layer's xn)
        // into s.aq81 as a side effect (true for architectures whose post-MoE tail runs
        // launch_add_rmsnorm2_q8 / add_rmsnorm3_q8). Muse Glimmer's tail is the sandwich-norm
        // pair launch_norm_then_add + a plain launch_rmsnorm (see the c.muse_glimmer branch
        // below) -- neither emits a Q8 side channel. Trusting xn_q8_ready==true here for L>0
        // left s.aq81 permanently stuck holding layer 0's Q8_1(xn): every K/V projection at
        // L=1..n_layers-1 (both wk_type=12 Q4_K and wv_type=14 Q6_K route through proj_xn's
        // mmvq_q4k/mmvq_q6k branches, which read s.aq81 unconditionally) ran against the wrong
        // layer's quantized activation. Force a fresh quantize every layer for muse_glimmer.
        bool xn_q8_ready = fnq && L > 0 && !c.muse_glimmer;
        auto prepare_xn_quant = [&](bool any_q4k, bool any_q6k, bool any_q80) {
            if (!s.gguf || !s.use_pq) return;
            if (xn_q8_ready) return;
            if (s.use_llama && (any_q4k || any_q80 || (s.use_q6mmvq && any_q6k))) {
                kernels::launch_quantize_q8_1_blocks(s.xn, s.aq81, H, st);
                xn_q8_ready = true;
            } else if (any_q4k) {
                kernels::launch_quantize_q8_1(s.xn, s.aq8, s.aq8_d, s.aq8_s, H, st);
            }
        };
        auto proj_xn = [&](const void* W, int t, void* y, int N, cudaStream_t pst) {
            if (s.gguf) {
                if (s.use_pq && t == 12) {
                    if (s.use_llama) kernels::launch_mmvq_q4k(s.aq81, W, y, N, H, pst);
                    else             kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s, W, y, N, H, pst);
                }
                else if (s.use_pq && s.use_llama && s.use_q6mmvq && t == 14)
                    kernels::launch_mmvq_q6k(s.aq81, W, y, N, H, pst);
                else if (s.use_pq && s.use_llama && t == 8)
                    kernels::launch_mmvq_q80(s.aq81, W, y, N, H, pst);
                else if (t) kernels::launch_gemv_q(s.xn, W, t, y, N, H, pst);
                else        kernels::launch_gemv(s.xn, W, y, N, H, pst);
            } else {
                kernels::launch_gemm(s.xn, W, y, 1, N, H, 1.f, 0.f, gc, pst);
            }
        };
        auto proj_from = [&](const void* x, const void* W, int t, void* y, int N, int K) {
            if (s.gguf) {
                if (s.use_pq && t == 12) {
                    if (s.use_llama) {
                        kernels::launch_quantize_q8_1_blocks(x, s.aq81, K, st);
                        kernels::launch_mmvq_q4k(s.aq81, W, y, N, K, st);
                    } else {
                        kernels::launch_quantize_q8_1(x, s.aq8, s.aq8_d, s.aq8_s, K, st);
                        kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s, W, y, N, K, st);
                    }
                } else if (s.use_pq && s.use_llama && s.use_q6mmvq && t == 14) {
                    kernels::launch_quantize_q8_1_blocks(x, s.aq81, K, st);
                    kernels::launch_mmvq_q6k(s.aq81, W, y, N, K, st);
                } else if (s.use_pq && s.use_llama && t == 8) {
                    kernels::launch_quantize_q8_1_blocks(x, s.aq81, K, st);
                    kernels::launch_mmvq_q80(s.aq81, W, y, N, K, st);
                } else if (t) kernels::launch_gemv_q(x, W, t, y, N, K, st);
                else          kernels::launch_gemv(x, W, y, N, K, st);
            } else {
                kernels::launch_gemm(x, W, y, 1, N, K, 1.f, 0.f, gc, st);
            }
        };

        if (w.linear_attn) {
            const bool any_q4k = (w.wqkv_type == 12 || w.wqkv_gate_type == 12 ||
                                  w.ssm_alpha_type == 12 || w.ssm_beta_type == 12);
            const bool any_q6k = (w.wqkv_type == 14 || w.wqkv_gate_type == 14 ||
                                  w.ssm_alpha_type == 14 || w.ssm_beta_type == 14);
            const bool any_q80 = (w.wqkv_type == 8 || w.wqkv_gate_type == 8 ||
                                  w.ssm_alpha_type == 8 || w.ssm_beta_type == 8);
            prepare_xn_quant(any_q4k, any_q6k, any_q80);
            const bool gdn_quad = s.use_gdn_quad && s.gguf && s.use_pq && s.use_llama && H == 2048
                               && w.wqkv_type == 12 && w.wqkv_gate_type == 12
                               && w.ssm_alpha_type == 12 && w.ssm_beta_type == 12;
            const bool gdn_pipelined = !gdn_quad && s.gguf && s.use_gdn_pipe;
            const bool gdn_fused_proj = [&] {
                static int fuse = -1;
                if (fuse < 0) { const char* e = getenv("SPARKINFER_GDN_QKVZ_FUSE");
                    fuse = (e && e[0] == '0') ? 0 : 1; }
                return fuse && s.gguf && s.use_pq && s.use_llama &&
                       w.wqkv_type == 12 && w.wqkv_gate_type == 12 &&
                       (H == 2048 || H == 4096) && s.linear_qkvdim > 0 && s.linear_vdim > 0;
            }();
            if (gdn_quad) {
                kernels::launch_gdn_quad_mmvq_q4k(s.aq81, w.wqkv, w.wqkv_gate, w.ssm_alpha, w.ssm_beta,
                    s.lin_qkv, s.lin_z, s.lin_alpha, s.lin_beta,
                    s.linear_qkvdim, s.linear_vdim, c.linear_v_heads, c.linear_v_heads, H, st);
            } else if (gdn_fused_proj && gdn_pipelined) {
                cudaEventRecord(s.ev_pipe_fork, st);
                cudaStreamWaitEvent(s.stream_v, s.ev_pipe_fork, 0);
                proj_xn(w.ssm_alpha, w.ssm_alpha_type, s.lin_alpha, c.linear_v_heads, s.stream_v);
                proj_xn(w.ssm_beta, w.ssm_beta_type, s.lin_beta, c.linear_v_heads, s.stream_v);
                cudaEventRecord(s.ev_gdn_ab, s.stream_v);
                kernels::launch_mmvq_gdn_qkv_z_pack2(s.aq81, w.wqkv, w.wqkv_gate,
                                                       s.lin_qkv, s.lin_z,
                                                       s.linear_qkvdim, s.linear_vdim, H, st);
            } else if (gdn_pipelined && !gdn_fused_proj) {
                cudaEventRecord(s.ev_pipe_fork, st);
                cudaStreamWaitEvent(s.stream_k, s.ev_pipe_fork, 0);
                cudaStreamWaitEvent(s.stream_v, s.ev_pipe_fork, 0);
                proj_xn(w.wqkv_gate, w.wqkv_gate_type, s.lin_z, s.linear_vdim, s.stream_k);
                cudaEventRecord(s.ev_gdn_z, s.stream_k);
                proj_xn(w.ssm_alpha, w.ssm_alpha_type, s.lin_alpha, c.linear_v_heads, s.stream_v);
                proj_xn(w.ssm_beta, w.ssm_beta_type, s.lin_beta, c.linear_v_heads, s.stream_v);
                cudaEventRecord(s.ev_gdn_ab, s.stream_v);
                proj_xn(w.wqkv, w.wqkv_type, s.lin_qkv, s.linear_qkvdim, st);
            } else if (gdn_fused_proj) {
                kernels::launch_mmvq_gdn_qkv_z_pack2(s.aq81, w.wqkv, w.wqkv_gate,
                                                       s.lin_qkv, s.lin_z,
                                                       s.linear_qkvdim, s.linear_vdim, H, st);
                proj_xn(w.ssm_alpha, w.ssm_alpha_type, s.lin_alpha, c.linear_v_heads, st);
                proj_xn(w.ssm_beta, w.ssm_beta_type, s.lin_beta, c.linear_v_heads, st);
            } else {
                proj_xn(w.wqkv, w.wqkv_type, s.lin_qkv, s.linear_qkvdim, st);
                proj_xn(w.wqkv_gate, w.wqkv_gate_type, s.lin_z, s.linear_vdim, st);
                proj_xn(w.ssm_alpha, w.ssm_alpha_type, s.lin_alpha, c.linear_v_heads, st);
                proj_xn(w.ssm_beta, w.ssm_beta_type, s.lin_beta, c.linear_v_heads, st);
            }

            bf16* conv_state = s.lin_conv_state +
                (size_t)L * (c.linear_conv_kernel - 1) * s.linear_qkvdim;
            // Fused conv_split + l2_norm: one kernel instead of three (SPARKINFER_GDN_FUSE=0 restores split).
            static int gdn_fuse = -1;
            if (gdn_fuse < 0) { const char* e = getenv("SPARKINFER_GDN_FUSE"); gdn_fuse = (e && e[0] == '0') ? 0 : 1; }
            if (gdn_fuse && c.linear_head_dim == 128 && c.linear_q_heads == 16 &&
                c.linear_v_heads == 32) {
                kernels::launch_qwen36_conv_split_l2norm_fused(s.lin_qkv, w.ssm_conv, conv_state,
                                                 s.lin_q, s.lin_k, s.lin_v,
                                                 c.linear_q_heads, c.linear_v_heads,
                                                 c.linear_head_dim, c.linear_conv_kernel,
                                                 c.rms_eps, st);
            } else {
                kernels::launch_qwen36_conv_split_l2(s.lin_qkv, w.ssm_conv, conv_state,
                                                 s.lin_q, s.lin_k, s.lin_v,
                                                 c.linear_q_heads, c.linear_v_heads,
                                                 c.linear_head_dim, c.linear_conv_kernel,
                                                 c.rms_eps, st);
            }
            if (gdn_pipelined) cudaStreamWaitEvent(st, s.ev_gdn_ab, 0);
            float* layer_state = s.lin_state +
                (size_t)L * c.linear_v_heads * c.linear_head_dim * c.linear_head_dim;
            kernels::launch_qwen36_gdn_ar(s.lin_q, s.lin_k, s.lin_v,
                                          s.lin_alpha, s.lin_beta, w.ssm_dt, w.ssm_a,
                                          layer_state, s.lin_gdn,
                                          c.linear_q_heads, c.linear_v_heads,
                                          c.linear_head_dim, st);
            if (gdn_pipelined && !gdn_fused_proj) cudaStreamWaitEvent(st, s.ev_gdn_z, 0);
            const bool gdn_gn_q8 = s.gguf && s.use_pq && s.use_llama &&
                                   (w.ssm_out_type == 12 || w.ssm_out_type == 8) &&
                                   c.linear_head_dim == 128;
            if (gdn_gn_q8) {
                static int gn_q8 = -1;
                if (gn_q8 < 0) {
                    const char* e = getenv("SPARKINFER_GDN_GNORM_Q8");
                    gn_q8 = (e && e[0] == '0') ? 0 : 1;
                }
                if (gn_q8) {
                    kernels::launch_qwen36_gated_norm_q8(s.lin_gdn, s.lin_z, w.ssm_norm, s.aq81,
                                                         c.linear_v_heads, c.linear_head_dim,
                                                         c.rms_eps, st);
                    if (w.ssm_out_type == 12)
                        kernels::launch_mmvq_q4k(s.aq81, w.ssm_out, s.ao, H, s.linear_vdim, st);
                    else
                        kernels::launch_mmvq_q80(s.aq81, w.ssm_out, s.ao, H, s.linear_vdim, st);
                } else {
                    kernels::launch_qwen36_gated_norm(s.lin_gdn, s.lin_z, w.ssm_norm, s.lin_norm,
                                                      c.linear_v_heads, c.linear_head_dim, c.rms_eps, st);
                    proj_from(s.lin_norm, w.ssm_out, w.ssm_out_type, s.ao, H, s.linear_vdim);
                }
            } else {
                kernels::launch_qwen36_gated_norm(s.lin_gdn, s.lin_z, w.ssm_norm, s.lin_norm,
                                                  c.linear_v_heads, c.linear_head_dim, c.rms_eps, st);
                proj_from(s.lin_norm, w.ssm_out, w.ssm_out_type, s.ao, H, s.linear_vdim);
            }
        } else {
            // ---- Q/K/V projection (q_has_gate-aware; q_has_gate=false is byte-identical to Qwen3-MoE) ----
            if (s.gguf) {
                const bool any_q4k = (w.wq_type == 12 || w.wk_type == 12 || w.wv_type == 12);
                const bool any_q6k = (w.wq_type == 14 || w.wk_type == 14 || w.wv_type == 14);
                const bool any_q80 = (w.wq_type == 8 || w.wk_type == 8 || w.wv_type == 8);
                prepare_xn_quant(any_q4k, any_q6k, any_q80);
                // Muse Glimmer keeps attn_gate as its own quantized tensor (w.wgate), so Q goes
                // straight to s.q and the gate straight to s.qgate -- no [q|gate] interleave to
                // build and no split to undo it. Every other model still fuses them into s.qraw.
                const bool sep_gate = (w.wgate != nullptr);
                void* q_dst = (w.q_has_gate && !sep_gate) ? s.qraw : s.q;
                const int nq = (w.q_has_gate && !sep_gate) ? s.qdim * 2 : s.qdim;
                // Muse Glimmer's Q and attn-gate are two separate Q4_K tensors projected
                // back-to-back on the SAME stream. Merge them into one launch: one fewer graph
                // node per layer, and the launch doubles to 18.4 MB which reads faster per byte.
                // K/V are untouched -- QKVSTREAM overlaps them on side streams and folding those
                // in measured -0.13%. SPARKINFER_MG_QG_FUSE=0 restores the split pair.
                static int mg_qg = -1;
                if (mg_qg < 0) { const char* e = getenv("SPARKINFER_MG_QG_FUSE"); mg_qg = (e && e[0] == '0') ? 0 : 1; }
                auto proj_q_gate = [&](cudaStream_t qs) {
                    if (mg_qg && c.muse_glimmer && sep_gate && s.use_pq && s.use_llama &&
                        w.wq_type == 12 && w.wgate_type == 12 && H == 6656 &&
                        kernels::launch_mmvq_q4k_kfixed2(s.aq81, w.wq, w.wgate, q_dst, s.qgate,
                                                         nq, s.qdim, H, qs))
                        return;
                    proj_xn(w.wq, w.wq_type, q_dst, nq, qs);
                    if (sep_gate) proj_xn(w.wgate, w.wgate_type, s.qgate, s.qdim, qs);
                };
                // The fused QKV kernel writes one contiguous q of width nq and knows nothing about
                // a separate gate tensor, so it cannot serve this path.
                const bool attn_qkv = !sep_gate && s.use_attn_qkv && s.use_pq && s.use_llama
                                   && (H == 2048 || H == 4096)
                                   && w.wq_type == 12 && w.wk_type == 12 && w.wv_type == 12;
                if (attn_qkv) {
                    kernels::launch_attn_qkv_mmvq_q4k(s.aq81, w.wq, w.wk, w.wv,
                        q_dst, s.k, s.v, nq, s.kvdim, s.kvdim, H, st);
                } else if (s.use_qkvstream) {
                    cudaEventRecord(s.ev_qkv, st);
                    cudaStreamWaitEvent(s.stream_k, s.ev_qkv, 0);
                    cudaStreamWaitEvent(s.stream_v, s.ev_qkv, 0);
                    proj_q_gate(st);
                    proj_xn(w.wk, w.wk_type, s.k, s.kvdim, s.stream_k);
                    proj_xn(w.wv, w.wv_type, s.v, s.kvdim, s.stream_v);
                    cudaEventRecord(s.ev_k, s.stream_k);
                    cudaEventRecord(s.ev_v, s.stream_v);
                    cudaStreamWaitEvent(st, s.ev_k, 0);
                    cudaStreamWaitEvent(st, s.ev_v, 0);
                } else {
                    proj_q_gate(st);
                    proj_xn(w.wk, w.wk_type, s.k, s.kvdim, st);
                    proj_xn(w.wv, w.wv_type, s.v, s.kvdim, st);
                }
            } else {
                kernels::launch_gemm(s.xn, w.wq, w.q_has_gate ? s.qraw : s.q,
                                     1, w.q_has_gate ? s.qdim * 2 : s.qdim, H, 1.f, 0.f, gc, st);
                kernels::launch_gemm(s.xn, w.wk, s.k, 1, s.kvdim, H, 1.f, 0.f, gc, st);
                kernels::launch_gemm(s.xn, w.wv, s.v, 1, s.kvdim, H, 1.f, 0.f, gc, st);
            }
            // ---- QK-norm + RoPE + KV-append ----
            const bool kv8 = s.kv->int8_kv();
            const int kv_elem = kv8 ? 1 : 2;
            void* kpool = (char*)s.kv->k_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
            void* vpool = (char*)s.kv->v_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
            void* kscale = kv8 ? (char*)s.kv->k_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
            void* vscale = kv8 ? (char*)s.kv->v_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
            const bool partial_rope = (c.rope_dim > 0 && c.rope_dim < c.head_dim);
            const bool qkgate_fuse = w.q_has_gate && partial_rope && kv8 && s.use_qkfuse && H == 2048;
            // w.wgate != nullptr means Q and the gate were projected straight into s.q / s.qgate
            // above, so there is no interleaved s.qraw to split.
            if (w.q_has_gate && !qkgate_fuse && !w.wgate)
                kernels::launch_qwen36_split_q_gate(s.qraw, s.q, s.qgate, c.n_q_heads, c.head_dim, st);
            dbg_bf16(s.q, s.qdim, 11, L);      // tag 11: Q, raw split, pre QK-norm
            dbg_bf16(s.qgate, s.qdim, 12, L);  // tag 12: attn gate proj, pre-sigmoid
            dbg_bf16(s.k, s.kvdim, 13, L);     // tag 13: K, raw, pre QK-norm

            if (!w.q_has_gate && !partial_rope && (s.use_attnin || kv8)) {
                // Qwen3-MoE frontier: fused int8 QK-norm + RoPE + KV-append (unchanged vs main)
                kernels::launch_qknorm_rope_kv_append(s.q, s.k, s.v, w.q_norm, w.k_norm, kpool, vpool,
                                                      btable, s.d_pos, 1, c.n_q_heads, c.n_kv_heads,
                                                      c.head_dim, c.rope_theta, c.rms_eps,
                                                      s.kv->block_size(), s.kv->max_blocks_per_seq(), st,
                                                      kscale, vscale, kv8 ? 1 : 0);
            } else {
                // Qwen3.6 (gated / partial-rotary): fuse QK-norm + partial-RoPE + KV when enabled.
                if (partial_rope && kv8) {
                    if (s.use_qkfuse && H == 2048) {
                        if (qkgate_fuse) {
                            kernels::launch_qknorm_rope_kv_partial_int8_gated(s.qraw, s.q, s.qgate, s.k, s.v,
                                w.q_norm, w.k_norm, kpool, vpool, kscale, vscale, btable, s.d_pos, 1,
                                c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_dim, c.rope_theta, c.rms_eps,
                                s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                        } else {
                            kernels::launch_qknorm_rope_kv_partial_int8(s.q, s.k, s.v, w.q_norm, w.k_norm,
                                kpool, vpool, kscale, vscale, btable, s.d_pos, 1,
                                c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_dim, c.rope_theta, c.rms_eps,
                                s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                        }
                    } else {
                        if (s.use_qkfuse)
                            kernels::launch_rmsnorm_qk(s.q, s.k, w.q_norm, w.k_norm, c.n_q_heads, c.n_kv_heads, c.head_dim, c.rms_eps, st);
                        else {
                            kernels::launch_rmsnorm(s.q, w.q_norm, s.q, c.n_q_heads,  c.head_dim, c.rms_eps, st);
                            kernels::launch_rmsnorm(s.k, w.k_norm, s.k, c.n_kv_heads, c.head_dim, c.rms_eps, st);
                        }
                        kernels::launch_rope_kv_append_partial_int8(s.q, s.k, s.v, kpool, vpool, kscale, vscale,
                            btable, s.d_pos, 1, c.n_q_heads, c.n_kv_heads,
                            c.head_dim, c.rope_dim, c.rope_theta,
                            s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                    }
                } else if (partial_rope && s.use_qkfuse) {
                    kernels::launch_qknorm_rope_kv_partial(s.q, s.k, s.v, w.q_norm, w.k_norm,
                        (bf16*)kpool, (bf16*)vpool, btable, s.d_pos, 1,
                        c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_dim,
                        c.rope_theta, c.rms_eps, s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                } else {
                    // Muse Glimmer: QK-norm + RoPE/append are two dependent graph nodes per layer
                    // (104/step) doing tiny work. Fuse them into one; bit-identical, see
                    // launch_muse_qknorm_rope_kv. SPARKINFER_MG_QKR_FUSE=0 restores the pair.
                    static int mg_qkr = -1;
                    if (mg_qkr < 0) { const char* e = getenv("SPARKINFER_MG_QKR_FUSE"); mg_qkr = (e && e[0] == '0') ? 0 : 1; }
                    const bool mg_qkr_fuse = mg_qkr && c.muse_glimmer && s.use_qkfuse && !partial_rope && !kv8;
                    if (mg_qkr_fuse) {
                        kernels::launch_muse_qknorm_rope_kv(
                            s.q, s.k, s.v, w.q_norm, w.k_norm, (bf16*)kpool, (bf16*)vpool, btable,
                            s.d_pos, w.swa ? s.d_pos : s.d_writepos,
                            c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta,
                            s.kv->block_size(), c.rms_eps, /*do_rope=*/w.swa != 0, st);
                    } else if (s.use_qkfuse)
                        kernels::launch_rmsnorm_qk(s.q, s.k, w.q_norm, w.k_norm, c.n_q_heads, c.n_kv_heads, c.head_dim, c.rms_eps, st);
                    else {
                        kernels::launch_rmsnorm(s.q, w.q_norm, s.q, c.n_q_heads,  c.head_dim, c.rms_eps, st);
                        kernels::launch_rmsnorm(s.k, w.k_norm, s.k, c.n_kv_heads, c.head_dim, c.rms_eps, st);
                    }
                    dbg_bf16(s.q, s.qdim, 20, L);   // tag 20: Q, post QK-norm, pre-RoPE
                    dbg_bf16(s.k, s.kvdim, 21, L);  // tag 21: K, post QK-norm, pre-RoPE
                    if (mg_qkr_fuse) {
                        // already done in one kernel above
                    } else if (c.muse_glimmer && !w.swa) {
                        // Global/NoPE layer: no rotation at all (Q/K are already QK-normed
                        // above) -- append K/V as-is. Every 4th layer per sliding_window_pattern.
                        launch_kv_append((bf16*)kpool, (bf16*)vpool, s.k, s.v, btable, s.d_writepos, 1,
                                         c.n_kv_heads, c.head_dim, s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                    } else if (partial_rope) {
                        kernels::launch_rope_kv_append_partial(s.q, s.k, s.v, (bf16*)kpool, (bf16*)vpool, btable, s.d_pos, 1,
                                                               c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_dim,
                                                               c.rope_theta, s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                    } else if (c.muse_glimmer) {
                        // Muse Glimmer's SWA layers need "normal" (consecutive-pair,
                        // LLAMA_ROPE_TYPE_NORM) rotation, NOT the NeoX (split-half) pairing
                        // every other kernel below implements -- llama.cpp's own
                        // llama_model_rope_type() puts LLM_ARCH_MUSE_GLIMMER in the same
                        // LLAMA_ROPE_TYPE_NORM bucket as LLM_ARCH_LLAMA, while every arch this
                        // codebase was actually built for (Qwen2/3/3MoE, Gemma) is
                        // LLAMA_ROPE_TYPE_NEOX. Reusing launch_rope_kv_append here rotated the
                        // wrong pair of dimensions together for every position > 0 (position 0
                        // is a no-op rotation under either convention, which is why this hid
                        // during the earliest single-token bring-up checks): Q/K stayed
                        // well-formed but phase-wrong, so attention still produced a plausible
                        // softmax over the wrong distribution instead of visibly breaking.
                        // Unconditional (not gated on s.use_ropekv) so a SPARKINFER_ROPEKV=0
                        // override can't silently fall through to the NeoX plain-launch_rope
                        // path in the final else below.
                        kernels::launch_rope_kv_append_normal(s.q, s.k, s.v, (bf16*)kpool, (bf16*)vpool, btable, s.d_pos, 1,
                                                              c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta,
                                                              s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                    } else if (s.use_ropekv) {
                        kernels::launch_rope_kv_append(s.q, s.k, s.v, (bf16*)kpool, (bf16*)vpool, btable, s.d_pos, 1,
                                                       c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta,
                                                       s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                    } else {
                        kernels::launch_rope(s.q, s.k, s.d_pos, 1, c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta, st);
                        launch_kv_append((bf16*)kpool, (bf16*)vpool, s.k, s.v, btable, s.d_writepos, 1,
                                         c.n_kv_heads, c.head_dim, s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                    }
                    dbg_bf16(s.q, s.qdim, 22, L);   // tag 22: Q, post RoPE-or-passthrough (SDPA input)
                    dbg_bf16(s.k, s.kvdim, 23, L);  // tag 23: K, post RoPE-or-passthrough (SDPA input)
                }
            }

            // ---- attention (Q8-emit only when output is not gated: the gate mutates attn after decode) ----
            static int attn_gq8 = -1;
            if (attn_gq8 < 0) { const char* e = getenv("SPARKINFER_ATTN_GQ8"); attn_gq8 = (e && e[0] == '0') ? 0 : 1; }
            // The H test is not a hidden-size requirement -- it is a proxy for "a gated combine has
            // been instantiated for this architecture's head_dim", and only the hd256 models ever
            // had one. fa_combine_gated_q8_kernel is HEAD_DIM-generic, so instantiating it at 128
            // lets Muse Glimmer (H=6656, head_dim=128) qualify too, folding the sigmoid gate and
            // the output quantize into the combine and deleting 2 graph nodes per layer (104/step).
            // SPARKINFER_MG_ATTN_GQ8=0 restores the split path for A/B.
            static int mg_gq8 = -1;
            if (mg_gq8 < 0) { const char* e = getenv("SPARKINFER_MG_ATTN_GQ8"); mg_gq8 = (e && e[0] == '0') ? 0 : 1; }
            const bool mg_gate_ok = c.muse_glimmer && mg_gq8 && c.head_dim == 128;
            const bool attn_gate_q8 = attn_gq8 && w.q_has_gate && s.gguf && s.use_pq && s.use_llama
                                      && (H == 2048 || H == 4096 || mg_gate_ok)
                                      && (w.wo_type == 12 || w.wo_type == 8) && (s.qdim % 32 == 0);
            const bool emit_attn_q8 = !w.q_has_gate && s.use_attnin && s.gguf && s.use_pq && s.use_llama && w.wo_type == 12;
            if (c.muse_glimmer && w.swa && s.swa_vtbl) {
                // Sliding-window layer: same dense flash-decode entry point, pointed at the
                // per-step pure sliding-window compact view (no sink, unlike the sparse-kv
                // path above) instead of the full KV. Mandatory every step at this context
                // regardless of length -- not gated on sparse_on/context-length like the
                // Qwythos/Qwen3.6 approximation.
                kernels::launch_flash_decode_split(s.q, kpool, vpool, s.swa_vtbl, s.swa_vlen,
                                                   s.attn, s.fa_m, s.fa_l, s.fa_acc, 1,
                                                   c.n_q_heads, c.n_kv_heads, c.head_dim,
                                                   s.kv->block_size(), s.swa_budget, s.swa_vsplits,
                                                   1.f / sqrtf((float)c.head_dim), st,
                                                   (emit_attn_q8 || attn_gate_q8) ? s.aq81 : nullptr,
                                                   s.swa_budget * s.kv->block_size(),
                                                   kscale, vscale, kv8 ? 1 : 0,
                                                   attn_gate_q8 ? s.qgate : nullptr,
                                                   mg_gate_ok ? 1 : 0);
            } else if (sparse_on && s.sparse_vtbl) {
                // GQA-8 (Qwen3.6): the same dense flash-decode entry point, pointed at the
                // per-step compact view — sink + last-window blocks, view seq_len carrying
                // the partial tail. The tuned int8-MMA kernel and fused combine run
                // unmodified; only the KV footprint changes (O(window) vs O(context)).
                // Host-side hints are view-sized constants, so the captured graph is stable:
                // max_blocks/seqlen describe the view, and sparse_vsplits keeps every MMA
                // split at >= 2 KV blocks while filling the 5090 (8 kv heads x 128 splits).
                kernels::launch_flash_decode_split(s.q, kpool, vpool, s.sparse_vtbl, s.sparse_vlen,
                                                   s.attn, s.fa_m, s.fa_l, s.fa_acc, 1,
                                                   c.n_q_heads, c.n_kv_heads, c.head_dim,
                                                   s.kv->block_size(), s.sparse_budget, s.sparse_vsplits,
                                                   1.f / sqrtf((float)c.head_dim), st,
                                                   (emit_attn_q8 || attn_gate_q8) ? s.aq81 : nullptr,
                                                   s.sparse_budget * s.kv->block_size(),
                                                   kscale, vscale, kv8 ? 1 : 0,
                                                   attn_gate_q8 ? s.qgate : nullptr);
            } else if (sparse_on) {
                kernels::launch_fa_kv_window_select(s.d_seqlen, s.sparse_sel, c.n_kv_heads,
                    s.kv->block_size(), s.sparse_budget, s.sparse_window, st);
                kernels::launch_flash_decode_split_sparse(s.q, kpool, vpool, btable, s.d_seqlen,
                    s.sparse_sel, s.fa_m, s.fa_l, s.fa_acc, c.n_q_heads, c.n_kv_heads, c.head_dim,
                    s.kv->block_size(), s.kv->max_blocks_per_seq(), s.n_splits, s.sparse_budget,
                    1.f / sqrtf((float)c.head_dim), kscale, vscale, st);
                kernels::launch_fa_combine_hd256(s.fa_m, s.fa_l, s.fa_acc, s.attn, c.n_q_heads,
                    s.n_splits, (emit_attn_q8 || attn_gate_q8) ? s.aq81 : nullptr, st,
                    attn_gate_q8 ? s.qgate : nullptr);
            } else {
            kernels::launch_flash_decode_split(s.q, kpool, vpool, btable, s.d_seqlen, s.attn,
                                               s.fa_m, s.fa_l, s.fa_acc, 1, c.n_q_heads, c.n_kv_heads, c.head_dim,
                                               s.kv->block_size(), s.kv->max_blocks_per_seq(), s.n_splits,
                                               1.f / sqrtf((float)c.head_dim), st,
                                               (emit_attn_q8 || attn_gate_q8) ? s.aq81 : nullptr, seqlen,
                                               kscale, vscale, kv8 ? 1 : 0,
                                               attn_gate_q8 ? s.qgate : nullptr,
                                               mg_gate_ok ? 1 : 0);
            }
            dbg_bf16(s.attn, s.qdim, 30, L);   // tag 30: SDPA output, pre-gate
            if (w.q_has_gate && !attn_gate_q8)
                kernels::launch_qwen36_mul_sigmoid(s.attn, s.qgate, s.qdim, st);
            dbg_bf16(s.attn, s.qdim, 31, L);   // tag 31: SDPA output, post sigmoid-gate

            // ---- O projection (main's int8 mmvq path) ----
            if (pf_win & 1) pf_join();   // window 1 lands here: w.wo is read next
            if (s.gguf && s.use_pq && w.wo_type == 12) {
                if (s.use_llama) {
                    if (!emit_attn_q8 && !attn_gate_q8) kernels::launch_quantize_q8_1_blocks(s.attn, s.aq81, s.qdim, st);
                    kernels::launch_mmvq_q4k(s.aq81, w.wo, s.ao, H, s.qdim, st);
                } else {
                    kernels::launch_quantize_q8_1(s.attn, s.aq8, s.aq8_d, s.aq8_s, s.qdim, st);
                    kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s, w.wo, s.ao, H, s.qdim, st);
                }
            }
            else if (s.gguf && s.use_pq && s.use_llama && w.wo_type == 8) {
                kernels::launch_quantize_q8_1_blocks(s.attn, s.aq81, s.qdim, st);
                kernels::launch_mmvq_q80(s.aq81, w.wo, s.ao, H, s.qdim, st);
            }
            else if (s.gguf && w.wo_type) kernels::launch_gemv_q(s.attn, w.wo, w.wo_type, s.ao, H, s.qdim, st);
            else if (s.gguf)         kernels::launch_gemv(s.attn, w.wo, s.ao, H, s.qdim, st);
            else                     kernels::launch_gemm(s.attn, w.wo, s.ao, 1, H, s.qdim, 1.f, 0.f, gc, st);
            dbg_bf16(s.ao, H, 40, L);   // tag 40: attn_o_proj output (post wo)
        }

        if (c.muse_glimmer) {
            // Sandwich norm: h = x + RMSNorm(ao) * post_attn_norm (norm the attention
            // output alone, not the sum -- see launch_norm_then_add), then hn = RMSNorm(h,
            // ffn_norm) is a genuine separate pre-FFN norm, not post_attn_norm doing double
            // duty like every other architecture here.
            //
            // The sandwich (post_attn_norm/post_ffn_norm) RMSNorm uses its OWN eps (1e-8,
            // upstream's `post_norm_eps` in muse-glimmer.cpp), distinct from the model's
            // normal rms_eps (1e-5, used for attn_norm/ffn_norm/q_norm/k_norm) -- reusing
            // c.rms_eps here silently gives wrong post-attn/post-ffn norms.
            // Both halves of the sandwich tail are single-CTA kernels over 6656 elements, so each
            // costs ~3.2 us of launch/reduction latency for 13 KB of traffic. Fuse them.
            // Window 2: the post-attn sandwich tail is a single CTA on a 170-SM device for ~3.4 us.
            // Cover it with the FFN gate/up matrices, which are streamed immediately after it.
            if (pf_win & 2) pf_fork(w.gate_q, w.up_q);
            if (muse_fuse_tail())
                kernels::launch_muse_sandwich_tail(s.x, s.ao, w.post_attn_norm, w.ffn_norm,
                                                   s.h, s.hn, 1, H, 1e-8f, c.rms_eps, st);
            else {
            kernels::launch_norm_then_add(s.x, s.ao, w.post_attn_norm, s.h, 1, H, 1e-8f, st);
            dbg_bf16(s.h, H, 50, L);   // tag 50: h = x + sandwich_norm(ao)  (post-attn residual)
            kernels::launch_rmsnorm(s.h, w.ffn_norm, s.hn, 1, H, c.rms_eps, st);
            }
            dbg_bf16(s.hn, H, 51, L);  // tag 51: hn = pre-FFN norm(h)
        } else if (fnq) {
            // fused: h = x + ao ; hn = RMSNorm(h, post_attn_norm). When fnq, also emit
            // Q8_1(hn) into aq81 so the MoE gate/up mmvq skips its own quantize node (the
            // router below reads bf16 hn).
            kernels::launch_add_rmsnorm2_q8(s.x, s.ao, w.post_attn_norm, s.h, s.hn, s.aq81, H, c.rms_eps, st);
        } else {
            kernels::launch_add_rmsnorm2(s.x, s.ao, w.post_attn_norm, s.h, s.hn, 1, H, c.rms_eps, st);
        }

        const bool qmoe = w.shared_gate_q && w.shared_up_q && w.shared_down_q
                       && w.shared_gate_qtype == 8 && c.hidden == 2048 && c.moe_ffn == 512;
        const bool shexp_pipelined = (c.n_shared > 0) && s.gguf && s.use_shexp_pipe;
        if (shexp_pipelined) {
            cudaEventRecord(s.ev_pipe_fork, st);
            cudaStreamWaitEvent(s.stream_k, s.ev_pipe_fork, 0);
            cudaStreamWaitEvent(s.stream_v, s.ev_pipe_fork, 0);
            if (w.shared_gate_inp) {
                if (s.use_pq && w.shared_gate_inp_type == 12) {
                    if (s.use_llama) {
                        if (!fnq) kernels::launch_quantize_q8_1_blocks(s.hn, s.aq81, H, s.stream_k);
                        if (H == 2048)
                            kernels::launch_mmvq_q4k_sigmoid(s.aq81, w.shared_gate_inp, s.d_shared_w, H, s.stream_k);
                        else
                            kernels::launch_mmvq_q4k(s.aq81, w.shared_gate_inp, s.shared_gate_tmp, 1, H, s.stream_k);
                    } else {
                        kernels::launch_quantize_q8_1(s.hn, s.aq8, s.aq8_d, s.aq8_s, H, s.stream_k);
                        kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s,
                                                        w.shared_gate_inp, s.shared_gate_tmp, 1, H, s.stream_k);
                    }
                } else if (s.use_pq && s.use_llama && s.use_q6mmvq && w.shared_gate_inp_type == 14) {
                    if (!fnq) kernels::launch_quantize_q8_1_blocks(s.hn, s.aq81, H, s.stream_k);
                    kernels::launch_mmvq_q6k(s.aq81, w.shared_gate_inp, s.shared_gate_tmp, 1, H, s.stream_k);
                } else if (w.shared_gate_inp_type) {
                    kernels::launch_gemv_q(s.hn, w.shared_gate_inp, w.shared_gate_inp_type,
                                           s.shared_gate_tmp, 1, H, s.stream_k);
                } else {
                    // Fused GEMV + sigmoid for the shared-expert gate scalar:
                    // writes fp32 sigmoid(gate) directly, eliminating the separate
                    // 1-thread sigmoid_scalar_kernel launch. SPARKINFER_GEMV_SIGMOID=0
                    // restores the split path for A/B.
                    static int gemv_sigmoid = -1;
                    if (gemv_sigmoid < 0) { const char* e = getenv("SPARKINFER_GEMV_SIGMOID");
                        gemv_sigmoid = (e && e[0] == '1') ? 1 : 0; }   // default off: fused dot != split-k GEMV
                    if (gemv_sigmoid) {
                        kernels::launch_gemv_sigmoid(s.hn, w.shared_gate_inp, s.shared_gate_tmp, s.d_shared_w, H, s.stream_k);
                    } else {
                        kernels::launch_gemv(s.hn, w.shared_gate_inp, s.shared_gate_tmp, 1, H, s.stream_k);
                        kernels::launch_qwen36_sigmoid_scalar(s.shared_gate_tmp, s.d_shared_w, s.stream_k);
                    }
                }
                // The dense gate branch above already applies sigmoid (either fused into
                // launch_gemv_sigmoid or as its split follow-up). Only quantized projections
                // that leave a raw scalar in shared_gate_tmp need this common epilogue.
                if (w.shared_gate_inp && w.shared_gate_inp_type != 0 &&
                    !(s.use_pq && s.use_llama && w.shared_gate_inp_type == 12 && H == 2048))
                    kernels::launch_qwen36_sigmoid_scalar(s.shared_gate_tmp, s.d_shared_w, s.stream_k);
            }
            if (qmoe) {
                // Pipelined shared overlaps stream_k with MoE on st — accum into routed here
                // races MoE (shared finishes first, MoE overwrites routed). Always write s.shared;
                // fold happens after both complete. SPARKINFER_SHEXP_ACCUM=1 only applies on the
                // non-pipelined path where MoE has already landed in routed.
                kernels::launch_shared_expert_q8_mmvq(
                    s.hn, fnq ? s.aq81 : nullptr,
                    w.shared_gate_q, w.shared_up_q, w.shared_down_q,
                    w.shared_gate_inp ? s.d_shared_w : nullptr,
                    s.shared, s.sx_h, s.sx_q8, H, c.moe_ffn, s.stream_k, false);
            } else {
                kernels::launch_gemv(s.hn, w.shared_gate, s.sh_gate, c.moe_ffn, H, s.stream_k);
                kernels::launch_gemv(s.hn, w.shared_up,   s.sh_up,   c.moe_ffn, H, s.stream_v);
                cudaEventRecord(s.ev_sx_gate, s.stream_v);
                cudaStreamWaitEvent(s.stream_k, s.ev_sx_gate, 0);
                kernels::launch_qwen36_shared_swiglu(s.sh_gate, s.sh_up, s.d_shared_w,
                                                     s.sh_h, c.moe_ffn, s.stream_k);
                kernels::launch_gemv(s.sh_h, w.shared_down, s.shared, H, c.moe_ffn, s.stream_k);
            }
            cudaEventRecord(s.ev_sx_done, s.stream_k);
        }

        if (c.dense_ffn) {
            // Qwen3.5 dense SwiGLU: keep gate/up/down quantized and run the same MMVQ
            // expert-FFN path as MoE decode — bf16 dequant+GEMV diverged ~40pp vs llama.cpp.
            //
            // input_q8 must be a valid Q8_1 quantization of s.hn (the FFN's actual input) or
            // null (letting the kernel quantize s.hn itself). `fnq` only guarantees that for
            // architectures whose post-attention step runs launch_add_rmsnorm2_q8, which emits
            // Q8_1(hn) as a side effect of computing hn. Muse Glimmer's sandwich norm does not:
            // its post-attn step is launch_norm_then_add (writes s.h, no Q8 emission) followed
            // by a plain launch_rmsnorm into s.hn (see the c.muse_glimmer branch above) -- s.aq81
            // at this point still holds Q8_1(xn) from this same layer's QKV-input quantize
            // (prepare_xn_quant, earlier in this iteration), not Q8_1(hn). Passing that stale
            // buffer here fed the gate/up MMVQ kernel the wrong activation vector entirely:
            // garbage FFN output that still looked like a confident (but wrong) distribution
            // downstream, rather than crashing or NaN-ing. Force nullptr for muse_glimmer so
            // launch_moe_expert_ffn_q4k quantizes s.hn fresh instead of trusting the stale cache.
            kernels::launch_moe_expert_ffn_q4k(s.hn, w.gate_q, w.up_q, w.down_q,
                                               w.gate_qtype, w.up_qtype, w.down_qtype,
                                               s.mf_ids, s.mf_weights, s.routed, s.mf_h, s.mf_out,
                                               1, c.top_k, H, c.moe_ffn,
                                               (fnq && !c.muse_glimmer) ? s.aq81 : nullptr, st);
        } else if (w.gate_q) {   // GGUF fused: route, then dequant-on-read only the top_k experts
            // The per-expert token counts only feed the batched-dispatch sort; the single-token
            // decode expert FFN reads ids/weights directly and never touches them. Zeroing that
            // buffer is a per-layer memset node in the replayed decode graph whose fixed cost far
            // outweighs the handful of atomics that fill it, so skip the count on this path.
            // SPARKINFER_MOE_COUNTS=1 restores the memset + on-device counting.
            static int moe_counts = -1;
            if (moe_counts < 0) { const char* mc = getenv("SPARKINFER_MOE_COUNTS"); moe_counts = (mc && mc[0] == '1') ? 1 : 0; }
            const bool rfuse = s.use_router_fused && !moe_counts && c.n_experts == 256 && (c.hidden % 8) == 0;
            if (rfuse) {
                // one kernel: router GEMV -> logits scratch, then in-kernel bitonic top-8 (last block)
                kernels::launch_router_fused(s.hn, w.router_w, s.mf_logits, s.mf_rc,
                                             s.mf_ids, s.mf_weights, c.n_experts, c.hidden, c.top_k, 1, st);
            } else {
                kernels::launch_gemv_f32(s.hn, w.router_w, s.mf_logits, c.n_experts, c.hidden, st);  // router_w native [E,H]
                if (moe_counts) cu(cudaMemsetAsync(s.mf_counts, 0, c.n_experts * sizeof(int), st), "mf counts");
                kernels::launch_moe_router(s.mf_logits, s.mf_ids, s.mf_weights,
                                           moe_counts ? s.mf_counts : nullptr,
                                           1, c.n_experts, c.top_k, 1, st);
            }
            kernels::launch_moe_expert_ffn_q4k(s.hn, w.gate_q, w.up_q, w.down_q,
                                               w.gate_qtype, w.up_qtype, w.down_qtype,
                                               s.mf_ids, s.mf_weights, s.routed, s.mf_h, s.mf_out,
                                               1, c.top_k, c.hidden, c.moe_ffn,
                                               fnq ? s.aq81 : nullptr, st);
        } else {
            s.engine->set_layer_weights(L, {w.router_w, w.gate, w.up, w.down});
            s.engine->forward(s.hn, s.routed, 1, L, st);
        }
        const void* shared_to_fold = nullptr;
        if (c.n_shared > 0) {
            const void* nextnorm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
            if (shexp_pipelined) {
                cudaStreamWaitEvent(st, s.ev_sx_done, 0);
                // (residual_add folded into add_rmsnorm3 below — #279)
                if (s.use_addnorm3) {
                    if (fnq)
                        kernels::launch_add_rmsnorm3_q8(s.h, s.routed, s.shared, nextnorm, s.x, s.xn, s.aq81, H, c.rms_eps, st);
                    else
                        kernels::launch_add_rmsnorm3(s.h, s.routed, s.shared, nextnorm, s.x, s.xn, 1, H, c.rms_eps, st);
                } else {
                    launch_residual_add(s.routed, s.shared, s.routed, H, st);
                    if (fnq)
                        kernels::launch_add_rmsnorm2_q8(s.h, s.routed, nextnorm, s.x, s.xn, s.aq81, H, c.rms_eps, st);
                    else
                        kernels::launch_add_rmsnorm2(s.h, s.routed, nextnorm, s.x, s.xn, 1, H, c.rms_eps, st);
                }
                dflash_maybe_capture_layer(L);
                continue;
            }
            if (w.shared_gate_inp) {
                if (s.gguf) {
                    if (s.use_pq && w.shared_gate_inp_type == 12) {
                        if (s.use_llama) {
                            if (!fnq) kernels::launch_quantize_q8_1_blocks(s.hn, s.aq81, H, st);
                            if (H == 2048)
                                kernels::launch_mmvq_q4k_sigmoid(s.aq81, w.shared_gate_inp, s.d_shared_w, H, st);
                            else
                                kernels::launch_mmvq_q4k(s.aq81, w.shared_gate_inp, s.shared_gate_tmp, 1, H, st);
                        } else {
                            kernels::launch_quantize_q8_1(s.hn, s.aq8, s.aq8_d, s.aq8_s, H, st);
                            kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s,
                                                            w.shared_gate_inp, s.shared_gate_tmp, 1, H, st);
                        }
                    } else if (s.use_pq && s.use_llama && s.use_q6mmvq && w.shared_gate_inp_type == 14) {
                        if (!fnq) kernels::launch_quantize_q8_1_blocks(s.hn, s.aq81, H, st);
                        kernels::launch_mmvq_q6k(s.aq81, w.shared_gate_inp, s.shared_gate_tmp, 1, H, st);
                    } else if (w.shared_gate_inp_type) {
                        kernels::launch_gemv_q(s.hn, w.shared_gate_inp, w.shared_gate_inp_type, s.shared_gate_tmp, 1, H, st);
                    } else {
                        static int gs2 = -1;
                        if (gs2 < 0) { const char* e = getenv("SPARKINFER_GEMV_SIGMOID");
                            gs2 = (e && e[0] == '1') ? 1 : 0; }
                        if (gs2) {
                            kernels::launch_gemv_sigmoid(s.hn, w.shared_gate_inp, s.shared_gate_tmp, s.d_shared_w, H, st);
                        } else {
                            kernels::launch_gemv(s.hn, w.shared_gate_inp, s.shared_gate_tmp, 1, H, st);
                            kernels::launch_qwen36_sigmoid_scalar(s.shared_gate_tmp, s.d_shared_w, st);
                        }
                    }
                } else {
                    kernels::launch_gemm(s.hn, w.shared_gate_inp, s.shared_gate_tmp, 1, 1, H, 1.f, 0.f, gc, st);
                    kernels::launch_qwen36_sigmoid_scalar(s.shared_gate_tmp, s.d_shared_w, st);
                }
            }
            if (s.gguf) {
                if (qmoe) {
                    // MoE already wrote routed; safe to accum shared down into it (opt-in).
                    static int shexp_accum = -1;
                    if (shexp_accum < 0) { const char* e = getenv("SPARKINFER_SHEXP_ACCUM");
                        shexp_accum = (e && e[0] == '1') ? 1 : 0; }
                    const bool sx_accum = shexp_accum != 0;
                    kernels::launch_shared_expert_q8_mmvq(
                        s.hn, fnq ? s.aq81 : nullptr,
                        w.shared_gate_q, w.shared_up_q, w.shared_down_q,
                        w.shared_gate_inp ? s.d_shared_w : nullptr,
                        sx_accum ? s.routed : s.shared, s.mf_h, s.aq81, H, c.moe_ffn, st,
                        sx_accum);
                    if (sx_accum) {
                        if (fnq)
                            kernels::launch_add_rmsnorm2_q8(s.h, s.routed, nextnorm, s.x, s.xn, s.aq81, H, c.rms_eps, st);
                        else
                            kernels::launch_add_rmsnorm2(s.h, s.routed, nextnorm, s.x, s.xn, 1, H, c.rms_eps, st);
                        dflash_maybe_capture_layer(L);
                        continue;
                    }
                } else {
                    kernels::launch_gemv(s.hn, w.shared_gate, s.sh_gate, c.moe_ffn, H, st);
                    kernels::launch_gemv(s.hn, w.shared_up,   s.sh_up,   c.moe_ffn, H, st);
                    kernels::launch_qwen36_shared_swiglu(s.sh_gate, s.sh_up, s.d_shared_w, s.sh_h, c.moe_ffn, st);
                    kernels::launch_gemv(s.sh_h, w.shared_down, s.shared, H, c.moe_ffn, st);
                }
            } else {
                // set_weights path: shared weights are [hidden,ffn]/[ffn,hidden] dense.
                kernels::launch_moe_expert_ffn(s.hn, w.shared_gate, w.shared_up, w.shared_down,
                                               s.d_shared_ids, s.d_shared_w, s.shared,
                                               1, 1, 1, H, c.moe_ffn, st);
            }
            if (s.use_addnorm3) shared_to_fold = s.shared;
            else launch_residual_add(s.routed, s.shared, s.routed, H, st);
        }
        const void* nextnorm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
        dbg_bf16(s.routed, H, 60, L);   // tag 60: routed = dense FFN(hn) output, pre sandwich-norm
        if (c.muse_glimmer) {
            // Sandwich norm: x = h + RMSNorm(routed) * post_ffn_norm (norm the FFN output
            // alone, not the sum -- mirrors the post-attention step above). No shared
            // expert on this architecture (dense_ffn, n_shared=0), so shared_to_fold is
            // always null here. xn = RMSNorm(x, nextnorm) is the next layer's ordinary
            // pre-attn norm (or final_norm on the last layer), unaffected by the sandwich.
            // Same 1e-8 post_norm_eps as the post-attn sandwich norm above -- see that comment.
            // Window 2 lands (the whole FFN has run since it was forked). Window 3: cover the
            // post-FFN sandwich tail with the next layer's attention projections.
            {
                if (pf_win & 2) pf_join();
                if (pf_win & 4) {
                    const Qwen35LayerWeights* nx = (L + 1 < c.n_layers) ? &s.w.layers[L + 1] : nullptr;
                    if (nx) pf_fork(nx->wq, nx->wgate ? nx->wgate : nx->wk);
                    else    pf_fork(s.w.lm_head, nullptr);
                }
            }
            if (muse_fuse_tail())
                kernels::launch_muse_sandwich_tail(s.h, s.routed, w.post_ffn_norm, nextnorm,
                                                   s.x, s.xn, 1, H, 1e-8f, c.rms_eps, st);
            else {
            kernels::launch_norm_then_add(s.h, s.routed, w.post_ffn_norm, s.x, 1, H, 1e-8f, st);
            dbg_bf16(s.x, H, 70, L);   // tag 70: x = h + sandwich_norm(routed)  (layer output)
            kernels::launch_rmsnorm(s.x, nextnorm, s.xn, 1, H, c.rms_eps, st);
            }
        } else if (shared_to_fold) {
            if (fnq)
                kernels::launch_add_rmsnorm3_q8(s.h, s.routed, shared_to_fold, nextnorm, s.x, s.xn, s.aq81, H, c.rms_eps, st);
            else
                kernels::launch_add_rmsnorm3(s.h, s.routed, shared_to_fold, nextnorm, s.x, s.xn, 1, H, c.rms_eps, st);
        } else if (fnq)
            kernels::launch_add_rmsnorm2_q8(s.h, s.routed, nextnorm, s.x, s.xn, s.aq81, H, c.rms_eps, st);
        else
            kernels::launch_add_rmsnorm2(s.h, s.routed, nextnorm, s.x, s.xn, 1, H, c.rms_eps, st);
        dflash_maybe_capture_layer(L);
    }
    pf_join();   // last layer's prefetch: must be joined before the capture ends or the graph is malformed
    // xn now holds RMSNorm(x_final, final_norm)
    dbg_bf16(s.xn, H, 80, -2);   // tag 80: final-norm output (lm_head input)
    dbg_xn_snapshot(s.xn, c.n_layers);   // extra slot: final-norm output, for lm_head cross-check
    if (!sample) {
        if (capturing_graph) {
            cu(cudaStreamEndCapture(st, &s.cu_prefill_graph), "end prefill capture");
            cu(cudaGraphInstantiate(&s.cu_prefill_exec, s.cu_prefill_graph, 0), "prefill graph instantiate");
            s.graph_prefill_ready = true;
            s.graph_prefill_attn_mode = attn_graph_mode;
            cu(cudaGraphLaunch(s.cu_prefill_exec, st), "prefill graph launch (first)");
        }
        cu(cudaStreamSynchronize(st), "prefill sync");
        return token_id;
    }
    // Third instance of the same stale-Q8_1-cache pattern as prepare_xn_quant's xn_q8_ready
    // (L>0) and the dense_ffn gate/up input above: `fnq` only promises aq81==Q8_1(xn) here
    // because the fnq path's final-layer tail is launch_add_rmsnorm2_q8/add_rmsnorm3_q8,
    // which writes xn AND emits Q8_1(xn) into aq81 as a side effect. Muse Glimmer's final-
    // layer tail is the c.muse_glimmer sandwich-norm branch above (launch_norm_then_add then
    // a plain launch_rmsnorm into s.xn) -- no Q8 side channel. `fnq` itself doesn't check
    // c.muse_glimmer (it's just s.gguf/use_fnq/use_pq/use_llama), so with fnq true this
    // silently fed the LM head a stale aq81 left over from the last layer's fresh
    // prepare_xn_quant(xn) quantize (a *different*, pre-final-norm activation vector) --
    // wrong logits on every single decode step. Force a fresh quantize for muse_glimmer.
    if (s.gguf && s.use_pq && s.use_llama && s.w.lm_head_type == 12) {
        if (!fnq || c.muse_glimmer) kernels::launch_quantize_q8_1_blocks(s.xn, s.aq81, H, st);
        kernels::launch_mmvq_q4k_f32(s.aq81, s.w.lm_head, s.logits, c.vocab, H, st);
    }
    else if (s.gguf && s.use_q6mmvq && s.w.lm_head_type == 14) {   // int8 Q6_K dp4a LM head (1 warp/row)
        if (!fnq || c.muse_glimmer) kernels::launch_quantize_q8_1_blocks(s.xn, s.aq81, H, st);  // else aq81 = Q8_1(xn) from final norm
        kernels::launch_gemv_q6k_dp4a_f32(s.aq81, s.w.lm_head, s.logits, c.vocab, H, st);
    }
    else if (s.gguf && s.w.lm_head_type) kernels::launch_gemv_q_f32(s.xn, s.w.lm_head, s.w.lm_head_type, s.logits, c.vocab, H, st);
    else if (s.gguf)                kernels::launch_gemv_f32(s.xn, s.w.lm_head, s.logits, c.vocab, H, st);  // lm_head native [vocab,H]
    else        kernels::launch_linear_f32(s.xn, s.w.lm_head, s.logits, 1, c.vocab, H, st);
    dbg_f32(s.logits, c.vocab, 90, -2);   // tag 90: raw logits (pre logit_scale/softcap)
    if (c.muse_glimmer && c.final_logit_softcapping > 0.f)
        kernels::launch_logit_softcap(s.logits, 1, c.vocab, c.logit_scale, c.final_logit_softcapping, st);
    dbg_f32(s.logits, c.vocab, 91, -2);   // tag 91: final logits (post logit_scale/softcap)
    // Always launched, never host-gated: logit_bias has NO inertness proof at temperature<=0 (an
    // arbitrary per-vocab additive bias CAN change the greedy-argmax winner on its own) -- see
    // qwen35.h's forward_token doc comment. Reads whatever is CURRENTLY in s.logit_bias, set once
    // per request by set_logit_bias (not refreshed here every decode step, unlike the scalar
    // params below) -- correct because activate_session() swaps s.logit_bias to the request's own
    // session buffer before any of this request's forward_token() calls run.
    kernels::launch_logit_bias(s.logits, s.logit_bias, c.vocab, st);
    // Always launched, never host-gated: presence/frequency penalty has NO inertness proof at
    // presence_penalty==0 && frequency_penalty==0 the way top_k/top_p do at temperature<=0 -- see
    // qwen35.h's forward_token doc comment. Applied BEFORE launch_topk_topp_mask's sort so the
    // penalized distribution flows through top_k/top_p truncation, temperature sampling, AND
    // logprobs reporting -- matches real-world OpenAI/vLLM behavior of reporting logprobs against
    // what was actually sampled from.
    kernels::launch_presence_frequency_penalty(s.logits, s.penalty_counts, c.vocab,
                                               s.d_sample_presence_penalty, s.d_sample_frequency_penalty, st);
    // Always launched, never host-gated on top_k/top_p/temperature: this call site is inside a
    // CUDA graph whose node topology is frozen at capture time and may be replayed for a LATER,
    // separate request (e.g. via use_prefix_session's shared seq_id=0) with different values. Both
    // kernels read their params from device memory refreshed above and no-op (or are provably
    // inert -- see qwen35.h's forward_token doc comment) when disabled/at temperature<=0.
    kernels::launch_topk_topp_mask(s.logits, c.vocab, s.d_vocab_iota, s.d_sorted_logits, s.d_sorted_idx,
                                   s.d_topk_exp, s.d_topk_cumsum, s.d_sort_temp, s.sort_temp_bytes,
                                   s.d_scan_temp, s.scan_temp_bytes,
                                   s.d_sample_top_k, s.d_sample_top_p, s.d_rank_by_id, st);
    kernels::launch_temperature_sample(s.logits, 1, c.vocab, s.d_sample_temp, s.d_sample_seed,
                                       s.d_sample_step, st);
    kernels::launch_argmax(s.logits, s.d_out_id, 1, c.vocab, st);
    // Always launched, never host-gated on logprobs: same CUDA-graph-captured-region rationale as
    // launch_topk_topp_mask/launch_temperature_sample above -- this call site may replay for a
    // LATER, different request whose logprobs setting differs. Negligible cost (single thread).
    kernels::launch_extract_chosen_logit(s.d_out_id, s.d_rank_by_id, s.d_sorted_logits,
                                         s.d_chosen_logit, st);
    // Always launched, unconditionally: records this step's sampled token into the CURRENT
    // session's running penalty count so the NEXT decode step's penalty application sees it.
    kernels::launch_increment_penalty_count(s.penalty_counts, s.d_out_id, st);
    if (s.bench_feedback_graph) kernels::launch_decode_feedback(s.d_scalars, s.d_out_id, st);

    if (capturing_graph && dflash_cap) {
        cu(cudaStreamEndCapture(st, &s.cu_dflash_graph), "end dflash capture");
        cu(cudaGraphInstantiate(&s.cu_dflash_exec, s.cu_dflash_graph, 0), "dflash graph instantiate");
        s.dflash_graph_ready = true;
        s.dflash_graph_attn_mode = attn_graph_mode;
        s.dflash_graph_sparse = sparse_on;
        {
            static int dbg = -1;
            if (dbg < 0) { const char* e = getenv("SPARKINFER_GRAPH_DEBUG"); dbg = (e && e[0]=='1') ? 1 : 0; }
            if (dbg) {
                const int mma_chunk = (s.n_splits > 0) ? (seqlen + s.n_splits - 1) / s.n_splits : 0;
                fprintf(stderr, "[dflash-graph] capture pos=%d seqlen=%d n_splits=%d attn_mode=%d mma_chunk=%d sparse=%d\n",
                        position, seqlen, s.n_splits, attn_graph_mode, mma_chunk, sparse_on ? 1 : 0);
            }
        }
        cu(cudaGraphLaunch(s.cu_dflash_exec, st), "dflash graph launch (first)");
    } else if (capturing_graph) {
        cu(cudaStreamEndCapture(st, &s.cu_graph), "end capture");
        cu(cudaGraphInstantiate(&s.cu_exec, s.cu_graph, 0), "graph instantiate");
        s.graph_ready = true;
        s.graph_attn_mode = attn_graph_mode;
        s.graph_sparse = sparse_on;
        static int graph_dbg = -1;
        if (graph_dbg < 0) {
            const char* e = getenv("SPARKINFER_GRAPH_DEBUG");
            graph_dbg = (e && e[0] == '1') ? 1 : 0;
        }
        if (graph_dbg) {
            const int mma_chunk = (s.n_splits > 0) ? (seqlen + s.n_splits - 1) / s.n_splits : 0;
            fprintf(stderr, "[graph] capture pos=%d seqlen=%d n_splits=%d attn_mode=%d mma_chunk=%d sparse=%d\n",
                    position, seqlen, s.n_splits, attn_graph_mode, mma_chunk, sparse_on ? 1 : 0);
        }
        cu(cudaGraphLaunch(s.cu_exec, st), "graph launch (first)");
    }

    cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "out_id");
    cu(cudaStreamSynchronize(st), "sync");
    if (mgdump) {
        std::vector<bf16> host((size_t)(c.n_layers + 1) * H);
        cu(cudaMemcpy(host.data(), s.dbg_xn_dump, host.size() * sizeof(bf16), cudaMemcpyDeviceToHost),
           "dbg_xn_dump readback");
        const char* path = getenv("SPARKINFER_MG_DUMP_FILE");
        FILE* f = fopen(path ? path : "/tmp/mg_xn_dump.bin", "wb");
        if (f) { fwrite(host.data(), sizeof(bf16), host.size(), f); fclose(f); }
        fprintf(stderr, "[mg-debug] dumped xn[layer,H=%d] for step=%d, %d layers -> %s\n",
                H, position, c.n_layers, path ? path : "/tmp/mg_xn_dump.bin");
    }
    return *s.h_out_id;
}

namespace {
bool prefill_samples_lmhead() {
    static int legacy = -1;
    if (legacy < 0) {
        const char* e = getenv("SPARKINFER_PREFILL_LEGACY");
        legacy = (e && e[0] == '1') ? 1 : 0;
    }
    return legacy != 0;
}

// Batched prefill (prefill_batched_run). Default ON; SPARKINFER_PREFILL_BATCHED=0 disables. Supports
// the Qwythos dense-hybrid AND the Qwen3.6-35B-A3B MoE hybrid (dense_ffn=false, n_experts>0) — both
// share the GDN + attention batched kernels; only the FFN differs. From position 0 only.
bool batched_prefill_enabled(bool gguf, const Qwen35Config& cfg, int n_tokens) {
    static int want_batched = -1, batched_maxctx = -1;
    if (want_batched < 0) {
        const char* e = getenv("SPARKINFER_PREFILL_BATCHED");
        want_batched = (e && e[0] == '0') ? 0 : 1;
        // 128k: the windowed prefill attention (#455) is O(N*window) and the FFN scratch is chunked
        // (prefill_batched_run), so the batched pass now fits VRAM and stays flat ~18k pp up to 128k
        // (vs the ~300 pp sequential fallback). Raised from 64k. SPARKINFER_PREFILL_BATCHED_MAXCTX overrides.
        const char* mc = getenv("SPARKINFER_PREFILL_BATCHED_MAXCTX");
        batched_maxctx = mc ? atoi(mc) : 131072;
    }
    // dense hybrid (Qwythos) or the Qwen3.6 MoE hybrid — prefill_batched_run validates the
    // MoE requirements (256 experts, quantized experts + router) itself and returns -1 to
    // fall back if unsupported.
    const bool ffn_ok = cfg.dense_ffn || cfg.n_experts > 0;
    // Muse Glimmer: windowed batched prefill IS implemented in prefill_batched_run -- per-layer
    // NoPE (rotary_dim=0) on the global layers, pure rolling-window attention (launch_prefill_attn_
    // swa_pure_bf16) on the SWA layers, bf16 KV write, and sandwich/embedding norm + logit softcap
    // parity with the decode path. DEFAULT ON: numerically equivalent to the token-loop prefill
    // (batched-vs-token-loop parity PPL 1.232 vs 1.233 & identical argmax; compute-sanitizer 0
    // errors; Qwen3.6 cross-model guard clean). SPARKINFER_MUSE_BATCHED=0 forces the sequential
    // token-loop fallback (kept for A/B and as an escape hatch).
    if (cfg.muse_glimmer) {
        static const int muse_batched = []{ const char* e = getenv("SPARKINFER_MUSE_BATCHED"); return (e && e[0] == '0') ? 0 : 1; }();
        if (!muse_batched) return false;
    }
    return want_batched && gguf && cfg.hybrid && ffn_ok && n_tokens > 0 &&
           n_tokens <= batched_maxctx;
}

// LMCache chunk size, tokens. Doubles as both the LOOKUP eligibility threshold (below this many
// tokens, always recompute locally rather than pay an IPC round trip) and the STORE alignment
// unit (a store range is rounded down to the nearest multiple, matching what the sidecar's own
// LMCache instance is configured with -- see bridge/lmcache_bridge.py's identically-named env
// var). Not negotiated from the sidecar's HELLO_ACK at runtime (that field exists on the wire
// but is currently informational-only on this side, see lmcache_bridge_client.cpp) -- both sides
// must be configured to agree; a mismatch degrades safely (misaligned STOREs are rejected by the
// sidecar, not corrupted) rather than crashing, so this is a real but non-catastrophic
// simplification, not a hidden correctness trap.
int lmcache_chunk_size_tokens() {
    static int chunk = [] {
        const char* e = getenv("SPARKINFER_LMCACHE_CHUNK_SIZE");
        const int v = e ? atoi(e) : 256;
        return v > 0 ? v : 256;
    }();
    return chunk;
}

BridgeKVLayout lmcache_layout_from_cfg(const Qwen35Config& cfg, const KVCacheManager& kv) {
    BridgeKVLayout layout;
    layout.num_layers = cfg.n_layers;
    layout.num_kv_heads = cfg.n_kv_heads;
    layout.head_dim = cfg.head_dim;
    layout.block_size = kv.block_size();
    layout.int8_kv = kv.int8_kv();
    layout.elem_bytes = kv.int8_kv() ? 1 : 2;
    // model_name is unused for the staging byte-math this layout feeds (stage_kv_to_shm /
    // restore_kv_from_shm never read it) -- the BridgeClient's own layout, fixed at construction
    // and used for HELLO, is what actually needs to match the sidecar's.
    return layout;
}

// Stores tokens[start, end) (rounded down to the nearest lmcache_chunk_size_tokens() boundary --
// a trailing partial chunk is silently dropped, not sent, since the sidecar would reject it as
// misaligned anyway) for seq_id to the external cache tier. No-op if no bridge is attached, the
// bridge is currently unreachable, or nothing chunk-aligned remains. Fire-and-forget: the actual
// network round trip happens on BridgeClient's own background thread, this call only blocks for
// the (synchronous, on-stream) cudaMemcpy staging into shm.
//
// Takes individual fields rather than Impl& -- Impl is a private nested type, so a free function
// outside the class can't name it in a signature even from within the same translation unit
// (only member functions, which is why every call site below reads `s.<field>` from inside an
// actual Qwen35Model member function's `Impl& s = *p_;`).
void lmcache_maybe_store(BridgeClient* bridge, const Qwen35Config& cfg, KVCacheManager& kv,
                         cudaStream_t stream, uint64_t seq_id, const std::vector<int>& tokens,
                         int start, int end) {
    // Deliberately not gated on bridge->is_alive(): that only becomes true as a side effect of
    // a prior successful handshake, so gating on it here would mean the very first store in a
    // freshly started process never fires (nothing has ever connected yet to make it true) --
    // found via the live E2E test (lmcache_e2e_gpu_test.cpp), which is exactly the class of bug
    // that kind of test exists to catch. store_async() (and the LOOKUP call in
    // ingest_prompt_range, same reasoning) already handle "is the bridge actually reachable"
    // internally via their own lazy-connect + cooldown/backoff -- that's what their docstrings
    // mean by "safe to call regardless."
    if (!bridge) return;
    const int chunk = lmcache_chunk_size_tokens();
    const int aligned_end = (end / chunk) * chunk;
    if (aligned_end <= start) return;

    static std::atomic<uint64_t> counter{0};
    const std::string shm_name = "/sparkinfer_kv_store_" + std::to_string(seq_id) + "_" +
                                 std::to_string(counter.fetch_add(1));
    const BridgeKVLayout layout = lmcache_layout_from_cfg(cfg, kv);
    if (!stage_kv_to_shm(kv, layout, seq_id, start, aligned_end, shm_name, stream)) return;
    bridge->store_async(tokens, start, aligned_end, shm_name);
}
} // namespace

Qwen35Model::BenchDecodeResult Qwen35Model::bench_decode(int warmup, int n, int context_tokens) {
    BenchDecodeResult out{};
    Impl& s = *p_;
    static int last_bench_ctx = -1;
    if (context_tokens != last_bench_ctx && s.graph_ready) {
        cudaGraphExecDestroy(s.cu_exec);
        cudaGraphDestroy(s.cu_graph);
        s.cu_exec = nullptr;
        s.cu_graph = nullptr;
        s.graph_ready = false;
    }
    last_bench_ctx = context_tokens;
    if (!s.kv->allocate(s.active_seq_id, s.cfg.max_seq)) { fprintf(stderr, "[bench] kv allocate failed\n"); return out; }
    int start_pos = context_tokens;
    if (const char* e = getenv("SPARKINFER_BENCH_START_POS")) {
        start_pos = atoi(e);
    }
    if (start_pos < 0) start_pos = 0;
    if (start_pos + warmup + n > s.cfg.max_seq) {
        fprintf(stderr, "[bench] requested ctx=%d warmup=%d n=%d exceeds max_seq=%d\n",
                start_pos, warmup, n, s.cfg.max_seq);
        s.kv->free(s.active_seq_id);
        return out;
    }
    static int bench_device_loop = -1;
    if (bench_device_loop < 0) {
        const char* e = getenv("SPARKINFER_BENCH_DEVICE_LOOP");
        bench_device_loop = (e && e[0] == '0') ? 0 : 1;
    }
    s.bench_feedback_graph = bench_device_loop != 0;
    int pos = 0, tok = 100;
    // Batched prefill: one weight-amortized GEMM pass fills the KV cache + Gated-DeltaNet state, then
    // decode continues from start_pos. Default ON for the dense hybrid; SPARKINFER_PREFILL_BATCHED=0
    // (or ctx > SPARKINFER_PREFILL_BATCHED_MAXCTX, default 64k — the O(N^2) prefill attention is still
    // naive) falls back to the token loop below, which is left byte-identical to main on purpose.
    bool batched_done = false;
    if (start_pos > 0) {
        if (batched_prefill_enabled(s.gguf, s.cfg, start_pos)) {
            std::vector<int> ids(start_pos);
            for (int i = 0; i < start_pos; i++) ids[i] = 100 + (i % 20000);   // deterministic pseudo-prompt
            auto pb0 = std::chrono::high_resolution_clock::now();
            int seed = prefill_batched(ids.data(), start_pos);
            cudaDeviceSynchronize();
            auto pb1 = std::chrono::high_resolution_clock::now();
            if (seed >= 0) {
                out.prefill_pp = start_pos / std::chrono::duration<double>(pb1 - pb0).count();
                pos = start_pos;
                tok = (seed < s.cfg.vocab) ? seed : 100;
                batched_done = true;
            }
        }
    }
    if (start_pos > 0 && !batched_done) {
        auto p0 = std::chrono::high_resolution_clock::now();
        for (; pos < start_pos; pos++) {
            tok = forward_token(tok, pos, prefill_samples_lmhead());
            if (tok < 0 || tok >= s.cfg.vocab) tok = 100;
        }
        cudaDeviceSynchronize();
        auto p1 = std::chrono::high_resolution_clock::now();
        out.prefill_pp = start_pos / std::chrono::duration<double>(p1 - p0).count();
    }
    for (int i = 0; i < warmup; i++) {
        tok = forward_token(tok, pos++, true);
        if (tok < 0 || tok >= s.cfg.vocab) tok = 100;
    }
    if (s.graph_ready) cu(cudaGraphUpload(s.cu_exec, s.stream), "bench graph upload");
    cudaDeviceSynchronize();

    if (bench_device_loop && s.graph_ready) {
        s.h_scalars[0] = tok;
        s.h_scalars[1] = pos;
        s.h_scalars[2] = pos;
        s.h_scalars[3] = pos + 1;
        cu(cudaMemcpyAsync(s.d_scalars, s.h_scalars, 4 * sizeof(int), cudaMemcpyHostToDevice, s.stream), "bench scalars");
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; i++) {
            cu(cudaGraphLaunch(s.cu_exec, s.stream), "bench graph launch");
        }
        cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, s.stream), "bench final out");
        cu(cudaStreamSynchronize(s.stream), "bench sync");
        auto t1 = std::chrono::high_resolution_clock::now();
        s.kv->free(s.active_seq_id);
        s.bench_feedback_graph = false;
        double secs = std::chrono::duration<double>(t1 - t0).count();
        out.decode_tps = n / secs;
        return out;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        tok = forward_token(tok, pos++, true);
        if (tok < 0 || tok >= s.cfg.vocab) tok = 100;
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    s.kv->free(s.active_seq_id);
    s.bench_feedback_graph = false;
    double secs = std::chrono::duration<double>(t1 - t0).count();
    out.decode_tps = n / secs;
    return out;
}

void Qwen35Model::invalidate_decode_graph() {
    Impl& s = *p_;
    if (s.graph_ready) {
        cudaGraphExecDestroy(s.cu_exec);
        cudaGraphDestroy(s.cu_graph);
        s.cu_exec = nullptr;
        s.cu_graph = nullptr;
        s.graph_ready = false;
        s.graph_attn_mode = -1;
        s.graph_sparse = false;
    }
    if (s.dflash_graph_ready) {
        cudaGraphExecDestroy(s.cu_dflash_exec);
        cudaGraphDestroy(s.cu_dflash_graph);
        s.cu_dflash_exec = nullptr;
        s.cu_dflash_graph = nullptr;
        s.dflash_graph_ready = false;
        s.dflash_graph_attn_mode = -1;
        s.dflash_graph_sparse = false;
    }
}

bool Qwen35Model::prompt_matches_prefix(const std::vector<int>& prompt) const {
    const Impl& s = *p_;
    if (!s.prefix_active || s.prefix_len <= 0) return false;
    if (prompt.size() < (size_t)s.prefix_len) return false;
    for (int i = 0; i < s.prefix_len; i++)
        if (prompt[(size_t)i] != s.prefix_tokens[(size_t)i]) return false;
    return true;
}

int Qwen35Model::ingest_prompt_range(const int* ids, int start, int end, int chunk_limit,
                                     int* out_pos) {
    Impl& s = *p_;
    if (!ids || end <= start) {
        if (out_pos) *out_pos = start;
        return -1;
    }
    const int n = end - start;
    // Batched GEMM prefill never chunks (no start_pos support in prefill_batched_run yet) --
    // only eligible on the very first call for this range (start==0) and always covers the
    // whole [0,end) in one pass regardless of chunk_limit.
    if (start == 0 && batched_prefill_enabled(s.gguf, s.cfg, n)) {
        int seed = prefill_batched(ids, n);
        if (seed >= 0 && seed < s.cfg.vocab) {
            if (out_pos) *out_pos = end;
            return seed;
        }
    }

    // External KV cache lookup (docs/lmcache_bridge_protocol.md). Deliberately gated on
    // batched_prefill_enabled() having already been tried-and-failed (or never eligible) above:
    // the batched path is ~100x faster than the token loop and only ever eligible from position
    // 0, so restoring a prefix from the bridge only when we were headed for the token loop
    // anyway means a partial hit can never downgrade what would otherwise be a fast batched pass
    // into a slower one for the remainder -- it's a strict win, never a trade-off. Only on the
    // very first call for this range (start==0); a chunked resumption call (start>0, continuing
    // a previous partial token-loop advance) never re-queries -- one round trip per prefill.
    // Deliberately not gated on lmcache_bridge->is_alive(): that only becomes true as a side
    // effect of a prior successful handshake, so gating on it here would mean the very first
    // lookup in a freshly started process never fires (found via the live E2E test,
    // lmcache_e2e_gpu_test.cpp -- exactly the class of bug that test exists to catch).
    // lookup() already handles "is the bridge actually reachable" internally via its own
    // lazy-connect + cooldown/backoff, bounded by the 5ms default timeout either way.
    int actual_start = start;
    if (start == 0 && s.lmcache_bridge && n >= lmcache_chunk_size_tokens()) {
        LookupResult res = s.lmcache_bridge->lookup(std::vector<int>(ids, ids + end));
        if (res.ok && res.matched_tokens > 0) {
            const BridgeKVLayout layout = lmcache_layout_from_cfg(s.cfg, *s.kv);
            bool restored = true;
            for (const BridgeKVChunk& c : res.chunks) {
                if (!restore_kv_from_shm(*s.kv, layout, s.active_seq_id, res.shm_name,
                                         c.shm_offset_bytes, c.start_tok, c.len_tok, s.stream)) {
                    restored = false;
                    break;
                }
            }
            // A partial restore failure means some positions in [0, matched_tokens) may hold
            // garbage KV -- never resume the token loop past whatever was verified fully
            // restored, and never resume past a failure point at all: fall back to recomputing
            // the entire range from 0 rather than risk attending against corrupted KV.
            //
            // Clamped to end-1, never end itself: a full hit (matched_tokens == end) would
            // otherwise skip the token loop entirely, and the token loop is what produces the
            // decode seed (the LM-head logits at the last prompt position) -- the KV cache lets
            // attention skip recomputing *past* positions, it doesn't let this function skip
            // computing the *current*/last position's own forward pass, which is what next's
            // value actually comes from. Found via lmcache_bench.cpp's benchmark, whose prompt
            // length happened to land on an exact chunk boundary (matched_tokens == end) --
            // earlier tests never hit this because their prompts weren't exact chunk multiples,
            // so the token loop always had at least one real remaining token regardless.
            if (restored) actual_start = std::min(res.matched_tokens, end - 1);
        }
    }

    int limit = end - actual_start;
    if (chunk_limit > 0 && limit > chunk_limit) limit = chunk_limit;
    const int stop = actual_start + limit;
    int next = -1;
    for (int i = actual_start; i < stop; i++) {
        // Only sample (compute logits/argmax) at the true end of the whole range, not at a
        // chunk boundary -- decode doesn't start until the full prompt is ingested, so
        // intermediate chunk-final tokens never need a logits pass (unless the legacy
        // every-step flag is set).
        const bool sample = prefill_samples_lmhead() || i + 1 == end;
        int r = forward_token(ids[i], i, sample);
        if (sample) next = r;
    }
    if (out_pos) *out_pos = stop;
    return (stop >= end) ? next : -1;
}

bool Qwen35Model::cache_prefix(const std::vector<int>& tokens) {
    Impl& s = *p_;
    clear_prefix_cache();
    if (tokens.empty()) return false;
    if (tokens.size() > (size_t)s.cfg.max_seq) return false;
    invalidate_decode_graph();
    if (!s.kv->allocate(s.active_seq_id, s.cfg.max_seq)) return false;
    const int n = (int)tokens.size();
    int next = ingest_prompt_range(tokens.data(), 0, n);
    if (next < 0 || next >= s.cfg.vocab) {
        s.kv->free(s.active_seq_id);
        return false;
    }
    cudaDeviceSynchronize();
    s.prefix_tokens = tokens;
    s.prefix_len = (int)tokens.size();
    s.prefix_next = next;
    s.prefix_active = true;
    return true;
}

void Qwen35Model::clear_prefix_cache() {
    Impl& s = *p_;
    if (s.prefix_active || s.kv->allocated_tokens(s.active_seq_id) > 0) {
        // Store to the external cache tier before the blocks it reads are freed below -- this
        // is the "prefix cache overwrite" eviction point (docs/lmcache_bridge_protocol.md's
        // STORE trigger list): the active prefix is about to be dropped for a different one.
        if (s.prefix_active)
            lmcache_maybe_store(s.lmcache_bridge, s.cfg, *s.kv, s.stream, s.active_seq_id,
                               s.prefix_tokens, 0, s.prefix_len);
        s.kv->free(s.active_seq_id);
        invalidate_decode_graph();
    }
    s.prefix_tokens.clear();
    s.prefix_len = 0;
    s.prefix_next = -1;
    s.prefix_active = false;
}

void Qwen35Model::release_prefix_session() {
    Impl& s = *p_;
    // Caller already freed session-0 KV. Keep the token fingerprint for matching;
    // mark inactive so the next request re-runs cache_prefix() instead of
    // attending against an empty block table.
    s.prefix_next = -1;
    s.prefix_active = false;
    invalidate_decode_graph();
}

int Qwen35Model::prefix_cached_len() const { return p_->prefix_active ? p_->prefix_len : 0; }

int Qwen35Model::prefix_seed_token() const {
    const Impl& s = *p_;
    return (s.prefix_active && s.prefix_next >= 0) ? s.prefix_next : -1;
}

double Qwen35Model::bench_ttft(const std::vector<int>& prompt) {
    Impl& s = *p_;
    if (prompt.empty()) return 0.;
    const bool reuse = prompt_matches_prefix(prompt);
    if (!reuse) {
        clear_prefix_cache();
        invalidate_decode_graph();
        if (!s.kv->allocate(s.active_seq_id, s.cfg.max_seq)) return -1.;
    } else if (!s.kv->allocate(s.active_seq_id, s.cfg.max_seq)) {
        return -1.;
    }
    const int start = reuse ? s.prefix_len : 0;
    if (getenv("SPARKINFER_DEBUG_PREFIX"))
        fprintf(stderr, "[prefix] ttft n=%zu start=%d reuse=%d cached=%d\n",
                prompt.size(), start, (int)reuse, s.prefix_len);
    s.bench_feedback_graph = false;
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    (void)ingest_prompt_range(prompt.data(), start, (int)prompt.size());
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    if (!reuse) {
        s.kv->free(s.active_seq_id);
        invalidate_decode_graph();
        if (s.graph_ready) {
            cu(cudaGraphExecDestroy(s.cu_exec), "ttft graph destroy exec");
            cu(cudaGraphDestroy(s.cu_graph), "ttft graph destroy");
            s.cu_exec = nullptr;
            s.cu_graph = nullptr;
            s.graph_ready = false;
        }
        if (s.graph_prefill_ready) {
            cu(cudaGraphExecDestroy(s.cu_prefill_exec), "ttft prefill graph destroy exec");
            cu(cudaGraphDestroy(s.cu_prefill_graph), "ttft prefill graph destroy");
            s.cu_prefill_exec = nullptr;
            s.cu_prefill_graph = nullptr;
            s.graph_prefill_ready = false;
            s.graph_prefill_attn_mode = -1;
        }
    }
    return std::chrono::duration<double>(t1 - t0).count();
}

// Thin adapter: hand the batched-prefill orchestration (qwen35_prefill.cpp) exactly the scratch
// buffers, streams and config it needs, so Impl stays private to this file.
int Qwen35Model::prefill_batched(const int* prompt_ids, int n) {
    Impl& s = *p_;
    auto it = s.sessions.find(s.active_seq_id);
    float* lin_state = (it != s.sessions.end()) ? it->second.lin_state : s.lin_state;
    bf16* lin_conv = (it != s.sessions.end()) ? it->second.lin_conv_state : s.lin_conv_state;
    Qwen35PrefillCtx ctx{ s.cfg, s.w, s.kv, s.stream, s.stream_k, s.stream_v, s.active_seq_id,
                          lin_state, lin_conv,
                          s.logits, s.d_out_id, s.h_out_id, s.gguf,
                          s.emb_norm_ones,
                          s.qdim, s.kvdim, s.linear_qdim, s.linear_vdim, s.linear_qkvdim,
                          s.moe_rs_gate, s.moe_rs_up, s.moe_rs_down, s.n_splits };
    return prefill_batched_run(ctx, prompt_ids, n);
}

int Qwen35Model::session_token_budget(size_t prompt_len, int max_new, int max_seq) {
    const long need = (long)prompt_len + max_new + 16;
    if (need <= 0) return 16;
    if (need > max_seq) return max_seq;
    return (int)need;
}

uint64_t Qwen35Model::open_session(int num_tokens, bool* alloc_failed) {
    Impl& s = *p_;
    if (num_tokens <= 0) return 0;
    const uint64_t seq_id = s.next_session_id.fetch_add(1);
    if (!s.kv->allocate(seq_id, num_tokens)) return 0;   // pool full -- normal, transient
    SessionBuffers buf;
    // Unconditional, every model -- unlike lin_state/lin_conv_state below (hybrid-only). This
    // fresh session serves exactly one request end-to-end before close_session() frees it (1:1
    // lifecycle via ContinuousBatchEngine::finish_job), so a one-time zero here is sufficient --
    // unlike session 0, which is reused across many DIFFERENT requests and needs an explicit
    // per-request reset (see reset_penalty_counts()).
    buf.penalty_counts = s.alloc<int>(s.cfg.vocab);
    buf.logit_bias = s.alloc<float>(s.cfg.vocab);
    bool alloc_ok = buf.penalty_counts != nullptr && buf.logit_bias != nullptr;
    if (s.cfg.hybrid) {
        buf.lin_state = s.alloc<float>((size_t)s.cfg.n_layers * s.cfg.linear_v_heads *
                                       s.cfg.linear_head_dim * s.cfg.linear_head_dim);
        buf.lin_conv_state = s.alloc<bf16>((size_t)s.cfg.n_layers *
                                           (s.cfg.linear_conv_kernel - 1) * s.linear_qkvdim);
        alloc_ok = alloc_ok && buf.lin_state && buf.lin_conv_state;
    }
    if (!alloc_ok) {
        // A real cudaMalloc failure (already logged by alloc<T>'s cu() wrapper as
        // "[qwen35] malloc: out of memory") -- not the KV pool being full, which was already
        // checked above. Surfaced separately so callers can tell "genuinely no capacity right
        // now" (retry later) apart from "device is out of memory" (permanent until restart).
        if (alloc_failed) *alloc_failed = true;
        s.kv->free(seq_id);
        if (buf.penalty_counts) cudaFree(buf.penalty_counts);
        if (buf.logit_bias) cudaFree(buf.logit_bias);
        if (buf.lin_state) cudaFree(buf.lin_state);
        if (buf.lin_conv_state) cudaFree(buf.lin_conv_state);
        return 0;
    }
    // Explicit zero, not relying on alloc<T>'s (plain cudaMalloc) zeroing guarantee -- unlike
    // lin_state/lin_conv_state, which are written wholesale before ever being read (GDN state is
    // computed fresh as the prefill/decode loop processes this session's own tokens), a stale,
    // nonzero penalty count would be silently, incorrectly wrong from token 1.
    cu(cudaMemsetAsync(buf.penalty_counts, 0, (size_t)s.cfg.vocab * sizeof(int), s.stream),
       "penalty_counts zero");
    // logit_bias: same one-time-zero-is-sufficient reasoning as penalty_counts above (1:1 fresh-
    // session lifecycle) -- the actual per-request VALUES are set later by set_logit_bias(), called
    // from submit_locked() right after this open_session() returns.
    cu(cudaMemsetAsync(buf.logit_bias, 0, (size_t)s.cfg.vocab * sizeof(float), s.stream),
       "logit_bias zero");
    s.sessions[seq_id] = buf;
    return seq_id;
}

void Qwen35Model::set_lmcache_bridge(BridgeClient* bridge) { p_->lmcache_bridge = bridge; }

void Qwen35Model::close_session(uint64_t seq_id, const std::vector<int>* store_tokens) {
    Impl& s = *p_;
    if (seq_id == 0) return;
    // Store to the external cache tier before freeing the blocks it reads -- this is the
    // "session close" eviction point (docs/lmcache_bridge_protocol.md's STORE trigger list).
    // store_tokens is null for most callers (this model class doesn't itself track a session's
    // original prompt; only ContinuousBatchEngine::finish_job has that context) -- a null
    // pointer is a pure no-op here, not a degraded path.
    if (store_tokens && !store_tokens->empty())
        lmcache_maybe_store(s.lmcache_bridge, s.cfg, *s.kv, s.stream, seq_id, *store_tokens, 0,
                           (int)store_tokens->size());
    s.kv->free(seq_id);
    auto it = s.sessions.find(seq_id);
    if (it != s.sessions.end()) {
        // open_session() only ever stores a freshly cudaMalloc'd, uniquely-owned buffer here
        // (seq_id == 0 -- the one case that could alias the model's persistent default buffers
        // -- returns before this point, see the guard at the top of this function). A `!=
        // s.lin_state` check used to gate these frees, but activate_session(seq_id) (called at
        // the top of every ContinuousBatchEngine::step_job, before finish_job/close_session runs
        // later in that same call) unconditionally aliases s.lin_state to *this* session's own
        // buffer -- so by the time close_session() ran, the two pointers were always equal and
        // the free was always skipped. Every hybrid-model (Muse Glimmer, Qwen3.6) request via the
        // server leaked its full lin_state/lin_conv_state allocation, permanently, confirmed live
        // as a fixed ~108 MiB/request leak independent of token count (#779).
        if (it->second.lin_state) cudaFree(it->second.lin_state);
        if (it->second.lin_conv_state) cudaFree(it->second.lin_conv_state);
        // Unconditional, every model -- mirrors the lin_state/lin_conv_state free above exactly.
        // The seq_id == 0 guard at the top of this function already protects
        // penalty_counts_default from ever being freed here, same aliasing-safety story as #779.
        if (it->second.penalty_counts) cudaFree(it->second.penalty_counts);
        // Unconditional, every model -- mirrors penalty_counts's free exactly, same aliasing-safety
        // story (the seq_id == 0 guard above already protects logit_bias_default).
        if (it->second.logit_bias) cudaFree(it->second.logit_bias);
        s.sessions.erase(it);
    }
    if (s.active_seq_id == seq_id) activate_session(0);
}

void Qwen35Model::activate_session(uint64_t seq_id) {
    Impl& s = *p_;
    if (s.active_seq_id == seq_id) return;
    s.active_seq_id = seq_id;
    auto it = s.sessions.find(seq_id);
    if (it == s.sessions.end() && seq_id == 0) it = s.sessions.find(0);  // defensive fallback
    if (it != s.sessions.end()) {
        if (s.cfg.hybrid) {
            s.lin_state = it->second.lin_state;
            s.lin_conv_state = it->second.lin_conv_state;
        }
        // penalty_counts/logit_bias swap for EVERY model (hybrid or not) -- unlike lin_state/
        // lin_conv_state above, these aren't architecture-specific, they're per-request sampling-
        // control state.
        s.penalty_counts = it->second.penalty_counts;
        s.logit_bias = it->second.logit_bias;
    }
    invalidate_decode_graph();
}

uint64_t Qwen35Model::active_session() const { return p_->active_seq_id; }

void Qwen35Model::reset_penalty_counts(uint64_t seq_id) {
    Impl& s = *p_;
    // Looked up via the sessions map directly, NOT s.penalty_counts (the "currently active"
    // pointer, which belongs to whatever the WORKER thread last swapped in via activate_session()
    // for some other job entirely) -- this is called from ContinuousBatchEngine::submit_locked()
    // on the HTTP-facing thread, so going through the map avoids racing/aliasing the currently-
    // active decode's scratch.
    auto it = s.sessions.find(seq_id);
    if (it == s.sessions.end() || !it->second.penalty_counts) return;   // defensive; should not happen
    cu(cudaMemsetAsync(it->second.penalty_counts, 0, (size_t)s.cfg.vocab * sizeof(int), s.stream),
       "penalty_counts reset");
}

void Qwen35Model::set_logit_bias(uint64_t seq_id, const std::vector<std::pair<int, float>>& bias) {
    Impl& s = *p_;
    // Looked up via the sessions map directly, same "HTTP-facing thread, not the worker's currently
    // active session" reasoning as reset_penalty_counts.
    auto it = s.sessions.find(seq_id);
    if (it == s.sessions.end() || !it->second.logit_bias) return;   // defensive; should not happen
    // Unconditional zero first, even for an empty bias -- session 0 is reused across unrelated
    // requests, so a request with no logit_bias must not inherit a PRIOR request's bias.
    cu(cudaMemsetAsync(it->second.logit_bias, 0, (size_t)s.cfg.vocab * sizeof(float), s.stream),
       "logit_bias reset");
    if (bias.empty()) return;
    const int k = (int)std::min<size_t>(bias.size(), (size_t)kMaxLogitBiasEntries);
    for (int i = 0; i < k; i++) {
        s.h_logit_bias_ids[i] = bias[i].first;
        s.h_logit_bias_vals[i] = bias[i].second;
    }
    // Safe to share the Impl-level scratch (not session-scoped): submit_locked, the only caller, is
    // always called with the engine mutex held, and the scatter launch below goes on s.stream (the
    // model's single compute stream) -- fully serialized with any other in-flight set_logit_bias.
    cu(cudaMemcpyAsync(s.d_logit_bias_ids, s.h_logit_bias_ids, k * sizeof(int),
                       cudaMemcpyHostToDevice, s.stream), "logit_bias ids");
    cu(cudaMemcpyAsync(s.d_logit_bias_vals, s.h_logit_bias_vals, k * sizeof(float),
                       cudaMemcpyHostToDevice, s.stream), "logit_bias vals");
    kernels::launch_scatter_logit_bias(it->second.logit_bias, s.d_logit_bias_ids, s.d_logit_bias_vals,
                                       k, s.cfg.vocab, s.stream);
}

// Greedy-only: this and dflash_generate() are never called from sparkinfer_server (which drives
// generation through ContinuousBatchEngine::step_job -> forward_token() directly, never through
// here -- confirmed, see inference_engine.cpp). Temperature sampling is deliberately NOT threaded
// into generate()/dflash_generate() or their CLI/bench callers (qwen3_gguf_generate,
// qwen3_gguf_dflash_bench, qwen3_gguf_dflash_check) for exactly this reason -- DFlash's verify
// step (qwen35_prefill.cpp's dflash_verify_short_run) requires exact greedy-argmax determinism
// against the draft model's own greedy proposal (see its own comments), and there is currently no
// guard here against calling this with a temperature-sampling caller. If DFlash is ever wired
// into step_job() (see inference_engine.h's TODO) or a CLI flag adds temperature/seed to these
// bench binaries, that future work must add its own guard here -- the HTTP-layer 400 in
// sparkinfer_server.cpp only protects the path that actually goes through step_job today.
std::vector<int> Qwen35Model::generate(const std::vector<int>& prompt, int max_new, ThermalGovernor* gov,
                                       double* out_ttft_s, double* out_decode_s) {
    Impl& s = *p_;
    if (s.dflash_draft) {
        const char* e = getenv("SPARKINFER_DFLASH");
        if (e && e[0] == '1') return dflash_generate(prompt, max_new, nullptr, gov);
    }
    std::vector<int> out;
    if (prompt.empty()) return out;

    const bool reuse = prompt_matches_prefix(prompt);
    const int budget = session_token_budget(prompt.size(), max_new, s.cfg.max_seq);
    uint64_t sid = 0;
    if (!reuse) {
        clear_prefix_cache();
        invalidate_decode_graph();
        sid = open_session(budget);
        if (!sid) {
            fprintf(stderr, "[qwen35] KV allocate failed (need %d tokens)\n", budget);
            return out;
        }
        activate_session(sid);
    } else {
        sid = s.active_seq_id;
        if (!s.kv->allocate(sid, budget)) {
            fprintf(stderr, "[qwen35] KV allocate failed (need %d tokens)\n", budget);
            return out;
        }
        activate_session(sid);
    }
    const int start = reuse ? s.prefix_len : 0;
    const size_t n = prompt.size();
    // AR's own decode graph gets captured on the LAST prefill call inside ingest_prompt_range
    // (the first one with sample=true) -- set the hint before that runs. See the matching
    // comment in forward_token's adaptive-split block for why AR needs this too, not just
    // DFlash: without it, AR would naturally transition mid-decode while a DFlash run over the
    // same prompt length freezes at the final tier from the start, so the two sides would use
    // different split counts for the same position and diverge from each other.
    s.final_seqlen_hint = (int)n + max_new;
    const auto t0 = std::chrono::steady_clock::now();
    int next = (start >= (int)n && reuse) ? s.prefix_next
                                            : ingest_prompt_range(prompt.data(), start, (int)n);
    const auto t1 = std::chrono::steady_clock::now();
    if (out_ttft_s) *out_ttft_s = std::chrono::duration<double>(t1 - t0).count();
    if (next < 0 || next >= s.cfg.vocab) {
        if (sid != 0) close_session(sid);
        else {
            s.kv->free(sid);
            if (reuse) release_prefix_session();
        }
        fprintf(stderr, "[qwen35] prompt prefill failed (start=%d n=%zu)\n", start, n);
        return out;
    }
    for (int i = 0; i < max_new; i++) {
        out.push_back(next);
        if (next == s.cfg.eos_id || (s.cfg.eos_id2 >= 0 && next == s.cfg.eos_id2)) break;
        next = forward_token(next, (int)prompt.size() + i, true);
        if (gov) gov->pace();
    }
    if (out_decode_s) *out_decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

    if (sid != 0) close_session(sid);
    else {
        s.kv->free(sid);
        if (reuse) release_prefix_session();
    }
    return out;
}

const void* Qwen35Model::embed_weights() const { return p_->w.embed_tokens; }
const void* Qwen35Model::lm_head_weights() const { return p_->w.lm_head; }
int Qwen35Model::lm_head_quant_type() const { return p_->w.lm_head_type; }

void Qwen35Model::set_dflash_draft(DFlashDraftModel* draft) { p_->dflash_draft = draft; }

void Qwen35Model::set_dflash_capture(bool on, const std::vector<int>& target_layer_ids, int max_rows) {
    Impl& s = *p_;
    s.dflash_capture = on;
    s.dflash_layer_ids = target_layer_ids;
    s.dflash_n_cap = (int)target_layer_ids.size();
    s.dflash_max_rows = std::max(1, max_rows);
    s.dflash_cap_row = 0;
    s.dflash_ctx_len = 0;
    invalidate_decode_graph();
    if (!on) return;
    const int H = s.cfg.hidden;
    const size_t row_elems = (size_t)s.dflash_n_cap * H;
    const size_t hidden_bytes = (size_t)s.dflash_max_rows * row_elems * sizeof(bf16);
    if (s.dflash_hidden) { cudaFree(s.dflash_hidden); s.dflash_hidden = nullptr; }
    if (s.dflash_context) { cudaFree(s.dflash_context); s.dflash_context = nullptr; }
    s.dflash_ctx_cap = s.cfg.max_seq;
    cu(cudaMalloc(&s.dflash_hidden, hidden_bytes), "dflash hidden");
    cu(cudaMalloc(&s.dflash_context, (size_t)s.dflash_ctx_cap * row_elems * sizeof(bf16)), "dflash ctx");
    if (s.cfg.hybrid && !s.spec_lin_snap) {
        const size_t ls = (size_t)s.cfg.n_layers * s.cfg.linear_v_heads *
                          s.cfg.linear_head_dim * s.cfg.linear_head_dim;
        const size_t cs = (size_t)s.cfg.n_layers * (s.cfg.linear_conv_kernel - 1) * s.linear_qkvdim;
        s.spec_lin_snap = s.alloc<float>(ls);
        s.spec_conv_snap = s.alloc<bf16>(cs);
    }
    if (const char* e = getenv("SPARKINFER_DFLASH_CAPTURE"); e && e[0] == '1')
        fprintf(stderr, "[dflash] capture on n_cap=%d max_rows=%d\n", s.dflash_n_cap, s.dflash_max_rows);
}

void Qwen35Model::set_dflash_capture_row(int row) { p_->dflash_cap_row = row; }

void Qwen35Model::dflash_stash_capture(int global_pos) {
    Impl& s = *p_;
    if (!s.dflash_hidden || !s.dflash_context || s.dflash_n_cap <= 0) return;
    if (global_pos < 0 || global_pos >= s.dflash_ctx_cap) return;
    const int H = s.cfg.hidden;
    const size_t row_elems = (size_t)s.dflash_n_cap * H;
    const bf16* src = s.dflash_hidden + (size_t)s.dflash_cap_row * row_elems;
    bf16* dst = s.dflash_context + (size_t)global_pos * row_elems;
    cu(cudaMemcpyAsync(dst, src, row_elems * sizeof(bf16), cudaMemcpyDeviceToDevice, s.stream),
       "dflash stash");
    if (global_pos + 1 > s.dflash_ctx_len) s.dflash_ctx_len = global_pos + 1;
}

const void* Qwen35Model::dflash_hidden_buffer() const { return p_->dflash_hidden; }
const void* Qwen35Model::dflash_context_buffer() const { return p_->dflash_context; }
int Qwen35Model::dflash_hidden_row_stride() const {
    return p_->dflash_n_cap * p_->cfg.hidden;
}
int Qwen35Model::dflash_context_len() const { return p_->dflash_ctx_len; }

void Qwen35Model::save_spec_snapshot() {
    Impl& s = *p_;
    const Qwen35Config& c = s.cfg;
    if (!s.spec_lin_snap || !c.hybrid) return;
    const size_t ls = (size_t)c.n_layers * c.linear_v_heads * c.linear_head_dim * c.linear_head_dim;
    const size_t cs = (size_t)c.n_layers * (c.linear_conv_kernel - 1) * s.linear_qkvdim;
    cu(cudaMemcpyAsync(s.spec_lin_snap, s.lin_state, ls * sizeof(float), cudaMemcpyDeviceToDevice, s.stream),
       "spec snap lin");
    cu(cudaMemcpyAsync(s.spec_conv_snap, s.lin_conv_state, cs * sizeof(bf16), cudaMemcpyDeviceToDevice, s.stream),
       "spec snap conv");
    cu(cudaStreamSynchronize(s.stream), "spec snap sync");
}

void Qwen35Model::restore_spec_snapshot() {
    Impl& s = *p_;
    const Qwen35Config& c = s.cfg;
    if (!s.spec_lin_snap || !c.hybrid) return;
    const size_t ls = (size_t)c.n_layers * c.linear_v_heads * c.linear_head_dim * c.linear_head_dim;
    const size_t cs = (size_t)c.n_layers * (c.linear_conv_kernel - 1) * s.linear_qkvdim;
    cu(cudaMemcpyAsync(s.lin_state, s.spec_lin_snap, ls * sizeof(float), cudaMemcpyDeviceToDevice, s.stream),
       "spec restore lin");
    cu(cudaMemcpyAsync(s.lin_conv_state, s.spec_conv_snap, cs * sizeof(bf16), cudaMemcpyDeviceToDevice, s.stream),
       "spec restore conv");
    cu(cudaStreamSynchronize(s.stream), "spec restore sync");
    invalidate_decode_graph();
}

bool Qwen35Model::verify_block(const int* token_ids, int n, int start_pos, int* out_argmax) {
    if (!token_ids || !out_argmax || n <= 0) return false;
    for (int i = 0; i < n; i++) {
        set_dflash_capture_row(i);
        out_argmax[i] = forward_token(token_ids[i], start_pos + i, true);
        if (out_argmax[i] < 0) return false;
    }
    return true;
}

void Qwen35Model::dflash_warm_verify(int n, int start_pos) {
    Impl& s = *p_;
    auto it = s.sessions.find(s.active_seq_id);
    float* lin_state = (it != s.sessions.end()) ? it->second.lin_state : s.lin_state;
    bf16* lin_conv = (it != s.sessions.end()) ? it->second.lin_conv_state : s.lin_conv_state;
    Qwen35PrefillCtx ctx{ s.cfg, s.w, s.kv, s.stream, s.stream_k, s.stream_v, s.active_seq_id,
                          lin_state, lin_conv, s.logits, s.d_out_id, s.h_out_id, s.gguf,
                          s.emb_norm_ones,
                          s.qdim, s.kvdim, s.linear_qdim, s.linear_vdim, s.linear_qkvdim,
                          s.moe_rs_gate, s.moe_rs_up, s.moe_rs_down, s.n_splits };
    // The recorded token ids and positions are irrelevant: the graph copies them from pinned host
    // buffers at replay, so only the shapes (n, and the pointer keys) have to match the real steps.
    std::vector<int> ids(n, 0);
    std::vector<int> argmax(n, 0);
    dflash_verify_short_run(ctx, ids.data(), n, start_pos, s.dflash_layer_ids.data(), s.dflash_n_cap,
                            s.dflash_hidden, argmax.data(), /*capture_only=*/true);
}

bool Qwen35Model::batched_forward(const int* token_ids, int n, int start_pos, bool /*resume_gdn*/,
                                  int* out_argmax, const void* dflash_capture_dst) {
    Impl& s = *p_;
    auto it = s.sessions.find(s.active_seq_id);
    float* lin_state = (it != s.sessions.end()) ? it->second.lin_state : s.lin_state;
    bf16* lin_conv = (it != s.sessions.end()) ? it->second.lin_conv_state : s.lin_conv_state;
    Qwen35PrefillCtx ctx{ s.cfg, s.w, s.kv, s.stream, s.stream_k, s.stream_v, s.active_seq_id,
                          lin_state, lin_conv, s.logits, s.d_out_id, s.h_out_id, s.gguf,
                          s.emb_norm_ones,
                          s.qdim, s.kvdim, s.linear_qdim, s.linear_vdim, s.linear_qkvdim,
                          s.moe_rs_gate, s.moe_rs_up, s.moe_rs_down, s.n_splits };
    const int consumed = dflash_verify_short_run(ctx, token_ids, n, start_pos,
                                                  s.dflash_layer_ids.data(), s.dflash_n_cap,
                                                  const_cast<void*>(dflash_capture_dst), out_argmax);
    return consumed > 0;
}

std::vector<int> Qwen35Model::dflash_generate(const std::vector<int>& prompt, int max_new,
                                              DFlashStats* stats, ThermalGovernor* gov) {
    Impl& s = *p_;
    std::vector<int> out;
    if (!s.dflash_draft || prompt.empty() || max_new <= 0) return out;
    DFlashDraftModel& draft = *s.dflash_draft;
    const DFlashDraftConfig& dc = draft.config();
    const int B = dc.block_size;
    const int mask_id = dc.mask_token_id;

    // The sequence length below which this path is byte-identical to main: the boundary between
    // main's behaviour and the exact long-context batched verify.
    static const int kCompactMaxSeq = []{
        const char* e = getenv("SPARKINFER_DFLASH_COMPACT_MAX_SEQ");
        return e ? atoi(e) : 384;
    }();
    // Sequence-length floor for the batched path. The acceptance EMA alone is not enough: 512-ctx
    // settles at tau 2.19, below the engage threshold, but a transient run of full blocks early in
    // the stream pushes it over and alpha 1/8 then takes ~8 steps to decay. A floor cannot be
    // spoofed by a transient, and it costs 4k nothing because 4k clears it from the first step.
    static const int kEngageMinSeq = []{
        const char* e = getenv("SPARKINFER_DFLASH_ENGAGE_MINSEQ");
        return e ? atoi(e) : 1024;
    }();

    // Speculating is a strict LOSS wherever the batched verify cannot engage.
    //
    // The token loop verifies with early exit, so a step that keeps `keep` tokens runs exactly
    // `keep` target forwards -- one per emitted token, the same count autoregressive decode runs --
    // and then pays a draft block on top. It never saves a target forward; only the batched verify
    // does, by collapsing them into one N-row pass. So in the band where the batched path is off,
    // DFlash is AR plus the draft, and the draft is pure overhead. Measured on RTX 5090 with the
    // batched path forced off, against this binary's own AR: 128-ctx 434.7 vs 516, 4k 385.9 vs 488,
    // 512-ctx 401 vs 511 -- slower at every one, by very close to draft_ms / tau per token.
    //
    // Whether the batched path can ever engage is decided by the prompt and max_new alone, before a
    // single token is generated: `start + remaining` is invariant across the loop, so a generation
    // that starts above kCompactMaxSeq (short-context compact path off) and ends below
    // kEngageMinSeq (long-context path off) spends every step on the token loop. Take the AR path
    // for that band outright -- decided here, before prefill, so nothing about the DFlash setup
    // (hidden-state capture, the draft's KV, the verify graph) is ever paid for.

    // 0=off, 1=force, 2=adaptive (default). #716 turned this off because the row-batched
    // compact-verify graph (dflash_verify_short_run) showed a numerical discrepancy from AR
    // "present in every row of a batch, including rows that still happen to land on the correct
    // token" (#712), which on some steps flipped the argmax and broke DFlash's lossless
    // guarantee. That gap is closed: the batched path was scoring its logits against a
    // per-row symmetric int8 REQUANTIZATION of the Q4_K LM head, while AR scores against the
    // native Q4_K weights -- a systematic per-row error, in exactly the "every row" shape
    // reported. dflash_verify_short_run now uses the native head (see the comment there), so
    // the batched path and AR read the same weights and greedy DFlash reproduces AR exactly.
    static const int compact_mode = []{
        const char* e = getenv("SPARKINFER_DFLASH_COMPACT_VERIFY");
        return e ? atoi(e) : 2;
    }();
    // Full-block accepts arrive in runs (the target and draft agree over a whole predictable
    // region), so one is already evidence the next step will accept a full block.
    //
    // Engage the row-batched verify only while the draft is actually landing full blocks. The
    // batched pass replaces the step's sequential target forwards with one N-row pass, so it pays
    // in proportion to how many forwards it collapses -- that is the ACCEPTANCE RATE, not the
    // context length. Measured on RTX 5090 over held-out prompts (mean accept tau, compact off ->
    // on): tau 5.32 -> 448.7 to 911.1 tok/s, but tau 2.40 -> 433.2 to 437.2, tau 2.00 -> 425.4 to
    // 396.1, and tau 1.87 -> 439.1 to 350.8. Below roughly half the block depth the batched pass
    // forwards the whole block to keep two tokens and the token loop's early exit is simply
    // cheaper, so engaging there costs throughput on exactly the prompts the draft handles worst.
    //
    // The previous score LATCHED: any partial accept of >= kStayKeep held it at the engage
    // threshold, so once armed it stayed armed straight through those low-acceptance stretches.
    // Require a RUN of full-block accepts to arm, and decay on every non-full block so it backs
    // off as soon as the draft stops landing blocks.
    // How many full-block accepts in a row arm the batched path. A run is evidence that the draft
    // is tracking the target, and the decay below still disarms on two non-full blocks, so the run
    // length only sets how long that evidence takes to accumulate -- and the wait is paid on every
    // prompt the draft handles well. On the held-out short prompt (tau 5.33, where batching is
    // worth +11%) a run of 3 spends the first few steps of a ~24-step generation on the token loop:
    // 806.1 tok/s at 3 against 878.7 at 1. A prompt the draft handles badly is unaffected either
    // way, because it never lands the full block that arms it at all -- measured at tau 2.08,
    // 437.2 -> 434.4, while blindly forcing the batched path there costs 15%.
    static const int kBlockScore = []{
        const char* e = getenv("SPARKINFER_DFLASH_BLOCK_SCORE");
        int v = e ? atoi(e) : 1;
        return v < 1 ? 1 : v;
    }();

    const int n_prompt = (int)prompt.size();
    const bool spec_never_pays = (n_prompt + B) > kCompactMaxSeq &&
                                 (n_prompt + max_new + B) < kEngageMinSeq &&
                                 compact_mode != 1;
    if (spec_never_pays) {
        // Plain autoregressive decode, set up the way generate() sets it up: no hidden-state
        // capture, no draft KV, no verify graph. Deciding before prefill rather than falling back
        // mid-stream is what makes this reach AR's own throughput instead of approaching it -- a
        // fallback still pays for the DFlash prefill it already ran.
        set_dflash_capture(false, {}, 0);
        const int ar_budget = session_token_budget(prompt.size(), max_new + B, s.cfg.max_seq);
        clear_prefix_cache();
        invalidate_decode_graph();
        const uint64_t ar_sid = open_session(ar_budget);
        if (!ar_sid) {
            fprintf(stderr, "[dflash] KV allocate failed (need %d)\n", ar_budget);
            return out;
        }
        activate_session(ar_sid);
        s.final_seqlen_hint = n_prompt + max_new;
        const auto ar_t0 = std::chrono::steady_clock::now();
        int ar_next = ingest_prompt_range(prompt.data(), 0, n_prompt);
        const auto ar_t1 = std::chrono::steady_clock::now();
        if (ar_next < 0 || ar_next >= s.cfg.vocab) {
            close_session(ar_sid);
            fprintf(stderr, "[dflash] prompt prefill failed (n=%d)\n", n_prompt);
            return out;
        }
        for (int i = 0; i < max_new; i++) {
            out.push_back(ar_next);
            if (ar_next == s.cfg.eos_id) break;
            ar_next = forward_token(ar_next, n_prompt + i, true);
            if (ar_next < 0) break;
            if (gov) gov->pace();
        }
        const auto ar_t2 = std::chrono::steady_clock::now();
        if (stats) {
            stats->steps = (int)out.size();
            stats->mean_accept = 1.0;   // one target forward per token, by definition
            stats->ttft_s = std::chrono::duration<double>(ar_t1 - ar_t0).count();
            stats->decode_s = std::chrono::duration<double>(ar_t2 - ar_t1).count();
        }
        close_session(ar_sid);
        return out;
    }

    // Head for the draft. The dual-head path keeps a native Q6_K copy so the draft's multi-row
    // MMVQ has something to chew on, but that kernel runs near HBM peak, so its runtime is just
    // its weight bytes -- and the target's own Q4_K copy is ~280 MB against the Q6_K's ~417 MB.
    // With a multi-row Q4_K MMVQ the draft prefers the smaller one. SPARKINFER_DFLASH_HEAD_Q4=0
    // restores the Q6_K copy (A/B).
    static const int head_q4 = []{ const char* e = getenv("SPARKINFER_DFLASH_HEAD_Q4");
                                   return (e && e[0] == '0') ? 0 : 1; }();
    const bool use_q4_head = head_q4 && lm_head_quant_type() == 12 && lm_head_weights();
    const void* draft_head = use_q4_head ? lm_head_weights()
                           : (s.dflash_lm_head ? s.dflash_lm_head : lm_head_weights());
    const int draft_head_type = use_q4_head ? lm_head_quant_type()
                              : (s.dflash_lm_head ? s.dflash_lm_head_type : lm_head_quant_type());
    draft.set_shared_weights(embed_weights(), draft_head, draft_head_type,
                             s.cfg.vocab, s.cfg.hidden);
    // Build the draft's quantized weights here, before prefill and well before the decode clock,
    // so this generation pays exactly what it did when load() built them eagerly. The point of
    // deferring them is the branch above: a generation that takes the autoregressive path returns
    // before this line and never materialises them at all.
    draft.ensure_quant();
    set_dflash_capture(true, dc.target_layer_ids, B);

    const int budget = session_token_budget(prompt.size(), max_new + B, s.cfg.max_seq);
    clear_prefix_cache();
    invalidate_decode_graph();
    uint64_t sid = open_session(budget);
    if (!sid) {
        fprintf(stderr, "[dflash] KV allocate failed (need %d)\n", budget);
        set_dflash_capture(false, {}, 0);
        return out;
    }
    activate_session(sid);

    const int n = (int)prompt.size();
    // The decode graph freezes n_splits on the LAST prefill call below (the first one with
    // sample=true), not at the start of the decode loop further down -- set the hint before
    // prefill runs so that freeze already bakes in the tier the whole generation will need.
    s.final_seqlen_hint = n + max_new;
    auto t0 = std::chrono::steady_clock::now();
    int next = -1;
    for (int i = 0; i < n; i++) {
        set_dflash_capture_row(0);
        const bool sample = (i + 1 == n);
        int r = forward_token(prompt[i], i, sample);
        dflash_stash_capture(i);
        if (sample) next = r;
    }
    auto t1 = std::chrono::steady_clock::now();
    if (next < 0 || next >= s.cfg.vocab) {
        close_session(sid);
        set_dflash_capture(false, {}, 0);
        return out;
    }

    draft.reset();
    int start = n;
    double accept_sum = 0;
    int steps = 0;
    const void* target_hidden = dflash_context_buffer();
    int th_len = n;

    std::vector<int> block(B), posterior(B), draft_ids(B);
    // Proposal depth (also sets the draft's active diffusion width, depth+1). 5 is the measured
    // optimum at short context: accept length rises only 5.33 -> 5.95 -> 6.43 going to depth 6 and
    // 7, while each extra verify row costs a flat ~0.74 ms, so depth 6 already loses there.
    //
    // That trade inverts once the KV is long. An extra verify row costs roughly one more set of
    // routed experts (each row picks its own 8 of 256, and they barely overlap), which is a fixed
    // price per row; what it buys is fewer decode steps, and every step re-reads the whole KV
    // cache. Short context: the KV read is negligible, the expert reads dominate, depth 5 wins.
    // Long context: the KV read dominates, and paying flat expert cost to remove whole steps wins.
    //
    // The draft is also running out of room to be asked: at 32k the accept length is 5.61 out of a
    // maximum of 6, so ~93% of steps are truncated by the request size rather than rejected by the
    // target. Depth 7 is the largest that keeps every batched path -- the compact verify itself
    // (dflash_verify_short_run bails above 8 rows), the row-batched Q4_K/Q6_K/Q8_0 GEMVs (MMAX=8 is
    // the widest instantiation) and the batched MoE all stop at 8 rows -- and it leaves the draft
    // untouched, because a width of depth+1 rounds up to the same 8-wide block either way.
    //
    // Measured, tok/s at depth 5 -> 7 (2 reps each, RTX 5090 @2550):
    //     128    800.2 -> 749.1  (-6.4%)      8192   432.5 -> 419.7  (-3.0%)
    //     4096   490.0 -> 429.3  (-12.4%)    12288   630.2 -> 677.7  (+7.5%)
    //     16384  559.8 -> 567.4  (+1.4%)     32768   497.7 -> 520.2  (+4.5%)
    // The crossover sits between 8k and 12k, so the floor is 12288 and everything below it keeps
    // the depth it has today. Acceptance is a property of the text as much as of the length (the
    // 8k prompt accepts 3.66, less than the 4k one's 3.91), so the floor is deliberately past the
    // last length measured to lose rather than at it.
    static const int kDeepMinSeq = []{
        const char* e = getenv("SPARKINFER_DFLASH_DEEP_MIN_SEQ");
        int v = e ? atoi(e) : 12288;
        return v < 1 ? 1 : v;
    }();
    // Explicit override wins; 0/unset selects by length. Keep in sync with dflash_draft.cpp, which
    // reads the same variable for its own default and is handed this value per block.
    static const int kProposalDepthEnv = []{
        const char* e = getenv("SPARKINFER_DFLASH_PROPOSALS");
        int v = e ? atoi(e) : 0;
        return v < 0 ? 0 : (v > 15 ? 15 : v);
    }();
    const int kProposalDepth = kProposalDepthEnv > 0 ? kProposalDepthEnv
                             : ((n + max_new) >= kDeepMinSeq ? 7 : 5);
    // ...but "full block accepted" is a proxy, and a lossy one. What actually decides whether the
    // batched pass pays is how many sequential target forwards it collapses -- that is the MEAN
    // accepted length, and it has no reason to sit at exactly B. Measured on RTX 5090 at the three
    // scored contexts (token loop -> batched):
    //
    //     ctx    tau    token loop   batched      verdict
    //     512    2.19   422.5        290.3        -31%  batched must stay OFF
    //     4096   3.91   396.6        639.5        +61%  batched must stay ON
    //     128    5.33   457.3        868.2        +90%  batched must stay ON
    //
    // Break-even is between tau 2.2 and 3.9. The full-block rule only climbs when keep == B, so a
    // 4k stream sitting at a perfectly profitable tau of 3.9 armed the batched path roughly 40% of
    // steps and banked +21% of the available +61%. Gating on a running mean of keep engages on the
    // condition that actually makes batching profitable, and still leaves 512 on the token loop.
    // The sequence length below which this PR is byte-identical to main. #720 shipped this as a
    // hard bound that kept the batched verify away from long context entirely (the #712 gap); it
    // is now the boundary between "main's behaviour, unchanged" and "the new exact long-context
    // path", so nothing already validated at short context moves.
    // Compared against keep_ema8 in the SCALED domain (kEngageKeep * 8), not by shifting the EMA
    // down first. Shifting floors it, which collapses the very gap the gate exists to resolve:
    // 512-ctx sits at tau 2.19 (ema8 ~17.5) and 4k at tau 3.91 (ema8 ~31.3), and both floor to the
    // same value. Scaled, the threshold lands cleanly between them at 24.
    // Sequence-length floor for the batched path. The EMA alone is not enough: 512-ctx settles at
    // tau 2.19 (ema8 ~17.5, below the threshold) but a transient run of full blocks early in the
    // stream pushes it over, and alpha 1/8 then takes ~8 steps to decay -- measured at 2.8-4.3%
    // lost to running the batched path where it is 31% SLOWER. A floor cannot be spoofed by a
    // transient, and it costs 4k nothing because 4k clears it from the first step.
    static const int kEngageKeep = []{
        const char* e = getenv("SPARKINFER_DFLASH_ENGAGE_KEEP");
        int v = e ? atoi(e) : 3;
        return v < 1 ? 1 : v;
    }();
    // The acceptance that makes batching pay is not a constant -- it falls as context grows. One
    // batched pass replaces `keep` sequential target forwards, and each of those forwards re-reads
    // the whole KV cache, so the token loop gets steadily more expensive with sequence length while
    // the N-row pass barely moves. A threshold calibrated where the token loop is cheap therefore
    // sits too high once the KV is long, and the batched path idles on steps where it was already
    // the better choice. Measured at 4k (tau 3.91): 3 -> 529.2 tok/s, 2 -> 542.1.
    static const int kEngageKeepLong = []{
        const char* e = getenv("SPARKINFER_DFLASH_ENGAGE_KEEP_LONG");
        int v = e ? atoi(e) : 2;
        return v < 1 ? 1 : v;
    }();
    int compact_score = 0;
    int keep_ema8 = 0;   // EMA of keep, alpha = 1/8, held x8 so the decode path needs no float
    int step_no = 0;
    bf16* th_scratch = nullptr;
    const int row_stride = dflash_hidden_row_stride();
    if (cudaMalloc(&th_scratch, (size_t)B * row_stride * sizeof(bf16)) != cudaSuccess) {
        close_session(sid);
        set_dflash_capture(false, {}, 0);
        draft.reset();
        return out;
    }
    // Build the verify replay graph before the decode clock starts. Capture records kernels
    // rather than running them, so this changes no state -- it just stops decode step 2 from
    // paying for graph construction.
    dflash_warm_verify(kProposalDepth + 1, start);
    auto t_decode0 = std::chrono::steady_clock::now();
    while ((int)out.size() < max_new) {
        // The context bound this used to carry (SPARKINFER_DFLASH_COMPACT_MAX_SEQ, default 384)
        // existed only to keep the batched path away from contexts where it diverged from AR --
        // the #712 gap. That gap was a real defect, not a property of batching: the batched GDN
        // conv summed its taps in a different order than the single-token decode conv, so every
        // k/v differed from AR's by ~1 ulp, and the GDN recurrence carried that difference across
        // decode steps until it flipped an argmax. With the two convs accumulating in the same
        // order the paths are bit-identical at every context, so the bound has nothing left to do
        // and the gate is now purely the acceptance test above -- engage where batching pays,
        // stay on the token loop where it does not, at any sequence length.
        // Below kCompactMaxSeq this is byte-identical to main: same acceptance-run rule, same
        // batched MoE. The change is purely additive above that bound, which main never reached.
        const bool short_ctx = (start + B) <= kCompactMaxSeq;
        // keep_ema8 ramps from zero at alpha = 1/8, so it needs 8-10 steps to reach the engage
        // threshold even on a stream that clears it comfortably. Measured at 4k (tau 3.91, ema8
        // settles at ~31 against a threshold of 24): the batched path ran on 20 of 33 steps, and
        // the 13 it missed were the warmup, not a genuine low-acceptance stretch -- worth 6.3% of
        // decode on a 128-token generation.
        //
        // The ramp exists so a transient run of full blocks cannot arm the batched path at 512-ctx
        // (tau 2.19), where it is 31% slower. kEngageMinSeq already excludes that case outright, so
        // seeding is scoped to sequences past the floor: start armed there, and let the existing
        // decay hand the stream back to the token loop within a couple of steps if the draft turns
        // out not to be landing blocks. Below the floor the ramp is untouched.
        if (step_no == 0 && (start + B) >= kEngageMinSeq) keep_ema8 = kEngageKeepLong * 8;
        const bool compact_verify = compact_mode == 1 ||
            (compact_mode != 0 && (short_ctx ? compact_score >= kBlockScore
                                             : ((start + B) >= kEngageMinSeq &&
                                                keep_ema8 >= kEngageKeepLong * 8)));
        block[0] = next;
        for (int i = 1; i < B; i++) block[i] = mask_id;

        // The target overwrites dflash_hidden while capturing verify row zero. Preserve the
        // accepted suffix before running that target forward concurrently with the independent
        // draft stream. The initial full-context buffer is separate and needs no copy.
        // Only the token-loop path runs verify row zero concurrently with the draft and so needs
        // the accepted suffix preserved. The compact path runs the draft to completion first, so
        // nothing can overwrite dflash_hidden underneath it and the copy plus its full-device
        // synchronize are pure overhead.
        const void* draft_hidden = target_hidden;
        if (!compact_verify && target_hidden == s.dflash_hidden) {
            cu(cudaMemcpyAsync(th_scratch, target_hidden,
                               (size_t)th_len * row_stride * sizeof(bf16),
                               cudaMemcpyDeviceToDevice, s.stream), "dflash overlap stash");
            cu(cudaStreamSynchronize(s.stream), "dflash overlap stash sync");
            draft_hidden = th_scratch;
        }

        // Enqueue verify token 0 first (one graph launch, ~10 us of host time), then issue the
        // draft block on its own stream. Ordering matters: the target must be in flight before
        // the draft's launches start, or the draft queues ahead of it. A deferred collect also
        // avoids spawning and joining a std::thread on every decode step.
        int p0 = -1;
        if (!compact_verify) {
            set_dflash_capture_row(0);
            s.defer_decode_sync = true;
            p0 = forward_token(block[0], start, true);
            s.defer_decode_sync = false;
        }
        const bool draft_ok = draft.forward_block(
            draft_hidden, th_len, block.data(), start, draft_ids.data(), nullptr, kProposalDepth);
        if (!compact_verify && p0 == kDFlashDeferred) {
            cu(cudaStreamSynchronize(s.stream), "verify0 sync");
            s.decode_pending = false;
            p0 = *s.h_out_id;
        }
        if (!draft_ok) {
            fprintf(stderr, "[dflash] draft forward failed at start=%d\n", start);
            break;
        }
        for (int i = 1; i <= kProposalDepth; i++) block[i] = draft_ids[i];

        // Incremental verify with early-exit: forward only the accepted prefix, stopping at the
        // first rejected proposal. forward_token advances GDN state + KV per token, so after
        // block[0..keep-1] the recurrent state and KV sit exactly at start+keep -- no snapshot /
        // restore / KV-truncate / replay needed (greedy speculative decoding is exact). Rejected
        // proposals (block[keep..B-1]) are never forwarded, saving ~B-keep target forwards/step.
        int accept = 0, keep = 1;
        bool vfail = false;
        if (compact_verify) {
            const int vn = kProposalDepth + 1;
            vfail = !batched_forward(block.data(), vn, start, false, posterior.data(), s.dflash_hidden);
            if (!vfail) {
                while (accept < kProposalDepth && block[accept + 1] == posterior[accept]) ++accept;
                keep = accept + 1;
            }
        } else {
            vfail = p0 < 0;
            posterior[0] = p0;
        }
        if (!compact_verify && !vfail && block[1] == p0) {
            for (int i = 1; i <= kProposalDepth; i++) {
                set_dflash_capture_row(i);
                const int p = forward_token(block[i], start + i, true);
                if (p < 0) { vfail = true; break; }
                posterior[i] = p;
                accept = i;
                keep = i + 1;
                if (i < kProposalDepth && block[i + 1] != p) break;
            }
        }
        if (vfail) { fprintf(stderr, "[dflash] verify failed at start=%d\n", start); break; }
        // Climb on a full-block accept, decay on anything less. The old rule latched the score at
        // the engage threshold for any partial accept of >= 2 tokens, which kept the batched path
        // armed through low-acceptance stretches -- precisely where it costs throughput. Decaying
        // instead means a prompt the draft handles badly drops back to the token loop within a
        // couple of steps and stays there, while a prompt it handles well re-arms just as quickly.
        // Ramp from zero rather than seeding on the first step. Seeding armed the batched path off
        // a single lucky block, which at 512-ctx (tau 2.19, where batching is 31% SLOWER) cost 5.6%
        // before the EMA had enough evidence to back off. Ramping costs ~3 token-loop steps at
        // short context and keeps 512 on the token loop throughout.
        keep_ema8 = keep_ema8 - (keep_ema8 >> 3) + keep;
        ++step_no;
        compact_score = keep == kProposalDepth + 1
                      ? std::min(compact_score + 1, kBlockScore + 1)
                      : std::max(compact_score - 1, 0);
        // forward_token() synchronizes after sampling, so the accepted capture rows are already
        // stable. The draft consumes only this newly accepted suffix; its KV cache retains all
        // earlier context. Hand the capture buffer over directly instead of copying it to a second
        // scratch allocation, stashing another unused full-context copy, and synchronizing again.

        bool stop = false;
        for (int i = 0; i < keep && (int)out.size() < max_new; i++) {
            out.push_back(block[i]);
            if (block[i] == s.cfg.eos_id) { stop = true; break; }
        }
        if (stop) break;
        // Bonus token becomes the next block seed (emitted on the following iteration).
        next = posterior[accept];
        if (next == s.cfg.eos_id) {
            if ((int)out.size() < max_new) out.push_back(next);
            break;
        }

        start += keep;
        accept_sum += (double)keep;
        steps++;
        target_hidden = s.dflash_hidden;
        th_len = keep;
        if (gov) gov->pace();
    }
    auto t_end = std::chrono::steady_clock::now();
    if (stats) {
        stats->steps = steps;
        stats->mean_accept = steps > 0 ? accept_sum / steps : 0;
        stats->ttft_s = std::chrono::duration<double>(t1 - t0).count();
        stats->decode_s = std::chrono::duration<double>(t_end - t_decode0).count();
    }
    close_session(sid);
    if (th_scratch) cudaFree(th_scratch);
    set_dflash_capture(false, {}, 0);
    draft.reset();
    return out;
}

// ----- weight loading from a sparkinfer weight directory -----
namespace {
void* load_bin(const std::string& path, std::vector<void*>& owned) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "[qwen35] missing weight: %s\n", path.c_str()); return nullptr; }
    std::streamsize n = f.tellg(); f.seekg(0);
    std::vector<char> host(n);
    f.read(host.data(), n);
    void* d = nullptr;
    if (cudaMalloc(&d, n) != cudaSuccess) return nullptr;
    cudaMemcpy(d, host.data(), n, cudaMemcpyHostToDevice);
    owned.push_back(d);
    return d;
}
}

bool Qwen35Model::load_weights(const std::string& dir) {
    Impl& s = *p_;
    auto L = [&](const std::string& n) { return load_bin(dir + "/" + n + ".bin", s.owned); };
    s.w.embed_tokens = L("embed_tokens");
    s.w.final_norm   = L("final_norm");
    s.w.lm_head      = L("lm_head");
    if (!s.w.embed_tokens || !s.w.final_norm || !s.w.lm_head) return false;
    s.w.layers.resize(s.cfg.n_layers);
    for (int i = 0; i < s.cfg.n_layers; i++) {
        std::string pfx = "layer_" + std::to_string(i) + ".";
        Qwen35LayerWeights& w = s.w.layers[i];
        w.input_norm     = L(pfx + "input_norm");
        w.wq = L(pfx + "wq"); w.wk = L(pfx + "wk"); w.wv = L(pfx + "wv"); w.wo = L(pfx + "wo");
        w.q_norm = L(pfx + "q_norm"); w.k_norm = L(pfx + "k_norm");
        w.post_attn_norm = L(pfx + "post_attn_norm");
        w.router_w = L(pfx + "router_w");
        w.gate = L(pfx + "gate"); w.up = L(pfx + "up"); w.down = L(pfx + "down");
        if (s.cfg.n_shared > 0) {
            w.shared_gate = L(pfx + "shared_gate"); w.shared_up = L(pfx + "shared_up"); w.shared_down = L(pfx + "shared_down");
        }
        if (!w.wq || !w.gate || !w.router_w) return false;
    }
    return true;
}

// ----- native GGUF load: dense -> bf16 (dequant + transpose), experts kept quantized -----
bool Qwen35Model::load_gguf(const std::string& path) {
    Impl& s = *p_;
    GGUF g;
    if (!g.open(path)) return false;
    const bool dense_file = g.tensor("blk.0.ffn_gate.weight") != nullptr &&
                            g.tensor("blk.0.ffn_gate_exps.weight") == nullptr;
    const bool hybrid_file = is_qwen35_or_qwen36_hybrid_moe(g) || dense_file;
    if (hybrid_file && !s.cfg.hybrid && !s.cfg.dense_ffn) {
        fprintf(stderr,
                "[qwen35] Qwen3.5/Qwen3.6 hybrid GGUF requires constructing "
                "Qwen35Model with cfg.hybrid=true and the GGUF metadata-derived "
                "head dimensions before load_gguf(), so scratch buffers and KV "
                "cache are sized correctly.\n");
        return false;
    }
    // Qwythos/Qwen3.6 hybrid-file backfill: assumes any dense-FFN or hybrid-detected GGUF
    // wants the Gated-DeltaNet SSM interleave (full_attn_interval defaulted to 4). Muse
    // Glimmer is also a dense-FFN GGUF (dense_file=true) but has NO linear/SSM layers at
    // all -- museglimmer_config_from_gguf already set every one of these fields correctly
    // (full_attn_interval=0 deliberately, to keep is_linear_layer() false for every layer),
    // so skip this backfill for it rather than let full_attn_interval<=0 get overwritten to
    // 4 and misroute layer 0 into looking for attn_qkv.weight/ssm_* tensors that don't exist.
    if ((hybrid_file || s.cfg.dense_ffn) && !s.cfg.muse_glimmer) {
        s.cfg.hybrid = true;
        if (dense_file) s.cfg.dense_ffn = true;
        if (s.cfg.full_attn_interval <= 0) s.cfg.full_attn_interval = 4;
        if (s.cfg.rope_dim <= 0 && s.cfg.head_dim == 256) s.cfg.rope_dim = 64;
        if (s.cfg.linear_q_heads <= 0) s.cfg.linear_q_heads = 16;
        if (s.cfg.linear_v_heads <= 0) s.cfg.linear_v_heads = 32;
        if (s.cfg.linear_head_dim <= 0) s.cfg.linear_head_dim = 128;
        if (s.cfg.linear_conv_kernel <= 0) s.cfg.linear_conv_kernel = 4;
    }
    const bool dense_ffn = g.tensor("blk.0.ffn_gate.weight") != nullptr &&
                           g.tensor("blk.0.ffn_gate_exps.weight") == nullptr;
    if (dense_ffn) {
        s.cfg.dense_ffn = true;
        s.cfg.n_experts = 1;
        s.cfg.top_k = 1;
        s.cfg.n_shared = 0;
        if (s.cfg.moe_ffn <= 0) {
            s.cfg.moe_ffn = (int)qwen_moe_meta_int(g, "feed_forward_length", 0);
            if (s.cfg.moe_ffn <= 0) {
                if (const GGUFTensor* gate = g.tensor("blk.0.ffn_gate.weight"))
                    if (gate->n_dims >= 2) s.cfg.moe_ffn = (int)gate->dims[1];
            }
        }
    }
    const Qwen35Config& c = s.cfg;
    const int H = c.hidden;
    s.gguf = true;   // dense weights kept native [out,in]; forward uses GEMV

    // Shared-expert tensors are optional in GGUF (Qwen3-30B-A3B has none). The
    // default config sets n_shared=1, so clamp it to what the file actually
    // contains before forward_token can launch a null-weight FFN.
    const bool gguf_has_shared =
        g.tensor("blk.0.ffn_gate_shexp.weight") != nullptr;
    if (hybrid_file && gguf_has_shared && s.cfg.n_shared == 0) s.cfg.n_shared = 1;
    if (c.n_shared > 0 && !gguf_has_shared) {
        fprintf(stderr,
                "[gguf] no shared-expert tensors; forcing n_shared=0 "
                "(safe for models without a shared FFN)\n");
        s.cfg.n_shared = 0;
    }
    // Shared-expert GEMV scratch [moe_ffn]. Allocated only on the GGUF path (native
    // [out,in] shared weights); the set_weights path keeps the moe_expert_ffn kernel.
    if (s.cfg.n_shared > 0 && !s.sh_gate) {
        s.sh_gate = s.alloc<bf16>(s.cfg.moe_ffn);
        s.sh_up   = s.alloc<bf16>(s.cfg.moe_ffn);
        s.sh_h    = s.alloc<bf16>(s.cfg.moe_ffn);
    }
    // upload raw quantized blocks, keep on device (for experts)
    auto dev_quant = [&](const std::string& name, int& qtype) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        if (!t) { fprintf(stderr, "[gguf] missing %s\n", name.c_str()); return nullptr; }
        if (!ggml_dequant_supported(t->ggml_type)) {
            fprintf(stderr, "[gguf] unsupported ggml type %d for %s\n", t->ggml_type, name.c_str());
            return nullptr;
        }
        qtype = t->ggml_type;
        void* d = nullptr;
        if (cudaMalloc(&d, t->n_bytes) != cudaSuccess) return nullptr;
        cudaMemcpy(d, t->data, t->n_bytes, cudaMemcpyHostToDevice);
        s.owned.push_back(d);
        return d;
    };
    // Optional Q6_K -> Q4_K requant: pay a load-time dequant+fit so decode reads
    // 4.5 instead of 6.5 bits/weight. The source Q6_K upload is freed after the
    // requant; qtype flips to 12 on success. Attention tensors use the Lloyd-max fit
    // (PR #353); FFN down keeps the affine fitter.
    auto is_attn_requant_name = [](const std::string& name) {
        return name.find(".attn_qkv.weight") != std::string::npos ||
               name.find(".attn_q.weight") != std::string::npos ||
               name.find(".attn_k.weight") != std::string::npos ||
               name.find(".attn_v.weight") != std::string::npos ||
               name.find(".attn_output.weight") != std::string::npos;
    };
    // allow_q5k is opt-in per call site, NOT a widening of the default set: dev_quant_down() also
    // routes through here with requant on by default, so accepting Q5_K unconditionally would
    // silently requantize the dense-FFN down tensor of any model that ships one.
    auto dev_quant_requant_q4k = [&](const std::string& name, int& qtype, bool req,
                                     bool allow_q5k = false) -> const void* {
        const void* q6 = dev_quant(name, qtype);
        const bool src_ok = (qtype == 14 || qtype == 8 || (allow_q5k && qtype == 13));
        if (!req || !src_ok || !q6) return q6;
        const int src_type = qtype;            // 14 (Q6_K), 8 (Q8_0) or 13 (Q5_K) -> Q4_K
        const GGUFTensor* t = g.tensor(name);
        const long nv = t->n_values;
        if (nv % 256 != 0) return q6;
        void* deq = nullptr;
        if (cudaMalloc(&deq, (size_t)nv * 2) != cudaSuccess) return q6;
        kernels::launch_gguf_dequant(src_type, q6, deq, nv, s.stream);
        void* q4 = nullptr;
        if (cudaMalloc(&q4, (size_t)(nv / 256) * 144) != cudaSuccess) { cudaFree(deq); return q6; }
        static int attn_lloyd = -1;
        if (attn_lloyd < 0) {
            const char* e = getenv("SPARKINFER_ATTN_REQUANT_LLOYD");
            attn_lloyd = (e && e[0] == '0') ? 0 : 1;
        }
        if (src_type == 8)
            kernels::launch_proj_requant_q4k_lloyd(deq, q4, nv, s.stream);
        else if (is_attn_requant_name(name) && attn_lloyd)
            kernels::launch_proj_requant_q4k_lloyd(deq, q4, nv, s.stream);
        else
            kernels::launch_ffn_down_requant_q4k(deq, q4, nv, s.stream);
        cudaStreamSynchronize(s.stream);
        cudaFree(deq);
        if (!s.owned.empty() && s.owned.back() == q6) { s.owned.pop_back(); cudaFree((void*)q6); }
        s.owned.push_back(q4);
        qtype = 12;
        return q4;
    };
    // Dense-FFN down: Q6_K in GGUF is requantized to Q4_K at load by default (~5% decode on
    // Qwythos). Set SPARKINFER_DOWN_REQUANT_Q4K=0 to keep native Q6_K reads.
    auto dev_quant_down = [&](const std::string& name, int& qtype) -> const void* {
        static int req = -1;
        if (req < 0) { const char* e = getenv("SPARKINFER_DOWN_REQUANT_Q4K"); req = (e && e[0] == '0') ? 0 : 1; }
        return dev_quant_requant_q4k(name, qtype, req != 0);
    };
    // Requantize a weight matrix to Q3_A (3.5 bits/weight -- Q4_K's asymmetric per-32 scale and
    // min, 3-bit quant plane, 112 B per 256 weights) at load, then decode it on-read through
    // si_vec_dot_q3_A, so the matrix is read 22.2% smaller on every decode token. Sources
    // Q4_K/Q5_K/Q6_K straight to Q3_A (one dequant, one fit). Falls back to the untouched source
    // on any failure.
    auto dev_quant_q3a = [&](const std::string& name, int& qtype,
                                 const void** prefill_q, int* prefill_qtype) -> const void* {
        const void* src = dev_quant(name, qtype);
        if (!src) return src;
        if (qtype != 12 && qtype != 13 && qtype != 14) return src;   // Q4_K / Q5_K / Q6_K
        const GGUFTensor* t = g.tensor(name);
        const long nv = t->n_values;
        if (nv % 256 != 0) return src;
        void* deq = nullptr;
        if (cudaMalloc(&deq, (size_t)nv * 2) != cudaSuccess) return src;
        kernels::launch_gguf_dequant(qtype, src, deq, nv, s.stream);
        void* q3 = nullptr;
        if (cudaMalloc(&q3, (size_t)(nv / 256) * 112) != cudaSuccess) { cudaFree(deq); return src; }
        kernels::launch_ffn_requant_q3a(deq, q3, nv, s.stream);
        cudaStreamSynchronize(s.stream);
        cudaFree(deq);
        // Keep the native source resident for batched prefill. Its established Q4_K/Q5_K/Q6_K
        // dequantizer is faster there; decode reads only the compact Q3_A copy.
        if (prefill_q) *prefill_q = src;
        if (prefill_qtype) *prefill_qtype = qtype;
        s.owned.push_back(q3);
        qtype = kernels::SI_QTYPE_Q3A;
        return q3;
    };
    // dense weight -> bf16 (optionally transpose [out,in] -> [in,out])
    auto dense = [&](const std::string& name, bool transpose) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        if (!t) { fprintf(stderr, "[gguf] missing %s\n", name.c_str()); return nullptr; }
        if (!ggml_dequant_supported(t->ggml_type)) {
            fprintf(stderr, "[gguf] unsupported ggml type %d for %s\n", t->ggml_type, name.c_str());
            return nullptr;
        }
        void* dq = nullptr; cudaMalloc(&dq, t->n_bytes);
        cudaMemcpy(dq, t->data, t->n_bytes, cudaMemcpyHostToDevice);
        void* tmp = nullptr; cudaMalloc(&tmp, (size_t)t->n_values * 2);
        kernels::launch_gguf_dequant(t->ggml_type, dq, tmp, t->n_values, s.stream);
        const void* result;
        if (transpose) {
            const int in = (int)t->dims[0], out = (int)t->dims[1];   // ggml ne0=in, ne1=out
            void* dst = nullptr; cudaMalloc(&dst, (size_t)t->n_values * 2); s.owned.push_back(dst);
            kernels::launch_transpose_bf16(tmp, dst, out, in, s.stream);   // [out,in]->[in,out]
            cudaStreamSynchronize(s.stream); cudaFree(tmp); cudaFree(dq);
            result = dst;
        } else {
            s.owned.push_back(tmp);
            cudaStreamSynchronize(s.stream); cudaFree(dq);
            result = tmp;
        }
        return result;
    };

    // Keep attention/lm_head weights quantized in VRAM and decode them on-read
    // (Q4_K -> int8 dp4a, Q6_K -> fp32 dequant) instead of expanding to bf16 at load.
    // Default ON: it feeds the dp4a GEMV path (~27% faster decode, gate-passing) and
    // uses ~1.5 GB less VRAM. Set SPARKINFER_QATTN=0 to load dense bf16 instead.
    const bool qattn = []{ const char* a = getenv("SPARKINFER_QATTN");
                           return !(a && a[0] == '0'); }();
    auto mode_is_off = [](const std::string& v) {
        return v.empty() || v == "0" || v == "false" || v == "FALSE" ||
               v == "off" || v == "OFF" || v == "no" || v == "NO";
    };
    const bool q35_dense9b_requant_default =
        c.dense_ffn && c.n_layers == 32 && H == 4096 && c.moe_ffn == 12288 &&
        c.top_k == 1 && c.full_attn_interval == 4 && []{
            const char* e = getenv("SPARKINFER_DOWN_REQUANT_Q4K");
            return !(e && e[0] == '0');
        }();
    auto env_enabled = [&](const char* name, bool def) {
        const char* v = getenv(name);
        return v ? !mode_is_off(std::string(v)) : def;
    };
    // Qwen3.6-35B-A3B UD ships its full-attention q/o projections as Q8_0. Requantize
    // them to Q4_K at load (Lloyd fit) so decode reads ~47% fewer bytes on those matvecs
    // (~+3.3% decode at short context, gate-passing). On by default for the Qwen3.6
    // fingerprint; a no-op on the dense Qwythos path (which uses its own qkv default).
    const bool q36_ud_requant_default = is_qwen35_or_qwen36_hybrid_moe(g);
    const char* attn_env = getenv("SPARKINFER_ATTN_REQUANT_Q4K");
    const std::string attn_requant_mode =
        attn_env ? std::string(attn_env)
                 : (q35_dense9b_requant_default ? std::string("qkv,v")
                    : (q36_ud_requant_default ? std::string("attn_q,attn_output,qkv,attn_gate,ssm_out")
                                              : std::string()));
    // Muse Glimmer dense FFN gate/up -> Q3_A on the 42 layers selected by calibration.
    // The ten sensitive layers remain native Q4_K; this passes the production distribution gate
    // (top1 >= 0.90 and KL <= 0.10 against llama.cpp) while reducing decode weight traffic.
    // Architecture-scoped because no other model served by this shared loader was calibrated.
    // SPARKINFER_MUSE_FFN_Q3A=0 restores the Q4_K load path exactly, so both arms of an A/B come
    // out of one binary.
    const char* ffn_q3a_layers_env = getenv("SPARKINFER_MUSE_FFN_Q3A_LAYERS");
    const std::string ffn_q3a_layers = ffn_q3a_layers_env
        ? std::string(ffn_q3a_layers_env)
        : (c.muse_glimmer ? std::string("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,33,34,39,40,41,42,43,44,45,46,47,48,49,50,51") : std::string());
    const char* ffn_q3a_env = getenv("SPARKINFER_MUSE_FFN_Q3A");
    const std::string ffn_q3a_mode =
        ffn_q3a_env ? std::string(ffn_q3a_env)
                    : (c.muse_glimmer ? std::string("ffn_gate,ffn_up") : std::string());
    auto list_token = [](const std::string& list, const char* want) {
        const std::string w(want);
        size_t p = 0;
        while (p < list.size()) {
            while (p < list.size() && (list[p] == ',' || list[p] == '+' || list[p] == ':' || list[p] == ' ')) ++p;
            size_t e = p;
            while (e < list.size() && list[e] != ',' && list[e] != '+' && list[e] != ':' && list[e] != ' ') ++e;
            if (e > p && list.compare(p, e - p, w) == 0) return true;
            p = e + 1;
        }
        return false;
    };
    auto ffn_q3a_on = [&](const char* which) {
        if (mode_is_off(ffn_q3a_mode)) return false;
        if (ffn_q3a_mode == "1" || list_token(ffn_q3a_mode, "all")) return true;
        return list_token(ffn_q3a_mode, which);
    };
    auto mode_token = [&](const char* want) {
        const std::string w(want);
        size_t p = 0;
        while (p < attn_requant_mode.size()) {
            while (p < attn_requant_mode.size() &&
                   (attn_requant_mode[p] == ',' || attn_requant_mode[p] == '+' ||
                    attn_requant_mode[p] == ':' || attn_requant_mode[p] == ' '))
                ++p;
            size_t e = p;
            while (e < attn_requant_mode.size() &&
                   attn_requant_mode[e] != ',' && attn_requant_mode[e] != '+' &&
                   attn_requant_mode[e] != ':' && attn_requant_mode[e] != ' ')
                ++e;
            if (e > p && attn_requant_mode.compare(p, e - p, w) == 0) return true;
            p = e + 1;
        }
        return false;
    };
    auto has_suffix = [](const std::string& s, const char* suffix) {
        const std::string t(suffix);
        return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
    };
    auto layer_index = [](const std::string& name) {
        if (name.compare(0, 4, "blk.") != 0) return -1;
        int layer = 0;
        size_t p = 4;
        if (p >= name.size() || name[p] < '0' || name[p] > '9') return -1;
        while (p < name.size() && name[p] >= '0' && name[p] <= '9') {
            layer = layer * 10 + (name[p] - '0');
            ++p;
        }
        return (p < name.size() && name[p] == '.') ? layer : -1;
    };
    auto int_list_has = [](const std::string& list, int want) {
        if (list.empty()) return true;
        size_t p = 0;
        while (p < list.size()) {
            while (p < list.size() && (list[p] == ',' || list[p] == '+' || list[p] == ':' || list[p] == ' '))
                ++p;
            int v = 0;
            bool any = false;
            while (p < list.size() && list[p] >= '0' && list[p] <= '9') {
                v = v * 10 + (list[p] - '0');
                any = true;
                ++p;
            }
            if (any && v == want) return true;
            while (p < list.size() && list[p] != ',' && list[p] != '+' && list[p] != ':' && list[p] != ' ')
                ++p;
        }
        return false;
    };
    const bool req_attn_all = !mode_is_off(attn_requant_mode) &&
        (attn_requant_mode == "1" || mode_token("all") || mode_token("true") || mode_token("TRUE") ||
         mode_token("on") || mode_token("ON") || mode_token("yes") || mode_token("YES"));
    // Qwythos Q4_K_M leaves one linear-attention QKV matrix in Q6_K at decode (layer 2 was the
    // sensitive outlier in early gates; included in default list after re-validation).
    const char* qkv_layers_env = getenv("SPARKINFER_ATTN_REQUANT_Q4K_QKV_LAYERS");
    const std::string qkv_requant_layers =
        qkv_layers_env ? std::string(qkv_layers_env)
                       : ((q35_dense9b_requant_default && !attn_env)
                            ? std::string("0,1,2,6,9,12,18,21,24,28,29,30")
                            : std::string());
    int qkv_requant_limit = -1;
    if (const char* ql = getenv("SPARKINFER_ATTN_REQUANT_Q4K_QKV_LIMIT")) {
        qkv_requant_limit = atoi(ql);
        if (qkv_requant_limit < 0) qkv_requant_limit = -1;
    }
    int qkv_requant_used = 0;
    // Qwen3.6 GDN ssm_out projections ship Q8_0; requant them to Q4_K by default (all
    // thirty out-projections). SPARKINFER_ATTN_REQUANT_Q4K_SSM_MINLAYER pins a lower
    // bound on the layer index — the early GDN layers seed the recurrent state and are
    // the most precision-sensitive, so raising this trades a little decode speed for a
    // higher fuzzed top-1 margin.
    int ssm_out_min_layer = 0;
    if (const char* e = getenv("SPARKINFER_ATTN_REQUANT_Q4K_SSM_MINLAYER"))
        ssm_out_min_layer = atoi(e);
    auto req_attn_q4 = [&](const std::string& name, int ggml_type) {
        if (mode_is_off(attn_requant_mode)) return false;
        if (req_attn_all) return true;
        if ((mode_token("qkv") || mode_token("linear")) && has_suffix(name, "attn_qkv.weight")) {
            if (!int_list_has(qkv_requant_layers, layer_index(name))) return false;
            if (ggml_type == 14 && qkv_requant_limit >= 0 && qkv_requant_used++ >= qkv_requant_limit)
                return false;
            return true;
        }
        if ((mode_token("v") || mode_token("attn_v")) && has_suffix(name, "attn_v.weight")) return true;
        if ((mode_token("q") || mode_token("attn_q")) && has_suffix(name, "attn_q.weight")) return true;
        if ((mode_token("k") || mode_token("attn_k")) && has_suffix(name, "attn_k.weight")) return true;
        if ((mode_token("o") || mode_token("out") || mode_token("attn_output")) &&
            has_suffix(name, "attn_output.weight")) return true;
        // Qwen3.6 GDN input projections ship Q8_0 (the single largest per-token weight read):
        // attn_qkv (wqkv, handled by the "qkv" token above) + attn_gate (the z gate). Requant
        // both to Q4_K so they route through the existing Q4_K fused GDN qkv+z kernel (~47% fewer
        // bytes). SPARKINFER_ATTN_REQUANT_Q4K=attn_q,attn_output restores the #353-only behavior.
        if (mode_token("attn_gate") && has_suffix(name, "attn_gate.weight")) return true;
        if (mode_token("ssm_out") && has_suffix(name, "ssm_out.weight"))
            return layer_index(name) >= ssm_out_min_layer;
        return false;
    };
    const bool dual_dflash_lm_head = env_enabled(
        "SPARKINFER_DFLASH_DUAL_LMHEAD",
        q36_ud_requant_default && !q35_dense9b_requant_default);
    const bool req_lm_q4 = env_enabled("SPARKINFER_LMHEAD_REQUANT_Q4K",
                                       q35_dense9b_requant_default || dual_dflash_lm_head);
    auto attn_w = [&](const std::string& name, int& type) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        if (qattn && t && (t->ggml_type == 12 || t->ggml_type == 14 || t->ggml_type == 8))
            return dev_quant_requant_q4k(name, type, req_attn_q4(name, t->ggml_type));
        type = 0; return dense(name, false);
    };
    auto attn_w_opt = [&](const std::string& name, int& type) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        if (!t) { type = 0; return nullptr; }
        if (qattn && (t->ggml_type == 12 || t->ggml_type == 14 || t->ggml_type == 8))
            return dev_quant_requant_q4k(name, type, req_attn_q4(name, t->ggml_type));
        type = 0; return dense(name, false);
    };
    // Muse Glimmer ships output.weight as Q5_K -- the only Q5_K tensor in the file -- and Q5_K was
    // not on this list, so the head fell through to `dense()` and was dequantized to bf16. The
    // decode LM head then read 2.69 GB every token (gemv_f32_sk) instead of 0.76 GB through the
    // Q4_K MMVQ path: 1.60 ms of a 12.4 ms step. Requanting it to Q4_K at load is the same
    // mechanism this codebase already applies to other models' heads (SPARKINFER_LMHEAD_REQUANT_Q4K).
    static const bool mg_lm_q5k = [] {
        const char* e = getenv("SPARKINFER_MUSE_LMHEAD_Q5K");
        return !(e && e[0] == '0');
    }();
    auto lm_w = [&](const std::string& name, int& type) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        const bool q5k_ok = mg_lm_q5k && s.cfg.muse_glimmer && t && t->ggml_type == 13;
        if (qattn && t && (t->ggml_type == 12 || t->ggml_type == 14 || t->ggml_type == 8 || q5k_ok))
            return dev_quant_requant_q4k(name, type, req_lm_q4 || q5k_ok, q5k_ok);
        type = 0; return dense(name, false);
    };
    auto dense_opt = [&](const std::string& name, bool transpose) -> const void* {
        return g.tensor(name) ? dense(name, transpose) : nullptr;
    };
    auto expect_dims = [&](const std::string& name, std::initializer_list<long> dims) -> bool {
        const GGUFTensor* t = g.tensor(name);
        if (!t) { fprintf(stderr, "[gguf] missing %s\n", name.c_str()); return false; }
        if (t->n_dims != (int)dims.size()) {
            fprintf(stderr, "[gguf] bad rank for %s: got %d want %zu\n",
                    name.c_str(), t->n_dims, dims.size());
            return false;
        }
        int i = 0;
        for (long want : dims) {
            if (t->dims[i] != want) {
                fprintf(stderr, "[gguf] bad shape for %s dim%d: got %ld want %ld\n",
                        name.c_str(), i, t->dims[i], want);
                return false;
            }
            i++;
        }
        return true;
    };
    auto expect_dims_opt = [&](const std::string& name, std::initializer_list<long> dims) -> bool {
        return !g.tensor(name) || expect_dims(name, dims);
    };

    s.w.embed_tokens = dense("token_embd.weight", false);     // [vocab,hidden] as-is
    s.w.final_norm   = dense("output_norm.weight", false);
    const char* lm = g.tensor("output.weight") ? "output.weight" : "token_embd.weight";  // tied fallback
    const GGUFTensor* lm_tensor = g.tensor(lm);
    if (dual_dflash_lm_head && req_lm_q4 && lm_tensor &&
        (lm_tensor->ggml_type == 14 || lm_tensor->ggml_type == 8)) {
        s.dflash_lm_head = dev_quant(lm, s.dflash_lm_head_type);
    }
    s.w.lm_head = lm_w(lm, s.w.lm_head_type);                 // native [vocab,hidden] for GEMV
    if (!s.w.embed_tokens || !s.w.final_norm || !s.w.lm_head) return false;

    s.w.layers.resize(c.n_layers);
    for (int i = 0; i < c.n_layers; i++) {
        std::string b = "blk." + std::to_string(i) + ".";
        Qwen35LayerWeights& w = s.w.layers[i];
        w.linear_attn = is_linear_layer(c, i);
        w.swa = (i < (int)c.swa_layers.size()) ? c.swa_layers[i] : false;
        if (c.muse_glimmer && getenv("SPARKINFER_MG_DEBUG"))
            fprintf(stderr, "[mg-debug] layer %d: linear_attn=%d swa=%d full_attn_interval=%d hybrid=%d\n",
                    i, (int)w.linear_attn, (int)w.swa, c.full_attn_interval, (int)c.hybrid);
        if (!expect_dims(b + "attn_norm.weight", {H})) return false;
        w.input_norm = dense(b + "attn_norm.weight", false);
        if (w.linear_attn) {
            if (!expect_dims(b + "attn_qkv.weight", {H, s.linear_qkvdim}) ||
                !expect_dims(b + "attn_gate.weight", {H, s.linear_vdim}) ||
                !expect_dims(b + "ssm_conv1d.weight", {c.linear_conv_kernel, s.linear_qkvdim}) ||
                !expect_dims(b + "ssm_dt.bias", {c.linear_v_heads}) ||
                !expect_dims(b + "ssm_a", {c.linear_v_heads}) ||
                !expect_dims(b + "ssm_beta.weight", {H, c.linear_v_heads}) ||
                !expect_dims(b + "ssm_alpha.weight", {H, c.linear_v_heads}) ||
                !expect_dims(b + "ssm_norm.weight", {c.linear_head_dim}) ||
                !expect_dims(b + "ssm_out.weight", {s.linear_vdim, H})) return false;
            w.wqkv = attn_w(b + "attn_qkv.weight", w.wqkv_type);
            w.wqkv_gate = attn_w(b + "attn_gate.weight", w.wqkv_gate_type);
            w.ssm_conv = dense(b + "ssm_conv1d.weight", false);
            w.ssm_dt = dense(b + "ssm_dt.bias", false);
            w.ssm_a = dense(b + "ssm_a", false);
            w.ssm_beta = attn_w(b + "ssm_beta.weight", w.ssm_beta_type);
            w.ssm_alpha = attn_w(b + "ssm_alpha.weight", w.ssm_alpha_type);
            w.ssm_norm = dense(b + "ssm_norm.weight", false);
            w.ssm_out = attn_w(b + "ssm_out.weight", w.ssm_out_type);
        } else {
            w.q_has_gate = c.hybrid;
            if (c.muse_glimmer) {
                // attn_q.weight and attn_gate.weight ship as two separate [H, qdim]
                // tensors (unlike Qwen3.6's GGUF, which pre-fuses them into one [H,
                // qdim*2] tensor before writing the file) -- dequantize both to bf16 and
                // interleave into the per-head [q|gate] layout split_q_gate_kernel
                // (qwen36.cu) expects. Load-time only; w.wq ends up dense bf16 (wq_type=0)
                // rather than kept-quantized, same as this path's non-gguf/dense fallback.
                if (!expect_dims(b + "attn_q.weight", {H, s.qdim}) ||
                    !expect_dims(b + "attn_gate.weight", {H, s.qdim}) ||
                    !expect_dims(b + "attn_k.weight", {H, s.kvdim}) ||
                    !expect_dims(b + "attn_v.weight", {H, s.kvdim}) ||
                    !expect_dims(b + "attn_output.weight", {s.qdim, H}) ||
                    !expect_dims(b + "attn_q_norm.weight", {c.head_dim}) ||
                    !expect_dims(b + "attn_k_norm.weight", {c.head_dim})) return false;
                // SPARKINFER_MUSE_QGATE_Q=0 restores the original load-time dequantize+interleave
                // (kept for a same-binary A/B; see the projection site for the matching switch).
                static const int kQGateQ = []{ const char* e = getenv("SPARKINFER_MUSE_QGATE_Q");
                                               return (e && e[0] == '0') ? 0 : 1; }();
                if (kQGateQ) {
                    // attn_q and attn_gate both ship Q4_K. Dequantizing them to bf16 so they can be
                    // interleaved into the [q|gate] layout split_q_gate_kernel wants costs 109 MB
                    // per layer against 31 MB kept quantized -- 4.08 GB of extra reads on EVERY
                    // decode token across 52 layers, which profiled as the single largest kernel in
                    // the 128-decode run (gemv_f32_sk, 29.8% of GPU time). The interleave only
                    // exists because Qwen3.6's GGUF pre-fuses q|gate into one tensor; Muse ships
                    // them separately, so keep both quantized and project each straight into its
                    // own destination, which also drops the split entirely.
                    w.wq    = attn_w(b + "attn_q.weight", w.wq_type);
                    w.wgate = attn_w(b + "attn_gate.weight", w.wgate_type);
                    if (!w.wq || !w.wgate) return false;
                } else {
                const void* qd = dense(b + "attn_q.weight", false);
                const void* gd = dense(b + "attn_gate.weight", false);
                if (!qd || !gd) return false;
                void* combined = nullptr;
                cu(cudaMalloc(&combined, (size_t)s.qdim * 2 * H * sizeof(bf16)), "qgate interleave alloc");
                // cu() only logs CUDA errors, it never aborts -- unlike qd/gd above, nothing
                // downstream checks `combined` before using it. On a real cudaMalloc failure
                // (OOM; trivially reproducible by running this load under compute-sanitizer,
                // whose shadow-memory overhead multiplies every allocation) `combined` stays
                // null/stale and launch_interleave_qgate_rows below writes through it --
                // out-of-bounds device writes rather than a clean load failure.
                if (!combined) return false;
                kernels::launch_interleave_qgate_rows(qd, gd, combined, c.n_q_heads, c.head_dim, H, s.stream);
                cu(cudaStreamSynchronize(s.stream), "qgate interleave sync");
                s.owned.push_back(combined);
                w.wq = combined;
                w.wq_type = 0;
                }
                w.wk = attn_w(b + "attn_k.weight", w.wk_type);
                w.wv = attn_w(b + "attn_v.weight", w.wv_type);
                w.wo = attn_w(b + "attn_output.weight", w.wo_type);
                w.q_norm = dense(b + "attn_q_norm.weight", false);
                w.k_norm = dense(b + "attn_k_norm.weight", false);
            } else {
                const int q_out = w.q_has_gate ? s.qdim * 2 : s.qdim;
                if (!expect_dims(b + "attn_q.weight", {H, q_out}) ||
                    !expect_dims(b + "attn_k.weight", {H, s.kvdim}) ||
                    !expect_dims(b + "attn_v.weight", {H, s.kvdim}) ||
                    !expect_dims(b + "attn_output.weight", {s.qdim, H}) ||
                    !expect_dims(b + "attn_q_norm.weight", {c.head_dim}) ||
                    !expect_dims(b + "attn_k_norm.weight", {c.head_dim})) return false;
                w.wq = attn_w(b + "attn_q.weight", w.wq_type);
                w.wk = attn_w(b + "attn_k.weight", w.wk_type);
                w.wv = attn_w(b + "attn_v.weight", w.wv_type);
                w.wo = attn_w(b + "attn_output.weight", w.wo_type);
                w.q_norm = dense(b + "attn_q_norm.weight", false);
                w.k_norm = dense(b + "attn_k_norm.weight", false);
            }
        }
        if (!expect_dims_opt(b + "attn_post_norm.weight", {H}) ||
            !expect_dims_opt(b + "post_attention_norm.weight", {H}) ||
            !expect_dims_opt(b + "ffn_norm.weight", {H}) ||
            !expect_dims_opt(b + "post_ffw_norm.weight", {H})) return false;
        if (c.muse_glimmer) {
            // Muse Glimmer ships post_attention_norm.weight AND ffn_norm.weight as distinct
            // tensors (sandwich norm, not one norm serving double duty) -- load both,
            // unlike the single-fallback-chain below every other architecture uses.
            w.post_attn_norm = dense(b + "post_attention_norm.weight", false);
            w.ffn_norm = dense(b + "ffn_norm.weight", false);
            w.post_ffn_norm = dense(b + "post_ffw_norm.weight", false);
        } else {
            w.post_attn_norm = dense_opt(b + "attn_post_norm.weight", false);
            if (!w.post_attn_norm) w.post_attn_norm = dense_opt(b + "post_attention_norm.weight", false);
            if (!w.post_attn_norm) w.post_attn_norm = dense(b + "ffn_norm.weight", false);
        }
        if (c.dense_ffn) {
            if (!expect_dims(b + "ffn_gate.weight", {H, c.moe_ffn}) ||
                !expect_dims(b + "ffn_up.weight", {H, c.moe_ffn}) ||
                !expect_dims(b + "ffn_down.weight", {c.moe_ffn, H})) return false;
            // Gate and up convert together or not at all because the compact MMVQ kernel
            // consumes equal-stride pairs; unselected layers retain the native Q4_K path.
            const bool gu3 = ffn_q3a_on("ffn_gate") && ffn_q3a_on("ffn_up") &&
                             int_list_has(ffn_q3a_layers, i);
            w.gate_q = gu3 ? dev_quant_q3a(b + "ffn_gate.weight", w.gate_qtype, &w.prefill_gate_q, &w.prefill_gate_qtype)
                           : dev_quant(b + "ffn_gate.weight", w.gate_qtype);
            w.up_q   = gu3 ? dev_quant_q3a(b + "ffn_up.weight", w.up_qtype, &w.prefill_up_q, &w.prefill_up_qtype)
                           : dev_quant(b + "ffn_up.weight", w.up_qtype);
            w.down_q = dev_quant_down(b + "ffn_down.weight", w.down_qtype);
        } else {
            if (!expect_dims(b + "ffn_gate_inp.weight", {H, c.n_experts})) return false;
            // Router weight: keep Q8_0 raw if present in the GGUF (half bandwidth, on-read GEMV)
            {
                const GGUFTensor* rt = g.tensor(b + "ffn_gate_inp.weight");
                if (qattn && rt && rt->ggml_type == 8) {
                    w.router_w = dev_quant(b + "ffn_gate_inp.weight", w.router_w_type);
                } else {
                    w.router_w = dense(b + "ffn_gate_inp.weight", false);
                    w.router_w_type = 0;
                }
            }
            w.gate_q = dev_quant(b + "ffn_gate_exps.weight", w.gate_qtype);   // kept quantized
            w.up_q   = dev_quant(b + "ffn_up_exps.weight",   w.up_qtype);
            w.down_q = dev_quant(b + "ffn_down_exps.weight", w.down_qtype);
            if (s.cfg.n_shared > 0) {
            if (!expect_dims(b + "ffn_gate_shexp.weight", {H, c.moe_ffn}) ||
                !expect_dims(b + "ffn_up_shexp.weight", {H, c.moe_ffn}) ||
                !expect_dims(b + "ffn_down_shexp.weight", {c.moe_ffn, H}) ||
                !expect_dims_opt(b + "ffn_gate_inp_shexp.weight", {H})) return false;
            // GGUF-native [out,in] layout (no transpose) so the shared expert runs as
            // three fast one-warp-per-row GEMVs instead of the single-block dense kernel.
            const bool qmoe = []{ const char* a = getenv("SPARKINFER_QMOE");
                                   return !(a && a[0] == '0'); }();
            if (qmoe) {
                w.shared_gate_q = dev_quant(b + "ffn_gate_shexp.weight", w.shared_gate_qtype);
                w.shared_up_q   = dev_quant(b + "ffn_up_shexp.weight",   w.shared_up_qtype);
                w.shared_down_q = dev_quant(b + "ffn_down_shexp.weight", w.shared_down_qtype);
            }
            if (!qmoe || !w.shared_gate_q || !w.shared_up_q || !w.shared_down_q ||
                w.shared_gate_qtype != 8) {
                w.shared_gate = dense(b + "ffn_gate_shexp.weight", false);
                w.shared_up   = dense(b + "ffn_up_shexp.weight", false);
                w.shared_down = dense(b + "ffn_down_shexp.weight", false);
            }
            w.shared_gate_inp = attn_w_opt(b + "ffn_gate_inp_shexp.weight", w.shared_gate_inp_type);
            const bool have_shared_q = w.shared_gate_q && w.shared_up_q && w.shared_down_q;
            const bool have_shared_d = w.shared_gate && w.shared_up && w.shared_down;
            if (!have_shared_q && !have_shared_d) return false;
            }
        }
        const bool have_attn = w.linear_attn
            ? (w.wqkv && w.wqkv_gate && w.ssm_conv && w.ssm_dt && w.ssm_a &&
               w.ssm_beta && w.ssm_alpha && w.ssm_norm && w.ssm_out)
            : (w.wq && w.wk && w.wv && w.wo && w.q_norm && w.k_norm);
        const bool have_ffn = c.dense_ffn
            ? (w.gate_q && w.up_q && w.down_q)
            : (w.router_w && w.gate_q && w.up_q && w.down_q);
        if (!have_attn || !w.input_norm || !w.post_attn_norm || !have_ffn) return false;
        if (c.muse_glimmer && (!w.ffn_norm || !w.post_ffn_norm)) return false;
        if (i == 0 || i == c.n_layers - 1) fprintf(stderr, "[gguf] layer %d loaded\n", i);
    }
    // ---- eager per-row int8 scales of the routed experts (fused quantized-B MoE prefill GEMM) ----
    // The batched prefill can run the routed GEMMs straight off the native GGUF expert weights
    // instead of materializing the whole int8 expert pool once per layer (prefill_moe_q.cu). That
    // needs the per-row int8 scale, which is a property of the FULL row (amax over all `cols`) and
    // so cannot be derived from a K-tile inside the GEMM. Compute it here with the same kernel the
    // materialize path uses and keep only its `scale` output: the fused GEMM then quantizes to the
    // identical int8 bytes by construction, not by re-deriving the scale.
    // ~120 MB for Qwen3.6-35B-A3B (40 layers x 256 experts x (512+512+2048) rows). Any failure
    // leaves the pointers null and the prefill simply keeps materializing.
    // SPARKINFER_PREFILL_MOE_QB=0 skips the precompute entirely.
    if (!c.dense_ffn && c.n_experts > 0 && c.moe_ffn > 0) {
        const char* qb_env = getenv("SPARKINFER_PREFILL_MOE_QB");
        if (!qb_env || qb_env[0] != '0') {
            const size_t rg = (size_t)c.n_experts * c.moe_ffn;   // gate/up rows per layer
            const size_t rd = (size_t)c.n_experts * H;           // down rows per layer
            const size_t ng = rg * (size_t)c.n_layers, nd = rd * (size_t)c.n_layers;
            const size_t tmp_bytes = 64u << 20;                  // int8 scratch, thrown away
            signed char* tmp = nullptr;
            bool ok = cudaMalloc(&s.moe_rs_gate, ng * sizeof(float)) == cudaSuccess;
            ok = ok && cudaMalloc(&s.moe_rs_up,   ng * sizeof(float)) == cudaSuccess;
            ok = ok && cudaMalloc(&s.moe_rs_down, nd * sizeof(float)) == cudaSuccess;
            ok = ok && cudaMalloc(&tmp, tmp_bytes) == cudaSuccess;
            auto fill = [&](int qtype, const void* src, float* dst, size_t rows, int cols) {
                const int blk = (qtype == 12) ? 144 : (qtype == 13) ? 176 : 210;
                const size_t rb = (size_t)(cols >> 8) * (size_t)blk;   // bytes per quantized row
                const size_t chunk = tmp_bytes / (size_t)cols;
                for (size_t r0 = 0; r0 < rows; r0 += chunk) {
                    const size_t nr = (rows - r0 < chunk) ? (rows - r0) : chunk;
                    if (!kernels::launch_gguf_dequant_rows_i8(
                            qtype, (const char*)src + r0 * rb, tmp, dst + r0,
                            (int)nr, cols, s.stream))
                        return false;
                }
                return true;
            };
            for (int i = 0; ok && i < c.n_layers; i++) {
                const Qwen35LayerWeights& lw = s.w.layers[i];
                if (!lw.gate_q || !lw.up_q || !lw.down_q) { ok = false; break; }
                ok = fill(lw.gate_qtype, lw.gate_q, s.moe_rs_gate + (size_t)i * rg, rg, H)
                  && fill(lw.up_qtype,   lw.up_q,   s.moe_rs_up   + (size_t)i * rg, rg, H)
                  && fill(lw.down_qtype, lw.down_q, s.moe_rs_down + (size_t)i * rd, rd, c.moe_ffn);
            }
            if (ok) ok = cudaStreamSynchronize(s.stream) == cudaSuccess;
            if (tmp) cudaFree(tmp);
            if (!ok) {
                cudaFree(s.moe_rs_gate); cudaFree(s.moe_rs_up); cudaFree(s.moe_rs_down);
                s.moe_rs_gate = s.moe_rs_up = s.moe_rs_down = nullptr;
                fprintf(stderr, "[prefill-moe] expert row-scale precompute unavailable "
                                "-> int8 materialize path\n");
            } else {
                fprintf(stderr, "[prefill-moe] expert int8 row scales ready (%.0f MB)\n",
                        (double)((2 * ng + nd) * sizeof(float)) / (1024.0 * 1024.0));
            }
        }
    }
    // ---- Muse Glimmer: eager per-row int8 scales for the fused quantized-B dense prefill GEMM ----
    // Muse's dense attn (wq/wgate/wk/wv/wo) and FFN gate/up ship as Q4_K. The batched prefill can
    // decode them to int8 inside the GEMM's B-stage (prefill_moe_q.cu's dense path) rather than
    // materializing the whole int8 weight per layer (dequant -> write W_i8 -> read W_i8 back). That
    // needs the per-output-row int8 scale -- amax over the FULL row -- which is computed here with
    // the SAME kernel the materialize path uses (launch_gguf_dequant_rows_i8, keeping only its
    // `scale`), so the fused GEMM's int8 bytes match the materialize path's by construction, not by
    // re-deriving the scale. Q6_K down (and any non-Q4/Q5 attn weight) is left null -> stays on the
    // materialize path. ~11 MB for Muse-30B. SPARKINFER_MUSE_PREFILL_QB=0 skips the precompute.
    if (c.muse_glimmer) {
        const char* qb_env = getenv("SPARKINFER_MUSE_PREFILL_QB");
        if (!qb_env || qb_env[0] != '0') {
            // Q6_K too: the dense fused GEMM decodes it now, so a Q6_K attn_v / ffn_down gets its
            // row scales here instead of falling back to the per-layer materialize.
            auto fusable = [](int t) { return t == 12 || t == 13 || t == 14; };  // Q4_K / Q5_K / Q6_K
            const int qd = c.n_q_heads * c.head_dim;                   // qdim (4096)
            const int kd = c.n_kv_heads * c.head_dim;                  // kvdim (256)
            const int ff = c.moe_ffn;                                  // dense FFN width (19968)
            // + H for ffn_down: the original layout reserved no slot for it, so every Q4_K down
            // fell back to the materialize path regardless of being a fusable type. Slots are
            // reserved for all eight; a slot stays unfilled (and its *_rs null) when that
            // weight's type is not fusable, so a Q6_K down still takes the materialize path.
            const size_t per_layer = (size_t)(2 * qd + 2 * kd + H + 2 * ff + H);  // rows/layer
            const size_t total = per_layer * (size_t)c.n_layers;
            const size_t tmp_bytes = 64u << 20;                        // int8 scratch, thrown away
            signed char* tmp = nullptr;
            bool ok = cudaMalloc(&s.muse_rs, total * sizeof(float)) == cudaSuccess;
            ok = ok && cudaMalloc(&tmp, tmp_bytes) == cudaSuccess;
            auto fill = [&](int qtype, const void* src, float* dst, size_t rows, int cols) -> bool {
                const int blk = (qtype == 12) ? 144 : (qtype == 13) ? 176 : 210;
                const size_t rb = (size_t)(cols >> 8) * (size_t)blk;   // bytes per quantized row
                const size_t chunk = tmp_bytes / (size_t)cols;        // rows/chunk fitting tmp
                for (size_t r0 = 0; r0 < rows; r0 += chunk) {
                    const size_t nr = (rows - r0 < chunk) ? (rows - r0) : chunk;
                    if (!kernels::launch_gguf_dequant_rows_i8(
                            qtype, (const char*)src + r0 * rb, tmp, dst + r0, (int)nr, cols, s.stream))
                        return false;
                }
                return true;
            };
            for (int i = 0; ok && i < c.n_layers; i++) {
                Qwen35LayerWeights& lw = s.w.layers[i];
                float* base = s.muse_rs + (size_t)i * per_layer;
                size_t off = 0;
                // Compute a weight's row scales into the pool and publish its *_rs pointer; the pool
                // slot is reserved for every weight (fixed layout) but filled only when fusable.
                auto place = [&](const void* W, int wt, const float** rs, int rows, int cols) {
                    float* dst = base + off; off += (size_t)rows;
                    if (ok && W && fusable(wt) && fill(wt, W, dst, (size_t)rows, cols)) *rs = dst;
                };
                place(lw.wq,     lw.wq_type,     &lw.wq_rs,    qd, H);
                place(lw.wgate,  lw.wgate_type,  &lw.wgate_rs, qd, H);
                place(lw.wk,     lw.wk_type,     &lw.wk_rs,    kd, H);
                place(lw.wv,     lw.wv_type,     &lw.wv_rs,    kd, H);
                place(lw.wo,     lw.wo_type,     &lw.wo_rs,    H,  qd);
                place(lw.prefill_gate_q ? lw.prefill_gate_q : lw.gate_q,
                      lw.prefill_gate_q ? lw.prefill_gate_qtype : lw.gate_qtype,
                      &lw.gate_rs, ff, H);
                place(lw.prefill_up_q ? lw.prefill_up_q : lw.up_q,
                      lw.prefill_up_q ? lw.prefill_up_qtype : lw.up_qtype,
                      &lw.up_rs, ff, H);
                place(lw.down_q, lw.down_qtype,  &lw.down_rs,  H,  ff);
            }
            if (ok) ok = cudaStreamSynchronize(s.stream) == cudaSuccess;
            if (tmp) cudaFree(tmp);
            if (!ok) {
                cudaFree(s.muse_rs); s.muse_rs = nullptr;
                for (int i = 0; i < c.n_layers; i++) {
                    Qwen35LayerWeights& lw = s.w.layers[i];
                    lw.wq_rs = lw.wgate_rs = lw.wk_rs = lw.wv_rs = lw.wo_rs = lw.gate_rs = lw.up_rs = nullptr;
                    lw.down_rs = nullptr;
                }
                fprintf(stderr, "[prefill-muse] dense row-scale precompute unavailable "
                                "-> int8 materialize path\n");
            } else {
                fprintf(stderr, "[prefill-muse] dense int8 row scales ready (%.0f MB)\n",
                        (double)(total * sizeof(float)) / (1024.0 * 1024.0));
            }
        }
    }
    // Optional native Blackwell FP4 copies for Muse's gate/up projections. This is eager
    // because scored prefill times the first pass. Gate/up native prefill copies that are distinct
    // from their compact Q3_A decode weights are released after conversion, keeping peak resident
    // memory within a 32-GB card. The normal GGUF pointers remain the correctness fallback.
    // On by default now that the kernels are in the default build: the conversion is gated on
    // Muse plus a successful sm_120a FP4 build, and the GGUF pointers stay as the fallback, so a
    // box that could not build the FP4 path simply never takes it. SPARKINFER_MUSE_PREFILL_NVFP4=0
    // forces the portable quantized projections back.
    const char* fp4_env = getenv("SPARKINFER_MUSE_PREFILL_NVFP4");
    if (c.muse_glimmer && (!fp4_env || fp4_env[0] != '0') &&
        kernels::prefill_nvfp4_supported(128, c.moe_ffn, H) &&
        kernels::prefill_nvfp4_supported(128, H, c.moe_ffn)) {
        void* tmp = nullptr;
        const size_t tmp_elems = (size_t)c.moe_ffn * H;
        bool ok = cudaMalloc(&tmp, tmp_elems * sizeof(bf16)) == cudaSuccess;
        int ready = 0;
        auto convert = [&](const void* src, int qtype, int rows, int cols,
                           const void** data, const void** sf) -> bool {
            void *d = nullptr, *scale = nullptr;
            if (!src || cudaMalloc(&d, kernels::prefill_nvfp4_data_bytes(rows, cols)) != cudaSuccess)
                return false;
            if (cudaMalloc(&scale, kernels::prefill_nvfp4_scale_bytes_b(rows, cols)) != cudaSuccess) {
                cudaFree(d); return false;
            }
            kernels::launch_gguf_dequant(qtype, src, tmp, (long)rows * cols, s.stream);
            if (!kernels::launch_prefill_nvfp4_quant_b(tmp, d, scale, rows, cols, s.stream) ||
                cudaStreamSynchronize(s.stream) != cudaSuccess) {
                cudaFree(d); cudaFree(scale); return false;
            }
            s.owned.push_back(d); s.owned.push_back(scale); *data = d; *sf = scale;
            return true;
        };
        auto release_prefill_copy = [&](const void*& p, const void* decode) {
            if (!p || p == decode) return;
            auto it = std::find(s.owned.begin(), s.owned.end(), const_cast<void*>(p));
            if (it != s.owned.end()) { cudaFree(*it); s.owned.erase(it); }
            p = nullptr;
        };
        for (int i = 0; ok && i < c.n_layers; ++i) {
            Qwen35LayerWeights& lw = s.w.layers[i];
            const void* g = lw.prefill_gate_q ? lw.prefill_gate_q : lw.gate_q;
            const void* u = lw.prefill_up_q ? lw.prefill_up_q : lw.up_q;
            const int gt = lw.prefill_gate_q ? lw.prefill_gate_qtype : lw.gate_qtype;
            const int ut = lw.prefill_up_q ? lw.prefill_up_qtype : lw.up_qtype;
            ok = convert(g, gt, c.moe_ffn, H, &lw.gate_fp4, &lw.gate_fp4_sf) &&
                 convert(u, ut, c.moe_ffn, H, &lw.up_fp4, &lw.up_fp4_sf);
            if (ok) {
                release_prefill_copy(lw.prefill_gate_q, lw.gate_q);
                release_prefill_copy(lw.prefill_up_q, lw.up_q);
                ++ready;
            }
        }
        if (tmp) cudaFree(tmp);
        fprintf(stderr, "[prefill-muse] SM120 NVFP4 FFN weights ready: %d/%d layers%s\n",
                ready, c.n_layers, ok ? "" : " (remaining layers use GGUF fallback)");
    }
    // decode scratch (mf_* / fa_*) is allocated in the constructor for all paths.
    return true;
}

} // namespace sparkinfer
