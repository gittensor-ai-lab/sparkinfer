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
#include "sparkinfer/thermal_governor.h"
#include "sparkinfer/kv_ops.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/kernels/attention.h"
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/fused.h"
#include "sparkinfer/kernels/moe.h"
#include "sparkinfer/kernels/quant.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <limits>

namespace sparkinfer {

namespace {
inline void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fprintf(stderr, "[qwen35] %s: %s\n", what, cudaGetErrorString(e));
}
using bf16 = unsigned short;

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
    long v = g.meta_int("qwen35moe." + key, missing);
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

struct Qwen35Model::Impl {
    Qwen35Config cfg;
    KVCacheManager* kv;
    moe::MoEEngine* engine;
    Qwen35Weights w;
    cudaStream_t stream{};
    cudaStream_t stream_k{}, stream_v{};         // side streams for concurrent K/V projection
    cudaEvent_t ev_qkv{}, ev_k{}, ev_v{};        // fork/join events (captured into the decode graph)
    cudaEvent_t ev_pipe_fork{}, ev_gdn_z{}, ev_gdn_ab{};
    cudaEvent_t ev_sx_gate{}, ev_sx_done{};
    uint64_t seq_id = 0;
    int qdim, kvdim;
    int linear_qdim = 0, linear_vdim = 0, linear_qkvdim = 0;
    bool gguf = false;   // true after load_gguf: dense weights are native [out,in], use GEMV
    // CUDA-graph capture of the decode compute (captured once, replayed each token)
    cudaGraph_t cu_graph{};
    cudaGraphExec_t cu_exec{};
    bool graph_ready = false;
    bool bench_feedback_graph = false;

    // scratch (bf16)
    bf16 *x, *xn, *q, *k, *v, *attn, *ao, *h, *hn, *routed, *shared;
    bf16 *qraw = nullptr, *qgate = nullptr;
    bf16 *lin_qkv = nullptr, *lin_q = nullptr, *lin_k = nullptr, *lin_v = nullptr;
    bf16 *lin_z = nullptr, *lin_alpha = nullptr, *lin_beta = nullptr;
    bf16 *lin_gdn = nullptr, *lin_norm = nullptr, *shared_gate_tmp = nullptr;
    bf16 *sh_gate = nullptr, *sh_up = nullptr, *sh_h = nullptr;   // shared-expert GEMV scratch [moe_ffn]
    bf16 *lin_conv_state = nullptr;
    float* lin_state = nullptr;
    float* logits;
    int *d_scalars, *d_tok, *d_out_id, *d_pos, *d_seqlen, *d_writepos, *d_shared_ids;
    int *h_scalars = nullptr, *h_out_id = nullptr;
    float* d_shared_w;
    std::vector<void*> owned;   // device buffers from load_weights / load_gguf
    // GGUF fused-expert decode scratch (allocated by load_gguf)
    float *mf_logits = nullptr, *mf_weights = nullptr, *mf_h = nullptr, *mf_out = nullptr;
    float *sx_h = nullptr;   // pipelined shared-expert h_scratch (avoids racing routed mf_h)
    void  *sx_q8 = nullptr;  // pipelined shared-expert Q8_1(h) for down (avoids racing aq81)
    int   *mf_ids = nullptr, *mf_counts = nullptr;
    // flash-decoding (KV-split) attention partials
    static constexpr int MAX_NSPLITS = 256;   // partials sized for this; adaptive n_splits <= this
    int n_splits = 32;
    bool adaptive_splits = true;              // scale n_splits with seq_len (decode graph re-captured on change)
    int split_chunk = 256;                    // target serial KV per split (SPARKINFER_SPLIT_CHUNK)
    float *fa_m = nullptr, *fa_l = nullptr, *fa_acc = nullptr;
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
    bool use_shexp_pipe = true; // default ON: overlap shared expert with routed MoE. =0 disables
    bool use_addnorm3 = true;   // default ON: fold routed+shared residual_add into post-MoE add_rmsnorm. =0 disables

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
    p_->d_scalars=p_->alloc<int>(4);
    p_->d_tok=p_->d_scalars + 0; p_->d_pos=p_->d_scalars + 1;
    p_->d_writepos=p_->d_scalars + 2; p_->d_seqlen=p_->d_scalars + 3;
    p_->d_out_id=p_->alloc<int>(1);
    cu(cudaHostAlloc(&p_->h_scalars, 4 * sizeof(int), cudaHostAllocDefault), "host scalars");
    cu(cudaHostAlloc(&p_->h_out_id, sizeof(int), cudaHostAllocDefault), "host out id");
    p_->d_shared_ids=p_->alloc<int>(1); p_->d_shared_w=p_->alloc<float>(1);
    int zero=0; float one=1.f;
    cu(cudaMemcpy(p_->d_shared_ids,&zero,sizeof(int),cudaMemcpyHostToDevice),"shared ids");
    cu(cudaMemcpy(p_->d_shared_w,&one,sizeof(float),cudaMemcpyHostToDevice),"shared w");
    // Fused-expert + flash-decoding decode scratch (batch 1). Allocated here so
    // EVERY load path (set_weights / load_weights / load_gguf) has it — not just
    // GGUF. (fa_* NULL here is what crashed flash_decode_split on the non-GGUF path.)
    p_->mf_logits  = p_->alloc<float>(cfg.n_experts);
    p_->mf_ids     = p_->alloc<int>(cfg.top_k);
    p_->mf_weights = p_->alloc<float>(cfg.top_k);
    p_->mf_counts  = p_->alloc<int>(cfg.n_experts);
    p_->mf_h       = p_->alloc<float>((size_t)cfg.top_k * cfg.moe_ffn);
    p_->mf_out     = p_->alloc<float>(cfg.hidden);
    if (cfg.n_shared > 0) {
        p_->sx_h  = p_->alloc<float>(cfg.moe_ffn);
        p_->sx_q8 = p_->alloc<char>(kernels::llama_q8_1_bytes(cfg.moe_ffn));
    }
    const size_t fa_n = (size_t)cfg.n_q_heads * Impl::MAX_NSPLITS;   // sized for the adaptive max
    p_->fa_m   = p_->alloc<float>(fa_n);
    p_->fa_l   = p_->alloc<float>(fa_n);
    p_->fa_acc = p_->alloc<float>(fa_n * cfg.head_dim);
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
    if (const char* e = getenv("SPARKINFER_SHEXP_PIPE")) p_->use_shexp_pipe = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_ADDNORM3")) p_->use_addnorm3 = !(e[0] == '0');
}

Qwen35Model::~Qwen35Model() {
    for (void* b : p_->owned) cudaFree(b);
    cudaFree(p_->x); cudaFree(p_->xn); cudaFree(p_->q); cudaFree(p_->k); cudaFree(p_->v);
    cudaFree(p_->attn); cudaFree(p_->ao); cudaFree(p_->h); cudaFree(p_->hn);
    cudaFree(p_->routed); cudaFree(p_->shared); cudaFree(p_->logits);
    // main's packed decode scalars (d_tok/d_pos/d_seqlen/d_writepos alias into d_scalars — not freed separately)
    cudaFree(p_->d_scalars); cudaFree(p_->d_out_id);
    cudaFreeHost(p_->h_scalars); cudaFreeHost(p_->h_out_id);
    cudaFree(p_->d_shared_ids); cudaFree(p_->d_shared_w);
    // Qwen3.6 Gated-DeltaNet buffers (allocated only for the hybrid model)
    cudaFree(p_->qraw); cudaFree(p_->qgate);
    cudaFree(p_->lin_qkv); cudaFree(p_->lin_q); cudaFree(p_->lin_k); cudaFree(p_->lin_v);
    cudaFree(p_->lin_z); cudaFree(p_->lin_alpha); cudaFree(p_->lin_beta);
    cudaFree(p_->lin_gdn); cudaFree(p_->lin_norm); cudaFree(p_->lin_conv_state); cudaFree(p_->lin_state);
    cudaFree(p_->shared_gate_tmp);
    cudaFree(p_->mf_logits); cudaFree(p_->mf_weights); cudaFree(p_->mf_h); cudaFree(p_->mf_out);
    cudaFree(p_->sx_h); cudaFree(p_->sx_q8);
    cudaFree(p_->mf_ids); cudaFree(p_->mf_counts);
    cudaFree(p_->fa_m); cudaFree(p_->fa_l); cudaFree(p_->fa_acc);
    cudaFree(p_->aq8); cudaFree(p_->aq8_d); cudaFree(p_->aq8_s); cudaFree(p_->aq81);
    if (p_->graph_ready) { cudaGraphExecDestroy(p_->cu_exec); cudaGraphDestroy(p_->cu_graph); }
    cudaEventDestroy(p_->ev_qkv); cudaEventDestroy(p_->ev_k); cudaEventDestroy(p_->ev_v);
    cudaEventDestroy(p_->ev_pipe_fork); cudaEventDestroy(p_->ev_gdn_z); cudaEventDestroy(p_->ev_gdn_ab);
    cudaEventDestroy(p_->ev_sx_gate); cudaEventDestroy(p_->ev_sx_done);
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

int Qwen35Model::forward_token(int token_id, int position) {
    Impl& s = *p_;
    const Qwen35Config& c = s.cfg;
    const int H = c.hidden;
    kernels::GemmConfig gc{};
    int seqlen = position + 1;
    cudaStream_t st = s.stream;

    s.h_scalars[0] = token_id;
    s.h_scalars[1] = position;
    s.h_scalars[2] = position;
    s.h_scalars[3] = seqlen;
    cu(cudaMemcpyAsync(s.d_scalars, s.h_scalars, 4 * sizeof(int), cudaMemcpyHostToDevice, st), "decode scalars");

    // Depth-adaptive KV-split: keep 32 splits for the short-context sweet spot, then jump to
    // the 128-split occupancy plateau on RTX 5090. The split grid is num_kv_heads*n_splits CTAs,
    // so 64 splits still underfills mid-context decode; 128 improves 512/2k/4k. Past the 16k
    // knee, use MAX_NSPLITS to keep each split's serial KV chunk bounded. Partials are sized for
    // MAX_NSPLITS, and the online-softmax combine is exact for any split count (accuracy unchanged).
    if (s.adaptive_splits) {
        int want = 32;
        if ((long)seqlen > 2L * s.split_chunk) want = 128;
        if ((long)seqlen > 64L * s.split_chunk) want = Impl::MAX_NSPLITS;
        if (want > Impl::MAX_NSPLITS) want = Impl::MAX_NSPLITS;
        if (want != s.n_splits) {                       // changed -> invalidate the captured graph
            s.n_splits = want;
            if (s.graph_ready) {
                cudaGraphExecDestroy(s.cu_exec); cudaGraphDestroy(s.cu_graph); s.graph_ready = false;
            }
        }
    }
    if (c.hybrid && position == 0) {
        cu(cudaMemsetAsync(s.lin_state, 0,
                           (size_t)c.n_layers * c.linear_v_heads * c.linear_head_dim * c.linear_head_dim * sizeof(float), st),
           "linear state reset");
        cu(cudaMemsetAsync(s.lin_conv_state, 0,
                           (size_t)c.n_layers * (c.linear_conv_kernel - 1) * s.linear_qkvdim * sizeof(bf16), st),
           "linear conv reset");
    }

    // Capture the decode compute into a CUDA graph on the first token, then
    // replay it every token (per-token inputs live in the d_tok/pos/seqlen/
    // writepos device buffers uploaded above, so replay produces fresh results).
    if (s.graph_ready) {
        cu(cudaGraphLaunch(s.cu_exec, st), "graph launch");
        cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "out_id");
        cu(cudaStreamSynchronize(st), "sync");
        return *s.h_out_id;
    }
    cu(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin capture");

    kernels::launch_embedding(s.d_tok, s.w.embed_tokens, s.x, 1, H, st);

    int* btable = s.kv->block_table(s.seq_id);
    // Prime: xn = RMSNorm(x, layer0.input_norm). Each layer's tail then fuses the
    // post-MoE residual with the NEXT layer's input norm (or final_norm), so the
    // per-layer input RMSNorm + two residual-adds collapse into two fused kernels.
    kernels::launch_rmsnorm(s.x, s.w.layers[0].input_norm, s.xn, 1, H, c.rms_eps, st);

    // When the fused norm+quant path is on, each layer's post-MoE add_rmsnorm2 also emits
    // Q8_1(xn) into aq81, so the next layer's QKV-input quantize (and the LM-head quantize)
    // are already done. Only the prime norm above still needs the standalone quant (layer 0).
    const bool fnq = s.gguf && s.use_fnq && s.use_pq && s.use_llama;

    for (int L = 0; L < c.n_layers; L++) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        bool xn_q8_ready = fnq && L > 0;
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
            const bool gdn_pipelined = s.gguf && s.use_gdn_pipe;
            if (gdn_pipelined) {
                cudaEventRecord(s.ev_pipe_fork, st);
                cudaStreamWaitEvent(s.stream_k, s.ev_pipe_fork, 0);
                cudaStreamWaitEvent(s.stream_v, s.ev_pipe_fork, 0);
                proj_xn(w.wqkv_gate, w.wqkv_gate_type, s.lin_z, s.linear_vdim, s.stream_k);
                cudaEventRecord(s.ev_gdn_z, s.stream_k);
                proj_xn(w.ssm_alpha, w.ssm_alpha_type, s.lin_alpha, c.linear_v_heads, s.stream_v);
                proj_xn(w.ssm_beta, w.ssm_beta_type, s.lin_beta, c.linear_v_heads, s.stream_v);
                cudaEventRecord(s.ev_gdn_ab, s.stream_v);
                proj_xn(w.wqkv, w.wqkv_type, s.lin_qkv, s.linear_qkvdim, st);
            } else {
                proj_xn(w.wqkv, w.wqkv_type, s.lin_qkv, s.linear_qkvdim, st);
                proj_xn(w.wqkv_gate, w.wqkv_gate_type, s.lin_z, s.linear_vdim, st);
                proj_xn(w.ssm_alpha, w.ssm_alpha_type, s.lin_alpha, c.linear_v_heads, st);
                proj_xn(w.ssm_beta, w.ssm_beta_type, s.lin_beta, c.linear_v_heads, st);
            }

            bf16* conv_state = s.lin_conv_state +
                (size_t)L * (c.linear_conv_kernel - 1) * s.linear_qkvdim;
            kernels::launch_qwen36_conv_split_l2(s.lin_qkv, w.ssm_conv, conv_state,
                                                 s.lin_q, s.lin_k, s.lin_v,
                                                 c.linear_q_heads, c.linear_v_heads,
                                                 c.linear_head_dim, c.linear_conv_kernel,
                                                 c.rms_eps, st);
            if (gdn_pipelined) cudaStreamWaitEvent(st, s.ev_gdn_ab, 0);
            float* layer_state = s.lin_state +
                (size_t)L * c.linear_v_heads * c.linear_head_dim * c.linear_head_dim;
            kernels::launch_qwen36_gdn_ar(s.lin_q, s.lin_k, s.lin_v,
                                          s.lin_alpha, s.lin_beta, w.ssm_dt, w.ssm_a,
                                          layer_state, s.lin_gdn,
                                          c.linear_q_heads, c.linear_v_heads,
                                          c.linear_head_dim, st);
            if (gdn_pipelined) cudaStreamWaitEvent(st, s.ev_gdn_z, 0);
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
                if (s.use_qkvstream) {
                    cudaEventRecord(s.ev_qkv, st);
                    cudaStreamWaitEvent(s.stream_k, s.ev_qkv, 0);
                    cudaStreamWaitEvent(s.stream_v, s.ev_qkv, 0);
                    proj_xn(w.wq, w.wq_type, w.q_has_gate ? s.qraw : s.q, w.q_has_gate ? s.qdim * 2 : s.qdim, st);
                    proj_xn(w.wk, w.wk_type, s.k, s.kvdim, s.stream_k);
                    proj_xn(w.wv, w.wv_type, s.v, s.kvdim, s.stream_v);
                    cudaEventRecord(s.ev_k, s.stream_k);
                    cudaEventRecord(s.ev_v, s.stream_v);
                    cudaStreamWaitEvent(st, s.ev_k, 0);
                    cudaStreamWaitEvent(st, s.ev_v, 0);
                } else {
                    proj_xn(w.wq, w.wq_type, w.q_has_gate ? s.qraw : s.q, w.q_has_gate ? s.qdim * 2 : s.qdim, st);
                    proj_xn(w.wk, w.wk_type, s.k, s.kvdim, st);
                    proj_xn(w.wv, w.wv_type, s.v, s.kvdim, st);
                }
            } else {
                kernels::launch_gemm(s.xn, w.wq, w.q_has_gate ? s.qraw : s.q,
                                     1, w.q_has_gate ? s.qdim * 2 : s.qdim, H, 1.f, 0.f, gc, st);
                kernels::launch_gemm(s.xn, w.wk, s.k, 1, s.kvdim, H, 1.f, 0.f, gc, st);
                kernels::launch_gemm(s.xn, w.wv, s.v, 1, s.kvdim, H, 1.f, 0.f, gc, st);
            }
            if (w.q_has_gate)
                kernels::launch_qwen36_split_q_gate(s.qraw, s.q, s.qgate, c.n_q_heads, c.head_dim, st);

            // ---- QK-norm + RoPE + KV-append ----
            const bool kv8 = s.kv->int8_kv();
            const int kv_elem = kv8 ? 1 : 2;
            void* kpool = (char*)s.kv->k_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
            void* vpool = (char*)s.kv->v_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
            void* kscale = kv8 ? (char*)s.kv->k_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
            void* vscale = kv8 ? (char*)s.kv->v_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
            const bool partial_rope = (c.rope_dim > 0 && c.rope_dim < c.head_dim);
            if (!w.q_has_gate && !partial_rope && (s.use_attnin || kv8)) {
                // Qwen3-MoE frontier: fused int8 QK-norm + RoPE + KV-append (unchanged vs main)
                kernels::launch_qknorm_rope_kv_append(s.q, s.k, s.v, w.q_norm, w.k_norm, kpool, vpool,
                                                      btable, s.d_pos, 1, c.n_q_heads, c.n_kv_heads,
                                                      c.head_dim, c.rope_theta, c.rms_eps,
                                                      s.kv->block_size(), s.kv->max_blocks_per_seq(), st,
                                                      kscale, vscale, kv8 ? 1 : 0);
            } else {
                // Qwen3.6 (gated / partial-rotary): fuse QK-norm + partial-RoPE + KV when enabled.
                if (partial_rope && s.use_qkfuse) {
                    kernels::launch_qknorm_rope_kv_partial(s.q, s.k, s.v, w.q_norm, w.k_norm,
                        (bf16*)kpool, (bf16*)vpool, btable, s.d_pos, 1,
                        c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_dim,
                        c.rope_theta, c.rms_eps, s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                } else {
                    if (s.use_qkfuse)
                        kernels::launch_rmsnorm_qk(s.q, s.k, w.q_norm, w.k_norm, c.n_q_heads, c.n_kv_heads, c.head_dim, c.rms_eps, st);
                    else {
                        kernels::launch_rmsnorm(s.q, w.q_norm, s.q, c.n_q_heads,  c.head_dim, c.rms_eps, st);
                        kernels::launch_rmsnorm(s.k, w.k_norm, s.k, c.n_kv_heads, c.head_dim, c.rms_eps, st);
                    }
                    if (partial_rope) {
                        kernels::launch_rope_kv_append_partial(s.q, s.k, s.v, (bf16*)kpool, (bf16*)vpool, btable, s.d_pos, 1,
                                                               c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_dim,
                                                               c.rope_theta, s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                    } else if (s.use_ropekv) {
                        kernels::launch_rope_kv_append(s.q, s.k, s.v, (bf16*)kpool, (bf16*)vpool, btable, s.d_pos, 1,
                                                       c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta,
                                                       s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                    } else {
                        kernels::launch_rope(s.q, s.k, s.d_pos, 1, c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta, st);
                        launch_kv_append((bf16*)kpool, (bf16*)vpool, s.k, s.v, btable, s.d_writepos, 1,
                                         c.n_kv_heads, c.head_dim, s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
                    }
                }
            }

            // ---- attention (Q8-emit only when output is not gated: the gate mutates attn after decode) ----
            const bool emit_attn_q8 = !w.q_has_gate && s.use_attnin && s.gguf && s.use_pq && s.use_llama && w.wo_type == 12;
            kernels::launch_flash_decode_split(s.q, kpool, vpool, btable, s.d_seqlen, s.attn,
                                               s.fa_m, s.fa_l, s.fa_acc, 1, c.n_q_heads, c.n_kv_heads, c.head_dim,
                                               s.kv->block_size(), s.kv->max_blocks_per_seq(), s.n_splits,
                                               1.f / sqrtf((float)c.head_dim), st,
                                               emit_attn_q8 ? s.aq81 : nullptr, seqlen, kscale, vscale, kv8 ? 1 : 0);
            if (w.q_has_gate)
                kernels::launch_qwen36_mul_sigmoid(s.attn, s.qgate, s.qdim, st);

            // ---- O projection (main's int8 mmvq path) ----
            if (s.gguf && s.use_pq && w.wo_type == 12) {
                if (s.use_llama) {
                    if (!emit_attn_q8) kernels::launch_quantize_q8_1_blocks(s.attn, s.aq81, s.qdim, st);
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
        }

        // fused: h = x + ao ; hn = RMSNorm(h, post_attn_norm). When fnq, also emit Q8_1(hn) into
        // aq81 so the MoE gate/up mmvq skips its own quantize node (the router below reads bf16 hn).
        if (fnq)
            kernels::launch_add_rmsnorm2_q8(s.x, s.ao, w.post_attn_norm, s.h, s.hn, s.aq81, H, c.rms_eps, st);
        else
            kernels::launch_add_rmsnorm2(s.x, s.ao, w.post_attn_norm, s.h, s.hn, 1, H, c.rms_eps, st);

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
                    kernels::launch_gemv(s.hn, w.shared_gate_inp, s.shared_gate_tmp, 1, H, s.stream_k);
                }
                kernels::launch_qwen36_sigmoid_scalar(s.shared_gate_tmp, s.d_shared_w, s.stream_k);
            }
            if (qmoe) {
                kernels::launch_shared_expert_q8_mmvq(
                    s.hn, fnq ? s.aq81 : nullptr,
                    w.shared_gate_q, w.shared_up_q, w.shared_down_q,
                    w.shared_gate_inp ? s.d_shared_w : nullptr,
                    s.shared, s.sx_h, s.sx_q8, H, c.moe_ffn, s.stream_k);
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

        if (w.gate_q) {   // GGUF fused: route, then dequant-on-read only the top_k experts
            kernels::launch_gemv_f32(s.hn, w.router_w, s.mf_logits, c.n_experts, c.hidden, st);  // router_w native [E,H]
            // The per-expert token counts only feed the batched-dispatch sort; the single-token
            // decode expert FFN reads ids/weights directly and never touches them. Zeroing that
            // buffer is a per-layer memset node in the replayed decode graph whose fixed cost far
            // outweighs the handful of atomics that fill it, so skip the count on this path.
            // SPARKINFER_MOE_COUNTS=1 restores the memset + on-device counting.
            static int moe_counts = -1;
            if (moe_counts < 0) { const char* mc = getenv("SPARKINFER_MOE_COUNTS"); moe_counts = (mc && mc[0] == '1') ? 1 : 0; }
            if (moe_counts) cu(cudaMemsetAsync(s.mf_counts, 0, c.n_experts * sizeof(int), st), "mf counts");
            kernels::launch_moe_router(s.mf_logits, s.mf_ids, s.mf_weights,
                                       moe_counts ? s.mf_counts : nullptr,
                                       1, c.n_experts, c.top_k, 1, st);
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
            if (shexp_pipelined) {
                cudaStreamWaitEvent(st, s.ev_sx_done, 0);
                const void* nextnorm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
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
                continue;
            }
            if (w.shared_gate_inp) {
                if (s.gguf) {
                    if (s.use_pq && w.shared_gate_inp_type == 12) {
                        if (s.use_llama) {
                            if (!fnq) kernels::launch_quantize_q8_1_blocks(s.hn, s.aq81, H, st);
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
                        kernels::launch_gemv(s.hn, w.shared_gate_inp, s.shared_gate_tmp, 1, H, st);
                    }
                } else {
                    kernels::launch_gemm(s.hn, w.shared_gate_inp, s.shared_gate_tmp, 1, 1, H, 1.f, 0.f, gc, st);
                }
                kernels::launch_qwen36_sigmoid_scalar(s.shared_gate_tmp, s.d_shared_w, st);
            }
            if (s.gguf) {
                if (qmoe) {
                    kernels::launch_shared_expert_q8_mmvq(
                        s.hn, fnq ? s.aq81 : nullptr,
                        w.shared_gate_q, w.shared_up_q, w.shared_down_q,
                        w.shared_gate_inp ? s.d_shared_w : nullptr,
                        s.shared, s.mf_h, s.aq81, H, c.moe_ffn, st);
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
        if (shared_to_fold) {
            if (fnq)
                kernels::launch_add_rmsnorm3_q8(s.h, s.routed, shared_to_fold, nextnorm, s.x, s.xn, s.aq81, H, c.rms_eps, st);
            else
                kernels::launch_add_rmsnorm3(s.h, s.routed, shared_to_fold, nextnorm, s.x, s.xn, 1, H, c.rms_eps, st);
        } else if (fnq)
            kernels::launch_add_rmsnorm2_q8(s.h, s.routed, nextnorm, s.x, s.xn, s.aq81, H, c.rms_eps, st);
        else
            kernels::launch_add_rmsnorm2(s.h, s.routed, nextnorm, s.x, s.xn, 1, H, c.rms_eps, st);
    }
    // xn now holds RMSNorm(x_final, final_norm)
    if (s.gguf && s.use_q6mmvq && s.w.lm_head_type == 14) {   // int8 Q6_K dp4a LM head (1 warp/row)
        if (!fnq) kernels::launch_quantize_q8_1_blocks(s.xn, s.aq81, H, st);  // else aq81 = Q8_1(xn) from final norm
        kernels::launch_gemv_q6k_dp4a_f32(s.aq81, s.w.lm_head, s.logits, c.vocab, H, st);
    }
    else if (s.gguf && s.w.lm_head_type) kernels::launch_gemv_q_f32(s.xn, s.w.lm_head, s.w.lm_head_type, s.logits, c.vocab, H, st);
    else if (s.gguf)                kernels::launch_gemv_f32(s.xn, s.w.lm_head, s.logits, c.vocab, H, st);  // lm_head native [vocab,H]
    else        kernels::launch_linear_f32(s.xn, s.w.lm_head, s.logits, 1, c.vocab, H, st);
    kernels::launch_argmax(s.logits, s.d_out_id, 1, c.vocab, st);
    if (s.bench_feedback_graph) kernels::launch_decode_feedback(s.d_scalars, s.d_out_id, st);

    cu(cudaStreamEndCapture(st, &s.cu_graph), "end capture");
    cu(cudaGraphInstantiate(&s.cu_exec, s.cu_graph, 0), "graph instantiate");
    s.graph_ready = true;
    cu(cudaGraphLaunch(s.cu_exec, st), "graph launch (first)");

    cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "out_id");
    cu(cudaStreamSynchronize(st), "sync");
    return *s.h_out_id;
}

double Qwen35Model::bench_decode(int warmup, int n, int context_tokens) {
    Impl& s = *p_;
    if (!s.kv->allocate(s.seq_id, s.cfg.max_seq)) { fprintf(stderr, "[bench] kv allocate failed\n"); return -1; }
    int start_pos = context_tokens;
    if (const char* e = getenv("SPARKINFER_BENCH_START_POS")) {
        start_pos = atoi(e);
    }
    if (start_pos < 0) start_pos = 0;
    if (start_pos + warmup + n > s.cfg.max_seq) {
        fprintf(stderr, "[bench] requested ctx=%d warmup=%d n=%d exceeds max_seq=%d\n",
                start_pos, warmup, n, s.cfg.max_seq);
        s.kv->free(s.seq_id);
        return -1;
    }
    static int bench_device_loop = -1;
    if (bench_device_loop < 0) {
        const char* e = getenv("SPARKINFER_BENCH_DEVICE_LOOP");
        bench_device_loop = (e && e[0] == '0') ? 0 : 1;
    }
    s.bench_feedback_graph = bench_device_loop != 0;
    int pos = 0, tok = 100;
    for (; pos < start_pos; pos++) { tok = forward_token(tok, pos); if (tok < 0 || tok >= s.cfg.vocab) tok = 100; }
    for (int i = 0; i < warmup; i++) { tok = forward_token(tok, pos++); if (tok < 0 || tok >= s.cfg.vocab) tok = 100; }
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
        s.kv->free(s.seq_id);
        s.bench_feedback_graph = false;
        double secs = std::chrono::duration<double>(t1 - t0).count();
        return n / secs;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) { tok = forward_token(tok, pos++); if (tok < 0 || tok >= s.cfg.vocab) tok = 100; }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    s.kv->free(s.seq_id);
    s.bench_feedback_graph = false;
    double secs = std::chrono::duration<double>(t1 - t0).count();
    return n / secs;
}

std::vector<int> Qwen35Model::generate(const std::vector<int>& prompt, int max_new, ThermalGovernor* gov) {
    Impl& s = *p_;
    std::vector<int> out;
    if (prompt.empty()) return out;
    if (!s.kv->allocate(s.seq_id, s.cfg.max_seq)) {
        fprintf(stderr, "[qwen35] KV allocate failed (pool too small for max_seq=%d)\n", s.cfg.max_seq);
        return out;
    }
    int next = -1;
    for (size_t i = 0; i < prompt.size(); i++) next = forward_token(prompt[i], (int)i);
    for (int i = 0; i < max_new; i++) {
        out.push_back(next);
        if (next == s.cfg.eos_id) break;
        next = forward_token(next, (int)prompt.size() + i);
        if (gov) gov->pace();   // thermally-adaptive decode pacing (accuracy-preserving; no-op if disabled)
    }
    s.kv->free(s.seq_id);
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
    const bool hybrid_file = is_qwen35_or_qwen36_hybrid_moe(g);
    if (hybrid_file && !s.cfg.hybrid) {
        fprintf(stderr,
                "[qwen35] Qwen3.5/Qwen3.6 hybrid GGUF requires constructing "
                "Qwen35Model with cfg.hybrid=true and the GGUF metadata-derived "
                "head dimensions before load_gguf(), so scratch buffers and KV "
                "cache are sized correctly.\n");
        return false;
    }
    if (hybrid_file) {
        s.cfg.hybrid = true;
        if (s.cfg.full_attn_interval <= 0) s.cfg.full_attn_interval = 4;
        if (s.cfg.rope_dim <= 0 && s.cfg.head_dim == 256) s.cfg.rope_dim = 64;
        if (s.cfg.linear_q_heads <= 0) s.cfg.linear_q_heads = 16;
        if (s.cfg.linear_v_heads <= 0) s.cfg.linear_v_heads = 32;
        if (s.cfg.linear_head_dim <= 0) s.cfg.linear_head_dim = 128;
        if (s.cfg.linear_conv_kernel <= 0) s.cfg.linear_conv_kernel = 4;
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
    auto attn_w = [&](const std::string& name, int& type) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        if (qattn && t && (t->ggml_type == 12 || t->ggml_type == 14 || t->ggml_type == 8)) return dev_quant(name, type);
        type = 0; return dense(name, false);
    };
    auto attn_w_opt = [&](const std::string& name, int& type) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        if (!t) { type = 0; return nullptr; }
        if (qattn && (t->ggml_type == 12 || t->ggml_type == 14 || t->ggml_type == 8)) return dev_quant(name, type);
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
    s.w.lm_head = attn_w(lm, s.w.lm_head_type);               // native [vocab,hidden] for GEMV
    if (!s.w.embed_tokens || !s.w.final_norm || !s.w.lm_head) return false;

    s.w.layers.resize(c.n_layers);
    for (int i = 0; i < c.n_layers; i++) {
        std::string b = "blk." + std::to_string(i) + ".";
        Qwen35LayerWeights& w = s.w.layers[i];
        w.linear_attn = is_linear_layer(c, i);
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
        if (!expect_dims_opt(b + "attn_post_norm.weight", {H}) ||
            !expect_dims_opt(b + "post_attention_norm.weight", {H}) ||
            !expect_dims_opt(b + "ffn_norm.weight", {H}) ||
            !expect_dims(b + "ffn_gate_inp.weight", {H, c.n_experts})) return false;
        // pre-MoE norm: Qwen3-MoE = "ffn_norm"; Qwen3.6 GGUFs name it "post_attention_norm".
        w.post_attn_norm = dense_opt(b + "attn_post_norm.weight", false);
        if (!w.post_attn_norm) w.post_attn_norm = dense_opt(b + "post_attention_norm.weight", false);
        if (!w.post_attn_norm) w.post_attn_norm = dense(b + "ffn_norm.weight", false);
        w.router_w = dense(b + "ffn_gate_inp.weight", false);   // native [E,H] for GEMV
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
        const bool have_attn = w.linear_attn
            ? (w.wqkv && w.wqkv_gate && w.ssm_conv && w.ssm_dt && w.ssm_a &&
               w.ssm_beta && w.ssm_alpha && w.ssm_norm && w.ssm_out)
            : (w.wq && w.wk && w.wv && w.wo && w.q_norm && w.k_norm);
        if (!have_attn || !w.input_norm || !w.post_attn_norm ||
            !w.router_w || !w.gate_q || !w.up_q || !w.down_q) return false;
        if (i == 0 || i == c.n_layers - 1) fprintf(stderr, "[gguf] layer %d loaded\n", i);
    }
    // decode scratch (mf_* / fa_*) is allocated in the constructor for all paths.
    return true;
}

} // namespace sparkinfer
