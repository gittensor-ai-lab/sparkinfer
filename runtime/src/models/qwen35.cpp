// Qwen3.5-35B-A3B single-sequence greedy decoder.
//
// Per token: embed -> [40x Qwen layer] -> final RMSNorm -> LM head -> argmax.
// Qwen layer: RMSNorm -> Q/K/V -> per-head QK-norm -> RoPE -> KV append ->
//             GQA flash decode -> O-proj -> residual -> RMSNorm ->
//             routed top-8 MoE (+ shared expert) -> residual.
// All steps run on one stream; only the sampled id is copied to the host, which
// autoregressive greedy decoding fundamentally requires.

#include "sparkinfer/models/qwen35.h"
#include "sparkinfer/kv_ops.h"
#include "sparkinfer/gguf.h"
#include "sparkinfer/spec_decode.h"
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
        case 14: // Q6_K
            return true;
        default:
            return false;
    }
}
}

struct Qwen35Model::Impl {
    Qwen35Config cfg;
    KVCacheManager* kv;
    moe::MoEEngine* engine;
    Qwen35Weights w;
    cudaStream_t stream{};
    uint64_t seq_id = 0;
    int qdim, kvdim;
    bool gguf = false;   // true after load_gguf: dense weights are native [out,in], use GEMV
    // CUDA-graph capture of the decode compute (captured once, replayed each token)
    cudaGraph_t cu_graph{};
    cudaGraphExec_t cu_exec{};
    bool graph_ready = false;

    // scratch (bf16)
    bf16 *x, *xn, *q, *k, *v, *attn, *ao, *h, *hn, *routed, *shared;
    float* logits;
    int *d_tok, *d_out_id, *d_pos, *d_seqlen, *d_writepos, *d_shared_ids;
    float* d_shared_w;
    std::vector<void*> owned;   // device buffers from load_weights / load_gguf
    // GGUF fused-expert decode scratch (allocated by load_gguf)
    float *mf_logits = nullptr, *mf_weights = nullptr, *mf_h = nullptr, *mf_out = nullptr;
    int   *mf_ids = nullptr, *mf_counts = nullptr;
    // flash-decoding (KV-split) attention partials
    int n_splits = 32;
    float *fa_m = nullptr, *fa_l = nullptr, *fa_acc = nullptr;
    // pre-quantized Q8_1 activation (computed once per projection input, shared across Q/K/V)
    signed char* aq8 = nullptr; float *aq8_d = nullptr, *aq8_s = nullptr;
    bool use_pq = true;   // SPARKINFER_PQ=0 disables the pre-quantized GEMV path
    void* aq81 = nullptr; // block_q8_1 activation for the faithful llama mmvq port
    bool use_llama = true; // default ON: faithful llama mmvq for Q4_K attn GEMVs (+9.7%, top1 0.99). =0 disables
    bool use_q6mmvq = true;  // default ON: int8 Q6_K mmvq for attn-V upgrades + LM head. =0 disables

    template <class T> T* alloc(size_t n) { void* p=nullptr; cu(cudaMalloc(&p, n*sizeof(T)), "malloc"); return (T*)p; }

    // --- speculative decoding (opt-in; see enable_speculative / forward_tokens_batch) ---
    // Batched verification scratch, fixed-width K = spec_k. The single-token
    // path above (x/xn/q/k/.../graph_ready) is untouched by any of this.
    int spec_k = 0;
    bool spec_ready = false;
    cudaGraph_t spec_graph{};
    cudaGraphExec_t spec_exec{};
    bool spec_graph_ready = false;
    bf16 *sk_x=nullptr, *sk_xn=nullptr, *sk_q=nullptr, *sk_k=nullptr, *sk_v=nullptr, *sk_attn=nullptr,
         *sk_ao=nullptr, *sk_h=nullptr, *sk_hn=nullptr, *sk_routed=nullptr, *sk_shared=nullptr;
    float* sk_logits = nullptr;
    int *sk_tok=nullptr, *sk_pos=nullptr, *sk_seqlen=nullptr, *sk_out_id=nullptr;
    int* sk_shared_ids = nullptr; float* sk_shared_w = nullptr;
    float *sk_mf_logits=nullptr, *sk_mf_weights=nullptr, *sk_mf_h=nullptr, *sk_mf_out=nullptr;
    int *sk_mf_ids=nullptr, *sk_mf_counts=nullptr;
    // Per-layer transposed copies of GGUF-native [out,in] dense weights, used
    // only by the batched verification path (forward_token's GEMV-on-read
    // path keeps using the original [out,in] copies). Empty for the non-GGUF
    // (load_weights) path, which already stores dense weights [in,out].
    std::vector<const void*> spec_wq, spec_wk, spec_wv, spec_wo, spec_router_w;
    const void* spec_lm_head = nullptr;
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
        int v = atoi(ns); if (v < 1) v = 1; if (v > 64) v = 64; p_->n_splits = v;
        fprintf(stderr, "[nsplits] flash-decode splits = %d (env override)\n", v);
    }
    p_->qdim = cfg.n_q_heads * cfg.head_dim;
    p_->kvdim = cfg.n_kv_heads * cfg.head_dim;
    cudaStreamCreate(&p_->stream);
    const int H = cfg.hidden;
    p_->x=p_->alloc<bf16>(H); p_->xn=p_->alloc<bf16>(H);
    p_->q=p_->alloc<bf16>(p_->qdim); p_->k=p_->alloc<bf16>(p_->kvdim); p_->v=p_->alloc<bf16>(p_->kvdim);
    p_->attn=p_->alloc<bf16>(p_->qdim); p_->ao=p_->alloc<bf16>(H);
    p_->h=p_->alloc<bf16>(H); p_->hn=p_->alloc<bf16>(H);
    p_->routed=p_->alloc<bf16>(H); p_->shared=p_->alloc<bf16>(H);
    p_->logits=p_->alloc<float>(cfg.vocab);
    p_->d_tok=p_->alloc<int>(1); p_->d_out_id=p_->alloc<int>(1);
    p_->d_pos=p_->alloc<int>(1); p_->d_seqlen=p_->alloc<int>(1); p_->d_writepos=p_->alloc<int>(1);
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
    const size_t fa_n = (size_t)cfg.n_q_heads * p_->n_splits;
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
}

Qwen35Model::~Qwen35Model() {
    for (void* b : p_->owned) cudaFree(b);
    cudaFree(p_->x); cudaFree(p_->xn); cudaFree(p_->q); cudaFree(p_->k); cudaFree(p_->v);
    cudaFree(p_->attn); cudaFree(p_->ao); cudaFree(p_->h); cudaFree(p_->hn);
    cudaFree(p_->routed); cudaFree(p_->shared); cudaFree(p_->logits);
    cudaFree(p_->d_tok); cudaFree(p_->d_out_id); cudaFree(p_->d_pos);
    cudaFree(p_->d_seqlen); cudaFree(p_->d_writepos); cudaFree(p_->d_shared_ids); cudaFree(p_->d_shared_w);
    cudaFree(p_->mf_logits); cudaFree(p_->mf_weights); cudaFree(p_->mf_h); cudaFree(p_->mf_out);
    cudaFree(p_->mf_ids); cudaFree(p_->mf_counts);
    cudaFree(p_->fa_m); cudaFree(p_->fa_l); cudaFree(p_->fa_acc);
    cudaFree(p_->aq8); cudaFree(p_->aq8_d); cudaFree(p_->aq8_s); cudaFree(p_->aq81);
    if (p_->graph_ready) { cudaGraphExecDestroy(p_->cu_exec); cudaGraphDestroy(p_->cu_graph); }
    if (p_->spec_graph_ready) { cudaGraphExecDestroy(p_->spec_exec); cudaGraphDestroy(p_->spec_graph); }
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

    cu(cudaMemcpyAsync(s.d_tok, &token_id, sizeof(int), cudaMemcpyHostToDevice, st), "tok");
    cu(cudaMemcpyAsync(s.d_pos, &position, sizeof(int), cudaMemcpyHostToDevice, st), "pos");
    cu(cudaMemcpyAsync(s.d_writepos, &position, sizeof(int), cudaMemcpyHostToDevice, st), "wpos");
    cu(cudaMemcpyAsync(s.d_seqlen, &seqlen, sizeof(int), cudaMemcpyHostToDevice, st), "slen");

    // Capture the decode compute into a CUDA graph on the first token, then
    // replay it every token (per-token inputs live in the d_tok/pos/seqlen/
    // writepos device buffers uploaded above, so replay produces fresh results).
    if (s.graph_ready) {
        cu(cudaGraphLaunch(s.cu_exec, st), "graph launch");
        int out_id = 0;
        cu(cudaMemcpyAsync(&out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "out_id");
        cu(cudaStreamSynchronize(st), "sync");
        return out_id;
    }
    cu(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin capture");

    kernels::launch_embedding(s.d_tok, s.w.embed_tokens, s.x, 1, H, st);

    int* btable = s.kv->block_table(s.seq_id);
    // Prime: xn = RMSNorm(x, layer0.input_norm). Each layer's tail then fuses the
    // post-MoE residual with the NEXT layer's input norm (or final_norm), so the
    // per-layer input RMSNorm + two residual-adds collapse into two fused kernels.
    kernels::launch_rmsnorm(s.x, s.w.layers[0].input_norm, s.xn, 1, H, c.rms_eps, st);

    for (int L = 0; L < c.n_layers; L++) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        if (s.gguf) {   // GGUF dense weights are native [out,in] -> coalesced GEMV
            // Q/K/V all read xn: quantize it to Q8_1 ONCE, then dp4a each Q4_K proj against it
            // (no per-block, per-GEMV re-quant). Q6_K/bf16 weights keep their existing path.
            const bool any_q4k = (w.wq_type == 12 || w.wk_type == 12 || w.wv_type == 12);
            const bool any_q6k = (w.wq_type == 14 || w.wk_type == 14 || w.wv_type == 14);
            if (s.use_pq && s.use_llama && (any_q4k || any_q6k))
                kernels::launch_quantize_q8_1_blocks(s.xn, s.aq81, H, st);   // shared Q8_1(xn) for Q4_K + Q6_K mmvq
            else if (s.use_pq && any_q4k)
                kernels::launch_quantize_q8_1(s.xn, s.aq8, s.aq8_d, s.aq8_s, H, st);
            auto proj = [&](const void* W, int t, void* y, int N) {
                if (s.use_pq && t == 12) {
                    if (s.use_llama) kernels::launch_mmvq_q4k(s.aq81, W, y, N, H, st);
                    else             kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s, W, y, N, H, st);
                }
                else if (s.use_q6mmvq && t == 14)
                    kernels::launch_mmvq_q6k(s.aq81, W, y, N, H, st);        // Q6_K mmvq (attn-V upgrades); reuses aq81

                else if (t) kernels::launch_gemv_q(s.xn, W, t, y, N, H, st);
                else        kernels::launch_gemv(s.xn, W, y, N, H, st);
            };
            proj(w.wq, w.wq_type, s.q, s.qdim);
            proj(w.wk, w.wk_type, s.k, s.kvdim);
            proj(w.wv, w.wv_type, s.v, s.kvdim);
        } else {
            kernels::launch_gemm(s.xn, w.wq, s.q, 1, s.qdim,  H, 1.f, 0.f, gc, st);
            kernels::launch_gemm(s.xn, w.wk, s.k, 1, s.kvdim, H, 1.f, 0.f, gc, st);
            kernels::launch_gemm(s.xn, w.wv, s.v, 1, s.kvdim, H, 1.f, 0.f, gc, st);
        }
        kernels::launch_rmsnorm(s.q, w.q_norm, s.q, c.n_q_heads,  c.head_dim, c.rms_eps, st);
        kernels::launch_rmsnorm(s.k, w.k_norm, s.k, c.n_kv_heads, c.head_dim, c.rms_eps, st);
        kernels::launch_rope(s.q, s.k, s.d_pos, 1, c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta, st);

        bf16* kpool = (bf16*)s.kv->k_pool() + (size_t)L * s.kv->layer_stride_elems();
        bf16* vpool = (bf16*)s.kv->v_pool() + (size_t)L * s.kv->layer_stride_elems();
        launch_kv_append(kpool, vpool, s.k, s.v, btable, s.d_writepos, 1,
                         c.n_kv_heads, c.head_dim, s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
        kernels::launch_flash_decode_split(s.q, kpool, vpool, btable, s.d_seqlen, s.attn,
                                           s.fa_m, s.fa_l, s.fa_acc, 1, c.n_q_heads, c.n_kv_heads, c.head_dim,
                                           s.kv->block_size(), s.kv->max_blocks_per_seq(), s.n_splits,
                                           1.f / sqrtf((float)c.head_dim), st);
        if (s.gguf && s.use_pq && w.wo_type == 12) {   // O proj reads attn: quantize once + dp4a
            if (s.use_llama) {
                kernels::launch_quantize_q8_1_blocks(s.attn, s.aq81, s.qdim, st);
                kernels::launch_mmvq_q4k(s.aq81, w.wo, s.ao, H, s.qdim, st);
            } else {
                kernels::launch_quantize_q8_1(s.attn, s.aq8, s.aq8_d, s.aq8_s, s.qdim, st);
                kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s, w.wo, s.ao, H, s.qdim, st);
            }
        }
        else if (s.gguf && w.wo_type) kernels::launch_gemv_q(s.attn, w.wo, w.wo_type, s.ao, H, s.qdim, st);
        else if (s.gguf)         kernels::launch_gemv(s.attn, w.wo, s.ao, H, s.qdim, st);
        else                     kernels::launch_gemm(s.attn, w.wo, s.ao, 1, H, s.qdim, 1.f, 0.f, gc, st);

        // fused: h = x + ao ; hn = RMSNorm(h, post_attn_norm)
        kernels::launch_add_rmsnorm2(s.x, s.ao, w.post_attn_norm, s.h, s.hn, 1, H, c.rms_eps, st);

        if (w.gate_q) {   // GGUF fused: route, then dequant-on-read only the top_k experts
            kernels::launch_gemv_f32(s.hn, w.router_w, s.mf_logits, c.n_experts, c.hidden, st);  // router_w native [E,H]
            cu(cudaMemsetAsync(s.mf_counts, 0, c.n_experts * sizeof(int), st), "mf counts");
            kernels::launch_moe_router(s.mf_logits, s.mf_ids, s.mf_weights, s.mf_counts,
                                       1, c.n_experts, c.top_k, 1, st);
            kernels::launch_moe_expert_ffn_q4k(s.hn, w.gate_q, w.up_q, w.down_q,
                                               w.gate_qtype, w.up_qtype, w.down_qtype,
                                               s.mf_ids, s.mf_weights, s.routed, s.mf_h, s.mf_out,
                                               1, c.top_k, c.hidden, c.moe_ffn, st);
        } else {
            s.engine->set_layer_weights(L, {w.router_w, w.gate, w.up, w.down});
            s.engine->forward(s.hn, s.routed, 1, L, st);
        }
        if (c.n_shared > 0) {
            kernels::launch_moe_expert_ffn(s.hn, w.shared_gate, w.shared_up, w.shared_down,
                                           s.d_shared_ids, s.d_shared_w, s.shared,
                                           1, 1, 1, H, c.moe_ffn, st);
            launch_residual_add(s.routed, s.shared, s.routed, H, st);
        }
        // fused: x = h + routed ; xn = RMSNorm(x, next input_norm or final_norm)
        const void* nextnorm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
        kernels::launch_add_rmsnorm2(s.h, s.routed, nextnorm, s.x, s.xn, 1, H, c.rms_eps, st);
    }
    // xn now holds RMSNorm(x_final, final_norm)
    if (s.gguf && s.use_q6mmvq && s.w.lm_head_type == 14) {   // int8 Q6_K dp4a LM head (1 warp/row)
        kernels::launch_quantize_q8_1_blocks(s.xn, s.aq81, H, st);
        kernels::launch_gemv_q6k_dp4a_f32(s.aq81, s.w.lm_head, s.logits, c.vocab, H, st);
    }
    else if (s.gguf && s.w.lm_head_type) kernels::launch_gemv_q_f32(s.xn, s.w.lm_head, s.w.lm_head_type, s.logits, c.vocab, H, st);
    else if (s.gguf)                kernels::launch_gemv_f32(s.xn, s.w.lm_head, s.logits, c.vocab, H, st);  // lm_head native [vocab,H]
    else        kernels::launch_linear_f32(s.xn, s.w.lm_head, s.logits, 1, c.vocab, H, st);
    kernels::launch_argmax(s.logits, s.d_out_id, 1, c.vocab, st);

    cu(cudaStreamEndCapture(st, &s.cu_graph), "end capture");
    cu(cudaGraphInstantiate(&s.cu_exec, s.cu_graph, 0), "graph instantiate");
    s.graph_ready = true;
    cu(cudaGraphLaunch(s.cu_exec, st), "graph launch (first)");

    int out_id = 0;
    cu(cudaMemcpyAsync(&out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "out_id");
    cu(cudaStreamSynchronize(st), "sync");
    return out_id;
}

double Qwen35Model::bench_decode(int warmup, int n) {
    Impl& s = *p_;
    if (!s.kv->allocate(s.seq_id, s.cfg.max_seq)) { fprintf(stderr, "[bench] kv allocate failed\n"); return -1; }
    int pos = 0, tok = 100;
    for (int i = 0; i < warmup; i++) { tok = forward_token(tok, pos++); if (tok < 0 || tok >= s.cfg.vocab) tok = 100; }
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) { tok = forward_token(tok, pos++); if (tok < 0 || tok >= s.cfg.vocab) tok = 100; }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    s.kv->free(s.seq_id);
    double secs = std::chrono::duration<double>(t1 - t0).count();
    return n / secs;
}

std::vector<int> Qwen35Model::generate(const std::vector<int>& prompt, int max_new) {
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
    const Qwen35Config& c = s.cfg;
    s.gguf = true;   // dense weights kept native [out,in]; forward uses GEMV
    GGUF g;
    if (!g.open(path)) return false;

    // Shared-expert tensors are optional in GGUF (Qwen3-30B-A3B has none). The
    // default config sets n_shared=1, so clamp it to what the file actually
    // contains before forward_token can launch a null-weight FFN.
    const bool gguf_has_shared =
        g.tensor("blk.0.ffn_gate_shexp.weight") != nullptr;
    if (c.n_shared > 0 && !gguf_has_shared) {
        fprintf(stderr,
                "[gguf] no shared-expert tensors; forcing n_shared=0 "
                "(safe for models without a shared FFN)\n");
        s.cfg.n_shared = 0;
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
        if (qattn && t && (t->ggml_type == 12 || t->ggml_type == 14)) return dev_quant(name, type);
        type = 0; return dense(name, false);
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
        w.input_norm = dense(b + "attn_norm.weight", false);
        w.wq = attn_w(b + "attn_q.weight", w.wq_type); w.wk = attn_w(b + "attn_k.weight", w.wk_type);
        w.wv = attn_w(b + "attn_v.weight", w.wv_type); w.wo = attn_w(b + "attn_output.weight", w.wo_type);
        w.q_norm = dense(b + "attn_q_norm.weight", false); w.k_norm = dense(b + "attn_k_norm.weight", false);
        w.post_attn_norm = dense(b + "ffn_norm.weight", false);
        w.router_w = dense(b + "ffn_gate_inp.weight", false);   // native [E,H] for GEMV
        w.gate_q = dev_quant(b + "ffn_gate_exps.weight", w.gate_qtype);   // kept quantized
        w.up_q   = dev_quant(b + "ffn_up_exps.weight",   w.up_qtype);
        w.down_q = dev_quant(b + "ffn_down_exps.weight", w.down_qtype);
        if (s.cfg.n_shared > 0) {
            w.shared_gate = dense(b + "ffn_gate_shexp.weight", true);
            w.shared_up   = dense(b + "ffn_up_shexp.weight", true);
            w.shared_down = dense(b + "ffn_down_shexp.weight", true);
            if (!w.shared_gate || !w.shared_up || !w.shared_down) return false;
        }
        if (!w.wq || !w.router_w || !w.gate_q || !w.up_q || !w.down_q) return false;
        if (i == 0 || i == c.n_layers - 1) fprintf(stderr, "[gguf] layer %d loaded\n", i);
    }
    // decode scratch (mf_* / fa_*) is allocated in the constructor for all paths.
    return true;
}

// ----- speculative decoding: batched verification + prompt-lookup draft loop -----
//
// forward_token() above is single-token (batch=1) and CUDA-graph-captured for
// that fixed shape. Speculative decoding needs a *second*, separately-captured
// graph for a fixed verification width K: draft up to K-1 tokens, run all K
// through one batched forward pass, and accept the longest prefix that agrees
// with the real (greedy) model output. Every kernel below already takes a
// batch/row count in its signature (it was just always called with 1); the one
// piece that does NOT already exist is a batched GEMM over GGUF's native
// [out,in] dense weight layout (only a single-row GEMV-on-read kernel does),
// so enable_speculative() pre-transposes one extra [in,out] bf16 copy per dense
// weight, once, and forward_tokens_batch() uses the existing launch_gemm /
// launch_linear_f32 / launch_moe_router_gemm against those copies. The
// per-token attention step still appends KV and runs flash-decode one row at a
// time (sequentially dependent within a layer -- token i's keys must already
// be in the cache before token i+1 can attend to them), but that is cheap
// relative to the weight reads the batching is meant to amortize.
//
// KV cache entries are appended for all K speculative positions whether or
// not they end up accepted; no rollback is needed because future calls always
// pass the correct (smaller) seq_len, so a rejected tail is simply never read,
// and gets overwritten the next time a real token lands on that position.

void Qwen35Model::enable_speculative(int k) {
    Impl& s = *p_;
    if (s.spec_ready) return;   // idempotent; k is fixed for the lifetime of the graph
    if (k < 2) { fprintf(stderr, "[qwen35] enable_speculative: k must be >= 2\n"); return; }
    const Qwen35Config& c = s.cfg;
    const int H = c.hidden;

    if (s.gguf) {
        for (const auto& w : s.w.layers) {
            if (w.wq_type || w.wk_type || w.wv_type || w.wo_type) {
                fprintf(stderr, "[qwen35] enable_speculative: SPARKINFER_QATTN on-read-quantized "
                                "attention weights are not supported by the batched verification path "
                                "yet; reload without SPARKINFER_QATTN to use speculative decoding.\n");
                return;
            }
        }
        if (s.w.lm_head_type) {
            fprintf(stderr, "[qwen35] enable_speculative: quantized lm_head is not supported by the "
                            "batched verification path yet.\n");
            return;
        }
    }

    auto alloc = [&](size_t bytes) { void* p=nullptr; cu(cudaMalloc(&p, bytes), "spec malloc"); s.owned.push_back(p); return p; };
    s.sk_x      = (bf16*)alloc((size_t)k*H*2);          s.sk_xn     = (bf16*)alloc((size_t)k*H*2);
    s.sk_q      = (bf16*)alloc((size_t)k*s.qdim*2);     s.sk_k      = (bf16*)alloc((size_t)k*s.kvdim*2);
    s.sk_v      = (bf16*)alloc((size_t)k*s.kvdim*2);    s.sk_attn   = (bf16*)alloc((size_t)k*s.qdim*2);
    s.sk_ao     = (bf16*)alloc((size_t)k*H*2);          s.sk_h      = (bf16*)alloc((size_t)k*H*2);
    s.sk_hn     = (bf16*)alloc((size_t)k*H*2);          s.sk_routed = (bf16*)alloc((size_t)k*H*2);
    s.sk_shared = (bf16*)alloc((size_t)k*H*2);
    s.sk_logits = (float*)alloc((size_t)k*c.vocab*sizeof(float));
    s.sk_tok    = (int*)alloc((size_t)k*sizeof(int));   s.sk_pos    = (int*)alloc((size_t)k*sizeof(int));
    s.sk_seqlen = (int*)alloc((size_t)k*sizeof(int));   s.sk_out_id = (int*)alloc((size_t)k*sizeof(int));
    s.sk_shared_ids = (int*)alloc((size_t)k*sizeof(int));
    s.sk_shared_w   = (float*)alloc((size_t)k*sizeof(float));
    s.sk_mf_logits  = (float*)alloc((size_t)k*c.n_experts*sizeof(float));
    s.sk_mf_ids     = (int*)alloc((size_t)k*c.top_k*sizeof(int));
    s.sk_mf_weights = (float*)alloc((size_t)k*c.top_k*sizeof(float));
    s.sk_mf_counts  = (int*)alloc((size_t)c.n_experts*sizeof(int));
    s.sk_mf_h       = (float*)alloc((size_t)k*c.top_k*c.moe_ffn*sizeof(float));
    s.sk_mf_out     = (float*)alloc((size_t)k*H*sizeof(float));

    std::vector<int> zeros(k, 0);
    std::vector<float> ones(k, 1.f);
    cu(cudaMemcpy(s.sk_shared_ids, zeros.data(), k*sizeof(int), cudaMemcpyHostToDevice), "sk shared ids");
    cu(cudaMemcpy(s.sk_shared_w, ones.data(), k*sizeof(float), cudaMemcpyHostToDevice), "sk shared w");

    if (s.gguf) {
        // [out,in] -> [in,out], once per dense weight, so the batched path can
        // use the plain row-major GEMM kernel (see launch_gemm's [K,N] B layout).
        auto transpose = [&](const void* src, int out, int in) -> const void* {
            void* dst = alloc((size_t)out*in*2);
            kernels::launch_transpose_bf16(src, dst, out, in, s.stream);
            return dst;
        };
        s.spec_wq.resize(c.n_layers); s.spec_wk.resize(c.n_layers);
        s.spec_wv.resize(c.n_layers); s.spec_wo.resize(c.n_layers);
        s.spec_router_w.resize(c.n_layers);
        for (int L = 0; L < c.n_layers; L++) {
            const Qwen35LayerWeights& w = s.w.layers[L];
            s.spec_wq[L]        = transpose(w.wq,        s.qdim,      H);
            s.spec_wk[L]        = transpose(w.wk,        s.kvdim,     H);
            s.spec_wv[L]        = transpose(w.wv,        s.kvdim,     H);
            s.spec_wo[L]        = transpose(w.wo,        H,           s.qdim);
            s.spec_router_w[L]  = transpose(w.router_w,  c.n_experts, H);
        }
        s.spec_lm_head = transpose(s.w.lm_head, c.vocab, H);
        cu(cudaStreamSynchronize(s.stream), "spec transpose sync");
    }

    s.spec_k = k;
    s.spec_ready = true;
}

bool Qwen35Model::forward_tokens_batch(const int* tokens, int n, int start_position, int* out_argmax) {
    Impl& s = *p_;
    if (!s.spec_ready || n != s.spec_k) {
        fprintf(stderr, "[qwen35] forward_tokens_batch: enable_speculative(%d) not called, or n=%d mismatch\n", n, n);
        return false;
    }
    const Qwen35Config& c = s.cfg;
    const int H = c.hidden, K = s.spec_k;
    kernels::GemmConfig gc{};
    cudaStream_t st = s.stream;

    std::vector<int> h_pos(K), h_seqlen(K);
    for (int i = 0; i < K; i++) { h_pos[i] = start_position + i; h_seqlen[i] = start_position + i + 1; }
    cu(cudaMemcpyAsync(s.sk_tok, tokens, K*sizeof(int), cudaMemcpyHostToDevice, st), "sk tok");
    cu(cudaMemcpyAsync(s.sk_pos, h_pos.data(), K*sizeof(int), cudaMemcpyHostToDevice, st), "sk pos");
    cu(cudaMemcpyAsync(s.sk_seqlen, h_seqlen.data(), K*sizeof(int), cudaMemcpyHostToDevice, st), "sk seqlen");

    if (s.spec_graph_ready) {
        cu(cudaGraphLaunch(s.spec_exec, st), "spec graph launch");
        cu(cudaMemcpyAsync(out_argmax, s.sk_out_id, K*sizeof(int), cudaMemcpyDeviceToHost, st), "sk out");
        cu(cudaStreamSynchronize(st), "sk sync");
        return true;
    }
    cu(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "sk begin capture");

    int* btable = s.kv->block_table(s.seq_id);
    kernels::launch_embedding(s.sk_tok, s.w.embed_tokens, s.sk_x, K, H, st);
    kernels::launch_rmsnorm(s.sk_x, s.w.layers[0].input_norm, s.sk_xn, K, H, c.rms_eps, st);

    for (int L = 0; L < c.n_layers; L++) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        const void* wq_g = s.gguf ? s.spec_wq[L] : w.wq;
        const void* wk_g = s.gguf ? s.spec_wk[L] : w.wk;
        const void* wv_g = s.gguf ? s.spec_wv[L] : w.wv;
        const void* wo_g = s.gguf ? s.spec_wo[L] : w.wo;
        kernels::launch_gemm(s.sk_xn, wq_g, s.sk_q, K, s.qdim,  H, 1.f, 0.f, gc, st);
        kernels::launch_gemm(s.sk_xn, wk_g, s.sk_k, K, s.kvdim, H, 1.f, 0.f, gc, st);
        kernels::launch_gemm(s.sk_xn, wv_g, s.sk_v, K, s.kvdim, H, 1.f, 0.f, gc, st);

        kernels::launch_rmsnorm(s.sk_q, w.q_norm, s.sk_q, K * c.n_q_heads,  c.head_dim, c.rms_eps, st);
        kernels::launch_rmsnorm(s.sk_k, w.k_norm, s.sk_k, K * c.n_kv_heads, c.head_dim, c.rms_eps, st);
        kernels::launch_rope(s.sk_q, s.sk_k, s.sk_pos, K, c.n_q_heads, c.n_kv_heads, c.head_dim, c.rope_theta, st);

        bf16* kpool = (bf16*)s.kv->k_pool() + (size_t)L * s.kv->layer_stride_elems();
        bf16* vpool = (bf16*)s.kv->v_pool() + (size_t)L * s.kv->layer_stride_elems();
        // Sequential across the K speculative positions within this layer:
        // token i's K/V must be in the cache before token i+1 attends to it.
        // Cheap relative to the batched weight reads above/below.
        for (int i = 0; i < K; i++) {
            launch_kv_append(kpool, vpool, s.sk_k + (size_t)i*s.kvdim, s.sk_v + (size_t)i*s.kvdim,
                             btable, s.sk_pos + i, 1, c.n_kv_heads, c.head_dim,
                             s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
            kernels::launch_flash_decode_split(s.sk_q + (size_t)i*s.qdim, kpool, vpool, btable, s.sk_seqlen + i,
                                               s.sk_attn + (size_t)i*s.qdim, s.fa_m, s.fa_l, s.fa_acc,
                                               1, c.n_q_heads, c.n_kv_heads, c.head_dim,
                                               s.kv->block_size(), s.kv->max_blocks_per_seq(), s.n_splits,
                                               1.f / sqrtf((float)c.head_dim), st);
        }
        kernels::launch_gemm(s.sk_attn, wo_g, s.sk_ao, K, H, s.qdim, 1.f, 0.f, gc, st);
        kernels::launch_add_rmsnorm2(s.sk_x, s.sk_ao, w.post_attn_norm, s.sk_h, s.sk_hn, K, H, c.rms_eps, st);

        if (w.gate_q) {
            const void* router_g = s.gguf ? s.spec_router_w[L] : w.router_w;
            kernels::launch_moe_router_gemm(s.sk_hn, router_g, s.sk_mf_logits, K, c.hidden, c.n_experts, st);
            cu(cudaMemsetAsync(s.sk_mf_counts, 0, c.n_experts * sizeof(int), st), "sk mf counts");
            kernels::launch_moe_router(s.sk_mf_logits, s.sk_mf_ids, s.sk_mf_weights, s.sk_mf_counts,
                                       K, c.n_experts, c.top_k, 1, st);
            kernels::launch_moe_expert_ffn_q4k(s.sk_hn, w.gate_q, w.up_q, w.down_q,
                                               w.gate_qtype, w.up_qtype, w.down_qtype,
                                               s.sk_mf_ids, s.sk_mf_weights, s.sk_routed, s.sk_mf_h, s.sk_mf_out,
                                               K, c.top_k, c.hidden, c.moe_ffn, st);
        } else {
            s.engine->set_layer_weights(L, {w.router_w, w.gate, w.up, w.down});
            s.engine->forward(s.sk_hn, s.sk_routed, K, L, st);
        }
        if (c.n_shared > 0) {
            kernels::launch_moe_expert_ffn(s.sk_hn, w.shared_gate, w.shared_up, w.shared_down,
                                           s.sk_shared_ids, s.sk_shared_w, s.sk_shared,
                                           K, 1, 1, H, c.moe_ffn, st);
            launch_residual_add(s.sk_routed, s.sk_shared, s.sk_routed, K * H, st);
        }
        const void* nextnorm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
        kernels::launch_add_rmsnorm2(s.sk_h, s.sk_routed, nextnorm, s.sk_x, s.sk_xn, K, H, c.rms_eps, st);
    }
    const void* lmw = s.gguf ? s.spec_lm_head : s.w.lm_head;
    kernels::launch_linear_f32(s.sk_xn, lmw, s.sk_logits, K, c.vocab, H, st);
    kernels::launch_argmax(s.sk_logits, s.sk_out_id, K, c.vocab, st);

    cu(cudaStreamEndCapture(st, &s.spec_graph), "sk end capture");
    cu(cudaGraphInstantiate(&s.spec_exec, s.spec_graph, 0), "sk graph instantiate");
    s.spec_graph_ready = true;
    cu(cudaGraphLaunch(s.spec_exec, st), "sk graph launch (first)");

    cu(cudaMemcpyAsync(out_argmax, s.sk_out_id, K*sizeof(int), cudaMemcpyDeviceToHost, st), "sk out (first)");
    cu(cudaStreamSynchronize(st), "sk sync (first)");
    return true;
}

std::vector<int> Qwen35Model::generate_speculative(const std::vector<int>& prompt, int max_new, int draft_k) {
    Impl& s = *p_;
    std::vector<int> out;
    if (prompt.empty() || draft_k < 2) return out;
    if (!s.spec_ready) enable_speculative(draft_k);
    if (!s.spec_ready || s.spec_k != draft_k) {
        fprintf(stderr, "[qwen35] generate_speculative: enable_speculative(%d) unavailable; "
                        "falling back to non-speculative generate()\n", draft_k);
        return generate(prompt, max_new);
    }
    if (!s.kv->allocate(s.seq_id, s.cfg.max_seq)) {
        fprintf(stderr, "[qwen35] generate_speculative: KV allocate failed\n");
        return out;
    }

    std::vector<int> committed = prompt;
    int next = -1;
    for (size_t i = 0; i < prompt.size(); i++) next = forward_token(prompt[i], (int)i);
    int cur = next;
    int pos = (int)prompt.size() - 1;

    long rounds = 0, total_drafted = 0, total_accepted = 0;
    bool stop = false;
    while (!stop && (int)out.size() < max_new) {
        std::vector<int> draft = propose_draft(committed, draft_k - 1);
        std::vector<int> batch_in; batch_in.reserve(draft_k);
        batch_in.push_back(cur);
        for (int d : draft) batch_in.push_back(d);
        while ((int)batch_in.size() < draft_k) batch_in.push_back(cur);   // pad to the fixed graph width

        std::vector<int> verified(draft_k);
        if (!forward_tokens_batch(batch_in.data(), draft_k, pos + 1, verified.data())) break;   // enable_speculative() above guarantees this doesn't happen

        AcceptResult r = accept_draft(draft, verified);
        rounds++; total_drafted += (long)draft.size();

        out.push_back(cur); committed.push_back(cur); pos++;
        stop = (cur == s.cfg.eos_id);
        for (int i = 0; i < r.accepted && !stop && (int)out.size() < max_new; i++) {
            out.push_back(draft[i]); committed.push_back(draft[i]); pos++;
            total_accepted++;
            stop = (draft[i] == s.cfg.eos_id);
        }
        cur = r.bonus_token;
    }
    if (rounds > 0) {
        fprintf(stderr, "[qwen35] speculative decode: %ld forward passes for %zu tokens, "
                        "draft accept rate %.1f%% (%ld/%ld)\n",
                rounds, out.size(), total_drafted ? 100.0 * total_accepted / total_drafted : 0.0,
                total_accepted, total_drafted);
    }
    if ((int)out.size() > max_new) out.resize(max_new);
    s.kv->free(s.seq_id);
    return out;
}

} // namespace sparkinfer
