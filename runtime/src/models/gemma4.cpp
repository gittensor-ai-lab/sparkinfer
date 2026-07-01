// Gemma 4 26B-A4B single-sequence greedy decoder.
//
// Per token: embed -> [30x Gemma layer, 5L1G local/global] -> final RMSNorm ->
// LM head -> argmax. Local layers use flash_decode_local_hd256 (sliding 1024);
// global layers use flash_decode_global_hd512 (full context, head_dim=512).

#include "sparkinfer/models/gemma4.h"
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
#include <unordered_map>

namespace sparkinfer {

namespace {
inline void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fprintf(stderr, "[gemma4] %s: %s\n", what, cudaGetErrorString(e));
}
using bf16 = unsigned short;

bool ggml_dequant_supported(int ggml_type) {
    switch (ggml_type) {
        case 0: case 1: case 8: case 12: case 14: return true;
        default: return false;
    }
}

constexpr int kBlockSize = 16;
constexpr int kLocalBlocks = 1024 / kBlockSize;   // 64
constexpr int kMaxSeqs = 256;

// Mixed local/global KV pool: per-layer geometry, shared physical block ids.
struct Gemma4KVCache {
    Gemma4Config cfg;
    int max_global_blocks = 0;
    int total_blocks = 0;
    void* k_pool = nullptr;
    void* v_pool = nullptr;
    std::vector<size_t> layer_off;   // bf16 element offset per layer in each pool
    int* d_global_tables = nullptr;  // [kMaxSeqs, max_global_blocks]
    int* d_local_tables  = nullptr;  // [kMaxSeqs, kLocalBlocks]
    std::vector<int> free_blocks;
    std::unordered_map<uint64_t, std::vector<int>> seq_global_blocks;
    std::unordered_map<uint64_t, std::vector<int>> seq_local_blocks;
    std::unordered_map<uint64_t, int> seq_slot;
    std::vector<int> free_slots;

    explicit Gemma4KVCache(const Gemma4Config& c, size_t pool_bytes) : cfg(c) {
        max_global_blocks = (c.max_seq + kBlockSize - 1) / kBlockSize;
        total_blocks = kLocalBlocks + max_global_blocks;

        size_t total_elems = 0;
        layer_off.resize(c.n_layers);
        for (int L = 0; L < c.n_layers; L++) {
            layer_off[L] = total_elems;
            const auto a = gemma4_layer_attn(L, c);
            total_elems += (size_t)total_blocks * kBlockSize * a.num_kv_heads * a.head_dim;
        }
        const size_t need = total_elems * 2 * sizeof(bf16);
        if (pool_bytes < need) {
            fprintf(stderr, "[gemma4-kv] pool %.1f MB < need %.1f MB — using minimum\n",
                    pool_bytes / 1e6, need / 1e6);
        }
        cu(cudaMalloc(&k_pool, total_elems * sizeof(bf16)), "k pool");
        cu(cudaMalloc(&v_pool, total_elems * sizeof(bf16)), "v pool");
        cu(cudaMalloc(&d_global_tables, (size_t)kMaxSeqs * max_global_blocks * sizeof(int)), "g tables");
        cu(cudaMalloc(&d_local_tables,  (size_t)kMaxSeqs * kLocalBlocks * sizeof(int)), "l tables");
        free_blocks.reserve(total_blocks);
        for (int i = total_blocks - 1; i >= 0; --i) free_blocks.push_back(i);
        for (int i = kMaxSeqs - 1; i >= 0; --i) free_slots.push_back(i);
    }

    ~Gemma4KVCache() {
        cudaFree(k_pool); cudaFree(v_pool);
        cudaFree(d_global_tables); cudaFree(d_local_tables);
    }

    void* k_layer(int L) const {
        return (bf16*)k_pool + layer_off[L];
    }
    void* v_layer(int L) const {
        return (bf16*)v_pool + layer_off[L];
    }

    bool allocate(uint64_t seq_id, int num_tokens) {
        const int gneed = (num_tokens + kBlockSize - 1) / kBlockSize;
        if (gneed > max_global_blocks) return false;

        auto& gblocks = seq_global_blocks[seq_id];
        int extra = 0;
        if (seq_local_blocks.find(seq_id) == seq_local_blocks.end()) extra += kLocalBlocks;
        extra += gneed - (int)gblocks.size();
        if (extra > 0 && (int)free_blocks.size() < extra) return false;
        if (seq_slot.find(seq_id) == seq_slot.end() && free_slots.empty()) return false;

        if (seq_local_blocks.find(seq_id) == seq_local_blocks.end()) {
            std::vector<int> local(kLocalBlocks);
            for (int i = 0; i < kLocalBlocks; i++) {
                local[i] = free_blocks.back(); free_blocks.pop_back();
            }
            seq_local_blocks[seq_id] = local;
            int slot;
            auto it = seq_slot.find(seq_id);
            if (it != seq_slot.end()) slot = it->second;
            else { slot = free_slots.back(); free_slots.pop_back(); seq_slot[seq_id] = slot; }
            cu(cudaMemcpy(d_local_tables + (size_t)slot * kLocalBlocks, local.data(),
                          kLocalBlocks * sizeof(int), cudaMemcpyHostToDevice), "local table");
        }
        while ((int)gblocks.size() < gneed) {
            gblocks.push_back(free_blocks.back()); free_blocks.pop_back();
        }
        int slot = seq_slot.at(seq_id);
        cu(cudaMemcpy(d_global_tables + (size_t)slot * max_global_blocks, gblocks.data(),
                      gblocks.size() * sizeof(int), cudaMemcpyHostToDevice), "global table");
        return true;
    }

    void free_seq(uint64_t seq_id) {
        auto it = seq_global_blocks.find(seq_id);
        if (it != seq_global_blocks.end()) {
            for (int b : it->second) free_blocks.push_back(b);
            seq_global_blocks.erase(it);
        }
        auto lt = seq_local_blocks.find(seq_id);
        if (lt != seq_local_blocks.end()) {
            for (int b : lt->second) free_blocks.push_back(b);
            seq_local_blocks.erase(lt);
        }
        auto s = seq_slot.find(seq_id);
        if (s != seq_slot.end()) {
            free_slots.push_back(s->second);
            seq_slot.erase(s);
        }
    }

    int* global_table(uint64_t seq_id) const {
        auto it = seq_slot.find(seq_id);
        return it == seq_slot.end() ? nullptr
               : d_global_tables + (size_t)it->second * max_global_blocks;
    }
    int* local_table(uint64_t seq_id) const {
        auto it = seq_slot.find(seq_id);
        return it == seq_slot.end() ? nullptr
               : d_local_tables + (size_t)it->second * kLocalBlocks;
    }
};

struct Gemma4Model::Impl {
    Gemma4Config cfg;
    Gemma4KVCache* kv = nullptr;
    moe::MoEEngine* engine = nullptr;
    Gemma4Weights w;
    cudaStream_t stream{};
    uint64_t seq_id = 0;
    bool gguf = false;

    int max_qdim = 0, max_kvdim = 0;
    cudaGraph_t cu_graph{};
    cudaGraphExec_t cu_exec{};
    bool graph_ready = false;

    bf16 *x, *xn, *q, *k, *v, *attn, *ao, *h, *hn, *routed, *shared;
    float* logits;
    int *d_tok, *d_out_id, *d_pos, *d_seqlen, *d_writepos, *d_writepos_local;
    int *d_shared_ids;
    float* d_shared_w;
    std::vector<void*> owned;

    float *mf_logits = nullptr, *mf_weights = nullptr, *mf_h = nullptr, *mf_out = nullptr;
    int   *mf_ids = nullptr, *mf_counts = nullptr;
    signed char* aq8 = nullptr;
    float *aq8_d = nullptr, *aq8_s = nullptr;
    void* aq81 = nullptr;
    bool use_pq = true, use_llama = true, use_q6mmvq = true, use_qkfuse = true;

    template <class T> T* alloc(size_t n) {
        void* p = nullptr; cu(cudaMalloc(&p, n * sizeof(T)), "malloc"); return (T*)p;
    }
};

Gemma4Model::Gemma4Model(const Gemma4Config& cfg, moe::MoEEngine* engine)
    : p_(new Impl()) {
    p_->cfg = cfg;
    p_->engine = engine;
    p_->max_qdim = cfg.n_q_heads * cfg.global_head_dim;
    p_->max_kvdim = cfg.local_n_kv_heads * cfg.local_head_dim;

    const size_t epb = (size_t)kBlockSize * cfg.local_n_kv_heads * cfg.local_head_dim;
    const size_t gblocks = (size_t)(cfg.max_seq + kBlockSize - 1) / kBlockSize + kLocalBlocks + 8;
    p_->kv = new Gemma4KVCache(cfg, cfg.n_layers * 2 * epb * 2 * gblocks);

    cudaStreamCreate(&p_->stream);
    const int H = cfg.hidden, V = cfg.vocab;
    p_->x = p_->alloc<bf16>(H); p_->xn = p_->alloc<bf16>(H);
    p_->q = p_->alloc<bf16>(p_->max_qdim);
    p_->k = p_->alloc<bf16>(p_->max_kvdim);
    p_->v = p_->alloc<bf16>(p_->max_kvdim);
    p_->attn = p_->alloc<bf16>(p_->max_qdim);
    p_->ao = p_->alloc<bf16>(H); p_->h = p_->alloc<bf16>(H); p_->hn = p_->alloc<bf16>(H);
    p_->routed = p_->alloc<bf16>(H); p_->shared = p_->alloc<bf16>(H);
    p_->logits = p_->alloc<float>(V);
    p_->d_tok = p_->alloc<int>(1); p_->d_out_id = p_->alloc<int>(1);
    p_->d_pos = p_->alloc<int>(1); p_->d_seqlen = p_->alloc<int>(1);
    p_->d_writepos = p_->alloc<int>(1); p_->d_writepos_local = p_->alloc<int>(1);
    p_->d_shared_ids = p_->alloc<int>(1); p_->d_shared_w = p_->alloc<float>(1);
    int zero = 0; float one = 1.f;
    cu(cudaMemcpy(p_->d_shared_ids, &zero, sizeof(int), cudaMemcpyHostToDevice), "shared ids");
    cu(cudaMemcpy(p_->d_shared_w, &one, sizeof(float), cudaMemcpyHostToDevice), "shared w");

    p_->mf_logits  = p_->alloc<float>(cfg.n_experts);
    p_->mf_ids     = p_->alloc<int>(cfg.top_k);
    p_->mf_weights = p_->alloc<float>(cfg.top_k);
    p_->mf_counts  = p_->alloc<int>(cfg.n_experts);
    p_->mf_h       = p_->alloc<float>((size_t)cfg.top_k * cfg.moe_ffn);
    p_->mf_out     = p_->alloc<float>(cfg.hidden);
    const int kmax = (p_->max_qdim > H) ? p_->max_qdim : H;
    p_->aq8 = p_->alloc<signed char>(kmax);
    p_->aq8_d = p_->alloc<float>(kmax >> 5);
    p_->aq8_s = p_->alloc<float>(kmax >> 5);
    p_->aq81 = p_->alloc<char>(kernels::llama_q8_1_bytes(kmax));

    if (const char* e = getenv("SPARKINFER_PQ"))     p_->use_pq     = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_LLAMA"))  p_->use_llama  = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_Q6MMVQ")) p_->use_q6mmvq = !(e[0] == '0');
    if (const char* e = getenv("SPARKINFER_QKFUSE")) p_->use_qkfuse = !(e[0] == '0');
}

Gemma4Model::~Gemma4Model() {
    Impl& s = *p_;
    for (void* b : s.owned) cudaFree(b);
    cudaFree(s.x); cudaFree(s.xn); cudaFree(s.q); cudaFree(s.k); cudaFree(s.v);
    cudaFree(s.attn); cudaFree(s.ao); cudaFree(s.h); cudaFree(s.hn);
    cudaFree(s.routed); cudaFree(s.shared); cudaFree(s.logits);
    cudaFree(s.d_tok); cudaFree(s.d_out_id); cudaFree(s.d_pos);
    cudaFree(s.d_seqlen); cudaFree(s.d_writepos); cudaFree(s.d_writepos_local);
    cudaFree(s.d_shared_ids); cudaFree(s.d_shared_w);
    cudaFree(s.mf_logits); cudaFree(s.mf_weights); cudaFree(s.mf_h); cudaFree(s.mf_out);
    cudaFree(s.mf_ids); cudaFree(s.mf_counts);
    cudaFree(s.aq8); cudaFree(s.aq8_d); cudaFree(s.aq8_s); cudaFree(s.aq81);
    if (s.graph_ready) { cudaGraphExecDestroy(s.cu_exec); cudaGraphDestroy(s.cu_graph); }
    cudaStreamDestroy(s.stream);
    delete s.kv;
    delete p_;
}

void Gemma4Model::set_weights(const Gemma4Weights& w) { p_->w = w; }
const Gemma4Config& Gemma4Model::config() const { return p_->cfg; }

void Gemma4Model::copy_logits(float* host_logits) const {
    cudaMemcpy(host_logits, p_->logits, (size_t)p_->cfg.vocab * sizeof(float), cudaMemcpyDeviceToHost);
}

int Gemma4Model::forward_token(int token_id, int position) {
    Impl& s = *p_;
    const Gemma4Config& c = s.cfg;
    const int H = c.hidden;
    kernels::GemmConfig gc{};
    const int seqlen = position + 1;
    const int local_wpos = position % c.local_window;
    cudaStream_t st = s.stream;

    cu(cudaMemcpyAsync(s.d_tok, &token_id, sizeof(int), cudaMemcpyHostToDevice, st), "tok");
    cu(cudaMemcpyAsync(s.d_pos, &position, sizeof(int), cudaMemcpyHostToDevice, st), "pos");
    cu(cudaMemcpyAsync(s.d_writepos, &position, sizeof(int), cudaMemcpyHostToDevice, st), "wpos");
    cu(cudaMemcpyAsync(s.d_writepos_local, &local_wpos, sizeof(int), cudaMemcpyHostToDevice, st), "lwpos");
    cu(cudaMemcpyAsync(s.d_seqlen, &seqlen, sizeof(int), cudaMemcpyHostToDevice, st), "slen");

    if (s.graph_ready) {
        cu(cudaGraphLaunch(s.cu_exec, st), "graph launch");
        int out_id = 0;
        cu(cudaMemcpyAsync(&out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "out_id");
        cu(cudaStreamSynchronize(st), "sync");
        return out_id;
    }
    cu(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "begin capture");

    kernels::launch_embedding(s.d_tok, s.w.embed_tokens, s.x, 1, H, st);
    kernels::launch_rmsnorm(s.x, s.w.layers[0].input_norm, s.xn, 1, H, c.rms_eps, st);

    int* gtable = s.kv->global_table(s.seq_id);
    int* ltable = s.kv->local_table(s.seq_id);

    for (int L = 0; L < c.n_layers; L++) {
        const Gemma4LayerWeights& w = s.w.layers[L];
        const Gemma4LayerAttn la = gemma4_layer_attn(L, c);
        const float attn_scale = 1.f / sqrtf((float)la.head_dim);

        if (s.gguf) {
            const bool any_q4k = (w.wq_type == 12 || w.wk_type == 12 || w.wv_type == 12);
            const bool any_q6k = (w.wq_type == 14 || w.wk_type == 14 || w.wv_type == 14);
            if (s.use_pq && s.use_llama && (any_q4k || any_q6k))
                kernels::launch_quantize_q8_1_blocks(s.xn, s.aq81, H, st);
            else if (s.use_pq && any_q4k)
                kernels::launch_quantize_q8_1(s.xn, s.aq8, s.aq8_d, s.aq8_s, H, st);
            auto proj = [&](const void* W, int t, void* y, int N) {
                if (s.use_pq && t == 12) {
                    if (s.use_llama) kernels::launch_mmvq_q4k(s.aq81, W, y, N, H, st);
                    else kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s, W, y, N, H, st);
                } else if (s.use_q6mmvq && t == 14)
                    kernels::launch_mmvq_q6k(s.aq81, W, y, N, H, st);
                else if (t) kernels::launch_gemv_q(s.xn, W, t, y, N, H, st);
                else kernels::launch_gemv(s.xn, W, y, N, H, st);
            };
            proj(w.wq, w.wq_type, s.q, la.qdim);
            proj(w.wk, w.wk_type, s.k, la.kvdim);
            proj(w.wv, w.wv_type, s.v, la.kvdim);
        } else {
            kernels::launch_gemm(s.xn, w.wq, s.q, 1, la.qdim, H, 1.f, 0.f, gc, st);
            kernels::launch_gemm(s.xn, w.wk, s.k, 1, la.kvdim, H, 1.f, 0.f, gc, st);
            kernels::launch_gemm(s.xn, w.wv, s.v, 1, la.kvdim, H, 1.f, 0.f, gc, st);
        }

        if (s.use_qkfuse)
            kernels::launch_rmsnorm_qk(s.q, s.k, w.q_norm, w.k_norm,
                                       c.n_q_heads, la.num_kv_heads, la.head_dim, c.rms_eps, st);
        else {
            kernels::launch_rmsnorm(s.q, w.q_norm, s.q, c.n_q_heads, la.head_dim, c.rms_eps, st);
            kernels::launch_rmsnorm(s.k, w.k_norm, s.k, la.num_kv_heads, la.head_dim, c.rms_eps, st);
        }
        kernels::launch_rope(s.q, s.k, s.d_pos, 1, c.n_q_heads, la.num_kv_heads,
                             la.head_dim, la.rope_theta, st);

        bf16* kpool = (bf16*)s.kv->k_layer(L);
        bf16* vpool = (bf16*)s.kv->v_layer(L);
        if (la.global) {
            launch_kv_append(kpool, vpool, s.k, s.v, gtable, s.d_writepos, 1,
                             la.num_kv_heads, la.head_dim, kBlockSize,
                             s.kv->max_global_blocks, st);
            kernels::launch_flash_decode_global_hd512(
                s.q, kpool, vpool, gtable, s.d_seqlen, s.attn,
                1, la.num_kv_heads, kBlockSize, s.kv->max_global_blocks, attn_scale, st);
        } else {
            launch_kv_append(kpool, vpool, s.k, s.v, ltable, s.d_writepos_local, 1,
                             la.num_kv_heads, la.head_dim, kBlockSize, kLocalBlocks, st);
            kernels::launch_flash_decode_local_hd256(
                s.q, kpool, vpool, ltable, s.d_seqlen, s.attn,
                1, la.num_kv_heads, kBlockSize, kLocalBlocks, attn_scale, st);
        }

        if (s.gguf && s.use_pq && w.wo_type == 12) {
            if (s.use_llama) {
                kernels::launch_quantize_q8_1_blocks(s.attn, s.aq81, la.qdim, st);
                kernels::launch_mmvq_q4k(s.aq81, w.wo, s.ao, H, la.qdim, st);
            } else {
                kernels::launch_quantize_q8_1(s.attn, s.aq8, s.aq8_d, s.aq8_s, la.qdim, st);
                kernels::launch_gemv_q_dp4a_pq(s.aq8, s.aq8_d, s.aq8_s, w.wo, s.ao, H, la.qdim, st);
            }
        } else if (s.gguf && w.wo_type)
            kernels::launch_gemv_q(s.attn, w.wo, w.wo_type, s.ao, H, la.qdim, st);
        else if (s.gguf)
            kernels::launch_gemv(s.attn, w.wo, s.ao, H, la.qdim, st);
        else
            kernels::launch_gemm(s.attn, w.wo, s.ao, 1, H, la.qdim, 1.f, 0.f, gc, st);

        kernels::launch_add_rmsnorm2(s.x, s.ao, w.post_attn_norm, s.h, s.hn, 1, H, c.rms_eps, st);

        if (w.gate_q) {
            kernels::launch_gemv_f32(s.hn, w.router_w, s.mf_logits, c.n_experts, c.hidden, st);
            static int moe_counts = -1;
            if (moe_counts < 0) {
                const char* mc = getenv("SPARKINFER_MOE_COUNTS");
                moe_counts = (mc && mc[0] == '1') ? 1 : 0;
            }
            if (moe_counts)
                cu(cudaMemsetAsync(s.mf_counts, 0, c.n_experts * sizeof(int), st), "mf counts");
            kernels::launch_moe_router(s.mf_logits, s.mf_ids, s.mf_weights,
                                       moe_counts ? s.mf_counts : nullptr,
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
        const void* nextnorm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
        kernels::launch_add_rmsnorm2(s.h, s.routed, nextnorm, s.x, s.xn, 1, H, c.rms_eps, st);
    }

    if (s.gguf && s.use_q6mmvq && s.w.lm_head_type == 14) {
        kernels::launch_quantize_q8_1_blocks(s.xn, s.aq81, H, st);
        kernels::launch_gemv_q6k_dp4a_f32(s.aq81, s.w.lm_head, s.logits, c.vocab, H, st);
    } else if (s.gguf && s.w.lm_head_type)
        kernels::launch_gemv_q_f32(s.xn, s.w.lm_head, s.w.lm_head_type, s.logits, c.vocab, H, st);
    else if (s.gguf)
        kernels::launch_gemv_f32(s.xn, s.w.lm_head, s.logits, c.vocab, H, st);
    else
        kernels::launch_linear_f32(s.xn, s.w.lm_head, s.logits, 1, c.vocab, H, st);
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

double Gemma4Model::bench_decode(int warmup, int n) {
    Impl& s = *p_;
    if (!s.kv->allocate(s.seq_id, s.cfg.max_seq)) {
        fprintf(stderr, "[gemma4] kv allocate failed\n"); return -1;
    }
    int pos = 0, tok = 100;
    for (int i = 0; i < warmup; i++) {
        tok = forward_token(tok, pos++);
        if (tok < 0 || tok >= s.cfg.vocab) tok = 100;
    }
    cudaDeviceSynchronize();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
        tok = forward_token(tok, pos++);
        if (tok < 0 || tok >= s.cfg.vocab) tok = 100;
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    s.kv->free_seq(s.seq_id);
    s.graph_ready = false;
    if (s.cu_exec) { cudaGraphExecDestroy(s.cu_exec); s.cu_exec = nullptr; }
    if (s.cu_graph) { cudaGraphDestroy(s.cu_graph); s.cu_graph = nullptr; }
    return n / std::chrono::duration<double>(t1 - t0).count();
}

std::vector<int> Gemma4Model::generate(const std::vector<int>& prompt, int max_new,
                                       ThermalGovernor* gov) {
    Impl& s = *p_;
    std::vector<int> out;
    if (prompt.empty()) return out;
    for (size_t i = 0; i < prompt.size(); i++) {
        if (prompt[i] < 0 || prompt[i] >= s.cfg.vocab) {
            fprintf(stderr, "[gemma4] prompt token %d out of vocab range [0,%d)\n",
                    prompt[i], s.cfg.vocab);
            return out;
        }
    }
    if (!s.kv->allocate(s.seq_id, s.cfg.max_seq)) {
        fprintf(stderr, "[gemma4] KV allocate failed (max_seq=%d)\n", s.cfg.max_seq);
        return out;
    }
    int next = -1;
    for (size_t i = 0; i < prompt.size(); i++) next = forward_token(prompt[i], (int)i);
    for (int i = 0; i < max_new; i++) {
        out.push_back(next);
        if (next == s.cfg.eos_id) break;
        if (next < 0 || next >= s.cfg.vocab) break;
        next = forward_token(next, (int)prompt.size() + i);
        if (gov) gov->pace();
    }
    s.kv->free_seq(s.seq_id);
    s.graph_ready = false;
    if (s.cu_exec) { cudaGraphExecDestroy(s.cu_exec); s.cu_exec = nullptr; }
    if (s.cu_graph) { cudaGraphDestroy(s.cu_graph); s.cu_graph = nullptr; }
    return out;
}

namespace {
void* load_bin(const std::string& path, std::vector<void*>& owned) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "[gemma4] missing weight: %s\n", path.c_str()); return nullptr; }
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

bool Gemma4Model::load_weights(const std::string& dir) {
    Impl& s = *p_;
    auto L = [&](const std::string& n) { return load_bin(dir + "/" + n + ".bin", s.owned); };
    s.w.embed_tokens = L("embed_tokens");
    s.w.final_norm   = L("final_norm");
    s.w.lm_head      = L("lm_head");
    if (!s.w.embed_tokens || !s.w.final_norm || !s.w.lm_head) return false;
    s.w.layers.resize(s.cfg.n_layers);
    for (int i = 0; i < s.cfg.n_layers; i++) {
        std::string pfx = "layer_" + std::to_string(i) + ".";
        Gemma4LayerWeights& w = s.w.layers[i];
        w.input_norm = L(pfx + "input_norm");
        w.wq = L(pfx + "wq"); w.wk = L(pfx + "wk"); w.wv = L(pfx + "wv"); w.wo = L(pfx + "wo");
        w.q_norm = L(pfx + "q_norm"); w.k_norm = L(pfx + "k_norm");
        w.post_attn_norm = L(pfx + "post_attn_norm");
        w.router_w = L(pfx + "router_w");
        w.gate = L(pfx + "gate"); w.up = L(pfx + "up"); w.down = L(pfx + "down");
        if (s.cfg.n_shared > 0) {
            w.shared_gate = L(pfx + "shared_gate");
            w.shared_up   = L(pfx + "shared_up");
            w.shared_down = L(pfx + "shared_down");
            if (!w.shared_gate || !w.shared_up || !w.shared_down) return false;
        }
        if (!w.wq || !w.gate || !w.router_w) return false;
    }
    return true;
}

bool Gemma4Model::load_gguf(const std::string& path) {
    Impl& s = *p_;
    const Gemma4Config& c = s.cfg;
    s.gguf = true;
    GGUF g;
    if (!g.open(path)) return false;

    const bool gguf_has_shared = g.tensor("blk.0.ffn_gate_shexp.weight") != nullptr;
    if (c.n_shared > 0 && !gguf_has_shared) {
        fprintf(stderr, "[gemma4-gguf] no shared-expert tensors; forcing n_shared=0\n");
        s.cfg.n_shared = 0;
    }

    auto dev_quant = [&](const std::string& name, int& qtype) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        if (!t) { fprintf(stderr, "[gemma4-gguf] missing %s\n", name.c_str()); return nullptr; }
        if (!ggml_dequant_supported(t->ggml_type)) {
            fprintf(stderr, "[gemma4-gguf] unsupported type %d for %s\n", t->ggml_type, name.c_str());
            return nullptr;
        }
        qtype = t->ggml_type;
        void* d = nullptr;
        if (cudaMalloc(&d, t->n_bytes) != cudaSuccess) return nullptr;
        cudaMemcpy(d, t->data, t->n_bytes, cudaMemcpyHostToDevice);
        s.owned.push_back(d);
        return d;
    };
    auto dense = [&](const std::string& name, bool transpose) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        if (!t) { fprintf(stderr, "[gemma4-gguf] missing %s\n", name.c_str()); return nullptr; }
        if (!ggml_dequant_supported(t->ggml_type)) return nullptr;
        void* dq = nullptr; cudaMalloc(&dq, t->n_bytes);
        cudaMemcpy(dq, t->data, t->n_bytes, cudaMemcpyHostToDevice);
        void* tmp = nullptr; cudaMalloc(&tmp, (size_t)t->n_values * 2);
        kernels::launch_gguf_dequant(t->ggml_type, dq, tmp, t->n_values, s.stream);
        const void* result;
        if (transpose) {
            const int in = (int)t->dims[0], out = (int)t->dims[1];
            void* dst = nullptr; cudaMalloc(&dst, (size_t)t->n_values * 2); s.owned.push_back(dst);
            kernels::launch_transpose_bf16(tmp, dst, out, in, s.stream);
            cudaStreamSynchronize(s.stream); cudaFree(tmp); cudaFree(dq);
            result = dst;
        } else {
            s.owned.push_back(tmp);
            cudaStreamSynchronize(s.stream); cudaFree(dq);
            result = tmp;
        }
        return result;
    };

    const bool qattn = []{ const char* a = getenv("SPARKINFER_QATTN");
                           return !(a && a[0] == '0'); }();
    auto attn_w = [&](const std::string& name, int& type) -> const void* {
        const GGUFTensor* t = g.tensor(name);
        if (qattn && t && (t->ggml_type == 12 || t->ggml_type == 14)) return dev_quant(name, type);
        type = 0; return dense(name, false);
    };

    s.w.embed_tokens = dense("token_embd.weight", false);
    s.w.final_norm   = dense("output_norm.weight", false);
    const char* lm = g.tensor("output.weight") ? "output.weight" : "token_embd.weight";
    s.w.lm_head = attn_w(lm, s.w.lm_head_type);
    if (!s.w.embed_tokens || !s.w.final_norm || !s.w.lm_head) return false;

    s.w.layers.resize(c.n_layers);
    for (int i = 0; i < c.n_layers; i++) {
        std::string b = "blk." + std::to_string(i) + ".";
        Gemma4LayerWeights& w = s.w.layers[i];
        w.input_norm = dense(b + "attn_norm.weight", false);
        w.wq = attn_w(b + "attn_q.weight", w.wq_type);
        w.wk = attn_w(b + "attn_k.weight", w.wk_type);
        w.wv = attn_w(b + "attn_v.weight", w.wv_type);
        w.wo = attn_w(b + "attn_output.weight", w.wo_type);
        w.q_norm = dense(b + "attn_q_norm.weight", false);
        w.k_norm = dense(b + "attn_k_norm.weight", false);
        w.post_attn_norm = dense(b + "ffn_norm.weight", false);
        w.router_w = dense(b + "ffn_gate_inp.weight", false);
        w.gate_q = dev_quant(b + "ffn_gate_exps.weight", w.gate_qtype);
        w.up_q   = dev_quant(b + "ffn_up_exps.weight",   w.up_qtype);
        w.down_q = dev_quant(b + "ffn_down_exps.weight", w.down_qtype);
        if (s.cfg.n_shared > 0) {
            w.shared_gate = dense(b + "ffn_gate_shexp.weight", true);
            w.shared_up   = dense(b + "ffn_up_shexp.weight", true);
            w.shared_down = dense(b + "ffn_down_shexp.weight", true);
            if (!w.shared_gate || !w.shared_up || !w.shared_down) return false;
        }
        if (!w.wq || !w.router_w || !w.gate_q || !w.up_q || !w.down_q) return false;
        if (i == 0 || i == c.n_layers - 1)
            fprintf(stderr, "[gemma4-gguf] layer %d loaded (%s)\n", i,
                    gemma4_is_global_layer(i) ? "global" : "local");
    }
    return true;
}

} // namespace sparkinfer
