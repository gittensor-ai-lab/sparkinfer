// Qwen3.5 Multi-Token Prediction (MTP / NextN) draft head.
// Implements the graph from llama.cpp graph_mtp for dense hybrid models (Qwythos 9B).

#include "sparkinfer/models/qwen35_mtp.h"

#include "sparkinfer/kv_ops.h"
#include "sparkinfer/kernels/attention.h"
#include "sparkinfer/kernels/fused.h"
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/moe.h"
#include "sparkinfer/kernels/quant.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace sparkinfer {
namespace mtp {

namespace {
using bf16 = __nv_bfloat16;

constexpr int MAX_NSPLITS = 256;

inline void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fprintf(stderr, "[mtp] %s: %s\n", what, cudaGetErrorString(e));
}

template <typename T>
T* dev_alloc(size_t n, std::vector<void*>& owned) {
    void* p = nullptr;
    cu(cudaMalloc(&p, n * sizeof(T)), "malloc");
    owned.push_back(p);
    return (T*)p;
}

kernels::GemmConfig gc() { return kernels::GemmConfig{}; }

}  // namespace

bool load_weights(GGUF&, int, Weights&, std::vector<void*>&, bool, const Qwen35Config&) {
    fprintf(stderr, "[mtp] load_weights: use Qwen35Model::load_gguf MTP path\n");
    return false;
}

void init_state(State& s, const Qwen35Config& cfg, KVCacheManager* kv, cudaStream_t stream) {
    s.kv = kv;
    s.stream = stream;
    s.qdim = cfg.n_q_heads * cfg.head_dim;
    s.kvdim = cfg.n_kv_heads * cfg.head_dim;
    const int H = cfg.hidden;
    if (const char* vt = getenv("SPARKINFER_MTP_VOCAB_TRIM"))
        s.vocab_trim = atoi(vt);
    if (s.vocab_trim <= 0) {
        static int def = -1;
        if (def < 0) {
            const char* e = getenv("SPARKINFER_MTP_FAST");
            // Default full vocab for acceptance; set SPARKINFER_MTP_FAST=32768 for faster drafts.
            def = e ? atoi(e) : 0;
        }
        s.vocab_trim = def;
    }
    if (s.vocab_trim > cfg.vocab) s.vocab_trim = 0;

    if (cfg.n_q_heads == cfg.n_kv_heads * 4 && cfg.head_dim == 256)
        s.n_splits = 160;
    else if (cfg.n_q_heads == cfg.n_kv_heads * 8 && cfg.head_dim == 256)
        s.n_splits = 160;

    s.x = dev_alloc<bf16>(H, s.owned);
    s.xn = dev_alloc<bf16>(H, s.owned);
    s.concat = dev_alloc<bf16>(H * 2, s.owned);
    s.qraw = dev_alloc<bf16>(s.qdim * 2, s.owned);
    s.q = dev_alloc<bf16>(s.qdim, s.owned);
    s.qgate = dev_alloc<bf16>(s.qdim, s.owned);
    s.k = dev_alloc<bf16>(s.kvdim, s.owned);
    s.v = dev_alloc<bf16>(s.kvdim, s.owned);
    s.attn = dev_alloc<bf16>(s.qdim, s.owned);
    s.ao = dev_alloc<bf16>(H, s.owned);
    s.h = dev_alloc<bf16>(H, s.owned);
    s.hn = dev_alloc<bf16>(H, s.owned);
    s.gate = dev_alloc<bf16>(cfg.moe_ffn, s.owned);
    s.up = dev_alloc<bf16>(cfg.moe_ffn, s.owned);
    s.fused = dev_alloc<bf16>(H, s.owned);
    const int vocab_out = s.vocab_trim > 0 ? s.vocab_trim : cfg.vocab;
    s.logits = dev_alloc<float>(vocab_out, s.owned);
    s.d_scalars = dev_alloc<int>(4, s.owned);
    s.d_out_id = dev_alloc<int>(1, s.owned);
    s.aq81 = dev_alloc<char>(kernels::llama_q8_1_bytes(H * 2), s.owned);
    const size_t fa_n = (size_t)cfg.n_q_heads * MAX_NSPLITS;
    s.fa_m = dev_alloc<float>(fa_n, s.owned);
    s.fa_l = dev_alloc<float>(fa_n, s.owned);
    s.fa_acc = dev_alloc<float>(fa_n * cfg.head_dim, s.owned);
    s.mf_ids = dev_alloc<int>(1, s.owned);
    s.mf_w = dev_alloc<float>(1, s.owned);
    s.mf_h = dev_alloc<float>(cfg.moe_ffn, s.owned);
    s.mf_out = dev_alloc<float>(H, s.owned);
    int zid = 0;
    float zw = 1.f;
    cu(cudaMemcpy(s.mf_ids, &zid, sizeof(int), cudaMemcpyHostToDevice), "mf id");
    cu(cudaMemcpy(s.mf_w, &zw, sizeof(float), cudaMemcpyHostToDevice), "mf w");
}

void reset(State& s) {
    s.pos = 0;
    if (s.kv) s.kv->free(s.seq_id);
}

int forward_step(State& s, const Qwen35Config& cfg,
                 int token_id, const void* main_hidden,
                 const void* embed_table, const void* lm_head, int lm_head_type,
                 int position, void* out_hidden) {
    if (!s.w.loaded || !s.kv) return -1;
    const Weights& w = s.w;
    const Qwen35LayerWeights& Lw = w.layer;
    const int H = cfg.hidden;
    const int seqlen = position + 1;
    cudaStream_t st = s.stream;

    if (cfg.n_q_heads == cfg.n_kv_heads * 4 && cfg.head_dim == 256) {
        int want = 160;
        if ((long)seqlen > 65536L) want = 192;
        if ((long)seqlen > 98304L) want = 128;
        if (want != s.n_splits) s.n_splits = want;
    }

    if (!s.kv->allocate(s.seq_id, cfg.max_seq)) {
        fprintf(stderr, "[mtp] KV allocate failed\n");
        return -1;
    }

    int h_scalars[4] = {token_id, position, position, seqlen};
    cu(cudaMemcpyAsync(s.d_scalars, h_scalars, 4 * sizeof(int), cudaMemcpyHostToDevice, st), "scalars");
    int* btable = s.kv->block_table(s.seq_id);
    const int L = 0;  // single MTP attention layer in its own KV cache

    // Token embedding
    kernels::launch_embedding(s.d_scalars, embed_table, (bf16*)s.x, 1, H, st);

    // eh_proj path: RMSNorm(embed), RMSNorm(main_hidden), concat, project
    kernels::launch_rmsnorm((bf16*)s.x, w.enorm, (bf16*)s.concat, 1, H, cfg.rms_eps, st);
    kernels::launch_rmsnorm(main_hidden, w.hnorm, (bf16*)s.concat + H, 1, H, cfg.rms_eps, st);
    kernels::launch_quantize_q8_1_blocks(s.concat, s.aq81, H * 2, st);
    if (w.eh_proj_type == 12)
        kernels::launch_mmvq_q4k(s.aq81, w.eh_proj, (bf16*)s.x, H, H * 2, st);
    else if (w.eh_proj_type)
        kernels::launch_gemv_q(s.concat, w.eh_proj, w.eh_proj_type, (bf16*)s.x, H, H * 2, st);
    else
        kernels::launch_gemv(s.concat, w.eh_proj, (bf16*)s.x, H, H * 2, st);

    cu(cudaMemcpyAsync(s.h, s.x, H * sizeof(bf16), cudaMemcpyDeviceToDevice, st), "inpSA");

    // attn_norm
    kernels::launch_rmsnorm((bf16*)s.x, Lw.input_norm, (bf16*)s.xn, 1, H, cfg.rms_eps, st);
    kernels::launch_quantize_q8_1_blocks(s.xn, s.aq81, H, st);

    // QKV projections (gated Q)
    const int nq = s.qdim * 2;
    if (Lw.wq_type == 12 && Lw.wk_type == 12 && Lw.wv_type == 12)
        kernels::launch_attn_qkv_mmvq_q4k(s.aq81, Lw.wq, Lw.wk, Lw.wv, (bf16*)s.qraw, (bf16*)s.k, (bf16*)s.v, nq, s.kvdim, s.kvdim, H, st);
    else {
        if (Lw.wq_type == 12) kernels::launch_mmvq_q4k(s.aq81, Lw.wq, (bf16*)s.qraw, nq, H, st);
        else if (Lw.wq_type) kernels::launch_gemv_q(s.xn, Lw.wq, Lw.wq_type, (bf16*)s.qraw, nq, H, st);
        else kernels::launch_gemv(s.xn, Lw.wq, (bf16*)s.qraw, nq, H, st);
        if (Lw.wk_type == 12) kernels::launch_mmvq_q4k(s.aq81, Lw.wk, (bf16*)s.k, s.kvdim, H, st);
        else if (Lw.wk_type) kernels::launch_gemv_q(s.xn, Lw.wk, Lw.wk_type, (bf16*)s.k, s.kvdim, H, st);
        else kernels::launch_gemv(s.xn, Lw.wk, (bf16*)s.k, s.kvdim, H, st);
        if (Lw.wv_type == 12) kernels::launch_mmvq_q4k(s.aq81, Lw.wv, (bf16*)s.v, s.kvdim, H, st);
        else if (Lw.wv_type) kernels::launch_gemv_q(s.xn, Lw.wv, Lw.wv_type, (bf16*)s.v, s.kvdim, H, st);
        else kernels::launch_gemv(s.xn, Lw.wv, (bf16*)s.v, s.kvdim, H, st);
    }

    const bool kv8 = s.kv->int8_kv();
    const int kv_elem = kv8 ? 1 : 2;
    void* kpool = (char*)s.kv->k_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
    void* vpool = (char*)s.kv->v_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
    void* kscale = kv8 ? (char*)s.kv->k_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
    void* vscale = kv8 ? (char*)s.kv->v_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
    const bool partial_rope = cfg.rope_dim > 0 && cfg.rope_dim < cfg.head_dim;

    kernels::launch_qwen36_split_q_gate((bf16*)s.qraw, (bf16*)s.q, (bf16*)s.qgate, cfg.n_q_heads, cfg.head_dim, st);
    if (partial_rope && kv8) {
        kernels::launch_qknorm_rope_kv_partial_int8_gated((bf16*)s.qraw, (bf16*)s.q, (bf16*)s.qgate, (bf16*)s.k, (bf16*)s.v,
            Lw.q_norm, Lw.k_norm, kpool, vpool, kscale, vscale, btable, s.d_scalars + 1, 1,
            cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rope_dim, cfg.rope_theta, cfg.rms_eps,
            s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
    } else if (partial_rope) {
        kernels::launch_qknorm_rope_kv_partial((bf16*)s.q, (bf16*)s.k, (bf16*)s.v, Lw.q_norm, Lw.k_norm,
            (bf16*)kpool, (bf16*)vpool, btable, s.d_scalars + 1, 1,
            cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rope_dim,
            cfg.rope_theta, cfg.rms_eps, s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
    } else {
        kernels::launch_rmsnorm_qk((bf16*)s.q, (bf16*)s.k, Lw.q_norm, Lw.k_norm, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rms_eps, st);
        kernels::launch_rope_kv_append((bf16*)s.q, (bf16*)s.k, (bf16*)s.v, (bf16*)kpool, (bf16*)vpool, btable,
            s.d_scalars + 1, 1, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim, cfg.rope_theta,
            s.kv->block_size(), s.kv->max_blocks_per_seq(), st);
    }

    kernels::launch_flash_decode_split((bf16*)s.q, kpool, vpool, btable, s.d_scalars + 3, (bf16*)s.attn,
        s.fa_m, s.fa_l, s.fa_acc, 1, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim,
        s.kv->block_size(), s.kv->max_blocks_per_seq(), s.n_splits,
        1.f / sqrtf((float)cfg.head_dim), st, nullptr, seqlen, kscale, vscale, kv8 ? 1 : 0, nullptr);
    kernels::launch_qwen36_mul_sigmoid((bf16*)s.attn, (bf16*)s.qgate, s.qdim, st);
    if (Lw.wo_type == 12) {
        kernels::launch_quantize_q8_1_blocks(s.attn, s.aq81, s.qdim, st);
        kernels::launch_mmvq_q4k(s.aq81, Lw.wo, (bf16*)s.ao, H, s.qdim, st);
    } else if (Lw.wo_type)
        kernels::launch_gemv_q(s.attn, Lw.wo, Lw.wo_type, (bf16*)s.ao, H, s.qdim, st);
    else
        kernels::launch_gemv(s.attn, Lw.wo, (bf16*)s.ao, H, s.qdim, st);

    launch_residual_add((bf16*)s.ao, (bf16*)s.h, (bf16*)s.x, H, st);

    // post-attn norm + dense SwiGLU FFN
    kernels::launch_rmsnorm((bf16*)s.x, Lw.post_attn_norm, (bf16*)s.hn, 1, H, cfg.rms_eps, st);
    kernels::launch_quantize_q8_1_blocks(s.hn, s.aq81, H, st);
    kernels::launch_moe_expert_ffn_q4k(s.hn, Lw.gate_q, Lw.up_q, Lw.down_q,
        Lw.gate_qtype, Lw.up_qtype, Lw.down_qtype, s.mf_ids, s.mf_w, (bf16*)s.fused, s.mf_h, s.mf_out,
        1, 1, H, cfg.moe_ffn, s.aq81, st);

    launch_residual_add((bf16*)s.fused, (bf16*)s.x, (bf16*)s.x, H, st);

    if (out_hidden)
        cu(cudaMemcpyAsync(out_hidden, s.x, (size_t)H * sizeof(bf16), cudaMemcpyDeviceToDevice, st),
           "mtp out hidden");

    // shared_head_norm + lm_head
    kernels::launch_rmsnorm((bf16*)s.x, w.shared_head_norm, (bf16*)s.xn, 1, H, cfg.rms_eps, st);
    kernels::launch_quantize_q8_1_blocks(s.xn, s.aq81, H, st);
    const int vocab_out = s.vocab_trim > 0 ? s.vocab_trim : cfg.vocab;
    if (lm_head_type == 12)
        kernels::launch_mmvq_q4k_f32(s.aq81, lm_head, s.logits, vocab_out, H, st);
    else if (lm_head_type == 14)
        kernels::launch_gemv_q6k_dp4a_f32(s.aq81, lm_head, s.logits, vocab_out, H, st);
    else if (lm_head_type)
        kernels::launch_gemv_q_f32(s.xn, lm_head, lm_head_type, s.logits, vocab_out, H, st);
    else
        kernels::launch_gemv_f32(s.xn, lm_head, s.logits, vocab_out, H, st);

    kernels::launch_argmax(s.logits, s.d_out_id, 1, vocab_out, st);
    int out_id = 0;
    cu(cudaMemcpyAsync(&out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "argmax");
    cu(cudaStreamSynchronize(st), "mtp sync");
    s.pos = position + 1;
    return out_id;
}

int copy_logits(State& s, const Qwen35Config& cfg, float* host_logits) {
    if (!host_logits || !s.logits) return 0;
    const int vocab_out = s.vocab_trim > 0 ? s.vocab_trim : cfg.vocab;
    cu(cudaMemcpyAsync(host_logits, s.logits, (size_t)vocab_out * sizeof(float),
                       cudaMemcpyDeviceToHost, s.stream), "mtp logits");
    cu(cudaStreamSynchronize(s.stream), "mtp logits sync");
    return vocab_out;
}

}  // namespace mtp
}  // namespace sparkinfer
