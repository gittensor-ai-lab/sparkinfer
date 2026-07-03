// DecodeRunner — wires the sparkinfer kernels + MoE engine into one MoE
// transformer decode layer. All steps run on the stream with no host sync, so a
// full layer (and a full model, looped) is CUDA-graph capturable.

#include "sparkinfer/decode.h"
#include "sparkinfer/kv_ops.h"
#include "sparkinfer/kernels/attention.h"
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/fused.h"

#include <cuda_runtime.h>
#include <cstdio>

namespace sparkinfer {

namespace {
inline void cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fprintf(stderr, "[decode] %s: %s\n", what, cudaGetErrorString(e));
}
using bf16 = unsigned short;
}

struct DecodeRunner::Impl {
    int hidden, qdim, kvdim, max_batch;
    AttnConfig attn;
    KVCacheManager* kv;
    moe::MoEEngine* moe;

    // scratch (bf16 unless noted)
    bf16 *xn, *q, *k, *v, *attnout, *ao, *h, *hn, *moeout;
    int *d_seq_lens, *d_write_pos;

    template <class T> T* alloc(size_t n) { void* p = nullptr; cu(cudaMalloc(&p, n * sizeof(T)), "malloc"); return (T*)p; }
};

DecodeRunner::DecodeRunner(int hidden, AttnConfig attn, KVCacheManager* kv,
                           moe::MoEEngine* moe, int max_batch)
    : p_(new Impl()) {
    p_->hidden = hidden; p_->attn = attn; p_->kv = kv; p_->moe = moe; p_->max_batch = max_batch;
    p_->qdim  = attn.num_q_heads  * attn.head_dim;
    p_->kvdim = attn.num_kv_heads * attn.head_dim;
    const int M = max_batch;
    p_->xn      = p_->alloc<bf16>((size_t)M * hidden);
    p_->q       = p_->alloc<bf16>((size_t)M * p_->qdim);
    p_->k       = p_->alloc<bf16>((size_t)M * p_->kvdim);
    p_->v       = p_->alloc<bf16>((size_t)M * p_->kvdim);
    p_->attnout = p_->alloc<bf16>((size_t)M * p_->qdim);
    p_->ao      = p_->alloc<bf16>((size_t)M * hidden);
    p_->h       = p_->alloc<bf16>((size_t)M * hidden);
    p_->hn      = p_->alloc<bf16>((size_t)M * hidden);
    p_->moeout  = p_->alloc<bf16>((size_t)M * hidden);
    p_->d_seq_lens  = p_->alloc<int>(M);
    p_->d_write_pos = p_->alloc<int>(M);
}

DecodeRunner::~DecodeRunner() {
    cudaFree(p_->xn); cudaFree(p_->q); cudaFree(p_->k); cudaFree(p_->v);
    cudaFree(p_->attnout); cudaFree(p_->ao); cudaFree(p_->h); cudaFree(p_->hn); cudaFree(p_->moeout);
    cudaFree(p_->d_seq_lens); cudaFree(p_->d_write_pos);
    delete p_;
}

void DecodeRunner::begin_step(const std::vector<int>& seq_lens_before) {
    const int n = (int)seq_lens_before.size();
    if (n <= 0 || n > p_->max_batch) {
        fprintf(stderr, "[decode] begin_step: num_seqs %d out of range (max_batch=%d)\n",
                n, p_->max_batch);
        return;
    }
    std::vector<int> after(n);
    for (int i = 0; i < n; i++) after[i] = seq_lens_before[i] + 1;   // include the new token
    cu(cudaMemcpy(p_->d_write_pos, seq_lens_before.data(), n * sizeof(int), cudaMemcpyHostToDevice), "wpos");
    cu(cudaMemcpy(p_->d_seq_lens,  after.data(),           n * sizeof(int), cudaMemcpyHostToDevice), "slens");
}

void DecodeRunner::decode_layer(int layer, void* x, int num_seqs,
                                const TransformerLayerWeights& w, cudaStream_t stream) {
    Impl& s = *p_;
    if (num_seqs <= 0) return;
    if (num_seqs > s.max_batch) {
        fprintf(stderr, "[decode] decode_layer: num_seqs %d exceeds max_batch %d — skipping\n",
                num_seqs, s.max_batch);
        return;
    }
    if (!w.attn_norm || !w.wq || !w.wk || !w.wv || !w.wo || !w.ffn_norm
        || !w.moe.router_w || !w.moe.gate_w || !w.moe.up_w || !w.moe.down_w) {
        fprintf(stderr, "[decode] decode_layer: layer %d has unset weight pointers — skipping\n", layer);
        return;
    }
    const int H = s.hidden, Q = s.qdim, KV = s.kvdim;
    kernels::GemmConfig gc{};

    // 1. pre-attention norm
    kernels::launch_rmsnorm(x, w.attn_norm, s.xn, num_seqs, H, 1e-6f, stream);

    // 2. Q/K/V projections
    kernels::launch_gemm(s.xn, w.wq, s.q, num_seqs, Q,  H, 1.f, 0.f, gc, stream);
    kernels::launch_gemm(s.xn, w.wk, s.k, num_seqs, KV, H, 1.f, 0.f, gc, stream);
    kernels::launch_gemm(s.xn, w.wv, s.v, num_seqs, KV, H, 1.f, 0.f, gc, stream);

    // 3. append new K/V into the paged cache for this layer
    bf16* kpool = (bf16*)s.kv->k_pool() + (size_t)layer * s.kv->layer_stride_elems();
    bf16* vpool = (bf16*)s.kv->v_pool() + (size_t)layer * s.kv->layer_stride_elems();
    int* btable = s.kv->block_table(0);   // batch occupies slots 0..num_seqs-1
    if (!btable) {
        fprintf(stderr, "[decode] decode_layer: seq 0 has no KV block table — skipping\n");
        return;
    }
    launch_kv_append(kpool, vpool, s.k, s.v, btable, s.d_write_pos,
                     num_seqs, s.attn.num_kv_heads, s.attn.head_dim,
                     s.kv->block_size(), s.kv->max_blocks_per_seq(), stream);

    // 4. GQA flash decode over the paged cache
    kernels::launch_flash_decode_gqa8(s.q, kpool, vpool, btable, s.d_seq_lens, s.attnout,
                                      num_seqs, s.attn.num_kv_heads, s.attn.head_dim,
                                      s.kv->block_size(), s.kv->max_blocks_per_seq(),
                                      s.attn.scale, stream);

    // 5. output projection + residual + post-attention norm
    kernels::launch_gemm(s.attnout, w.wo, s.ao, num_seqs, H, Q, 1.f, 0.f, gc, stream);
    launch_residual_add(x, s.ao, s.h, num_seqs * H, stream);          // h = x + attn
    kernels::launch_rmsnorm(s.h, w.ffn_norm, s.hn, num_seqs, H, 1e-6f, stream);

    // 6. sync-free MoE FFN + residual
    s.moe->set_layer_weights(layer, w.moe);
    s.moe->forward(s.hn, s.moeout, num_seqs, layer, stream);
    launch_residual_add(s.h, s.moeout, x, num_seqs * H, stream);      // x = h + moe
}

} // namespace sparkinfer
