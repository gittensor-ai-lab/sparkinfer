// Batched prompt prefill for the Qwen3.5 dense-hybrid (Qwythos) model.
//
// forward_token ingests a prompt one token at a time, so every prompt token pays a full
// bandwidth-bound weight reload for each projection (a GEMV). prefill_batched_run() instead runs
// the whole prompt through the layer stack in one pass: the weight-bound Q/K/V/O + dense-SwiGLU-FFN
// projections become tensor-core (cp.async, wmma) GEMMs, the Gated-DeltaNet recurrence runs as a
// single sequential scan over all N tokens, and the full-attention layers fill the paged int8 KV
// cache in the exact layout the decode path reads. It fills the same KV cache and recurrent/conv
// state a forward_token loop would, so a subsequent decode is numerically faithful.
//
// This is its own translation unit — it reaches nothing but the explicit Qwen35PrefillCtx, so it
// shares no code with the decode path (qwen35.cpp keeps Impl private).

#include "qwen35_prefill.h"
#include "sparkinfer/kernels/prefill.h"
#include "sparkinfer/kernels/fused.h"
#include "sparkinfer/kernels/quant.h"
#include "sparkinfer/kernels/gemm.h"
#include "sparkinfer/kernels/prefill_i8.h"
#include "sparkinfer/kernels/moe.h"
#include "sparkinfer/kernels/attention.h"
#include "sparkinfer/kv_ops.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

namespace sparkinfer {

namespace {
using bf16 = unsigned short;
inline void pf_cu(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fprintf(stderr, "[prefill] %s: %s\n", what, cudaGetErrorString(e));
}
// int8 GEMM scratch must cover the largest projection (n_out x K), not just the MoE FFN dims.
inline int pf_max_n_out(int lqkv, int wide, int qdim, int kvdim, int H, int lvdim, int ffn, int vh) {
    return std::max({lqkv, wide, qdim, kvdim, H, lvdim, ffn, vh});
}
inline int pf_max_k_in(int H, int lvdim, int qdim) {
    return std::max({H, lvdim, qdim});
}
inline void pf_reset_hybrid_state(const Qwen35PrefillCtx& s, int n_layers, int vh, int lhd,
                                int lqkv, int conv_k, cudaStream_t st) {
    const Qwen35Config& c = s.cfg;
    if (!c.hybrid) return;
    pf_cu(cudaMemsetAsync(s.lin_state, 0,
        (size_t)n_layers * vh * lhd * lhd * sizeof(float), st), "lin state reset");
    pf_cu(cudaMemsetAsync(s.lin_conv_state, 0,
        (size_t)n_layers * (conv_k - 1) * lqkv * sizeof(bf16), st), "lin conv reset");
}
inline bool pf_addnorm3() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("SPARKINFER_ADDNORM3"); v = (e && e[0] == '0') ? 0 : 1; }
    return v != 0;
}
inline bool pf_fnq(bool gguf) {
    static int v = -1;
    if (v < 0) { const char* e = getenv("SPARKINFER_FNQ"); v = (e && e[0] == '0') ? 0 : 1; }
    return gguf && v;
}
// Match forward_token adaptive KV-split count for flash-decode parity.
inline int pf_adaptive_splits(int seqlen, const Qwen35Config& c, int split_chunk = 256) {
    int want = 32;
    if ((long)seqlen > 2L * split_chunk) want = 128;
    if ((long)seqlen > 28L * split_chunk && (long)seqlen <= 48L * split_chunk) want = 256;
    if ((long)seqlen > 64L * split_chunk) want = 256;
    if (c.head_dim == 256 && c.n_kv_heads > 0 && want >= 128) {
        if (c.n_q_heads == c.n_kv_heads * 8) want = 160;
        else if (c.n_q_heads == c.n_kv_heads * 4) {
            if ((long)seqlen > 98304L)      want = 128;
            else if ((long)seqlen > 65536L) want = 192;
            else                            want = 160;
        }
    }
    return want;
}
inline bool pf_mma_attn_ok(int seqlen, int n_splits, int block_size) {
    static int famma256 = -1;
    if (famma256 < 0) { const char* e = getenv("SPARKINFER_FAMMA"); famma256 = (e && e[0] == '0') ? 0 : 1; }
    const int mma_chunk = (n_splits > 0) ? (seqlen + n_splits - 1) / n_splits : 0;
    return famma256 && seqlen > 512 && block_size == 16 && mma_chunk >= 32;
}
// Token-loop flash decode for each prompt position — matches forward_token attention exactly.
inline void pf_flash_attn_tokens(
    const bf16* qb, const bf16* qg, bf16* att, signed char* kpool, signed char* vpool,
    void* kscale, void* vscale, const int* btable, int N, int qdim,
    int n_q_heads, int n_kv_heads, int head_dim, int bs, int mbs,
    float scale, bool q_has_gate, const Qwen35Config& cfg,
    float* fa_m, float* fa_l, float* fa_acc, int* d_scalars, int* d_seqlen,
    const int* prompt_ids, cudaStream_t st) {
    static int split_chunk = -1;
    if (split_chunk < 0) {
        const char* e = getenv("SPARKINFER_SPLIT_CHUNK");
        split_chunk = e ? atoi(e) : 256;
        if (split_chunk <= 0) split_chunk = 256;
    }
    for (int t = 0; t < N; t++) {
        const int seqlen = t + 1;
        const int n_splits = pf_adaptive_splits(seqlen, cfg, split_chunk);
        if (pf_mma_attn_ok(seqlen, n_splits, bs)) {
            const size_t pn = (size_t)n_q_heads * (size_t)n_splits;
            pf_cu(cudaMemsetAsync(fa_m, 0, pn * sizeof(float), st), "prefill fa_m zero");
            pf_cu(cudaMemsetAsync(fa_l, 0, pn * sizeof(float), st), "prefill fa_l zero");
            pf_cu(cudaMemsetAsync(fa_acc, 0, pn * (size_t)head_dim * sizeof(float), st), "prefill fa_acc zero");
        }
        if (d_scalars) {
            int hsc[4] = { prompt_ids[t], t, t, seqlen };
            pf_cu(cudaMemcpyAsync(d_scalars, hsc, 4 * sizeof(int), cudaMemcpyHostToDevice, st), "prefill scalars");
        } else {
            pf_cu(cudaMemcpyAsync(d_seqlen, &seqlen, sizeof(int), cudaMemcpyHostToDevice, st), "prefill seqlen");
        }
        pf_cu(cudaStreamSynchronize(st), "prefill seqlen sync");
        const bf16* qt = qb + (size_t)t * (size_t)qdim;
        const bf16* gt = qg + (size_t)t * (size_t)qdim;
        bf16* at = att + (size_t)t * (size_t)qdim;
        kernels::launch_flash_decode_split(qt, kpool, vpool, btable, d_seqlen, at,
            fa_m, fa_l, fa_acc, 1, n_q_heads, n_kv_heads, head_dim, bs, mbs, n_splits, scale, st,
            nullptr, seqlen, kscale, vscale, 1, nullptr, true);
        pf_cu(cudaStreamSynchronize(st), "prefill flash sync");
        if (q_has_gate)
            kernels::launch_qwen36_mul_sigmoid(at, gt, qdim, st);
    }
}
// Gated-DeltaNet: per-token decode kernels (conv + gdn_ar) — matches forward_token recurrence.
inline void pf_gdn_decode_tokens(
    const bf16* qkv, const bf16* lz, const bf16* la, const bf16* lb, const Qwen35LayerWeights& w,
    bf16* conv_state, float* layer_state, bf16* gq, bf16* gk, bf16* gv, bf16* gdn_out, bf16* lnrm,
    int N, const Qwen35Config& c, int vh, int lqkv, int lvdim, int lqd, cudaStream_t st) {
    const int lhd = c.linear_head_dim;
    static int gdn_fuse = -1;
    if (gdn_fuse < 0) { const char* e = getenv("SPARKINFER_GDN_FUSE"); gdn_fuse = (e && e[0] == '0') ? 0 : 1; }
    for (int t = 0; t < N; t++) {
        const bf16* qkv_t = qkv + (size_t)t * (size_t)lqkv;
        bf16* gq_t = gq + (size_t)t * (size_t)lqd;
        bf16* gk_t = gk + (size_t)t * (size_t)lqd;
        bf16* gv_t = gv + (size_t)t * (size_t)lvdim;
        bf16* out_t = gdn_out + (size_t)t * (size_t)lvdim;
        bf16* ln_t = lnrm + (size_t)t * (size_t)lvdim;
        if (gdn_fuse && lhd == 128 && c.linear_q_heads == 16 && vh == 32)
            kernels::launch_qwen36_conv_split_l2norm_fused(qkv_t, w.ssm_conv, conv_state, gq_t, gk_t, gv_t,
                c.linear_q_heads, vh, lhd, c.linear_conv_kernel, c.rms_eps, st);
        else
            kernels::launch_qwen36_conv_split_l2(qkv_t, w.ssm_conv, conv_state, gq_t, gk_t, gv_t,
                c.linear_q_heads, vh, lhd, c.linear_conv_kernel, c.rms_eps, st);
        kernels::launch_qwen36_gdn_ar(gq_t, gk_t, gv_t, la + (size_t)t * vh, lb + (size_t)t * vh,
            w.ssm_dt, w.ssm_a, layer_state, out_t, c.linear_q_heads, vh, lhd, st);
        kernels::launch_qwen36_gated_norm(out_t, lz + (size_t)t * (size_t)lvdim, w.ssm_norm, ln_t,
                                          vh, lhd, c.rms_eps, st);
    }
}
// Simple device-buffer arena: all-or-nothing allocation with one free() at the end.
struct Arena {
    std::vector<void*> bufs;
    bool ok = true;
    template <class T> T* alloc(size_t n) {
        void* p = nullptr;
        if (n == 0) n = 1;
        if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) { ok = false; return nullptr; }
        bufs.push_back(p);
        return static_cast<T*>(p);
    }
    void free_all() { for (void* b : bufs) cudaFree(b); bufs.clear(); }
};
} // namespace

int prefill_batched_run(const Qwen35PrefillCtx& s, const int* prompt_ids, int n) {
    const Qwen35Config& c = s.cfg;
    // Only the Qwen3.5 dense-hybrid path is supported (GGUF-native, quantized weights).
    if (!s.gguf || !c.hybrid || !c.dense_ffn || n <= 0) return -1;
    if (c.head_dim != 256 || c.linear_head_dim != 128) return -1;   // kernels specialize these

    const int H = c.hidden;
    const int N = n;
    cudaStream_t st = s.stream;

    const int qdim = s.qdim, kvdim = s.kvdim;            // full-attn: 4096 / 1024
    const int lqkv = s.linear_qkvdim;                    // 8192
    const int lvdim = s.linear_vdim;                     // 4096
    const int vh   = c.linear_v_heads;                   // 32
    const int ffn  = c.moe_ffn;                          // 12288
    const int wide = 2 * qdim;                           // 8192 (qraw); also >= lqkv
    const int max_n_out = pf_max_n_out(lqkv, wide, qdim, kvdim, H, lvdim, ffn, vh);
    const int max_k_in  = pf_max_k_in(H, lvdim, qdim);
    const size_t maxw = (size_t)max_n_out * (size_t)max_k_in;
    bf16* lin_conv_state = static_cast<bf16*>(s.lin_conv_state);

    // ---- scratch ----
    Arena a;
    bf16* x    = a.alloc<bf16>((size_t)N * H);
    bf16* xn   = a.alloc<bf16>((size_t)N * H);
    bf16* hbuf = a.alloc<bf16>((size_t)N * H);
    bf16* hn   = a.alloc<bf16>((size_t)N * H);
    bf16* ao   = a.alloc<bf16>((size_t)N * H);
    bf16* b8   = a.alloc<bf16>((size_t)N * wide);        // qraw / lin_qkv (8192)
    bf16* lz   = a.alloc<bf16>((size_t)N * lvdim);       // lin_z (4096)
    bf16* qb   = a.alloc<bf16>((size_t)N * qdim);        // full q (4096)
    bf16* qg   = a.alloc<bf16>((size_t)N * qdim);        // full q-gate (4096)
    bf16* kf   = a.alloc<bf16>((size_t)N * kvdim);       // full k (1024)
    bf16* vf   = a.alloc<bf16>((size_t)N * kvdim);       // full v (1024)
    bf16* gq   = a.alloc<bf16>((size_t)N * s.linear_qdim);   // gdn q (2048)
    bf16* gk   = a.alloc<bf16>((size_t)N * s.linear_qdim);   // gdn k (2048)
    bf16* gv   = a.alloc<bf16>((size_t)N * lvdim);       // gdn v (4096)
    bf16* att  = a.alloc<bf16>((size_t)N * (size_t)std::max(qdim, lvdim));
    bf16* lnrm = a.alloc<bf16>((size_t)N * lvdim);       // lin_norm (4096)
    bf16* la   = a.alloc<bf16>((size_t)N * vh);          // lin_alpha (32)
    bf16* lb   = a.alloc<bf16>((size_t)N * vh);          // lin_beta (32)
    bf16* ffg  = a.alloc<bf16>((size_t)N * ffn);         // ffn gate (12288)
    bf16* ffu  = a.alloc<bf16>((size_t)N * ffn);         // ffn up
    bf16* ffh  = a.alloc<bf16>((size_t)N * ffn);         // ffn silu(gate)*up
    bf16* wbuf = a.alloc<bf16>(maxw);                    // dequantized-weight scratch (reused)
    int*  d_ids = a.alloc<int>((size_t)N);
    static constexpr int PF_MAX_NSPLITS = 256;
    float* fa_m = s.fa_m;
    float* fa_l = s.fa_l;
    float* fa_acc = s.fa_acc;
    int* d_seqlen = s.d_seqlen;
    if (!fa_m || !fa_l || !fa_acc || !d_seqlen) {
        a.free_all();
        fprintf(stderr, "[prefill] flash-decode buffers missing -> fallback\n");
        return -1;
    }
    if (!a.ok) { a.free_all(); fprintf(stderr, "[prefill] scratch alloc failed (ctx=%d) -> fallback\n", N); return -1; }
    // int8 tensor-core projections (prefill_gemm_i8): ~2x the bf16 GEMM at int8==bf16 output fidelity
    // (GGUF weights are already Q4_K/Q6_K -> int8 weight-quant is lossless vs what's stored). Default
    // ON at every batched context; SPARKINFER_PREFILL_I8=0 disables (A/B). The int8 scratch lives in
    // its own arena so an alloc failure at huge N degrades to the bf16 GEMMs, not to the token loop.
    const char* _pi8 = getenv("SPARKINFER_PREFILL_I8");
    bool use_i8 = !(_pi8 && _pi8[0] == '0');
    Arena a8;
    signed char* A_i8 = use_i8 ? a8.alloc<signed char>((size_t)N * (size_t)max_k_in) : nullptr;
    signed char* W_i8 = use_i8 ? a8.alloc<signed char>(maxw) : nullptr;
    float* sx = use_i8 ? a8.alloc<float>((size_t)N) : nullptr;
    float* sw = use_i8 ? a8.alloc<float>((size_t)max_n_out) : nullptr;
    if (use_i8 && !a8.ok) { a8.free_all(); use_i8 = false; }

    pf_cu(cudaMemcpyAsync(d_ids, prompt_ids, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, st), "prefill ids");
    pf_reset_hybrid_state(s, c.n_layers, vh, c.linear_head_dim, lqkv, c.linear_conv_kernel, st);

    // Dequantize a native GGUF weight [n_out,K] to bf16 scratch; return a bf16 [n_out,K] ptr.
    auto dq = [&](const void* W, int wtype, int n_out, int K) -> const void* {
        if (wtype == 0) return W;   // already bf16 dense
        kernels::launch_gguf_dequant(wtype, W, wbuf, (long)n_out * K, st);
        return wbuf;
    };
    // C[N,n_out] = A[N,K] @ W^T  (W native quantized [n_out,K]).
    auto proj = [&](const bf16* A, const void* W, int wtype, bf16* C, int n_out, int K) {
        // int8 only for the big weight-bound projections; keep the tiny per-v-head gate
        // projections (ssm_alpha/ssm_beta, n_out == v_heads) in bf16 — they feed the GDN
        // sigmoid gates, where per-row int8 quant of a 32-wide weight costs more accuracy
        // than the negligible time it saves.
        if (use_i8 && n_out >= 128) {
            kernels::launch_prefill_quantize_rows_i8(A, A_i8, sx, N, K, st);
            // fused Q4_K/Q6_K -> int8 rows skips the dequant-to-bf16 scratch round trip
            if (!kernels::launch_gguf_dequant_rows_i8(wtype, W, W_i8, sw, n_out, K, st)) {
                const void* wb = dq(W, wtype, n_out, K);
                kernels::launch_prefill_quantize_rows_i8(wb, W_i8, sw, n_out, K, st);
            }
            kernels::launch_prefill_gemm_i8(A_i8, W_i8, sx, sw, C, N, n_out, K, st);
        } else {
            kernels::launch_prefill_gemm(A, dq(W, wtype, n_out, K), C, N, n_out, K, st);
        }
    };

    const int* btable = s.kv->block_table(s.seq_id);
    const int  bs = s.kv->block_size();
    const int  mbs = s.kv->max_blocks_per_seq();
    const bool kv8 = s.kv->int8_kv();
    const int  kv_elem = kv8 ? 1 : 2;
    const float rope_theta = c.rope_theta, eps = c.rms_eps;
    const int rope_dim = (c.rope_dim > 0) ? c.rope_dim : c.head_dim;
    const float attn_scale = 1.f / sqrtf((float)c.head_dim);

    // embed -> x, prime xn = RMSNorm(x, layer0.input_norm)
    kernels::launch_embedding(d_ids, s.w.embed_tokens, x, N, H, st);
    kernels::launch_rmsnorm(x, s.w.layers[0].input_norm, xn, N, H, eps, st);

    for (int L = 0; L < c.n_layers; L++) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        if (w.linear_attn) {
            // ---- Gated DeltaNet linear-attention layer ----
            proj(xn, w.wqkv,      w.wqkv_type,      b8, lqkv,  H);   // qkv
            proj(xn, w.wqkv_gate, w.wqkv_gate_type, lz, lvdim, H);   // z gate
            proj(xn, w.ssm_alpha, w.ssm_alpha_type, la, vh,    H);
            proj(xn, w.ssm_beta,  w.ssm_beta_type,  lb, vh,    H);
            bf16* conv_state = lin_conv_state + (size_t)L * (c.linear_conv_kernel - 1) * lqkv;
            kernels::launch_prefill_gdn_conv(b8, w.ssm_conv, conv_state, gq, gk, gv,
                N, c.linear_q_heads, vh, c.linear_head_dim, c.linear_conv_kernel, eps, st);
            float* layer_state = s.lin_state + (size_t)L * vh * c.linear_head_dim * c.linear_head_dim;
            kernels::launch_prefill_gdn_scan(gq, gk, gv, la, lb, w.ssm_dt, w.ssm_a,
                layer_state, att, N, c.linear_q_heads, vh, c.linear_head_dim, st);
            kernels::launch_prefill_gated_norm(att, lz, w.ssm_norm, lnrm, N, vh, c.linear_head_dim, eps, st);
            proj(lnrm, w.ssm_out, w.ssm_out_type, ao, H, lvdim);
        } else {
            // ---- full softmax-attention layer (q_has_gate, partial RoPE, int8 KV) ----
            proj(xn, w.wq, w.wq_type, b8, wide,  H);                 // qraw = [q|gate] per head
            proj(xn, w.wk, w.wk_type, kf, kvdim, H);
            proj(xn, w.wv, w.wv_type, vf, kvdim, H);
            kernels::launch_prefill_split_q_gate(b8, qb, qg, N, c.n_q_heads, c.head_dim, st);
            signed char* kpool = (signed char*)s.kv->k_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
            signed char* vpool = (signed char*)s.kv->v_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
            void* kscale = kv8 ? (char*)s.kv->k_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
            void* vscale = kv8 ? (char*)s.kv->v_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
            if (!kv8) { a.free_all(); a8.free_all(); fprintf(stderr, "[prefill] batched prefill requires int8 KV\n"); return -1; }
            kernels::launch_prefill_qknorm_rope_kv_int8(qb, kf, vf, w.q_norm, w.k_norm,
                kpool, vpool, kscale, vscale, btable, N, c.n_q_heads, c.n_kv_heads, c.head_dim,
                rope_dim, rope_theta, eps, bs, mbs, st);
            pf_flash_attn_tokens(qb, qg, att, kpool, vpool, kscale, vscale, btable, N, qdim,
                c.n_q_heads, c.n_kv_heads, c.head_dim, bs, mbs, attn_scale,
                w.q_has_gate, c, fa_m, fa_l, fa_acc, s.d_scalars, d_seqlen, prompt_ids, st);
            proj(att, w.wo, w.wo_type, ao, H, qdim);
        }

        // h = x + ao ; hn = RMSNorm(h, post_attn_norm)
        kernels::launch_prefill_add(x, ao, hbuf, (long)N * H, st);
        kernels::launch_rmsnorm(hbuf, w.post_attn_norm, hn, N, H, eps, st);

        // dense SwiGLU FFN
        proj(hn, w.gate_q, w.gate_qtype, ffg, ffn, H);
        proj(hn, w.up_q,   w.up_qtype,   ffu, ffn, H);
        kernels::launch_prefill_swiglu(ffg, ffu, ffh, (long)N * ffn, st);
        proj(ffh, w.down_q, w.down_qtype, ao, H, ffn);

        // x = h + ffn_out ; xn = RMSNorm(x, next_input_norm)  (final_norm on the last layer)
        kernels::launch_prefill_add(hbuf, ao, x, (long)N * H, st);
        const void* next_norm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
        kernels::launch_rmsnorm(x, next_norm, xn, N, H, eps, st);
    }

    // Seed for the first decode step: argmax at the last prompt position (xn already = final norm).
    const bf16* xn_last = xn + (size_t)(N - 1) * H;
    if (s.w.lm_head_type)
        kernels::launch_gemv_q_f32(xn_last, s.w.lm_head, s.w.lm_head_type, s.logits, c.vocab, H, st);
    else
        kernels::launch_gemv_f32(xn_last, s.w.lm_head, s.logits, c.vocab, H, st);
    kernels::launch_argmax(s.logits, s.d_out_id, 1, c.vocab, st);
    pf_cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "prefill seed");
    pf_cu(cudaStreamSynchronize(st), "prefill sync");
    int seed = *s.h_out_id;

    a.free_all();
    a8.free_all();
    return seed;
}

int prefill_batched_moe_run(const Qwen35PrefillCtx& s, const int* prompt_ids, int n) {
    const Qwen35Config& c = s.cfg;
    // Qwen3.6 MoE hybrid: batched attention stack, per-token routed FFN (Phase 1).
    if (!s.gguf || !c.hybrid || c.dense_ffn || n <= 0) return -1;
    if (c.head_dim != 256 || c.linear_head_dim != 128) return -1;
    if (c.n_experts != 256 || (c.hidden % 8) != 0 || c.top_k <= 0) return -1;
    for (int L = 0; L < c.n_layers; L++) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        if (!w.gate_q || !w.router_w) return -1;
    }

    const int H = c.hidden;
    const int N = n;
    cudaStream_t st = s.stream;

    const int qdim = s.qdim, kvdim = s.kvdim;
    const int lqkv = s.linear_qkvdim;
    const int lvdim = s.linear_vdim;
    const int vh   = c.linear_v_heads;
    const int ffn  = c.moe_ffn;
    const int topk = c.top_k;
    const int wide = 2 * qdim;
    const int max_n_out = pf_max_n_out(lqkv, wide, qdim, kvdim, H, lvdim, ffn, vh);
    const int max_k_in  = pf_max_k_in(H, lvdim, qdim);
    const size_t maxw = (size_t)max_n_out * (size_t)max_k_in;
    bf16* lin_conv_state = static_cast<bf16*>(s.lin_conv_state);

    Arena a;
    bf16* x    = a.alloc<bf16>((size_t)N * H);
    bf16* xn   = a.alloc<bf16>((size_t)N * H);
    bf16* hbuf = a.alloc<bf16>((size_t)N * H);
    bf16* hn   = a.alloc<bf16>((size_t)N * H);
    bf16* ao   = a.alloc<bf16>((size_t)N * H);
    bf16* b8   = a.alloc<bf16>((size_t)N * wide);
    bf16* lz   = a.alloc<bf16>((size_t)N * lvdim);
    bf16* qb   = a.alloc<bf16>((size_t)N * qdim);
    bf16* qg   = a.alloc<bf16>((size_t)N * qdim);
    bf16* kf   = a.alloc<bf16>((size_t)N * kvdim);
    bf16* vf   = a.alloc<bf16>((size_t)N * kvdim);
    bf16* gq   = a.alloc<bf16>((size_t)N * s.linear_qdim);
    bf16* gk   = a.alloc<bf16>((size_t)N * s.linear_qdim);
    bf16* gv   = a.alloc<bf16>((size_t)N * lvdim);
    bf16* att  = a.alloc<bf16>((size_t)N * (size_t)std::max(qdim, lvdim));
    bf16* lnrm = a.alloc<bf16>((size_t)N * lvdim);
    bf16* la   = a.alloc<bf16>((size_t)N * vh);
    bf16* lb   = a.alloc<bf16>((size_t)N * vh);
    bf16* wbuf = a.alloc<bf16>(maxw);
    // Per-token MoE FFN scratch (reused across tokens and layers).
    bf16* routed = a.alloc<bf16>((size_t)H);
    bf16* shared = a.alloc<bf16>((size_t)H);
    bf16* shared_gate_tmp = a.alloc<bf16>((size_t)H);
    float* mf_logits  = a.alloc<float>((size_t)c.n_experts);
    float* mf_weights = a.alloc<float>((size_t)topk);
    float* mf_h       = a.alloc<float>((size_t)topk * ffn);
    float* mf_out     = a.alloc<float>((size_t)H);
    float* d_shared_w = a.alloc<float>(1);
    int*   mf_ids     = a.alloc<int>((size_t)topk);
    unsigned int* mf_rc = a.alloc<unsigned int>(1);
    char*  sh_q8_buf   = a.alloc<char>(kernels::llama_q8_1_bytes(H));
    char*  hn_q8       = a.alloc<char>(kernels::llama_q8_1_bytes(H));
    int*  d_ids = a.alloc<int>((size_t)N);
    static constexpr int PF_MAX_NSPLITS = 256;
    float* fa_m = s.fa_m;
    float* fa_l = s.fa_l;
    float* fa_acc = s.fa_acc;
    int* d_seqlen = s.d_seqlen;
    if (!fa_m || !fa_l || !fa_acc || !d_seqlen) {
        a.free_all();
        fprintf(stderr, "[prefill-moe] flash-decode buffers missing -> fallback\n");
        return -1;
    }
    if (!a.ok) { a.free_all(); fprintf(stderr, "[prefill-moe] scratch alloc failed (ctx=%d) -> fallback\n", N); return -1; }
    pf_cu(cudaMemsetAsync(mf_rc, 0, sizeof(unsigned int), st), "mf_rc zero");

    const char* _pi8 = getenv("SPARKINFER_PREFILL_I8");
    bool use_i8 = !(_pi8 && _pi8[0] == '0');
    Arena a8;
    signed char* A_i8 = use_i8 ? a8.alloc<signed char>((size_t)N * (size_t)max_k_in) : nullptr;
    signed char* W_i8 = use_i8 ? a8.alloc<signed char>(maxw) : nullptr;
    float* sx = use_i8 ? a8.alloc<float>((size_t)N) : nullptr;
    float* sw = use_i8 ? a8.alloc<float>((size_t)max_n_out) : nullptr;
    if (use_i8 && !a8.ok) { a8.free_all(); use_i8 = false; }

    pf_cu(cudaMemcpyAsync(d_ids, prompt_ids, (size_t)N * sizeof(int), cudaMemcpyHostToDevice, st), "prefill-moe ids");
    pf_reset_hybrid_state(s, c.n_layers, vh, c.linear_head_dim, lqkv, c.linear_conv_kernel, st);

    auto dq = [&](const void* W, int wtype, int n_out, int K) -> const void* {
        if (wtype == 0) return W;
        kernels::launch_gguf_dequant(wtype, W, wbuf, (long)n_out * K, st);
        return wbuf;
    };
    auto proj = [&](const bf16* A, const void* W, int wtype, bf16* C, int n_out, int K) {
        if (use_i8 && n_out >= 128) {
            kernels::launch_prefill_quantize_rows_i8(A, A_i8, sx, N, K, st);
            if (!kernels::launch_gguf_dequant_rows_i8(wtype, W, W_i8, sw, n_out, K, st)) {
                const void* wb = dq(W, wtype, n_out, K);
                kernels::launch_prefill_quantize_rows_i8(wb, W_i8, sw, n_out, K, st);
            }
            kernels::launch_prefill_gemm_i8(A_i8, W_i8, sx, sw, C, N, n_out, K, st);
        } else {
            kernels::launch_prefill_gemm(A, dq(W, wtype, n_out, K), C, N, n_out, K, st);
        }
    };

    const int* btable = s.kv->block_table(s.seq_id);
    const int  bs = s.kv->block_size();
    const int  mbs = s.kv->max_blocks_per_seq();
    const bool kv8 = s.kv->int8_kv();
    const int  kv_elem = kv8 ? 1 : 2;
    const float rope_theta = c.rope_theta, eps = c.rms_eps;
    const int rope_dim = (c.rope_dim > 0) ? c.rope_dim : c.head_dim;
    const float attn_scale = 1.f / sqrtf((float)c.head_dim);
    const bool fnq = pf_fnq(s.gguf);

    kernels::launch_embedding(d_ids, s.w.embed_tokens, x, N, H, st);
    kernels::launch_rmsnorm(x, s.w.layers[0].input_norm, xn, N, H, eps, st);

    for (int L = 0; L < c.n_layers; L++) {
        const Qwen35LayerWeights& w = s.w.layers[L];
        if (w.linear_attn) {
            proj(xn, w.wqkv,      w.wqkv_type,      b8, lqkv,  H);
            proj(xn, w.wqkv_gate, w.wqkv_gate_type, lz, lvdim, H);
            proj(xn, w.ssm_alpha, w.ssm_alpha_type, la, vh,    H);
            proj(xn, w.ssm_beta,  w.ssm_beta_type,  lb, vh,    H);
            bf16* conv_state = lin_conv_state + (size_t)L * (c.linear_conv_kernel - 1) * lqkv;
            float* layer_state = s.lin_state + (size_t)L * vh * c.linear_head_dim * c.linear_head_dim;
            pf_gdn_decode_tokens(b8, lz, la, lb, w, conv_state, layer_state, gq, gk, gv, att, lnrm,
                N, c, vh, lqkv, lvdim, s.linear_qdim, st);
            proj(lnrm, w.ssm_out, w.ssm_out_type, ao, H, lvdim);
        } else {
            proj(xn, w.wq, w.wq_type, b8, wide,  H);
            proj(xn, w.wk, w.wk_type, kf, kvdim, H);
            proj(xn, w.wv, w.wv_type, vf, kvdim, H);
            kernels::launch_prefill_split_q_gate(b8, qb, qg, N, c.n_q_heads, c.head_dim, st);
            signed char* kpool = (signed char*)s.kv->k_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
            signed char* vpool = (signed char*)s.kv->v_pool() + (size_t)L * s.kv->layer_stride_elems() * kv_elem;
            void* kscale = kv8 ? (char*)s.kv->k_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
            void* vscale = kv8 ? (char*)s.kv->v_scale_pool() + (size_t)L * s.kv->scale_layer_stride_elems() * 2 : nullptr;
            if (!kv8) { a.free_all(); a8.free_all(); fprintf(stderr, "[prefill-moe] batched prefill requires int8 KV\n"); return -1; }
            kernels::launch_prefill_qknorm_rope_kv_int8(qb, kf, vf, w.q_norm, w.k_norm,
                kpool, vpool, kscale, vscale, btable, N, c.n_q_heads, c.n_kv_heads, c.head_dim,
                rope_dim, rope_theta, eps, bs, mbs, st);
            pf_flash_attn_tokens(qb, qg, att, kpool, vpool, kscale, vscale, btable, N, qdim,
                c.n_q_heads, c.n_kv_heads, c.head_dim, bs, mbs, attn_scale,
                w.q_has_gate, c, fa_m, fa_l, fa_acc, s.d_scalars, d_seqlen, prompt_ids, st);
            proj(att, w.wo, w.wo_type, ao, H, qdim);
        }

        const void* next_norm = (L + 1 < c.n_layers) ? s.w.layers[L + 1].input_norm : s.w.final_norm;
        const bool fnq_per_tok = fnq && N > 128;
        if (!fnq_per_tok) {
            kernels::launch_prefill_add(x, ao, hbuf, (long)N * H, st);
            kernels::launch_rmsnorm(hbuf, w.post_attn_norm, hn, N, H, eps, st);
        }
        for (int t = 0; t < N; t++) {
            bf16* ao_t    = ao    + (size_t)t * H;
            bf16* hbuf_t  = hbuf  + (size_t)t * H;
            bf16* hn_t    = hn    + (size_t)t * H;
            bf16* x_t     = x     + (size_t)t * H;
            bf16* xn_t    = xn    + (size_t)t * H;
            if (fnq_per_tok)
                kernels::launch_add_rmsnorm2_q8(x_t, ao_t, w.post_attn_norm, hbuf_t, hn_t, hn_q8, H, eps, st);
            else if (fnq)
                kernels::launch_quantize_q8_1_blocks(hn_t, hn_q8, H, st);
            pf_cu(cudaMemsetAsync(routed, 0, (size_t)H * sizeof(bf16), st), "prefill-moe routed zero");
            if (c.n_shared > 0 && w.shared_gate_q) {
                const float* dw = nullptr;
                if (w.shared_gate_inp) {
                    if (w.shared_gate_inp_type == 12 && H == 2048)
                        kernels::launch_mmvq_q4k_sigmoid(hn_q8, w.shared_gate_inp, d_shared_w, H, st);
                    else if (w.shared_gate_inp_type)
                        kernels::launch_mmvq_q4k(hn_q8, w.shared_gate_inp, shared_gate_tmp, 1, H, st);
                    else {
                        kernels::launch_gemv(hn_t, w.shared_gate_inp, shared_gate_tmp, 1, H, st);
                        kernels::launch_qwen36_sigmoid_scalar(shared_gate_tmp, d_shared_w, st);
                    }
                    if (w.shared_gate_inp_type && !(w.shared_gate_inp_type == 12 && H == 2048))
                        kernels::launch_qwen36_sigmoid_scalar(shared_gate_tmp, d_shared_w, st);
                    dw = d_shared_w;
                }
                kernels::launch_shared_expert_q8_mmvq(
                    hn_t, hn_q8, w.shared_gate_q, w.shared_up_q, w.shared_down_q,
                    dw, shared, mf_h, sh_q8_buf, H, ffn, st, false);
            }
            kernels::launch_router_fused(hn_t, w.router_w, mf_logits, mf_rc,
                                         mf_ids, mf_weights, c.n_experts, H, topk, 1, st);
            kernels::launch_moe_expert_ffn_q4k(hn_t, w.gate_q, w.up_q, w.down_q,
                                               w.gate_qtype, w.up_qtype, w.down_qtype,
                                               mf_ids, mf_weights, routed, mf_h, mf_out,
                                               1, topk, H, ffn, fnq ? hn_q8 : nullptr, st);
            if (c.n_shared > 0 && w.shared_gate_q) {
                if (pf_addnorm3()) {
                    if (fnq)
                        kernels::launch_add_rmsnorm3_q8(hbuf_t, routed, shared, next_norm, x_t, xn_t, hn_q8, H, eps, st);
                    else
                        kernels::launch_add_rmsnorm3(hbuf_t, routed, shared, next_norm, x_t, xn_t, 1, H, eps, st);
                } else {
                    launch_residual_add(routed, shared, routed, H, st);
                    if (fnq)
                        kernels::launch_add_rmsnorm2_q8(hbuf_t, routed, next_norm, x_t, xn_t, hn_q8, H, eps, st);
                    else
                        kernels::launch_add_rmsnorm2(hbuf_t, routed, next_norm, x_t, xn_t, 1, H, eps, st);
                }
            } else if (fnq) {
                kernels::launch_add_rmsnorm2_q8(hbuf_t, routed, next_norm, x_t, xn_t, hn_q8, H, eps, st);
            } else {
                kernels::launch_add_rmsnorm2(hbuf_t, routed, next_norm, x_t, xn_t, 1, H, eps, st);
            }
        }
    }

    const bf16* xn_last = xn + (size_t)(N - 1) * H;
    if (s.w.lm_head_type)
        kernels::launch_gemv_q_f32(xn_last, s.w.lm_head, s.w.lm_head_type, s.logits, c.vocab, H, st);
    else
        kernels::launch_gemv_f32(xn_last, s.w.lm_head, s.logits, c.vocab, H, st);
    kernels::launch_argmax(s.logits, s.d_out_id, 1, c.vocab, st);
    pf_cu(cudaMemcpyAsync(s.h_out_id, s.d_out_id, sizeof(int), cudaMemcpyDeviceToHost, st), "prefill-moe seed");
    pf_cu(cudaStreamSynchronize(st), "prefill-moe sync");
    if (cudaGetLastError() != cudaSuccess) {
        a.free_all();
        a8.free_all();
        fprintf(stderr, "[prefill-moe] cuda error after sync -> fallback\n");
        return -1;
    }
    int seed = *s.h_out_id;

    a.free_all();
    a8.free_all();
    return seed;
}

} // namespace sparkinfer
